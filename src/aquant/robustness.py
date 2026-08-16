from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from aquant.backtest import BacktestEngine
from aquant.config import AppConfig
from aquant.data import MarketDataRepository

DEFAULT_MA_CANDIDATES: tuple[dict[str, float | int], ...] = (
    {
        "short_window": 5,
        "long_window": 20,
        "ma_signal_buffer": 0.0,
        "ma_confirmation_days": 1,
    },
    {
        "short_window": 5,
        "long_window": 20,
        "ma_signal_buffer": 0.003,
        "ma_confirmation_days": 2,
    },
    {
        "short_window": 5,
        "long_window": 20,
        "ma_signal_buffer": 0.005,
        "ma_confirmation_days": 2,
    },
    {
        "short_window": 5,
        "long_window": 20,
        "ma_signal_buffer": 0.01,
        "ma_confirmation_days": 3,
    },
    {
        "short_window": 3,
        "long_window": 15,
        "ma_signal_buffer": 0.003,
        "ma_confirmation_days": 2,
    },
    {
        "short_window": 5,
        "long_window": 30,
        "ma_signal_buffer": 0.005,
        "ma_confirmation_days": 2,
    },
    {
        "short_window": 8,
        "long_window": 30,
        "ma_signal_buffer": 0.003,
        "ma_confirmation_days": 2,
    },
    {
        "short_window": 10,
        "long_window": 40,
        "ma_signal_buffer": 0.005,
        "ma_confirmation_days": 2,
    },
)


def _chronological_folds(
    histories: dict[str, pd.DataFrame],
    benchmark: pd.DataFrame,
    fold_count: int,
    *,
    start: date | None = None,
    end: date | None = None,
) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    if fold_count < 2:
        raise ValueError("稳健性分析至少需要两个时间分段")
    if not histories or benchmark.empty:
        raise ValueError("稳健性分析缺少股票或基准行情")
    history_dates: dict[str, pd.DatetimeIndex] = {}
    for symbol, frame in histories.items():
        dates = pd.DatetimeIndex(pd.to_datetime(frame["date"]).dt.normalize().dropna().unique())
        if dates.empty:
            raise ValueError(f"{symbol} 没有可用于稳健性分析的行情")
        history_dates[symbol] = dates.sort_values()
    benchmark_dates = pd.DatetimeIndex(
        pd.to_datetime(benchmark["date"]).dt.normalize().dropna().unique()
    ).sort_values()
    if benchmark_dates.empty:
        raise ValueError("基准没有可用于稳健性分析的行情")
    common_start = max([dates.min() for dates in history_dates.values()] + [benchmark_dates.min()])
    common_end = min([dates.max() for dates in history_dates.values()] + [benchmark_dates.max()])
    if start:
        common_start = max(common_start, pd.Timestamp(start))
    if end:
        common_end = min(common_end, pd.Timestamp(end))
    calendar = benchmark_dates[
        (benchmark_dates >= common_start) & (benchmark_dates <= common_end)
    ]
    if len(calendar) < fold_count * 20:
        raise ValueError("股票与基准的共同历史不足以进行时间分段稳健性分析")
    folds: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    for chunk in np.array_split(calendar, fold_count):
        fold_start, fold_end = chunk[0], chunk[-1]
        for symbol, dates in history_dates.items():
            if not ((dates >= fold_start) & (dates <= fold_end)).any():
                raise ValueError(f"{symbol} 在 {fold_start.date()} 至 {fold_end.date()} 没有行情")
        folds.append((fold_start, fold_end, len(chunk)))
    return folds


def analyze_ma_robustness(
    config: AppConfig,
    histories: dict[str, pd.DataFrame],
    benchmark: pd.DataFrame,
    *,
    candidates: tuple[dict[str, float | int], ...] = DEFAULT_MA_CANDIDATES,
    fold_count: int = 3,
    start: date | None = None,
    end: date | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """在连续时间段上比较候选参数，不自动把最优值写回正式配置。"""
    folds = _chronological_folds(
        histories,
        benchmark,
        fold_count,
        start=start,
        end=end,
    )
    rows: list[dict[str, object]] = []
    for candidate_index, candidate in enumerate(candidates, start=1):
        for fold_index, (fold_start, fold_end, trading_days) in enumerate(folds, start=1):
            run_config = config.model_copy(deep=True)
            run_config.strategy.name = "ma_cross"
            for name, value in candidate.items():
                setattr(run_config.strategy, name, value)
            result = BacktestEngine(run_config).run(
                histories,
                benchmark,
                start=fold_start.date(),
                end=fold_end.date(),
            )
            rows.append(
                {
                    "candidate": candidate_index,
                    **candidate,
                    "fold": fold_index,
                    "start": fold_start.date(),
                    "end": fold_end.date(),
                    "trading_days": trading_days,
                    "backtest_run_id": result.run_id,
                    **result.metrics,
                }
            )
    details = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    parameter_columns = [
        "short_window",
        "long_window",
        "ma_signal_buffer",
        "ma_confirmation_days",
    ]
    for candidate, group in details.groupby("candidate", sort=True):
        excess = group["excess_total_return"]
        annual_excess = group["excess_annual_return"]
        summary_rows.append(
            {
                "candidate": int(candidate),
                **{column: group.iloc[0][column] for column in parameter_columns},
                "folds": len(group),
                "positive_excess_folds": int(annual_excess.gt(0).sum()),
                "median_excess_return": float(excess.median()),
                "worst_excess_return": float(excess.min()),
                "median_excess_annual_return": float(annual_excess.median()),
                "lower_quartile_excess_annual_return": float(
                    annual_excess.quantile(0.25)
                ),
                "worst_excess_annual_return": float(annual_excess.min()),
                "median_information_ratio": float(group["information_ratio"].median()),
                "median_annual_return": float(group["annual_return"].median()),
                "worst_max_drawdown": float(group["max_drawdown"].min()),
                "median_annualized_turnover": float(group["annualized_turnover"].median()),
                "median_cost_ratio": float(group["cost_ratio"].median()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        [
            "positive_excess_folds",
            "lower_quartile_excess_annual_return",
            "worst_excess_annual_return",
            "median_information_ratio",
            "median_annualized_turnover",
        ],
        ascending=[False, False, False, False, True],
    )
    summary.insert(0, "robustness_rank", range(1, len(summary) + 1))
    return summary.reset_index(drop=True), details


def load_and_analyze_ma_robustness(
    config: AppConfig,
    *,
    start: date | None = None,
    end: date | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    repository = MarketDataRepository(config)
    histories = {symbol: repository.load(symbol, "qfq") for symbol in config.market.watchlist}
    benchmark = repository.load(config.market.benchmark, "", benchmark=True)
    return analyze_ma_robustness(
        config,
        histories,
        benchmark,
        start=start,
        end=end,
    )


def save_ma_robustness(
    summary: pd.DataFrame,
    details: pd.DataFrame,
    config: AppConfig,
) -> Path:
    fingerprint = {
        "config": config.model_dump(mode="json"),
        "summary": summary.to_dict("records"),
        "backtest_runs": sorted(details["backtest_run_id"].astype(str).unique()),
        "folds": details[["fold", "start", "end", "trading_days"]]
        .drop_duplicates()
        .to_dict("records"),
    }
    analysis_id = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    output_dir = config.storage.output_dir / "robustness" / analysis_id
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "summary.csv", index=False)
    details.to_csv(output_dir / "folds.csv", index=False)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {"analysis_id": analysis_id, **fingerprint},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )
    return output_dir
