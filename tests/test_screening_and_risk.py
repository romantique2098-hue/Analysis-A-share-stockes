from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from conftest import make_prices

from aquant.backtest import BacktestEngine
from aquant.data import DataQualityReport, MarketDataRepository, atomic_write_parquet
from aquant.screening import MultiFactorScorer, ScreeningEngine, UniverseService
from aquant.strategy import BollingerMeanReversionStrategy, TrendBreakoutStrategy


def test_trend_breakout_has_atr_stop_and_exits(app_config) -> None:
    history = make_prices("000001", periods=100, pattern="trend")
    history.loc[history.index[-5] :, ["open", "close", "high", "low"]] *= 0.65
    strategy = TrendBreakoutStrategy(app_config.strategy)
    result = strategy.generate_signals(history)
    assert result["signal"].eq(1).any()
    assert result.loc[result["signal"] == 1, "atr_stop"].notna().any()
    assert result["signal"].iloc[-1] == 0
    assert result["breakout_high"].notna().any()


def test_bollinger_mean_reversion_combines_oversold_signals(app_config) -> None:
    history = make_prices("000001", periods=100, pattern="flat")
    history.loc[60:65, "close"] *= 0.78
    history["open"] = history["close"] * 0.999
    history["high"] = history[["open", "close"]].max(axis=1) * 1.01
    history["low"] = history[["open", "close"]].min(axis=1) * 0.99
    result = BollingerMeanReversionStrategy(app_config.strategy).generate_signals(history)
    assert result["signal"].eq(1).any()
    assert result["bollinger_lower"].notna().any()
    assert result["short_return"].min() <= app_config.strategy.oversold_return


def test_multifactor_score_is_industry_neutral(app_config) -> None:
    frame = pd.DataFrame(
        {
            "symbol": [f"00000{i}" for i in range(1, 7)],
            "industry": ["A"] * 3 + ["B"] * 3,
            "momentum_factor": [1, 2, 3, 10, 20, 30],
            "low_volatility_factor": [3, 2, 1, 30, 20, 10],
            "liquidity_factor": [1, 2, 3, 10, 20, 30],
            "pe_ratio": [10, 12, 14, 20, 22, 24],
            "pb_ratio": [1, 2, 3, 2, 3, 4],
            "roe": [8, 10, 12, 15, 18, 20],
            "profit_growth": [5, 10, 15, 20, 25, 30],
            "debt_ratio": [50, 40, 30, 60, 50, 40],
        }
    )
    result = MultiFactorScorer(app_config).score(frame)
    for column in [
        "momentum_factor_z",
        "value_factor_z",
        "quality_factor_z",
        "low_volatility_factor_z",
        "liquidity_factor_z",
    ]:
        means = result.groupby("industry")[column].mean()
        assert np.allclose(means, 0, atol=1e-10)
    assert result["total_score"].notna().all()


def test_screening_persists_ranked_results(app_config) -> None:
    app_config.screening.universe_source = "local"
    app_config.screening.minimum_average_amount = 0
    app_config.screening.selection_count = 3
    app_config.screening.technical_preselect_count = 5
    app_config.screening.max_per_industry = 2
    repository = MarketDataRepository(app_config)
    symbols = ["000001", "000002", "000003", "000004", "000005"]
    for number, symbol in enumerate(symbols, start=1):
        history = make_prices(symbol, periods=180, pattern="trend")
        history["close"] *= 1 + number * np.linspace(0, 0.3, len(history))
        history["open"] = history["close"] * 0.999
        history["high"] = history[["open", "close"]].max(axis=1) * 1.01
        history["low"] = history[["open", "close"]].min(axis=1) * 0.99
        atomic_write_parquet(history, repository.path_for(symbol, "qfq"))
    universe = pd.DataFrame(
        {
            "symbol": symbols,
            "name": [f"测试{number}" for number in range(5)],
            "is_st": False,
            "updated_at": "2026-01-01",
        }
    )
    atomic_write_parquet(universe, app_config.storage.universe_file)
    factors = pd.DataFrame(
        {
            "symbol": symbols,
            "factor_date": pd.Timestamp("2026-01-01"),
            "industry": ["A", "A", "A", "B", "B"],
            "pe_ratio": [10, 12, 14, 20, 22],
            "pb_ratio": [1, 1.2, 1.4, 2, 2.2],
            "roe": [10, 11, 12, 13, 14],
            "profit_growth": [5, 6, 7, 8, 9],
            "debt_ratio": [40, 38, 36, 34, 32],
        }
    )
    atomic_write_parquet(factors, app_config.storage.factor_file)
    result = ScreeningEngine(app_config, repository).run(force=True)
    assert result.status == "SUCCESS"
    assert len(result.results) == 5
    assert len(result.selected) == 3
    assert result.output_dir is not None
    assert (result.output_dir / "screening.parquet").exists()
    assert result.results["rank"].notna().all()
    assert len(result.selected[result.selected["industry"] == "A"]) <= 2


def test_universe_sync_normalizes_codes(app_config, monkeypatch) -> None:
    monkeypatch.setattr(
        "aquant.screening.ak.stock_info_a_code_name",
        lambda: pd.DataFrame(
            {"code": [1, "600519", "000002"], "name": ["测试", "贵州茅台", "*ST测试"]}
        ),
    )
    result = UniverseService(app_config).sync()
    assert result["symbol"].tolist() == ["000001", "000002", "600519"]
    assert result.set_index("symbol").loc["000002", "is_st"]


def test_universe_batch_defaults_to_research_prices_only(app_config, monkeypatch) -> None:
    symbols = ["000001", "000002", "000003"]
    universe = pd.DataFrame(
        {
            "symbol": symbols,
            "name": symbols,
            "is_st": False,
            "updated_at": "2026-01-01",
        }
    )
    atomic_write_parquet(universe, app_config.storage.universe_file)
    repository = MarketDataRepository(app_config)
    calls: list[tuple[str, str]] = []

    def update_symbol(symbol, adjustment, start, end):
        calls.append((symbol, adjustment))
        return DataQualityReport(symbol, adjustment, 100, start, end)

    monkeypatch.setattr(repository, "update_symbol", update_symbol)
    service = UniverseService(app_config, repository)

    reports = service.update_next_batch(2)
    service.update_next_batch(1, include_raw=True)

    assert [(report.symbol, report.adjustment) for report in reports] == [
        ("000001", "qfq"),
        ("000002", "qfq"),
    ]
    assert calls == [
        ("000001", "qfq"),
        ("000002", "qfq"),
        ("000003", ""),
        ("000003", "qfq"),
    ]


def test_weekly_screening_is_due_only_once(app_config, monkeypatch) -> None:
    app_config.screening.schedule = "weekly"
    app_config.screening.schedule_weekday = 5
    engine = ScreeningEngine(app_config)
    monkeypatch.setattr(engine.store, "latest_screen_date", lambda: date(2025, 1, 3))

    assert not engine.is_due(date(2025, 1, 9))
    assert engine.is_due(date(2025, 1, 10))


class AlwaysLongStrategy:
    name = "always_long"

    def generate_signals(self, history, parameters=None):
        result = history[["symbol", "date", "close"]].copy()
        result["signal"] = 1
        return result


def test_portfolio_drawdown_triggers_risk_off(app_config) -> None:
    app_config.portfolio.max_portfolio_drawdown = 0.03
    app_config.portfolio.drawdown_cooldown_days = 10
    history = make_prices("000001", periods=100, pattern="trend")
    history.loc[70:, ["open", "close", "high", "low"]] *= np.linspace(1, 0.45, 30)[:, None]
    result = BacktestEngine(app_config, AlwaysLongStrategy()).run({"000001": history})
    assert result.equity_curve["risk_off"].any()
    assert result.equity_curve["portfolio_drawdown"].min() <= -0.03
    assert not result.equity_curve["risk_off"].iloc[-1]


def test_backtest_run_id_hashes_complete_history(app_config) -> None:
    history = make_prices("000001", periods=100, pattern="trend")
    revised = history.copy()
    revised.loc[40, ["open", "close", "high", "low"]] *= 1.02

    first = BacktestEngine(app_config, AlwaysLongStrategy()).run({"000001": history})
    second = BacktestEngine(app_config, AlwaysLongStrategy()).run({"000001": revised})

    assert first.run_id != second.run_id
    assert first.parameters["data_fingerprints"] != second.parameters["data_fingerprints"]


def test_comparison_curves_do_not_backfill_future_prices(app_config) -> None:
    early = make_prices("000001", periods=100, start="2025-01-02", pattern="trend")
    late = make_prices("000002", periods=40, start="2025-03-03", pattern="trend")
    benchmark = make_prices("000300", periods=40, start="2025-03-03", pattern="trend")

    result = BacktestEngine(app_config, AlwaysLongStrategy()).run(
        {"000001": early, "000002": late}, benchmark
    )
    curve = result.equity_curve.set_index("date")
    before_late = pd.Timestamp("2025-02-03")
    expected = float(early.set_index("date").loc[before_late, "close"] / early.iloc[0]["close"])

    assert curve.loc[before_late, "buy_hold_curve"] == pytest.approx(expected)
    assert pd.isna(curve.loc[before_late, "benchmark_curve"])
