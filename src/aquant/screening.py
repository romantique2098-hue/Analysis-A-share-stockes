from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd

from aquant.config import AppConfig
from aquant.data import DataQualityReport, MarketDataRepository, atomic_write_parquet
from aquant.industries import IndustryClassificationService
from aquant.storage import StateStore
from aquant.strategy import BollingerMeanReversionStrategy, TrendBreakoutStrategy


@dataclass
class ScreeningResult:
    run_id: str
    as_of_date: date
    status: str
    results: pd.DataFrame
    output_dir: Path | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def selected(self) -> pd.DataFrame:
        if self.results.empty or "selected" not in self.results:
            return self.results.iloc[0:0]
        return self.results[self.results["selected"]].copy()


class UniverseService:
    def __init__(
        self,
        config: AppConfig,
        repository: MarketDataRepository | None = None,
        store: StateStore | None = None,
    ) -> None:
        self.config = config
        self.repository = repository or MarketDataRepository(config)
        self.store = store or StateStore(config.storage.database)
        self.logger = logging.getLogger("aquant.universe")

    def sync(self) -> pd.DataFrame:
        raw = ak.stock_info_a_code_name()
        if raw.empty or not {"code", "name"} <= set(raw.columns):
            raise RuntimeError("A股代码表为空或字段发生变化")
        universe = raw[["code", "name"]].rename(columns={"code": "symbol"}).copy()
        universe["symbol"] = universe["symbol"].astype(str).str.strip().str.zfill(6)
        universe.loc[~universe["symbol"].str.fullmatch(r"\d{6}"), "symbol"] = np.nan
        universe = universe.dropna(subset=["symbol", "name"])
        universe["is_st"] = universe["name"].astype(str).str.upper().str.contains("ST|退")
        universe["updated_at"] = datetime.now().isoformat(timespec="seconds")
        universe = universe.drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)
        atomic_write_parquet(universe, self.config.storage.universe_file)
        return universe

    def load(self) -> pd.DataFrame:
        if not self.config.storage.universe_file.exists():
            return self.sync()
        data = pd.read_parquet(self.config.storage.universe_file)
        data["symbol"] = data["symbol"].astype(str).str.zfill(6)
        return data.drop_duplicates("symbol").reset_index(drop=True)

    def available_symbols(self) -> list[str]:
        source = self.config.screening.universe_source
        if source == "watchlist":
            return list(self.config.market.watchlist)
        local = sorted(path.stem for path in self.config.storage.adjusted_dir.glob("*.parquet"))
        if source == "local":
            return local
        universe = set(self.load()["symbol"])
        return [symbol for symbol in local if symbol in universe]

    def coverage(self) -> dict[str, int]:
        if not self.config.storage.universe_file.exists():
            return {"universe": 0, "adjusted": 0, "raw": 0, "both": 0}
        universe = {
            str(symbol).zfill(6)
            for symbol in pd.read_parquet(self.config.storage.universe_file, columns=["symbol"])[
                "symbol"
            ]
        }
        adjusted = {
            path.stem for path in self.config.storage.adjusted_dir.glob("*.parquet")
        } & universe
        raw = {path.stem for path in self.config.storage.raw_dir.glob("*.parquet")} & universe
        return {
            "universe": len(universe),
            "adjusted": len(adjusted),
            "raw": len(raw),
            "both": len(adjusted & raw),
        }

    def update_next_batch(
        self,
        batch_size: int | None = None,
        *,
        include_raw: bool = False,
        progress_callback: Callable[[int, int, DataQualityReport], None] | None = None,
    ) -> list[DataQualityReport]:
        universe = self.load()
        symbols = universe.loc[~universe["is_st"], "symbol"].tolist()
        if not symbols:
            return []
        size = batch_size or self.config.screening.universe_batch_size
        cursor = int(self.store.get_state("universe_update_cursor", "0") or 0) % len(symbols)
        indices = [(cursor + offset) % len(symbols) for offset in range(min(size, len(symbols)))]
        batch = [symbols[index] for index in indices]
        start = date.fromisoformat(self.config.market.start_date)
        end = datetime.now().date()
        reports: list[DataQualityReport] = []
        for completed, symbol in enumerate(batch, start=1):
            try:
                if include_raw:
                    reports.append(self.repository.update_symbol(symbol, "", start, end))
                reports.append(self.repository.update_symbol(symbol, "qfq", start, end))
            except Exception as exc:
                self.logger.error("股票宇宙批量更新 %s 失败：%s", symbol, exc)
                reports.append(
                    DataQualityReport(
                        symbol=symbol,
                        adjustment="raw+qfq" if include_raw else "qfq",
                        rows=0,
                        start_date=None,
                        end_date=None,
                        used_cache=True,
                        warnings=[str(exc)],
                    )
                )
            if progress_callback:
                progress_callback(completed, len(batch), reports[-1])
        self.store.set_state("universe_update_cursor", str((cursor + len(batch)) % len(symbols)))
        return reports


class FundamentalFactorService:
    """按技术预选候选拉取财务数据，并用本地快照做失败回退。"""

    COLUMNS = [
        "symbol",
        "factor_date",
        "industry",
        "pe_ratio",
        "pb_ratio",
        "roe",
        "profit_growth",
        "debt_ratio",
    ]

    def __init__(self, config: AppConfig, repository: MarketDataRepository) -> None:
        self.config = config
        self.repository = repository
        self.logger = logging.getLogger("aquant.factors")

    def load(self) -> pd.DataFrame:
        path = self.config.storage.factor_file
        if not path.exists():
            return pd.DataFrame(columns=self.COLUMNS)
        data = pd.read_parquet(path)
        data["symbol"] = data["symbol"].astype(str).str.zfill(6)
        return data.drop_duplicates("symbol", keep="last")

    @staticmethod
    def _number(row: pd.Series, *columns: str) -> float:
        for column in columns:
            if column in row and pd.notna(row[column]):
                return float(row[column])
        return np.nan

    def _fetch_symbol(self, symbol: str) -> dict[str, object]:
        financial = ak.stock_financial_analysis_indicator(
            symbol=symbol, start_year=str(datetime.now().year - 2)
        )
        if financial.empty:
            raise RuntimeError("财务指标为空")
        financial["日期"] = pd.to_datetime(financial["日期"])
        latest = financial.sort_values("日期").iloc[-1]
        factor_date = pd.Timestamp(latest["日期"])
        eps = self._number(latest, "摊薄每股收益(元)", "加权每股收益(元)")
        book_value = self._number(latest, "每股净资产_调整前(元)", "每股净资产_调整后(元)")
        month = max(factor_date.month, 3)
        annualized_eps = eps * 12 / month if eps > 0 else np.nan
        price = float(self.repository.load(symbol, "qfq")["close"].iloc[-1])
        return {
            "symbol": symbol,
            "factor_date": factor_date,
            # 行业分类由独立服务维护。保留此列只为兼容现有因子快照。
            "industry": np.nan,
            "pe_ratio": price / annualized_eps if annualized_eps > 0 else np.nan,
            "pb_ratio": price / book_value if book_value > 0 else np.nan,
            "roe": self._number(latest, "净资产收益率(%)", "加权净资产收益率(%)"),
            "profit_growth": self._number(latest, "净利润增长率(%)"),
            "debt_ratio": self._number(latest, "资产负债率(%)"),
        }

    def update(self, symbols: list[str]) -> tuple[pd.DataFrame, list[str]]:
        cached = self.load().set_index("symbol", drop=False)
        rows: list[dict[str, object]] = []
        warnings: list[str] = []
        for symbol in symbols:
            try:
                row = self._fetch_symbol(symbol)
                if symbol in cached.index:
                    cached_industry = cached.loc[symbol].get("industry")
                    if pd.notna(cached_industry) and str(cached_industry) != "未分类":
                        row["industry"] = cached_industry
                rows.append(row)
            except Exception as exc:
                warnings.append(f"{symbol} 基本面更新失败：{exc}")
                if symbol in cached.index:
                    rows.append(cached.loc[symbol].to_dict())
        updated = pd.DataFrame(rows, columns=self.COLUMNS)
        untouched = cached.loc[~cached.index.isin(symbols)].reset_index(drop=True)
        frames = [frame for frame in [untouched, updated] if not frame.empty]
        combined = (
            pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=self.COLUMNS)
        )
        if not combined.empty:
            atomic_write_parquet(combined[self.COLUMNS], self.config.storage.factor_file)
        return combined, warnings


class MultiFactorScorer:
    FACTORS = [
        "momentum_factor",
        "value_factor",
        "quality_factor",
        "low_volatility_factor",
        "liquidity_factor",
    ]

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    @staticmethod
    def _row_mean(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
        return frame[columns].mean(axis=1, skipna=True)

    @staticmethod
    def _neutral_zscore(values: pd.Series, industry: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce")
        valid = numeric.dropna()
        if valid.empty:
            return pd.Series(np.nan, index=values.index)
        lower, upper = valid.quantile([0.025, 0.975])
        clipped = numeric.clip(lower, upper)
        global_std = clipped.std(ddof=0)
        global_score = (
            (clipped - clipped.mean()) / global_std
            if global_std and pd.notna(global_std)
            else pd.Series(0.0, index=values.index)
        )

        def score_group(group: pd.Series) -> pd.Series:
            if group.count() < 3 or not group.std(ddof=0):
                return global_score.loc[group.index]
            return (group - group.mean()) / group.std(ddof=0)

        return clipped.groupby(industry.fillna("未分类"), group_keys=False).apply(score_group)

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        data = frame.copy()
        positive_pe = data["pe_ratio"].where(data["pe_ratio"] > 0)
        positive_pb = data["pb_ratio"].where(data["pb_ratio"] > 0)
        data["value_factor"] = self._row_mean(
            pd.DataFrame({"earnings_yield": 1 / positive_pe, "book_yield": 1 / positive_pb}),
            ["earnings_yield", "book_yield"],
        )
        data["quality_factor"] = self._row_mean(
            pd.DataFrame(
                {
                    "roe": data["roe"],
                    "growth": data["profit_growth"],
                    "balance_sheet": -data["debt_ratio"],
                }
            ),
            ["roe", "growth", "balance_sheet"],
        )
        weights = {
            "momentum_factor": self.config.screening.momentum_weight,
            "value_factor": self.config.screening.value_weight,
            "quality_factor": self.config.screening.quality_weight,
            "low_volatility_factor": self.config.screening.low_volatility_weight,
            "liquidity_factor": self.config.screening.liquidity_weight,
        }
        numerator = pd.Series(0.0, index=data.index)
        denominator = pd.Series(0.0, index=data.index)
        for factor, weight in weights.items():
            data[f"{factor}_z"] = self._neutral_zscore(data[factor], data["industry"])
            available = data[factor].notna()
            numerator += data[f"{factor}_z"].fillna(0) * weight
            denominator += available.astype(float) * weight
        data["factor_coverage"] = data[self.FACTORS].notna().sum(axis=1)
        data["total_score"] = numerator / denominator.replace(0, np.nan)
        return data


class ScreeningEngine:
    def __init__(
        self,
        config: AppConfig,
        repository: MarketDataRepository | None = None,
        store: StateStore | None = None,
    ) -> None:
        self.config = config
        self.repository = repository or MarketDataRepository(config)
        self.store = store or StateStore(config.storage.database)
        self.universe = UniverseService(config, self.repository, self.store)
        self.fundamentals = FundamentalFactorService(config, self.repository)
        self.industries = IndustryClassificationService(config, self.store)

    def is_due(self, today: date | None = None) -> bool:
        today = today or datetime.now().date()
        latest = self.store.latest_screen_date()
        if latest is None:
            return True
        schedule = self.config.screening.schedule
        if schedule == "daily":
            return latest < today
        if schedule == "weekly":
            return today.isoweekday() >= self.config.screening.schedule_weekday and (
                latest.isocalendar()[:2] != today.isocalendar()[:2]
            )
        return (latest.year, latest.month) != (today.year, today.month)

    def _technical_snapshot(
        self,
        symbols: list[str],
        as_of: date | None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        trend_strategy = TrendBreakoutStrategy(self.config.strategy)
        mean_strategy = BollingerMeanReversionStrategy(self.config.strategy)
        periods = (
            self.config.strategy.momentum_short_period,
            self.config.strategy.momentum_medium_period,
            self.config.strategy.momentum_long_period,
        )
        total = len(symbols)
        for completed, symbol in enumerate(symbols, start=1):
            if progress_callback and (
                completed == 1 or completed % 25 == 0 or completed == total
            ):
                progress_callback(completed, total, symbol)
            try:
                history = self.repository.load(symbol, "qfq")
            except Exception:
                continue
            if as_of:
                history = history[pd.to_datetime(history["date"]).dt.date <= as_of]
            if len(history) < self.config.screening.minimum_history_days:
                continue
            history = history.sort_values("date")
            close = history["close"].astype(float)
            returns = {
                period: close.iloc[-1] / close.iloc[-period - 1] - 1
                for period in periods
                if len(close) > period
            }
            if len(returns) != len(periods):
                continue
            daily_return = close.pct_change(fill_method=None)
            annual_volatility = daily_return.tail(60).std() * np.sqrt(252)
            average_amount = history["amount"].tail(20).mean()
            trend = trend_strategy.generate_signals(history).iloc[-1]
            mean_reversion = mean_strategy.generate_signals(history).iloc[-1]
            rows.append(
                {
                    "symbol": symbol,
                    "price_date": pd.Timestamp(history["date"].iloc[-1]),
                    "is_fresh": (
                        (as_of or datetime.now().date())
                        - pd.Timestamp(history["date"].iloc[-1]).date()
                    ).days
                    <= self.config.market.max_stale_days,
                    "history_days": len(history),
                    "close": close.iloc[-1],
                    "return_20": returns[periods[0]],
                    "return_60": returns[periods[1]],
                    "return_120": returns[periods[2]],
                    "momentum_factor": 0.2 * returns[periods[0]]
                    + 0.3 * returns[periods[1]]
                    + 0.5 * returns[periods[2]],
                    "annual_volatility": annual_volatility,
                    "low_volatility_factor": -annual_volatility,
                    "average_amount_20": average_amount,
                    "liquidity_factor": np.log1p(average_amount),
                    "trend_signal": int(trend["signal"]),
                    "breakout_level": trend["breakout_high"],
                    "atr_stop": trend["atr_stop"],
                    "mean_reversion_signal": int(mean_reversion["signal"]),
                    "rsi": mean_reversion["rsi"],
                    "bollinger_lower": mean_reversion["bollinger_lower"],
                }
            )
        return pd.DataFrame(rows)

    def run(
        self,
        *,
        force: bool = False,
        refresh_universe: bool = False,
        refresh_fundamentals: bool = False,
        as_of: date | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> ScreeningResult:
        today = as_of or datetime.now().date()
        if not force and not self.is_due(today):
            return ScreeningResult("", today, "SKIPPED", pd.DataFrame())
        if refresh_universe:
            self.universe.sync()
        symbols = self.universe.available_symbols()
        technical = self._technical_snapshot(symbols, as_of, progress_callback)
        if technical.empty:
            raise RuntimeError("没有满足历史长度要求的本地行情，无法筛选")
        universe = self.universe.load()[["symbol", "name", "is_st"]]
        technical = technical.merge(universe, on="symbol", how="left")
        technical["name"] = technical["name"].fillna(technical["symbol"])
        technical["is_st"] = technical["is_st"].fillna(False).astype(bool)
        technical["eligible"] = (
            technical["average_amount_20"].ge(self.config.screening.minimum_average_amount)
            & technical["is_fresh"]
        )
        if self.config.screening.exclude_st:
            technical["eligible"] &= ~technical["is_st"]
        preselected = (
            technical[technical["eligible"]]
            .nlargest(
                self.config.screening.technical_preselect_count,
                "momentum_factor",
            )["symbol"]
            .tolist()
        )
        warnings: list[str] = []
        if self.config.screening.universe_source == "all_a":
            total_universe = len(universe)
            if len(technical) < total_universe:
                warnings.append(
                    f"全A股覆盖 {len(technical)}/{total_universe}；请持续运行 universe update 扩充"
                )
        if refresh_fundamentals:
            fundamentals, factor_warnings = self.fundamentals.update(preselected)
            warnings.extend(factor_warnings)
        else:
            fundamentals = self.fundamentals.load()
            if fundamentals.empty:
                warnings.append("基本面快照为空，本次仅按可用技术/风险因子评分")
        # 行业分类覆盖整个研究行情宇宙，不再只依附于技术预选股的财务请求。
        # 独立服务还会合并旧因子快照和 SQLite 手工修正。
        industry_snapshot = self.industries.resolve(technical["symbol"].tolist())
        factor_values = fundamentals.drop(columns=["industry"], errors="ignore")
        data = technical.merge(factor_values, on="symbol", how="left")
        data = data.merge(industry_snapshot, on="symbol", how="left")
        data["industry"] = data["industry"].fillna("未分类")
        data["industry_status"] = data["industry_status"].fillna("missing")
        data["industry_source"] = data["industry_source"].fillna("")
        data["industry_error"] = data["industry_error"].fillna("")
        classified = int(data["industry"].ne("未分类").sum())
        if classified < len(data):
            warnings.append(
                f"行业分类覆盖 {classified}/{len(data)}；可在自动选股页后台回填或手工修正"
            )
        scored = MultiFactorScorer(self.config).score(data)
        scored.loc[~scored["eligible"], "total_score"] = np.nan
        scored["rank"] = scored["total_score"].rank(ascending=False, method="first")
        scored["selected"] = False
        industry_counts: dict[str, int] = {}
        selected = 0
        for index in scored.sort_values("rank").index:
            if not scored.loc[index, "eligible"] or pd.isna(scored.loc[index, "rank"]):
                continue
            industry = str(scored.loc[index, "industry"])
            if industry_counts.get(industry, 0) >= self.config.screening.max_per_industry:
                continue
            scored.loc[index, "selected"] = True
            industry_counts[industry] = industry_counts.get(industry, 0) + 1
            selected += 1
            if selected >= self.config.screening.selection_count:
                break
        scored = scored.sort_values(["selected", "rank"], ascending=[False, True])
        scored = scored.reset_index(drop=True)
        data_date = pd.to_datetime(scored["price_date"]).max().date()
        fingerprint = {
            "as_of": data_date.isoformat(),
            "symbols": scored["symbol"].tolist(),
            "config": self.config.screening.model_dump(),
            "input_digest": hashlib.sha256(
                pd.util.hash_pandas_object(
                    scored.drop(
                        columns=["industry_error", "industry_updated_at"], errors="ignore"
                    )
                    .sort_values("symbol")
                    .reindex(
                        sorted(
                            scored.drop(
                                columns=["industry_error", "industry_updated_at"],
                                errors="ignore",
                            ).columns
                        ),
                        axis=1,
                    ),
                    index=False,
                ).values.tobytes()
            ).hexdigest(),
        }
        run_id = hashlib.sha256(
            json.dumps(fingerprint, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()[:16]
        output_dir = self.config.storage.output_dir / "screens" / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        scored.to_parquet(output_dir / "screening.parquet", index=False)
        scored.to_csv(output_dir / "screening.csv", index=False)
        (output_dir / "summary.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "as_of_date": data_date.isoformat(),
                    "universe_size": len(scored),
                    "eligible_count": int(scored["eligible"].sum()),
                    "selected_count": int(scored["selected"].sum()),
                    "warnings": warnings,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self.store.save_screening_run(run_id, data_date, "SUCCESS", scored, output_dir, warnings)
        return ScreeningResult(run_id, data_date, "SUCCESS", scored, output_dir, warnings)
