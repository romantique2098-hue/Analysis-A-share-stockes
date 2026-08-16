from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Protocol

import akshare as ak
import pandas as pd

from aquant.config import AppConfig

REQUIRED_COLUMNS = ["symbol", "date", "open", "close", "high", "low", "volume", "amount"]
COLUMN_ALIASES = {
    "日期": "date",
    "股票代码": "symbol",
    "代码": "symbol",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
}


class DataValidationError(ValueError):
    """行情数据未通过安全校验。"""


class DataUnavailableError(RuntimeError):
    """数据源与本地缓存均不可用。"""


@dataclass
class DataQualityReport:
    symbol: str
    adjustment: str
    rows: int
    start_date: date | None
    end_date: date | None
    used_cache: bool = False
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "adjustment": self.adjustment or "raw",
            "rows": self.rows,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "used_cache": self.used_cache,
            "warnings": "；".join(self.warnings),
        }


class DataProvider(Protocol):
    def fetch_daily(
        self, symbol: str, start: date, end: date, adjustment: str = ""
    ) -> pd.DataFrame: ...

    def fetch_index_daily(self, symbol: str, start: date, end: date) -> pd.DataFrame: ...


def normalize_daily_data(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise DataValidationError(f"{symbol} 返回空行情")
    data = frame.rename(columns=COLUMN_ALIASES).copy()
    if "symbol" not in data:
        data["symbol"] = symbol
    missing = set(REQUIRED_COLUMNS) - set(data.columns)
    if missing:
        raise DataValidationError(f"{symbol} 缺少字段：{sorted(missing)}")
    data = data[REQUIRED_COLUMNS].copy()
    data["symbol"] = data["symbol"].fillna(symbol).astype(str).str.extract(r"(\d{6})")[0]
    data["symbol"] = data["symbol"].fillna(symbol).str.zfill(6)
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    for column in ["open", "close", "high", "low", "volume", "amount"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data[REQUIRED_COLUMNS].isna().any().any():
        bad = data.columns[data.isna().any()].tolist()
        raise DataValidationError(f"{symbol} 存在无法解析的字段：{bad}")
    if not data["symbol"].eq(symbol).all():
        raise DataValidationError(f"返回数据代码与请求代码 {symbol} 不一致")
    prices = data[["open", "close", "high", "low"]]
    invalid_prices = (
        prices.le(0).any(axis=1)
        | data["low"].gt(data["high"])
        | data["open"].lt(data["low"])
        | data["open"].gt(data["high"])
        | data["close"].lt(data["low"])
        | data["close"].gt(data["high"])
    )
    if invalid_prices.any():
        dates = data.loc[invalid_prices, "date"].dt.strftime("%Y-%m-%d").head(3).tolist()
        raise DataValidationError(f"{symbol} 存在异常 OHLC：{dates}")
    if data[["volume", "amount"]].lt(0).any().any():
        raise DataValidationError(f"{symbol} 成交量或成交额为负数")
    return (
        data.sort_values("date")
        .drop_duplicates(["symbol", "date"], keep="last")
        .reset_index(drop=True)
    )


class AkShareDataProvider:
    def __init__(self, timeout: float = 15, retries: int = 3) -> None:
        self.timeout = timeout
        self.retries = retries
        self.logger = logging.getLogger("aquant.data")
        self.stock_primary_available: bool | None = None
        self.index_primary_available: bool | None = None

    def _retry(self, operation: str, callback) -> pd.DataFrame:
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                return callback()
            except Exception as exc:  # 第三方接口异常类型不稳定，统一在边界处理。
                last_error = exc
                self.logger.warning("%s 第 %s/%s 次失败：%s", operation, attempt, self.retries, exc)
                if attempt < self.retries:
                    time.sleep(min(2 ** (attempt - 1), 4))
        raise DataUnavailableError(f"{operation} 失败：{last_error}") from last_error

    def fetch_daily(
        self, symbol: str, start: date, end: date, adjustment: str = ""
    ) -> pd.DataFrame:
        if adjustment not in {"", "qfq"}:
            raise ValueError("第一版仅支持未复权和前复权行情")
        try:
            if self.stock_primary_available is False:
                raise DataUnavailableError("本进程已确认东方财富股票接口不可用")
            raw = self._retry(
                f"下载 {symbol} ({adjustment or 'raw'})",
                lambda: ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=start.strftime("%Y%m%d"),
                    end_date=end.strftime("%Y%m%d"),
                    adjust=adjustment,
                    timeout=self.timeout,
                ),
            )
            result = normalize_daily_data(raw, symbol)
            result.attrs["source"] = "eastmoney"
            self.stock_primary_available = True
            return result
        except Exception as primary_error:
            self.stock_primary_available = False
            market_symbol = f"sh{symbol}" if symbol.startswith(("5", "6", "9")) else f"sz{symbol}"
            self.logger.warning("东方财富接口不可用，切换新浪备用源：%s", primary_error)
            raw = self._retry(
                f"备用下载 {symbol} ({adjustment or 'raw'})",
                lambda: ak.stock_zh_a_daily(
                    symbol=market_symbol,
                    start_date=start.strftime("%Y%m%d"),
                    end_date=end.strftime("%Y%m%d"),
                    adjust=adjustment,
                ),
            )
            result = normalize_daily_data(raw, symbol)
            result.attrs["source"] = "sina-fallback"
            return result

    def fetch_index_daily(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        try:
            if self.index_primary_available is False:
                raise DataUnavailableError("本进程已确认东方财富指数接口不可用")
            raw = self._retry(
                f"下载指数 {symbol}",
                lambda: ak.index_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=start.strftime("%Y%m%d"),
                    end_date=end.strftime("%Y%m%d"),
                ),
            )
            result = normalize_daily_data(raw, symbol)
            result.attrs["source"] = "eastmoney"
            self.index_primary_available = True
            return result
        except Exception as primary_error:
            self.index_primary_available = False
            self.logger.warning("东方财富指数接口不可用，切换新浪备用源：%s", primary_error)
            market_symbol = f"sh{symbol}" if symbol.startswith("0") else f"sz{symbol}"
            raw = self._retry(
                f"备用下载指数 {symbol}", lambda: ak.stock_zh_index_daily(symbol=market_symbol)
            )
            if "amount" not in raw:
                raw["amount"] = 0.0
            raw = raw[
                (pd.to_datetime(raw["date"]).dt.date >= start)
                & (pd.to_datetime(raw["date"]).dt.date <= end)
            ]
            result = normalize_daily_data(raw, symbol)
            result.attrs["source"] = "sina-fallback"
            return result


def atomic_write_parquet(data: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        data.to_parquet(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


class MarketDataRepository:
    def __init__(self, config: AppConfig, provider: DataProvider | None = None) -> None:
        self.config = config
        self.provider = provider or AkShareDataProvider(
            timeout=config.data.timeout_seconds, retries=config.data.retries
        )
        self.logger = logging.getLogger("aquant.repository")

    def path_for(self, symbol: str, adjustment: str = "qfq", benchmark: bool = False) -> Path:
        if benchmark:
            return self.config.storage.benchmark_dir / f"{symbol}.parquet"
        directory = (
            self.config.storage.adjusted_dir if adjustment == "qfq" else self.config.storage.raw_dir
        )
        return directory / f"{symbol}.parquet"

    def load(self, symbol: str, adjustment: str = "qfq", benchmark: bool = False) -> pd.DataFrame:
        path = self.path_for(symbol, adjustment, benchmark)
        if not path.exists():
            raise DataUnavailableError(f"本地行情不存在：{path}")
        return normalize_daily_data(pd.read_parquet(path), symbol)

    def _report(
        self, symbol: str, adjustment: str, data: pd.DataFrame, used_cache: bool = False
    ) -> DataQualityReport:
        warnings: list[str] = []
        gaps = data["date"].diff().dt.days.dropna()
        if (gaps > 15).any():
            warnings.append("存在超过 15 个自然日的行情间隔，可能包含长期停牌")
        if data.attrs.get("source") == "sina-fallback":
            warnings.append("东方财富不可用，本次使用新浪备用源")
        return DataQualityReport(
            symbol=symbol,
            adjustment=adjustment,
            rows=len(data),
            start_date=data["date"].min().date(),
            end_date=data["date"].max().date(),
            used_cache=used_cache,
            warnings=warnings,
        )

    def update_symbol(
        self,
        symbol: str,
        adjustment: str,
        start: date,
        end: date,
        *,
        benchmark: bool = False,
    ) -> DataQualityReport:
        path = self.path_for(symbol, adjustment, benchmark)
        existing: pd.DataFrame | None = None
        if path.exists():
            existing = self.load(symbol, adjustment, benchmark)
        fetch_start = start
        # 前复权因子会在除权日改变历史价格，因此每次完整刷新；原始行情只重叠更新尾部。
        if existing is not None and adjustment == "" and not benchmark:
            fetch_start = max(
                start,
                existing["date"].max().date() - timedelta(days=self.config.data.overlap_days),
            )
        try:
            fetched = (
                self.provider.fetch_index_daily(symbol, fetch_start, end)
                if benchmark
                else self.provider.fetch_daily(symbol, fetch_start, end, adjustment)
            )
            source = fetched.attrs.get("source")
            if existing is not None and fetch_start > start:
                combined = pd.concat([existing, fetched], ignore_index=True)
                fetched = normalize_daily_data(combined, symbol)
            if source:
                fetched.attrs["source"] = source
            atomic_write_parquet(fetched, path)
            return self._report(symbol, adjustment, fetched)
        except Exception as exc:
            if existing is None:
                raise
            report = self._report(symbol, adjustment, existing, used_cache=True)
            report.warnings.append(f"更新失败，继续使用缓存：{exc}")
            self.logger.error("%s 更新失败，使用缓存：%s", symbol, exc)
            return report

    def update_all(
        self,
        end: date | None = None,
        *,
        progress_callback: Callable[[int, int, DataQualityReport], None] | None = None,
    ) -> list[DataQualityReport]:
        self.config.ensure_directories()
        end = end or datetime.now().date()
        start = date.fromisoformat(self.config.market.start_date)
        reports: list[DataQualityReport] = []

        def safely_update(
            symbol: str, adjustment: str, *, benchmark: bool = False
        ) -> DataQualityReport:
            try:
                return self.update_symbol(symbol, adjustment, start, end, benchmark=benchmark)
            except Exception as exc:
                self.logger.error("%s 没有可用行情：%s", symbol, exc)
                return DataQualityReport(
                    symbol=symbol,
                    adjustment=adjustment,
                    rows=0,
                    start_date=None,
                    end_date=None,
                    used_cache=True,
                    warnings=[f"更新失败且无可用缓存：{exc}"],
                )

        work_items = [
            (symbol, adjustment, False)
            for symbol in self.config.market.watchlist
            for adjustment in ("", "qfq")
        ]
        work_items.append((self.config.market.benchmark, "", True))
        for completed, (symbol, adjustment, benchmark) in enumerate(work_items, start=1):
            report = safely_update(symbol, adjustment, benchmark=benchmark)
            reports.append(report)
            if progress_callback:
                progress_callback(completed, len(work_items), report)
        return reports

    def latest_dates(self, adjustment: str = "qfq") -> dict[str, date | None]:
        result: dict[str, date | None] = {}
        for symbol in self.config.market.watchlist:
            try:
                result[symbol] = self.load(symbol, adjustment)["date"].max().date()
            except (DataUnavailableError, DataValidationError):
                result[symbol] = None
        return result

    def stale_symbols(self, as_of: date | None = None) -> dict[str, str]:
        as_of = as_of or datetime.now().date()
        stale: dict[str, str] = {}
        adjusted = self.latest_dates("qfq")
        raw = self.latest_dates("")
        for symbol in self.config.market.watchlist:
            adjusted_latest = adjusted[symbol]
            raw_latest = raw[symbol]
            if adjusted_latest is None or raw_latest is None:
                missing = "前复权" if adjusted_latest is None else "未复权"
                stale[symbol] = f"缺少{missing}行情"
                continue
            oldest = min(adjusted_latest, raw_latest)
            if (as_of - oldest).days > self.config.market.max_stale_days:
                stale[symbol] = f"最旧行情为 {oldest.isoformat()}"
        return stale
