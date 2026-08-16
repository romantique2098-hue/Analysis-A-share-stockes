from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from conftest import make_prices

from aquant.config import (
    MarketConfig,
    StrategyConfig,
    extend_watchlist,
    load_config,
    save_config,
)
from aquant.data import (
    AkShareDataProvider,
    DataUnavailableError,
    DataValidationError,
    MarketDataRepository,
    atomic_write_parquet,
    normalize_daily_data,
)


def test_config_rejects_invalid_strategy_windows() -> None:
    with pytest.raises(ValueError, match="短均线"):
        StrategyConfig(short_window=20, long_window=5)
    with pytest.raises(ValueError, match="动量跳过"):
        StrategyConfig(momentum_lookback=20, momentum_skip_recent=20)
    with pytest.raises(ValueError, match="RSI 入场"):
        StrategyConfig(rsi_entry=60, rsi_exit=55)


def test_config_normalizes_and_deduplicates_symbols() -> None:
    market = MarketConfig(watchlist=["1", "000001", "600519"])
    assert market.watchlist == ["000001", "600519"]


def test_extend_watchlist_deduplicates_and_persists(app_config) -> None:
    added = extend_watchlist(app_config, ["600519", "300750", "1", "300750"])

    assert added == ["300750"]
    assert app_config.market.watchlist == ["000001", "600519", "000858", "300750"]
    assert extend_watchlist(app_config, ["300750", "600519"]) == []

    save_config(app_config)
    reloaded = load_config(app_config.config_path)
    assert reloaded.market.watchlist == app_config.market.watchlist


def test_normalize_deduplicates_and_sorts() -> None:
    data = make_prices("000001", periods=5)
    scrambled = pd.concat([data.iloc[::-1], data.iloc[[2]]], ignore_index=True)
    result = normalize_daily_data(scrambled, "000001")
    assert len(result) == 5
    assert result["date"].is_monotonic_increasing


def test_normalize_rejects_bad_ohlc() -> None:
    data = make_prices("000001", periods=5)
    data.loc[2, "low"] = data.loc[2, "high"] + 1
    with pytest.raises(DataValidationError, match="异常 OHLC"):
        normalize_daily_data(data, "000001")


def test_atomic_write_keeps_readable_file(tmp_path: Path) -> None:
    path = tmp_path / "bars.parquet"
    data = make_prices("000001", periods=5)
    atomic_write_parquet(data, path)
    pd.testing.assert_frame_equal(pd.read_parquet(path), data)
    assert not list(tmp_path.glob("*.tmp"))


def test_update_all_reports_missing_cache_instead_of_crashing(app_config) -> None:
    class FailingProvider:
        def fetch_daily(self, symbol, start, end, adjustment=""):
            raise DataUnavailableError("offline")

        def fetch_index_daily(self, symbol, start, end):
            raise DataUnavailableError("offline")

    reports = MarketDataRepository(app_config, FailingProvider()).update_all()
    assert len(reports) == 7
    assert all(report.used_cache and report.rows == 0 for report in reports)


def test_freshness_requires_raw_and_adjusted_files(app_config) -> None:
    repository = MarketDataRepository(app_config)
    for symbol in app_config.market.watchlist:
        atomic_write_parquet(make_prices(symbol, periods=5), repository.path_for(symbol, "qfq"))
    stale = repository.stale_symbols(as_of=pd.Timestamp("2025-01-10").date())
    assert set(stale) == set(app_config.market.watchlist)
    assert all("未复权" in message for message in stale.values())


def test_provider_circuit_breaker_skips_failed_primary(monkeypatch) -> None:
    calls = {"primary": 0, "fallback": 0}

    def primary(**kwargs):
        calls["primary"] += 1
        raise RuntimeError("primary offline")

    def fallback(**kwargs):
        calls["fallback"] += 1
        symbol = kwargs["symbol"][-6:]
        return make_prices(symbol, periods=5)

    monkeypatch.setattr("aquant.data.ak.stock_zh_a_hist", primary)
    monkeypatch.setattr("aquant.data.ak.stock_zh_a_daily", fallback)
    provider = AkShareDataProvider(retries=1)
    start = pd.Timestamp("2025-01-01").date()
    end = pd.Timestamp("2025-01-31").date()

    provider.fetch_daily("000001", start, end)
    provider.fetch_daily("000002", start, end)

    assert calls == {"primary": 1, "fallback": 2}
