from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aquant.config import AppConfig, MarketConfig, StorageConfig


def make_prices(
    symbol: str,
    periods: int = 80,
    start: str = "2025-01-02",
    pattern: str = "trend",
) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=periods)
    if pattern == "trend":
        close = np.concatenate(
            [np.linspace(20, 16, periods // 3), np.linspace(16, 30, periods - periods // 3)]
        )
    elif pattern == "fall":
        close = np.linspace(30, 10, periods)
    else:
        close = np.full(periods, 20.0)
    open_price = close * 0.998
    return pd.DataFrame(
        {
            "symbol": symbol,
            "date": dates,
            "open": open_price,
            "close": close,
            "high": np.maximum(open_price, close) * 1.01,
            "low": np.minimum(open_price, close) * 0.99,
            "volume": np.full(periods, 1_000_000.0),
            "amount": close * 1_000_000.0,
        }
    )


@pytest.fixture
def app_config(tmp_path: Path) -> AppConfig:
    config = AppConfig(
        project_root=tmp_path,
        config_path=tmp_path / "config.yaml",
        market=MarketConfig(
            watchlist=["000001", "600519", "000858"],
            benchmark="000300",
            start_date="2025-01-01",
            max_stale_days=5000,
        ),
        storage=StorageConfig(),
    ).resolve_paths()
    config.ensure_directories()
    return config


@pytest.fixture
def three_histories() -> dict[str, pd.DataFrame]:
    return {
        "000001": make_prices("000001", pattern="trend"),
        "600519": make_prices("600519", pattern="trend"),
        "000858": make_prices("000858", pattern="fall"),
    }
