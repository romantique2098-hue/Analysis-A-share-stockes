from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from aquant.config import load_config
from aquant.data import AkShareDataProvider, atomic_write_parquet, normalize_daily_data
from aquant.strategy import MovingAverageCrossStrategy


def download_daily_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    config = load_config()
    provider = AkShareDataProvider(config.data.timeout_seconds, config.data.retries)
    data = provider.fetch_daily(
        symbol,
        pd.Timestamp(start_date).date(),
        pd.Timestamp(end_date).date(),
        adjustment="",
    )
    atomic_write_parquet(data, config.storage.raw_dir / f"{symbol}.parquet")
    print(f"共得到 {len(data)} 行；保存位置：{config.storage.raw_dir / f'{symbol}.parquet'}")
    return data


def build_features(symbol: str) -> pd.DataFrame:
    config = load_config()
    adjusted = config.storage.adjusted_dir / f"{symbol}.parquet"
    raw = config.storage.raw_dir / f"{symbol}.parquet"
    input_path = adjusted if adjusted.exists() else raw
    data = normalize_daily_data(pd.read_parquet(input_path), symbol)
    strategy = MovingAverageCrossStrategy(config.strategy.short_window, config.strategy.long_window)
    features = strategy.generate_signals(data).rename(
        columns={"ma_short": "ma5", "ma_long": "ma20"}
    )
    merged = data.merge(features.drop(columns="close"), on=["symbol", "date"], how="left")
    merged["position"] = merged["signal"].shift(1, fill_value=0).astype("int8")
    merged["trade"] = merged["position"].diff().fillna(0).astype("int8")
    merged["buy_trade"] = (merged["trade"] == 1).astype("int8")
    merged["sell_trade"] = (merged["trade"] == -1).astype("int8")
    merged["daily_return"] = merged["close"].pct_change(fill_method=None).fillna(0)
    merged["strategy_return"] = merged["position"] * merged["daily_return"]
    merged["buy_hold_curve"] = (1 + merged["daily_return"]).cumprod()
    merged["strategy_curve"] = (1 + merged["strategy_return"]).cumprod()
    merged["net_strategy_curve"] = merged["strategy_curve"]
    output = config.storage.features_dir / f"{symbol}.parquet"
    atomic_write_parquet(merged, output)
    print(f"特征数据已保存到：{output}")
    return merged


def plot_backtest(symbol: str, show: bool = True) -> Path:
    config = load_config()
    input_path = config.storage.features_dir / f"{symbol}.parquet"
    data = build_features(symbol) if not input_path.exists() else pd.read_parquet(input_path)
    figure, axis = plt.subplots(figsize=(12, 6))
    for column, label in [
        ("buy_hold_curve", "Buy and Hold"),
        ("strategy_curve", "Strategy Before Costs"),
        ("net_strategy_curve", "Strategy After Costs"),
    ]:
        if column in data:
            axis.plot(data["date"], data[column], label=label, linewidth=2)
    axis.axhline(1, color="gray", linestyle="--", linewidth=1)
    axis.set(title=f"{symbol} Backtest Equity Curves", xlabel="Date", ylabel="Portfolio Value")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.autofmt_xdate()
    figure.tight_layout()
    output = config.storage.output_dir / f"{symbol}_backtest.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)
    print(f"回测图已经保存到：{output}")
    return output
