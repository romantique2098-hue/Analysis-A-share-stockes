from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252
CURVE_COLUMNS = (
    "net_curve",
    "gross_curve",
    "buy_hold_curve",
    "benchmark_curve",
    "relative_curve",
)


def calendar_year_analysis(curve: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """按自然年拆解收益、仓位、换手和成本，避免只看回测终点。"""
    if curve.empty or "date" not in curve:
        return pd.DataFrame()
    data = curve.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    columns = [column for column in CURVE_COLUMNS if column in data]
    daily_returns = data[columns].pct_change(fill_method=None)
    rows: list[dict[str, float | int]] = []
    for year, indices in data.groupby(data["date"].dt.year).groups.items():
        row: dict[str, float | int] = {"year": int(year)}
        year_returns = daily_returns.loc[indices]
        for column in columns:
            valid = year_returns[column].dropna()
            row[f"{column}_return"] = (
                float((1 + valid).prod() - 1) if not valid.empty else 0.0
            )
        if {"market_value", "equity"} <= set(data):
            equity = data.loc[indices, "equity"].replace(0, pd.NA)
            row["average_exposure"] = float((data.loc[indices, "market_value"] / equity).mean())
        if "daily_turnover" in data:
            row["turnover"] = float(data.loc[indices, "daily_turnover"].sum())
        rows.append(row)

    result = pd.DataFrame(rows)
    if trades.empty or "date" not in trades:
        result["trade_count"] = 0
        result["traded_notional"] = 0.0
        result["transaction_costs"] = 0.0
        return result
    executions = trades.copy()
    executions["year"] = pd.to_datetime(executions["date"]).dt.year
    cost_columns = [
        column for column in ["commission", "stamp_duty", "slippage"] if column in executions
    ]
    executions["transaction_costs"] = executions[cost_columns].sum(axis=1)
    grouped = (
        executions.groupby("year")
        .agg(
            trade_count=("symbol", "size"),
            traded_notional=("notional", "sum"),
            transaction_costs=("transaction_costs", "sum"),
        )
        .reset_index()
    )
    return result.merge(grouped, on="year", how="left").fillna(
        {"trade_count": 0, "traded_notional": 0.0, "transaction_costs": 0.0}
    )


def rolling_relative_analysis(curve: pd.DataFrame, window: int = TRADING_DAYS) -> pd.DataFrame:
    """计算只使用当日及过去数据的滚动相对表现。"""
    required = {"date", "net_curve", "benchmark_curve"}
    if curve.empty or not required <= set(curve):
        return pd.DataFrame()
    data = curve[["date", "net_curve", "benchmark_curve"]].copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    returns = data[["net_curve", "benchmark_curve"]].pct_change(fill_method=None)
    active = returns["net_curve"] - returns["benchmark_curve"]
    active_std = active.rolling(window, min_periods=window).std()
    active_mean = active.rolling(window, min_periods=window).mean()
    benchmark_variance = returns["benchmark_curve"].rolling(
        window, min_periods=window
    ).var()
    information_ratio = (
        active_mean.div(active_std.where(active_std.ne(0))) * math.sqrt(TRADING_DAYS)
    ).where(active_std.ne(0), 0.0)
    beta = (
        returns["net_curve"]
        .rolling(window, min_periods=window)
        .cov(returns["benchmark_curve"])
        .div(benchmark_variance.where(benchmark_variance.ne(0)))
    ).where(benchmark_variance.ne(0), 0.0)
    result = pd.DataFrame(
        {
            "date": data["date"],
            "strategy_return": data["net_curve"] / data["net_curve"].shift(window) - 1,
            "benchmark_return": data["benchmark_curve"]
            / data["benchmark_curve"].shift(window)
            - 1,
            "tracking_error": active_std * math.sqrt(TRADING_DAYS),
            "information_ratio": information_ratio,
            "beta": beta,
        }
    )
    result["excess_return"] = result["strategy_return"] - result["benchmark_return"]
    return result.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
