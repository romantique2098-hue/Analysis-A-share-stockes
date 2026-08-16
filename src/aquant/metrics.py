from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _finite_or_zero(value: float) -> float:
    numeric = float(value)
    return numeric if math.isfinite(numeric) else 0.0


def calculate_metrics(
    curve: pd.DataFrame,
    *,
    risk_free_rate: float = 0.02,
    total_costs: float = 0.0,
    turnover_notional: float = 0.0,
    trade_count: int = 0,
) -> dict[str, float]:
    if curve.empty or "net_curve" not in curve:
        raise ValueError("净值序列为空")
    values = curve["net_curve"].astype(float).dropna()
    if values.empty:
        raise ValueError("净值序列没有有效值")
    returns = values.pct_change(fill_method=None).dropna()
    total_return = float(values.iloc[-1] / values.iloc[0] - 1)
    start = pd.Timestamp(curve["date"].iloc[0])
    end = pd.Timestamp(curve["date"].iloc[-1])
    elapsed_days = max((end - start).days, 1)
    elapsed_years = elapsed_days / 365.25
    annual_return = float((1 + total_return) ** (365.25 / elapsed_days) - 1)
    daily_std = _finite_or_zero(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    volatility = daily_std * math.sqrt(TRADING_DAYS)
    risk_free_daily = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    std = daily_std
    sharpe = (
        float((returns.mean() - risk_free_daily) / std * math.sqrt(TRADING_DAYS)) if std else 0.0
    )
    downside = returns.where(returns < 0, 0.0)
    downside_std = (
        _finite_or_zero(np.sqrt(np.mean(np.square(downside)))) if len(downside) else 0.0
    )
    sortino = (
        float((returns.mean() - risk_free_daily) / downside_std * math.sqrt(TRADING_DAYS))
        if downside_std
        else 0.0
    )
    drawdown = values / values.cummax() - 1
    max_drawdown = float(drawdown.min())
    calmar = float(annual_return / abs(max_drawdown)) if max_drawdown else 0.0
    win_rate = float((returns > 0).mean()) if len(returns) else 0.0
    average_equity = float(curve["equity"].mean()) if "equity" in curve else float(values.mean())
    turnover = float(turnover_notional / average_equity) if average_equity else 0.0
    metrics = {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "win_rate": win_rate,
        "turnover": turnover,
        "annualized_turnover": turnover / elapsed_years,
        "total_costs": float(total_costs),
        "trade_count": float(trade_count),
    }
    initial_equity = float(curve["equity"].iloc[0]) if "equity" in curve else 1.0
    metrics["cost_ratio"] = float(total_costs / initial_equity) if initial_equity else 0.0
    if {"market_value", "equity"} <= set(curve):
        exposure = curve["market_value"] / curve["equity"].replace(0, np.nan)
        metrics["average_exposure"] = _finite_or_zero(exposure.mean())
        metrics["maximum_exposure"] = _finite_or_zero(exposure.max())
    if {"cash", "equity"} <= set(curve):
        final_equity = float(curve["equity"].iloc[-1])
        metrics["final_cash_weight"] = (
            _finite_or_zero(curve["cash"].iloc[-1] / final_equity) if final_equity else 0.0
        )
    if "gross_curve" in curve and curve["gross_curve"].notna().any():
        gross = curve["gross_curve"].dropna().astype(float)
        gross_total_return = float(gross.iloc[-1] / gross.iloc[0] - 1)
        metrics["gross_total_return"] = gross_total_return
        metrics["cost_drag"] = gross_total_return - total_return
    if "buy_hold_curve" in curve and curve["buy_hold_curve"].notna().any():
        buy_hold = curve["buy_hold_curve"].dropna().astype(float)
        buy_hold_total_return = float(buy_hold.iloc[-1] / buy_hold.iloc[0] - 1)
        metrics["buy_hold_total_return"] = buy_hold_total_return
        metrics["timing_vs_buy_hold"] = total_return - buy_hold_total_return
    if "benchmark_curve" in curve and curve["benchmark_curve"].notna().any():
        aligned = pd.DataFrame(
            {
                "strategy": curve["net_curve"].astype(float),
                "benchmark": curve["benchmark_curve"].astype(float),
                "date": pd.to_datetime(curve["date"]),
            }
        ).dropna()
        aligned_strategy = aligned["strategy"]
        benchmark = aligned["benchmark"]
        aligned_strategy_total_return = float(
            aligned_strategy.iloc[-1] / aligned_strategy.iloc[0] - 1
        )
        benchmark_total_return = float(benchmark.iloc[-1] / benchmark.iloc[0] - 1)
        benchmark_days = max((aligned["date"].iloc[-1] - aligned["date"].iloc[0]).days, 1)
        aligned_strategy_annual_return = float(
            (1 + aligned_strategy_total_return) ** (365.25 / benchmark_days) - 1
        )
        benchmark_annual_return = float(
            (1 + benchmark_total_return) ** (365.25 / benchmark_days) - 1
        )
        metrics["benchmark_total_return"] = benchmark_total_return
        metrics["benchmark_annual_return"] = benchmark_annual_return
        benchmark_returns = benchmark.pct_change(fill_method=None).dropna()
        benchmark_daily_std = (
            _finite_or_zero(benchmark_returns.std(ddof=1))
            if len(benchmark_returns) > 1
            else 0.0
        )
        metrics["benchmark_annual_volatility"] = benchmark_daily_std * math.sqrt(
            TRADING_DAYS
        )
        metrics["benchmark_max_drawdown"] = float((benchmark / benchmark.cummax() - 1).min())
        metrics["excess_total_return"] = (
            aligned_strategy_total_return - metrics["benchmark_total_return"]
        )
        metrics["relative_total_return"] = float(
            (aligned["strategy"].iloc[-1] / aligned["strategy"].iloc[0])
            / (benchmark.iloc[-1] / benchmark.iloc[0])
            - 1
        )
        metrics["excess_annual_return"] = (
            aligned_strategy_annual_return - benchmark_annual_return
        )

        paired_returns = aligned[["strategy", "benchmark"]].pct_change(
            fill_method=None
        ).dropna()
        active_returns = paired_returns["strategy"] - paired_returns["benchmark"]
        active_std = (
            _finite_or_zero(active_returns.std(ddof=1)) if len(active_returns) > 1 else 0.0
        )
        metrics["tracking_error"] = active_std * math.sqrt(TRADING_DAYS)
        metrics["information_ratio"] = (
            float(active_returns.mean() / active_std * math.sqrt(TRADING_DAYS))
            if active_std
            else 0.0
        )
        benchmark_variance = (
            _finite_or_zero(paired_returns["benchmark"].var(ddof=1))
            if len(paired_returns) > 1
            else 0.0
        )
        beta = (
            float(
                paired_returns["strategy"].cov(paired_returns["benchmark"])
                / benchmark_variance
            )
            if benchmark_variance
            else 0.0
        )
        metrics["beta"] = beta
        metrics["annual_alpha"] = _finite_or_zero(
            (
                paired_returns["strategy"].mean()
                - risk_free_daily
                - beta * (paired_returns["benchmark"].mean() - risk_free_daily)
            )
            * TRADING_DAYS
        )
        strategy_std = (
            _finite_or_zero(paired_returns["strategy"].std(ddof=1))
            if len(paired_returns) > 1
            else 0.0
        )
        benchmark_std = benchmark_daily_std
        metrics["benchmark_correlation"] = (
            float(paired_returns["strategy"].corr(paired_returns["benchmark"]))
            if strategy_std and benchmark_std
            else 0.0
        )
    return metrics
