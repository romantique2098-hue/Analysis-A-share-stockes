from __future__ import annotations

import math

import pandas as pd

from aquant.metrics import calculate_metrics
from aquant.research import calendar_year_analysis, rolling_relative_analysis


def test_calendar_year_analysis_includes_year_boundary_return_and_costs() -> None:
    curve = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-12-30", "2024-12-31", "2025-01-02", "2025-01-03"]
            ),
            "net_curve": [1.0, 1.1, 1.21, 1.089],
            "gross_curve": [1.0, 1.1, 1.22, 1.10],
            "benchmark_curve": [1.0, 1.05, 1.05, 1.05],
            "equity": [100_000, 110_000, 121_000, 108_900],
            "market_value": [50_000, 55_000, 60_500, 54_450],
            "daily_turnover": [0.0, 0.1, 0.2, 0.0],
        }
    )
    trades = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-12-31", "2025-01-02"]),
            "symbol": ["000001", "000001"],
            "notional": [10_000.0, 20_000.0],
            "commission": [5.0, 5.0],
            "stamp_duty": [0.0, 10.0],
            "slippage": [2.0, 4.0],
        }
    )

    annual = calendar_year_analysis(
        curve.sample(frac=1, random_state=7),
        trades,
    ).set_index("year")

    assert math.isclose(annual.loc[2024, "net_curve_return"], 0.10)
    assert math.isclose(annual.loc[2025, "net_curve_return"], -0.01)
    assert annual.loc[2025, "trade_count"] == 1
    assert math.isclose(annual.loc[2025, "transaction_costs"], 19.0)


def test_rolling_relative_analysis_handles_identical_curves_without_infinity() -> None:
    dates = pd.bdate_range("2025-01-02", periods=10)
    values = pd.Series([1.0, 1.01, 1.00, 1.03, 1.02, 1.04, 1.05, 1.04, 1.06, 1.07])
    curve = pd.DataFrame(
        {"date": dates, "net_curve": values, "benchmark_curve": values}
    )

    rolling = rolling_relative_analysis(
        curve.sample(frac=1, random_state=11),
        window=3,
    )

    assert not rolling.empty
    assert rolling["information_ratio"].eq(0.0).all()
    assert rolling["tracking_error"].eq(0.0).all()
    assert rolling.select_dtypes("number").map(math.isfinite).all().all()


def test_metrics_are_finite_for_single_observation() -> None:
    curve = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02"]),
            "net_curve": [1.0],
            "benchmark_curve": [1.0],
            "equity": [100_000.0],
            "cash": [100_000.0],
            "market_value": [0.0],
        }
    )

    metrics = calculate_metrics(curve)

    assert metrics["annual_volatility"] == 0.0
    assert metrics["benchmark_annual_volatility"] == 0.0
    assert all(math.isfinite(value) for value in metrics.values())


def test_excess_metrics_use_same_aligned_period_as_benchmark() -> None:
    curve = pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-02", periods=3),
            "net_curve": [1.0, 2.0, 2.2],
            "benchmark_curve": [float("nan"), 1.0, 1.1],
            "equity": [100_000.0, 200_000.0, 220_000.0],
        }
    )

    metrics = calculate_metrics(curve)

    assert math.isclose(metrics["excess_total_return"], 0.0, abs_tol=1e-12)
    assert math.isclose(metrics["relative_total_return"], 0.0, abs_tol=1e-12)
    assert math.isclose(metrics["excess_annual_return"], 0.0, abs_tol=1e-12)
