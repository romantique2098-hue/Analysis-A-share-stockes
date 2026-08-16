from __future__ import annotations

import math

import pandas as pd

from aquant.backtest import BacktestEngine
from aquant.broker import limit_target_weights
from aquant.strategy import MovingAverageCrossStrategy


class OneDayPulseStrategy:
    name = "one_day_pulse"

    def generate_signals(self, history, parameters=None):
        result = history[["symbol", "date", "close"]].copy()
        first_date = result.groupby("symbol")["date"].transform("min")
        result["signal"] = result["date"].eq(first_date).astype(int)
        return result


class ExitThenReenterStrategy:
    name = "exit_then_reenter"

    def generate_signals(self, history, parameters=None):
        result = history[["symbol", "date", "close"]].copy()
        offset = result.groupby("symbol").cumcount()
        result["signal"] = offset.ne(1).astype(int)
        return result


def test_moving_average_needs_full_history(three_histories) -> None:
    data = three_histories["000001"]
    result = MovingAverageCrossStrategy(5, 20).generate_signals(data)
    assert result.loc[:18, "signal"].eq(0).all()
    assert result.loc[:18, "ma_long"].isna().all()


def test_moving_average_buffer_and_confirmation_reduce_whipsaw() -> None:
    dates = pd.bdate_range("2025-01-02", periods=80)
    close = pd.Series([100 + (index % 6 - 3) * 0.35 for index in range(80)])
    history = pd.DataFrame({"symbol": "000001", "date": dates, "close": close})
    baseline = MovingAverageCrossStrategy(3, 8).generate_signals(history)
    filtered = MovingAverageCrossStrategy(
        3,
        8,
        signal_buffer=0.005,
        confirmation_days=2,
    ).generate_signals(history)

    baseline_changes = baseline["signal"].diff().abs().fillna(0).sum()
    filtered_changes = filtered["signal"].diff().abs().fillna(0).sum()
    assert filtered_changes < baseline_changes


def test_backtest_executes_after_signal_and_uses_lots(app_config, three_histories) -> None:
    result = BacktestEngine(app_config).run(three_histories)
    buy_trades = result.trades[result.trades["side"] == "BUY"]
    assert not buy_trades.empty
    assert (buy_trades["quantity"] % app_config.portfolio.lot_size == 0).all()
    first_signal = result.signals[result.signals["signal"] == 1]["date"].min()
    assert pd.Timestamp(buy_trades["date"].min()) > pd.Timestamp(first_signal)


def test_costs_reduce_equity_and_metrics_are_complete(app_config, three_histories) -> None:
    result = BacktestEngine(app_config).run(
        three_histories,
        benchmark=three_histories["000001"],
    )
    assert result.equity_curve["net_curve"].iloc[-1] <= result.equity_curve["gross_curve"].iloc[-1]
    expected = {
        "annual_return",
        "annual_volatility",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "win_rate",
        "turnover",
        "total_costs",
        "benchmark_annual_return",
        "information_ratio",
        "tracking_error",
        "beta",
        "annual_alpha",
        "average_exposure",
        "annualized_turnover",
    }
    assert expected <= result.metrics.keys()
    assert result.metrics["total_costs"] > 0
    assert result.equity_curve["cash"].ge(0).all()
    assert result.equity_curve["cumulative_costs"].is_monotonic_increasing
    assert math.isclose(
        result.metrics["cost_drag"],
        result.metrics["total_costs"] / app_config.portfolio.initial_cash,
    )
    expected_gross = (
        result.equity_curve["equity"] + result.equity_curve["cumulative_costs"]
    ) / app_config.portfolio.initial_cash
    pd.testing.assert_series_equal(
        result.equity_curve["gross_curve"],
        expected_gross,
        check_names=False,
    )


def test_backtest_is_deterministic(app_config, three_histories) -> None:
    engine = BacktestEngine(app_config)
    first = engine.run(three_histories)
    second = engine.run(three_histories)
    assert first.run_id == second.run_id
    pd.testing.assert_frame_equal(first.equity_curve, second.equity_curve)
    pd.testing.assert_frame_equal(first.trades, second.trades)


def test_custom_strategy_identity_is_part_of_run_id(app_config, three_histories) -> None:
    first = BacktestEngine(app_config, OneDayPulseStrategy()).run(three_histories)
    second = BacktestEngine(app_config, ExitThenReenterStrategy()).run(three_histories)

    assert first.run_id != second.run_id
    assert first.parameters["strategy_implementation"]["name"] == "one_day_pulse"
    assert "OneDayPulseStrategy" in first.parameters["strategy_implementation"]["class"]


def test_same_day_high_low_do_not_control_open_execution(app_config, three_histories) -> None:
    histories = {symbol: frame.copy() for symbol, frame in three_histories.items()}
    for frame in histories.values():
        frame["high"] = frame["open"]
        frame["low"] = frame["open"]
        frame["volume"] = 1_000.0

    result = BacktestEngine(app_config).run(histories)

    assert not result.trades.empty


def test_zero_volume_bar_is_treated_as_suspended(app_config, three_histories) -> None:
    histories = {symbol: frame.copy() for symbol, frame in three_histories.items()}
    for frame in histories.values():
        frame["volume"] = 0.0

    result = BacktestEngine(app_config).run(histories)

    assert result.trades.empty


def test_new_signal_replaces_unfilled_stale_target(app_config, three_histories) -> None:
    app_config.portfolio.min_holding_days = 5
    app_config.portfolio.max_portfolio_drawdown = 0.99
    history = {"000001": three_histories["000001"].head(15)}

    result = BacktestEngine(app_config, ExitThenReenterStrategy()).run(history)

    assert not result.trades.empty
    assert not result.trades["side"].eq("SELL").any()


def test_start_date_keeps_indicator_warmup(app_config, three_histories) -> None:
    full = BacktestEngine(app_config).run(three_histories)
    start = full.signals["date"].iloc[30].date()
    sliced = BacktestEngine(app_config).run(three_histories, start=start)
    expected = full.signals[full.signals["date"].dt.date >= start].reset_index(drop=True)
    pd.testing.assert_frame_equal(sliced.signals.reset_index(drop=True), expected)


def test_run_id_fingerprints_pre_start_indicator_history(app_config, three_histories) -> None:
    start = three_histories["000001"]["date"].iloc[30].date()
    first = BacktestEngine(app_config).run(three_histories, start=start)
    revised = {symbol: frame.copy() for symbol, frame in three_histories.items()}
    revised["000001"].loc[10, "close"] *= 1.01

    second = BacktestEngine(app_config).run(revised, start=start)

    assert first.run_id != second.run_id


def test_target_limiter_applies_threshold_and_turnover_budget() -> None:
    opened = limit_target_weights(
        {},
        {"000001": 0.2, "000002": 0.2},
        minimum_change=0.02,
        max_turnover=0.2,
    )
    assert opened == {"000001": 0.1, "000002": 0.1}

    unchanged = limit_target_weights(
        {"000001": 0.20},
        {"000001": 0.19},
        minimum_change=0.02,
        max_turnover=0.2,
    )
    assert unchanged == {"000001": 0.20}

    exited = limit_target_weights(
        {"000001": 0.30},
        {"000001": 0.0},
        minimum_change=0.02,
        max_turnover=0.2,
    )
    assert exited == {"000001": 0.0}


def test_backtest_enforces_minimum_holding_period(app_config, three_histories) -> None:
    app_config.portfolio.min_holding_days = 3
    history = {"000001": three_histories["000001"].head(15)}

    result = BacktestEngine(app_config, OneDayPulseStrategy()).run(history)
    trades = result.trades.reset_index(drop=True)

    assert list(trades["side"]) == ["BUY", "SELL"]
    calendar = list(history["000001"]["date"])
    buy_index = calendar.index(trades.loc[0, "date"])
    sell_index = calendar.index(trades.loc[1, "date"])
    assert sell_index - buy_index >= app_config.portfolio.min_holding_days


def test_backtest_stages_entries_with_daily_turnover_cap(app_config, three_histories) -> None:
    app_config.portfolio.max_daily_turnover = 0.2
    result = BacktestEngine(app_config).run(three_histories)
    active_days = result.equity_curve[result.equity_curve["daily_turnover"] > 0]

    assert len(active_days) >= 2
    assert active_days["daily_turnover"].max() <= 0.205
