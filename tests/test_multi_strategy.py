from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aquant.backtest import BacktestEngine, load_and_compare_strategies
from aquant.cli import build_parser
from aquant.config import StrategyConfig
from aquant.data import MarketDataRepository, atomic_write_parquet
from aquant.strategy import (
    STRATEGY_NAMES,
    CrossSectionalMomentumStrategy,
    EnsembleStrategy,
    RSIMeanReversionStrategy,
)


def linear_history(symbol: str, start: float, end: float, periods: int = 100) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=periods)
    close = np.linspace(start, end, periods)
    open_price = close * 0.999
    return pd.DataFrame(
        {
            "symbol": symbol,
            "date": dates,
            "open": open_price,
            "close": close,
            "high": np.maximum(open_price, close) * 1.01,
            "low": np.minimum(open_price, close) * 0.99,
            "volume": 1_000_000.0,
            "amount": close * 1_000_000.0,
        }
    )


@pytest.fixture
def ranked_histories() -> dict[str, pd.DataFrame]:
    return {
        "000001": linear_history("000001", 10, 35),
        "600519": linear_history("600519", 10, 20),
        "000858": linear_history("000858", 10, 6),
    }


def combined(histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(histories.values(), ignore_index=True)


def test_cross_sectional_momentum_selects_only_strongest(ranked_histories) -> None:
    strategy = CrossSectionalMomentumStrategy(lookback=20, skip_recent=0, top_n=1)
    result = strategy.generate_signals(combined(ranked_histories))
    latest = result[result["date"] == result["date"].max()].set_index("symbol")
    assert latest.loc["000001", "signal"] == 1
    assert latest.loc["600519", "signal"] == 0
    assert latest.loc["000858", "signal"] == 0
    assert latest.loc["000001", "momentum_rank"] == 1


def test_momentum_rank_buffer_keeps_existing_positive_holder() -> None:
    dates = pd.bdate_range("2025-01-02", periods=40)
    first = 10 * np.power(1.01, np.arange(40))
    second_returns = np.r_[np.full(20, 1.005), np.full(20, 1.02)]
    second = 10 * np.cumprod(second_returns)
    history = pd.concat(
        [
            pd.DataFrame({"symbol": "000001", "date": dates, "close": first}),
            pd.DataFrame({"symbol": "000002", "date": dates, "close": second}),
        ],
        ignore_index=True,
    )
    baseline = CrossSectionalMomentumStrategy(
        lookback=5,
        skip_recent=0,
        top_n=1,
        periods=(5, 5, 5),
        rank_buffer=0,
    ).generate_signals(history)
    buffered = CrossSectionalMomentumStrategy(
        lookback=5,
        skip_recent=0,
        top_n=1,
        periods=(5, 5, 5),
        rank_buffer=1,
    ).generate_signals(history)
    switched_dates = baseline.loc[
        (baseline["symbol"] == "000002") & (baseline["signal"] == 1), "date"
    ]
    comparison_date = switched_dates.iloc[0]
    buffered_day = buffered[buffered["date"] == comparison_date].set_index("symbol")

    assert buffered_day.loc["000001", "momentum_rank"] == 2
    assert buffered_day.loc["000001", "momentum_score"] > 0
    assert buffered_day.loc["000001", "signal"] == 1
    assert buffered_day.loc["000002", "signal"] == 0


def test_rsi_reversion_enters_on_fall_and_exits_on_recovery() -> None:
    data = linear_history("000001", 20, 8, periods=40)
    recovery = linear_history("000001", 8, 25, periods=40)
    recovery["date"] = pd.bdate_range(data["date"].max() + pd.Timedelta(days=1), periods=40)
    history = pd.concat([data, recovery], ignore_index=True)
    result = RSIMeanReversionStrategy(window=5, entry=30, exit=55).generate_signals(history)
    assert result["signal"].eq(1).any()
    assert result["signal"].iloc[-1] == 0
    first_entry = result.index[result["signal"] == 1][0]
    assert result.loc[first_entry : first_entry + 2, "signal"].eq(1).all()


def test_future_prices_do_not_change_past_momentum_signals(ranked_histories) -> None:
    history = combined(ranked_histories)
    strategy = CrossSectionalMomentumStrategy(20, 2, 1)
    baseline = strategy.generate_signals(history)
    cutoff = baseline["date"].drop_duplicates().iloc[70]
    changed = history.copy()
    changed.loc[changed["date"] > cutoff, "close"] *= 10
    recalculated = strategy.generate_signals(changed)
    columns = ["symbol", "date", "signal", "momentum_rank"]
    pd.testing.assert_frame_equal(
        baseline.loc[baseline["date"] <= cutoff, columns].reset_index(drop=True),
        recalculated.loc[recalculated["date"] <= cutoff, columns].reset_index(drop=True),
    )


def ensemble_config(name: str) -> StrategyConfig:
    return StrategyConfig(
        name=name,
        short_window=3,
        long_window=8,
        momentum_lookback=15,
        momentum_skip_recent=2,
        momentum_top_n=1,
        rsi_window=5,
        rsi_entry=35,
        rsi_exit=55,
        ensemble_volatility_window=10,
    )


def test_equal_ensemble_uses_two_of_three_vote(ranked_histories) -> None:
    result = EnsembleStrategy(
        ensemble_config("ensemble_equal"), risk_weighted=False
    ).generate_signals(combined(ranked_histories))
    weight_columns = [
        "trend_signal_weight",
        "momentum_signal_weight",
        "reversion_signal_weight",
    ]
    assert np.allclose(result[weight_columns], 1 / 3)
    expected = result["ensemble_score"].ge(0.5).astype("int8")
    pd.testing.assert_series_equal(result["signal"], expected, check_names=False)


def test_risk_ensemble_weights_are_valid_and_causal(ranked_histories) -> None:
    strategy = EnsembleStrategy(ensemble_config("ensemble_risk"), risk_weighted=True)
    history = combined(ranked_histories)
    result = strategy.generate_signals(history)
    weights = result[
        [
            "trend_signal_weight",
            "momentum_signal_weight",
            "reversion_signal_weight",
        ]
    ]
    assert weights.notna().all().all()
    assert weights.ge(0).all().all()
    assert np.allclose(weights.sum(axis=1), 1.0)
    cutoff = result["date"].drop_duplicates().iloc[70]
    changed = history.copy()
    changed.loc[changed["date"] > cutoff, "close"] *= 4
    recalculated = strategy.generate_signals(changed)
    causal_columns = ["symbol", "date", "ensemble_score", "signal"]
    pd.testing.assert_frame_equal(
        result.loc[result["date"] <= cutoff, causal_columns].reset_index(drop=True),
        recalculated.loc[recalculated["date"] <= cutoff, causal_columns].reset_index(drop=True),
    )


def test_all_strategies_run_through_backtest(app_config, ranked_histories) -> None:
    run_ids: set[str] = set()
    for name in STRATEGY_NAMES:
        run_config = app_config.model_copy(deep=True)
        run_config.strategy = ensemble_config(name)
        result = BacktestEngine(run_config).run(ranked_histories)
        assert not result.equity_curve.empty
        assert result.parameters["strategy"]["name"] == name
        run_ids.add(result.run_id)
    assert len(run_ids) == len(STRATEGY_NAMES)


def test_strategy_comparison_and_cli(app_config, ranked_histories) -> None:
    repository = MarketDataRepository(app_config)
    for symbol, frame in ranked_histories.items():
        atomic_write_parquet(frame, repository.path_for(symbol, "qfq"))
    benchmark = linear_history("000300", 100, 120)
    atomic_write_parquet(benchmark, repository.path_for("000300", "", benchmark=True))
    app_config.strategy = ensemble_config("ma_cross")
    metrics, curves, results = load_and_compare_strategies(app_config)
    assert set(metrics["strategy"]) == set(STRATEGY_NAMES)
    assert set(curves.columns) == {"date", *STRATEGY_NAMES}
    assert set(results) == set(STRATEGY_NAMES)
    parsed = build_parser().parse_args(["backtest", "--strategy", "momentum"])
    assert parsed.strategy == "momentum"
    assert build_parser().parse_args(["backtest", "--compare-all"]).compare_all
    assert build_parser().parse_args(["backtest", "--ma-robustness"]).ma_robustness
    with pytest.raises(SystemExit):
        build_parser().parse_args(["backtest", "--compare-all", "--ma-robustness"])
    assert build_parser().parse_args(["universe", "status"]).universe_command == "status"
    assert build_parser().parse_args(["jobs", "--limit", "5"]).limit == 5
