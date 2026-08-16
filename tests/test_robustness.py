from __future__ import annotations

from aquant.robustness import analyze_ma_robustness, save_ma_robustness

CANDIDATE = {
    "short_window": 5,
    "long_window": 20,
    "ma_signal_buffer": 0.0,
    "ma_confirmation_days": 1,
}


def test_ma_robustness_uses_chronological_folds(app_config, three_histories) -> None:
    candidates = (
        CANDIDATE,
        {
            "short_window": 5,
            "long_window": 20,
            "ma_signal_buffer": 0.005,
            "ma_confirmation_days": 2,
        },
    )
    summary, details = analyze_ma_robustness(
        app_config,
        three_histories,
        three_histories["000001"],
        candidates=candidates,
        fold_count=2,
    )

    assert len(summary) == 2
    assert len(details) == 4
    assert details.groupby("fold")["start"].nunique().eq(1).all()
    fold_lengths = details[["fold", "trading_days"]].drop_duplicates()["trading_days"]
    assert fold_lengths.max() - fold_lengths.min() <= 1
    assert {
        "positive_excess_folds",
        "worst_excess_return",
        "lower_quartile_excess_annual_return",
        "worst_excess_annual_return",
    } <= set(summary)
    assert details["backtest_run_id"].notna().all()

    output = save_ma_robustness(summary, details, app_config)
    assert (output / "summary.csv").exists()
    assert (output / "folds.csv").exists()
    assert (output / "manifest.json").exists()


def test_ma_robustness_uses_common_available_period(app_config, three_histories) -> None:
    histories = {symbol: frame.copy() for symbol, frame in three_histories.items()}
    histories["600519"] = histories["600519"].iloc[20:].reset_index(drop=True)
    common_start = histories["600519"]["date"].min()

    _, details = analyze_ma_robustness(
        app_config,
        histories,
        three_histories["000001"],
        candidates=(CANDIDATE,),
        fold_count=2,
    )

    assert details["start"].min() >= common_start.date()
