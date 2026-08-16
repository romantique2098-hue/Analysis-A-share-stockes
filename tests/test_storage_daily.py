from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest

from aquant.daily import PaperTradingService, run_daily
from aquant.data import MarketDataRepository, atomic_write_parquet
from aquant.models import Order, PortfolioSnapshot, Position, Side, Signal
from aquant.storage import StateStore


def seed_repository(app_config, histories) -> MarketDataRepository:
    repository = MarketDataRepository(app_config)
    for symbol, frame in histories.items():
        atomic_write_parquet(frame, repository.path_for(symbol, ""))
        atomic_write_parquet(frame, repository.path_for(symbol, "qfq"))
    return repository


def test_signal_storage_is_idempotent(app_config) -> None:
    store = StateStore(app_config.storage.database)
    signal = Signal("ma_cross", "000001", pd.Timestamp("2025-01-02").date(), 1, 0.2)
    assert store.add_signals([signal]) == 1
    assert store.add_signals([signal]) == 0


def test_daily_run_does_not_duplicate_orders(app_config, three_histories) -> None:
    repository = seed_repository(app_config, three_histories)
    store = StateStore(app_config.storage.database)
    service = PaperTradingService(app_config, repository, store)
    as_of = max(frame["date"].max().date() for frame in three_histories.values())
    first = service.run(as_of=as_of)
    second = service.run(as_of=as_of)
    assert first.status == "SUCCESS"
    assert len(first.new_orders) > 0
    assert second.status == "SUCCESS"
    assert len(second.new_orders) == 0
    assert len(store.table("orders")) == len(first.new_orders)


def test_existing_order_is_not_reported_as_new_when_weight_changed(
    app_config, three_histories
) -> None:
    repository = seed_repository(app_config, three_histories)
    store = StateStore(app_config.storage.database)
    service = PaperTradingService(app_config, repository, store)
    signal_date, signals, _, _ = service._latest_signals()
    first_orders = service._generate_orders(signal_date, signals)
    assert first_orders

    changed_weights = [
        Signal(
            strategy=signal.strategy,
            symbol=signal.symbol,
            signal_date=signal.signal_date,
            value=signal.value,
            target_weight=(
                min(signal.target_weight + 0.01, 1.0) if signal.target_weight > 0 else 0.0
            ),
        )
        for signal in signals
    ]
    assert service._generate_orders(signal_date, changed_weights) == []


def test_pending_orders_fill_on_next_bar(app_config, three_histories) -> None:
    repository = seed_repository(app_config, three_histories)
    store = StateStore(app_config.storage.database)
    service = PaperTradingService(app_config, repository, store)
    last_date = max(frame["date"].max().date() for frame in three_histories.values())
    first = service.run(as_of=last_date)
    assert first.new_orders
    for symbol, frame in three_histories.items():
        new_date = frame["date"].max() + pd.offsets.BDay(1)
        last = frame.iloc[-1].copy()
        last["date"] = new_date
        last["open"] = float(last["close"]) * 1.001
        last["high"] = max(float(last["open"]), float(last["close"])) * 1.01
        last["low"] = min(float(last["open"]), float(last["close"])) * 0.99
        extended = pd.concat([frame, pd.DataFrame([last])], ignore_index=True)
        atomic_write_parquet(extended, repository.path_for(symbol, ""))
        atomic_write_parquet(extended, repository.path_for(symbol, "qfq"))
    second = service.run(as_of=(last_date + timedelta(days=3)))
    assert second.fills
    assert not store.table("fills").empty
    assert all(fill.trade_date > first.run_date for fill in second.fills)


def test_minimum_holding_period_blocks_early_sale(app_config, three_histories) -> None:
    repository = seed_repository(app_config, three_histories)
    store = StateStore(app_config.storage.database)
    store.ensure_account(app_config.portfolio.initial_cash)
    trade_date = three_histories["000001"]["date"].iloc[-1].date()
    signal_date = three_histories["000001"]["date"].iloc[-2].date()
    store.save_position(Position("000001", 100, 20.0, trade_date))
    store.add_orders(
        [
            Order(
                order_id="minimum-holding",
                strategy="ma_cross",
                symbol="000001",
                signal_date=signal_date,
                side=Side.SELL,
                target_weight=0.0,
            )
        ]
    )
    service = PaperTradingService(app_config, repository, store)
    assert service._execute_pending() == []
    pending = store.pending_orders()
    assert len(pending) == 1
    assert "最短持有 3 个交易日" in pending[0].message


def test_daily_orders_respect_turnover_budget(app_config, three_histories) -> None:
    repository = seed_repository(app_config, three_histories)
    store = StateStore(app_config.storage.database)
    service = PaperTradingService(app_config, repository, store)
    signal_date = three_histories["000001"]["date"].max().date()
    signals = [
        Signal("ma_cross", symbol, signal_date, 1, 0.2)
        for symbol in app_config.market.watchlist
    ]

    orders = service._generate_orders(signal_date, signals)

    assert len(orders) == len(signals)
    assert sum(order.target_weight for order in orders) == pytest.approx(
        app_config.portfolio.max_daily_turnover
    )


def test_removed_watchlist_holding_gets_liquidation_order(app_config, three_histories) -> None:
    repository = seed_repository(app_config, three_histories)
    store = StateStore(app_config.storage.database)
    store.ensure_account(app_config.portfolio.initial_cash)
    store.save_position(Position("000858", 100, 20.0))
    app_config.market.watchlist = ["000001", "600519"]
    service = PaperTradingService(app_config, repository, store)
    signal_date, signals, _, _ = service._latest_signals()

    orders = service._generate_orders(signal_date, signals)

    liquidation = [order for order in orders if order.symbol == "000858"]
    assert len(liquidation) == 1
    assert liquidation[0].side == Side.SELL
    assert liquidation[0].target_weight == 0


def test_paper_drawdown_guard_is_idempotent(app_config, three_histories) -> None:
    repository = seed_repository(app_config, three_histories)
    store = StateStore(app_config.storage.database)
    service = PaperTradingService(app_config, repository, store)
    signal_date = max(frame["date"].max().date() for frame in three_histories.values())
    store.save_snapshot(
        PortfolioSnapshot(
            snapshot_date=signal_date - timedelta(days=1),
            cash=app_config.portfolio.initial_cash * 1.25,
            market_value=0,
            equity=app_config.portfolio.initial_cash * 1.25,
        )
    )

    _, first_signals, first_active, drawdown = service._latest_signals()
    remaining = store.get_state("paper_risk_off_remaining")
    _, second_signals, second_active, _ = service._latest_signals()

    assert drawdown == pytest.approx(-app_config.portfolio.max_portfolio_drawdown)
    assert first_active and second_active
    assert all(signal.target_weight == 0 for signal in first_signals + second_signals)
    assert store.get_state("paper_risk_off_remaining") == remaining


def test_run_daily_saves_reviewable_report(app_config, three_histories) -> None:
    seed_repository(app_config, three_histories)

    result = run_daily(app_config, update_data=False)

    assert result.report_dir is not None
    assert (result.report_dir / "summary.json").exists()
    daily_run = StateStore(app_config.storage.database).table("daily_runs", 1).iloc[0]
    assert daily_run["deployment_id"] == result.deployment_id
    assert daily_run["strategy_name"] == result.strategy_name
    for name in ["positions", "orders", "fills", "signals", "portfolio_snapshots"]:
        assert (result.report_dir / f"{name}.csv").exists()
        assert (result.report_dir / f"{name}.parquet").exists()
