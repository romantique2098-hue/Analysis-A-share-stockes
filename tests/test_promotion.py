from __future__ import annotations

import pytest

from aquant.backtest import BacktestEngine, save_backtest_result
from aquant.daily import PaperTradingService
from aquant.data import MarketDataRepository, atomic_write_parquet
from aquant.models import Order, OrderStatus, Side
from aquant.promotion import PaperStrategyService
from aquant.storage import StateStore


def save_candidate(app_config, histories, name: str):
    run_config = app_config.model_copy(deep=True)
    run_config.strategy.name = name
    run_config.portfolio.initial_cash = 250_000.0
    result = BacktestEngine(run_config).run(histories)
    save_backtest_result(result, run_config)
    return result


def test_backtest_strategy_is_explicitly_promoted_and_used_by_paper(
    app_config, three_histories
) -> None:
    store = StateStore(app_config.storage.database)
    service = PaperStrategyService(app_config, store)
    initial = service.current()
    candidate = save_candidate(app_config, three_histories, "rsi_reversion")

    promoted = service.promote_backtest(candidate.run_id, "样本外复核通过")

    assert promoted["strategy_name"] == "rsi_reversion"
    assert promoted["source_backtest_run_id"] == candidate.run_id
    assert promoted["note"] == "样本外复核通过"
    history = service.history()
    assert set(history["status"]) == {"ACTIVE", "RETIRED"}
    assert history.loc[
        history["deployment_id"].eq(initial["deployment_id"]), "status"
    ].iloc[0] == "RETIRED"

    paper = PaperTradingService(app_config, store=store)
    assert paper.config.strategy.name == "rsi_reversion"
    assert paper.config.portfolio.initial_cash == app_config.portfolio.initial_cash
    assert paper.deployment["deployment_id"] == promoted["deployment_id"]

    with pytest.raises(RuntimeError, match="已经是当前模拟盘策略的来源"):
        service.promote_backtest(candidate.run_id)


def test_bootstrap_can_be_linked_to_same_parameter_backtest(
    app_config, three_histories
) -> None:
    store = StateStore(app_config.storage.database)
    service = PaperStrategyService(app_config, store)
    bootstrap = service.current()
    candidate = save_candidate(app_config, three_histories, "ma_cross")

    promoted = service.promote_backtest(candidate.run_id)

    assert promoted["status"] == "ACTIVE"
    assert promoted["source_backtest_run_id"] == candidate.run_id
    assert promoted["deployment_id"] != bootstrap["deployment_id"]


def test_pending_orders_schedule_promotion_without_generating_more_old_orders(
    app_config, three_histories
) -> None:
    store = StateStore(app_config.storage.database)
    service = PaperStrategyService(app_config, store)
    service.current()
    candidate = save_candidate(app_config, three_histories, "trend_breakout")
    repository = MarketDataRepository(app_config)
    for symbol, frame in three_histories.items():
        atomic_write_parquet(frame, repository.path_for(symbol, ""))
        atomic_write_parquet(frame, repository.path_for(symbol, "qfq"))
    signal_date = max(frame["date"].max().date() for frame in three_histories.values())
    store.add_orders(
        [
            Order(
                order_id="old-strategy-order",
                strategy="ma_cross",
                symbol="000001",
                signal_date=signal_date,
                side=Side.BUY,
                target_weight=0.2,
            )
        ]
    )

    scheduled = service.promote_backtest(candidate.run_id)
    result = PaperTradingService(app_config, repository, store).run(as_of=signal_date)

    assert scheduled["status"] == "PENDING"
    assert service.current()["strategy_name"] == "ma_cross"
    assert result.new_orders == []
    assert "本次不生成新的旧策略订单" in result.message

    old_order = store.pending_orders()[0]
    old_order.status = OrderStatus.REJECTED
    store.update_order(old_order)
    activated = service.activate_pending_if_ready()
    assert activated is not None
    assert activated["strategy_name"] == "trend_breakout"
    assert service.current()["strategy_name"] == "trend_breakout"


def test_promotion_is_blocked_by_background_queue(app_config, three_histories) -> None:
    store = StateStore(app_config.storage.database)
    service = PaperStrategyService(app_config, store)
    service.current()
    candidate = save_candidate(app_config, three_histories, "momentum")
    store.create_background_job(
        job_id="queued-screen",
        kind="screen",
        lock_key="aquant-long-task",
        payload={"arguments": {}},
        log_path=app_config.storage.log_dir / "jobs" / "queued-screen.log",
    )

    with pytest.raises(RuntimeError, match="正在运行或排队"):
        service.promote_backtest(candidate.run_id)

    assert service.current()["strategy_name"] == "ma_cross"


def test_deployment_key_prevents_same_strategy_parameter_collisions(
    app_config, three_histories
) -> None:
    store = StateStore(app_config.storage.database)
    repository = MarketDataRepository(app_config)
    for symbol, frame in three_histories.items():
        atomic_write_parquet(frame, repository.path_for(symbol, ""))
        atomic_write_parquet(frame, repository.path_for(symbol, "qfq"))

    first_paper = PaperTradingService(app_config, repository, store)
    signal_date, first_signals, _, _ = first_paper._latest_signals()
    first_paper._generate_orders(signal_date, first_signals)
    for order in store.pending_orders():
        order.status = OrderStatus.REJECTED
        store.update_order(order)

    run_config = app_config.model_copy(deep=True)
    run_config.strategy.short_window = 3
    candidate = BacktestEngine(run_config).run(three_histories)
    save_backtest_result(candidate, run_config, store)
    PaperStrategyService(app_config, store).promote_backtest(candidate.run_id)

    second_paper = PaperTradingService(app_config, repository, store)
    second_date, second_signals, _, _ = second_paper._latest_signals()
    second_paper._generate_orders(second_date, second_signals)

    assert first_paper.execution_strategy_key != second_paper.execution_strategy_key
    signal_strategies = set(store.table("signals")["strategy"])
    assert first_paper.execution_strategy_key in signal_strategies
    assert second_paper.execution_strategy_key in signal_strategies


def test_promotion_rejects_tampered_backtest_artifacts(
    app_config, three_histories
) -> None:
    store = StateStore(app_config.storage.database)
    service = PaperStrategyService(app_config, store)
    service.current()
    candidate = save_candidate(app_config, three_histories, "rsi_reversion")
    metrics_path = app_config.storage.output_dir / "backtests" / candidate.run_id / "metrics.json"
    metrics_path.write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="不一致"):
        service.promote_backtest(candidate.run_id)
