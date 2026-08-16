from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from aquant.broker import (
    execution_price,
    limit_target_weights,
    round_down_lot,
    transaction_fees,
)
from aquant.config import AppConfig
from aquant.data import DataQualityReport, MarketDataRepository
from aquant.models import (
    Fill,
    Order,
    OrderStatus,
    PortfolioSnapshot,
    Position,
    Side,
    Signal,
)
from aquant.promotion import PaperStrategyService
from aquant.storage import StateStore
from aquant.strategy import create_strategy


@dataclass
class DailyRunResult:
    run_date: date
    status: str
    message: str
    fills: list[Fill] = field(default_factory=list)
    new_orders: list[Order] = field(default_factory=list)
    snapshot: PortfolioSnapshot | None = None
    report_dir: Path | None = None
    deployment_id: str = ""
    strategy_name: str = ""


class PaperTradingService:
    def __init__(
        self,
        config: AppConfig,
        repository: MarketDataRepository | None = None,
        store: StateStore | None = None,
    ) -> None:
        self.store = store or StateStore(config.storage.database)
        self.deployment_service = PaperStrategyService(config, self.store)
        self.deployment = self.deployment_service.current()
        self.config = config.model_copy(deep=True)
        self.repository = repository or MarketDataRepository(self.config)
        self.store.ensure_account(self.config.portfolio.initial_cash)
        self.logger = logging.getLogger("aquant.daily")
        self._apply_deployment(self.deployment)

    def _apply_deployment(self, deployment: dict[str, object]) -> None:
        self.deployment = deployment
        self.config.strategy = self.deployment_service.parameters(deployment)
        self.execution_strategy_key = (
            f"{self.config.strategy.name}@{deployment['deployment_id']}"
        )
        self.strategy = create_strategy(self.config.strategy)
        self.store.save_strategy(
            self.execution_strategy_key,
            {
                **self.config.strategy.model_dump(),
                "deployment_id": str(deployment["deployment_id"]),
            },
        )

    @staticmethod
    def _first_tradable_bar(frame: pd.DataFrame, after: date) -> pd.Series | None:
        candidates = frame[pd.to_datetime(frame["date"]).dt.date > after].sort_values("date")
        for _, bar in candidates.iterrows():
            if float(bar["volume"]) > 0 and float(bar["high"]) != float(bar["low"]):
                return bar
        return None

    def _marks(self, day: date, use_open: bool = False) -> dict[str, float]:
        marks: dict[str, float] = {}
        column = "open" if use_open else "close"
        symbols = set(self.config.market.watchlist) | set(self.store.positions())
        for symbol in symbols:
            try:
                frame = self.repository.load(symbol, "")
            except Exception:
                continue
            candidates = frame[pd.to_datetime(frame["date"]).dt.date <= day]
            if not candidates.empty:
                marks[symbol] = float(candidates.sort_values("date").iloc[-1][column])
        return marks

    def _current_equity(self, day: date) -> float:
        marks = self._marks(day)
        return self.store.get_cash() + sum(
            position.quantity * marks.get(symbol, position.average_cost)
            for symbol, position in self.store.positions().items()
        )

    def _portfolio_risk_off(self, signal_date: date) -> tuple[bool, float]:
        """按唯一信号日推进模拟盘回撤冷静期，重复运行不会重复扣减。"""
        current_equity = self._current_equity(signal_date)
        historical_peak = self.store.peak_equity() or self.config.portfolio.initial_cash
        peak_equity = max(historical_peak, current_equity)
        drawdown = current_equity / peak_equity - 1 if peak_equity else 0.0

        last_date = self.store.get_state("paper_risk_off_last_signal_date")
        if last_date == signal_date.isoformat():
            active = self.store.get_state("paper_risk_off_last_active", "0") == "1"
            return active, drawdown

        remaining = int(self.store.get_state("paper_risk_off_remaining", "0") or 0)
        armed = self.store.get_state("paper_risk_off_armed", "1") == "1"
        threshold = self.config.portfolio.max_portfolio_drawdown
        if drawdown > -(threshold / 2):
            armed = True
        if remaining == 0 and armed and drawdown <= -threshold + 1e-12:
            remaining = self.config.portfolio.drawdown_cooldown_days
            armed = False

        active = remaining > 0
        if active:
            remaining -= 1
        self.store.set_state("paper_risk_off_remaining", str(remaining))
        self.store.set_state("paper_risk_off_armed", "1" if armed else "0")
        self.store.set_state("paper_risk_off_last_signal_date", signal_date.isoformat())
        self.store.set_state("paper_risk_off_last_active", "1" if active else "0")
        return active, drawdown

    def _execute_pending(self) -> list[Fill]:
        pending = self.store.pending_orders()
        candidates: list[tuple[date, Order, pd.Series]] = []
        positions_at_start = self.store.positions()
        for order in pending:
            try:
                raw = self.repository.load(order.symbol, "")
            except Exception as exc:
                order.message = f"等待行情：{exc}"
                self.store.update_order(order)
                continue
            position = positions_at_start.get(order.symbol)
            force_exit = order.message.startswith(("风险熔断", "股票池移除"))
            reducing = order.target_weight == 0 or order.side == Side.SELL
            bar = None
            possible = raw[pd.to_datetime(raw["date"]).dt.date > order.signal_date].sort_values(
                "date"
            )
            for _, candidate in possible.iterrows():
                if float(candidate["volume"]) <= 0 or float(candidate["high"]) == float(
                    candidate["low"]
                ):
                    continue
                candidate_date = pd.Timestamp(candidate["date"]).date()
                if reducing and position and position.last_buy_date and not force_exit:
                    held_sessions = int(
                        (
                            (pd.to_datetime(raw["date"]).dt.date > position.last_buy_date)
                            & (pd.to_datetime(raw["date"]).dt.date <= candidate_date)
                        ).sum()
                    )
                    if held_sessions < self.config.portfolio.min_holding_days:
                        continue
                bar = candidate
                break
            if bar is None:
                order.message = (
                    f"等待满足最短持有 {self.config.portfolio.min_holding_days} 个交易日"
                    if reducing and position and position.last_buy_date and not force_exit
                    else "等待下一条可交易行情"
                )
                self.store.update_order(order)
                continue
            candidates.append((pd.Timestamp(bar["date"]).date(), order, bar))

        def sort_key(item: tuple[date, Order, pd.Series]) -> tuple[date, int, str]:
            reducing = item[1].target_weight == 0 or item[1].side == Side.SELL
            return item[0], 0 if reducing else 1, item[1].symbol

        fills: list[Fill] = []
        for trade_date, order, bar in sorted(candidates, key=sort_key):
            positions = self.store.positions()
            position = positions.get(order.symbol, Position(order.symbol, 0, 0.0))
            cash = self.store.get_cash()
            marks = self._marks(trade_date, use_open=True)
            marks[order.symbol] = float(bar["open"])
            equity = cash + sum(
                item.quantity * marks.get(symbol, item.average_cost)
                for symbol, item in positions.items()
            )
            target_quantity = round_down_lot(
                equity * order.target_weight / float(bar["open"]),
                self.config.portfolio.lot_size,
            )
            difference = target_quantity - position.quantity
            if difference == 0:
                order.status = OrderStatus.SKIPPED
                order.message = "已处于目标仓位"
                self.store.update_order(order)
                continue
            side = Side.BUY if difference > 0 else Side.SELL
            quantity = abs(difference)
            price, slip_per_share = execution_price(float(bar["open"]), side, self.config.costs)
            if side == Side.BUY:
                affordable = round_down_lot(cash / price, self.config.portfolio.lot_size)
                quantity = min(quantity, affordable)
                while quantity > 0:
                    notional = quantity * price
                    commission, _ = transaction_fees(notional, side, self.config.costs)
                    if notional + commission <= cash:
                        break
                    quantity -= self.config.portfolio.lot_size
                if quantity <= 0:
                    order.status = OrderStatus.REJECTED
                    order.message = "可用现金不足一个交易单位"
                    self.store.update_order(order)
                    continue
            notional = quantity * price
            commission, stamp = transaction_fees(notional, side, self.config.costs)
            if side == Side.BUY:
                old_quantity = position.quantity
                new_quantity = old_quantity + quantity
                position.average_cost = (
                    old_quantity * position.average_cost + notional + commission
                ) / new_quantity
                position.quantity = new_quantity
                position.last_buy_date = trade_date
                cash -= notional + commission
            else:
                quantity = min(quantity, position.quantity)
                notional = quantity * price
                commission, stamp = transaction_fees(notional, side, self.config.costs)
                position.quantity -= quantity
                cash += notional - commission - stamp
            order.side = side
            order.quantity = quantity
            order.status = OrderStatus.FILLED
            order.message = f"{trade_date.isoformat()} 模拟成交"
            fill_id = hashlib.sha256(f"{order.order_id}:{trade_date}".encode()).hexdigest()[:20]
            fill = Fill(
                fill_id=fill_id,
                order_id=order.order_id,
                symbol=order.symbol,
                trade_date=trade_date,
                side=side,
                quantity=quantity,
                price=price,
                commission=commission,
                stamp_duty=stamp,
                slippage=slip_per_share * quantity,
            )
            self.store.settle_fill(fill, order, position, cash)
            fills.append(fill)
        return fills

    def _latest_signals(self) -> tuple[date, list[Signal], bool, float]:
        signal_values: dict[str, int] = {}
        signal_scores: dict[str, float] = {}
        annual_volatilities: dict[str, float] = {}
        latest_dates: list[date] = []
        histories: list[pd.DataFrame] = []
        for symbol in self.config.market.watchlist:
            history = self.repository.load(symbol, "qfq")
            histories.append(history)
            volatility = history["close"].pct_change(fill_method=None).tail(
                self.config.portfolio.volatility_window
            ).std() * (252**0.5)
            annual_volatilities[symbol] = float(volatility) if pd.notna(volatility) else 0.0
        generated = self.strategy.generate_signals(pd.concat(histories, ignore_index=True))
        for symbol in self.config.market.watchlist:
            symbol_signals = generated[generated["symbol"] == symbol]
            if symbol_signals.empty:
                raise ValueError(f"{symbol} 未生成策略信号")
            row = symbol_signals.sort_values("date").iloc[-1]
            signal_values[symbol] = int(row["signal"])
            signal_scores[symbol] = float(
                row.get("ensemble_score", row.get("momentum_score", row["signal"]))
            )
            latest_dates.append(pd.Timestamp(row["date"]).date())
        signal_date = max(latest_dates)
        active = sorted(
            [symbol for symbol, value in signal_values.items() if value == 1],
            key=lambda symbol: (signal_scores[symbol], symbol),
            reverse=True,
        )[: self.config.portfolio.max_positions]
        if (
            active
            and self.config.portfolio.allocation_method == "inverse_volatility"
            and all(annual_volatilities[symbol] > 0 for symbol in active)
        ):
            inverse = {symbol: 1 / annual_volatilities[symbol] for symbol in active}
            total_inverse = sum(inverse.values())
            base_weights = {symbol: inverse[symbol] / total_inverse for symbol in active}
        else:
            base_weights = {symbol: 1 / len(active) if active else 0.0 for symbol in active}
        estimated_volatility = (
            sum((base_weights[symbol] * annual_volatilities[symbol]) ** 2 for symbol in active)
            ** 0.5
        )
        volatility_scale = (
            min(
                1.0,
                self.config.portfolio.target_annual_volatility / estimated_volatility,
            )
            if estimated_volatility
            else 1.0
        )
        exposure = (1 - self.config.portfolio.cash_buffer) * volatility_scale
        risk_off, drawdown = self._portfolio_risk_off(signal_date)
        target_weights = {
            symbol: min(
                self.config.portfolio.max_position_weight,
                base_weights.get(symbol, 0.0) * exposure,
            )
            for symbol in self.config.market.watchlist
        }
        if risk_off:
            target_weights = dict.fromkeys(self.config.market.watchlist, 0.0)
        return (
            signal_date,
            [
                Signal(
                    strategy=self.execution_strategy_key,
                    symbol=symbol,
                    signal_date=signal_date,
                    value=signal_values[symbol],
                    target_weight=target_weights[symbol],
                )
                for symbol in self.config.market.watchlist
            ],
            risk_off,
            drawdown,
        )

    def _generate_orders(
        self,
        signal_date: date,
        signals: list[Signal],
        *,
        risk_off: bool = False,
    ) -> list[Order]:
        positions = self.store.positions()
        signal_symbols = {signal.symbol for signal in signals}
        self.store.add_signals(signals)
        marks = self._marks(signal_date)
        equity = self.store.get_cash() + sum(
            position.quantity * marks.get(symbol, position.average_cost)
            for symbol, position in positions.items()
        )
        current_weights = {
            symbol: (
                position.quantity * marks.get(symbol, position.average_cost) / equity
                if equity
                else 0.0
            )
            for symbol, position in positions.items()
        }
        desired_weights = {signal.symbol: signal.target_weight for signal in signals}
        desired_weights.update({symbol: 0.0 for symbol in set(positions) - signal_symbols})
        target_weights = limit_target_weights(
            current_weights,
            desired_weights,
            minimum_change=self.config.portfolio.minimum_rebalance_weight_change,
            max_turnover=self.config.portfolio.max_daily_turnover,
            force_exit=risk_off,
        )
        pending_symbols = {order.symbol for order in self.store.pending_orders()}
        orders: list[Order] = []
        for signal in signals:
            current = positions.get(signal.symbol, Position(signal.symbol, 0, 0.0)).quantity
            current_weight = current_weights.get(signal.symbol, 0.0)
            target_weight = target_weights.get(signal.symbol, 0.0)
            if signal.symbol in pending_symbols:
                continue
            if target_weight == 0 and current == 0:
                continue
            if abs(target_weight - current_weight) < 1e-12:
                continue
            side = Side.BUY if target_weight > current_weight else Side.SELL
            order_id = hashlib.sha256(
                f"{signal.strategy}:{signal.symbol}:{signal_date.isoformat()}".encode()
            ).hexdigest()[:20]
            orders.append(
                Order(
                    order_id=order_id,
                    strategy=signal.strategy,
                    symbol=signal.symbol,
                    signal_date=signal_date,
                    side=side,
                    target_weight=target_weight,
                    message=("风险熔断强制清仓" if risk_off and target_weight == 0 else ""),
                )
            )
        for symbol in sorted(set(positions) - signal_symbols):
            if symbol in pending_symbols:
                continue
            order_id = hashlib.sha256(
                f"{self.execution_strategy_key}:{symbol}:{signal_date.isoformat()}".encode()
            ).hexdigest()[:20]
            orders.append(
                Order(
                    order_id=order_id,
                    strategy=self.execution_strategy_key,
                    symbol=symbol,
                    signal_date=signal_date,
                    side=Side.SELL,
                    target_weight=0.0,
                    message="股票池移除清仓",
                )
            )
        existing = self.store.existing_order_ids([order.order_id for order in orders])
        new_orders = [order for order in orders if order.order_id not in existing]
        self.store.add_orders(new_orders)
        return new_orders

    def _snapshot(self, day: date) -> PortfolioSnapshot:
        cash = self.store.get_cash()
        marks = self._marks(day)
        market_value = sum(
            position.quantity * marks.get(symbol, position.average_cost)
            for symbol, position in self.store.positions().items()
        )
        snapshot = PortfolioSnapshot(day, cash, market_value, cash + market_value)
        self.store.save_snapshot(snapshot)
        return snapshot

    def run(
        self,
        *,
        as_of: date | None = None,
        update_reports: list[DataQualityReport] | None = None,
    ) -> DailyRunResult:
        as_of = as_of or datetime.now().date()
        fills = self._execute_pending()
        activated = self.deployment_service.activate_pending_if_ready()
        if activated is not None:
            self._apply_deployment(activated)
        pending_deployment = self.deployment_service.pending()
        remaining_old_orders = self.store.pending_orders()
        if pending_deployment is not None and remaining_old_orders:
            latest_available = [
                value for value in self.repository.latest_dates().values() if value
            ]
            run_date = max(latest_available) if latest_available else as_of
            snapshot = self._snapshot(run_date)
            message = (
                f"策略晋级等待中：旧策略还有 {len(remaining_old_orders)} 笔待成交订单；"
                "本次不生成新的旧策略订单"
            )
            self.store.record_daily_run(
                run_date,
                "SUCCESS",
                message,
                deployment_id=str(self.deployment["deployment_id"]),
                strategy_name=self.config.strategy.name,
            )
            return DailyRunResult(
                run_date,
                "SUCCESS",
                message,
                fills,
                [],
                snapshot,
                deployment_id=str(self.deployment["deployment_id"]),
                strategy_name=self.config.strategy.name,
            )
        stale = self.repository.stale_symbols(as_of)
        update_failed = [report for report in (update_reports or []) if report.used_cache]
        latest_available = [value for value in self.repository.latest_dates().values() if value]
        run_date = max(latest_available) if latest_available else as_of
        if stale or update_failed:
            reasons = [f"{symbol}: {reason}" for symbol, reason in stale.items()]
            reasons.extend(f"{report.symbol}: 使用缓存" for report in update_failed)
            message = "数据未通过新鲜度检查，未生成新订单；" + "；".join(reasons)
            snapshot = self._snapshot(run_date)
            self.store.record_daily_run(
                run_date,
                "BLOCKED",
                message,
                deployment_id=str(self.deployment["deployment_id"]),
                strategy_name=self.config.strategy.name,
            )
            return DailyRunResult(
                run_date,
                "BLOCKED",
                message,
                fills,
                [],
                snapshot,
                deployment_id=str(self.deployment["deployment_id"]),
                strategy_name=self.config.strategy.name,
            )
        signal_date, signals, risk_off, drawdown = self._latest_signals()
        orders = self._generate_orders(signal_date, signals, risk_off=risk_off)
        snapshot = self._snapshot(signal_date)
        message = f"记录 {len(fills)} 笔成交，生成 {len(orders)} 笔待处理目标订单"
        if risk_off:
            message += f"；组合回撤 {drawdown:.2%}，冷静期内目标仓位归零"
        self.store.record_daily_run(
            signal_date,
            "SUCCESS",
            message,
            deployment_id=str(self.deployment["deployment_id"]),
            strategy_name=self.config.strategy.name,
        )
        return DailyRunResult(
            signal_date,
            "SUCCESS",
            message,
            fills,
            orders,
            snapshot,
            deployment_id=str(self.deployment["deployment_id"]),
            strategy_name=self.config.strategy.name,
        )


def run_daily(config: AppConfig, update_data: bool = True) -> DailyRunResult:
    repository = MarketDataRepository(config)
    reports = repository.update_all() if update_data else None
    store = StateStore(config.storage.database)
    store.backup(config.storage.output_dir / "backups")
    service = PaperTradingService(config, repository, store)
    result = service.run(update_reports=reports)
    result.report_dir = save_daily_report(result, service.config, store)
    return result


def save_daily_report(
    result: DailyRunResult,
    config: AppConfig,
    store: StateStore,
) -> Path:
    output_dir = config.storage.output_dir / "paper" / result.run_date.strftime("%Y%m%d")
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = result.snapshot
    summary = {
        "run_date": result.run_date.isoformat(),
        "status": result.status,
        "message": result.message,
        "fills": len(result.fills),
        "new_orders": len(result.new_orders),
        "cash": snapshot.cash if snapshot else None,
        "market_value": snapshot.market_value if snapshot else None,
        "equity": snapshot.equity if snapshot else None,
        "paper_strategy_deployment_id": result.deployment_id,
        "paper_strategy_name": result.strategy_name,
        "strategy": config.strategy.model_dump(),
        "portfolio": config.portfolio.model_dump(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    for table in ["positions", "orders", "fills", "signals", "portfolio_snapshots"]:
        frame = store.table(table, limit=100_000)
        if table in {"orders", "signals"} and not frame.empty:
            frame = frame[frame["signal_date"] == result.run_date.isoformat()]
        if table == "fills" and not frame.empty:
            frame = frame[frame["trade_date"] == result.run_date.isoformat()]
        frame.to_csv(output_dir / f"{table}.csv", index=False)
        frame.to_parquet(output_dir / f"{table}.parquet", index=False)
    return output_dir
