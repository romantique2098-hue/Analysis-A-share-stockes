from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import uuid
from datetime import date
from pathlib import Path

import pandas as pd

from aquant.broker import execution_price, limit_target_weights, round_down_lot, transaction_fees
from aquant.config import AppConfig, CostConfig
from aquant.data import MarketDataRepository
from aquant.metrics import calculate_metrics
from aquant.models import BacktestResult, Position, Side
from aquant.research import calendar_year_analysis, rolling_relative_analysis
from aquant.storage import StateStore
from aquant.strategy import STRATEGY_NAMES, Strategy, create_strategy

BACKTEST_ENGINE_VERSION = 3
BACKTEST_ARTIFACTS = (
    "equity_curve.parquet",
    "trades.csv",
    "signals.parquet",
    "annual_returns.csv",
    "rolling_relative.parquet",
    "metrics.json",
    "parameters.json",
)


class BacktestEngine:
    def __init__(self, config: AppConfig, strategy: Strategy | None = None) -> None:
        self.config = config
        self.strategy = strategy or create_strategy(config.strategy)

    @staticmethod
    def _frame_fingerprint(frame: pd.DataFrame) -> str:
        stable = frame.reset_index(drop=True).copy()
        if "date" in stable:
            stable["date"] = pd.to_datetime(stable["date"]).dt.strftime("%Y-%m-%d")
        stable = stable.reindex(sorted(stable.columns), axis=1)
        hashed = pd.util.hash_pandas_object(stable, index=False).values.tobytes()
        return hashlib.sha256(hashed).hexdigest()

    def _prepare(
        self, histories: dict[str, pd.DataFrame], start: date | None, end: date | None
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        filtered: dict[str, pd.DataFrame] = {}
        full_histories: list[pd.DataFrame] = []
        for symbol, frame in histories.items():
            data = frame.copy()
            data["date"] = pd.to_datetime(data["date"]).dt.normalize()
            data = data.sort_values("date").drop_duplicates("date", keep="last")
            full_histories.append(data)
            if start:
                data = data[data["date"].dt.date >= start]
            if end:
                data = data[data["date"].dt.date <= end]
            if data.empty:
                raise ValueError(f"{symbol} 在指定区间没有行情")
            filtered[symbol] = data.set_index("date", drop=False)
        combined_history = pd.concat(full_histories, ignore_index=True)
        generated = self.strategy.generate_signals(combined_history)
        if start:
            generated = generated[generated["date"].dt.date >= start]
        if end:
            generated = generated[generated["date"].dt.date <= end]
        return filtered, generated.sort_values(["date", "symbol"]).reset_index(drop=True)

    def _simulate(
        self,
        histories: dict[str, pd.DataFrame],
        signals: pd.DataFrame,
        costs: CostConfig,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        calendar = sorted(set().union(*(set(frame.index) for frame in histories.values())))
        signal_lookup = {
            (pd.Timestamp(row.date), row.symbol): int(row.signal)
            for row in signals.itertuples(index=False)
        }
        score_column = next(
            (column for column in ["ensemble_score", "momentum_score"] if column in signals),
            None,
        )
        score_lookup = (
            {
                (pd.Timestamp(row.date), row.symbol): float(getattr(row, score_column))
                for row in signals.itertuples(index=False)
                if pd.notna(getattr(row, score_column))
            }
            if score_column
            else {}
        )
        volatility_lookup: dict[tuple[pd.Timestamp, str], float] = {}
        for symbol, frame in histories.items():
            returns = frame["close"].pct_change(fill_method=None)
            volatility = returns.rolling(
                self.config.portfolio.volatility_window,
                min_periods=self.config.portfolio.volatility_window,
            ).std() * (252**0.5)
            volatility_lookup.update(
                {
                    (pd.Timestamp(timestamp), symbol): float(value)
                    for timestamp, value in volatility.items()
                    if pd.notna(value) and value > 0
                }
            )

        def target_weights(active_symbols: list[str], timestamp: pd.Timestamp) -> dict[str, float]:
            ranked = sorted(
                active_symbols,
                key=lambda symbol: (
                    score_lookup.get((timestamp, symbol), 0.0),
                    symbol,
                ),
                reverse=True,
            )[: self.config.portfolio.max_positions]
            if not ranked:
                return {symbol: 0.0 for symbol in histories}
            volatilities = {symbol: volatility_lookup.get((timestamp, symbol)) for symbol in ranked}
            if self.config.portfolio.allocation_method == "inverse_volatility" and all(
                volatilities.values()
            ):
                inverse = {symbol: 1 / float(volatilities[symbol]) for symbol in ranked}
                inverse_sum = sum(inverse.values())
                base = {symbol: value / inverse_sum for symbol, value in inverse.items()}
            else:
                base = {symbol: 1 / len(ranked) for symbol in ranked}
            estimated_volatility = (
                sum(
                    (base[symbol] * float(volatilities[symbol])) ** 2
                    for symbol in ranked
                    if volatilities[symbol]
                )
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
            targets = {
                symbol: min(
                    self.config.portfolio.max_position_weight,
                    base.get(symbol, 0.0) * exposure,
                )
                for symbol in histories
            }
            return targets

        cash = self.config.portfolio.initial_cash
        positions: dict[str, Position] = {}
        last_closes: dict[str, float] = {}
        states = {symbol: 0 for symbol in histories}
        desired_targets = {symbol: 0.0 for symbol in histories}
        pending_targets: dict[str, float] | None = None
        pending_force_exit = False
        last_buy_indices: dict[str, int] = {}
        curve_rows: list[dict[str, object]] = []
        trades: list[dict[str, object]] = []
        peak_equity = self.config.portfolio.initial_cash
        risk_off_until = -1
        was_risk_off = False
        risk_off_armed = True

        for day_index, timestamp in enumerate(calendar):
            day = timestamp.date()
            for symbol, frame in histories.items():
                if timestamp in frame.index:
                    last_closes[symbol] = float(frame.loc[timestamp, "close"])

            traded_notional = 0.0
            if pending_targets is not None:
                open_values: dict[str, float] = {}
                for symbol, frame in histories.items():
                    if timestamp not in frame.index:
                        continue
                    open_price = float(frame.loc[timestamp, "open"])
                    if pd.notna(open_price) and open_price > 0:
                        open_values[symbol] = open_price
                mark_values = {
                    symbol: open_values.get(symbol, last_closes.get(symbol, 0.0))
                    for symbol in histories
                }
                equity_open = cash + sum(
                    position.quantity * mark_values.get(symbol, 0.0)
                    for symbol, position in positions.items()
                )
                desired: dict[str, int] = {}
                for symbol, weight in pending_targets.items():
                    current = positions.get(symbol, Position(symbol, 0, 0.0)).quantity
                    if symbol not in open_values:
                        continue
                    desired[symbol] = round_down_lot(
                        equity_open * weight / open_values[symbol],
                        self.config.portfolio.lot_size,
                    )
                # 先卖后买，确保买入资金只来自已到账现金。
                for symbol in sorted(desired):
                    current = positions.get(symbol, Position(symbol, 0, 0.0)).quantity
                    quantity = max(current - desired[symbol], 0)
                    if quantity <= 0:
                        continue
                    bar = histories[symbol].loc[timestamp]
                    # 零成交量仅作为停牌/全天无成交的代理；不使用正成交量大小或
                    # 当天最高、最低价决定开盘订单能否成交。
                    if "volume" in bar and float(bar["volume"]) <= 0:
                        continue
                    if (
                        not pending_force_exit
                        and day_index - last_buy_indices.get(symbol, -10_000)
                        < self.config.portfolio.min_holding_days
                    ):
                        continue
                    position = positions[symbol]
                    price, slip_per_share = execution_price(open_values[symbol], Side.SELL, costs)
                    notional = quantity * price
                    commission, stamp = transaction_fees(notional, Side.SELL, costs)
                    cash += notional - commission - stamp
                    position.quantity -= quantity
                    if position.quantity == 0:
                        positions.pop(symbol)
                        last_buy_indices.pop(symbol, None)
                    traded_notional += notional
                    trades.append(
                        {
                            "date": timestamp,
                            "symbol": symbol,
                            "side": Side.SELL.value,
                            "quantity": quantity,
                            "price": price,
                            "notional": notional,
                            "commission": commission,
                            "stamp_duty": stamp,
                            "slippage": slip_per_share * quantity,
                        }
                    )
                for symbol in sorted(desired):
                    current_position = positions.get(symbol, Position(symbol, 0, 0.0))
                    requested = max(desired[symbol] - current_position.quantity, 0)
                    if requested <= 0:
                        continue
                    bar = histories[symbol].loc[timestamp]
                    if "volume" in bar and float(bar["volume"]) <= 0:
                        continue
                    price, slip_per_share = execution_price(open_values[symbol], Side.BUY, costs)
                    affordable = round_down_lot(cash / price, self.config.portfolio.lot_size)
                    quantity = min(requested, affordable)
                    while quantity > 0:
                        notional = quantity * price
                        commission, _ = transaction_fees(notional, Side.BUY, costs)
                        if notional + commission <= cash:
                            break
                        quantity -= self.config.portfolio.lot_size
                    if quantity <= 0:
                        continue
                    notional = quantity * price
                    commission, stamp = transaction_fees(notional, Side.BUY, costs)
                    cash -= notional + commission
                    old_quantity = current_position.quantity
                    new_quantity = old_quantity + quantity
                    current_position.average_cost = (
                        current_position.average_cost * old_quantity + notional + commission
                    ) / new_quantity
                    current_position.quantity = new_quantity
                    current_position.last_buy_date = day
                    positions[symbol] = current_position
                    last_buy_indices[symbol] = day_index
                    traded_notional += notional
                    trades.append(
                        {
                            "date": timestamp,
                            "symbol": symbol,
                            "side": Side.BUY.value,
                            "quantity": quantity,
                            "price": price,
                            "notional": notional,
                            "commission": commission,
                            "stamp_duty": stamp,
                            "slippage": slip_per_share * quantity,
                        }
                    )
                # 日频数据无法在开盘时知道当天完整成交量、最高价或最低价，因此只要有
                # 有效开盘价便按开盘撮合。未完成目标会在当日收盘后基于最新信号、权益
                # 和持仓重新生成，避免旧订单长期阻塞后续调仓。
                pending_targets = None
                pending_force_exit = False

            market_value = sum(
                position.quantity * last_closes.get(symbol, position.average_cost)
                for symbol, position in positions.items()
            )
            equity = cash + market_value
            peak_equity = max(peak_equity, equity)
            portfolio_drawdown = equity / peak_equity - 1
            threshold = self.config.portfolio.max_portfolio_drawdown
            if portfolio_drawdown > -(threshold / 2):
                risk_off_armed = True
            if (
                portfolio_drawdown <= -threshold + 1e-12
                and day_index > risk_off_until
                and risk_off_armed
            ):
                risk_off_until = day_index + self.config.portfolio.drawdown_cooldown_days - 1
                risk_off_armed = False
            risk_off = day_index <= risk_off_until
            curve_rows.append(
                {
                    "date": timestamp,
                    "cash": cash,
                    "market_value": market_value,
                    "equity": equity,
                    "portfolio_drawdown": portfolio_drawdown,
                    "risk_off": risk_off,
                    "daily_turnover": traded_notional / equity if equity else 0.0,
                }
            )

            changed_today = False
            for symbol in histories:
                value = signal_lookup.get((timestamp, symbol))
                if value is not None and value != states[symbol]:
                    states[symbol] = value
                    changed_today = True
            risk_state_changed = risk_off != was_risk_off
            if risk_off:
                desired_targets = {symbol: 0.0 for symbol in histories}
                pending_targets = dict(desired_targets)
                pending_force_exit = True
            elif changed_today or risk_state_changed:
                active = [symbol for symbol, value in states.items() if value == 1]
                desired_targets = target_weights(active, timestamp)

            if pending_targets is None:
                current_weights = {
                    symbol: (
                        positions.get(symbol, Position(symbol, 0, 0.0)).quantity
                        * last_closes.get(symbol, 0.0)
                        / equity
                        if equity
                        else 0.0
                    )
                    for symbol in histories
                }
                limited_targets = limit_target_weights(
                    current_weights,
                    desired_targets,
                    minimum_change=self.config.portfolio.minimum_rebalance_weight_change,
                    max_turnover=self.config.portfolio.max_daily_turnover,
                )
                if any(
                    abs(limited_targets[symbol] - current_weights[symbol]) > 1e-12
                    for symbol in histories
                ):
                    pending_targets = {
                        symbol: limited_targets[symbol]
                        for symbol in histories
                        if abs(limited_targets[symbol] - current_weights[symbol]) > 1e-12
                    }
            was_risk_off = risk_off

        curve = pd.DataFrame(curve_rows)
        curve["net_curve"] = curve["equity"] / self.config.portfolio.initial_cash
        return curve, pd.DataFrame(trades)

    def run(
        self,
        histories: dict[str, pd.DataFrame],
        benchmark: pd.DataFrame | None = None,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> BacktestResult:
        prepared, signals = self._prepare(histories, start, end)
        net_curve, trades = self._simulate(prepared, signals, self.config.costs)
        if trades.empty:
            cumulative_costs = pd.Series(0.0, index=net_curve.index)
        else:
            executions = trades.copy()
            executions["date"] = pd.to_datetime(executions["date"]).dt.normalize()
            executions["transaction_costs"] = executions[
                ["commission", "stamp_duty", "slippage"]
            ].sum(axis=1)
            costs_by_date = executions.groupby("date")["transaction_costs"].sum()
            cumulative_costs = (
                pd.to_datetime(net_curve["date"])
                .map(costs_by_date)
                .fillna(0.0)
                .cumsum()
            )
        net_curve["cumulative_costs"] = cumulative_costs.to_numpy()
        net_curve["gross_curve"] = (
            net_curve["equity"] + net_curve["cumulative_costs"]
        ) / self.config.portfolio.initial_cash

        normalized_stocks: list[pd.Series] = []
        for symbol, frame in prepared.items():
            close = frame["close"].reindex(net_curve["date"]).ffill()
            first_valid = close.first_valid_index()
            if first_valid is not None:
                normalized_stocks.append((close / close.loc[first_valid]).rename(symbol))
        net_curve["buy_hold_curve"] = pd.concat(normalized_stocks, axis=1).mean(axis=1).to_numpy()
        benchmark_fingerprint: str | None = None
        if benchmark is not None and not benchmark.empty:
            benchmark_data = benchmark.copy()
            benchmark_data["date"] = pd.to_datetime(benchmark_data["date"]).dt.normalize()
            benchmark_effective = benchmark_data[
                benchmark_data["date"].between(
                    net_curve["date"].min(), net_curve["date"].max()
                )
            ]
            benchmark_fingerprint = self._frame_fingerprint(benchmark_effective)
            benchmark_close = benchmark_data.set_index("date")["close"]
            aligned = benchmark_close.reindex(net_curve["date"]).ffill()
            first_valid = aligned.first_valid_index()
            if first_valid is not None:
                net_curve["benchmark_curve"] = (aligned / aligned.loc[first_valid]).to_numpy()
                relative = net_curve["net_curve"] / net_curve["benchmark_curve"]
                first_relative = relative.first_valid_index()
                if first_relative is not None:
                    net_curve["relative_curve"] = relative / relative.loc[first_relative]

        total_costs = float(
            trades.get("commission", pd.Series(dtype=float)).sum()
            + trades.get("stamp_duty", pd.Series(dtype=float)).sum()
            + trades.get("slippage", pd.Series(dtype=float)).sum()
        )
        turnover_notional = float(trades.get("notional", pd.Series(dtype=float)).sum())
        metrics = calculate_metrics(
            net_curve,
            risk_free_rate=self.config.portfolio.risk_free_rate,
            total_costs=total_costs,
            turnover_notional=turnover_notional,
            trade_count=len(trades),
        )
        parameters = {
            "backtest_engine_version": BACKTEST_ENGINE_VERSION,
            "strategy": self.config.strategy.model_dump(),
            "strategy_implementation": {
                "name": getattr(self.strategy, "name", type(self.strategy).__name__),
                "class": (
                    f"{type(self.strategy).__module__}.{type(self.strategy).__qualname__}"
                ),
                "signal_fingerprint": self._frame_fingerprint(signals),
            },
            "portfolio": self.config.portfolio.model_dump(),
            "costs": self.config.costs.model_dump(),
            "symbols": sorted(histories),
            "start": net_curve["date"].min().date().isoformat(),
            "end": net_curve["date"].max().date().isoformat(),
            "data_fingerprints": {
                symbol: self._frame_fingerprint(
                    frame[
                        pd.to_datetime(frame["date"]).dt.normalize()
                        <= net_curve["date"].max()
                    ]
                )
                for symbol, frame in sorted(histories.items())
            },
            "benchmark_fingerprint": benchmark_fingerprint,
        }
        run_id = hashlib.sha256(
            json.dumps(parameters, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()[:16]
        return BacktestResult(run_id, metrics, net_curve, trades, signals, parameters)


def load_and_run_backtest(config: AppConfig) -> BacktestResult:
    repository = MarketDataRepository(config)
    histories = {symbol: repository.load(symbol, "qfq") for symbol in config.market.watchlist}
    benchmark = repository.load(config.market.benchmark, "", benchmark=True)
    return BacktestEngine(config).run(histories, benchmark)


def load_saved_backtest(output_dir: Path) -> BacktestResult:
    """从持久化目录恢复回测，供页面刷新或后台任务完成后展示。"""
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    parameters = json.loads((output_dir / "parameters.json").read_text(encoding="utf-8"))
    try:
        trades = pd.read_csv(output_dir / "trades.csv")
    except pd.errors.EmptyDataError:
        trades = pd.DataFrame()
    return BacktestResult(
        run_id=output_dir.name,
        metrics={key: float(value) for key, value in metrics.items()},
        equity_curve=pd.read_parquet(output_dir / "equity_curve.parquet"),
        trades=trades,
        signals=pd.read_parquet(output_dir / "signals.parquet"),
        parameters=parameters,
    )


def load_and_compare_strategies(
    config: AppConfig,
    names: tuple[str, ...] = STRATEGY_NAMES,
    *,
    start: date | None = None,
    end: date | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, BacktestResult]]:
    repository = MarketDataRepository(config)
    histories = {symbol: repository.load(symbol, "qfq") for symbol in config.market.watchlist}
    benchmark = repository.load(config.market.benchmark, "", benchmark=True)
    results: dict[str, BacktestResult] = {}
    metric_rows: list[dict[str, object]] = []
    curve_frames: list[pd.DataFrame] = []
    for name in names:
        run_config = config.model_copy(deep=True)
        run_config.strategy.name = name
        result = BacktestEngine(run_config).run(histories, benchmark, start=start, end=end)
        results[name] = result
        metric_rows.append({"strategy": name, "run_id": result.run_id, **result.metrics})
        curve_frames.append(
            result.equity_curve[["date", "net_curve"]].rename(columns={"net_curve": name})
        )
    metrics = pd.DataFrame(metric_rows)
    curves = curve_frames[0]
    for frame in curve_frames[1:]:
        curves = curves.merge(frame, on="date", how="outer")
    return metrics, curves.sort_values("date"), results


def save_backtest_result(
    result: BacktestResult, config: AppConfig, store: StateStore | None = None
) -> Path:
    output_dir = config.storage.output_dir / "backtests" / result.run_id
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    def checksum(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def complete_existing() -> bool:
        manifest_path = output_dir / "manifest.json"
        if not manifest_path.exists():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("run_id") != result.run_id:
                return False
            checksums = manifest["checksums"]
            return all(
                (output_dir / name).exists()
                and checksums.get(name) == checksum(output_dir / name)
                for name in BACKTEST_ARTIFACTS
            )
        except (KeyError, OSError, TypeError, ValueError):
            return False

    if not complete_existing():
        temporary_dir = Path(
            tempfile.mkdtemp(prefix=f".{result.run_id}-", dir=output_dir.parent)
        )
        backup_dir: Path | None = None
        try:
            result.equity_curve.to_parquet(
                temporary_dir / "equity_curve.parquet", index=False
            )
            result.trades.to_csv(temporary_dir / "trades.csv", index=False)
            result.signals.to_parquet(temporary_dir / "signals.parquet", index=False)
            annual = calendar_year_analysis(result.equity_curve, result.trades)
            rolling = rolling_relative_analysis(result.equity_curve)
            annual.to_csv(temporary_dir / "annual_returns.csv", index=False)
            rolling.to_parquet(temporary_dir / "rolling_relative.parquet", index=False)
            (temporary_dir / "metrics.json").write_text(
                json.dumps(result.metrics, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            (temporary_dir / "parameters.json").write_text(
                json.dumps(result.parameters, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            checksums = {
                name: checksum(temporary_dir / name) for name in BACKTEST_ARTIFACTS
            }
            (temporary_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "run_id": result.run_id,
                        "backtest_engine_version": BACKTEST_ENGINE_VERSION,
                        "checksums": checksums,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            if output_dir.exists():
                backup_dir = output_dir.parent / (
                    f".{result.run_id}-backup-{uuid.uuid4().hex}"
                )
                output_dir.replace(backup_dir)
            try:
                temporary_dir.replace(output_dir)
            except Exception:
                if backup_dir is not None and backup_dir.exists() and not output_dir.exists():
                    backup_dir.replace(output_dir)
                raise
            if backup_dir is not None and backup_dir.exists():
                shutil.rmtree(backup_dir)
        finally:
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir)
    store = store or StateStore(config.storage.database)
    store.save_backtest(
        result.run_id,
        result.equity_curve["date"].max().date(),
        result.parameters,
        result.metrics,
        output_dir,
    )
    return output_dir


def save_strategy_comparison(
    metrics: pd.DataFrame,
    curves: pd.DataFrame,
    config: AppConfig,
) -> Path:
    fingerprint = metrics[["strategy", "run_id"]].sort_values("strategy").to_dict("records")
    comparison_id = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[
        :16
    ]
    output_dir = config.storage.output_dir / "comparisons" / comparison_id
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    curves.to_parquet(output_dir / "equity_curves.parquet", index=False)
    (output_dir / "runs.json").write_text(
        json.dumps(fingerprint, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_dir
