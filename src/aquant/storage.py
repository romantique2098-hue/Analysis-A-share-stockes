from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from aquant.models import Fill, Order, OrderStatus, PortfolioSnapshot, Position, Side, Signal

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS strategy_instances (
    name TEXT PRIMARY KEY,
    parameters_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS paper_strategy_deployments (
    deployment_id TEXT PRIMARY KEY,
    strategy_name TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    source_backtest_run_id TEXT,
    source_data_end_date TEXT,
    source_metrics_json TEXT NOT NULL DEFAULT '{}',
    source_parameters_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL CHECK(status IN ('ACTIVE', 'PENDING', 'RETIRED')),
    requested_at TEXT NOT NULL,
    activated_at TEXT,
    retired_at TEXT,
    note TEXT NOT NULL DEFAULT ''
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_paper_strategy_active
ON paper_strategy_deployments(status) WHERE status = 'ACTIVE';
CREATE UNIQUE INDEX IF NOT EXISTS uq_paper_strategy_pending
ON paper_strategy_deployments(status) WHERE status = 'PENDING';
CREATE TABLE IF NOT EXISTS signals (
    strategy TEXT NOT NULL,
    symbol TEXT NOT NULL,
    signal_date TEXT NOT NULL,
    value INTEGER NOT NULL,
    target_weight REAL NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (strategy, symbol, signal_date)
);
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    strategy TEXT NOT NULL,
    symbol TEXT NOT NULL,
    signal_date TEXT NOT NULL,
    side TEXT NOT NULL,
    target_weight REAL NOT NULL,
    quantity INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    message TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (strategy, symbol, signal_date)
);
CREATE TABLE IF NOT EXISTS fills (
    fill_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL UNIQUE REFERENCES orders(order_id),
    symbol TEXT NOT NULL,
    trade_date TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    price REAL NOT NULL,
    commission REAL NOT NULL,
    stamp_duty REAL NOT NULL,
    slippage REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS positions (
    symbol TEXT PRIMARY KEY,
    quantity INTEGER NOT NULL,
    average_cost REAL NOT NULL,
    last_buy_date TEXT
);
CREATE TABLE IF NOT EXISTS account_state (
    account TEXT PRIMARY KEY,
    cash REAL NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    account TEXT NOT NULL,
    snapshot_date TEXT NOT NULL,
    cash REAL NOT NULL,
    market_value REAL NOT NULL,
    equity REAL NOT NULL,
    PRIMARY KEY (account, snapshot_date)
);
CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    data_end_date TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    output_dir TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS daily_runs (
    run_date TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    message TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deployment_id TEXT,
    strategy_name TEXT
);
CREATE TABLE IF NOT EXISTS screen_runs (
    run_id TEXT PRIMARY KEY,
    as_of_date TEXT NOT NULL,
    status TEXT NOT NULL,
    universe_size INTEGER NOT NULL,
    eligible_count INTEGER NOT NULL,
    selected_count INTEGER NOT NULL,
    output_dir TEXT NOT NULL,
    warnings_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS screen_results (
    run_id TEXT NOT NULL REFERENCES screen_runs(run_id),
    symbol TEXT NOT NULL,
    rank INTEGER,
    selected INTEGER NOT NULL,
    total_score REAL,
    industry TEXT,
    factors_json TEXT NOT NULL,
    PRIMARY KEY (run_id, symbol)
);
CREATE TABLE IF NOT EXISTS background_jobs (
    job_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    lock_key TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('QUEUED', 'RUNNING', 'SUCCESS', 'FAILED')),
    payload_json TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT '',
    pid INTEGER,
    log_path TEXT NOT NULL,
    result_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    started_at TEXT,
    heartbeat_at TEXT,
    finished_at TEXT
);
DROP INDEX IF EXISTS uq_background_jobs_active_lock;
CREATE UNIQUE INDEX IF NOT EXISTS uq_background_jobs_running_lock
ON background_jobs(lock_key) WHERE status = 'RUNNING';
CREATE UNIQUE INDEX IF NOT EXISTS uq_background_jobs_active_kind
ON background_jobs(kind) WHERE status IN ('QUEUED', 'RUNNING');
CREATE TABLE IF NOT EXISTS industry_classifications (
    symbol TEXT PRIMARY KEY,
    industry TEXT,
    industry_code TEXT,
    industry_level TEXT,
    industry_standard TEXT,
    industry_source TEXT,
    industry_as_of TEXT,
    status TEXT NOT NULL CHECK(status IN ('ok', 'empty', 'error')),
    error TEXT NOT NULL DEFAULT '',
    attempted_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS industry_overrides (
    symbol TEXT PRIMARY KEY,
    industry TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS kv_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

_INDUSTRY_UPSERT_SQL = """
INSERT INTO industry_classifications (
    symbol, industry, industry_code, industry_level, industry_standard,
    industry_source, industry_as_of, status, error, attempted_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(symbol) DO UPDATE SET
    industry = CASE
        WHEN excluded.status = 'ok' AND NULLIF(TRIM(excluded.industry), '') IS NOT NULL
             AND (industry_classifications.industry_as_of IS NULL
                  OR (excluded.industry_as_of IS NOT NULL
                      AND excluded.industry_as_of >= industry_classifications.industry_as_of))
        THEN excluded.industry ELSE industry_classifications.industry END,
    industry_code = CASE
        WHEN excluded.status = 'ok' AND NULLIF(TRIM(excluded.industry), '') IS NOT NULL
             AND (industry_classifications.industry_as_of IS NULL
                  OR (excluded.industry_as_of IS NOT NULL
                      AND excluded.industry_as_of >= industry_classifications.industry_as_of))
        THEN excluded.industry_code ELSE industry_classifications.industry_code END,
    industry_level = CASE
        WHEN excluded.status = 'ok' AND NULLIF(TRIM(excluded.industry), '') IS NOT NULL
             AND (industry_classifications.industry_as_of IS NULL
                  OR (excluded.industry_as_of IS NOT NULL
                      AND excluded.industry_as_of >= industry_classifications.industry_as_of))
        THEN excluded.industry_level ELSE industry_classifications.industry_level END,
    industry_standard = CASE
        WHEN excluded.status = 'ok' AND NULLIF(TRIM(excluded.industry), '') IS NOT NULL
             AND (industry_classifications.industry_as_of IS NULL
                  OR (excluded.industry_as_of IS NOT NULL
                      AND excluded.industry_as_of >= industry_classifications.industry_as_of))
        THEN excluded.industry_standard ELSE industry_classifications.industry_standard END,
    industry_source = CASE
        WHEN excluded.status = 'ok' AND NULLIF(TRIM(excluded.industry), '') IS NOT NULL
             AND (industry_classifications.industry_as_of IS NULL
                  OR (excluded.industry_as_of IS NOT NULL
                      AND excluded.industry_as_of >= industry_classifications.industry_as_of))
        THEN excluded.industry_source ELSE industry_classifications.industry_source END,
    industry_as_of = CASE
        WHEN excluded.status = 'ok' AND NULLIF(TRIM(excluded.industry), '') IS NOT NULL
             AND (industry_classifications.industry_as_of IS NULL
                  OR (excluded.industry_as_of IS NOT NULL
                      AND excluded.industry_as_of >= industry_classifications.industry_as_of))
        THEN excluded.industry_as_of ELSE industry_classifications.industry_as_of END,
    status = excluded.status,
    error = excluded.error,
    attempted_at = excluded.attempted_at
"""


def _record_value(record: Mapping[str, object] | object, key: str) -> object | None:
    if isinstance(record, Mapping):
        return record.get(key)
    return getattr(record, key, None)


def _text(value: object | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _iso_text(value: object | None, *, default: str | None = None) -> str | None:
    if value is None or pd.isna(value):
        return default
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return value.isoformat()
    return _text(value) or default


def _normalise_industry_record(
    record: Mapping[str, object] | object,
) -> tuple[
    str,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str,
    str,
    str,
]:
    symbol = _text(_record_value(record, "symbol"))
    if symbol is None:
        raise ValueError("行业分类记录缺少 symbol")
    if symbol.endswith(".0") and symbol[:-2].isdigit():
        symbol = symbol[:-2]
    if symbol.isdigit() and len(symbol) < 6:
        symbol = symbol.zfill(6)

    industry = _text(_record_value(record, "industry"))
    status = (_text(_record_value(record, "status")) or ("ok" if industry else "empty")).lower()
    if status not in {"ok", "empty", "error"}:
        raise ValueError(f"无效的行业分类状态：{status}")
    # “ok”但没有行业名称也不能视为成功，否则可能清空已有有效分类。
    if status == "ok" and industry is None:
        status = "empty"

    attempted_at = _iso_text(
        _record_value(record, "attempted_at"),
        default=datetime.now().isoformat(timespec="seconds"),
    )
    return (
        symbol,
        industry,
        _text(_record_value(record, "industry_code")),
        _text(_record_value(record, "industry_level")),
        _text(_record_value(record, "industry_standard")),
        _text(_record_value(record, "industry_source")),
        _iso_text(_record_value(record, "industry_as_of")),
        status,
        _text(_record_value(record, "error")) or "",
        attempted_at or datetime.now().isoformat(timespec="seconds"),
    )


class StateStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            daily_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(daily_runs)").fetchall()
            }
            if "deployment_id" not in daily_columns:
                connection.execute("ALTER TABLE daily_runs ADD COLUMN deployment_id TEXT")
            if "strategy_name" not in daily_columns:
                connection.execute("ALTER TABLE daily_runs ADD COLUMN strategy_name TEXT")

    def ensure_account(self, initial_cash: float, account: str = "paper") -> None:
        now = datetime.now().isoformat(timespec="seconds")
        with self.connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO account_state(account, cash, updated_at) VALUES (?, ?, ?)",
                (account, initial_cash, now),
            )

    def get_cash(self, account: str = "paper") -> float:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT cash FROM account_state WHERE account = ?", (account,)
            ).fetchone()
        if row is None:
            raise RuntimeError("模拟账户尚未初始化")
        return float(row["cash"])

    def set_cash(self, cash: float, account: str = "paper") -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE account_state SET cash = ?, updated_at = ? WHERE account = ?",
                (cash, datetime.now().isoformat(timespec="seconds"), account),
            )

    def save_strategy(self, name: str, parameters: dict[str, object]) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO strategy_instances(name, parameters_json, updated_at)
                   VALUES (?, ?, ?) ON CONFLICT(name) DO UPDATE SET
                   parameters_json=excluded.parameters_json, updated_at=excluded.updated_at""",
                (
                    name,
                    json.dumps(parameters, ensure_ascii=False, sort_keys=True),
                    datetime.now().isoformat(),
                ),
            )

    def ensure_paper_strategy_deployment(
        self,
        *,
        deployment_id: str,
        strategy_name: str,
        parameters: dict[str, object],
        note: str,
    ) -> dict[str, object]:
        """Create the initial deployment without treating migration as a strategy switch."""
        now = datetime.now().isoformat(timespec="seconds")
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            active = connection.execute(
                """SELECT * FROM paper_strategy_deployments
                   WHERE status='ACTIVE' LIMIT 1"""
            ).fetchone()
            if active is None:
                connection.execute(
                    """INSERT INTO paper_strategy_deployments
                       (deployment_id, strategy_name, parameters_json,
                        source_backtest_run_id, source_data_end_date,
                        source_metrics_json, source_parameters_json, status,
                        requested_at, activated_at, retired_at, note)
                       VALUES (?, ?, ?, NULL, NULL, '{}', '{}', 'ACTIVE',
                               ?, ?, NULL, ?)""",
                    (
                        deployment_id,
                        strategy_name,
                        json.dumps(parameters, ensure_ascii=False, sort_keys=True),
                        now,
                        now,
                        note,
                    ),
                )
                active = connection.execute(
                    """SELECT * FROM paper_strategy_deployments
                       WHERE deployment_id=?""",
                    (deployment_id,),
                ).fetchone()
        if active is None:
            raise RuntimeError("无法初始化模拟盘策略部署")
        return dict(active)

    def active_paper_strategy_deployment(self) -> dict[str, object] | None:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT * FROM paper_strategy_deployments
                   WHERE status='ACTIVE' LIMIT 1"""
            ).fetchone()
        return dict(row) if row else None

    def paper_strategy_deployments(self, limit: int = 50) -> pd.DataFrame:
        with self.connect() as connection:
            return pd.read_sql_query(
                """SELECT * FROM paper_strategy_deployments
                   ORDER BY requested_at DESC LIMIT ?""",
                connection,
                params=(max(1, int(limit)),),
            )

    def promote_paper_strategy(
        self,
        *,
        deployment_id: str,
        strategy_name: str,
        parameters: dict[str, object],
        source_backtest_run_id: str,
        source_data_end_date: str,
        source_metrics: dict[str, object],
        source_parameters: dict[str, object],
        note: str,
    ) -> dict[str, object]:
        """Atomically promote one validated backtest while the paper account is idle."""
        now = datetime.now().isoformat(timespec="seconds")
        parameters_json = json.dumps(parameters, ensure_ascii=False, sort_keys=True)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            active_job = connection.execute(
                """SELECT job_id, kind FROM background_jobs
                   WHERE status IN ('QUEUED', 'RUNNING')
                   ORDER BY rowid LIMIT 1"""
            ).fetchone()
            if active_job is not None:
                raise RuntimeError(
                    f"后台任务 {active_job['job_id']}（{active_job['kind']}）"
                    "正在运行或排队，暂不能切换模拟盘策略"
                )
            existing_pending = connection.execute(
                """SELECT deployment_id FROM paper_strategy_deployments
                   WHERE status='PENDING' LIMIT 1"""
            ).fetchone()
            if existing_pending is not None:
                raise RuntimeError(
                    f"已有待生效策略部署 {existing_pending['deployment_id']}，"
                    "不能重复提交晋级"
                )
            pending_count = connection.execute(
                "SELECT COUNT(*) AS count FROM orders WHERE status='PENDING'"
            ).fetchone()["count"]
            active = connection.execute(
                """SELECT * FROM paper_strategy_deployments
                   WHERE status='ACTIVE' LIMIT 1"""
            ).fetchone()
            if (
                active is not None
                and active["source_backtest_run_id"] == source_backtest_run_id
            ):
                raise RuntimeError("所选回测已经是当前模拟盘策略的来源")
            status = "PENDING" if pending_count else "ACTIVE"
            activated_at = None if pending_count else now
            if status == "ACTIVE":
                connection.execute(
                    """UPDATE paper_strategy_deployments
                       SET status='RETIRED', retired_at=? WHERE status='ACTIVE'""",
                    (now,),
                )
            connection.execute(
                """INSERT INTO paper_strategy_deployments
                   (deployment_id, strategy_name, parameters_json,
                    source_backtest_run_id, source_data_end_date,
                    source_metrics_json, source_parameters_json, status,
                    requested_at, activated_at, retired_at, note)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)""",
                (
                    deployment_id,
                    strategy_name,
                    parameters_json,
                    source_backtest_run_id,
                    source_data_end_date,
                    json.dumps(source_metrics, ensure_ascii=False, sort_keys=True),
                    json.dumps(source_parameters, ensure_ascii=False, sort_keys=True),
                    status,
                    now,
                    activated_at,
                    note,
                ),
            )
            promoted = connection.execute(
                """SELECT * FROM paper_strategy_deployments
                   WHERE deployment_id=?""",
                (deployment_id,),
            ).fetchone()
        if promoted is None:
            raise RuntimeError("模拟盘策略晋级失败")
        return dict(promoted)

    def activate_pending_paper_strategy(self) -> dict[str, object] | None:
        """Activate a scheduled deployment once every old-strategy order is resolved."""
        now = datetime.now().isoformat(timespec="seconds")
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            pending_deployment = connection.execute(
                """SELECT * FROM paper_strategy_deployments
                   WHERE status='PENDING' LIMIT 1"""
            ).fetchone()
            if pending_deployment is None:
                return None
            pending_orders = connection.execute(
                "SELECT COUNT(*) AS count FROM orders WHERE status='PENDING'"
            ).fetchone()["count"]
            if pending_orders:
                return None
            connection.execute(
                """UPDATE paper_strategy_deployments
                   SET status='RETIRED', retired_at=? WHERE status='ACTIVE'""",
                (now,),
            )
            connection.execute(
                """UPDATE paper_strategy_deployments
                   SET status='ACTIVE', activated_at=?
                   WHERE deployment_id=? AND status='PENDING'""",
                (now, pending_deployment["deployment_id"]),
            )
            activated = connection.execute(
                """SELECT * FROM paper_strategy_deployments
                   WHERE deployment_id=?""",
                (pending_deployment["deployment_id"],),
            ).fetchone()
        return dict(activated) if activated else None

    def pending_paper_strategy_deployment(self) -> dict[str, object] | None:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT * FROM paper_strategy_deployments
                   WHERE status='PENDING' LIMIT 1"""
            ).fetchone()
        return dict(row) if row else None

    def add_signals(self, signals: list[Signal]) -> int:
        inserted = 0
        with self.connect() as connection:
            for signal in signals:
                cursor = connection.execute(
                    """INSERT OR IGNORE INTO signals
                       (strategy, symbol, signal_date, value, target_weight, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        signal.strategy,
                        signal.symbol,
                        signal.signal_date.isoformat(),
                        signal.value,
                        signal.target_weight,
                        datetime.now().isoformat(timespec="seconds"),
                    ),
                )
                inserted += cursor.rowcount
        return inserted

    def latest_signal_values(self, strategy: str) -> dict[str, int]:
        query = """
            SELECT s.symbol, s.value FROM signals s
            JOIN (SELECT symbol, MAX(signal_date) AS max_date FROM signals
                  WHERE strategy = ? GROUP BY symbol) latest
              ON s.symbol = latest.symbol AND s.signal_date = latest.max_date
            WHERE s.strategy = ?
        """
        with self.connect() as connection:
            rows = connection.execute(query, (strategy, strategy)).fetchall()
        return {row["symbol"]: int(row["value"]) for row in rows}

    def latest_signal_states(self, strategy: str) -> dict[str, tuple[int, float]]:
        query = """
            SELECT s.symbol, s.value, s.target_weight FROM signals s
            JOIN (SELECT symbol, MAX(signal_date) AS max_date FROM signals
                  WHERE strategy = ? GROUP BY symbol) latest
              ON s.symbol = latest.symbol AND s.signal_date = latest.max_date
            WHERE s.strategy = ?
        """
        with self.connect() as connection:
            rows = connection.execute(query, (strategy, strategy)).fetchall()
        return {row["symbol"]: (int(row["value"]), float(row["target_weight"])) for row in rows}

    def add_orders(self, orders: list[Order]) -> int:
        inserted = 0
        now = datetime.now().isoformat(timespec="seconds")
        with self.connect() as connection:
            for order in orders:
                cursor = connection.execute(
                    """INSERT OR IGNORE INTO orders
                       (order_id, strategy, symbol, signal_date, side, target_weight,
                        quantity, status, message, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        order.order_id,
                        order.strategy,
                        order.symbol,
                        order.signal_date.isoformat(),
                        order.side.value,
                        order.target_weight,
                        order.quantity,
                        order.status.value,
                        order.message,
                        now,
                        now,
                    ),
                )
                inserted += cursor.rowcount
        return inserted

    def existing_order_ids(self, order_ids: list[str]) -> set[str]:
        if not order_ids:
            return set()
        placeholders = ",".join("?" for _ in order_ids)
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT order_id FROM orders WHERE order_id IN ({placeholders})",
                order_ids,
            ).fetchall()
        return {str(row["order_id"]) for row in rows}

    def pending_orders(self) -> list[Order]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM orders WHERE status = 'PENDING' ORDER BY signal_date, side DESC"
            ).fetchall()
        return [
            Order(
                order_id=row["order_id"],
                strategy=row["strategy"],
                symbol=row["symbol"],
                signal_date=date.fromisoformat(row["signal_date"]),
                side=Side(row["side"]),
                target_weight=float(row["target_weight"]),
                quantity=int(row["quantity"]),
                status=OrderStatus(row["status"]),
                message=row["message"],
            )
            for row in rows
        ]

    def update_order(self, order: Order) -> None:
        with self.connect() as connection:
            connection.execute(
                """UPDATE orders SET quantity=?, status=?, message=?, updated_at=?
                   WHERE order_id=?""",
                (
                    order.quantity,
                    order.status.value,
                    order.message,
                    datetime.now().isoformat(timespec="seconds"),
                    order.order_id,
                ),
            )

    def add_fill(self, fill: Fill) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO fills
                   (fill_id, order_id, symbol, trade_date, side, quantity, price,
                    commission, stamp_duty, slippage)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    fill.fill_id,
                    fill.order_id,
                    fill.symbol,
                    fill.trade_date.isoformat(),
                    fill.side.value,
                    fill.quantity,
                    fill.price,
                    fill.commission,
                    fill.stamp_duty,
                    fill.slippage,
                ),
            )

    def settle_fill(
        self,
        fill: Fill,
        order: Order,
        position: Position,
        cash: float,
        account: str = "paper",
    ) -> None:
        """在同一事务中结算现金、持仓、订单和成交。"""
        with self.connect() as connection:
            connection.execute(
                "UPDATE account_state SET cash = ?, updated_at = ? WHERE account = ?",
                (cash, datetime.now().isoformat(timespec="seconds"), account),
            )
            if position.quantity <= 0:
                connection.execute("DELETE FROM positions WHERE symbol = ?", (position.symbol,))
            else:
                connection.execute(
                    """INSERT INTO positions(symbol, quantity, average_cost, last_buy_date)
                       VALUES (?, ?, ?, ?) ON CONFLICT(symbol) DO UPDATE SET
                       quantity=excluded.quantity, average_cost=excluded.average_cost,
                       last_buy_date=excluded.last_buy_date""",
                    (
                        position.symbol,
                        position.quantity,
                        position.average_cost,
                        position.last_buy_date.isoformat() if position.last_buy_date else None,
                    ),
                )
            connection.execute(
                """UPDATE orders SET side=?, quantity=?, status=?, message=?, updated_at=?
                   WHERE order_id=?""",
                (
                    order.side.value,
                    order.quantity,
                    order.status.value,
                    order.message,
                    datetime.now().isoformat(timespec="seconds"),
                    order.order_id,
                ),
            )
            connection.execute(
                """INSERT OR IGNORE INTO fills
                   (fill_id, order_id, symbol, trade_date, side, quantity, price,
                    commission, stamp_duty, slippage)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    fill.fill_id,
                    fill.order_id,
                    fill.symbol,
                    fill.trade_date.isoformat(),
                    fill.side.value,
                    fill.quantity,
                    fill.price,
                    fill.commission,
                    fill.stamp_duty,
                    fill.slippage,
                ),
            )

    def positions(self) -> dict[str, Position]:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM positions WHERE quantity > 0").fetchall()
        return {
            row["symbol"]: Position(
                symbol=row["symbol"],
                quantity=int(row["quantity"]),
                average_cost=float(row["average_cost"]),
                last_buy_date=(
                    date.fromisoformat(row["last_buy_date"]) if row["last_buy_date"] else None
                ),
            )
            for row in rows
        }

    def save_position(self, position: Position) -> None:
        with self.connect() as connection:
            if position.quantity <= 0:
                connection.execute("DELETE FROM positions WHERE symbol = ?", (position.symbol,))
                return
            connection.execute(
                """INSERT INTO positions(symbol, quantity, average_cost, last_buy_date)
                   VALUES (?, ?, ?, ?) ON CONFLICT(symbol) DO UPDATE SET
                   quantity=excluded.quantity, average_cost=excluded.average_cost,
                   last_buy_date=excluded.last_buy_date""",
                (
                    position.symbol,
                    position.quantity,
                    position.average_cost,
                    position.last_buy_date.isoformat() if position.last_buy_date else None,
                ),
            )

    def save_snapshot(self, snapshot: PortfolioSnapshot, account: str = "paper") -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO portfolio_snapshots
                   (account, snapshot_date, cash, market_value, equity) VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(account, snapshot_date) DO UPDATE SET cash=excluded.cash,
                   market_value=excluded.market_value, equity=excluded.equity""",
                (
                    account,
                    snapshot.snapshot_date.isoformat(),
                    snapshot.cash,
                    snapshot.market_value,
                    snapshot.equity,
                ),
            )

    def peak_equity(self, account: str = "paper") -> float | None:
        """返回账户历史最高权益，供模拟盘组合回撤风控使用。"""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT MAX(equity) AS peak FROM portfolio_snapshots WHERE account = ?",
                (account,),
            ).fetchone()
        return float(row["peak"]) if row and row["peak"] is not None else None

    def save_backtest(
        self,
        run_id: str,
        data_end_date: date,
        parameters: dict[str, object],
        metrics: dict[str, float],
        output_dir: Path,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO backtest_runs
                   (run_id, created_at, data_end_date, parameters_json, metrics_json, output_dir)
                   VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(run_id) DO UPDATE SET
                   created_at=excluded.created_at, metrics_json=excluded.metrics_json,
                   output_dir=excluded.output_dir""",
                (
                    run_id,
                    datetime.now().isoformat(timespec="seconds"),
                    data_end_date.isoformat(),
                    json.dumps(parameters, ensure_ascii=False, sort_keys=True),
                    json.dumps(metrics, ensure_ascii=False, sort_keys=True),
                    str(output_dir),
                ),
            )

    def backtest_run(self, run_id: str) -> dict[str, object] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM backtest_runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
        return dict(row) if row else None

    def record_daily_run(
        self,
        run_date: date,
        status: str,
        message: str,
        *,
        deployment_id: str = "",
        strategy_name: str = "",
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO daily_runs
                   (run_date, status, message, updated_at, deployment_id, strategy_name)
                   VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(run_date) DO UPDATE SET
                   status=excluded.status, message=excluded.message,
                   updated_at=excluded.updated_at,
                   deployment_id=excluded.deployment_id,
                   strategy_name=excluded.strategy_name""",
                (
                    run_date.isoformat(),
                    status,
                    message,
                    datetime.now().isoformat(timespec="seconds"),
                    deployment_id or None,
                    strategy_name or None,
                ),
            )

    def get_state(self, key: str, default: str | None = None) -> str | None:
        with self.connect() as connection:
            row = connection.execute("SELECT value FROM kv_state WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def set_state(self, key: str, value: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO kv_state(key, value, updated_at) VALUES (?, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET value=excluded.value,
                   updated_at=excluded.updated_at""",
                (key, value, datetime.now().isoformat(timespec="seconds")),
            )

    def upsert_industry_classification(
        self,
        record: Mapping[str, object] | object,
    ) -> None:
        """保存一次行业查询结果，失败查询不会清空上一次有效分类。"""
        payload = _normalise_industry_record(record)
        with self.connect() as connection:
            connection.execute(_INDUSTRY_UPSERT_SQL, payload)

    def upsert_industry_classifications(
        self,
        records: list[dict[str, object]],
    ) -> int:
        """在单一事务中批量保存行业查询结果。"""
        if not records:
            return 0
        payloads = [_normalise_industry_record(record) for record in records]
        with self.connect() as connection:
            connection.executemany(_INDUSTRY_UPSERT_SQL, payloads)
        return len(payloads)

    def industry_classifications(
        self,
        symbols: list[str] | tuple[str, ...] | None = None,
    ) -> pd.DataFrame:
        parameters: list[object] = []
        where = ""
        if symbols is not None:
            normalised = []
            for symbol in symbols:
                text = str(symbol).strip()
                normalised.append(text.zfill(6) if text.isdigit() else text)
            normalised = list(dict.fromkeys(normalised))
            if len(normalised) > 800:
                chunks = [
                    self.industry_classifications(tuple(normalised[offset : offset + 800]))
                    for offset in range(0, len(normalised), 800)
                ]
                return pd.concat(chunks, ignore_index=True).sort_values(
                    "symbol", ignore_index=True
                )
            if not normalised:
                where = "WHERE 0"
            else:
                placeholders = ",".join("?" for _ in normalised)
                where = f"WHERE symbol IN ({placeholders})"
                parameters.extend(normalised)
        with self.connect() as connection:
            return pd.read_sql_query(
                f"""SELECT symbol, industry, industry_code, industry_level,
                           industry_standard, industry_source, industry_as_of,
                           status, error, attempted_at
                    FROM industry_classifications {where}
                    ORDER BY symbol""",
                connection,
                params=parameters,
            )

    def set_industry_override(self, symbol: str, industry: str, note: str = "") -> None:
        symbol = str(symbol).strip()
        if symbol.endswith(".0") and symbol[:-2].isdigit():
            symbol = symbol[:-2]
        if symbol.isdigit():
            symbol = symbol.zfill(6)
        industry = industry.strip()
        if not symbol:
            raise ValueError("股票代码不能为空")
        if not industry:
            raise ValueError("人工行业分类不能为空")
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO industry_overrides(symbol, industry, note, updated_at)
                   VALUES (?, ?, ?, ?) ON CONFLICT(symbol) DO UPDATE SET
                   industry=excluded.industry, note=excluded.note,
                   updated_at=excluded.updated_at""",
                (
                    symbol,
                    industry,
                    note.strip(),
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )

    def remove_industry_override(self, symbol: str) -> bool:
        symbol = str(symbol).strip()
        if symbol.endswith(".0") and symbol[:-2].isdigit():
            symbol = symbol[:-2]
        if symbol.isdigit():
            symbol = symbol.zfill(6)
        with self.connect() as connection:
            cursor = connection.execute(
                "DELETE FROM industry_overrides WHERE symbol = ?",
                (symbol,),
            )
        return cursor.rowcount > 0

    def industry_overrides(self) -> pd.DataFrame:
        with self.connect() as connection:
            return pd.read_sql_query(
                """SELECT symbol, industry, note, updated_at
                    FROM industry_overrides ORDER BY symbol""",
                connection,
            )

    def latest_screen_date(self) -> date | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT MAX(as_of_date) AS latest FROM screen_runs WHERE status = 'SUCCESS'"
            ).fetchone()
        return date.fromisoformat(row["latest"]) if row and row["latest"] else None

    def save_screening_run(
        self,
        run_id: str,
        as_of_date: date,
        status: str,
        results: pd.DataFrame,
        output_dir: Path,
        warnings: list[str],
    ) -> None:
        eligible_count = int(results.get("eligible", pd.Series(dtype=bool)).sum())
        selected_count = int(results.get("selected", pd.Series(dtype=bool)).sum())
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO screen_runs
                   (run_id, as_of_date, status, universe_size, eligible_count,
                    selected_count, output_dir, warnings_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(run_id) DO UPDATE SET status=excluded.status,
                   eligible_count=excluded.eligible_count,
                   selected_count=excluded.selected_count,
                   output_dir=excluded.output_dir, warnings_json=excluded.warnings_json""",
                (
                    run_id,
                    as_of_date.isoformat(),
                    status,
                    len(results),
                    eligible_count,
                    selected_count,
                    str(output_dir),
                    json.dumps(warnings, ensure_ascii=False),
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
            connection.execute("DELETE FROM screen_results WHERE run_id = ?", (run_id,))
            for row in results.to_dict("records"):
                factor_payload = {
                    key: value
                    for key, value in row.items()
                    if key
                    not in {
                        "symbol",
                        "rank",
                        "selected",
                        "total_score",
                        "industry",
                    }
                    and not pd.isna(value)
                }
                connection.execute(
                    """INSERT INTO screen_results
                       (run_id, symbol, rank, selected, total_score, industry, factors_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        run_id,
                        row["symbol"],
                        int(row["rank"]) if pd.notna(row.get("rank")) else None,
                        int(bool(row.get("selected", False))),
                        float(row["total_score"]) if pd.notna(row.get("total_score")) else None,
                        row.get("industry"),
                        json.dumps(factor_payload, ensure_ascii=False, default=str),
                    ),
                )

    def create_background_job(
        self,
        *,
        job_id: str,
        kind: str,
        lock_key: str,
        payload: dict[str, object],
        log_path: Path,
    ) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        try:
            with self.connect() as connection:
                connection.execute(
                    """INSERT INTO background_jobs
                       (job_id, kind, lock_key, status, payload_json, progress, message,
                        log_path, result_json, created_at, heartbeat_at)
                       VALUES (?, ?, ?, 'QUEUED', ?, 0, ?, ?, '{}', ?, ?)""",
                    (
                        job_id,
                        kind,
                        lock_key,
                        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str),
                        "等待后台进程启动",
                        str(log_path),
                        now,
                        now,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            with self.connect() as connection:
                active = connection.execute(
                    """SELECT job_id, kind, status FROM background_jobs
                       WHERE kind = ? AND status IN ('QUEUED', 'RUNNING')
                       ORDER BY created_at DESC LIMIT 1""",
                    (kind,),
                ).fetchone()
            if active:
                raise RuntimeError(
                    f"已有同类后台任务 {active['job_id']}"
                    f"（{active['kind']}，{active['status']}）正在运行或排队"
                ) from exc
            raise

    def claim_background_job(self, job_id: str, pid: int) -> bool:
        now = datetime.now().isoformat(timespec="seconds")
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            job = connection.execute(
                """SELECT rowid, lock_key, status FROM background_jobs
                   WHERE job_id=?""",
                (job_id,),
            ).fetchone()
            if job is None or job["status"] != "QUEUED":
                return False
            running = connection.execute(
                """SELECT 1 FROM background_jobs
                   WHERE lock_key=? AND status='RUNNING' LIMIT 1""",
                (job["lock_key"],),
            ).fetchone()
            first_queued = connection.execute(
                """SELECT job_id FROM background_jobs
                   WHERE lock_key=? AND status='QUEUED'
                   ORDER BY rowid ASC LIMIT 1""",
                (job["lock_key"],),
            ).fetchone()
            if running is not None or first_queued is None or first_queued["job_id"] != job_id:
                return False
            cursor = connection.execute(
                """UPDATE background_jobs SET status='RUNNING', pid=?, progress=0,
                   message='后台任务已启动', started_at=?, heartbeat_at=?
                   WHERE job_id=? AND status='QUEUED'""",
                (pid, now, now, job_id),
            )
        return cursor.rowcount == 1

    def touch_queued_background_job(self, job_id: str, message: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """UPDATE background_jobs SET message=?, heartbeat_at=?
                   WHERE job_id=? AND status='QUEUED'""",
                (
                    message,
                    datetime.now().isoformat(timespec="seconds"),
                    job_id,
                ),
            )

    def attach_background_job_process(self, job_id: str, pid: int) -> None:
        with self.connect() as connection:
            connection.execute(
                """UPDATE background_jobs SET pid=?, heartbeat_at=?
                   WHERE job_id=? AND status='QUEUED'""",
                (
                    pid,
                    datetime.now().isoformat(timespec="seconds"),
                    job_id,
                ),
            )

    def update_background_job(
        self,
        job_id: str,
        *,
        progress: float | None = None,
        message: str | None = None,
    ) -> None:
        assignments = ["heartbeat_at=?"]
        values: list[object] = [datetime.now().isoformat(timespec="seconds")]
        if progress is not None:
            assignments.append("progress=?")
            values.append(max(0.0, min(float(progress), 1.0)))
        if message is not None:
            assignments.append("message=?")
            values.append(message)
        values.append(job_id)
        with self.connect() as connection:
            connection.execute(
                f"UPDATE background_jobs SET {', '.join(assignments)} "
                "WHERE job_id=? AND status='RUNNING'",
                values,
            )

    def finish_background_job(
        self,
        job_id: str,
        *,
        result: dict[str, object],
        message: str,
    ) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        with self.connect() as connection:
            connection.execute(
                """UPDATE background_jobs SET status='SUCCESS', progress=1,
                   message=?, result_json=?, heartbeat_at=?, finished_at=?
                   WHERE job_id=? AND status='RUNNING'""",
                (
                    message,
                    json.dumps(result, ensure_ascii=False, sort_keys=True, default=str),
                    now,
                    now,
                    job_id,
                ),
            )

    def fail_background_job(self, job_id: str, message: str) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        with self.connect() as connection:
            connection.execute(
                """UPDATE background_jobs SET status='FAILED', message=?,
                   heartbeat_at=?, finished_at=? WHERE job_id=?
                   AND status IN ('QUEUED', 'RUNNING')""",
                (message, now, now, job_id),
            )

    def background_job(self, job_id: str) -> dict[str, object] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM background_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        return dict(row) if row else None

    def background_jobs(
        self,
        *,
        kinds: list[str] | tuple[str, ...] | None = None,
        limit: int = 50,
    ) -> pd.DataFrame:
        parameters: list[object] = []
        where = ""
        if kinds:
            placeholders = ",".join("?" for _ in kinds)
            where = f"WHERE kind IN ({placeholders})"
            parameters.extend(kinds)
        parameters.append(limit)
        with self.connect() as connection:
            return pd.read_sql_query(
                f"""SELECT * FROM background_jobs {where}
                    ORDER BY created_at DESC LIMIT ?""",
                connection,
                params=parameters,
            )

    def active_background_job(self, lock_key: str) -> dict[str, object] | None:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT * FROM background_jobs WHERE lock_key=?
                   AND status IN ('QUEUED', 'RUNNING')
                   ORDER BY CASE status WHEN 'RUNNING' THEN 0 ELSE 1 END,
                            rowid ASC LIMIT 1""",
                (lock_key,),
            ).fetchone()
        return dict(row) if row else None

    def table(self, name: str, limit: int = 500) -> pd.DataFrame:
        allowed = {
            "signals",
            "orders",
            "fills",
            "positions",
            "portfolio_snapshots",
            "paper_strategy_deployments",
            "backtest_runs",
            "daily_runs",
            "screen_runs",
            "screen_results",
            "background_jobs",
            "industry_classifications",
            "industry_overrides",
        }
        if name not in allowed:
            raise ValueError("不允许读取未知表")
        with self.connect() as connection:
            query = f"SELECT * FROM {name} ORDER BY rowid DESC LIMIT ?"
            return pd.read_sql_query(query, connection, params=(limit,))

    def backup(self, destination_dir: Path) -> Path:
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"aquant-{datetime.now():%Y%m%d-%H%M%S}.sqlite3"
        with self.connect() as source, sqlite3.connect(destination) as target:
            source.backup(target)
        return destination
