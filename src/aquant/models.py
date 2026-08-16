from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import Any

import pandas as pd


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(StrEnum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    SKIPPED = "SKIPPED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class PriceBar:
    symbol: str
    date: date
    open: float
    close: float
    high: float
    low: float
    volume: float
    amount: float


@dataclass(frozen=True)
class Signal:
    strategy: str
    symbol: str
    signal_date: date
    value: int
    target_weight: float


@dataclass
class Order:
    order_id: str
    strategy: str
    symbol: str
    signal_date: date
    side: Side
    target_weight: float
    quantity: int = 0
    status: OrderStatus = OrderStatus.PENDING
    message: str = ""


@dataclass(frozen=True)
class Fill:
    fill_id: str
    order_id: str
    symbol: str
    trade_date: date
    side: Side
    quantity: int
    price: float
    commission: float
    stamp_duty: float
    slippage: float


@dataclass
class Position:
    symbol: str
    quantity: int
    average_cost: float
    last_buy_date: date | None = None


@dataclass(frozen=True)
class PortfolioSnapshot:
    snapshot_date: date
    cash: float
    market_value: float
    equity: float


@dataclass
class BacktestResult:
    run_id: str
    metrics: dict[str, float]
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    signals: pd.DataFrame
    parameters: dict[str, Any] = field(default_factory=dict)
