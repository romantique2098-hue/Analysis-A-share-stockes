from __future__ import annotations

import math

from aquant.config import CostConfig
from aquant.models import Side


def limit_target_weights(
    current_weights: dict[str, float],
    desired_weights: dict[str, float],
    *,
    minimum_change: float,
    max_turnover: float,
    force_exit: bool = False,
) -> dict[str, float]:
    """生成一次可执行的目标权重；减仓优先，新增风险仓位受换手预算约束。"""
    symbols = sorted(set(current_weights) | set(desired_weights))
    current = {symbol: max(float(current_weights.get(symbol, 0.0)), 0.0) for symbol in symbols}
    desired = {symbol: max(float(desired_weights.get(symbol, 0.0)), 0.0) for symbol in symbols}
    if force_exit:
        return dict.fromkeys(symbols, 0.0)

    targets = dict(current)
    for symbol in symbols:
        old = current[symbol]
        new = desired[symbol]
        difference = new - old
        if difference < 0 and (new == 0 or abs(difference) >= minimum_change):
            # 退出和风险减仓优先；微小的仓位漂移不触发调仓。
            targets[symbol] = new

    reductions = sum(max(current[symbol] - targets[symbol], 0.0) for symbol in symbols)
    buy_budget = max(max_turnover - reductions, 0.0)
    increases = {
        symbol: desired[symbol] - current[symbol]
        for symbol in symbols
        if desired[symbol] - current[symbol] >= minimum_change
    }
    total_increase = sum(increases.values())
    scale = min(1.0, buy_budget / total_increase) if total_increase > 0 else 0.0
    for symbol, increase in increases.items():
        targets[symbol] = current[symbol] + increase * scale
    return targets


def round_down_lot(quantity: float, lot_size: int) -> int:
    if quantity <= 0:
        return 0
    return int(math.floor(quantity / lot_size) * lot_size)


def execution_price(open_price: float, side: Side, costs: CostConfig) -> tuple[float, float]:
    rate = costs.slippage_bps / 10_000
    multiplier = 1 + rate if side == Side.BUY else 1 - rate
    price = open_price * multiplier
    return price, abs(price - open_price)


def transaction_fees(notional: float, side: Side, costs: CostConfig) -> tuple[float, float]:
    if notional <= 0:
        return 0.0, 0.0
    commission = max(notional * costs.commission_rate, costs.minimum_commission)
    stamp_duty = notional * costs.stamp_duty_rate if side == Side.SELL else 0.0
    return commission, stamp_duty
