from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from aquant.config import StrategyConfig

STRATEGY_NAMES = (
    "ma_cross",
    "trend_breakout",
    "momentum",
    "rsi_reversion",
    "mean_reversion",
    "ensemble_equal",
    "ensemble_risk",
    "first_board_pullback",
)
BASE_COLUMNS = ["symbol", "date", "close"]


class Strategy(Protocol):
    name: str

    def generate_signals(
        self, history: pd.DataFrame, parameters: dict[str, object] | None = None
    ) -> pd.DataFrame: ...


def _prepare_history(history: pd.DataFrame) -> pd.DataFrame:
    required = set(BASE_COLUMNS)
    if missing := required - set(history.columns):
        raise ValueError(f"策略输入缺少字段：{sorted(missing)}")
    data = history.copy()
    data["symbol"] = data["symbol"].astype(str).str.zfill(6)
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    if data[list(required)].isna().any().any() or data["close"].le(0).any():
        raise ValueError("策略输入包含无效代码、日期或收盘价")
    return (
        data.sort_values(["symbol", "date"])
        .drop_duplicates(["symbol", "date"], keep="last")
        .reset_index(drop=True)
    )


def _calculate_rsi(data: pd.DataFrame, window: int) -> pd.Series:
    delta = data.groupby("symbol", sort=False)["close"].diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = gains.groupby(data["symbol"], sort=False).transform(
        lambda values: values.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    )
    average_loss = losses.groupby(data["symbol"], sort=False).transform(
        lambda values: values.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    )
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + relative_strength)
    rsi.loc[average_loss.eq(0) & average_gain.gt(0)] = 100.0
    rsi.loc[average_loss.eq(0) & average_gain.eq(0)] = 50.0
    return rsi


class MovingAverageCrossStrategy:
    name = "ma_cross"

    def __init__(
        self,
        short_window: int = 5,
        long_window: int = 20,
        signal_buffer: float = 0.0,
        confirmation_days: int = 1,
    ) -> None:
        if (
            short_window < 2
            or long_window <= short_window
            or not 0 <= signal_buffer < 0.2
            or not 1 <= confirmation_days <= 20
        ):
            raise ValueError(
                "均线参数不合法：短周期至少为 2 且小于长周期，"
                "缓冲带须在 [0, 20%)，确认天数须在 1 到 20 之间"
            )
        self.short_window = short_window
        self.long_window = long_window
        self.signal_buffer = signal_buffer
        self.confirmation_days = confirmation_days

    def generate_signals(
        self, history: pd.DataFrame, parameters: dict[str, object] | None = None
    ) -> pd.DataFrame:
        parameters = parameters or {}
        short = int(parameters.get("short_window", self.short_window))
        long = int(parameters.get("long_window", self.long_window))
        buffer = float(parameters.get("ma_signal_buffer", self.signal_buffer))
        confirmation = int(parameters.get("ma_confirmation_days", self.confirmation_days))
        if (
            short < 2
            or long <= short
            or not 0 <= buffer < 0.2
            or not 1 <= confirmation <= 20
        ):
            raise ValueError(
                "均线参数不合法：短周期至少为 2 且小于长周期，"
                "缓冲带须在 [0, 20%)，确认天数须在 1 到 20 之间"
            )
        data = _prepare_history(history)
        grouped_close = data.groupby("symbol", sort=False)["close"]
        data["ma_short"] = grouped_close.transform(
            lambda values: values.rolling(short, min_periods=short).mean()
        )
        data["ma_long"] = grouped_close.transform(
            lambda values: values.rolling(long, min_periods=long).mean()
        )
        data["ma_spread"] = data["ma_short"] / data["ma_long"] - 1
        data["signal"] = 0
        for _, indices in data.groupby("symbol", sort=False).groups.items():
            holding = 0
            entry_streak = 0
            exit_streak = 0
            for index in indices:
                spread = data.loc[index, "ma_spread"]
                if pd.isna(spread):
                    entry_streak = 0
                    exit_streak = 0
                elif holding == 0:
                    entry_streak = entry_streak + 1 if spread > buffer else 0
                    if entry_streak >= confirmation:
                        holding = 1
                        entry_streak = 0
                else:
                    exit_streak = exit_streak + 1 if spread <= -buffer else 0
                    if exit_streak >= confirmation:
                        holding = 0
                        exit_streak = 0
                data.loc[index, "signal"] = holding
        data["signal"] = data["signal"].astype("int8")
        return data[[*BASE_COLUMNS, "ma_short", "ma_long", "ma_spread", "signal"]]


class TrendBreakoutStrategy:
    """均线/突破入场，均线转弱或 ATR 移动止损退出。"""

    name = "trend_breakout"

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    def generate_signals(
        self, history: pd.DataFrame, parameters: dict[str, object] | None = None
    ) -> pd.DataFrame:
        if missing := {"high", "low"} - set(history.columns):
            raise ValueError(f"趋势突破策略缺少字段：{sorted(missing)}")
        data = _prepare_history(history)
        grouped = data.groupby("symbol", sort=False)
        data["ma_short"] = grouped["close"].transform(
            lambda values: values.rolling(
                self.config.short_window, min_periods=self.config.short_window
            ).mean()
        )
        data["ma_long"] = grouped["close"].transform(
            lambda values: values.rolling(
                self.config.long_window, min_periods=self.config.long_window
            ).mean()
        )
        data["breakout_high"] = grouped["high"].transform(
            lambda values: (
                values.shift(1)
                .rolling(
                    self.config.breakout_window,
                    min_periods=self.config.breakout_window,
                )
                .max()
            )
        )
        previous_close = grouped["close"].shift(1)
        true_range = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - previous_close).abs(),
                (data["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        data["atr"] = true_range.groupby(data["symbol"], sort=False).transform(
            lambda values: values.rolling(
                self.config.atr_window, min_periods=self.config.atr_window
            ).mean()
        )
        data["signal"] = 0
        data["trailing_peak"] = np.nan
        data["atr_stop"] = np.nan
        for _, indices in data.groupby("symbol", sort=False).groups.items():
            holding = 0
            peak = np.nan
            for index in indices:
                row = data.loc[index]
                breakout = pd.notna(row["breakout_high"]) and row["close"] > row["breakout_high"]
                ma_up = pd.notna(row["ma_long"]) and row["ma_short"] > row["ma_long"]
                if holding == 0 and (breakout or ma_up):
                    holding = 1
                    peak = row["close"]
                elif holding == 1:
                    peak = max(float(peak), row["close"])
                    stop = peak - self.config.atr_stop_multiplier * row["atr"]
                    ma_down = pd.notna(row["ma_long"]) and row["ma_short"] < row["ma_long"]
                    if (pd.notna(stop) and row["close"] < stop) or ma_down:
                        holding = 0
                        peak = np.nan
                data.loc[index, "signal"] = holding
                data.loc[index, "trailing_peak"] = peak
                if holding and pd.notna(row["atr"]):
                    data.loc[index, "atr_stop"] = (
                        peak - self.config.atr_stop_multiplier * row["atr"]
                    )
        data["signal"] = data["signal"].astype("int8")
        return data[
            [
                *BASE_COLUMNS,
                "ma_short",
                "ma_long",
                "breakout_high",
                "atr",
                "trailing_peak",
                "atr_stop",
                "signal",
            ]
        ]


class CrossSectionalMomentumStrategy:
    """在同一交易日按 20/60/120 日复合动量横向排名。"""

    name = "momentum"

    def __init__(
        self,
        lookback: int = 60,
        skip_recent: int = 5,
        top_n: int = 2,
        periods: tuple[int, int, int] | None = None,
        weights: tuple[float, float, float] = (0.2, 0.3, 0.5),
        rank_buffer: int = 0,
    ) -> None:
        if (
            lookback < 5
            or skip_recent < 0
            or skip_recent >= lookback
            or top_n < 1
            or rank_buffer < 0
        ):
            raise ValueError("动量参数不合法")
        self.lookback = lookback
        self.skip_recent = skip_recent
        self.top_n = top_n
        self.periods = periods or (lookback, lookback, lookback)
        self.weights = weights
        self.rank_buffer = rank_buffer

    def generate_signals(
        self, history: pd.DataFrame, parameters: dict[str, object] | None = None
    ) -> pd.DataFrame:
        parameters = parameters or {}
        lookback = int(parameters.get("momentum_lookback", self.lookback))
        skip_recent = int(parameters.get("momentum_skip_recent", self.skip_recent))
        top_n = int(parameters.get("momentum_top_n", self.top_n))
        rank_buffer = int(parameters.get("momentum_rank_buffer", self.rank_buffer))
        if (
            lookback < 5
            or skip_recent < 0
            or skip_recent >= lookback
            or top_n < 1
            or rank_buffer < 0
        ):
            raise ValueError("动量参数不合法")
        data = _prepare_history(history)
        grouped_close = data.groupby("symbol", sort=False)["close"]
        recent_reference = grouped_close.shift(skip_recent)
        labels = ("short", "medium", "long")
        component_columns: list[str] = []
        for label, period in zip(labels, self.periods, strict=True):
            column = f"momentum_{label}"
            data[column] = recent_reference / grouped_close.shift(period + skip_recent) - 1
            component_columns.append(column)
        complete = data[component_columns].notna().all(axis=1)
        data["momentum_score"] = sum(
            data[column] * weight
            for column, weight in zip(component_columns, self.weights, strict=True)
        ).where(complete)
        data["momentum_rank"] = data.groupby("date")["momentum_score"].rank(
            ascending=False, method="first"
        )
        if rank_buffer == 0:
            eligible = data["momentum_score"].gt(0) & data["momentum_rank"].le(top_n)
            data["signal"] = eligible.fillna(False).astype("int8")
        else:
            data["signal"] = 0
            holders: set[str] = set()
            for _, indices in data.groupby("date", sort=True).groups.items():
                daily = data.loc[indices].sort_values("momentum_rank")
                keep = {
                    str(row.symbol)
                    for row in daily.itertuples()
                    if row.symbol in holders
                    and pd.notna(row.momentum_score)
                    and row.momentum_score > 0
                    and row.momentum_rank <= top_n + rank_buffer
                }
                candidates = [
                    str(row.symbol)
                    for row in daily.itertuples()
                    if row.symbol not in keep
                    and pd.notna(row.momentum_score)
                    and row.momentum_score > 0
                ]
                holders = keep | set(candidates[: max(top_n - len(keep), 0)])
                data.loc[indices, "signal"] = data.loc[indices, "symbol"].isin(holders).astype(int)
            data["signal"] = data["signal"].astype("int8")
        return data[
            [
                *BASE_COLUMNS,
                *component_columns,
                "momentum_score",
                "momentum_rank",
                "signal",
            ]
        ]


class RSIMeanReversionStrategy:
    name = "rsi_reversion"

    def __init__(self, window: int = 14, entry: float = 30, exit: float = 55) -> None:
        if window < 2 or not 0 <= entry < exit <= 100:
            raise ValueError("RSI 参数不合法")
        self.window = window
        self.entry = entry
        self.exit = exit

    def generate_signals(
        self, history: pd.DataFrame, parameters: dict[str, object] | None = None
    ) -> pd.DataFrame:
        parameters = parameters or {}
        window = int(parameters.get("rsi_window", self.window))
        entry = float(parameters.get("rsi_entry", self.entry))
        exit = float(parameters.get("rsi_exit", self.exit))
        if window < 2 or not 0 <= entry < exit <= 100:
            raise ValueError("RSI 参数不合法")
        data = _prepare_history(history)
        data["rsi"] = _calculate_rsi(data, window)
        data["signal"] = 0
        for _, indices in data.groupby("symbol", sort=False).groups.items():
            holding = 0
            for index in indices:
                value = data.loc[index, "rsi"]
                if pd.notna(value):
                    if holding == 0 and value <= entry:
                        holding = 1
                    elif holding == 1 and value >= exit:
                        holding = 0
                data.loc[index, "signal"] = holding
        data["signal"] = data["signal"].astype("int8")
        return data[[*BASE_COLUMNS, "rsi", "signal"]]


class BollingerMeanReversionStrategy:
    """RSI、布林带与短期超跌共同驱动的均值回归策略。"""

    name = "mean_reversion"

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    def generate_signals(
        self, history: pd.DataFrame, parameters: dict[str, object] | None = None
    ) -> pd.DataFrame:
        data = _prepare_history(history)
        data["rsi"] = _calculate_rsi(data, self.config.rsi_window)
        grouped_close = data.groupby("symbol", sort=False)["close"]
        data["bollinger_mid"] = grouped_close.transform(
            lambda values: values.rolling(
                self.config.bollinger_window,
                min_periods=self.config.bollinger_window,
            ).mean()
        )
        rolling_std = grouped_close.transform(
            lambda values: values.rolling(
                self.config.bollinger_window,
                min_periods=self.config.bollinger_window,
            ).std()
        )
        data["bollinger_lower"] = data["bollinger_mid"] - self.config.bollinger_std * rolling_std
        data["bollinger_upper"] = data["bollinger_mid"] + self.config.bollinger_std * rolling_std
        data["short_return"] = grouped_close.pct_change(
            self.config.oversold_window, fill_method=None
        )
        data["signal"] = 0
        for _, indices in data.groupby("symbol", sort=False).groups.items():
            holding = 0
            for index in indices:
                row = data.loc[index]
                oversold = (
                    (pd.notna(row["rsi"]) and row["rsi"] <= self.config.rsi_entry)
                    or (pd.notna(row["bollinger_lower"]) and row["close"] < row["bollinger_lower"])
                    or (
                        pd.notna(row["short_return"])
                        and row["short_return"] <= self.config.oversold_return
                    )
                )
                recovered = (pd.notna(row["rsi"]) and row["rsi"] >= self.config.rsi_exit) or (
                    pd.notna(row["bollinger_mid"]) and row["close"] >= row["bollinger_mid"]
                )
                if holding == 0 and oversold:
                    holding = 1
                elif holding == 1 and recovered:
                    holding = 0
                data.loc[index, "signal"] = holding
        data["signal"] = data["signal"].astype("int8")
        return data[
            [
                *BASE_COLUMNS,
                "rsi",
                "bollinger_mid",
                "bollinger_lower",
                "bollinger_upper",
                "short_return",
                "signal",
            ]
        ]


class EnsembleStrategy:
    def __init__(self, config: StrategyConfig, risk_weighted: bool = False) -> None:
        self.config = config
        self.risk_weighted = risk_weighted
        self.name = "ensemble_risk" if risk_weighted else "ensemble_equal"
        self.trend = TrendBreakoutStrategy(config)
        self.momentum = CrossSectionalMomentumStrategy(
            config.momentum_lookback,
            config.momentum_skip_recent,
            config.momentum_top_n,
            periods=(
                config.momentum_short_period,
                config.momentum_medium_period,
                config.momentum_long_period,
            ),
            rank_buffer=config.momentum_rank_buffer,
        )
        self.reversion = BollingerMeanReversionStrategy(config)

    def generate_signals(
        self, history: pd.DataFrame, parameters: dict[str, object] | None = None
    ) -> pd.DataFrame:
        data = _prepare_history(history)
        keys = BASE_COLUMNS
        trend = self.trend.generate_signals(data)[[*keys, "signal"]].rename(
            columns={"signal": "trend_signal"}
        )
        momentum = self.momentum.generate_signals(data)[[*keys, "signal"]].rename(
            columns={"signal": "momentum_signal"}
        )
        reversion = self.reversion.generate_signals(data)[[*keys, "signal"]].rename(
            columns={"signal": "reversion_signal"}
        )
        combined = trend.merge(momentum, on=keys, how="inner").merge(
            reversion, on=keys, how="inner"
        )
        components = ["trend_signal", "momentum_signal", "reversion_signal"]
        if self.risk_weighted:
            daily_return = combined.groupby("symbol", sort=False)["close"].pct_change()
            inverse_volatility: dict[str, pd.Series] = {}
            for component in components:
                prior_position = combined.groupby("symbol", sort=False)[component].shift(1)
                realized_return = prior_position * daily_return
                volatility = realized_return.groupby(combined["symbol"], sort=False).transform(
                    lambda values: values.rolling(
                        self.config.ensemble_volatility_window,
                        min_periods=self.config.ensemble_volatility_window,
                    ).std()
                )
                inverse_volatility[component] = 1 / volatility.replace(0, np.nan)
            inverse = pd.DataFrame(inverse_volatility, index=combined.index)
            ready = inverse.notna().all(axis=1) & inverse.gt(0).all(axis=1)
            normalized = inverse.div(inverse.sum(axis=1), axis=0)
            weights = normalized.where(ready, 1 / len(components))
        else:
            weights = pd.DataFrame(1 / len(components), index=combined.index, columns=components)
        for component in components:
            combined[f"{component}_weight"] = weights[component]
        combined["ensemble_score"] = sum(
            combined[component] * weights[component] for component in components
        )
        combined["signal"] = (combined["ensemble_score"] >= self.config.ensemble_threshold).astype(
            "int8"
        )
        return combined[
            [
                *BASE_COLUMNS,
                *components,
                *(f"{component}_weight" for component in components),
                "ensemble_score",
                "signal",
            ]
        ]


class FirstBoardPullbackStrategy:
    """首板回马枪（用户本对话给定的规则，字面实现）。

    时间记号：B = 首板日（T-1），T = 买入日（B 的下一交易日）。
    引擎是“收盘信号 -> 次日开盘执行”，因此入场信号记在 B 日，
    引擎在 T 开盘买入，成交价就是 T 开盘价——这是开盘策略在
    该引擎下的唯一不失真映射，代价是信号生成时必须看 T 的开盘价。
    为防把这误读为未来函数，特此限定：入场判断只用 T 的开盘价，
    绝不用 T 的收盘价/最高价/最低价/成交量；B 日的涨幅、是否一字板、
    最高价均在 B 收盘时已知。退出则按常规收盘信号处理（见下），
    另由引擎的盘中止盈止损扩展在触及当日按止盈/止损价了结。

    选股（B 日）：主板涨幅 9.8%~10.2%，创业板/科创板 19.8%~20.2%，
    且开盘价 != 收盘价（非一字板）。
    买入（T 日）：T 开盘 > B 最高价（跳空高开确认），且
    T 开盘涨幅（相对 B 收盘）<= 7%，且 T 开盘不涨停
    （涨停预过滤，引擎的买入涨停校验是第二道安全网）。
    卖出：入场价 = T 开盘价；持有期间任一交易日 high >= 入场×1.08
    止盈，low <= 入场×0.95 止损，持有满 5 个交易日（T+5）收盘了结。
    信号：B 日至退出日前一日为 1，退出日及之后为 0；
    退出日信号翻空后，引擎在次日开盘的卖出尝试因已无仓位而不成交，
    盘中止盈止损扩展已在退出日当日完成实际了结。
    """

    name = "first_board_pullback"
    enable_intraday_stops = True

    def __init__(
        self,
        take_profit: float = 0.08,
        stop_loss: float = 0.05,
        max_holding_days: int = 5,
        max_open_rise: float = 0.07,
    ) -> None:
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.max_holding_days = max_holding_days
        self.max_open_rise = max_open_rise

    @staticmethod
    def _is_chinext_like(code: str) -> bool:
        return code.startswith(("300", "688"))

    def generate_signals(
        self, history: pd.DataFrame, parameters: dict[str, object] | None = None
    ) -> pd.DataFrame:
        parameters = parameters or {}
        take_profit = float(parameters.get("take_profit", self.take_profit))
        stop_loss = float(parameters.get("stop_loss", self.stop_loss))
        max_holding = int(parameters.get("max_holding_days", self.max_holding_days))
        max_open_rise = float(parameters.get("max_open_rise", self.max_open_rise))
        for col in ("open", "high", "low"):
            if col not in history.columns:
                raise ValueError(f"首板策略需要字段：{col}")
        data = _prepare_history(history)
        for col in ("open", "high", "low"):
            data[col] = pd.to_numeric(data[col], errors="coerce")
        data["signal"] = 0
        data["is_board"] = False
        data["entry_price"] = np.nan
        data["exit_reason"] = ""
        for _, indices in data.groupby("symbol", sort=False).groups.items():
            idx = list(indices)
            # 按该股交易日顺序逐日推进，不用任何未来收盘价。
            holding = False
            entry_price = np.nan
            buy_idx_pos = -1  # 在 idx 列表中的位置（买入日 T）
            for pos, index in enumerate(idx):
                row = data.loc[index]
                if pos == 0:
                    continue
                prev = data.loc[idx[pos - 1]]
                board_ret = row["close"] / prev["close"] - 1 if prev["close"] else np.nan
                code = str(row["symbol"])
                lo, hi = (0.198, 0.202) if self._is_chinext_like(code) else (0.098, 0.102)
                is_board = (
                    pd.notna(board_ret)
                    and lo <= board_ret <= hi
                    and row["open"] != row["close"]
                )
                data.loc[index, "is_board"] = bool(is_board)
                if holding:
                    # 退出判断只用当日 high/low/close（当日收盘时已知），
                    # 且只在买入日之后的交易日执行。
                    if pos > buy_idx_pos:
                        exit_reason = ""
                        if row["high"] >= entry_price * (1 + take_profit):
                            exit_reason = "take_profit"
                        elif row["low"] <= entry_price * (1 - stop_loss):
                            exit_reason = "stop_loss"
                        elif pos - buy_idx_pos >= max_holding:
                            exit_reason = "max_holding"
                        if exit_reason:
                            data.loc[index, "signal"] = 0
                            data.loc[index, "exit_reason"] = exit_reason
                            holding = False
                            entry_price = np.nan
                            continue
                    data.loc[index, "signal"] = 1
                    data.loc[index, "entry_price"] = entry_price
                    continue
                # 未持有：只有当日本身是首板日，且下一交易日满足开盘条件，
                # 才在当日（B 日）记入场信号，引擎于下一交易日开盘买入。
                if not is_board or pos + 1 >= len(idx):
                    continue
                nxt = data.loc[idx[pos + 1]]
                open_rise = nxt["open"] / row["close"] - 1 if row["close"] else np.nan
                if not (
                    pd.notna(open_rise)
                    and nxt["open"] > row["high"]
                    and 0 < open_rise <= max_open_rise
                ):
                    continue
                # 涨停预过滤：T 开盘涨幅达到涨停阈值则放弃本次（引擎另有安全网）。
                thresh = 0.198 if self._is_chinext_like(code) else 0.098
                if open_rise >= thresh:
                    continue
                holding = True
                entry_price = float(nxt["open"])
                buy_idx_pos = pos + 1
                data.loc[index, "signal"] = 1
                data.loc[index, "entry_price"] = entry_price
        data["signal"] = data["signal"].astype("int8")
        cols = [*BASE_COLUMNS, "open", "high", "low"]
        cols += ["signal", "is_board", "entry_price", "exit_reason"]
        return data[cols]


def create_strategy(config: StrategyConfig) -> Strategy:
    if config.name == "first_board_pullback":
        return FirstBoardPullbackStrategy()
    if config.name == "ma_cross":
        return MovingAverageCrossStrategy(
            config.short_window,
            config.long_window,
            config.ma_signal_buffer,
            config.ma_confirmation_days,
        )
    if config.name == "trend_breakout":
        return TrendBreakoutStrategy(config)
    if config.name == "momentum":
        return CrossSectionalMomentumStrategy(
            config.momentum_lookback,
            config.momentum_skip_recent,
            config.momentum_top_n,
            periods=(
                config.momentum_short_period,
                config.momentum_medium_period,
                config.momentum_long_period,
            ),
            rank_buffer=config.momentum_rank_buffer,
        )
    if config.name == "rsi_reversion":
        return RSIMeanReversionStrategy(config.rsi_window, config.rsi_entry, config.rsi_exit)
    if config.name == "mean_reversion":
        return BollingerMeanReversionStrategy(config)
    if config.name == "ensemble_equal":
        return EnsembleStrategy(config, risk_weighted=False)
    if config.name == "ensemble_risk":
        return EnsembleStrategy(config, risk_weighted=True)
    raise ValueError(f"未知策略：{config.name}；可选值：{', '.join(STRATEGY_NAMES)}")
