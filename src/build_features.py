from pathlib import Path
import pandas as pd
import argparse

# 无论从哪个目录启动脚本，都先定位到项目根目录，再由根目录拼出数据路径。
# 这样可以避免把“/Users/.../A股分析”写死，项目移动后代码仍然可用。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
# 预留给加工后数据的目录；当前代码还没有执行保存操作。
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
COMMISSION_RATE = 0.0003
STAMP_DUTY_RATE = 0.0005
SLIPPAGE_RATE = 0.0002

def build_features(symbol:str) ->pd.DataFrame:
    """计算收益率，均线和最基本的均线策略信号"""
    # f-string 会把传入的股票代码放进文件名，例如 000001.parquet。
    input_path = RAW_DATA_DIR / f"{symbol}.parquet"

    # Parquet 是硬盘上的文件；read_parquet 将它读成 pandas DataFrame。
    data = pd.read_parquet(input_path)

    # 按日期排序，保证后面的 shift、pct_change 和 rolling 都按时间先后计算。
    # 同一只股票在同一交易日只保留一行，然后重新生成连续的行索引。
    data=(
        data.sort_values("date")
        .drop_duplicates(subset=["symbol", "date"])
        .reset_index(drop=True)
    )

    # shift(1) 将收盘价向下移一行，使每行都能同时拿到“今日”与“昨日”收盘价。
    data["previous_close"] = data["close"].shift(1)

    # 手动套用公式：(今日收盘价 - 昨日收盘价) / 昨日收盘价。
    # 这一列是教学辅助列，用来和 pct_change 的结果互相验证。
    data["daily_return_manual"] = (data["close"] - data["previous_close"]) / data["previous_close"]

    # pandas 的正式写法；fill_method=None 表示遇到缺失价格时不自动向前填充。
    data["daily_return"] = data["close"].pct_change(fill_method=None)

    # rolling 创建向前滚动的 5 个交易日窗口，mean 对窗口内的收盘价求平均。
    # min_periods=5 表示历史数据不足 5 行时保留 NaN，不伪造均线。
    data["ma5"] = (
        data["close"]
        .rolling(window=5, min_periods=5)
        .mean()
        )

    # 20 日均线变化更慢，用作与短期 5 日均线比较的长期参考。
    data["ma20"] = (
        data["close"]
        .rolling(window=20, min_periods=20)
        .mean()
    )

    # 比较 ma5 和 ma20 生成信号。注意：括号位置会决定 astype 究竟转换哪一部分。
    data["signal"] = (
        (data["ma5"] > data["ma20"])
        .astype("int8")
    )

    # 当日收盘后才能计算当日信号，所以将信号向后移一天作为实际仓位。
    # 第一行没有上一日信号，fill_value=0 明确规定回测从空仓开始。
    data["position"] = data["signal"].shift(1, fill_value=0)

    data["trade"] = (
        data["position"]
        .diff()
        .fillna(0)
        .astype("int8")
    )

    data["buy_trade"] = (data["trade"] == 1).astype("int8")
    data["sell_trade"] = (data["trade"] == -1).astype("int8")

    data["turnover"] = data["trade"].abs()
    data["commission_cost"] = data["turnover"] * COMMISSION_RATE
    data["stamp_duty_cost"] = data["sell_trade"] * STAMP_DUTY_RATE
    data["slippage_cost"] = data["turnover"] * SLIPPAGE_RATE
    data["total_cost"] = data["commission_cost"] + data["stamp_duty_cost"] + data["slippage_cost"]

    # 仓位为 1 时承担股票涨跌；仓位为 0 时策略当日收益为 0。
    data["strategy_return"] = data["position"] * data["daily_return"]
    data["net_strategy_return"] = data["strategy_return"] - data["total_cost"]

    # 将每日的收益倍数 (1 + 日收益率) 连续相乘，得到从 1 元出发的累计净值。
    # buy_hold_curve 是一直持有的基准，strategy_curve 是均线策略净值。
    data["buy_hold_curve"] = (1 + data["daily_return"]).cumprod()
    data["strategy_curve"] = (1 + data["strategy_return"]).cumprod()
    data["net_strategy_curve"] = (1 + data["net_strategy_return"]).cumprod()

    # iloc[-1] 表示按位置取净值列的最后一个值。
    final_buy_hold = data["buy_hold_curve"].iloc[-1]
    final_strategy = data["strategy_curve"].iloc[-1]
    final_net_strategy = data["net_strategy_curve"].iloc[-1]
    cost_drag = final_strategy - final_net_strategy
    print(f"{symbol}的买入持有策略最终收益为 {final_buy_hold:.4f}, 成本前均线策略最终收益为 {final_strategy:.4f}, 成本后均线策略最终收益:{final_net_strategy:.4f}")
    print(f"交易成本造成的净值损耗:{cost_drag:.4f}")

    # cummax 记录“截至当日曾出现过的最高净值”，是计算回撤的基准。
    data["buy_hold_peak"] = data["buy_hold_curve"].cummax()
    data["strategy_peak"] = data["strategy_curve"].cummax()
    data["net_strategy_peak"] = data["net_strategy_curve"].cummax()

    # 回撤 = 当前净值 / 历史最高净值 - 1；创新高时为 0，低于高点时为负数。
    data["buy_hold_drawdown"] = (data["buy_hold_curve"] / data["buy_hold_peak"] - 1)
    data["strategy_drawdown"] = (data["strategy_curve"] / data["strategy_peak"] - 1)
    data["net_strategy_drawdown"] = data["net_strategy_curve"] / data["net_strategy_peak"] - 1

    # 回撤列中最小（最负）的数就是整段回测的最大回撤。
    buy_hold_max_drawdown = (data["buy_hold_drawdown"].min())
    strategy_max_drawdown = (data["strategy_drawdown"].min())
    net_strategy_max_drawdown = data["net_strategy_drawdown"].min()
    print(f"买入持有最大回撤{buy_hold_max_drawdown:.2%}, 均线策略最大回撤{strategy_max_drawdown:.2%}，成本后均线策略最大回撤：{net_strategy_max_drawdown:.2%}")

    # 用首尾交易日的时间差计算回测持续年数，365.25 把闰年纳入长期平均。
    start_day = data["date"].iloc[0]
    end_day = data["date"].iloc[-1]

    elapsed_days = (end_day - start_day).days
    elapsed_years = elapsed_days / 365.25

    # 年化收益率是与最终净值等价的每年复利增长速度，不是总收益率简单除以年数。
    buy_hold_annual_return = (final_buy_hold ** (1 / elapsed_years)) - 1
    strategy_annual_return = (final_strategy ** (1 / elapsed_years)) - 1
    net_strategy_annual_return = (final_net_strategy **(1 / elapsed_years)) - 1

    print(f"买入持有年化收益率:{buy_hold_annual_return:.2%}")
    print(f"均线策略年化收益率:{strategy_annual_return:.2%}")
    print(f"成本后均线策略年化收益率：{net_strategy_annual_return:.2%}")

    # 用每年约 252 个交易日作为年化假设；大写表示这是配置常量。
    TRADING_DAYS_PER_YEAR = 252

    # std 计算日收益率的标准差，用来衡量收益每天的不稳定程度。
    buy_hold_daily_volatility = data["daily_return"].std()
    strategy_daily_volatility = data["strategy_return"].std()
    net_strategy_daily_volatility = data["net_strategy_return"].std()

    # 在常用的简化假设下，年化波动率 = 日波动率 × 根号(252)。
    annualized_buy_hold_volatility = buy_hold_daily_volatility * (TRADING_DAYS_PER_YEAR ** 0.5)
    annualized_strategy_volatility = strategy_daily_volatility * (TRADING_DAYS_PER_YEAR ** 0.5)
    annualized_net_strategy_volatility = net_strategy_daily_volatility * (TRADING_DAYS_PER_YEAR ** 0.5)

    print(f"买入持有年化波动率{annualized_buy_hold_volatility:.2%}")
    print(f"均线策略年化波动率{annualized_strategy_volatility:.2%}")
    print(f"成本后均线策略年化波动率{annualized_net_strategy_volatility:.2%}")

    RISK_FREE_ANNUAL_RETURN = 0.02
    risk_free_daily_return = (1 + RISK_FREE_ANNUAL_RETURN) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    buy_hold_daily_return_excess = data["daily_return"] - risk_free_daily_return
    strategy_daily_return_excess = data["strategy_return"] - risk_free_daily_return
    net_strategy_daily_return_excess = data["net_strategy_return"] - risk_free_daily_return

    buy_hold_sharpo = buy_hold_daily_return_excess.mean() / buy_hold_daily_volatility * (TRADING_DAYS_PER_YEAR ** 0.5)
    strategy_sharpo = strategy_daily_return_excess.mean() / strategy_daily_volatility * (TRADING_DAYS_PER_YEAR ** 0.5)
    net_strategy_sharpo = net_strategy_daily_return_excess.mean() / net_strategy_daily_volatility * (TRADING_DAYS_PER_YEAR ** 0.5)
    print(f"买入持有夏普比率：{buy_hold_sharpo:.2f}")
    print(f"均线策略夏普比率：{strategy_sharpo:.2f}")
    print(f"成本后均线策略夏普比率：{net_strategy_sharpo:.2f}")

    FEATURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    buy_count = int(data["buy_trade"].sum())
    sell_count = int(data["sell_trade"].sum())
    trade_action_count = int(data["turnover"].sum())

    has_open_position = bool(data["position"].iloc[-1] == 1)
    print(f"买入次数：{buy_count}")
    print(f"卖出次数：{sell_count}")
    print(f"期末是否持仓：{has_open_position}")

    time_in_market = data["position"].mean()
    print(f"持仓时间比例：{time_in_market:.2%}")

    trade_log = data.loc[
        data["trade"] != 0,
        [
            "date",
            "close",
            "trade",
            "commission_cost",
            "stamp_duty_cost",
            "slippage_cost",
            "total_cost",
        ],
    ].copy()

    trade_log["action"] = trade_log["trade"].map({1:"买入", -1:"卖出"})



    output_path = FEATURES_DIR / f"{symbol}.parquet"
    data.to_parquet(output_path, index=False)
    print(f"特征数据已经保存到{output_path}")

    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="计算指定股票的量化特征")
    parser.add_argument(
        "symbol",
        type=str,
    )
    args = parser.parse_args()
    features_data = build_features(args.symbol)
