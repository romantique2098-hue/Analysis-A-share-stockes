from pathlib import Path
import argparse
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = PROJECT_ROOT / "data" / "features"

COST_SCENARIOS = {
    "low": {
        "commission": 0.0001,
        "stamp_duty": 0.0005,
        "slippage": 0.0001,
    },
    "base": {
        "commission": 0.0003,
        "stamp_duty": 0.0005,
        "slippage": 0.0002,
    },
    "high": {
        "commission": 0.0005,
        "stamp_duty": 0.0005,
        "slippage": 0.0010,
    },
}


def load_features(
    symbol: str,
) -> pd.DataFrame:
    input_path = FEATURE_DIR / f"{symbol}.parquet"
    data = pd.read_parquet(input_path)

    return data


def calculate_scenario(
    data: pd.DataFrame,
    commission_rate: float,
    stamp_duty_rate: float,
    slippage_rate: float,
) -> pd.Series:
    turnover = data["trade"].abs()

    sell_trade = (
        data["trade"] == -1
    ).astype("int8")

    commission_cost = (
        turnover * commission_rate
    )

    stamp_duty_cost = (
        sell_trade * stamp_duty_rate
    )

    slippage_cost = (
        turnover * slippage_rate
    )

    total_cost = (
        commission_cost
        + stamp_duty_cost
        + slippage_cost
    )

    net_return = (
        data["strategy_return"]
        - total_cost
    )

    net_curve = (
        1 + net_return
    ).cumprod()

    return net_curve


def run_sensitivity(
    symbol: str,
) -> None:
    data = load_features(symbol)

    gross_final_value = data["strategy_curve"].iloc[-1]
    print(f"成本前最终净值：{gross_final_value:.4f}")

    for scenario_name, rates in COST_SCENARIOS.items():
        net_curve = calculate_scenario(
            data=data,
            commission_rate=rates["commission"],
            stamp_duty_rate=rates["stamp_duty"],
            slippage_rate=rates["slippage"],
        )

        final_value = net_curve.iloc[-1]
        cost_drag = gross_final_value - final_value
        net_peak = net_curve.cummax()
        net_drawdown = net_curve / net_peak - 1
        net_max_drawdown = net_drawdown.min()
        print(f"\n成本情景：{scenario_name}")
        print(f"成本后最终净值：{final_value:.4f}")
        print(f"成本损失：{cost_drag:.4f}")
        print(f"成本后最大回撤：{net_max_drawdown:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="比较指定股票在不同交易成本情景下的回测结果"
    )
    parser.add_argument(
        "symbol",
        type=str,
        help="六位股票代码，例如 000001",
    )
    args = parser.parse_args()
    run_sensitivity(args.symbol)
