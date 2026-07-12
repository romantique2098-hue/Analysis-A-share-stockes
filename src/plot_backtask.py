from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
OUTPUT_DIR = PROJECT_ROOT / "output"

def plot_backtest(symbol:str) -> None:
    input_path = FEATURES_DIR / f"symbol.parquet"
    data = pd.read_parquet(input_path)

    figure, axis = plt.subplots(figsize=(12,6))
    axis.plot(
        data["date"],
        data["buy_hold_curve"],
        label="Buy and Hold",
        linewidth=2,
    )

    axis.plot(
        data["date"],
        data["strategy_curve"],
        label="Strategy Before Costs",
        linewidth=2,
    )

    axis.plot(
        data["date"],
        data["net_strategy_curve"],
        label="Strategy After Costs",
        linewidth=2,
    )
    axis.axhline(
        y=1,
        color="gray",
        linestyle="--",
        linewidth=1,
    )

    axis.set_title(
        f"{symbol} Backtest Equity Curves"
    )

    axis.set_xlabel("Date")
    axis.set_ylabel("Portfolio Value")

    axis.legend()
    axis.grid(
        alpha=0.25
    )

    figure.autofmt_xdate()
    figure.tight_layout()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / f"{symbol}_backtest.png"
    )

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    print(f"回测图已经保存到：{output_path}")

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="绘制指定股票的回测净值曲线"
    )

    parser.add_argument(
        "symbol",
        type=str,
    )

    args = parser.parse_args()

    plot_backtest(args.symbol)