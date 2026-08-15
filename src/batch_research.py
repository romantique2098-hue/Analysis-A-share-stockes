from pathlib import Path

import pandas as pd

from parameter_sensitivity import (
    WALK_FORWARD_TEST_YEARS,
    build_parameters_grid,
    calculate_buy_hold_metrics,
    calculate_metrics,
    load_features,
    run_walk_forward_analysis,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
OUTPUT_DIR = PROJECT_ROOT / "output"


def discover_symbols():
    """从特征数据文件名中自动发现所有待研究股票代码。"""
    feature_paths = FEATURES_DIR.glob("*.parquet")
    symbols = sorted(
        path.stem
        for path in feature_paths
    )

    if not symbols:
        raise FileNotFoundError(
            f"没有在 {FEATURES_DIR} 中找到特征文件"
        )

    return symbols


def analyze_symbol(
    symbol,
    parameter_pairs,
):
    """对一只股票运行滚动样本外验证，并返回一行汇总结果。"""
    features = load_features(symbol)
    (
        fold_table,
        walk_forward_daily,
    ) = run_walk_forward_analysis(
        data=features,
        parameter_pairs=parameter_pairs,
        test_years=WALK_FORWARD_TEST_YEARS,
    )

    strategy_metrics = calculate_metrics(
        walk_forward_daily
    )
    benchmark_metrics = calculate_buy_hold_metrics(
        walk_forward_daily
    )

    final_value_advantage = (
        strategy_metrics["final_value"]
        - benchmark_metrics["final_value"]
    )
    drawdown_improvement = (
        strategy_metrics["max_drawdown"]
        - benchmark_metrics["max_drawdown"]
    )

    profitable_test_years = int(
        (
            fold_table["test_final_value"]
            > 1
        ).sum()
    )
    positive_sharpe_years = int(
        (
            fold_table["test_sharpe"]
            > 0
        ).sum()
    )

    return {
        "symbol": symbol,
        "test_days": len(walk_forward_daily),
        "test_years": len(fold_table),
        "profitable_test_years": profitable_test_years,
        "positive_sharpe_years": positive_sharpe_years,
        "strategy_final_value": strategy_metrics["final_value"],
        "benchmark_final_value": benchmark_metrics["final_value"],
        "final_value_advantage": final_value_advantage,
        "strategy_max_drawdown": strategy_metrics["max_drawdown"],
        "benchmark_max_drawdown": benchmark_metrics["max_drawdown"],
        "drawdown_improvement": drawdown_improvement,
        "strategy_sharpe": strategy_metrics["sharpe"],
        "benchmark_sharpe": benchmark_metrics["sharpe"],
        "strategy_profitable": (
            strategy_metrics["final_value"] > 1
        ),
        "beats_benchmark": (
            final_value_advantage > 0
        ),
        "reduces_drawdown": (
            drawdown_improvement > 0
        ),
    }


def run_batch_research():
    """依次研究所有已准备好的股票，并整理成跨股票结果表。"""
    symbols = discover_symbols()
    parameter_pairs = build_parameters_grid()
    results = []

    print(
        f"发现 {len(symbols)} 只待研究股票："
        f"{symbols}"
    )

    for number, symbol in enumerate(
        symbols,
        start=1,
    ):
        print(
            f"[{number}/{len(symbols)}] "
            f"正在分析 {symbol}"
        )

        symbol_result = analyze_symbol(
            symbol=symbol,
            parameter_pairs=parameter_pairs,
        )
        results.append(symbol_result)

    result_table = (
        pd.DataFrame(results)
        .sort_values(
            "final_value_advantage",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return result_table


def save_batch_results(result_table):
    """把跨股票结果保存为CSV文件，并返回保存路径。"""
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path = (
        OUTPUT_DIR
        / "multi_stock_walk_forward.csv"
    )
    result_table.to_csv(
        output_path,
        index=False,
    )

    return output_path


if __name__ == "__main__":
    batch_result = run_batch_research()
    output_path = save_batch_results(
        batch_result
    )

    total_count = len(batch_result)
    profitable_count = int(
        batch_result[
            "strategy_profitable"
        ].sum()
    )
    beat_count = int(
        batch_result[
            "beats_benchmark"
        ].sum()
    )
    reduced_drawdown_count = int(
        batch_result[
            "reduces_drawdown"
        ].sum()
    )

    print("\n跨股票研究结果：")
    print(
        batch_result
        .round(4)
        .to_string(index=False)
    )

    print("\n跨股票汇总：")
    print(
        f"策略盈利："
        f"{profitable_count}/{total_count}"
    )
    print(
        f"战胜买入持有："
        f"{beat_count}/{total_count}"
    )
    print(
        f"降低最大回撤："
        f"{reduced_drawdown_count}/{total_count}"
    )
    print(f"结果已保存到：{output_path}")
