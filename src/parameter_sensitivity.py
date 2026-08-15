from pathlib import Path
import pandas as pd

SHORT_WINDOWS = [5, 10, 20]
LONG_WINDOWS = [20, 60, 120]

COMMISSION_RATE = 0.0003
STAMP_DUTY_RATE = 0.0005
SLIPPAGE_RATE = 0.0002
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_ANNUAL_RETURN = 0.02
SPLIT_DATE = "2024-01-01"
WALK_FORWARD_TEST_YEARS = [2022, 2023, 2024, 2025]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
OUTPUT_DIR = PROJECT_ROOT / "output"

def load_features(symbol):
    input_path = FEATURES_DIR / f"{symbol}.parquet"
    data = pd.read_parquet(input_path)

    required_columns = {
        "date",
        "close",
        "daily_return",
    }

    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(
            f"特征数据缺少字段：{missing_columns}"
        )

    data = (
        data.sort_values("date")
        .reset_index(drop=True)
    )

    return data

def build_parameters_grid():
    parameters_pair = []
    for short_windows in SHORT_WINDOWS:
        for long_windows in LONG_WINDOWS:
            if short_windows >= long_windows:
                continue
            parameters_pair.append(
                (short_windows, long_windows)
            )
    return parameters_pair

def calculate_strategy(
    data,
    short_window,
    long_window,
):
    if short_window >= long_window:
        raise ValueError(
            "短期均线天数必须小于长期均线天数"
        )

    experiment = data[
        [
            "date",
            "close",
            "daily_return",
        ]
    ].copy()

    experiment["short_ma"] = (
        experiment["close"]
        .rolling(
            window=short_window,
            min_periods=short_window,
        )
        .mean()
    )

    experiment["long_ma"] = (
        experiment["close"]
        .rolling(
            window=long_window,
            min_periods=long_window,
        )
        .mean()
    )

    experiment["signal"] = (
        experiment["short_ma"]
        > experiment["long_ma"]
    ).astype("int8")

    experiment["position"] = (
        experiment["signal"]
        .shift(1, fill_value=0)
        .astype("int8")
    )

    experiment["trade"] = (
        experiment["position"]
        .diff()
        .fillna(0)
        .astype("int8")
    )

    experiment["turnover"] = (
        experiment["trade"].abs()
    )

    experiment["sell_trade"] = (
        experiment["trade"] == -1
    ).astype("int8")

    experiment["commission_cost"] = (
        experiment["turnover"]
        * COMMISSION_RATE
    )

    experiment["stamp_duty_cost"] = (
        experiment["sell_trade"]
        * STAMP_DUTY_RATE
    )

    experiment["slippage_cost"] = (
        experiment["turnover"]
        * SLIPPAGE_RATE
    )

    experiment["total_cost"] = (
        experiment["commission_cost"]
        + experiment["stamp_duty_cost"]
        + experiment["slippage_cost"]
    )

    experiment["gross_return"] = (
        experiment["position"]
        * experiment["daily_return"].fillna(0)
    )

    experiment["net_return"] = (
        experiment["gross_return"]
        - experiment["total_cost"]
    )

    experiment["net_curve"] = (
        1 + experiment["net_return"]
    ).cumprod()

    return experiment


def calculate_metrics(period_data):
    if period_data.empty:
        raise ValueError(
            "无法计算空数据的绩效指标"
        )

    period_returns = (
        period_data["net_return"]
        .fillna(0)
    )
    net_curve = (
        1 + period_returns
    ).cumprod()
    final_value = net_curve.iloc[-1]

    # 净值默认从 1 出发，所以历史高点不应低于 1。
    peak = net_curve.cummax().clip(lower=1)
    drawdown = net_curve / peak - 1
    max_drawdown = drawdown.min()

    start_date = period_data["date"].iloc[0]
    end_date = period_data["date"].iloc[-1]
    elapsed_days = (end_date - start_date).days

    if elapsed_days <= 0:
        raise ValueError(
            "计算年化收益至少需要两个不同日期"
        )

    annual_return = (
        final_value
        ** (365.25 / elapsed_days)
        - 1
    )

    daily_volatility = period_returns.std()
    risk_free_daily_return = (
        (1 + RISK_FREE_ANNUAL_RETURN)
        ** (1 / TRADING_DAYS_PER_YEAR)
        - 1
    )
    daily_excess_return = (
        period_returns
        - risk_free_daily_return
    )

    if (
        pd.isna(daily_volatility)
        or daily_volatility == 0
    ):
        sharpe = float("nan")
    else:
        sharpe = (
            daily_excess_return.mean()
            / daily_volatility
            * (TRADING_DAYS_PER_YEAR ** 0.5)
        )

    trade_count = int(
        period_data["turnover"].sum()
    )

    return {
        "final_value": final_value,
        "annual_return": annual_return,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "trade_count": trade_count,
    }


def calculate_buy_hold_metrics(period_data):
    """用同一段样本外日期计算买入持有基准的绩效。"""
    benchmark_data = period_data[
        [
            "date",
            "daily_return",
        ]
    ].copy()

    # 滚动样本外从上一交易日收盘后开始，所以基准也视为期初已经持股。
    # 这里比较的是后复权价格收益，不另外扣除一次性建仓成本。
    benchmark_data["net_return"] = (
        benchmark_data["daily_return"]
        .fillna(0)
    )
    benchmark_data["turnover"] = 0

    return calculate_metrics(benchmark_data)


def build_benchmark_comparison(
    strategy_metrics,
    buy_hold_metrics,
):
    """把策略与买入持有的核心指标整理成便于比较的表格。"""
    return pd.DataFrame(
        [
            {
                "portfolio": "滚动均线策略",
                "final_value": strategy_metrics["final_value"],
                "annual_return": strategy_metrics["annual_return"],
                "max_drawdown": strategy_metrics["max_drawdown"],
                "sharpe": strategy_metrics["sharpe"],
            },
            {
                "portfolio": "买入持有",
                "final_value": buy_hold_metrics["final_value"],
                "annual_return": buy_hold_metrics["annual_return"],
                "max_drawdown": buy_hold_metrics["max_drawdown"],
                "sharpe": buy_hold_metrics["sharpe"],
            },
        ]
    )


def run_all_parameters(
    data,
    parameter_pairs,
):
    results = []

    for short_window, long_window in parameter_pairs:
        experiment = calculate_strategy(
            data=data,
            short_window=short_window,
            long_window=long_window,
        )
        metrics = calculate_metrics(experiment)

        results.append(
            {
                "short_window": short_window,
                "long_window": long_window,
                "final_value": metrics["final_value"],
                "annual_return": metrics["annual_return"],
                "max_drawdown": metrics["max_drawdown"],
                "sharpe": metrics["sharpe"],
                "trade_count": metrics["trade_count"],
            }
        )

    result_table = (
        pd.DataFrame(results)
        .sort_values(
            "sharpe",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return result_table


def split_experiment(
    experiment,
    split_date,
):
    split_timestamp = pd.Timestamp(split_date)

    in_sample = experiment.loc[
        experiment["date"] < split_timestamp
    ].copy()
    out_of_sample = experiment.loc[
        experiment["date"] >= split_timestamp
    ].copy()

    if in_sample.empty:
        raise ValueError(
            "样本内数据为空，请检查切分日期"
        )
    if out_of_sample.empty:
        raise ValueError(
            "样本外数据为空，请检查切分日期"
        )

    return in_sample, out_of_sample


def run_sample_analysis(
    data,
    parameter_pairs,
    split_date,
):
    results = []

    for short_window, long_window in parameter_pairs:
        experiment = calculate_strategy(
            data=data,
            short_window=short_window,
            long_window=long_window,
        )
        in_sample, out_of_sample = split_experiment(
            experiment=experiment,
            split_date=split_date,
        )
        in_metrics = calculate_metrics(in_sample)
        out_metrics = calculate_metrics(out_of_sample)

        results.append(
            {
                "short_window": short_window,
                "long_window": long_window,
                "in_final_value": in_metrics["final_value"],
                "in_annual_return": in_metrics["annual_return"],
                "in_max_drawdown": in_metrics["max_drawdown"],
                "in_sharpe": in_metrics["sharpe"],
                "in_trade_count": in_metrics["trade_count"],
                "out_final_value": out_metrics["final_value"],
                "out_annual_return": out_metrics["annual_return"],
                "out_max_drawdown": out_metrics["max_drawdown"],
                "out_sharpe": out_metrics["sharpe"],
                "out_trade_count": out_metrics["trade_count"],
            }
        )

    result_table = (
        pd.DataFrame(results)
        .sort_values(
            "in_sharpe",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return result_table

def run_walk_forward_analysis(
    data,
    parameter_pairs,
    test_years,
):
    fold_results = []
    out_of_sample_parts = []

    for test_year in test_years:
        test_start = pd.Timestamp(
            f"{test_year}-01-01"
        )
        test_end = pd.Timestamp(
            f"{test_year + 1}-01-01"
        )
        train_data = data.loc[
            data["date"] < test_start
        ].copy()
        available_data = data.loc[
            data["date"] < test_end
        ].copy()

        train_results = run_all_parameters(
            data=train_data,
            parameter_pairs=parameter_pairs,
        )

        best_parameter = train_results.iloc[0]

        short_window = int(best_parameter["short_window"])
        long_window = int(best_parameter["long_window"])

        experiment = calculate_strategy(
            data=available_data,
            short_window=short_window,
            long_window=long_window,
        )

        test_data = experiment.loc[
            (experiment["date"] >= test_start)
            &
            (experiment["date"] < test_end)
        ].copy()

        test_metrics = calculate_metrics(
            test_data
        )

        fold_results.append(
            {
                "test_year": test_year,
                "short_window": short_window,
                "long_window": long_window,
                "train_sharpe": best_parameter["sharpe"],
                "test_final_value": test_metrics["final_value"],
                "test_sharpe": test_metrics["sharpe"],
            }
        )

        # 保存每个考试年度的每日结果，稍后按日期纵向拼接。
        test_data["test_year"] = test_year
        test_data["selected_short_window"] = short_window
        test_data["selected_long_window"] = long_window
        out_of_sample_parts.append(test_data)

    fold_table = pd.DataFrame(fold_results)
    out_of_sample_data = (
        pd.concat(
            out_of_sample_parts,
            ignore_index=True,
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    # 两条曲线使用完全相同的样本外日期，才能进行公平比较。
    out_of_sample_data["walk_forward_curve"] = (
        1 + out_of_sample_data["net_return"]
    ).cumprod()
    out_of_sample_data["buy_hold_curve"] = (
        1
        + out_of_sample_data["daily_return"]
        .fillna(0)
    ).cumprod()

    return fold_table, out_of_sample_data



def add_diagnostic_columns(result_table):
    diagnostic_table = result_table.copy()
    diagnostic_table["sharpe_change"] = (
        diagnostic_table["out_sharpe"]
        - diagnostic_table["in_sharpe"]
    )
    diagnostic_table["out_profitable"] = (
        diagnostic_table["out_final_value"] > 1
    )
    diagnostic_table["out_positive_sharpe"] = (
        diagnostic_table["out_sharpe"] > 0
    )

    return diagnostic_table


def build_metric_matrix(
    result_table,
    metric_name,
):
    metric_matrix = (
        result_table.pivot(
            index="short_window",
            columns="long_window",
            values=metric_name,
        )
        .sort_index()
        .sort_index(axis=1)
    )

    return metric_matrix


def save_results(
    result_table,
    symbol,
    split_date,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    safe_split_date = split_date.replace("-", "")
    output_path = (
        OUTPUT_DIR
        / (
            f"{symbol}_"
            f"parameter_sensitivity_"
            f"{safe_split_date}.csv"
        )
    )
    result_table.to_csv(
        output_path,
        index=False,
    )

    return output_path



if __name__ == "__main__":
    symbol = "000001"

    features = load_features(symbol)
    parameter_pairs = build_parameters_grid()

    start_date = features["date"].min()
    end_date = features["date"].max()

    print(f"股票代码：{symbol}")
    print(f"数据行数：{len(features)}")
    print(f"数据范围：{start_date.date()} 至 {end_date.date()}")
    print(f"有效参数组合数量：{len(parameter_pairs)}")

    print("\n准备测试的参数组合：")

    for number, pair in enumerate(
        parameter_pairs,
        start=1,
    ):
        short_window, long_window = pair

        print(
            f"{number:02d}. "
            f"短期均线={short_window}，"
            f"长期均线={long_window}"
        )

    test_short_window = 10
    test_long_window = 60

    test_result = calculate_strategy(
        data=features,
        short_window=test_short_window,
        long_window=test_long_window,
    )

    final_net_value = (
        test_result["net_curve"].iloc[-1]
    )

    trade_count = int(
        test_result["turnover"].sum()
    )

    position_values = sorted(
        int(value)
        for value in test_result["position"].unique()
    )

    trade_values = sorted(
        int(value)
        for value in test_result["trade"].unique()
    )

    print("\n单组参数回测检查：")
    print(
        f"测试参数："
        f"{test_short_window}/"
        f"{test_long_window}"
    )
    print(f"成本后最终净值：{final_net_value:.4f}")
    print(f"交易动作次数：{trade_count}")
    print(f"仓位可能值：{position_values}")
    print(f"交易可能值：{trade_values}")

    print("\n最近五行计算结果：")
    print(
        test_result[
            [
                "date",
                "close",
                "short_ma",
                "long_ma",
                "signal",
                "position",
                "trade",
                "net_return",
                "net_curve",
            ]
        ]
        .tail()
        .to_string(index=False)
    )

    assert (
        test_result["position"]
        .isin([0, 1])
        .all()
    )
    assert (
        test_result["trade"]
        .isin([-1, 0, 1])
        .all()
    )
    assert (
        test_result["net_curve"]
        .notna()
        .all()
    )
    print("单组参数自动检查：全部通过")

    full_result_table = run_all_parameters(
        data=features,
        parameter_pairs=parameter_pairs,
    )

    assert len(full_result_table) == len(parameter_pairs)
    assert not full_result_table[
        ["short_window", "long_window"]
    ].duplicated().any()
    assert (
        full_result_table["final_value"] > 0
    ).all()

    print("\n全部历史参数比较：")
    print(
        full_result_table
        .round(4)
        .to_string(index=False)
    )

    split_timestamp = pd.Timestamp(SPLIT_DATE)
    in_sample_rows = int(
        (
            features["date"] < split_timestamp
        ).sum()
    )
    out_sample_rows = int(
        (
            features["date"] >= split_timestamp
        ).sum()
    )

    assert (
        in_sample_rows + out_sample_rows
        == len(features)
    )
    assert in_sample_rows > 0
    assert out_sample_rows > 0

    print("\n时间切分检查：")
    print(f"切分日期：{SPLIT_DATE}")
    print(f"样本内行数：{in_sample_rows}")
    print(f"样本外行数：{out_sample_rows}")

    sample_result_table = run_sample_analysis(
        data=features,
        parameter_pairs=parameter_pairs,
        split_date=SPLIT_DATE,
    )

    assert (
        len(sample_result_table)
        == len(parameter_pairs)
    )
    assert sample_result_table[
        "in_sharpe"
    ].is_monotonic_decreasing

    display_columns = [
        "short_window",
        "long_window",
        "in_final_value",
        "in_max_drawdown",
        "in_sharpe",
        "out_final_value",
        "out_max_drawdown",
        "out_sharpe",
    ]

    print("\n样本内与样本外比较：")
    print(
        sample_result_table[
            display_columns
        ]
        .round(4)
        .to_string(index=False)
    )

    diagnostic_table = add_diagnostic_columns(
        sample_result_table
    )
    in_sharpe_matrix = build_metric_matrix(
        result_table=diagnostic_table,
        metric_name="in_sharpe",
    )
    out_sharpe_matrix = build_metric_matrix(
        result_table=diagnostic_table,
        metric_name="out_sharpe",
    )

    print("\n样本内夏普矩阵：")
    print(
        in_sharpe_matrix
        .round(4)
        .to_string()
    )
    print("\n样本外夏普矩阵：")
    print(
        out_sharpe_matrix
        .round(4)
        .to_string()
    )

    total_parameter_count = len(diagnostic_table)
    profitable_count = int(
        diagnostic_table[
            "out_profitable"
        ].sum()
    )
    positive_sharpe_count = int(
        diagnostic_table[
            "out_positive_sharpe"
        ].sum()
    )
    best_parameter = diagnostic_table.iloc[0]

    print("\n稳健性检查：")
    print(
        f"参数组合总数："
        f"{total_parameter_count}"
    )
    print(
        f"样本外最终净值大于1："
        f"{profitable_count}/"
        f"{total_parameter_count}"
    )
    print(
        f"样本外夏普大于0："
        f"{positive_sharpe_count}/"
        f"{total_parameter_count}"
    )
    print(
        f"样本内冠军："
        f"{int(best_parameter['short_window'])}/"
        f"{int(best_parameter['long_window'])}"
    )
    print(
        f"冠军样本内夏普："
        f"{best_parameter['in_sharpe']:.4f}"
    )
    print(
        f"冠军样本外夏普："
        f"{best_parameter['out_sharpe']:.4f}"
    )
    print(
        f"冠军夏普变化："
        f"{best_parameter['sharpe_change']:.4f}"
    )

    output_path = save_results(
        result_table=diagnostic_table,
        symbol=symbol,
        split_date=SPLIT_DATE,
    )

    assert output_path.exists()
    assert len(diagnostic_table) == len(parameter_pairs)
    assert diagnostic_table[
        "out_profitable"
    ].dtype == bool
    assert diagnostic_table[
        "out_positive_sharpe"
    ].dtype == bool

    print(f"\n完整结果已保存到：{output_path}")
    print("结果保存与诊断检查：全部通过")

    (
        walk_forward_table,
        walk_forward_daily,
    ) = run_walk_forward_analysis(
        data=features,
        parameter_pairs=parameter_pairs,
        test_years=WALK_FORWARD_TEST_YEARS,
    )
    print("\n滚动样本外验证：")
    print(
        walk_forward_table
        .round(4)
        .to_string(index=False)
    )

    # 连续数据必须按日期递增，而且四个考试年度之间不能有重复日期。
    assert walk_forward_daily[
        "date"
    ].is_monotonic_increasing
    assert not walk_forward_daily[
        "date"
    ].duplicated().any()

    strategy_metrics = calculate_metrics(
        walk_forward_daily
    )
    buy_hold_metrics = calculate_buy_hold_metrics(
        walk_forward_daily
    )
    comparison_table = build_benchmark_comparison(
        strategy_metrics=strategy_metrics,
        buy_hold_metrics=buy_hold_metrics,
    )

    # 验证保存的净值曲线和绩效函数重新计算的最终净值一致。
    assert abs(
        walk_forward_daily[
            "walk_forward_curve"
        ].iloc[-1]
        - strategy_metrics["final_value"]
    ) < 1e-12
    assert abs(
        walk_forward_daily[
            "buy_hold_curve"
        ].iloc[-1]
        - buy_hold_metrics["final_value"]
    ) < 1e-12

    print("\n连续样本外绩效：")
    print(f"交易日数量：{len(walk_forward_daily)}")
    print(
        f"最终净值："
        f"{strategy_metrics['final_value']:.4f}"
    )
    print(
        f"年化收益率："
        f"{strategy_metrics['annual_return']:.2%}"
    )
    print(
        f"最大回撤："
        f"{strategy_metrics['max_drawdown']:.2%}"
    )
    print(
        f"夏普比率："
        f"{strategy_metrics['sharpe']:.4f}"
    )
    print(
        f"交易动作次数："
        f"{strategy_metrics['trade_count']}"
    )

    print("\n策略与买入持有基准对照：")
    print(
        comparison_table
        .round(4)
        .to_string(index=False)
    )

    final_value_advantage = (
        strategy_metrics["final_value"]
        - buy_hold_metrics["final_value"]
    )
    drawdown_improvement = (
        strategy_metrics["max_drawdown"]
        - buy_hold_metrics["max_drawdown"]
    )

    print(
        f"策略相对买入持有的最终净值优势："
        f"{final_value_advantage:.4f}"
    )
    print(
        f"策略最大回撤改善："
        f"{drawdown_improvement:.2%}"
    )
