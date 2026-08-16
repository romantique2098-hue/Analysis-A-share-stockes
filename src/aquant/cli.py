from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from aquant.backtest import (
    BacktestEngine,
    load_and_compare_strategies,
    load_and_run_backtest,
    save_backtest_result,
    save_strategy_comparison,
)
from aquant.config import DEFAULT_CONFIG_PATH, AppConfig, load_config
from aquant.daily import run_daily
from aquant.data import MarketDataRepository
from aquant.industries import IndustryClassificationService
from aquant.jobs import JOB_LABELS, BackgroundJobManager, tail_job_log
from aquant.logging import configure_logging
from aquant.robustness import load_and_analyze_ma_robustness, save_ma_robustness
from aquant.screening import ScreeningEngine, UniverseService
from aquant.storage import StateStore
from aquant.strategy import STRATEGY_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aquant", description="A股量化分析与模拟盘")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="配置文件路径")
    parser.add_argument("--verbose", action="store_true", help="显示调试日志")
    subparsers = parser.add_subparsers(dest="command", required=True)
    data_parser = subparsers.add_parser("data", help="行情数据操作")
    data_subparsers = data_parser.add_subparsers(dest="data_command", required=True)
    data_subparsers.add_parser("update", help="更新股票池和基准行情")

    universe_parser = subparsers.add_parser("universe", help="A股股票宇宙操作")
    universe_subparsers = universe_parser.add_subparsers(dest="universe_command", required=True)
    universe_subparsers.add_parser("sync", help="同步全部A股代码与名称")
    universe_subparsers.add_parser("status", help="查看代码宇宙与本地行情覆盖率")
    universe_update = universe_subparsers.add_parser("update", help="轮换更新一批全市场历史行情")
    universe_update.add_argument("--batch-size", type=int, help="本批股票数量")
    universe_update.add_argument(
        "--include-raw",
        action="store_true",
        help="同时更新未复权行情；全市场筛选通常不需要",
    )

    industry_parser = subparsers.add_parser("industry", help="A股行业分类操作")
    industry_subparsers = industry_parser.add_subparsers(
        dest="industry_command", required=True
    )
    industry_subparsers.add_parser("status", help="查看行业分类覆盖率")
    industry_update = industry_subparsers.add_parser(
        "update", help="优先全量更新，并逐股补全一批未分类股票"
    )
    industry_update.add_argument("--batch-size", type=int, help="逐股回填数量")
    industry_update.add_argument(
        "--skip-bulk", action="store_true", help="跳过申万全量接口，直接逐股回填"
    )
    industry_set = industry_subparsers.add_parser("set", help="手工设置单只股票行业")
    industry_set.add_argument("symbol", help="六位股票代码")
    industry_set.add_argument("industry", help="行业名称")
    industry_set.add_argument("--note", default="", help="修正说明")
    industry_remove = industry_subparsers.add_parser("remove", help="删除手工行业覆盖")
    industry_remove.add_argument("symbol", help="六位股票代码")

    backtest = subparsers.add_parser("backtest", help="执行并保存组合回测")
    backtest.add_argument("--strategy", choices=STRATEGY_NAMES, help="覆盖配置中的策略")
    backtest_mode = backtest.add_mutually_exclusive_group()
    backtest_mode.add_argument(
        "--compare-all", action="store_true", help="对比全部内置策略"
    )
    backtest_mode.add_argument(
        "--ma-robustness",
        action="store_true",
        help="按连续时间段检验均线候选参数稳健性",
    )
    backtest.add_argument("--start", help="回测开始日期 YYYY-MM-DD")
    backtest.add_argument("--end", help="回测结束日期 YYYY-MM-DD")

    daily = subparsers.add_parser("daily-run", help="运行每日模拟盘流程")
    daily.add_argument("--skip-update", action="store_true", help="只使用本地行情")
    subparsers.add_parser("app", help="启动 Streamlit 本地网页")
    subparsers.add_parser("doctor", help="检查依赖、配置、数据库和行情")
    screen = subparsers.add_parser("screen", help="按计划执行股票筛选")
    screen.add_argument("--force", action="store_true", help="忽略周期，立即执行")
    screen.add_argument("--refresh-universe", action="store_true", help="筛选前同步A股代码表")
    screen.add_argument("--refresh-fundamentals", action="store_true", help="更新技术预选股基本面")
    jobs = subparsers.add_parser("jobs", help="查看独立后台任务")
    jobs.add_argument("--limit", type=int, default=20, help="最多显示多少条任务")
    jobs.add_argument("--show", help="显示指定任务的完整状态和最近日志")
    return parser


def _load(args: argparse.Namespace) -> AppConfig:
    config = load_config(args.config)
    config.ensure_directories()
    configure_logging(config.storage.log_dir, args.verbose)
    return config


def _ensure_no_background_conflict(config: AppConfig) -> None:
    active = BackgroundJobManager(config).active_job()
    if active:
        raise RuntimeError(
            f"后台任务 {active['job_id']}（{active['kind']}）正在运行；"
            "请等待完成后再执行前台耗时命令"
        )


def command_data_update(config: AppConfig) -> int:
    reports = MarketDataRepository(config).update_all()
    print(pd.DataFrame(report.as_dict() for report in reports).to_string(index=False))
    failed = [report for report in reports if report.used_cache]
    if failed:
        print(f"警告：{len(failed)} 组行情更新失败，保留了上一次缓存。", file=sys.stderr)
        return 2
    return 0


def command_backtest(config: AppConfig, args: argparse.Namespace) -> int:
    start = pd.Timestamp(args.start).date() if args.start else None
    end = pd.Timestamp(args.end).date() if args.end else None
    if args.ma_robustness:
        summary, details = load_and_analyze_ma_robustness(
            config,
            start=start,
            end=end,
        )
        output = save_ma_robustness(summary, details, config)
        print(summary.to_string(index=False))
        print(f"均线稳健性分析目录：{output}")
        return 0
    if args.compare_all:
        metrics, curves, results = load_and_compare_strategies(config, start=start, end=end)
        for result in results.values():
            save_backtest_result(result, config)
        output = save_strategy_comparison(metrics, curves, config)
        columns = [
            "strategy",
            "total_return",
            "annual_return",
            "max_drawdown",
            "sharpe",
            "total_costs",
        ]
        print(metrics[columns].to_string(index=False))
        print(f"策略对比目录：{output}")
        return 0
    if args.strategy and args.strategy != config.strategy.name:
        config.strategy.name = args.strategy
    if args.start or args.end:
        repository = MarketDataRepository(config)
        histories = {symbol: repository.load(symbol, "qfq") for symbol in config.market.watchlist}
        benchmark = repository.load(config.market.benchmark, "", benchmark=True)
        result = BacktestEngine(config).run(
            histories,
            benchmark,
            start=start,
            end=end,
        )
    else:
        result = load_and_run_backtest(config)
    output = save_backtest_result(result, config)
    print(f"回测编号：{result.run_id}")
    print(pd.Series(result.metrics, name="value").to_string())
    print(f"结果目录：{output}")
    return 0


def command_daily(config: AppConfig, skip_update: bool) -> int:
    result = run_daily(config, update_data=not skip_update)
    print(f"{result.run_date} [{result.status}] {result.message}")
    if result.deployment_id:
        print(
            f"模拟盘策略：{result.strategy_name}，部署编号：{result.deployment_id}"
        )
    if result.snapshot:
        print(
            f"现金 {result.snapshot.cash:.2f}，持仓市值 {result.snapshot.market_value:.2f}，"
            f"总权益 {result.snapshot.equity:.2f}"
        )
    if result.report_dir:
        print(f"模拟盘报告：{result.report_dir}")
    if config.screening.enabled:
        try:
            screening = ScreeningEngine(config).run(
                force=False,
                refresh_fundamentals=True,
            )
            if screening.status == "SUCCESS":
                print(
                    f"自动筛选 {screening.run_id}：从 {len(screening.results)} 只中选出 "
                    f"{len(screening.selected)} 只"
                )
                for warning in screening.warnings:
                    print(f"筛选警告：{warning}", file=sys.stderr)
        except Exception as exc:
            print(f"筛选警告：每日模拟盘已完成，但自动筛选失败：{exc}", file=sys.stderr)
    return 0 if result.status == "SUCCESS" else 2


def command_universe(config: AppConfig, args: argparse.Namespace) -> int:
    service = UniverseService(config)
    if args.universe_command == "sync":
        universe = service.sync()
        print(f"已同步 {len(universe)} 只A股：{config.storage.universe_file}")
        return 0
    if args.universe_command == "status":
        coverage = service.coverage()
        ratio = coverage["adjusted"] / coverage["universe"] if coverage["universe"] else 0.0
        print(f"代码宇宙：{coverage['universe']} 只")
        print(f"前复权研究行情：{coverage['adjusted']} 只（{ratio:.2%}）")
        print(f"未复权实际行情：{coverage['raw']} 只")
        print(f"两类行情齐全：{coverage['both']} 只")
        return 0
    reports = service.update_next_batch(args.batch_size, include_raw=args.include_raw)
    failed = sum(report.used_cache for report in reports)
    symbols = len({report.symbol for report in reports})
    coverage = service.coverage()
    ratio = coverage["adjusted"] / coverage["universe"] if coverage["universe"] else 0.0
    print(
        f"本批更新 {symbols} 只股票，异常/缓存 {failed} 组；"
        f"研究行情覆盖 {coverage['adjusted']}/{coverage['universe']}（{ratio:.2%}）"
    )
    return 2 if failed else 0


def command_industry(config: AppConfig, args: argparse.Namespace) -> int:
    store = StateStore(config.storage.database)
    service = IndustryClassificationService(config, store)
    if args.industry_command == "set":
        store.set_industry_override(args.symbol, args.industry, args.note)
        print(f"已将 {str(args.symbol).zfill(6)} 手工归类为：{args.industry}")
        return 0
    if args.industry_command == "remove":
        removed = store.remove_industry_override(args.symbol)
        print("已删除手工分类。" if removed else "该股票没有手工分类。")
        return 0

    universe = UniverseService(config).load()
    if config.screening.exclude_st and "is_st" in universe:
        universe = universe[~universe["is_st"].fillna(False).astype(bool)]
    symbols = universe["symbol"].astype(str).str.zfill(6).tolist()
    if args.industry_command == "status":
        coverage = service.coverage(symbols)
        ratio = coverage["classified"] / coverage["total"] if coverage["total"] else 0
        print(
            f"行业分类：{coverage['classified']}/{coverage['total']}（{ratio:.2%}）；"
            f"未分类 {coverage['unclassified']}，接口错误 {coverage['errors']}，"
            f"手工修正 {coverage['manual']}"
        )
        return 0

    result = service.update(
        symbols,
        batch_size=args.batch_size or config.screening.industry_batch_size,
        try_bulk=not args.skip_bulk,
        progress_callback=lambda completed, total, symbol, status: print(
            f"\r逐股行业回填 {completed}/{total}：{symbol} {status}", end="", flush=True
        ),
    )
    if result.attempted:
        print()
    coverage = service.coverage(symbols)
    for warning in result.warnings:
        print(f"警告：{warning}", file=sys.stderr)
    print(
        f"申万全量写入 {result.bulk_classified}；逐股尝试 {result.attempted}，"
        f"成功 {result.succeeded}，失败 {result.failed}；"
        f"当前覆盖 {coverage['classified']}/{coverage['total']}"
    )
    return 2 if result.failed else 0


def command_screen(config: AppConfig, args: argparse.Namespace) -> int:
    result = ScreeningEngine(config).run(
        force=args.force,
        refresh_universe=args.refresh_universe,
        refresh_fundamentals=args.refresh_fundamentals,
    )
    if result.status == "SKIPPED":
        print("尚未到配置的筛选周期，本次跳过。使用 --force 可立即执行。")
        return 0
    columns = [
        "rank",
        "symbol",
        "name",
        "industry",
        "total_score",
        "trend_signal",
        "mean_reversion_signal",
        "return_20",
        "return_60",
        "return_120",
    ]
    print(result.selected[columns].to_string(index=False))
    for warning in result.warnings:
        print(f"警告：{warning}", file=sys.stderr)
    print(f"筛选结果：{result.output_dir}")
    return 0


def command_jobs(config: AppConfig, args: argparse.Namespace) -> int:
    manager = BackgroundJobManager(config)
    jobs = manager.jobs(limit=max(1, int(args.limit)))
    if args.show:
        selected = manager.store.background_job(args.show)
        if selected is None:
            print(f"没有找到后台任务：{args.show}", file=sys.stderr)
            return 2
        print(pd.Series(selected).to_string())
        log = tail_job_log(selected)
        if log:
            print("\n最近日志：")
            print(log)
        return 0
    if jobs.empty:
        print("尚无后台任务。")
        return 0
    display = jobs[["job_id", "kind", "status", "progress", "message", "created_at"]].copy()
    display["kind"] = display["kind"].map(JOB_LABELS).fillna(display["kind"])
    display["status"] = display["status"].map(
        {"QUEUED": "排队中", "RUNNING": "运行中", "SUCCESS": "已完成", "FAILED": "失败"}
    ).fillna(display["status"])
    display["progress"] = display["progress"].map(lambda value: f"{float(value):.0%}")
    print(display.to_string(index=False))
    return 0


def command_app(config: AppConfig) -> int:
    environment = os.environ.copy()
    environment["AQUANT_CONFIG"] = str(config.config_path)
    app_path = Path(__file__).with_name("app.py")
    return subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(app_path)], env=environment
    )


def command_doctor(config: AppConfig) -> int:
    problems: list[str] = []
    for module in ["akshare", "pandas", "pyarrow", "pydantic", "yaml", "streamlit"]:
        status = "OK" if importlib.util.find_spec(module) else "缺失"
        print(f"依赖 {module:<12} {status}")
        if status != "OK":
            problems.append(f"缺少依赖 {module}")
    try:
        StateStore(config.storage.database)
        print(f"数据库             OK ({config.storage.database})")
    except Exception as exc:
        problems.append(f"数据库不可用：{exc}")
    repository = MarketDataRepository(config)
    adjusted_dates = repository.latest_dates("qfq")
    raw_dates = repository.latest_dates("")
    for symbol in config.market.watchlist:
        adjusted_latest = adjusted_dates[symbol]
        raw_latest = raw_dates[symbol]
        print(f"行情 {symbol:<8} 前复权 {adjusted_latest or '缺失'} 未复权 {raw_latest or '缺失'}")
        if adjusted_latest is None:
            problems.append(f"{symbol} 缺少前复权行情")
        if raw_latest is None:
            problems.append(f"{symbol} 缺少未复权行情")
    try:
        benchmark = repository.load(config.market.benchmark, "", benchmark=True)
        benchmark_latest = benchmark["date"].max().date()
        print(f"基准 {config.market.benchmark:<8} {benchmark_latest}")
    except Exception as exc:
        problems.append(f"基准 {config.market.benchmark} 不可用：{exc}")
    if config.screening.enabled:
        if config.storage.universe_file.exists():
            universe = pd.read_parquet(config.storage.universe_file)
            local_symbols = {path.stem for path in config.storage.adjusted_dir.glob("*.parquet")}
            covered = universe["symbol"].astype(str).str.zfill(6).isin(local_symbols).sum()
            print(f"股票宇宙           {len(universe)} 只，本地覆盖 {covered} 只")
        else:
            print("股票宇宙           缺失")
            if config.screening.universe_source == "all_a":
                problems.append("全 A 股代码表缺失，请运行 aquant universe sync")
        if config.storage.factor_file.exists():
            factors = pd.read_parquet(config.storage.factor_file)
            print(f"因子快照           {len(factors)} 只")
        else:
            print("因子快照           尚未生成（首次筛选时可刷新）")
        if config.storage.universe_file.exists():
            industry_universe = universe
            if config.screening.exclude_st and "is_st" in industry_universe:
                industry_universe = industry_universe[
                    ~industry_universe["is_st"].fillna(False).astype(bool)
                ]
            symbols = industry_universe["symbol"].astype(str).str.zfill(6).tolist()
            industry_coverage = IndustryClassificationService(
                config, StateStore(config.storage.database)
            ).coverage(symbols)
            print(
                f"行业分类           {industry_coverage['classified']}/"
                f"{industry_coverage['total']} 只"
            )
    stale = repository.stale_symbols()
    for symbol, message in stale.items():
        problem = f"{symbol} {message}"
        already_missing = any(item.startswith(f"{symbol} 缺少") for item in problems)
        if problem not in problems and not already_missing:
            problems.append(problem)
    if problems:
        print("\n检查未通过：", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 2
    print("\n检查通过，可以运行回测和每日任务。")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = _load(args)
        foreground_long_task = args.command in {"data", "backtest", "daily-run", "screen"} or (
            args.command == "universe" and args.universe_command != "status"
        ) or (
            args.command == "industry" and args.industry_command == "update"
        )
        if foreground_long_task:
            _ensure_no_background_conflict(config)
        if args.command == "data":
            return command_data_update(config)
        if args.command == "universe":
            return command_universe(config, args)
        if args.command == "industry":
            return command_industry(config, args)
        if args.command == "backtest":
            return command_backtest(config, args)
        if args.command == "daily-run":
            return command_daily(config, args.skip_update)
        if args.command == "app":
            return command_app(config)
        if args.command == "doctor":
            return command_doctor(config)
        if args.command == "screen":
            return command_screen(config, args)
        if args.command == "jobs":
            return command_jobs(config, args)
    except KeyboardInterrupt:
        print("操作已取消。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
