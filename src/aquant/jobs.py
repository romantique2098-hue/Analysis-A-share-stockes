from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from aquant.backtest import (
    load_and_compare_strategies,
    load_and_run_backtest,
    save_backtest_result,
    save_strategy_comparison,
)
from aquant.config import AppConfig, load_config
from aquant.daily import run_daily
from aquant.data import DataQualityReport, MarketDataRepository, atomic_write_parquet
from aquant.industries import IndustryClassificationService
from aquant.logging import configure_logging
from aquant.robustness import load_and_analyze_ma_robustness, save_ma_robustness
from aquant.screening import ScreeningEngine, UniverseService
from aquant.storage import StateStore

JOB_LABELS = {
    "data_update": "更新全部行情",
    "universe_sync": "同步A股代码",
    "universe_update": "轮换更新行情",
    "industry_update": "补全行业分类",
    "screen": "多因子筛选",
    "backtest": "策略回测",
    "compare": "策略对比",
    "robustness": "均线稳健性分析",
    "daily": "每日模拟盘",
}
GLOBAL_JOB_LOCK = "aquant-long-task"
ACTIVE_STATUSES = {"QUEUED", "RUNNING"}


def _write_json_atomic(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_job_result(job: dict[str, object] | pd.Series) -> dict[str, object]:
    raw = job.get("result_json", "{}")
    if not raw:
        return {}
    try:
        return json.loads(str(raw))
    except (TypeError, ValueError):
        return {}


def tail_job_log(job: dict[str, object] | pd.Series, lines: int = 120) -> str:
    path = Path(str(job.get("log_path", "")))
    if not path.exists():
        return ""
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


class BackgroundJobManager:
    """提交脱离 Streamlit 生命周期的本地后台进程，并从 SQLite 恢复状态。"""

    def __init__(self, config: AppConfig, store: StateStore | None = None) -> None:
        self.config = config
        self.store = store or StateStore(config.storage.database)
        self.log_dir = config.storage.log_dir / "jobs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _pid_alive(pid: int | None) -> bool:
        if not pid or pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        try:
            state = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(pid)],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            ).stdout.strip()
            return bool(state) and not state.startswith("Z")
        except (OSError, subprocess.SubprocessError):
            return True

    def reconcile(self, grace_seconds: int = 30) -> None:
        jobs = self.store.background_jobs(limit=100)
        if jobs.empty:
            return
        now = datetime.now()
        for row in jobs[jobs["status"].isin(ACTIVE_STATUSES)].to_dict("records"):
            heartbeat_raw = row.get("heartbeat_at") or row.get("created_at")
            try:
                heartbeat = datetime.fromisoformat(str(heartbeat_raw))
            except ValueError:
                heartbeat = now - timedelta(seconds=grace_seconds + 1)
            if (now - heartbeat).total_seconds() <= grace_seconds:
                continue
            pid = int(row["pid"]) if pd.notna(row.get("pid")) else None
            if not self._pid_alive(pid):
                self.store.fail_background_job(
                    str(row["job_id"]),
                    "后台进程已退出且没有完成记录；任务已标记为失败，可重新提交。",
                )

    def submit(
        self,
        kind: str,
        *,
        arguments: dict[str, object] | None = None,
        job_config: AppConfig | None = None,
    ) -> str:
        if kind not in JOB_LABELS:
            raise ValueError(f"未知后台任务：{kind}")
        self.reconcile()
        effective_config = job_config or self.config
        if effective_config.storage.database != self.config.storage.database:
            raise ValueError("后台任务配置必须使用与页面相同的状态数据库")
        job_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"
        log_path = self.log_dir / f"{job_id}.log"
        payload: dict[str, object] = {
            "config": effective_config.model_dump(mode="json"),
            "arguments": arguments or {},
        }
        self.store.create_background_job(
            job_id=job_id,
            kind=kind,
            lock_key=GLOBAL_JOB_LOCK,
            payload=payload,
            log_path=log_path,
        )
        command = [
            sys.executable,
            "-m",
            "aquant.jobs",
            "worker",
            "--config",
            str(self.config.config_path),
            "--job-id",
            job_id,
        ]
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        try:
            with log_path.open("ab", buffering=0) as log_file:
                process = subprocess.Popen(
                    command,
                    cwd=str(self.config.project_root),
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                )
            self.store.attach_background_job_process(job_id, process.pid)
        except Exception as exc:
            self.store.fail_background_job(job_id, f"无法启动后台进程：{exc}")
            raise
        return job_id

    def jobs(
        self,
        kinds: list[str] | tuple[str, ...] | None = None,
        *,
        limit: int = 20,
    ) -> pd.DataFrame:
        self.reconcile()
        return self.store.background_jobs(kinds=kinds, limit=limit)

    def active_job(self) -> dict[str, object] | None:
        self.reconcile()
        return self.store.active_background_job(GLOBAL_JOB_LOCK)


ProgressCallback = Callable[[float, str], None]


def _report_progress(
    update: ProgressCallback,
    prefix: str,
) -> Callable[[int, int, DataQualityReport], None]:
    def callback(completed: int, total: int, report: DataQualityReport) -> None:
        state = "缓存/异常" if report.used_cache else "完成"
        update(
            0.05 + 0.9 * completed / max(total, 1),
            f"{prefix} {completed}/{total}：{report.symbol} {state}",
        )

    return callback


def _execute_job(
    kind: str,
    config: AppConfig,
    arguments: dict[str, Any],
    job_dir: Path,
    update: ProgressCallback,
) -> tuple[dict[str, object], str]:
    repository = MarketDataRepository(config)
    store = StateStore(config.storage.database)
    if kind == "data_update":
        update(0.02, "开始更新股票池和基准行情")
        reports = repository.update_all(
            progress_callback=_report_progress(update, "行情更新")
        )
        frame = pd.DataFrame(report.as_dict() for report in reports)
        reports_path = job_dir / "quality_reports.parquet"
        atomic_write_parquet(frame, reports_path)
        failed = int(frame["used_cache"].sum()) if not frame.empty else 0
        return (
            {
                "reports_path": str(reports_path),
                "total": len(frame),
                "failed": failed,
            },
            f"全部行情更新完成：{len(frame)} 组，异常/缓存 {failed} 组",
        )
    if kind == "universe_sync":
        update(0.1, "正在从数据源同步A股代码")
        universe = UniverseService(config, repository, store).sync()
        return (
            {"universe_file": str(config.storage.universe_file), "count": len(universe)},
            f"已同步 {len(universe)} 只A股代码",
        )
    if kind == "universe_update":
        update(0.02, "开始轮换更新全市场行情")
        reports = UniverseService(config, repository, store).update_next_batch(
            int(arguments.get("batch_size") or config.screening.universe_batch_size),
            include_raw=bool(arguments.get("include_raw", False)),
            progress_callback=_report_progress(update, "轮换更新"),
        )
        frame = pd.DataFrame(report.as_dict() for report in reports)
        reports_path = job_dir / "quality_reports.parquet"
        atomic_write_parquet(frame, reports_path)
        failed = int(frame["used_cache"].sum()) if not frame.empty else 0
        symbols = frame["symbol"].nunique() if not frame.empty else 0
        return (
            {
                "reports_path": str(reports_path),
                "symbols": int(symbols),
                "failed": failed,
            },
            f"轮换更新完成：{symbols} 只，异常/缓存 {failed} 组",
        )
    if kind == "industry_update":
        universe_scope = "非 ST A股" if config.screening.exclude_st else "A股（含 ST）"
        update(0.02, f"正在读取{universe_scope}代码宇宙")
        universe = UniverseService(config, repository, store).load()
        if config.screening.exclude_st and "is_st" in universe:
            universe = universe[~universe["is_st"].fillna(False).astype(bool)]
        symbols = universe["symbol"].astype(str).str.zfill(6).drop_duplicates().tolist()
        # 先补最近入选和高排名股票，随后再覆盖其余代码宇宙。
        screen_runs = store.table("screen_runs", 1)
        if not screen_runs.empty:
            screening_path = Path(str(screen_runs.iloc[0]["output_dir"])) / "screening.parquet"
            if screening_path.exists():
                try:
                    recent = pd.read_parquet(
                        screening_path,
                        columns=["symbol", "selected", "rank"],
                    )
                    recent["symbol"] = recent["symbol"].astype(str).str.zfill(6)
                    recent = recent.sort_values(
                        ["selected", "rank"], ascending=[False, True], na_position="last"
                    )
                    priority = recent["symbol"].drop_duplicates().tolist()
                    symbol_set = set(symbols)
                    priority = [symbol for symbol in priority if symbol in symbol_set]
                    priority_set = set(priority)
                    symbols = priority + [
                        symbol for symbol in symbols if symbol not in priority_set
                    ]
                except Exception as exc:
                    print(f"最近筛选文件无法用于行业优先级，改按代码宇宙顺序：{exc}")
        batch_size = int(
            arguments.get("batch_size") or config.screening.industry_batch_size
        )

        def industry_progress(
            completed: int,
            total: int,
            symbol: str,
            status: str,
        ) -> None:
            status_label = {
                "ok": "已分类",
                "empty": "接口无结果",
                "error": "接口错误",
            }.get(status, status)
            update(
                0.12 + 0.83 * completed / max(total, 1),
                f"行业回填 {completed}/{total}：{symbol} {status_label}",
            )

        update(0.06, "正在尝试批量行业分类，失败时将自动改用逐股回填")
        result = IndustryClassificationService(config, store).update(
            symbols,
            batch_size=batch_size,
            try_bulk=bool(arguments.get("try_bulk", True)),
            progress_callback=industry_progress,
        )
        if result.bulk_classified == 0 and result.succeeded == 0 and result.failed:
            raise RuntimeError(
                f"行业批量源不可用，逐股回填 {result.failed} 只也全部失败；"
                "已保留上一次有效分类"
            )
        update(0.97, "行业分类结果已写入本地数据库")
        payload = {
            "requested": result.requested,
            "attempted": result.attempted,
            "succeeded": result.succeeded,
            "failed": result.failed,
            "bulk_classified": result.bulk_classified,
            "warnings": result.warnings,
        }
        return (
            payload,
            "行业分类补全完成："
            f"代码宇宙 {result.requested} 只，"
            f"批量分类 {result.bulk_classified} 只，"
            f"逐股成功 {result.succeeded} 只，失败 {result.failed} 只",
        )
    if kind == "screen":
        update(0.05, "正在计算技术指标与多因子排名")

        def screen_progress(completed: int, total: int, symbol: str) -> None:
            update(
                0.05 + 0.82 * completed / max(total, 1),
                f"技术指标 {completed}/{total}：{symbol}",
            )

        result = ScreeningEngine(config, repository, store).run(
            force=True,
            refresh_fundamentals=bool(arguments.get("refresh_fundamentals", False)),
            progress_callback=screen_progress,
        )
        update(0.95, "筛选结果正在落盘")
        return (
            {
                "run_id": result.run_id,
                "output_dir": str(result.output_dir) if result.output_dir else "",
                "selected_count": len(result.selected),
                "warnings": result.warnings,
            },
            f"筛选 {result.run_id} 完成，入选 {len(result.selected)} 只",
        )
    if kind == "backtest":
        update(0.1, "正在执行组合回测")
        result = load_and_run_backtest(config)
        update(0.9, "正在保存回测结果")
        output = save_backtest_result(result, config, store)
        return (
            {"run_id": result.run_id, "output_dir": str(output), "metrics": result.metrics},
            f"回测 {result.run_id} 完成",
        )
    if kind == "compare":
        update(0.05, "正在执行全部内置策略")
        metrics, curves, results = load_and_compare_strategies(config)
        for completed, result in enumerate(results.values(), start=1):
            save_backtest_result(result, config, store)
            update(
                0.7 + 0.2 * completed / max(len(results), 1),
                f"正在保存策略结果 {completed}/{len(results)}",
            )
        output = save_strategy_comparison(metrics, curves, config)
        return (
            {"output_dir": str(output), "strategy_count": len(results)},
            f"{len(results)} 个策略对比完成",
        )
    if kind == "robustness":
        update(0.05, "正在执行连续时间分段稳健性分析")
        summary, details = load_and_analyze_ma_robustness(config)
        update(0.9, "正在保存稳健性分析")
        output = save_ma_robustness(summary, details, config)
        return (
            {"output_dir": str(output), "candidate_count": len(summary)},
            f"{len(summary)} 组均线参数稳健性分析完成",
        )
    if kind == "daily":
        update(0.05, "正在执行每日模拟盘流程")
        result = run_daily(config, update_data=bool(arguments.get("update_data", True)))
        payload = {
            "run_date": result.run_date,
            "status": result.status,
            "message": result.message,
            "report_dir": str(result.report_dir) if result.report_dir else "",
            "deployment_id": result.deployment_id,
            "strategy_name": result.strategy_name,
        }
        if result.status != "SUCCESS":
            raise RuntimeError(result.message)
        return payload, result.message
    raise ValueError(f"未知后台任务：{kind}")


def run_worker(config_path: Path, job_id: str) -> int:
    base_config = load_config(config_path)
    store = StateStore(base_config.storage.database)
    row = store.background_job(job_id)
    if row is None:
        print(f"后台任务不存在：{job_id}", flush=True)
        return 2
    manager = BackgroundJobManager(base_config, store)
    last_queue_heartbeat = 0.0
    last_waiting_message = ""
    while not store.claim_background_job(job_id, os.getpid()):
        row = store.background_job(job_id)
        if row is None:
            print(f"后台任务不存在：{job_id}", flush=True)
            return 2
        if row["status"] != "QUEUED":
            print(f"后台任务无法领取，当前状态：{row['status']}", flush=True)
            return 2
        now = time.monotonic()
        if now - last_queue_heartbeat >= 5:
            manager.reconcile()
            current = store.active_background_job(GLOBAL_JOB_LOCK)
            if current and current["job_id"] != job_id:
                waiting_for = JOB_LABELS.get(
                    str(current["kind"]), str(current["kind"])
                )
                message = f"排队等待“{waiting_for}”完成"
            else:
                message = "已进入队列，等待后台调度"
            store.touch_queued_background_job(job_id, message)
            if message != last_waiting_message:
                print(message, flush=True)
                last_waiting_message = message
            last_queue_heartbeat = now
        time.sleep(1)
    row = store.background_job(job_id)
    if row is None:
        print(f"后台任务不存在：{job_id}", flush=True)
        return 2
    payload = json.loads(str(row["payload_json"]))
    config = AppConfig.model_validate(payload["config"]).resolve_paths()
    config.ensure_directories()
    configure_logging(config.storage.log_dir, verbose=False)
    arguments = dict(payload.get("arguments") or {})
    stop_heartbeat = threading.Event()

    def heartbeat() -> None:
        while not stop_heartbeat.wait(5):
            try:
                store.update_background_job(job_id)
            except Exception:
                traceback.print_exc()

    thread = threading.Thread(target=heartbeat, name=f"job-heartbeat-{job_id}", daemon=True)
    thread.start()

    def update(progress: float, message: str) -> None:
        print(f"[{progress:6.1%}] {message}", flush=True)
        store.update_background_job(job_id, progress=progress, message=message)

    job_dir = config.storage.output_dir / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    try:
        print(f"开始后台任务 {job_id}：{row['kind']}", flush=True)
        result, message = _execute_job(
            str(row["kind"]),
            config,
            arguments,
            job_dir,
            update,
        )
        result_file = job_dir / "result.json"
        result = {**result, "result_file": str(result_file)}
        _write_json_atomic(result, result_file)
        store.finish_background_job(job_id, result=result, message=message)
        print(message, flush=True)
        return 0
    except Exception as exc:
        traceback.print_exc()
        store.fail_background_job(job_id, f"{type(exc).__name__}: {exc}")
        return 1
    finally:
        stop_heartbeat.set()
        thread.join(timeout=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AQuant 后台任务工作进程")
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--config", type=Path, required=True)
    worker.add_argument("--job-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "worker":
        return run_worker(args.config, args.job_id)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
