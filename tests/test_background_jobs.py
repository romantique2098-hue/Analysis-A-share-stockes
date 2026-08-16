from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from aquant.config import save_config
from aquant.data import MarketDataRepository, atomic_write_parquet
from aquant.jobs import BackgroundJobManager, parse_job_result
from aquant.storage import StateStore


def test_background_job_queue_and_state_machine(app_config) -> None:
    store = StateStore(app_config.storage.database)
    first_log = app_config.storage.log_dir / "jobs" / "first.log"
    store.create_background_job(
        job_id="first",
        kind="backtest",
        lock_key="global",
        payload={"arguments": {}},
        log_path=first_log,
    )
    store.create_background_job(
        job_id="second",
        kind="data_update",
        lock_key="global",
        payload={"arguments": {}},
        log_path=app_config.storage.log_dir / "jobs" / "second.log",
    )
    with pytest.raises(RuntimeError, match="已有同类后台任务"):
        store.create_background_job(
            job_id="duplicate",
            kind="backtest",
            lock_key="global",
            payload={"arguments": {}},
            log_path=app_config.storage.log_dir / "jobs" / "duplicate.log",
        )

    assert store.claim_background_job("first", 12345)
    assert not store.claim_background_job("second", 23456)
    store.update_background_job("first", progress=0.4, message="处理中")
    store.finish_background_job("first", result={"value": 1}, message="完成")
    finished = store.background_job("first")
    assert finished is not None
    assert finished["status"] == "SUCCESS"
    assert json.loads(str(finished["result_json"])) == {"value": 1}
    assert store.claim_background_job("second", 23456)
    store.finish_background_job("second", result={}, message="完成")

    store.create_background_job(
        job_id="next",
        kind="data_update",
        lock_key="global",
        payload={"arguments": {}},
        log_path=app_config.storage.log_dir / "jobs" / "next.log",
    )


def test_reconcile_releases_crashed_job_lock(app_config) -> None:
    store = StateStore(app_config.storage.database)
    store.create_background_job(
        job_id="orphan",
        kind="backtest",
        lock_key="aquant-long-task",
        payload={"arguments": {}},
        log_path=app_config.storage.log_dir / "jobs" / "orphan.log",
    )
    stale = (datetime.now() - timedelta(minutes=5)).isoformat(timespec="seconds")
    with store.connect() as connection:
        connection.execute(
            "UPDATE background_jobs SET heartbeat_at=?, pid=NULL WHERE job_id='orphan'",
            (stale,),
        )

    BackgroundJobManager(app_config, store).reconcile(grace_seconds=1)

    job = store.background_job("orphan")
    assert job is not None
    assert job["status"] == "FAILED"


def test_detached_background_backtest_waits_for_queue_and_completes(
    app_config,
    three_histories,
) -> None:
    config = app_config.model_copy(deep=True)
    config.config_path = config.project_root / "config" / "settings.yaml"
    save_config(config)
    repository = MarketDataRepository(config)
    for symbol, frame in three_histories.items():
        atomic_write_parquet(frame, repository.path_for(symbol, "qfq"))
    benchmark = three_histories["000001"].copy()
    benchmark["symbol"] = config.market.benchmark
    atomic_write_parquet(
        benchmark,
        repository.path_for(config.market.benchmark, "", benchmark=True),
    )

    store = StateStore(config.storage.database)
    store.create_background_job(
        job_id="blocker",
        kind="screen",
        lock_key="aquant-long-task",
        payload={"arguments": {}},
        log_path=config.storage.log_dir / "jobs" / "blocker.log",
    )
    assert store.claim_background_job("blocker", os.getpid())

    manager = BackgroundJobManager(config, store)
    job_id = manager.submit("backtest")
    del manager
    time.sleep(0.25)
    queued = store.background_job(job_id)
    assert queued is not None
    assert queued["status"] == "QUEUED"
    store.finish_background_job("blocker", result={}, message="测试阻塞任务完成")

    deadline = time.monotonic() + 20
    job = store.background_job(job_id)
    while job and job["status"] in {"QUEUED", "RUNNING"} and time.monotonic() < deadline:
        time.sleep(0.1)
        job = store.background_job(job_id)

    assert job is not None
    if job["status"] != "SUCCESS":
        log_path = Path(str(job["log_path"]))
        pytest.fail(
            f"后台任务未成功：{job['status']} {job['message']}\n"
            f"{log_path.read_text(encoding='utf-8') if log_path.exists() else ''}"
        )
    result = parse_job_result(job)
    output_dir = Path(str(result["output_dir"]))
    assert (output_dir / "manifest.json").exists()
    assert Path(str(result["result_file"])).exists()
