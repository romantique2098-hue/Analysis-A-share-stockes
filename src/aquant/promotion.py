from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

import pandas as pd

from aquant.backtest import BACKTEST_ARTIFACTS, BACKTEST_ENGINE_VERSION
from aquant.config import AppConfig, StrategyConfig
from aquant.storage import StateStore


class PaperStrategyService:
    """Resolve and explicitly promote the independently deployed paper strategy."""

    def __init__(self, config: AppConfig, store: StateStore | None = None) -> None:
        self.config = config
        self.store = store or StateStore(config.storage.database)

    @staticmethod
    def parameters(deployment: dict[str, object] | pd.Series) -> StrategyConfig:
        try:
            payload = json.loads(str(deployment["parameters_json"]))
            return StrategyConfig.model_validate(payload)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("已部署的模拟盘策略参数损坏") from exc

    def ensure_initialized(self) -> dict[str, object]:
        current = self.store.active_paper_strategy_deployment()
        if current is not None:
            self.parameters(current)
            return current
        deployment_id = f"bootstrap-{uuid.uuid4().hex[:12]}"
        return self.store.ensure_paper_strategy_deployment(
            deployment_id=deployment_id,
            strategy_name=self.config.strategy.name,
            parameters=self.config.strategy.model_dump(),
            note="从配置文件登记的初始模拟盘策略",
        )

    def current(self) -> dict[str, object]:
        return self.ensure_initialized()

    def effective_config(self) -> AppConfig:
        effective = self.config.model_copy(deep=True)
        effective.strategy = self.parameters(self.current())
        return effective

    def history(self, limit: int = 50) -> pd.DataFrame:
        self.ensure_initialized()
        return self.store.paper_strategy_deployments(limit)

    def pending(self) -> dict[str, object] | None:
        return self.store.pending_paper_strategy_deployment()

    def activate_pending_if_ready(self) -> dict[str, object] | None:
        return self.store.activate_pending_paper_strategy()

    def promote_backtest(self, run_id: str, note: str = "") -> dict[str, object]:
        source = self.store.backtest_run(run_id)
        if source is None:
            raise RuntimeError(f"没有找到回测记录：{run_id}")
        output_dir = Path(str(source["output_dir"]))
        parameters_path = output_dir / "parameters.json"
        manifest_path = output_dir / "manifest.json"
        if not parameters_path.exists() or not manifest_path.exists():
            raise RuntimeError("回测产物不完整，不能晋级为模拟盘策略")
        try:
            database_parameters = json.loads(str(source["parameters_json"]))
            artifact_parameters = json.loads(parameters_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            source_metrics = json.loads(str(source["metrics_json"]))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("回测记录或产物无法解析，不能执行晋级") from exc
        if not isinstance(manifest, dict) or not isinstance(source_metrics, dict):
            raise RuntimeError("回测记录或清单格式无效，不能执行晋级")

        def checksum(path: Path) -> str:
            digest = hashlib.sha256()
            with path.open("rb") as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        checksums = manifest.get("checksums")
        expected_run_id = hashlib.sha256(
            json.dumps(
                database_parameters,
                sort_keys=True,
                ensure_ascii=False,
            ).encode()
        ).hexdigest()[:16]
        artifacts_match = isinstance(checksums, dict) and all(
            (output_dir / name).exists()
            and checksums.get(name) == checksum(output_dir / name)
            for name in BACKTEST_ARTIFACTS
        )
        if (
            manifest.get("run_id") != run_id
            or manifest.get("backtest_engine_version") != BACKTEST_ENGINE_VERSION
            or expected_run_id != run_id
            or artifact_parameters != database_parameters
            or not artifacts_match
        ):
            raise RuntimeError("回测数据库记录与落盘产物不一致，不能执行晋级")
        strategy_payload = database_parameters.get("strategy")
        if not isinstance(strategy_payload, dict):
            raise RuntimeError("回测记录缺少完整策略参数")
        try:
            strategy = StrategyConfig.model_validate(strategy_payload)
        except ValueError as exc:
            raise RuntimeError(f"回测策略参数无效：{exc}") from exc
        deployment_id = f"deploy-{uuid.uuid4().hex[:16]}"
        return self.store.promote_paper_strategy(
            deployment_id=deployment_id,
            strategy_name=strategy.name,
            parameters=strategy.model_dump(),
            source_backtest_run_id=run_id,
            source_data_end_date=str(source["data_end_date"]),
            source_metrics=source_metrics,
            source_parameters=database_parameters,
            note=note.strip(),
        )
