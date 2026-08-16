from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SOURCE_PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = (
    SOURCE_PROJECT_ROOT if (SOURCE_PROJECT_ROOT / "pyproject.toml").exists() else Path.cwd()
)
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"
PACKAGED_CONFIG_PATH = Path(__file__).with_name("default_settings.yaml")


class MarketConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    watchlist: list[str] = Field(default_factory=lambda: ["000001"])
    benchmark: str = "000300"
    start_date: str = "2020-01-01"
    max_stale_days: int = Field(default=7, ge=1)

    @field_validator("watchlist")
    @classmethod
    def validate_watchlist(cls, values: list[str]) -> list[str]:
        normalized = list(dict.fromkeys(str(value).zfill(6) for value in values))
        if not normalized or any(not value.isdigit() or len(value) != 6 for value in normalized):
            raise ValueError("股票池至少包含一个六位数字代码")
        return normalized

    @field_validator("benchmark")
    @classmethod
    def validate_benchmark(cls, value: str) -> str:
        value = str(value).zfill(6)
        if not value.isdigit() or len(value) != 6:
            raise ValueError("基准代码必须是六位数字")
        return value


class StrategyConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    name: str = "ma_cross"
    short_window: int = Field(default=5, ge=2)
    long_window: int = Field(default=20, ge=3)
    ma_signal_buffer: float = Field(default=0.003, ge=0, lt=0.2)
    ma_confirmation_days: int = Field(default=2, ge=1, le=20)
    breakout_window: int = Field(default=20, ge=5)
    atr_window: int = Field(default=14, ge=2)
    atr_stop_multiplier: float = Field(default=3.0, gt=0)
    momentum_lookback: int = Field(default=60, ge=5)
    momentum_skip_recent: int = Field(default=5, ge=0)
    momentum_top_n: int = Field(default=2, ge=1)
    momentum_rank_buffer: int = Field(default=2, ge=0)
    momentum_short_period: int = Field(default=20, ge=5)
    momentum_medium_period: int = Field(default=60, ge=10)
    momentum_long_period: int = Field(default=120, ge=20)
    rsi_window: int = Field(default=14, ge=2)
    rsi_entry: float = Field(default=30.0, ge=0, le=100)
    rsi_exit: float = Field(default=55.0, ge=0, le=100)
    bollinger_window: int = Field(default=20, ge=5)
    bollinger_std: float = Field(default=2.0, gt=0)
    oversold_window: int = Field(default=5, ge=2)
    oversold_return: float = Field(default=-0.08, ge=-1, lt=0)
    ensemble_threshold: float = Field(default=0.5, gt=0, le=1)
    ensemble_volatility_window: int = Field(default=20, ge=5)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        allowed = {
            "ma_cross",
            "trend_breakout",
            "momentum",
            "rsi_reversion",
            "mean_reversion",
            "ensemble_equal",
            "ensemble_risk",
        }
        if value not in allowed:
            raise ValueError(f"未知策略：{value}")
        return value

    @model_validator(mode="after")
    def validate_windows(self) -> StrategyConfig:
        if self.short_window >= self.long_window:
            raise ValueError("短均线周期必须小于长均线周期")
        if self.momentum_skip_recent >= self.momentum_lookback:
            raise ValueError("动量跳过周期必须小于回看周期")
        if not (
            self.momentum_short_period < self.momentum_medium_period < self.momentum_long_period
        ):
            raise ValueError("动量周期必须满足短期 < 中期 < 长期")
        if self.rsi_entry >= self.rsi_exit:
            raise ValueError("RSI 入场阈值必须小于退出阈值")
        return self


class PortfolioConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    initial_cash: float = Field(default=100_000.0, gt=0)
    max_position_weight: float = Field(default=0.20, gt=0, le=1)
    lot_size: int = Field(default=100, ge=1)
    risk_free_rate: float = Field(default=0.02, ge=-1, le=1)
    allocation_method: str = "inverse_volatility"
    volatility_window: int = Field(default=60, ge=10)
    target_annual_volatility: float = Field(default=0.15, gt=0, le=1)
    cash_buffer: float = Field(default=0.05, ge=0, lt=1)
    max_portfolio_drawdown: float = Field(default=0.20, gt=0, lt=1)
    drawdown_cooldown_days: int = Field(default=20, ge=1)
    max_positions: int = Field(default=20, ge=1)
    min_holding_days: int = Field(default=3, ge=1)
    minimum_rebalance_weight_change: float = Field(default=0.02, ge=0, lt=1)
    max_daily_turnover: float = Field(default=0.20, gt=0, le=2)

    @field_validator("allocation_method")
    @classmethod
    def validate_allocation_method(cls, value: str) -> str:
        if value not in {"equal", "inverse_volatility"}:
            raise ValueError("allocation_method 必须是 equal 或 inverse_volatility")
        return value


class CostConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    commission_rate: float = Field(default=0.00025, ge=0)
    minimum_commission: float = Field(default=5.0, ge=0)
    stamp_duty_rate: float = Field(default=0.0005, ge=0)
    slippage_bps: float = Field(default=5.0, ge=0)


class StorageConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    raw_dir: Path = Path("data/raw")
    adjusted_dir: Path = Path("data/adjusted")
    benchmark_dir: Path = Path("data/benchmark")
    features_dir: Path = Path("data/features")
    output_dir: Path = Path("output")
    database: Path = Path("state/aquant.sqlite3")
    log_dir: Path = Path("logs")
    universe_file: Path = Path("data/universe.parquet")
    factor_file: Path = Path("data/factors/latest.parquet")


class DataConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    provider: str = "akshare"
    timeout_seconds: float = Field(default=15, gt=0)
    retries: int = Field(default=3, ge=1, le=10)
    overlap_days: int = Field(default=15, ge=1)


class ScreeningConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    enabled: bool = True
    universe_source: str = "local"
    schedule: str = "weekly"
    schedule_weekday: int = Field(default=5, ge=1, le=7)
    selection_count: int = Field(default=20, ge=1)
    technical_preselect_count: int = Field(default=100, ge=1)
    minimum_history_days: int = Field(default=130, ge=30)
    minimum_average_amount: float = Field(default=20_000_000.0, ge=0)
    exclude_st: bool = True
    universe_batch_size: int = Field(default=200, ge=1, le=1000)
    industry_batch_size: int = Field(default=200, ge=1, le=1000)
    max_per_industry: int = Field(default=4, ge=1)
    momentum_weight: float = Field(default=0.25, ge=0)
    value_weight: float = Field(default=0.20, ge=0)
    quality_weight: float = Field(default=0.20, ge=0)
    low_volatility_weight: float = Field(default=0.20, ge=0)
    liquidity_weight: float = Field(default=0.15, ge=0)

    @field_validator("universe_source")
    @classmethod
    def validate_universe_source(cls, value: str) -> str:
        if value not in {"watchlist", "local", "all_a"}:
            raise ValueError("universe_source 必须是 watchlist、local 或 all_a")
        return value

    @field_validator("schedule")
    @classmethod
    def validate_schedule(cls, value: str) -> str:
        if value not in {"daily", "weekly", "monthly"}:
            raise ValueError("schedule 必须是 daily、weekly 或 monthly")
        return value

    @model_validator(mode="after")
    def validate_factor_weights(self) -> ScreeningConfig:
        total = (
            self.momentum_weight
            + self.value_weight
            + self.quality_weight
            + self.low_volatility_weight
            + self.liquidity_weight
        )
        if total <= 0:
            raise ValueError("多因子权重之和必须大于 0")
        if self.technical_preselect_count < self.selection_count:
            raise ValueError("技术预选数量不能小于最终入选数量")
        return self


class AppConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    market: MarketConfig = Field(default_factory=MarketConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    costs: CostConfig = Field(default_factory=CostConfig)
    screening: ScreeningConfig = Field(default_factory=ScreeningConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    project_root: Path = PROJECT_ROOT
    config_path: Path = DEFAULT_CONFIG_PATH

    def resolve_paths(self) -> AppConfig:
        for name in (
            "raw_dir",
            "adjusted_dir",
            "benchmark_dir",
            "features_dir",
            "output_dir",
            "database",
            "log_dir",
            "universe_file",
            "factor_file",
        ):
            path = getattr(self.storage, name)
            if not path.is_absolute():
                setattr(self.storage, name, self.project_root / path)
        return self

    def ensure_directories(self) -> None:
        for path in (
            self.storage.raw_dir,
            self.storage.adjusted_dir,
            self.storage.benchmark_dir,
            self.storage.features_dir,
            self.storage.output_dir,
            self.storage.database.parent,
            self.storage.log_dir,
            self.storage.universe_file.parent,
            self.storage.factor_file.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path | None = None) -> AppConfig:
    config_path = Path(path or DEFAULT_CONFIG_PATH).expanduser().resolve()
    read_path = config_path
    if not read_path.exists() and config_path == DEFAULT_CONFIG_PATH.resolve():
        read_path = PACKAGED_CONFIG_PATH
    if not read_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    payload = yaml.safe_load(read_path.read_text(encoding="utf-8")) or {}
    root = config_path.parent.parent if config_path.parent.name == "config" else PROJECT_ROOT
    config = AppConfig.model_validate(
        {**payload, "project_root": root, "config_path": config_path}
    ).resolve_paths()
    return config


def extend_watchlist(config: AppConfig, symbols: list[str]) -> list[str]:
    """Merge symbols into the watchlist and return the newly added normalized codes."""
    existing = list(config.market.watchlist)
    market_payload = config.market.model_dump()
    market_payload["watchlist"] = [*existing, *symbols]
    merged = MarketConfig.model_validate(market_payload).watchlist
    existing_set = set(existing)
    added = [symbol for symbol in merged if symbol not in existing_set]
    if added:
        config.market.watchlist = merged
    return added


def save_config(config: AppConfig, path: str | Path | None = None) -> Path:
    output = Path(path or config.config_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = config.model_dump(
        exclude={"project_root", "config_path"}, mode="json"
    )
    storage = payload["storage"]
    for key, value in list(storage.items()):
        absolute = Path(value)
        try:
            storage[key] = str(absolute.relative_to(config.project_root))
        except ValueError:
            storage[key] = str(absolute)
    serialized = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(serialized)
            temporary_path = Path(temporary.name)
        temporary_path.replace(output)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return output
