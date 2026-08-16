from __future__ import annotations

import io
import logging
import re
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date, datetime

import akshare as ak
import pandas as pd

from aquant.config import AppConfig
from aquant.storage import StateStore

# 申万行业分类标准 2021 的一级行业代码。个股历史接口返回六位行业代码，
# 前两位对应一级行业。保留少量旧版前缀，便于尚无新版记录的股票回退。
SW_LEVEL1_INDUSTRIES: dict[str, str] = {
    "11": "农林牧渔",
    "21": "煤炭",  # 旧版“采掘”中的煤炭股票
    "22": "基础化工",
    "23": "钢铁",
    "24": "有色金属",
    "27": "电子",
    "28": "汽车",
    "33": "家用电器",
    "34": "食品饮料",
    "35": "纺织服饰",
    "36": "轻工制造",
    "37": "医药生物",
    "41": "公用事业",
    "42": "交通运输",
    "43": "房地产",
    "44": "银行",  # 申万旧版银行代码
    "45": "商贸零售",
    "46": "社会服务",
    "48": "银行",
    "49": "非银金融",
    "51": "综合",
    "61": "建筑材料",
    "62": "建筑装饰",
    "63": "电力设备",
    "64": "机械设备",
    "65": "国防军工",
    "71": "计算机",
    "72": "传媒",
    "73": "通信",
    "74": "煤炭",
    "75": "石油石化",
    "76": "环保",
    "77": "美容护理",
}

IndustryProgressCallback = Callable[[int, int, str, str], None]


@dataclass
class IndustryUpdateResult:
    requested: int
    attempted: int
    succeeded: int
    failed: int
    bulk_classified: int = 0
    warnings: list[str] = field(default_factory=list)


class IndustryClassificationService:
    """独立维护行业分类，并按“手工覆盖 > 数据源 > 旧因子缓存”解析。"""

    RESOLVED_COLUMNS = [
        "symbol",
        "industry",
        "industry_code",
        "industry_level",
        "industry_standard",
        "industry_source",
        "industry_as_of",
        "industry_status",
        "industry_error",
        "industry_updated_at",
        "is_manual",
    ]

    def __init__(self, config: AppConfig, store: StateStore | None = None) -> None:
        self.config = config
        self.store = store or StateStore(config.storage.database)
        self.logger = logging.getLogger("aquant.industries")

    @staticmethod
    def _symbols(symbols: Iterable[str] | None) -> list[str]:
        if symbols is None:
            return []
        normalized = [str(symbol).strip().zfill(6) for symbol in symbols]
        return list(dict.fromkeys(symbol for symbol in normalized if symbol.isdigit()))

    @staticmethod
    def _industry_from_sw_code(value: object) -> tuple[str, str]:
        match = re.search(r"(\d{6})", str(value))
        if not match:
            return "", ""
        code = match.group(1)
        return SW_LEVEL1_INDUSTRIES.get(code[:2], ""), code

    def _legacy_factors(self) -> pd.DataFrame:
        path = self.config.storage.factor_file
        if not path.exists():
            return pd.DataFrame(columns=["symbol", "industry"])
        try:
            frame = pd.read_parquet(path, columns=["symbol", "industry"])
        except (OSError, ValueError, KeyError):
            return pd.DataFrame(columns=["symbol", "industry"])
        frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
        frame["industry"] = frame["industry"].astype("string").str.strip()
        frame = frame[
            frame["industry"].notna()
            & frame["industry"].ne("")
            & frame["industry"].ne("未分类")
        ]
        return frame.drop_duplicates("symbol", keep="last")

    def status(self, symbols: Iterable[str] | None = None) -> pd.DataFrame:
        explicit_symbols = symbols is not None
        requested = self._symbols(symbols)
        if explicit_symbols and not requested:
            return pd.DataFrame(columns=self.RESOLVED_COLUMNS)
        provider = self.store.industry_classifications(requested if explicit_symbols else None)
        if provider.empty:
            provider = pd.DataFrame(
                columns=[
                    "symbol",
                    "industry",
                    "industry_code",
                    "industry_level",
                    "industry_standard",
                    "industry_source",
                    "industry_as_of",
                    "status",
                    "error",
                    "attempted_at",
                ]
            )
        provider = provider.copy()
        if not provider.empty:
            provider["symbol"] = provider["symbol"].astype(str).str.zfill(6)

        legacy = self._legacy_factors()
        if requested:
            legacy = legacy[legacy["symbol"].isin(requested)]
        legacy = legacy.rename(columns={"industry": "legacy_industry"})

        overrides = self.store.industry_overrides()
        if overrides.empty:
            overrides = pd.DataFrame(columns=["symbol", "industry", "note", "updated_at"])
        overrides = overrides.rename(
            columns={"industry": "manual_industry", "updated_at": "manual_updated_at"}
        )
        if not overrides.empty:
            overrides["symbol"] = overrides["symbol"].astype(str).str.zfill(6)
            if requested:
                overrides = overrides[overrides["symbol"].isin(requested)]

        if requested:
            result = pd.DataFrame({"symbol": requested})
        else:
            all_symbols = set(provider.get("symbol", pd.Series(dtype=str)).tolist())
            all_symbols.update(legacy.get("symbol", pd.Series(dtype=str)).tolist())
            all_symbols.update(overrides.get("symbol", pd.Series(dtype=str)).tolist())
            result = pd.DataFrame({"symbol": sorted(all_symbols)})
        result = result.merge(provider, on="symbol", how="left")
        result = result.merge(legacy, on="symbol", how="left")
        result = result.merge(overrides, on="symbol", how="left")

        provider_industry = result.get("industry", pd.Series(index=result.index, dtype="object"))
        provider_industry = provider_industry.where(
            provider_industry.notna()
            & provider_industry.astype("string").str.strip().ne("")
            & provider_industry.astype("string").str.strip().ne("未分类")
        )
        legacy_industry = result.get(
            "legacy_industry", pd.Series(index=result.index, dtype="object")
        )
        manual_industry = result.get(
            "manual_industry", pd.Series(index=result.index, dtype="object")
        )
        has_manual = manual_industry.notna() & manual_industry.astype("string").str.strip().ne("")
        has_provider = provider_industry.notna()
        has_legacy = legacy_industry.notna()
        has_provider_row = result.get(
            "status", pd.Series(index=result.index, dtype="object")
        ).notna()
        resolved = pd.Series(index=result.index, dtype="object")
        resolved.loc[has_legacy] = legacy_industry.loc[has_legacy]
        resolved.loc[has_provider] = provider_industry.loc[has_provider]
        resolved.loc[has_manual] = manual_industry.loc[has_manual]
        result["industry"] = resolved.fillna("未分类")

        result["is_manual"] = has_manual
        result["industry_source"] = result.get(
            "industry_source", pd.Series(index=result.index, dtype="object")
        )
        result.loc[has_legacy & ~has_provider, "industry_source"] = "legacy_factor"
        result.loc[has_manual, "industry_source"] = "manual"
        result["industry_status"] = result.get(
            "status", pd.Series(index=result.index, dtype="object")
        ).fillna("missing")
        # 数据源上次失败时继续使用旧有效分类，但保留 error 状态供页面解释。
        result.loc[has_legacy & result["industry_status"].eq("missing"), "industry_status"] = "ok"
        result.loc[has_manual, "industry_status"] = "ok"
        result["industry_error"] = result.get(
            "error", pd.Series(index=result.index, dtype="object")
        ).fillna("")
        result.loc[has_legacy & ~has_provider_row, "industry_error"] = ""
        result.loc[has_manual, "industry_error"] = ""
        result["industry_updated_at"] = result.get(
            "attempted_at", pd.Series(index=result.index, dtype="object")
        )
        result.loc[has_manual, "industry_updated_at"] = result.loc[
            has_manual, "manual_updated_at"
        ]
        for column in [
            "industry_code",
            "industry_level",
            "industry_standard",
            "industry_as_of",
        ]:
            if column not in result:
                result[column] = None
        return result.reindex(columns=self.RESOLVED_COLUMNS)

    def resolve(self, symbols: Iterable[str]) -> pd.DataFrame:
        return self.status(symbols)

    def coverage(self, symbols: Iterable[str]) -> dict[str, int]:
        frame = self.status(symbols)
        classified = int(frame["industry"].ne("未分类").sum()) if not frame.empty else 0
        return {
            "total": len(frame),
            "classified": classified,
            "unclassified": len(frame) - classified,
            "manual": int(frame["is_manual"].sum()) if not frame.empty else 0,
            "errors": int(frame["industry_status"].eq("error").sum()) if not frame.empty else 0,
        }

    def _normalize_sw_bulk(self, raw: pd.DataFrame, as_of: date) -> list[dict[str, object]]:
        required = {"symbol", "start_date", "industry_code"}
        if raw.empty or not required <= set(raw.columns):
            missing = sorted(required - set(raw.columns))
            raise RuntimeError(f"申万行业批量数据为空或缺少字段：{', '.join(missing)}")
        frame = raw[list(required | ({"update_time"} & set(raw.columns)))].copy()
        frame["symbol"] = (
            frame["symbol"].astype(str).str.extract(r"(\d+)", expand=False).str.zfill(6)
        )
        frame.loc[~frame["symbol"].str.fullmatch(r"\d{6}", na=False), "symbol"] = pd.NA
        frame["start_date"] = pd.to_datetime(frame["start_date"], errors="coerce")
        frame = frame[
            frame["symbol"].notna()
            & frame["start_date"].notna()
            & frame["start_date"].dt.date.le(as_of)
        ].copy()
        frame = frame.sort_values(["symbol", "start_date"]).drop_duplicates(
            "symbol", keep="last"
        )
        now = datetime.now().isoformat(timespec="seconds")
        records: list[dict[str, object]] = []
        for row in frame.itertuples(index=False):
            industry, code = self._industry_from_sw_code(row.industry_code)
            if not industry:
                continue
            records.append(
                {
                    "symbol": row.symbol,
                    "industry": industry,
                    "industry_code": code,
                    "industry_level": "一级",
                    "industry_standard": "申万行业分类2021",
                    "industry_source": "akshare_sw_bulk",
                    "industry_as_of": pd.Timestamp(row.start_date).date().isoformat(),
                    "status": "ok",
                    "error": "",
                    "attempted_at": now,
                }
            )
        if not records:
            raise RuntimeError("申万行业批量数据中没有可识别的当前行业")
        return records

    def _network_call(self, function: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        """给 AkShare 中没有 timeout 参数的行业接口补上进程级超时。"""
        can_alarm = (
            threading.current_thread() is threading.main_thread()
            and hasattr(signal, "SIGALRM")
            and hasattr(signal, "setitimer")
        )
        if not can_alarm:
            return function()

        def timeout_handler(signum, frame) -> None:  # noqa: ARG001
            raise TimeoutError(f"行业接口超过 {self.config.data.timeout_seconds:g} 秒未返回")

        previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, self.config.data.timeout_seconds)
        try:
            return function()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)

    def _fetch_sw_bulk(self) -> pd.DataFrame:
        """读取申万全量表；AkShare 失败时用系统 curl 处理站点证书链。"""
        try:
            return self._network_call(ak.stock_industry_clf_hist_sw)
        except Exception as akshare_error:
            url = (
                "https://www.swsresearch.com/swindex/pdf/SwClass2021/"
                "StockClassifyUse_stock.xls?download=1"
            )
            try:
                completed = subprocess.run(
                    [
                        "curl",
                        "--fail",
                        "--location",
                        "--silent",
                        "--show-error",
                        "--max-time",
                        str(max(1, int(self.config.data.timeout_seconds))),
                        "--user-agent",
                        "Mozilla/5.0",
                        url,
                    ],
                    check=True,
                    capture_output=True,
                    timeout=self.config.data.timeout_seconds + 2,
                )
                raw = pd.read_excel(
                    io.BytesIO(completed.stdout),
                    dtype={"股票代码": "str", "行业代码": "str"},
                )
                return raw.rename(
                    columns={
                        "股票代码": "symbol",
                        "计入日期": "start_date",
                        "行业代码": "industry_code",
                        "更新日期": "update_time",
                    }
                )
            except Exception as curl_error:
                raise RuntimeError(
                    "申万全量行业下载失败；"
                    f"AkShare={type(akshare_error).__name__}: {akshare_error}; "
                    f"curl={type(curl_error).__name__}: {curl_error}"
                ) from curl_error

    def _fetch_cninfo(self, symbol: str, as_of: date) -> dict[str, object]:
        raw = self._network_call(
            lambda: ak.stock_industry_change_cninfo(
                symbol=symbol,
                start_date="19900101",
                end_date=as_of.strftime("%Y%m%d"),
            )
        )
        now = datetime.now().isoformat(timespec="seconds")
        if raw.empty:
            return {
                "symbol": symbol,
                "industry": None,
                "industry_source": "akshare_cninfo",
                "status": "empty",
                "error": "行业接口返回空数据",
                "attempted_at": now,
            }
        data = raw.copy()
        data["变更日期"] = pd.to_datetime(data.get("变更日期"), errors="coerce")
        data = data[data["变更日期"].dt.date.le(as_of)]
        standard = data.get("分类标准", pd.Series(index=data.index, dtype="object")).astype(str)
        code = data.get("分类标准编码", pd.Series(index=data.index, dtype="object")).astype(str)
        current_sw = data[code.eq("008003") | standard.str.contains("申银万国", na=False)]
        current_sw = current_sw[~standard.loc[current_sw.index].str.contains("旧", na=False)]
        old_sw = data[code.eq("008018") | standard.str.contains("申银万国.*旧", na=False)]
        selected_standard = ""
        if not current_sw.empty:
            selected = current_sw.sort_values("变更日期").iloc[-1]
            selected_standard = str(selected.get("分类标准", "申银万国行业分类标准"))
        elif not old_sw.empty:
            selected = old_sw.sort_values("变更日期").iloc[-1]
            selected_standard = str(selected.get("分类标准", "申银万国行业分类标准(旧)"))
        else:
            fallback_order = ["中证行业分类标准", "中国上市公司协会", "证监会行业分类"]
            selected = None
            for name in fallback_order:
                candidates = data[standard.str.contains(name, na=False)]
                if not candidates.empty:
                    selected = candidates.sort_values("变更日期").iloc[-1]
                    selected_standard = str(selected.get("分类标准", name))
                    break
            if selected is None:
                return {
                    "symbol": symbol,
                    "industry": None,
                    "industry_source": "akshare_cninfo",
                    "status": "empty",
                    "error": "没有可识别的行业分类标准",
                    "attempted_at": now,
                }

        industry, normalized_code = self._industry_from_sw_code(selected.get("行业编码", ""))
        level = "一级"
        if not industry:
            for column in ["行业大类", "行业中类", "行业次类", "行业门类"]:
                value = str(selected.get(column, "")).strip()
                if value and value.lower() != "nan":
                    industry = value
                    level = "数据源原始层级"
                    break
        if not industry:
            return {
                "symbol": symbol,
                "industry": None,
                "industry_source": "akshare_cninfo",
                "status": "empty",
                "error": "行业记录缺少名称",
                "attempted_at": now,
            }
        raw_code = str(selected.get("行业编码", ""))
        return {
            "symbol": symbol,
            "industry": industry,
            "industry_code": normalized_code or raw_code,
            "industry_level": level,
            "industry_standard": selected_standard,
            "industry_source": "akshare_cninfo",
            "industry_as_of": pd.Timestamp(selected["变更日期"]).date().isoformat(),
            "status": "ok",
            "error": "",
            "attempted_at": now,
        }

    def update(
        self,
        symbols: Iterable[str],
        *,
        batch_size: int = 200,
        try_bulk: bool = True,
        as_of: date | None = None,
        progress_callback: IndustryProgressCallback | None = None,
    ) -> IndustryUpdateResult:
        requested = self._symbols(symbols)
        target_date = as_of or datetime.now().date()
        warnings: list[str] = []
        bulk_classified = 0
        if try_bulk:
            try:
                records = self._normalize_sw_bulk(self._fetch_sw_bulk(), target_date)
                self.store.upsert_industry_classifications(records)
                bulk_classified = len(records)
            except Exception as exc:
                message = f"申万全量行业接口不可用，已改用逐股回填：{type(exc).__name__}: {exc}"
                self.logger.warning(message)
                warnings.append(message)

        # 旧因子文件只作为显示回退，不能阻止统一申万口径的独立分类刷新。
        provider = self.store.industry_classifications(requested)
        provider_ok = set(
            provider.loc[
                provider["status"].eq("ok") & provider["industry"].notna(), "symbol"
            ].astype(str)
        )
        attempted = set(provider["symbol"].astype(str))
        manual = set(
            self.store.industry_overrides()
            .get("symbol", pd.Series(dtype=str))
            .astype(str)
        )
        never_attempted = [
            symbol
            for symbol in requested
            if symbol not in provider_ok and symbol not in attempted and symbol not in manual
        ]
        retryable = [
            symbol for symbol in requested if symbol not in provider_ok and symbol not in manual
            and symbol in attempted
        ]
        # 先推进从未查询的股票；接口失败项排到其后，避免一个持续故障的代码
        # 阻塞整个宇宙。requested 的顺序由调用方用最新筛选结果做优先级。
        pending = never_attempted + retryable
        batch = pending[: max(1, min(int(batch_size), len(pending)))] if pending else []

        succeeded = 0
        failed = 0
        for completed, symbol in enumerate(batch, start=1):
            record: dict[str, object] | None = None
            error: Exception | None = None
            for attempt in range(min(2, self.config.data.retries)):
                try:
                    record = self._fetch_cninfo(symbol, target_date)
                    error = None
                    break
                except Exception as exc:
                    error = exc
                    if attempt == 0:
                        time.sleep(0.25)
            if record is None:
                record = {
                    "symbol": symbol,
                    "industry": None,
                    "industry_source": "akshare_cninfo",
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}" if error else "未知错误",
                    "attempted_at": datetime.now().isoformat(timespec="seconds"),
                }
            self.store.upsert_industry_classification(record)
            if record.get("status") == "ok":
                succeeded += 1
            else:
                failed += 1
            if progress_callback:
                progress_callback(completed, len(batch), symbol, str(record.get("status")))

        return IndustryUpdateResult(
            requested=len(requested),
            attempted=len(batch),
            succeeded=succeeded,
            failed=failed,
            bulk_classified=bulk_classified,
            warnings=warnings,
        )
