from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest
from conftest import make_prices

from aquant.data import MarketDataRepository, atomic_write_parquet
from aquant.industries import IndustryClassificationService
from aquant.screening import ScreeningEngine
from aquant.storage import StateStore


@pytest.fixture(autouse=True)
def forbid_real_industry_network(monkeypatch) -> None:
    """行业分类测试必须完全使用固定数据，避免误访真实接口。"""

    def unexpected_network_call(*args, **kwargs):
        raise AssertionError("测试不应访问真实行业数据接口")

    monkeypatch.setattr(
        "aquant.industries.ak.stock_industry_clf_hist_sw",
        unexpected_network_call,
    )
    monkeypatch.setattr(
        "aquant.industries.ak.stock_industry_change_cninfo",
        unexpected_network_call,
    )


def test_sw_bulk_selects_latest_record_on_or_before_as_of(app_config) -> None:
    service = IndustryClassificationService(app_config)
    raw = pd.DataFrame(
        {
            "symbol": ["000001", "000001", "000001", "600519"],
            "start_date": ["2020-01-01", "2024-03-01", "2027-01-01", "2021-06-01"],
            "industry_code": ["480100", "710100", "720100", "340300"],
            "update_time": ["2020-01-02", "2024-03-02", "2027-01-02", "2021-06-02"],
        }
    )

    records = service._normalize_sw_bulk(raw, date(2025, 12, 31))
    result = pd.DataFrame(records).set_index("symbol")

    assert result.loc["000001", "industry"] == "计算机"
    assert result.loc["000001", "industry_code"] == "710100"
    assert result.loc["000001", "industry_as_of"] == "2024-03-01"
    assert result.loc["000001", "industry_level"] == "一级"
    assert result.loc["600519", "industry"] == "食品饮料"


def test_sw_bulk_uses_system_curl_when_akshare_download_fails(
    app_config, monkeypatch
) -> None:
    monkeypatch.setattr(
        "aquant.industries.ak.stock_industry_clf_hist_sw",
        lambda: (_ for _ in ()).throw(RuntimeError("证书链错误")),
    )
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        assert kwargs["check"]
        assert kwargs["capture_output"]
        return SimpleNamespace(stdout=b"fixed-industry-workbook")

    monkeypatch.setattr("aquant.industries.subprocess.run", fake_run)
    monkeypatch.setattr(
        "aquant.industries.pd.read_excel",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "股票代码": ["000001"],
                "计入日期": ["2021-07-30"],
                "行业代码": ["480301"],
                "更新日期": ["2021-07-31"],
            }
        ),
    )

    result = IndustryClassificationService(app_config)._fetch_sw_bulk()

    assert result.columns.tolist() == [
        "symbol",
        "start_date",
        "industry_code",
        "update_time",
    ]
    assert result.iloc[0]["symbol"] == "000001"
    assert calls and calls[0][0] == "curl"
    assert "?download=1" in calls[0][-1]


def test_cninfo_prefers_current_sw_standard_over_newer_association_record(
    app_config, monkeypatch
) -> None:
    raw = pd.DataFrame(
        {
            "变更日期": ["2024-01-10", "2025-06-30"],
            "分类标准编码": ["008003", "008021"],
            "分类标准": ["申银万国行业分类标准", "中国上市公司协会行业分类"],
            "行业编码": ["710100", "I65"],
            "行业门类": ["计算机", "信息传输、软件和信息技术服务业"],
            "行业大类": ["计算机设备", "软件和信息技术服务业"],
            "行业中类": [None, None],
            "行业次类": [None, None],
        }
    )
    monkeypatch.setattr(
        "aquant.industries.ak.stock_industry_change_cninfo",
        lambda **kwargs: raw.copy(),
    )

    record = IndustryClassificationService(app_config)._fetch_cninfo(
        "000001", date(2025, 12, 31)
    )

    assert record["status"] == "ok"
    assert record["industry"] == "计算机"
    assert record["industry_code"] == "710100"
    assert record["industry_standard"] == "申银万国行业分类标准"
    assert record["industry_as_of"] == "2024-01-10"


def test_manual_override_takes_priority_over_provider_classification(app_config) -> None:
    store = StateStore(app_config.storage.database)
    store.upsert_industry_classification(
        {
            "symbol": "000001",
            "industry": "银行",
            "industry_code": "480100",
            "industry_level": "一级",
            "industry_standard": "申万行业分类",
            "industry_source": "fixed_test_data",
            "industry_as_of": "2025-01-01",
            "status": "ok",
            "error": "",
            "attempted_at": "2025-01-02T00:00:00",
        }
    )
    store.set_industry_override("000001", "金融科技", "人工复核")

    resolved = IndustryClassificationService(app_config, store).resolve(["000001"])
    row = resolved.iloc[0]

    assert row["industry"] == "金融科技"
    assert row["industry_source"] == "manual"
    assert bool(row["is_manual"])
    assert row["industry_status"] == "ok"


def test_explicit_empty_symbol_list_does_not_mean_all_symbols(app_config) -> None:
    store = StateStore(app_config.storage.database)
    store.set_industry_override("000001", "银行")

    resolved = IndustryClassificationService(app_config, store).status([])

    assert resolved.empty
    assert resolved.columns.tolist() == IndustryClassificationService.RESOLVED_COLUMNS


def test_failed_classification_does_not_erase_last_valid_industry(app_config) -> None:
    store = StateStore(app_config.storage.database)
    store.upsert_industry_classification(
        {
            "symbol": "600519",
            "industry": "食品饮料",
            "industry_code": "340300",
            "industry_level": "一级",
            "industry_standard": "申万行业分类",
            "industry_source": "fixed_test_data",
            "industry_as_of": "2025-01-01",
            "status": "ok",
            "error": "",
            "attempted_at": "2025-01-02T00:00:00",
        }
    )

    store.upsert_industry_classification(
        {
            "symbol": "600519",
            "industry": None,
            "industry_source": "akshare_cninfo",
            "status": "error",
            "error": "TimeoutError: 请求超时",
            "attempted_at": "2025-02-02T00:00:00",
        }
    )
    row = store.industry_classifications(["600519"]).iloc[0]

    assert row["industry"] == "食品饮料"
    assert row["industry_code"] == "340300"
    assert row["industry_source"] == "fixed_test_data"
    assert row["status"] == "error"
    assert row["error"] == "TimeoutError: 请求超时"

    store.upsert_industry_classification(
        {
            "symbol": "600519",
            "industry": "旧行业",
            "industry_code": "110000",
            "industry_source": "stale_bulk",
            "industry_as_of": "2024-12-31",
            "status": "ok",
            "attempted_at": "2025-03-01T00:00:00",
        }
    )
    preserved = store.industry_classifications(["600519"]).iloc[0]
    assert preserved["industry"] == "食品饮料"
    assert preserved["industry_code"] == "340300"
    assert preserved["industry_source"] == "fixed_test_data"
    assert preserved["industry_as_of"] == "2025-01-01"


def test_screening_uses_independent_industry_for_stock_without_factor_snapshot(
    app_config,
) -> None:
    app_config.screening.universe_source = "local"
    app_config.screening.minimum_average_amount = 0
    app_config.screening.selection_count = 2
    app_config.screening.technical_preselect_count = 2
    repository = MarketDataRepository(app_config)
    symbols = ["000001", "000002"]
    for offset, symbol in enumerate(symbols):
        history = make_prices(symbol, periods=180, pattern="trend")
        history["close"] *= 1 + offset * 0.1
        history["open"] = history["close"] * 0.999
        history["high"] = history[["open", "close"]].max(axis=1) * 1.01
        history["low"] = history[["open", "close"]].min(axis=1) * 0.99
        atomic_write_parquet(history, repository.path_for(symbol, "qfq"))

    atomic_write_parquet(
        pd.DataFrame(
            {
                "symbol": symbols,
                "name": ["测试银行", "测试电子"],
                "is_st": [False, False],
                "updated_at": ["2026-01-01", "2026-01-01"],
            }
        ),
        app_config.storage.universe_file,
    )
    # 基本面快照故意只放 000001，000002 的行业必须来自独立分类表。
    atomic_write_parquet(
        pd.DataFrame(
            {
                "symbol": ["000001"],
                "factor_date": [pd.Timestamp("2025-01-01")],
                "industry": ["银行"],
                "pe_ratio": [10.0],
                "pb_ratio": [1.0],
                "roe": [12.0],
                "profit_growth": [8.0],
                "debt_ratio": [40.0],
            }
        ),
        app_config.storage.factor_file,
    )
    store = StateStore(app_config.storage.database)
    store.upsert_industry_classification(
        {
            "symbol": "000002",
            "industry": "电子",
            "industry_code": "270100",
            "industry_level": "一级",
            "industry_standard": "申万行业分类",
            "industry_source": "fixed_test_data",
            "industry_as_of": "2025-01-01",
            "status": "ok",
            "error": "",
            "attempted_at": "2025-01-02T00:00:00",
        }
    )

    result = ScreeningEngine(app_config, repository, store).run(force=True)
    classified = result.results.set_index("symbol")

    assert classified.loc["000002", "industry"] == "电子"
    assert classified.loc["000002", "industry_source"] == "fixed_test_data"
    assert classified.loc["000002", "industry_status"] == "ok"
