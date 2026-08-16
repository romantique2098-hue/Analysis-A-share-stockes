from __future__ import annotations

from datetime import date

import pandas as pd

from aquant.backtest import BacktestEngine, save_backtest_result
from aquant.config import load_config, save_config
from aquant.models import Position
from aquant.storage import StateStore


def test_three_symbol_backtest_persists_all_artifacts(app_config, three_histories) -> None:
    result = BacktestEngine(app_config).run(three_histories)
    store = StateStore(app_config.storage.database)
    output = save_backtest_result(result, app_config, store)
    assert (output / "equity_curve.parquet").exists()
    assert (output / "trades.csv").exists()
    assert (output / "signals.parquet").exists()
    assert (output / "metrics.json").exists()
    assert (output / "annual_returns.csv").exists()
    assert (output / "rolling_relative.parquet").exists()
    assert (output / "manifest.json").exists()
    manifest = (output / "manifest.json").read_text(encoding="utf-8")
    assert save_backtest_result(result, app_config, store) == output
    assert (output / "manifest.json").read_text(encoding="utf-8") == manifest
    assert len(store.table("backtest_runs")) == 1


def test_streamlit_pages_smoke() -> None:
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    app = Path(__file__).parents[1] / "src" / "aquant" / "app.py"
    result = AppTest.from_file(str(app)).run(timeout=10)
    assert not result.exception
    for page in ["数据与股票池", "自动选股", "回测研究", "模拟盘"]:
        result.sidebar.radio[0].set_value(page).run(timeout=10)
        assert not result.exception, page


def test_backtest_page_only_shows_selected_strategy_parameters() -> None:
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    app = Path(__file__).parents[1] / "src" / "aquant" / "app.py"
    result = AppTest.from_file(str(app)).run(timeout=10)
    result.sidebar.radio[0].set_value("回测研究").run(timeout=10)

    trend = {
        "短均线",
        "长均线",
        "突破回看期",
        "ATR 周期",
        "ATR 止损倍数",
    }
    momentum = {
        "跳过最近交易日",
        "动量入选数量",
        "动量排名缓冲",
        "短期动量周期",
        "中期动量周期",
        "长期动量周期",
    }
    mean_reversion = {
        "RSI 周期",
        "RSI 入场",
        "RSI 退出",
        "布林带周期",
        "布林带标准差",
        "短期超跌周期",
        "超跌收益阈值",
    }
    expected_by_strategy = {
        "均线趋势": {"短均线", "长均线", "均线缓冲带", "均线确认交易日"},
        "趋势突破与波动止损": trend,
        "横截面动量": momentum,
        "RSI 均值回归": {"RSI 周期", "RSI 入场", "RSI 退出"},
        "布林带超跌反弹": mean_reversion,
        "多策略等权投票": trend | momentum | mean_reversion | {"组合入场阈值"},
        "多策略风险加权": trend
        | momentum
        | mean_reversion
        | {"组合入场阈值", "风险权重窗口"},
    }

    for strategy_label, expected_labels in expected_by_strategy.items():
        selector = next(item for item in result.selectbox if item.label == "策略")
        result = selector.set_value(strategy_label).run(timeout=10)
        assert not result.exception, strategy_label
        parameter_box = next(
            item for item in result.expander if item.label == "策略参数"
        )
        actual_labels = {item.label for item in parameter_box.number_input}
        assert actual_labels == expected_labels, strategy_label


def test_auto_screening_can_add_selected_stock_to_watchlist(
    app_config, monkeypatch
) -> None:
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    app_config.config_path = app_config.project_root / "config" / "settings.yaml"
    save_config(app_config)
    output_dir = app_config.storage.output_dir / "screens" / "ui-test"
    output_dir.mkdir(parents=True)
    results = pd.DataFrame(
        [
            {
                "rank": 1,
                "symbol": "300750",
                "name": "宁德时代",
                "industry": "电力设备",
                "selected": True,
                "eligible": True,
                "total_score": 0.91,
                "factor_coverage": 1.0,
                "trend_signal": 1,
                "mean_reversion_signal": 0,
                "return_20": 0.08,
                "return_60": 0.16,
                "return_120": 0.25,
                "annual_volatility": 0.22,
                "average_amount_20": 2_000_000_000.0,
            },
            {
                "rank": 2,
                "symbol": "000001",
                "name": "平安银行",
                "industry": "银行",
                "selected": True,
                "eligible": True,
                "total_score": 0.82,
                "factor_coverage": 1.0,
                "trend_signal": 1,
                "mean_reversion_signal": 0,
                "return_20": 0.05,
                "return_60": 0.12,
                "return_120": 0.18,
                "annual_volatility": 0.18,
                "average_amount_20": 1_000_000_000.0,
            },
        ]
    )
    results.to_parquet(output_dir / "screening.parquet", index=False)
    store = StateStore(app_config.storage.database)
    store.save_screening_run(
        "ui-test", date(2025, 1, 10), "SUCCESS", results, output_dir, []
    )
    monkeypatch.setenv("AQUANT_CONFIG", str(app_config.config_path))

    app = Path(__file__).parents[1] / "src" / "aquant" / "app.py"
    result = AppTest.from_file(str(app)).run(timeout=10)
    result = result.sidebar.radio[0].set_value("自动选股").run(timeout=10)
    assert not result.exception

    chooser = next(
        item for item in result.multiselect if item.label == "选择要加入的股票"
    )
    assert chooser.options == ["300750 · 宁德时代 · 电力设备"]
    result = chooser.set_value(["300750"]).run(timeout=10)
    add_button = next(item for item in result.button if item.label == "加入所选股票")
    assert not add_button.disabled
    result = add_button.click().run(timeout=10)

    assert not result.exception
    assert load_config(app_config.config_path).market.watchlist == [
        "000001",
        "600519",
        "000858",
        "300750",
    ]


def test_running_screen_only_disables_the_same_background_task(
    app_config, monkeypatch
) -> None:
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    app_config.config_path = app_config.project_root / "config" / "settings.yaml"
    save_config(app_config)
    store = StateStore(app_config.storage.database)
    store.create_background_job(
        job_id="active-screen",
        kind="screen",
        lock_key="aquant-long-task",
        payload={"arguments": {}},
        log_path=app_config.storage.log_dir / "jobs" / "active-screen.log",
    )
    monkeypatch.setenv("AQUANT_CONFIG", str(app_config.config_path))

    app = Path(__file__).parents[1] / "src" / "aquant" / "app.py"
    result = AppTest.from_file(str(app)).run(timeout=10)
    result = result.sidebar.radio[0].set_value("自动选股").run(timeout=10)
    buttons = {item.label: item for item in result.button}
    assert buttons["后台执行筛选"].disabled
    assert not buttons["后台同步A股代码"].disabled
    assert not buttons["后台轮换更新行情"].disabled

    result = result.sidebar.radio[0].set_value("数据与股票池").run(timeout=10)
    update_button = next(
        item for item in result.button if item.label == "后台更新全部行情"
    )
    assert not update_button.disabled

    result = result.sidebar.radio[0].set_value("回测研究").run(timeout=10)
    research_buttons = {item.label: item for item in result.button}
    assert not research_buttons["后台运行所选策略"].disabled
    assert not research_buttons["后台对比全部策略"].disabled
    assert not research_buttons["后台均线稳健性分析"].disabled

    result = result.sidebar.radio[0].set_value("模拟盘").run(timeout=10)
    daily_button = next(
        item for item in result.button if item.label == "后台运行每日流程"
    )
    assert not daily_button.disabled


def test_saved_backtest_can_be_promoted_from_research_page(
    app_config, three_histories, monkeypatch
) -> None:
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    app_config.config_path = app_config.project_root / "config" / "settings.yaml"
    save_config(app_config)
    run_config = app_config.model_copy(deep=True)
    run_config.strategy.name = "rsi_reversion"
    candidate = BacktestEngine(run_config).run(three_histories)
    store = StateStore(app_config.storage.database)
    store.ensure_account(app_config.portfolio.initial_cash)
    store.save_position(Position("000001", 100, 20.0))
    save_backtest_result(candidate, run_config, store)
    monkeypatch.setenv("AQUANT_CONFIG", str(app_config.config_path))

    app = Path(__file__).parents[1] / "src" / "aquant" / "app.py"
    result = AppTest.from_file(str(app)).run(timeout=10)
    result = result.sidebar.radio[0].set_value("回测研究").run(timeout=10)
    assert not result.exception
    saved_selector = next(
        item for item in result.selectbox if item.label == "已保存回测"
    )
    assert any(candidate.run_id in option for option in saved_selector.options)
    confirm = next(
        item
        for item in result.checkbox
        if item.label == "我已复核回测结果，并确认切换后保留当前资金与持仓"
    )
    promotion_button = next(
        item for item in result.button if item.label == "晋级为模拟盘策略"
    )
    assert promotion_button.disabled

    result = confirm.set_value(True).run(timeout=10)
    promotion_button = next(
        item for item in result.button if item.label == "晋级为模拟盘策略"
    )
    assert not promotion_button.disabled
    result = promotion_button.click().run(timeout=10)

    assert not result.exception
    deployed = StateStore(app_config.storage.database).active_paper_strategy_deployment()
    assert deployed is not None
    assert deployed["strategy_name"] == "rsi_reversion"
    assert deployed["source_backtest_run_id"] == candidate.run_id
    assert StateStore(app_config.storage.database).positions()["000001"].quantity == 100
    assert load_config(app_config.config_path).strategy.name == "ma_cross"
