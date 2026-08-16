from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from aquant.backtest import load_saved_backtest
from aquant.charts import build_zoomable_line_chart
from aquant.config import (
    DEFAULT_CONFIG_PATH,
    AppConfig,
    StrategyConfig,
    extend_watchlist,
    load_config,
    save_config,
)
from aquant.data import MarketDataRepository
from aquant.industries import IndustryClassificationService
from aquant.jobs import (
    ACTIVE_STATUSES,
    JOB_LABELS,
    BackgroundJobManager,
    parse_job_result,
    tail_job_log,
)
from aquant.promotion import PaperStrategyService
from aquant.research import calendar_year_analysis, rolling_relative_analysis
from aquant.screening import UniverseService
from aquant.storage import StateStore

st.set_page_config(page_title="AQuant", page_icon="📈", layout="wide")
st.title("AQuant · A股量化分析与模拟盘")
st.caption("日频研究工具。所有订单均为模拟记录，不会连接券商或发出真实委托。")

config_path = Path(os.environ.get("AQUANT_CONFIG", DEFAULT_CONFIG_PATH))
config = load_config(config_path)
config.ensure_directories()
repository = MarketDataRepository(config)
store = StateStore(config.storage.database)
store.ensure_account(config.portfolio.initial_cash)
job_manager = BackgroundJobManager(config, store)
paper_strategy_service = PaperStrategyService(config, store)
paper_strategy_deployment = paper_strategy_service.current()

page = st.sidebar.radio("功能", ["总览", "数据与股票池", "自动选股", "回测研究", "模拟盘"])
st.sidebar.caption(f"配置：{config.config_path}")

METRIC_LABELS = {
    "total_return": "费后累计收益",
    "annual_return": "费后年化收益",
    "annual_volatility": "年化波动率",
    "sharpe": "夏普比率",
    "sortino": "Sortino 比率",
    "max_drawdown": "最大回撤",
    "calmar": "Calmar 比率",
    "win_rate": "日胜率",
    "turnover": "累计换手",
    "annualized_turnover": "年化换手",
    "total_costs": "累计交易成本",
    "trade_count": "成交笔数",
    "cost_ratio": "成本/初始资金",
    "average_exposure": "平均仓位",
    "maximum_exposure": "最高仓位",
    "final_cash_weight": "期末现金占比",
    "gross_total_return": "费前累计收益",
    "cost_drag": "成本收益拖累",
    "buy_hold_total_return": "股票池买入持有收益",
    "timing_vs_buy_hold": "择时相对买入持有",
    "benchmark_total_return": "沪深300累计收益",
    "benchmark_annual_return": "沪深300年化收益",
    "benchmark_annual_volatility": "沪深300年化波动",
    "benchmark_max_drawdown": "沪深300最大回撤",
    "excess_total_return": "累计超额收益（差值）",
    "relative_total_return": "相对沪深300收益（复合）",
    "excess_annual_return": "年化超额收益",
    "tracking_error": "年化跟踪误差",
    "information_ratio": "信息比率",
    "beta": "市场 Beta",
    "annual_alpha": "年化 Alpha",
    "benchmark_correlation": "与沪深300相关性",
}

PERCENT_METRICS = {
    "total_return",
    "annual_return",
    "annual_volatility",
    "max_drawdown",
    "win_rate",
    "cost_ratio",
    "average_exposure",
    "maximum_exposure",
    "final_cash_weight",
    "gross_total_return",
    "cost_drag",
    "buy_hold_total_return",
    "timing_vs_buy_hold",
    "benchmark_total_return",
    "benchmark_annual_return",
    "benchmark_annual_volatility",
    "benchmark_max_drawdown",
    "excess_total_return",
    "relative_total_return",
    "excess_annual_return",
    "tracking_error",
    "annual_alpha",
}

ANNUAL_COLUMN_LABELS = {
    "year": "年份",
    "net_curve_return": "费后收益",
    "gross_curve_return": "费前收益",
    "buy_hold_curve_return": "股票池买入持有",
    "benchmark_curve_return": "沪深300收益",
    "relative_curve_return": "相对沪深300收益",
    "average_exposure": "平均仓位",
    "turnover": "累计换手",
    "trade_count": "成交笔数",
    "traded_notional": "成交金额",
    "transaction_costs": "交易成本",
}

JOB_STATUS_LABELS = {
    "QUEUED": "排队中",
    "RUNNING": "运行中",
    "SUCCESS": "已完成",
    "FAILED": "失败",
}

INDUSTRY_SOURCE_LABELS = {
    "akshare_sw_bulk": "申万全量分类",
    "akshare_cninfo": "巨潮逐股回填",
    "legacy_factor": "旧因子快照",
    "manual": "手工修正",
}

STRATEGY_LABELS = {
    "均线趋势": "ma_cross",
    "趋势突破与波动止损": "trend_breakout",
    "横截面动量": "momentum",
    "RSI 均值回归": "rsi_reversion",
    "布林带超跌反弹": "mean_reversion",
    "多策略等权投票": "ensemble_equal",
    "多策略风险加权": "ensemble_risk",
}
STRATEGY_NAME_LABELS = {name: label for label, name in STRATEGY_LABELS.items()}
STRATEGY_PARAMETER_FIELDS = {
    "ma_cross": [
        "short_window",
        "long_window",
        "ma_signal_buffer",
        "ma_confirmation_days",
    ],
    "trend_breakout": [
        "short_window",
        "long_window",
        "breakout_window",
        "atr_window",
        "atr_stop_multiplier",
    ],
    "momentum": [
        "momentum_skip_recent",
        "momentum_top_n",
        "momentum_rank_buffer",
        "momentum_short_period",
        "momentum_medium_period",
        "momentum_long_period",
    ],
    "rsi_reversion": ["rsi_window", "rsi_entry", "rsi_exit"],
    "mean_reversion": [
        "rsi_window",
        "rsi_entry",
        "rsi_exit",
        "bollinger_window",
        "bollinger_std",
        "oversold_window",
        "oversold_return",
    ],
}
STRATEGY_PARAMETER_FIELDS["ensemble_equal"] = list(
    dict.fromkeys(
        STRATEGY_PARAMETER_FIELDS["trend_breakout"]
        + STRATEGY_PARAMETER_FIELDS["momentum"]
        + STRATEGY_PARAMETER_FIELDS["mean_reversion"]
        + ["ensemble_threshold"]
    )
)
STRATEGY_PARAMETER_FIELDS["ensemble_risk"] = [
    *STRATEGY_PARAMETER_FIELDS["ensemble_equal"],
    "ensemble_volatility_window",
]
STRATEGY_PARAMETER_LABELS = {
    "short_window": "短均线",
    "long_window": "长均线",
    "ma_signal_buffer": "均线缓冲带",
    "ma_confirmation_days": "均线确认交易日",
    "breakout_window": "突破回看期",
    "atr_window": "ATR 周期",
    "atr_stop_multiplier": "ATR 止损倍数",
    "momentum_skip_recent": "跳过最近交易日",
    "momentum_top_n": "动量入选数量",
    "momentum_rank_buffer": "动量排名缓冲",
    "momentum_short_period": "短期动量周期",
    "momentum_medium_period": "中期动量周期",
    "momentum_long_period": "长期动量周期",
    "rsi_window": "RSI 周期",
    "rsi_entry": "RSI 入场",
    "rsi_exit": "RSI 退出",
    "bollinger_window": "布林带周期",
    "bollinger_std": "布林带标准差",
    "oversold_window": "短期超跌周期",
    "oversold_return": "超跌收益阈值",
    "ensemble_threshold": "组合入场阈值",
    "ensemble_volatility_window": "风险权重窗口",
}


def format_metric_value(name: str, value: float) -> str:
    if name in PERCENT_METRICS:
        return f"{value:.2%}"
    if name in {"turnover", "annualized_turnover"}:
        return f"{value:.2f} 倍"
    if name == "total_costs":
        return f"¥{value:,.2f}"
    if name == "trade_count":
        return f"{int(value)}"
    return f"{value:.3f}"


def format_annual_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.rename(columns=ANNUAL_COLUMN_LABELS).copy()
    percent_columns = [
        label
        for label in [
            "费后收益",
            "费前收益",
            "股票池买入持有",
            "沪深300收益",
            "相对沪深300收益",
            "平均仓位",
        ]
        if label in display
    ]
    for column in percent_columns:
        display[column] = display[column].map(lambda value: f"{float(value):.2%}")
    if "累计换手" in display:
        display["累计换手"] = display["累计换手"].map(
            lambda value: f"{float(value):.2f} 倍"
        )
    for column in ["成交金额", "交易成本"]:
        if column in display:
            display[column] = display[column].map(lambda value: f"¥{float(value):,.2f}")
    if "成交笔数" in display:
        display["成交笔数"] = display["成交笔数"].astype(int)
    return display


def submit_background_job(
    kind: str,
    *,
    arguments: dict[str, object] | None = None,
    job_config: AppConfig | None = None,
) -> str | None:
    try:
        job_id = job_manager.submit(
            kind,
            arguments=arguments,
            job_config=job_config,
        )
        st.success(
            f"后台任务 {job_id} 已提交；如果已有任务正在运行，它会自动排队。"
            "现在可以切换页面或关闭浏览器，任务仍会继续。"
        )
        return job_id
    except Exception as exc:
        st.error(f"后台任务提交失败：{exc}")
        return None


def latest_successful_job(kind: str) -> pd.Series | None:
    jobs = job_manager.jobs([kind], limit=20)
    successful = jobs[jobs["status"] == "SUCCESS"] if not jobs.empty else jobs
    return successful.iloc[0] if not successful.empty else None


@st.fragment(run_every="3s")
def render_background_jobs(kinds: list[str], *, key: str) -> None:
    jobs = job_manager.jobs(kinds, limit=10)
    st.caption("任务由独立后台进程执行；切换页面、刷新或关闭浏览器不会终止任务。")
    if jobs.empty:
        st.info("还没有相关后台任务。")
    else:
        display = jobs[
            ["job_id", "kind", "status", "progress", "message", "created_at", "finished_at"]
        ].copy()
        display["kind"] = display["kind"].map(JOB_LABELS).fillna(display["kind"])
        display["status"] = display["status"].map(JOB_STATUS_LABELS).fillna(display["status"])
        display["progress"] = display["progress"].map(lambda value: f"{float(value):.0%}")
        display = display.rename(
            columns={
                "job_id": "任务编号",
                "kind": "任务",
                "status": "状态",
                "progress": "进度",
                "message": "当前阶段/结果",
                "created_at": "提交时间",
                "finished_at": "完成时间",
            }
        )
        st.dataframe(display, use_container_width=True, hide_index=True)
        running_jobs = jobs[jobs["status"] == "RUNNING"]
        latest = running_jobs.iloc[0] if not running_jobs.empty else jobs.iloc[0]
        status_token = "|".join(
            f"{row.job_id}:{row.status}"
            for row in jobs[["job_id", "status"]].itertuples(index=False)
        )
        token_key = f"background-job-token-{key}"
        previous_token = st.session_state.get(token_key)
        st.session_state[token_key] = status_token
        if previous_token and previous_token != status_token:
            st.rerun(scope="app")
        if latest["status"] in ACTIVE_STATUSES:
            st.progress(float(latest["progress"]), text=str(latest["message"]))
        elif latest["status"] == "FAILED":
            st.error(f"最近任务失败：{latest['message']}")
        log = tail_job_log(latest)
        if log:
            with st.expander(f"最近任务日志 · {latest['job_id']}"):
                st.code(log, language="text")
    if st.button("刷新任务状态与结果", key=f"refresh-jobs-{key}"):
        st.rerun()


def dataframe_download(label: str, frame: pd.DataFrame, filename: str) -> None:
    csv_column, parquet_column = st.columns(2)
    csv_column.download_button(
        f"{label} CSV",
        frame.to_csv(index=False).encode("utf-8-sig"),
        f"{filename}.csv",
        mime="text/csv",
    )
    parquet = io.BytesIO()
    frame.to_parquet(parquet, index=False)
    parquet_column.download_button(
        f"{label} Parquet",
        parquet.getvalue(),
        f"{filename}.parquet",
        mime="application/octet-stream",
    )


def strategy_line_chart(
    frame: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    labels: dict[str, str] | None = None,
    y_title: str = "净值",
    value_format: str = ".4f",
) -> None:
    chart = build_zoomable_line_chart(
        frame,
        columns=columns,
        labels=labels,
        y_title=y_title,
        value_format=value_format,
    )
    st.vega_lite_chart(
        chart.data,
        chart.spec,
        use_container_width=True,
        theme=None,
    )
    st.caption("双指捏合缩放 · 双指滚动页面 · 图内拖动平移 · 双击恢复完整范围")


background_jobs = job_manager.jobs(limit=100)
active_background_jobs = (
    background_jobs[background_jobs["status"].isin(ACTIVE_STATUSES)]
    if not background_jobs.empty
    else background_jobs
)
active_background_kinds = (
    set(active_background_jobs["kind"].astype(str))
    if not active_background_jobs.empty
    else set()
)
active_background_job = job_manager.active_job()


def background_kind_is_active(kind: str) -> bool:
    return kind in active_background_kinds


with st.sidebar.expander("后台任务中心", expanded=active_background_job is not None):
    if active_background_job is None:
        st.caption("当前没有运行或排队中的耗时任务")
    else:
        active_kind = JOB_LABELS.get(
            str(active_background_job["kind"]), str(active_background_job["kind"])
        )
        active_status = JOB_STATUS_LABELS.get(
            str(active_background_job["status"]), str(active_background_job["status"])
        )
        st.write(f"{active_kind} · {active_status}")
        st.progress(
            float(active_background_job["progress"]),
            text=str(active_background_job["message"]),
        )
        st.caption(f"任务编号：{active_background_job['job_id']}")
        queued_count = int((active_background_jobs["status"] == "QUEUED").sum())
        if queued_count:
            st.caption(f"排队中：{queued_count} 个任务，将按提交顺序执行。")
    st.caption("不同任务可以继续提交并排队；切换页面或关闭浏览器不会停止任务。")


if page == "总览":
    dates = repository.latest_dates()
    latest = max((value for value in dates.values() if value), default=None)
    snapshots = store.table("portfolio_snapshots")
    positions = store.table("positions")
    cash = store.get_cash()
    ordered_snapshots = snapshots.sort_values("snapshot_date")
    equity = float(ordered_snapshots.iloc[-1]["equity"]) if not snapshots.empty else cash
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("组合权益", f"¥{equity:,.2f}")
    col2.metric("可用现金", f"¥{cash:,.2f}")
    col3.metric("持仓数量", len(positions))
    col4.metric("最新行情", str(latest or "缺失"))
    stale = repository.stale_symbols()
    if stale:
        st.error("数据告警：" + "；".join(f"{key} {value}" for key, value in stale.items()))
    else:
        st.success("股票池行情新鲜度检查通过")
    if not snapshots.empty:
        chart = ordered_snapshots.copy()
        chart["drawdown"] = chart["equity"] / chart["equity"].cummax() - 1
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("当前回撤", f"{chart['drawdown'].iloc[-1]:.2%}")
        r2.metric("历史最大回撤", f"{chart['drawdown'].min():.2%}")
        r3.metric("目标年化波动", f"{config.portfolio.target_annual_volatility:.1%}")
        r4.metric("风险配置", config.portfolio.allocation_method)
        st.line_chart(chart, x="snapshot_date", y="equity")
    screen_runs = store.table("screen_runs", 1)
    if not screen_runs.empty:
        latest_screen = screen_runs.iloc[0]
        st.caption(
            f"最近选股：{latest_screen['as_of_date']}，覆盖 {latest_screen['universe_size']} 只，"
            f"入选 {latest_screen['selected_count']} 只。"
        )
    st.subheader("最近运行")
    st.dataframe(store.table("daily_runs", 20), use_container_width=True, hide_index=True)

elif page == "数据与股票池":
    st.subheader("股票池")
    symbols_text = st.text_input(
        "六位股票代码，以逗号分隔", value=", ".join(config.market.watchlist)
    )
    col1, col2 = st.columns(2)
    if col1.button("保存股票池", use_container_width=True):
        symbols = [
            item.strip() for item in symbols_text.replace("，", ",").split(",") if item.strip()
        ]
        config.market.watchlist = symbols
        save_config(config)
        st.success("股票池已写入配置；刷新页面后生效。")
    if col2.button(
        "后台更新全部行情",
        type="primary",
        use_container_width=True,
        disabled=background_kind_is_active("data_update"),
    ):
        submit_background_job("data_update")
    render_background_jobs(["data_update"], key="data-update")
    reports_frame: pd.DataFrame | None = None
    quality_job = latest_successful_job("data_update")
    if quality_job is not None:
        reports_path = Path(str(parse_job_result(quality_job).get("reports_path", "")))
        if reports_path.exists():
            reports_frame = pd.read_parquet(reports_path)
    if reports_frame is not None and not reports_frame.empty:
        st.subheader("最近一次后台更新质量报告")
        st.dataframe(reports_frame, use_container_width=True, hide_index=True)
        dataframe_download("导出质量报告", reports_frame, "data-quality")
    st.subheader("本地数据状态")
    st.dataframe(
        pd.DataFrame(
            [
                {"symbol": symbol, "latest_adjusted_date": value}
                for symbol, value in repository.latest_dates().items()
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

elif page == "自动选股":
    st.subheader("定时多因子选股")
    universe_service = UniverseService(config, repository, store)
    coverage = universe_service.coverage()
    if config.storage.universe_file.exists():
        universe_frame = universe_service.load()
        if config.screening.exclude_st and "is_st" in universe_frame:
            universe_frame = universe_frame[
                ~universe_frame["is_st"].fillna(False).astype(bool)
            ]
        universe_frame = universe_frame.drop_duplicates("symbol").copy()
        universe_frame["symbol"] = universe_frame["symbol"].astype(str).str.zfill(6)
    else:
        universe_frame = pd.DataFrame(columns=["symbol", "name", "is_st"])
    industry_service = IndustryClassificationService(config, store)
    industry_symbols = universe_frame.get("symbol", pd.Series(dtype=str)).tolist()
    industry_coverage = industry_service.coverage(industry_symbols)
    industry_status = industry_service.status(industry_symbols)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("代码宇宙", coverage["universe"])
    c2.metric("研究行情覆盖", coverage["adjusted"])
    c3.metric(
        "覆盖率",
        f"{coverage['adjusted'] / coverage['universe']:.2%}" if coverage["universe"] else "0.00%",
    )
    c4.metric("未复权行情", coverage["raw"])
    st.caption(
        f"来源：{config.screening.universe_source}；筛选周期：{config.screening.schedule}；"
        f"目标 {config.screening.selection_count} 只；单行业最多 "
        f"{config.screening.max_per_industry} 只。"
    )
    industry_scope = "非 ST 代码宇宙" if config.screening.exclude_st else "代码宇宙（含 ST）"
    st.markdown(f"#### 行业分类覆盖（{industry_scope}）")
    i1, i2, i3, i4 = st.columns(4)
    i1.metric(
        "已分类 / 总数",
        f"{industry_coverage['classified']:,} / {industry_coverage['total']:,}",
    )
    i2.metric("未分类", f"{industry_coverage['unclassified']:,}")
    i3.metric("接口错误", f"{industry_coverage['errors']:,}")
    i4.metric("手工分类", f"{industry_coverage['manual']:,}")
    st.caption(
        "“未分类”表示本地还没有可用的行业记录，不是该公司没有所属行业。"
        "后台补全会先尝试全量批量映射，再轮换查询未分类股票；"
        "手工修正始终优先。"
    )
    batch_size = st.number_input(
        "本次轮换更新数量",
        min_value=1,
        max_value=1000,
        value=config.screening.universe_batch_size,
    )
    industry_batch_size = st.number_input(
        "本次行业逐股回填数量",
        min_value=1,
        max_value=1000,
        value=config.screening.industry_batch_size,
        help="批量行业映射不受此数量限制；只有批量接口未覆盖的股票才按该数量轮换回填。",
    )
    include_raw = st.checkbox(
        "同时下载未复权行情（仅模拟盘候选需要，全市场筛选可不勾选）",
        value=False,
    )
    refresh_fundamentals = st.checkbox(
        "更新技术预选股的估值与质量数据",
        value=False,
        help="行业分类由独立后台任务维护，不再依赖基本面更新。",
    )
    action1, action2, action3 = st.columns(3)
    if action1.button(
        "后台同步A股代码",
        use_container_width=True,
        disabled=background_kind_is_active("universe_sync"),
    ):
        submit_background_job("universe_sync")
    if action2.button(
        "后台轮换更新行情",
        use_container_width=True,
        disabled=background_kind_is_active("universe_update"),
    ):
        submit_background_job(
            "universe_update",
            arguments={"batch_size": int(batch_size), "include_raw": include_raw},
        )
    if action3.button(
        "后台执行筛选",
        type="primary",
        use_container_width=True,
        disabled=background_kind_is_active("screen"),
    ):
        submit_background_job(
            "screen",
            arguments={"refresh_fundamentals": refresh_fundamentals},
        )
    if st.button(
        "后台补全行业分类",
        use_container_width=True,
        disabled=background_kind_is_active("industry_update") or not industry_symbols,
    ):
        submit_background_job(
            "industry_update",
            arguments={"batch_size": int(industry_batch_size), "try_bulk": True},
        )
    st.caption("行业补全完成后重新执行一次筛选，已保存的筛选结果才会使用新分类。")
    render_background_jobs(
        ["universe_sync", "universe_update", "industry_update", "screen"],
        key="screening",
    )
    batch_job = latest_successful_job("universe_update")
    if batch_job is not None:
        batch_reports_path = Path(
            str(parse_job_result(batch_job).get("reports_path", ""))
        )
        if batch_reports_path.exists():
            with st.expander("最近一批轮换更新明细"):
                st.dataframe(
                    pd.read_parquet(batch_reports_path),
                    use_container_width=True,
                    hide_index=True,
                )

    st.markdown("#### 未分类明细")
    if industry_coverage["total"] == 0:
        st.info("尚无代码宇宙，请先执行“后台同步A股代码”。")
    elif industry_coverage["unclassified"] == 0:
        st.success(f"当前{industry_scope}已全部完成行业分类。")
    else:
        unclassified = industry_status[
            industry_status["industry"].eq("未分类")
        ].copy()
        if "name" in universe_frame.columns:
            unclassified = unclassified.merge(
                universe_frame[["symbol", "name"]],
                on="symbol",
                how="left",
            )
        else:
            unclassified["name"] = ""
        status_labels = {
            "missing": "尚未查询",
            "empty": "接口无可用分类",
            "error": "接口错误",
            "ok": "已分类",
        }
        unclassified["industry_status"] = unclassified["industry_status"].map(
            status_labels
        ).fillna(unclassified["industry_status"])
        unclassified["industry_source"] = unclassified["industry_source"].map(
            INDUSTRY_SOURCE_LABELS
        ).fillna(unclassified["industry_source"])
        unclassified = unclassified.rename(
            columns={
                "symbol": "股票代码",
                "name": "股票名称",
                "industry_status": "状态",
                "industry_source": "数据源",
                "industry_error": "错误/说明",
                "industry_updated_at": "最近查询时间",
            }
        )
        st.dataframe(
            unclassified[
                ["股票代码", "股票名称", "状态", "数据源", "错误/说明", "最近查询时间"]
            ],
            use_container_width=True,
            hide_index=True,
            height=420,
        )

    override_flash = st.session_state.pop("industry_override_flash", None)
    with st.expander("手工修正行业分类", expanded=bool(override_flash)):
        if override_flash:
            st.success(override_flash)
        st.caption(
            "可以使用申万一级行业，也可以建立自己的分类；行业名称完全相同的股票"
            "才会在多因子评分中视为同一组。"
        )
        if not industry_symbols:
            st.info("请先同步 A 股代码。")
        else:
            symbol_names = (
                universe_frame.set_index("symbol")["name"].astype(str).to_dict()
                if "name" in universe_frame.columns
                else {}
            )
            selected_symbol = st.selectbox(
                "股票",
                industry_symbols,
                format_func=lambda symbol: (
                    f"{symbol} · {symbol_names.get(symbol, '')}".rstrip(" ·")
                ),
                key="industry-override-symbol",
            )
            selected_status = industry_status[
                industry_status["symbol"].eq(selected_symbol)
            ]
            if not selected_status.empty:
                current = selected_status.iloc[0]
                source = current["industry_source"]
                source_text = (
                    INDUSTRY_SOURCE_LABELS.get(str(source), str(source))
                    if pd.notna(source) and str(source).strip()
                    else "无"
                )
                st.caption(
                    f"当前分类：{current['industry']}；来源：{source_text}"
                )
            overrides = store.industry_overrides()
            current_override = overrides[overrides["symbol"].eq(selected_symbol)]
            manual_industry = (
                str(current_override.iloc[0]["industry"])
                if not current_override.empty
                else ""
            )
            manual_note = (
                str(current_override.iloc[0]["note"])
                if not current_override.empty
                else ""
            )
            value_key = f"industry-override-value-{selected_symbol}"
            note_key = f"industry-override-note-{selected_symbol}"
            manual_value = st.text_input(
                "行业名称",
                value=manual_industry,
                placeholder="例如：电子",
                key=value_key,
            )
            manual_note_value = st.text_input(
                "备注（可选）",
                value=manual_note,
                placeholder="例如：根据公司主营业务手工修正",
                key=note_key,
            )
            save_override, remove_override = st.columns(2)
            if save_override.button(
                "保存手工分类",
                type="primary",
                use_container_width=True,
                disabled=not manual_value.strip(),
            ):
                store.set_industry_override(
                    selected_symbol,
                    manual_value,
                    manual_note_value,
                )
                st.session_state.pop(value_key, None)
                st.session_state.pop(note_key, None)
                st.session_state["industry_override_flash"] = (
                    f"已将 {selected_symbol} 手工分类为“{manual_value.strip()}”。"
                    "重新执行筛选后，结果会使用该分类。"
                )
                st.rerun()
            if remove_override.button(
                "删除手工覆盖",
                use_container_width=True,
                disabled=current_override.empty,
            ):
                store.remove_industry_override(selected_symbol)
                st.session_state.pop(value_key, None)
                st.session_state.pop(note_key, None)
                st.session_state["industry_override_flash"] = (
                    f"已删除 {selected_symbol} 的手工覆盖，恢复使用数据源分类。"
                    "重新执行筛选后，结果会同步更新。"
                )
                st.rerun()
    screening_frame = None
    screen_runs = store.table("screen_runs", 1)
    if not screen_runs.empty:
        path = Path(screen_runs.iloc[0]["output_dir"]) / "screening.parquet"
        if path.exists():
            screening_frame = pd.read_parquet(path)
    if screening_frame is not None and not screening_frame.empty:
        selected = screening_frame[screening_frame["selected"]].copy()
        display = [
            "rank",
            "symbol",
            "name",
            "industry",
            "total_score",
            "factor_coverage",
            "trend_signal",
            "mean_reversion_signal",
            "return_20",
            "return_60",
            "return_120",
            "annual_volatility",
            "average_amount_20",
        ]
        st.subheader("入选结果")
        st.dataframe(selected[display], use_container_width=True, hide_index=True)

        watchlist_flash = st.session_state.pop("watchlist_add_flash", None)
        if watchlist_flash:
            st.success(watchlist_flash)
        current_watchlist = set(config.market.watchlist)
        candidate_symbols = list(
            dict.fromkeys(
                symbol
                for symbol in selected["symbol"].astype(str).str.zfill(6).tolist()
                if symbol not in current_watchlist
            )
        )
        st.markdown("#### 加入股票池")
        st.caption(
            f"当前股票池 {len(current_watchlist)} 只；"
            f"本次入选中有 {len(candidate_symbols)} 只尚未加入。"
        )
        if candidate_symbols:
            candidate_rows = selected.copy()
            candidate_rows["symbol"] = candidate_rows["symbol"].astype(str).str.zfill(6)
            candidate_rows = candidate_rows.drop_duplicates("symbol").set_index("symbol")

            def format_candidate(symbol: str) -> str:
                row = candidate_rows.loc[symbol]
                raw_name = row.get("name", "")
                raw_industry = row.get("industry", "")
                name = str(raw_name).strip() if pd.notna(raw_name) else ""
                industry = str(raw_industry).strip() if pd.notna(raw_industry) else ""
                details = " · ".join(value for value in (name, industry) if value)
                return f"{symbol} · {details}" if details else symbol

            chosen_symbols = st.multiselect(
                "选择要加入的股票",
                candidate_symbols,
                format_func=format_candidate,
                placeholder="可以选择一只或多只本次入选股票",
            )
            if st.button(
                "加入所选股票",
                type="primary",
                disabled=not chosen_symbols,
                use_container_width=True,
            ):
                latest_config = load_config(config_path)
                added_symbols = extend_watchlist(latest_config, chosen_symbols)
                if added_symbols:
                    save_config(latest_config)
                    st.session_state["watchlist_add_flash"] = (
                        f"已将 {len(added_symbols)} 只股票加入股票池："
                        f"{', '.join(added_symbols)}。"
                        "请在“数据与股票池”页面更新行情后再运行模拟盘。"
                    )
                else:
                    st.session_state["watchlist_add_flash"] = (
                        "所选股票已经在股票池中，没有重复添加。"
                    )
                st.rerun()
        else:
            st.info("本次入选股票已经全部在股票池中。")
        dataframe_download("导出筛选结果", screening_frame, "stock-screening")
        with st.expander("查看全部评分"):
            st.dataframe(screening_frame, use_container_width=True, hide_index=True)

elif page == "回测研究":
    st.subheader("多策略组合回测")
    current_label = next(
        label for label, name in STRATEGY_LABELS.items() if name == config.strategy.name
    )
    selected_label = st.selectbox(
        "策略", list(STRATEGY_LABELS), index=list(STRATEGY_LABELS).index(current_label)
    )
    selected_strategy = STRATEGY_LABELS[selected_label]
    strategy_values = config.strategy.model_dump()
    strategy_values["name"] = selected_strategy

    def render_ma_cross_parameters() -> None:
        c1, c2 = st.columns(2)
        strategy_values["short_window"] = int(
            c1.number_input("短均线", min_value=2, value=config.strategy.short_window)
        )
        strategy_values["long_window"] = int(
            c2.number_input("长均线", min_value=3, value=config.strategy.long_window)
        )
        c3, c4 = st.columns(2)
        strategy_values["ma_signal_buffer"] = float(
            c3.number_input(
                "均线缓冲带",
                min_value=0.0,
                max_value=0.199,
                value=config.strategy.ma_signal_buffer,
                format="%.4f",
                help="短均线高于长均线超过该比例才入场，低于相同比例才退出。",
            )
        )
        strategy_values["ma_confirmation_days"] = int(
            c4.number_input(
                "均线确认交易日",
                min_value=1,
                max_value=20,
                value=config.strategy.ma_confirmation_days,
                help="入场或退出条件必须连续满足多少个交易日。",
            )
        )

    def render_trend_parameters() -> None:
        c1, c2, c3 = st.columns(3)
        strategy_values["short_window"] = int(
            c1.number_input("短均线", min_value=2, value=config.strategy.short_window)
        )
        strategy_values["long_window"] = int(
            c2.number_input("长均线", min_value=3, value=config.strategy.long_window)
        )
        strategy_values["breakout_window"] = int(
            c3.number_input(
                "突破回看期", min_value=5, value=config.strategy.breakout_window
            )
        )
        c4, c5 = st.columns(2)
        strategy_values["atr_window"] = int(
            c4.number_input("ATR 周期", min_value=2, value=config.strategy.atr_window)
        )
        strategy_values["atr_stop_multiplier"] = float(
            c5.number_input(
                "ATR 止损倍数",
                min_value=0.1,
                value=config.strategy.atr_stop_multiplier,
            )
        )

    def render_momentum_parameters() -> None:
        c1, c2, c3 = st.columns(3)
        strategy_values["momentum_skip_recent"] = int(
            c1.number_input(
                "跳过最近交易日",
                min_value=0,
                max_value=config.strategy.momentum_lookback - 1,
                value=config.strategy.momentum_skip_recent,
            )
        )
        strategy_values["momentum_top_n"] = int(
            c2.number_input(
                "动量入选数量", min_value=1, value=config.strategy.momentum_top_n
            )
        )
        strategy_values["momentum_rank_buffer"] = int(
            c3.number_input(
                "动量排名缓冲", min_value=0, value=config.strategy.momentum_rank_buffer
            )
        )
        c5, c6, c7 = st.columns(3)
        strategy_values["momentum_short_period"] = int(
            c5.number_input(
                "短期动量周期", min_value=5, value=config.strategy.momentum_short_period
            )
        )
        strategy_values["momentum_medium_period"] = int(
            c6.number_input(
                "中期动量周期",
                min_value=10,
                value=config.strategy.momentum_medium_period,
            )
        )
        strategy_values["momentum_long_period"] = int(
            c7.number_input(
                "长期动量周期", min_value=20, value=config.strategy.momentum_long_period
            )
        )

    def render_rsi_parameters() -> None:
        c1, c2, c3 = st.columns(3)
        strategy_values["rsi_window"] = int(
            c1.number_input("RSI 周期", min_value=2, value=config.strategy.rsi_window)
        )
        strategy_values["rsi_entry"] = float(
            c2.number_input(
                "RSI 入场",
                min_value=0.0,
                max_value=100.0,
                value=config.strategy.rsi_entry,
            )
        )
        strategy_values["rsi_exit"] = float(
            c3.number_input(
                "RSI 退出",
                min_value=0.0,
                max_value=100.0,
                value=config.strategy.rsi_exit,
            )
        )

    def render_mean_reversion_parameters() -> None:
        render_rsi_parameters()
        c1, c2, c3 = st.columns(3)
        strategy_values["bollinger_window"] = int(
            c1.number_input(
                "布林带周期", min_value=5, value=config.strategy.bollinger_window
            )
        )
        strategy_values["bollinger_std"] = float(
            c2.number_input(
                "布林带标准差", min_value=0.1, value=config.strategy.bollinger_std
            )
        )
        strategy_values["oversold_window"] = int(
            c3.number_input(
                "短期超跌周期", min_value=2, value=config.strategy.oversold_window
            )
        )
        strategy_values["oversold_return"] = float(
            st.number_input(
                "超跌收益阈值",
                min_value=-1.0,
                max_value=-0.001,
                value=config.strategy.oversold_return,
                format="%.3f",
            )
        )

    with st.expander("策略参数", expanded=True):
        if selected_strategy == "ma_cross":
            render_ma_cross_parameters()
        elif selected_strategy == "trend_breakout":
            render_trend_parameters()
        elif selected_strategy == "momentum":
            render_momentum_parameters()
        elif selected_strategy == "rsi_reversion":
            render_rsi_parameters()
        elif selected_strategy == "mean_reversion":
            render_mean_reversion_parameters()
        else:
            st.caption("趋势突破组件")
            render_trend_parameters()
            st.caption("横截面动量组件")
            render_momentum_parameters()
            st.caption("均值回归组件")
            render_mean_reversion_parameters()
            c1, c2 = st.columns(2)
            strategy_values["ensemble_threshold"] = float(
                c1.number_input(
                    "组合入场阈值",
                    min_value=0.01,
                    max_value=1.0,
                    value=config.strategy.ensemble_threshold,
                )
            )
            if selected_strategy == "ensemble_risk":
                strategy_values["ensemble_volatility_window"] = int(
                    c2.number_input(
                        "风险权重窗口",
                        min_value=5,
                        value=config.strategy.ensemble_volatility_window,
                    )
                )

    with st.expander("交易执行与组合风控", expanded=False):
        c1, c2 = st.columns(2)
        commission = c1.number_input(
            "佣金率",
            min_value=0.0,
            value=config.costs.commission_rate,
            format="%.5f",
        )
        slippage = c2.number_input(
            "单边滑点(bps)", min_value=0.0, value=config.costs.slippage_bps
        )
        c3, c4 = st.columns(2)
        target_volatility = c3.number_input(
            "组合目标年化波动",
            min_value=0.01,
            max_value=1.0,
            value=config.portfolio.target_annual_volatility,
        )
        maximum_drawdown = c4.number_input(
            "组合回撤熔断阈值",
            min_value=0.01,
            max_value=0.99,
            value=config.portfolio.max_portfolio_drawdown,
        )
        c5, c6, c7 = st.columns(3)
        min_holding_days = c5.number_input(
            "最短持有交易日",
            min_value=1,
            value=config.portfolio.min_holding_days,
        )
        minimum_rebalance = c6.number_input(
            "最小调仓差异",
            min_value=0.0,
            max_value=0.99,
            value=config.portfolio.minimum_rebalance_weight_change,
        )
        max_daily_turnover = c7.number_input(
            "普通调仓单日换手上限",
            min_value=0.01,
            max_value=2.0,
            value=config.portfolio.max_daily_turnover,
        )

    try:
        run_strategy = StrategyConfig.model_validate(strategy_values)
        parameter_error = None
    except ValueError as exc:
        run_strategy = config.strategy
        parameter_error = str(exc)
        st.error(parameter_error)
    run_config = config.model_copy(deep=True)
    run_config.strategy = run_strategy
    run_config.costs.commission_rate = float(commission)
    run_config.costs.slippage_bps = float(slippage)
    run_config.portfolio.target_annual_volatility = float(target_volatility)
    run_config.portfolio.max_portfolio_drawdown = float(maximum_drawdown)
    run_config.portfolio.min_holding_days = int(min_holding_days)
    run_config.portfolio.minimum_rebalance_weight_change = float(minimum_rebalance)
    run_config.portfolio.max_daily_turnover = float(max_daily_turnover)

    run_column, compare_column, robustness_column = st.columns(3)
    st.caption(
        "“对比全部策略”会采用当前策略页面中可见的修改；"
        "其他策略继续使用 config/settings.yaml 中的参数。"
    )
    research_disabled = bool(parameter_error)
    if run_column.button(
        "后台运行所选策略",
        type="primary",
        use_container_width=True,
        disabled=research_disabled or background_kind_is_active("backtest"),
    ):
        submit_background_job("backtest", job_config=run_config)
    if compare_column.button(
        "后台对比全部策略",
        use_container_width=True,
        disabled=research_disabled or background_kind_is_active("compare"),
    ):
        submit_background_job("compare", job_config=run_config)
    if robustness_column.button(
        "后台均线稳健性分析",
        use_container_width=True,
        disabled=research_disabled or background_kind_is_active("robustness"),
    ):
        submit_background_job("robustness", job_config=run_config)

    render_background_jobs(
        ["backtest", "compare", "robustness"],
        key="research",
    )

    robustness_summary = None
    robustness_details = None
    robustness_job = latest_successful_job("robustness")
    if robustness_job is not None:
        robustness_dir = Path(
            str(parse_job_result(robustness_job).get("output_dir", ""))
        )
        if (robustness_dir / "summary.csv").exists():
            robustness_summary = pd.read_csv(robustness_dir / "summary.csv")
            robustness_details = pd.read_csv(robustness_dir / "folds.csv")
    if robustness_summary is not None:
        st.subheader("均线参数稳健性")
        st.caption(
            "排名优先考虑跨时间段一致性、年化超额收益下四分位与最差表现，"
            "不会自动改写正式配置。"
            "当前结果仍基于今天的股票池，可能存在选股与幸存者偏差，不等同于真正样本外验证。"
        )
        st.dataframe(robustness_summary, use_container_width=True, hide_index=True)
        dataframe_download("导出稳健性摘要", robustness_summary, "ma-robustness-summary")
        if robustness_details is not None:
            with st.expander("查看各时间段明细"):
                st.dataframe(robustness_details, use_container_width=True, hide_index=True)

    comparison_metrics = None
    comparison_curves = None
    comparison_job = latest_successful_job("compare")
    if comparison_job is not None:
        comparison_dir = Path(
            str(parse_job_result(comparison_job).get("output_dir", ""))
        )
        if (comparison_dir / "metrics.csv").exists():
            comparison_metrics = pd.read_csv(comparison_dir / "metrics.csv")
            comparison_curves = pd.read_parquet(
                comparison_dir / "equity_curves.parquet"
            )
    if comparison_metrics is not None and comparison_curves is not None:
        st.subheader("策略对比")
        display_columns = [
            "strategy",
            "total_return",
            "annual_return",
            "max_drawdown",
            "sharpe",
            "turnover",
            "total_costs",
        ]
        st.dataframe(comparison_metrics[display_columns], use_container_width=True, hide_index=True)
        strategy_line_chart(
            comparison_curves.set_index("date"),
            labels=STRATEGY_NAME_LABELS,
        )
        dataframe_download("导出策略对比", comparison_metrics, "strategy-comparison")

    result = None
    selected_backtest_row: pd.Series | None = None
    saved_backtests = store.table("backtest_runs", 100)
    backtest_job = latest_successful_job("backtest")
    preferred_run_id = (
        str(parse_job_result(backtest_job).get("run_id", ""))
        if backtest_job is not None
        else ""
    )
    if saved_backtests.empty:
        st.info("尚无已保存回测。完成一次回测后，可在这里复核并晋级策略。")
    else:
        saved_records = {
            str(row["run_id"]): row for _, row in saved_backtests.iterrows()
        }

        def format_saved_backtest(run_id: str) -> str:
            row = saved_records[run_id]
            try:
                payload = json.loads(str(row["parameters_json"]))
                strategy_name = str(payload.get("strategy", {}).get("name", "未知"))
            except (TypeError, ValueError, json.JSONDecodeError):
                strategy_name = "未知"
            strategy_label = STRATEGY_NAME_LABELS.get(strategy_name, strategy_name)
            return (
                f"{row['created_at']} · {strategy_label} · "
                f"数据截至 {row['data_end_date']} · {run_id}"
            )

        saved_run_ids = list(saved_records)
        selected_run_id = st.selectbox(
            "已保存回测",
            saved_run_ids,
            index=(
                saved_run_ids.index(preferred_run_id)
                if preferred_run_id in saved_run_ids
                else 0
            ),
            format_func=format_saved_backtest,
        )
        selected_backtest_row = saved_records[selected_run_id]
        selected_backtest_dir = Path(str(selected_backtest_row["output_dir"]))
        if (selected_backtest_dir / "manifest.json").exists():
            try:
                result = load_saved_backtest(selected_backtest_dir)
            except (OSError, TypeError, ValueError) as exc:
                st.error(f"已保存回测无法读取：{exc}")
        else:
            st.error("已保存回测缺少 manifest.json，不能展示或晋级。")
    if result is not None:
        headline = st.columns(5)
        headline[0].metric("费后累计收益", f"{result.metrics['total_return']:.2%}")
        headline[1].metric(
            "累计超额收益",
            f"{result.metrics.get('excess_total_return', 0.0):.2%}",
        )
        headline[2].metric(
            "信息比率",
            f"{result.metrics.get('information_ratio', 0.0):.3f}",
        )
        headline[3].metric("最大回撤", f"{result.metrics['max_drawdown']:.2%}")
        headline[4].metric("成本拖累", f"{result.metrics.get('cost_drag', 0.0):.2%}")
        metrics = pd.DataFrame(
            [
                {
                    "指标": METRIC_LABELS.get(key, key),
                    "数值": format_metric_value(key, value),
                }
                for key, value in result.metrics.items()
            ]
        )
        st.dataframe(metrics, use_container_width=True, hide_index=True)

        st.subheader("模拟盘策略晋级")
        promotion_flash = st.session_state.pop("strategy_promotion_flash", None)
        if promotion_flash:
            st.success(promotion_flash)
        st.caption(
            "回测结果不会自动改变模拟盘。晋级只复制策略参数；"
            "当前现金、持仓、组合风控和交易成本保持不变。"
        )
        try:
            candidate_strategy = StrategyConfig.model_validate(
                result.parameters["strategy"]
            )
            current_deployment = paper_strategy_service.current()
            current_paper_strategy = paper_strategy_service.parameters(
                current_deployment
            )
            candidate_error = None
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            candidate_strategy = None
            current_deployment = paper_strategy_deployment
            current_paper_strategy = paper_strategy_service.parameters(
                current_deployment
            )
            candidate_error = str(exc)
            st.error(f"候选策略参数无法用于晋级：{candidate_error}")

        if candidate_strategy is not None:
            current_source = str(
                current_deployment.get("source_backtest_run_id") or "原配置"
            )
            summary = st.columns(4)
            summary[0].metric(
                "当前模拟盘策略",
                STRATEGY_NAME_LABELS.get(
                    current_paper_strategy.name,
                    current_paper_strategy.name,
                ),
            )
            summary[1].metric(
                "待晋级策略",
                STRATEGY_NAME_LABELS.get(
                    candidate_strategy.name,
                    candidate_strategy.name,
                ),
            )
            summary[2].metric("当前来源", current_source)
            summary[3].metric("候选回测", result.run_id)

            with st.expander("待晋级参数对比", expanded=False):
                current_values = current_paper_strategy.model_dump()
                candidate_values = candidate_strategy.model_dump()
                fields = STRATEGY_PARAMETER_FIELDS[candidate_strategy.name]
                comparison_rows = [
                    {
                        "参数": "策略",
                        "当前模拟盘": STRATEGY_NAME_LABELS.get(
                            current_paper_strategy.name,
                            current_paper_strategy.name,
                        ),
                        "待晋级": STRATEGY_NAME_LABELS.get(
                            candidate_strategy.name,
                            candidate_strategy.name,
                        ),
                    }
                ]
                comparison_rows.extend(
                    {
                        "参数": STRATEGY_PARAMETER_LABELS.get(field, field),
                        "当前模拟盘": current_values.get(field),
                        "待晋级": candidate_values.get(field),
                    }
                    for field in fields
                )
                st.dataframe(
                    pd.DataFrame(comparison_rows),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(
                    "候选收益指标来自该回测当时保存的股票池、组合风控和成本参数；"
                    "本次不会复制这些设置。"
                )

            pending_deployment = paper_strategy_service.pending()
            pending_orders_count = len(store.pending_orders())
            active_task_count = len(active_background_jobs)
            already_current = (
                str(current_deployment.get("source_backtest_run_id") or "")
                == result.run_id
            )
            already_scheduled = (
                pending_deployment is not None
                and str(pending_deployment.get("source_backtest_run_id") or "")
                == result.run_id
            )
            if active_task_count:
                st.warning(
                    f"还有 {active_task_count} 个后台任务正在运行或排队；"
                    "全部完成后才能提交策略晋级。"
                )
            if pending_deployment is not None:
                pending_name = STRATEGY_NAME_LABELS.get(
                    str(pending_deployment["strategy_name"]),
                    str(pending_deployment["strategy_name"]),
                )
                st.warning(
                    f"已有待生效部署：{pending_name} · "
                    f"{pending_deployment['deployment_id']}。"
                )
            elif pending_orders_count:
                st.info(
                    f"当前有 {pending_orders_count} 笔待成交订单。提交晋级后将先停止"
                    "生成旧策略新订单；旧订单全部成交、跳过或拒绝后，新策略自动生效。"
                )
            if already_current:
                st.info("该回测已经是当前模拟盘策略的正式来源。")
            elif already_scheduled:
                st.info("该回测已经提交晋级，正在等待旧策略订单处理完成。")

            promotion_note = st.text_input(
                "晋级备注（可选）",
                placeholder="例如：滚动验证通过，批准进入模拟盘",
                key=f"promotion-note-{result.run_id}",
            )
            promotion_confirmed = st.checkbox(
                "我已复核回测结果，并确认切换后保留当前资金与持仓",
                key=f"promotion-confirm-{result.run_id}",
            )
            promotion_disabled = (
                not promotion_confirmed
                or bool(candidate_error)
                or bool(active_task_count)
                or pending_deployment is not None
                or already_current
            )
            if st.button(
                "晋级为模拟盘策略",
                type="primary",
                use_container_width=True,
                disabled=promotion_disabled,
            ):
                try:
                    promoted = paper_strategy_service.promote_backtest(
                        result.run_id,
                        promotion_note,
                    )
                    if promoted["status"] == "PENDING":
                        message = (
                            f"已提交 {candidate_strategy.name} 晋级；"
                            "旧策略待成交订单处理完成后自动生效。"
                        )
                    else:
                        message = (
                            f"已将 {candidate_strategy.name} 晋级为模拟盘策略，"
                            "从下一次每日流程开始使用。"
                        )
                    st.session_state["strategy_promotion_flash"] = message
                    st.rerun()
                except RuntimeError as exc:
                    st.error(f"策略晋级失败：{exc}")

        curves = result.equity_curve.set_index("date")
        columns = [
            name
            for name in [
                "net_curve",
                "gross_curve",
                "buy_hold_curve",
                "benchmark_curve",
                "relative_curve",
            ]
            if name in curves
        ]
        strategy_line_chart(
            curves,
            columns=columns,
            labels={
                "net_curve": "费后净值",
                "gross_curve": "固定成交·加回直接成本",
                "buy_hold_curve": "买入持有",
                "benchmark_curve": "沪深300",
                "relative_curve": "相对沪深300净值",
            },
        )
        annual = calendar_year_analysis(result.equity_curve, result.trades)
        rolling = rolling_relative_analysis(result.equity_curve)
        st.subheader("分年度与滚动验证")
        st.dataframe(
            format_annual_analysis(annual),
            use_container_width=True,
            hide_index=True,
        )
        if not rolling.empty:
            rolling_left, rolling_right = st.columns(2)
            with rolling_left:
                strategy_line_chart(
                    rolling.set_index("date"),
                    columns=["excess_return"],
                    labels={"excess_return": "滚动一年超额收益"},
                    y_title="滚动一年超额收益",
                    value_format=".2%",
                )
            with rolling_right:
                strategy_line_chart(
                    rolling.set_index("date"),
                    columns=["information_ratio"],
                    labels={"information_ratio": "滚动一年信息比率"},
                    y_title="滚动一年信息比率",
                    value_format=".3f",
                )
        st.subheader("成交明细")
        st.dataframe(result.trades, use_container_width=True, hide_index=True)
        dataframe_download("导出成交", result.trades, f"{result.run_id}-trades")

elif page == "模拟盘":
    st.subheader("每日模拟盘")
    current_deployment = paper_strategy_service.current()
    current_strategy = paper_strategy_service.parameters(current_deployment)
    deployment_summary = st.columns(4)
    deployment_summary[0].metric(
        "当前模拟盘策略",
        STRATEGY_NAME_LABELS.get(current_strategy.name, current_strategy.name),
    )
    deployment_summary[1].metric(
        "来源回测",
        str(current_deployment.get("source_backtest_run_id") or "原配置"),
    )
    deployment_summary[2].metric(
        "回测数据截至",
        str(current_deployment.get("source_data_end_date") or "未记录"),
    )
    deployment_summary[3].metric(
        "启用时间",
        str(current_deployment.get("activated_at") or "等待生效"),
    )
    st.caption(
        "模拟盘读取独立部署策略，不会自动跟随回测排名或 settings.yaml 的研究默认值。"
        "已有持仓不会被清空，新策略通过后续目标仓位正常接管。"
        "订单和信号中的“策略@部署编号”用于区分同名策略的不同参数版本。"
    )
    scheduled_deployment = paper_strategy_service.pending()
    if scheduled_deployment is not None:
        scheduled_label = STRATEGY_NAME_LABELS.get(
            str(scheduled_deployment["strategy_name"]),
            str(scheduled_deployment["strategy_name"]),
        )
        st.info(
            f"待生效策略：{scheduled_label} · "
            f"来源回测 {scheduled_deployment['source_backtest_run_id']}。"
            f"旧策略还有 {len(store.pending_orders())} 笔待成交订单。"
        )
    with st.expander("当前模拟盘策略参数"):
        current_values = current_strategy.model_dump()
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "参数": STRATEGY_PARAMETER_LABELS.get(field, field),
                        "数值": current_values[field],
                    }
                    for field in STRATEGY_PARAMETER_FIELDS[current_strategy.name]
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    with st.expander("策略部署历史"):
        deployment_history = paper_strategy_service.history()
        history_rows = []
        for _, deployment in deployment_history.iterrows():
            try:
                source_metrics = json.loads(str(deployment["source_metrics_json"]))
            except (TypeError, ValueError, json.JSONDecodeError):
                source_metrics = {}
            history_rows.append(
                {
                    "状态": {
                        "ACTIVE": "当前",
                        "PENDING": "待生效",
                        "RETIRED": "已退役",
                    }.get(str(deployment["status"]), str(deployment["status"])),
                    "策略": STRATEGY_NAME_LABELS.get(
                        str(deployment["strategy_name"]),
                        str(deployment["strategy_name"]),
                    ),
                    "部署编号": deployment["deployment_id"],
                    "来源回测": deployment["source_backtest_run_id"] or "原配置",
                    "数据截至": deployment["source_data_end_date"] or "",
                    "申请时间": deployment["requested_at"],
                    "启用时间": deployment["activated_at"] or "",
                    "停用时间": deployment["retired_at"] or "",
                    "年化收益": (
                        f"{float(source_metrics['annual_return']):.2%}"
                        if "annual_return" in source_metrics
                        else ""
                    ),
                    "Sharpe": (
                        f"{float(source_metrics['sharpe']):.3f}"
                        if "sharpe" in source_metrics
                        else ""
                    ),
                    "最大回撤": (
                        f"{float(source_metrics['max_drawdown']):.2%}"
                        if "max_drawdown" in source_metrics
                        else ""
                    ),
                    "备注": deployment["note"],
                }
            )
        st.dataframe(
            pd.DataFrame(history_rows),
            use_container_width=True,
            hide_index=True,
        )
    st.warning("运行会更新行情并写入模拟信号、订单、成交和持仓，不会真实下单。")
    if st.button(
        "后台运行每日流程",
        type="primary",
        disabled=background_kind_is_active("daily"),
    ):
        submit_background_job("daily", arguments={"update_data": True})
    render_background_jobs(["daily"], key="daily")
    tabs = st.tabs(["持仓", "待办与订单", "成交", "信号", "资金曲线"])
    with tabs[0]:
        positions = store.table("positions")
        st.dataframe(positions, use_container_width=True, hide_index=True)
        dataframe_download("导出持仓", positions, "paper-positions")
    with tabs[1]:
        st.dataframe(store.table("orders"), use_container_width=True, hide_index=True)
    with tabs[2]:
        st.dataframe(store.table("fills"), use_container_width=True, hide_index=True)
    with tabs[3]:
        st.dataframe(store.table("signals"), use_container_width=True, hide_index=True)
    with tabs[4]:
        snapshots = store.table("portfolio_snapshots")
        if snapshots.empty:
            st.info("尚无模拟盘快照。")
        else:
            st.line_chart(snapshots.sort_values("snapshot_date"), x="snapshot_date", y="equity")
