from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

PINCH_ZOOM_EVENT = "wheel![event.ctrlKey]"


@dataclass(frozen=True)
class ZoomableLineChart:
    data: pd.DataFrame
    spec: dict[str, object]


def build_zoomable_line_chart(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    labels: dict[str, str] | None = None,
    y_title: str = "净值",
    value_format: str = ".4f",
) -> ZoomableLineChart:
    """创建只响应触控板捏合、不拦截普通双指滚动的时间序列图。"""
    if frame.empty:
        raise ValueError("折线图数据不能为空")
    selected = list(columns) if columns is not None else list(frame.columns)
    missing = set(selected) - set(frame.columns)
    if missing:
        raise ValueError(f"折线图缺少字段：{sorted(missing)}")

    labels = labels or {}
    renamed = frame[selected].rename(
        columns={column: labels.get(column, column) for column in selected}
    )
    index_name = frame.index.name or "date"
    data = renamed.rename_axis(index_name).reset_index().melt(
        id_vars=[index_name],
        var_name="series",
        value_name="value",
    )
    if not pd.api.types.is_datetime64_any_dtype(data[index_name]):
        converted = pd.to_datetime(data[index_name], errors="coerce")
        if converted.notna().all():
            data[index_name] = converted

    spec: dict[str, object] = {
        "mark": {"type": "line", "strokeWidth": 2, "clip": True},
        "encoding": {
            "x": {"field": index_name, "type": "temporal", "title": None},
            "y": {
                "field": "value",
                "type": "quantitative",
                "title": y_title,
                "scale": {"zero": False},
            },
            "color": {
                "field": "series",
                "type": "nominal",
                "title": None,
                "legend": {"orient": "top", "direction": "horizontal"},
            },
            "tooltip": [
                {
                    "field": index_name,
                    "type": "temporal",
                    "title": "日期",
                    "format": "%Y-%m-%d",
                },
                {"field": "series", "type": "nominal", "title": "曲线"},
                {
                    "field": "value",
                    "type": "quantitative",
                    "title": y_title,
                    "format": value_format,
                },
            ],
        },
        "params": [
            {
                "name": "chart_viewport",
                "select": {
                    "type": "interval",
                    "encodings": ["x", "y"],
                    "zoom": PINCH_ZOOM_EVENT,
                    "clear": "dblclick",
                },
                "bind": "scales",
            }
        ],
        "height": 440,
    }
    return ZoomableLineChart(data, spec)
