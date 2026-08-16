from __future__ import annotations

import pandas as pd

from aquant.charts import PINCH_ZOOM_EVENT, build_zoomable_line_chart


def test_strategy_chart_supports_trackpad_zoom_and_pan() -> None:
    frame = pd.DataFrame(
        {"net_curve": [1.0, 1.02, 1.01]},
        index=pd.date_range("2025-01-01", periods=3),
    )

    figure = build_zoomable_line_chart(
        frame,
        columns=["net_curve"],
        labels={"net_curve": "费后净值"},
        y_title="累计净值",
        value_format=".3f",
    )

    specification = figure.spec
    parameter = specification["params"][0]
    selection = parameter["select"]
    assert selection["zoom"] == PINCH_ZOOM_EVENT
    assert selection["clear"] == "dblclick"
    assert parameter["bind"] == "scales"
    assert selection["encodings"] == ["x", "y"]
    assert set(figure.data["series"]) == {"费后净值"}
    assert specification["encoding"]["y"]["title"] == "累计净值"
    assert specification["encoding"]["tooltip"][-1]["format"] == ".3f"
