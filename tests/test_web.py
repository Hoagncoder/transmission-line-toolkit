from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

import ui_helpers as ui  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from tltoolkit import recommend_matching  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "web" / "app.py")


def run_app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=60).run()


# ----------------------------------------------------------------- helpers
@pytest.mark.parametrize(
    "vswr, level",
    [(1.0, "success"), (1.4, "success"), (1.9, "info"), (2.6, "warning"), (9.0, "error")],
)
def test_vswr_quality_levels(vswr, level):
    assert ui.vswr_quality(vswr)[1] == level


def test_vswr_quality_infinite_is_error():
    assert ui.vswr_quality(float("inf"))[1] == "error"


def test_fmt_number_handles_infinity():
    assert ui.fmt_number(float("inf"), 2, "dB") == "∞ dB"
    assert ui.fmt_number(2.0, 2) == "2.00"


def test_clip_for_plot_breaks_singularities():
    import numpy as np

    out = ui.clip_for_plot(np.array([1.0, 1e9, -1e9, np.inf, 5.0]), 100.0)
    assert out[0] == 1.0 and out[4] == 5.0
    assert np.isnan(out[1:4]).all()


def test_recommendation_rows_for_every_method():
    report = recommend_matching(50, 100 + 50j, 1e9)
    assert len(report.recommendations) == 3
    for rec in report.recommendations:
        rows = ui.recommendation_rows(rec)
        assert len(rows) == len(rec.solutions) >= 1
        assert all(r["Verified"] == "✓" for r in rows)


def test_presets_are_valid_loads():
    for r, x in ui.PRESETS.values():
        assert r >= 0 and isinstance(x, float)


# --------------------------------------------------------------------- app
def test_app_runs_with_defaults():
    at = run_app()
    assert not at.exception
    assert len(at.tabs) == 3
    assert [m.label for m in at.metric] == ["VSWR", "|Γ|", "Return loss", "Mismatch loss"]
    assert at.metric[0].value == "2.618"


@pytest.mark.parametrize("preset", list(ui.PRESETS))
def test_every_preset_runs_without_error(preset):
    at = run_app()
    at.selectbox(key="preset").select(preset).run()
    assert not at.exception
    assert at.number_input(key="rl").value == ui.PRESETS[preset][0]
    assert at.number_input(key="xl").value == ui.PRESETS[preset][1]


def test_short_circuit_explains_why_no_match():
    at = run_app()
    at.selectbox(key="preset").select("Short circuit").run()
    assert not at.exception
    assert any("absorb no power" in i.value for i in at.info)
    assert not at.dataframe


def test_manual_edit_switches_preset_to_custom():
    at = run_app()
    at.number_input(key="xl").set_value(-30.0).run()
    assert not at.exception
    assert at.selectbox(key="preset").value == ui.CUSTOM


def test_unit_change_and_matched_load():
    at = run_app()
    at.selectbox(key="funit").select("MHz").run()
    at.selectbox(key="preset").select("Already matched (50)").run()
    assert not at.exception
    assert at.metric[0].value == "1.000"
    assert any("already matched" in i.value for i in at.info)
