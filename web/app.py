"""Streamlit web app for transmission-line analysis and impedance matching.

Run from the repository root:  streamlit run web/app.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:  # works after `pip install -e .`, otherwise fall back to the source tree
    import tltoolkit  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(HERE.parent / "src"))

import streamlit as st  # noqa: E402
import ui_helpers as ui  # noqa: E402

from tltoolkit import (  # noqa: E402
    phase_velocity,
    recommend_matching,
)

st.set_page_config(
    page_title="Transmission Line Toolkit",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state and callbacks
# ---------------------------------------------------------------------------
DEFAULTS = {
    "line_impedance": None,
    "load_resistance": None,
    "load_reactance": None,
    "frequency_hz": None,
    "relative_permittivity": None,
    "matching_method": "Lumped Element (L, C)",
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)

MATCHING_METHODS = {
    "Lumped Element (L, C)": ("L-network (lumped elements)", None),
    "Shorted Stub": ("Single shunt stub (short-circuited)", "short"),
    "Opened Stub": ("Single shunt stub (open-circuited)", "open"),
    "Quarter-Wave Transformer": ("Quarter-wave transformer", None),
}
MATCHING_DIAGRAMS = {
    "Lumped Element (L, C)": (
        HERE / "assets" / "matching" / "lumped-element.png",
        "Lumped-element matching using a capacitor or an inductor.",
    ),
    "Shorted Stub": (
        HERE / "assets" / "matching" / "shorted-stub.png",
        "Single shunt matching with a short-circuited stub.",
    ),
    "Opened Stub": (
        HERE / "assets" / "matching" / "opened-stub.png",
        "Single shunt matching with an open-circuited stub.",
    ),
    "Quarter-Wave Transformer": (
        HERE / "assets" / "matching" / "quarter-wave-transformer.png",
        "Quarter-wave transformer matching.",
    ),
}


# ---------------------------------------------------------------------------
# System parameters
# ---------------------------------------------------------------------------
st.title("Transmission Line Toolkit")
with st.sidebar:
    st.header("System Parameters")
    rl = st.number_input(
        "Real part of $Z_L$ (Ω)",
        min_value=0.0,
        max_value=1e6,
        step=1.0,
        value=None,
        key="load_resistance",
        help="Resistive part of the load. Must be ≥ 0 (passive load).",
    )
    xl = st.number_input(
        "Imaginary part of $Z_L$ (Ω)",
        min_value=-1e6,
        max_value=1e6,
        step=1.0,
        value=None,
        key="load_reactance",
        help="Reactive part: positive = inductive, negative = capacitive.",
    )
    z0 = st.number_input(
        "Line impedance Z0 (Ω)",
        min_value=1.0,
        max_value=1000.0,
        step=1.0,
        value=None,
        key="line_impedance",
    )
    freq = st.number_input(
        "Frequency (Hz)",
        min_value=0.001,
        step=1e6,
        format="%.3e",
        value=None,
        key="frequency_hz",
    )
    dielectric_constant = st.number_input(
        "Dielectric constant εr",
        min_value=1.0,
        max_value=1000.0,
        step=0.1,
        format="%.2f",
        value=None,
        key="relative_permittivity",
        help="Relative permittivity; assumes a non-magnetic line, so vp = c / sqrt(εr).",
    )

if any(value is None for value in (rl, xl, z0, freq, dielectric_constant)):
    st.info("Enter all five system parameters to run the analysis.")
    st.stop()

# ---------------------------------------------------------------------------
# Computation (all maths lives in tltoolkit)
# ---------------------------------------------------------------------------
zl = complex(rl, xl)
matching_method = st.session_state["matching_method"]
recommendation_name, stub_termination = MATCHING_METHODS[matching_method]
try:
    vp = phase_velocity(1.0 / math.sqrt(dielectric_constant))
    report = recommend_matching(
        z0,
        zl,
        freq,
        vp=vp,
        include_stub=stub_termination is not None,
        stub_termination=stub_termination or "open",
    )
except ValueError as exc:
    st.error(f"Invalid input: {exc}")
    st.stop()

analysis = report.analysis

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
st.caption(
    f"Z0 = {z0:g} Ω · ZL = {rl:g} {'+' if xl >= 0 else '-'} j{abs(xl):g} Ω · "
    f"f = {freq:g} Hz · εr = {dielectric_constant:.2f}"
)

tab_analysis, tab_matching = st.tabs(["Analysis", "Matching"])

# ---------------------------------------------------------------------------
# Tab 1: analysis
# ---------------------------------------------------------------------------
with tab_analysis:
    label, level = ui.vswr_quality(analysis.vswr)
    getattr(st, level)(f"**{label}** (VSWR = {ui.fmt_number(analysis.vswr, 3)})")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("VSWR", ui.fmt_number(analysis.vswr, 3))
    m2.metric("|Γ|", f"{analysis.gamma_mag:.4f}")
    m3.metric("Return loss", ui.fmt_number(analysis.return_loss_db, 2, "dB"))
    m4.metric("Mismatch loss", ui.fmt_number(analysis.mismatch_loss_db, 3, "dB"))

    st.progress(
        analysis.delivered_power_percent / 100.0,
        text=(
            f"{analysis.delivered_power_percent:.2f} % of the power reaches the load, "
            f"{analysis.reflected_power_percent:.2f} % is reflected"
        ),
    )

    with st.expander("Details"):
        gamma = analysis.gamma
        zn = zl / z0
        st.markdown(
            f"- Reflection coefficient Γ = {gamma.real:.5f} {'+' if gamma.imag >= 0 else '-'} "
            f"j{abs(gamma.imag):.5f} = {abs(gamma):.5f} ∠ {analysis.gamma_angle_deg:.2f}°\n"
            f"- Normalised load z = ZL / Z0 = {zn.real:.4f} {'+' if zn.imag >= 0 else '-'} "
            f"j{abs(zn.imag):.4f}\n"
            f"- Wavelength on the line: {vp / freq * 1000:.3f} mm"
        )

    with st.expander("Assumptions and limitations"):
        st.markdown(
            "- Matching designs assume a **lossless** main line with real Z0 and a **single "
            "design frequency**; bandwidth is not optimised.\n"
            "- Lumped components are ideal (no parasitics).\n"
            "- Open, short and purely reactive loads absorb no power, so no passive lossless "
            "network can match them.\n"
            "- The line is assumed non-magnetic, so its phase velocity is calculated as "
            "c / sqrt(εr)."
        )

# ---------------------------------------------------------------------------
# Tab 2: matching
# ---------------------------------------------------------------------------
with tab_matching:
    st.radio(
        "Select Matching Method",
        list(MATCHING_METHODS),
        horizontal=True,
        key="matching_method",
    )
    selected_recommendation = next(
        (rec for rec in report.recommendations if rec.method == recommendation_name), None
    )
    if selected_recommendation is not None:
        rec = selected_recommendation
        with st.container(border=True):
            st.subheader(rec.method)
            st.caption(rec.reason)
            st.dataframe(ui.recommendation_rows(rec), hide_index=True)
        st.caption(
            "Every solution is verified by recomputing the impedance (or admittance) after "
            "matching; the residual error should be close to zero."
        )
    else:
        if not report.recommendations:
            notes = report.notes
        elif matching_method == "Lumped Element (L, C)":
            notes = [note for note in report.notes if note.startswith("L-network")]
        else:
            notes = [note for note in report.notes if note.startswith("No single-stub")]
        if notes:
            for note in notes:
                st.info(note)
        else:
            st.info("No matching design is available for this method and these parameters.")
    image_path, image_caption = MATCHING_DIAGRAMS[matching_method]
    st.image(str(image_path), caption=image_caption, width=700)

# ---------------------------------------------------------------------------
