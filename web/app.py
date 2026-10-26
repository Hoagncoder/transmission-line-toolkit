"""Streamlit web app for transmission-line analysis and impedance matching.

Run from the repository root:  streamlit run web/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:  # works after `pip install -e .`, otherwise fall back to the source tree
    import tltoolkit  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(HERE.parent / "src"))

import cmath  # noqa: E402

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
import ui_helpers as ui  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from tltoolkit import (  # noqa: E402
    __version__,
    phase_velocity,
    real_impedance_points,
    recommend_matching,
    sweep_impedance,
)

st.set_page_config(
    page_title="Transmission Line Toolkit",
    page_icon="📡",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state and callbacks
# ---------------------------------------------------------------------------
DEFAULTS = {
    "z0": 50.0,
    "load_r": None,
    "load_x": None,
    "fval": 1.0,
    "funit": "GHz",
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
# Sidebar: inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Parameters")
    z0 = st.number_input(
        "Line impedance Z0 (Ω)", min_value=1.0, max_value=1000.0, step=1.0, key="z0"
    )
    col_r, col_x = st.columns(2)
    rl = col_r.number_input(
        "Load R (Ω)",
        min_value=0.0,
        max_value=1e6,
        step=1.0,
        value=None,
        key="load_r",
        help="Resistive part of the load. Must be ≥ 0 (passive load).",
    )
    xl = col_x.number_input(
        "Load X (Ω)",
        min_value=-1e6,
        max_value=1e6,
        step=1.0,
        value=None,
        key="load_x",
        help="Reactive part: positive = inductive, negative = capacitive.",
    )
    col_f, col_u = st.columns([2, 1])
    fval = col_f.number_input("Frequency", min_value=0.001, step=0.1, format="%.3f", key="fval")
    funit = col_u.selectbox("Unit", list(ui.FREQ_UNITS), key="funit")

    with st.expander("Advanced options"):
        vf = st.slider("Velocity factor", 0.3, 1.0, 1.0, 0.01)
        st.markdown("**Sweep**")
        max_lambda = st.slider("Length (wavelengths)", 0.25, 3.0, 1.0, 0.25)
        points = st.slider("Points", 101, 2001, 501, 100)
        alpha = st.number_input(
            "Attenuation α (Np/m)", min_value=0.0, max_value=10.0, value=0.0, step=0.01
        )

if rl is None or xl is None:
    st.info("Enter both Load R and Load X to run the analysis.")
    st.stop()

# ---------------------------------------------------------------------------
# Computation (all maths lives in tltoolkit)
# ---------------------------------------------------------------------------
zl = complex(rl, xl)
freq = fval * ui.FREQ_UNITS[funit]
matching_method = st.session_state["matching_method"]
recommendation_name, stub_termination = MATCHING_METHODS[matching_method]
try:
    vp = phase_velocity(vf)
    report = recommend_matching(
        z0,
        zl,
        freq,
        vp=vp,
        include_stub=stub_termination is not None,
        stub_termination=stub_termination or "open",
    )
    sweep = sweep_impedance(
        z0, zl, freq, max_length_lambda=max_lambda, points=points, vp=vp, alpha=alpha
    )
except ValueError as exc:
    st.error(f"Invalid input: {exc}")
    st.stop()

analysis = report.analysis

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📡 Transmission Line Toolkit")
st.caption(
    f"Z0 = {z0:g} Ω · ZL = {rl:g} {'+' if xl >= 0 else '-'} j{abs(xl):g} Ω · "
    f"f = {fval:g} {funit} · velocity factor {vf:.2f}"
)

tab_analysis, tab_matching, tab_sweep = st.tabs(["📊 Analysis", "🔧 Matching", "📈 Sweep"])

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
    image_path, image_caption = MATCHING_DIAGRAMS[matching_method]
    st.image(str(image_path), caption=image_caption, width=700)
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

# ---------------------------------------------------------------------------
# Tab 3: sweep
# ---------------------------------------------------------------------------
with tab_sweep:
    limit = 10.0 * max(z0, abs(zl))
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Input impedance along the line", "Voltage standing-wave pattern"),
    )
    x = sweep.distance_lambda
    fig.add_trace(
        go.Scatter(x=x, y=ui.clip_for_plot(sweep.zin.real, limit), name="Re(Zin)"), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=x, y=ui.clip_for_plot(sweep.zin.imag, limit), name="Im(Zin)"), row=1, col=1
    )
    fig.add_hline(y=z0, line_dash="dash", line_color="gray", annotation_text="Z0", row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=sweep.voltage_magnitude,
            name="|V|",
            line={"color": "#2ca02c"},
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    marks = []
    if abs(zl.imag) > 1e-9 and zl.real > 0:
        try:
            base = [p.distance_lambda for p in real_impedance_points(z0, zl, freq, vp)]
        except ValueError:
            base = []
        for b in base:
            k = 0
            while b + 0.5 * k <= max_lambda:
                marks.append(b + 0.5 * k)
                k += 1
    for m in marks:
        fig.add_vline(x=m, line_dash="dot", line_color="rgba(120,120,120,0.6)", row="all", col=1)

    fig.update_xaxes(title_text="Distance from load (wavelengths)", row=2, col=1)
    fig.update_yaxes(title_text="Impedance (Ω)", row=1, col=1)
    fig.update_yaxes(title_text="|V| (normalised)", row=2, col=1)
    fig.update_layout(height=640, hovermode="x unified", margin={"t": 60, "b": 40})
    st.plotly_chart(fig)

    if marks:
        st.caption(
            "Dotted lines mark the planes where Zin is purely real - "
            "the places where a quarter-wave transformer can be inserted."
        )
    if cmath.isclose(analysis.gamma_mag, 1.0, abs_tol=1e-9):
        st.caption("Values far above the plot range (near singularities) are omitted.")

    st.download_button(
        "⬇ Download sweep as CSV",
        data=sweep.to_csv_text(),
        file_name="sweep.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
with st.expander("Assumptions and limitations"):
    st.markdown(
        "- Matching designs assume a **lossless** main line with real Z0 and a **single design "
        "frequency**; bandwidth is not optimised.\n"
        "- Lumped components are ideal (no parasitics).\n"
        "- Open, short and purely reactive loads absorb no power, so no passive lossless "
        "network can match them.\n"
        "- Attenuation α is applied to the sweep only, not to the matching designs."
    )
st.caption(f"transmission-line-toolkit v{__version__}")
