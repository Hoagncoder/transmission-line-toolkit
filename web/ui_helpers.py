"""UI-independent helpers for the Streamlit app.

Everything here is a plain function (no Streamlit calls), so it can be unit-tested.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from tltoolkit.analysis import Recommendation
from tltoolkit.matching import LNetworkSolution, QuarterWaveResult, StubSolution

FREQ_UNITS: Dict[str, float] = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}


def fmt_number(value: float, digits: int = 4, unit: str = "") -> str:
    """Format a float, showing the infinity sign for ``inf``."""
    if math.isinf(value):
        text = "∞"
    else:
        text = f"{value:.{digits}f}"
    return f"{text} {unit}".strip()


def vswr_quality(vswr: float) -> Tuple[str, str]:
    """Return ``(label, level)`` where level is a Streamlit status name."""
    if vswr <= 1.05:
        return "Matched", "success"
    if vswr <= 1.5:
        return "Excellent match", "success"
    if vswr <= 2.0:
        return "Good match", "info"
    if vswr <= 3.0:
        return "Fair match - matching network recommended", "warning"
    return "Poor match - matching network strongly recommended", "error"


def clip_for_plot(values: np.ndarray, limit: float) -> np.ndarray:
    """Replace huge values (near singularities) by NaN so plot lines break cleanly."""
    out = np.array(values, dtype=float, copy=True)
    out[~np.isfinite(out) | (np.abs(out) > limit)] = np.nan
    return out


def _mark(matched: bool) -> str:
    return "✓" if matched else "✗"


def recommendation_rows(rec: Recommendation) -> List[Dict[str, str]]:
    """Table rows (one dict per solution) for a matching recommendation."""
    rows: List[Dict[str, str]] = []
    for i, sol in enumerate(rec.solutions, start=1):
        if isinstance(sol, QuarterWaveResult):
            lam = sol.distance_to_load_lambda or 0.0
            metres = sol.distance_to_load_m or 0.0
            rows.append(
                {
                    "#": str(i),
                    "Position from load (λ)": f"{lam:.5f}",
                    "Position from load (mm)": f"{metres * 1000:.3f}",
                    "Zin at that position (Ω)": f"{sol.r_load_ohm:.4f}",
                    "Transformer Z1 (Ω)": f"{sol.z1_ohm:.4f}",
                    "Residual error (Ω)": f"{sol.error_ohm:.1e}",
                    "Verified": _mark(sol.matched),
                }
            )
        elif isinstance(sol, LNetworkSolution):
            rows.append(
                {
                    "#": str(i),
                    "Topology": sol.topology.value,
                    "Series element": str(sol.series_component or "-"),
                    "Shunt element": str(sol.shunt_component or "-"),
                    "Residual error (Ω)": f"{sol.error_ohm:.1e}",
                    "Verified": _mark(sol.matched),
                }
            )
        elif isinstance(sol, StubSolution):
            rows.append(
                {
                    "#": str(i),
                    "Stub position d (λ)": f"{sol.d_lambda:.5f}",
                    "Stub position d (mm)": f"{sol.d_m * 1000:.3f}",
                    "Stub length l (λ)": f"{sol.l_lambda:.5f}",
                    "Stub length l (mm)": f"{sol.l_m * 1000:.3f}",
                    "Stub type": f"{sol.termination}-circuited",
                    "Residual error": f"{sol.error:.1e}",
                    "Verified": _mark(sol.matched),
                }
            )
    return rows
