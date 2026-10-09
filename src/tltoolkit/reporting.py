"""Human-readable text formatting for analysis and matching results."""

from __future__ import annotations

import math
from typing import List

import numpy as np

from .analysis import LoadAnalysis, MatchingReport, SweepResult
from .matching import LNetworkSolution, QuarterWaveResult, StubSolution

__all__ = ["format_analysis", "format_complex", "format_report", "format_sweep_summary"]


def format_complex(z: complex, digits: int = 4) -> str:
    """Format a complex impedance as ``R + jX`` (ohms), handling ``inf``."""
    if math.isinf(z.real) or math.isinf(z.imag):
        return "inf"
    sign = "-" if z.imag < 0 and round(abs(z.imag), digits) > 0 else "+"
    return f"{z.real:.{digits}f} {sign} j{abs(z.imag):.{digits}f}"


def _db(value: float) -> str:
    return "inf" if math.isinf(value) else f"{value:.4f}"


def format_analysis(a: LoadAnalysis) -> str:
    """Multi-line summary of a :class:`LoadAnalysis`."""
    vswr = "inf" if math.isinf(a.vswr) else f"{a.vswr:.4f}"
    return "\n".join(
        [
            "Load analysis",
            f"  Z0                 : {a.z0:.4f} Ohm",
            f"  ZL                 : {format_complex(a.zl)} Ohm",
            f"  Gamma              : {format_complex(a.gamma, 5)}",
            f"  |Gamma|            : {a.gamma_mag:.5f}",
            f"  angle(Gamma)       : {a.gamma_angle_deg:.3f} deg",
            f"  VSWR               : {vswr}",
            f"  Return loss        : {_db(a.return_loss_db)} dB",
            f"  Mismatch loss      : {_db(a.mismatch_loss_db)} dB",
            f"  Reflected power    : {a.reflected_power_percent:.3f} %",
            f"  Delivered power    : {a.delivered_power_percent:.3f} %",
        ]
    )


def _format_quarter_wave(i: int, s: QuarterWaveResult) -> List[str]:
    lines = [f"  Solution {i}:"]
    if s.distance_to_load_m is not None:
        lines.append(
            f"    Place at d = {s.distance_to_load_lambda:.5f} lambda "
            f"({s.distance_to_load_m:.5f} m) from the load, where Zin = {s.r_load_ohm:.4f} Ohm"
        )
    else:
        lines.append(f"    Place directly at the load (RL = {s.r_load_ohm:.4f} Ohm)")
    lines.append(f"    Z1 = {s.z1_ohm:.4f} Ohm, length = {s.length_lambda:.2f} lambda")
    lines.append(_check_line(s.zin_verified, s.error_ohm, s.matched))
    return lines


def _format_l_network(i: int, s: LNetworkSolution) -> List[str]:
    lines = [f"  Solution {i}: {s.topology.value}"]
    if s.series_component is not None:
        lines.append(f"    Series: X = {s.series_reactance_ohm:+.4f} Ohm -> {s.series_component}")
    if s.shunt_component is not None:
        lines.append(f"    Shunt : B = {s.shunt_susceptance_s:+.6f} S -> {s.shunt_component}")
    lines.append(_check_line(s.zin_verified, s.error_ohm, s.matched))
    return lines


def _format_stub(i: int, s: StubSolution) -> List[str]:
    return [
        f"  Solution {i}:",
        f"    Stub position d = {s.d_lambda:.5f} lambda ({s.d_m:.5f} m) from the load",
        f"    Stub length   l = {s.l_lambda:.5f} lambda ({s.l_m:.5f} m), {s.termination}-circuited",
        f"    Line admittance at stub = {format_complex(s.line_admittance_normalized)} "
        "(normalised)",
        f"    Check: Y_total/Y0 = {format_complex(s.y_total_normalized, 6)}, "
        f"error = {s.error:.2e} -> {'OK' if s.matched else 'FAILED'}",
    ]


def _check_line(zin: complex, error: float, matched: bool) -> str:
    return (
        f"    Check: Zin = {format_complex(zin, 6)} Ohm, error = {error:.2e} Ohm "
        f"-> {'OK' if matched else 'FAILED'}"
    )


def format_report(report: MatchingReport) -> str:
    """Multi-line summary of a :class:`MatchingReport`."""
    out = [format_analysis(report.analysis)]
    for rec in report.recommendations:
        out.append("")
        out.append(f"{rec.method}")
        out.append(f"  {rec.reason}")
        for i, sol in enumerate(rec.solutions, start=1):
            if isinstance(sol, QuarterWaveResult):
                out.extend(_format_quarter_wave(i, sol))
            elif isinstance(sol, LNetworkSolution):
                out.extend(_format_l_network(i, sol))
            elif isinstance(sol, StubSolution):
                out.extend(_format_stub(i, sol))
    if report.notes:
        out.append("")
        out.append("Notes")
        out.extend(f"  - {n}" for n in report.notes)
    return "\n".join(out)


def format_sweep_summary(s: SweepResult) -> str:
    """Short summary of a :class:`SweepResult`."""
    finite = s.vswr[np.isfinite(s.vswr)]
    vswr_txt = "inf" if finite.size == 0 else f"{finite.min():.4f} .. {finite.max():.4f}"
    return "\n".join(
        [
            f"Sweep: {len(s.distance_m)} points, d = 0 .. {s.distance_lambda[-1]:.3f} lambda "
            f"({s.distance_m[-1]:.4f} m)",
            f"  Zin(d = 0)   : {format_complex(complex(s.zin[0]))} Ohm",
            f"  Zin(d = end) : {format_complex(complex(s.zin[-1]))} Ohm",
            f"  VSWR range   : {vswr_txt}",
            f"  |V| range    : {s.voltage_magnitude.min():.4f} .. {s.voltage_magnitude.max():.4f}",
        ]
    )
