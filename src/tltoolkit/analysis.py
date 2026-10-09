"""High-level analysis: load figures of merit, matching advice and line sweeps."""

from __future__ import annotations

import cmath
import csv
import io
import math
from dataclasses import dataclass, field
from typing import List

import numpy as np

from .line import (
    C0,
    input_impedance,
    mismatch_loss_db,
    reflection_coefficient,
    return_loss_db,
    validate_frequency,
    validate_load,
    validate_phase_velocity,
    validate_z0,
    vswr_from_gamma,
    wavelength,
)
from .matching import (
    LNetworkSolution,
    QuarterWaveResult,
    StubSolution,
    l_network_match,
    quarter_wave_after_real_point,
    quarter_wave_transformer,
    single_stub_match,
)

__all__ = [
    "LoadAnalysis",
    "MatchingReport",
    "Recommendation",
    "SweepResult",
    "analyze_load",
    "recommend_matching",
    "sweep_impedance",
]


# ---------------------------------------------------------------------------
# Load analysis
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LoadAnalysis:
    """Figures of merit of load ``zl`` on a line of impedance ``z0``."""

    z0: float
    zl: complex
    gamma: complex
    gamma_mag: float
    gamma_angle_deg: float
    vswr: float
    return_loss_db: float
    mismatch_loss_db: float
    reflected_power_percent: float
    delivered_power_percent: float


def analyze_load(z0: float, zl: complex) -> LoadAnalysis:
    """Compute reflection coefficient, VSWR, return/mismatch loss and power split."""
    z0 = validate_z0(z0)
    zl = validate_load(zl)
    gamma = reflection_coefficient(z0, zl)
    mag = abs(gamma)
    return LoadAnalysis(
        z0=z0,
        zl=zl,
        gamma=gamma,
        gamma_mag=mag,
        gamma_angle_deg=math.degrees(cmath.phase(gamma)),
        vswr=vswr_from_gamma(gamma),
        return_loss_db=return_loss_db(gamma),
        mismatch_loss_db=mismatch_loss_db(gamma),
        reflected_power_percent=100.0 * min(1.0, mag**2),
        delivered_power_percent=100.0 * max(0.0, 1.0 - mag**2),
    )


# ---------------------------------------------------------------------------
# Matching recommendations
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Recommendation:
    """One matching technique with its verified solutions."""

    method: str
    reason: str
    solutions: list


@dataclass(frozen=True)
class MatchingReport:
    """Analysis of a load plus every matching technique that applies to it."""

    analysis: LoadAnalysis
    recommendations: List[Recommendation] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def recommend_matching(
    z0: float,
    zl: complex,
    freq: float,
    vp: float = C0,
    include_stub: bool = True,
    stub_termination: str = "open",
) -> MatchingReport:
    """Analyse the load and collect every applicable, verified matching design.

    Reasons why a technique could not be applied are returned in ``notes``
    instead of being silently dropped.
    """
    z0 = validate_z0(z0)
    zl = validate_load(zl)
    freq = validate_frequency(freq)
    vp = validate_phase_velocity(vp)
    analysis = analyze_load(z0, zl)
    report = MatchingReport(analysis=analysis)

    if analysis.gamma_mag < 1e-9:
        report.notes.append("The load is already matched; no matching network is needed.")
        return report
    if cmath.isinf(zl) or zl.real <= 1e-12:
        report.notes.append(
            "Open, short and purely reactive loads absorb no power, so no passive "
            "lossless network can match them."
        )
        return report

    # 1) Quarter-wave transformer
    if abs(zl.imag) < 1e-9 * max(1.0, abs(zl)):
        qw: List[QuarterWaveResult] = [quarter_wave_transformer(z0, zl.real)]
        reason = "Purely resistive load: place the transformer directly at the load."
    else:
        qw = quarter_wave_after_real_point(z0, zl, freq, vp)
        reason = "Complex load: place the transformer where the line impedance is purely real."
    report.recommendations.append(Recommendation("Quarter-wave transformer", reason, qw))

    # 2) L-network
    try:
        l_solutions: List[LNetworkSolution] = [
            s for s in l_network_match(z0, zl, freq) if s.matched
        ]
    except ValueError as exc:
        l_solutions = []
        report.notes.append(f"L-network not applicable: {exc}")
    if l_solutions:
        report.recommendations.append(
            Recommendation(
                "L-network (lumped elements)",
                "Compact solution; element values are valid at this single frequency.",
                l_solutions,
            )
        )

    # 3) Single stub
    if include_stub:
        stubs: List[StubSolution] = [
            s for s in single_stub_match(z0, zl, freq, vp, stub_termination) if s.matched
        ]
        if stubs:
            report.recommendations.append(
                Recommendation(
                    f"Single shunt stub ({stub_termination}-circuited)",
                    "Distributed solution, convenient for microstrip and coaxial lines.",
                    stubs,
                )
            )
        else:
            report.notes.append("No single-stub solution was found for this load.")
    return report


# ---------------------------------------------------------------------------
# Sweep along the line
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SweepResult:
    """Quantities along the line, sampled from the load (d = 0) towards the source.

    ``vswr`` is the VSWR seen looking into the line at each plane; it is constant
    on a lossless line and decreases with distance on a lossy one.
    ``voltage_magnitude`` is the standing-wave pattern ``|V(d)|`` normalised to the
    incident-wave amplitude at the load.
    """

    z0: float
    zl: complex
    freq: float
    distance_m: np.ndarray
    distance_lambda: np.ndarray
    zin: np.ndarray
    gamma_mag: np.ndarray
    vswr: np.ndarray
    voltage_magnitude: np.ndarray

    def to_csv_text(self) -> str:
        """Return the sweep as CSV text."""
        header = [
            "distance_m",
            "distance_lambda",
            "zin_real_ohm",
            "zin_imag_ohm",
            "gamma_mag",
            "vswr",
            "voltage_magnitude",
        ]
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(header)
        for i in range(len(self.distance_m)):
            writer.writerow(
                [
                    f"{self.distance_m[i]:.9g}",
                    f"{self.distance_lambda[i]:.9g}",
                    f"{self.zin[i].real:.9g}",
                    f"{self.zin[i].imag:.9g}",
                    f"{self.gamma_mag[i]:.9g}",
                    f"{self.vswr[i]:.9g}",
                    f"{self.voltage_magnitude[i]:.9g}",
                ]
            )
        return buffer.getvalue()

    def save_csv(self, path: str) -> None:
        """Write the sweep to a CSV file."""
        with open(path, "w", newline="", encoding="utf-8") as fh:
            fh.write(self.to_csv_text())


def sweep_impedance(
    z0: float,
    zl: complex,
    freq: float,
    max_length_lambda: float = 0.5,
    points: int = 501,
    vp: float = C0,
    alpha: float = 0.0,
) -> SweepResult:
    """Sample ``Zin``, ``|Gamma|``, VSWR and ``|V|`` from the load up to ``max_length_lambda``."""
    z0 = validate_z0(z0)
    zl = validate_load(zl)
    lam = wavelength(freq, vp)
    if max_length_lambda <= 0:
        raise ValueError("max_length_lambda must be > 0.")
    if points < 2:
        raise ValueError("points must be >= 2.")

    d = np.linspace(0.0, max_length_lambda * lam, int(points))
    zin = np.asarray(input_impedance(z0, zl, d, freq, vp, alpha), dtype=complex)

    with np.errstate(divide="ignore", invalid="ignore"):
        gamma_plane = (zin - z0) / (zin + z0)
        mag = np.abs(gamma_plane)
        vswr = np.where(mag < 1.0, (1.0 + mag) / (1.0 - mag), np.inf)

    gamma_load = reflection_coefficient(z0, zl)
    gamma_prop = alpha + 1j * 2.0 * math.pi / lam
    voltage = np.abs(np.exp(gamma_prop * d) + gamma_load * np.exp(-gamma_prop * d))
    return SweepResult(
        z0=z0,
        zl=zl,
        freq=float(freq),
        distance_m=d,
        distance_lambda=d / lam,
        zin=zin,
        gamma_mag=mag,
        vswr=vswr,
        voltage_magnitude=voltage,
    )
