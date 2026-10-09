"""Impedance-matching techniques.

Every function returns plain numeric results (dataclasses) and verifies its own
solution by recomputing the impedance (or admittance) seen after matching.
Formatting for humans lives in :mod:`tltoolkit.reporting`.

Assumptions
-----------
* The main line is lossless with real characteristic impedance ``z0``.
* Lumped components are ideal and the design is valid at a single frequency.
* The load must be finite with ``Re(ZL) > 0``; a purely reactive, open or
  shorted load absorbs no power and cannot be matched with a lossless network.
"""

from __future__ import annotations

import cmath
import dataclasses
import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from .line import (
    C0,
    input_impedance,
    input_impedance_electrical,
    validate_frequency,
    validate_load,
    validate_phase_velocity,
    validate_z0,
    wavelength,
)

__all__ = [
    "Component",
    "LNetworkSolution",
    "QuarterWaveResult",
    "RealPoint",
    "StubSolution",
    "Topology",
    "l_network_match",
    "quarter_wave_after_real_point",
    "quarter_wave_transformer",
    "real_impedance_points",
    "reactance_to_component",
    "single_stub_match",
    "susceptance_to_component",
]

#: A solution counts as "matched" if the residual error is below this fraction of Z0.
MATCH_TOL = 1e-6


def _require_matchable_load(zl: complex) -> complex:
    zl = validate_load(zl)
    if cmath.isinf(zl) or zl.real <= 0:
        raise ValueError(
            "Matching requires a finite load with Re(ZL) > 0 "
            "(open, short and purely reactive loads absorb no power)."
        )
    return zl


# ---------------------------------------------------------------------------
# Lumped components
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Component:
    """An ideal lumped element: ``kind`` is ``"inductor"`` or ``"capacitor"``.

    ``value`` is in henries (inductor) or farads (capacitor).
    """

    kind: str
    value: float

    def __str__(self) -> str:
        if self.kind == "inductor":
            return f"inductor L = {self.value * 1e9:.4f} nH"
        return f"capacitor C = {self.value * 1e12:.4f} pF"


def reactance_to_component(x_ohm: float, omega: float) -> Optional[Component]:
    """Series reactance -> element: ``X > 0`` is an inductor, ``X < 0`` a capacitor."""
    if abs(x_ohm) < 1e-9:
        return None
    if x_ohm > 0:
        return Component("inductor", x_ohm / omega)
    return Component("capacitor", -1.0 / (omega * x_ohm))


def susceptance_to_component(b_siemens: float, omega: float) -> Optional[Component]:
    """Shunt susceptance -> element: ``B > 0`` is a capacitor, ``B < 0`` an inductor."""
    if abs(b_siemens) < 1e-12:
        return None
    if b_siemens > 0:
        return Component("capacitor", b_siemens / omega)
    return Component("inductor", -1.0 / (omega * b_siemens))


# ---------------------------------------------------------------------------
# Quarter-wave transformer
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RealPoint:
    """A point on the line where the input impedance is purely real."""

    distance_m: float
    distance_lambda: float
    resistance_ohm: float


@dataclass(frozen=True)
class QuarterWaveResult:
    """A lambda/4 transformer design.

    ``distance_to_load_*`` is ``None`` when the transformer sits directly at the
    (real) load, otherwise it is the distance from the load to the transformer.
    """

    z1_ohm: float
    r_load_ohm: float
    zin_verified: complex
    error_ohm: float
    matched: bool
    length_lambda: float = 0.25
    distance_to_load_m: Optional[float] = None
    distance_to_load_lambda: Optional[float] = None


def quarter_wave_transformer(z0: float, r_load: complex) -> QuarterWaveResult:
    """Design a lambda/4 transformer ``Z1 = sqrt(Z0 * RL)`` for a purely resistive load."""
    z0 = validate_z0(z0)
    r = complex(r_load)
    if r.real <= 0 or abs(r.imag) > 1e-9 * max(1.0, abs(r.real)):
        raise ValueError(
            "A quarter-wave transformer needs a purely resistive load (RL > 0, XL = 0). "
            "For a complex load use quarter_wave_after_real_point()."
        )
    r_val = r.real
    z1 = math.sqrt(z0 * r_val)
    zin = complex(input_impedance_electrical(z1, r_val, math.pi / 2.0))
    error = abs(zin - z0)
    return QuarterWaveResult(
        z1_ohm=z1,
        r_load_ohm=r_val,
        zin_verified=zin,
        error_ohm=error,
        matched=error < MATCH_TOL * z0,
    )


def real_impedance_points(z0: float, zl: complex, freq: float, vp: float = C0) -> List[RealPoint]:
    """Distances ``d`` in ``[0, lambda/2)`` where ``Im(Zin(d)) = 0`` (lossless line).

    Solved analytically. With ``t = tan(beta*d)`` and ``ZL = R + jX`` the condition
    reduces to ``Z0*X*t^2 - (Z0^2 - |ZL|^2)*t - Z0*X = 0``. A purely resistive load
    is already real at ``d = 0`` and again at ``d = lambda/4``.
    """
    z0 = validate_z0(z0)
    zl = _require_matchable_load(zl)
    lam = wavelength(freq, vp)
    beta = 2.0 * math.pi / lam
    r, x = zl.real, zl.imag

    if abs(x) < 1e-9 * max(1.0, abs(zl)):
        return [RealPoint(0.0, 0.0, r), RealPoint(lam / 4.0, 0.25, z0**2 / r)]

    a = z0 * x
    b = -(z0**2 - abs(zl) ** 2)
    c = -z0 * x
    sd = math.sqrt(b * b - 4.0 * a * c)  # always > 0 because a*c = -(z0*x)^2 < 0
    points = []
    for t in ((-b + sd) / (2.0 * a), (-b - sd) / (2.0 * a)):
        d = (math.atan(t) % math.pi) / beta
        zin = complex(input_impedance(z0, zl, d, freq, vp))
        points.append(RealPoint(d, d / lam, zin.real))
    return sorted(points, key=lambda p: p.distance_m)


def quarter_wave_after_real_point(
    z0: float, zl: complex, freq: float, vp: float = C0
) -> List[QuarterWaveResult]:
    """Lambda/4 transformer designs placed at the points where ``Zin`` is real."""
    results = []
    for p in real_impedance_points(z0, zl, freq, vp):
        design = quarter_wave_transformer(z0, p.resistance_ohm)
        results.append(
            dataclasses.replace(
                design,
                distance_to_load_m=p.distance_m,
                distance_to_load_lambda=p.distance_lambda,
            )
        )
    return results


# ---------------------------------------------------------------------------
# L-network
# ---------------------------------------------------------------------------
class Topology(Enum):
    """Where the lumped elements of an L-network are placed."""

    NONE = "already matched, no element needed"
    SERIES_ONLY = "single series element"
    SHUNT_AT_LOAD = "shunt element across the load, series element towards the source"
    SERIES_AT_LOAD = "series element next to the load, shunt element towards the source"


@dataclass(frozen=True)
class LNetworkSolution:
    """One L-network solution, with numeric values and a self-check."""

    topology: Topology
    series_reactance_ohm: float
    shunt_susceptance_s: float
    series_component: Optional[Component]
    shunt_component: Optional[Component]
    zin_verified: complex
    error_ohm: float
    matched: bool


def _l_zin(topology: Topology, zl: complex, x: float, b: float) -> complex:
    if topology is Topology.SHUNT_AT_LOAD:
        return 1.0 / (1.0 / zl + 1j * b) + 1j * x
    if topology is Topology.SERIES_AT_LOAD:
        return 1.0 / (1.0 / (zl + 1j * x) + 1j * b)
    return zl + 1j * x


def l_network_match(z0: float, zl: complex, freq: float) -> List[LNetworkSolution]:
    """Match ``zl`` to the real impedance ``z0`` with a two-element L-network.

    Follows Pozar, *Microwave Engineering*, section 5.1. Both topologies are
    tried; every returned solution has been verified by recomputing the
    impedance seen at the input of the network.
    """
    z0 = validate_z0(z0)
    zl = _require_matchable_load(zl)
    omega = 2.0 * math.pi * validate_frequency(freq)
    rl, xl = zl.real, zl.imag
    tol = MATCH_TOL * z0

    if abs(zl - z0) < tol:
        return [LNetworkSolution(Topology.NONE, 0.0, 0.0, None, None, zl, abs(zl - z0), True)]

    candidates = []  # (topology, X in ohm, B in siemens)

    if abs(rl - z0) < tol:
        candidates.append((Topology.SERIES_ONLY, -xl, 0.0))

    # Shunt element across the load, series element towards the source.
    mag2 = rl**2 + xl**2
    radicand = (rl / z0) * (mag2 - z0 * rl)
    if radicand >= 0:
        root = math.sqrt(radicand)
        for sign in (1.0, -1.0):
            b = (xl + sign * root) / mag2
            if abs(b) < 1e-15:
                continue
            x = 1.0 / b + xl * z0 / rl - z0 / (b * rl)
            candidates.append((Topology.SHUNT_AT_LOAD, x, b))

    # Series element next to the load, shunt element towards the source.
    if rl < z0:
        root = math.sqrt(rl * (z0 - rl))
        for sign in (1.0, -1.0):
            x = sign * root - xl
            b = sign * math.sqrt((z0 - rl) / rl) / z0
            candidates.append((Topology.SERIES_AT_LOAD, x, b))

    solutions = []
    for topology, x, b in candidates:
        zin = _l_zin(topology, zl, x, b)
        error = abs(zin - z0)
        solutions.append(
            LNetworkSolution(
                topology=topology,
                series_reactance_ohm=x,
                shunt_susceptance_s=b,
                series_component=reactance_to_component(x, omega),
                shunt_component=susceptance_to_component(b, omega),
                zin_verified=zin,
                error_ohm=error,
                matched=error < tol,
            )
        )
    return solutions


# ---------------------------------------------------------------------------
# Single shunt stub
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StubSolution:
    """One single-stub design.

    ``d_*`` is the distance from the load to the stub, ``l_*`` the stub length.
    Admittances are normalised to ``Y0 = 1/Z0``; ``y_total_normalized`` is the
    admittance at the junction after adding the stub and should equal 1.
    """

    d_m: float
    d_lambda: float
    l_m: float
    l_lambda: float
    termination: str
    line_admittance_normalized: complex
    stub_susceptance_normalized: float
    y_total_normalized: complex
    error: float
    matched: bool


def single_stub_match(
    z0: float,
    zl: complex,
    freq: float,
    vp: float = C0,
    termination: str = "open",
) -> List[StubSolution]:
    """Single shunt-stub matching of a load on a lossless line (closed-form).

    Finds the distances ``d`` where the normalised admittance has real part 1,
    then the stub length that cancels the remaining susceptance. Returns up to
    two solutions sorted by distance; an empty list means the load is already
    matched.
    """
    if termination not in ("open", "short"):
        raise ValueError("termination must be 'open' or 'short'.")
    z0 = validate_z0(z0)
    zl = _require_matchable_load(zl)
    validate_phase_velocity(vp)
    lam = wavelength(freq, vp)
    beta = 2.0 * math.pi / lam
    if abs(zl - z0) < MATCH_TOL * z0:
        return []

    y = z0 / zl
    g, b = y.real, y.imag
    # Re(y_in(d)) = 1 with t = tan(beta*d)  ->  A t^2 + B t + C = 0
    a_coef, b_coef, c_coef = g * g + b * b - g, -2.0 * b, 1.0 - g

    thetas = []
    if abs(a_coef) < 1e-12:
        if abs(b_coef) > 1e-12:
            thetas.append(math.atan(-c_coef / b_coef) % math.pi)
        thetas.append(math.pi / 2.0)  # the second root is t -> infinity
    else:
        disc = b_coef**2 - 4.0 * a_coef * c_coef
        if disc >= -1e-12:
            sd = math.sqrt(max(0.0, disc))
            for t in ((-b_coef + sd) / (2.0 * a_coef), (-b_coef - sd) / (2.0 * a_coef)):
                thetas.append(math.atan(t) % math.pi)

    solutions: List[StubSolution] = []
    for theta in sorted(thetas):
        d = theta / beta
        if any(abs(d - s.d_m) < 1e-9 * lam for s in solutions):
            continue
        y_line = z0 / complex(input_impedance(z0, zl, d, freq, vp))
        if termination == "open":
            theta_stub = math.atan(-y_line.imag) % math.pi
            b_stub = math.tan(theta_stub)
        else:
            theta_stub = math.atan2(1.0, y_line.imag) % math.pi
            b_stub = -1.0 / math.tan(theta_stub)
        length = theta_stub / beta
        y_total = y_line + 1j * b_stub
        error = abs(y_total - 1.0)
        solutions.append(
            StubSolution(
                d_m=d,
                d_lambda=d / lam,
                l_m=length,
                l_lambda=length / lam,
                termination=termination,
                line_admittance_normalized=y_line,
                stub_susceptance_normalized=b_stub,
                y_total_normalized=y_total,
                error=error,
                matched=error < MATCH_TOL,
            )
        )
    return solutions[:2]
