"""Basic transmission-line relations.

Conventions
-----------
* Impedances are in ohms, frequencies in hertz, lengths in metres.
* ``z0`` is the characteristic impedance of the line (real and positive).
* ``zl`` is the load impedance. It may be complex, must satisfy ``Re(zl) >= 0``
  (passive load), and ``float("inf")`` represents an open circuit.
* Distances ``d`` are measured from the load towards the source.
* Time convention is ``exp(+j*w*t)`` and the propagation constant is
  ``gamma = alpha + j*beta`` with ``beta = 2*pi*f/vp``.
"""

from __future__ import annotations

import cmath
import math
from typing import Union

import numpy as np

__all__ = [
    "C0",
    "electrical_length_deg",
    "input_impedance",
    "input_impedance_electrical",
    "mismatch_loss_db",
    "phase_velocity",
    "reflection_coefficient",
    "return_loss_db",
    "validate_frequency",
    "validate_load",
    "validate_phase_velocity",
    "validate_z0",
    "vswr_from_gamma",
    "vswr_from_impedance",
    "wavelength",
]

#: Speed of light in vacuum (m/s).
C0 = 299_792_458.0

ArrayLike = Union[float, np.ndarray]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def validate_z0(z0: complex) -> float:
    """Return ``z0`` as a float, or raise ``ValueError`` if it is not real and > 0."""
    z = complex(z0)
    if abs(z.imag) > 1e-12 or not z.real > 0 or math.isinf(z.real):
        raise ValueError("Z0 must be a finite, positive real number (ohms).")
    return z.real


def validate_load(zl: complex) -> complex:
    """Return ``zl`` as a complex number, or raise if it is not a passive load."""
    z = complex(zl)
    if math.isnan(z.real) or math.isnan(z.imag):
        raise ValueError("ZL must not be NaN.")
    if z.real < 0:
        raise ValueError("Re(ZL) must be >= 0 for a passive load.")
    return z


def validate_frequency(freq: float) -> float:
    """Return ``freq`` if it is a finite number > 0 Hz."""
    if not freq > 0 or math.isinf(freq):
        raise ValueError("Frequency must be a finite number > 0 Hz.")
    return float(freq)


def validate_phase_velocity(vp: float) -> float:
    """Return ``vp`` if it is a finite number > 0 m/s."""
    if not vp > 0 or math.isinf(vp):
        raise ValueError("Phase velocity must be a finite number > 0 m/s.")
    return float(vp)


def phase_velocity(velocity_factor: float = 1.0) -> float:
    """Phase velocity (m/s) of a line with the given velocity factor in (0, 1]."""
    if not 0.0 < velocity_factor <= 1.0:
        raise ValueError("velocity_factor must be in the range (0, 1].")
    return C0 * velocity_factor


# ---------------------------------------------------------------------------
# Reflection and mismatch figures of merit
# ---------------------------------------------------------------------------
def reflection_coefficient(z0: float, zl: complex) -> complex:
    """Voltage reflection coefficient ``(ZL - Z0) / (ZL + Z0)``."""
    z0 = validate_z0(z0)
    zl = complex(zl)
    if cmath.isinf(zl):
        return complex(1.0, 0.0)
    den = zl + z0
    if abs(den) < 1e-15:
        raise ValueError("ZL = -Z0 is not a physically meaningful load.")
    return (zl - z0) / den


def vswr_from_gamma(gamma: complex) -> float:
    """VSWR ``(1 + |Gamma|) / (1 - |Gamma|)``; ``inf`` when ``|Gamma| >= 1``."""
    mag = abs(gamma)
    if mag >= 1.0:
        return float("inf")
    return (1.0 + mag) / (1.0 - mag)


def vswr_from_impedance(z0: float, zl: complex) -> float:
    """VSWR produced by load ``zl`` on a line of impedance ``z0``."""
    return vswr_from_gamma(reflection_coefficient(z0, zl))


def return_loss_db(gamma: complex) -> float:
    """Return loss in dB (larger is better); ``inf`` for a perfect match."""
    mag = abs(gamma)
    if mag == 0:
        return float("inf")
    return -20.0 * math.log10(mag) + 0.0  # "+ 0.0" turns -0.0 into 0.0


def mismatch_loss_db(gamma: complex) -> float:
    """Power lost to reflection, ``-10*log10(1 - |Gamma|^2)`` in dB."""
    mag = abs(gamma)
    if mag >= 1.0:
        return float("inf")
    return -10.0 * math.log10(1.0 - mag**2) + 0.0


# ---------------------------------------------------------------------------
# Geometry along the line
# ---------------------------------------------------------------------------
def wavelength(freq: float, vp: float = C0) -> float:
    """Wavelength (m) on a line with phase velocity ``vp`` at ``freq`` Hz."""
    return validate_phase_velocity(vp) / validate_frequency(freq)


def electrical_length_deg(length: float, freq: float, vp: float = C0) -> float:
    """Electrical length in degrees (modulo 360) of a physical ``length`` in metres."""
    beta = 2.0 * math.pi * validate_frequency(freq) / validate_phase_velocity(vp)
    return math.degrees(beta * length) % 360.0


def _zin(z0: float, zl: complex, gl: ArrayLike) -> ArrayLike:
    """``Zin = Z0 (ZL + Z0 tanh(gl)) / (Z0 + ZL tanh(gl))`` for ``gl = gamma * length``."""
    th = np.tanh(np.asarray(gl, dtype=complex))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if cmath.isinf(zl):
            zin = z0 / th
        else:
            zin = z0 * (zl + z0 * th) / (z0 + zl * th)
    if zin.ndim == 0:
        z = complex(zin)
        return complex(float("inf")) if cmath.isnan(z) else z
    return zin


def input_impedance(
    z0: float,
    zl: complex,
    length: ArrayLike,
    freq: float,
    vp: float = C0,
    alpha: float = 0.0,
) -> ArrayLike:
    """Impedance seen looking towards the load from distance ``length`` (m) away.

    ``length`` may be a float or a NumPy array. ``alpha`` is the attenuation
    constant in Np/m (0 for a lossless line). Returns ``complex`` for a scalar
    length and a complex array otherwise.
    """
    z0 = validate_z0(z0)
    zl = validate_load(zl)
    freq = validate_frequency(freq)
    vp = validate_phase_velocity(vp)
    if alpha < 0:
        raise ValueError("alpha (attenuation) must be >= 0 Np/m.")
    length = np.asarray(length, dtype=float)
    if np.any(length < 0):
        raise ValueError("Line length must not be negative.")
    beta = 2.0 * math.pi * freq / vp
    return _zin(z0, zl, (alpha + 1j * beta) * length)


def input_impedance_electrical(z0: float, zl: complex, theta_rad: ArrayLike) -> ArrayLike:
    """Like :func:`input_impedance`, for a lossless line of electrical length ``theta_rad``."""
    return _zin(validate_z0(z0), validate_load(zl), 1j * np.asarray(theta_rad, dtype=float))
