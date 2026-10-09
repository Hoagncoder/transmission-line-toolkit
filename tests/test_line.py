import math

import numpy as np
import pytest

from tltoolkit import line

Z0 = 50.0
F = 1e9
LAM = line.wavelength(F)


def test_gamma_matched_short_open():
    assert line.reflection_coefficient(50, 50) == 0
    assert line.reflection_coefficient(50, 0) == -1
    assert line.reflection_coefficient(50, float("inf")) == 1


def test_gamma_known_value():
    assert line.reflection_coefficient(50, 100) == pytest.approx(1 / 3)


def test_vswr_known_values():
    assert line.vswr_from_impedance(50, 100) == pytest.approx(2.0)
    assert line.vswr_from_impedance(50, 50) == pytest.approx(1.0)
    assert line.vswr_from_impedance(50, 0) == math.inf


def test_return_and_mismatch_loss():
    assert line.return_loss_db(0.1) == pytest.approx(20.0)
    assert line.return_loss_db(0) == math.inf
    assert line.return_loss_db(1) == 0.0
    assert math.copysign(1, line.return_loss_db(1)) == 1.0  # not -0.0
    assert line.mismatch_loss_db(1 / 3) == pytest.approx(-10 * math.log10(8 / 9))
    assert line.mismatch_loss_db(1) == math.inf


def test_wavelength_and_validation():
    assert line.wavelength(1e9, 3e8) == pytest.approx(0.3)
    with pytest.raises(ValueError):
        line.wavelength(0)
    with pytest.raises(ValueError):
        line.wavelength(1e9, -1)


def test_phase_velocity():
    assert line.phase_velocity(0.5) == pytest.approx(line.C0 / 2)
    with pytest.raises(ValueError):
        line.phase_velocity(1.5)


def test_electrical_length():
    assert line.electrical_length_deg(LAM / 4, F) == pytest.approx(90.0)
    full = line.electrical_length_deg(LAM, F)
    assert min(full, 360.0 - full) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("zl", [100, 25 + 30j, 10 - 40j, 0, 1e6])
def test_zin_zero_length_and_half_wave(zl):
    assert line.input_impedance(Z0, zl, 0.0, F) == pytest.approx(complex(zl))
    assert line.input_impedance(Z0, zl, LAM / 2, F) == pytest.approx(complex(zl), abs=1e-4)


@pytest.mark.parametrize("zl", [100, 25 + 30j, 10 - 40j])
def test_zin_quarter_wave_inverts(zl):
    assert line.input_impedance(Z0, zl, LAM / 4, F) == pytest.approx(Z0**2 / complex(zl))


def test_short_and_open_stubs_at_eighth_wave():
    assert line.input_impedance(Z0, 0, LAM / 8, F) == pytest.approx(1j * Z0)
    assert line.input_impedance(Z0, float("inf"), LAM / 8, F) == pytest.approx(-1j * Z0)


def test_open_circuit_at_zero_length_is_infinite():
    assert line.input_impedance(Z0, float("inf"), 0.0, F) == complex(math.inf)


def test_matched_line_is_constant():
    for d in (0.0, 0.123, 1.0):
        assert line.input_impedance(Z0, Z0, d, F) == pytest.approx(Z0)


def test_lossy_line_tends_to_z0():
    zin = line.input_impedance(Z0, 200 + 100j, 100.0, F, alpha=0.1)
    assert zin == pytest.approx(Z0, abs=1e-6)


def test_array_input_matches_scalar():
    d = np.linspace(0, LAM, 7)
    arr = line.input_impedance(Z0, 100 + 50j, d, F)
    for i, di in enumerate(d):
        assert arr[i] == pytest.approx(line.input_impedance(Z0, 100 + 50j, di, F))


def test_invalid_inputs():
    with pytest.raises(ValueError):
        line.input_impedance(Z0, 100, -1.0, F)
    with pytest.raises(ValueError):
        line.input_impedance(-50, 100, 0.1, F)
    with pytest.raises(ValueError):
        line.input_impedance(50 + 5j, 100, 0.1, F)
    with pytest.raises(ValueError):
        line.input_impedance(Z0, -10 + 0j, 0.1, F)
    with pytest.raises(ValueError):
        line.input_impedance(Z0, 100, 0.1, F, alpha=-1)
    with pytest.raises(ValueError):
        line.reflection_coefficient(50, -50)
