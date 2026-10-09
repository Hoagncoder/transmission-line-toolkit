import math

import pytest

from tltoolkit import matching
from tltoolkit.line import input_impedance, wavelength

Z0 = 50.0
F = 1e9
W = 2 * math.pi * F
COMPLEX_LOADS = [100 + 50j, 25 - 30j, 10 + 80j, 200 - 150j, 50 + 40j, 50 - 40j, 5 + 2j, 500 + 300j]
REAL_LOADS = [100, 25, 200, 5, 1000]
ALL_LOADS = COMPLEX_LOADS + REAL_LOADS + [50]


# ---------------------------------------------------------------- quarter-wave
@pytest.mark.parametrize("r", REAL_LOADS)
def test_quarter_wave_real_load(r):
    res = matching.quarter_wave_transformer(Z0, r)
    assert res.z1_ohm == pytest.approx(math.sqrt(Z0 * r))
    assert res.matched and res.zin_verified == pytest.approx(Z0)


def test_quarter_wave_rejects_complex_or_nonpositive():
    with pytest.raises(ValueError):
        matching.quarter_wave_transformer(Z0, 100 + 10j)
    with pytest.raises(ValueError):
        matching.quarter_wave_transformer(Z0, 0)


@pytest.mark.parametrize("zl", COMPLEX_LOADS)
def test_real_impedance_points_are_real(zl):
    points = matching.real_impedance_points(Z0, zl, F)
    assert len(points) == 2
    assert points == sorted(points, key=lambda p: p.distance_m)
    lam = wavelength(F)
    for p in points:
        assert 0 <= p.distance_m < lam / 2
        zin = input_impedance(Z0, zl, p.distance_m, F)
        assert zin.imag == pytest.approx(0.0, abs=1e-6)
        assert zin.real == pytest.approx(p.resistance_ohm)
        assert p.resistance_ohm > 0


def test_real_impedance_points_for_real_load():
    points = matching.real_impedance_points(Z0, 100, F)
    assert [p.distance_lambda for p in points] == [0.0, 0.25]
    assert points[1].resistance_ohm == pytest.approx(25.0)


@pytest.mark.parametrize("zl", COMPLEX_LOADS)
def test_quarter_wave_after_real_point(zl):
    for res in matching.quarter_wave_after_real_point(Z0, zl, F):
        assert res.matched
        assert res.distance_to_load_m is not None
        # independent check: line to the real point, then the transformer
        r = input_impedance(Z0, zl, res.distance_to_load_m, F).real
        assert res.z1_ohm == pytest.approx(math.sqrt(Z0 * r))


# ------------------------------------------------------------------ L-network
def _series_z(comp):
    if comp is None:
        return 0j
    return 1j * W * comp.value if comp.kind == "inductor" else 1 / (1j * W * comp.value)


def _shunt_y(comp):
    if comp is None:
        return 0j
    return 1 / (1j * W * comp.value) if comp.kind == "inductor" else 1j * W * comp.value


@pytest.mark.parametrize("zl", ALL_LOADS)
def test_l_network_all_solutions_match(zl):
    solutions = matching.l_network_match(Z0, zl, F)
    assert solutions, "at least one solution must exist"
    for s in solutions:
        assert s.matched, f"{s.topology} failed for {zl}: error {s.error_ohm}"
        zl_c = complex(zl)
        zs, ys = _series_z(s.series_component), _shunt_y(s.shunt_component)
        # independent rebuild of Zin from the *component values*
        if s.topology is matching.Topology.SHUNT_AT_LOAD:
            zin = 1 / (1 / zl_c + ys) + zs
        elif s.topology is matching.Topology.SERIES_AT_LOAD:
            zin = 1 / (1 / (zl_c + zs) + ys)
        else:
            zin = zl_c + zs
        assert zin == pytest.approx(Z0, abs=1e-6)


def test_l_network_already_matched():
    (s,) = matching.l_network_match(Z0, 50, F)
    assert s.topology is matching.Topology.NONE and s.matched


def test_l_network_equal_resistance_uses_series_element():
    solutions = matching.l_network_match(Z0, 50 + 30j, F)
    assert any(s.topology is matching.Topology.SERIES_ONLY for s in solutions)


@pytest.mark.parametrize("bad", [0, 10j, float("inf")])
def test_matching_rejects_unmatchable_loads(bad):
    with pytest.raises(ValueError):
        matching.l_network_match(Z0, bad, F)
    with pytest.raises(ValueError):
        matching.single_stub_match(Z0, bad, F)


def test_component_conversion():
    assert matching.reactance_to_component(10.0, W).kind == "inductor"
    assert matching.reactance_to_component(-10.0, W).kind == "capacitor"
    assert matching.susceptance_to_component(0.01, W).kind == "capacitor"
    assert matching.susceptance_to_component(-0.01, W).kind == "inductor"
    assert matching.reactance_to_component(0.0, W) is None


# ------------------------------------------------------------------ single stub
@pytest.mark.parametrize("termination", ["open", "short"])
@pytest.mark.parametrize("zl", COMPLEX_LOADS + REAL_LOADS)
def test_single_stub_matches(zl, termination):
    solutions = matching.single_stub_match(Z0, zl, F, termination=termination)
    assert solutions
    for s in solutions:
        assert s.matched
        # independent check using real open/short stub impedances
        y_line = 1 / input_impedance(Z0, zl, s.d_m, F)
        stub_load = float("inf") if termination == "open" else 0.0
        y_stub = 1 / input_impedance(Z0, stub_load, s.l_m, F)
        assert (y_line + y_stub) * Z0 == pytest.approx(1.0, abs=1e-6)


def test_single_stub_known_values():
    # ZL = 100 + j50 on 50 Ohm: classic textbook geometry (d ~ 0.1988 / 0.0369 lambda region)
    solutions = matching.single_stub_match(Z0, 100 + 50j, F, termination="short")
    assert len(solutions) == 2
    assert all(0 <= s.d_lambda < 0.5 and 0 < s.l_lambda < 0.5 for s in solutions)


def test_single_stub_matched_load_and_bad_termination():
    assert matching.single_stub_match(Z0, 50, F) == []
    with pytest.raises(ValueError):
        matching.single_stub_match(Z0, 100, F, termination="loaded")


def test_single_stub_velocity_factor_scales_lengths():
    fast = matching.single_stub_match(Z0, 100 + 50j, F)
    slow = matching.single_stub_match(Z0, 100 + 50j, F, vp=0.5 * 299_792_458.0)
    assert slow[0].d_lambda == pytest.approx(fast[0].d_lambda)
    assert slow[0].d_m == pytest.approx(0.5 * fast[0].d_m)
