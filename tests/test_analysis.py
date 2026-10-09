import math

import numpy as np
import pytest

from tltoolkit import analysis

Z0 = 50.0
F = 1e9


def test_analyze_load_known_values():
    a = analysis.analyze_load(Z0, 100)
    assert a.gamma_mag == pytest.approx(1 / 3)
    assert a.vswr == pytest.approx(2.0)
    assert a.return_loss_db == pytest.approx(9.5424, abs=1e-3)
    assert a.mismatch_loss_db == pytest.approx(0.5115, abs=1e-3)
    assert a.reflected_power_percent == pytest.approx(100 / 9)
    assert a.delivered_power_percent == pytest.approx(800 / 9)


def test_analyze_open_and_short():
    for zl in (0, float("inf")):
        a = analysis.analyze_load(Z0, zl)
        assert a.vswr == math.inf and a.delivered_power_percent == 0.0


def test_recommend_matched_load():
    r = analysis.recommend_matching(Z0, 50, F)
    assert not r.recommendations and "already matched" in r.notes[0]


@pytest.mark.parametrize("zl", [0, float("inf"), 30j])
def test_recommend_unmatchable_loads_explain_why(zl):
    r = analysis.recommend_matching(Z0, zl, F)
    assert not r.recommendations and r.notes


@pytest.mark.parametrize("zl", [100, 25, 100 + 50j, 20 - 35j])
def test_recommend_returns_only_verified_solutions(zl):
    r = analysis.recommend_matching(Z0, zl, F)
    assert [x.method.split(" (")[0] for x in r.recommendations][:2] == [
        "Quarter-wave transformer",
        "L-network",
    ]
    for rec in r.recommendations:
        assert rec.solutions
        assert all(s.matched for s in rec.solutions)


def test_recommend_can_skip_stub():
    r = analysis.recommend_matching(Z0, 100 + 50j, F, include_stub=False)
    assert all("stub" not in x.method.lower() for x in r.recommendations)


def test_sweep_lossless_is_consistent():
    s = analysis.sweep_impedance(Z0, 100 + 50j, F, max_length_lambda=0.5, points=2001)
    assert s.zin.shape == s.distance_m.shape == (2001,)
    assert s.zin[0] == pytest.approx(100 + 50j)
    assert np.allclose(s.gamma_mag, s.gamma_mag[0])
    assert np.allclose(s.vswr, s.vswr[0])
    # standing-wave ratio from the voltage pattern equals the VSWR
    assert s.voltage_magnitude.max() / s.voltage_magnitude.min() == pytest.approx(
        s.vswr[0], rel=1e-3
    )


def test_sweep_lossy_vswr_decreases():
    s = analysis.sweep_impedance(Z0, 200, F, max_length_lambda=5, points=200, alpha=0.2)
    assert s.vswr[-1] < s.vswr[0]


def test_sweep_invalid_arguments():
    with pytest.raises(ValueError):
        analysis.sweep_impedance(Z0, 100, F, points=1)
    with pytest.raises(ValueError):
        analysis.sweep_impedance(Z0, 100, F, max_length_lambda=0)


def test_sweep_csv(tmp_path):
    path = tmp_path / "sweep.csv"
    analysis.sweep_impedance(Z0, 100 + 50j, F, points=11).save_csv(str(path))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 12 and lines[0].startswith("distance_m")
