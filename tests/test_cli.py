import pytest

from tltoolkit import cli


@pytest.mark.parametrize(
    "text, expected",
    [
        ("75", 75),
        ("100+50j", 100 + 50j),
        ("100+j50", 100 + 50j),
        ("100 - j25", 100 - 25j),
        ("j50", 50j),
        ("12.5-j3.5", 12.5 - 3.5j),
        ("1e2+j2e1", 100 + 20j),
    ],
)
def test_parse_impedance(text, expected):
    assert cli.parse_impedance(text) == expected


def test_parse_impedance_inf_and_errors():
    assert cli.parse_impedance("inf") == complex(float("inf"))
    with pytest.raises(ValueError):
        cli.parse_impedance("abc")


@pytest.mark.parametrize(
    "text, expected",
    [("1e9", 1e9), ("1GHz", 1e9), ("900 MHz", 9e8), ("2.4ghz", 2.4e9), ("50kHz", 5e4), ("60", 60)],
)
def test_parse_frequency(text, expected):
    assert cli.parse_frequency(text) == pytest.approx(expected)


@pytest.mark.parametrize("bad", ["abc", "-1", "0", "1THzz"])
def test_parse_frequency_errors(bad):
    with pytest.raises(ValueError):
        cli.parse_frequency(bad)


def test_analyze_command(capsys):
    assert cli.main(["analyze", "--zl", "100+50j"]) == 0
    out = capsys.readouterr().out
    assert "VSWR" in out and "Return loss" in out


def test_match_command(capsys):
    assert cli.main(["match", "--zl", "100+j50", "--freq", "1GHz"]) == 0
    out = capsys.readouterr().out
    for text in ("Quarter-wave transformer", "L-network", "Single shunt stub", "-> OK"):
        assert text in out
    assert "FAILED" not in out


def test_match_options(capsys):
    argv = ["match", "--zl", "75", "--freq", "2.4GHz", "--vf", "0.66"]
    argv += ["--stub-termination", "short"]
    assert cli.main(argv) == 0
    assert "short-circuited" in capsys.readouterr().out
    assert cli.main(["match", "--zl", "75", "--freq", "1GHz", "--no-stub"]) == 0
    assert "stub" not in capsys.readouterr().out.lower()


def test_zin_command(capsys):
    assert cli.main(["zin", "--zl", "100", "--freq", "1GHz", "--length-lambda", "0.25"]) == 0
    assert "25.00000" in capsys.readouterr().out


def test_zin_requires_length(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["zin", "--zl", "100", "--freq", "1GHz"])
    assert exc.value.code == 2


def test_invalid_input_gives_exit_code_2(capsys):
    assert cli.main(["analyze", "--zl", "abc"]) == 2
    assert "error:" in capsys.readouterr().err
    assert cli.main(["match", "--zl", "100", "--freq", "-5"]) == 2
    assert cli.main(["analyze", "--z0", "-50", "--zl", "100"]) == 2


def test_sweep_command_csv_and_plot(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    csv_path, png_path = tmp_path / "s.csv", tmp_path / "s.png"
    argv = [
        "sweep", "--zl", "100+50j", "--freq", "1GHz", "--max-lambda", "1",
        "--csv", str(csv_path), "--plot", str(png_path),
    ]  # fmt: skip
    assert cli.main(argv) == 0
    assert csv_path.exists() and png_path.stat().st_size > 1000


def test_interactive():
    # defaults for Z0 and velocity factor; first ZL is a typo, so everything is asked again
    answers = iter(["", "abc", "", "100+j50", "1GHz", ""])
    outputs = []
    assert cli.run_interactive(input_fn=lambda _: next(answers), output_fn=outputs.append) == 0
    assert any("error:" in o for o in outputs)
    assert any("Load analysis" in o for o in outputs)
