"""Command-line interface: ``tlmatch`` (or ``python -m tltoolkit``)."""

from __future__ import annotations

import argparse
import re
import sys
from typing import Callable, List, Optional

from . import __version__
from .analysis import analyze_load, recommend_matching, sweep_impedance
from .line import (
    electrical_length_deg,
    input_impedance,
    phase_velocity,
    reflection_coefficient,
    vswr_from_gamma,
    wavelength,
)
from .reporting import format_analysis, format_complex, format_report, format_sweep_summary

__all__ = ["main", "parse_frequency", "parse_impedance"]

_REACTIVE_FIRST = re.compile(r"([+-]?)j(\d+\.?\d*(?:e[+-]?\d+)?)")
_FREQ = re.compile(r"^([0-9.]+(?:e[+-]?\d+)?)(hz|khz|mhz|ghz|thz)?$")
_FREQ_UNITS = {None: 1.0, "hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9, "thz": 1e12}


def parse_impedance(text: str) -> complex:
    """Parse ``75``, ``100+50j``, ``100+j50``, ``100 - j25``, ``j50`` or ``inf``."""
    s = text.strip().lower().replace(" ", "")
    if s in ("inf", "+inf", "open"):
        return complex(float("inf"))
    converted = _REACTIVE_FIRST.sub(r"\1\2j", s)
    try:
        return complex(converted)
    except ValueError:
        raise ValueError(
            f"cannot parse impedance {text!r}; use forms like 75, 100+50j or 100+j50"
        ) from None


def parse_frequency(text: str) -> float:
    """Parse ``1e9``, ``900MHz``, ``2.4 GHz`` ... into hertz."""
    s = text.strip().lower().replace(" ", "")
    m = _FREQ.match(s)
    if not m:
        raise ValueError(f"cannot parse frequency {text!r}; use forms like 1e9, 900MHz or 2.4GHz")
    value = float(m.group(1)) * _FREQ_UNITS[m.group(2)]
    if value <= 0:
        raise ValueError("frequency must be > 0")
    return value


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------
def _vp(args: argparse.Namespace) -> float:
    return phase_velocity(args.vf)


def _cmd_analyze(args: argparse.Namespace) -> int:
    z0, zl = parse_impedance(args.z0).real, parse_impedance(args.zl)
    print(format_analysis(analyze_load(z0, zl)))
    return 0


def _cmd_match(args: argparse.Namespace) -> int:
    z0, zl = parse_impedance(args.z0).real, parse_impedance(args.zl)
    report = recommend_matching(
        z0,
        zl,
        parse_frequency(args.freq),
        vp=_vp(args),
        include_stub=not args.no_stub,
        stub_termination=args.stub_termination,
    )
    print(format_report(report))
    return 0


def _cmd_zin(args: argparse.Namespace) -> int:
    z0, zl = parse_impedance(args.z0).real, parse_impedance(args.zl)
    freq, vp = parse_frequency(args.freq), _vp(args)
    lam = wavelength(freq, vp)
    length = args.length_m if args.length_m is not None else args.length_lambda * lam
    zin = complex(input_impedance(z0, zl, length, freq, vp, args.alpha))
    gamma = reflection_coefficient(z0, zin)
    print(f"Wavelength        : {lam:.6f} m")
    print(f"Line length       : {length:.6f} m = {length / lam:.5f} lambda")
    print(f"Electrical length : {electrical_length_deg(length, freq, vp):.3f} deg")
    print(f"Zin               : {format_complex(zin, 5)} Ohm")
    print(f"|Gamma| at input  : {abs(gamma):.5f}   VSWR: {vswr_from_gamma(gamma):.4f}")
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    z0, zl = parse_impedance(args.z0).real, parse_impedance(args.zl)
    sweep = sweep_impedance(
        z0,
        zl,
        parse_frequency(args.freq),
        max_length_lambda=args.max_lambda,
        points=args.points,
        vp=_vp(args),
        alpha=args.alpha,
    )
    print(format_sweep_summary(sweep))
    if args.csv:
        sweep.save_csv(args.csv)
        print(f"Saved CSV  : {args.csv}")
    if args.plot or args.show:
        from .plotting import plot_sweep

        plot_sweep(sweep, save_path=args.plot, show=args.show)
        if args.plot:
            print(f"Saved plot : {args.plot}")
    return 0


def run_interactive(
    input_fn: Callable[[str], str] = input, output_fn: Callable[[str], None] = print
) -> int:
    """Prompt for the parameters one by one, then run the matching report."""

    def ask(prompt: str, default: Optional[str] = None) -> str:
        suffix = f" [{default}]" if default is not None else ""
        while True:
            answer = input_fn(f"{prompt}{suffix}: ").strip()
            if answer:
                return answer
            if default is not None:
                return default

    while True:
        try:
            z0 = parse_impedance(ask("Characteristic impedance Z0 (Ohm)", "50")).real
            zl = parse_impedance(ask("Load impedance ZL (e.g. 100+50j)"))
            freq = parse_frequency(ask("Frequency (e.g. 1GHz)", "1GHz"))
            vf = float(ask("Velocity factor (0-1]", "1.0"))
            report = recommend_matching(z0, zl, freq, vp=phase_velocity(vf))
        except ValueError as exc:
            output_fn(f"error: {exc} -- please try again.")
            continue
        break
    output_fn("")
    output_fn(format_report(report))
    return 0


def _cmd_interactive(args: argparse.Namespace) -> int:
    try:
        return run_interactive()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    load = argparse.ArgumentParser(add_help=False)
    load.add_argument("--z0", default="50", help="line impedance in ohms (default: 50)")
    load.add_argument("--zl", required=True, help="load impedance, e.g. 75, 100+50j, 100+j50")

    line = argparse.ArgumentParser(add_help=False)
    line.add_argument("--freq", required=True, help="frequency, e.g. 1e9, 900MHz, 2.4GHz")
    line.add_argument("--vf", type=float, default=1.0, help="velocity factor in (0, 1] (default 1)")

    parser = argparse.ArgumentParser(
        prog="tlmatch",
        description="Transmission-line analysis and impedance matching.",
        epilog="Tip: for a negative leading sign write --zl=-j50 (with '=').",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    p = sub.add_parser("analyze", parents=[load], help="reflection coefficient, VSWR, losses")
    p.set_defaults(func=_cmd_analyze)

    p = sub.add_parser("match", parents=[load, line], help="design matching networks")
    p.add_argument("--no-stub", action="store_true", help="skip single-stub designs")
    p.add_argument("--stub-termination", choices=("open", "short"), default="open")
    p.set_defaults(func=_cmd_match)

    p = sub.add_parser("zin", parents=[load, line], help="input impedance at a given length")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--length-m", type=float, help="line length in metres")
    group.add_argument("--length-lambda", type=float, help="line length in wavelengths")
    p.add_argument("--alpha", type=float, default=0.0, help="attenuation in Np/m (default 0)")
    p.set_defaults(func=_cmd_zin)

    p = sub.add_parser("sweep", parents=[load, line], help="sweep Zin and |V| along the line")
    p.add_argument("--max-lambda", type=float, default=0.5, help="sweep length in wavelengths")
    p.add_argument("--points", type=int, default=501)
    p.add_argument("--alpha", type=float, default=0.0, help="attenuation in Np/m (default 0)")
    p.add_argument("--csv", metavar="FILE", help="save the sweep as CSV")
    p.add_argument("--plot", metavar="FILE", help="save a figure (needs matplotlib)")
    p.add_argument("--show", action="store_true", help="open the figure window")
    p.set_defaults(func=_cmd_sweep)

    p = sub.add_parser("interactive", help="enter the parameters step by step")
    p.set_defaults(func=_cmd_interactive)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point. Returns a process exit code (0 ok, 2 invalid input)."""
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
