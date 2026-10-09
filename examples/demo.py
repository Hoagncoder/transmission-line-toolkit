"""Demo: analyse a load, print matching designs and save a sweep plot.

Run from the repository root:  python examples/demo.py
"""

from pathlib import Path

from tltoolkit import recommend_matching, sweep_impedance
from tltoolkit.reporting import format_report

Z0 = 50.0
ZL = 100.0 + 50.0j
FREQ = 1e9

report = recommend_matching(Z0, ZL, FREQ)
print(format_report(report))

try:
    from tltoolkit.plotting import plot_sweep
except ImportError:
    print("\n(matplotlib not installed - skipping the plot)")
else:
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    sweep = sweep_impedance(Z0, ZL, FREQ, max_length_lambda=1.0)
    plot_sweep(sweep, save_path=str(out / "sweep.png"))
    print(f"\nSaved plot to {out / 'sweep.png'}")
