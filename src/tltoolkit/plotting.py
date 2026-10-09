"""Plots of line sweeps. Requires matplotlib (``pip install transmission-line-toolkit[plot]``)."""

from __future__ import annotations

from typing import Optional

from .analysis import SweepResult

__all__ = ["plot_impedance_sweep", "plot_standing_wave", "plot_sweep"]


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install 'transmission-line-toolkit[plot]'"
        ) from exc
    return plt


def plot_impedance_sweep(sweep: SweepResult, ax) -> None:
    """Draw ``Re(Zin)`` and ``Im(Zin)`` versus distance on ``ax``."""
    ax.plot(sweep.distance_lambda, sweep.zin.real, label="Re(Zin)")
    ax.plot(sweep.distance_lambda, sweep.zin.imag, label="Im(Zin)")
    ax.axhline(sweep.z0, color="gray", linestyle="--", linewidth=0.8, label="Z0")
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_xlabel("Distance from load (wavelengths)")
    ax.set_ylabel("Impedance (Ohm)")
    ax.set_title("Input impedance along the line")
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_standing_wave(sweep: SweepResult, ax) -> None:
    """Draw the voltage standing-wave pattern ``|V(d)|`` on ``ax``."""
    ax.plot(sweep.distance_lambda, sweep.voltage_magnitude, color="tab:green")
    ax.set_xlabel("Distance from load (wavelengths)")
    ax.set_ylabel("|V| (normalised to incident wave)")
    ax.set_title("Voltage standing-wave pattern")
    ax.grid(True, alpha=0.3)


def plot_sweep(sweep: SweepResult, save_path: Optional[str] = None, show: bool = False):
    """Two-panel figure: input impedance and standing-wave pattern.

    Saves to ``save_path`` if given and calls ``plt.show()`` if ``show`` is true.
    Returns the matplotlib ``Figure``.
    """
    plt = _pyplot()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    plot_impedance_sweep(sweep, ax1)
    plot_standing_wave(sweep, ax2)
    ax1.set_xlabel("")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig
