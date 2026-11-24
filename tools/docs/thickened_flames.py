"""Figure of the thickened flame of non-premixed and partially premixed flames
(docs/guide/les.md, sensor = "reaction"): the flame table of
examples/triple_flame/tfles (tools/setup/flame_table.py: Cantera flames of
H2/N2 against air) and what the model takes from it, with the formulas of
src/physics/tfles.h:

- tfles_flame_table: s_L, delta_L and omega_max against the mixture fraction,
  the flammable range, and the largest thickening F_max = max(1, n_res Delta /
  delta_L) on the example's mesh (Delta = V^(1/2), the 2D default filter width).

    python tools/docs/thickened_flames.py
"""
import csv
import tomllib
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import figstyle
from figstyle import label

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "triple_flame" / "tfles"
N_RES = 5  # ThickenedFlame's default, which the example keeps


def flame_table():
    rows = [r for r in csv.DictReader(line for line in (EXAMPLE / "h2n2_air.csv").open() if not line.startswith("#"))]
    return {k: np.array([float(r[k]) for r in rows]) for k in ("Z", "s_L", "delta_L", "omega_max", "phi")}


def cell_size():
    mesh = tomllib.loads((EXAMPLE / "input.toml").read_text())["mesh"]
    return (mesh["Lx"] / mesh["Nx"] * mesh["Ly"] / mesh["Ny"]) ** 0.5  # filter_width = "volume", 2D


def tfles_flame_table(P):
    t = flame_table()
    Z, Z_st = t["Z"], float(np.interp(1.0, t["phi"], t["Z"]))
    delta = cell_size()
    F_max = np.maximum(1.0, N_RES * delta / t["delta_L"])
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), sharex=True, gridspec_kw=dict(wspace=0.32))
    curves = [(t["s_L"], r"$s_L$ (m/s)", "Laminar flame speed", P["flame"]),
              (1e3 * t["delta_L"], r"$\delta_L$ (mm)", "Thermal thickness", P["teal"]),
              (F_max, r"$F_{max}$", rf"Thickening, $\Delta$ = {1e3 * delta:.3f} mm", P["plum"])]
    for ax, (y, ylabel, title, color) in zip(axes, curves):
        for lo, hi in ((0, Z[0]), (Z[-1], 1)):
            ax.axvspan(lo, hi, color=P["faint"], lw=0)
        ax.plot(Z, y, color=color, lw=2)
        ax.axvline(Z_st, color=P["muted"], ls=":", lw=1.2)
        ax.set_xlim(0, 0.8)
        ax.set_ylim(0, None)
        ax.set_xlabel("mixture fraction Z")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=13)
    top = axes[0].get_ylim()[1]
    label(axes[0], Z_st + 0.01, 0.06 * top, r"$Z_{st}$", P, size=12, ha="left")
    top = axes[2].get_ylim()[1]
    label(axes[2], 0.5 * (Z[0] + Z[-1]), 0.12 * top, rf"$n_{{res}}$ = {N_RES}", P, size=12, color=P["muted"])
    return fig


FIGURES = {"tfles_flame_table": tfles_flame_table}


def figures(names=None):
    for name, draw in FIGURES.items():
        if not names or name in names:
            figstyle.save(name, draw)


if __name__ == "__main__":
    import sys
    figures(sys.argv[1:])
