#!/usr/bin/env python3
"""Results of the cylinder tutorial against reference computations.

    python plot_results.py RUN_DIR [--t-start 60] [--out DIR]

RUN_DIR holds a run of input.toml (forces_cylinder.csv, solut/cylinder.vtkhdf.series).
From --t-start on, prints the Strouhal number (from the lift's period), the
mean drag coefficient and the lift amplitude against the references, and
writes the figures cylinder_history (drag and lift against time) and
cylinder_field (vorticity of the last snapshot).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import matplotlib.pyplot as plt  # noqa: E402
import snapshots  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from tutfig import render  # noqa: E402

U, D = 0.2, 1.0
Q = 0.5 * U**2 * D  # dynamic pressure times the diameter (density 1)
# Incompressible references at Re = 100 (the validation page lists them)
REF = {"St": (0.164, 0.166), "Cd": (1.33, 1.35), "Cl": (0.33, 0.34)}


def coefficients(run):
    d = np.genfromtxt(os.path.join(run, "forces_cylinder.csv"), delimiter=",", names=True)
    cd = (d["Fx_pressure"] + d["Fx_viscous"]) / Q
    cl = (d["Fy_pressure"] + d["Fy_viscous"]) / Q
    return d["t"], cd, cl


def strouhal(t, cl):
    """Shedding frequency * D / U from the upward zero crossings of the lift."""
    s = cl - cl.mean()
    i = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]
    tc = t[i] - s[i] * (t[i + 1] - t[i]) / (s[i + 1] - s[i])
    if len(tc) < 2:
        return float("nan")  # not shedding (yet)
    return (len(tc) - 1) / (tc[-1] - tc[0]) * D / U


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("--t-start", type=float, default=60.0)
    ap.add_argument("--out", help="output directory (default: docs/images/tutorials)")
    args = ap.parse_args()
    t, cd, cl = coefficients(args.run)
    sel = t >= args.t_start
    st = strouhal(t[sel], cl[sel])
    mean_cd = cd[sel].mean()
    amp = 0.5 * (cl[sel].max() - cl[sel].min())
    print(f"t >= {args.t_start:g}: St {st:.4f}, mean Cd {mean_cd:.4f}, Cl amplitude {amp:.4f} "
          f"(references: St {REF['St'][0]}-{REF['St'][1]}, Cd {REF['Cd'][0]}-{REF['Cd'][1]}, "
          f"Cl {REF['Cl'][0]}-{REF['Cl'][1]})")
    pts, conn, offs, a = snapshots.read(snapshots.last(os.path.join(args.run, "solut", "cylinder")))
    polys, _ = snapshots.cells(pts, conn, offs)
    polys = [p[:, :2] for p in polys]
    w = a["VORTICITY"]

    def history(th):
        fig, (ax, bx) = plt.subplots(2, 1, figsize=(9, 4.8), sharex=True)
        ax.plot(t, cd, color=th["series"][0])
        ax.axhspan(*REF["Cd"], color=th["ink2"], alpha=0.25, lw=0)
        ax.set_ylabel("C_D")
        ax.set_ylim(1.2, 1.5)
        bx.plot(t, cl, color=th["series"][1])
        for sign in (1, -1):
            bx.axhline(sign * sum(REF["Cl"]) / 2, color=th["ink2"], ls=":", lw=1.2)
        bx.axvline(args.t_start, color=th["ink2"], ls="--", lw=1)
        ax.set_title(f"from t = {args.t_start:g}: St = {st:.3f}, mean C_D = {mean_cd:.3f}, C_L' = {amp:.3f} "
                     "(references shaded and dotted)", fontsize=11)
        bx.set_ylabel("C_L")
        bx.set_xlabel("t a / D")
        bx.set_ylim(-0.6, 0.6)
        return fig

    def field(th):
        fig, ax = plt.subplots(figsize=(10, 3.6))
        pc = PolyCollection(polys, array=w, cmap="RdBu_r", edgecolors="none", antialiased=False, rasterized=True)
        pc.set_clim(-0.6, 0.6)
        ax.add_collection(pc)
        fig.colorbar(pc, ax=ax, shrink=0.85, label="vorticity ω D / a")
        ax.add_patch(plt.Circle((0, 0), 0.5, color=th["ink2"]))
        ax.set_xlim(-2, 16)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal")
        ax.set_xlabel("x / D")
        ax.set_ylabel("y / D")
        ax.grid(False)
        return fig

    render(history, args.out, "cylinder_history")
    render(field, args.out, "cylinder_field")


if __name__ == "__main__":
    main()
