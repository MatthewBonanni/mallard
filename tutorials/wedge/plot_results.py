#!/usr/bin/env python3
"""Results of the wedge tutorial against oblique-shock theory.

    python plot_results.py RUN_DIR [RUN_DIR ...] [--labels A B ...] [--out DIR]

Each RUN_DIR holds a finished run of input.toml (solut/wedge.pvd,
probe_line.csv); the last one is drawn in the field plot. Prints, per run, the
shock angle (fitted through the corner to where the pressure crosses halfway
up the theoretical jump along horizontal lines) and the mean pressure ratio
behind the shock, against theory, and writes the figures wedge_field,
wedge_history and wedge_validation.
"""
import argparse
import math
import os
import sys
import tomllib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import snapshots  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from oblique_shock import inflow_mach, solve  # noqa: E402
from tutfig import label_box, render  # noqa: E402

CORNER, RAMP = 0.5, 8.0


def load(run):
    pts, conn, offs, a = snapshots.read(snapshots.last(os.path.join(run, "solut", "wedge.pvd")))
    polys, c = snapshots.cells(pts, conn, offs)
    return [p[:, :2] for p in polys], c[:, :2], a


def shock_angle(c, p, p1, p2):
    """Angle (degrees) of the line through the corner fitted to the shock points along y = 0.3 .. 1.0."""
    interp = mtri.LinearTriInterpolator(mtri.Triangulation(c[:, 0], c[:, 1]), p)
    mid = 0.5 * (p1 + p2)
    xs, ys = [], []
    for y in np.linspace(0.3, 1.0, 15):
        x = np.linspace(CORNER, 2.0, 3000)
        v = np.asarray(interp(x, np.full_like(x, y)).filled(np.nan))
        i = np.where((v[:-1] < mid) & (v[1:] >= mid))[0]
        if len(i):
            k = i[0]
            xs.append(x[k] + (mid - v[k]) * (x[k + 1] - x[k]) / (v[k + 1] - v[k]))
            ys.append(y)
    dx, dy = np.array(xs) - CORNER, np.array(ys)
    return math.degrees(math.atan((dx @ dy) / (dx @ dx))), np.array(xs), dy


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--out", help="output directory (default: docs/images/tutorials)")
    args = ap.parse_args()
    labels = args.labels or [os.path.basename(os.path.normpath(r)) for r in args.runs]
    mach, gamma = inflow_mach(os.path.join(args.runs[-1], "input.toml"))
    th = solve(mach, RAMP, gamma)
    beta = math.degrees(th["beta"])
    with open(os.path.join(args.runs[-1], "input.toml"), "rb") as f:
        p1 = next(b for b in tomllib.load(f)["boundaries"] if b["name"] == "left")["p"]
    print(f"theory: M = {mach:.4f}, shock angle {beta:.2f} deg, p2/p1 {th['p']:.4f}")

    profiles = []
    for run, label in zip(args.runs, labels):
        _, c, a = load(run)
        b, xs, ys = shock_angle(c, a["P"], p1, th["p"] * p1)
        x, y = c[:, 0], c[:, 1]
        ramp_y = np.maximum(x - CORNER, 0) * math.tan(math.radians(RAMP))
        behind = (x > 0.8) & (y > ramp_y + 0.05) & (y < (x - CORNER) * math.tan(th["beta"]) - 0.08)
        ratio = a["P"][behind].mean() / p1
        print(f"{label}: shock angle {b:.2f} deg ({b - beta:+.2f}), p2/p1 {ratio:.4f} "
              f"({100 * (ratio / th['p'] - 1):+.2f}%)")
        d = np.genfromtxt(os.path.join(run, "probe_line.csv"), delimiter=",", names=True)
        profiles.append((label, d))
    polys, c, a = load(args.runs[-1])
    _, xs, ys = shock_angle(c, a["P"], p1, th["p"] * p1)
    speed = np.hypot(a["U_X"], a["U_Y"])
    R = 101325.0 / (1.225 * 298.15)
    mach_field = speed / np.sqrt(gamma * R * a["T"])

    def field(t):
        fig, ax = plt.subplots(figsize=(8.6, 5.2))
        pc = PolyCollection(polys, array=mach_field, cmap="viridis", edgecolors="none", antialiased=False,
                            rasterized=True)
        ax.add_collection(pc)
        fig.colorbar(pc, ax=ax, shrink=0.8, label="Mach number")
        xx = np.array([CORNER, 2.0])
        ax.plot(xx, (xx - CORNER) * math.tan(th["beta"]), color=t["surface"], lw=1.6, ls="--")
        ax.plot(xs, ys, "o", ms=4, mfc="none", color=t["surface"])
        ax.text(1.0, 1.05, f"theory: β = {beta:.2f}°", color=t["ink"], fontsize=11, bbox=label_box(t))
        ax.plot([1.5, 1.5], [0.15, 1.45], color=t["flame"], lw=1.4)
        ax.text(1.53, 0.2, "probe line", color=t["flame"], fontsize=10, bbox=label_box(t))
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 1.5)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(False)
        return fig

    def history(t):
        label, d = profiles[-1]
        fig, ax = plt.subplots(figsize=(8, 3.6))
        times = np.unique(d["t"])
        pick = [times[np.abs(times - ms * 1e-3).argmin()] for ms in (1, 2, 3, 4, 20)]
        for k, tt in enumerate(pick):
            s = d[d["t"] == tt]
            ax.plot(s["y"], s["P"] / p1, color=t["series"][k % len(t["series"])], label=f"t = {1e3 * tt:.1f} ms")
        ax.set_xlabel("y at x = 1.5 [m]")
        ax.set_ylabel("p / p₁")
        ax.legend(ncol=2, fontsize=10)
        return fig

    def validation(t):
        fig, ax = plt.subplots(figsize=(8, 3.6))
        y_shock = (1.5 - CORNER) * math.tan(th["beta"])
        y = np.linspace(0.15, 1.45, 400)
        ax.plot(y, np.where(y < y_shock, th["p"], 1.0), color=t["ink2"], lw=2, label="theory")
        for k, (label, d) in enumerate(profiles):
            s = d[d["t"] == d["t"].max()]
            ax.plot(s["y"], s["P"] / p1, "o-", ms=3, lw=1, color=t["series"][k], label=label)
        ax.set_xlabel("y at x = 1.5 [m]")
        ax.set_ylabel("p / p₁")
        ax.legend()
        return fig

    render(field, args.out, "wedge_field")
    render(history, args.out, "wedge_history")
    render(validation, args.out, "wedge_validation")


if __name__ == "__main__":
    main()
