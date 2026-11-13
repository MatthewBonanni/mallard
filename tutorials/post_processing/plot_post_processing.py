#!/usr/bin/env python3
"""Figures of the from-run-to-figure tutorial, read with the Python package.

    python plot_post_processing.py RUN_DIR... [--out DIR]

Each RUN_DIR holds a run of input.toml on one mesh, post-processed: the
probes of probes.toml by mallard post extract in extract/, the averages of
its last snapshots by mallard post average in average/, and in the finest
run the Q-criterion by mallard post derive in derived/. Prints, for each run,
its cells, the drag coefficient from [[forces]] and from the momentum flowing
out through the circle of radius 5, the separation angle, the length of the
recirculation and the largest RMS of U_X over the averaged snapshots; then
runs mallard post convergence on the runs. Writes the figures
post_processing_wall (wall pressure and skin friction against the angle, and
U_X along the wake axis), post_processing_field (U_X, the Q-criterion and the
RMS of U_X of the finest run) and post_processing_convergence (the drag
coefficient's history and its convergence with the mesh).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import mallard  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import LogNorm, Normalize  # noqa: E402
from tutfig import render  # noqa: E402

U, D = 0.2, 1.0
Q = 0.5 * U**2 * D  # dynamic pressure times the diameter (density 1)
DRAG = ["forces_cylinder:Fx_pressure", "forces_cylinder:Fx_viscous"]
# Steady flow at Re = 40, incompressible: Dennis & Chang (1970), Fornberg (1980),
# Taira & Colonius (2007); Coutanceau & Bouard (1977) measured
REFERENCES = {"C_D": (1.50, 1.54), "L / D": (2.13, 2.35), "separation angle [deg]": (53.7, 55.6)}


def crossing(x, f):
    """Where f first crosses zero upward, interpolated linearly; NaN if it does not."""
    i = np.where((f[:-1] < 0) & (f[1:] >= 0))[0]
    if len(i) == 0:
        return float("nan")
    i = i[0]
    return x[i] - f[i] * (x[i + 1] - x[i]) / (f[i + 1] - f[i])


def cells(f):
    g = f.geometry()
    return [g.points[g.connectivity[a:b], :2] for a, b in zip(g.offsets[:-1], g.offsets[1:])]


class Run:
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(os.path.normpath(path))
        forces = mallard.open(os.path.join(path, "forces_cylinder.csv"))
        self.t = forces.read_all("t")
        self.cd = (forces.read_all("Fx_pressure") + forces.read_all("Fx_viscous")) / Q
        snapshots = mallard.open(os.path.join(path, "solut", "wake.vtkhdf"))
        self.cells = snapshots.mesh["cells"]

        wall = mallard.open(os.path.join(path, "extract", "probe_wall.csv"))
        x, y = wall.read("x"), wall.read("y")
        upper = y >= 0
        # The angle from the front stagnation point, along the upper half
        theta = np.degrees(np.arctan2(y, -x))[upper]
        order = np.argsort(theta)
        self.theta = theta[order]
        self.cp = wall.read("CP")[upper][order]
        self.cf = wall.read("CF")[upper][order]
        # CF is signed along the free stream: it turns negative where the flow separates
        self.separation = 180 - crossing(self.theta, -self.cf)

        axis = mallard.open(os.path.join(path, "extract", "probe_axis.csv"))
        self.x = axis.read("x")
        self.u = axis.read("U_X") / U
        self.length = crossing(self.x, self.u) - 0.5 * D

        contour = mallard.open(os.path.join(path, "extract", "probe_contour_reductions.csv"))
        self.cd_contour = -contour.read("integral_MOMENTUM_FLUX_X")[0] / Q

        average = mallard.open(os.path.join(path, "average", "average.vtkhdf"))
        self.rms = average.read("RMS_U_X") / U


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", help="output directory (default: docs/images/tutorials)")
    args = ap.parse_args()
    runs = sorted((Run(p) for p in args.runs), key=lambda r: r.cells)
    fine = runs[-1]

    print(f"{'run':8} {'cells':>7} {'C_D':>7} {'C_D circle':>10} {'L / D':>6} {'separation':>10} {'RMS U_X / U':>11}")
    for r in runs:
        print(f"{r.name:8} {r.cells:7,} {r.cd[-1]:7.4f} {r.cd_contour:10.4f} {r.length:6.3f} "
              f"{r.separation:9.2f}° {r.rms.max():11.2e}")
    for name, (lo, hi) in REFERENCES.items():
        print(f"reference {name}: {lo} to {hi}")

    study = None
    if len(runs) >= 2:
        study = mallard.post.convergence([r.path for r in runs], quantity=DRAG)
        for q in study["quantities"]:
            print(f"{q['name']}: {q['verdict']}")

    q_snap = mallard.open(os.path.join(fine.path, "derived", "derived.vtkhdf"))
    u_snap = mallard.open(os.path.join(fine.path, "solut", "wake.vtkhdf"))
    polys = cells(u_snap)
    centers = u_snap.geometry().vertex_averages()
    ux = u_snap.read("U")[:, 0] / U
    qc = q_snap.read("Q_CRITERION") * (D / U) ** 2

    def wall(th):
        fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(12, 3.6))
        for k, r in enumerate(runs):
            label = f"{r.name}, {r.cells:,} cells"
            ax.plot(r.theta, r.cp, color=th["series"][k], label=label)
            bx.plot(r.theta, r.cf, color=th["series"][k], label=label)
            cx.plot(r.x, r.u, color=th["series"][k], label=label)
        ax.set_xlabel("angle from the front [deg]")
        ax.set_ylabel("C_p")
        ax.set_title("wall pressure", fontsize=11)
        bx.axhline(0, color=th["ink2"], lw=0.8)
        bx.axvline(180 - fine.separation, color=th["ink2"], ls="--", lw=1)
        bx.set_xlabel("angle from the front [deg]")
        bx.set_ylabel("C_f")
        bx.set_title(f"skin friction: separation {fine.separation:.1f}° from the back", fontsize=11)
        cx.axhline(0, color=th["ink2"], lw=0.8)
        cx.set_xlabel("x / D")
        cx.set_ylabel("U_X / U on the axis")
        cx.set_title(f"wake axis: L / D = {fine.length:.2f}", fontsize=11)
        cx.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        return fig

    def field(th):
        fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(15, 3.4), sharey=True)
        for a, values, cmap, norm, label in ((ax, ux, "viridis", Normalize(-0.2, 1.3), "U_X / U"),
                                             (bx, qc, "PuOr_r", Normalize(-0.1, 0.1), "Q D² / U²"),
                                             (cx, fine.rms, "magma", LogNorm(1e-4, 1e-1), "RMS of U_X / U")):
            pc = PolyCollection(polys, array=values, cmap=cmap, norm=norm, edgecolors="none", antialiased=False,
                                rasterized=True)
            a.add_collection(pc)
            a.add_patch(plt.Circle((0, 0), 0.5, color=th["ink2"]))
            a.set_xlim(-1.5, 4.5)
            a.set_ylim(-1.5, 1.5)
            a.set_aspect("equal")
            a.grid(False)
            a.set_xlabel("x / D")
            fig.colorbar(pc, ax=a, shrink=0.8, label=label)
        near = (np.abs(centers[:, 0] - 1.5) < 3.5) & (np.abs(centers[:, 1]) < 2)
        ax.tricontour(centers[near, 0], centers[near, 1], ux[near], levels=[0], colors=[th["surface"]],
                      linewidths=1.2)
        ax.set_ylabel("y / D")
        ax.set_title(f"{fine.name}: U_X, the recirculation outlined", fontsize=11)
        bx.set_title("Q-criterion: mallard post derive", fontsize=11)
        cx.set_title("RMS of U_X: mallard post average", fontsize=11)
        return fig

    def convergence(th):
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 3.6))
        for k, r in enumerate(runs):
            ax.plot(r.t, r.cd, color=th["series"][k], label=f"{r.name}, {r.cells:,} cells")
        ax.set_ylim(1.4, 2.0)
        ax.set_xlabel("t a / D")
        ax.set_ylabel("C_D")
        ax.set_title("drag coefficient while the wake settles", fontsize=11)
        ax.legend(loc="upper right", fontsize=8)
        h = np.array([r.cells ** -0.5 for r in runs]) / fine.cells ** -0.5
        bx.plot(h, [r.cd[-1] for r in runs], "o-", color=th["series"][0], label="runs")
        if study:
            parts = {q["name"]: q for q in study["quantities"]}
            fine_cd = sum(parts[n]["values"][-1] for n in DRAG) / Q
            extrapolated = sum(parts[n]["estimate"]["extrapolated"] for n in DRAG) / Q
            band = sum(parts[n]["estimate"]["gci"] * abs(parts[n]["values"][-1]) for n in DRAG) / Q
            bx.plot([0], [extrapolated], "s", color=th["series"][1], label="extrapolated")
            bx.errorbar([1], [fine_cd], yerr=band, color=th["series"][1], capsize=4, label="fine run's GCI band")
        lo, hi = REFERENCES["C_D"]
        bx.axhspan(lo, hi, color=th["grid"], label="incompressible references")
        bx.set_xlim(-0.1, h.max() + 0.2)
        bx.set_xlabel("h / h_fine")
        bx.set_ylabel("C_D")
        bx.set_title("mallard post convergence", fontsize=11)
        bx.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        return fig

    render(wall, args.out, "post_processing_wall")
    render(field, args.out, "post_processing_field")
    render(convergence, args.out, "post_processing_convergence")


if __name__ == "__main__":
    main()
