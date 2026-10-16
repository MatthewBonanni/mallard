#!/usr/bin/env python3
"""Results of the airfoil tutorial against the High-Order CFD Workshop.

    python plot_results.py RUN_DIR [RUN_DIR ...] [--labels A B ...] [--out DIR]

Each RUN_DIR holds a run of input.toml (forces_airfoil.csv, solut/); list
them from coarse to fine. Prints, per run, the drag and lift coefficients
averaged over the last 10% of the run and their change over it, against the
workshop's reference values, and writes the figures airfoil_history (drag
and lift over time), airfoil_field (Mach number and streamlines of the last
run), airfoil_surface (pressure and skin-friction coefficients along both
surfaces) and airfoil_convergence (drag and lift against the cell size).

Cp = (p - p_inf) / q_inf at the cells next to the wall; Cf = tau_w / q_inf
with tau_w = mu u_t / d from the tangential velocity u_t of those cells and
the distance d of their centroids from the wall, positive where the flow
near the wall runs toward the trailing edge.
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
from tutfig import label_box, render  # noqa: E402

ALPHA = np.radians(1.0)
MACH, GAMMA, MU = 0.5, 1.4, 1.0e-4
Q = 0.5 * MACH**2  # rho = 1, U = M
P_INF = 1 / GAMMA
# Reference: adjoint-adapted p = 2 DG solutions of the 1st International
# Workshop on High-Order CFD Methods (2012), case C1.3, M = 0.5, alpha = 1 deg,
# Re = 5000, as reported by the University of Michigan XFlow group
CD_REF, CL_REF = 0.055316802, 0.018273615


def coefficients(run):
    d = np.genfromtxt(os.path.join(run, "forces_airfoil.csv"), delimiter=",", names=True)
    fx = d["Fx_pressure"] + d["Fx_viscous"]
    fy = d["Fy_pressure"] + d["Fy_viscous"]
    cd = (fx * np.cos(ALPHA) + fy * np.sin(ALPHA)) / Q
    cl = (-fx * np.sin(ALPHA) + fy * np.cos(ALPHA)) / Q
    cdp = (d["Fx_pressure"] * np.cos(ALPHA) + d["Fy_pressure"] * np.sin(ALPHA)) / Q
    return d["t"] * MACH, cd, cl, cdp  # time in chords per free-stream speed


def wall(run):
    """Cells next to the airfoil: wall-face midpoints, Cp, Cf, upper-surface flags."""
    pts, conn, offs, a = snapshots.read(snapshots.last(os.path.join(run, "solut", "flow.pvd")))
    polys, c = snapshots.cells(pts, conn, offs)
    starts = np.concatenate([[0], offs[:-1]])
    owner = {}
    for k, (s, e) in enumerate(zip(starts, offs)):
        nodes = conn[s:e]
        for i in range(len(nodes)):
            key = tuple(sorted((nodes[i], nodes[(i + 1) % len(nodes)])))
            owner[key] = -1 if key in owner else k
    rows = []
    for (i, j), k in owner.items():
        if k < 0:
            continue
        p0, p1 = pts[i, :2], pts[j, :2]
        mid = 0.5 * (p0 + p1)
        if not (-0.01 <= mid[0] <= 1.01 and abs(mid[1]) < 0.08):
            continue
        t = (p1 - p0) / np.linalg.norm(p1 - p0)
        if t[0] < 0:
            t = -t
        n = np.array([-t[1], t[0]])
        d = np.dot(c[k, :2] - mid, n)
        if d < 0:
            n, d = -n, -d
        u = a["U"][k, :2]
        rows.append((mid[0], mid[1], (a["P"][k] - P_INF) / Q, MU * np.dot(u, t) / d / Q))
    rows = np.array(sorted(rows))
    return rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]


def separation(x, y, cf, upper):
    """First x/c past mid-chord where Cf turns negative on one surface."""
    sel = (y > 0) == upper
    xs, cs = x[sel], cf[sel]
    i = np.where((xs[:-1] > 0.5) & (cs[:-1] > 0) & (cs[1:] <= 0))[0]
    return xs[i[0]] - cs[i[0]] * (xs[i[0] + 1] - xs[i[0]]) / (cs[i[0] + 1] - cs[i[0]]) if i.size else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--out", help="output directory (default: docs/images/tutorials)")
    args = ap.parse_args()
    labels = args.labels or [os.path.basename(os.path.normpath(r)) for r in args.runs]

    histories, finals, walls, cells = [], [], [], []
    for run, label in zip(args.runs, labels):
        t, cd, cl, cdp = coefficients(run)
        last = t >= 0.9 * t[-1]
        finals.append((cd[last].mean(), cl[last].mean(), cdp[last].mean()))
        histories.append((t, cd, cl))
        walls.append(wall(run))
        x, y, cp, cf = walls[-1]
        cells.append(int(np.sum(y > 0)))
        print(f"{label}: cd {finals[-1][0]:.5f} ({100 * (finals[-1][0] / CD_REF - 1):+.2f}%, "
              f"change over the last 10% {np.ptp(cd[last]):.1e}), cl {finals[-1][1]:.5f} "
              f"({100 * (finals[-1][1] / CL_REF - 1):+.1f}%, {np.ptp(cl[last]):.1e}), pressure drag "
              f"{finals[-1][2]:.5f}; separation x/c {separation(x, y, cf, True):.3f} (upper), "
              f"{separation(x, y, cf, False):.3f} (lower); t = {t[-1]:.1f} c/U")

    def history(th):
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 3.8))
        for k, ((t, cd, cl), label) in enumerate(zip(histories, labels)):
            ax.plot(t, cd, color=th["series"][k], lw=1.4, label=label)
            bx.plot(t, cl, color=th["series"][k], lw=1.4, label=label)
        ax.axhline(CD_REF, color=th["ink"], ls="--", lw=1.2, label="workshop reference")
        bx.axhline(CL_REF, color=th["ink"], ls="--", lw=1.2, label="workshop reference")
        ax.set_ylim(0.9 * CD_REF, 1.25 * CD_REF)
        bx.set_ylim(-0.02, 0.06)
        for a, name in ((ax, "drag coefficient"), (bx, "lift coefficient")):
            a.set_xlabel("t U / c")
            a.set_ylabel(name)
        ax.legend(fontsize=10)
        fig.tight_layout()
        return fig

    pts, conn, offs, a = snapshots.read(snapshots.last(os.path.join(args.runs[-1], "solut", "flow.pvd")))
    polys, cent = snapshots.cells(pts, conn, offs)
    mach = np.linalg.norm(a["U"][:, :2], axis=1) / np.sqrt(GAMMA * a["P"] / a["RHO"])

    def field(th):
        from scipy.interpolate import griddata
        fig, axs = plt.subplots(2, 1, figsize=(10, 7.4), gridspec_kw={"height_ratios": [1, 1]})
        views = [((-0.4, 2.2), (-0.45, 0.45)), ((0.7, 1.25), (-0.09, 0.09))]
        for ax, (xl, yl) in zip(axs, views):
            near = (cent[:, 0] > xl[0] - 0.3) & (cent[:, 0] < xl[1] + 0.3) & (np.abs(cent[:, 1]) < yl[1] + 0.3)
            pc = PolyCollection([polys[i][:, :2] for i in np.where(near)[0]], array=mach[near], cmap=th["cmap"],
                                edgecolors="none", linewidths=0, antialiased=False, rasterized=True)
            pc.set_clim(0, 0.65)
            ax.add_collection(pc)
            gx, gy = np.meshgrid(np.linspace(*xl, 300), np.linspace(*yl, 160))
            sel = near
            u = griddata(cent[sel, :2], a["U"][sel, 0], (gx, gy), method="linear")
            v = griddata(cent[sel, :2], a["U"][sel, 1], (gx, gy), method="linear")
            ax.streamplot(gx, gy, u, v, color=th["fg"], linewidth=0.5, density=1.6 if xl[0] > 0 else 1.0,
                          arrowsize=0.6)
            ax.set_xlim(*xl)
            ax.set_ylim(*yl)
            ax.set_aspect("equal")
            ax.grid(False)
        axs[0].set_title(f"Mach number and streamlines ({labels[-1]})")
        axs[1].set_title("Trailing edge: the separated flow near the wall")
        cb = fig.colorbar(pc, ax=axs, shrink=0.6)
        cb.set_label("Mach number")
        cb.outline.set_edgecolor(th["axis"])
        return fig

    def surface(th):
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4))
        for k, ((x, y, cp, cf), label) in enumerate(zip(walls, labels)):
            for upper, ls in ((True, "-"), (False, "--")):
                sel = (y > 0) == upper
                side = "upper" if upper else "lower"
                ax.plot(x[sel], cp[sel], color=th["series"][k], ls=ls, lw=1.3,
                        label=f"{label}, {side}" if k == len(walls) - 1 else None)
                bx.plot(x[sel], cf[sel], color=th["series"][k], ls=ls, lw=1.3,
                        label=label if upper else None)
        ax.invert_yaxis()
        ax.set_xlabel("x / c")
        ax.set_ylabel("Cp")
        ax.legend(fontsize=10)
        ax.set_title("Pressure coefficient")
        bx.axhline(0, color=th["ink"], lw=0.8)
        bx.set_ylim(-0.02, 0.12)
        bx.set_xlabel("x / c")
        bx.set_ylabel("Cf")
        bx.legend(fontsize=10, title="solid: upper, dashed: lower", title_fontsize=9)
        bx.set_title("Skin friction")
        xs = separation(*walls[-1][:2], walls[-1][3], True)
        if np.isfinite(xs):
            bx.annotate(f"separation, x/c = {xs:.2f}", (xs, 0), (0.45, 0.04), color=th["ink2"], fontsize=10,
                        arrowprops={"arrowstyle": "->", "color": th["ink2"]}, bbox=label_box(th))
        fig.tight_layout()
        return fig

    def convergence(th):
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 3.6))
        h = 1 / np.array(cells, dtype=float)
        f = np.array(finals)
        for a, k, ref, name in ((ax, 0, CD_REF, "drag coefficient"), (bx, 1, CL_REF, "lift coefficient")):
            a.plot(h, f[:, k], "o-", color=th["series"][0], ms=7, label="Mallard")
            a.axhline(ref, color=th["ink"], ls="--", lw=1.2, label=f"workshop reference, {ref:.4f}")
            for hi, fi, label in zip(h, f[:, k], labels):
                a.annotate(label, (hi, fi), (6, 6), textcoords="offset points", fontsize=10, color=th["ink2"])
            a.set_xlim(0, 1.15 * h.max())
            a.set_xlabel("1 / (faces on each surface)")
            a.set_ylabel(name)
            a.legend(fontsize=10, loc="lower right")
        fig.tight_layout()
        return fig

    render(history, args.out, "airfoil_history")
    render(field, args.out, "airfoil_field")
    render(surface, args.out, "airfoil_surface")
    render(convergence, args.out, "airfoil_convergence")


if __name__ == "__main__":
    main()
