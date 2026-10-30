#!/usr/bin/env python3
"""Results of the cone tutorial against the exact Taylor-Maccoll solution.

    python plot_results.py RUN_DIR [RUN_DIR ...] [--labels A B ...] [--out DIR]

Each RUN_DIR holds a finished run of input.toml (solut/, probe_layer.csv,
forces_cone.csv); the last one is drawn in the field plots. Prints, per run,
the shock angle (where the probe line across the layer at x = 0.8 crosses
halfway up the exact pressure jump) and the mean surface
pressure coefficient over 0.3 <= x <= 0.95, against the exact values, and
writes the figures cone_history, cone_field and cone_validation.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import matplotlib.pyplot as plt  # noqa: E402
import snapshots  # noqa: E402
from tutfig import render  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from taylor_maccoll import solve  # noqa: E402

GAMMA, MACH, CONE = 1.4, 2.0, 15.0
P_INF = 1 / GAMMA
Q_INF = 0.5 * MACH**2  # 0.5 rho U^2 with rho = 1, U = M


def load(run, prefix):
    """Cells (node coordinates), centroids and arrays of the last snapshot of series solut/PREFIX."""
    pts, conn, offs, arrays = snapshots.read(snapshots.last(os.path.join(run, "solut", prefix)))
    polys, c = snapshots.cells(pts, conn, offs)
    return polys, c, arrays


def probe(run):
    return np.genfromtxt(os.path.join(run, "probe_layer.csv"), delimiter=",", names=True)


def shock_angle(d, p_shock):
    t = d["t"]
    final = d[t == t.max()]
    theta = np.degrees(np.arctan2(np.hypot(final["y"], final["z"]), final["x"]))
    p = final["P"] / P_INF
    mid = 0.5 * (1 + p_shock)  # halfway up the exact pressure jump
    i = np.where((p[:-1] >= mid) & (p[1:] < mid))[0][-1]
    return theta[i] + (mid - p[i]) * (theta[i + 1] - theta[i]) / (p[i + 1] - p[i])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--out", help="output directory (default: docs/images/tutorials)")
    args = ap.parse_args()
    labels = args.labels or [os.path.basename(os.path.normpath(r)) for r in args.runs]
    beta, tm = solve(MACH, CONE, GAMMA)
    cp_exact = (tm["p"][0] - 1) / (0.5 * GAMMA * MACH**2)
    print(f"Taylor-Maccoll: shock {np.degrees(beta):.3f} deg, surface Cp {cp_exact:.5f}")

    surfaces, probes = [], []
    for run, label in zip(args.runs, labels):
        _, c, a = load(run, "cone")
        cp = (a["P"] - P_INF) / Q_INF
        surfaces.append((c[:, 0], cp))
        d = probe(run)
        probes.append(d)
        sel = (c[:, 0] >= 0.3) & (c[:, 0] <= 0.95)
        b = shock_angle(d, tm["p"][-1])
        print(f"{label}: shock {b:.2f} deg ({b - np.degrees(beta):+.2f}), surface Cp {cp[sel].mean():.5f} "
              f"({100 * (cp[sel].mean() / cp_exact - 1):+.2f}%), spread {cp[sel].std():.5f}")

    vol = load(args.runs[-1], "flow")
    plane = load(args.runs[-1], "plane")

    def history(t):
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 3.6))
        for k, (d, label) in enumerate(zip(probes, labels)):
            for pt, ls in ((5, "-"), (25, "--")):
                s = d[d["point"] == pt]
                theta = np.degrees(np.arctan2(np.hypot(s["y"][0], s["z"][0]), s["x"][0]))
                ax.plot(s["t"], s["P"] / P_INF, color=t["series"][k], ls=ls,
                        label=f"{label}, θ = {theta:.0f}°" if k == len(probes) - 1 else None)
            f = np.genfromtxt(os.path.join(args.runs[k], "forces_cone.csv"), delimiter=",", names=True)
            area = np.pi * np.tan(np.radians(CONE)) ** 2 / 4  # base area of the quarter cone
            bx.plot(f["t"], f["Fx_pressure"] / (Q_INF * area) - P_INF / Q_INF, color=t["series"][k], label=label)
        ax.set_xlabel("t (cone lengths / sound speed)")
        ax.set_ylabel("p / p∞ at two probe points, x = 0.8")
        ax.legend(fontsize=8)
        bx.axhline(cp_exact, color=t["ink2"], lw=1, ls=":", label="Taylor-Maccoll")
        bx.set_xlabel("t")
        bx.set_ylabel("pressure drag coefficient (base area)")
        bx.legend(fontsize=8)
        fig.suptitle("Monitoring: the flow settles after about two passes over the cone (t ≈ 1)", color=t["ink"])
        fig.tight_layout()
        return fig

    def field(t):
        polys, c, a = plane
        pts = np.vstack(polys)
        fig, ax = plt.subplots(figsize=(8, 4.4))
        pc = PolyCollection([q[:, :2] for q in polys], array=a["P"] / P_INF, cmap=t["cmap"], edgecolors="none",
                            linewidths=0, antialiased=False, rasterized=True)
        pc.set_clim(1.0, tm["p"][0] * 1.02)
        ax.add_collection(pc)
        x1 = pts[:, 0].max()
        ax.plot([0, x1], [0, np.tan(beta) * x1], color=t["red"], lw=1.4, ls="--")
        ax.set_xlim(pts[:, 0].min(), x1)
        ax.set_ylim(0, pts[:, 1].max())
        ax.set_aspect("equal")
        ax.grid(False)
        cb = fig.colorbar(pc, ax=ax, shrink=0.8)
        cb.set_label("p / p∞")
        cb.outline.set_edgecolor(t["axis"])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Pressure on the symmetry plane z = 0 ({labels[-1]}); dashed: exact shock")
        return fig

    def validation(t):
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 4))
        _, c, a = vol
        r = np.hypot(c[:, 1], c[:, 2])
        sel = (c[:, 0] > 0.4) & (c[:, 0] < 0.95)
        theta = np.degrees(np.arctan2(r[sel], c[sel, 0]))
        ax.scatter(theta, a["P"][sel] / P_INF, s=1.5, color=t["series"][0], alpha=0.25, linewidths=0,
                   label=f"cells with 0.4 < x < 0.95 ({labels[-1]})", rasterized=True)
        th = np.concatenate([tm["theta"], [np.degrees(beta), np.degrees(beta), 45]])
        pe = np.concatenate([tm["p"], [tm["p"][-1], 1, 1]])
        ax.plot(th, pe, color=t["ink"], lw=1.5, label="Taylor-Maccoll (exact)")
        ax.set_xlim(CONE - 1, 45)
        ax.set_xlabel("θ, angle from the axis (degrees)")
        ax.set_ylabel("p / p∞")
        ax.legend(fontsize=8, markerscale=6, loc="upper right")
        ax.set_title("Conical flow: every cell against its angle")
        for k, ((x, cp), label) in enumerate(zip(surfaces, labels)):
            order = np.argsort(x)
            xb = np.linspace(0, 1, 41)
            idx = np.digitize(x[order], xb)
            mean = [cp[order][idx == i].mean() if np.any(idx == i) else np.nan for i in range(1, len(xb))]
            bx.plot(0.5 * (xb[1:] + xb[:-1]), mean, color=t["series"][k], marker="o", ms=3, lw=1.2, label=label)
        bx.axhline(cp_exact, color=t["ink"], lw=1.5, label=f"Taylor-Maccoll, Cp = {cp_exact:.4f}")
        bx.set_xlabel("x along the cone")
        bx.set_ylabel("surface Cp (averaged around the cone)")
        bx.set_ylim(cp_exact * 0.8, cp_exact * 1.1)
        bx.legend(fontsize=8)
        bx.set_title("Surface pressure")
        fig.tight_layout()
        return fig

    render(history, args.out, "cone_history")
    render(field, args.out, "cone_field")
    render(validation, args.out, "cone_validation")


if __name__ == "__main__":
    main()
