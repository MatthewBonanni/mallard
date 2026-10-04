"""Outcome of flame-vortex interaction runs (tools/flame_vortex.py) on the spectral diagram.

    python tools/flame_vortex_analysis.py RUN_DIR [RUN_DIR ...] [--mechanism mechanisms/h2o2.yaml]
        [--phase ohmech] [--plot OUT.png] [--csv OUT.csv]

For each output of RUN_DIR/solut/flame.pvd (the half channel y > 0):
  Q      the heat release over that of the planar flame, int HRR dA / (L_y int
         HRR_planar dx) (Cantera's planar flame, RUN_DIR/planar_full.csv);
  L      the length of the flame, the isotherm T_u + (T_b - T_u) / 2, over the
         channel height;
  q_min  the weakest burning along that isotherm: at each of its points the
         largest heat release within delta_L / 2, over the planar flame's peak,
         and the smallest of these along the isotherm (1 for a planar flame);
  pocket whether fresh gas (T below the isotherm) has been cut off from the
         inlet.
Each run is then classified as Poinsot, Veynante & Candel (1991) do:
  quenched  the front is locally extinguished, q_min < 0.1;
  pocket    a pocket of fresh gas is cut off, without quenching;
  wrinkled  Q or L changes by 5% or more;
  no effect the total heat release changes by less than 5% (below their
            cut-off limit).
Prints a table per run and the outcomes; --plot draws the runs on the
spectral diagram (u' / S_L against r / delta_L) with the line Ka(r) =
(u' / r) / (S_L / delta_L) = 1 that bounds Poinsot et al.'s quenching zone
for large vortices, and the histories of Q and q_min.
"""
import argparse
import os
import re
import sys

import cantera as ct
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triple_flame_analysis import grid  # noqa: E402

OUTCOMES = ["no effect", "wrinkled", "pocket", "quenched"]
MARKERS = {"no effect": ("o", "#9aa0a6"), "wrinkled": ("s", "#4fc3f7"), "pocket": ("D", "#ffb74d"),
           "quenched": ("X", "#ef5350")}


def planar(run_dir, mechanism, phase):
    d = np.loadtxt(os.path.join(run_dir, "planar_full.csv"), delimiter=",", skiprows=1)
    gas = ct.Solution(mechanism, phase)
    hrr = np.empty(d.shape[0])
    for i, row in enumerate(d):
        gas.TDY = row[2], row[3], row[4:]
        hrr[i] = gas.heat_release_rate
    x, T = d[:, 0], d[:, 2]
    return dict(I=np.trapezoid(hrr, x), peak=hrr.max(), S_L=d[0, 1], T_u=T[0], T_b=T[-1],
                delta=(T[-1] - T[0]) / np.gradient(T, x).max())


def isotherm(xs, ys, T, level):
    """Points of the isotherm as a list of (n, 2) arrays (matplotlib's contouring)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure()
    cs = plt.contour(xs, ys, T.T, levels=[level])
    segs = [s for s in cs.allsegs[0] if len(s) > 1]
    plt.close(fig)
    return segs


def frame_metrics(path, p):
    t, xs, ys, a = grid(path, ["T", "HRR"])
    dx = xs[1] - xs[0]
    H = ys.size * dx
    Q = a["HRR"].sum() * dx * dx / (H * p["I"])
    T_mid = p["T_u"] + 0.5 * (p["T_b"] - p["T_u"])
    segs = isotherm(xs, ys, a["T"], T_mid)
    L = sum(np.hypot(*np.diff(s, axis=0).T).sum() for s in segs) / H
    w = max(1, int(round(0.5 * p["delta"] / dx)))
    near = ndimage.maximum_filter(a["HRR"], size=2 * w + 1, mode="nearest") / p["peak"]
    q_min = 1.0
    for s in segs:
        i = np.clip(np.rint((s[:, 0] - xs[0]) / dx).astype(int), 0, xs.size - 1)
        j = np.clip(np.rint((s[:, 1] - ys[0]) / dx).astype(int), 0, ys.size - 1)
        # Away from the top plane, where the planar flame's isotherm ends
        keep = ys[j] < ys[-1] - 2 * p["delta"]
        if keep.any():
            q_min = min(q_min, near[i[keep], j[keep]].min())
    fresh, n = ndimage.label(a["T"] < T_mid)
    inlet = set(np.unique(fresh[0, :])) - {0}
    pocket = any(k not in inlet for k in range(1, n + 1))
    return t, Q, L, q_min, pocket


def parameters(run_dir):
    head = open(os.path.join(run_dir, "input.toml")).readline()
    m = re.search(r"r / delta = ([0-9.e+-]+), u' / S_L = ([0-9.e+-]+)", head)
    return float(m.group(1)), float(m.group(2))


def classify(rows):
    if rows[:, 3].min() < 0.1:
        return "quenched"
    if rows[:, 4].any():
        return "pocket"
    if np.abs(rows[:, 1] - rows[0, 1]).max() >= 0.05 * rows[0, 1] or np.abs(rows[:, 2] - rows[0, 2]).max() >= 0.05:
        return "wrinkled"
    return "no effect"


def analyze(run_dir, p):
    pvd = os.path.join(run_dir, "solut", "flame.pvd")
    files = re.findall(r'file="([^"]+)"', open(pvd).read())
    return np.array([frame_metrics(os.path.join(run_dir, "solut", f), p) for f in files])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    ap.add_argument("--mechanism", default=os.path.join(root, "mechanisms", "h2o2.yaml"))
    ap.add_argument("--phase", default="ohmech")
    ap.add_argument("--plot")
    ap.add_argument("--csv")
    args = ap.parse_args()
    results = []
    for run in args.runs:
        p = planar(run, args.mechanism, args.phase)
        r, u = parameters(run)
        rows = analyze(run, p)
        tau = p["delta"] / p["S_L"]
        outcome = classify(rows)
        print(f"{run}: r / delta_L = {r:g}, u' / S_L = {u:g}, Ka(r) = {u / r:.2f}: {outcome}")
        print(f"{'t/tau':>7} {'Q':>7} {'L':>7} {'q_min':>7} pocket")
        for row in rows[:: max(1, len(rows) // 20)]:
            print(f"{row[0] / tau:7.2f} {row[1]:7.3f} {row[2]:7.3f} {row[3]:7.3f} {'yes' if row[4] else ''}")
        print(f"  Q {rows[:, 1].min():.3f}-{rows[:, 1].max():.3f}, L up to {rows[:, 2].max():.3f}, "
              f"q_min down to {rows[:, 3].min():.3f}")
        results.append(dict(run=run, r=r, u=u, rows=rows, tau=tau, outcome=outcome))
    if args.csv:
        with open(args.csv, "w") as fh:
            fh.write("run,r/delta,u'/S_L,outcome,Q_min,Q_max,L_max,q_min\n")
            for res in results:
                rows = res["rows"]
                fh.write(f"{res['run']},{res['r']:g},{res['u']:g},{res['outcome']},{rows[:, 1].min():.4f},"
                         f"{rows[:, 1].max():.4f},{rows[:, 2].max():.4f},{rows[:, 3].min():.4f}\n")
    if args.plot:
        plot(results, args.plot)


def spectral_axes(ax, fg="black"):
    r = np.logspace(-1, 2, 50)
    ax.plot(r, r, color=fg, ls="--", lw=1)
    ax.text(30, 22, "Ka(r) = 1", color=fg, fontsize=9, rotation=38)
    ax.add_patch(__import__("matplotlib").patches.Rectangle((0.81, 1), 11 - 0.81, 99, fill=False, ec="#888",
                                                             ls=":", lw=1))
    ax.text(0.85, 75, "range of Poinsot et al.", color="#888", fontsize=8)
    ax.plot(4.8, 28, "*", color="#888", ms=10)
    ax.text(5.3, 30, "their quenching example\n(with heat losses)", color="#888", fontsize=8, va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.3, 50)
    ax.set_ylim(0.5, 150)
    ax.set_xlabel("vortex pair size r / delta_L")
    ax.set_ylabel("vortex velocity u' / S_L")


def plot(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    spectral_axes(axs[0])
    for res in results:
        m, c = MARKERS[res["outcome"]]
        axs[0].plot(res["r"], res["u"], m, color=c, ms=10, mec="k")
    for o in OUTCOMES:
        m, c = MARKERS[o]
        axs[0].plot([], [], m, color=c, mec="k", label=o)
    axs[0].legend(loc="lower right", fontsize=9)
    axs[0].set_title("Spectral diagram")
    for res in results:
        t = res["rows"][:, 0] / res["tau"]
        lab = f"r = {res['r']:g}, u' = {res['u']:g}"
        axs[1].plot(t, res["rows"][:, 1], label=lab)
        axs[2].plot(t, res["rows"][:, 3], label=lab)
    axs[1].set_xlabel("t S_L / delta_L")
    axs[1].set_ylabel("heat release / planar flame's")
    axs[2].set_xlabel("t S_L / delta_L")
    axs[2].set_ylabel("weakest burning along the front, q_min")
    axs[2].axhline(0.1, color="#888", ls=":")
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
