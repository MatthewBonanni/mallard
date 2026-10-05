"""Outcome of flame-vortex interaction runs (tools/flame_vortex.py) on the spectral diagram.

    python tools/flame_vortex_analysis.py RUN_DIR [RUN_DIR ...] [--mechanism mechanisms/h2o2.yaml]
        [--phase ohmech] [--control RUN_DIR ...] [--plot OUT.png] [--csv OUT.csv]

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
  wrinkled  Q changes by 5% or more (their cut-off limit: about 5% change in
            the total reaction rate);
  no effect Q changes by less than 5%.
Q is compared with its value at the same time in the --control run on the same
mesh (a run of tools/flame_vortex.py with --u 0: the planar flame alone, which
relaxes by a few percent from Cantera's profile in the first flame time), or
without one with its first output.
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
    head = open(os.path.join(run_dir, "input.toml")).read()
    m = re.search(r"r / delta = ([0-9.e+-]+), u' / S_L = ([0-9.e+-]+)", head)
    return float(m.group(1)), float(m.group(2))


def cell_size(run_dir):
    text = open(os.path.join(run_dir, "input.toml")).read()
    return float(re.search(r"Lx = (\S+)", text).group(1)) / int(re.search(r"Nx = (\d+)", text).group(1))


def load_controls(dirs, mechanism, phase):
    return [(cell_size(d), analyze(d, planar(d, mechanism, phase))) for d in dirs or []]


def control_for(run_dir, controls):
    """The control run on the same mesh as run_dir (its cell size), if any."""
    dx = cell_size(run_dir)
    match = [rows for dx_c, rows in controls if abs(dx_c / dx - 1) < 1e-6]
    return match[0] if match else None


def relative_Q(rows, control=None):
    """Q over the planar flame's at the same time (the control run's), or over the first output's."""
    if control is None:
        return rows[:, 1] / rows[0, 1]
    return rows[:, 1] / np.interp(rows[:, 0], control[:, 0], control[:, 1])


def classify(rows, control=None):
    if rows[:, 3].min() < 0.1:
        return "quenched"
    if rows[:, 4].any():
        return "pocket"
    if np.abs(relative_Q(rows, control) - 1).max() >= 0.05:
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
    ap.add_argument("--control", nargs="+")
    ap.add_argument("--plot")
    ap.add_argument("--csv")
    args = ap.parse_args()
    controls = load_controls(args.control, args.mechanism, args.phase)
    results = []
    for run in args.runs:
        p = planar(run, args.mechanism, args.phase)
        r, u = parameters(run)
        rows = analyze(run, p)
        tau = p["delta"] / p["S_L"]
        control = control_for(run, controls)
        outcome = classify(rows, control)
        Qr = relative_Q(rows, control)
        print(f"{run}: r / delta_L = {r:g}, u' / S_L = {u:g}, Ka(r) = {u / r:.2f}: {outcome}")
        print(f"{'t/tau':>7} {'Q':>7} {'Q/Q_0':>7} {'L':>7} {'q_min':>7} pocket")
        for row, q in list(zip(rows, Qr))[:: max(1, len(rows) // 20)]:
            print(f"{row[0] / tau:7.2f} {row[1]:7.3f} {q:7.3f} {row[2]:7.3f} {row[3]:7.3f} {'yes' if row[4] else ''}")
        print(f"  Q / Q_0 {Qr.min():.3f}-{Qr.max():.3f}, L up to {rows[:, 2].max():.3f}, "
              f"q_min down to {rows[:, 3].min():.3f}")
        results.append(dict(run=run, r=r, u=u, rows=rows, Qr=Qr, tau=tau, outcome=outcome))
    if args.csv:
        with open(args.csv, "w") as fh:
            fh.write("run,r/delta,u'/S_L,Ka(r),outcome,Q/Q0_min,Q/Q0_max,L_max,q_min,pocket\n")
            for res in results:
                rows = res["rows"]
                fh.write(f"{res['run']},{res['r']:g},{res['u']:g},{res['u'] / res['r']:.3g},{res['outcome']},"
                         f"{res['Qr'].min():.4f},{res['Qr'].max():.4f},{rows[:, 2].max():.4f},"
                         f"{rows[:, 3].min():.4f},{int(rows[:, 4].any())}\n")
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
    cmap = plt.get_cmap("tab20")
    for i, res in enumerate(results):
        t = res["rows"][:, 0] / res["tau"]
        lab = f"r = {res['r']:g}, u' = {res['u']:g}"
        axs[1].plot(t, res["Qr"], color=cmap(i % 20), label=lab)
        axs[2].plot(t, res["rows"][:, 3], color=cmap(i % 20), label=lab)
    axs[1].axhspan(0.95, 1.05, color="#ddd", zorder=0)
    axs[1].set_xlabel("t S_L / delta_L")
    axs[1].set_ylabel("heat release / planar flame's (grey: cut-off, 5%)")
    axs[2].set_xlabel("t S_L / delta_L")
    axs[2].set_ylabel("weakest burning along the front, q_min")
    axs[2].axhline(0.1, color="#888", ls=":")
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
