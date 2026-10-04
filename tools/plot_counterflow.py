"""Counterflow diffusion flame runs (V10) against Cantera.

    python tools/plot_counterflow.py OUT_PREFIX --run LABEL RUN_DIR [--run ...]
        [--reference examples/counterflow_diffusion/reference] [--history]

Each RUN_DIR holds a run of tools/counterflow_setup.py; its symmetry-plane
output (solut/plane_*.vtu) gives the stagnation line: the cells next to the
axis, extrapolated to r = 0 with the cells one further out (f = f0 + b r^2).
On it: the peak temperature (parabola through the hottest cell and its
neighbors) and the local strain rates K_ox and K_f (the largest -du/dx on the
oxidizer side of the peak temperature and on the fuel side of the stagnation
point), as tools/counterflow_reference.py measures Cantera's.

Writes OUT_PREFIX_strain.png (peak T against K_ox: Cantera's sweep up to
extinction, and the runs), OUT_PREFIX_profiles.png (T, u and major species
along the stagnation line against Cantera's flame of the same K_ox, from
reference/K<K_ox>.csv written by counterflow_reference.py --match, aligned
at the peak temperature) and, with --history, OUT_PREFIX_history.png (peak T
and K_ox over time), and prints the comparison table.
"""
import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mallard_vtu import read_vtu_cells  # noqa: E402

SPECIES = ["H2", "O2", "H2O", "OH"]


def peak(x, T):
    """Peak of T through the parabola of its largest sample and neighbors."""
    i = int(np.clip(np.argmax(T), 1, T.size - 2))
    c = np.polyfit(x[i - 1:i + 2] - x[i], T[i - 1:i + 2], 2)
    if c[0] >= 0:
        return T.max(), x[np.argmax(T)]
    xm = -c[1] / (2 * c[0])
    return np.polyval(c, xm), x[i] + xm


def local_strains(x, u, T):
    dudx = np.gradient(u, x)
    i_T = np.argmax(T)
    i_s = np.argmin(np.abs(u[: i_T + 1]))
    return np.max(-dudx[i_T:]), np.max(-dudx[: max(i_s, 1)])


def stagnation_line(path):
    """x and the fields on the axis from a z = 0 symmetry-plane VTU."""
    pts, conn, offs, _, arrays = read_vtu_cells(path)
    c = pts[conn.reshape(-1, 4)].mean(axis=1)
    x, y = np.round(c[:, 0], 12), np.round(c[:, 1], 12)
    ys = np.unique(y)
    rows = []
    for yy in ys[:2]:
        sel = np.nonzero(y == yy)[0]
        sel = sel[np.argsort(x[sel])]
        rows.append(sel)
    # cell centers of the plane's cells are at z = dz0 / 2, the first ones at y = dy0 / 2 = dz0 / 2
    z0 = ys[0]
    r2 = [ys[0] ** 2 + z0 ** 2, ys[1] ** 2 + z0 ** 2]
    w = r2[0] / (r2[1] - r2[0])
    out = {"x": x[rows[0]], "TIME": arrays["TIME"]}
    for name, v in arrays.items():
        if name == "TIME":
            continue
        if v.ndim == 2:
            v = v[:, 0]
        out[name] = v[rows[0]] - w * (v[rows[1]] - v[rows[0]])
    return out


def run_series(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "solut", "plane_*.vtu")))
    if not files:
        raise SystemExit(f"{run_dir}: no solut/plane_*.vtu")
    return files


def measure(line):
    T_max, x_T = peak(line["x"], line["T"])
    K_ox, K_f = local_strains(line["x"], line["U"], line["T"])
    return T_max, x_T, K_ox, K_f


def load_reference(path):
    with open(path) as f:
        header = f.readline()
        names = f.readline().strip().split(",")
    data = np.loadtxt(path, delimiter=",", skiprows=2)
    return header, {n: data[:, i] for i, n in enumerate(names)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--run", nargs=2, action="append", metavar=("LABEL", "DIR"), required=True)
    ap.add_argument("--reference", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                        "examples", "counterflow_diffusion", "reference"))
    ap.add_argument("--history", action="store_true")
    args = ap.parse_args()

    sweep = np.genfromtxt(os.path.join(args.reference, "strain.csv"), delimiter=",", names=True, skip_header=1)
    K_ref, T_ref = sweep["K_ox"], sweep["T_max"]
    matched = {}
    for path in glob.glob(os.path.join(args.reference, "K*.csv")):
        header, prof = load_reference(path)
        K = float(re.search(r"K_ox = ([0-9.eE+-]+)", header).group(1))
        matched[K] = prof

    results = []
    for label, run_dir in args.run:
        files = run_series(run_dir)
        line = stagnation_line(files[-1])
        T_max, x_T, K_ox, K_f = measure(line)
        hist = None
        if args.history:
            hist = np.array([(stagnation_line(f)["TIME"],) + measure(stagnation_line(f)) for f in files])
        T_cantera = np.interp(K_ox, K_ref, T_ref) if K_ox <= K_ref.max() else np.nan
        results.append(dict(label=label, line=line, T_max=T_max, x_T=x_T, K_ox=K_ox, K_f=K_f, hist=hist,
                            T_cantera=T_cantera, t=line["TIME"]))

    print(f"{'run':<28s} {'t [ms]':>8s} {'K_ox':>8s} {'K_f':>8s} {'T_max':>8s} {'Cantera':>8s} {'error':>7s}")
    for r in results:
        err = (r["T_max"] - r["T_cantera"]) / r["T_cantera"]
        print(f"{r['label']:<28s} {r['t'] * 1e3:8.3f} {r['K_ox']:8.1f} {r['K_f']:8.1f} {r['T_max']:8.1f} "
              f"{r['T_cantera']:8.1f} {100 * err:6.2f}%")

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(K_ref, T_ref, "k-", lw=1.5, label="Cantera (axisymmetric similarity solution)")
    ax.plot(K_ref[-1], T_ref[-1], "kx", ms=9, mew=2, label=f"Cantera extinction, $K_{{ox}}$ = {K_ref[-1]:.0f} 1/s")
    band = np.linspace(K_ref.min(), K_ref.max(), 200)
    T_band = np.interp(band, K_ref, T_ref)
    ax.fill_between(band, 0.98 * T_band, 1.02 * T_band, color="0.85", lw=0, label="±2%")
    for r in results:
        ax.plot(r["K_ox"], r["T_max"], "o", ms=7, label=f"Mallard, {r['label']}")
    ax.set_xscale("log")
    ax.set_xlabel(r"local strain rate $K_{ox}$ = max $(-\partial u / \partial x)$ ahead of the flame [1/s]")
    ax.set_ylabel("peak temperature [K]")
    ax.set_title("H$_2$/N$_2$ (1:3) against air, 300 K, 1 atm")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out + "_strain.png", dpi=150)

    with_ref = [r for r in results if matched]
    if with_ref:
        fig, axes = plt.subplots(len(with_ref), 2, figsize=(11, 3.4 * len(with_ref)), squeeze=False)
        for row, r in zip(axes, with_ref):
            K = min(matched, key=lambda k: abs(k - r["K_ox"]))
            ref = matched[K]
            z_T = peak(ref["z"], ref["T"])[1]
            x = (r["line"]["x"] - r["x_T"]) * 1e3
            zr = (ref["z"] - z_T) * 1e3
            ax = row[0]
            ax.plot(zr, ref["T"], "k-", lw=1.2, label=f"Cantera, $K_{{ox}}$ = {K:.0f} 1/s")
            ax.plot(x, r["line"]["T"], "C3o", ms=3, label=f"Mallard, $K_{{ox}}$ = {r['K_ox']:.0f} 1/s")
            ax.set_ylabel("T [K]")
            ax2 = ax.twinx()
            ax2.plot(zr, ref["u"], "k--", lw=1)
            ax2.plot(x, r["line"]["U"], "C0s", ms=2.5)
            ax2.set_ylabel("u [m/s] (dashed, squares)")
            ax.set_title(r["label"], fontsize=10)
            ax.legend(fontsize=8, loc="upper left")
            ax = row[1]
            for k, s in enumerate(SPECIES):
                scale = 10.0 if s == "OH" else 1.0
                lab = f"{s}" + (" x10" if scale != 1 else "")
                ax.plot(zr, scale * ref["Y_" + s], "-", color=f"C{k}", lw=1.2, label=lab)
                ax.plot(x, scale * r["line"]["Y_" + s], "o", color=f"C{k}", ms=2.5)
            ax.set_ylabel("mass fraction")
            ax.legend(fontsize=8, ncol=2)
            for a in row:
                a.set_xlim(-4, 4)
                a.set_xlabel("x - x(T$_{max}$) [mm]")
                a.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out + "_profiles.png", dpi=150)

    if args.history:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        for r in results:
            h = r["hist"]
            axes[0].plot(h[:, 0] * 1e3, h[:, 1], label=r["label"])
            axes[1].plot(h[:, 0] * 1e3, h[:, 3], label=r["label"])
        axes[0].set_ylabel("peak T [K]")
        axes[1].set_ylabel("K_ox [1/s]")
        for a in axes:
            a.set_xlabel("t [ms]")
            a.grid(True, alpha=0.3)
            a.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.out + "_history.png", dpi=150)


if __name__ == "__main__":
    main()
