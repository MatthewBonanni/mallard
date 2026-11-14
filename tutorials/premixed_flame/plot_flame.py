#!/usr/bin/env python3
"""Figures of the premixed-flame tutorial.

    python plot_flame.py setup RUN_DIR [--out DIR]
    python plot_flame.py results RUN_DIR [RUN_DIR ...] [--out DIR]

setup: the strip and its boundary zones (from RUN_DIR/strip.msh) above the
initial temperature and Cantera's flame (premixed_flame_setup).
results: for the first RUN_DIR (the stoichiometric flame), the consumption
speed over time (premixed_flame_history) and the temperature, heat release
and H2, O2, H2O and OH profiles against Cantera's (premixed_flame_profiles);
with several RUN_DIRs (flames at different equivalence ratios, from
flame_setup.py), the flame speed against Cantera's (premixed_flame_speed).
Prints each run's consumption speed over its last flame time.
"""
import argparse
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import matplotlib.pyplot as plt  # noqa: E402
import snapshots  # noqa: E402
from tutfig import label_box, render  # noqa: E402
from gmsh_mesh import read_msh  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402

W = {"H2": 2.016, "O2": 31.998}


def reference(run):
    text = open(os.path.join(run, "reference.csv")).readline()
    s_l = float(re.search(r"S_L = ([0-9.]+)", text).group(1))
    delta = float(re.search(r"delta = ([0-9.e+-]+) m", text).group(1))
    phi = float(re.search(r"phi = ([0-9.]+)", text).group(1))
    return phi, s_l, delta, np.genfromtxt(os.path.join(run, "reference.csv"), delimiter=",", names=True, skip_header=1)


def profile(path):
    pts, conn, offs, a = snapshots.read(path)
    _, c = snapshots.cells(pts, conn, offs)
    order = np.argsort(c[:, 0])
    return c[order, 0], {k: v[order] for k, v in a.items() if np.ndim(v) == 1 and v.size == len(order)}, a["TIME"]


def consumption_speed(x, a):
    """-int W omega_k dx / (rho_u (Y_k,u - Y_k,b)) of the deficient reactant."""
    n_h2, n_o2 = a["Y_H2"][0] / W["H2"], a["Y_O2"][0] / W["O2"]
    k = "H2" if n_o2 >= 0.5 * n_h2 * (1 - 1e-9) else "O2"
    edges = np.concatenate([[x[0] - 0.5 * (x[1] - x[0])], 0.5 * (x[1:] + x[:-1]), [x[-1] + 0.5 * (x[-1] - x[-2])]])
    rate = -np.sum(a["OMEGA_" + k] * np.diff(edges))
    return rate / (a["RHO"][0] * (a["Y_" + k][0] - a["Y_" + k][-1]))


def history(run):
    t, s = [], []
    for time, path in snapshots.series(os.path.join(run, "solut", "flame.pvd")):
        x, a, _ = profile(path)
        t.append(time)
        s.append(consumption_speed(x, a))
    return np.array(t), np.array(s)


def align(x, T):
    i = np.argmax(np.gradient(T, x))
    return x - x[i]


def setup(args):
    run = args.runs[0]
    xyz, cells, zones = read_msh(os.path.join(run, "strip.msh"))
    phi, s_l, delta, ref = reference(run)
    x, a, _ = profile(snapshots.series(os.path.join(run, "solut", "flame.pvd"))[0][1])
    L, h = xyz[:, 0].max(), xyz[:, 1].max()

    def draw(t):
        fig, (ax, bx) = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True, gridspec_kw={"height_ratios": [1, 2.2]})
        s = t["series"]
        mm = 1e3
        ax.add_collection(PolyCollection([xyz[c][:, :2] * mm for c in cells], facecolors="none",
                                         edgecolors=t["mesh"], linewidths=0.3, rasterized=True))
        colors = {"inlet": s[0], "outlet": s[1], "sides": s[2]}
        for name, faces in zones.items():
            ax.add_collection(LineCollection([xyz[f][:, :2] * mm for f in faces], colors=colors[name], linewidths=3,
                                             label=f'"{name}"'))
        ax.set_ylim(-0.6 * h * mm, 1.6 * h * mm)
        ax.set_yticks([])
        ax.grid(False)
        ax.legend(loc="upper center", ncol=3, fontsize=11, bbox_to_anchor=(0.5, 1.45))
        ax.set_title(f"{len(cells)} cells, {delta / (L / len(cells)):.0f} per thermal thickness δ", pad=34)
        ax.annotate("", (0.9, 0.5 * h * mm), (0.0, 0.5 * h * mm),
                    arrowprops={"arrowstyle": "-|>", "color": t["ink"], "lw": 1.5})
        ax.text(0.95, 0.5 * h * mm, f"fresh H2/air, φ = {phi:g}, at S_L", va="center", ha="left", fontsize=11,
                bbox=label_box(t))
        mid = 0.5 * (ref["T"][0] + ref["T"][-1])
        xr = ref["x"] + np.interp(mid, a["T"], x) - np.interp(mid, ref["T"], ref["x"])
        bx.plot(x * mm, a["T"], color=s[0], label="initial state (tanh profile)")
        bx.plot(xr * mm, ref["T"], color=t["ink"], ls="--", lw=1.4, label="Cantera's flame")
        bx.axvspan(0, 6 * delta * mm, color=t["faint"], zorder=-1)
        bx.text(3 * delta * mm, 1900, "6 δ upstream", ha="center", color=t["ink2"])
        bx.text((6 + 4.5) * delta * mm, 900, "9 δ downstream", ha="center", color=t["ink2"])
        bx.set_xlim(-0.02 * L * mm, 1.02 * L * mm)
        bx.set_xlabel("x (mm)")
        bx.set_ylabel("T (K)")
        bx.legend(loc="lower right")
        fig.tight_layout()
        return fig

    render(draw, args.out, "premixed_flame_setup")


def results(args):
    main_run = args.runs[0]
    phi, s_l, delta, ref = reference(main_run)
    t, s = history(main_run)
    t_flame = delta / s_l
    x, a, time = profile(snapshots.last(os.path.join(main_run, "solut", "flame.pvd")))
    last = t >= t[-1] - t_flame
    print(f"phi = {phi:g}: consumption speed {s[last].mean():.4f} m/s over the last flame time, "
          f"Cantera {s_l:.4f} ({100 * (s[last].mean() / s_l - 1):+.2f}%)")

    def draw_history(th):
        fig, ax = plt.subplots(figsize=(8, 3.6))
        ax.plot(t / t_flame, s / s_l, color=th["series"][0], marker="o", ms=3, label="Mallard, consumption speed")
        ax.axhline(1, color=th["ink"], ls="--", lw=1.2, label="Cantera")
        ax.axvspan(t[-1] / t_flame - 1, t[-1] / t_flame, color=th["faint"], zorder=-1)
        ax.text(t[-1] / t_flame - 0.5, 1.12, "averaged", ha="center", color=th["ink2"], fontsize=11)
        ax.set_ylim(0.85, 1.15)
        ax.set_xlabel("t / (δ / S_L)")
        ax.set_ylabel("S_c / S_L")
        ax.legend(loc="lower right")
        return fig

    def draw_profiles(th):
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4))
        xa = align(x, a["T"]) / delta
        xr = ref["x"] / delta
        s = th["series"]
        ax.plot(xa, a["T"], color=s[0], label="T, Mallard")
        ax.plot(xr, ref["T"], color=th["ink"], ls="--", lw=1.2, label="Cantera")
        ax2 = ax.twinx()
        ax2.plot(xa, a["HRR"] / 1e9, color=s[1], label="heat release, Mallard")
        ax2.plot(xr, ref["hrr"] / 1e9, color=th["ink"], ls=":", lw=1.2)
        ax2.set_ylabel("heat release rate (GW/m³)", color=s[1])
        ax2.spines["right"].set_visible(True)
        ax2.grid(False)
        ax.set_xlim(-4, 8)
        ax.set_xlabel("(x − x_f) / δ")
        ax.set_ylabel("T (K)", color=s[0])
        ax.set_title("Temperature and heat release")
        ax.legend(loc="center right", fontsize=10)
        species = [("H2", s[0]), ("O2", s[2]), ("H2O", s[3]), ("OH", s[1])]
        for k, color in species:
            scale = 10 if k == "OH" else 1
            bx.plot(xa, a["Y_" + k] * scale, color=color, label=f"{k}" + (" × 10" if scale > 1 else ""))
            bx.plot(xr, ref["Y_" + k] * scale, color=th["ink"], ls="--", lw=1)
        bx.set_xlim(-4, 8)
        bx.set_xlabel("(x − x_f) / δ")
        bx.set_ylabel("mass fraction")
        bx.set_title("Species (dashed: Cantera)")
        bx.legend(fontsize=10, ncol=2)
        fig.tight_layout()
        return fig

    render(draw_history, args.out, "premixed_flame_history")
    render(draw_profiles, args.out, "premixed_flame_profiles")
    if len(args.runs) < 2:
        return
    rows = []
    for run in args.runs:
        p, sl, d, _ = reference(run)
        tt, ss = history(run)
        rows.append((p, sl, ss[tt >= tt[-1] - d / sl].mean()))
    rows.sort()
    for p, sl, sc in rows:
        print(f"phi = {p:g}: S_c = {sc:.4f} m/s, Cantera {sl:.4f} ({100 * (sc / sl - 1):+.2f}%)")
    rows = np.array(rows)

    def draw_speed(th):
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
        phis = np.linspace(rows[0, 0], rows[-1, 0], 200)
        ax.plot(phis, np.interp(phis, rows[:, 0], rows[:, 1]), color=th["ink"], lw=1.2, label="Cantera")
        ax.plot(rows[:, 0], rows[:, 2], "o", color=th["series"][0], ms=7, label="Mallard")
        ax.set_xlabel("equivalence ratio φ")
        ax.set_ylabel("S_L (m/s)")
        ax.set_title("H2/air, 300 K, 1 atm")
        ax.legend()
        bx.bar(rows[:, 0], 100 * (rows[:, 2] / rows[:, 1] - 1), width=0.12, color=th["series"][0])
        bx.axhline(0, color=th["ink"], lw=0.8)
        bx.set_xlabel("φ")
        bx.set_ylabel("error (%)")
        bx.set_ylim(-1, 1)
        bx.set_title("Mallard vs Cantera")
        fig.tight_layout()
        return fig

    render(draw_speed, args.out, "premixed_flame_speed")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=["setup", "results"])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", help="output directory (default: docs/images/tutorials)")
    args = ap.parse_args()
    setup(args) if args.what == "setup" else results(args)


if __name__ == "__main__":
    main()
