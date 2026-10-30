#!/usr/bin/env python3
"""The forced-inlet example (examples/forced_inlet): x-t diagrams of the
velocity and temperature fluctuations along the duct, whose slopes are the
acoustic (u + c) and entropy (u) wave speeds, and the inlet cell's velocity
and temperature against the forcing.

    plot_forced_inlet.py PREFIX [--figure]

PREFIX is the series' prefix (solut/duct). --figure draws
docs/images/guide/forced_inlet-{light,dark}.svg.
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../tools/docs"))
import figstyle  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../tools"))
from mallard.vtu import read_vtu_cells  # noqa: E402

U0, T0 = 50.0, 300.0


def u_target(t):
    return U0 * (1.0 + 0.1 * np.sin(2.0 * np.pi * 1000.0 * t))


def T_target(t):
    return T0 * (1.0 + 0.05 * np.sin(2.0 * np.pi * 250.0 * t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--figure", action="store_true")
    args = ap.parse_args()
    files = sorted(glob.glob(args.prefix + "_[0-9]*.vtu"))
    times, u, T = [], [], []
    x = None
    for f in files:
        pts, conn, offs, types, a = read_vtu_cells(f)
        starts = np.concatenate([[0], offs[:-1]])
        cx = np.array([pts[conn[s:e], 0].mean() for s, e in zip(starts, offs)])
        order = np.argsort(cx)
        x = cx[order]
        times.append(a["TIME"])
        u.append(a["U_X"][order])
        T.append(a["T"][order])
    t, u, T = np.array(times), np.array(u), np.array(T)
    c = np.sqrt(1.4 * 287.0 * T0)
    late = t > 0.01
    print(f"inlet cell: velocity amplitude {np.ptp(u[late, 0]) / 2:.2f} m/s (forcing 5.00), "
          f"temperature amplitude {np.ptp(T[late, 0]) / 2:.2f} K (forcing 15.00)")
    if not args.figure:
        return

    def draw(P):
        plt = figstyle.plt
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), gridspec_kw=dict(width_ratios=[1, 1, 1.25], wspace=0.45))
        ms = 1e3
        for ax, field, ref, label, cmap in ((axes[0], u, U0, r"$u'$ [m/s]: acoustic waves at $u + c$", "RdBu_r"),
                                            (axes[1], T, T0, r"$T'$ [K]: entropy waves at $u$", "RdBu_r")):
            amp = np.max(np.abs(field - ref))
            mesh = ax.pcolormesh(x, ms * t, field - ref, cmap=cmap, vmin=-amp, vmax=amp, shading="auto",
                                 rasterized=True)
            cb = fig.colorbar(mesh, ax=ax, fraction=0.05, pad=0.03)
            cb.outline.set_edgecolor(P["muted"])
            ax.set_title(label, fontsize=12)
            ax.set_xlabel("x [m]")
            ax.set_ylabel("t [ms]")
        ax = axes[2]
        tt = np.linspace(0.0, t[-1], 2000)
        ax.plot(ms * tt, u_target(tt) / U0 - 1.0, color=P["blue"], lw=1.0, label=r"$u_t / U - 1$ (target)")
        ax.plot(ms * t, u[:, 0] / U0 - 1.0, "o", ms=2.5, mfc="none", color=P["blue"], label="first cell")
        ax.plot(ms * tt, T_target(tt) / T0 - 1.0, color=P["flame"], lw=1.0, label=r"$T_t / T_0 - 1$ (target)")
        ax.plot(ms * t, T[:, 0] / T0 - 1.0, "s", ms=2.5, mfc="none", color=P["flame"], label="first cell")
        ax.set_xlabel("t [ms]")
        ax.set_ylabel("relative fluctuation")
        ax.set_xlim(10.0, ms * t[-1])
        ax.legend(loc="upper left", ncol=2, fontsize=9)
        ax.set_ylim(-0.12, 0.2)
        return fig

    figstyle.save("forced_inlet", draw)


if __name__ == "__main__":
    main()
