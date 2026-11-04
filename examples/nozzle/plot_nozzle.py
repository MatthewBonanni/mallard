#!/usr/bin/env python3
"""Compare the nozzle example (examples/nozzle: stagnation inlet, p0 = T0 = 1,
gamma = 1.4, back pressure 0.75 p0) with the quasi-one-dimensional solution:
isentropic flow through the throat, a normal shock in the diverging part
where the subsonic flow behind it leaves at the back pressure, and the
section-averaged pressure, Mach number and stagnation pressure of the run.

    plot_nozzle.py SNAPSHOT [--p-back 0.75] [--figure]

Prints the mass flow against the choked one, the shock position and the
stagnation-pressure ratio across the shock.
"""
import argparse

import numpy as np
from scipy.optimize import brentq

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../tools/docs"))
import figstyle  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../tools"))
from mallard.vtu import read_vtu_cells  # noqa: E402

G = 1.4
A_EXIT = 1.6


def height(x):
    return 1.0 + 0.6 * (0.5 * x - 1.0) ** 2


def area_ratio(M):
    return (2.0 / (G + 1.0) * (1.0 + 0.5 * (G - 1.0) * M * M)) ** (0.5 * (G + 1.0) / (G - 1.0)) / M


def mach(A, supersonic):
    if A <= 1.0:
        return 1.0
    return brentq(lambda M: area_ratio(M) - A, 1.0, 10.0) if supersonic else brentq(lambda M: area_ratio(M) - A, 1e-9, 1.0)


def total_pressure_ratio(M):
    a = (G + 1.0) * M * M / ((G - 1.0) * M * M + 2.0)
    b = (G + 1.0) / (2.0 * G * M * M - (G - 1.0))
    return a ** (G / (G - 1.0)) * b ** (1.0 / (G - 1.0))


def static_ratio(M):
    return (1.0 + 0.5 * (G - 1.0) * M * M) ** (-G / (G - 1.0))


def quasi_1d(p_back):
    """Shock position and the exact Mach number, pressure and stagnation pressure along x."""
    def exit_pressure(A_s):
        r = total_pressure_ratio(mach(A_s, True))
        return r * static_ratio(mach(A_EXIT * r, False))

    A_s = brentq(lambda A: exit_pressure(A) - p_back, 1.0 + 1e-9, A_EXIT)
    x_s = 2.0 + 2.0 * np.sqrt((A_s - 1.0) / 0.6)
    r = total_pressure_ratio(mach(A_s, True))
    x = np.linspace(0.0, 4.0, 801)
    M = np.array([mach(height(xi), xi > 2.0) if xi < x_s else mach(height(xi) * r, False) for xi in x])
    p0 = np.where(x < x_s, 1.0, r)
    return x_s, r, x, M, p0 * np.array([static_ratio(m) for m in M]), p0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("--figure", action="store_true", help="also draw docs/images/guide/nozzle_validation-{light,dark}.svg")
    ap.add_argument("--p-back", type=float, default=0.75)
    args = ap.parse_args()

    pts, conn, offs, types, arrays = read_vtu_cells(args.snapshot)
    starts = np.concatenate([[0], offs[:-1]])
    quads = np.array([conn[s:e] for s, e in zip(starts, offs)])
    xy = pts[quads][:, :, :2]
    # Shoelace area and centroid of each quadrilateral
    x0, y0 = xy[..., 0], xy[..., 1]
    x1, y1 = np.roll(x0, -1, axis=1), np.roll(y0, -1, axis=1)
    cross = x0 * y1 - x1 * y0
    area = 0.5 * cross.sum(axis=1)
    cx = ((x0 + x1) * cross).sum(axis=1) / (6.0 * area)

    rho, ux, uy = arrays["RHO"], arrays["U_X"], arrays["U_Y"]
    p, T = arrays["P"], arrays["T"]
    M2 = (ux ** 2 + uy ** 2) / (G * T)
    p0_local = p * (1.0 + 0.5 * (G - 1.0) * M2) ** (G / (G - 1.0))
    mdot = np.sum(rho * ux * area) / 4.0
    T_star = 2.0 / (G + 1.0)
    mdot_choked = T_star ** (G / (G - 1.0)) / T_star * np.sqrt(G * T_star)

    columns = np.unique(np.round(cx, 9))
    col = np.searchsorted(columns, np.round(cx, 9))
    weight = np.bincount(col, area)
    avg = {name: np.bincount(col, v * area) / weight for name, v in
           (("p", p), ("M", np.sqrt(M2)), ("p0", p0_local))}

    x_s, r, x, M_exact, p_exact, p0_exact = quasi_1d(args.p_back)
    p1 = static_ratio(mach(height(x_s), True))
    M2s = np.sqrt((1.0 + 0.5 * (G - 1.0) * mach(height(x_s), True) ** 2) /
                  (G * mach(height(x_s), True) ** 2 - 0.5 * (G - 1.0)))
    mid = 0.5 * (p1 + r * static_ratio(M2s))
    i = np.nonzero((avg["p"][:-1] < mid) & (avg["p"][1:] >= mid) & (columns[:-1] > 2.0))[0][0]
    x_num = columns[i] + (mid - avg["p"][i]) / (avg["p"][i + 1] - avg["p"][i]) * (columns[i + 1] - columns[i])
    dx = columns[1] - columns[0]
    behind = avg["p0"][columns > x_s + 6 * dx].mean()
    # Ahead of the curved, captured shock (its foot at the wall lies upstream of the axis)
    ahead = np.abs(avg["p0"][columns < x_num - 20 * dx] - 1.0).max()
    print(f"mass flow {mdot:.5f}, choked {mdot_choked:.5f} ({100 * (mdot / mdot_choked - 1):+.2f}%)")
    print(f"shock at x = {x_num:.4f}, quasi-1D {x_s:.4f} ({(x_num - x_s) / dx:+.2f} cells)")
    print(f"stagnation pressure: within {ahead:.1e} of p0 ahead of the shock; "
          f"{behind:.4f} behind it, quasi-1D {r:.4f}")

    def draw(P):
        from matplotlib.collections import PolyCollection
        plt = figstyle.plt
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 4.2), gridspec_kw={"width_ratios": [1.0, 1.7]})
        # The Mach number in the nozzle, mirrored about the axis
        for sign in (1.0, -1.0):
            polys = xy.copy()
            polys[..., 1] *= sign
            pc = PolyCollection(polys, array=np.sqrt(M2), cmap=P["cmap"], edgecolors="face", linewidths=0.2)
            pc.set_clim(0.0, 1.8)
            pc.set_rasterized(True)
            ax0.add_collection(pc)
        xw = np.linspace(0.0, 4.0, 200)
        ax0.plot(xw, height(xw), color=P["line"], lw=1.2)
        ax0.plot(xw, -height(xw), color=P["line"], lw=1.2)
        ax0.axvline(x_s, color=P["muted"], ls=":", lw=1.0)
        ax0.set_xlim(0.0, 4.0)
        ax0.set_ylim(-1.75, 1.75)
        ax0.set_aspect("equal")
        ax0.set_xlabel("x")
        ax0.set_ylabel("y")
        cb = fig.colorbar(pc, ax=ax0, orientation="horizontal", fraction=0.05, pad=0.18)
        cb.set_label("Mach number")
        cb.outline.set_edgecolor(P["muted"])

        every = slice(None, None, max(1, len(columns) // 80))
        for key, exact, color, name in (("M", M_exact, P["blue"], "Mach number"), ("p", p_exact, P["flame"], "p / p0"),
                                        ("p0", p0_exact, P["teal"], "stagnation pressure / p0")):
            ax1.plot(x, exact, color=color, lw=1.6, label=name + ", quasi-1D")
            ax1.plot(columns[every], avg[key][every], "o", color=color, ms=3, mfc="none", mew=1.0,
                     label="Mallard, section averages" if key == "M" else None)
        ax1.axvline(x_s, color=P["muted"], ls=":", lw=1.0)
        ax1.set_xlabel("x")
        ax1.set_xlim(0.0, 4.0)
        ax1.set_ylim(0.0, 1.8)
        ax1.legend(loc="upper left", fontsize=10, ncol=2)
        return fig

    if args.figure:
        figstyle.save("nozzle_validation", draw)

if __name__ == "__main__":
    main()
