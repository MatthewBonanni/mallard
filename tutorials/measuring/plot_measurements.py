#!/usr/bin/env python3
"""Measurements of the measuring-a-flow tutorial.

    python plot_measurements.py RUN_DIR [--t-start 60] [--out DIR]

RUN_DIR holds a run of input.toml. From --t-start on, prints
- the drag and lift coefficients from [[forces]] and from the wall probe's
  traction integrals;
- the mass balance of the box (d/dt of its mass plus the net mass flux out
  through its four sides) and the mean net enthalpy flux out of it;
- the drag and lift from the box's momentum balance (the momentum flux,
  pressure included, through its sides plus d/dt of its momentum);
- the Strouhal number from the lift's zero crossings and from the peak of the
  wake probe's spectrum;
and writes the figures measuring_balance (mass residual and the two drags
against time), measuring_spectrum (spectrum of the wake's U_Y) and
measuring_phase (the phase-averaged U_Y in the box at four phases).
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
from tutfig import render  # noqa: E402

U, D = 0.2, 1.0
Q = 0.5 * U**2 * D  # dynamic pressure times the diameter (density 1)
SIDES = ("west", "east", "south", "north")


def table(run, name):
    return np.genfromtxt(os.path.join(run, name), delimiter=",", names=True, comments="#")


def ddt(t, f):
    return np.gradient(f, t)


def crossings(t, s):
    """Times of the upward zero crossings of s, interpolated linearly."""
    i = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]
    return t[i] - s[i] * (t[i + 1] - t[i]) / (s[i + 1] - s[i])


def spectrum(t, v):
    """One-sided power spectrum of v (Hann window, mean removed) on a uniform grid of t."""
    dt = np.median(np.diff(t))
    tu = np.arange(t[0], t[-1], dt)
    vu = np.interp(tu, t, v)
    vu = (vu - vu.mean()) * np.hanning(len(vu))
    n = 8 * len(vu)  # zero padding: a smoother peak to locate
    f = np.fft.rfftfreq(n, dt)
    return f, np.abs(np.fft.rfft(vu, n)) ** 2


def peak(f, s):
    """Frequency of the spectrum's peak, refined by a parabola through its neighbors."""
    k = int(np.argmax(s[1:])) + 1
    a, b, c = np.log(s[k - 1:k + 2])
    return f[k] + 0.5 * (a - c) / (a - 2 * b + c) * (f[1] - f[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("--t-start", type=float, default=60.0)
    ap.add_argument("--out", help="output directory (default: docs/images/tutorials)")
    args = ap.parse_args()
    run = args.run

    forces = table(run, "forces_cylinder.csv")
    t = forces["t"]
    cd = (forces["Fx_pressure"] + forces["Fx_viscous"]) / Q
    cl = (forces["Fy_pressure"] + forces["Fy_viscous"]) / Q
    wall = table(run, "probe_wall_reductions.csv")
    wall = wall[np.isin(wall["step"], forces["step"])]
    cd_wall = (wall["integral_TRACTION_PRESSURE_X"] + wall["integral_TRACTION_VISCOUS_X"]) / Q
    cl_wall = (wall["integral_TRACTION_PRESSURE_Y"] + wall["integral_TRACTION_VISCOUS_Y"]) / Q

    box = table(run, "probe_box_reductions.csv")
    tb = box["t"]
    sides = {s: table(run, f"probe_{s}_reductions.csv") for s in SIDES}
    flux = {q: sum(sides[s][f"integral_{q}"] for s in SIDES)
            for q in ("MASS_FLUX", "MOMENTUM_FLUX_X", "MOMENTUM_FLUX_Y", "ENTHALPY_FLUX")}
    mass = box["integral_RHO"]
    residual = ddt(tb, mass) + flux["MASS_FLUX"]
    inflow = -sides["west"]["integral_MASS_FLUX"]
    # The force on the cylinder is minus the rate of change of the box's
    # momentum and its flux out (viscous stresses on the sides neglected)
    cd_box = -(ddt(tb, box["favre_U_X"] * mass) + flux["MOMENTUM_FLUX_X"]) / Q
    cl_box = -(ddt(tb, box["favre_U_Y"] * mass) + flux["MOMENTUM_FLUX_Y"]) / Q

    sel = t >= args.t_start
    selb = tb >= args.t_start
    # Interior points only: np.gradient is one-sided at the ends
    selb[[0, -1]] = False
    tc = crossings(t[sel], cl[sel] - cl[sel].mean())
    st_lift = (len(tc) - 1) / (tc[-1] - tc[0]) * D / U if len(tc) > 1 else float("nan")
    wake = table(run, "probe_wake.csv")
    selw = wake["t"] >= args.t_start
    f, s = spectrum(wake["t"][selw], wake["U_Y"][selw])
    st_spec = peak(f, s) * D / U

    print(f"t >= {args.t_start:g}, {len(tc) - 1} shedding periods")
    print(f"wall:     mean Cd {cd[sel].mean():.4f} ([[forces]]), {cd_wall[sel].mean():.4f} (wall probe); "
          f"largest difference {np.abs(cd_wall - cd).max():.2e}")
    print(f"mass:     box {mass[selb].mean():.4f}, inflow {inflow[selb].mean():.5f} per unit time, "
          f"residual d/dt + net outflow: RMS {np.sqrt(np.mean(residual[selb] ** 2)):.2e} "
          f"({np.sqrt(np.mean(residual[selb] ** 2)) / inflow[selb].mean():.1e} of the inflow)")
    print(f"energy:   mean net enthalpy outflow {flux['ENTHALPY_FLUX'][selb].mean():.3e}, "
          f"{flux['ENTHALPY_FLUX'][selb].mean() / -sides['west']['integral_ENTHALPY_FLUX'][selb].mean():.1e} "
          "of the inflow")
    print(f"momentum: mean Cd {cd_box[selb].mean():.4f} from the box against {cd[sel].mean():.4f} on the wall; "
          f"Cl amplitude {0.5 * np.ptp(cl_box[selb]):.4f} against {0.5 * np.ptp(cl[sel]):.4f}")
    print(f"Strouhal: {st_lift:.4f} from the lift's crossings, {st_spec:.4f} from the wake spectrum")

    phase_snap = snapshots.last(os.path.join(run, "solut", "box"))
    pts, conn, offs, a = snapshots.read(phase_snap)
    polys, _ = snapshots.cells(pts, conn, offs)
    polys = [p[:, :2] for p in polys]
    bins = sorted({k.split("_")[1] for k in a if k.startswith("PHASE_") and k.endswith("_U_Y")})

    def balance(th):
        fig, (ax, bx) = plt.subplots(2, 1, figsize=(9, 4.8), sharex=True)
        ax.plot(tb[1:-1], residual[1:-1] / inflow[1:-1], color=th["series"][0])
        ax.set_ylabel("mass residual / inflow")
        ax.set_title("box -2 < x < 4, -2 < y < 2: mass balance and drag", fontsize=11)
        bx.plot(t, cd, color=th["series"][0], label="wall ([[forces]])")
        bx.plot(tb[1:-1], cd_box[1:-1], color=th["series"][1], lw=1, label="box momentum balance")
        bx.axvline(args.t_start, color=th["ink2"], ls="--", lw=1)
        bx.set_ylim(1.0, 1.7)
        bx.set_ylabel("C_D")
        bx.set_xlabel("t a / D")
        bx.legend(loc="upper right", fontsize=9)
        return fig

    def spec(th):
        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.semilogy(f * D / U, s / s.max(), color=th["series"][0])
        ax.axvline(st_spec, color=th["ink2"], ls="--", lw=1)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(1e-6, 2)
        ax.set_xlabel("St = f D / U")
        ax.set_ylabel("power of U_Y at (3, 0.5)")
        ax.set_title(f"peak St = {st_spec:.3f}; its harmonics follow", fontsize=11)
        return fig

    def phase(th):
        show = bins[:: max(1, len(bins) // 4)][:4]
        fig, axes = plt.subplots(2, 2, figsize=(9, 5), sharex=True, sharey=True)
        for ax, b in zip(axes.flat, show):
            pc = PolyCollection(polys, array=a[f"PHASE_{b}_U_Y"], cmap="RdBu_r", edgecolors="none",
                                antialiased=False, rasterized=True)
            pc.set_clim(-0.12, 0.12)
            ax.add_collection(pc)
            ax.add_patch(plt.Circle((0, 0), 0.5, color=th["ink2"]))
            ax.set_xlim(-2, 4)
            ax.set_ylim(-2, 2)
            ax.set_aspect("equal")
            ax.grid(False)
            ax.set_title(f"phase bin {int(b)} of {len(bins)}", fontsize=10)
        fig.colorbar(pc, ax=axes, shrink=0.85, label="phase-averaged U_Y / a")
        return fig

    render(balance, args.out, "measuring_balance")
    render(spec, args.out, "measuring_spectrum")
    render(phase, args.out, "measuring_phase")


if __name__ == "__main__":
    main()
