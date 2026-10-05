#!/usr/bin/env python3
"""Steady axisymmetric flow past a sphere (examples/sphere_axisymmetric):
drag coefficient, separation angle and wake length against the literature,
and the drag history.

    plot_sphere_axisymmetric.py OUTPUT.png SOLUT_DIR --re 100 [--U 0.2] [--D 1]

The forces are per radian of the revolved sphere (2 pi times them is the
force on the sphere). The separation angle is measured from the front
stagnation point, where the wall shear of the first ring of cells changes
sign; the wake length L is the distance from the rear of the sphere to where
the axial velocity next to the axis turns positive again.
"""
import argparse
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mallard_vtu import read_vtu_cells


def clift(re):
    """Correlations of Clift, Grace & Weber (1978) for 20 < Re < 260: drag and separation angle."""
    cd = 24.0 / re * (1.0 + 0.1935 * re ** 0.6305)
    theta = 180.0 - 42.5 * np.log(re / 20.0) ** 0.483
    return cd, theta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("solut")
    ap.add_argument("--re", type=float, required=True)
    ap.add_argument("--U", type=float, default=0.2)
    ap.add_argument("--D", type=float, default=1.0)
    args = ap.parse_args()
    U, D = args.U, args.D

    forces = np.loadtxt(os.path.join(args.solut, "forces_sphere.csv"), delimiter=",", ndmin=2, skiprows=1)
    t = forces[:, 1]
    cd = 2.0 * np.pi * (forces[:, 2] + forces[:, 4]) / (0.5 * U ** 2 * np.pi * D ** 2 / 4)
    cd_p = 2.0 * np.pi * forces[:, 2] / (0.5 * U ** 2 * np.pi * D ** 2 / 4)

    pvd = glob.glob(os.path.join(args.solut, "sphere*.pvd"))[0]
    last = re.findall(r'file="([^"]+)"', open(pvd).read())[-1]
    pts, conn, offs, _, data = read_vtu_cells(os.path.join(args.solut, last))
    starts = np.concatenate([[0], offs[:-1]])
    cells = [pts[conn[s:e], :2] for s, e in zip(starts, offs)]
    center = np.array([c.mean(axis=0) for c in cells])
    r_min = np.array([np.hypot(c[:, 0], c[:, 1]).min() for c in cells])
    y_min = np.array([c[:, 1].min() for c in cells])
    ux, ur = data["U_X"], data["U_Y"]

    # Separation: tangential velocity of the wall ring of cells (polar angle phi from the rear axis)
    ring = r_min < 0.5 * D * (1 + 1e-9)
    phi = np.degrees(np.arctan2(center[ring, 1], center[ring, 0]))
    ph = np.radians(phi)
    u_t = ux[ring] * np.sin(ph) - ur[ring] * np.cos(ph)  # positive from the front toward the rear
    order = np.argsort(phi)
    phi, u_t = phi[order], u_t[order]
    k = np.nonzero((u_t[:-1] < 0) & (u_t[1:] >= 0))[0]
    theta_sep = 180.0 - np.interp(0.0, [u_t[k[-1]], u_t[k[-1] + 1]], [phi[k[-1]], phi[k[-1] + 1]]) if len(k) else np.nan

    # Wake: axial velocity in the cells next to the axis behind the sphere
    axis = (y_min < 1e-12) & (center[:, 0] > 0)
    order = np.argsort(center[axis, 0])
    xa, ua = center[axis, 0][order], ux[axis][order]
    k = np.nonzero((ua[:-1] < 0) & (ua[1:] >= 0))[0]
    x_end = np.interp(0.0, [ua[k[0]], ua[k[0] + 1]], [xa[k[0]], xa[k[0] + 1]]) if len(k) else np.nan
    wake = (x_end - 0.5 * D) / D

    cd_ref, theta_ref = clift(args.re)
    window = t > t[-1] - 20.0
    print(f"Re = {args.re:g}, t = {t[-1]:g}")
    print(f"  Cd = {cd[-1]:.4f} (pressure {cd_p[-1]:.4f}; variation over the last 20 time units "
          f"{np.ptp(cd[window]):.1e}); Clift, Grace & Weber: {cd_ref:.4f}")
    print(f"  separation angle = {theta_sep:.1f} deg from the front; Clift, Grace & Weber: {theta_ref:.1f}")
    print(f"  wake length L / D = {wake:.3f}")

    fig, (ax_c, ax_u) = plt.subplots(1, 2, figsize=(11, 4))
    ax_c.plot(t, cd, label="Mallard")
    ax_c.axhline(cd_ref, color="k", ls="--", lw=0.8, label="Clift, Grace & Weber")
    ax_c.set(xlabel="t", ylabel="Cd", ylim=(0.5 * cd_ref, 1.5 * cd_ref), title=f"Re = {args.re:g}")
    ax_c.legend()
    ax_u.plot(xa / D, ua / U, ".-", ms=2)
    ax_u.axhline(0.0, color="k", lw=0.5)
    ax_u.set(xlabel="x / D", ylabel="u_x / U on the axis", xlim=(0.5, 5), title=f"wake length L / D = {wake:.2f}")
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)


if __name__ == "__main__":
    main()
