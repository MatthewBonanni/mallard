#!/usr/bin/env python3
"""Bow-shock standoff of supersonic flow over a sphere against Billig's correlation.

    plot_sphere.py SLICES_DIR [--mach 3] [--gamma 1.4] [-o sphere.png]

SLICES_DIR holds the z0 series of examples/sphere_mach3 (the meridian plane
z = 0, sphere of radius 0.5 at the origin, free stream along +x with p = 1).
Along the stagnation line the shock is where the pressure crosses the middle
of the normal-shock jump. Prints the standoff Delta / R of every snapshot
against Billig (J. Spacecraft Rockets 4(6), 1967),
Delta / R = 0.143 exp(3.24 / M^2), and plots its history with the pressure
along the stagnation line at the last snapshot.
"""
import argparse
import os
import re

import numpy as np

R_SPHERE = 0.5


def billig(mach):
    return 0.143 * np.exp(3.24 / mach ** 2)


def normal_shock_pressure(mach, gamma):
    return 1.0 + 2.0 * gamma / (gamma + 1.0) * (mach ** 2 - 1.0)


def series(slices, name="z0"):
    text = open(os.path.join(slices, name + ".pvd")).read()
    entries = re.findall(r'timestep="([^"]+)"[^>]*file="([^"]+)"', text)
    # A run still in progress may list a snapshot not yet copied
    return [(float(t), os.path.join(slices, f)) for t, f in entries if os.path.exists(os.path.join(slices, f))]


def stagnation_line(mesh, half_width=0.03):
    """x and pressure of the cells along the stagnation line, upstream of the sphere."""
    c = mesh.cell_centers().points
    sel = (np.abs(c[:, 1]) < half_width) & (c[:, 0] < -R_SPHERE)
    order = np.argsort(c[sel, 0])
    return c[sel, 0][order], np.asarray(mesh.cell_data["P"])[sel][order]


def shock_x(x, p, level):
    """Most upstream crossing of the pressure level, scanning downstream."""
    for k in range(len(x) - 1):
        if p[k] < level <= p[k + 1]:
            return x[k] + (level - p[k]) / (p[k + 1] - p[k]) * (x[k + 1] - x[k])
    return np.nan


def standoff(mesh, mach, gamma):
    x, p = stagnation_line(mesh)
    xs = shock_x(x, p, 0.5 * (1.0 + normal_shock_pressure(mach, gamma)))
    return (-R_SPHERE - xs) / R_SPHERE


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pyvista as pv

    ap = argparse.ArgumentParser()
    ap.add_argument("slices")
    ap.add_argument("--mach", type=float, default=3.0)
    ap.add_argument("--gamma", type=float, default=1.4)
    ap.add_argument("-o", "--output", default="sphere.png")
    args = ap.parse_args()

    ref = billig(args.mach)
    snaps = [s for s in series(args.slices) if s[0] > 0]
    times, deltas = [], []
    for t, path in snaps:
        mesh = pv.read(path)
        times.append(t)
        deltas.append(standoff(mesh, args.mach, args.gamma))
    print(f"Billig: Delta / R = {ref:.4f}")
    for t, d in list(zip(times, deltas))[:: max(1, len(times) // 20)] + [(times[-1], deltas[-1])]:
        print(f"t = {t:.2f}: Delta / R = {d:.4f} ({d / ref - 1:+.1%})")

    fig, (ax_t, ax_x) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax_t.plot(times, deltas, label="Mallard")
    ax_t.axhline(ref, color="k", ls="--", label="Billig")
    ax_t.set(xlabel="t", ylabel="$\\Delta / R$", title="Shock standoff")
    x, p = stagnation_line(mesh)
    ax_x.plot(x, p, ".-", ms=3, label="Mallard")
    ax_x.axvline(-R_SPHERE * (1 + ref), color="k", ls="--", label="Billig shock")
    ax_x.set(xlabel="x", ylabel="p / $p_\\infty$", title=f"Stagnation line, t = {times[-1]:g}")
    for ax in (ax_t, ax_x):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
