#!/usr/bin/env python3
"""Strouhal number, mean drag and lift of the sphere at Re = 300 against the literature.

    plot_sphere_re300.py FORCES.csv [FORCES.csv ...] [--t-start 600] [--u 0.2]
        [--label NAME ...] [-o sphere_re300.png]

FORCES.csv is the [[forces]] output of examples/sphere_re300 (sphere of
diameter 1, free-stream density 1 and speed u along +x). The coefficients are
C = F / (rho u^2 pi D^2 / 8). The lift is the force component normal to the
stream; its mean direction defines the plane of symmetry of the wake, and the
Strouhal number St = f D / u is the frequency of the lift along that direction,
from the upward zero crossings of its fluctuation over a whole number of
periods after t-start. Averages are taken
over the same whole periods. Prints a table against the references and plots
Cd(t) and Cl(t) of every run.
"""
import argparse

import numpy as np

REFERENCES = [
    # name, Cd, Cl, St
    ("Johnson & Patel 1999", 0.656, 0.069, 0.137),
    ("Tomboulides et al. 1993", 0.671, None, 0.136),
    ("Kim, Kim & Choi 2001", 0.657, 0.067, 0.134),
    ("Constantinescu & Squires 2003", 0.655, 0.065, 0.136),
]


def load(path, u):
    # step, t, pressure force (3), viscous force (3); a run resumed from a
    # restart appends rows without a header
    data = np.loadtxt(path, delimiter=",", comments="step")
    q = 0.5 * u ** 2 * np.pi / 4.0
    t = data[:, 1]
    F = data[:, 2:5] + data[:, 5:8]
    # A restarted run repeats the steps after the restart: keep the last occurrence
    _, last = np.unique(t[::-1], return_index=True)
    keep = np.sort(len(t) - 1 - last)
    return t[keep], F[keep] / q


def analyze(t, C, t_start):
    sel = t >= t_start
    t, C = t[sel], C[sel]
    cd = C[:, 0]
    lift = C[:, 1:]
    direction = lift.mean(0)
    direction /= np.linalg.norm(direction)
    cl = lift @ direction
    # Whole periods between the first and last upward zero crossings of the
    # lift fluctuation
    f = cl - cl.mean()
    up = np.where((f[:-1] < 0) & (f[1:] >= 0))[0]
    t_up = t[up] - f[up] * (t[up + 1] - t[up]) / (f[up + 1] - f[up])
    if len(t_up) < 3:
        raise SystemExit(f"fewer than two lift periods after t = {t_start}")
    n_periods = len(t_up) - 1
    period = (t_up[-1] - t_up[0]) / n_periods
    window = (t >= t_up[0]) & (t <= t_up[-1])
    tw = t[window]

    def mean(v):
        return np.trapezoid(v[window], tw) / (tw[-1] - tw[0])

    cl_vec = np.array([mean(lift[:, 0]), mean(lift[:, 1])])
    return {
        "period": period, "n_periods": n_periods, "t0": t_up[0], "t1": t_up[-1],
        "cd": mean(cd), "cl": np.linalg.norm(cl_vec),
        "cd_amp": 0.5 * (cd[window].max() - cd[window].min()),
        "cl_amp": 0.5 * (cl[window].max() - cl[window].min()),
        "angle": np.degrees(np.arctan2(direction[1], direction[0])),
        "t": t, "cd_t": cd, "cl_t": cl,
    }


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("forces", nargs="+")
    ap.add_argument("--t-start", type=float, default=600.0)
    ap.add_argument("--u", type=float, default=0.2)
    ap.add_argument("--label", nargs="*")
    ap.add_argument("-o", "--output", default="sphere_re300.png")
    args = ap.parse_args()
    labels = args.label or args.forces

    print(f"{'':32s} {'St':>7s} {'Cd':>7s} {'Cl':>7s}")
    for name, cd, cl, st in REFERENCES:
        print(f"{name:32s} {st:7.3f} {cd:7.3f} {cl if cl is not None else float('nan'):7.3f}")
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    t_max = 0.0
    for path, label in zip(args.forces, labels):
        t, C = load(path, args.u)
        r = analyze(t, C, args.t_start)
        st = 1.0 / (r["period"] * args.u)
        print(f"{label:32s} {st:7.4f} {r['cd']:7.4f} {r['cl']:7.4f}"
              f"   ({r['n_periods']} periods, t = {r['t0']:.0f}-{r['t1']:.0f};"
              f" amplitudes Cd {r['cd_amp']:.4f}, Cl {r['cl_amp']:.4f}; lift at {r['angle']:.0f} deg from y)")
        tu = t * args.u
        t_max = max(t_max, tu[-1])
        axes[0].plot(tu, C[:, 0], lw=1, label=label)
        lift_dir = C[:, 1:] @ np.array([np.cos(np.radians(r["angle"])), np.sin(np.radians(r["angle"]))])
        axes[1].plot(tu, lift_dir, lw=1, label=label)
    for ax, name in zip(axes, ("drag", "lift")):
        ref = REFERENCES[0][1] if name == "drag" else REFERENCES[0][2]
        ax.axhline(ref, color="k", ls="--", lw=0.8, label="Johnson & Patel 1999 (mean)")
        ax.axvspan(args.t_start * args.u, t_max, color="0.9", zorder=0)
        ax.set_xlim(0, t_max)
        ax.set_ylabel(f"C_{name[0].upper()}")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("t U / D")
    axes[0].set_ylim(0.6, 0.75)
    axes[1].set_ylim(0.0, 0.15)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
