#!/usr/bin/env python3
"""Kinetic energy and dissipation rate of the Taylor-Green vortex against a reference.

    plot_taylor_green.py integrals.csv [more.csv ...] [--label NAME ...]
        [--ref spectral_Re1600_512.gdiag] [--mu 0.000625] [-o tgv.png]

Each CSV is the [integrals] output of a Taylor-Green run on the octant
[0, pi]^3 (rho0 = V0 = L = 1), or with --full-box on the periodic box of side
2 pi. Prints, per run, the peak of eps = -dE/dt and of the enstrophy-based
eps = 2 mu E_omega, with the reference's.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OCTANT_VOLUME = np.pi ** 3
BOX_VOLUME = (2 * np.pi) ** 3


def load(path, mu, volume=OCTANT_VOLUME):
    """Time, normalized kinetic energy, -dE/dt and 2 mu enstrophy."""
    d = np.genfromtxt(path, delimiter=",", names=True)
    t, i = np.unique(d["t"], return_index=True)
    E = d["kinetic_energy"][i] / volume
    eps = -np.gradient(E, t)
    eps_omega = 2.0 * mu * d["enstrophy"][i] / volume
    return t, E, eps, eps_omega


def peak(t, y):
    """Peak of y(t), refined by a parabola through the three samples around it."""
    k = int(np.argmax(y))
    if 0 < k < len(y) - 1:
        a, b, _ = np.polyfit(t[k - 1:k + 2], y[k - 1:k + 2], 2)
        tp = -b / (2 * a)
        return np.polyval([a, b, _], tp), tp
    return y[k], t[k]


def load_reference(path):
    d = np.loadtxt(path, comments="#")
    return d[:, 0], d[:, 1], d[:, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--label", nargs="*")
    ap.add_argument("--ref")
    ap.add_argument("--mu", type=float, default=1.0 / 1600)
    ap.add_argument("--full-box", action="store_true", help="runs on the periodic box of side 2 pi")
    ap.add_argument("-o", "--output", default="tgv.png")
    args = ap.parse_args()
    labels = args.label or args.csv

    fig, (ax_e, ax_d) = plt.subplots(1, 2, figsize=(11, 4.2))
    if args.ref:
        tr, Er, er = load_reference(args.ref)
        pv, pt = peak(tr, er)
        print(f"reference: peak eps = {pv:.5f} at t = {pt:.2f}")
        ax_e.plot(tr, Er, "k-", lw=2.2, label="Spectral DNS 512$^3$")
        ax_d.plot(tr, er, "k-", lw=2.2, label="Spectral DNS 512$^3$")
    for path, label in zip(args.csv, labels):
        t, E, eps, eps_w = load(path, args.mu, BOX_VOLUME if args.full_box else OCTANT_VOLUME)
        pv, pt = peak(t, eps)
        wv, wt = peak(t, eps_w)
        print(f"{label}: peak -dE/dt = {pv:.5f} at t = {pt:.2f}; "
              f"peak 2 mu enstrophy = {wv:.5f} at t = {wt:.2f}")
        line, = ax_e.plot(t, E, lw=1.5, label=label)
        ax_d.plot(t, eps, lw=1.5, color=line.get_color(), label=f"{label}: $-dE/dt$")
        ax_d.plot(t, eps_w, lw=1.2, ls="--", color=line.get_color(),
                  label=f"{label}: $2\\mu\\,\\mathcal{{E}}$")
    ax_e.set(xlabel="$t\\,V_0/L$", ylabel="$E_k$", xlim=(0, 20))
    ax_d.set(xlabel="$t\\,V_0/L$", ylabel="$\\epsilon$", xlim=(0, 20), ylim=(0, None))
    for ax in (ax_e, ax_d):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
