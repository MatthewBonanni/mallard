#!/usr/bin/env python3
"""The heated-duct example (examples/heated_duct): the steady run against
Rayleigh flow, and the closed duct's mass against mdot t for the mass-flow
inlet and for upt.

    plot_heated_duct.py STEADY.vtu CLOSED_PREFIX CLOSED_UPT_PREFIX [--figure]

STEADY.vtu is the last snapshot of input.toml; the prefixes are those of the
closed runs' series (closed.toml, closed_upt.toml). Prints the inlet pressure
and exit state against Rayleigh flow and the closed duct's mass balance;
--figure draws docs/images/guide/{heated_duct,mass_flow_held}-{light,dark}.svg.
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs"))
import figstyle  # noqa: E402
from mallard_vtu import read_vtu_cells  # noqa: E402

GAMMA, R = 1.4, 287.0
CP = GAMMA * R / (GAMMA - 1.0)
G, T1, P_OUT = 100.0, 300.0, 1.0e5   # input.toml
Q, X0, X1 = 1.0e8, 0.4, 0.6
MDOT_CLOSED, AREA = 2.0, 0.1         # closed.toml


def state_after(p1, heat):
    """Rayleigh flow: [p, u, T] after heat per unit area `heat`, from the inlet at p1."""
    u1 = G * R * T1 / p1
    impulse = p1 + G * u1
    H = CP * T1 + 0.5 * u1 * u1 + heat / G
    a, b = 0.5 - CP / R, CP * impulse / (G * R)
    u = (-b + np.sqrt(b * b + 4.0 * a * H)) / (2.0 * a)
    return impulse - G * u, u, (impulse - G * u) * u / (G * R)


def rayleigh_inlet_pressure():
    lo, hi = P_OUT, 3.0 * P_OUT
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if state_after(mid, Q * (X1 - X0))[0] > P_OUT:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def cells(path):
    pts, conn, offs, types, arrays = read_vtu_cells(path)
    starts = np.concatenate([[0], offs[:-1]])
    quads = np.array([conn[s:e] for s, e in zip(starts, offs)])
    xy = pts[quads][:, :, :2]
    x0, y0 = xy[..., 0], xy[..., 1]
    x1, y1 = np.roll(x0, -1, axis=1), np.roll(y0, -1, axis=1)
    area = 0.5 * (x0 * y1 - x1 * y0).sum(axis=1)
    cx = xy[..., 0].mean(axis=1)
    order = np.argsort(cx)
    return cx[order], area[order], {k: (v[order] if np.ndim(v) else v) for k, v in arrays.items()}


def series(prefix):
    files = sorted(glob.glob(prefix + "_[0-9]*.vtu"))
    t, mass, rho_in = [], [], []
    for f in files:
        x, area, a = cells(f)
        t.append(a["TIME"])
        mass.append(np.sum(a["RHO"] * area))
        rho_in.append(a["RHO"][0])
    t, mass = np.array(t), np.array(mass)
    return t, mass - mass[0], np.array(rho_in)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("steady")
    ap.add_argument("closed")
    ap.add_argument("closed_upt")
    ap.add_argument("--figure", action="store_true")
    args = ap.parse_args()

    p1 = rayleigh_inlet_pressure()
    x, _, s = cells(args.steady)
    heat = Q * np.clip(x - X0, 0.0, X1 - X0)
    exact = np.array([state_after(p1, h) for h in heat])
    for name, key, col in (("p", "P", 0), ("u", "U_X", 1), ("T", "T", 2)):
        err = np.max(np.abs(s[key] - exact[:, col]) / np.abs(exact[:, col]))
        print(f"{name}: max relative difference from Rayleigh flow {err:.2e}")
    print(f"inlet pressure {s['P'][0]:.1f} Pa, Rayleigh {p1:.1f}; exit T {s['T'][-1]:.2f} K, Rayleigh {exact[-1, 2]:.2f}")

    t, added, rho_in = series(args.closed)
    t_u, added_u, rho_in_u = series(args.closed_upt)
    held = np.max(np.abs(added[1:] / (MDOT_CLOSED * t[1:]) - 1.0))
    drift = added_u[-1] / (MDOT_CLOSED * t_u[-1]) - 1.0
    print(f"closed duct: mass added within {held:.1e} of mdot t (inlet density {rho_in[0]:.3f} -> {rho_in[-1]:.3f}); "
          f"upt {100 * drift:+.1f}% after {1e3 * t_u[-1]:.0f} ms")
    if not args.figure:
        return

    def heated(P):
        fig, axes = figstyle.plt.subplots(1, 3, figsize=(11.0, 3.4), gridspec_kw=dict(wspace=0.32))
        xe = np.linspace(0.0, 1.0, 401)
        he = Q * np.clip(xe - X0, 0.0, X1 - X0)
        ex = np.array([state_after(p1, h) for h in he])
        every = slice(None, None, max(1, len(x) // 50))
        for ax, (label, key, col, scale, color) in zip(axes, (("p [kPa]", "P", 0, 1e-3, P["blue"]),
                                                              ("u [m/s]", "U_X", 1, 1.0, P["flame"]),
                                                              ("T [K]", "T", 2, 1.0, P["teal"]))):
            ax.axvspan(X0, X1, color=P["faint"], lw=0)
            ax.plot(xe, scale * ex[:, col], color=color, label="Rayleigh flow")
            ax.plot(x[every], scale * s[key][every], "o", ms=3.5, mfc="none", mew=1.0, color=color, label="Mallard")
            ax.set_xlabel("x [m]")
            ax.set_ylabel(label)
            ax.set_xlim(0.0, 1.0)
        axes[0].legend(loc="upper right")
        return fig

    def held_fig(P):
        fig, (ax0, ax1) = figstyle.plt.subplots(1, 2, figsize=(10.0, 3.6), gridspec_kw=dict(wspace=0.32))
        ms = 1e3
        ax0.plot(ms * t, MDOT_CLOSED * t, color=P["muted"], lw=1.2, ls="--", label=r"$\dot m\, t$")
        ax0.plot(ms * t, added, "o", ms=4, mfc="none", mew=1.1, color=P["teal"], label="mass_flow_inlet")
        ax0.plot(ms * t_u, added_u, "s", ms=4, mfc="none", mew=1.1, color=P["flame"], label="upt")
        ax0.set_xlabel("t [ms]")
        ax0.set_ylabel("mass added [kg per m depth]")
        ax0.legend(loc="upper left")
        ax1.plot(ms * t, rho_in, color=P["teal"], label="mass_flow_inlet")
        ax1.plot(ms * t_u, rho_in_u, color=P["flame"], label="upt")
        ax1.set_xlabel("t [ms]")
        ax1.set_ylabel(r"inlet density [kg/m$^3$]")
        ax1.legend(loc="upper left")
        return fig

    figstyle.save("heated_duct", heated)
    figstyle.save("mass_flow_held", held_fig)


if __name__ == "__main__":
    main()
