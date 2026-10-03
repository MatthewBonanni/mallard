#!/usr/bin/env python3
"""Exact Sedov-Taylor point blast in 3D, and comparison with Mallard runs.

    sedov.py [--gamma 1.4]                         print xi0 of R = xi0 (E t^2 / rho0)^(1/5)
    sedov.py SOLUT_DIR --energy E [--rho0 1] [-o sedov.png]

The similarity solution behind a strong spherical shock in a uniform gas at
rest: with u = R' f(eta), rho = rho0 g(eta), p = rho0 R'^2 h(eta) and
eta = r / R(t), mass, momentum and entropy conservation reduce to three ODEs,
integrated from the Rankine-Hugoniot state at eta = 1 to the center. Energy
conservation fixes xi0 (1.0328 for gamma = 1.4, 1.1517 for 5/3).

With a series of VTU (or PVTU) snapshots of an octant run of
examples/sedov_3d (total blast energy E in the full sphere), prints the
shock radius against R(t) for every snapshot and plots radial density
scatter against the exact profile at the last one.
"""
import argparse
import glob
import os
import re

import numpy as np
from scipy.integrate import solve_ivp


def similarity_profiles(gamma, n=2000):
    """eta, f, g, h from the shock (eta = 1) inward."""
    def rhs(eta, y):
        f, g, h = y
        A = np.array([[g, f - eta, 0.0],
                      [f - eta, 0.0, 1.0 / g],
                      [0.0, -gamma / g, 1.0 / h]])
        b = np.array([-2.0 * g * f / eta, 1.5 * f, 3.0 / (f - eta)])
        return np.linalg.solve(A, b)

    y1 = [2.0 / (gamma + 1.0), (gamma + 1.0) / (gamma - 1.0), 2.0 / (gamma + 1.0)]
    eta = 1.0 - np.geomspace(1e-12, 1.0 - 1e-4, n)
    sol = solve_ivp(rhs, (1.0, eta[-1]), y1, t_eval=eta, rtol=1e-11, atol=1e-14, method="LSODA")
    return sol.t, sol.y[0], sol.y[1], sol.y[2]


def xi0(gamma):
    """R = xi0 (E t^2 / rho0)^(1/5) from E = 4 pi rho0 R^3 R'^2 int (g f^2 / 2 + h / (gamma - 1)) eta^2."""
    eta, f, g, h = similarity_profiles(gamma)
    integrand = (0.5 * g * f ** 2 + h / (gamma - 1.0)) * eta ** 2
    I = -np.trapezoid(integrand, eta)
    return (25.0 / (16.0 * np.pi * I)) ** 0.2


def exact_density(r, t, energy, rho0, gamma):
    R = xi0(gamma) * (energy * t ** 2 / rho0) ** 0.2
    eta, _, g, _ = similarity_profiles(gamma)
    x = np.asarray(r) / R
    return np.where(x < 1.0, rho0 * np.interp(x, eta[::-1], g[::-1]), rho0), R


def snapshots(solut):
    pvd = glob.glob(os.path.join(solut, "*.pvd"))[0]
    entries = re.findall(r'timestep="([^"]+)"[^>]*file="([^"]+)"', open(pvd).read())
    return [(float(t), os.path.join(solut, f)) for t, f in entries]


def read_cells(path):
    from mallard_vtu import read_vtu_cells
    files = [path]
    if path.endswith(".pvtu"):
        files = [os.path.join(os.path.dirname(path), p)
                 for p in re.findall(r'<Piece Source="([^"]+)"', open(path).read())]
    centers, rho = [], []
    for f in files:
        pts, conn, offs, types, arrays = read_vtu_cells(f)
        starts = np.concatenate([[0], offs[:-1]])
        for size in np.unique(offs - starts):
            sel = np.nonzero(offs - starts == size)[0]
            centers.append(pts[conn[starts[sel, None] + np.arange(size)]].mean(axis=1))
            rho.append(arrays["RHO"][sel])
    return np.vstack(centers), np.concatenate(rho)


def shock_radius(r, rho, rho0):
    """Radius where the shell-averaged density first rises to the midpoint of the
    jump, scanning inward from outside."""
    edges = np.linspace(0.0, r.max(), 400)
    idx = np.digitize(r, edges)
    mean = np.array([rho[idx == k].mean() if np.any(idx == k) else np.nan for k in range(1, len(edges))])
    centers = 0.5 * (edges[1:] + edges[:-1])
    peak = np.nanargmax(mean)
    level = rho0 + 0.5 * (mean[peak] - rho0)
    for k in range(len(mean) - 1, peak, -1):
        if mean[k - 1] >= level > mean[k]:
            return centers[k - 1] + (mean[k - 1] - level) / (mean[k - 1] - mean[k]) * (centers[k] - centers[k - 1])
    return centers[peak]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("solut", nargs="?")
    ap.add_argument("--gamma", type=float, default=1.4)
    ap.add_argument("--energy", type=float, default=1.0)
    ap.add_argument("--rho0", type=float, default=1.0)
    ap.add_argument("-o", "--output", default="sedov.png")
    args = ap.parse_args()
    x0 = xi0(args.gamma)
    print(f"gamma = {args.gamma}: xi0 = {x0:.5f} (alpha = xi0^-5 = {x0 ** -5:.5f})")
    if not args.solut:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snaps = [s for s in snapshots(args.solut) if s[0] > 0]
    print("t, R_numerical, R_exact, relative error")
    times, radii = [], []
    for t, path in snaps:
        c, rho = read_cells(path)
        r = np.linalg.norm(c, axis=1)
        R_num = shock_radius(r, rho, args.rho0)
        R_ex = x0 * (args.energy * t ** 2 / args.rho0) ** 0.2
        times.append(t)
        radii.append(R_num)
        print(f"{t:.4f} {R_num:.4f} {R_ex:.4f} {R_num / R_ex - 1:+.4f}")

    t_last = snaps[-1][0]
    fig, (ax_r, ax_p) = plt.subplots(1, 2, figsize=(11, 4.2))
    ts = np.linspace(0, max(times) * 1.05, 200)
    ax_r.plot(ts, x0 * (args.energy * ts ** 2 / args.rho0) ** 0.2, "k-", label="Sedov-Taylor")
    ax_r.plot(times, radii, "o", ms=4, label="Mallard")
    ax_r.set(xlabel="t", ylabel="Shock radius R", title="Shock radius")
    rr = np.linspace(0, r.max(), 1000)
    rho_ex, R = exact_density(rr, t_last, args.energy, args.rho0, args.gamma)
    ax_p.plot(r, rho, ",", color="C0", alpha=0.3)
    ax_p.plot([], [], ".", color="C0", label="Mallard cells")
    ax_p.plot(rr, rho_ex, "k-", label="Exact")
    ax_p.set(xlabel="r", ylabel="Density", title=f"t = {t_last:g}")
    for ax in (ax_r, ax_p):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
