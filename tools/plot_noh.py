#!/usr/bin/env python3
"""Spherical Noh problem (gamma = 5/3) from an axisymmetric run against the
exact solution: density against the spherical radius, the shock radius in
angular sectors of the meridian plane, the post-shock plateau and the
pre-shock compression.

    plot_noh.py OUTPUT.png RUN.vtu
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mallard_vtu import read_vtu_cells


def main():
    output, path = sys.argv[1], sys.argv[2]
    pts, conn, offs, _, data = read_vtu_cells(path)
    t = data["TIME"]
    starts = np.concatenate([[0], offs[:-1]])
    centers = np.array([pts[conn[s:e], :2].mean(axis=0) for s, e in zip(starts, offs)])
    R = np.hypot(centers[:, 0], centers[:, 1])
    theta = np.degrees(np.arctan2(centers[:, 1], centers[:, 0]))
    rho = data["RHO"]
    R_shock = t / 3.0

    # Shock radius per 15-degree sector: where rho crosses the middle of the jump at the shock,
    # from 64 down to the pre-shock (1 + t / R)^2 = 16
    print(f"t = {t:g}: exact shock radius {R_shock:.4f}")
    for lo in range(0, 90, 15):
        sel = (theta >= lo) & (theta < lo + 15) & (R < 2 * R_shock)
        order = np.argsort(R[sel])
        r_s, rho_s = R[sel][order], rho[sel][order]
        k = np.nonzero((rho_s[:-1] >= 40.0) & (rho_s[1:] < 40.0))[0]
        R_num = np.interp(40.0, [rho_s[k[-1] + 1], rho_s[k[-1]]], [r_s[k[-1] + 1], r_s[k[-1]]]) if len(k) else np.nan
        print(f"  theta {lo:2d}-{lo + 15:2d} deg: shock at {R_num:.4f}")
    plateau = (R > 0.25 * R_shock) & (R < 0.75 * R_shock)
    print(f"post-shock density (R in [{0.25 * R_shock:.3f}, {0.75 * R_shock:.3f}]): mean {rho[plateau].mean():.3f}, "
          f"min {rho[plateau].min():.3f}, max {rho[plateau].max():.3f} (exact 64)")
    ahead = (R > 1.3 * R_shock) & (R < 0.9)
    pre = (1 + t / R[ahead]) ** 2
    print(f"pre-shock density: max relative error {np.abs(rho[ahead] / pre - 1).max():.3e}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(R, rho, ",", alpha=0.4, label="Mallard cells")
    rr = np.linspace(1e-3, R.max(), 1000)
    ax.plot(rr, np.where(rr < R_shock, 64.0, (1 + t / rr) ** 2), "k-", lw=1, label="exact")
    ax.set(xlabel="R = sqrt(x^2 + r^2)", ylabel="density", title=f"Spherical Noh, t = {t:g}")
    ax.legend()
    fig.savefig(output, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
