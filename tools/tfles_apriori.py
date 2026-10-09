#!/usr/bin/env python3
"""A priori test of the TFLES efficiency on the DNS of examples/flame_turbulence.

    tfles_apriori.py DNS.restart [--ratios 4,8,16] [--n-res 5] [--beta 0.5]

From a DNS restart file (224 x 128 x 128 cells of h = delta_L / 10), for each
LES ratio R (LES cells of R^3 DNS cells, Delta = R h, F = n_res Delta /
delta_L):

- the resolved wrinkling the efficiency must restore, Xi = <|grad c|> /
  |grad <c>| with <.> the top-hat filter of width F delta_L (the thickened
  flame's filter width) over y, z (periodic) and x, summed over the flame
  (0.05 < <c> < 0.95) with weight |grad <c>|: the ratio of the DNS flame
  surface to the filtered one within each filter volume;
- what the model predicts from the LES-resolved field, the box-filtered DNS
  on the LES mesh: Colin et al.'s subgrid velocity u' = 2 Delta^3
  |lap(curl u)| (as in Mallard), Charlette et al.'s E(F, u' / S_L, Re) with
  beta, weighted the same way; also with u' at the scale F delta_L,
  u' (F delta_L / Delta)^(1/3) (Kolmogorov scaling from the grid scale).

c = (Y_H2,u - Y_H2) / (Y_H2,u - Y_H2,b). Prints one row per ratio.
"""
import argparse
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isotropic_turbulence_restart import read_restart  # noqa: E402

DELTA_L, S_L = 3.301397619449605e-04, 2.33237442
Y_U, Y_B = 2.85223875e-02, 1.16716361e-03
NU_U = 2.16e-5
C_K = 1.5


def charlette(F, u, nu, beta):
    """Charlette, Meneveau & Veynante (2002) efficiency, as ThickenedFlame::wrinkling."""
    F = np.maximum(F, 1.0 + 1e-12)
    pi43 = np.pi ** (4.0 / 3.0)
    r = np.maximum(u, 1e-12) / S_L
    re = np.maximum(u, 1e-12) * F * DELTA_L / nu
    f_u = 4.0 * np.sqrt(27.0 * C_K / 110.0) * (18.0 * C_K / 55.0) * r * r
    f_d = np.sqrt(27.0 * C_K * pi43 / 110.0 * (F ** (4.0 / 3.0) - 1.0))
    f_re = np.sqrt(9.0 / 55.0 * np.exp(-1.5 * C_K * pi43 / re)) * np.sqrt(re)
    a = 0.6 + 0.2 * np.exp(-0.1 * r) - 0.2 * np.exp(-0.01 * F)
    inner = (f_u ** -a + f_d ** -a) ** (-1.0 / a)
    gamma = (inner ** -1.4 + f_re ** -1.4) ** (-1.0 / 1.4)
    return (1.0 + np.minimum(F - 1.0, gamma * r)) ** beta


def block(a, r):
    nx, ny, nz = a.shape
    return a[: nx // r * r].reshape(nx // r, r, ny // r, r, nz // r, r).mean(axis=(1, 3, 5))


def grad(a, h):
    """Central differences, periodic in y and z, one-sided at the x ends."""
    gx = np.gradient(a, h, axis=0)
    gy = (np.roll(a, -1, 1) - np.roll(a, 1, 1)) / (2 * h)
    gz = (np.roll(a, -1, 2) - np.roll(a, 1, 2)) / (2 * h)
    return gx, gy, gz


def laplacian(a, h):
    lx = np.gradient(np.gradient(a, h, axis=0), h, axis=0)
    ly = (np.roll(a, -1, 1) - 2 * a + np.roll(a, 1, 1)) / h ** 2
    lz = (np.roll(a, -1, 2) - 2 * a + np.roll(a, 1, 2)) / h ** 2
    return lx + ly + lz


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("restart")
    ap.add_argument("--ratios", default="4,8,16")
    ap.add_argument("--n-res", type=float, default=5.0)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--shape", default="224,128,128")
    args = ap.parse_args()
    f, t, _ = read_restart(args.restart)
    shape = tuple(int(v) for v in args.shape.split(","))
    h = DELTA_L / 10.0
    rho = f["RHO"].reshape(shape)
    u = [f[f"RHOU_{c}"].reshape(shape) / rho for c in "XYZ"]
    T = f["T_SEED"].reshape(shape)
    c = np.clip((Y_U - f["RHOY_H2"].reshape(shape) / rho) / (Y_U - Y_B), 0.0, 1.0)
    gx, gy, gz = grad(c, h)
    mag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    print(f"t = {t * 1e3:.3f} ms; DNS flame surface int |grad c| dV / L^2 = {mag.sum() * h / (shape[1] * shape[2]):.3f}")
    print("R  Delta/dL   F   Xi(DNS)  E(Colin)  E(Colin at F dL)  u'_Delta/S_L  E^(1/beta)-1 (beta=1)  beta fit")
    for r in [int(v) for v in args.ratios.split(",")]:
        F = max(1.0, args.n_res * r / 10.0)
        width = int(round(F * 10))  # F delta_L in DNS cells
        modes = ("nearest", "wrap", "wrap")
        cf = ndimage.uniform_filter(c, size=width, mode=modes)
        magf = ndimage.uniform_filter(mag, size=width, mode=modes)
        fx, fy, fz = grad(cf, h)
        gradf = np.sqrt(fx ** 2 + fy ** 2 + fz ** 2)
        flame = (cf > 0.05) & (cf < 0.95)
        xi = magf[flame].sum() / gradf[flame].sum()
        # The model on the LES mesh
        H = r * h
        rho_l = block(rho, r)
        ul = [block(rho * v, r) / rho_l for v in u]
        cl = block(c, r)
        Tl = block(T, r)
        g = [grad(v, H) for v in ul]  # g[i][j] = d u_i / d x_j
        w = [g[2][1] - g[1][2], g[0][2] - g[2][0], g[1][0] - g[0][1]]
        lap = np.sqrt(sum(laplacian(wi, H) ** 2 for wi in w))
        u_prime = 2.0 * H ** 3 * lap
        theta = np.clip((Tl - 300.0) / (2384.28 - 300.0), 0, 1)
        omega = np.minimum(1.0, theta * (1 - theta) / (0.05 * 0.95))
        Fl = 1.0 + (F - 1.0) * omega
        nu = NU_U * (Tl / 300.0) ** 1.7
        E = charlette(Fl, u_prime, nu, args.beta)
        E_scaled = charlette(Fl, u_prime * (F * DELTA_L / H) ** (1.0 / 3.0), nu, args.beta)
        lx, ly, lz = grad(cl, H)
        wgt = np.sqrt(lx ** 2 + ly ** 2 + lz ** 2) * ((cl > 0.05) & (cl < 0.95))
        # Colin's estimate against the DNS's own subgrid velocity at the scale F delta_L, in the fresh gas
        fresh = (cl < 0.01) & (Tl < 320.0)
        res2 = 0.0
        for v in u:
            vf = ndimage.uniform_filter(v, size=width, mode=modes)
            res2 = res2 + ndimage.uniform_filter((v - vf) ** 2, size=width, mode=modes)
        u_true = np.sqrt(block(res2 / 3.0, r))
        colin_fresh = u_prime[fresh].mean() / S_L
        # Sigma's eddy viscosity on the same field: nu_t / Delta is a velocity at the grid scale
        G = np.empty(Tl.shape + (3, 3))
        for i in range(3):
            for j in range(3):
                G[..., i, j] = g[i][j]
        sv = np.linalg.svd(G, compute_uv=False)
        D = np.where(sv[..., 0] > 0, sv[..., 2] * (sv[..., 0] - sv[..., 1]) * (sv[..., 1] - sv[..., 2]) /
                     np.maximum(sv[..., 0], 1e-30) ** 2, 0.0)
        nut = (1.35 * H) ** 2 * D
        sigma_fresh = (nut / H)[fresh].mean() / S_L
        true_fresh = u_true[fresh].mean() / S_L
        e1 = np.sum(charlette(Fl, u_prime, nu, 1.0) * wgt) / wgt.sum()
        beta_fit = np.log(xi) / np.log(e1) if e1 > 1.0 + 1e-9 else np.nan
        print(f"{r:2d} {H / DELTA_L:6.2f} {F:5.1f} {xi:8.3f} {np.sum(E * wgt) / wgt.sum():9.3f} "
              f"{np.sum(E_scaled * wgt) / wgt.sum():17.3f} {np.sum(u_prime * wgt) / wgt.sum() / S_L:12.3f} "
              f"{e1 - 1:22.3f} {beta_fit:9.3f}   fresh gas: Colin u'/S_L {colin_fresh:.3f}, Sigma nu_t/(Delta S_L) {sigma_fresh:.3f}, DNS u'(F dL)/S_L {true_fresh:.3f}")


if __name__ == "__main__":
    main()
