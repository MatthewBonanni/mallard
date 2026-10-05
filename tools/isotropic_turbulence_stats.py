#!/usr/bin/env python3
"""Statistics and spectra of decaying isotropic turbulence runs from their restart files.

    isotropic_turbulence_stats.py RESTART [RESTART ...] -o stats.csv
        [--spectra DIR] [--fields DIR --fields-n 128] [--filter-n 64]
        [--no-deconvolve] [--k0 4] [--mt 0.6] [--re-lambda 100] [--gamma 1.4]

Each RESTART is a snapshot of examples/isotropic_turbulence on N^3 hexahedra
of the box [0, 2 pi]^3 (see tools/isotropic_turbulence_restart.py for the
parameters). The cell averages are first turned into point values by
dividing each Fourier mode by its cell-average factor sinc(k_i h / 2)
(exact for a band-limited field; --no-deconvolve skips it). Appends one row
per snapshot to the CSV (time order is the caller's), with the normalizations
of Johnsen et al. (2010): <u_i u_i> / (3 u_rms0^2), enstrophy <w_i w_i> and
dilatation <theta^2> over u_rms0^2 / lambda0^2, <T'^2> / ((gamma - 1) T0 Mt0^2)^2,
<rho'^2> / Mt0^4 and <p'^2> / (gamma p0 Mt0^2)^2. Three sets of columns:

  full    on the N^3 grid with spectral derivatives, the DNS statistics;
  fd6     spectrally filtered to filter-n^3 (modes |k_i| < filter-n / 2) and
          differentiated with sixth-order central differences, the protocol
          of Johnsen et al.'s reference (Subramaniam et al. 2019, appendix F);
  spec    filtered the same way, with spectral derivatives (the protocol of
          Subramaniam et al.'s 512^3 reference).

Also Mt, Re_lambda, lambda, the dissipation rate eps = <mu (w^2 + 4/3 theta^2)> / <rho>,
the Kolmogorov scale eta = (nu^3 / eps)^(1/4) and k_max eta (k_max = N / 2),
the most negative dilatation and the largest local Mach number.

--spectra DIR writes DIR/spectra_<step>.npz: shell spectra (shells of unit
width around integer k) of the velocity (E = sum |u_hat|^2 / 2, and its
solenoidal and dilatational parts), vorticity, dilatation, density,
temperature and pressure fluctuations. --fields DIR writes DIR/fields_<step>.npz
for the animation: Q-criterion, vorticity magnitude and dilatation resampled
to fields-n^3 (float16), and the dilatation and density gradient magnitude on
the plane z = pi at full resolution.
"""
import argparse
import os

import numpy as np

from isotropic_turbulence_restart import cell_average_factor, parameters, read_restart, wavenumbers


def fft(f):
    return np.fft.fftn(f, axes=(-3, -2, -1))


def ifft(f):
    return np.real(np.fft.ifftn(f, axes=(-3, -2, -1)))


def derivative_multipliers(n, scheme):
    """i k_eff along x, y, z for spectral (Nyquist dropped) or sixth-order central differences."""
    k = np.fft.fftfreq(n, 1.0 / n)
    h = 2 * np.pi / n
    if scheme == "spectral":
        keff = np.where(np.abs(k) == n // 2, 0.0, k)
    elif scheme == "fd6":
        keff = (45 * 2 * np.sin(k * h) - 9 * 2 * np.sin(2 * k * h) + 2 * np.sin(3 * k * h)) / (60 * h)
    else:
        raise ValueError(scheme)
    return 1j * keff[:, None, None], 1j * keff[None, :, None], 1j * keff[None, None, :]


def point_fields(cons, n, gamma, deconvolve=True):
    """Point values rho, u (3, n, n, n), p and T (R = 1) from the cells' conservatives."""
    def grid(name):
        f = cons[name].reshape(n, n, n)
        return ifft(fft(f) / cell_average_factor(n)) if deconvolve else f
    rho = grid("RHO")
    m = np.stack([grid(v) for v in ("RHOU_X", "RHOU_Y", "RHOU_Z")])
    E = grid("RHOE")
    u = m / rho
    p = (gamma - 1.0) * (E - 0.5 * (m * u).sum(axis=0))
    return rho, u, p, p / rho


def truncate(f_hat, n, m):
    """Fourier coefficients of an n^3 grid kept for |k_i| < m / 2 on an m^3 grid (same normalization of values)."""
    keep = m // 2 - 1
    idx = np.arange(-keep, keep + 1)
    out = np.zeros(f_hat.shape[:-3] + (m, m, m), dtype=complex)
    src = np.ix_(idx % n, idx % n, idx % n)
    dst = np.ix_(idx % m, idx % m, idx % m)
    out[(...,) + dst] = f_hat[(...,) + src] * (m / n) ** 3
    return out


def gradients(u_hat, D):
    """du_i/dx_j as g[i][j] from the coefficients of u."""
    return [[ifft(D[j] * u_hat[i]) for j in range(3)] for i in range(3)]


def statistics(rho, u, p, T, D, par):
    """Normalized one-point statistics of point fields with derivative multipliers D."""
    g = gradients(fft(u), D)
    w = np.stack([g[2][1] - g[1][2], g[0][2] - g[2][0], g[1][0] - g[0][1]])
    theta = g[0][0] + g[1][1] + g[2][2]
    u0, lam0, mt = par["u_rms"], par["lambda"], par["mt"]
    s = {
        "kinetic_energy": (u ** 2).sum(axis=0).mean() / (3 * u0 ** 2),
        "enstrophy": (w ** 2).sum(axis=0).mean() / (u0 ** 2 / lam0 ** 2),
        "dilatation_variance": (theta ** 2).mean() / (u0 ** 2 / lam0 ** 2),
        "temperature_variance": T.var() / ((par["gamma"] - 1) * par["T0"] * mt ** 2) ** 2,
        "density_variance": rho.var() / mt ** 4,
        "pressure_variance": p.var() / (par["gamma"] * par["p0"] * mt ** 2) ** 2,
    }
    return s, g, w, theta


def flow_scales(rho, u, p, T, g, w, theta, par, n):
    """Mt, Re_lambda, lambda, eps, eta, k_max eta, min dilatation and max Mach."""
    mu = par["mu"] * (T / par["T0"]) ** 0.75
    c = np.sqrt(par["gamma"] * p / rho)
    uu = (u ** 2).sum(axis=0)
    lam = np.sqrt(sum((u[i] ** 2).mean() for i in range(3)) / sum((g[i][i] ** 2).mean() for i in range(3)))
    eps = (mu * ((w ** 2).sum(axis=0) + 4.0 / 3.0 * theta ** 2)).mean() / rho.mean()
    nu = mu.mean() / rho.mean()
    eta = (nu ** 3 / eps) ** 0.25
    return {
        "Mt": np.sqrt(uu.mean()) / c.mean(),
        "Re_lambda": rho.mean() * np.sqrt(uu.mean() / 3) * lam / mu.mean(),
        "lambda": lam,
        "eps": eps,
        "eta": eta,
        "kmax_eta": n / 2 * eta,
        "theta_min": theta.min() / (par["u_rms"] / par["lambda"]),
        "mach_max": np.sqrt(uu / c ** 2).max(),
    }


def shell_spectrum(f_hat, n):
    """Sum of |f_hat|^2 / 2 (values normalized as grid means) over shells around integer |k|, k = 0..n/2."""
    kx, ky, kz = wavenumbers(n)
    shell = np.rint(np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)).astype(int).ravel()
    e = 0.5 * (np.abs(f_hat) ** 2 / n ** 6)
    e = e.reshape(-1, n ** 3).sum(axis=0) if e.ndim == 4 else e.ravel()
    return np.bincount(shell, weights=e, minlength=n)[:n // 2 + 1]


def spectra(rho, u, p, T, n):
    D = derivative_multipliers(n, "spectral")
    kx, ky, kz = wavenumbers(n)
    u_hat = fft(u)
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    k2[0, 0, 0] = 1.0
    div = kx * u_hat[0] + ky * u_hat[1] + kz * u_hat[2]
    u_d = np.stack([kx * div, ky * div, kz * div]) / k2
    w_hat = np.stack([D[1] * u_hat[2] - D[2] * u_hat[1], D[2] * u_hat[0] - D[0] * u_hat[2],
                      D[0] * u_hat[1] - D[1] * u_hat[0]])
    theta_hat = D[0] * u_hat[0] + D[1] * u_hat[1] + D[2] * u_hat[2]
    out = {"k": np.arange(n // 2 + 1), "velocity": shell_spectrum(u_hat, n),
           "solenoidal": shell_spectrum(u_hat - u_d, n), "dilatational": shell_spectrum(u_d, n),
           "vorticity": shell_spectrum(w_hat, n), "dilatation": shell_spectrum(theta_hat, n)}
    for name, f in (("density", rho), ("temperature", T), ("pressure", p)):
        f_hat = fft(f)
        f_hat[0, 0, 0] = 0.0
        out[name] = shell_spectrum(f_hat, n)
    return out


def animation_fields(rho, u, n, m):
    """Q-criterion, |vorticity| and dilatation on an m^3 grid, and full-resolution
    dilatation and |grad rho| on the plane z = pi."""
    u_hat = fft(u)
    um = truncate(u_hat, n, m) if m < n else u_hat
    mm = min(m, n)
    g = gradients(um, derivative_multipliers(mm, "spectral"))
    S2 = sum((0.5 * (g[i][j] + g[j][i])) ** 2 for i in range(3) for j in range(3))
    W2 = sum((0.5 * (g[i][j] - g[j][i])) ** 2 for i in range(3) for j in range(3))
    w = np.sqrt((g[2][1] - g[1][2]) ** 2 + (g[0][2] - g[2][0]) ** 2 + (g[1][0] - g[0][1]) ** 2)
    out = {"Q": (0.5 * (W2 - S2)).astype(np.float16), "vorticity": w.astype(np.float16),
           "dilatation": (g[0][0] + g[1][1] + g[2][2]).astype(np.float16)}
    D = derivative_multipliers(n, "spectral")
    z = n // 2
    theta = sum(ifft(D[i] * u_hat[i])[:, :, z] for i in range(3))
    r_hat = fft(rho)
    grad_rho = np.sqrt(sum(ifft(D[i] * r_hat)[:, :, z] ** 2 for i in range(3)))
    out["slice_dilatation"] = theta.astype(np.float32)
    out["slice_grad_rho"] = grad_rho.astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("restart", nargs="+")
    ap.add_argument("-o", "--output", required=True, help="CSV to append to")
    ap.add_argument("--spectra")
    ap.add_argument("--fields")
    ap.add_argument("--fields-n", type=int, default=128)
    ap.add_argument("--filter-n", type=int, default=64)
    ap.add_argument("--no-deconvolve", action="store_true")
    ap.add_argument("--k0", type=float, default=4.0)
    ap.add_argument("--mt", type=float, default=0.6)
    ap.add_argument("--re-lambda", type=float, default=100.0)
    ap.add_argument("--gamma", type=float, default=1.4)
    args = ap.parse_args()
    par = parameters(args.k0, args.mt, args.re_lambda, args.gamma)
    for d in (args.spectra, args.fields):
        if d:
            os.makedirs(d, exist_ok=True)

    for path in args.restart:
        cons, t, step = read_restart(path)
        n = round(len(cons["RHO"]) ** (1 / 3))
        rho, u, p, T = point_fields(cons, n, par["gamma"], not args.no_deconvolve)
        del cons
        row = {"step": step, "t": t, "t_over_tau": t / par["tau"], "n": n}
        full, g, w, theta = statistics(rho, u, p, T, derivative_multipliers(n, "spectral"), par)
        row.update({f"{k}_full": v for k, v in full.items()})
        row.update(flow_scales(rho, u, p, T, g, w, theta, par, n))
        del g, w, theta
        m = min(args.filter_n, n)
        filtered = [ifft(truncate(fft(f), n, m)) for f in (rho, u, p, T)]
        for scheme in ("fd6", "spectral"):
            s, *_ = statistics(*filtered, derivative_multipliers(m, scheme), par)
            tag = "fd6" if scheme == "fd6" else "spec"
            row.update({f"{k}_{tag}": v for k, v in s.items() if k in
                        ("kinetic_energy", "enstrophy", "dilatation_variance", "temperature_variance",
                         "density_variance", "pressure_variance")})
        new = not os.path.exists(args.output)
        with open(args.output, "a") as f:
            if new:
                f.write(",".join(row) + "\n")
            f.write(",".join(f"{v:.10g}" if isinstance(v, float) else str(v) for v in row.values()) + "\n")
        if args.spectra:
            np.savez(os.path.join(args.spectra, f"spectra_{step:08d}.npz"), t=t, t_over_tau=t / par["tau"], n=n,
                     **spectra(rho, u, p, T, n))
        if args.fields:
            np.savez_compressed(os.path.join(args.fields, f"fields_{step:08d}.npz"), t=t, t_over_tau=t / par["tau"],
                                n=n, **animation_fields(rho, u, n, args.fields_n))
        print(f"{path}: t / tau = {t / par['tau']:.3f}, KE {row['kinetic_energy_full']:.4f}, "
              f"enstrophy {row['enstrophy_full']:.3f} (fd6 {row['enstrophy_fd6']:.3f}), "
              f"dilatation {row['dilatation_variance_full']:.4f}, Mt {row['Mt']:.4f}, "
              f"k_max eta {row['kmax_eta']:.3f}", flush=True)


if __name__ == "__main__":
    main()
