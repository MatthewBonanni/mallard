#!/usr/bin/env python3
"""Initial state of decaying compressible isotropic turbulence as a Mallard restart file.

    isotropic_turbulence_restart.py N OUT.restart [--seed 1] [--k0 4] [--mt 0.6]
        [--re-lambda 100] [--gamma 1.4]

The case of Johnsen et al. (J. Comput. Phys. 229, 2010, section 3.5) on the
periodic box [0, 2 pi]^3 with N^3 hexahedra (mesh type "cartesian", cells
numbered (i * N + j) * N + k with x slowest): a random solenoidal velocity
field with the energy spectrum E(k) ~ k^4 exp(-2 (k / k0)^2), uniform density
rho0 = 1 and pressure p0 = 1 / gamma (R = 1, so T0 = 1 / gamma and the mean
sound speed is 1), scaled to the turbulent Mach number
Mt = sqrt(<u_i u_i>) / <c>.

Every Fourier mode below |k_i| = 32 has a fixed magnitude
|u_hat| ~ k exp(-(k / k0)^2) (energy E(k) / (4 pi k^2) per mode), a random
direction normal to k and random phases, drawn from the seed independently of
N, so runs on different meshes start from the same band-limited field (N >= 64;
coarser meshes keep the modes they can hold, which carry all but ~1e-9 of the
energy). Cells take the exact averages of that field (each mode times
sinc(k_i h / 2) per direction), and the energy is set so that the cell
pressure is exactly p0.

Prints the derived parameters: u_rms0 = Mt / sqrt(3), lambda0 = 2 / k0 (the
Taylor microscale of the continuous spectrum, with the value of the drawn
field), mu_ref = u_rms0 lambda0 / Re_lambda at T0 for the power law
mu = mu_ref (T / T0)^(3/4), and tau = lambda0 / u_rms0.
"""
import argparse
import struct

import numpy as np

K_MAX = 31  # modes |k_i| <= K_MAX are drawn, independently of the mesh


def parameters(k0=4.0, mt=0.6, re_lambda=100.0, gamma=1.4):
    """Reference quantities of the case (rho0 = 1, p0 = 1 / gamma, R = 1, c0 = 1)."""
    u_rms = mt / np.sqrt(3.0)
    lam = 2.0 / k0
    return {"gamma": gamma, "rho0": 1.0, "p0": 1.0 / gamma, "T0": 1.0 / gamma, "c0": 1.0, "k0": k0, "mt": mt,
            "u_rms": u_rms, "lambda": lam, "mu": u_rms * lam / re_lambda, "tau": lam / u_rms,
            "re_lambda": re_lambda}


def wavenumbers(n):
    """Integer wavenumbers of an n^3 FFT on the box of side 2 pi, as broadcastable axes."""
    k = np.fft.fftfreq(n, 1.0 / n)
    return k[:, None, None], k[None, :, None], k[None, None, :]


def master_modes(seed, k0):
    """Solenoidal Fourier coefficients on the cube |k_i| <= K_MAX (index k + K_MAX),
    Hermitian so the field is real, with |u_hat(k)| = k exp(-(k / k0)^2)."""
    rng = np.random.default_rng(seed)
    m = 2 * K_MAX + 1
    k1 = np.arange(-K_MAX, K_MAX + 1, dtype=float)
    kx, ky, kz = np.meshgrid(k1, k1, k1, indexing="ij")
    kk = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    g = rng.standard_normal((3, m, m, m)) + 1j * rng.standard_normal((3, m, m, m))
    kv = np.stack([kx, ky, kz])
    with np.errstate(invalid="ignore", divide="ignore"):
        g -= kv * (kv * g).sum(axis=0) / kk ** 2
        g *= kk * np.exp(-(kk / k0) ** 2) / np.sqrt((np.abs(g) ** 2).sum(axis=0))
    g[:, K_MAX, K_MAX, K_MAX] = 0.0
    # Hermitian: u_hat(-k) = conj(u_hat(k)), keeping the half with k > 0 lexicographically
    flipped = np.conj(g[:, ::-1, ::-1, ::-1])
    upper = (kx > 0) | ((kx == 0) & (ky > 0)) | ((kx == 0) & (ky == 0) & (kz > 0))
    return np.where(upper, g, flipped)


def velocity_hat(n, seed, k0):
    """Coefficients of the point-value velocity on an n^3 FFT grid (numpy layout,
    normalized as np.fft.fftn of the grid values), from the master modes that fit."""
    modes = master_modes(seed, k0)
    keep = min(K_MAX, n // 2 - 1)
    u_hat = np.zeros((3, n, n, n), dtype=complex)
    idx = np.arange(-keep, keep + 1)
    sub = modes[:, (idx + K_MAX)[:, None, None], (idx + K_MAX)[None, :, None], (idx + K_MAX)[None, None, :]]
    u_hat[:, (idx % n)[:, None, None], (idx % n)[None, :, None], (idx % n)[None, None, :]] = sub * n ** 3
    return u_hat


def cell_average_factor(n):
    """Fourier multiplier taking point values to averages over cells of side 2 pi / n."""
    kx, ky, kz = wavenumbers(n)
    h = 2 * np.pi / n
    return np.sinc(kx * h / (2 * np.pi)) * np.sinc(ky * h / (2 * np.pi)) * np.sinc(kz * h / (2 * np.pi))


def taylor_microscale(u_hat, n):
    """lambda = sqrt(<u_x^2> / <(du_x/dx)^2>), averaged over the three directions."""
    k = wavenumbers(n)
    num = sum((np.abs(u_hat[i]) ** 2).sum() for i in range(3))
    den = sum((np.abs(k[i] * u_hat[i]) ** 2).sum() for i in range(3))
    return np.sqrt(num / den)


def write_restart(path, fields, t=0.0, step=0):
    """Version-2 double-precision restart: magic, header, names, then each field's cells."""
    names = list(fields)
    n_cells = fields[names[0]].size
    with open(path, "wb") as f:
        f.write(b"MALLARD-RESTART\0")
        f.write(struct.pack("<IIQQQd", 2, 8, n_cells, len(names), step, t))
        for name in names:
            f.write(struct.pack("<I", len(name)) + name.encode())
        for name in names:
            f.write(np.ascontiguousarray(fields[name], dtype="<f8").tobytes())


def read_restart(path):
    """(fields by name, t, step) of a version-2 double-precision restart file."""
    with open(path, "rb") as f:
        if f.read(16) != b"MALLARD-RESTART\0":
            raise SystemExit(f"{path}: not a Mallard restart file")
        version, real_size, n_cells, n_vars, step, t = struct.unpack("<IIQQQd", f.read(40))
        if version != 2 or real_size != 8:
            raise SystemExit(f"{path}: expected a double-precision restart file of version 2")
        names = []
        for _ in range(n_vars):
            (length,) = struct.unpack("<I", f.read(4))
            names.append(f.read(length).decode())
        data = np.fromfile(f, dtype="<f8", count=n_cells * n_vars).reshape(n_vars, n_cells)
    return dict(zip(names, data)), t, step


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("n", type=int, help="cells per direction")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--k0", type=float, default=4.0)
    ap.add_argument("--mt", type=float, default=0.6)
    ap.add_argument("--re-lambda", type=float, default=100.0)
    ap.add_argument("--gamma", type=float, default=1.4)
    args = ap.parse_args()
    n = args.n
    par = parameters(args.k0, args.mt, args.re_lambda, args.gamma)

    u_hat = velocity_hat(n, args.seed, args.k0)
    # Parseval: <u_i u_i> of the point values; scale it to (Mt c0)^2
    uu = (np.abs(u_hat) ** 2).sum() / n ** 6
    u_hat *= par["mt"] * par["c0"] / np.sqrt(uu)
    lam = taylor_microscale(u_hat, n)
    u_hat *= cell_average_factor(n)
    u = np.real(np.fft.ifftn(u_hat, axes=(1, 2, 3)))
    del u_hat

    rho0, p0 = par["rho0"], par["p0"]
    fields = {"RHO": np.full(n ** 3, rho0)}
    for d, name in enumerate(["RHOU_X", "RHOU_Y", "RHOU_Z"]):
        fields[name] = rho0 * u[d].ravel()
    fields["RHOE"] = p0 / (par["gamma"] - 1.0) + 0.5 * rho0 * (u ** 2).sum(axis=0).ravel()
    write_restart(args.out, fields)

    print(f"wrote {args.out}: {n}^3 cells, seed {args.seed}")
    print(f"u_rms0 = {par['u_rms']:.10g}, lambda0 = {par['lambda']:.6g} (drawn field: {lam:.6g}), "
          f"cell-average Mt = {np.sqrt((u ** 2).sum(axis=0).mean()) / par['c0']:.6g}")
    print(f"[physics] gamma = {par['gamma']}, p_ref = {p0:.16g}, T_ref = {par['T0']:.16g}, rho_ref = 1.0, "
          f"mu = {par['mu']:.16g}, viscosity_model = \"power_law\", T_mu_ref = {par['T0']:.16g}, "
          "viscosity_exponent = 0.75, Pr = 0.7")
    print(f"tau = lambda0 / u_rms0 = {par['tau']:.16g}; t / tau = 4 at t = {4 * par['tau']:.16g}")


if __name__ == "__main__":
    main()
