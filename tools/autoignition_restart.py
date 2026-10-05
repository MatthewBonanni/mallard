"""Initial state of a thermally stratified autoignition run (examples/autoignition_2d).

    python tools/autoignition_restart.py RUN_DIR [--t-rms 15] [--n 400] [--length 4.1e-3]
        [--T0 1070] [--p0 41] [--phi 0.1] [--l-T 1.25e-3] [--u-rms 0.5] [--l-e 1.25e-3]
        [--seed 1] [--t-stop 4.8e-3] [--outputs 192] [--restart-interval 2e-4] [--cfl 0.4]
        [--mechanism mechanisms/h2o2.yaml --phase ohmech]

A periodic square of side --length and N x N cells holds lean H2/air (--phi)
at uniform pressure --p0 (atm) and composition, with the temperature
T0 + T'(x, y) and the velocity of decaying turbulence, as in Chen et al. and
Hawkes et al., Combust. Flame 145 (2006). T' and the two velocity components
are random fields of the Passot-Pouquet spectrum

    E(k) ~ (k / k_e)^4 exp(-2 (k / k_e)^2),   k_e = 2 pi / l_e,

with random phases from --seed: T' has the RMS --t-rms and its most energetic
length --l-T; the velocity is solenoidal (from a stream function), with the
RMS --u-rms per component and most energetic length --l-e. The same seed gives
the same field shapes at every amplitude, so runs of different --t-rms differ
only in amplitude. Cells take the fields' values at their centers (the fields
hold only wavelengths far above the cell size).

Writes RUN_DIR/autoignition.restart (float64, double builds) and
RUN_DIR/input.toml, and prints the field's statistics with the spontaneous-
propagation speed of Sankaran et al. (2005) for its RMS temperature gradient.
"""
import argparse
import os
import struct

import cantera as ct
import numpy as np


def passot_pouquet_modes(n, length, l_e, rng):
    """Random-phase Fourier amplitudes on an n x n periodic grid, and the wavenumbers."""
    k1 = 2 * np.pi * np.fft.fftfreq(n, d=length / n)
    kx, ky = np.meshgrid(k1, k1, indexing="ij")
    k = np.hypot(kx, ky)
    k_e = 2 * np.pi / l_e
    energy = (k / k_e) ** 4 * np.exp(-2 * (k / k_e) ** 2)
    # E(k) is the energy of the shell of radius k: per mode it is E / (2 pi k) in 2D
    amp = np.zeros_like(k)
    amp[k > 0] = np.sqrt(energy[k > 0] / (2 * np.pi * k[k > 0]))
    return amp * np.exp(2j * np.pi * rng.uniform(size=k.shape)), kx, ky, k


def temperature_field(n, length, l_T, t_rms, rng):
    modes, _, _, _ = passot_pouquet_modes(n, length, l_T, rng)
    f = np.real(np.fft.ifft2(modes))
    f -= f.mean()
    return f * t_rms / f.std()


def velocity_field(n, length, l_e, u_rms, rng):
    modes, kx, ky, k = passot_pouquet_modes(n, length, l_e, rng)
    psi = np.zeros_like(modes)
    psi[k > 0] = modes[k > 0] / k[k > 0]
    u = np.real(np.fft.ifft2(1j * ky * psi))
    v = np.real(np.fft.ifft2(-1j * kx * psi))
    scale = u_rms / np.sqrt(0.5 * (u.var() + v.var()))
    return u * scale, v * scale


def write_restart(path, names, fields):
    n = fields[0].size
    with open(path, "wb") as f:
        f.write(b"MALLARD-RESTART\0")
        f.write(struct.pack("<IIQQQd", 2, 8, n, len(names), 0, 0.0))
        for name in names:
            f.write(struct.pack("<I", len(name)) + name.encode())
        for field in fields:
            f.write(np.asarray(field, dtype="<f8").tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--t-rms", type=float, default=15.0)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--length", type=float, default=4.1e-3)
    ap.add_argument("--T0", type=float, default=1070.0)
    ap.add_argument("--p0", type=float, default=41.0)
    ap.add_argument("--phi", type=float, default=0.1)
    ap.add_argument("--l-T", type=float, default=1.25e-3)
    ap.add_argument("--u-rms", type=float, default=0.5)
    ap.add_argument("--l-e", type=float, default=1.25e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--t-stop", type=float, default=4.8e-3)
    ap.add_argument("--outputs", type=int, default=192)
    ap.add_argument("--restart-interval", type=float, default=0.2e-3)
    ap.add_argument("--cfl", type=float, default=0.4)
    ap.add_argument("--mechanism", default=os.path.join(os.path.dirname(__file__), "..", "mechanisms", "h2o2.yaml"))
    ap.add_argument("--phase", default="ohmech")
    args = ap.parse_args()

    n, length = args.n, args.length
    rng_T = np.random.default_rng(args.seed)
    rng_u = np.random.default_rng(args.seed + 1000)
    dT = temperature_field(n, length, args.l_T, args.t_rms, rng_T) if args.t_rms > 0 else np.zeros((n, n))
    u, v = velocity_field(n, length, args.l_e, args.u_rms, rng_u) if args.u_rms > 0 else (np.zeros((n, n)),) * 2

    gas = ct.Solution(args.mechanism, args.phase)
    gas.set_equivalence_ratio(args.phi, "H2", "O2:1, N2:3.76")
    Y = gas.Y.copy()
    p = args.p0 * ct.one_atm
    T = args.T0 + dT
    # rho and e are smooth in T: interpolate them from a fine table
    Tt = np.linspace(T.min() - 1, T.max() + 1, 512)
    rho_t, e_t = np.empty_like(Tt), np.empty_like(Tt)
    for i, Ti in enumerate(Tt):
        gas.TPY = Ti, p, Y
        rho_t[i], e_t[i] = gas.density, gas.int_energy_mass
    rho = np.interp(T, Tt, rho_t)
    e = np.interp(T, Tt, e_t)

    # Generated cartesian meshes number cells with y fastest: field[i_x, i_y].ravel()
    names = ["RHO", "RHOU_X", "RHOU_Y", "RHOE"] + ["RHOY_" + s for s in gas.species_names] + ["T_SEED"]
    fields = [rho, rho * u, rho * v, rho * (e + 0.5 * (u * u + v * v))]
    fields += [rho * Yk for Yk in Y] + [T]
    os.makedirs(args.run_dir, exist_ok=True)
    write_restart(os.path.join(args.run_dir, "autoignition.restart"), names, [f.ravel() for f in fields])

    dx = length / n
    gx, gy = np.gradient(T, dx, edge_order=2)
    grad_rms = np.sqrt(np.mean(gx ** 2 + gy ** 2))
    X = ", ".join(f'"{s}" = {x:.17g}' for s, x in zip(gas.species_names, gas.X) if x > 0)
    mech_path = os.path.relpath(os.path.abspath(args.mechanism), os.path.abspath(args.run_dir))
    with open(os.path.join(args.run_dir, "input.toml"), "w") as f:
        f.write(f"""# Autoignition of H2/air (phi = {args.phi:g}, {args.T0:g} K, {args.p0:g} atm) with T' = {args.t_rms:g} K
# (l_T = {args.l_T * 1e3:g} mm), u' = {args.u_rms:g} m/s (l_e = {args.l_e * 1e3:g} mm), seed {args.seed}:
# {n} x {n} cells of {dx * 1e6:.3f} um, from tools/autoignition_restart.py

[run]
t_stop = {args.t_stop:.17g}
cfl = {args.cfl}

[mesh]
type = "cartesian"
Nx = {n}
Ny = {n}
Lx = {length:.17g}
Ly = {length:.17g}
periodic = ["x", "y"]

[initialize]
type = "restart"
file = "autoignition.restart"

[numerics]
riemann_solver = "HLLC"
time_integrator = "SSPRK3"

[numerics.face_reconstruction]
type = "MUSCL"

[physics]
type = "navier_stokes"
gas = "mixture"
mechanism = "{mech_path}"
phase = "{args.phase}"
transport = "mixture_averaged"

[chemistry]
rtol = 1.0e-6
atol = 1.0e-10
fuse_half_steps = true

[output]
check_interval = 1000

[[write_data]]
prefix = "./solut/autoignition"
format = "vtu"
time_interval = {args.t_stop / args.outputs:.17g}
variables = ["RHO", "U", "P", "T", "HRR", "Y_H2", "Y_H2O", "Y_OH", "Y_HO2", "OMEGA_H2O", "X_H2O", "D_H2O"]

[[write_data]]
prefix = "./restart/autoignition"
format = "restart"
time_interval = {args.restart_interval:.17g}
""")
    print(f"{args.run_dir}: {n} x {n} cells of {dx * 1e6:.3f} um; T = {T.mean():.2f} K +- {T.std():.3f} K "
          f"(min {T.min():.1f}, max {T.max():.1f}), |grad T|_rms = {grad_rms:.4g} K/m; "
          f"u' = {np.sqrt(0.5 * (u.var() + v.var())):.4f} m/s")


if __name__ == "__main__":
    main()
