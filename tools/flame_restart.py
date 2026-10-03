"""A premixed flame run for Mallard from a Cantera FreeFlame (V8), 1D or 2D.

    python tools/flame_restart.py FULL.csv MECHANISM PHASE CELLS_PER_DELTA RUN_DIR
        [--cfl 0.4] [--transport mixture_averaged] [--flame-times 3] [--t-stop T]
        [--outputs 30] [--dim 2] [--upstream 6] [--downstream 9] [--ref-delta D]
        [--inlet-factor 1] [--ny NY --perturb A --modes 8 --seed 1]

FULL.csv is a flame's full solution from tools/flame_reference.py (x, u, T,
rho, Y). The run is a strip of Nx cells (with an aspect ratio of 10, so that
the cross-stream faces do not limit the time step) from --upstream thermal
thicknesses upstream of the flame (the maximum of dT/dx) to --downstream
downstream, in the flame's frame: the fresh mixture enters on the left at
Cantera's flame speed times --inlet-factor (upt) and leaves through a
pressure outlet at 1 atm. Each cell starts from the average of Cantera's
profile over 16 points. --ref-delta sizes the cells and the domain with that
thickness instead of the flame's own (to give flames of different transport
models the same mesh). With --ny the run is a 2D channel of NY square cells
across, periodic in y, whose flame is displaced along x by a sum of
random-phase modes 1..--modes of the width, scaled to a largest displacement
of --perturb reference thicknesses. Writes RUN_DIR/input.toml and
RUN_DIR/flame.restart (float64, double builds); the output series
RUN_DIR/solut/flame.pvd is what tools/flame_speed.py reads.
"""
import argparse
import os
import struct

import cantera as ct
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("full")
    ap.add_argument("mechanism")
    ap.add_argument("phase")
    ap.add_argument("cells_per_delta", type=float)
    ap.add_argument("run_dir")
    ap.add_argument("--cfl", type=float, default=0.4)
    ap.add_argument("--transport", default="mixture_averaged")
    ap.add_argument("--flame-times", type=float, default=3.0)
    ap.add_argument("--outputs", type=int, default=30)
    ap.add_argument("--dim", type=int, default=2)
    ap.add_argument("--upstream", type=float, default=6.0)
    ap.add_argument("--downstream", type=float, default=9.0)
    ap.add_argument("--ref-delta", type=float)
    ap.add_argument("--ny", type=int, default=1)
    ap.add_argument("--perturb", type=float, default=0.0)
    ap.add_argument("--modes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--t-stop", type=float)
    ap.add_argument("--inlet-factor", type=float, default=1.0)
    args = ap.parse_args()
    if args.ny > 1 and args.dim != 2:
        ap.error("--ny is for 2D runs")

    with open(args.full) as f:
        names = f.readline().strip().split(",")
    data = np.loadtxt(args.full, delimiter=",", skiprows=1)
    x, u, T, rho = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    species = [n[2:] for n in names[4:]]
    Y = data[:, 4:]
    gas = ct.Solution(args.mechanism, args.phase)
    assert gas.species_names == species, "species differ between the flame and the mechanism"
    p = ct.one_atm
    dTdx = np.gradient(T, x)
    i_f = np.argmax(dTdx)
    delta = (T[-1] - T[0]) / dTdx[i_f]
    S_L = u[0]
    ref = args.ref_delta or delta
    dx = ref / args.cells_per_delta
    nx = int(round((args.upstream + args.downstream) * args.cells_per_delta))
    L = nx * dx
    x_flame = args.upstream * ref
    xs = x - x[i_f] + x_flame  # Cantera's grid in the run's frame
    e = np.empty(x.size)
    for i in range(x.size):
        gas.TPY = T[i], p, Y[i]
        e[i] = gas.int_energy_mass

    def profile(q, xq):
        return np.interp(xq, xs, q)  # constant beyond Cantera's domain

    n_sub = 16
    ns = len(species)
    cons = np.zeros((nx, 3 + ns))
    T_seed = np.zeros(nx)
    for c in range(nx):
        xq = (c + (np.arange(n_sub) + 0.5) / n_sub) * dx
        r, v, ee = profile(rho, xq), profile(u, xq), profile(e, xq)
        rY = np.column_stack([r * profile(Y[:, k], xq) for k in range(ns)])
        cons[c, 0] = r.mean()
        cons[c, 1] = (r * v).mean()
        cons[c, 2] = (r * (ee + 0.5 * v * v)).mean()
        cons[c, 3:] = rY.mean(axis=0)
        T_seed[c] = profile(T, xq).mean()

    ny = args.ny
    n = nx * ny
    if ny > 1:
        # Flame displaced by x_f(y): random-phase modes 1..MODES of the periodic width,
        # scaled to a largest displacement of PERTURB reference thicknesses
        rng = np.random.default_rng(args.seed)
        yc = (np.arange(ny) + 0.5) / ny
        k = np.arange(1, args.modes + 1)
        shift = (rng.uniform(0.5, 1.0, k.size)[:, None] *
                 np.sin(2 * np.pi * (k[:, None] * yc[None, :] + rng.uniform(0, 1, k.size)[:, None]))).sum(axis=0)
        shift *= args.perturb * ref / np.abs(shift).max()
        cols = np.empty((nx, ny, 3 + len(species)))
        seeds = np.empty((nx, ny))
        xc = (np.arange(nx) + 0.5) * dx
        for j in range(ny):
            for v in range(cols.shape[2]):
                cols[:, j, v] = np.interp(xc - shift[j], xc, cons[:, v])
            seeds[:, j] = np.interp(xc - shift[j], xc, T_seed)
        cons = cols.reshape(n, -1)  # cells numbered with y fastest
        T_seed = seeds.reshape(-1)

    os.makedirs(args.run_dir, exist_ok=True)
    dim = args.dim
    fields_names = ["RHO", "RHOU_X", "RHOU_Y"] + (["RHOU_Z"] if dim == 3 else []) + ["RHOE"]
    fields_names += ["RHOY_" + s for s in species] + ["T_SEED"]
    zeros = np.zeros(n)
    fields = [cons[:, 0], cons[:, 1], zeros] + ([zeros] if dim == 3 else []) + [cons[:, 2]]
    fields += [cons[:, 3 + k] for k in range(ns)] + [T_seed]
    with open(os.path.join(args.run_dir, "flame.restart"), "wb") as f:
        f.write(b"MALLARD-RESTART\0")
        f.write(struct.pack("<IIQQQd", 2, 8, n, len(fields_names), 0, 0.0))
        for name in fields_names:
            f.write(struct.pack("<I", len(name)) + name.encode())
        for field in fields:
            f.write(np.asarray(field, dtype="<f8").tobytes())

    t_stop = args.t_stop or args.flame_times * delta / S_L
    # The fresh mixture without the traces that diffuse into Cantera's inlet
    X0 = Y[0] / gas.molecular_weights
    X0[X0 < 1e-8 * X0.sum()] = 0.0
    X0 /= X0.sum()
    X_in = ", ".join(f'"{s}" = {v:.17g}' for s, v in zip(species, X0) if v > 0.0)
    h = ny * dx if ny > 1 else 10.0 * dx
    zero = ", 0.0" if dim == 3 else ""
    extra_bc = ('[[boundaries]]\nname = "back"\ntype = "symmetry"\n'
                '[[boundaries]]\nname = "front"\ntype = "symmetry"\n') if dim == 3 else ""
    mesh_z = f"Nz = 1\nLz = {h:.17g}\n" if dim == 3 else ""
    bottom_top = '[[boundaries]]\nname = "bottom"\ntype = "symmetry"\n\n[[boundaries]]\nname = "top"\ntype = "symmetry"\n'
    mech_path = os.path.relpath(os.path.abspath(args.mechanism), os.path.abspath(args.run_dir))
    with open(os.path.join(args.run_dir, "input.toml"), "w") as f:
        f.write(f"""# Premixed flame (V8): {os.path.basename(args.full)}, {args.cells_per_delta:g} cells per
# thermal thickness ({delta * 1e3:.4f} mm), Cantera S_L = {S_L:.6f} m/s

[run]
t_stop = {t_stop:.17g}
cfl = {args.cfl}

[mesh]
type = "cartesian"
Nx = {nx}
Ny = {ny}
Lx = {L:.17g}
Ly = {h:.17g}
{'periodic = ["y"]' + chr(10) if ny > 1 else ""}{mesh_z}
[initialize]
type = "restart"
file = "flame.restart"

[[boundaries]]
name = "left"
type = "upt"
u = [{args.inlet_factor * S_L:.17g}, 0.0{zero}]
p = {p}
T = {T[0]:.17g}
X = {{ {X_in} }}

[[boundaries]]
name = "right"
type = "p_out"
p = {p}

{"" if ny > 1 else bottom_top}{extra_bc}
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
transport = "{args.transport}"

[chemistry]
rtol = 1.0e-6
atol = 1.0e-10

[output]
check_interval = 2000

[[write_data]]
prefix = "./solut/flame"
format = "vtu"
time_interval = {t_stop / args.outputs:.17g}
variables = ["RHO", "U_X", "P", "T", "HRR", "Y_*", "OMEGA_*"]
""")
    print(f"{args.run_dir}: {nx} x {ny} cells of {dx * 1e6:.3f} um, delta = {delta * 1e3:.4f} mm, S_L = {S_L:.5f} m/s, "
          f"t_stop = {t_stop:.4e} s")


if __name__ == "__main__":
    main()
