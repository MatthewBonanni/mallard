"""A triple (tribrachial) flame run: a premixed front meets a fuel/air mixing layer.

    python tools/triple_flame.py MECHANISM PHASE RUN_DIR --mixing D0
        [--fuel H2:0.3,N2:0.7] [--oxidizer O2:1,N2:3.76] [--T 300] [--cells 12]
        [--width W] [--upstream 12] [--downstream 30] [--u-in 1.6] [--flame-times 12]
        [--outputs 120] [--cfl 0.8] [--transport mixture_averaged] [--planar FULL.csv]

The fresh gas enters on the left at a uniform velocity --u-in times S_L (upt),
at T and 1 atm, with the mixture fraction Z (kg of fuel stream per kg) of a
tanh mixing layer across the channel,

    Z(y) = (1 + tanh((y - y_c) / w)) / 2,

centered so that the stoichiometric value Z_st lies on the channel's
midline, with the mixing thickness delta_M = 1 / (dZ/dy at Z_st) =
D0 delta_L (Ruetsch, Vervisch & Linan 1995, whose overall change of Z is
also 1). S_L, delta_L (the thermal thickness (T_b - T_u) / max dT/dx) and
rho_u / rho_b are those of the planar Cantera flame at Z_st, computed here
(or read from --planar, a full solution as tools/flame_reference.py writes,
which this tool also writes as RUN_DIR/planar_full.csv for 1D runs of
tools/flame_restart.py). The sides are symmetry planes --width delta_L apart
(default 3 delta_M + 24 delta_L), and the burnt gas leaves through a pressure
outlet at 1 atm.

Initial state: fresh gas up to --upstream delta_L from the inlet, then each
row's adiabatic equilibrium products (Cantera, HP) for --downstream delta_L,
joined by a tanh of width delta_L / 2 (temperature and mass fractions); the
velocity is u_in rho_u / rho on every row. The planar front frays into a
triple flame whose leading edge settles near the stoichiometric line; with
u_in close to its propagation speed U_F it stays in the channel, and
tools/triple_flame_analysis.py measures U_F = u_in - dx_tip/dt.

Writes RUN_DIR/input.toml and RUN_DIR/flame.restart (float64); outputs
RUN_DIR/solut/flame.pvd (density, velocity, T, HRR and the mass fractions of
H2, O2, H2O and N2; the analysis takes Z from N2, which does not react).
"""
import argparse
import os
import struct

import cantera as ct
import numpy as np

P = ct.one_atm


def planar_flame(gas, T, fuel, oxidizer, Z, width=0.03):
    gas.TP = T, P
    gas.set_mixture_fraction(Z, fuel, oxidizer)
    flame = ct.FreeFlame(gas, width=width)
    flame.set_refine_criteria(ratio=2.0, slope=0.02, curve=0.02, prune=0.002)
    flame.solve(loglevel=0, auto=True)
    return flame


def read_full(path):
    with open(path) as f:
        names = f.readline().strip().split(",")
    return names, np.loadtxt(path, delimiter=",", skiprows=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mechanism")
    ap.add_argument("phase")
    ap.add_argument("run_dir")
    ap.add_argument("--mixing", type=float, required=True, help="delta_M / delta_L at the inlet")
    ap.add_argument("--fuel", default="H2:0.3,N2:0.7")
    ap.add_argument("--oxidizer", default="O2:1,N2:3.76")
    ap.add_argument("--T", type=float, default=300.0)
    ap.add_argument("--cells", type=float, default=12)
    ap.add_argument("--width", type=float)
    ap.add_argument("--upstream", type=float, default=12.0)
    ap.add_argument("--downstream", type=float, default=30.0)
    ap.add_argument("--u-in", type=float, default=1.6)
    ap.add_argument("--flame-times", type=float, default=12.0)
    ap.add_argument("--outputs", type=int, default=120)
    ap.add_argument("--cfl", type=float, default=0.8)
    ap.add_argument("--transport", default="mixture_averaged")
    ap.add_argument("--planar")
    args = ap.parse_args()

    model = {"mixture_averaged": "mixture-averaged", "unity_lewis": "unity-Lewis-number"}[args.transport]
    gas = ct.Solution(args.mechanism, args.phase, transport_model=model)
    gas.TP = args.T, P
    gas.set_equivalence_ratio(1.0, args.fuel, args.oxidizer)
    Z_st = gas.mixture_fraction(args.fuel, args.oxidizer)
    species = gas.species_names
    gas.TPX = args.T, P, args.fuel
    Y_F = gas.Y.copy()
    gas.TPX = args.T, P, args.oxidizer
    Y_O = gas.Y.copy()

    os.makedirs(args.run_dir, exist_ok=True)
    planar = os.path.join(args.run_dir, "planar_full.csv")
    if args.planar:
        names, d = read_full(args.planar)
    else:
        f = planar_flame(gas, args.T, args.fuel, args.oxidizer, Z_st)
        d = np.column_stack([f.grid, f.velocity, f.T, f.density, f.Y.T])
        names = ["x", "u", "T", "rho"] + ["Y_" + s for s in species]
    np.savetxt(planar, d, delimiter=",", header=",".join(names), comments="")
    x, u, T = d[:, 0], d[:, 1], d[:, 2]
    S_L, delta = u[0], (T[-1] - T[0]) / np.gradient(T, x).max()
    sigma = d[0, 3] / d[-1, 3]

    dx = delta / args.cells
    delta_M = args.mixing * delta
    width = args.width if args.width else 3 * args.mixing + 24
    ny = int(round(width * args.cells))
    nx = int(round((args.upstream + args.downstream) * args.cells))
    Lx, Ly = nx * dx, ny * dx
    w = 2 * Z_st * (1 - Z_st) * delta_M
    y_c = 0.5 * Ly - w * np.arctanh(2 * Z_st - 1)
    u_in = args.u_in * S_L
    x_0 = args.upstream * delta

    yc = (np.arange(ny) + 0.5) * dx
    xc = (np.arange(nx) + 0.5) * dx
    Z = 0.5 * (1 + np.tanh((yc - y_c) / w))
    s = 0.5 * (1 + np.tanh((xc - x_0) / (0.25 * delta)))
    ns = len(species)
    n = nx * ny
    cons = np.empty((nx, ny, 3 + ns))
    T_seed = np.empty((nx, ny))
    for j in range(ny):
        Y_u = Z[j] * Y_F + (1 - Z[j]) * Y_O
        gas.TPY = args.T, P, Y_u
        rho_u = gas.density
        gas.equilibrate("HP")
        T_b, Y_b = gas.T, gas.Y.copy()
        for i in range(nx):
            Ti = (1 - s[i]) * args.T + s[i] * T_b
            Yi = (1 - s[i]) * Y_u + s[i] * Y_b
            gas.TPY = Ti, P, Yi
            rho = gas.density
            ui = u_in * rho_u / rho
            cons[i, j, 0] = rho
            cons[i, j, 1] = rho * ui
            cons[i, j, 2] = rho * (gas.int_energy_mass + 0.5 * ui * ui)
            cons[i, j, 3:] = rho * gas.Y
            T_seed[i, j] = Ti
    cons = cons.reshape(n, -1)  # cells numbered with y fastest
    T_seed = T_seed.reshape(-1)

    field_names = ["RHO", "RHOU_X", "RHOU_Y", "RHOE"] + ["RHOY_" + sp for sp in species] + ["T_SEED"]
    fields = [cons[:, 0], cons[:, 1], np.zeros(n), cons[:, 2]] + [cons[:, 3 + k] for k in range(ns)] + [T_seed]
    with open(os.path.join(args.run_dir, "flame.restart"), "wb") as fh:
        fh.write(b"MALLARD-RESTART\0")
        fh.write(struct.pack("<IIQQQd", 2, 8, n, len(field_names), 0, 0.0))
        for name in field_names:
            fh.write(struct.pack("<I", len(name)) + name.encode())
        for field in fields:
            fh.write(np.asarray(field, dtype="<f8").tobytes())

    z_expr = f"0.5 * (1 + tanh((y - {y_c:.17g}) / {w:.17g}))"
    entries = ", ".join(f'"{sp}" = "({z_expr}) * {Y_F[k]:.17g} + (1 - ({z_expr})) * {Y_O[k]:.17g}"'
                        for k, sp in enumerate(species) if Y_F[k] > 0 or Y_O[k] > 0)
    t_stop = args.flame_times * delta / S_L
    mech_path = os.path.relpath(os.path.abspath(args.mechanism), os.path.abspath(args.run_dir))
    with open(os.path.join(args.run_dir, "input.toml"), "w") as fh:
        fh.write(f"""# Triple flame: fuel {args.fuel} / oxidizer {args.oxidizer} at {args.T:g} K, 1 atm; Z_st = {Z_st:.5f}.
# Planar flame at Z_st (Cantera): S_L = {S_L:.5f} m/s, delta_L = {delta * 1e3:.4f} mm, rho_u / rho_b = {sigma:.4f}.
# Inlet mixing thickness delta_M = {args.mixing:g} delta_L, inflow {args.u_in:g} S_L, {args.cells:g} cells per delta_L.

[run]
t_stop = {t_stop:.17g}
cfl = {args.cfl}

[mesh]
type = "cartesian"
Nx = {nx}
Ny = {ny}
Lx = {Lx:.17g}
Ly = {Ly:.17g}

[initialize]
type = "restart"
file = "flame.restart"

[[boundaries]]
name = "left"
type = "upt"
u = [{u_in:.17g}, 0.0]
p = {P}
T = {args.T:g}
Y = {{ {entries} }}

[[boundaries]]
name = "right"
type = "p_out"
p = {P}

[[boundaries]]
name = "bottom"
type = "symmetry"

[[boundaries]]
name = "top"
type = "symmetry"

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
variables = ["RHO", "U", "T", "HRR", "Y_H2", "Y_O2", "Y_H2O", "Y_N2"]

[[write_data]]
prefix = "./restart/flame"
format = "restart"
time_interval = {t_stop / 6:.17g}
""")
    print(f"{args.run_dir}: Z_st = {Z_st:.4f}, S_L = {S_L:.4f} m/s, delta_L = {delta * 1e3:.4f} mm, "
          f"rho_u / rho_b = {sigma:.3f}, sqrt = {np.sqrt(sigma):.3f}; {nx} x {ny} cells of {dx * 1e6:.2f} um, "
          f"delta_M = {delta_M * 1e3:.3f} mm, u_in = {u_in:.4f} m/s, t_stop = {t_stop * 1e3:.3f} ms")


if __name__ == "__main__":
    main()
