#!/usr/bin/env python3
"""Set up a one-dimensional H2/air flame run from Cantera's freely propagating flame.

    python flame_setup.py PHI RUN_DIR [--cells-per-delta 20] [--flame-times 4]

Solves Cantera's FreeFlame of H2/air at equivalence ratio PHI, 300 K and
1 atm with mixture-averaged transport (mechanisms/h2o2.yaml, as Mallard),
and writes to RUN_DIR:

  reference.csv  Cantera's flame: x (relative to the maximum of dT/dx), u, T,
                 heat release rate, H2, O2, H2O and OH mass fractions; first line: S_L, T_b and the thermal
                 thickness delta = (T_b - T_u) / max dT/dx
  input.toml     Mallard's input: the flame in its own frame on a strip from
                 6 delta upstream of the flame to 9 delta downstream, fresh
                 gas entering at S_L, started from a tanh profile of
                 thickness delta between the fresh and the burnt state
  mesh.sh        the gmsh command for the strip (strip.geo)

The input of the tutorial, input.toml next to this script, is the output for
PHI = 1.
"""
import argparse
import os

import cantera as ct
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MECH = os.path.join(HERE, "..", "..", "mechanisms", "h2o2.yaml")
T_U, P = 300.0, ct.one_atm
UPSTREAM, DOWNSTREAM = 6, 9  # domain in thermal thicknesses


def cantera_flame(phi):
    gas = ct.Solution(MECH, "ohmech", transport_model="mixture-averaged")
    gas.set_equivalence_ratio(phi, "H2", "O2:1.0, N2:3.76")
    gas.TP = T_U, P
    flame = ct.FreeFlame(gas, width=0.03)
    flame.set_refine_criteria(ratio=2.0, slope=0.02, curve=0.02, prune=0.002)
    flame.solve(loglevel=0, auto=True)
    return flame


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("phi", type=float)
    ap.add_argument("run_dir")
    ap.add_argument("--cells-per-delta", type=int, default=20)
    ap.add_argument("--flame-times", type=float, default=4.0)
    ap.add_argument("--mechanism", default="../../mechanisms/h2o2.yaml",
                    help="mechanism path as the input file should give it (relative to RUN_DIR)")
    args = ap.parse_args()
    os.makedirs(args.run_dir, exist_ok=True)

    f = cantera_flame(args.phi)
    x, T = f.grid, f.T
    dTdx = np.gradient(T, x)
    delta = (T[-1] - T[0]) / dTdx.max()
    s_l, t_b = f.velocity[0], T[-1]
    i_max = np.argmax(dTdx)
    xs = np.linspace(x[i_max] - 4 * delta, x[i_max] + 8 * delta, 400)
    with open(os.path.join(args.run_dir, "reference.csv"), "w") as fh:
        fh.write(f"# Cantera {ct.__version__} FreeFlame, H2/air, phi = {args.phi}, T_u = {T_U} K, 1 atm, "
                 f"mixture-averaged: S_L = {s_l:.6f} m/s, T_b = {t_b:.2f} K, delta = {delta:.6e} m\n")
        species = ["H2", "O2", "H2O", "OH"]
        fh.write("x,u,T,hrr," + ",".join("Y_" + k for k in species) + "\n")
        columns = [f.velocity, T, f.heat_release_rate] + [f.Y[f.gas.species_index(k)] for k in species]
        for xi in xs:
            fh.write(f"{xi - x[i_max]:.6e}," + ",".join(f"{np.interp(xi, x, c):.6e}" for c in columns) + "\n")

    names = f.gas.species_names
    X_u, X_b = np.where(f.X[:, 0] > 1e-12, f.X[:, 0], 0.0), f.X[:, -1]
    keep = [k for k, s in enumerate(names) if s != "N2" and max(X_u[k], X_b[k]) > 1e-4]
    n = int(round((UPSTREAM + DOWNSTREAM) * args.cells_per_delta))
    length = (UPSTREAM + DOWNSTREAM) * delta
    x_f = UPSTREAM * delta
    t_flame = delta / s_l
    c = f"0.5 * (1 + tanh((x - {x_f:.6e}) / {0.5 * delta:.6e}))"
    expansion = f.density[0] / f.density[-1]
    fresh = ", ".join(f'"{names[k]}" = {X_u[k]:.10g}' for k in range(len(names)) if X_u[k] > 0)
    mix = "\n".join(f'"{names[k]}" = "' + (f"{X_u[k]:.6g} {'-' if X_b[k] < X_u[k] else '+'} {abs(X_b[k] - X_u[k]):.6g}" if X_u[k] > 0 else f"{X_b[k]:.6g}")
                           + f' * {c}"' for k in keep)
    text = f"""# Freely propagating H2/air flame at phi = {args.phi:g}, 300 K and 1 atm with
# mixture-averaged transport, in the flame's frame, against Cantera's
# FreeFlame. Tutorial: docs/tutorials/premixed_flame.md. Written by
#   python flame_setup.py {args.phi:g} RUN_DIR
# Cantera: S_L = {s_l:.4f} m/s, T_b = {t_b:.1f} K, thermal thickness
# delta = {delta * 1e3:.4f} mm. Strip of {n} cells ({args.cells_per_delta} per delta) from
# {UPSTREAM} delta upstream of the flame to {DOWNSTREAM} delta downstream: fresh gas
# enters on the left at S_L and leaves through a pressure outlet; the run
# lasts {args.flame_times:g} flame times delta / S_L = {args.flame_times:g} x {t_flame * 1e3:.4f} ms.
#
# Initial state: temperature and mole fractions blend from the fresh to the
# burnt gas of Cantera's flame along c(x) = (1 + tanh(2 (x - x_f) / delta)) / 2
# at x_f = {x_f * 1e3:.4f} mm, with the velocity of a steady flame of that density.

[run]
t_stop = {args.flame_times * t_flame:.6e}
cfl = 1.0

[mesh]
type = "file"
filename = "strip.msh"

[initialize]
type = "analytical"
p = "{P}"
T = "{T_U} + {t_b - T_U:.4f} * {c}"
u = ["{s_l:.6f} * (1 + {expansion - 1:.5f} * {c})", "0"]
balance = "N2"

[initialize.X]
{mix}

[[boundaries]]
name = "inlet"
type = "upt"
u = [{s_l:.6f}, 0.0]
p = {P}
T = {T_U}
X = {{ {fresh} }}

[[boundaries]]
name = "outlet"
type = "p_out"
p = {P}

[[boundaries]]
name = "sides"
type = "symmetry"

[numerics]
riemann_solver = "HLLC"
time_integrator = "SSPRK3"
check_nan = true

[numerics.face_reconstruction]
type = "MUSCL"

[physics]
type = "navier_stokes"
gas = "mixture"
mechanism = "{args.mechanism}"
phase = "ohmech"
transport = "mixture_averaged"

[chemistry]
rtol = 1.0e-6
atol = 1.0e-10

[output]
check_interval = 5000

[[write_data]]
prefix = "./solut/flame"
format = "vtu"
time_interval = {t_flame / 10:.6e}
variables = ["RHO", "U_X", "P", "T", "HRR", "Y_*", "OMEGA_*"]
"""
    with open(os.path.join(args.run_dir, "input.toml"), "w") as fh:
        fh.write(text)
    geo = os.path.join(HERE, "strip.geo")
    with open(os.path.join(args.run_dir, "mesh.sh"), "w") as fh:
        fh.write(f"gmsh {geo} -2 -setnumber L {length:.6e} -setnumber n {n} -o strip.msh\n")
    print(f"phi = {args.phi:g}: S_L = {s_l:.5f} m/s, T_b = {t_b:.1f} K, delta = {delta * 1e3:.4f} mm; "
          f"{n} cells over {length * 1e3:.4f} mm, t_stop = {args.flame_times * t_flame * 1e3:.4f} ms")


if __name__ == "__main__":
    main()
