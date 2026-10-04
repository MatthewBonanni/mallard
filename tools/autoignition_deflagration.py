"""Laminar deflagration speed and thickness of an autoigniting mixture (examples/autoignition_2d).

    python tools/autoignition_deflagration.py make RUN_DIR --Tu 1070 --p 41 [--dx 2e-6]
        [--length 1.5e-3] [--t-stop 3e-4] [--phi 0.1] [--frozen 15]
    python tools/autoignition_deflagration.py measure RUN_DIR... [--csv OUT.csv]

A steady laminar flame does not exist in a mixture that autoignites in a time
comparable to its flame time (lean H2/air at 1070 K and 41 atm: tau_ig = 3.6
ms, flame time ~1 ms): Cantera's FreeFlame lets the incoming gas react all
the way from its inlet, and its speed then depends on the domain length. The
deflagration speed S_L of the regime criterion of Sankaran et al. (2005) is
instead measured here with Mallard: a 1D strip (symmetry at x = 0, a pressure
outlet at x = --length) of fresh mixture at --Tu and --p (atm) behind a
0.1 mm layer of its equilibrium products (constant enthalpy and pressure),
joined by a 30 um tanh profile of temperature and mole fractions, with no
chemistry in cells below Tu + --frozen K
([chemistry] T_frozen), so that the fresh gas cannot autoignite and the front
can only propagate by conduction and diffusion.

`make` writes RUN_DIR/input.toml. `measure` reads each run's series
(solut/defl.pvd) and prints the consumption speed

    S_c = -int omega_H2 dx / (rho_u Y_H2,u)

over time, its mean over the last half of the run (S_L), and the thermal
thickness delta = (T_b - T_u) / max dT/dx of the front; --csv collects them.
"""
import argparse
import os
import re
import sys

import cantera as ct
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mallard_vtu import read_quads  # noqa: E402


def make(args):
    gas = ct.Solution(args.mechanism, args.phase)
    gas.set_equivalence_ratio(args.phi, "H2", "O2:1, N2:3.76")
    Xu = gas.X.copy()
    gas.TP = args.Tu, args.p * ct.one_atm
    gas.equilibrate("HP")
    Xb, Tb = gas.X.copy(), gas.T
    s = "0.5 * (1 - tanh((x - 1.0e-4) / 3.0e-5))"
    X = ", ".join(f'{n} = "{xu:.17g} + {xb - xu:.17g} * {s}"'
                  for n, xu, xb in zip(gas.species_names, Xu, Xb) if max(xu, xb) > 1e-12)
    nx = int(round(args.length / args.dx))
    p = args.p * ct.one_atm
    os.makedirs(args.run_dir, exist_ok=True)
    mech = os.path.relpath(os.path.abspath(args.mechanism), os.path.abspath(args.run_dir))
    with open(os.path.join(args.run_dir, "input.toml"), "w") as f:
        f.write(f"""# Deflagration into frozen lean H2/air (phi = {args.phi:g}) at {args.Tu:g} K, {args.p:g} atm:
# {nx} cells of {args.dx * 1e6:g} um, from tools/autoignition_deflagration.py

[run]
t_stop = {args.t_stop:.17g}
cfl = 0.8

[mesh]
type = "cartesian"
Nx = {nx}
Ny = 1
Lx = {nx * args.dx:.17g}
Ly = {10 * args.dx:.17g}

[initialize]
type = "analytical"
T = "{args.Tu:.17g} + {Tb - args.Tu:.17g} * {s}"
p = "{p:.17g}"
u = ["0", "0"]
X = {{ {X} }}

[[boundaries]]
name = "left"
type = "symmetry"

[[boundaries]]
name = "right"
type = "p_out"
p = {p:.17g}

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
mechanism = "{mech}"
phase = "{args.phase}"
transport = "mixture_averaged"

[chemistry]
rtol = 1.0e-6
atol = 1.0e-10
T_frozen = {args.Tu + args.frozen:.17g}

[output]
check_interval = 20000

[[write_data]]
prefix = "./solut/defl"
format = "vtu"
time_interval = {args.t_stop / 30:.17g}
variables = ["RHO", "U", "P", "T", "HRR", "Y_H2", "OMEGA_H2"]
""")


def measure_run(run_dir):
    pvd = os.path.join(run_dir, "solut", "defl.pvd")
    files = re.findall(r'file="([^"]+)"', open(pvd).read())
    rows = []
    for name in files:
        t, c, f = read_quads(os.path.join(run_dir, "solut", name), ["RHO", "T", "Y_H2", "OMEGA_H2"])
        order = np.argsort(c[:, 0])
        xs = c[order, 0]
        rho, T, Y, w = (f[k][order] for k in ["RHO", "T", "Y_H2", "OMEGA_H2"])
        dx = xs[1] - xs[0]
        u = -1  # fresh gas: the right end
        S_c = -w.sum() * dx / (rho[u] * Y[u])
        dTdx = np.abs(np.gradient(T, dx))
        delta = (T.max() - T[u]) / dTdx.max()
        rows.append((t, S_c, delta, T[u], xs[np.argmax(dTdx)]))
    return np.array(rows), dx


def measure(args):
    out = []
    for run_dir in args.run_dirs:
        rows, dx = measure_run(run_dir)
        late = rows[:, 0] >= 0.5 * rows[-1, 0]
        S_L, delta = rows[late, 1].mean(), rows[late, 2].mean()
        spread = rows[late, 1].std() / S_L
        print(f"{run_dir}: dx = {dx * 1e6:.3g} um, S_L = {S_L:.4f} m/s (+- {100 * spread:.1f}% over the last half), "
              f"delta = {delta * 1e6:.1f} um = {delta / dx:.1f} cells, T_u = {rows[-1, 3]:.1f} K")
        for r in rows:
            print(f"    t = {r[0] * 1e6:8.2f} us  S_c = {r[1]:.4f} m/s  delta = {r[2] * 1e6:6.1f} um  x_f = {r[4] * 1e3:.4f} mm")
        out.append((dx, rows[-1, 3], S_L, delta))
    if args.csv:
        np.savetxt(args.csv, np.array(out), delimiter=",", header="dx,T_u,S_L,delta", comments="")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("run_dir")
    m.add_argument("--Tu", type=float, required=True)
    m.add_argument("--p", type=float, required=True)
    m.add_argument("--dx", type=float, default=2e-6)
    m.add_argument("--length", type=float, default=1.5e-3)
    m.add_argument("--t-stop", type=float, default=3e-4)
    m.add_argument("--phi", type=float, default=0.1)
    m.add_argument("--frozen", type=float, default=15.0)
    m.add_argument("--mechanism", default=os.path.join(os.path.dirname(__file__), "..", "mechanisms", "h2o2.yaml"))
    m.add_argument("--phase", default="ohmech")
    s = sub.add_parser("measure")
    s.add_argument("run_dirs", nargs="+")
    s.add_argument("--csv")
    args = ap.parse_args()
    make(args) if args.cmd == "make" else measure(args)


if __name__ == "__main__":
    main()
