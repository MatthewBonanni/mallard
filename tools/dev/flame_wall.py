"""Head-on quenching of a laminar premixed flame against a cold isothermal wall.

    python tools/dev/flame_wall.py setup RUN_DIR [--phi 1.0] [--dx 4e-6] [--T-wall 300]
        [--gap 0.4e-3] [--burnt 0.8e-3] [--cfl 1.0] [--soret]
    python tools/dev/flame_wall.py analyze RUN_DIR [RUN_DIR ...]

setup computes the H2/air flame (mechanisms/h2o2.yaml, mixture-averaged
transport, 300 K, 1 atm) with Cantera's FreeFlame and writes RUN_DIR/input.toml
and RUN_DIR/flame.restart: a 1D strip with an isothermal wall at x = 0, fresh
gas at rest between the wall and the flame (its steepest temperature gradient
--gap from the wall) and burnt gas moving away from it to a pressure outlet
--burnt behind the flame. The flame runs into the wall at S_L and quenches.
Probes record the boundary cell (every 5 steps) and the whole strip (every 200
steps); RUN_DIR/flame.json keeps the flame's properties.

analyze reads the probes and reports, per run:
  - the wall heat flux q_w(t) = lambda_w (T_1 - T_w) / (dx / 2), with
    lambda_w the mean of the mixture conductivity at the boundary cell's T_1
    and at T_w (its composition), as the solver's wall flux;
  - the flame position: the wall distance of the largest H2 consumption rate
    (Cantera's rates at each probe point), and the quenching distance x_q, its
    minimum;
  - the quenching Peclet numbers x_q / delta_L, with delta_L = lambda_u /
    (rho_u cp_u S_L), and x_q / delta_T (the thermal thickness
    (T_b - T_u) / max dT/dx), the normalized peak wall heat flux
    phi_q = max q_w / (rho_u cp_u S_L (T_b - T_u)), and the estimate
    phi_q Pe_q (delta_L) of the conduction across the quenching distance,
    int_Tw^Tb lambda dT / (x_q rho_u cp_u S_L (T_b - T_u)) (Cantera's
    burnt-gas conductivity), which the measured phi_q should approach.
Writes RUN_DIR/wall.csv (t, q_w, x_flame).
"""
import argparse
import json
import os
import struct

import cantera as ct
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MECH = os.path.join(ROOT, "mechanisms", "h2o2.yaml")
PHASE = "ohmech"
P = ct.one_atm
T_U = 300.0


def flame(phi, soret):
    gas = ct.Solution(MECH, PHASE, transport_model="mixture-averaged")
    gas.set_equivalence_ratio(phi, "H2", "O2:1.0, N2:3.76")
    gas.TP = T_U, P
    f = ct.FreeFlame(gas, width=0.02)
    f.soret_enabled = soret
    f.set_refine_criteria(ratio=2.0, slope=0.02, curve=0.02, prune=0.002)
    f.solve(loglevel=0, auto=True)
    return f


def setup(args):
    f = flame(args.phi, args.soret)
    gas = f.gas
    x, u, T, rho, Y = f.grid, f.velocity, f.T, f.density, f.Y.T
    S_L = u[0]
    dTdx = np.gradient(T, x)
    i_f = np.argmax(dTdx)
    delta_T = (T[-1] - T[0]) / dTdx[i_f]
    gas.TPY = T[0], P, Y[0]
    delta_L = gas.thermal_conductivity / (rho[0] * gas.cp_mass * S_L)
    species = gas.species_names
    ns = len(species)
    e = np.empty(x.size)
    for i in range(x.size):
        gas.TPY = T[i], P, Y[i]
        e[i] = gas.int_energy_mass
    dx = args.dx
    nx = int(round((args.gap + args.burnt) / dx))
    L = nx * dx
    xs = x - x[i_f] + args.gap
    u_lab = u - S_L  # fresh gas at rest

    def profile(q, xq):
        return np.interp(xq, xs, q)

    n_sub = 16
    cons = np.zeros((nx, 3 + ns))
    T_seed = np.zeros(nx)
    for c in range(nx):
        xq = (c + (np.arange(n_sub) + 0.5) / n_sub) * dx
        r, v, ee = profile(rho, xq), profile(u_lab, xq), profile(e, xq)
        cons[c, 0] = r.mean()
        cons[c, 1] = (r * v).mean()
        cons[c, 2] = (r * (ee + 0.5 * v * v)).mean()
        cons[c, 3:] = [(r * profile(Y[:, k], xq)).mean() for k in range(ns)]
        T_seed[c] = profile(T, xq).mean()
    os.makedirs(args.run_dir, exist_ok=True)
    names = ["RHO", "RHOU_X", "RHOU_Y", "RHOE"] + ["RHOY_" + s for s in species] + ["T_SEED"]
    zeros = np.zeros(nx)
    fields = [cons[:, 0], cons[:, 1], zeros, cons[:, 2]] + [cons[:, 3 + k] for k in range(ns)] + [T_seed]
    with open(os.path.join(args.run_dir, "flame.restart"), "wb") as fh:
        fh.write(b"MALLARD-RESTART\0")
        fh.write(struct.pack("<IIQQQd", 2, 8, nx, len(names), 0, 0.0))
        for name in names:
            fh.write(struct.pack("<I", len(name)) + name.encode())
        for field in fields:
            fh.write(np.asarray(field, dtype="<f8").tobytes())
    t_stop = 1.3 * args.gap / S_L
    variables = ", ".join(f'"Y_{s}"' for s in species)
    mech_path = os.path.relpath(MECH, os.path.abspath(args.run_dir))
    with open(os.path.join(args.run_dir, "input.toml"), "w") as fh:
        fh.write(f"""# Head-on quenching of the H2/air flame (phi = {args.phi}) against an isothermal
# wall at {args.T_wall} K (tools/dev/flame_wall.py): S_L = {S_L:.5f} m/s, delta_T = {delta_T * 1e6:.1f} um,
# delta_L = lambda_u / (rho_u cp_u S_L) = {delta_L * 1e6:.2f} um; {nx} cells of {dx * 1e6:g} um

[run]
t_stop = {t_stop:.17g}
cfl = {args.cfl}

[mesh]
type = "cartesian"
Nx = {nx}
Ny = 1
Lx = {L:.17g}
Ly = {10 * dx:.17g}

[initialize]
type = "restart"
file = "flame.restart"

[[boundaries]]
name = "left"
type = "wall_isothermal"
T = {args.T_wall}

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
phase = "{PHASE}"
transport = "mixture_averaged"
{"soret = true" + chr(10) if args.soret else ""}
[chemistry]
rtol = 1.0e-6
atol = 1.0e-10

[output]
check_interval = 5000

[[probes]]
name = "wall"
variables = ["T", "P", {variables}]
point = [{0.5 * dx:.17g}, {5 * dx:.17g}]
interval = 5

[[probes]]
name = "strip"
variables = ["T", "P", {variables}]
start = [{0.5 * dx:.17g}, {5 * dx:.17g}]
end = [{L - 0.5 * dx:.17g}, {5 * dx:.17g}]
n_points = {nx}
interval = 200
""")
    with open(os.path.join(args.run_dir, "flame.json"), "w") as fh:
        json.dump({"phi": args.phi, "soret": args.soret, "S_L": S_L, "T_u": T[0], "T_b": T[-1], "delta_T": delta_T,
                   "delta_L": delta_L, "dx": dx, "T_wall": args.T_wall, "Y_u": list(Y[0]), "Y_b": list(Y[-1]),
                   "cantera": ct.__version__}, fh, indent=1)
    print(f"{args.run_dir}: {nx} cells of {dx * 1e6:g} um, S_L = {S_L:.5f} m/s, delta_T = {delta_T * 1e6:.1f} um, "
          f"delta_L = {delta_L * 1e6:.2f} um, t_stop = {t_stop:.3e} s")


def read_probe(path):
    """Rows step, t, point, x, y, T, P, Y_k (the order setup lists them in)."""
    rows = np.loadtxt(path, delimiter=",", ndmin=2)
    return rows[:, 1], rows[:, 3], rows[:, 5], rows[:, 6], rows[:, 7:]


def analyze(run_dir):
    with open(os.path.join(run_dir, "flame.json")) as fh:
        ref = json.load(fh)
    gas = ct.Solution(MECH, PHASE, transport_model="mixture-averaged")
    species = gas.species_names
    i_H2 = gas.species_index("H2")
    dx, T_w = ref["dx"], ref["T_wall"]

    def conductivity(T, Y):
        gas.TPY = T, P, Y
        return gas.thermal_conductivity

    t, _, T1, _, Y1 = read_probe(os.path.join(run_dir, "probe_wall.csv"))
    q = np.array([0.5 * (conductivity(T, Y) + conductivity(T_w, Y)) * (T - T_w) / (0.5 * dx) for T, Y in zip(T1, Y1)])
    ts, xs, Ts, ps, Ys = read_probe(os.path.join(run_dir, "probe_strip.csv"))
    times = np.unique(ts)
    x_flame = []
    for tt in times:
        m = ts == tt
        rate = np.empty(m.sum())
        for j, (T, p, Y) in enumerate(zip(Ts[m], ps[m], Ys[m])):
            gas.TPY = T, p, Y
            rate[j] = gas.net_production_rates[i_H2]
        x_flame.append(xs[m][np.argmin(rate)])  # most negative: largest consumption
    x_flame = np.array(x_flame)
    gas.TPY = ref["T_u"], P, ref["Y_u"]
    rho_u, cp_u = gas.density, gas.cp_mass
    S_L, T_u, T_b = ref["S_L"], ref["T_u"], ref["T_b"]
    power = rho_u * cp_u * S_L * (T_b - T_u)
    i_q = np.argmax(q)
    x_q = x_flame.min()
    Tg = np.linspace(T_w, T_b, 200)
    lam = np.array([conductivity(T, ref["Y_b"]) for T in Tg])
    conduction = np.trapezoid(lam, Tg) / x_q
    with open(os.path.join(run_dir, "wall.csv"), "w") as fh:
        fh.write("t,q_w,x_flame\n")
        for tt, qq in zip(t, q):
            fh.write(f"{tt:.9e},{qq:.6e},{np.interp(tt, times, x_flame):.6e}\n")
    out = {"dx_um": dx * 1e6, "q_max_MW_m2": q.max() / 1e6, "t_q": t[i_q], "x_q_um": x_q * 1e6,
           "x_flame_at_q_max_um": np.interp(t[i_q], times, x_flame) * 1e6,
           "Pe_q_delta_L": x_q / ref["delta_L"], "Pe_q_delta_T": x_q / ref["delta_T"],
           "phi_q": q.max() / power, "phi_q_Pe_q": q.max() / power * x_q / ref["delta_L"],
           "conduction_estimate_phi_q": conduction / power}
    print(run_dir + ": " + ", ".join(f"{k} = {v:.4g}" for k, v in out.items()))
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("setup")
    s.add_argument("run_dir")
    s.add_argument("--phi", type=float, default=1.0)
    s.add_argument("--dx", type=float, default=4e-6)
    s.add_argument("--T-wall", type=float, default=300.0)
    s.add_argument("--gap", type=float, default=0.4e-3)
    s.add_argument("--burnt", type=float, default=0.8e-3)
    s.add_argument("--cfl", type=float, default=1.0)
    s.add_argument("--soret", action="store_true")
    a = sub.add_parser("analyze")
    a.add_argument("run_dirs", nargs="+")
    args = ap.parse_args()
    if args.command == "setup":
        setup(args)
    else:
        for d in args.run_dirs:
            analyze(d)


if __name__ == "__main__":
    main()
