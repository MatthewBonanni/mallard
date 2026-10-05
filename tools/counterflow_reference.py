"""Counterflow diffusion flames from Cantera for V10 (docs/design/chemistry.md).

    python tools/counterflow_reference.py [OUT_DIR]
    python tools/counterflow_reference.py [OUT_DIR] --profile U_OX [U_OX ...]
    python tools/counterflow_reference.py [OUT_DIR] --match K_OX [K_OX ...]

Cantera's CounterflowDiffusionFlame (the axisymmetric stagnation-flow
similarity solution with plug-flow nozzles) of H2/N2 (X_H2 = 0.25) against
air, both at 300 K and 1 atm, nozzles 10 mm apart, momentum-balanced jets
(rho_f U_f^2 = rho_o U_o^2), mechanisms/h2o2.yaml with mixture-averaged
transport. Writes to OUT_DIR (default examples/counterflow_diffusion/reference):

  strain.csv     the strain-rate sweep: oxidizer and fuel nozzle velocities,
                 global strain (U_o + U_f) / L, local strains K_ox and K_f
                 (the axial velocity gradient -du/dz ahead of the flame on
                 each side: its first maximum going from the nozzle towards
                 the flame), peak temperature and its position,
                 stagnation point and the spread rate V = v / r at the
                 peak temperature (V_T, the strain the flame itself
                 sees); up to the last burning flame, whose K_ox is the
                 extinction strain rate (bisected on U_o to 0.5%)
  U<U_OX>.csv    with --profile: the full solution (z, u, spread rate V =
                 v / r, T, rho, Y) at that oxidizer velocity, which is the
                 initial state of Mallard's runs (tools/counterflow_setup.py)
  K<K_OX>.csv    with --match: the same for the flame whose K_ox is K_OX
                 (to 0.1%, found by iterating on U_o from strain.csv): the
                 overlays of the comparison (tools/plot_counterflow.py)

z runs from the fuel nozzle (z = 0) to the oxidizer nozzle (z = L).
"""
import argparse
import os

import cantera as ct
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MECH, PHASE = os.path.join(ROOT, "mechanisms", "h2o2.yaml"), "ohmech"
FUEL = {"H2": 0.25, "N2": 0.75}
OXIDIZER = {"O2": 0.21, "N2": 0.79}
T_IN, P, GAP = 300.0, ct.one_atm, 0.01
TRANSPORT = "mixture-averaged"


def densities(gas):
    gas.TPX = T_IN, P, FUEL
    rho_f = gas.density
    gas.TPX = T_IN, P, OXIDIZER
    return rho_f, gas.density


def local_strains(z, u):
    """K_ox, K_f: the local strain rates ahead of the flame, the first maxima
    of -du/dz met going from each nozzle towards the flame (before the
    flame's dilatation reverses the gradient)."""
    g = -np.gradient(u, z)
    i = g.size - 1
    while i > 0 and g[i - 1] >= g[i]:
        i -= 1
    j = 0
    while j < g.size - 1 and g[j + 1] >= g[j]:
        j += 1
    return g[i], g[j]


def make_flame():
    gas = ct.Solution(MECH, PHASE, transport_model=TRANSPORT)
    gas.TP = T_IN, P
    f = ct.CounterflowDiffusionFlame(gas, width=GAP)
    f.fuel_inlet.X, f.fuel_inlet.T = FUEL, T_IN
    f.oxidizer_inlet.X, f.oxidizer_inlet.T = OXIDIZER, T_IN
    f.set_refine_criteria(ratio=2.0, slope=0.05, curve=0.08, prune=0.02)
    f.set_max_grid_points(f.flame, 5000)
    return f


def solve(f, U_o, guess=None, first=False):
    """Solve at oxidizer velocity U_o from the current solution, or from
    guess (a previous solution) when given; first: from Cantera's guess."""
    rho_f, rho_o = densities(f.gas)
    U_f = U_o * np.sqrt(rho_o / rho_f)
    if guess is not None:
        f.from_array(guess)
    f.fuel_inlet.mdot = rho_f * U_f
    f.oxidizer_inlet.mdot = rho_o * U_o
    f.solve(loglevel=0, auto=first)
    return U_f


def row(f, U_o, U_f):
    z, u, T = f.grid, f.velocity, f.T
    K_ox, K_f = local_strains(z, u)
    i_s = np.argmin(np.abs(u))
    i_T = np.argmax(T)
    return (U_o, U_f, (U_o + U_f) / GAP, K_ox, K_f, T[i_T], z[i_T], z[i_s], f.spread_rate[i_T], z.size)


def sweep(out):
    f = make_flame()
    rows = []
    U = 0.1
    last = None
    while True:
        U_f = solve(f, U, first=last is None)
        if f.extinct():
            break
        last = f.to_array()
        rows.append(row(f, U, U_f))
        print("U_o = %.4f m/s: K_ox = %.1f 1/s, T_max = %.1f K" % (U, rows[-1][3], rows[-1][5]), flush=True)
        U *= 1.25 if U < 3.0 else 1.08
    lo, hi = rows[-1][0], U
    while (hi - lo) / lo > 0.005:
        mid = 0.5 * (lo + hi)
        U_f = solve(f, mid, last)
        if f.extinct():
            hi = mid
        else:
            last = f.to_array()
            lo = mid
            rows.append(row(f, mid, U_f))
            print("U_o = %.4f m/s: K_ox = %.1f 1/s, T_max = %.1f K" % (mid, rows[-1][3], rows[-1][5]), flush=True)
    rows.sort()
    with open(os.path.join(out, "strain.csv"), "w") as fh:
        fh.write(f"# Cantera {ct.__version__} CounterflowDiffusionFlame, h2o2.yaml/{PHASE}, {TRANSPORT}; "
                 f"fuel H2:N2 = 0.25:0.75, air O2:N2 = 0.21:0.79, {T_IN} K, {P} Pa, gap {GAP} m; "
                 f"momentum-balanced; extinct above U_o = {hi:.5f} m/s (last row: extinction)\n")
        fh.write("U_o,U_f,a_global,K_ox,K_f,T_max,z_T_max,z_stag,V_T,points\n")
        for r in rows:
            fh.write(",".join(f"{v:.8g}" for v in r[:-1]) + f",{r[-1]}\n")


def burning(f, U_o):
    """Solve at U_o by continuation from U_o = 0.1 m/s."""
    U = min(0.1, U_o)
    U_f = solve(f, U, first=True)
    while U < U_o:
        U = min(U * 1.5, U_o)
        U_f = solve(f, U)
    if f.extinct():
        raise RuntimeError(f"extinct at U_o = {U_o}")
    return U_f


def write_profile(f, U_o, U_f, path):
    r = row(f, U_o, U_f)
    with open(path, "w") as fh:
        fh.write(f"# Cantera {ct.__version__} CounterflowDiffusionFlame as strain.csv; U_o = {U_o:.8g} m/s, "
                 f"U_f = {U_f:.8g} m/s, K_ox = {r[3]:.6g} 1/s, K_f = {r[4]:.6g} 1/s, T_max = {r[5]:.6g} K, "
                 f"V_T = {r[8]:.6g} 1/s\n")
        fh.write("z,u,V,T,rho," + ",".join("Y_" + s for s in f.gas.species_names) + "\n")
        data = np.column_stack([f.grid, f.velocity, f.spread_rate, f.T, f.density, f.Y.T])
        np.savetxt(fh, data, delimiter=",", fmt="%.8e")
    print(f"{path}: U_o = {U_o:.5f} m/s, K_ox = {r[3]:.1f} 1/s, T_max = {r[5]:.1f} K, {f.grid.size} points")


def profile(U_o, out):
    f = make_flame()
    U_f = burning(f, U_o)
    write_profile(f, U_o, U_f, os.path.join(out, f"U{U_o:g}.csv"))


def match(K, out):
    """The flame whose K_ox is K (to 0.1%), starting from strain.csv."""
    sweep = np.genfromtxt(os.path.join(out, "strain.csv"), delimiter=",", names=True, skip_header=1)
    U = float(np.exp(np.interp(np.log(K), np.log(sweep["K_ox"]), np.log(sweep["U_o"]))))
    f = make_flame()
    U_f = burning(f, U)
    for _ in range(20):
        K_U = row(f, U, U_f)[3]
        if abs(K_U / K - 1.0) < 1e-3:
            break
        U *= K / K_U
        U_f = solve(f, U)
    else:
        raise RuntimeError(f"no flame with K_ox = {K}")
    write_profile(f, U, U_f, os.path.join(out, f"K{K:.0f}.csv"))


def main():
    ap = argparse.ArgumentParser(description="Cantera counterflow diffusion flames for V10")
    ap.add_argument("out", nargs="?", default=os.path.join(ROOT, "examples", "counterflow_diffusion", "reference"))
    ap.add_argument("--profile", type=float, nargs="+", metavar="U_OX",
                    help="write the full solutions at these oxidizer velocities instead of the sweep")
    ap.add_argument("--match", type=float, nargs="+", metavar="K_OX",
                    help="write the full solutions with these local strain rates K_ox (from strain.csv)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for U_o in args.profile or []:
        profile(U_o, args.out)
    for K in args.match or []:
        match(K, args.out)
    if not args.profile and not args.match:
        sweep(args.out)


if __name__ == "__main__":
    main()
