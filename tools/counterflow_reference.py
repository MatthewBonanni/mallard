"""Counterflow diffusion flames from Cantera for V10 (docs/design/chemistry.md).

    python tools/counterflow_reference.py [OUT_DIR]
    python tools/counterflow_reference.py --profile U_OX [U_OX ...] [OUT_DIR]

Cantera's CounterflowDiffusionFlame (the axisymmetric stagnation-flow
similarity solution with plug-flow nozzles) of H2/N2 (X_H2 = 0.25) against
air, both at 300 K and 1 atm, nozzles 10 mm apart, momentum-balanced jets
(rho_f U_f^2 = rho_o U_o^2), mechanisms/h2o2.yaml with mixture-averaged
transport. Writes to OUT_DIR (default examples/counterflow_diffusion/reference):

  strain.csv     the strain-rate sweep: oxidizer and fuel nozzle velocities,
                 global strain (U_o + U_f) / L, local strains K_ox and K_f
                 (the largest axial velocity gradient -du/dz ahead of the
                 flame on each side), peak temperature and its position,
                 stagnation point; up to the last burning flame, whose K_ox
                 is the extinction strain rate (bisected on U_o to 0.5%)
  U<U_OX>.csv    with --profile: the full solution (z, u, spread rate V =
                 v / r, T, rho, Y) at that oxidizer velocity, which is the
                 initial state of Mallard's runs (tools/counterflow_setup.py)
                 and the overlay of the comparison (tools/plot_counterflow.py)

z runs from the fuel nozzle (z = 0) to the oxidizer nozzle (z = L).
"""
import os
import sys

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


def local_strains(z, u, T):
    """K_ox, K_f: the largest -du/dz on the oxidizer side of the peak
    temperature and on the fuel side of the stagnation point."""
    dudz = np.gradient(u, z)
    i_T = np.argmax(T)
    i_s = np.argmin(np.abs(u))
    return np.max(-dudz[i_T:]), np.max(-dudz[: max(i_s, 1)])


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
    K_ox, K_f = local_strains(z, u, T)
    i_s = np.argmin(np.abs(u))
    return (U_o, U_f, (U_o + U_f) / GAP, K_ox, K_f, T.max(), z[np.argmax(T)], z[i_s], z.size)


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
        fh.write("U_o,U_f,a_global,K_ox,K_f,T_max,z_T_max,z_stag,points\n")
        for r in rows:
            fh.write(",".join(f"{v:.8g}" for v in r[:-1]) + f",{r[-1]}\n")


def profile(U_o, out):
    f = make_flame()
    U = min(0.1, U_o)
    solve(f, U, first=True)
    while U < U_o:
        U = min(U * 1.5, U_o)
        U_f = solve(f, U)
    U_f = solve(f, U_o)
    if f.extinct():
        raise RuntimeError(f"extinct at U_o = {U_o}")
    r = row(f, U_o, U_f)
    path = os.path.join(out, f"U{U_o:g}.csv")
    with open(path, "w") as fh:
        fh.write(f"# Cantera {ct.__version__} CounterflowDiffusionFlame as strain.csv; U_o = {U_o:.8g} m/s, "
                 f"U_f = {U_f:.8g} m/s, K_ox = {r[3]:.6g} 1/s, K_f = {r[4]:.6g} 1/s, T_max = {r[5]:.6g} K\n")
        fh.write("z,u,V,T,rho," + ",".join("Y_" + s for s in f.gas.species_names) + "\n")
        data = np.column_stack([f.grid, f.velocity, f.spread_rate, f.T, f.density, f.Y.T])
        np.savetxt(fh, data, delimiter=",", fmt="%.10e")
    print(f"{path}: K_ox = {r[3]:.1f} 1/s, T_max = {r[5]:.1f} K, {f.grid.size} points")


def main():
    args = sys.argv[1:]
    default_out = os.path.join(ROOT, "examples", "counterflow_diffusion", "reference")
    if args and args[0] == "--profile":
        values = []
        rest = args[1:]
        while rest:
            try:
                values.append(float(rest[0]))
                rest = rest[1:]
            except ValueError:
                break
        out = rest[0] if rest else default_out
        os.makedirs(out, exist_ok=True)
        for U_o in values:
            profile(U_o, out)
        return
    out = args[0] if args else default_out
    os.makedirs(out, exist_ok=True)
    sweep(out)


if __name__ == "__main__":
    main()
