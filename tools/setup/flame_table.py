"""Laminar premixed flame properties against mixture fraction, for the thickened flame of
non-premixed and partially premixed combustion ([les.combustion] flame_table, docs/guide/les.md).

    python tools/setup/flame_table.py MECHANISM PHASE OUT.csv --fuel H2:0.5,N2:0.5 [--oxidizer O2:1,N2:3.76]
        [--species H2] [--T 300] [--p 101325] [--dz 0.01] [--transport mixture_averaged]
        [--min-speed 0.02] [--width 0.03] [--profiles DIR] [--full DIR]

Solves Cantera FreeFlames of the fresh mixtures Y = Z Y_fuel + (1 - Z) Y_oxidizer (compositions as mole
fractions, at T and p), starting at the stoichiometric mixture fraction and stepping by --dz towards each
side, each flame from its neighbor's solution, until a flame has no solution or burns slower than
--min-speed (m/s): the flammable range of the table. Writes one row per flame:

  Z          Bilger's mixture fraction of the fuel and oxidizer streams (Cantera's, as Mallard's)
  s_L        laminar flame speed (m/s)
  delta_L    thermal thickness (T_b - T_u) / max dT/dx (m)
  omega_max  the largest consumption rate of --species (default: the fuel stream's most abundant
             species with C or H) through the flame, kg/m^3/s, positive
  phi, T_b   equivalence ratio and burnt temperature, for reference

Mallard interpolates the table linearly in Z and does not thicken mixtures outside its range. Use the run's
mechanism, transport model and fresh-gas state, and the same streams as [les.combustion] fuel and oxidizer.
--profiles DIR also writes each flame's x, T and consumption rate (DIR/Z<Z>.csv), and --full DIR its full
solution (x, u, T, rho, Y; DIR/Z<Z>_full.csv), the initial state of tools/setup/flame_restart.py.
"""
import argparse
import os

import cantera as ct
import numpy as np


def solve(gas, state, width, previous):
    gas.TPY = state
    flame = ct.FreeFlame(gas, width=width)
    flame.set_max_grid_points(flame.flame, 5000)
    flame.set_refine_criteria(ratio=2.0, slope=0.02, curve=0.04, prune=0.002)
    if previous is not None:
        xr = (previous.grid - previous.grid[0]) / (previous.grid[-1] - previous.grid[0])
        flame.set_initial_guess()
        flame.flame.set_profile("velocity", xr, previous.velocity)
        flame.flame.set_profile("T", xr, np.maximum(previous.T, state[0]))
        for k, name in enumerate(gas.species_names):
            flame.flame.set_profile(name, xr, previous.Y[k])
    flame.solve(loglevel=0, auto=previous is None)
    return flame


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mechanism")
    ap.add_argument("phase")
    ap.add_argument("out")
    ap.add_argument("--fuel", required=True)
    ap.add_argument("--oxidizer", default="O2:1,N2:3.76")
    ap.add_argument("--species")
    ap.add_argument("--T", type=float, default=300.0)
    ap.add_argument("--p", type=float, default=ct.one_atm)
    ap.add_argument("--dz", type=float, default=0.01)
    ap.add_argument("--transport", default="mixture_averaged")
    ap.add_argument("--min-speed", type=float, default=0.02)
    ap.add_argument("--width", type=float, default=0.03)
    ap.add_argument("--profiles")
    ap.add_argument("--full")
    args = ap.parse_args()

    model = {"mixture_averaged": "mixture-averaged", "unity_lewis": "unity-Lewis-number"}[args.transport]
    gas = ct.Solution(args.mechanism, args.phase, transport_model=model)
    gas.TPX = args.T, args.p, args.fuel
    if args.species is None:
        burns = [k for k, s in enumerate(gas.species()) if s.composition.get("C", 0) + s.composition.get("H", 0) > 0]
        args.species = gas.species_names[max(burns, key=lambda k: gas.Y[k])]
    k_sensor = gas.species_index(args.species)
    W = gas.molecular_weights[k_sensor]
    gas.set_equivalence_ratio(1.0, args.fuel, args.oxidizer)
    z_st = gas.mixture_fraction(args.fuel, args.oxidizer)
    for d in (args.profiles, args.full):
        if d:
            os.makedirs(d, exist_ok=True)

    rows = []
    for direction in (-1, 1):
        previous = None
        for i in range(0 if direction < 0 else 1, 1000):
            Z = z_st + direction * i * args.dz
            if not 0.0 < Z < 1.0:
                break
            gas.TP = args.T, args.p
            gas.set_mixture_fraction(Z, args.fuel, args.oxidizer)
            state = gas.TPY
            phi = gas.equivalence_ratio(args.fuel, args.oxidizer)
            try:
                flame = solve(gas, state, args.width, previous)
                if previous is not None and flame.velocity[0] < args.min_speed:
                    flame = solve(gas, state, args.width, None)  # not a flame carried over from its neighbor
            except ct.CanteraError:
                print(f"Z = {Z:.4f} (phi = {phi:.3f}): no flame")
                break
            x, T = flame.grid, flame.T
            s_L = flame.velocity[0]
            if not s_L >= args.min_speed or T[-1] < T[0] + 100.0:
                print(f"Z = {Z:.4f} (phi = {phi:.3f}): s_L = {s_L:.4f} m/s, below --min-speed")
                break
            delta = (T[-1] - T[0]) / np.gradient(T, x).max()
            rate = -flame.net_production_rates[k_sensor] * W
            rows.append((Z, s_L, delta, rate.max(), phi, T[-1]))
            print(f"Z = {Z:.4f} (phi = {phi:.3f}): s_L = {s_L:.4f} m/s, delta_L = {delta * 1e3:.4f} mm, "
                  f"omega_max = {rate.max():.4g} kg/m^3/s, T_b = {T[-1]:.1f} K, {x.size} points")
            if args.profiles:
                np.savetxt(os.path.join(args.profiles, f"Z{Z:.4f}.csv"), np.column_stack([x, T, rate]),
                           delimiter=",", header="x,T,omega", comments="")
            if args.full:
                np.savetxt(os.path.join(args.full, f"Z{Z:.4f}_full.csv"),
                           np.column_stack([x, flame.velocity, T, flame.density, flame.Y.T]), delimiter=",",
                           header="x,u,T,rho," + ",".join("Y_" + s for s in gas.species_names), comments="")
            previous = flame
    rows.sort()
    with open(args.out, "w") as f:
        f.write(f"# Cantera {ct.__version__} FreeFlame, {os.path.basename(args.mechanism)} ({args.phase}), "
                f"{args.transport}, fuel {args.fuel}, oxidizer {args.oxidizer}, T = {args.T} K, p = {args.p} Pa, "
                f"omega_max of {args.species}\n")
        f.write("Z,s_L,delta_L,omega_max,phi,T_b\n")
        for r in rows:
            f.write(",".join(f"{v:.8g}" for v in r) + "\n")
    print(f"{args.out}: {len(rows)} flames, Z = {rows[0][0]:.4f} to {rows[-1][0]:.4f}")


if __name__ == "__main__":
    main()
