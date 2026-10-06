"""Premixed laminar flames from Cantera for V8 (docs/design/chemistry.md).

    python tools/flame_reference.py [OUT_DIR [FULL_DIR]]

Freely propagating flames (Cantera FreeFlame) of H2/air (mechanisms/h2o2.yaml)
and CH4/air (mechanisms/gri30.yaml) at 300 K and 1 atm for phi = 0.6 to 1.4,
with mixture-averaged and unity-Lewis-number transport (Cantera's default
mole-fraction flux basis, as Mallard's). Writes to OUT_DIR (default
examples/premixed_flame/reference):

  flames.csv          fuel, phi, transport, flame speed S_L, burnt T,
                      thermal thickness (T_b - T_u) / max dT/dx, grid points
  <case>.csv          x, u, T and heat release rate of each flame, on 200
                      points around the flame, for overlays

and, with FULL_DIR, each flame's full solution (x, u, T, rho, Y) as
FULL_DIR/<case>_full.csv: the initial state of Mallard's runs
(tools/flame_restart.py). <case> is e.g. "h2_phi1.0_mix".

    python tools/flame_reference.py --single FUEL PHI T_U FULL_DIR [--radiation] [--width W]

computes only the flames of one mixture at T_U (both transport models, on a
3 cm domain or --width W m) and writes their full solutions, e.g.
FULL_DIR/h2_phi0.4_T700_mix_full.csv; --radiation turns on Cantera's optically
thin radiation (H2O and CO2, no background term; "_rad" names), and FUEL
ch4_bfer is CH4/air with the two-step mechanisms/ch4_bfer.yaml.
"""
import os
import sys

import cantera as ct
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FUELS = {
    "h2": ("mechanisms/h2o2.yaml", "ohmech", "H2", 0.02),
    "ch4": ("mechanisms/gri30.yaml", "gri30", "CH4", 0.05),
    "ch4_bfer": ("mechanisms/ch4_bfer.yaml", "gas", "CH4", 0.05),
}
PHIS = [0.6, 0.8, 1.0, 1.2, 1.4]
MODELS = {"mix": "mixture-averaged", "unity": "unity-Lewis-number"}
T_U, P = 300.0, ct.one_atm


def case_name(fuel, phi, model):
    return f"{fuel}_phi{phi:.1f}_{model}"


def solve(fuel, phi, model, T_u=T_U, width=None, radiation=False):
    mech, phase, species, _ = FUELS[fuel]
    gas = ct.Solution(os.path.join(ROOT, mech), phase, transport_model=MODELS[model])
    gas.set_equivalence_ratio(phi, species, "O2:1.0, N2:3.76")
    gas.TP = T_u, P
    flame = ct.FreeFlame(gas, width=width or FUELS[fuel][3])
    flame.set_refine_criteria(ratio=2.0, slope=0.02, curve=0.02, prune=0.002)
    flame.radiation_enabled = radiation
    flame.solve(loglevel=0, auto=True)
    return flame


def write_full(f, path):
    data = np.column_stack([f.grid, f.velocity, f.T, f.density, f.Y.T])
    header = "x,u,T,rho," + ",".join("Y_" + s for s in f.gas.species_names)
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def single(fuel, phi, T_u, full, radiation=False, width=0.03):
    os.makedirs(full, exist_ok=True)
    for model in MODELS:
        f = solve(fuel, phi, model, T_u, width=width, radiation=radiation)
        delta = (f.T.max() - f.T[0]) / np.gradient(f.T, f.grid).max()
        name = f"{fuel}_phi{phi:g}_T{T_u:.0f}_{model}" + ("_rad" if radiation else "")
        print(f"{name}: S_L = {f.velocity[0]:.5f} m/s, T_b = {f.T[-1]:.1f} K, delta = {delta * 1e3:.4f} mm, "
              f"rho_u / rho_b = {f.density[0] / f.density[-1]:.3f}")
        write_full(f, os.path.join(full, name + "_full.csv"))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--single":
        width = float(sys.argv[sys.argv.index("--width") + 1]) if "--width" in sys.argv else 0.03
        single(sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5], "--radiation" in sys.argv, width)
        return
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "examples", "premixed_flame", "reference")
    full = sys.argv[2] if len(sys.argv) > 2 else None
    os.makedirs(out, exist_ok=True)
    if full:
        os.makedirs(full, exist_ok=True)
    rows = []
    for fuel in FUELS:
        for model in MODELS:
            for phi in PHIS:
                f = solve(fuel, phi, model)
                x, T = f.grid, f.T
                dTdx = np.gradient(T, x)
                delta = (T[-1] - T[0]) / dTdx.max()
                name = case_name(fuel, phi, model)
                rows.append((fuel, phi, model, f.velocity[0], T[-1], delta, x.size))
                print(f"{name}: S_L = {f.velocity[0]:.5f} m/s, T_b = {T[-1]:.1f} K, "
                      f"delta = {delta * 1e3:.4f} mm, {x.size} points")
                i_max = np.argmax(dTdx)
                xs = np.linspace(x[i_max] - 4 * delta, x[i_max] + 8 * delta, 200)
                hrr = f.heat_release_rate
                with open(os.path.join(out, name + ".csv"), "w") as fh:
                    fh.write(f"# Cantera {ct.__version__} FreeFlame, {FUELS[fuel][0]}, phi = {phi}, "
                             f"{MODELS[model]}; x relative to the max of dT/dx\n")
                    fh.write("x,u,T,hrr\n")
                    for xi in xs:
                        fh.write(f"{xi - x[i_max]:.6e},{np.interp(xi, x, f.velocity):.6e},"
                                 f"{np.interp(xi, x, T):.6e},{np.interp(xi, x, hrr):.6e}\n")
                if full:
                    write_full(f, os.path.join(full, name + "_full.csv"))
    with open(os.path.join(out, "flames.csv"), "w") as fh:
        fh.write(f"# Cantera {ct.__version__} FreeFlame, T_u = {T_U} K, p = {P} Pa, air O2:N2 = 1:3.76\n")
        fh.write("fuel,phi,transport,S_L,T_b,delta_T,points\n")
        for r in rows:
            fh.write(f"{r[0]},{r[1]:.1f},{r[2]},{r[3]:.8f},{r[4]:.4f},{r[5]:.8e},{r[6]}\n")


if __name__ == "__main__":
    main()
