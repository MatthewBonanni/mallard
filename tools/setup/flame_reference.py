"""Premixed laminar flames from Cantera for V8 (docs/design/chemistry.md).

    python tools/setup/flame_reference.py [OUT_DIR [FULL_DIR]]

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
(tools/setup/flame_restart.py). <case> is e.g. "h2_phi1.0_mix".

    python tools/setup/flame_reference.py --single FUEL PHI T_U FULL_DIR [--radiation] [--width W]

computes only the flames of one mixture at T_U (both transport models, on a
3 cm domain or --width W m) and writes their full solutions, e.g.
FULL_DIR/h2_phi0.4_T700_mix_full.csv; --radiation turns on Cantera's optically
thin radiation (H2O and CO2, no background term; "_rad" names), and FUEL
ch4_bfer is CH4/air with the two-step mechanisms/ch4_bfer.yaml.

    python tools/setup/flame_reference.py --soret [OUT_DIR [FULL_DIR]]

computes the lean-to-stoichiometric H2/air flames of thermal diffusion
(phi = SORET_PHIS, on a 3 cm domain) with mixture-averaged transport with and
without the Soret effect ("mix_soret", "mix"), the transport Mallard runs, and
for comparison with multicomponent transport with and without it ("multi_soret",
"multi"). Writes OUT_DIR/soret_flames.csv (as flames.csv) and the profiles of
the mixture-averaged flames with Soret, with species mass fractions
(<case>.csv, e.g. h2_phi0.5_mix_soret.csv), and with FULL_DIR the full
solutions of the mixture-averaged flames.

    python tools/setup/flame_reference.py --convergence

prints S_L of every flame above on each refinement stage of converge() and on
one more (slope = curve = 0.000625).

Every flame is solved on a grid refined in stages to slope = curve = 0.00125
(converge()): Cantera's convection is first-order upwind, so S_L depends on
the grid, and one more halving changes it by at most 0.07% (--convergence).
"""
import os
import sys

import cantera as ct
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FUELS = {
    "h2": ("mechanisms/h2o2.yaml", "ohmech", "H2", 0.02),
    "ch4": ("mechanisms/gri30.yaml", "gri30", "CH4", 0.05),
    "ch4_bfer": ("mechanisms/ch4_bfer.yaml", "gas", "CH4", 0.05),
}
V8_FUELS = ["h2", "ch4"]
PHIS = [0.6, 0.8, 1.0, 1.2, 1.4]
MODELS = {"mix": "mixture-averaged", "unity": "unity-Lewis-number"}
SORET_PHIS = [0.4, 0.5, 0.6, 1.0]
SORET_MODELS = {"mix": ("mixture-averaged", False), "mix_soret": ("mixture-averaged", True),
                "multi": ("multicomponent", False), "multi_soret": ("multicomponent", True)}
PROFILE_SPECIES = ["H2", "O2", "H2O", "H", "O", "OH", "HO2"]
T_U, P = 300.0, ct.one_atm


def case_name(fuel, phi, model):
    return f"{fuel}_phi{phi:.1f}_{model}"


def solve(fuel, phi, model, T_u=T_U, width=None, soret=False, radiation=False):
    mech, phase, species, _ = FUELS[fuel]
    gas = ct.Solution(os.path.join(ROOT, mech), phase, transport_model=MODELS.get(model, model))
    gas.set_equivalence_ratio(phi, species, "O2:1.0, N2:3.76")
    gas.TP = T_u, P
    flame = ct.FreeFlame(gas, width=width or FUELS[fuel][3])
    flame.soret_enabled = soret
    flame.radiation_enabled = radiation
    return converge(flame)


CRITERIA = [0.02, 0.01, 0.005, 0.0025, 0.00125]  # slope = curve of the refinement stages


def refinements(flame, criteria=CRITERIA):
    """Solve a FreeFlame on grids refined in stages, slope = curve = each of criteria; yields after each."""
    flame.set_max_grid_points(flame.flame, 100000)
    for i, c in enumerate(criteria):
        flame.set_refine_criteria(ratio=2.0, slope=c, curve=c, prune=c / 10 if i == 0 else 0.0)
        flame.solve(loglevel=0, auto=i == 0, refine_grid=True)
        yield c


def converge(flame):
    """Solve a FreeFlame on the grid of slope = curve = 0.00125.

    Cantera's convection is first-order upwind: S_L on the grid of 0.02 is
    up to 2.4% off (lean H2/air); one more halving than 0.00125 changes it
    by at most 0.07% (--convergence).
    """
    for _ in refinements(flame):
        pass
    return flame


def convergence():
    """S_L of every flame on each refinement stage and on one more (slope = curve = 0.000625)."""
    cases = [(fuel, phi, model, False, FUELS[fuel][3]) for fuel in V8_FUELS for model in MODELS for phi in PHIS]
    cases += [("h2", phi, "mix", soret, 0.03) for phi in SORET_PHIS for soret in (False, True)]
    for fuel, phi, model, soret, width in cases:
        mech, phase, species, _ = FUELS[fuel]
        gas = ct.Solution(os.path.join(ROOT, mech), phase, transport_model=MODELS[model])
        gas.set_equivalence_ratio(phi, species, "O2:1.0, N2:3.76")
        gas.TP = T_U, P
        flame = ct.FreeFlame(gas, width=width)
        flame.soret_enabled = soret
        s = [(c, flame.velocity[0], flame.grid.size) for c in refinements(flame, CRITERIA + [CRITERIA[-1] / 2])]
        print(f"{case_name(fuel, phi, model)}{'_soret' if soret else ''}: "
              + ", ".join(f"{c:g}: {v:.6f} ({n})" for c, v, n in s)
              + f"; 0.02 vs {CRITERIA[-1]:g}: {100 * (s[0][1] / s[-2][1] - 1):+.3f}%, "
              f"{CRITERIA[-1]:g} vs {CRITERIA[-1] / 2:g}: {100 * (s[-1][1] / s[-2][1] - 1):+.3f}%", flush=True)


def write_full(f, path):
    data = np.column_stack([f.grid, f.velocity, f.T, f.density, f.Y.T])
    header = "x,u,T,rho," + ",".join("Y_" + s for s in f.gas.species_names)
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def single(fuel, phi, T_u, full, radiation=False, width=0.03):
    os.makedirs(full, exist_ok=True)
    for model in MODELS:
        try:
            f = solve(fuel, phi, model, T_u, width=width, radiation=radiation)
        except ct.CanteraError as e:
            print(f"{fuel} phi = {phi:g}, {MODELS[model]}: no solution ({str(e).strip().splitlines()[-1]})")
            continue
        delta = (f.T.max() - f.T[0]) / np.gradient(f.T, f.grid).max()
        name = f"{fuel}_phi{phi:g}_T{T_u:.0f}_{model}" + ("_rad" if radiation else "")
        print(f"{name}: S_L = {f.velocity[0]:.5f} m/s, T_b = {f.T[-1]:.1f} K, delta = {delta * 1e3:.4f} mm, "
              f"rho_u / rho_b = {f.density[0] / f.density[-1]:.3f}")
        write_full(f, os.path.join(full, name + "_full.csv"))


def soret_flames(out, full):
    os.makedirs(out, exist_ok=True)
    if full:
        os.makedirs(full, exist_ok=True)
    rows = []
    for phi in SORET_PHIS:
        for name, (model, soret) in SORET_MODELS.items():
            f = solve("h2", phi, model, width=0.03, soret=soret)
            x, T = f.grid, f.T
            dTdx = np.gradient(T, x)
            delta = (T[-1] - T[0]) / dTdx.max()
            case = case_name("h2", phi, name)
            rows.append(("h2", phi, name, f.velocity[0], T[-1], delta, x.size))
            print(f"{case}: S_L = {f.velocity[0]:.5f} m/s, T_b = {T[-1]:.1f} K, "
                  f"delta = {delta * 1e3:.4f} mm, {x.size} points")
            if not name.startswith("mix"):
                continue
            if name == "mix_soret":
                i_max = np.argmax(dTdx)
                xs = np.linspace(x[i_max] - 4 * delta, x[i_max] + 8 * delta, 200)
                columns = [f.velocity, T, f.heat_release_rate] + [f.Y[f.gas.species_index(s)] for s in PROFILE_SPECIES]
                with open(os.path.join(out, case + ".csv"), "w") as fh:
                    fh.write(f"# Cantera {ct.__version__} FreeFlame, {FUELS['h2'][0]}, phi = {phi}, "
                             f"mixture-averaged with the Soret effect; x relative to the max of dT/dx\n")
                    fh.write("x,u,T,hrr," + ",".join("Y_" + s for s in PROFILE_SPECIES) + "\n")
                    for xi in xs:
                        fh.write(f"{xi - x[i_max]:.6e}," + ",".join(f"{np.interp(xi, x, c):.6e}" for c in columns) + "\n")
            if full:
                write_full(f, os.path.join(full, case + "_full.csv"))
    with open(os.path.join(out, "soret_flames.csv"), "w") as fh:
        fh.write(f"# Cantera {ct.__version__} FreeFlame, T_u = {T_U} K, p = {P} Pa, air O2:N2 = 1:3.76, 3 cm domain\n")
        fh.write("fuel,phi,transport,S_L,T_b,delta_T,points\n")
        for r in rows:
            fh.write(f"{r[0]},{r[1]:.1f},{r[2]},{r[3]:.8f},{r[4]:.4f},{r[5]:.8e},{r[6]}\n")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--convergence":
        convergence()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "--soret":
        out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "examples", "premixed_flame", "reference")
        soret_flames(out, sys.argv[3] if len(sys.argv) > 3 else None)
        return
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
    for fuel in V8_FUELS:
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
