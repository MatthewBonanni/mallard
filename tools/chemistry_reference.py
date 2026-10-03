"""Reference data from Cantera for Mallard's chemistry tests.

Writes small CSV files to test/data/chemistry/ for the mechanisms in
mechanisms/ and test/data/chemistry/test_mechanism.yaml, so that the tests
themselves need no Cantera. Run from the repository root:

    python tools/chemistry_reference.py

Files (one set per mechanism and phase, prefix <name>):
  <name>_species.csv  species, molecular weight, thermo model, Cantera's
                      coefficient array (';'-separated)
  <name>_thermo.csv   cp/R, h/RT, s/R of each species at random temperatures,
                      some outside the fitted range (extrapolation)
  <name>_mixture.csv  mixture cp, cv, h, e per unit mass and gas constant at
                      random temperatures and mass fractions
"""
import os
import sys

import cantera as ct
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "test", "data", "chemistry")

CASES = [
    # name, file, phase, mixture states
    ("h2o2", "mechanisms/h2o2.yaml", "ohmech", 100),
    ("gri30", "mechanisms/gri30.yaml", "gri30", 40),
    ("airNASA9", "mechanisms/airNASA9.yaml", "airNASA9", 60),
    ("test_air_cp", "test/data/chemistry/test_mechanism.yaml", "air-cp", 10),
    ("test_mixed", "test/data/chemistry/test_mechanism.yaml", "mixed", 40),
]


def fmt(x):
    return "%.17g" % x


def header(f, gas, path):
    f.write(f"# Cantera {ct.__version__}, {path} phase {gas.name}\n")


def write_case(name, path, phase, n_states, rng):
    gas = ct.Solution(os.path.join(ROOT, path), phase)
    with open(os.path.join(OUT, f"{name}_species.csv"), "w") as f:
        header(f, gas, path)
        f.write("species,molecular_weight,model,coeffs\n")
        for sp in gas.species():
            model = type(sp.thermo).__name__
            coeffs = ";".join(fmt(c) for c in sp.thermo.coeffs)
            f.write(f"{sp.name},{fmt(gas.molecular_weights[gas.species_index(sp.name)])},{model},{coeffs}\n")

    with open(os.path.join(OUT, f"{name}_thermo.csv"), "w") as f:
        header(f, gas, path)
        f.write("species,T,cp_R,h_RT,s_R\n")
        for k, sp in enumerate(gas.species()):
            t_min = max(sp.thermo.min_temp, 50.0)
            t_max = min(sp.thermo.max_temp, 3.0e4)
            temps = list(rng.uniform(t_min, t_max, 4)) + [0.9 * t_min, 1.05 * t_max]
            for T in temps:
                cp = sp.thermo.cp(T) / ct.gas_constant
                h = sp.thermo.h(T) / (ct.gas_constant * T)
                s = sp.thermo.s(T) / ct.gas_constant
                f.write(f"{k},{fmt(T)},{fmt(cp)},{fmt(h)},{fmt(s)}\n")

    t_lo = max(min(sp.thermo.min_temp for sp in gas.species()), 200.0)
    t_hi = min(max(sp.thermo.max_temp for sp in gas.species()), 6000.0)
    with open(os.path.join(OUT, f"{name}_mixture.csv"), "w") as f:
        header(f, gas, path)
        cols = ["T", "cp", "cv", "h", "e", "R"] + [f"Y_{s}" for s in gas.species_names]
        f.write(",".join(cols) + "\n")
        for _ in range(n_states):
            T = rng.uniform(t_lo, t_hi)
            Y = rng.dirichlet(np.full(gas.n_species, 0.5))
            gas.TPY = T, ct.one_atm, Y
            R = ct.gas_constant / gas.mean_molecular_weight
            row = [T, gas.cp_mass, gas.cv_mass, gas.enthalpy_mass, gas.int_energy_mass, R] + list(gas.Y)
            f.write(",".join(fmt(x) for x in row) + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(20261002)
    names = sys.argv[1:]
    for name, path, phase, n_states in CASES:
        if names and name not in names:
            continue
        write_case(name, path, phase, n_states, rng)
        print("wrote", name)


if __name__ == "__main__":
    main()
