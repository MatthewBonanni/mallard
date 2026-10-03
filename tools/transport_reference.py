"""Transport reference data from Cantera for Mallard's transport tests.

Writes small CSV files to test/data/chemistry/ so that the tests need no
Cantera. Run from the repository root:

    python tools/transport_reference.py

Files (one set per mechanism, prefix <name>):
  <name>_species_transport.csv  per temperature: each species' viscosity and
                                conductivity, and the binary diffusion
                                coefficient of each pair (k <= j) at 1 atm
  <name>_mixture_transport.csv  at random temperatures, pressures and mass
                                fractions: the mixture-averaged viscosity,
                                conductivity and diffusion coefficients, and
                                the unity-Lewis-number diffusion coefficient
"""
import os

import cantera as ct
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "test", "data", "chemistry")

CASES = [
    # name, file, phase, temperatures, mixture states
    ("h2o2", "mechanisms/h2o2.yaml", "ohmech", 12, 60),
    ("gri30", "mechanisms/gri30.yaml", "gri30", 4, 30),
]


def write(path, header, columns, rows):
    with open(path, "w") as f:
        f.write(f"# {header}\n")
        f.write(",".join(columns) + "\n")
        for r in rows:
            f.write(",".join(f"{v:.16e}" for v in r) + "\n")


def main():
    rng = np.random.default_rng(20261003)
    for name, mech, phase, n_T, n_mix in CASES:
        gas = ct.Solution(os.path.join(ROOT, mech), phase, transport_model="mixture-averaged")
        unity = ct.Solution(os.path.join(ROOT, mech), phase, transport_model="unity-Lewis-number")
        ns = gas.n_species
        names = gas.species_names
        header = f"Cantera {ct.__version__}, {mech} phase {phase}"
        T_lo, T_hi = gas.min_temp, gas.max_temp
        pairs = [(k, j) for k in range(ns) for j in range(k, ns)]

        columns = ["T"] + [f"mu_{s}" for s in names] + [f"lambda_{s}" for s in names]
        columns += [f"D_{names[k]}_{names[j]}" for k, j in pairs]
        rows = []
        for T in np.linspace(T_lo + 13.7, T_hi - 21.3, n_T):
            gas.TPX = T, ct.one_atm, {names[0]: 1.0}
            mu = gas.species_viscosities
            sqrt_T, log_T = np.sqrt(T), np.log(T)
            lam = [sqrt_T * np.polyval(gas.get_thermal_conductivity_polynomial(k)[::-1], log_T) for k in range(ns)]
            D = gas.binary_diff_coeffs
            rows.append([T] + list(mu) + lam + [D[k, j] for k, j in pairs])
        write(os.path.join(OUT, f"{name}_species_transport.csv"), header, columns, rows)

        columns = ["T", "p"] + [f"Y_{s}" for s in names] + ["mu", "lambda"]
        columns += [f"D_{s}" for s in names] + ["D_unity"]
        rows = []
        for i in range(n_mix):
            T = rng.uniform(T_lo, T_hi)
            p = ct.one_atm * 10 ** rng.uniform(-1, 1)
            Y = rng.uniform(0, 1, ns) ** 4
            Y[rng.uniform(0, 1, ns) < 0.3] = 0.0  # some absent species
            if i == 0:
                Y = np.zeros(ns)  # one pure species
                Y[ns - 1] = 1.0
            Y /= Y.sum()
            gas.TPY = T, p, Y
            unity.TPY = T, p, Y
            rows.append([T, p] + list(Y) + [gas.viscosity, gas.thermal_conductivity] + list(gas.mix_diff_coeffs) +
                        [unity.mix_diff_coeffs[0]])
        write(os.path.join(OUT, f"{name}_mixture_transport.csv"), header, columns, rows)
        print(f"{name}: {ns} species")


if __name__ == "__main__":
    main()
