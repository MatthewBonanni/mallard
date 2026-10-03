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
  <name>_rates.csv    net rates of progress and production rates at random
                      temperatures, densities and mass fractions (mechanisms
                      with reactions)
  <name>_ignition.csv adiabatic constant-volume ignition of fuel/air: initial
                      state, ignition delay (time of max dT/dt), T at 0.5 and
                      2 delays, and the UV equilibrium state
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


KINETICS = [
    # name, file, phase, states
    ("h2o2", "mechanisms/h2o2.yaml", "ohmech", 40),
    ("gri30", "mechanisms/gri30.yaml", "gri30", 12),
    ("test_kinetics", "test/data/chemistry/test_kinetics.yaml", "gas", 40),
]


def write_rates(name, path, phase, n_states):
    rng = np.random.default_rng(sum(map(ord, name)))
    gas = ct.Solution(os.path.join(ROOT, path), phase)
    with open(os.path.join(OUT, f"{name}_rates.csv"), "w") as f:
        header(f, gas, path)
        cols = (["T", "rho"] + [f"Y_{s}" for s in gas.species_names] +
                [f"q_{i}" for i in range(gas.n_reactions)] + [f"omega_{s}" for s in gas.species_names])
        f.write(",".join(cols) + "\n")
        for _ in range(n_states):
            T = rng.uniform(800.0, 2500.0)
            p = 10.0 ** rng.uniform(4.0, 6.0)
            Y = rng.dirichlet(np.full(gas.n_species, 0.5))
            gas.TPY = T, p, Y
            row = ([T, gas.density] + list(gas.Y) + list(gas.net_rates_of_progress) +
                   list(gas.net_production_rates))
            f.write(",".join("%.16e" % x for x in row) + "\n")


IGNITION = [
    # name, file, phase, fuel, O2 per fuel at phi = 1
    ("h2o2", "mechanisms/h2o2.yaml", "ohmech", "H2", 0.5),
    ("gri30", "mechanisms/gri30.yaml", "gri30", "CH4", 2.0),
]


def dT_dt(gas):
    """Temperature rate of an adiabatic constant-volume reactor."""
    return -np.dot(gas.partial_molar_int_energies, gas.net_production_rates) / (gas.density * gas.cv_mass)


def peak_time(t, g):
    """Time of the maximum of g: vertex of the parabola through the largest sample and its neighbors."""
    i = int(np.argmax(g))
    (t0, t1, t2), (g0, g1, g2) = t[i - 1:i + 2], g[i - 1:i + 2]
    num = (t1 - t0) ** 2 * (g1 - g2) - (t1 - t2) ** 2 * (g1 - g0)
    den = (t1 - t0) * (g1 - g2) - (t1 - t2) * (g1 - g0)
    return t1 - 0.5 * num / den


def reactor(gas):
    r = ct.IdealGasReactor(gas, clone=False)
    net = ct.ReactorNet([r])
    net.rtol, net.atol = 1e-12, 1e-22
    net.max_steps = 1000000
    return r, net


def write_ignition(name, path, phase, fuel, o2_per_fuel):
    gas = ct.Solution(os.path.join(ROOT, path), phase)
    with open(os.path.join(OUT, f"{name}_ignition.csv"), "w") as f:
        header(f, gas, path)
        cols = (["T0", "p0", "phi", "rho"] + [f"Y0_{s}" for s in gas.species_names] +
                ["tau", "T_half_tau", "T_2tau", "T_eq"] + [f"Yeq_{s}" for s in gas.species_names])
        f.write(",".join(cols) + "\n")
        for p_atm in (1.0, 10.0):
            for phi in (0.5, 1.0, 2.0):
                for T0 in (1000.0, 1200.0, 1500.0):
                    X = {fuel: phi, "O2": o2_per_fuel, "N2": 3.76 * o2_per_fuel}
                    gas.TPX = T0, p_atm * ct.one_atm, X
                    initial = gas.TDY
                    r, net = reactor(gas)
                    t, g = [0.0], [dT_dt(r.phase)]
                    while r.phase.T < T0 + 400.0 or g[-1] > 0.01 * max(g):
                        net.step()
                        t.append(net.time)
                        g.append(dT_dt(r.phase))
                    tau = peak_time(np.array(t), np.array(g))
                    gas.TDY = initial
                    r, net = reactor(gas)
                    net.advance(0.5 * tau)
                    T_half = r.phase.T
                    net.advance(2.0 * tau)
                    T_2 = r.phase.T
                    gas.TDY = initial
                    gas.equilibrate("UV")
                    row = ([T0, p_atm * ct.one_atm, phi, initial[1]] + list(initial[2]) +
                           [tau, T_half, T_2, gas.T] + list(gas.Y))
                    f.write(",".join("%.16e" % x for x in row) + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(20261002)
    names = sys.argv[1:]
    for name, path, phase, n_states in CASES:
        if names and name not in names:
            continue
        write_case(name, path, phase, n_states, rng)
        print("wrote", name)
    for name, path, phase, n_states in KINETICS:
        if names and name + "_rates" not in names:
            continue
        write_rates(name, path, phase, n_states)
        print("wrote", name, "rates")
    for name, path, phase, fuel, o2 in IGNITION:
        if names and name + "_ignition" not in names:
            continue
        write_ignition(name, path, phase, fuel, o2)
        print("wrote", name, "ignition")


if __name__ == "__main__":
    main()
