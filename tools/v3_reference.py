"""V3 reference data: constant-volume ignition with large mechanisms (Cantera).

    python tools/v3_reference.py

Writes, in the format of tools/chemistry_reference.py's <name>_ignition.csv
(initial state, ignition delay as the time of max dT/dt, T at 0.5 and 2
delays, the UV equilibrium state):
  test/data/chemistry/ndodecane_ignition.csv    n-dodecane/air, Wang et al.
      (100 species), 20 atm, phi 0.5-2, T0 1000-1400 K: the unit tests
  benchmarks/chemistry/nhexane_ignition.csv     n-hexane/air, NUIG (1268
      species, mechanisms/external, see tools/cantera_mechanisms.py), 10 atm,
      phi 1, T0 1000-1400 K, without the compositions: tools/v3_check.py
"""
import os
import sys

import cantera as ct
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chemistry_reference import dT_dt, header, peak_time, reactor  # noqa: E402

CASES = [
    # output, mechanism, phase, fuel, O2 / N2 names, O2 per fuel, pressures [atm], phis, T0s
    ("test/data/chemistry/ndodecane_ignition.csv", "mechanisms/nDodecane_Reitz.yaml", "nDodecane_IG", "c12h26",
     ("o2", "n2"), 18.5, (20.0,), (0.5, 1.0, 2.0), (1000.0, 1200.0, 1400.0), True),
    ("benchmarks/chemistry/nhexane_ignition.csv", "mechanisms/external/n-hexane-NUIG-2015.yaml", "gas", "NC6H14",
     ("O2", "N2"), 9.5, (10.0,), (1.0,), (1000.0, 1200.0, 1400.0), False),
]


def main():
    for out, path, phase, fuel, (o2, n2), o2_per_fuel, pressures, phis, T0s, compositions in CASES:
        gas = ct.Solution(os.path.join(ROOT, path), phase)
        with open(os.path.join(ROOT, out), "w") as f:
            header(f, gas, path)
            species = gas.species_names if compositions else []
            cols = (["T0", "p0", "phi", "rho"] + [f"Y0_{s}" for s in species] +
                    ["tau", "T_half_tau", "T_2tau", "T_eq"] + [f"Yeq_{s}" for s in species])
            f.write(",".join(cols) + "\n")
            for p_atm in pressures:
                for phi in phis:
                    for T0 in T0s:
                        gas.TPX = T0, p_atm * ct.one_atm, {fuel: phi, o2: o2_per_fuel, n2: 3.76 * o2_per_fuel}
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
                        row = ([T0, p_atm * ct.one_atm, phi, initial[1]] + (list(initial[2]) if compositions else []) +
                               [tau, T_half, T_2, gas.T] + (list(gas.Y) if compositions else []))
                        f.write(",".join("%.16e" % x for x in row) + "\n")
                        print(f"{out}: T0 {T0} phi {phi}: tau {tau:.6e} s")


if __name__ == "__main__":
    main()
