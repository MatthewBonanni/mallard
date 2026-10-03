"""V3 with the large mechanism: MallardReactor's ignition against Cantera's.

    python tools/v3_check.py MALLARD_REACTOR [WORK_DIR]

Runs MallardReactor (sparse LU) for each n-hexane/air case of
benchmarks/chemistry/nhexane_ignition.csv (tools/v3_reference.py; the
mechanism from tools/cantera_mechanisms.py) and compares the ignition delay
(within 0.5%) and T at 0.5 and 2 delays (within 1%).
"""
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    exe = os.path.abspath(sys.argv[1])
    work = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "runs", "v3")
    os.makedirs(work, exist_ok=True)
    ref_file = os.path.join(ROOT, "benchmarks", "chemistry", "nhexane_ignition.csv")
    with open(ref_file) as f:
        f.readline()
        cols = f.readline().strip().split(",")
    ref = np.loadtxt(ref_file, delimiter=",", skiprows=2, ndmin=2)
    c = {name: i for i, name in enumerate(cols)}
    mech = os.path.join(ROOT, "mechanisms", "external", "n-hexane-NUIG-2015.yaml")
    ok = True
    for row in ref:
        T0, p0, phi, tau = row[c["T0"]], row[c["p0"]], row[c["phi"]], row[c["tau"]]
        name = f"nhexane_T{T0:.0f}_phi{phi:.1f}"
        out = os.path.join(work, name + ".csv")
        with open(os.path.join(work, name + ".toml"), "w") as f:
            f.write(f'[physics]\ngas = "mixture"\nmechanism = "{mech}"\n'
                    f"[reactor]\ntype = \"constant_volume\"\nT = {T0}\np = {p0}\n"
                    f"X = {{ NC6H14 = {phi}, O2 = 9.5, N2 = 35.72 }}\nend_time = {2.0 * tau:.17g}\n"
                    f"output_interval = {0.5 * tau:.17g}\noutput = \"{out}\"\n")
        log = subprocess.run([exe, "-i", os.path.join(work, name + ".toml")], capture_output=True, text=True).stdout
        delay = float([line for line in log.splitlines() if "Ignition delay" in line][0].split()[2])
        wall = [line for line in log.splitlines() if "Wall time" in line][0].split()[2]
        data = np.loadtxt(out, delimiter=",", skiprows=1, usecols=(0, 1))
        T_half, T_2 = np.interp(0.5 * tau, data[:, 0], data[:, 1]), np.interp(2.0 * tau, data[:, 0], data[:, 1])
        errors = (delay / tau - 1, T_half / row[c["T_half_tau"]] - 1, T_2 / row[c["T_2tau"]] - 1)
        good = abs(errors[0]) < 5e-3 and abs(errors[1]) < 1e-2 and abs(errors[2]) < 1e-2
        ok &= good
        print(f"{name}: tau {delay:.6e} s vs {tau:.6e} ({100 * errors[0]:+.4f}%), T(tau/2) {100 * errors[1]:+.3f}%, "
              f"T(2 tau) {100 * errors[2]:+.3f}%, {wall} s {'ok' if good else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
