"""Initial state of a 1D detonation run from a ZND profile, as a Mallard restart file.

    python tools/znd_restart.py ZND.csv MECHANISM PHASE NX LX X_SHOCK OUT.restart [DIM]

ZND.csv comes from tools/detonation_reference.py. The mesh is Mallard's
generated "cartesian" box with NX x 1 (x 1) cells over [0, LX]; the shock
sits at X_SHOCK moving to +x at D_CJ (read from the file's header). Behind it
each cell takes the ZND state at its distance from the shock, in the lab
frame (beyond the profile's end, the end state); ahead of it the unburnt gas
at rest (the ZND file's first state upstream: T0, p0 and the initial
composition). Cell averages are taken over 16 points per cell, so the shock
lands inside a cell as a mixed state. Values are written as float64 (double
builds).
"""
import re
import struct
import sys

import cantera as ct
import numpy as np


def main():
    znd_file, mech, phase = sys.argv[1], sys.argv[2], sys.argv[3]
    nx, lx, x_shock, out = int(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), sys.argv[7]
    dim = int(sys.argv[8]) if len(sys.argv) > 8 else 2
    header = open(znd_file).readline()
    D = float(re.search(r"D_CJ = ([0-9.eE+-]+)", header).group(1))
    T0 = float(re.search(r"T0 = ([0-9.eE+-]+)", header).group(1))
    p0 = float(re.search(r"p0 = ([0-9.eE+-]+)", header).group(1))
    X0 = re.search(r"X = (.*), T0", header).group(1)
    data = np.loadtxt(znd_file, delimiter=",", skiprows=2)
    xi, rho_z, u_z = data[:, 0], data[:, 4], data[:, 5]
    Y_z = data[:, 6:]
    gas = ct.Solution(mech, phase)
    ns = gas.n_species
    gas.TPX = T0, p0, X0
    rho0, Y0, e0 = gas.density, gas.Y.copy(), gas.int_energy_mass
    # Conserved quantities per unit volume at the ZND points: rho, rho u, rho E, rho Y
    e_z = np.empty(xi.size)
    T_z = data[:, 2]
    for i in range(xi.size):
        gas.TDY = T_z[i], rho_z[i], Y_z[i]
        e_z[i] = gas.int_energy_mass
    u_lab = D - u_z

    def state(x):
        s = x_shock - x
        if s < 0.0:
            return rho0, 0.0, rho0 * e0, rho0 * Y0, T0
        j = min(np.searchsorted(xi, s), xi.size - 1)
        r = np.interp(s, xi, rho_z)
        u = np.interp(s, xi, u_lab)
        e = np.interp(s, xi, e_z)
        Y = np.array([np.interp(s, xi, Y_z[:, k]) for k in range(ns)]) if j < xi.size - 1 else Y_z[-1]
        T = np.interp(s, xi, T_z)
        return r, r * u, r * (e + 0.5 * u * u), r * Y, T

    dx = lx / nx
    n_sub = 16
    cons = np.zeros((nx, 3 + ns))
    T_seed = np.zeros(nx)
    for c in range(nx):
        acc = np.zeros(3 + ns)
        Tm = 0.0
        for q in range(n_sub):
            r, ru, rE, rY, T = state((c + (q + 0.5) / n_sub) * dx)
            acc += np.concatenate(([r, ru, rE], rY))
            Tm += T / n_sub
        cons[c] = acc / n_sub
        T_seed[c] = Tm
    names = ["RHO", "RHOU_X", "RHOU_Y"] + (["RHOU_Z"] if dim == 3 else []) + ["RHOE"]
    names += ["RHOY_" + s for s in gas.species_names] + ["T_SEED"]
    zeros = np.zeros(nx)
    fields = [cons[:, 0], cons[:, 1], zeros] + ([zeros] if dim == 3 else []) + [cons[:, 2]]
    fields += [cons[:, 3 + k] for k in range(ns)] + [T_seed]
    with open(out, "wb") as f:
        f.write(b"MALLARD-RESTART\0")
        f.write(struct.pack("<IIQQQd", 2, 8, nx, len(names), 0, 0.0))
        for name in names:
            f.write(struct.pack("<I", len(name)) + name.encode())
        for field in fields:
            f.write(np.asarray(field, dtype="<f8").tobytes())
    print(f"wrote {out}: {nx} cells, D_CJ = {D} m/s, shock at {x_shock} m")


if __name__ == "__main__":
    main()
