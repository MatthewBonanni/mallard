"""0D references for the stratified autoignition runs (examples/autoignition_2d).

    python tools/autoignition_reference.py OUT_DIR [--T0 1070] [--p0 41] [--phi 0.1]
        [--restart RUN_DIR/autoignition.restart ...] [--zones 40] [--t-end 8e-3]
        [--mechanism mechanisms/h2o2.yaml --phase ohmech]

Writes, with Cantera:

- OUT_DIR/homogeneous.csv: the homogeneous constant-volume reactor at the mean
  state (t, T, p, HRR), whose ignition delay tau_0 (the time of the largest
  dT/dt) scales the runs' time;
- OUT_DIR/ignition_delay.csv: the ignition delay against the initial
  temperature at p0 (T, tau, dtau/dT), for the spontaneous-propagation speed
  S_sp = 1 / |grad tau| = 1 / (|dtau/dT| |grad T|) of Sankaran et al. (2005);
- for each --restart, OUT_DIR/multizone_<run>.csv: the multizone model of the
  run (t, p, mean HRR): the initial temperatures binned into --zones zones of
  equal mass that react as adiabatic closed reactors, without transport, at a
  common pressure in the fixed total volume (so that the zones that ignite
  first compress the others), as the zero-transport limit of the run. A run
  that burns only by spontaneous ignition fronts follows it; deflagrations
  (molecular transport) burn the colder zones sooner.
"""
import argparse
import os
import struct

import cantera as ct
import numpy as np
from scipy.integrate import solve_ivp


def read_restart(path):
    with open(path, "rb") as f:
        raw = f.read()
    _, _, n, nv, _, _ = struct.unpack_from("<IIQQQd", raw, 16)
    off = 16 + struct.calcsize("<IIQQQd")
    names = []
    for _ in range(nv):
        (ln,) = struct.unpack_from("<I", raw, off)
        names.append(raw[off + 4:off + 4 + ln].decode())
        off += 4 + ln
    data = np.frombuffer(raw, "<f8", n * nv, off).reshape(nv, n)
    return dict(zip(names, data))


def homogeneous(gas, T, p, Y, t_end):
    gas.TPY = T, p, Y
    r = ct.IdealGasReactor(gas, clone=True)
    net = ct.ReactorNet([r])
    rows = [(0.0, r.T, r.phase.P, r.phase.heat_release_rate)]
    while net.time < t_end:
        net.step()
        rows.append((net.time, r.T, r.phase.P, r.phase.heat_release_rate))
    rows = np.array(rows)
    tau = rows[np.argmax(np.gradient(rows[:, 1], rows[:, 0])), 0]
    return rows, tau


def multizone(mech, phase, T_zones, p, Y, t_end):
    """Equal-mass closed zones at a common pressure in a fixed volume."""
    states = ct.SolutionArray(ct.Solution(mech, phase), len(T_zones))
    states.TPY = T_zones, p, Y
    W = states.molecular_weights
    nz, ns = len(T_zones), W.size

    def rates(y):
        T = y[:nz]
        Yk = np.clip(y[nz:nz + nz * ns].reshape(nz, ns), 0.0, None)
        P = y[-1]
        states.TPY = T, P, Yk
        rho = states.density
        dY = states.net_production_rates * W / rho[:, None]
        u_k = states.partial_molar_int_energies / W  # J/kg per species
        return T, P, Yk, rho, dY, u_k

    def rhs(_, y):
        T, P, Yk, rho, dY, u_k = rates(y)
        v = 1.0 / rho
        cp = states.cp_mass
        Wbar = states.mean_molecular_weight
        g = Wbar * (dY / W).sum(axis=1)  # d ln(1/Wbar) / dt
        # Energy of each closed zone: cv dT + sum u_k dY_k = -P dv, with
        # dv / v = dT / T - dP / P + g, gives dT = A + B dP
        A = (-(u_k * dY).sum(axis=1) - P * v * g) / cp
        B = v / cp
        # Fixed total volume: sum v (dT / T - dP / P + g) = 0 over equal masses
        dP = -(v * (A / T + g)).sum() / (v * (B / T - 1.0 / P)).sum()
        dT = A + B * dP
        return np.concatenate([dT, dY.ravel(), [dP]])

    y0 = np.concatenate([T_zones, np.tile(Y, nz), [p]])
    sol = solve_ivp(rhs, (0.0, t_end), y0, method="BDF", rtol=1e-7, atol=1e-12, dense_output=True)
    t = np.linspace(0.0, t_end, 2001)
    hrr = np.empty_like(t)
    P = np.empty_like(t)
    for i, ti in enumerate(t):
        T, Pi, Yk, rho, _, _ = rates(sol.sol(ti))
        # Mean HRR per unit volume: sum of zone volumes (1 / rho per unit mass) times HRR
        hrr[i] = (states.heat_release_rate / rho).sum() / (1.0 / rho).sum()
        P[i] = Pi
    return t, P, hrr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--T0", type=float, default=1070.0)
    ap.add_argument("--p0", type=float, default=41.0)
    ap.add_argument("--phi", type=float, default=0.1)
    ap.add_argument("--restart", action="append", default=[])
    ap.add_argument("--zones", type=int, default=40)
    ap.add_argument("--t-end", type=float, default=8e-3)
    ap.add_argument("--mechanism", default=os.path.join(os.path.dirname(__file__), "..", "mechanisms", "h2o2.yaml"))
    ap.add_argument("--phase", default="ohmech")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    gas = ct.Solution(args.mechanism, args.phase)
    gas.set_equivalence_ratio(args.phi, "H2", "O2:1, N2:3.76")
    Y = gas.Y.copy()
    p = args.p0 * ct.one_atm

    rows, tau0 = homogeneous(gas, args.T0, p, Y, args.t_end)
    np.savetxt(os.path.join(args.out_dir, "homogeneous.csv"), rows, delimiter=",",
               header="t,T,p,HRR", comments="")
    i = np.argmax(rows[:, 3])
    print(f"homogeneous: tau_0 = {tau0 * 1e3:.4f} ms, peak HRR {rows[i, 3]:.4e} W/m^3 at {rows[i, 0] * 1e3:.4f} ms, "
          f"T_end = {rows[-1, 1]:.1f} K, p_end = {rows[-1, 2] / ct.one_atm:.2f} atm")

    Ts = np.arange(args.T0 - 120, args.T0 + 121, 5.0)
    taus = np.array([homogeneous(gas, T, p, Y, 0.05)[1] for T in Ts])
    np.savetxt(os.path.join(args.out_dir, "ignition_delay.csv"),
               np.column_stack([Ts, taus, np.gradient(taus, Ts)]), delimiter=",", header="T,tau,dtau_dT",
               comments="")
    print(f"dtau/dT at T0: {np.interp(args.T0, Ts, np.gradient(taus, Ts)) * 1e6:.2f} us/K")

    for path in args.restart:
        f = read_restart(path)
        T = np.sort(f["T_SEED"])
        rho = f["RHO"][np.argsort(f["T_SEED"])]
        # Equal-mass zones: mass-weighted mean temperature of each mass slice
        cm = np.cumsum(rho) / rho.sum()
        edges = np.searchsorted(cm, np.linspace(0, 1, args.zones + 1)[1:-1])
        T_zones = np.array([np.average(s, weights=w) for s, w in
                            zip(np.split(T, edges), np.split(rho, edges))])
        t, P, hrr = multizone(args.mechanism, args.phase, T_zones, p, Y, args.t_end)
        run = os.path.basename(os.path.dirname(os.path.abspath(path)))
        np.savetxt(os.path.join(args.out_dir, f"multizone_{run}.csv"), np.column_stack([t, P, hrr]),
                   delimiter=",", header="t,p,HRR", comments="")
        k = np.argmax(hrr)
        print(f"multizone {run}: T' = {T.std():.2f} K, peak HRR {hrr[k]:.4e} W/m^3 at {t[k] * 1e3:.4f} ms "
              f"({t[k] / tau0:.3f} tau_0), p_end = {P[-1] / ct.one_atm:.2f} atm")


if __name__ == "__main__":
    main()
