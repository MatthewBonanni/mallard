"""CJ speed and ZND structure of a planar detonation, from Cantera.

The Chapman-Jouguet speed is the minimum wave speed over the equilibrium
Hugoniot; the ZND structure integrates the steady reacting flow behind a
frozen shock moving at that speed (the formulation of Shepherd's Shock and
Detonation Toolbox: thermicity sigma, 1 - M^2 in the shock frame). Usage:

    python tools/detonation_reference.py MECHANISM PHASE "X" T0 P0 [OUT.csv]

e.g. python tools/detonation_reference.py mechanisms/h2o2.yaml ohmech \\
         "H2:2, O2:1, AR:7" 298 6670 znd.csv

Prints D_CJ, the von Neumann state, the induction length (distance from the
shock to the maximum heat release rate) and the CJ state; writes the ZND profile
(x, t, T, p, rho, u in the shock frame, Y) if OUT is given.
"""
import sys

import cantera as ct
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar


def equilibrium_hugoniot(gas, h1, p1, v1, v2, T_guess):
    """Equilibrium state at specific volume v2 on the Hugoniot of (h1, p1, v1); returns p2."""
    Y1 = gas.Y.copy()

    def residual(T):
        gas.TDY = T, 1.0 / v2, Y1
        gas.equilibrate("TV")
        return gas.enthalpy_mass - h1 - 0.5 * (gas.P - p1) * (v1 + v2)

    T = brentq(residual, 0.5 * T_guess, 3.0 * T_guess, xtol=1e-12)
    residual(T)
    p2 = gas.P
    gas.TDY = gas.T, gas.density, Y1  # leave the composition unburnt for the next call
    return p2, T


def cj_speed(gas):
    T1, p1, v1, h1 = gas.T, gas.P, 1.0 / gas.density, gas.enthalpy_mass
    state = gas.TPY

    def speed(ratio):
        gas.TPY = state
        p2, _ = equilibrium_hugoniot(gas, h1, p1, v1, v1 / ratio, 3000.0)
        return v1 * np.sqrt((p2 - p1) / (v1 - v1 / ratio))

    result = minimize_scalar(speed, bounds=(1.3, 2.5), method="bounded", options={"xatol": 1e-10})
    gas.TPY = state
    return result.fun, result.x


def frozen_shock(gas, D):
    """Post-shock state of a frozen (non-reacting) shock at speed D: sets gas and returns its velocity in the shock frame."""
    rho1, p1, h1, Y1 = gas.density, gas.P, gas.enthalpy_mass, gas.Y.copy()
    m = rho1 * D

    def residual(ratio):
        rho2 = rho1 * ratio
        u2 = m / rho2
        p2 = p1 + m * D - m * u2
        gas.DPY = rho2, p2, Y1
        return gas.enthalpy_mass + 0.5 * u2 ** 2 - h1 - 0.5 * D ** 2

    ratio = brentq(residual, 1.5, 15.0, xtol=1e-14)
    residual(ratio)
    return m / (rho1 * ratio)


def znd(gas, D, t_end):
    u_vn = frozen_shock(gas, D)
    W = gas.molecular_weights

    def rhs(t, z):
        p, rho, u, x = z[:4]
        Y = z[4:]
        gas.DPY = rho, p, Y
        omega = gas.net_production_rates
        dY = W * omega / rho
        cp = gas.cp_mass
        sigma = np.dot(gas.mean_molecular_weight / W - gas.partial_molar_enthalpies / (W * cp * gas.T), dY)
        a2 = gas.cp_mass / gas.cv_mass * p / rho
        eta = 1.0 - u * u / a2
        return np.concatenate(([-rho * u * u * sigma / eta, -rho * sigma / eta, u * sigma / eta, u], dY))

    z0 = np.concatenate(([gas.P, gas.density, u_vn, 0.0], gas.Y))
    sol = solve_ivp(rhs, (0.0, t_end), z0, method="Radau", rtol=1e-10, atol=1e-14, dense_output=False)
    return sol


def main():
    mech, phase, X, T0, p0 = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5])
    out = sys.argv[6] if len(sys.argv) > 6 else None
    gas = ct.Solution(mech, phase)
    gas.TPX = T0, p0, X
    D, ratio = cj_speed(gas)
    print(f"D_CJ = {D:.3f} m/s (density ratio {ratio:.4f})")
    gas.TPX = T0, p0, X
    sol = znd(gas, D, 1e-3)
    p, rho, u, x = sol.y[:4]
    Ts, hrr = [], []
    for i in range(sol.t.size):
        gas.DPY = rho[i], p[i], sol.y[4:, i]
        Ts.append(gas.T)
        hrr.append(-np.dot(gas.partial_molar_enthalpies, gas.net_production_rates))
    Ts = np.array(Ts)
    i_max = int(np.argmax(hrr))
    print(f"von Neumann: T = {Ts[0]:.2f} K, p = {p[0]:.1f} Pa, u = {u[0]:.2f} m/s (shock frame)")
    print(f"induction length (max heat release rate) = {x[i_max] * 1e3:.4f} mm")
    print(f"end state: T = {Ts[-1]:.2f} K, p = {p[-1]:.1f} Pa, x = {x[-1] * 1e3:.2f} mm")
    if out:
        with open(out, "w") as f:
            f.write("# ZND profile, " + f"{mech} {phase} X = {X}, T0 = {T0} K, p0 = {p0} Pa, D_CJ = {D:.6f} m/s\n")
            f.write("x,t,T,p,rho,u," + ",".join("Y_" + s for s in gas.species_names) + "\n")
            for i in range(sol.t.size):
                row = [x[i], sol.t[i], Ts[i], p[i], rho[i], u[i]] + list(sol.y[4:, i])
                f.write(",".join("%.10e" % v for v in row) + "\n")


if __name__ == "__main__":
    main()
