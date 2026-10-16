#!/usr/bin/env python3
"""Exact inviscid supersonic flow over a sharp cone at zero incidence
(Taylor & Maccoll 1933): the flow between the attached conical shock and the
cone depends only on the angle theta from the axis.

    python taylor_maccoll.py [--mach 2] [--cone 15] [--gamma 1.4] [--csv taylor_maccoll.csv]

Prints the shock angle and the surface pressure coefficient and Mach number,
and writes the solution between cone and shock (theta in degrees, p / p_inf,
rho / rho_inf, Mach, u_x / U_inf) to the CSV file.

Velocities are scaled by the maximum speed V_max = sqrt(2 h0); with f = V_r
and f' = V_theta the Taylor-Maccoll equation is integrated inward from the
state behind the oblique shock until V_theta = 0, which happens at the cone
surface; the shock angle is found by root finding on that surface angle.
"""
import argparse

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


def oblique_shock(M1, beta, g):
    """Mach number, deflection, pressure and density ratios behind an oblique shock of angle beta."""
    Mn1 = M1 * np.sin(beta)
    p21 = 1 + 2 * g / (g + 1) * (Mn1**2 - 1)
    r21 = (g + 1) * Mn1**2 / ((g - 1) * Mn1**2 + 2)
    delta = np.arctan(2 / np.tan(beta) * (Mn1**2 - 1) / (M1**2 * (g + np.cos(2 * beta)) + 2))
    Mn2 = np.sqrt((1 + 0.5 * (g - 1) * Mn1**2) / (g * Mn1**2 - 0.5 * (g - 1)))
    M2 = Mn2 / np.sin(beta - delta)
    return M2, delta, p21, r21


def v_of_mach(M, g):
    return 1 / np.sqrt(2 / ((g - 1) * M**2) + 1)


def mach_of_v(V, g):
    return np.sqrt(2 / (g - 1) * V**2 / (1 - V**2))


def rhs(theta, y, g):
    Vr, Vt = y
    a2 = 0.5 * (g - 1) * (1 - Vr**2 - Vt**2)  # (speed of sound / V_max)^2
    dVt = (Vt**2 * Vr - a2 * (2 * Vr + Vt / np.tan(theta))) / (a2 - Vt**2)
    return [Vt, dVt]


def integrate(M1, beta, g, dense=False):
    M2, delta, _, _ = oblique_shock(M1, beta, g)
    V2 = v_of_mach(M2, g)
    y0 = [V2 * np.cos(beta - delta), -V2 * np.sin(beta - delta)]
    surface = lambda th, y, g: y[1]  # noqa: E731
    surface.terminal = True
    return solve_ivp(rhs, (beta, 1e-3), y0, args=(g,), events=surface, rtol=1e-11, atol=1e-13,
                     dense_output=dense)


def solve(M1, cone_deg, g=1.4):
    """Shock angle (rad) and the solution between cone and shock."""
    theta_c = np.radians(cone_deg)

    def cone_angle(beta):
        sol = integrate(M1, beta, g)
        return (sol.t_events[0][0] if sol.t_events[0].size else 0.0) - theta_c

    mu = np.arcsin(1 / M1)
    betas = np.linspace(mu + 1e-4, np.radians(89), 400)
    vals = [cone_angle(b) for b in betas]
    i = next(i for i in range(len(betas) - 1) if vals[i] < 0 <= vals[i + 1])  # weak shock
    beta = brentq(cone_angle, betas[i], betas[i + 1], xtol=1e-14)
    sol = integrate(M1, beta, g, dense=True)
    M2, delta, p21, r21 = oblique_shock(M1, beta, g)
    theta = np.linspace(theta_c, beta, 200)
    Vr, Vt = sol.sol(theta)
    V = np.hypot(Vr, Vt)
    M = mach_of_v(V, g)
    # Isentropic between shock and cone: p, rho from the stagnation state behind the shock
    p0_2 = p21 * (1 + 0.5 * (g - 1) * M2**2) ** (g / (g - 1))
    p = p0_2 * (1 + 0.5 * (g - 1) * M**2) ** (-g / (g - 1))
    rho = r21 * (p / p21) ** (1 / g)
    ux = (Vr * np.cos(theta) - Vt * np.sin(theta)) / v_of_mach(M1, g)
    return beta, {"theta": np.degrees(theta), "p": p, "rho": rho, "mach": M, "ux": ux}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mach", type=float, default=2.0)
    ap.add_argument("--cone", type=float, default=15.0, help="cone half-angle (degrees)")
    ap.add_argument("--gamma", type=float, default=1.4)
    ap.add_argument("--csv", default="taylor_maccoll.csv")
    args = ap.parse_args()
    beta, f = solve(args.mach, args.cone, args.gamma)
    cp = (f["p"][0] - 1) / (0.5 * args.gamma * args.mach**2)
    print(f"M = {args.mach}, cone {args.cone} deg: shock angle {np.degrees(beta):.4f} deg, "
          f"surface p/p_inf {f['p'][0]:.5f}, Cp {cp:.5f}, Mach {f['mach'][0]:.4f}")
    np.savetxt(args.csv, np.column_stack([f[k] for k in ("theta", "p", "rho", "mach", "ux")]), delimiter=",",
               header="theta_deg,p_over_pinf,rho_over_rhoinf,mach,ux_over_Uinf", comments="", fmt="%.10g")


if __name__ == "__main__":
    main()
