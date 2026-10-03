"""Exact Riemann solution for thermally perfect gas mixtures of frozen composition.

Each side keeps its own composition (the contact separates them); the
thermodynamics come from Cantera with the mechanism the run uses. Shocks
satisfy the Rankine-Hugoniot conditions h2 - h1 = (p2 - p1)(v1 + v2)/2,
rarefactions follow the isentrope with the frozen sound speed
a^2 = (cp/cv) p / rho.

    python tools/mixture_riemann.py

writes the reference solutions of Mallard's multicomponent shock tube test
(test/data/chemistry/shock_tube_<N>.csv: the exact state at the centers of N
uniform cells on [0, 1]).
"""
import os

import cantera as ct
import numpy as np
from scipy.optimize import brentq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Side:
    """One initial state: a Cantera gas at fixed composition."""

    def __init__(self, mech, phase, p, T, u, X):
        self.gas = ct.Solution(mech, phase)
        self.gas.TPX = T, p, X
        self.X = self.gas.X.copy()
        self.p, self.u, self.T = p, u, T
        self.rho = self.gas.density
        self.h = self.gas.enthalpy_mass
        self.s = self.gas.entropy_mass
        self.a = self.sound_speed()
        # Isentrope table for rarefactions: p decreasing from the initial pressure
        self.fan_p = np.geomspace(p, p * 0.02, 2001)
        rho_a = []
        for q in self.fan_p:
            self.gas.SPX = self.s, q, self.X
            rho_a.append(self.gas.density * self.sound_speed())
        rho_a = np.array(rho_a)
        # integral of dp / (rho a) from p down to each fan pressure (trapezoid in log p, fine grid)
        dp = -np.diff(self.fan_p)
        mid = 0.5 * (1.0 / rho_a[1:] + 1.0 / rho_a[:-1])
        self.fan_integral = np.concatenate([[0.0], np.cumsum(mid * dp)])

    def sound_speed(self):
        g = self.gas
        return np.sqrt(g.cp_mass / g.cv_mass * g.P / g.density)

    def isentrope(self, q):
        self.gas.SPX = self.s, q, self.X
        return self.gas.density, self.gas.T, self.sound_speed()

    def velocity_jump(self, q):
        """Toro's f_K: u* = u_L - f_L(q) = u_R + f_R(q); positive across a shock, negative across a fan."""
        if q <= self.p:
            return -np.interp(-q, -self.fan_p, self.fan_integral)
        v1 = 1.0 / self.rho
        T2 = self.shock_temperature(q)
        v2 = 1.0 / self.gas.density
        m = np.sqrt((q - self.p) / (v1 - v2))
        return (q - self.p) / m

    def shock_temperature(self, q):
        v1 = 1.0 / self.rho

        def residual(T2):
            self.gas.TPX = T2, q, self.X
            return self.gas.enthalpy_mass - self.h - 0.5 * (q - self.p) * (v1 + 1.0 / self.gas.density)

        T2 = brentq(residual, self.T, self.T * (1.0 + 2.0 * q / self.p), xtol=1e-12, rtol=1e-15)
        self.gas.TPX = T2, q, self.X
        return T2

    def shock_mass_flux(self, q):
        self.shock_temperature(q)
        return np.sqrt((q - self.p) / (1.0 / self.rho - 1.0 / self.gas.density))


def solve(L, R):
    def f(q):
        return (L.u - L.velocity_jump(q)) - (R.u + R.velocity_jump(q))

    lo, hi = min(L.p, R.p), max(L.p, R.p)
    while f(lo) < 0.0:
        lo *= 0.5
    while f(hi) > 0.0:
        hi *= 2.0
    p_star = brentq(f, lo, hi, xtol=1e-10, rtol=1e-14)
    u_star = L.u - L.velocity_jump(p_star)
    return p_star, u_star


def sample(side, sign, p_star, u_star, xi):
    """State at xi = x / t on this side of the contact (sign -1 left, +1 right)."""
    g = side.gas
    if p_star > side.p:
        m = side.shock_mass_flux(p_star)
        S = side.u + sign * m / side.rho
        if sign * (xi - S) > 0:
            return side.rho, side.u, side.p, side.T
        T2 = side.shock_temperature(p_star)
        return g.density, u_star, p_star, T2
    head = side.u + sign * side.a
    rho_s, T_s, a_s = side.isentrope(p_star)
    tail = u_star + sign * a_s
    if sign * (xi - head) > 0:
        return side.rho, side.u, side.p, side.T
    if sign * (xi - tail) < 0:
        return rho_s, u_star, p_star, T_s

    def fan(q):
        _, _, a = side.isentrope(q)
        u = side.u - sign * np.interp(-q, -side.fan_p, side.fan_integral)
        return u + sign * a - xi

    q = brentq(fan, p_star, side.p, xtol=1e-10, rtol=1e-14)
    rho, T, _ = side.isentrope(q)
    u = side.u - sign * np.interp(-q, -side.fan_p, side.fan_integral)
    return rho, u, q, T


def shock_tube(mech, phase, left, right, x0, t, x):
    L = Side(mech, phase, *left)
    R = Side(mech, phase, *right)
    p_star, u_star = solve(L, R)
    rows = []
    for xc in x:
        xi = (xc - x0) / t
        side, sign = (L, -1) if xi < u_star else (R, 1)
        rho, u, p, T = sample(side, sign, p_star, u_star, xi)
        side.gas.TPX = T, p, side.X
        rows.append([xc, rho, u, p, T] + list(side.gas.Y))
    return np.array(rows), L.gas.species_names, p_star, u_star


SHOCK_TUBE = dict(
    mech=os.path.join(ROOT, "mechanisms", "h2o2.yaml"),
    phase="ohmech",
    # p, T, u, X
    left=(1.0e5, 1000.0, 0.0, {"H2": 2.0, "O2": 1.0, "AR": 7.0}),
    right=(1.0e4, 300.0, 0.0, {"N2": 1.0}),
    x0=0.5,
    t=2.0e-4,
)


def main():
    c = SHOCK_TUBE
    for n in (50, 100, 200, 400):
        x = (np.arange(n) + 0.5) / n
        rows, names, p_star, u_star = shock_tube(c["mech"], c["phase"], c["left"], c["right"], c["x0"], c["t"], x)
        path = os.path.join(ROOT, "test", "data", "chemistry", f"shock_tube_{n}.csv")
        with open(path, "w") as f:
            f.write(f"# Cantera {ct.__version__}, exact frozen-mixture Riemann solution, mechanisms/h2o2.yaml, "
                    f"t = {c['t']}, p* = {p_star:.10g}, u* = {u_star:.10g}\n")
            f.write(",".join(["x", "rho", "u", "p", "T"] + [f"Y_{s}" for s in names]) + "\n")
            for r in rows:
                f.write(",".join("%.12g" % v for v in r) + "\n")
        print("wrote", path, "p* =", p_star, "u* =", u_star)


if __name__ == "__main__":
    main()
