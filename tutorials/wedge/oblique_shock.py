#!/usr/bin/env python3
"""Attached oblique shock on a ramp, from the theta-beta-Mach relation.

    python oblique_shock.py [input.toml] [--mach M] [--ramp 8] [--gamma 1.4]

With an input file, the Mach number is that of its inflow (the `left`
boundary's u and T) with Mallard's gas constant, R = p_ref / (rho_ref T_ref).
Prints the weak shock angle beta, the pressure, density and temperature
ratios across the shock and the Mach number behind it, or that the shock
detaches (the ramp is steeper than the largest deflection at that Mach number).
"""
import argparse
import math
import tomllib

from scipy.optimize import brentq, minimize_scalar


def deflection(beta, mach, gamma):
    """Flow deflection theta behind an oblique shock of angle beta (radians)."""
    m2 = (mach * math.sin(beta)) ** 2
    return math.atan(2 / math.tan(beta) * (m2 - 1) / (mach**2 * (gamma + math.cos(2 * beta)) + 2))


def solve(mach, ramp_deg, gamma=1.4):
    """The weak shock angle (radians) and ratios across it; None when the shock detaches."""
    theta = math.radians(ramp_deg)
    mu = math.asin(1 / mach)
    best = minimize_scalar(lambda b: -deflection(b, mach, gamma), bounds=(mu, math.pi / 2), method="bounded")
    if deflection(best.x, mach, gamma) < theta:
        return None
    beta = brentq(lambda b: deflection(b, mach, gamma) - theta, mu + 1e-12, best.x)
    mn2 = (mach * math.sin(beta)) ** 2
    p = 1 + 2 * gamma / (gamma + 1) * (mn2 - 1)
    rho = (gamma + 1) * mn2 / ((gamma - 1) * mn2 + 2)
    mn_after = math.sqrt((1 + 0.5 * (gamma - 1) * mn2) / (gamma * mn2 - 0.5 * (gamma - 1)))
    return {"beta": beta, "p": p, "rho": rho, "T": p / rho, "mach2": mn_after / math.sin(beta - theta),
            "theta_max": math.degrees(deflection(best.x, mach, gamma))}


def inflow_mach(path):
    """Mach number and gamma of an input file's `left` inflow."""
    with open(path, "rb") as f:
        inp = tomllib.load(f)
    ph = inp["physics"]
    gamma = ph.get("gamma", 1.4)
    R = ph["p_ref"] / (ph["rho_ref"] * ph["T_ref"])
    left = next(b for b in inp["boundaries"] if b["name"] == "left")
    return left["u"][0] / math.sqrt(gamma * R * left["T"]), gamma


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("input", nargs="?", help="input file whose inflow sets the Mach number")
    ap.add_argument("--mach", type=float)
    ap.add_argument("--ramp", type=float, default=8.0, help="ramp angle in degrees (the wedge mesh: 8)")
    ap.add_argument("--gamma", type=float, default=1.4)
    args = ap.parse_args()
    mach, gamma = (args.mach, args.gamma) if args.mach else inflow_mach(args.input or "input.toml")
    s = solve(mach, args.ramp, gamma)
    if s is None:
        print(f"M = {mach:.4f}, ramp {args.ramp:g} deg: no attached shock (detached)")
        return
    print(f"M = {mach:.4f}, ramp {args.ramp:g} deg: shock angle {math.degrees(s['beta']):.2f} deg, "
          f"p2/p1 {s['p']:.4f}, rho2/rho1 {s['rho']:.4f}, T2/T1 {s['T']:.4f}, M2 {s['mach2']:.4f} "
          f"(largest deflection {s['theta_max']:.2f} deg)")


if __name__ == "__main__":
    main()
