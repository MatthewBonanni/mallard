#!/usr/bin/env python3
"""The particle-centrifuge example (examples/particle_centrifuge): the particles
at the end of the run against their exact spirals in solid-body rotation.

    check_particle_centrifuge.py particles.csv [--input input.toml]

In gas rotating at Omega, a particle with Stokes drag and no gravity moves by
tau z'' + z' = i Omega z (z = x + i y), so z(t) = a e^(l1 t) + b e^(l2 t), with
l1, l2 the roots of tau l^2 + l - i Omega = 0 and a + b = z0, a l1 + b l2 =
i Omega z0 (released at the gas velocity). spiral() returns z(t) and the first
time |z| reaches the outer radius, where the wall catches it. Each particle the
spiral keeps inside must be in the file within 3e-3 of its exact position, and
each the wall caught must be missing from it. Omega and the radii are read from
the input: Omega from u, the outer radius from [mesh]. Exits 1 on any failure.
"""
import argparse
import cmath
import csv
import math
import re
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_particles(path):
    with open(path) as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
    return [dict(zip(rows[0], map(float, r))) for r in rows[1:]]


def roots(omega, tau):
    s = cmath.sqrt(1.0 + 4j * omega * tau)
    return (-1.0 + s) / (2.0 * tau), (-1.0 - s) / (2.0 * tau)


def spiral(z0, omega, tau, t):
    """z(t) of a particle released at z0 with the gas velocity i Omega z0."""
    l1, l2 = roots(omega, tau)
    a = z0 * (1j * omega - l2) / (l1 - l2)
    b = z0 - a
    return a * cmath.exp(l1 * t) + b * cmath.exp(l2 * t)


def arrival(z0, omega, tau, r_outer, t_end, dt=1e-3):
    """The first time |z| reaches r_outer before t_end, or None."""
    t = 0.0
    while t < t_end:
        t += dt
        if abs(spiral(z0, omega, tau, t)) >= r_outer:
            return t
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv", help="the particles file of the run ([particles] file)")
    ap.add_argument("--input", default=str(HERE / "input.toml"), help="the run's input file")
    args = ap.parse_args()
    cfg = tomllib.loads(Path(args.input).read_text())
    p, ph = cfg["particles"], cfg["physics"]
    omega = float(re.match(r"-\s*([0-9.eE+-]+)\s*\*\s*y", cfg["initialize"]["u"][0]).group(1))
    tau = p["density"] * p["diameter"] ** 2 / (18.0 * ph["mu"])
    r_outer, t = cfg["mesh"]["r_outer"], cfg["run"]["t_stop"]
    l1, _ = roots(omega, tau)
    print(f"Omega tau = {omega * tau:g}: the spirals grow at {l1.real:.4f} and turn at {l1.imag:.4f} "
          f"({l1.imag / omega:.3f} Omega)")
    rows = {int(r["id"]): r for r in read_particles(args.csv)}
    failed = 0
    for k, (x0, y0) in enumerate(p["points"]):
        z0 = complex(x0, y0)
        hit = arrival(z0, omega, tau, r_outer, t)
        if hit is not None:
            ok = k not in rows
            print(f"  particle {k} (r0 = {abs(z0):g}): caught by the outer wall at t = {hit:.2f}"
                  f"{'' if ok else '  FAIL: still in the file'}")
        elif k not in rows:
            ok = False
            print(f"  particle {k} (r0 = {abs(z0):g}): FAIL: missing, but its spiral stays inside")
        else:
            z = spiral(z0, omega, tau, t)
            err = abs(complex(rows[k]["x"], rows[k]["y"]) - z)
            ok = err < 3e-3
            print(f"  particle {k} (r0 = {abs(z0):g}): r = {abs(z):.4f}, {err:.1e} from its spiral"
                  f"{'' if ok else '  FAIL'}")
        failed += not ok
    print(f"{len(p['points']) - failed} of {len(p['points'])} particles as their spirals predict")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
