#!/usr/bin/env python3
"""The particle-settling example (examples/particle_settling): the velocities of
the particles at the end of the run against their terminal velocity.

    check_particle_settling.py droplets [--input input.toml]
    check_particle_settling.py smoke --input smoke.toml

Reads the particles' diameter, density, drag law and gravity, the gas viscosity
and state and t_stop from the input (by default the input.toml next to this
script for droplets, smoke.toml for smoke), and computes the terminal
velocity v_t = tau g (1 - rho / rho_p) / f(Re(v_t)) with the drag law's f
(Stokes: f = 1; Schiller-Naumann: 1 + 0.15 Re^0.687) by fixed-point
iteration, and the Stokes value for comparison. Particles whose relaxation time
is below the run's step relax to it exponentially within each step; all
approach it as e^(-t / tau). A particle more than 1e-3 v_t off (plus the rest
of its transient) fails. Exits 1 on any failure.
"""
import argparse
import math
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_particles(prefix):
    """id, x, y, v_x, v_y of the particles of the run's last snapshot, PREFIX_particles_<n>.vtkhdf."""
    import h5py

    path = Path(prefix)
    files = sorted(path.parent.glob(path.name + "_particles_*.vtkhdf"), key=lambda f: int(f.stem.rsplit("_", 1)[1]))
    if not files:
        sys.exit(f"no {prefix}_particles_<n>.vtkhdf: run the example first")
    with h5py.File(files[-1], "r") as f:
        g = f["VTKHDF"]
        points, ids, v = g["Points"][:], g["PointData/ID"][:], g["PointData/V"][:]
    return [{"id": int(i), "x": float(p[0]), "y": float(p[1]), "v_x": float(u[0]), "v_y": float(u[1])}
            for i, p, u in zip(ids, points, v)]


def terminal_velocity(d, rho_p, rho, mu, g, drag):
    """|v_t| and the relaxation time at it, for a drag law of [particles] drag."""
    tau_st = rho_p * d * d / (18.0 * mu)
    g_eff = g * (1.0 - rho / rho_p)
    v = tau_st * g_eff
    f = 1.0
    for _ in range(500):
        if drag == "schiller_naumann":
            re = rho * v * d / mu
            f = 1.0 + 0.15 * re ** 0.687 if re < 1000 else 0.44 * re / 24.0
        v = tau_st * g_eff / f
    return v, tau_st / f, tau_st * g_eff


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("prefix", help="the prefix of the run's particle snapshots ([[write_data]] prefix)")
    ap.add_argument("--input", help="the run's input file (default: input.toml, or smoke.toml for smoke)")
    args = ap.parse_args()
    inp = Path(args.input) if args.input else HERE / ("smoke.toml" if "smoke" in Path(args.prefix).name else "input.toml")
    cfg = tomllib.loads(inp.read_text())
    p, ph, init = cfg["particles"], cfg["physics"], cfg["initialize"]
    rho = init["p"] / (ph["p_ref"] / (ph["rho_ref"] * ph["T_ref"]) * init["T"])
    g = math.hypot(*p["gravity"])
    drag = p.get("drag", "schiller_naumann")
    v_t, tau, v_stokes = terminal_velocity(p["diameter"], p["density"], rho, ph["mu"], g, drag)
    t = cfg["run"]["t_stop"]
    re = rho * v_t * p["diameter"] / ph["mu"]
    print(f"{inp.name}: d = {p['diameter'] * 1e6:g} um, tau = {tau * 1e3:.4g} ms, Re_p = {re:.3g}")
    print(f"terminal velocity: {v_t:.6g} m/s ({drag}), Stokes' law {v_stokes:.6g} m/s")
    rows = read_particles(args.prefix)
    transient = math.exp(-t / tau)
    tol = 1e-3 + transient
    failed = 0
    for r in rows:
        v = -r["v_y"]
        err = abs(v - v_t) / v_t
        ok = err <= tol and abs(r["v_x"]) <= 1e-12 * v_t
        failed += not ok
        print(f"  particle {int(r['id'])}: v = {v:.6g} m/s, {err:.1e} of v_t{'' if ok else '  FAIL'}")
    print(f"{len(rows) - failed} of {len(rows)} particles within {tol:.1e} of v_t (rest of the transient {transient:.1e})")
    return 1 if failed or not rows else 0


if __name__ == "__main__":
    sys.exit(main())
