#!/usr/bin/env python3
"""The particle-bounce example (examples/particle_bounce): the particles at the
end of the run against their exact trajectories.

    check_particle_bounce.py particles.csv [--input input.toml]

With Stokes drag in gas at rest, the motion between bounces is exact: along
each axis, dv/dt = -v / tau + g', so v(s) = v_t + (v0 - v_t) e^(-s/tau) and
x(s) = x0 + v_t s + (v0 - v_t) tau (1 - e^(-s/tau)), with v_t = tau g' along
gravity and 0 across it. trajectory() steps from bounce to bounce, finding the
next time a coordinate reaches a wall by bisection, and applies the wall's
restitution coefficients there ([[particles.walls]]; 1 elsewhere). Mallard
locates each bounce within its step, exactly for drag in gas at rest under
gravity: a particle farther than 1e-6 from its exact position fails. Exits 1
on any failure.
"""
import argparse
import csv
import math
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAMES = {(1, 0): "bottom", (1, 1): "top", (0, 0): "left", (0, 1): "right"}


def read_particles(path):
    with open(path) as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
    return [dict(zip(rows[0], map(float, r))) for r in rows[1:]]


def walls_of(cfg):
    """{zone: (e_n, e_t, stick)} from [[particles.walls]]."""
    out = {}
    for w in cfg["particles"].get("walls", []):
        for z in w["zones"]:
            out[z] = (w.get("restitution", 1.0), w.get("tangential_restitution", 1.0),
                      w.get("interaction", "rebound") == "stick")
    return out


def trajectory(x0, v0, t_end, tau, g_eff, box, walls, samples=None):
    """Position, velocity and number of bounces at t_end (positions at samples["times"] appended to
    samples["points"]), or None if the particle stuck to a wall."""
    x, v, t, bounces = list(x0), list(v0), 0.0, 0
    vt = [tau * g_eff[0], tau * g_eff[1]]

    def state(s):
        e = math.exp(-s / tau)
        return ([x[d] + vt[d] * s + (v[d] - vt[d]) * tau * (1.0 - e) for d in range(2)],
                [vt[d] + (v[d] - vt[d]) * e for d in range(2)])

    def outside(s):
        p, _ = state(s)
        return any(p[d] < box[d][0] or p[d] > box[d][1] for d in range(2))

    while True:
        # The first time within the rest of the run, at a resolution of 1e-4, then bisection
        horizon = t_end - t
        n = max(1, int(horizon / 1e-4))
        hit = next((k * horizon / n for k in range(1, n + 1) if outside(k * horizon / n)), None)
        if samples is not None:
            for ts in [s for s in samples["times"] if t <= s < t + (hit if hit is not None else horizon + 1e-12)]:
                samples["points"].append(state(ts - t)[0])
        if hit is None:
            return (*state(horizon), bounces)
        a, b = hit - horizon / n, hit
        for _ in range(60):
            m = 0.5 * (a + b)
            a, b = (a, m) if outside(m) else (m, b)
        p, u = state(b)
        for d in range(2):
            for side in range(2):
                if (p[d] < box[d][0]) if side == 0 else (p[d] > box[d][1]):
                    e_n, e_t, stick = walls.get(NAMES[(d, side)], (1.0, 1.0, False))
                    if stick:
                        return None
                    u[d] = -e_n * u[d]
                    u[1 - d] *= e_t
                    p[d] = box[d][side]
        x, v, t, bounces = p, u, t + b, bounces + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv", help="the particles file of the run ([particles] file)")
    ap.add_argument("--input", default=str(HERE / "input.toml"), help="the run's input file")
    args = ap.parse_args()
    cfg = tomllib.loads(Path(args.input).read_text())
    p, ph, mesh = cfg["particles"], cfg["physics"], cfg["mesh"]
    rho = 1.0
    tau = p["density"] * p["diameter"] ** 2 / (18.0 * ph["mu"])
    g_eff = [g * (1.0 - rho / p["density"]) for g in p["gravity"]]
    box = [(0.0, mesh["Lx"]), (0.0, mesh["Ly"])]
    rows = {int(r["id"]): r for r in read_particles(args.csv)}
    failed = 0
    for k, seed in enumerate(p["points"]):
        exact = trajectory(seed, p["velocity"], cfg["run"]["t_stop"], tau, g_eff, box, walls_of(cfg))
        if exact is None or k not in rows:
            ok = exact is None and k not in rows
            print(f"  particle {k}: {'stuck, as exact' if ok else 'FAIL: stuck in one and not the other'}")
            failed += not ok
            continue
        r = rows[k]
        err = math.hypot(r["x"] - exact[0][0], r["y"] - exact[0][1])
        ok = err < 1e-6
        failed += not ok
        print(f"  particle {k}: ({r['x']:.5f}, {r['y']:.5f}), exact ({exact[0][0]:.5f}, {exact[0][1]:.5f}) after "
              f"{exact[2]} bounces: {err:.1e} off{'' if ok else '  FAIL'}")
    print(f"{len(p['points']) - failed} of {len(p['points'])} particles within 1e-6 of the exact trajectories")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
