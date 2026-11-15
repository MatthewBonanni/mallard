#!/usr/bin/env python3
"""Inviscid flow over a sphere, axisymmetric, with straight and curved walls.

    axisymmetric_sphere.py RUN_DIR --mallard BUILD/src/mallard [--mach 3 | --mach 0.3]
        [--levels 1 2 4] [--geometry straight curved] [--schemes TENO3] [--t-stop T] [--threads 4]

The meridian half-plane of a sphere of diameter 1 (make_cylinder_mesh.py
--half, 32 L x 16 L quadrilaterals at level L, the sphere and the far field
P2 lines), free stream rho = 1.4, p = 1 (sound speed 1), HLL fluxes.

Subsonic (--mach < 1, far field at r = 8, from the incompressible potential
flow with isentropic pressure): the flow is smooth and isentropic, so the
entropy p / rho^gamma of the reconstruction at the wall's Gauss points stays
that of the free stream; its error is reported over the whole wall (mean,
weighted by the revolved area, and max).

Mach 3 (far field at r = 2.5), against exact inviscid values:

- the stagnation pressure, the pitot pressure behind the normal shock
  (Rayleigh: 12.061 at Mach 3);
- the entropy along the forebody wall (90 to 180 degrees from the rear), which
  is that behind the normal shock: the wall is the stagnation streamline;

and the shock standoff against Billig's correlation (0.205 R at Mach 3, an
experimental fit). Results go to RUN_DIR/results.json and a table.
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys

GAMMA, MACH = 1.4, 3.0
RHO_INF, P_INF = 1.4, 1.0

INPUT = """[run]
t_stop = {t_stop}
cfl = 0.4

[mesh]
type = "file"
filename = "sphere.msh"
curved_geometry = {curved}

[initialize]
type = "analytical"
{init}
[[boundaries]]
name = "cylinder"
type = "symmetry"

[[boundaries]]
name = "farfield"
type = "farfield"
u = [{mach}, 0.0]
p = 1.0
T = 0.7142857142857143

[[boundaries]]
name = "axis"
type = "symmetry"

[numerics]
riemann_solver = "HLL"
time_integrator = "SSPRK3"
check_nan = true

[numerics.face_reconstruction]
{scheme}

[physics]
type = "euler"
axisymmetric = true
gamma = 1.4
p_ref = 1.0
T_ref = 0.7142857142857143
rho_ref = 1.4

[output]
check_interval = 1000

[[probes]]
name = "wall"
boundary = "cylinder"
fields = ["P", "RHO"]
sampling = "{sampling}"
time_interval = {t_half}
{axis}"""

AXIS_PROBE = """
[[probes]]
name = "axis"
fields = ["RHO"]
start = [-1.0, 0.001]
end = [-0.5, 0.001]
n_points = 2001
time_interval = {t_half}
"""

SUPERSONIC_INIT = """rho = "1.4"
u = ["3 * (1 - exp(-4 * (x^2 + y^2 - 0.25)))", "0.0"]
p = "1.0"
"""

# Potential flow around the sphere, c^2 from Bernoulli, isentropic
SUBSONIC_INIT = """rho = "1.4 * (1 + 0.2 * ({m}^2 - ({ux})^2 - ({uy})^2))^2.5"
u = ["{ux}", "{uy}"]
p = "(1 + 0.2 * ({m}^2 - ({ux})^2 - ({uy})^2))^3.5"
"""


def pitot_pressure():
    m2 = MACH * MACH
    return P_INF * ((GAMMA + 1) / 2 * m2) ** (GAMMA / (GAMMA - 1)) / (
        (2 * GAMMA / (GAMMA + 1) * m2 - (GAMMA - 1) / (GAMMA + 1)) ** (1 / (GAMMA - 1)))


def shock_entropy():
    """p / rho^gamma behind the normal shock."""
    m2 = MACH * MACH
    p2 = P_INF * (1 + 2 * GAMMA / (GAMMA + 1) * (m2 - 1))
    rho2 = RHO_INF * (GAMMA + 1) * m2 / ((GAMMA - 1) * m2 + 2)
    return p2 / rho2 ** GAMMA


def last_rows(path):
    with open(path) as f:
        rows = [r for r in csv.DictReader(line for line in f if not line.startswith("#"))]
    step = rows[-1]["step"]
    return [r for r in rows if r["step"] == step]


def column(row, name):
    for key, value in row.items():
        if key == name or key.startswith(name + " ") or key.startswith(name + "["):
            return float(value)
    raise KeyError(f"{name} not in {list(row)}")


def subsonic_init(m):
    k = "0.125 / (x^2 + y^2)^2.5"
    ux, uy = f"{m} * (1 - {k} * (x^2 - 0.5 * y^2))", f"-{m} * 1.5 * {k} * x * y"
    return SUBSONIC_INIT.format(m=m, ux=ux, uy=uy)


def run_case(case_dir, args, scheme, level, geometry):
    os.makedirs(case_dir, exist_ok=True)
    subsonic = args.mach < 1.0
    tools = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mesh")
    subprocess.run([sys.executable, os.path.join(tools, "make_cylinder_mesh.py"), os.path.join(case_dir, "sphere.msh"),
                    "--half", "--n-theta", str(32 * level), "--n-r", str(16 * level),
                    "--r-far", "8" if subsonic else "2.5", "--dr0", str((0.05 if subsonic else 0.04) / level),
                    "--order", "2"], check=True, stdout=subprocess.DEVNULL)
    recon = 'type = "MUSCL"' if scheme == "MUSCL" else f'type = "TENO"\norder = {scheme[4:]}'
    with open(os.path.join(case_dir, "input.toml"), "w") as f:
        f.write(INPUT.format(t_stop=args.t_stop, t_half=args.t_stop / 2, mach=args.mach,
                             curved="false" if geometry == "straight" else "true", scheme=recon,
                             init=subsonic_init(args.mach) if subsonic else SUPERSONIC_INIT,
                             sampling="point" if subsonic else "cell",
                             axis="" if subsonic else AXIS_PROBE.format(t_half=args.t_stop / 2)))
    with open(os.path.join(case_dir, "log.txt"), "w") as log:
        rc = subprocess.run([args.mallard, "-i", "input.toml", f"--kokkos-num-threads={args.threads}"], cwd=case_dir,
                            stdout=log, stderr=subprocess.STDOUT).returncode
    out = {"scheme": scheme, "level": level, "geometry": geometry}
    if rc != 0:
        out["failed"] = True
        return out
    wall = last_rows(os.path.join(case_dir, "probe_wall.csv"))
    if subsonic:
        entropy = [abs(column(r, "P") / column(r, "RHO") ** GAMMA * RHO_INF ** GAMMA / P_INF - 1) for r in wall]
        weights = [column(r, "area") for r in wall]
        out.update({"failed": False, "entropy_error_max": max(entropy),
                    "entropy_error_mean": sum(e * w for e, w in zip(entropy, weights)) / sum(weights)})
        return out
    s_exact, p0 = shock_entropy(), pitot_pressure()
    forebody = [r for r in wall if column(r, "x") < 0.0]
    stagnation = min(forebody, key=lambda r: abs(column(r, "y")))
    entropy = [abs(column(r, "P") / column(r, "RHO") ** GAMMA / s_exact - 1) for r in forebody]
    axis = last_rows(os.path.join(case_dir, "probe_axis.csv"))
    rho_mid = 0.5 * (RHO_INF + RHO_INF * (GAMMA + 1) * MACH ** 2 / ((GAMMA - 1) * MACH ** 2 + 2))
    shock = next(column(r, "x") for r in axis if column(r, "RHO") > rho_mid)
    out.update({"failed": False, "p_stagnation_error": column(stagnation, "P") / p0 - 1,
                "entropy_error_max": max(entropy), "entropy_error_mean": sum(entropy) / len(entropy),
                "standoff_over_R": (-0.5 - shock) / 0.5})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--mallard", required=True)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--geometry", nargs="+", default=["straight", "curved"])
    ap.add_argument("--schemes", nargs="+", default=["TENO3"])
    ap.add_argument("--mach", type=float, default=3.0)
    ap.add_argument("--t-stop", type=float, default=None)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    global MACH
    MACH = args.mach
    args.mallard = os.path.abspath(args.mallard)
    if args.t_stop is None:
        args.t_stop = 6.0 if args.mach >= 1.0 else 40.0
    os.makedirs(args.run_dir, exist_ok=True)
    results_file = os.path.join(args.run_dir, "results.json")
    results = json.load(open(results_file)) if os.path.exists(results_file) else []
    done = {(r["scheme"], r["level"], r["geometry"]) for r in results}
    for scheme in args.schemes:
        for geometry in args.geometry:
            for level in args.levels:
                if (scheme, level, geometry) in done:
                    continue
                r = run_case(os.path.join(args.run_dir, f"{scheme}_{geometry}_l{level}"), args, scheme, level, geometry)
                results.append(r)
                json.dump(results, open(results_file, "w"), indent=1)
                print(json.dumps(r), flush=True)
    print(f"{'scheme':>6} {'walls':>8} {'cells':>7} {'p0 error':>10} {'entropy max':>12} {'mean':>10} {'standoff/R':>10}")
    for r in sorted(results, key=lambda r: (r["scheme"], r["geometry"], r["level"])):
        if r.get("failed"):
            print(f"{r['scheme']:>6} {r['geometry']:>8} {512 * r['level'] ** 2:>7} failed")
            continue
        p0, standoff = r.get("p_stagnation_error"), r.get("standoff_over_R")
        print(f"{r['scheme']:>6} {r['geometry']:>8} {512 * r['level'] ** 2:>7} "
              f"{'-' if p0 is None else f'{p0:.2e}':>10} {r['entropy_error_max']:>12.2e} "
              f"{r['entropy_error_mean']:>10.2e} {'-' if standoff is None else f'{standoff:.4f}':>10}")
    if MACH >= 1.0:
        print(f"exact: p0 = {pitot_pressure():.4f}, Billig standoff/R = {0.143 * math.exp(3.24 / MACH ** 2):.4f}")

if __name__ == "__main__":
    main()
