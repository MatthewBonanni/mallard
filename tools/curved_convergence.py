#!/usr/bin/env python3
"""Convergence of the steady isentropic vortex between two circular walls.

    curved_convergence.py RUN_DIR --mallard BUILD/src/Mallard [--orders 3 4 5 6]
        [--levels 4 8 16] [--geometry straight curved] [--triangles] [--t-stop 12]
        [--mach 2.25] [--threads 4] [--launcher 'mpirun -n 1'] [--extra 'TOML lines']

The annulus 1 <= r <= 1.384 holds the irrotational vortex u_theta = M / r
(M = 2.25 at the inner wall by default, rho = 1, p = 1 / gamma there), an
exact steady solution of the Euler equations with slip walls (Aftosmis et al.
1994, here around the full circle). Each run starts from the exact cell
averages and marches to t_stop; the error is that of the final density
averages against the initial ones. Geometries: "straight" (polygonal walls),
"curved" (walls projected onto the circles), "curved_mirrors" (the same, with
TENO mirror images across the curved walls), "p2" / "p3" (walls from
high-order Gmsh lines). Results go to RUN_DIR/results.json and a table.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mallard_vtu import read_vtu_cells  # noqa: E402

GAMMA = 1.4
R_IN, R_OUT = 1.0, 1.384

INPUT = """[run]
t_stop = {t_stop}
cfl = {cfl}

[mesh]
type = "file"
filename = "annulus.msh"
curved_geometry = {curved}
{shapes}
[initialize]
type = "analytical"
rho = "(1 + 0.2 * {M2} * (1 - 1 / (x^2 + y^2)))^2.5"
u = ["-{M} * y / (x^2 + y^2)", "{M} * x / (x^2 + y^2)"]
p = "(1 + 0.2 * {M2} * (1 - 1 / (x^2 + y^2)))^3.5 / 1.4"

[[boundaries]]
name = "inner"
type = "symmetry"

[[boundaries]]
name = "outer"
type = "symmetry"

[numerics]
riemann_solver = "HLLC"
time_integrator = "SSPRK3"
check_nan = true

[numerics.face_reconstruction]
type = "{scheme}"
{order}
{extra}

[physics]
type = "euler"
gamma = 1.4
p_ref = 1.0
T_ref = 1.0
rho_ref = 1.4

[output]
check_interval = 1000

[[write_data]]
prefix = "./solut/annulus"
format = "vtu"
time_interval = {t_out}
variables = ["RHO", "P"]
"""

SHAPES = """
[[mesh.curved]]
zone = "inner"
shape = "circle"
center = [0.0, 0.0]
radius = {r_in}

[[mesh.curved]]
zone = "outer"
shape = "circle"
center = [0.0, 0.0]
radius = {r_out}
"""


def polygon_areas(points, conn, offsets):
    areas = np.empty(len(offsets))
    start = 0
    for c, end in enumerate(offsets):
        p = points[conn[start:end], :2]
        x, y = p[:, 0], p[:, 1]
        areas[c] = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        start = end
    return areas


def run_case(case_dir, args, order, level, geometry):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(case_dir, exist_ok=True)
    n_theta = 16 * level
    mesh_order = {"p2": 2, "p3": 3}.get(geometry, 1)
    cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_annulus_mesh.py"),
           os.path.join(case_dir, "annulus.msh"), "--n-r", str(level), "--n-theta", str(n_theta),
           "--r-in", str(R_IN), "--r-out", str(R_OUT), "--order", str(mesh_order)]
    if args.triangles:
        cmd.append("--triangles")
    if args.jitter:
        cmd += ["--jitter", str(args.jitter)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    scheme = "MUSCL" if order == 2 else "TENO"
    text = INPUT.format(
        t_stop=args.t_stop, cfl=args.cfl, curved="false" if geometry == "straight" else "true",
        shapes=SHAPES.format(r_in=R_IN, r_out=R_OUT) if geometry.startswith("curved") else "",
        M=args.mach, M2=args.mach * args.mach, scheme=scheme, order="" if order == 2 else f"order = {order}",
        extra=args.extra.replace("\\n", "\n") + ("\ncurved_mirrors = true" if geometry == "curved_mirrors" else ""),
        t_out=args.t_stop / 4)
    with open(os.path.join(case_dir, "input.toml"), "w") as f:
        f.write(text)
    run = [args.mallard, "-i", "input.toml", f"--kokkos-num-threads={args.threads}"]
    if args.launcher:
        run = args.launcher.split() + run
    with open(os.path.join(case_dir, "log.txt"), "w") as log:
        rc = subprocess.run(run, cwd=case_dir, stdout=log, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        return {"order": order, "level": level, "geometry": geometry, "failed": True}
    files = sorted(f for f in os.listdir(os.path.join(case_dir, "solut")) if f.endswith(".vtu") and "_" in f)
    first = read_vtu_cells(os.path.join(case_dir, "solut", files[0]))
    last = read_vtu_cells(os.path.join(case_dir, "solut", files[-1]))
    prev = read_vtu_cells(os.path.join(case_dir, "solut", files[-2]))
    points, conn, offsets = first[0], first[1], first[2]
    areas = polygon_areas(points, conn, offsets)
    err = np.abs(last[4]["RHO"] - first[4]["RHO"])
    change = np.abs(last[4]["RHO"] - prev[4]["RHO"]).max()
    return {"order": order, "level": level, "geometry": geometry, "n_cells": int(len(areas)),
            "L1": float(np.sum(err * areas) / np.sum(areas)), "Linf": float(err.max()),
            "last_change": float(change), "failed": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--mallard", required=True)
    ap.add_argument("--orders", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--levels", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--geometry", nargs="+", default=["straight", "curved"])
    ap.add_argument("--triangles", action="store_true")
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--t-stop", type=float, default=12.0)
    ap.add_argument("--cfl", type=float, default=0.4)
    ap.add_argument("--mach", type=float, default=2.25)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--launcher", default="", help="Command prefix, e.g. 'mpirun -n 1'")
    ap.add_argument("--extra", default="")
    args = ap.parse_args()
    args.mallard = os.path.abspath(args.mallard)
    os.makedirs(args.run_dir, exist_ok=True)
    results_file = os.path.join(args.run_dir, "results.json")
    results = json.load(open(results_file)) if os.path.exists(results_file) else []
    done = {(r["order"], r["level"], r["geometry"]) for r in results}
    for geometry in args.geometry:
        for order in args.orders:
            for level in args.levels:
                if (order, level, geometry) in done:
                    continue
                case = os.path.join(args.run_dir, f"{geometry}_o{order}_n{level}")
                r = run_case(case, args, order, level, geometry)
                results.append(r)
                json.dump(results, open(results_file, "w"), indent=1)
                print(json.dumps(r), flush=True)
    print(f"{'geometry':>9} {'order':>5} {'n_r':>4} {'cells':>7} {'L1':>10} {'rate':>5} {'Linf':>10} {'rate':>5}")
    for geometry in args.geometry:
        for order in args.orders:
            rows = sorted((r for r in results if r["geometry"] == geometry and r["order"] == order
                           and r["level"] in args.levels), key=lambda r: r["level"])
            prev = None
            for r in rows:
                if r.get("failed"):
                    print(f"{geometry:>9} {order:>5} {r['level']:>4} failed")
                    prev = None
                    continue
                rate1 = rate2 = ""
                if prev is not None:
                    f = np.log(r["level"] / prev["level"])
                    rate1 = f"{np.log(prev['L1'] / r['L1']) / f:5.2f}"
                    rate2 = f"{np.log(prev['Linf'] / r['Linf']) / f:5.2f}"
                print(f"{geometry:>9} {order:>5} {r['level']:>4} {r['n_cells']:>7} {r['L1']:10.3e} {rate1:>5} "
                      f"{r['Linf']:10.3e} {rate2:>5}")
                prev = r


if __name__ == "__main__":
    main()
