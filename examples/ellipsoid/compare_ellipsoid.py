#!/usr/bin/env python3
"""Flat against curved wall faces on the spheroid of examples/ellipsoid.

    compare_ellipsoid.py RUN_DIR --mallard BUILD3D/src/mallard [--scales 1 1.4 2]
        [--geometry straight curved] [--t-stop 40] [--order 5] [--threads 2]
        [--launcher 'mpirun -n 2'] [-o ellipsoid.png]

For each mesh scale (make_ellipsoid_mesh.py --scale) and geometry, runs
examples/ellipsoid/input.toml as it is (curved) or with [mesh] curved_geometry
= false (flat faces), in RUN_DIR/<geometry>_s<scale> unless that run finished,
and reports the drag coefficient (exact: 0), averaged over the last quarter of
the run and based on the frontal area pi b^2; the entropy error s / s_inf - 1
(s = p / rho^gamma; exact: 0) at the end, its volume-weighted RMS and maximum
within the spheroid scaled by 3; and the RMS difference of the wall pressure
coefficient from incompressible potential flow (Lamb, Hydrodynamics, ch. V).
Prints a table, and with -o draws the wall pressure and the errors against
the mesh size.
"""
import argparse
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
from mallard.vtu import read_vtu_cells  # noqa: E402

GAMMA = 1.4
A, B = 1.0, 0.5
U = 0.3
P_INF = 1.0 / GAMMA


def potential_k(a, b):
    """Lamb's k for a prolate spheroid moving along its axis: the surface
    velocity is (1 + k) times the tangential part of the free stream."""
    e = np.sqrt(1.0 - (b / a) ** 2)
    alpha = 2.0 * (1.0 - e * e) / e ** 3 * (0.5 * np.log((1.0 + e) / (1.0 - e)) - e)
    return alpha / (2.0 - alpha)


def cp_potential(x, y, z):
    n = np.stack([x / A ** 2, y / B ** 2, z / B ** 2], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return 1.0 - (1.0 + potential_k(A, B)) ** 2 * (1.0 - n[:, 0] ** 2)


def cell_centroids(points, conn, offsets):
    starts = np.concatenate([[0], offsets[:-1]])
    return np.array([points[conn[s:e]].mean(axis=0) for s, e in zip(starts, offsets)])


def tet_volumes(points, conn, offsets):
    starts = np.concatenate([[0], offsets[:-1]])
    assert np.all(offsets - starts == 4), "tetrahedra only"
    p = points[conn.reshape(-1, 4)]
    return np.abs(np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]))) / 6.0


def last_file(directory, prefix):
    files = sorted(f for f in os.listdir(directory) if re.fullmatch(prefix + r"_\d+\.vtu", f))
    return os.path.join(directory, files[-1])


def run_case(case_dir, args, scale, geometry):
    os.makedirs(case_dir, exist_ok=True)
    # One mesh per scale for both geometries (Gmsh's parallel 3D mesher is not deterministic)
    mesh = os.path.join(args.run_dir, f"ellipsoid_s{scale:g}.msh")
    if not os.path.exists(mesh):
        subprocess.run([sys.executable, os.path.join(HERE, "make_ellipsoid_mesh.py"), mesh, "--scale", str(scale)],
                       check=True, stdout=subprocess.DEVNULL)
    text = open(os.path.join(HERE, "input.toml")).read()
    text = re.sub(r'(?m)^filename = "ellipsoid.msh"$', f'filename = "{os.path.abspath(mesh)}"', text)
    if geometry == "straight":
        text = text.replace("\n[mesh]\n", "\n[mesh]\ncurved_geometry = false\n", 1)
    text = re.sub(r"(?m)^t_stop = .*$", f"t_stop = {args.t_stop}", text)
    text = re.sub(r"(?m)^order = .*$", f"order = {args.order}", text)
    text = re.sub(r"(?m)^time_interval = .*$", f"time_interval = {args.t_stop / 4}", text)
    with open(os.path.join(case_dir, "input.toml"), "w") as f:
        f.write(text)
    cmd = args.launcher.split() + [args.mallard, "-i", "input.toml", f"--kokkos-num-threads={args.threads}"]
    with open(os.path.join(case_dir, "log.txt"), "w") as log:
        subprocess.run(cmd, cwd=case_dir, stdout=log, stderr=subprocess.STDOUT)


def finished(case_dir):
    log = os.path.join(case_dir, "log.txt")
    return os.path.exists(log) and "t_stop = " in open(log).read().split("== Summary")[-1]


def analyze(case_dir, scale, geometry):
    if not finished(case_dir):
        return {"scale": scale, "geometry": geometry, "failed": True}
    solut = os.path.join(case_dir, "solut")

    forces = np.genfromtxt(os.path.join(solut, "forces_ellipsoid.csv"), delimiter=",", names=True)
    late = forces["t"] >= 0.75 * forces["t"][-1]
    cd = 4.0 * forces["Fx_pressure"][late].mean() / (0.5 * U * U * np.pi * B * B)

    points, conn, offsets, _, fields = read_vtu_cells(last_file(solut, "ellipsoid"))
    volume = tet_volumes(points, conn, offsets)
    centroid = cell_centroids(points, conn, offsets)
    ds = fields["P"] / fields["RHO"] ** GAMMA / P_INF - 1.0
    near = np.sum((centroid / np.array([A, B, B])) ** 2, axis=1) < 9.0
    s_rms = np.sqrt(np.sum(volume[near] * ds[near] ** 2) / np.sum(volume[near]))

    wpoints, wconn, woffsets, _, wfields = read_vtu_cells(last_file(solut, "wall"))
    wc = cell_centroids(wpoints, wconn, woffsets)
    cp = (wfields["P"] - P_INF) / (0.5 * U * U)
    cp_rms = np.sqrt(np.mean((cp - cp_potential(*wc.T)) ** 2))
    order = np.argsort(wc[:, 0])
    return {"scale": scale, "geometry": geometry, "n_cells": int(len(volume)), "Cd": float(cd),
            "entropy_rms": float(s_rms), "entropy_max": float(np.abs(ds[near]).max()), "cp_rms": float(cp_rms),
            "wall_x": wc[order, 0].tolist(), "wall_cp": cp[order].tolist(), "failed": False}


def plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))
    ok = [r for r in results if not r["failed"]]
    finest = max(r["scale"] for r in ok)
    x = np.linspace(-A, A, 401)
    y = B * np.sqrt(np.clip(1.0 - (x / A) ** 2, 0.0, None))
    ax0.plot(x, cp_potential(x, y, 0 * x), "k-", lw=1, label="potential flow (incompressible)")
    for r in ok:
        if r["scale"] == finest:
            ax0.plot(r["wall_x"], r["wall_cp"], ".", ms=2, label=f"{r['geometry']} faces")
    ax0.set_xlabel("x / a")
    ax0.set_ylabel("Cp")
    ax0.invert_yaxis()
    ax0.legend()
    for geometry, marker in (("straight", "s"), ("curved", "o")):
        rows = sorted((r for r in ok if r["geometry"] == geometry), key=lambda r: r["scale"])
        if rows:
            h = [1.0 / r["scale"] for r in rows]
            ax1.loglog(h, [r["entropy_rms"] for r in rows], marker + "-", label=f"entropy error, {geometry}")
            ax1.loglog(h, [abs(r["Cd"]) for r in rows], marker + "--", label=f"|Cd|, {geometry}")
    ax1.set_xlabel("mesh size (1 / scale)")
    ax1.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--mallard", required=True)
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 1.4, 2.0])
    ap.add_argument("--geometry", nargs="+", default=["straight", "curved"])
    ap.add_argument("--t-stop", type=float, default=40.0)
    ap.add_argument("--order", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--launcher", default="", help="Command prefix, e.g. 'mpirun -n 2'")
    ap.add_argument("-o", "--output", default="")
    args = ap.parse_args()
    args.mallard = os.path.abspath(args.mallard)
    os.makedirs(args.run_dir, exist_ok=True)
    results = []
    for scale in args.scales:
        for geometry in args.geometry:
            case_dir = os.path.join(args.run_dir, f"{geometry}_s{scale:g}")
            if not finished(case_dir):
                run_case(case_dir, args, scale, geometry)
            results.append(analyze(case_dir, scale, geometry))
    print(f"{'scale':>5} {'geometry':>9} {'cells':>7} {'Cd':>10} {'s_rms':>10} {'s_max':>10} {'Cp_rms':>8}")
    for r in sorted(results, key=lambda r: (r["scale"], r["geometry"])):
        if r["failed"]:
            print(f"{r['scale']:5g} {r['geometry']:>9} failed")
            continue
        print(f"{r['scale']:5g} {r['geometry']:>9} {r['n_cells']:7d} {r['Cd']:10.2e} {r['entropy_rms']:10.2e} "
              f"{r['entropy_max']:10.2e} {r['cp_rms']:8.4f}")
    if args.output:
        plot(results, args.output)


if __name__ == "__main__":
    main()
