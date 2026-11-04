#!/usr/bin/env python3
"""Animate flat against curved wall faces on the spheroid of examples/ellipsoid.

    animate_ellipsoid.py STRAIGHT_DIR CURVED_DIR OUTPUT_BASE [--width 1920]
        [--fps 20] [--hold 2] [--t-start 0]

STRAIGHT_DIR and CURVED_DIR are runs of input.toml with [mesh]
curved_geometry = false and true (e.g. compare_ellipsoid.py's straight_s4 and
curved_s4). Each frame shows the entropy error |s / s_inf - 1| on the meridian
plane y = 0 (mirrored about the axis) of both runs, the wall pressure
coefficient against incompressible potential flow, and the drag coefficients
so far. Writes OUTPUT_BASE.mp4 (H.264, CRF 18) and OUTPUT_BASE_poster.jpg (the
last frame).
"""
import argparse
import glob
import os
import re
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools", "post"))
from animate import write_mp4  # noqa: E402
from compare_ellipsoid import A, B, GAMMA, P_INF, U, cell_centroids, cp_potential  # noqa: E402
from mallard.vtu import read_vtu_cells  # noqa: E402

BG = "#0d1117"
FG = "#e6edf3"
GRID = "#30363d"
FLAT = "#f2a04d"
CURVED = "#5fcfc1"


def frames(run, prefix):
    files = glob.glob(os.path.join(run, "solut", prefix + "_*.vtu"))
    return sorted(files, key=lambda f: int(re.search(r"_(\d+)\.vtu$", f).group(1)))


def polygons(points, conn, offsets):
    starts = np.concatenate([[0], offsets[:-1]])
    return [points[conn[s:e]][:, [0, 2]] for s, e in zip(starts, offsets)]


def slice_panel(ax, path, title):
    points, conn, offsets, _, f = read_vtu_cells(path)
    polys = polygons(points, conn, offsets)
    err = np.clip(np.abs(f["P"] / f["RHO"] ** GAMMA / P_INF - 1.0), 1e-7, None)
    for sign in (1.0, -1.0):
        mirrored = [p * np.array([1.0, sign]) for p in polys]
        ax.add_collection(PolyCollection(mirrored, array=err, cmap="magma", norm=LogNorm(1e-6, 1e-1),
                                         edgecolors="face", linewidths=0.2))
    x = np.linspace(-A, A, 300)
    z = B * np.sqrt(np.clip(1.0 - (x / A) ** 2, 0.0, None))
    ax.fill(np.concatenate([x, x[::-1]]), np.concatenate([z, -z[::-1]]), color="#21262d", zorder=3)
    ax.set_xlim(-2.0, 4.5)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, color=FG, fontsize=15, loc="left")
    return f["TIME"]


def wall_cp(path):
    points, conn, offsets, _, f = read_vtu_cells(path)
    c = cell_centroids(points, conn, offsets)
    order = np.argsort(c[:, 0])
    return c[order, 0], ((f["P"] - P_INF) / (0.5 * U * U))[order]


def drag(run):
    f = np.genfromtxt(os.path.join(run, "solut", "forces_ellipsoid.csv"), delimiter=",", names=True)
    return f["t"], 4.0 * f["Fx_pressure"] / (0.5 * U * U * np.pi * B * B)


def style(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=FG, labelsize=11)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.grid(color=GRID, lw=0.6)


def frame(k, args, runs, cd, size, dpi):
    fig = plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi, facecolor=BG)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], left=0.02, right=0.97, top=0.9, bottom=0.08,
                            wspace=0.12, hspace=0.25)
    t = 0.0
    for row, (name, color, slices, _) in enumerate(runs):
        ax = fig.add_subplot(grid[row, 0])
        ax.set_facecolor(BG)
        t = slice_panel(ax, slices[k], name)
        ax.title.set_color(color)
    sm = plt.cm.ScalarMappable(cmap="magma", norm=LogNorm(1e-6, 1e-1))
    cax = fig.add_axes([0.06, 0.045, 0.45, 0.015])
    bar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    bar.set_label("entropy error |s / s∞ - 1|  (exact: 0)", color=FG, fontsize=12)
    bar.ax.tick_params(colors=FG, labelsize=10)

    ax = fig.add_subplot(grid[0, 1])
    style(ax)
    x = np.linspace(-A, A, 401)
    ax.plot(x, cp_potential(x, B * np.sqrt(np.clip(1 - (x / A) ** 2, 0, None)), 0 * x), color=FG, lw=1.5,
            label="potential flow (incompressible)")
    for name, color, _, walls in runs:
        wx, cp = wall_cp(walls[k])
        ax.plot(wx, cp, ".", ms=1.5, color=color, label=name)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(1.3, -1.6)
    ax.set_xlabel("x / a")
    ax.set_ylabel("wall Cp")
    ax.legend(loc="lower center", fontsize=10, facecolor=BG, edgecolor=GRID, labelcolor=FG, markerscale=6)

    ax = fig.add_subplot(grid[1, 1])
    style(ax)
    for (name, color, _, _), (tc, c) in zip(runs, cd):
        m = tc <= t
        ax.plot(tc[m], c[m], color=color, lw=1.2, label=name)
    ax.axhline(0.0, color=FG, lw=0.8)
    ax.set_xlim(0.0, max(tc[-1] for tc, _ in cd))
    ax.set_ylim(-0.3, 0.3)
    ax.set_xlabel("t c / a")
    ax.set_ylabel("drag coefficient  (exact: 0)")
    fig.suptitle(f"Mach 0.3 past a 2:1 spheroid, the same 627,507 tetrahedra: flat or curved wall faces    "
                 f"t = {t:5.2f}", color=FG, fontsize=16, x=0.02, ha="left")
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("straight")
    ap.add_argument("curved")
    ap.add_argument("output")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--hold", type=float, default=2.0, help="Seconds the last frame is held")
    ap.add_argument("--t-start", type=float, default=0.0)
    args = ap.parse_args()
    import imageio.v2 as imageio

    runs = [("flat faces", FLAT, frames(args.straight, "y0"), frames(args.straight, "wall")),
            ("curved faces", CURVED, frames(args.curved, "y0"), frames(args.curved, "wall"))]
    n = min(min(len(s), len(w)) for _, _, s, w in runs)
    cd = [drag(args.straight), drag(args.curved)]
    size, dpi = (args.width, args.width * 9 // 16), 100
    with tempfile.TemporaryDirectory() as tmp:
        count = 0
        image = None
        for k in range(n):
            if read_vtu_cells(runs[1][2][k])[4]["TIME"] < args.t_start:
                continue
            image = frame(k, args, runs, cd, size, dpi)
            imageio.imwrite(os.path.join(tmp, f"f{count:05d}.png"), image)
            count += 1
        for _ in range(int(args.hold * args.fps)):
            imageio.imwrite(os.path.join(tmp, f"f{count:05d}.png"), image)
            count += 1
        write_mp4(os.path.join(tmp, "f%05d.png"), args.output + ".mp4", args.fps)
    imageio.imwrite(args.output + "_poster.jpg", image, quality=92)


if __name__ == "__main__":
    main()
