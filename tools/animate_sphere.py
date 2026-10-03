#!/usr/bin/env python3
"""Animate supersonic flow over a sphere: Mach number, schlieren and the bow shock in 3D.

    animate_sphere.py SLICES_DIR OUTPUT_BASE [--mach 3] [--gamma 1.4]
        [--width 1920] [--fps 15] [--orbit 70] [--hold 2] [--gif-width 0]

SLICES_DIR holds the y0, z0 and wall series of examples/sphere_mach3: the
meridian planes y = 0 and z = 0 of the quarter domain and the sphere surface.
The flow is axisymmetric, so the z = 0 plane, mirrored, gives the Mach number
on a full horizontal cut; the y = 0 plane shows numerical schlieren; and the
bow shock (the pressure contour p = 2 p_inf on the meridian) revolved about
the axis gives a translucent 3D shock surface over the sphere. The side panel
traces the shock standoff against Billig's correlation. Writes
OUTPUT_BASE.mp4 (H.264, CRF 18), OUTPUT_BASE_still.png and, with --gif-width,
OUTPUT_BASE.gif. Needs pyvista besides the packages in tools/README.md.
"""
import argparse
import os
import tempfile

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from animate import write_gif, write_mp4
from plot_sphere import R_SPHERE, billig, series, stagnation_line, standoff

BG = "#0d1117"
FG = "#e6edf3"
GRID = "#30363d"
WINDOW = (-1.3, 2.4, -1.9, 1.9, -0.01, 1.9)


def mirror(mesh, axes):
    parts = [mesh]
    for axis in axes:
        n = [0.0, 0.0, 0.0]
        n[axis] = 1.0
        parts += [p.reflect(n, point=(0, 0, 0)) for p in parts]
    return pv.merge(parts)


def with_mach(mesh, gamma):
    rho = np.asarray(mesh.cell_data["RHO"])
    p = np.asarray(mesh.cell_data["P"])
    u = np.asarray(mesh.cell_data["U"])
    mesh.cell_data["MACH"] = np.linalg.norm(u, axis=1) / np.sqrt(gamma * p / rho)
    return mesh


def schlieren(mesh, k=12.0):
    pts = mesh.cell_data_to_point_data()
    grad = pts.compute_derivative(scalars="RHO", gradient="G").point_data["G"]
    g = np.linalg.norm(grad, axis=1)
    pts.point_data["SCHLIEREN"] = np.exp(-k * g / max(np.percentile(g, 99.5), 1e-12))
    pts.point_data["SCHLIEREN"] = 1.0 - pts.point_data["SCHLIEREN"]
    return pts


def shock_surface(z0, level):
    """Bow shock: the meridian contour p = level revolved over the upper half space."""
    line = z0.cell_data_to_point_data().contour([level], scalars="P")
    if line.n_points == 0:
        return None
    line = line.extract_largest().extract_surface(algorithm="dataset_surface")
    return line.extrude_rotate(resolution=72, angle=180.0, rotation_axis=(1, 0, 0),
                                                 capping=False)


def window(mesh):
    return mesh.clip_box(WINDOW, invert=False)


def render_3d(plotter, z0, y0, wall, shock, azimuth):
    plotter.clear_actors()
    plotter.add_mesh(window(mirror(z0, [1])), scalars="MACH", cmap="magma_r", clim=(0.0, 3.2),
                     show_scalar_bar=False, lighting=False)
    plotter.add_mesh(window(y0), scalars="SCHLIEREN", cmap="gray", clim=(0.0, 1.0), show_scalar_bar=False,
                     lighting=False, opacity=0.9)
    plotter.add_mesh(mirror(wall, [1, 2]), color="#c9d1d9", smooth_shading=True, specular=0.8, specular_power=30)
    if shock is not None:
        plotter.add_mesh(window(shock), color="#58a6ff", opacity=0.22, smooth_shading=True, specular=0.5)
    r = 8.5
    a = np.radians(azimuth)
    focal = (0.45, 0.0, 0.25)
    plotter.camera_position = [(focal[0] + r * np.cos(a), focal[1] + r * np.sin(a), 0.5 * r), focal, (0, 0, 1)]
    plotter.camera.view_angle = 30
    plotter.reset_camera_clipping_range()
    return plotter.screenshot(return_img=True)


def render_panel(t, history, line, mach, t_end, size, dpi):
    ref = billig(mach)
    fig = plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi, facecolor=BG)
    ax_d = fig.add_axes([0.17, 0.53, 0.76, 0.27], facecolor=BG)
    ax_x = fig.add_axes([0.17, 0.09, 0.76, 0.33], facecolor=BG)
    ax_d.axhline(ref, color="#8b949e", lw=2.2, ls="--", label=f"Billig: {ref:.3f}")
    if history:
        h = np.array(history)
        ax_d.plot(h[:, 0], h[:, 1], color="#ff9e3d", lw=2.6, label="Mallard")
        ax_d.plot(h[-1, 0], h[-1, 1], "o", color="#ff9e3d", ms=8)
    ax_d.set_xlim(0, t_end)
    ax_d.set_ylim(0, 0.4)
    ax_d.set_xlabel("$t\\,u_\\infty / D$", color=FG, fontsize=14)
    ax_d.set_ylabel("Standoff $\\Delta / R$", color=FG, fontsize=14)
    x, p = line
    ax_x.plot(x, p, color="#ff9e3d", lw=2.4, label="Mallard, stagnation line")
    ax_x.axvline(-R_SPHERE * (1 + ref), color="#8b949e", lw=2.2, ls="--", label="Billig shock position")
    ax_x.set_xlim(-1.5, -0.5)
    ax_x.set_ylim(0, 13.5)
    ax_x.set_xlabel("x / D", color=FG, fontsize=14)
    ax_x.set_ylabel("$p / p_\\infty$", color=FG, fontsize=14)
    for ax in (ax_d, ax_x):
        ax.tick_params(colors=FG, labelsize=11)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.grid(color=GRID, lw=0.8)
        ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=11, loc="upper left")
    fig.text(0.08, 0.93, f"Mach {mach:g} flow over a sphere", color=FG, fontsize=20, weight="bold")
    fig.text(0.08, 0.885, f"t = {t:5.2f}", color=FG, fontsize=16, family="monospace")
    fig.text(0.36, 0.885, "Mach number, schlieren, bow shock", color="#8b949e", fontsize=11)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slices")
    ap.add_argument("output")
    ap.add_argument("--mach", type=float, default=3.0)
    ap.add_argument("--gamma", type=float, default=1.4)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--orbit", type=float, default=70.0)
    ap.add_argument("--hold", type=float, default=2.0)
    ap.add_argument("--gif-width", type=int, default=0)
    ap.add_argument("--every", type=int, default=1)
    args = ap.parse_args()

    names = ["z0", "y0", "wall"]
    lists = [series(args.slices, n) for n in names]
    n = min(len(s) for s in lists)
    frames = list(range(0, n, args.every))
    t_end = lists[0][n - 1][0]

    W, H = args.width, round(args.width * 9 / 16)
    w3d = round(W * 0.62)
    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True, window_size=(w3d, H))
    plotter.set_background(BG)
    plotter.enable_anti_aliasing("ssaa")
    history = []
    with tempfile.TemporaryDirectory() as tmp:
        for k, i in enumerate(frames):
            t = lists[0][i][0]
            z0 = with_mach(pv.read(lists[0][i][1]).extract_surface(algorithm="dataset_surface"), args.gamma)
            y0 = schlieren(pv.read(lists[1][i][1]).extract_surface(algorithm="dataset_surface"))
            wall = pv.read(lists[2][i][1]).extract_surface(algorithm="dataset_surface")
            if t > 0:
                history.append((t, standoff(z0, args.mach, args.gamma)))
            az = -115 + args.orbit * k / max(len(frames) - 1, 1)
            left = render_3d(plotter, z0, y0, wall, shock_surface(z0, 2.0), az)
            right = render_panel(t, history, stagnation_line(z0), args.mach, t_end, (W - w3d, H), 100)
            frame = np.concatenate([left[:H, :w3d], right[:H]], axis=1)
            imageio.imwrite(os.path.join(tmp, f"f{k:05d}.png"), frame)
            print(f"frame {k + 1}/{len(frames)} t = {t:.2f}", flush=True)
        for j in range(round(args.hold * args.fps)):
            imageio.imwrite(os.path.join(tmp, f"f{len(frames) + j:05d}.png"), frame)
        write_mp4(os.path.join(tmp, "f%05d.png"), args.output + ".mp4", args.fps)
        if args.gif_width > 0:
            write_gif(os.path.join(tmp, "f%05d.png"), args.output + ".gif", args.fps, args.gif_width)
    imageio.imwrite(args.output + "_still.png", frame)
    print(f"wrote {args.output}.mp4")


if __name__ == "__main__":
    main()
