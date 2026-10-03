#!/usr/bin/env python3
"""Animate the 3D Sedov-Taylor blast: density on the symmetry planes, the exact shock.

    animate_sedov.py PLANES_DIR OUTPUT_BASE [--energy 1] [--gamma 1.4]
        [--width 1920] [--fps 15] [--orbit 120] [--gif-width 0]

PLANES_DIR holds the x0, y0 and z0 surface series of examples/sedov_3d
(density on the planes x = 0, y = 0 and z = 0 of the octant). Each quarter
plane is mirrored into a full disk, giving three orthogonal cuts through the
blast; a translucent sphere marks the exact shock radius
R(t) = xi0 (E t^2 / rho0)^(1/5) while the camera orbits. The side panel
traces the shock radius measured on the planes against R(t), and the density
on the planes against the exact similarity profile. Writes OUTPUT_BASE.mp4
(H.264, CRF 18), OUTPUT_BASE_still.png (last frame) and, with --gif-width,
OUTPUT_BASE.gif. Needs pyvista besides the packages in tools/README.md.
"""
import argparse
import os
import re
import tempfile

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from animate import write_gif, write_mp4
from sedov import shock_radius, similarity_profiles, xi0

BG = "#0d1117"
FG = "#e6edf3"
GRID = "#30363d"
DISK_RADIUS = 1.08  # The planes are cut at 1.2 R(t), so the cutaway grows with the blast


def series(planes, name):
    text = open(os.path.join(planes, name + ".pvd")).read()
    return [(float(t), os.path.join(planes, f)) for t, f in re.findall(r'timestep="([^"]+)"[^>]*file="([^"]+)"', text)]


def read_plane(path):
    """Faces of one plane (VTU or PVTU) as PolyData with cell data RHO."""
    return pv.read(path).extract_surface(algorithm="dataset_surface")


def mirrored(poly, normal_axis, radius):
    """Mirror a quarter plane (normal along normal_axis) across its two in-plane
    axes, keeping the cells within radius of the origin."""
    r = np.linalg.norm(poly.cell_centers().points, axis=1)
    parts = [poly.extract_cells(np.nonzero(r < radius)[0]).extract_surface(algorithm="dataset_surface")]
    for axis in range(3):
        if axis == normal_axis:
            continue
        n = [0.0, 0.0, 0.0]
        n[axis] = 1.0
        parts += [p.reflect(n, point=(0, 0, 0)) for p in parts]
    return pv.merge(parts)


def render_3d(plotter, planes, R_exact, azimuth, rho_max):
    plotter.clear_actors()
    for poly in planes:
        plotter.add_mesh(poly, scalars="RHO", cmap="magma", clim=(0.0, rho_max), show_scalar_bar=False,
                         lighting=False)
    shell = pv.Sphere(radius=R_exact, theta_resolution=96, phi_resolution=96)
    plotter.add_mesh(shell, color="#58a6ff", opacity=0.12, smooth_shading=True, specular=0.6)
    globe = pv.Sphere(radius=R_exact, theta_resolution=24, phi_resolution=13)
    plotter.add_mesh(globe, style="wireframe", color="#58a6ff", opacity=0.35, line_width=1.2)
    r = 5.2
    a = np.radians(azimuth)
    plotter.camera_position = [(r * np.cos(a), r * np.sin(a), 0.6 * r), (0, 0, 0), (0, 0, 1)]
    plotter.camera.view_angle = 30
    plotter.reset_camera_clipping_range()
    return plotter.screenshot(return_img=True)


def render_panel(t, history, R_of, r_cells, rho_cells, profile, size, dpi):
    fig = plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi, facecolor=BG)
    ax_r = fig.add_axes([0.17, 0.53, 0.76, 0.27], facecolor=BG)
    ax_p = fig.add_axes([0.17, 0.09, 0.76, 0.33], facecolor=BG)
    ts = np.linspace(0, history[-1][0] if history else t, 200)
    t_end = max(t, 1e-9)
    ax_r.plot(np.linspace(0, R_of.t_stop, 200), R_of(np.linspace(0, R_of.t_stop, 200)), color="#8b949e", lw=2.2,
              label="Sedov-Taylor $\\xi_0 (E t^2/\\rho_0)^{1/5}$")
    if history:
        h = np.array(history)
        ax_r.plot(h[:, 0], h[:, 1], color="#ff9e3d", lw=2.6, label="Mallard")
        ax_r.plot(h[-1, 0], h[-1, 1], "o", color="#ff9e3d", ms=8)
    ax_r.set_xlim(0, R_of.t_stop)
    ax_r.set_ylim(0, 1.05)
    ax_r.set_xlabel("t", color=FG, fontsize=14)
    ax_r.set_ylabel("Shock radius", color=FG, fontsize=14)
    eta, g = profile
    R = R_of(t_end)
    ax_p.plot(r_cells, rho_cells, ".", color="#ff9e3d", ms=2.5, alpha=0.35, label="Mallard (plane cells)")
    rr = np.linspace(0, 1.1, 600)
    ax_p.plot(rr, np.where(rr < R, np.interp(rr / R, eta[::-1], g[::-1]), 1.0), color="#8b949e", lw=2.2,
              label="Exact")
    ax_p.set_xlim(0, 1.1)
    ax_p.set_ylim(0, 6.5)
    ax_p.set_xlabel("r", color=FG, fontsize=14)
    ax_p.set_ylabel("Density", color=FG, fontsize=14)
    for ax in (ax_r, ax_p):
        ax.tick_params(colors=FG, labelsize=11)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.grid(color=GRID, lw=0.8)
        ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=11, loc="upper left", markerscale=4)
    fig.text(0.08, 0.93, "Sedov-Taylor blast wave", color=FG, fontsize=20, weight="bold")
    fig.text(0.08, 0.885, f"t = {t:5.3f}", color=FG, fontsize=16, family="monospace")
    fig.text(0.40, 0.885, "density on the symmetry planes; blue: exact shock", color="#8b949e", fontsize=11)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("planes")
    ap.add_argument("output")
    ap.add_argument("--energy", type=float, default=1.0)
    ap.add_argument("--rho0", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.4)
    ap.add_argument("--rho-max", type=float, default=5.0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--orbit", type=float, default=120.0)
    ap.add_argument("--gif-width", type=int, default=0)
    ap.add_argument("--every", type=int, default=1)
    args = ap.parse_args()

    names = ["x0", "y0", "z0"]
    lists = [series(args.planes, n) for n in names]
    n_frames = min(len(s) for s in lists)
    x0c = xi0(args.gamma)

    def R_of(t):
        return x0c * (args.energy * np.asarray(t) ** 2 / args.rho0) ** 0.2

    R_of.t_stop = lists[0][n_frames - 1][0]
    eta, _, g, _ = similarity_profiles(args.gamma)
    profile = (eta, g * args.rho0)

    W, H = args.width, round(args.width * 9 / 16)
    w3d = round(W * 0.6)
    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True, window_size=(w3d, H))
    plotter.set_background(BG)
    plotter.enable_anti_aliasing("ssaa")
    history = []
    frames = list(range(0, n_frames, args.every))
    with tempfile.TemporaryDirectory() as tmp:
        for k, i in enumerate(frames):
            t = lists[0][i][0]
            polys, r_all, rho_all = [], [], []
            for axis, s in enumerate(lists):
                p = read_plane(s[i][1])
                c = p.cell_centers().points
                r_all.append(np.linalg.norm(c, axis=1))
                rho_all.append(np.asarray(p.cell_data["RHO"]))
                polys.append(mirrored(p, axis, min(DISK_RADIUS, 1.2 * float(R_of(max(t, 1e-6))) + 0.03)))
            r_cells, rho_cells = np.concatenate(r_all), np.concatenate(rho_all)
            if t > 0:
                history.append((t, shock_radius(r_cells, rho_cells, args.rho0)))
            az = 30 + args.orbit * k / max(len(frames) - 1, 1)
            left = render_3d(plotter, polys, float(R_of(max(t, 1e-6))), az, args.rho_max)
            right = render_panel(t, history, R_of, r_cells, rho_cells, profile, (W - w3d, H), 100)
            frame = np.concatenate([left[:H, :w3d], right[:H]], axis=1)
            imageio.imwrite(os.path.join(tmp, f"f{k:05d}.png"), frame)
            print(f"frame {k + 1}/{len(frames)} t = {t:.3f}", flush=True)
        write_mp4(os.path.join(tmp, "f%05d.png"), args.output + ".mp4", args.fps)
        if args.gif_width > 0:
            write_gif(os.path.join(tmp, "f%05d.png"), args.output + ".gif", args.fps, args.gif_width)
    imageio.imwrite(args.output + "_still.png", frame)
    print(f"wrote {args.output}.mp4")


if __name__ == "__main__":
    main()
