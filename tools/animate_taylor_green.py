#!/usr/bin/env python3
"""Animate the Taylor-Green vortex: Q-criterion isosurfaces and the dissipation rate.

    animate_taylor_green.py SOLUT_DIR integrals.csv OUTPUT_BASE
        [--ref spectral_Re1600_512.gdiag] [--q-factor 0.5] [--width 1920] [--fps 15]
        [--orbit 150] [--gif-width 720] [--compare CSV LABEL ...] [--hold 2]

SOLUT_DIR holds the VTU (or PVTU) series of a run of
examples/taylor_green_3d, with U on the octant [0, pi]^3 of hexahedra. Each
snapshot is mirrored into the full periodic box [-pi, pi]^3 through the
vortex's symmetry planes; isosurfaces of Q = (|Omega|^2 - |S|^2) / 2 are
colored by vorticity magnitude while the camera orbits the box, next to the
dissipation rate -dE/dt tracing the reference. Writes OUTPUT_BASE.mp4
(H.264, CRF 18), OUTPUT_BASE.gif (if --gif-width > 0) and a still of the
dissipation peak, OUTPUT_BASE_still.png. Needs pyvista besides the packages in
tools/README.md.
"""
import argparse
import glob
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
from mallard_vtu import read_vtu_cells
from plot_taylor_green import OCTANT_VOLUME, load, load_reference

BG = "#0d1117"
FG = "#e6edf3"


def snapshot_files(solut):
    pvd = glob.glob(os.path.join(solut, "*.pvd"))
    if pvd:
        text = open(pvd[0]).read()
        entries = re.findall(r'timestep="([^"]+)"[^>]*file="([^"]+)"', text)
        return [(float(t), os.path.join(os.path.dirname(pvd[0]), f)) for t, f in entries]
    files = sorted(glob.glob(os.path.join(solut, "*.vtu")))
    return [(None, f) for f in files]


def read_octant(path):
    """Cell-centered velocity of a uniform hexahedral octant as an (n, n, n, 3) array."""
    if path.endswith(".pvtu"):
        pieces = re.findall(r'<Piece Source="([^"]+)"', open(path).read())
        parts = [read_vtu_cells(os.path.join(os.path.dirname(path), p)) for p in pieces]
    else:
        parts = [read_vtu_cells(path)]
    centers, U, t = [], [], None
    for pts, conn, offs, types, arrays in parts:
        starts = np.concatenate([[0], offs[:-1]])
        centers.append(pts[conn[starts[:, None] + np.arange(8)]].mean(axis=1))
        U.append(arrays["U"])
        t = arrays.get("TIME", t)
    centers, U = np.vstack(centers), np.vstack(U)
    n = round(len(U) ** (1 / 3))
    h = np.pi / n
    ijk = np.clip(np.round(centers / h - 0.5).astype(int), 0, n - 1)
    grid = np.empty((n, n, n, 3))
    grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = U
    return grid, t


def mirror(octant):
    """Extend the octant to the full box: across each plane x_d = 0 the normal
    velocity component is odd and the others even (and likewise across pi,
    which periodicity identifies with -pi)."""
    u = octant
    for d in range(3):
        flipped = np.flip(u, axis=d).copy()
        flipped[..., d] *= -1.0
        u = np.concatenate([flipped, u], axis=d)
    return u


def q_and_vorticity(u, h):
    """Q-criterion and vorticity magnitude by central differences on the periodic box."""
    g = np.empty(u.shape[:3] + (3, 3))
    for k in range(3):
        for i in range(3):
            g[..., k, i] = (np.roll(u[..., k], -1, axis=i) - np.roll(u[..., k], 1, axis=i)) / (2 * h)
    S = 0.5 * (g + np.swapaxes(g, -1, -2))
    W = 0.5 * (g - np.swapaxes(g, -1, -2))
    Q = 0.5 * ((W ** 2).sum(axis=(-1, -2)) - (S ** 2).sum(axis=(-1, -2)))
    omega = np.stack([g[..., 2, 1] - g[..., 1, 2], g[..., 0, 2] - g[..., 2, 0], g[..., 1, 0] - g[..., 0, 1]], -1)
    return Q, np.linalg.norm(omega, axis=-1)


def render_3d(plotter, u, q_factor, azimuth):
    """Isosurface Q = q_factor * <|omega|^2> / 2 (the mean enstrophy density), so
    the surfaces track the most intense structures as the vortex breaks down,
    colored by |omega| from 0 to 3x its RMS."""
    n = u.shape[0]
    h = 2 * np.pi / n
    Q, omega = q_and_vorticity(u, h)
    grid = pv.ImageData(dimensions=(n, n, n), spacing=(h, h, h), origin=(-np.pi + h / 2,) * 3)
    grid.point_data["Q"] = Q.ravel(order="F")
    grid.point_data["omega"] = omega.ravel(order="F")
    mean_w2 = float(np.mean(omega ** 2))
    plotter.clear_actors()
    surf = grid.contour([q_factor * 0.5 * mean_w2], scalars="Q")
    if surf.n_points > 0:
        plotter.add_mesh(surf, scalars="omega", cmap="inferno", clim=(0.0, 3.0 * np.sqrt(mean_w2)),
                         smooth_shading=True, specular=0.4, specular_power=20, show_scalar_bar=False)
    plotter.add_mesh(pv.Box(bounds=(-np.pi, np.pi) * 3).outline(), color="#8b949e", line_width=1.5)
    r = 19.0
    a = np.radians(azimuth)
    plotter.camera_position = [(r * np.cos(a), r * np.sin(a), 0.55 * r), (0, 0, -0.2), (0, 0, 1)]
    plotter.camera.view_angle = 30
    plotter.reset_camera_clipping_range()
    return plotter.screenshot(return_img=True)


def render_panel(t_now, curves, ref, size, dpi):
    """Dissipation rate up to t_now over the full reference curve."""
    fig = plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi, facecolor=BG)
    ax = fig.add_axes([0.2, 0.14, 0.74, 0.62], facecolor=BG)
    if ref is not None:
        ax.plot(ref[0], ref[1], color="#8b949e", lw=2.5, label="Spectral DNS $512^3$")
    for (t, eps, label, color) in curves:
        m = t <= t_now
        ax.plot(t[m], eps[m], color=color, lw=2.8, label=label)
        if m.any():
            ax.plot(t[m][-1], eps[m][-1], "o", color=color, ms=9)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 0.0145)
    ax.set_xlabel("$t\\,V_0/L$", color=FG, fontsize=15)
    ax.set_ylabel("$\\epsilon = -dE_k/dt$", color=FG, fontsize=15)
    ax.tick_params(colors=FG, labelsize=12)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, -2))
    ax.yaxis.get_offset_text().set_color(FG)
    for s in ax.spines.values():
        s.set_color("#30363d")
    ax.grid(color="#30363d", lw=0.8)
    ax.legend(facecolor=BG, edgecolor="#30363d", labelcolor=FG, fontsize=12, loc="upper right")
    fig.text(0.08, 0.92, "Taylor-Green vortex, Re = 1600", color=FG, fontsize=20, weight="bold")
    fig.text(0.08, 0.875, f"t = {t_now:5.2f}", color=FG, fontsize=16, family="monospace")
    fig.text(0.08, 0.83, "Q-criterion isosurfaces colored by |vorticity|", color="#8b949e", fontsize=12)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("solut")
    ap.add_argument("integrals")
    ap.add_argument("output")
    ap.add_argument("--ref")
    ap.add_argument("--label", default="Mallard TENO5")
    ap.add_argument("--q-factor", type=float, default=0.5,
                    help="Isosurface level in units of the mean enstrophy density")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--orbit", type=float, default=150.0, help="Camera orbit over the whole animation, degrees")
    ap.add_argument("--gif-width", type=int, default=720)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--compare", nargs=2, action="append", default=[], metavar=("CSV", "LABEL"),
                    help="Another run's integrals to trace in the panel (e.g. a coarser mesh)")
    ap.add_argument("--hold", type=float, default=2.0, help="Seconds to hold the last frame")
    args = ap.parse_args()

    files = snapshot_files(args.solut)[::args.every]
    t, E, eps, _ = load(args.integrals, 1.0 / 1600)
    curves = [(t, eps, args.label, "#ff9e3d")]
    for (path, label), color in zip(args.compare, ["#58a6ff", "#3fb950", "#d2a8ff"]):
        tc, _, ec, _ = load(path, 1.0 / 1600)
        curves.append((tc, ec, label, color))
    ref = None
    if args.ref:
        tr, _, er = load_reference(args.ref)
        ref = (tr, er)

    W, H = args.width, round(args.width * 9 / 16)
    w3d = round(W * 0.62)
    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True, window_size=(w3d, H))
    plotter.set_background(BG)
    plotter.enable_anti_aliasing("ssaa")
    still_t = 9.0
    still, still_dt = None, np.inf
    with tempfile.TemporaryDirectory() as tmp:
        for k, (t_file, path) in enumerate(files):
            octant, t_vtu = read_octant(path)
            t_now = t_vtu if t_vtu is not None else t_file
            az = 35 + args.orbit * k / max(len(files) - 1, 1)
            left = render_3d(plotter, mirror(octant), args.q_factor, az)
            right = render_panel(t_now, curves, ref, (W - w3d, H), 100)
            frame = np.concatenate([left[:H, :w3d], right[:H]], axis=1)
            imageio.imwrite(os.path.join(tmp, f"f{k:05d}.png"), frame)
            if abs(t_now - still_t) < still_dt:
                still, still_dt = frame, abs(t_now - still_t)
            print(f"frame {k + 1}/{len(files)} t = {t_now:.2f}", flush=True)
        for j in range(round(args.hold * args.fps)):
            imageio.imwrite(os.path.join(tmp, f"f{len(files) + j:05d}.png"), frame)
        write_mp4(os.path.join(tmp, "f%05d.png"), args.output + ".mp4", args.fps)
        if args.gif_width > 0:
            write_gif(os.path.join(tmp, "f%05d.png"), args.output + ".gif", args.fps, args.gif_width)
    imageio.imwrite(args.output + "_still.png", still)
    print(f"wrote {args.output}.mp4")


if __name__ == "__main__":
    main()
