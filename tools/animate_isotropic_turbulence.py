#!/usr/bin/env python3
"""Animate decaying isotropic turbulence: Q-criterion isosurfaces, shocklets and statistics.

    animate_isotropic_turbulence.py FIELDS_DIR stats.csv OUTPUT_BASE
        [--ref examples/isotropic_turbulence/reference/johnsen2010_filtered256.csv]
        [--q-factor 1.0] [--width 1920] [--fps 15] [--orbit 120] [--hold 2]
        [--gif-width 640] [--still 1.0] [--title ...]

FIELDS_DIR holds the fields_<step>.npz files of a run written by
tools/isotropic_turbulence_stats.py --fields, stats.csv its statistics.
Left: isosurfaces Q = q_factor <|omega|^2> / 2 colored by the dilatation
theta (red: compression, blue: expansion), while the camera orbits the box. Right: the
dilatation on the plane z = pi, where the eddy shocklets are the thin sheets
of strong compression, and the enstrophy and dilatation variance (filtered to
64^3 as the reference) tracing Johnsen et al.'s. Writes OUTPUT_BASE.mp4 (H.264,
CRF 18), OUTPUT_BASE.gif (if --gif-width > 0) and a still at t / tau = --still,
OUTPUT_BASE_still.png. Needs pyvista besides the packages in tools/README.md.
"""
import argparse
import glob
import os
import tempfile

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pyvista as pv

from animate import write_gif, write_mp4
from isotropic_turbulence_restart import parameters
from plot_isotropic_turbulence import load_csv

BG = "#0d1117"
FG = "#e6edf3"
MUTED = "#8b949e"
GRID = "#30363d"
THETA_CLIM = 4.0              # dilatation color range, in units of u_rms0 / lambda0 (3D)
SLICE_RANGE = (-10.0, 4.0)    # and on the slice, where the shocklets reach about -25


def render_3d(plotter, f, q_factor, azimuth, theta_scale):
    n = f["Q"].shape[0]
    h = 2 * np.pi / n
    grid = pv.ImageData(dimensions=(n, n, n), spacing=(h, h, h), origin=(h / 2,) * 3)
    Q = f["Q"].astype(np.float32)
    grid.point_data["Q"] = Q.ravel(order="F")
    grid.point_data["theta"] = (f["dilatation"].astype(np.float32) / theta_scale).ravel(order="F")
    w2 = float(np.mean(f["vorticity"].astype(np.float32) ** 2))
    plotter.clear_actors()
    surf = grid.contour([q_factor * 0.5 * w2], scalars="Q")
    if surf.n_points > 0:
        plotter.add_mesh(surf, scalars="theta", cmap="RdBu", clim=(-THETA_CLIM, THETA_CLIM), smooth_shading=True,
                         specular=0.3, specular_power=15, show_scalar_bar=False)
    plotter.add_mesh(pv.Box(bounds=(0, 2 * np.pi) * 3).outline(), color=MUTED, line_width=1.5)
    c = np.pi
    r = 18.5
    a = np.radians(azimuth)
    plotter.camera_position = [(c + r * np.cos(a), c + r * np.sin(a), c + 0.5 * r), (c, c, c - 0.3), (0, 0, 1)]
    plotter.camera.view_angle = 30
    plotter.reset_camera_clipping_range()
    return plotter.screenshot(return_img=True)


def render_panel(f, t_now, stats, ref, size, dpi, theta_scale, title):
    fig = plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi, facecolor=BG)
    fig.text(0.06, 0.955, title, color=FG, fontsize=19, weight="bold")
    fig.text(0.06, 0.918, f"t / τ = {t_now:4.2f}", color=FG, fontsize=15, family="monospace")
    fig.text(0.06, 0.888, "Q isosurfaces colored by dilatation θ (red: compression); right: θ on z = π",
             color=MUTED, fontsize=11)
    ax = fig.add_axes([0.1, 0.36, 0.8, 0.5], facecolor=BG)
    s = f["slice_dilatation"].T / theta_scale
    norm = TwoSlopeNorm(vmin=SLICE_RANGE[0], vcenter=0.0, vmax=SLICE_RANGE[1])
    im = ax.imshow(s, origin="lower", extent=(0, 2 * np.pi, 0, 2 * np.pi), cmap="RdBu", norm=norm,
                   interpolation="bilinear")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GRID)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("θ λ₀ / u₀ (red: compression)", color=FG, fontsize=11)
    cb.ax.tick_params(colors=FG, labelsize=9)
    cb.outline.set_edgecolor(GRID)
    for k, (key, label, ylim) in enumerate([("enstrophy", "⟨ω²⟩ λ₀²/u₀²", 32),
                                             ("dilatation_variance", "⟨θ²⟩ λ₀²/u₀²",
                                              1.6)]):
        a = fig.add_axes([0.1 + 0.46 * k, 0.07, 0.36, 0.2], facecolor=BG)
        if ref is not None:
            a.plot(ref["t_over_tau"], ref[key], "o", mfc="none", mec=MUTED, ms=5)
        m = stats["t_over_tau"] <= t_now + 1e-9
        a.plot(stats["t_over_tau"][m], stats[f"{key}_fd6"][m], color="#ff9e3d", lw=2.2)
        if m.any():
            a.plot(stats["t_over_tau"][m][-1], stats[f"{key}_fd6"][m][-1], "o", color="#ff9e3d", ms=7)
        a.set_xlim(0, 4)
        a.set_ylim(0, ylim)
        a.set_title(label, color=FG, fontsize=11)
        a.set_xlabel("t / τ", color=FG, fontsize=10)
        a.tick_params(colors=FG, labelsize=9)
        for sp in a.spines.values():
            sp.set_color(GRID)
        a.grid(color=GRID, lw=0.6)
    fig.text(0.1, 0.005, "lines: this run filtered to 64³; circles: Johnsen et al. 2010",
             color=MUTED, fontsize=9)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fields")
    ap.add_argument("stats")
    ap.add_argument("output")
    ap.add_argument("--ref", default=os.path.join(os.path.dirname(__file__), "..", "examples", "isotropic_turbulence",
                                                  "reference", "johnsen2010_filtered256.csv"))
    ap.add_argument("--q-factor", type=float, default=1.0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--orbit", type=float, default=120.0)
    ap.add_argument("--hold", type=float, default=2.0)
    ap.add_argument("--gif-width", type=int, default=640)
    ap.add_argument("--gif-every", type=int, default=2, help="Use every n-th frame in the GIF")
    ap.add_argument("--still", type=float, default=1.0, help="t / tau of the still")
    ap.add_argument("--title", default="Isotropic turbulence, Mt = 0.6, Reλ = 100")
    args = ap.parse_args()

    par = parameters()
    theta_scale = par["u_rms"] / par["lambda"]
    files = sorted(glob.glob(os.path.join(args.fields, "fields_*.npz")))
    stats = np.genfromtxt(args.stats, delimiter=",", names=True)
    stats = stats[np.argsort(stats["t"])]
    ref = load_csv(args.ref) if args.ref else None

    W, H = args.width, round(args.width * 9 / 16)
    w3d = round(W * 0.58)
    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True, window_size=(w3d, H))
    plotter.set_background(BG)
    plotter.enable_anti_aliasing("ssaa")
    still, still_dt = None, np.inf
    with tempfile.TemporaryDirectory() as tmp:
        k = 0
        for k, path in enumerate(files):
            f = np.load(path)
            t_now = float(f["t_over_tau"])
            az = 30 + args.orbit * k / max(len(files) - 1, 1)
            left = render_3d(plotter, f, args.q_factor, az, theta_scale)
            right = render_panel(f, t_now, stats, ref, (W - w3d, H), 100, theta_scale, args.title)
            frame = np.concatenate([left[:H, :w3d], right[:H]], axis=1)
            imageio.imwrite(os.path.join(tmp, f"f{k:05d}.png"), frame)
            if abs(t_now - args.still) < still_dt:
                still, still_dt = frame, abs(t_now - args.still)
            print(f"frame {k + 1}/{len(files)} t/tau = {t_now:.2f}", flush=True)
        n_frames = len(files)
        for j in range(round(args.hold * args.fps)):
            imageio.imwrite(os.path.join(tmp, f"f{n_frames + j:05d}.png"), frame)
        write_mp4(os.path.join(tmp, "f%05d.png"), args.output + ".mp4", args.fps)
        if args.gif_width > 0:
            gif_dir = os.path.join(tmp, "gif")
            os.makedirs(gif_dir)
            total = n_frames + round(args.hold * args.fps)
            for j, i in enumerate(range(0, total, args.gif_every)):
                os.link(os.path.join(tmp, f"f{i:05d}.png"), os.path.join(gif_dir, f"g{j:05d}.png"))
            write_gif(os.path.join(gif_dir, "g%05d.png"), args.output + ".gif", args.fps / args.gif_every,
                      args.gif_width, colors=128)
    imageio.imwrite(args.output + "_still.png", still)
    print(f"wrote {args.output}.mp4")


if __name__ == "__main__":
    main()
