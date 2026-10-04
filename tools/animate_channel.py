#!/usr/bin/env python3
"""Animate the near-wall vortices of the turbulent channel (examples/channel_retau180).

    python tools/animate_channel.py input.toml frames/channel_*.restart --out media/channel_retau180

Each restart file is one frame: isosurfaces of the Q criterion,
Q = (|Omega|^2 - |S|^2) / 2, in the lower half of the channel (the
quasi-streamwise and hairpin vortices of the wall cycle), colored by the
streamwise velocity, over a translucent plane of u near the wall
(--slice-yplus) showing the low- and high-speed streaks. Velocity gradients
are second-order central differences on the stretched grid (periodic in x
and z). Writes OUT.mp4 (H.264, CRF 18, --fps frames per second, the last
frame held --hold seconds), OUT.png (the last frame) and OUT.gif (half size).
Needs pyvista and imageio-ffmpeg.
"""
import argparse
import os
import sys
import tomllib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from channel_init import grid, read_restart  # noqa: E402


def velocity_gradient(u, x, y, z):
    """d u_i / d x_j at the cell centers: (3, 3, nx, ny, nz)."""
    g = np.empty((3, 3) + u[0].shape)
    dx, dz = x[1] - x[0], z[1] - z[0]
    for i in range(3):
        g[i, 0] = (np.roll(u[i], -1, 0) - np.roll(u[i], 1, 0)) / (2 * dx)
        g[i, 1] = np.gradient(u[i], y, axis=1)
        g[i, 2] = (np.roll(u[i], -1, 2) - np.roll(u[i], 1, 2)) / (2 * dz)
    return g


def q_criterion(g):
    s = 0.5 * (g + g.transpose(1, 0, 2, 3, 4))
    w = 0.5 * (g - g.transpose(1, 0, 2, 3, 4))
    return 0.5 * ((w**2).sum((0, 1)) - (s**2).sum((0, 1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("input")
    ap.add_argument("frames", nargs="+")
    ap.add_argument("--out", required=True, help="output path without extension")
    ap.add_argument("--fps", type=float, default=15)
    ap.add_argument("--hold", type=float, default=2.0, help="seconds the last frame is held")
    ap.add_argument("--u-tau", type=float, default=1 / 15.68, help="friction velocity in units of U_b")
    ap.add_argument("--q-plus", type=float, default=0.01, help="isosurface level of Q nu^2 / u_tau^4")
    ap.add_argument("--slice-yplus", type=float, default=12.0, help="wall distance of the streak plane")
    ap.add_argument("--size", type=int, nargs=2, default=(1920, 1080))
    args = ap.parse_args()
    import imageio.v2 as imageio
    import pyvista as pv

    pv.OFF_SCREEN = True
    with open(args.input, "rb") as f:
        inp = tomllib.load(f)
    n, L, (x, y, z) = grid(inp)
    nu = float(inp["physics"]["mu"]) / float(inp["physics"]["rho_ref"])
    h = 0.5 * L[1]
    u_tau = args.u_tau
    q_level = args.q_plus * u_tau**4 / nu**2
    ny_half = n[1] // 2
    yh = y[:ny_half]
    j_slice = int(np.argmin(np.abs(yh * u_tau / nu - args.slice_yplus)))

    frames = sorted(args.frames, key=lambda p: read_restart(p)[0])
    images = []
    for k, path in enumerate(frames):
        step, t, fields = read_restart(path)
        rho = fields["RHO"].reshape(n)
        u = np.stack([fields[f"RHOU_{c}"].reshape(n) / rho for c in "XYZ"])
        q = q_criterion(velocity_gradient(u, x, y, z))[:, :ny_half]
        ux = u[0][:, :ny_half]
        mesh = pv.RectilinearGrid(x, yh, z)
        mesh.point_data["Q"] = q.ravel(order="F")
        mesh.point_data["u"] = ux.ravel(order="F")
        surf = mesh.contour([q_level], scalars="Q")
        plane = pv.RectilinearGrid(x, yh[j_slice:j_slice + 1], z)
        plane.point_data["u"] = ux[:, j_slice:j_slice + 1].ravel(order="F")

        p = pv.Plotter(off_screen=True, window_size=list(args.size))
        p.set_background("#0d1117")
        clim = (0.3, 1.2)
        bar = {"color": "white", "vertical": False, "width": 0.25, "height": 0.035, "position_y": 0.03,
               "title_font_size": 16, "label_font_size": 13, "n_labels": 2, "fmt": "%.2f"}
        p.add_mesh(plane, scalars="u", cmap="RdBu_r", clim=(0.2, 0.9), opacity=0.85,
                   scalar_bar_args=dict(bar, title=f"u / U_b at y+ = {yh[j_slice] * u_tau / nu:.0f}",
                                        position_x=0.40))
        p.add_mesh(surf, scalars="u", cmap="viridis", clim=clim, smooth_shading=True, specular=0.3,
                   scalar_bar_args=dict(bar, title="u / U_b on Q isosurfaces", position_x=0.70))
        p.add_mesh(pv.Box(bounds=(0, L[0], 0, h, 0, L[2])).extract_feature_edges(), color="#999999", line_width=1)
        p.camera_position = [(0.6 * L[0], 7.5 * h, -2.9 * L[2]), (0.48 * L[0], -0.3 * h, 0.5 * L[2]), (0, 1, 0)]
        p.camera.zoom(1.0)
        p.add_text(f"Channel flow, Re_tau = {u_tau * h / nu:.0f}: Q isosurfaces (Q+ = {args.q_plus:g}) colored by u, "
                   f"streaks at y+ = {yh[j_slice] * u_tau / nu:.0f}", position="upper_left", font_size=12,
                   color="white")
        p.add_text(f"t u_tau / h = {t * u_tau / h:6.2f}", position="lower_left", font_size=12, color="white")
        images.append(p.screenshot(return_img=True))
        p.close()
        print(f"frame {k + 1}/{len(frames)}: {os.path.basename(path)}, t = {t:.3f}, "
              f"{surf.n_cells} isosurface triangles", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    imageio.imwrite(args.out + ".png", images[-1])
    hold = [images[-1]] * int(round(args.hold * args.fps))
    with imageio.get_writer(args.out + ".mp4", fps=args.fps, codec="libx264", quality=None,
                            pixelformat="yuv420p", ffmpeg_params=["-crf", "18", "-preset", "slow"],
                            macro_block_size=8) as w:
        for img in images + hold:
            w.append_data(img)
    small = [img[::2, ::2] for img in images + hold]
    imageio.mimsave(args.out + ".gif", small, duration=1 / args.fps, loop=0)
    print(f"wrote {args.out}.mp4, .png, .gif")


if __name__ == "__main__":
    main()
