#!/usr/bin/env python3
"""Animate the sphere wake at Re = 300: Q-criterion isosurfaces and the force history.

    animate_sphere_re300.py SOLUT_DIR FORCES.csv OUTPUT_BASE [--q 0.02] [--u 0.2]
        [--spacing 0.02] [--sigma 1.5] [--width 1920] [--fps 15] [--orbit 120]
        [--hold 2] [--gif-width 640] [--gif-every 2] [--t-start 600] [--prefix sphere]
    animate_sphere_re300.py --extract SOLUT_DIR SURFACE_DIR [--q 0.02] [--u 0.2]

SOLUT_DIR holds the volume VTU (or PVTU) series of examples/sphere_re300 with
U and Q, the Q-criterion from the solver's velocity gradients. Each snapshot
is resampled near the wake onto a uniform grid of the given spacing (in D),
smoothed by a Gaussian of sigma grid spacings, and Q is contoured there at
--q (in units of (U / D)^2, U the free-stream speed), colored by the
streamwise velocity, with the sphere, while the camera orbits the wake;
the panels on the right trace the drag and lift coefficients from t-start
(solver time, D / a) on. The tetrahedra make a piecewise-constant field
whose contours would follow their facets; the resampling and filter remove
that without moving the isosurfaces by more than about sigma * spacing.
Writes OUTPUT_BASE.mp4 (H.264, CRF 18), OUTPUT_BASE.gif (if --gif-width >
0) and a still of the last frame, OUTPUT_BASE_still.png. Needs pyvista besides
the packages in tools/README.md.

--extract only writes the isosurfaces of each snapshot (q_*.vtp and
q.pvd), which are far smaller than the volume files: run it where the
solution is, then animate SURFACE_DIR with --prefix q.
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
from PIL import Image
from scipy.ndimage import gaussian_filter

from animate import write_gif, write_mp4
from plot_sphere_re300 import REFERENCES, load

BG = "#0d1117"
FG = "#e6edf3"
MUTED = "#8b949e"
GRID = "#30363d"
BOUNDS = (-1.0, 12.0, -2.5, 2.5, -2.5, 2.5)
R_SHEATH = 0.6


def snapshot_files(solut, prefix):
    text = open(os.path.join(solut, prefix + ".pvd")).read()
    entries = re.findall(r'timestep="([^"]+)"[^>]*file="([^"]+)"', text)
    return [(float(t), os.path.join(solut, f)) for t, f in entries if os.path.exists(os.path.join(solut, f))]


def q_surface(path, q_level, spacing, sigma):
    mesh = pv.read(path)
    if path.endswith(".vtp"):
        return mesh
    if "Q" not in mesh.cell_data:
        raise SystemExit(f"{path} has no Q: add \"Q\" to the variables of [[write_data]]")
    lo, hi = np.array(BOUNDS[0::2]), np.array(BOUNDS[1::2])
    margin = 4 * (sigma + 1) * spacing
    c = mesh.cell_centers().points
    near = np.all((c >= lo - margin) & (c <= hi + margin), axis=1)
    mesh = mesh.extract_cells(np.flatnonzero(near))
    for name in list(mesh.cell_data.keys()):
        if name not in ("Q", "U"):
            del mesh.cell_data[name]
    # The pieces of a parallel output repeat the points they share
    mesh = mesh.clean().cell_data_to_point_data(pass_cell_data=False)
    n = np.round((hi - lo) / spacing).astype(int) + 1
    grid = pv.ImageData(dimensions=n, spacing=(spacing,) * 3, origin=lo)
    grid = grid.sample(mesh)
    shape = tuple(n[::-1])  # ImageData points run fastest in x
    valid = np.asarray(grid.point_data["vtkValidPointMask"]).reshape(shape) > 0
    q = np.where(valid, np.asarray(grid.point_data["Q"]).reshape(shape), 0.0)
    ux = np.where(valid, np.asarray(grid.point_data["U"])[:, 0].reshape(shape), 0.0)
    # The boundary layer carries Q ~ 0 of either sign: leave it out so that it
    # does not wrap the sphere in a sheath
    r = np.linalg.norm(np.asarray(grid.points), axis=1).reshape(shape)
    q[r < R_SHEATH] = min(q_level, 0.0) - 1.0
    if sigma > 0:
        q = gaussian_filter(q, sigma)
        ux = gaussian_filter(ux, sigma)
    out = pv.ImageData(dimensions=n, spacing=(spacing,) * 3, origin=lo)
    out.point_data["Q"] = q.ravel()
    out.point_data["UX"] = ux.ravel()
    surf = out.contour([q_level], scalars="Q")
    for name in list(surf.point_data.keys()):
        if name != "UX":
            del surf.point_data[name]
    return surf


def extract(solut, out, prefix, q_level, spacing, sigma):
    os.makedirs(out, exist_ok=True)
    entries = []
    for k, (t, path) in enumerate(snapshot_files(solut, prefix)):
        name = f"q_{k:05d}.vtp"
        q_surface(path, q_level, spacing, sigma).save(os.path.join(out, name))
        entries.append(f'    <DataSet timestep="{t:.10g}" file="{name}"/>')
        print(f"{name}: t = {t:.4g}", flush=True)
    with open(os.path.join(out, "q.pvd"), "w") as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n  <Collection>\n'
                + "\n".join(entries) + "\n  </Collection>\n</VTKFile>\n")


def render_3d(plotter, surf, u, angle):
    plotter.clear_actors()
    if surf.n_points > 0:
        plotter.add_mesh(surf, scalars="UX", cmap="RdYlBu_r", clim=(-0.2 * u, 1.2 * u), smooth_shading=True,
                         specular=0.35, specular_power=20, show_scalar_bar=False)
    plotter.add_mesh(pv.Sphere(radius=0.5, theta_resolution=96, phi_resolution=48), color="#c9d1d9",
                     smooth_shading=True, specular=0.5)
    focus = np.array([4.5, 0.0, 0.0])
    a = np.radians(angle)
    r = 17.0
    eye = focus + r * np.array([-0.25, 0.97 * np.sin(a), 0.97 * np.cos(a)])
    up = np.array([0.0, np.cos(a), -np.sin(a)])
    plotter.camera_position = [tuple(eye), tuple(focus), tuple(up)]
    plotter.camera.view_angle = 32
    plotter.reset_camera_clipping_range()
    img = plotter.screenshot(return_img=True)
    size = tuple(plotter.window_size)
    if img.shape[1::-1] != size:  # High-DPI displays render at a multiple of the window size
        img = np.asarray(Image.fromarray(img).resize(size, Image.LANCZOS))
    return img


def render_panels(t_now, tu, cd, cl, t_start, size, dpi, title_lines):
    fig = plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi, facecolor=BG)
    fig.text(0.08, 0.94, title_lines[0], color=FG, fontsize=19, weight="bold")
    fig.text(0.08, 0.905, title_lines[1], color=MUTED, fontsize=12)
    fig.text(0.08, 0.875, f"t U / D = {t_now:6.1f}", color=FG, fontsize=15, family="monospace")
    m = (tu >= t_start) & (tu <= t_now)
    window = tu >= t_start
    for k, (y, name, ref, ylim) in enumerate([
            (cd, "$C_D$", REFERENCES[0][1], (0.62, 0.70)), (cl, "$C_L$", REFERENCES[0][2], (0.0, 0.14))]):
        ax = fig.add_axes([0.18, 0.49 - 0.41 * k, 0.76, 0.32], facecolor=BG)
        ax.axhline(ref, color=MUTED, lw=1.6, ls="--", label="Johnson & Patel 1999, mean")
        ax.plot(tu[m], y[m], color="#ff9e3d", lw=2.2, label="Mallard")
        if m.any():
            ax.plot(tu[m][-1], y[m][-1], "o", color="#ff9e3d", ms=8)
        ax.set_xlim(tu[window][0], tu[window][-1])
        ax.set_ylim(*ylim)
        ax.set_ylabel(name, color=FG, fontsize=16)
        ax.tick_params(colors=FG, labelsize=11)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.grid(color=GRID, lw=0.8)
        if k == 0:
            ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=11, loc="upper right")
        else:
            ax.set_xlabel("$t\\,U/D$", color=FG, fontsize=15)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", nargs=2, metavar=("SOLUT_DIR", "SURFACE_DIR"),
                    help="Only write the isosurfaces of every snapshot")
    ap.add_argument("solut", nargs="?")
    ap.add_argument("forces", nargs="?")
    ap.add_argument("output", nargs="?")
    ap.add_argument("--prefix", default="sphere")
    ap.add_argument("--q", type=float, default=0.02, help="Isosurface level of Q, in (U / D)^2")
    ap.add_argument("--u", type=float, default=0.2, help="Free-stream speed")
    ap.add_argument("--spacing", type=float, default=0.02, help="Sampling grid spacing, in D")
    ap.add_argument("--sigma", type=float, default=1.5, help="Gaussian filter width, in grid spacings")
    ap.add_argument("--t-start", type=float, default=600.0, help="Start of the force panels (solver time)")
    ap.add_argument("--subtitle", help="Default: the isosurface level and coloring")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--orbit", type=float, default=120.0, help="Camera orbit about the stream axis, degrees")
    ap.add_argument("--hold", type=float, default=2.0, help="Seconds to hold the last frame")
    ap.add_argument("--gif-width", type=int, default=640)
    ap.add_argument("--gif-every", type=int, default=2, help="Use every n-th frame in the GIF")
    args = ap.parse_args()
    if args.extract:
        extract(*args.extract, args.prefix, args.q * args.u ** 2, args.spacing, args.sigma)
        return
    if not args.output:
        ap.error("SOLUT_DIR, FORCES.csv and OUTPUT_BASE are required")

    files = snapshot_files(args.solut, args.prefix)
    t, C = load(args.forces, args.u)
    tu = t * args.u
    direction = C[t >= args.t_start, 1:].mean(0)
    direction /= np.linalg.norm(direction)
    cl = C[:, 1:] @ direction

    W, H = args.width, round(args.width * 9 / 16)
    w3d = round(W * 0.64)
    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True, window_size=(w3d, H))
    plotter.set_background(BG)
    plotter.enable_anti_aliasing("ssaa")
    subtitle = args.subtitle or f"Isosurfaces Q = {args.q:g} (U/D)$^2$, colored by streamwise velocity"
    title = ("Sphere at Re = 300, M = 0.2", subtitle)
    with tempfile.TemporaryDirectory() as tmp:
        for k, (t_file, path) in enumerate(files):
            surf = q_surface(path, args.q * args.u ** 2, args.spacing, args.sigma)
            angle = args.orbit * k / max(len(files) - 1, 1)
            left = render_3d(plotter, surf, args.u, angle)
            right = render_panels(t_file * args.u, tu, C[:, 0], cl, args.t_start * args.u, (W - w3d, H), 100, title)
            frame = np.concatenate([left[:H, :w3d], right[:H]], axis=1)
            imageio.imwrite(os.path.join(tmp, f"f{k:05d}.png"), frame)
            if k % args.gif_every == 0:
                imageio.imwrite(os.path.join(tmp, f"g{k // args.gif_every:05d}.png"), frame)
            print(f"frame {k + 1}/{len(files)} t U / D = {t_file * args.u:.1f}", flush=True)
        n_gif = len(glob.glob(os.path.join(tmp, "g*.png")))
        for j in range(round(args.hold * args.fps)):
            imageio.imwrite(os.path.join(tmp, f"f{len(files) + j:05d}.png"), frame)
        for j in range(round(args.hold * args.fps / args.gif_every)):
            imageio.imwrite(os.path.join(tmp, f"g{n_gif + j:05d}.png"), frame)
        write_mp4(os.path.join(tmp, "f%05d.png"), args.output + ".mp4", args.fps)
        if args.gif_width > 0:
            write_gif(os.path.join(tmp, "g%05d.png"), args.output + ".gif", args.fps / args.gif_every,
                      args.gif_width, colors=128)
    imageio.imwrite(args.output + "_still.png", frame)
    print(f"wrote {args.output}.mp4")


if __name__ == "__main__":
    main()
