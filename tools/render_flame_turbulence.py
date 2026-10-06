#!/usr/bin/env python3
"""Render the flame in decaying turbulence (examples/flame_turbulence): flame surface and vortices.

    render_flame_turbulence.py FLAME_NPZ [FLAME_NPZ ...] --out BASE [--labels A,B] [--q-factor 1.0]
        [--y-u 0.0285223875] [--width 1600] [--movie DIR_OR_GLOB --fps 8]

FLAME_NPZ are snapshots written by tools/flame_turbulence.py extract (T,
Y_H2, Q, HRR on the cell lattice). Each panel shows the flame surface, the
isosurface c = 1 - Y_H2 / Y_H2,u = 0.5 colored by the heat release rate, and
the vortices of the fresh gas, isosurfaces Q = q_factor <Q+> (the mean of the
positive Q where c < 0.05), in gray. Several snapshots are drawn side by side
(e.g. the DNS and the LES at the same time) in one still, BASE.png; with
--movie, each snapshot matching the glob (sorted) becomes a frame of
BASE.mp4 and BASE.gif. Needs pyvista.
"""
import argparse
import glob
import os
import tempfile

import numpy as np


def scene(plotter, path, q_factor, y_u, label, clim, args_window=None, crop=None):
    import pyvista as pv
    d = dict(np.load(path))
    h = float(d["h"])
    if crop:
        # The last crop metres along x (the DNS box at the fresh-gas end of an extended LES box)
        keep = int(round(crop / h))
        for k, v in d.items():
            if np.ndim(v) == 3:
                d[k] = v[-keep:]
    c = np.clip(1.0 - d["Y_H2"] / y_u, 0.0, 1.0)
    grid = pv.ImageData(dimensions=np.array(c.shape) + 1, spacing=(h * 1e3,) * 3)
    grid.cell_data["c"] = c.ravel(order="F")
    grid.cell_data["HRR"] = np.log10(np.maximum(d["HRR"], 1.0)).ravel(order="F")
    grid.cell_data["Q"] = d["Q"].ravel(order="F")
    pts = grid.cell_data_to_point_data()
    flame = pts.contour([0.5], scalars="c")
    if flame.n_points:
        plotter.add_mesh(flame, scalars="HRR", cmap="inferno", clim=clim, smooth_shading=True,
                         scalar_bar_args={"title": "log10 HRR [W/m^3]"})
    fresh = (c < 0.05) & (d["Q"] > 0)
    if fresh.any():
        level = q_factor * float(d["Q"][fresh].mean())
        vort = pts.contour([level], scalars="Q")
        if vort.n_points:
            cc = vort.point_data["c"] if "c" in vort.point_data else None
            if cc is not None:
                vort = vort.extract_points(cc < 0.05, adjacent_cells=False) if (cc < 0.05).any() else vort
            plotter.add_mesh(vort, color="lightsteelblue", opacity=0.35, smooth_shading=True)
    plotter.add_mesh(grid.outline(), color="gray")
    plotter.add_text(f"{label}  t = {float(d['t']) * 1e3:.3f} ms", font_size=10, color="black")
    # x (propagation, burnt gas on the left) across the image, viewed from slightly above and in front
    lx, ly, lz = (np.array(c.shape) * h * 1e3)
    center = np.array([lx, ly, lz]) / 2
    if args_window is not None:
        center[0] = args_window
    plotter.camera_position = [tuple(center + np.array([1.9 * ly, -1.9 * ly, 1.3 * lz])), tuple(center), (0, 0, 1)]
    plotter.camera.zoom(1.0)


def render(paths, labels, out, args):
    import pyvista as pv
    pv.OFF_SCREEN = True
    n = len(paths)
    p = pv.Plotter(shape=(1, n), off_screen=True, window_size=(args.width, int(args.width * 0.55 / max(1, n - 1 or 1))))
    p.set_background("white")
    for i, (path, label) in enumerate(zip(paths, labels)):
        p.subplot(0, i)
        scene(p, path, args.q_factor, args.y_u, label, (args.hrr_min, args.hrr_max), crop=args.crop)
    p.screenshot(out)
    p.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshots", nargs="*")
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels")
    ap.add_argument("--q-factor", type=float, default=1.0)
    ap.add_argument("--y-u", type=float, default=2.85223875e-02)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--hrr-min", type=float, default=9.0)
    ap.add_argument("--hrr-max", type=float, default=10.8)
    ap.add_argument("--movie", help="glob of the snapshots of one run, one frame each")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--crop", type=float, help="draw only the last CROP metres along x")
    args = ap.parse_args()
    if args.snapshots:
        labels = args.labels.split(",") if args.labels else [os.path.basename(s) for s in args.snapshots]
        render(args.snapshots, labels, args.out + ".png", args)
        print(f"wrote {args.out}.png")
    if args.movie:
        import imageio.v2 as imageio
        frames = sorted(glob.glob(args.movie))
        label = args.labels.split(",")[0] if args.labels else ""
        with tempfile.TemporaryDirectory() as tmp:
            images = []
            for i, f in enumerate(frames):
                png = os.path.join(tmp, f"{i:04d}.png")
                render([f], [label], png, args)
                images.append(imageio.imread(png))
            imageio.mimwrite(args.out + ".mp4", images, fps=args.fps, codec="libx264", quality=8)
            imageio.mimwrite(args.out + ".gif", [im[::2, ::2] for im in images], duration=1000 // args.fps, loop=0)
        print(f"wrote {args.out}.mp4, {args.out}.gif ({len(frames)} frames)")


if __name__ == "__main__":
    main()
