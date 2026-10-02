#!/usr/bin/env python3
"""Render a Mallard VTU time series to MP4/GIF: density and numerical schlieren.

    animate.py SERIES_DIR OUTPUT_STEM [--var RHO] [--res 1000] [--fps 20]

Frames are rasterized by mapping every pixel to the cell containing it, which
works for any mesh (triangles, quads, mixed).
"""
import argparse
import glob
import os

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation
from scipy.ndimage import gaussian_filter

from mallard_vtu import read_vtu


def pixel_to_cell(pts, tris, tri_cell, res):
    """Cell index for every pixel of a res x res image of the bounding box."""
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    xs = np.linspace(x0, x1, res)
    ys = np.linspace(y0, y1, res)
    X, Y = np.meshgrid(xs, ys)
    tri = Triangulation(pts[:, 0], pts[:, 1], tris)
    finder = tri.get_trifinder()
    t = finder(X, Y)
    valid = t >= 0
    cell = np.zeros_like(t)
    cell[valid] = tri_cell[t[valid]]
    return cell, valid, (x0, x1, y0, y1)


def contour_levels(values, lo, hi, n=30):
    """Evenly spaced levels, nudged off plateaus: a level that coincides with a
    large nearly uniform region would draw speckles from its tiny noise."""
    levels = np.linspace(lo, hi, n)[1:-1]
    spacing = (hi - lo) / (n - 1)
    for i, level in enumerate(levels):
        if np.mean(np.abs(values - level) < 0.1 * spacing) > 0.01:
            levels[i] = level + 0.5 * spacing
    return np.unique(levels[levels < hi])


def schlieren(img, valid):
    smooth = gaussian_filter(np.where(valid, img, np.nanmean(img[valid])), 0.8)
    gy, gx = np.gradient(smooth)
    g = np.hypot(gx, gy)
    g /= np.percentile(g[valid], 99.7) + 1e-30
    return np.exp(-6.0 * g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("series_dir")
    ap.add_argument("output_stem")
    ap.add_argument("--var", default="RHO")
    ap.add_argument("--res", type=int, default=1000)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--title", default="2D Riemann problem, configuration 3")
    ap.add_argument("--subtitle", default="")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.series_dir, "*.vtu")))
    pts, tris, tri_cell, data = read_vtu(files[0])
    cell, valid, extent = pixel_to_cell(pts, tris, tri_cell, args.res)

    # Fixed color range over the whole series
    lo, hi = np.inf, -np.inf
    for f in files:
        d = read_vtu(f)[3][args.var]
        lo, hi = min(lo, d.min()), max(hi, d.max())

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": "#e8e8e8",
                         "axes.labelcolor": "#e8e8e8", "xtick.color": "#9a9a9a", "ytick.color": "#9a9a9a"})
    frames = []
    for f in files:
        d = read_vtu(f)[3]
        img = np.where(valid, d[args.var][cell], np.nan)
        fig, axs = plt.subplots(1, 2, figsize=(14, 7.2), facecolor="#101014")
        for ax in axs:
            ax.set_facecolor("#101014")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        im = axs[0].imshow(img, origin="lower", extent=extent, cmap="turbo", vmin=lo, vmax=hi,
                           interpolation="nearest")
        axs[0].contour(np.linspace(extent[0], extent[1], args.res), np.linspace(extent[2], extent[3], args.res),
                       gaussian_filter(np.nan_to_num(img, nan=lo), 0.8), levels=contour_levels(img[valid], lo, hi),
                       colors="k", linewidths=0.25, alpha=0.5)
        axs[0].set_title("Density", fontsize=13)
        cb = fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.02)
        cb.outline.set_visible(False)
        axs[1].imshow(schlieren(img, valid), origin="lower", extent=extent, cmap="bone", vmin=0, vmax=1,
                      interpolation="bilinear")
        axs[1].set_title("Numerical schlieren", fontsize=13)
        t = d.get("TIME", float("nan"))
        fig.suptitle(f"{args.title}    t = {t:.3f}", fontsize=15, y=0.97)
        if args.subtitle:
            fig.text(0.5, 0.025, args.subtitle, ha="center", fontsize=10, color="#9a9a9a")
        fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.06, wspace=0.08)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        frames.append(frame)
        print(f"rendered {os.path.basename(f)} (t = {t:.3f})", flush=True)

    # Hold the final frame for a moment
    frames += [frames[-1]] * args.fps
    imageio.mimwrite(args.output_stem + ".mp4", frames, fps=args.fps, codec="libx264", quality=9,
                     macro_block_size=8)
    small = [f[::2, ::2] for f in frames]
    imageio.mimwrite(args.output_stem + ".gif", small, duration=1000 / args.fps, loop=0)
    imageio.imwrite(args.output_stem + "_final.png", frames[-1])
    print("wrote", args.output_stem + ".mp4", args.output_stem + ".gif")


if __name__ == "__main__":
    main()
