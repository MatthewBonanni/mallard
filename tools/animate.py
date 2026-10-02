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


def pixel_to_cell(pts, tris, tri_cell, res, xlim=None, ylim=None):
    """Cell index for every pixel of an image of the bounding box whose longer
    side has res pixels."""
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    if xlim:
        x0, x1 = xlim
    if ylim:
        y0, y1 = ylim
    scale = res / max(x1 - x0, y1 - y0)
    xs = np.linspace(x0, x1, int(round((x1 - x0) * scale)))
    ys = np.linspace(y0, y1, int(round((y1 - y0) * scale)))
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


class Rasterizer:
    """Maps cell data to an image. 'nearest' shows each pixel's cell value;
    'linear' averages cells to nodes (area-weighted) and interpolates linearly
    inside each triangle, which keeps gradients (vorticity, schlieren) free of
    cell-edge artifacts on coarse meshes."""

    def __init__(self, pts, tris, tri_cell, cell, valid, extent, mode):
        self.mode, self.cell, self.valid, self.tris = mode, cell, valid, tris
        if mode != "linear":
            return
        n_cells = tri_cell.max() + 1
        p = pts[tris]
        area = 0.5 * np.abs((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) -
                            (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
        self.tri_cell, self.tri_area, self.n_nodes = tri_cell, area, len(pts)
        self.node_weight = np.zeros(len(pts))
        np.add.at(self.node_weight, tris.ravel(), np.repeat(area, 3))
        # Pixel -> triangle and barycentric weights
        tri = Triangulation(pts[:, 0], pts[:, 1], tris)
        ny, nx = cell.shape
        X, Y = np.meshgrid(np.linspace(extent[0], extent[1], nx), np.linspace(extent[2], extent[3], ny))
        t = tri.get_trifinder()(X, Y)
        self.pix_tri = np.where(t >= 0, t, 0)
        a, b, c = pts[tris[self.pix_tri, 0]], pts[tris[self.pix_tri, 1]], pts[tris[self.pix_tri, 2]]
        det = (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (c[..., 0] - a[..., 0]) * (b[..., 1] - a[..., 1])
        l1 = ((X - a[..., 0]) * (c[..., 1] - a[..., 1]) - (c[..., 0] - a[..., 0]) * (Y - a[..., 1])) / det
        l2 = ((b[..., 0] - a[..., 0]) * (Y - a[..., 1]) - (X - a[..., 0]) * (b[..., 1] - a[..., 1])) / det
        self.lam = np.stack([1 - l1 - l2, l1, l2], axis=-1)

    def __call__(self, cell_values):
        if self.mode != "linear":
            return np.where(self.valid, cell_values[self.cell], np.nan)
        node = np.zeros(self.n_nodes)
        np.add.at(node, self.tris.ravel(), np.repeat(cell_values[self.tri_cell] * self.tri_area, 3))
        node /= np.maximum(self.node_weight, 1e-300)
        img = np.einsum("ijk,ijk->ij", node[self.tris[self.pix_tri]], self.lam)
        return np.where(self.valid, img, np.nan)


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
    ap.add_argument("--xlim", type=float, nargs=2, default=None, help="Crop the x range")
    ap.add_argument("--ylim", type=float, nargs=2, default=None, help="Crop the y range")
    ap.add_argument("--every", type=int, default=1, help="Use every n-th snapshot")
    ap.add_argument("--second", default="schlieren",
                    help="Second panel: schlieren (of density) or a variable name, e.g. P")
    ap.add_argument("--interp", choices=["nearest", "linear"], default="nearest",
                    help="Pixel values: the cell's value, or linear interpolation through node averages")
    ap.add_argument("--cmap", default="turbo")
    ap.add_argument("--glob", default="*.vtu", help="Snapshot file pattern within the series directory")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.series_dir, args.glob)))[::args.every]
    pts, tris, tri_cell, data = read_vtu(files[0])
    cell, valid, extent = pixel_to_cell(pts, tris, tri_cell, args.res, args.xlim, args.ylim)
    raster = Rasterizer(pts, tris, tri_cell, cell, valid, extent, args.interp)
    aspect = (extent[1] - extent[0]) / (extent[3] - extent[2])
    vertical = aspect > 1.8

    dx = (extent[1] - extent[0]) / (cell.shape[1] - 1)
    dy = (extent[3] - extent[2]) / (cell.shape[0] - 1)

    def field(d):
        """Image of the requested variable; VORTICITY is derived from U_X, U_Y."""
        if args.var == "VORTICITY":
            u = gaussian_filter(np.nan_to_num(raster(d["U_X"])), 1.0)
            v = gaussian_filter(np.nan_to_num(raster(d["U_Y"])), 1.0)
            w = np.gradient(v, dx, axis=1) - np.gradient(u, dy, axis=0)
            return np.where(valid, w, np.nan)
        return raster(d[args.var])

    # Fixed color range over the whole series (within the displayed region)
    lo, hi = np.inf, -np.inf
    for f in files:
        img = field(read_vtu(f)[3])
        if args.var == "VORTICITY":
            m = np.nanpercentile(np.abs(img), 99.5)
            lo, hi = min(lo, -m), max(hi, m)
        else:
            lo, hi = min(lo, np.nanmin(img)), max(hi, np.nanmax(img))
    if args.var == "VORTICITY":
        lo, hi = -max(-lo, hi), max(-lo, hi)
    second_range = None
    if args.second != "schlieren":
        vals = [raster(read_vtu(f)[3][args.second]) for f in files[len(files) // 2:]]
        second_range = (np.nanpercentile(vals, 0.5), np.nanpercentile(vals, 99.5))

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": "#e8e8e8",
                         "axes.labelcolor": "#e8e8e8", "xtick.color": "#9a9a9a", "ytick.color": "#9a9a9a"})
    frames = []
    for f in files:
        d = read_vtu(f)[3]
        img = field(d)
        if vertical:
            fig, axs = plt.subplots(2, 1, figsize=(14, 2 * 13 / aspect + 1.6), facecolor="#101014")
        else:
            fig, axs = plt.subplots(1, 2, figsize=(14, 7.2), facecolor="#101014")
        for ax in axs:
            ax.set_facecolor("#101014")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        im = axs[0].imshow(img, origin="lower", extent=extent, cmap=args.cmap, vmin=lo, vmax=hi,
                           interpolation="nearest")
        if args.var != "VORTICITY":
            axs[0].contour(np.linspace(extent[0], extent[1], img.shape[1]), np.linspace(extent[2], extent[3], img.shape[0]),
                           gaussian_filter(np.nan_to_num(img, nan=lo), 0.8), levels=contour_levels(img[valid], lo, hi),
                           colors="k", linewidths=0.25, alpha=0.5)
        axs[0].set_title({"RHO": "Density", "P": "Pressure", "VORTICITY": "Vorticity"}.get(args.var, args.var), fontsize=13)
        cb = fig.colorbar(im, ax=axs[0], fraction=0.046 if not vertical else 0.015, pad=0.01)
        cb.outline.set_visible(False)
        if args.second == "schlieren":
            rho_img = raster(d["RHO"]) if "RHO" in d else img
            axs[1].imshow(schlieren(rho_img, valid), origin="lower", extent=extent, cmap="bone", vmin=0, vmax=1,
                          interpolation="bilinear")
            axs[1].set_title("Numerical schlieren", fontsize=13)
        else:
            second = raster(d[args.second])
            im2 = axs[1].imshow(second, origin="lower", extent=extent, cmap="cividis",
                                vmin=second_range[0], vmax=second_range[1], interpolation="bilinear")
            axs[1].set_title({"P": "Pressure", "RHO": "Density"}.get(args.second, args.second), fontsize=13)
            cb2 = fig.colorbar(im2, ax=axs[1], fraction=0.046 if not vertical else 0.015, pad=0.01)
            cb2.outline.set_visible(False)
        t = d.get("TIME", float("nan"))
        fig.suptitle(f"{args.title}    t = {t:.3f}", fontsize=15, y=0.97)
        if args.subtitle:
            fig.text(0.5, 0.025, args.subtitle, ha="center", fontsize=10, color="#9a9a9a")
        if vertical:
            fig.subplots_adjust(left=0.02, right=0.93, top=0.9, bottom=0.06, hspace=0.18)
        else:
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
