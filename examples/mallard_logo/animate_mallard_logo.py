#!/usr/bin/env python3
"""Animation of the mallard_logo example: the density and its numerical schlieren, one frame per snapshot.

    animate_mallard_logo.py OUT.mp4 SNAPSHOT_DIR [SNAPSHOT_DIR ...] [--fps 20] [--crf 27] [--poster FILE.jpg --poster-t T]
        [--png FILE] [--frames-dir DIR]

Reads every .vtkhdf snapshot of the directories and orders them by the TIME
each carries, not by file name, so the pieces of a run continued from restarts
(in several directories, or numbered again from 0) join into one sequence; a
time already seen is skipped. Prints the extremes of the density over the
sequence. The MP4 is H.264 (CRF 27, about 4 MB) at 1920 x 1080 and holds the last frame for
2 s; --poster writes the frame nearest --poster-t, --png the last.
"""
import argparse
import glob
import os
import sys
import tempfile

import h5py
import imageio.v2 as imageio
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tools", "post"))
from animate import Rasterizer, ffmpeg, pixel_to_cell, schlieren  # noqa: E402
from mallard.vtu import read_vtu_cells  # noqa: E402

BG = "#0d0f14"
FG = "#e8e8e8"
MUTED = "#aab"
BODY = "#2a2f3a"
WIDTH = 1800  # pixels across the domain


def snapshots(dirs):
    """(time, path) of every snapshot of the directories, by time, each time once."""
    seen = {}
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.vtkhdf"))):
            with h5py.File(f, "r") as h:
                g = h["VTKHDF"]
                if "Steps" in g:
                    continue  # a series file of boundary data, not a snapshot
                t = float(g["FieldData/TIME"][0])
            key = round(t, 9)
            if key not in seen:
                seen[key] = (t, f)
    return [seen[k] for k in sorted(seen)]


def density(path):
    with h5py.File(path, "r") as h:
        return h["VTKHDF/CellData/RHO"][...]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--fps", type=float, default=20)
    ap.add_argument("--crf", type=int, default=27, help="H.264 quality: lower is better and larger")
    ap.add_argument("--poster")
    ap.add_argument("--poster-t", type=float, default=3.6)
    ap.add_argument("--png")
    ap.add_argument("--frames-dir")
    args = ap.parse_args()

    shots = snapshots(args.dirs)
    print(f"{len(shots)} snapshots, t = {shots[0][0]:g} to {shots[-1][0]:g}")
    # The straight-sided triangles of the cells' corners: the fans of the curved
    # cells' outlines overlap their neighbors, which pixel lookup cannot take
    pts, conn, offs, _, _ = read_vtu_cells(shots[0][1])
    starts = np.concatenate([[0], offs[:-1]])
    tris = np.stack([conn[starts], conn[starts + 1], conn[starts + 2]], axis=1).astype(np.int64)
    pts, tri_cell = pts[:, :2], np.arange(len(tris))
    cell, valid, extent = pixel_to_cell(pts, tris, tri_cell, WIDTH)
    raster = Rasterizer(pts, tris, tri_cell, cell, valid, extent, "nearest")
    n_cells = tri_cell.max() + 1
    norm = LogNorm(0.25, 8.0)
    cmap = plt.get_cmap("cividis").copy()
    cmap.set_bad(BODY)
    gray = plt.get_cmap("gray").copy()
    gray.set_bad(BODY)

    frames = args.frames_dir or tempfile.mkdtemp()
    os.makedirs(frames, exist_ok=True)
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    lo, hi = np.inf, -np.inf
    poster = min(range(len(shots)), key=lambda i: abs(shots[i][0] - args.poster_t))
    h = (extent[3] - extent[2]) / (extent[1] - extent[0]) * 0.9 * 19.2 / 10.8
    for n, (t, f) in enumerate(shots):
        rho = density(f)
        if rho.size != n_cells:
            raise SystemExit(f"{f}: {rho.size} cells, not the {n_cells} of the first snapshot")
        lo, hi = min(lo, rho.min()), max(hi, rho.max())
        img = raster(rho)
        fig.clf()
        fig.patch.set_facecolor(BG)
        ax = fig.add_axes([0.05, 0.855 - h, 0.9, h])
        ax.imshow(img, origin="lower", extent=extent, cmap=cmap, norm=norm, interpolation="nearest")
        bx = fig.add_axes([0.05, 0.815 - 2 * h, 0.9, h])
        bx.imshow(np.where(valid, schlieren(img, valid), np.nan), origin="lower", extent=extent, cmap=gray,
                  vmin=0, vmax=1, interpolation="nearest")
        for a, label in ((ax, "density"), (bx, "numerical schlieren")):
            a.set_axis_off()
            a.text(0.0, 1.01, label, transform=a.transAxes, color=MUTED, fontsize=13, va="bottom")
        cax = fig.add_axes([0.05, 0.775 - 2 * h, 0.3, 0.012])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
        cb.set_ticks([0.25, 0.5, 1, 2, 4, 8])
        cb.set_ticklabels(["0.25", "0.5", "1", "2", "4", "8"])
        cb.ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        cb.set_label("density / ambient density", color=FG, fontsize=12)
        cb.ax.tick_params(colors=FG, labelsize=11)
        cb.outline.set_edgecolor("#555")
        fig.text(0.05, 0.945, "The Mallard logo as an initial condition", color=FG, fontsize=22, weight="bold")
        fig.text(0.05, 0.91, f"A Mach 1.5 shock through the logo painted in gas and around the logo's duck, a "
                 f"curved wall; Euler, TENO5 on {n_cells:,} cubic triangles", color=MUTED, fontsize=14)
        fig.text(0.95, 0.945, f"t = {t:4.2f}", color=FG, fontsize=16, ha="right", family="monospace")
        fig.savefig(os.path.join(frames, f"frame_{n:05d}.png"), facecolor=BG)
        if n == poster and args.poster:
            imageio.imwrite(args.poster, imageio.imread(os.path.join(frames, f"frame_{n:05d}.png"))[:, :, :3],
                            quality=90)
        if n % 25 == 0:
            print(f"frame {n + 1} / {len(shots)}", flush=True)
    last = os.path.join(frames, f"frame_{len(shots) - 1:05d}.png")
    for k in range(int(round(2 * args.fps))):
        hold = os.path.join(frames, f"frame_{len(shots) + k:05d}.png")
        if os.path.exists(hold):
            os.remove(hold)
        os.link(last, hold)
    ffmpeg("-framerate", str(args.fps), "-i", os.path.join(frames, "frame_%05d.png"), "-vf",
           "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(args.crf),
           "-preset", "slow", "-movflags", "+faststart", args.out)
    print(f"density over the sequence: {lo:.3f} to {hi:.3f}")
    print(f"wrote {args.out}" + (f" and {args.poster} (t = {shots[poster][0]:g})" if args.poster else ""))
    if args.png:
        imageio.imwrite(args.png, imageio.imread(last)[:, :, :3])


if __name__ == "__main__":
    main()
