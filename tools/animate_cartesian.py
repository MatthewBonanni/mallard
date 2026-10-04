#!/usr/bin/env python3
"""Animate a large Cartesian (generated quad mesh) run: density shaded by a
numerical schlieren, beside the schlieren itself, rendered pixel-exact.

    animate_cartesian.py extract SERIES_DIR GRID_DIR [--var RHO] [--delete] [--follow]
    animate_cartesian.py render GRID_DIR OUTPUT_STEM [--width 3840 --height 2160] [--layout side]
    animate_cartesian.py still GRID_DIR/NAME.npy OUTPUT_STEM

extract turns each snapshot of SERIES_DIR (VTU or the PVTU of a distributed
run) into GRID_DIR/NAME.npy, the cell values on the (ny, nx) grid in float32,
and appends its time to GRID_DIR/times.txt. Only snapshots listed in the .pvd
series are read, so files still being written are skipped; --delete removes
the VTU pieces afterwards and --follow keeps polling until the run's log says
it finished, which keeps the disk use of long runs bounded.

render computes every frame at the grid's resolution and area-averages it to
the panel size, so meshes far finer than the screen show no aliasing; text is
drawn by matplotlib around the pixel-exact panels. still writes the final
state at one pixel per cell (density and schlieren PNGs).
"""
import argparse
import glob
import os
import re
import shutil
import tempfile
import time
from multiprocessing import Pool

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter

from animate import write_gif, write_mp4
from mallard_vtu import grid_fields

BG = "#101014"
FG = "#e8e8e8"


def listed_snapshots(series_dir):
    """Snapshots of the .pvd series (written after each snapshot is complete)."""
    out = []
    for pvd in glob.glob(os.path.join(series_dir, "*.pvd")):
        for name in re.findall(r'file="([^"]+)"', open(pvd).read()):
            out.append(os.path.join(series_dir, name))
    return out


def extract(args):
    os.makedirs(args.grid_dir, exist_ok=True)
    times = os.path.join(args.grid_dir, "times.txt")
    while True:
        finished = args.follow and os.path.exists(args.follow) and re.search(r"^\W*Stopped", open(args.follow).read(), re.M) is not None
        for path in listed_snapshots(args.series_dir):
            stem = os.path.splitext(os.path.basename(path))[0]
            out = os.path.join(args.grid_dir, stem + ".npy")
            if os.path.exists(out) or not os.path.exists(path):
                continue
            t, xs, ys, f = grid_fields(path, [args.var])
            np.save(out + ".tmp.npy", f[args.var].T.astype(np.float32))
            os.replace(out + ".tmp.npy", out)
            with open(times, "a") as fh:
                fh.write(f"{stem} {t:.10g}\n")
            if args.delete:
                pieces = re.findall(r'Source="([^"]+)"', open(path).read()) if path.endswith(".pvtu") else []
                for p in pieces:
                    os.remove(os.path.join(os.path.dirname(path), p))
                os.remove(path)
            print(f"extracted {stem} (t = {t:.4f}, {xs.size} x {ys.size})", flush=True)
        if not args.follow or finished:
            return
        time.sleep(20)


def schlieren(rho, sigma=0.8, pct=99.7):
    smooth = gaussian_filter(rho, sigma, mode="nearest")
    gy, gx = np.gradient(smooth)
    g = np.hypot(gx, gy)
    g /= np.percentile(g[::4, ::4], pct) + 1e-30
    return np.exp(-6.0 * g)


def downsample(img, n):
    """Area average of img (ny, nx[, c]) to n x n pixels."""
    ny, nx = img.shape[:2]
    if (ny, nx) == (n, n):
        return img
    # exact box averaging over fractional pixel footprints, separable
    def weights(m):
        edges = np.linspace(0, m, n + 1)
        w = np.zeros((n, m), dtype=np.float32)
        for i in range(n):
            a, b = edges[i], edges[i + 1]
            lo, hi = int(np.floor(a)), int(np.ceil(b))
            for k in range(lo, min(hi, m)):
                w[i, k] = min(b, k + 1) - max(a, k)
            w[i] /= w[i].sum()
        return w
    wy, wx = weights(ny), weights(nx)
    if img.ndim == 2:
        return wy @ img @ wx.T
    return np.stack([wy @ img[..., c] @ wx.T for c in range(img.shape[2])], axis=-1)


def shaded_density(rho, norm, cmap, shading, shade=None):
    shade = schlieren(rho) if shade is None else shade
    rgb = plt.get_cmap(cmap)(norm(rho))[..., :3].astype(np.float32)
    rgb *= (1 - shading) + shading * shade[..., None]
    return rgb


def to_uint8(rgb):
    return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)


class Frame:
    def __init__(self, args, times, vrange):
        self.a, self.times, self.vrange = args, times, vrange

    def __call__(self, job):
        i, stem = job
        a = self.a
        out = os.path.join(a.frames_dir, f"frame_{i:05d}.png")
        if os.path.exists(out):
            return out
        rho = np.load(os.path.join(a.grid_dir, stem + ".npy")).astype(np.float32)
        norm = Normalize(*self.vrange, clip=True)
        shade = schlieren(rho)
        k = a.height / 2160
        side = a.layout == "side"
        title_h, foot_h = int(150 * k), int(110 * k)
        cb_w, gap = int(30 * k), int(110 * k)
        n = a.height - title_h - foot_h
        if side:
            n = min(n, (a.width - gap - 3 * cb_w - 2 * int(40 * k)) // 2)
        dens = to_uint8(downsample(shaded_density(rho, norm, a.cmap, a.shading, shade), n))
        fig = plt.figure(figsize=(a.width / 100, a.height / 100), dpi=100, facecolor=BG)
        if side:
            schl = downsample(shade, n)
            schl = to_uint8(plt.get_cmap("bone")(schl)[..., :3])
            total = 2 * n + gap
            x0 = (a.width - total) // 2
            panels = [(dens, x0), (schl, x0 + n + gap)]
        else:
            x0 = (a.width - n - 3 * cb_w) // 2
            panels = [(dens, x0)]
        y0 = foot_h + (a.height - title_h - foot_h - n) // 2
        for img, x in panels:
            fig.figimage(img[::-1], xo=x, yo=y0, origin="upper")
        fs = 26 * k
        W, H = a.width, a.height
        cax = fig.add_axes([(x0 + n + 0.5 * cb_w) / W, y0 / H, cb_w / W, n / H])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=a.cmap), cax=cax)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=0.8 * fs, colors="#9a9a9a", length=3 * k)
        if side:
            fig.text((x0 + n / 2) / W, (y0 + n + 18 * k) / H, "Density", ha="center", va="bottom", fontsize=fs, color=FG)
            fig.text((x0 + 1.5 * n + gap) / W, (y0 + n + 18 * k) / H, "Numerical schlieren", ha="center",
                     va="bottom", fontsize=fs, color=FG)
        else:
            cb.set_label("Density", fontsize=fs, color=FG)
        fig.text(0.5, 1 - 45 * k / H, f"{a.title}    t = {self.times[stem]:.3f}", ha="center", va="top",
                 fontsize=1.3 * fs, color=FG)
        if a.subtitle:
            fig.text(0.5, 0.45 * foot_h / H, a.subtitle, ha="center", va="center", fontsize=0.85 * fs, color="#9a9a9a")
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        imageio.imwrite(out + ".tmp.png", frame)
        os.replace(out + ".tmp.png", out)
        return out


def read_times(grid_dir):
    times = {}
    for line in open(os.path.join(grid_dir, "times.txt")):
        stem, t = line.split()
        times[stem] = float(t)
    return dict(sorted(times.items()))


def render(args):
    times = read_times(args.grid_dir)
    stems = list(times)[::-1][::args.every][::-1]
    if args.vmin is None or args.vmax is None:
        lo, hi = np.inf, -np.inf
        for s in stems[::max(1, len(stems) // 20)] + stems[-1:]:
            rho = np.load(os.path.join(args.grid_dir, s + ".npy"), mmap_mode="r")[::4, ::4]
            lo, hi = min(lo, float(rho.min())), max(hi, float(rho.max()))
        args.vmin = lo if args.vmin is None else args.vmin
        args.vmax = hi if args.vmax is None else args.vmax
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    keep = args.frames_dir is not None
    tmp = tempfile.TemporaryDirectory()
    args.frames_dir = args.frames_dir or tmp.name
    os.makedirs(args.frames_dir, exist_ok=True)
    with Pool(args.workers) as pool:
        frames = []
        for f in pool.imap(Frame(args, times, (args.vmin, args.vmax)), list(enumerate(stems))):
            frames.append(f)
            print("rendered", os.path.basename(f), flush=True)
    seq = frames + [frames[-1]] * args.fps
    for i, src in enumerate(seq):
        os.link(src, os.path.join(tmp.name, f"seq_{i:05d}.png"))
    write_mp4(os.path.join(tmp.name, "seq_%05d.png"), args.output_stem + ".mp4", args.fps)
    if args.gif_width:
        gseq = frames[::-1][::args.gif_every][::-1]
        gseq = gseq + [gseq[-1]] * args.fps
        for i, src in enumerate(gseq):
            os.link(src, os.path.join(tmp.name, f"gif_{i:05d}.png"))
        write_gif(os.path.join(tmp.name, "gif_%05d.png"), args.output_stem + ".gif", args.fps, args.gif_width)
    shutil.copy(frames[-1], args.output_stem + "_final.png")
    if not keep:
        tmp.cleanup()
    print("wrote", args.output_stem + ".mp4", flush=True)


def still(args):
    rho = np.load(args.npy).astype(np.float32)
    lo = rho.min() if args.vmin is None else args.vmin
    hi = rho.max() if args.vmax is None else args.vmax
    shade = schlieren(rho)
    imageio.imwrite(args.output_stem + "_density.png",
                    to_uint8(shaded_density(rho, Normalize(lo, hi, clip=True), args.cmap, args.shading, shade))[::-1])
    imageio.imwrite(args.output_stem + "_schlieren.png", to_uint8(shade)[::-1])
    print("wrote", args.output_stem + "_density.png", args.output_stem + "_schlieren.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("series_dir")
    e.add_argument("grid_dir")
    e.add_argument("--var", default="RHO")
    e.add_argument("--delete", action="store_true", help="Delete each snapshot once extracted")
    e.add_argument("--follow", metavar="LOG", help="Keep polling until this run log reports the end of the run")
    r = sub.add_parser("render")
    r.add_argument("grid_dir")
    r.add_argument("output_stem")
    r.add_argument("--width", type=int, default=3840)
    r.add_argument("--height", type=int, default=2160)
    r.add_argument("--layout", choices=["side", "single"], default="side",
                   help="Shaded density beside the schlieren, or the shaded density alone")
    r.add_argument("--fps", type=int, default=30)
    r.add_argument("--every", type=int, default=1, help="Use every n-th snapshot, counting back from the last")
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--frames-dir", help="Keep the PNG frames here (and skip frames already rendered)")
    r.add_argument("--gif-width", type=int, help="Also write a GIF of this width")
    r.add_argument("--gif-every", type=int, default=2)
    for p in (r,):
        p.add_argument("--title", default="2D Riemann problem, configuration 3")
        p.add_argument("--subtitle", default="")
    for p in (r, s := sub.add_parser("still")):
        p.add_argument("--cmap", default="turbo")
        p.add_argument("--shading", type=float, default=0.6, help="Schlieren shading strength of the density")
        p.add_argument("--vmin", type=float)
        p.add_argument("--vmax", type=float)
    s.add_argument("npy")
    s.add_argument("output_stem")
    args = ap.parse_args()
    {"extract": extract, "render": render, "still": still}[args.cmd](args)


if __name__ == "__main__":
    main()
