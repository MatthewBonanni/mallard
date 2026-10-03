"""Animation of a 2D cellular detonation run (examples/detonation_2d).

    python tools/animate_detonation_2d.py SERIES.pvd OUT.mp4 [--window 0.16] [--fps 15]
        [--gif OUT.gif] [--png OUT.png] [--title TEXT]

Each frame (1920x1080) shows, in a window that follows the front: the
pressure field and the numerical soot foil (P_MAX, the largest
pressure each cell has seen) building up behind the front; below them, the
whole foil so far with the window marked. The run must write P and P_MAX on
a generated "cartesian" mesh. The MP4 is H.264 (CRF 18) and holds the last
frame for 2 s; --gif writes a small GIF of the same frames, --png the last
frame.
"""
import argparse
import os
import re
import sys

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mallard_vtu import grid_fields  # noqa: E402
from soot_foil import front_position  # noqa: E402

BG = "#0d0f14"
FG = "#e8e8e8"


def soot(foil, p0):
    """Soot-foil shading: log of P_MAX relative to its mean across the channel at each x."""
    f = np.log(np.maximum(foil, p0))
    return (f - f.mean(axis=1, keepdims=True)).T


def render(fig, t, x, y, p, foil, p0, x_front, x_lo_all, args, D):
    fig.clf()
    fig.patch.set_facecolor(BG)
    W = (y[-1] + 0.5 * (y[1] - y[0])) * 1e3
    lo = max(x_front + 0.01 - args.window, x[0])
    hi = lo + args.window
    win = (x >= lo) & (x <= hi)
    ext = [lo * 1e3, hi * 1e3, 0, W]
    lp = (p[win] / p0).T
    sf = soot(foil[win], p0)
    seen = x[win] <= x_front
    sf[:, ~seen] = np.nan
    left, width = 0.045, 0.91
    ax1 = fig.add_axes([left, 0.585, width, 0.30])
    ax2 = fig.add_axes([left, 0.265, width, 0.30])
    ax3 = fig.add_axes([left, 0.06, width, 0.13])
    cm_soot = matplotlib.colormaps["gray_r"].copy()
    cm_soot.set_bad(BG)
    im1 = ax1.imshow(lp, origin="lower", extent=ext, cmap="inferno", vmin=args.p_min, vmax=args.p_max,
                     aspect="auto", interpolation="bilinear")
    ax2.imshow(sf, origin="lower", extent=ext, cmap=cm_soot, vmin=args.foil_lo, vmax=args.foil_hi,
               aspect="auto", interpolation="bilinear")
    full = (x >= x_lo_all) & (x <= x_front)
    fsf = soot(foil[full], p0)
    ax3.imshow(fsf, origin="lower", extent=[x[full][0] * 1e3, x[full][-1] * 1e3, 0, W], cmap=cm_soot,
               vmin=args.foil_lo, vmax=args.foil_hi, aspect="auto", interpolation="bilinear")
    ax3.set_xlim(x_lo_all * 1e3, x[-1] * 1e3)
    ax3.set_facecolor(BG)
    ax3.add_patch(Rectangle((lo * 1e3, 0), args.window * 1e3, W, fill=False, ec="#4fc3f7", lw=1.5))
    for ax in (ax1, ax2, ax3):
        ax.tick_params(colors=FG, labelsize=11)
        for s in ax.spines.values():
            s.set_color("#555")
        ax.set_ylabel("y [mm]", color=FG, fontsize=12)
    ax1.set_xticklabels([])
    ax3.set_xlabel("x [mm]", color=FG, fontsize=12)
    fig.text(left, 0.895, "Pressure p / p0", color=FG, fontsize=14)
    fig.text(left + width, 0.895, f"t = {t * 1e6:6.1f} us    front speed {D:6.0f} m/s", color=FG,
             fontsize=14, ha="right", family="monospace")
    fig.text(left, 0.205, "Numerical soot foil (peak pressure of each cell): the whole run so far", color=FG,
             fontsize=13)
    ax2.text(0.01, 0.92, "Soot foil", transform=ax2.transAxes, color="#222", fontsize=13,
             bbox=dict(fc="white", ec="none", alpha=0.7))
    fig.text(0.5, 0.955, args.title, color=FG, fontsize=20, ha="center", weight="bold")
    fig.text(0.5, 0.925, args.subtitle, color="#aab", fontsize=13, ha="center")
    cax = fig.add_axes([left + width + 0.005, 0.585, 0.008, 0.30])
    cb = fig.colorbar(im1, cax=cax)
    cb.ax.tick_params(colors=FG, labelsize=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pvd")
    ap.add_argument("out")
    ap.add_argument("--window", type=float, default=0.16)
    ap.add_argument("--fps", type=float, default=15)
    ap.add_argument("--gif")
    ap.add_argument("--png")
    ap.add_argument("--p-min", type=float, default=12.0)
    ap.add_argument("--p-max", type=float, default=34.0)
    ap.add_argument("--foil-lo", type=float, default=-0.12)
    ap.add_argument("--foil-hi", type=float, default=0.2)
    ap.add_argument("--title", default="Cellular detonation in 2H2-O2-7Ar at 6.67 kPa")
    ap.add_argument("--subtitle", default="Mallard: MUSCL-HLLC, finite-rate H2/O2 chemistry "
                                          "(10 species, 29 reactions), 1.6M cells")
    args = ap.parse_args()
    base = os.path.dirname(args.pvd)
    files = re.findall(r'file="([^"]+)"', open(args.pvd).read())
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=None,
                                ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p", "-preset", "slow"],
                                macro_block_size=8)
    gif_frames = []
    p0, x_lo_all, prev = None, None, None
    frame = None
    for k, f in enumerate(files):
        t, x, y, fld = grid_fields(os.path.join(base, f), ["P", "P_MAX"])
        p = fld["P"]
        p0 = p.min() if p0 is None else p0
        xf = front_position(x, p, p0)
        if x_lo_all is None:
            x_lo_all = xf - 0.01
        D = (xf - prev[1]) / (t - prev[0]) if prev else float("nan")
        prev = (t, xf)
        render(fig, t, x, y, p, fld["P_MAX"], p0, xf, x_lo_all, args, D)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        writer.append_data(frame)
        if args.gif and k % 2 == 0:
            gif_frames.append(frame[::3, ::3])
        print(f"frame {k + 1}/{len(files)}: t = {t * 1e6:.1f} us, front at {xf * 1e3:.1f} mm", flush=True)
    for _ in range(int(round(2 * args.fps))):
        writer.append_data(frame)
    writer.close()
    if args.png:
        imageio.imwrite(args.png, frame)
    if args.gif:
        imageio.mimsave(args.gif, gif_frames + [gif_frames[-1]] * 10, duration=1000 / 10, loop=0)


if __name__ == "__main__":
    main()
