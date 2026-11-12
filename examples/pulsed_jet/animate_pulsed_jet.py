#!/usr/bin/env python3
"""Animation of the pulsed-jet example (examples/pulsed_jet): one frame per snapshot.

    animate_pulsed_jet.py PREFIX PROBE OUT.mp4 [--fps 15] [--png FILE] [--poster FILE.jpg] [--subtitle TEXT]

PREFIX is the snapshots' prefix (solut/jet), PROBE the inlet probe
(solut/probe_inlet.csv). On top, the azimuthal vorticity mirrored about the
axis over the whole domain, with each ring numbered as
examples/pulsed_jet/plot_pulsed_jet.py tracks it; below, the rings' core
positions and the inlet cell's velocity against U(t), drawn up to the frame's
time. The MP4 is H.264 (CRF 18) at 1920 x 1080 and holds the last frame for
2 s; --png and --poster write the last frame.
"""
import argparse
import os
import sys

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_pulsed_jet import D, U, U0, grid, read_index, track  # noqa: E402

BG = "#0d0f14"
FG = "#e8e8e8"
MUTED = "#aab"
COLORS = ("#64b5f6", "#ffb74d", "#4db6ac", "#ce93d8", "#aed581", "#ef9a9a")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("probe")
    ap.add_argument("out")
    ap.add_argument("--fps", type=float, default=15)
    ap.add_argument("--png")
    ap.add_argument("--poster")
    ap.add_argument("--subtitle", help="the line under the title (default: the mesh)")
    args = ap.parse_args()

    files = [f for _, f in read_index(args.prefix)]
    x, y, w, _ = grid(files[-1])
    threshold = 0.25 * w[(x > 1.0) & (x < 9.0)].max()
    dA = (x[0, 1] - x[0, 0]) * (y[1, 0] - y[0, 0])
    tracks = {k: np.array(rows) for k, rows in track(files, threshold, dA).items()}
    lim = 0.5 * w.max()
    subtitle = args.subtitle or (f"Axisymmetric, Mach 0.2; a stroke of L / D = 3 per pulse; {w.size:,} quads "
                                 f"(D / {D / (x[0, 1] - x[0, 0]):.0f})")
    probe = np.genfromtxt(args.probe, delimiter=",", names=True)
    tp, up = probe["t"], probe["U_X"]
    t_end = tp[-1]
    tf = np.linspace(0.0, t_end, 2000)
    extent = [x[0, 0] - 0.5 * (x[0, 1] - x[0, 0]), x[0, -1] + 0.5 * (x[0, 1] - x[0, 0]), -y[-1, 0], y[-1, 0]]

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=None,
                                ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p", "-preset", "slow"],
                                macro_block_size=8)
    frame = None
    for n, f in enumerate(files):
        x, y, w, t = grid(f)
        fig.clf()
        fig.patch.set_facecolor(BG)
        ax = fig.add_axes([0.045, 0.43, 0.89, 0.46])
        im = ax.imshow(np.vstack([-w[::-1], w]), origin="lower", extent=extent, cmap="RdBu_r", vmin=-lim, vmax=lim,
                       interpolation="bilinear")
        ax.set_ylim(-2.0, 2.0)
        ax.axvline(9.0, color="#666", lw=1.0, ls="--")
        ax.text(10.5, 1.8, "sponge", color="#666", fontsize=13, ha="center", va="top")
        for k, r in tracks.items():
            now = r[np.isclose(r[:, 0], t)]
            if now.size:
                ax.text(now[0, 1], now[0, 2] + 0.45, str(k + 1), color=COLORS[k % len(COLORS)], fontsize=15,
                        ha="center", va="bottom", weight="bold")
        ax.set_xlabel("x / D", color=FG, fontsize=13)
        ax.set_ylabel("r / D", color=FG, fontsize=13)
        cax = fig.add_axes([0.945, 0.43, 0.01, 0.46])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(r"azimuthal vorticity $\omega_\theta D / U_0$", color=FG, fontsize=12)
        cb.ax.tick_params(colors=FG)
        cb.outline.set_edgecolor("#555")
        cb.set_ticks(np.linspace(-lim, lim, 5))
        cb.set_ticklabels([f"{v * D / U0:.1f}" for v in np.linspace(-lim, lim, 5)])

        axr = fig.add_axes([0.045, 0.075, 0.42, 0.27])
        for k, r in tracks.items():
            sel = r[:, 0] <= t + 1e-9
            if sel.any():
                c = COLORS[k % len(COLORS)]
                axr.plot(r[sel, 0] * U0 / D, r[sel, 1] / D, color=c, lw=2, label=f"ring {k + 1}")
                if np.isclose(r[sel, 0][-1], t):
                    axr.plot(r[sel, 0][-1] * U0 / D, r[sel, 1][-1] / D, "o", color=c, ms=6)
        axr.set_xlim(0.0, t_end * U0 / D)
        axr.set_ylim(0.0, 9.5)
        axr.set_xlabel(r"$t\,U_0 / D$", color=FG, fontsize=13)
        axr.set_ylabel(r"core position $x / D$", color=FG, fontsize=13)
        if axr.lines:
            axr.legend(loc="upper left", facecolor=BG, edgecolor="#555", labelcolor=FG, fontsize=11, ncol=2)

        axu = fig.add_axes([0.53, 0.075, 0.42, 0.27])
        axu.plot(tf * U0 / D, U(tf) / U0, color="#777", lw=1.2, label=r"$U(t)$ (target)")
        sel = tp <= t + 1e-9
        axu.plot(tp[sel] * U0 / D, up[sel] / U0, color=COLORS[0], lw=2, label="inlet cell")
        axu.set_xlim(0.0, t_end * U0 / D)
        axu.set_ylim(-0.05, 1.3)
        axu.set_xlabel(r"$t\,U_0 / D$", color=FG, fontsize=13)
        axu.set_ylabel(r"$u / U_0$ at the nozzle", color=FG, fontsize=13)
        axu.legend(loc="upper right", facecolor=BG, edgecolor="#555", labelcolor=FG, fontsize=11, ncol=2)
        for a in (ax, axr, axu):
            a.set_facecolor(BG)
            a.tick_params(colors=FG, labelsize=11)
            for s in a.spines.values():
                s.set_color("#555")

        fig.text(0.045, 0.955, "Pulsed jet launching a train of vortex rings", color=FG, fontsize=20, weight="bold")
        fig.text(0.045, 0.922, subtitle, color=MUTED, fontsize=13)
        fig.text(0.955, 0.955, f"t U0 / D = {t * U0 / D:5.1f}", color=FG, fontsize=15, ha="right", family="monospace")
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        writer.append_data(frame)
        if n % 20 == 0:
            print(f"frame {n + 1} / {len(files)}", flush=True)
    for _ in range(int(round(2 * args.fps))):
        writer.append_data(frame)
    writer.close()
    print(f"wrote {args.out}")
    if args.png:
        imageio.imwrite(args.png, frame)
    if args.poster:
        imageio.imwrite(args.poster, frame, quality=90)


if __name__ == "__main__":
    main()
