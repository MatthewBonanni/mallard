"""Animation of stratified autoignition runs side by side (examples/autoignition_2d).

    python tools/animate_autoignition.py OUT.mp4 REFERENCE_DIR --run LABEL RUN_DIR [--run ...]
        [--t-start 1.4e-3] [--fps 15] [--gif OUT.gif] [--png OUT.png] [--png-time T]

Each run (tools/autoignition_restart.py, output series solut/autoignition.pvd
with T and HRR) gets a column: its temperature above, its heat release rate
(log scale) below, at the same physical time in every column (a finished run
holds its last output). The strip at the bottom traces each run's mean heat
release rate over its multizone prediction (REFERENCE_DIR/multizone_<run>.csv
from tools/autoignition_reference.py, dashed) and the homogeneous reactor's,
against t / tau_0. Frames are 1920x1080; the MP4 is H.264 (CRF 18) and holds
the last frame for 2 s; --png writes the frame nearest --png-time (default
the last), --gif every second frame at a third of the size.
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
from matplotlib.colors import LogNorm  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mallard_vtu import grid_fields  # noqa: E402

BG = "#0d0f14"
FG = "#e8e8e8"
COLORS = ["#4fc3f7", "#b0e57c", "#ffd166", "#ff8a3d", "#f06292"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("reference")
    ap.add_argument("--run", nargs=2, action="append", metavar=("LABEL", "RUN_DIR"), required=True)
    ap.add_argument("--t-start", type=float, default=1.4e-3)
    ap.add_argument("--fps", type=float, default=15)
    ap.add_argument("--T-range", type=float, nargs=2, default=[1000.0, 1560.0])
    ap.add_argument("--hrr-range", type=float, nargs=2, default=[1e8, 2e11])
    ap.add_argument("--gif")
    ap.add_argument("--png")
    ap.add_argument("--png-time", type=float)
    ap.add_argument("--title", default="Autoignition of thermally stratified lean H2/air: spontaneous ignition fronts "
                                       "and deflagrations")
    ap.add_argument("--subtitle", default="phi = 0.1, 1070 K, 41 atm, u' = 0.5 m/s; 4.1 mm periodic square, "
                                          "400 x 400 cells (10 um), detailed H2/O2 chemistry, mixture-averaged "
                                          "transport")
    args = ap.parse_args()

    hom = np.genfromtxt(os.path.join(args.reference, "homogeneous.csv"), delimiter=",", names=True)
    tau0 = hom["t"][np.argmax(np.gradient(hom["T"], hom["t"]))]
    h_ref = hom["HRR"].max()
    runs = []
    for label, run_dir in args.run:
        pvd = open(os.path.join(run_dir, "solut", "autoignition.pvd")).read()
        files = re.findall(r'file="([^"]+)"', pvd)
        times = np.array([float(t) for t in re.findall(r'timestep="([^"]+)"', pvd)])
        run = os.path.basename(os.path.normpath(run_dir))
        mz = np.loadtxt(os.path.join(args.reference, f"multizone_{run}.csv"), delimiter=",", skiprows=1)
        runs.append(dict(label=label, dir=run_dir, files=files, times=times, mz=mz, t=[], hrr=[], last=None))
    # Mean HRR of every output, for the traces
    dt = np.median(np.diff(runs[0]["times"]))
    t_end = max(r["times"][-1] for r in runs)
    frame_times = np.arange(args.t_start, t_end + 0.5 * dt, dt)
    i_png = len(frame_times) - 1 if args.png_time is None else int(np.argmin(np.abs(frame_times - args.png_time)))
    peak = 0.0
    for r in runs:
        for f in r["files"]:
            t, _, _, fld = grid_fields(os.path.join(r["dir"], "solut", f), ["HRR"])
            r["t"].append(t)
            r["hrr"].append(fld["HRR"].mean())
        r["t"], r["hrr"] = np.array(r["t"]), np.array(r["hrr"])
        peak = max(peak, r["hrr"].max(), r["mz"][:, 2].max())

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=None,
                                ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p", "-preset", "slow", "-threads", "4"],
                                macro_block_size=8)
    gif_frames, frame = [], None
    n = len(runs)
    size = 0.34  # panel height (fraction of the figure height)
    w = size * 1080 / 1920
    gap = 0.012
    x0 = 0.5 - 0.5 * (n * w + (n - 1) * gap) - 0.02
    norm_h = LogNorm(*args.hrr_range)
    for k, tf in enumerate(frame_times):
        fig.clf()
        fig.patch.set_facecolor(BG)
        for i, r in enumerate(runs):
            j = int(np.argmin(np.abs(r["times"] - tf)))  # a finished run holds its last output
            t, x, y, fld = grid_fields(os.path.join(r["dir"], "solut", r["files"][j]), ["T", "HRR"])
            ext = [0, (x[-1] + x[0]) * 1e3, 0, (y[-1] + y[0]) * 1e3]
            for row, (name, kw) in enumerate([("T", dict(cmap="RdYlBu_r", vmin=args.T_range[0], vmax=args.T_range[1])),
                                              ("HRR", dict(cmap="inferno", norm=norm_h))]):
                ax = fig.add_axes([x0 + i * (w + gap), 0.535 - row * (size + 0.012), w, size])
                data = fld[name] if name == "T" else np.clip(fld[name], args.hrr_range[0], None)
                im = ax.imshow(data.T, origin="lower", extent=ext, aspect="equal", interpolation="bilinear", **kw)
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_color("#555")
                if row == 0:
                    ax.set_title(r["label"], color=COLORS[i], fontsize=17, pad=6)
                if i == n - 1:
                    cax = fig.add_axes([x0 + n * (w + gap) + 0.004, 0.555 - row * (size + 0.012), 0.007, size - 0.04])
                    cb = fig.colorbar(im, cax=cax)
                    cb.ax.tick_params(colors=FG, labelsize=11)
                    cb.set_label("T [K]" if row == 0 else "heat release rate [W/m$^3$]", color=FG, fontsize=13)
        axp = fig.add_axes([0.06, 0.055, 0.88, 0.105])
        axp.set_facecolor(BG)
        axp.plot(hom["t"] / tau0, hom["HRR"] / h_ref, color="#888", lw=1, ls=":")
        for i, r in enumerate(runs):
            axp.plot(r["mz"][:, 0] / tau0, r["mz"][:, 2] / h_ref, color=COLORS[i], lw=1, ls="--", alpha=0.8)
            s = r["t"] <= tf + 1e-9
            axp.plot(r["t"][s] / tau0, r["hrr"][s] / h_ref, color=COLORS[i], lw=2.4)
            if s.any():
                axp.plot(r["t"][s][-1] / tau0, r["hrr"][s][-1] / h_ref, "o", color=COLORS[i])
        axp.axvline(tf / tau0, color="#666", lw=1)
        axp.set_xlim(args.t_start / tau0, t_end / tau0)
        axp.set_ylim(0, 1.1 * peak / h_ref)
        axp.set_xlabel(r"$t / \tau_0$", color=FG, fontsize=13, labelpad=1)
        axp.set_ylabel("mean HRR", color=FG, fontsize=12)
        axp.tick_params(colors=FG, labelsize=11)
        for s in axp.spines.values():
            s.set_color("#555")
        axp.text(0.995, 1.03, "mean heat release rate over the homogeneous reactor's peak: DNS (solid), "
                 "multizone without transport (dashed), homogeneous (dotted)", transform=axp.transAxes,
                 color="#aab", ha="right", va="bottom", fontsize=11)
        fig.text(0.5, 0.955, args.title, color=FG, fontsize=20, ha="center", weight="bold")
        fig.text(0.5, 0.922, args.subtitle, color="#aab", fontsize=13, ha="center")
        fig.text(0.985, 0.955, f"t = {tf * 1e3:5.3f} ms\n= {tf / tau0:5.3f} tau_0", color=FG, fontsize=14, ha="right",
                 va="top", family="monospace")
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        writer.append_data(frame)
        if args.gif and k % 2 == 0:
            gif_frames.append(frame[::3, ::3])
        if args.png and k == i_png:
            imageio.imwrite(args.png, frame)
        print(f"frame {k + 1}/{len(frame_times)}: t = {tf * 1e3:.3f} ms", flush=True)
    for _ in range(int(round(2 * args.fps))):
        writer.append_data(frame)
    writer.close()
    if args.gif:
        imageio.mimsave(args.gif, gif_frames + [gif_frames[-1]] * 10, duration=1000 / 10, loop=0)


if __name__ == "__main__":
    main()
