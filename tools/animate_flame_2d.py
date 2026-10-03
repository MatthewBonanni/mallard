"""Side-by-side animation of 2D premixed flame runs (examples/flame_2d).

    python tools/animate_flame_2d.py OUT.mp4 --run LABEL SERIES.pvd FULL.csv [--run ...]
        [--mechanism mechanisms/h2o2.yaml --phase ohmech] [--fps 15] [--gif OUT.gif]
        [--png OUT.png] [--title TEXT] [--subtitle TEXT] [--tile 2] [--t-range 0.08]

Each run is a periodic channel from tools/flame_restart.py --ny, writing T
and HRR on a generated "cartesian" mesh; FULL.csv is its planar Cantera
flame (tools/flame_reference.py's full solution). Each frame (1920x1080)
shows the temperature of every run over the adiabatic flame temperature
(diverging scale over 1 +- --t-range, default 0.08: red where differential
diffusion makes the burnt gas superadiabatic) with heat-release contours at
0.5 and 1 times the planar flame's peak, the channel
tiled --tile times across (it is periodic), and below them the consumption
speed of each run against time,

    S_c / S_L = (int HRR dA / L_y) / int HRR_planar dx,

the heat release per unit width over that of the planar Cantera flame:
1 for a planar flame, larger as the front wrinkles and its curved parts burn
faster. Prints the same series as a table. The MP4 is H.264 (CRF 18) and holds
the last frame for 2 s.
"""
import argparse
import os
import re
import sys

import cantera as ct
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mallard_vtu import grid_fields  # noqa: E402

BG = "#0d0f14"
FG = "#e8e8e8"
COLORS = ["#ff8a3d", "#4fc3f7", "#b0e57c"]


def planar_hrr_integral(full, mechanism, phase):
    with open(full) as f:
        names = f.readline().strip().split(",")
    d = np.loadtxt(full, delimiter=",", skiprows=1)
    gas = ct.Solution(mechanism, phase)
    hrr = np.empty(d.shape[0])
    for i in range(d.shape[0]):
        gas.TDY = d[i, 2], d[i, 3], d[i, 4:]
        hrr[i] = gas.heat_release_rate
    return np.trapezoid(hrr, d[:, 0]), hrr.max(), d[0, 1], d[0, 2], d[-1, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--run", nargs=3, action="append", metavar=("LABEL", "PVD", "FULL"), required=True)
    ap.add_argument("--mechanism", default="mechanisms/h2o2.yaml")
    ap.add_argument("--phase", default="ohmech")
    ap.add_argument("--fps", type=float, default=15)
    ap.add_argument("--tile", type=int, default=1)
    ap.add_argument("--t-range", type=float, default=0.08)
    ap.add_argument("--gif")
    ap.add_argument("--png")
    ap.add_argument("--title", default="Lean H2/air flames: thermodiffusive instability")
    ap.add_argument("--subtitle", default="")
    args = ap.parse_args()
    runs = []
    for label, pvd, full in args.run:
        files = re.findall(r'file="([^"]+)"', open(pvd).read())
        I_ref, H_ref, S_L, T_u, T_b = planar_hrr_integral(full, args.mechanism, args.phase)
        t_last = float(re.findall(r'timestep="([^"]+)"', open(pvd).read())[-1])
        runs.append(dict(label=label, base=os.path.dirname(pvd), files=files, t_last=t_last, I_ref=I_ref, H_ref=H_ref, S_L=S_L,
                         T_u=T_u, T_b=T_b, t=[], sc=[]))
    n_frames = min(len(r["files"]) for r in runs)
    t_end = 1e3 * min(r["t_last"] for r in runs)
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=None,
                                ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p", "-preset", "slow"],
                                macro_block_size=8)
    gif_frames, frame = [], None
    print("t [ms] " + " ".join(f"{r['label'][:18]:>18}" for r in runs))
    for k in range(n_frames):
        fig.clf()
        fig.patch.set_facecolor(BG)
        n = len(runs)
        gap = 0.03
        w = (0.92 - gap * (n - 1)) / n
        for i, r in enumerate(runs):
            t, x, y, fld = grid_fields(os.path.join(r["base"], r["files"][k]), ["T", "HRR"])
            dx, dy = x[1] - x[0], y[1] - y[0]
            Ly = y.size * dy
            sc = fld["HRR"].sum() * dx * dy / Ly / r["I_ref"]
            r["t"].append(t)
            r["sc"].append(sc)
            T = np.tile(fld["T"], (1, args.tile)).T
            H = np.tile(fld["HRR"], (1, args.tile)).T
            ax = fig.add_axes([0.04 + i * (w + gap), 0.36, w, 0.52])
            ext = [x[0] * 1e3 - 0.5 * dx * 1e3, x[-1] * 1e3 + 0.5 * dx * 1e3, 0, Ly * args.tile * 1e3]
            im = ax.imshow(T / r["T_b"], origin="lower", extent=ext, cmap="RdBu_r", vmin=1.0 - args.t_range,
                           vmax=1.0 + args.t_range, aspect="equal", interpolation="bilinear")
            yy = (np.arange(T.shape[0]) + 0.5) * dy * 1e3
            ax.contour(x * 1e3, yy, H / r["H_ref"], levels=[0.5, 1.0], colors=["#ffd166", "#ffffff"],
                       linewidths=[0.8, 1.4])
            ax.set_title(r["label"], color=COLORS[i], fontsize=16, pad=8)
            ax.tick_params(colors=FG, labelsize=10)
            for s in ax.spines.values():
                s.set_color("#555")
            ax.set_xlabel("x [mm]  (fresh gas enters from the left)", color=FG, fontsize=11)
            if i == 0:
                ax.set_ylabel("y [mm]  (periodic)", color=FG, fontsize=11)
        print(f"{runs[0]['t'][-1] * 1e3:6.3f} " + " ".join(f"{r['sc'][-1]:18.3f}" for r in runs), flush=True)
        cax = fig.add_axes([0.965, 0.40, 0.008, 0.44])
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(colors=FG, labelsize=10)
        fig.text(0.04, 0.293, "Color: temperature over the adiabatic flame temperature (red: superadiabatic). "
                 "Lines: heat release rate at 0.5 and 1 times the planar flame's peak", color="#aab", fontsize=12)
        axp = fig.add_axes([0.08, 0.07, 0.84, 0.19])
        axp.set_facecolor(BG)
        for i, r in enumerate(runs):
            axp.plot(np.array(r["t"]) * 1e3, r["sc"], color=COLORS[i], lw=2.2, label=r["label"])
            axp.plot(r["t"][-1] * 1e3, r["sc"][-1], "o", color=COLORS[i])
        axp.axhline(1.0, color="#777", lw=1, ls="--")
        axp.set_xlim(0, t_end)
        top = max(max(r["sc"]) for r in runs)
        axp.set_ylim(0.0, max(2.0, 1.15 * top))
        axp.set_xlabel("t [ms]", color=FG, fontsize=12)
        axp.set_ylabel("S_c / S_L", color=FG, fontsize=12)
        axp.tick_params(colors=FG)
        for s in axp.spines.values():
            s.set_color("#555")
        axp.legend(loc="upper left", facecolor=BG, edgecolor="#555", labelcolor=FG, fontsize=11)
        axp.text(0.995, 0.06, "consumption speed over the planar flame's (dashed)", transform=axp.transAxes,
                 color="#aab", ha="right", fontsize=11)
        fig.text(0.5, 0.955, args.title, color=FG, fontsize=20, ha="center", weight="bold")
        fig.text(0.5, 0.922, args.subtitle, color="#aab", fontsize=13, ha="center")
        fig.text(0.96, 0.955, f"t = {runs[0]['t'][-1] * 1e3:5.3f} ms", color=FG, fontsize=15, ha="right",
                 family="monospace")
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        writer.append_data(frame)
        if args.gif and k % 2 == 0:
            gif_frames.append(frame[::3, ::3])
    for _ in range(int(round(2 * args.fps))):
        writer.append_data(frame)
    writer.close()
    if args.png:
        imageio.imwrite(args.png, frame)
    if args.gif:
        imageio.mimsave(args.gif, gif_frames + [gif_frames[-1]] * 10, duration=1000 / 10, loop=0)


if __name__ == "__main__":
    main()
