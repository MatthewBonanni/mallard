"""Animation of flame-vortex interaction runs (tools/flame_vortex.py) on one clock.

    python tools/animate_flame_vortex.py OUT.mp4 RUN_DIR [RUN_DIR ...] [--cols 3] [--fps 15]
        [--gif OUT.gif] [--png OUT.png] [--title TEXT] [--subtitle TEXT] [--t-max T]

Each run is a panel: the temperature over the planar flame's burnt-gas
temperature, mirrored about the pair's axis to show the whole pair, with the
heat release rate at 0.25 times the planar peak (white) and vorticity contours
(blue and pink, the two signs), in thermal thicknesses delta_L around the
flame's initial position. All panels share the time t S_L / delta_L (a run
that ended holds its last frame). On the right, the spectral diagram of
Poinsot, Veynante & Candel (1991) with each run's outcome
(tools/flame_vortex_analysis.py) and the histories of the heat release. The
MP4 is H.264 (CRF 18) and holds the last frame for 2 s.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flame_vortex_analysis import MARKERS, analyze, classify, parameters, planar  # noqa: E402
from triple_flame_analysis import grid  # noqa: E402

BG = "#0d0f14"
FG = "#e8e8e8"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("runs", nargs="+")
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    ap.add_argument("--mechanism", default=os.path.join(root, "mechanisms", "h2o2.yaml"))
    ap.add_argument("--phase", default="ohmech")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--fps", type=float, default=15)
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--t-max", type=float)
    ap.add_argument("--gif")
    ap.add_argument("--png")
    ap.add_argument("--title", default="Premixed flame-vortex interactions")
    ap.add_argument("--subtitle", default="")
    args = ap.parse_args()

    runs = []
    for d in args.runs:
        p = planar(d, args.mechanism, args.phase)
        pvd = open(os.path.join(d, "solut", "flame.pvd")).read()
        files = re.findall(r'file="([^"]+)"', pvd)
        times = np.array([float(s) for s in re.findall(r'timestep="([^"]+)"', pvd)])
        rows = analyze(d, p)
        r, u = parameters(d)
        tau = p["delta"] / p["S_L"]
        head = open(os.path.join(d, "input.toml")).readline()
        x_v = float(re.search(r"pair at x = ([0-9.]+) mm", head).group(1)) * 1e-3
        runs.append(dict(dir=d, p=p, files=files, t=times / tau, rows=rows, r=r, u=u, tau=tau, x_v=x_v,
                         outcome=classify(rows)))
        print(f"{d}: r = {r:g}, u' = {u:g}: {runs[-1]['outcome']}", flush=True)
    runs.sort(key=lambda q: (q["r"], q["u"]))
    t_max = args.t_max or max(q["t"][-1] for q in runs)
    clock = np.linspace(0, t_max, args.frames)
    n = len(runs)
    cols = args.cols
    rows_n = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=None,
                                ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p", "-preset", "slow"],
                                macro_block_size=8)
    gif_frames, frame = [], None
    left, right, bottom, top = 0.03, 0.66, 0.05, 0.86
    pw, ph = (right - left) / cols, (top - bottom) / rows_n
    for k, tc in enumerate(clock):
        fig.clf()
        fig.patch.set_facecolor(BG)
        for i, q in enumerate(runs):
            j = int(np.clip(np.searchsorted(q["t"], tc + 1e-9) - 1, 0, len(q["files"]) - 1))
            t, xs, ys, a = grid(os.path.join(q["dir"], "solut", q["files"][j]), ["T", "HRR", "U"])
            dL = q["p"]["delta"]
            dx = xs[1] - xs[0]
            ux, uy = a["U"][:, :, 0], a["U"][:, :, 1]
            omega = np.gradient(uy, dx, axis=0) - np.gradient(ux, dx, axis=1)
            T = np.concatenate([a["T"][:, ::-1], a["T"]], axis=1) / q["p"]["T_b"]
            H = np.concatenate([a["HRR"][:, ::-1], a["HRR"]], axis=1) / q["p"]["peak"]
            W = np.concatenate([-omega[:, ::-1], omega], axis=1)
            yy = np.concatenate([-ys[::-1], ys]) / dL
            xx = (xs - q["x_v"]) / dL
            c, rr = i % cols, i // cols
            ax = fig.add_axes([left + c * pw + 0.006, top - (rr + 1) * ph + 0.035, pw - 0.012, ph - 0.07])
            ax.imshow(T.T, origin="lower", cmap="magma", vmin=0.15, vmax=1.05, aspect="equal",
                      extent=[xx[0], xx[-1], yy[0], yy[-1]], interpolation="bilinear")
            w_max = q["u"] * q["p"]["S_L"] / (q["r"] * dL / 4)
            ax.contour(xx, yy, W.T, levels=[-0.6 * w_max, -0.2 * w_max], colors=["#64b5f6"], linewidths=0.7)
            ax.contour(xx, yy, W.T, levels=[0.2 * w_max, 0.6 * w_max], colors=["#f48fb1"], linewidths=0.7)
            ax.contour(xx, yy, H.T, levels=[0.25], colors=["white"], linewidths=1.0)
            ax.set_xlim(xx[0], xx[-1])
            ax.set_ylim(yy[0], yy[-1])
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(MARKERS[q["outcome"]][1])
                s.set_linewidth(2)
            ax.set_title(f"r = {q['r']:g} delta_L, u' = {q['u']:g} S_L: {q['outcome']}", color=FG, fontsize=12,
                         pad=4)
        axd = fig.add_axes([0.71, 0.47, 0.27, 0.39])
        axd.set_facecolor(BG)
        from flame_vortex_analysis import spectral_axes
        spectral_axes(axd, fg="#bbb")
        for q in runs:
            m, col = MARKERS[q["outcome"]]
            axd.plot(q["r"], q["u"], m, color=col, ms=10, mec="white")
        for o, (m, col) in MARKERS.items():
            axd.plot([], [], m, color=col, mec="white", label=o)
        axd.legend(loc="lower right", facecolor=BG, edgecolor="#555", labelcolor=FG, fontsize=9)
        axd.tick_params(colors=FG, which="both")
        axd.xaxis.label.set_color(FG)
        axd.yaxis.label.set_color(FG)
        for s in axd.spines.values():
            s.set_color("#555")
        axq = fig.add_axes([0.71, 0.07, 0.27, 0.3])
        axq.set_facecolor(BG)
        for q in runs:
            t = q["rows"][:, 0] / q["tau"]
            sel = t <= tc + 1e-9
            col = MARKERS[q["outcome"]][1]
            axq.plot(t[sel], q["rows"][sel, 1], color=col, lw=1.5)
        axq.axhline(1.0, color="#777", ls="--", lw=1)
        axq.set_xlim(0, t_max)
        axq.set_ylim(0, max(2.0, 1.1 * max(q["rows"][:, 1].max() for q in runs)))
        axq.set_xlabel("t S_L / delta_L", color=FG)
        axq.set_ylabel("heat release / planar flame's", color=FG)
        axq.tick_params(colors=FG)
        for s in axq.spines.values():
            s.set_color("#555")
        fig.text(0.5, 0.955, args.title, color=FG, fontsize=20, ha="center", weight="bold")
        fig.text(0.5, 0.922, args.subtitle, color="#aab", fontsize=13, ha="center")
        fig.text(0.03, 0.015, "Color: temperature / burnt gas's.  White: heat release at 0.25 of the planar peak.  "
                 "Blue / pink: vorticity of either sign.  x from the pair's start, in delta_L; fresh gas enters "
                 "from the left at S_L", color="#aab", fontsize=11)
        fig.text(0.98, 0.955, f"t = {tc:5.2f} delta_L / S_L", color=FG, fontsize=14, ha="right", family="monospace")
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
