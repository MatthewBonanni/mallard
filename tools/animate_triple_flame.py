"""Animation of triple-flame runs (tools/triple_flame.py) side by side.

    python tools/animate_triple_flame.py OUT.mp4 --run LABEL RUN_DIR[,CONTINUATION] [--run ...]
        [--fuel H2:0.3,N2:0.7] [--oxidizer O2:1,N2:3.76] [--fps 15] [--gif OUT.gif]
        [--png OUT.png] [--title TEXT] [--subtitle TEXT] [--half-height 2.2]

Each frame (1920x1080) shows, for every run, the heat release rate over the
planar stoichiometric flame's peak with the stoichiometric line (white) and the
lines of equivalence ratio 0.5 and 2 (dashed), streamlines of the fresh gas
diverging ahead of the tip, and below them each run's propagation speed against
the inflow, U_F / S_L = (u_in - dx_tip/dt) / S_L (smoothed over a few
outputs; dashed: u_in / S_L), with the heat-release limit sqrt(rho_u / rho_b) of Ruetsch,
Vervisch & Linan (1995). Lengths are in thermal thicknesses delta_L of the
planar flame; the window spans --half-height mixing thicknesses (plus 8
delta_L) on each side of the midline. A run continued from one of its restart
files in another directory (e.g. with a different inflow velocity) is given
as RUN_DIR,CONTINUATION: its outputs follow the first run's up to the restart
time. The MP4 is H.264 (CRF 18) and holds the last frame for 2 s.
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
from triple_flame_analysis import grid, leading_edge, run_info, stream_n2  # noqa: E402

BG = "#0d0f14"
FG = "#e8e8e8"
COLORS = ["#ff8a3d", "#4fc3f7", "#b0e57c"]


def phi_to_Z(phi, Z_st):
    return phi * Z_st / (1 - Z_st + phi * Z_st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--run", nargs=2, action="append", metavar=("LABEL", "RUN_DIR"), required=True)
    ap.add_argument("--fuel", default="H2:0.3,N2:0.7")
    ap.add_argument("--oxidizer", default="O2:1,N2:3.76")
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    ap.add_argument("--mechanism", default=os.path.join(root, "mechanisms", "h2o2.yaml"))
    ap.add_argument("--phase", default="ohmech")
    ap.add_argument("--fps", type=float, default=15)
    ap.add_argument("--half-height", type=float, default=2.2)
    ap.add_argument("--gif")
    ap.add_argument("--png")
    ap.add_argument("--title", default="Triple flames in a hydrogen/air mixing layer")
    ap.add_argument("--subtitle", default="")
    args = ap.parse_args()

    gas = ct.Solution(args.mechanism, args.phase)
    gas.TP = 300.0, ct.one_atm
    gas.set_equivalence_ratio(1.0, args.fuel, args.oxidizer)
    Z_st = gas.mixture_fraction(args.fuel, args.oxidizer)
    N2_F, N2_O = stream_n2(args.fuel, args.mechanism, args.phase), stream_n2(args.oxidizer, args.mechanism, args.phase)
    runs = []
    for label, dirs in args.run:
        frames = []  # (path, time, inflow velocity, segment)
        for seg, d in enumerate(dirs.split(",")):
            pvd = open(os.path.join(d, "solut", "flame.pvd")).read()
            u_in = run_info(d, args.mechanism, args.phase)["u_in"]
            times = [float(s) for s in re.findall(r'timestep="([^"]+)"', pvd)]
            files = re.findall(r'file="([^"]+)"', pvd)
            if frames:
                frames = [f for f in frames if f[1] < times[0] - 1e-12]
            frames += [(os.path.join(d, "solut", f), t, u_in, seg) for f, t in zip(files, times)]
        info = run_info(dirs.split(",")[0], args.mechanism, args.phase)
        runs.append(dict(label=label, info=info, frames=frames, t=[], x=[], U=[], seg=[], u_in=[]))
    n_frames = min(len(r["frames"]) for r in runs)
    tau = runs[0]["info"]["delta"] / runs[0]["info"]["S_L"]
    t_end = max(r["frames"][-1][1] for r in runs) / tau
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=None,
                                ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p", "-preset", "slow"],
                                macro_block_size=8)
    gif_frames, frame = [], None
    n = len(runs)
    for k in range(n_frames):
        fig.clf()
        fig.patch.set_facecolor(BG)
        gap = 0.04
        w = (0.90 - gap * (n - 1)) / n
        for i, r in enumerate(runs):
            info = r["info"]
            dL = info["delta"]
            path, _, u_in, seg = r["frames"][k]
            t, xs, ys, a = grid(path, ["T", "HRR", "U", "Y_N2"])
            T_mid = info["T_u"] + 0.5 * (info["T_b"] - info["T_u"])
            x_tip, y_tip, _ = leading_edge(xs, ys, a["T"], T_mid)
            r["t"].append(t / tau)
            r["x"].append(x_tip)
            r["seg"].append(seg)
            r["u_in"].append(u_in / info["S_L"])
            same = np.array(r["seg"]) == seg
            if same.sum() >= 3:
                tt, xx = np.array(r["t"])[same] * tau, np.array(r["x"])[same]
                m = min(len(tt), 9)
                v = np.polyfit(tt[-m:], xx[-m:], 1)[0]
                r["U"].append((u_in - v) / info["S_L"])
            else:
                r["U"].append(np.nan)
            Z = (a["Y_N2"] - N2_O) / (N2_F - N2_O)
            yc = 0.5 * (ys[0] + ys[-1])
            half = args.half_height * info["mixing"] * dL + 8 * dL
            sel = np.abs(ys - yc) <= half
            X, Y = xs / dL, (ys[sel] - yc) / dL
            ax = fig.add_axes([0.05 + i * (w + gap), 0.36, w, 0.52])
            ax.set_facecolor("black")
            ax.imshow((a["HRR"][:, sel] / info["hrr_max"]).T, origin="lower", cmap="inferno", vmin=0, vmax=1.2,
                      extent=[X[0], X[-1], Y[0], Y[-1]], aspect="equal", interpolation="bilinear")
            ax.contour(X, Y, a["T"][:, sel].T, levels=[info["T_u"] + 0.1 * (info["T_b"] - info["T_u"])],
                       colors=["#5c6bc0"], linewidths=0.8)
            ax.contour(X, Y, Z[:, sel].T, levels=[Z_st], colors=["white"], linewidths=1.3)
            ax.contour(X, Y, Z[:, sel].T, levels=[phi_to_Z(0.5, Z_st), phi_to_Z(2.0, Z_st)], colors=["#bbbbbb"],
                       linewidths=0.8, linestyles="--")
            U = a["U"][:, sel]
            seeds = np.column_stack([np.full(9, X[2]), np.linspace(-0.8, 0.8, 9) * Y[-1]])
            ax.streamplot(X, Y, U[:, :, 0].T, U[:, :, 1].T, start_points=seeds, color="#8fd3ff", linewidth=0.7,
                          arrowsize=0.7, density=4, broken_streamlines=False)
            ax.plot(x_tip / dL, (y_tip - yc) / dL, "o", ms=5, mfc="none", mec="#00e5ff")
            ax.set_xlim(X[0], X[-1])
            ax.set_ylim(Y[0], Y[-1])
            ax.set_title(r["label"], color=COLORS[i], fontsize=16, pad=8)
            ax.tick_params(colors=FG, labelsize=10)
            for s in ax.spines.values():
                s.set_color("#555")
            ax.set_xlabel("x / delta_L  (fresh gas enters from the left)", color=FG, fontsize=11)
            if i == 0:
                ax.set_ylabel("y / delta_L  (fuel-rich side up)", color=FG, fontsize=11)
        fig.text(0.05, 0.293, "Color: heat release rate / stoichiometric planar flame's peak.  White: stoichiometric "
                 "line; dashed: equivalence ratio 0.5 and 2; blue: 10% temperature rise; light blue: streamlines; "
                 "circle: leading edge", color="#aab", fontsize=12)
        axp = fig.add_axes([0.08, 0.07, 0.84, 0.19])
        axp.set_facecolor(BG)
        for i, r in enumerate(runs):
            axp.plot(r["t"], r["U"], color=COLORS[i], lw=2.2, label=r["label"])
            axp.plot(r["t"][-1], r["U"][-1], "o", color=COLORS[i])
            axp.plot(r["t"], r["u_in"], color=COLORS[i], lw=1, ls="--")
        s = np.sqrt(runs[0]["info"]["sigma"])
        axp.axhline(s, color="#ccc", lw=1, ls=":")
        axp.text(0.2, s + 0.05, "sqrt(rho_u / rho_b): heat-release limit for weak gradients", color="#ccc", fontsize=10)
        axp.axhline(1.0, color="#777", lw=1, ls="--")
        axp.set_xlim(0, t_end)
        axp.set_ylim(0.0, s + 0.5)
        axp.set_xlabel("t S_L / delta_L", color=FG, fontsize=12)
        axp.set_ylabel("U_F / S_L", color=FG, fontsize=12)
        axp.tick_params(colors=FG)
        for sp in axp.spines.values():
            sp.set_color("#555")
        axp.legend(loc="lower right", facecolor=BG, edgecolor="#555", labelcolor=FG, fontsize=11)
        fig.text(0.5, 0.955, args.title, color=FG, fontsize=20, ha="center", weight="bold")
        fig.text(0.5, 0.922, args.subtitle, color="#aab", fontsize=13, ha="center")
        fig.text(0.96, 0.955, f"t = {runs[0]['t'][-1]:5.2f} delta_L / S_L", color=FG, fontsize=14, ha="right",
                 family="monospace")
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        writer.append_data(frame)
        if args.gif and k % 2 == 0:
            gif_frames.append(frame[::3, ::3])
        print(f"frame {k + 1}/{n_frames}: " + ", ".join(f"{r['label']}: U_F/S_L = {r['U'][-1]:.3f}" for r in runs),
              flush=True)
    for _ in range(int(round(2 * args.fps))):
        writer.append_data(frame)
    writer.close()
    if args.png:
        imageio.imwrite(args.png, frame)
    if args.gif:
        imageio.mimsave(args.gif, gif_frames + [gif_frames[-1]] * 10, duration=1000 / 10, loop=0)


if __name__ == "__main__":
    main()
