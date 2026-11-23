"""Edge trajectories and animation of the shock_curtain run: a Mach 1.66 shock hitting a curtain of glass beads.

    python examples/shock_curtain/animate_shock_curtain.py extract RUN_DIR   (writes RUN_DIR/frames.npz)
    python examples/shock_curtain/animate_shock_curtain.py edges RUN_DIR [RUN_DIR ...] [--plot OUT.png]
    python examples/shock_curtain/animate_shock_curtain.py render OUT.mp4 RUN_DIR [--fps 24] [--crf 18]
        [--poster OUT.jpg] [--poster-t T]

extract reads a run's restart frames (frames/frame_*.restart, which carry the particles) into one compact file:
the gas averaged over the height of the tube (the flow is planar) and every bead's position and velocity. edges
prints the curtain's upstream and downstream edges, the positions that 1% and 99% of the beads have not passed,
against the schlieren edges of Wagner et al. (wagner_edges.csv), as displacements from their initial positions
against the time since the shock reached the curtain. render draws the test section (numerical schlieren of the
gas, the beads colored by their velocity), the pressure along the whole tube, the edge trajectories against the
experiment, and the beads' volume fraction. Frames are drawn at 3840 x 2160; render writes OUT.mp4 at that size and
OUT_1080p.mp4 at half of it (H.264), and the poster as OUT.jpg, OUT.png and OUT_1080p.jpg (T in microseconds after
the impact; OUT "-" skips the video).
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
from mallard import restart  # noqa: E402

X_CURTAIN = 0.2  # m, the curtain's upstream edge
THICKNESS = 2e-3  # m, the curtain's initial thickness
W_SHOCK = 572.91  # m/s, the incident shock speed
T_IMPACT = 3e-3 / W_SHOCK  # s, the shock starts 3 mm upstream of the curtain
P1 = 82700.0  # Pa, the gas ahead of the shock
QUANTILES = (0.01, 0.99)


def read_input(run_dir):
    text = open(os.path.join(run_dir, "input.toml")).read()

    def num(key, table=None):
        block = text if table is None else text[text.index(f"[{table}]"):]
        return float(re.search(rf"^{key} = ([0-9.eE+-]+)", block, re.M).group(1))

    nx, ny = int(num("Nx")), int(num("Ny"))
    return dict(nx=nx, ny=ny, lx=num("Lx"), ly=num("Ly"), d=num("diameter"), w=num("weight"),
                gamma=num("gamma"), rgas=num("p_ref") / (num("rho_ref") * num("T_ref")))


def extract(run_dir):
    cfg = read_input(run_dir)
    files = sorted(glob.glob(os.path.join(run_dir, "frames", "frame_*.restart")))
    first = restart.read(files[0])
    n = int(first.arrays["PARTICLE_ID"].max()) + 1
    nf, nx, ny = len(files), cfg["nx"], cfg["ny"]
    t = np.empty(nf)
    gas = {k: np.empty((nf, nx), dtype=np.float32) for k in ("rho", "u", "p")}
    pos = np.full((nf, n, 2), np.nan, dtype=np.float32)
    vel = np.full((nf, n), np.nan, dtype=np.float32)
    for k, f in enumerate(files):
        r = restart.read(f)
        t[k] = r.t
        # cells of the Cartesian mesh in global order: y fastest
        rho = r["RHO"].reshape(nx, ny)
        mx, my, e = (r[v].reshape(nx, ny) for v in ("RHOU_X", "RHOU_Y", "RHOE"))
        p = (cfg["gamma"] - 1) * (e - 0.5 * (mx * mx + my * my) / rho)
        gas["rho"][k], gas["u"][k], gas["p"][k] = rho.mean(1), (mx / rho).mean(1), p.mean(1)
        ids = r.arrays["PARTICLE_ID"].astype(np.int64)
        pos[k, ids] = r.arrays["PARTICLE_X"].reshape(len(ids), 2)
        vel[k, ids] = r.arrays["PARTICLE_V"].reshape(len(ids), 2)[:, 0]
    np.savez_compressed(os.path.join(run_dir, "frames.npz"), t=t, pos=pos, vel=vel, **gas,
                        **{k: v for k, v in cfg.items()})
    print(f"{run_dir}: {nf} frames, {n} particles, t = {t[0] * 1e6:g} to {t[-1] * 1e6:g} us")


def edges_of(run):
    """Times since the impact (us) and the edges' displacements (mm) from their initial positions."""
    x = run["pos"][:, :, 0].astype(np.float64)
    q = np.nanquantile(x, QUANTILES, axis=1).T
    return (run["t"] - T_IMPACT) * 1e6, (q - q[0]) * 1e3


def wagner():
    return np.genfromtxt(os.path.join(HERE, "wagner_edges.csv"), delimiter=",", names=True)


def edges(args):
    data = wagner()
    runs = [(d, np.load(os.path.join(d, "frames.npz"))) for d in args.runs]
    print("t (us)   Wagner et al.: upstream, downstream (mm)   " + "   ".join(os.path.basename(os.path.normpath(d))
                                                                          for d, _ in runs))
    curves = []
    for d, r in runs:
        ts, e = edges_of(r)
        curves.append((d, ts, e))
    for row in data:
        if row["t_us"] % 50:
            continue
        line = f"{row['t_us']:6.0f}   {row['upstream_mm']:6.2f} {row['downstream_mm']:6.2f}"
        for _, ts, e in curves:
            if row["t_us"] <= ts[-1]:
                line += f"   {np.interp(row['t_us'], ts, e[:, 0]):6.2f} {np.interp(row['t_us'], ts, e[:, 1]):6.2f}"
        print(line)
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
        for key, c in (("upstream", "C0"), ("downstream", "C3")):
            ax.fill_between(data["t_us"], data[key + "_lo_mm"], data[key + "_hi_mm"], color=c, alpha=0.25, lw=0)
            ax.plot(data["t_us"], data[key + "_mm"], c, lw=1, ls="--")
        for i, (d, ts, e) in enumerate(curves):
            ax.plot(ts, e[:, 0], "C0", lw=1.5, alpha=1 - 0.3 * i, label=f"{d} upstream")
            ax.plot(ts, e[:, 1], "C3", lw=1.5, alpha=1 - 0.3 * i, label=f"{d} downstream")
        ax.set(xlabel="t after impact (us)", ylabel="edge displacement (mm)", xlim=(0, 700), ylim=(0, 60))
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(args.plot)


BG = "#0d0f14"
FG = "#e8e8e8"
DIM = "#b8bcc6"
EDGE = "#4a505c"
UP_C, DOWN_C = "#5ab4ff", "#ff7a59"
W, HT = 3840, 2160  # the master frame; render also encodes a copy at half this size
MARGIN = 46
WX0, WX1 = -20e-3, 60e-3  # the test-section window, from the curtain's upstream edge
PW, PH = W - 2 * MARGIN, 1000
PX = (WX1 - WX0) / PW  # m per pixel
SS = 2  # supersampling of the bead dots
DOT_RADIUS = 1.7  # in master pixels
V_MAX = 100.0  # m/s, top of the velocity colormap


def hex_rgb(c):
    return np.array([int(c[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0


class Section:
    """A slice PH pixels tall of the test section at mid-height: numerical schlieren of the gas under the beads,
    drawn as round dots colored by their streamwise velocity, anti-aliased by supersampling."""

    def __init__(self, run, cmap):
        self.cmap = cmap
        nx = int(run["nx"])
        self.y0 = 0.5 * (float(run["ly"]) - PH * PX)
        self.xc = (np.arange(nx) + 0.5) * float(run["lx"]) / nx - X_CURTAIN
        self.xp = WX0 + (np.arange(PW) + 0.5) * PX
        g = np.abs(np.gradient(run["rho"][0].astype(np.float64), self.xc))
        self.g_ref = float(g.max())  # the incident shock
        r = DOT_RADIUS * SS
        k = int(np.ceil(r))
        dy, dx = np.mgrid[-k:k + 1, -k:k + 1]
        inside = dx ** 2 + dy ** 2 <= r * r
        self.offsets = np.column_stack([dy[inside], dx[inside]])

    def __call__(self, rho, pos, vel):
        bg, light = hex_rgb(BG), np.array([0.80, 0.86, 0.95])
        g = np.abs(np.gradient(rho.astype(np.float64), self.xc))
        s = np.interp(self.xp, self.xc, 1 - np.exp(-8.0 * g / self.g_ref))
        img = np.broadcast_to(bg + 0.9 * s[None, :, None] * (light - bg), (PH, PW, 3)).copy()
        ok = ~np.isnan(pos[:, 0])
        p, v = pos[ok], vel[ok]
        order = np.argsort(v)  # faster beads on top
        p, v = p[order], v[order]
        cols = self.cmap(np.clip(v / V_MAX, 0, 1))[:, :3].astype(np.float32)
        hs, ws = PH * SS, PW * SS
        cover = np.zeros((hs, ws), dtype=np.float32)
        color = np.zeros((hs, ws, 3), dtype=np.float32)
        iy = np.floor((p[:, 1] - self.y0) / PX * SS).astype(np.int64)
        ix = np.floor((p[:, 0] - X_CURTAIN - WX0) / PX * SS).astype(np.int64)
        for oy, ox in self.offsets:
            jy, jx = iy + oy, ix + ox
            m = (jy >= 0) & (jy < hs) & (jx >= 0) & (jx < ws)
            cover[jy[m], jx[m]] = 1.0
            color[jy[m], jx[m]] = cols[m]
        a = cover.reshape(PH, SS, PW, SS).mean(axis=(1, 3))[..., None]
        c = color.reshape(PH, SS, PW, SS, 3).sum(axis=(1, 3)) / np.maximum(a * SS * SS, 1e-9)
        img = img * (1 - a) + c * a
        return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)[::-1]


def shocks(xc, p, t_us, curtain):
    """The shocks of a pressure profile, as (position, name): jumps of more than 0.25 p1 over 4 cells outside the
    curtain (x0, x1). A shock moving downstream has the higher pressure on its left."""
    jump = p[4:] - p[:-4]
    out = []
    idx = np.where(np.abs(jump) > 0.25 * P1)[0]
    for g in np.split(idx, np.where(np.diff(idx) > 8)[0] + 1) if len(idx) else []:
        i = g[np.argmax(np.abs(jump[g]))]
        if curtain[0] - 1e-3 < xc[i + 2] < curtain[1] + 1e-3:
            continue
        if jump[i] > 0:
            name = "reflected shock"
        else:
            name = "incident shock" if t_us < 0 else "transmitted shock"
        out.append((xc[i + 2], name))
    return out


def volume_fraction(x, run, edges_m):
    """The beads' volume fraction, averaged over the height, on the bins edges_m (from the curtain's edge)."""
    vp = np.pi / 6 * float(run["d"]) ** 3
    n, _ = np.histogram(x - X_CURTAIN, bins=edges_m)
    return n * float(run["w"]) * vp / (np.diff(edges_m) * float(run["ly"]))


def render(args):
    import imageio.v2 as imageio
    import matplotlib
    from PIL import Image

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.colors import Normalize

    run = np.load(os.path.join(args.run, "frames.npz"))
    pos_all, vel_all = run["pos"], run["vel"]
    data = wagner()
    t_us, e = edges_of(run)
    cmap = colormaps["plasma"]
    section = Section(run, cmap)
    dpi = 200
    fig = plt.figure(figsize=(W / dpi, HT / dpi), dpi=dpi)
    fig.patch.set_facecolor(BG)
    y_sec = HT - 290 - PH
    im = fig.figimage(np.zeros((PH, PW, 3), dtype=np.uint8), xo=MARGIN, yo=y_sec, origin="upper")

    def style(ax, title):
        ax.set_facecolor(BG)
        for s in ax.spines.values():
            s.set_color(EDGE)
        ax.tick_params(colors=DIM, labelsize=11)
        ax.xaxis.label.set_color(DIM)
        ax.yaxis.label.set_color(DIM)
        ax.grid(color="#262b34", lw=0.6)
        ax.set_title(title, color=FG, fontsize=13.5, loc="left", pad=10)

    axs = fig.add_axes([MARGIN / W, y_sec / HT, PW / W, PH / HT])
    axs.set_xlim(WX0 * 1e3, WX1 * 1e3)
    axs.set_ylim(section.y0 * 1e3, (section.y0 + PH * PX) * 1e3)
    axs.patch.set_alpha(0)
    axs.set_zorder(3)  # over the figimage
    for s in axs.spines.values():
        s.set_color(EDGE)
    axs.tick_params(colors=DIM, labelsize=11, direction="in")
    axs.set_yticks([])
    axs.set_xlabel("x (mm from the curtain's initial upstream edge)", fontsize=12, color=DIM)
    exp_lines = [axs.axvline(0, color=c, lw=1.6, ls=(0, (6, 4)), alpha=0.9) for c in (UP_C, DOWN_C)]
    exp_text = [axs.text(0, 0, "", color=c, fontsize=11.5, ha=ha, va="top", zorder=5)
                for c, ha in ((UP_C, "right"), (DOWN_C, "left"))]
    labels = [axs.text(0, 0, "", color=FG, fontsize=12, ha="center", va="bottom", zorder=5,
                       bbox=dict(boxstyle="round,pad=0.3", fc=BG, ec="none", alpha=0.75)) for _ in range(3)]

    cax = fig.add_axes([(W - MARGIN - 760) / W, (y_sec + PH + 40) / HT, 760 / W, 20 / HT])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, V_MAX), cmap=cmap), cax=cax, orientation="horizontal")
    cb.outline.set_edgecolor(EDGE)
    cax.tick_params(colors=DIM, labelsize=11)
    cax.set_title("bead velocity (m/s)", color=DIM, fontsize=12, loc="left", pad=5)

    y_p, h_p, gap = 110, y_sec - 110 - 210, 130
    w_p = (PW - 2 * gap) // 3
    xc = section.xc * 1e3
    axp = fig.add_axes([(MARGIN + 90) / W, y_p / HT, (w_p + 170) / W, h_p / HT])
    style(axp, "Pressure along the tube (shaded: the slice above)")
    axp.axvspan(WX0 * 1e3, WX1 * 1e3, color="#ffffff", alpha=0.06, lw=0)
    p_line, = axp.plot(xc, run["p"][0] / P1, color="#9fe7ff", lw=1.8)
    axp.set_xlim(xc[0], xc[-1])
    axp.set_ylim(0.5, 5.0)
    axp.set_xlabel("x (mm)", fontsize=12)
    axp.set_ylabel("p / p₁", fontsize=12)

    x_e = MARGIN + w_p + 260 + gap
    w_e = w_p - 130
    axe = fig.add_axes([x_e / W, y_p / HT, w_e / W, h_p / HT])
    style(axe, "Curtain edges against the experiment")
    for key, c in (("upstream", UP_C), ("downstream", DOWN_C)):
        axe.fill_between(data["t_us"], data[key + "_lo_mm"], data[key + "_hi_mm"], color=c, alpha=0.22, lw=0)
        axe.plot(data["t_us"], data[key + "_mm"], color=c, lw=1.3, ls=(0, (4, 3)))
    up_line, = axe.plot([], [], color=UP_C, lw=2.8)
    dn_line, = axe.plot([], [], color=DOWN_C, lw=2.8)
    axe.set_xlim(0, 700)
    axe.set_ylim(0, 60)
    axe.set_xlabel("t after impact (µs)", fontsize=12)
    axe.set_ylabel("edge displacement (mm)", fontsize=12)
    axe.text(0.03, 0.95, "downstream edge", color=DOWN_C, transform=axe.transAxes, fontsize=12, va="top")
    axe.text(0.03, 0.87, "upstream edge", color=UP_C, transform=axe.transAxes, fontsize=12, va="top")
    axe.text(0.97, 0.04, "solid: Mallard\ndashed: Wagner et al.,\n4 runs (band: range)", color=DIM,
             transform=axe.transAxes, fontsize=10.5, va="bottom", ha="right")

    x_v = x_e + w_e + gap
    axv = fig.add_axes([x_v / W, y_p / HT, (W - MARGIN - x_v) / W, h_p / HT])
    style(axv, "Bead volume fraction (gray: initial)")
    bins = np.arange(WX0, WX1 + 1e-9, 0.25e-3)
    mid = 0.5 * (bins[1:] + bins[:-1]) * 1e3
    axv.plot(mid, volume_fraction(pos_all[0][:, 0], run, bins), color="#6b7280", lw=1.2)
    vf_line, = axv.plot(mid, np.zeros_like(mid), color="#ffc34d", lw=2.0)
    axv.set_xlim(WX0 * 1e3, WX1 * 1e3)
    axv.set_ylim(0, 0.5)
    axv.set_xlabel("x (mm)", fontsize=12)

    fig.text(MARGIN / W, 0.972, "A shock wave hitting a curtain of glass beads", color=FG, fontsize=26, va="top",
             fontweight="bold")
    fig.text(MARGIN / W, 0.93, "Mach 1.66 into a 2 mm curtain of 115 µm soda-lime beads at 19% volume fraction, after "
             "the Sandia multiphase shock tube (Wagner et al., 2012)", color=DIM, fontsize=14, va="top")
    fig.text(MARGIN / W, 0.903, f"Numerical schlieren of the gas, {len(vel_all[0]):,} parcels coupled two ways "
             "as point particles (dilute model). Mallard, TENO5", color=DIM, fontsize=14, va="top")
    clock_text = fig.text(1 - MARGIN / W, 0.972, "", color=FG, fontsize=22, va="top", ha="right",
                          family="monospace")

    def draw(k):
        im.set_data(section(run["rho"][k], pos_all[k], vel_all[k]))
        p = run["p"][k]
        p_line.set_ydata(p / P1)
        x_beads = pos_all[k][:, 0] - X_CURTAIN
        curtain = (np.nanmin(x_beads), np.nanmax(x_beads))
        found = [(x, n) for x, n in shocks(section.xc, p, t_us[k], curtain) if WX0 + 4e-3 < x < WX1 - 4e-3]
        ytxt = section.y0 * 1e3 + 0.8
        for lab, item in zip(labels, found + [None] * 3):
            if item is None:
                lab.set_visible(False)
            else:
                lab.set_visible(True)
                lab.set_position((item[0] * 1e3, ytxt))
                lab.set_text(item[1])
        top = (section.y0 + PH * PX) * 1e3 - 0.8
        thick = THICKNESS * 1e3
        for line, txt, key, base, dx, name in ((exp_lines[0], exp_text[0], "upstream_mm", 0.0, -0.4, "upstream"),
                                               (exp_lines[1], exp_text[1], "downstream_mm", thick, 0.4,
                                                "downstream")):
            on = t_us[k] >= 0
            x = base + np.interp(max(t_us[k], 0.0), data["t_us"], data[key])
            line.set_visible(on)
            txt.set_visible(on)
            line.set_xdata([x, x])
            txt.set_position((x + dx, top))
            txt.set_text(f"experiment's\n{name} edge")
        sel = t_us[: k + 1] >= 0
        up_line.set_data(t_us[: k + 1][sel], e[: k + 1, 0][sel])
        dn_line.set_data(t_us[: k + 1][sel], e[: k + 1, 1][sel])
        vf_line.set_ydata(volume_fraction(pos_all[k][:, 0], run, bins))
        clock_text.set_text(f"t = {t_us[k]:4.0f} µs after impact")
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()

    def half(frame):
        return np.asarray(Image.fromarray(frame).resize((W // 2, HT // 2), Image.LANCZOS))

    if args.poster:
        k = int(np.argmin(np.abs(t_us - args.poster_t))) if args.poster_t is not None else len(t_us) - 1
        frame = draw(k)
        stem, ext = os.path.splitext(args.poster)
        imageio.imwrite(stem + ".png", frame)
        imageio.imwrite(args.poster, frame, quality=95)
        imageio.imwrite(stem + "_1080p" + ext, half(frame), quality=95)
    if args.out != "-":
        stem, ext = os.path.splitext(args.out)
        opts = dict(fps=args.fps, codec="libx264", quality=None, macro_block_size=8)
        params = ["-crf", str(args.crf), "-pix_fmt", "yuv420p", "-preset", "slow", "-movflags", "+faststart"]
        full = imageio.get_writer(args.out, ffmpeg_params=params, **opts)
        small = imageio.get_writer(stem + "_1080p" + ext, ffmpeg_params=params, **opts)
        frame = None
        for k in range(len(t_us)):
            frame = draw(k)
            full.append_data(frame)
            small.append_data(half(frame))
        last = half(frame)
        for _ in range(int(args.fps * 2)):
            full.append_data(frame)
            small.append_data(last)
        full.close()
        small.close()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("run")
    g = sub.add_parser("edges")
    g.add_argument("runs", nargs="+")
    g.add_argument("--plot")
    r = sub.add_parser("render")
    r.add_argument("out")
    r.add_argument("run")
    r.add_argument("--fps", type=float, default=24)
    r.add_argument("--poster")
    r.add_argument("--poster-t", type=float)
    r.add_argument("--crf", type=int, default=18)
    args = ap.parse_args()
    if args.cmd == "extract":
        extract(args.run)
    elif args.cmd == "edges":
        edges(args)
    else:
        render(args)


if __name__ == "__main__":
    main()
