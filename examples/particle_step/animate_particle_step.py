"""Animation and statistics of the particle_step runs (run_particle_step.py): the particles of each Stokes
number over the backward-facing step, over the vorticity of the gas.

    python examples/particle_step/animate_particle_step.py extract RUN_DIR   (writes RUN_DIR/frames.npz)
    python examples/particle_step/animate_particle_step.py profiles OUT.png RUN_DIR [RUN_DIR ...] [--from T]
    python examples/particle_step/animate_particle_step.py render OUT.mp4 RUN_DIR [RUN_DIR ...] [--fps 24]
        [--crf 18] [--poster OUT.jpg] [--poster-t T]

extract reads a run's frames (frames/frame_*.restart, which carry the particles, and the snapshots of the gas
frames/step_*.vtu) into one compact file: every particle's position and velocity by id, and the vorticity and
streamwise velocity of the gas on a uniform grid of the window. profiles prints and plots, at the stations
x / H = 2, 4, 6, 8 and 10 and from T (in U0 t / H since the seeding) on, the gas's mean streamwise velocity, each
run's particle mean streamwise velocity and its particle concentration against the inlet's, and the share of
the particles in the recirculation zone. render draws one panel per run on one clock, the first run's
vorticity under every panel (the particles do not act on the gas, so every run has the same flow), each
particle a round dot. Frames are drawn at 3840 x 2160; render writes OUT.mp4 at that size and OUT_1080p.mp4 at
half of it (H.264), and the poster as OUT.jpg, OUT.png and OUT_1080p.jpg (T in U0 t / H since the seeding;
OUT "-" skips the video).
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from mallard import restart, vtu  # noqa: E402

U0, TAU_F = 0.3, 5.0 / 0.3
X0, X1, Y0, Y1, H = -2.0, 20.0, 0.0, 2.5, 0.0125
STATIONS = (2.0, 4.0, 6.0, 8.0, 10.0)
BG = "#0d0f14"
FG = "#e8e8e8"


def stokes(run_dir):
    """The Stokes number tau / (5 H / U0) of a run's input (0 for tracers)."""
    text = open(os.path.join(run_dir, "input.toml")).read()
    if re.search(r'^type = "tracer"', text, re.M):
        return 0.0
    d = float(re.search(r"^diameter = ([0-9.eE+-]+)", text, re.M).group(1))
    rho_p = float(re.search(r"^density = ([0-9.eE+-]+)", text, re.M).group(1))
    mu = float(re.search(r"^mu = ([0-9.eE+-]+)", text, re.M).group(1))
    return rho_p * d * d / (18 * mu) / TAU_F


def extract(run_dir):
    from scipy.spatial import Delaunay

    files = sorted(glob.glob(os.path.join(run_dir, "frames", "frame_*.restart")))
    first = restart.read(files[0])
    n = int(first.arrays["PARTICLE_ID"].max()) + 1
    solid = "PARTICLE_V" in first.arrays
    t = np.empty(len(files))
    pos = np.full((len(files), n, 2), np.nan, dtype=np.float32)
    vel = np.full((len(files), n, 2), np.nan, dtype=np.float32)
    for k, f in enumerate(files):
        r = restart.read(f)
        ids = r.arrays["PARTICLE_ID"].astype(np.int64)
        t[k] = r.t
        pos[k, ids] = r.arrays["PARTICLE_X"].reshape(len(ids), 2)
        if solid:
            vel[k, ids] = r.arrays["PARTICLE_V"].reshape(len(ids), 2)
    out = dict(t=t, pos=pos, stokes=stokes(run_dir))
    if solid:
        out["vel"] = vel
    snaps = sorted(glob.glob(os.path.join(run_dir, "frames", "step_*.vtu")))
    if snaps:
        xs = np.arange(X0, X1 + H / 2, H)
        ys = np.arange(Y0 + H / 2, Y1, H)
        gx, gy = np.meshgrid(xs, ys)
        q = np.column_stack([gx.ravel(), gy.ravel()])
        weights = None
        g_t = np.empty(len(snaps))
        w = np.empty((len(snaps), ys.size, xs.size), dtype=np.float16)
        u = np.empty((len(snaps), ys.size, xs.size), dtype=np.float16)
        for k, f in enumerate(snaps):
            pts, conn, offs, _, a = vtu.read_vtu_cells(f)
            if weights is None:
                c = pts[conn.reshape(-1, 4)].mean(axis=1)[:, :2]
                tri = Delaunay(c)
                s = tri.find_simplex(q)
                b = np.einsum("nij,nj->ni", tri.transform[s, :2], q - tri.transform[s, 2])
                weights = (tri.simplices[s], np.column_stack([b, 1 - b.sum(axis=1)]),
                           (s < 0) | ((q[:, 0] < 0) & (q[:, 1] < 1)))
            idx, bw, outside = weights
            for name, dst in (("VORTICITY", w), ("U", u)):
                field = a[name] if a[name].ndim == 1 else a[name][:, 0]
                v = (field[idx] * bw).sum(axis=1)
                v[outside] = np.nan
                dst[k] = v.reshape(ys.size, xs.size)
            g_t[k] = a["TIME"][0] if np.ndim(a["TIME"]) else a["TIME"]
        out.update(g_t=g_t, w=w, u=u)
    np.savez_compressed(os.path.join(run_dir, "frames.npz"), **out)
    print(f"{run_dir}: {len(files)} particle frames, {n} particles, {len(snaps)} gas frames, St = {out['stokes']:g}")


def profiles(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [(d, np.load(os.path.join(d, "frames.npz"))) for d in args.runs]
    gas = next(r for _, r in runs if "u" in r)
    t0 = runs[0][1]["t"][0]
    ys = np.arange(Y0 + H / 2, Y1, H)
    xs = np.arange(X0, X1 + H / 2, H)
    keep = (gas["g_t"] - t0) * U0 >= args.start
    u_mean = np.nanmean(gas["u"][keep].astype(np.float32), axis=0)
    edges = np.linspace(0.0, 2.5, 26)
    yc = 0.5 * (edges[1:] + edges[:-1])
    half = 0.25
    fig, axes = plt.subplots(2, len(STATIONS), figsize=(3 * len(STATIONS), 7), sharey=True)
    colors = {0.0: "#1b9e77", 1.0: "#d95f02", 4.0: "#e7298a"}
    print(f"statistics from U0 t / H = {args.start:g} after the seeding")
    for k, xs_ in enumerate(STATIONS):
        i = int(np.argmin(np.abs(xs - xs_)))
        axes[0, k].plot(u_mean[:, i] / U0, ys, "k-", lw=2, label="gas")
        axes[0, k].axvline(0, color="0.7", lw=0.8)
        axes[0, k].set_title(f"x / H = {xs_:g}")
        axes[0, k].set_xlabel("U / U0")
        axes[1, k].set_xlabel("concentration / inlet")
    floor = u_mean[2, :]
    cross = np.flatnonzero((floor[:-1] < 0) & (floor[1:] >= 0) & (xs[:-1] > 0))
    reattach = xs[cross[-1] + 1] if len(cross) else np.nan
    print(f"gas: mean reattachment (U at y = {ys[2]:.3f} turns positive) at x / H = {reattach:.2f}")
    for d, r in runs:
        st = round(float(r["stokes"]), 6)
        sel = (r["t"] - t0) * U0 >= args.start
        pos = r["pos"][sel]
        if "vel" in r:
            vel = r["vel"][sel]
        else:  # tracers: the gas velocity, by central differences of the positions
            p = r["pos"]
            dt = np.gradient(r["t"])[:, None, None]
            vel = (np.gradient(p, axis=0) / dt)[sel]
        n_inlet = np.count_nonzero(np.isfinite(r["pos"][0, :, 0]))
        area_inlet = 19.0 * 1.46
        density_inlet = n_inlet / area_inlet
        frames = pos.shape[0]
        inside = np.isfinite(pos[..., 0])
        recirc = inside & (pos[..., 0] > 0) & (pos[..., 0] < reattach) & (pos[..., 1] < 1)
        downstream = inside & (pos[..., 0] > 0) & (pos[..., 0] < reattach)
        share = recirc.sum() / max(downstream.sum(), 1)
        print(f"St = {st:g}: {frames} frames; particles below the step's height over 0 < x < x_r: "
              f"{100 * share:.1f}% of those over 0 < x < x_r")
        line = []
        for k, xs_ in enumerate(STATIONS):
            m = inside & (np.abs(pos[..., 0] - xs_) < half)
            y = pos[..., 1][m]
            vx = vel[..., 0][m]
            count, _ = np.histogram(y, edges)
            total, _ = np.histogram(y, edges, weights=vx)
            conc = count / (frames * 2 * half * np.diff(edges)) / density_inlet
            mean = np.where(count >= 20, total / np.maximum(count, 1), np.nan) / U0
            axes[0, k].plot(mean, yc, "o-", ms=3, color=colors.get(st), label=f"St = {st:g}")
            axes[1, k].plot(conc, yc, "-", color=colors.get(st), label=f"St = {st:g}")
            low = yc < 0.5
            line.append(f"x/H={xs_:g}: V/U0 at y<0.5 {np.nanmean(mean[low]):+.3f}, conc {np.mean(conc[low]):.2f}")
        print("   " + "; ".join(line))
    axes[0, 0].set_ylabel("y / H")
    axes[1, 0].set_ylabel("y / H")
    axes[0, -1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("run")
    p = sub.add_parser("profiles")
    p.add_argument("out")
    p.add_argument("runs", nargs="+")
    p.add_argument("--from", dest="start", type=float, default=15.0)
    r = sub.add_parser("render")
    r.add_argument("out")
    r.add_argument("runs", nargs="+")
    r.add_argument("--fps", type=float, default=24)
    r.add_argument("--poster")
    r.add_argument("--poster-t", type=float)
    r.add_argument("--crf", type=int, default=18)
    args = ap.parse_args()
    if args.cmd == "extract":
        extract(args.run)
    elif args.cmd == "profiles":
        profiles(args)
    else:
        render(args)


COLORS = {0.0: "#9fe7ff", 1.0: "#ffc34d", 4.0: "#ff6f91"}
WX0, WX1, WY0, WY1 = -1.0, 14.0, 0.0, 2.5
W, HT = 3840, 2160  # the master frame; render also encodes a copy at half this size
MARGIN, GAP, TOP = 46, 26, 184
PW = W - 2 * MARGIN
PH = round(PW * (WY1 - WY0) / (WX1 - WX0) / 2) * 2
SS = 3  # supersampling of the particle dots and the step's edge
DOT_RADIUS = 1.5  # in master pixels
NX_SEED, LANE_ROWS = 760, 6


def hex_rgb(c):
    return np.array([int(c[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0


class Panels:
    """Panel images of PW x PH pixels: the vorticity, dimmed, under the particles drawn as round dots in the
    run's color, anti-aliased by supersampling, and the step's solid block."""

    def __init__(self):
        from scipy.ndimage import map_coordinates

        self.map_coordinates = map_coordinates
        px = (WX1 - WX0) / PW
        x = WX0 + (np.arange(PW) + 0.5) * px
        y = WY0 + (np.arange(PH) + 0.5) * px
        gx, gy = np.meshgrid(x, y)
        self.coords = np.array([(gy - (Y0 + H / 2)) / H, (gx - X0) / H])
        xs = WX0 + (np.arange(PW * SS) + 0.5) * px / SS
        ys = WY0 + (np.arange(PH * SS) + 0.5) * px / SS
        block = ((xs[None, :] < 0) & (ys[:, None] < 1)).astype(float)
        self.block = self.down(block)[..., None]
        r = DOT_RADIUS * SS
        k = int(np.ceil(r))
        dy, dx = np.mgrid[-k:k + 1, -k:k + 1]
        inside = dx ** 2 + dy ** 2 <= r * r
        self.offsets = np.column_stack([dy[inside], dx[inside]])
        self.scale = SS / px

    @staticmethod
    def down(a):
        return a.reshape(PH, SS, PW, SS).mean(axis=(1, 3))

    def dots(self, p):
        """Coverage of the dots of the particles p in each pixel, 0 to 1."""
        mask = np.zeros((PH * SS, PW * SS), dtype=bool)
        iy = np.floor((p[:, 1] - WY0) * self.scale).astype(np.int64)
        ix = np.floor((p[:, 0] - WX0) * self.scale).astype(np.int64)
        for oy, ox in self.offsets:
            jy, jx = iy + oy, ix + ox
            ok = (jy >= 0) & (jy < PH * SS) & (jx >= 0) & (jx < PW * SS)
            mask[jy[ok], jx[ok]] = True
        return self.down(mask.astype(np.float32))

    def __call__(self, w, pos, color):
        v = self.map_coordinates(np.nan_to_num(w.astype(np.float32)), self.coords, order=1, mode="nearest")
        bg = hex_rgb(BG)
        strength = np.clip(np.abs(v) / 2.5, 0, 1)[..., None] ** 0.8
        tone = np.where((v > 0)[..., None], hex_rgb("#c0392b"), hex_rgb("#2e6fd8"))
        img = bg + 0.5 * strength * (tone - bg)
        # alternate lanes of the seeding lattice's rows in two shades, as dye streaks
        lane = (np.arange(len(pos)) // NX_SEED // LANE_ROWS) % 2
        for shade, sel in ((0.5, lane == 1), (1.0, lane == 0)):
            a = self.dots(pos[sel & ~np.isnan(pos[:, 0])])[..., None]
            img = img * (1 - a) + shade * hex_rgb(color) * a
        img = img * (1 - self.block) + hex_rgb("#3a3f4b") * self.block
        return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)[::-1]


def label(st):
    return "Tracers, St = 0" if st == 0 else f"St = {st:g}"


def render(args):
    import imageio.v2 as imageio
    import matplotlib
    from PIL import Image

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [np.load(os.path.join(d, "frames.npz")) for d in args.runs]
    gas = next(r for r in runs if "w" in r)
    t0 = runs[0]["t"][0]
    clock = runs[0]["t"]
    dpi = 200
    fig = plt.figure(figsize=(W / dpi, HT / dpi), dpi=dpi)
    fig.patch.set_facecolor(BG)
    panels = Panels()
    slots, ims = [], []
    for i, r in enumerate(runs):
        xo, yo = MARGIN, HT - TOP - (i + 1) * PH - i * GAP
        st = round(float(r["stokes"]), 6)
        ims.append(fig.figimage(np.zeros((PH, PW, 3), dtype=np.uint8), xo=xo, yo=yo, origin="upper"))
        fig.text((xo + PW - 24) / W, (yo + PH - 16) / HT, label(st), color=COLORS.get(st, FG), fontsize=17,
                 va="top", ha="right", fontweight="bold", zorder=5)
        slots.append((r, st))
    fig.text(MARGIN / W, 1 - 34 / HT, "Solid particles over a backward-facing step", color=FG, fontsize=26,
             va="top", fontweight="bold")
    fig.text(MARGIN / W, 1 - 112 / HT, "Laminar inflow at Re_H = 3000, Mach 0.3, expansion 5:3 (after Fessler & "
             "Eaton): 45,600 particles per Stokes number St = τU0/5H, over the vorticity of the gas "
             "(red: counterclockwise). Mallard, TENO5, 29,280 cells",
             color="#b8bcc6", fontsize=13.5, va="top")
    clock_text = fig.text(1 - MARGIN / W, 1 - 34 / HT, "", color=FG, fontsize=20, va="top", ha="right",
                          family="monospace")

    def draw(k):
        t = clock[k]
        j = int(np.argmin(np.abs(gas["g_t"] - t)))
        for im, (r, st) in zip(ims, slots):
            kk = int(np.argmin(np.abs(r["t"] - t)))
            im.set_data(panels(gas["w"][j], r["pos"][kk], COLORS.get(st, FG)))
        clock_text.set_text(f"t U0/H = {(t - t0) * U0:5.1f}")
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()

    def half(frame):
        return np.asarray(Image.fromarray(frame).resize((W // 2, HT // 2), Image.LANCZOS))

    if args.poster:
        k = int(np.argmin(np.abs(clock - (t0 + args.poster_t / U0)))) if args.poster_t is not None else len(clock) - 1
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
        for k in range(len(clock)):
            frame = draw(k)
            full.append_data(frame)
            small.append_data(half(frame))
        last = half(frame)
        for _ in range(int(args.fps * 2)):
            full.append_data(frame)
            small.append_data(last)
        full.close()
        small.close()


if __name__ == "__main__":
    main()
