"""Animation of the particle_wake runs (run_particle_wake.py): the particles of each Stokes number in the
vortex street, over the vorticity of the gas.

    python examples/particle_wake/animate_particle_wake.py extract RUN_DIR   (writes RUN_DIR/frames.npz)
    python examples/particle_wake/animate_particle_wake.py render OUT.mp4 RUN_DIR [RUN_DIR ...] [--fps 24]
        [--crf 18] [--poster OUT.jpg] [--poster-t T]

extract reads a run's frames (frames/frame_*.restart, which carry the particles, and the vorticity snapshots
frames/wake_*.vtu) into one compact file: every particle's position by id, and the vorticity on a uniform
grid of the window. render draws one panel per run on one clock, the first run's vorticity under every
panel (the particles do not act on the gas, so every run has the same flow), each particle a round dot. Frames
are drawn at 3840 x 2160; render writes OUT.mp4 at that size and OUT_1080p.mp4 at half of it (H.264), and the
poster as OUT.jpg, OUT.png and OUT_1080p.jpg (T in U t / D since the seeding; OUT "-" skips the video).
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from mallard import restart, vtu  # noqa: E402

X0, X1, Y0, Y1, H = -3.0, 15.0, -4.0, 4.0, 0.0125
BG = "#0d0f14"
FG = "#e8e8e8"


def stokes(run_dir):
    """The Stokes number tau U / D of a run's input (0 for tracers)."""
    text = open(os.path.join(run_dir, "input.toml")).read()
    if re.search(r'^type = "tracer"', text, re.M):
        return 0.0
    d = float(re.search(r"^diameter = ([0-9.eE+-]+)", text, re.M).group(1))
    rho_p = float(re.search(r"^density = ([0-9.eE+-]+)", text, re.M).group(1))
    mu = float(re.search(r"^mu = ([0-9.eE+-]+)", text, re.M).group(1))
    return rho_p * d * d / (18 * mu) * 0.2


def extract(run_dir):
    from scipy.spatial import Delaunay

    files = sorted(glob.glob(os.path.join(run_dir, "frames", "frame_*.restart")))
    first = restart.read(files[0])
    n = int(first.arrays["PARTICLE_ID"].max()) + 1
    t = np.empty(len(files))
    pos = np.full((len(files), n, 2), np.nan, dtype=np.float32)
    for k, f in enumerate(files):
        r = restart.read(f)
        ids = r.arrays["PARTICLE_ID"].astype(np.int64)
        t[k] = r.t
        pos[k, ids] = r.arrays["PARTICLE_X"].reshape(len(ids), 2)
    out = dict(t=t, pos=pos, stokes=stokes(run_dir))
    snaps = sorted(glob.glob(os.path.join(run_dir, "frames", "wake_*.vtu")))
    if snaps:
        xs = np.arange(X0, X1 + H / 2, H)
        ys = np.arange(Y0, Y1 + H / 2, H)
        gx, gy = np.meshgrid(xs, ys)
        q = np.column_stack([gx.ravel(), gy.ravel()])
        weights = None
        w_t, w = np.empty(len(snaps)), np.empty((len(snaps), ys.size, xs.size), dtype=np.float16)
        for k, f in enumerate(snaps):
            pts, conn, offs, _, a = vtu.read_vtu_cells(f)
            if weights is None:
                c = pts[conn.reshape(-1, 4)].mean(axis=1)[:, :2]
                tri = Delaunay(c)
                s = tri.find_simplex(q)
                b = np.einsum("nij,nj->ni", tri.transform[s, :2], q - tri.transform[s, 2])
                weights = (tri.simplices[s], np.column_stack([b, 1 - b.sum(axis=1)]),
                           (s < 0) | (np.hypot(q[:, 0], q[:, 1]) < 0.5))
            idx, bw, outside = weights
            v = (a["VORTICITY"][idx] * bw).sum(axis=1)
            v[outside] = np.nan
            w_t[k] = a["TIME"][0] if np.ndim(a["TIME"]) else a["TIME"]
            w[k] = v.reshape(ys.size, xs.size)
        out.update(w_t=w_t, w=w)
    np.savez_compressed(os.path.join(run_dir, "frames.npz"), **out)
    print(f"{run_dir}: {len(files)} particle frames, {n} particles, {len(snaps)} vorticity frames, "
          f"St = {out['stokes']:g}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("run")
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
    else:
        render(args)


COLORS = {0.0: "#9fe7ff", 0.25: "#7dffb0", 1.0: "#ffc34d", 4.0: "#ff6f91"}
WX0, WX1, WY0, WY1 = -1.5, 11.5, -3.25, 3.25
W, HT = 3840, 2160  # the master frame; render also encodes a copy at half this size
MARGIN, GAP, TOP = 46, 46, 222
PW = (W - 2 * MARGIN - GAP) // 2 // 2 * 2
PH = PW // 2
SS = 3  # supersampling of the particle dots and the cylinder's edge
DOT_RADIUS = 1.5  # in master pixels
NX_SEED, LANE_ROWS = 670, 8


def hex_rgb(c):
    return np.array([int(c[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0


class Panels:
    """Panel images of PW x PH pixels: the vorticity, dimmed, under the particles drawn as round dots in the
    run's color, anti-aliased by supersampling."""

    def __init__(self):
        from scipy.ndimage import map_coordinates

        self.map_coordinates = map_coordinates
        px = (WX1 - WX0) / PW
        x = WX0 + (np.arange(PW) + 0.5) * px
        y = WY0 + (np.arange(PH) + 0.5) * px
        gx, gy = np.meshgrid(x, y)
        self.coords = np.array([(gy - Y0) / H, (gx - X0) / H])
        xs = WX0 + (np.arange(PW * SS) + 0.5) * px / SS
        ys = WY0 + (np.arange(PH * SS) + 0.5) * px / SS
        disc = (xs[None, :] ** 2 + ys[:, None] ** 2 < 0.25).astype(float)
        self.cylinder = self.down(disc)[..., None]
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
        v = self.map_coordinates(np.nan_to_num(w.astype(np.float32)), self.coords, order=1)
        bg = hex_rgb(BG)
        strength = np.clip(np.abs(v) / 1.2, 0, 1)[..., None] ** 0.7
        tone = np.where((v > 0)[..., None], hex_rgb("#c0392b"), hex_rgb("#2e6fd8"))
        img = bg + 0.6 * strength * (tone - bg)
        # alternate lanes of the seeding lattice's rows in two shades, as dye streaks
        lane = (np.arange(len(pos)) // NX_SEED // LANE_ROWS) % 2
        for shade, sel in ((0.5, lane == 1), (1.0, lane == 0)):
            a = self.dots(pos[sel & ~np.isnan(pos[:, 0])])[..., None]
            img = img * (1 - a) + shade * hex_rgb(color) * a
        img = img * (1 - self.cylinder) + hex_rgb("#3a3f4b") * self.cylinder
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
        c, rr = i % 2, i // 2
        xo, yo = MARGIN + c * (PW + GAP), HT - TOP - (rr + 1) * PH - rr * GAP
        st = round(float(r["stokes"]), 6)
        ims.append(fig.figimage(np.zeros((PH, PW, 3), dtype=np.uint8), xo=xo, yo=yo, origin="upper"))
        fig.text((xo + 24) / W, (yo + PH - 18) / HT, label(st), color=COLORS.get(st, FG), fontsize=17,
                 va="top", fontweight="bold", zorder=5)
        slots.append((r, st))
    fig.text(MARGIN / W, 0.965, "Solid particles in a Kármán vortex street", color=FG, fontsize=26, va="top",
             fontweight="bold")
    fig.text(MARGIN / W, 0.918, "Cylinder at Re = 100, Mach 0.2: 134,000 particles per Stokes number St = τU/D, "
             "seeded upstream, over the vorticity of the gas (red: counterclockwise). Mallard, TENO5, 246,000 cells",
             color="#b8bcc6", fontsize=13.5, va="top")
    clock_text = fig.text(1 - MARGIN / W, 0.965, "", color=FG, fontsize=20, va="top", ha="right",
                          family="monospace")

    def draw(k):
        t = clock[k]
        j = int(np.argmin(np.abs(gas["w_t"] - t)))
        for im, (r, st) in zip(ims, slots):
            kk = int(np.argmin(np.abs(r["t"] - t)))
            im.set_data(panels(gas["w"][j], r["pos"][kk], COLORS.get(st, FG)))
        clock_text.set_text(f"t U/D = {(t - t0) * 0.2:5.1f}")
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()

    def half(frame):
        return np.asarray(Image.fromarray(frame).resize((W // 2, HT // 2), Image.LANCZOS))

    if args.poster:
        k = int(np.argmin(np.abs(clock - (t0 + args.poster_t / 0.2)))) if args.poster_t is not None else len(clock) - 1
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
