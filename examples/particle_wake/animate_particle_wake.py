"""Animation of the particle_wake runs (run_particle_wake.py): the particles of each Stokes number in the
vortex street, over the vorticity of the gas.

    python examples/particle_wake/animate_particle_wake.py extract RUN_DIR   (writes RUN_DIR/frames.npz)
    python examples/particle_wake/animate_particle_wake.py render OUT.mp4 RUN_DIR [RUN_DIR ...] [--fps 24]
        [--poster OUT.jpg] [--poster-t T]

extract reads a run's frames (frames/frame_*.restart, which carry the particles, and the vorticity snapshots
frames/wake_*.vtu) into one compact file: every particle's position by id, and the vorticity on a uniform
grid of the window. render draws one panel per run on one clock, the first run's vorticity under every
panel (the particles do not act on the gas, so every run has the same flow). The MP4 is H.264 (CRF 18).
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
    args = ap.parse_args()
    if args.cmd == "extract":
        extract(args.run)
    else:
        render(args)


COLORS = {0.0: "#9fe7ff", 0.25: "#7dffb0", 1.0: "#ffc34d", 4.0: "#ff6f91"}
WX0, WX1, WY0, WY1 = -1.5, 11.5, -3.25, 3.25
PX = 0.0125
SPACING = 23.4 / 669  # of the seeding lattice
NX_SEED, LANE_ROWS = 670, 8


def hex_rgb(c):
    return np.array([int(c[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0


def panel(w, pos, color):
    """RGB image of one panel: the vorticity, dimmed, under the particle density in the run's color."""
    from scipy.ndimage import gaussian_filter

    nx, ny = int(round((WX1 - WX0) / PX)), int(round((WY1 - WY0) / PX))
    i0, j0 = int(round((WX0 - X0) / H)), int(round((WY0 - Y0) / H))
    v = w[j0:j0 + ny, i0:i0 + nx].astype(float)
    inside = np.isnan(v)
    bg = hex_rgb(BG)
    strength = np.clip(np.abs(np.nan_to_num(v)) / 1.2, 0, 1)[..., None] ** 0.7
    tone = np.where((np.nan_to_num(v) > 0)[..., None], hex_rgb("#c0392b"), hex_rgb("#2e6fd8"))
    img = bg + 0.6 * strength * (tone - bg)
    # alternate lanes of the seeding lattice's rows in two shades, as dye streaks
    lane = (np.arange(len(pos)) // NX_SEED // LANE_ROWS) % 2
    for shade, sel in ((1.0, lane == 0), (0.5, lane == 1)):
        p = pos[sel & ~np.isnan(pos[:, 0])]
        count, _, _ = np.histogram2d(p[:, 1], p[:, 0], bins=[ny, nx], range=[[WY0, WY1], [WX0, WX1]])
        # particles per seeded area (1: the density of the lattice), smoothed over the lattice spacing
        density = gaussian_filter(count, 2.2) * (SPACING / PX) ** 2
        a = (1 - np.exp(-0.6 * density))[..., None]
        img = img * (1 - a) + shade * hex_rgb(color) * a
    img[inside] = hex_rgb("#3a3f4b")
    return img


def label(st):
    return "Tracers, St = 0" if st == 0 else f"St = {st:g}"


def render(args):
    import imageio.v2 as imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [np.load(os.path.join(d, "frames.npz")) for d in args.runs]
    gas = next(r for r in runs if "w" in r)
    t0 = runs[0]["t"][0]
    clock = runs[0]["t"]
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor(BG)
    cols, rows = 2, -(-len(runs) // 2)
    left, right, bottom, top, gap = 0.012, 0.988, 0.02, 0.885, 0.012
    pw, ph = (right - left - gap) / cols, (top - bottom - gap * (rows - 1)) / rows
    axes, ims, stamps = [], [], []
    for i, r in enumerate(runs):
        c, rr = i % cols, i // cols
        ax = fig.add_axes([left + c * (pw + gap), top - (rr + 1) * ph - rr * gap, pw, ph])
        ax.set_axis_off()
        st = round(float(r["stokes"]), 6)
        ims.append(ax.imshow(np.zeros((2, 2, 3)), origin="lower", extent=[WX0, WX1, WY0, WY1], aspect="equal",
                             interpolation="antialiased"))
        ax.text(0.012, 0.965, label(st), transform=ax.transAxes, color=COLORS.get(st, FG), fontsize=17,
                va="top", fontweight="bold")
        axes.append((ax, r, st))
    fig.text(0.012, 0.965, "Solid particles in a Kármán vortex street", color=FG, fontsize=26, va="top",
             fontweight="bold")
    fig.text(0.012, 0.918, "Cylinder at Re = 100, Mach 0.2: 134,000 particles per Stokes number St = τU/D, "
             "seeded upstream, over the vorticity of the gas (red: counterclockwise). Mallard, TENO5, 246,000 cells",
             color="#b8bcc6", fontsize=13.5, va="top")
    clock_text = fig.text(0.988, 0.965, "", color=FG, fontsize=20, va="top", ha="right", family="monospace")

    def draw(k):
        t = clock[k]
        j = int(np.argmin(np.abs(gas["w_t"] - t)))
        for im, (ax, r, st) in zip(ims, axes):
            kk = int(np.argmin(np.abs(r["t"] - t)))
            im.set_data(panel(gas["w"][j], r["pos"][kk], COLORS.get(st, FG)))
        clock_text.set_text(f"t U/D = {(t - t0) * 0.2:5.1f}")
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()

    if args.poster:
        k = int(np.argmin(np.abs(clock - (t0 + args.poster_t / 0.2)))) if args.poster_t is not None else len(clock) - 1
        imageio.imwrite(args.poster, draw(k), quality=92)
    if args.out != "-":
        writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=None,
                                    ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p", "-preset", "slow"],
                                    macro_block_size=8)
        for k in range(len(clock)):
            writer.append_data(draw(k))
        for _ in range(int(args.fps * 2)):
            writer.append_data(draw(len(clock) - 1))
        writer.close()


if __name__ == "__main__":
    main()
