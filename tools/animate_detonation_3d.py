#!/usr/bin/env python3
"""Animate a 3D cellular detonation in a duct (examples/detonation_3d): pressure isosurfaces and soot foils.

    animate_detonation_3d.py INPUT FRAME_DIR OUTPUT_BASE [--levels 60 120 240] [--fps 15] [--hold 2]
        [--width 1920] [--gif-width 720] [--still-frame N] [--view-length 0.06]

FRAME_DIR holds the frames of tools/detonation_window.py (`run --frames`):
the pressure behind the front and P_MAX on the side walls. Each frame shows,
on the left, the detonation front as isosurfaces of pressure (kPa, colored
by level) in a cut-away of the duct whose two far walls (y = 0 and z = 0)
carry the numerical soot foil written so far, with the camera following the
front; on the right, the soot foils of all four walls unrolled over the
whole run so far (from the window's foil chunks, foil/ next to INPUT).
Writes OUTPUT_BASE.mp4 (H.264, CRF 18, holding the last frame for --hold
seconds), OUTPUT_BASE.gif (if --gif-width > 0) and OUTPUT_BASE_still.png
(the frame --still-frame, default the last). Needs pyvista.
"""
import argparse
import glob
import json
import os
import shutil
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from animate import write_gif, write_mp4
from detonation_window import Case

BG = "#0b1014"
FG = "#e6edf3"
WALLS = ("bottom", "back", "top", "front")
WALL_LABELS = {"bottom": "y = 0", "top": "y = W", "back": "z = 0", "front": "z = W"}


def soot_cmap():
    """Light paper, dark where the pressure peaked (as on a smoked foil)."""
    return matplotlib.colors.LinearSegmentedColormap.from_list("soot", ["#f2ead8", "#b9a888", "#5a4a36", "#120c06"])


def foil_history(case):
    state = case.load_state()
    chunks = []
    for c in state["chunks"]:
        with np.load(os.path.join(case.foil_dir, c["file"])) as z:
            chunks.append((c["first_cell"], {w: z[w] for w in z.files}))
    return chunks


def full_foils(chunks, frame):
    """Foils of the duct up to this frame: chunks dropped before it, then the frame's own walls."""
    shift = int(frame["shift_cells"])
    parts = {w: [] for w in WALLS}
    for first, strips in chunks:
        if first + next(iter(strips.values())).shape[0] <= shift:
            for w in WALLS:
                parts[w].append(strips[w])
    for w in WALLS:
        parts[w].append(frame["pmax_" + w])
    return {w: np.concatenate(parts[w]) for w in WALLS}


def render_3d(frame, foils, levels, view_length, size, p0, p_ref):
    dx = float(frame["dx"])
    ly, lz = float(frame["size"][1]), float(frame["size"][2])
    shift, i0, i_front = int(frame["shift_cells"]), int(frame["i0"]), int(frame["i_front"])
    p = frame["p_kpa"].astype(np.float32)
    x_front = (shift + i_front + 0.5) * dx
    x_lo = x_front - view_length
    pl = pv.Plotter(off_screen=True, window_size=size)
    pl.set_background(BG)
    grid = pv.ImageData(dimensions=np.array(p.shape) + 1, spacing=(dx, dx, dx), origin=((shift + i0) * dx, 0, 0))
    grid.cell_data["p"] = p.ravel(order="F")
    grid = grid.cell_data_to_point_data()
    cmap = plt.get_cmap("inferno")
    for k, level in enumerate(levels):
        iso = grid.contour([level], scalars="p")
        if iso.n_points:
            color = cmap(0.35 + 0.6 * k / max(1, len(levels) - 1))[:3]
            pl.add_mesh(iso, color=color, opacity=[0.35, 0.6, 1.0][min(k, 2)] if len(levels) > 1 else 1.0,
                        smooth_shading=True, specular=0.4)
    # Soot foils on the far walls, over the stretch of duct in view
    n = foils["bottom"].shape[0]
    i_lo = max(0, int(x_lo / dx))
    i_hi = min(n, shift + i_front + 1)
    soot = soot_cmap()
    norm = matplotlib.colors.LogNorm(p_ref[0], p_ref[1])
    for wall, normal in (("bottom", "y"), ("back", "z")):
        strip = foils[wall][i_lo:i_hi]
        if strip.size == 0:
            continue
        rgb = (soot(norm(np.clip(strip, p_ref[0], p_ref[1]) / 1000))[:, :, :3] * 255).astype(np.uint8)
        nx_s, nt = strip.shape
        if normal == "y":
            plane = pv.ImageData(dimensions=(nx_s + 1, 1, nt + 1), spacing=(dx, 1, dx), origin=(i_lo * dx, 0, 0))
        else:
            plane = pv.ImageData(dimensions=(nx_s + 1, nt + 1, 1), spacing=(dx, dx, 1), origin=(i_lo * dx, 0, 0))
        plane.cell_data["rgb"] = rgb.reshape(-1, 3, order="F")
        pl.add_mesh(plane, scalars="rgb", rgb=True, lighting=False)
    outline = pv.Box(bounds=(x_lo, x_front + 0.004, 0, ly, 0, lz))
    pl.add_mesh(outline, style="wireframe", color="#56606b", line_width=1)
    focus = np.array([x_front - 0.3 * view_length, 0.5 * ly, 0.5 * lz])
    pl.camera.focal_point = focus
    pl.camera.position = focus + np.array([0.55, 1.05, 0.85]) * view_length * 1.35
    pl.camera.up = (0, 0, 1)
    pl.enable_anti_aliasing("ssaa")
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


def compose(img, frame, foils, levels, p_ref, path, width):
    dx = float(frame["dx"])
    t = float(frame["t"])
    shift, i_front = int(frame["shift_cells"]), int(frame["i_front"])
    x_front = (shift + i_front + 0.5) * dx
    fig = plt.figure(figsize=(width / 100, width * 9 / 16 / 100), dpi=100, facecolor=BG)
    ax3 = fig.add_axes([0.0, 0.0, 0.62, 0.9])
    ax3.imshow(img)
    ax3.axis("off")
    fig.text(0.02, 0.95, "Cellular detonation in a 3 cm square duct, 2H$_2$-O$_2$-7Ar at 6.67 kPa", color=FG,
             fontsize=17, weight="bold")
    fig.text(0.02, 0.915, f"t = {t * 1e6:6.1f} µs    front at x = {x_front * 100:5.1f} cm    "
             f"pressure isosurfaces {', '.join(f'{v:g}' for v in levels)} kPa; soot foils on the far walls",
             color=FG, fontsize=11)
    n = foils["bottom"].shape[0]
    soot = soot_cmap()
    norm = matplotlib.colors.LogNorm(p_ref[0], p_ref[1])
    for k, wall in enumerate(WALLS):
        ax = fig.add_axes([0.645, 0.08 + 0.205 * (3 - k), 0.34, 0.17], facecolor=BG)
        ax.imshow(foils[wall].T / 1000, origin="lower", cmap=soot, norm=norm, aspect="auto",
                  extent=(0, n * dx * 100, 0, float(frame["size"][1]) * 100), interpolation="antialiased")
        ax.set_ylabel(WALL_LABELS[wall], color=FG, fontsize=9)
        ax.tick_params(colors=FG, labelsize=8)
        for s in ax.spines.values():
            s.set_color("#56606b")
        if k < 3:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("x (cm)", color=FG, fontsize=9)
    fig.text(0.645, 0.915, "Numerical soot foils (P$_{max}$) of the four walls", color=FG, fontsize=11)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("frame_dir")
    ap.add_argument("output_base")
    ap.add_argument("--levels", type=float, nargs="+", default=[60.0, 120.0, 240.0])
    ap.add_argument("--foil-range", type=float, nargs=2, default=[60.0, 600.0], help="P_MAX color range, kPa")
    ap.add_argument("--view-length", type=float, default=0.06)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--hold", type=float, default=2.0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--gif-width", type=int, default=720)
    ap.add_argument("--still-frame", type=int, default=-1)
    ap.add_argument("--every", type=int, default=1)
    args = ap.parse_args()
    case = Case(args.input)
    chunks = foil_history(case)
    files = sorted(glob.glob(os.path.join(args.frame_dir, "frame_*.npz")))[::args.every]
    tmp = tempfile.mkdtemp()
    size = (int(args.width * 0.62), int(args.width * 9 / 16 * 0.9))
    for k, f in enumerate(files):
        with np.load(f) as z:
            frame = {key: z[key] for key in z.files}
        foils = full_foils(chunks, frame)
        img = render_3d(frame, foils, args.levels, args.view_length, size, None, args.foil_range)
        compose(img, frame, foils, args.levels, args.foil_range, os.path.join(tmp, f"f{k:05d}.png"), args.width)
        print(f"{k + 1}/{len(files)}", end="\r", flush=True)
    last = len(files) - 1
    for h in range(int(args.hold * args.fps)):
        shutil.copy(os.path.join(tmp, f"f{last:05d}.png"), os.path.join(tmp, f"f{last + 1 + h:05d}.png"))
    still = args.still_frame if args.still_frame >= 0 else last
    shutil.copy(os.path.join(tmp, f"f{still:05d}.png"), args.output_base + "_still.png")
    write_mp4(os.path.join(tmp, "f%05d.png"), args.output_base + ".mp4", args.fps)
    if args.gif_width > 0:
        write_gif(os.path.join(tmp, "f%05d.png"), args.output_base + ".gif", args.fps, args.gif_width)
    shutil.rmtree(tmp)
    print(f"\nwrote {args.output_base}.mp4")


if __name__ == "__main__":
    main()
