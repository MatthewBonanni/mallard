#!/usr/bin/env python3
"""Animate the shock / helium-bubble interaction (examples/shock_bubble_3d).

    animate_shock_bubble.py INPUT FRAME_DIR probe_axis.csv OUTPUT_BASE [--fps 15] [--hold 2]
        [--width 1920] [--gif-width 720] [--still-frame N] [--orbit 60]

FRAME_DIR holds the frames of tools/shock_bubble.py (`frames`). Each frame
shows the helium (isosurface Y_HE = 0.4, mirrored from the computed quarter
into the full bubble) with the vortex sheet and ring (isosurfaces of the
vorticity magnitude, colored by it) under a slowly orbiting camera; the
numerical schlieren and the helium on the symmetry plane z = 0, mirrored
about the axis; and the x-t diagram of the axis (upstream interface and
air jet, downstream interface, vortex ring) traced over the lines of the
velocities measured by Haas & Sturtevant (1987), in the experiment's units.
Writes OUTPUT_BASE.mp4 (H.264, CRF 18, holding the last frame for --hold
seconds), OUTPUT_BASE.gif (if --gif-width > 0) and OUTPUT_BASE_still.png.
Needs pyvista.
"""
import argparse
import glob
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
from shock_bubble import D_EXPERIMENT, HS_SPHERE, crossings, geometry, read_probe

BG = "#0b1014"
FG = "#e6edf3"


def mirror(a, axes):
    """Mirror a quarter-domain array (x, y, z) across y = 0 and/or z = 0 into the full section."""
    for ax in axes:
        a = np.concatenate([np.flip(a, axis=ax), a], axis=ax)
    return a


def render_3d(frame, size, azimuth, x_range, vort_level):
    dx = float(frame["dx"])
    lo = frame["lo"]
    y_he = mirror(frame["y_he"].astype(np.float32), (1, 2))
    vort = mirror(frame["vort"].astype(np.float32), (1, 2))
    ny, nz = y_he.shape[1] // 2, y_he.shape[2] // 2
    origin = (lo[0] * dx, -(lo[1] + ny) * dx, -(lo[2] + nz) * dx)
    grid = pv.ImageData(dimensions=np.array(y_he.shape) + 1, spacing=(dx, dx, dx), origin=origin)
    grid.cell_data["y"] = y_he.ravel(order="F")
    grid.cell_data["w"] = vort.ravel(order="F")
    grid = grid.cell_data_to_point_data()
    pl = pv.Plotter(off_screen=True, window_size=size)
    pl.set_background(BG)
    he = grid.contour([0.4], scalars="y")
    if he.n_points:
        pl.add_mesh(he, color="#7fd3ff", opacity=0.35, smooth_shading=True, specular=0.5)
    w = grid.contour([vort_level], scalars="w")
    if w.n_points:
        pl.add_mesh(w, scalars="y", cmap="magma", clim=(0, 0.7), smooth_shading=True, show_scalar_bar=False,
                    specular=0.3)
    d = float(frame["d"])
    center = np.array([0.5 * (x_range[0] + x_range[1]), 0.0, 0.0])
    r = 3.6 * d
    a = np.radians(azimuth)
    pl.camera.focal_point = center
    pl.camera.position = center + np.array([-0.15 * r, -r * np.cos(a), r * np.sin(a)])
    pl.camera.up = (0, -np.sin(a), np.cos(a))
    pl.camera.view_angle = 30
    pl.enable_anti_aliasing("ssaa")
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("frame_dir")
    ap.add_argument("probe")
    ap.add_argument("output_base")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--hold", type=float, default=2.0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--gif-width", type=int, default=720)
    ap.add_argument("--still-frame", type=int, default=-1)
    ap.add_argument("--orbit", type=float, default=60.0, help="degrees of camera orbit over the animation")
    ap.add_argument("--vort", type=float, default=15.0, help="vorticity isosurface, units of u_p / D")
    args = ap.parse_args()
    case = Case(args.input)
    xb, d, xs, ws = geometry(case)
    scale = D_EXPERIMENT / d
    t_hit = (xb - d / 2 - xs) / ws
    files = sorted(glob.glob(os.path.join(args.frame_dir, "frame_*.npz")))
    probe = read_probe(args.probe)
    pt = np.array(sorted(probe))
    ui, di = [], []
    for t in pt:
        a = probe[t]
        c = crossings(a[:, 0], a[:, 1], 0.5 * a[:, 1].max()) if a[:, 1].max() > 0.1 else np.array([])
        ui.append((c.min() - xb) * scale * 1e3 if c.size else np.nan)
        di.append((c.max() - xb) * scale * 1e3 if c.size else np.nan)
    pt_ms = (pt - t_hit) * scale * 1e3
    ui, di = np.array(ui), np.array(di)
    ring_t, ring_x = [], []
    for f in files:
        with np.load(f) as z:
            ring_t.append((float(z["t"]) - t_hit) * scale * 1e3)
            ring_x.append((float(z["ring"][0]) - xb) * scale * 1e3)
    ring_t, ring_x = np.array(ring_t), np.array(ring_x)
    t_end = pt_ms.max()
    tmp = tempfile.mkdtemp()
    size = (int(args.width * 0.5), int(args.width * 9 / 16 * 0.86))
    lx = case.size[0]
    for k, f in enumerate(files):
        with np.load(f) as z:
            frame = {key: z[key] for key in z.files}
        t_ms = (float(frame["t"]) - t_hit) * scale * 1e3
        xs_box = (frame["lo"][0] * frame["dx"], (frame["lo"][0] + frame["y_he"].shape[0]) * frame["dx"])
        img = render_3d(frame, size, 15 + args.orbit * k / max(1, len(files) - 1), xs_box, args.vort)
        fig = plt.figure(figsize=(args.width / 100, args.width * 9 / 16 / 100), dpi=100, facecolor=BG)
        fig.text(0.02, 0.95, "Mach 1.25 shock on a helium bubble in air (Haas & Sturtevant 1987), "
                 "Navier-Stokes at Re = 1.5$\\times$10$^3$", color=FG, fontsize=17, weight="bold")
        fig.text(0.02, 0.915, f"t = {t_ms:5.3f} ms (experiment scale, 4.5 cm bubble)    helium: Y = 0.4 surface; "
                 f"vortex sheet and ring: |$\\omega$| = {args.vort:g} u$_p$/D, colored by helium fraction",
                 color=FG, fontsize=11)
        ax3 = fig.add_axes([0.0, 0.0, 0.5, 0.88])
        ax3.imshow(img)
        ax3.axis("off")
        # Schlieren and helium on the symmetry plane, mirrored about the axis
        sch = frame["schlieren"].astype(np.float32)
        sch = np.concatenate([np.flip(sch, 1), sch], 1)
        hel = frame["y_he_plane"].astype(np.float32)
        hel = np.concatenate([np.flip(hel, 1), hel], 1)
        dxm = float(frame["dx"]) * scale * 1e3
        ny = sch.shape[1] // 2
        ext = (-xb * scale * 1e3, (lx - xb) * scale * 1e3, -ny * dxm, ny * dxm)
        axs = fig.add_axes([0.52, 0.53, 0.46, 0.35], facecolor=BG)
        axs.imshow(np.exp(-sch.T / 4.0), origin="lower", cmap="gray", extent=ext, vmin=0, vmax=1, aspect="equal")
        axs.contour(np.linspace(ext[0], ext[1], hel.shape[0]), np.linspace(ext[2], ext[3], hel.shape[1]), hel.T,
                    [0.4], colors=["#7fd3ff"], linewidths=0.8)
        axs.set_xlim(-40, (lx - xb) * scale * 1e3)
        axs.set_title("numerical schlieren and helium (Y = 0.4) on the symmetry plane", color=FG, fontsize=10)
        axs.tick_params(colors=FG, labelsize=8)
        axs.set_xlabel("x (mm, experiment scale)", color=FG, fontsize=9)
        # x-t diagram on the axis
        axt = fig.add_axes([0.56, 0.08, 0.40, 0.36], facecolor=BG)
        sel = pt_ms <= t_ms
        axt.plot(pt_ms[sel], ui[sel], color="#ffb000", lw=2, label="upstream interface / air jet")
        axt.plot(pt_ms[sel], di[sel], color="#7fd3ff", lw=2, label="downstream interface")
        rs = ring_t <= t_ms
        axt.plot(ring_t[rs & (ring_t > HS_SPHERE["t_v"])], ring_x[rs & (ring_t > HS_SPHERE["t_v"])], "o",
                 color="#ff5c8a", ms=3, label="vortex ring core")
        tt = np.array([0, t_end])
        x0 = -d / 2 * scale * 1e3
        axt.plot(tt, x0 + HS_SPHERE["V_ui"] * tt, "--", color="#ffb000", lw=0.8, alpha=0.7,
                 label="H&S: V$_{ui}$ = 190 m/s, V$_v$ = 165 m/s")
        x_ring0 = np.interp(HS_SPHERE["t_v"], ring_t, ring_x) if ring_t.size else 0.0
        tv = np.array([HS_SPHERE["t_v"], t_end])
        axt.plot(tv, x_ring0 + HS_SPHERE["V_v"] * (tv - HS_SPHERE["t_v"]), "--", color="#ff5c8a", lw=0.8, alpha=0.7)
        axt.set_xlim(0, t_end)
        axt.set_ylim(-30, (lx - xb) * scale * 1e3)
        axt.set_xlabel("t (ms)", color=FG, fontsize=9)
        axt.set_ylabel("x on the axis (mm)", color=FG, fontsize=9)
        axt.tick_params(colors=FG, labelsize=8)
        for s in list(axt.spines.values()) + list(axs.spines.values()):
            s.set_color("#56606b")
        axt.legend(fontsize=7, loc="upper left", facecolor=BG, labelcolor=FG, edgecolor="#56606b")
        fig.savefig(os.path.join(tmp, f"f{k:05d}.png"), facecolor=BG)
        plt.close(fig)
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
