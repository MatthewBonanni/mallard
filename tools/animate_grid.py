#!/usr/bin/env python3
"""Animate several Mallard cases side by side in one figure, synchronized in
normalized time t / t_end, and write an MP4 and a looping GIF.

    animate_grid.py CONFIG.json OUTPUT_STEM [--workers 4]

CONFIG lists rows of panels; the panels of a row share one height and fill the
figure width in proportion to their aspect ratios:

    {"width": 6, "dpi": 300, "fps": 30, "gif_width": 1500, "gif_colors": 64,
     "footer": "...",
     "rows": [[{"series": "runs/dmr/solut", "title": "Double Mach reflection",
                "xlim": [0, 3]}],
              [{"series": "runs/riemann/solut", "title": "2D Riemann problem"},
               {"type": "wall", "series": "runs/vst/solut", "glob": "vst_*.vtu",
                "xlim": [0.45, 1], "ylim": [0, 130], "title": "Wall density",
                "reference": {"label": "Zhou et al.", "points": [[0.3, 39.8], ...]}}]]}

Field panels ("type": "field", the default) show density (or "var") in color,
shaded by a numerical schlieren. Wall panels plot "var" along the first row of
cells of a Cartesian mesh (y = min) against reference points. Frames panels
show a sequence of pre-rendered images ("series" holds 00000.png, 00001.png,
... and optionally times.txt), e.g. 3D views. Curve panels trace a time series
("data": CSV with columns t and y) up to the current time, marked by a
vertical line, over reference curves ("reference": {"label", "file": a CSV
of t and y, optional "color", "alpha", "lw", "ls"}, or a list of them);
"t_end" maps the normalized time to t. Detonation panels show a 2D detonation run's pressure
over the channel and, below it, the numerical soot foil (P_MAX) behind the
front ("series": the run's .pvd; "xlim" in mesh units). "gif_width" and
"gif_colors" shrink the GIF, "gif_dither" sets its dither (ffmpeg paletteuse),
"mp4_width" scales the MP4, and "background" sets the figure color (to match
pre-rendered frames).

Every frame is one snapshot of each case: a case with N snapshots is sampled
at the nearest normalized time, so cases written with the same number of
snapshots line up exactly. The last frame is held for --hold seconds by
lengthening its GIF delay rather than by repeating it.
"""
import argparse
import glob
import json
import os
import re
import shutil
import tempfile
from multiprocessing import Pool

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from animate import Rasterizer, pixel_to_cell, schlieren, write_gif, write_mp4
from animate_detonation_2d import soot
from mallard_vtu import grid_fields, read_vtu
from soot_foil import front_position

BG = "#101014"
FG = "#e8e8e8"
DIM = "#9a9a9a"


def snapshot_files(panel):
    if panel.get("type") == "detonation":
        base = os.path.dirname(panel["series"])
        return [os.path.join(base, f) for f in re.findall(r'file="([^"]+)"', open(panel["series"]).read())]
    pattern = "*.png" if panel.get("type") == "frames" else panel.get("glob", "*.vtu")
    files = sorted(glob.glob(os.path.join(panel["series"], pattern)))
    if not files:
        raise SystemExit(f"no snapshots in {panel['series']}")
    return files


def panel_aspect(panel):
    if panel.get("type", "field") in ("wall", "curve"):
        return panel.get("aspect", 1.0)
    if panel.get("type") == "detonation":
        _, x, y, _ = grid_fields(snapshot_files(panel)[0], ["P"])
        x0, x1 = panel.get("xlim") or (x[0], x[-1])
        return (x1 - x0) / ((y[-1] - y[0] + y[1] - y[0]) * (2 + DetonationPanel.GAP))
    if panel.get("type") == "frames":
        h, w = imageio.imread(snapshot_files(panel)[0]).shape[:2]
        return w / h
    pts = read_vtu(snapshot_files(panel)[0])[0]
    x0, x1 = panel.get("xlim") or (pts[:, 0].min(), pts[:, 0].max())
    y0, y1 = panel.get("ylim") or (pts[:, 1].min(), pts[:, 1].max())
    return (x1 - x0) / (y1 - y0)


def layout(config):
    """Axes rectangles (figure fractions) and the figure height in inches."""
    width = config["width"]
    pad, title_h = 0.04, config.get("title_height", 0.17)
    footer_h = 0.2 if config.get("footer") else 0.05
    rows = []
    for row in config["rows"]:
        aspects = [panel_aspect(p) for p in row]
        h = (width - pad * (len(row) + 1)) / sum(aspects)
        rows.append((aspects, h))
    height = sum(h + title_h + pad for _, h in rows) + footer_h
    rects, top = [], height
    for (aspects, h), row in zip(rows, config["rows"]):
        top -= title_h
        x = pad
        for a, p in zip(aspects, row):
            rects.append((p, [x / width, (top - h) / height, a * h / width, h / height]))
            x += a * h + pad
        top -= h + pad
    return rects, height, footer_h


class FieldPanel:
    def __init__(self, panel, rect, fig_px):
        self.p = panel
        self.files = snapshot_files(panel)
        pts, tris, tri_cell, _ = read_vtu(self.files[0])
        px = max(rect[2] * fig_px[0], rect[3] * fig_px[1])
        res = int(panel.get("supersample", 1.5) * px)
        cell, self.valid, self.extent = pixel_to_cell(pts, tris, tri_cell, res, panel.get("xlim"), panel.get("ylim"))
        self.raster = Rasterizer(pts, tris, tri_cell, cell, self.valid, self.extent, panel.get("interp", "nearest"))
        var = panel.get("var", "RHO")
        lo, hi = np.inf, -np.inf
        for f in self.files[::max(1, len(self.files) // 20)] + [self.files[-1]]:
            img = self.raster(read_vtu(f)[3][var])
            lo, hi = min(lo, np.nanmin(img)), max(hi, np.nanmax(img))
        self.norm = Normalize(*(panel.get("range") or (lo, hi)))
        self.cmap = plt.get_cmap(panel.get("cmap", "turbo"))

    def draw(self, ax, k, n):
        f = self.files[round(k * (len(self.files) - 1) / max(n - 1, 1))]
        d = read_vtu(f)[3]
        img = self.raster(d[self.p.get("var", "RHO")])
        rgb = self.cmap(self.norm(np.nan_to_num(img, nan=self.norm.vmin)))[..., :3]
        shade = schlieren(self.raster(d["RHO"]), self.valid)
        rgb *= (1 - self.p.get("shading", 0.6)) + self.p.get("shading", 0.6) * shade[..., None]
        ax.imshow(rgb, origin="lower", extent=self.extent, interpolation="antialiased")
        ax.set_xlim(self.extent[:2])
        ax.set_ylim(self.extent[2:])
        return d.get("TIME", np.nan)


class WallPanel:
    def __init__(self, panel, rect, fig_px):
        self.p = panel
        self.files = snapshot_files(panel)
        pts, tris, tri_cell, _ = read_vtu(self.files[0])
        n_cells = tri_cell.max() + 1
        c = np.zeros((n_cells, 2))
        np.add.at(c, tri_cell, pts[tris].mean(axis=1))
        c /= np.bincount(tri_cell, minlength=n_cells)[:, None]
        y0 = c[:, 1].min()
        row = np.nonzero(c[:, 1] < y0 + 1e-6 * (c[:, 1].max() - y0) + 1e-12)[0]
        self.cells = row[np.argsort(c[row, 0])]
        self.x = c[self.cells, 0]

    def draw(self, ax, k, n):
        f = self.files[round(k * (len(self.files) - 1) / max(n - 1, 1))]
        d = read_vtu(f)[3]
        ax.plot(self.x, d[self.p.get("var", "RHO")][self.cells], color="#f2b134", lw=1.0, label="Mallard")
        ref = self.p.get("reference")
        if ref:
            r = np.asarray(ref["points"])
            ax.plot(r[:, 0], r[:, 1], "o", ms=2.6, mfc="none", mec=FG, mew=0.6, label=ref["label"])
        ax.set_xlim(self.p.get("xlim") or (self.x[0], self.x[-1]))
        if self.p.get("ylim"):
            ax.set_ylim(self.p["ylim"])
        style_plot_axes(ax, self.p)
        ax.legend(loc=self.p.get("legend_loc", "upper left"), fontsize=4.5, frameon=False, labelcolor=FG,
                  handlelength=1.2, borderaxespad=0.2)
        return d.get("TIME", np.nan)


def style_plot_axes(ax, p):
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(DIM)
        ax.spines[s].set_linewidth(0.5)
    ax.tick_params(colors=DIM, labelsize=4.5, width=0.5, length=2, pad=1)
    if p.get("xlabel"):
        ax.set_xlabel(p["xlabel"], fontsize=5, color=DIM, labelpad=1)
    if p.get("ylabel"):
        ax.set_ylabel(p["ylabel"], fontsize=5, color=DIM, labelpad=1)


class FramesPanel:
    def __init__(self, panel, rect, fig_px):
        self.p = panel
        self.files = snapshot_files(panel)
        times = os.path.join(panel["series"], "times.txt")
        self.times = np.loadtxt(times) if os.path.exists(times) else None

    def draw(self, ax, k, n):
        i = round(k * (len(self.files) - 1) / max(n - 1, 1))
        ax.imshow(imageio.imread(self.files[i]), interpolation="antialiased")
        return self.times[i] if self.times is not None else np.nan


class CurvePanel:
    def __init__(self, panel, rect, fig_px):
        self.p = panel
        d = np.genfromtxt(panel["data"], delimiter=",", names=True)
        self.t, self.y = d["t"], d["y"]
        refs = panel.get("reference") or []
        self.refs = [(r, np.genfromtxt(r["file"], delimiter=",", names=True))
                     for r in (refs if isinstance(refs, list) else [refs])]

    def draw(self, ax, k, n):
        t_now = self.p["t_end"] * k / max(n - 1, 1)
        for r, d in self.refs:
            ax.plot(d["t"], d["y"], color=r.get("color", FG), lw=r.get("lw", 0.8), alpha=r.get("alpha", 0.75),
                    ls=r.get("ls", "-"), label=r["label"])
        ax.axvline(t_now, color=DIM, lw=0.4, alpha=0.6)
        m = self.t <= t_now + 1e-9
        ax.plot(self.t[m], self.y[m], color="#f2b134", lw=1.0, label=self.p.get("label", "Mallard"))
        if m.any():
            ax.plot(self.t[m][-1], self.y[m][-1], "o", ms=2.2, color="#f2b134")
        ax.set_xlim(self.p.get("xlim") or (0, self.p["t_end"]))
        if self.p.get("ylim"):
            ax.set_ylim(self.p["ylim"])
        style_plot_axes(ax, self.p)
        ax.legend(loc=self.p.get("legend_loc", "upper right"), fontsize=4.5, frameon=False, labelcolor=FG,
                  handlelength=1.2, borderaxespad=0.2)
        return t_now


class DetonationPanel:
    GAP = 0.06

    def __init__(self, panel, rect, fig_px):
        self.p = panel
        self.files = snapshot_files(panel)
        _, x, y, fld = grid_fields(self.files[0], ["P"])
        self.p0 = fld["P"].min()
        dy = y[1] - y[0]
        self.xlim = panel.get("xlim") or (x[0], x[-1])
        self.ylim = (y[0] - dy / 2, y[-1] + dy / 2)

    def draw(self, ax, k, n):
        t, x, y, fld = grid_fields(self.files[round(k * (len(self.files) - 1) / max(n - 1, 1))], ["P", "P_MAX"])
        x_front = front_position(x, fld["P"], self.p0)
        win = (x >= self.xlim[0]) & (x <= self.xlim[1])
        ext = [*self.xlim, *self.ylim]
        h = (1 - self.GAP / (2 + self.GAP)) / 2
        top, bottom = ax.inset_axes([0, 1 - h, 1, h]), ax.inset_axes([0, 0, 1, h])
        p_lo, p_hi = self.p.get("p_range", (12, 34))
        # The fresh gas ahead of the front lies far below the pressure range: background, like the foil
        pressure = (fld["P"][win] / self.p0).T
        pressure[:, x[win] > x_front] = np.nan
        p_cmap = plt.get_cmap("inferno").copy()
        p_cmap.set_bad(BG)
        top.imshow(pressure, origin="lower", extent=ext, cmap=p_cmap, vmin=p_lo, vmax=p_hi,
                   aspect="auto", interpolation="antialiased")
        foil = soot(fld["P_MAX"][win], self.p0)
        foil[:, x[win] > x_front] = np.nan
        cmap = plt.get_cmap("gray_r").copy()
        cmap.set_bad(BG)
        f_lo, f_hi = self.p.get("foil_range", (-0.12, 0.2))
        bottom.imshow(foil, origin="lower", extent=ext, cmap=cmap, vmin=f_lo, vmax=f_hi, aspect="auto",
                      interpolation="antialiased")
        for a, label in ((top, "pressure"), (bottom, "soot foil")):
            a.set_axis_off()
            a.text(0.006, 0.94, label, transform=a.transAxes, fontsize=4.5, color="white", ha="left", va="top",
                   bbox=dict(boxstyle="round,pad=0.25", fc=BG, ec="none", alpha=0.6))
        return t


PANELS = {"field": FieldPanel, "wall": WallPanel, "frames": FramesPanel, "curve": CurvePanel,
          "detonation": DetonationPanel}


def render(args):
    config, k, n, path = args
    rects, height, footer_h = _layout
    fig = plt.figure(figsize=(config["width"], height), dpi=config["dpi"], facecolor=BG)
    for (p, rect), obj in zip(rects, _panels):
        if isinstance(obj, (WallPanel, CurvePanel)):
            # room for tick labels inside the slot
            ml, mb = p.get("margin_left", 0.2) / config["width"], 0.16 / height
            ax = fig.add_axes([rect[0] + ml, rect[1] + mb, rect[2] - ml - 0.08 / config["width"], rect[3] - mb])
        else:
            ax = fig.add_axes(rect)
            ax.set_axis_off()
        t = obj.draw(ax, k, n)
        fig.text(rect[0], rect[1] + rect[3] + 0.006, p.get("title", ""), fontsize=6, color=FG, ha="left", va="bottom")
        if p.get("show_time", isinstance(obj, (FieldPanel, FramesPanel, DetonationPanel))):
            ax.text(0.985, 0.975, f"t = {t:.3f}", transform=ax.transAxes, fontsize=4.5, color="white", ha="right",
                    va="top", family="DejaVu Sans Mono",
                    bbox=dict(boxstyle="round,pad=0.25", fc=BG, ec="none", alpha=0.6))
    if config.get("footer"):
        fig.text(0.5, footer_h / 2 / height, config["footer"], fontsize=5, color=DIM, ha="center", va="center")
    fig.savefig(path, dpi=config["dpi"], facecolor=BG)
    plt.close(fig)
    return path


def init_worker(config, fig_px):
    global _panels, _layout, BG
    BG = config.get("background", BG)
    _layout = layout(config)
    _panels = [PANELS[p.get("type", "field")](p, rect, fig_px)
               for p, rect in _layout[0]]


def set_last_delay(gif, seconds):
    """Lengthen the delay of the GIF's final frame (its last Graphic Control Extension)."""
    data = bytearray(open(gif, "rb").read())
    i = data.rfind(b"\x21\xf9\x04")
    if i >= 0:
        data[i + 4:i + 6] = int(round(seconds * 100)).to_bytes(2, "little")
        open(gif, "wb").write(data)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("output_stem")
    ap.add_argument("--frames", type=int, default=None, help="Number of frames (default: most snapshots of any case)")
    ap.add_argument("--start", type=float, default=0.0, help="Begin at this normalized time")
    ap.add_argument("--hold", type=float, default=1.0, help="Seconds to hold the final GIF frame")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    config = json.load(open(args.config))
    global BG
    BG = config.get("background", BG)
    config.setdefault("dpi", 300)
    config.setdefault("fps", 30)
    plt.rcParams.update({"font.family": "DejaVu Sans"})

    n_total = args.frames or max(len(snapshot_files(p)) for row in config["rows"] for p in row
                                 if p.get("type") != "curve")
    first = int(round(args.start * (n_total - 1)))
    _, height, _ = layout(config)
    fig_px = (config["width"] * config["dpi"], height * config["dpi"])
    tmp = tempfile.mkdtemp()
    jobs = [(config, k, n_total, os.path.join(tmp, f"{k - first:05d}.png")) for k in range(first, n_total)]
    with Pool(args.workers, initializer=init_worker, initargs=(config, fig_px)) as pool:
        for i, _ in enumerate(pool.imap(render, jobs)):
            if i % 25 == 0:
                print(f"frame {i + 1}/{len(jobs)}", flush=True)
    pattern = os.path.join(tmp, "%05d.png")
    write_mp4(pattern, args.output_stem + ".mp4", config["fps"], config.get("mp4_width"))
    write_gif(pattern, args.output_stem + ".gif", config["fps"], config.get("gif_width"),
              config.get("gif_colors", 256), config.get("gif_dither", "bayer:bayer_scale=5"))
    set_last_delay(args.output_stem + ".gif", args.hold)
    shutil.copy(jobs[-1][3], args.output_stem + "_final.png")
    shutil.rmtree(tmp)
    print("wrote", args.output_stem + ".mp4", args.output_stem + ".gif", f"({len(jobs)} frames)")


if __name__ == "__main__":
    main()
