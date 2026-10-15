#!/usr/bin/env python3
"""Figures of [initialize] type = "remap" (docs/design/remap.md), each written
in a light and a dark variant (NAME_light.png, NAME_dark.png).

    plot_remap.py meshes SOURCE.vtu TARGET.vtu [--window x0 x1 y0 y1] [-o remap_meshes]
    plot_remap.py fields SOURCE.vtu TARGET.vtu [--var RHO] [-o remap_fields]
    plot_remap.py tgv REFERENCE.csv COARSE.csv REMAP.csv [--ref spectral.gdiag] [-o remap_tgv]

meshes overlays a 2D source and target mesh in a window and shades the pieces
of one target cell cut by the source cells (the supermesh the remap integrates
over); fields shows a variable before (source) and after (target) the remap,
with both along the diagonal y = x. Their inputs are the coarse run's last
snapshot and the remapped run's first (examples/riemann_2d_quads, coarse.toml
and remap.toml). tgv plots the dissipation rate of the Taylor-Green vortex
from the [integrals] files of the fine run from t = 0 (input.toml), the coarse
run (coarse.toml) and the remapped fine run (remap.toml) of
examples/taylor_green_3d.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mallard_vtu import read_vtu, read_vtu_cells  # noqa: E402

THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", muted="#52514e", grid="#e4e3df",
                  series=["#2a78d6", "#eb6834", "#1baf7a"], fine="#9b9a95"),
    "dark": dict(surface="#1a1a19", text="#ffffff", muted="#c3c2b7", grid="#34332f",
                 series=["#3987e5", "#d95926", "#199e70"], fine="#6d6c66"),
}
CMAP = "viridis"


def style(theme):
    plt.rcParams.update({
        "figure.facecolor": theme["surface"], "axes.facecolor": theme["surface"],
        "savefig.facecolor": theme["surface"], "text.color": theme["text"],
        "axes.labelcolor": theme["text"], "axes.edgecolor": theme["muted"],
        "xtick.color": theme["muted"], "ytick.color": theme["muted"],
        "axes.grid": False, "grid.color": theme["grid"], "font.size": 10,
        "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    })


def polygons(path):
    """Cells of a 2D VTU as lists of (x, y) vertices, and its cell arrays."""
    pts, conn, offs, _, arrays = read_vtu_cells(path)
    starts = np.concatenate([[0], offs[:-1]])
    return [pts[conn[a:b], :2] for a, b in zip(starts, offs)], arrays


def clip(subject, a, b):
    """Part of convex polygon subject left of the directed line a -> b."""
    def side(p):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    out = []
    for i, p in enumerate(subject):
        q = subject[(i + 1) % len(subject)]
        sp, sq = side(p), side(q)
        if sp >= 0:
            out.append(p)
        if (sp >= 0) != (sq >= 0):
            out.append(p + sp / (sp - sq) * (q - p))
    return out


def intersect(target, source):
    """Intersection of two convex polygons (any orientation)."""
    area = 0.5 * np.sum(source[:, 0] * np.roll(source[:, 1], -1) - np.roll(source[:, 0], -1) * source[:, 1])
    s = source if area > 0 else source[::-1]
    piece = list(target)
    for i in range(len(s)):
        if not piece:
            break
        piece = clip(piece, s[i], s[(i + 1) % len(s)])
    return np.array(piece)


def in_window(cell, w):
    return cell[:, 0].max() > w[0] and cell[:, 0].min() < w[1] and cell[:, 1].max() > w[2] and cell[:, 1].min() < w[3]


def meshes(args, theme):
    source, _ = polygons(args.source)
    target, _ = polygons(args.target)
    w = args.window
    source = [c for c in source if in_window(c, w)]
    target = [c for c in target if in_window(c, w)]
    center = np.array([0.5 * (w[0] + w[1]), 0.5 * (w[2] + w[3])])
    def area(p):
        return 0.5 * abs(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))

    def cut(c):
        return [p for p in (intersect(c, s) for s in source) if len(p) >= 3 and area(p) > 0.02 * area(c)]

    # The target cell cut into the most pieces, nearest the center
    chosen = max(target, key=lambda c: (len(cut(c)), -np.linalg.norm(c.mean(axis=0) - center)))
    pieces = cut(chosen)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.add_collection(PolyCollection(pieces, facecolors=[theme["series"][1 + k % 2] for k in range(len(pieces))],
                                     edgecolors=theme["surface"], linewidths=1.2, alpha=0.9, zorder=1))
    ax.add_collection(PolyCollection(target, facecolors="none", edgecolors=theme["fine"], linewidths=0.6, zorder=2))
    ax.add_collection(PolyCollection(source, facecolors="none", edgecolors=theme["series"][0], linewidths=1.8,
                                     zorder=3))
    ax.add_collection(LineCollection([np.vstack([chosen, chosen[:1]])], colors=theme["text"], linewidths=1.6,
                                     zorder=4))
    ax.set_xlim(w[0], w[1])
    ax.set_ylim(w[2], w[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    handles = [plt.Line2D([], [], color=theme["series"][0], lw=1.8, label="source cells"),
               plt.Line2D([], [], color=theme["fine"], lw=0.8, label="target cells"),
               plt.Line2D([], [], color=theme["text"], lw=1.6, label="one target cell"),
               plt.Rectangle((0, 0), 1, 1, color=theme["series"][1], label="its pieces in each source cell")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2)
    fig.tight_layout()
    return fig


def tri_field(path, var):
    """Triangulation of a 2D VTU (each cell fanned) with the cell value on each triangle."""
    pts, tris, cell_of_tri, arrays = read_vtu(path)
    tri = mtri.Triangulation(pts[:, 0], pts[:, 1], tris)
    return tri, np.asarray(arrays[var])[cell_of_tri], arrays.get("TIME")


def fields(args, theme):
    src_tri, src, t = tri_field(args.source, args.var)
    tgt_tri, tgt, _ = tri_field(args.target, args.var)
    lo, hi = min(src.min(), tgt.min()), max(src.max(), tgt.max())
    fig = plt.figure(figsize=(11.5, 4.4))
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.3], height_ratios=[1, 0.05], wspace=0.15, hspace=0.12)
    for k, (tri, v) in enumerate([(src_tri, src), (tgt_tri, tgt)]):
        ax = fig.add_subplot(grid[0, k])
        im = ax.tripcolor(tri, v, cmap=CMAP, vmin=lo, vmax=hi, rasterized=True)
        ax.plot(np.array(args.cut) / np.sqrt(2), np.array(args.cut) / np.sqrt(2), color=theme["surface"], lw=1.6)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
    fig.axes[0].set_title(f"before: {args.source_label}", color=theme["text"])
    fig.axes[1].set_title(f"after: {args.target_label}", color=theme["text"])
    cbar = fig.colorbar(im, cax=fig.add_subplot(grid[1, :2]), orientation="horizontal")
    cbar.set_label(args.var)
    cbar.outline.set_visible(False)
    ax = fig.add_subplot(grid[0, 2])
    s = np.linspace(args.cut[0], args.cut[1], 4001) / np.sqrt(2)
    for (tri, v, label), color in zip([(src_tri, src, "before"), (tgt_tri, tgt, "after")], theme["series"]):
        finder = tri.get_trifinder()
        k = finder(s, s)
        ax.plot(np.sqrt(2) * s[k >= 0], v[k[k >= 0]], color=color, lw=2 if label == "after" else 1.4,
                label=label, drawstyle="steps-mid")
    ax.set_xlabel("distance along y = x")
    ax.set_ylabel(args.var)
    ax.grid(True)
    ax.legend(loc="lower right")
    if t is not None:
        fig.suptitle(f"t = {t:g}", color=theme["muted"], x=0.06, ha="left", fontsize=9)
    return fig


def tgv(args, theme):
    from plot_taylor_green import load, load_reference
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    if args.ref:
        t_ref, eps_ref = load_reference(args.ref)[:2]
        ax.plot(t_ref, eps_ref, color=theme["muted"], lw=1.2, ls="--", label="spectral DNS, 512³ (full box)")
    labels = [args.labels[0], args.labels[1], args.labels[2]]
    for path, label, color, lw in zip([args.reference, args.coarse, args.remap], labels, theme["series"],
                                      [2.0, 1.6, 2.0]):
        t, _, eps, _ = load(path, args.mu)
        keep = t <= args.t_max
        ax.plot(t[keep], eps[keep], color=color, lw=lw, label=label)
    ax.axvline(args.t_remap, color=theme["grid"], lw=1.0, zorder=0)
    ax.text(args.t_remap, ax.get_ylim()[1], " remap", color=theme["muted"], va="top", fontsize=9)
    ax.set_xlim(0, args.t_max)
    ax.set_xlabel("t")
    ax.set_ylabel("dissipation rate  -dE/dt")
    ax.grid(True)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="figure", required=True)
    p = sub.add_parser("meshes")
    p.add_argument("source")
    p.add_argument("target")
    p.add_argument("--window", type=float, nargs=4, default=[0.555, 0.595, 0.475, 0.515])
    p.add_argument("-o", default="remap_meshes")
    p = sub.add_parser("fields")
    p.add_argument("source")
    p.add_argument("target")
    p.add_argument("--var", default="RHO")
    p.add_argument("--source-label", default="10,000 quadrilaterals")
    p.add_argument("--target-label", default="131,072 triangles")
    p.add_argument("--cut", type=float, nargs=2, default=[0.85, 1.25], help="range of the distance along y = x")
    p.add_argument("-o", default="remap_fields")
    p = sub.add_parser("tgv")
    p.add_argument("reference")
    p.add_argument("coarse")
    p.add_argument("remap")
    p.add_argument("--labels", nargs=3, default=["64³ from t = 0", "32³", "64³ remapped from 32³ at t = 6"])
    p.add_argument("--ref")
    p.add_argument("--mu", type=float, default=0.000625)
    p.add_argument("--t-remap", type=float, default=6.0)
    p.add_argument("--t-max", type=float, default=12.0)
    p.add_argument("-o", default="remap_tgv")
    args = parser.parse_args()
    draw = {"meshes": meshes, "fields": fields, "tgv": tgv}[args.figure]
    for name, theme in THEMES.items():
        style(theme)
        fig = draw(args, theme)
        fig.savefig(f"{args.o}_{name}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"{args.o}_{name}.png")


if __name__ == "__main__":
    main()
