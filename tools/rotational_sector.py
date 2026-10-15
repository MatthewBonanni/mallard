#!/usr/bin/env python3
"""Figures of rotational periodicity (examples/rotational_sector), each in a
light and a dark version on a transparent background.

    rotational_sector.py diagram [--out DIR]
        A sector of an annulus with its periodic pair, the rotation's axis and
        angle, and a velocity turned across the pair.

    rotational_sector.py compare SECTOR.vtu ANNULUS.vtu --sectors N [--out DIR]
        The full annulus against the sector's solution turned into each of its
        N copies (2D, or the first axial layer of cells in 3D): the largest
        differences of density, momentum and energy over all cells, printed
        and plotted.

Figures go to DIR (default docs/images) as rotational_periodicity_{light,dark}.png
and rotational_sector_comparison_{light,dark}.png.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Arc, FancyArrowPatch

THEMES = {
    "light": {"fg": "#1f2937", "faint": "#9ca3af", "a": "#2563eb", "b": "#ea580c", "vec": "#059669"},
    "dark": {"fg": "#e5e7eb", "faint": "#6b7280", "a": "#60a5fa", "b": "#fb923c", "vec": "#34d399"},
}


def style(ax, theme):
    ax.set_aspect("equal")
    ax.axis("off")
    ax.title.set_color(theme["fg"])


def save(fig, out, name, mode):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{name}_{mode}.png")
    fig.savefig(path, dpi=200, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(path)


def turn(x, y, angle):
    c, s = np.cos(angle), np.sin(angle)
    return c * x - s * y, s * x + c * y


def cell(r0, r1, t0, t1, n=8):
    """Outline of the annular cell [r0, r1] x [t0, t1] (straight-sided, as the mesh)."""
    t = np.array([t0, t1])
    xy = [(r0 * np.cos(t0), r0 * np.sin(t0)), (r0 * np.cos(t1), r0 * np.sin(t1)),
          (r1 * np.cos(t1), r1 * np.sin(t1)), (r1 * np.cos(t0), r1 * np.sin(t0))]
    return np.array(xy)


def diagram(out):
    sectors, n_t, n_r, r_in, r_out = 8, 6, 4, 1.0, 2.0
    alpha = 2 * np.pi / sectors
    dt, dr = alpha / n_t, (r_out - r_in) / n_r
    radii = np.linspace(r_in, r_out, n_r + 1)
    for mode, theme in THEMES.items():
        fig, ax = plt.subplots(figsize=(7.2, 6.0))
        style(ax, theme)
        # The other sectors, faint
        for k in range(1, sectors):
            for i in range(n_t + 1):
                t = k * alpha + i * dt
                ax.plot([r_in * np.cos(t), r_out * np.cos(t)], [r_in * np.sin(t), r_out * np.sin(t)],
                        color=theme["faint"], lw=0.4, alpha=0.5)
            for r in radii:
                t = k * alpha + np.linspace(0, alpha, n_t + 1)
                ax.plot(r * np.cos(t), r * np.sin(t), color=theme["faint"], lw=0.4, alpha=0.5)
        # The sector
        for i in range(1, n_t):
            t = i * dt
            ax.plot([r_in * np.cos(t), r_out * np.cos(t)], [r_in * np.sin(t), r_out * np.sin(t)],
                    color=theme["fg"], lw=0.9)
        for r in radii:
            t = np.linspace(0, alpha, n_t + 1)
            ax.plot(r * np.cos(t), r * np.sin(t), color=theme["fg"], lw=0.9)
        # The periodic pair
        ax.plot([r_in, r_out], [0, 0], color=theme["a"], lw=3.2, solid_capstyle="round")
        ax.plot([r_in * np.cos(alpha), r_out * np.cos(alpha)], [r_in * np.sin(alpha), r_out * np.sin(alpha)],
                color=theme["b"], lw=3.2, solid_capstyle="round")
        ax.text(r_out + 0.08, -0.1, "zone A", color=theme["a"], fontsize=11, ha="left", va="top")
        ax.text(r_out * np.cos(alpha) - 0.15, r_out * np.sin(alpha) + 0.3, "zone B = R(α) A", color=theme["b"],
                fontsize=11, ha="left")
        # Axis and angle
        ax.plot(0, 0, marker="o", color=theme["fg"], ms=5)
        ax.text(-0.08, -0.12, "axis\n(center)", color=theme["fg"], fontsize=9.5, ha="right", va="top")
        for t in (0.0, alpha):
            ax.plot([0, r_in * np.cos(t)], [0, r_in * np.sin(t)], color=theme["fg"], lw=0.8, ls=(0, (3, 3)))
        ax.add_patch(Arc((0, 0), 0.9, 0.9, theta1=0, theta2=np.degrees(alpha), color=theme["fg"], lw=1.0))
        ax.text(0.5, 0.06, "α", color=theme["fg"], fontsize=12, ha="left", va="bottom")
        ax.text(-0.3, 0.42, "α = 360° / N\n(here N = 8)", color=theme["fg"], fontsize=10, ha="center", va="center")
        ax.add_patch(FancyArrowPatch((2.25, 0.12), (2.25 * np.cos(alpha) + 0.05, 2.25 * np.sin(alpha) - 0.05),
                                     connectionstyle="arc3,rad=0.38", arrowstyle="-|>", mutation_scale=14,
                                     color=theme["fg"], lw=1.0))
        ax.text(2.42, 0.95, "R(α)", color=theme["fg"], fontsize=11)
        # A cell next to B and its image beyond A: the velocity turns with it
        j = 2
        r0, r1 = radii[j], radii[j + 1]
        last = cell(r0, r1, alpha - dt, alpha)
        image = cell(r0, r1, -dt, 0.0)
        ax.fill(last[:, 0], last[:, 1], color=theme["vec"], alpha=0.25, lw=0)
        ax.fill(image[:, 0], image[:, 1], color=theme["vec"], alpha=0.12, lw=0)
        ax.plot(*np.vstack([image, image[:1]]).T, color=theme["vec"], lw=1.2, ls=(0, (3, 2)))
        rc = 0.5 * (r0 + r1)
        tc = alpha - 0.5 * dt
        cx, cy = rc * np.cos(tc), rc * np.sin(tc)
        u = np.array(turn(0.25, 0.42, tc))  # radial and azimuthal components
        ax.add_patch(FancyArrowPatch((cx, cy), (cx + u[0], cy + u[1]), arrowstyle="-|>", mutation_scale=14,
                                     color=theme["vec"], lw=2.0))
        ax.text(cx + u[0] + 0.03, cy + u[1], "u", color=theme["vec"], fontsize=12, va="bottom")
        ix, iy = turn(cx, cy, -alpha)
        v = turn(u[0], u[1], -alpha)
        ax.add_patch(FancyArrowPatch((ix, iy), (ix + v[0], iy + v[1]), arrowstyle="-|>", mutation_scale=14,
                                     color=theme["vec"], lw=2.0, ls="--"))
        ax.text(ix + v[0] + 0.04, iy + v[1] - 0.04, "R(α)ᵀ u", color=theme["vec"], fontsize=12, va="top")
        ax.text(0.0, -2.35, "The cell next to B, seen from the cells next to A, is its copy turned by R(α)ᵀ:\n"
                "its velocity, momentum and gradients turn with it; scalars carry over unchanged.",
                color=theme["fg"], fontsize=9.5, ha="center", va="top")
        ax.set_xlim(-2.4, 3.2)
        ax.set_ylim(-2.6, 2.3)
        save(fig, out, "rotational_periodicity", mode)


def read_cells(path):
    from mallard_vtu import read_vtu_cells
    pts, conn, offs, types, arrays = read_vtu_cells(path)
    starts = np.concatenate([[0], offs[:-1]])
    polys = [pts[conn[a:b]] for a, b in zip(starts, offs)]
    centers = np.array([p.mean(axis=0) for p in polys])
    return polys, centers, arrays


def compare(sector_path, annulus_path, sectors, out):
    alpha = 2 * np.pi / sectors
    ps, cs, fs = read_cells(sector_path)
    pa, ca, fa = read_cells(annulus_path)
    h = 1e-6 * np.ptp(ca[:, 0])
    key = lambda x: (round(x[0] / h), round(x[1] / h), round(x[2] / h))
    index = {key(x): i for i, x in enumerate(cs)}
    theta = np.mod(np.arctan2(ca[:, 1], ca[:, 0]), 2 * np.pi)
    copies = np.floor(theta / alpha + 1e-9).astype(int) % sectors
    match = np.empty(len(ca), dtype=int)
    for i, (x, k) in enumerate(zip(ca, copies)):
        xs, ys = turn(x[0], x[1], -k * alpha)
        match[i] = index[key((xs, ys, x[2]))]
    rho_s, rho_a = fs["RHO"][match], fa["RHO"]
    E_s, E_a = fs["RHOE"][match], fa["RHOE"]
    mx, my = turn(fs["RHOU"][match, 0], fs["RHOU"][match, 1], copies * alpha)
    d_rho = np.abs(rho_a - rho_s)
    d_m = np.max(np.abs(np.stack([fa["RHOU"][:, 0] - mx, fa["RHOU"][:, 1] - my, fa["RHOU"][:, 2] - fs["RHOU"][match, 2]])),
                 axis=0)
    d_E = np.abs(E_a - E_s)
    d = np.maximum(d_rho, np.maximum(d_m, d_E))
    print(f"{len(ca)} annulus cells, {len(cs)} sector cells, {sectors} copies")
    print(f"max |d rho| {d_rho.max():.3e}, max |d rho u| {d_m.max():.3e}, max |d rho E| {d_E.max():.3e}")
    # 3D: the first axial layer
    layer = ca[:, 2] <= ca[:, 2].min() + 1e-9 + 0.5 * h if np.ptp(ca[:, 2]) > 0 else np.ones(len(ca), bool)
    polys_a = [p[:, :2] for p, keep in zip(pa, layer) if keep]
    if np.ptp(ca[:, 2]) > 0:
        polys_a = [p[np.argsort(p[:, 2])[:4]][:, :2] for p, keep in zip(pa, layer) if keep]
        polys_a = [q[np.argsort(np.arctan2(q[:, 1] - q[:, 1].mean(), q[:, 0] - q[:, 0].mean()))] for q in polys_a]
    for mode, theme in THEMES.items():
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
        panels = [(rho_a[layer], "Full annulus: density"), (rho_s[layer], f"Sector, turned into its {sectors} copies"),
                  (np.log10(np.maximum(d[layer], 1e-18)), "log10 of the largest difference")]
        for ax, (values, title) in zip(axes, panels):
            style(ax, theme)
            log = title.startswith("log10")
            coll = PolyCollection(polys_a, array=values, cmap="magma" if log else "viridis", edgecolors="none")
            if not log:
                coll.set_clim(rho_a.min(), rho_a.max())
            ax.add_collection(coll)
            ax.autoscale_view()
            ax.set_title(title, fontsize=11)
            cb = fig.colorbar(coll, ax=ax, shrink=0.8)
            cb.ax.tick_params(colors=theme["fg"], labelsize=8)
            cb.outline.set_edgecolor(theme["faint"])
        for t in (0.0, alpha):
            r = np.array([np.hypot(ca[:, 0], ca[:, 1]).min() * 0.9, np.hypot(ca[:, 0], ca[:, 1]).max() * 1.05])
            for ax in axes[:2]:
                ax.plot(r * np.cos(t), r * np.sin(t), color=theme["b"], lw=1.6)
        axes[2].text(0.5, -0.06, f"max over all cells: {d.max():.1e}", transform=axes[2].transAxes, ha="center",
                     va="top", color=theme["fg"], fontsize=10)
        save(fig, out, "rotational_sector_comparison", mode)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    d = sub.add_parser("diagram")
    d.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "docs", "images"))
    c = sub.add_parser("compare")
    c.add_argument("sector")
    c.add_argument("annulus")
    c.add_argument("--sectors", type=int, required=True)
    c.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "docs", "images"))
    args = ap.parse_args()
    if args.command == "diagram":
        diagram(args.out)
    else:
        compare(args.sector, args.annulus, args.sectors, args.out)


if __name__ == "__main__":
    main()
