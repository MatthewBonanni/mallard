#!/usr/bin/env python3
"""Schematics of Mallard's inlet conditions for the documentation: the
characteristics that meet a subsonic inflow face in the x-t plane, which of
them the condition imposes and which come from the interior, and how the
boundary state follows.

    plot_inlet_schematics.py [OUTPUT_DIR] [--only NAME ...]

Writes OUTPUT_DIR/inlet_<name>.png (default docs/images) for each inlet.
"""
import argparse
import os

import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

import docs_style

INLETS = {
    "stagnation": {
        "title": "stagnation_inlet",
        "imposed": ["p0, T0", "flow direction", "composition (mixtures)"],
        "interior": ["outgoing Riemann invariant u − 2c / (γ_c − 1)", "(γ_c at T0; exact for constant γ)"],
        "result": [
            "speed |u| from the invariant",
            "T from h(T) + |u|²/2 = h(T0)",
            "p from s(T, p) = s(T0, p0)",
            "thermally perfect: h(T), s(T) of the mixture",
        ],
        "waves": {"u": "imposed", "u+c": "imposed", "u-c": "interior"},
    },
}


def arrow(ax, start, end, color, lw=2.0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0))


def draw(spec, path):
    plt = docs_style.plt
    fig = plt.figure(figsize=(9.0, 3.6))
    ax = fig.add_axes([0.0, 0.0, 0.48, 1.0])
    ax.set_xlim(-1.25, 1.6)
    ax.set_ylim(-0.32, 1.35)
    ax.axis("off")
    # Exterior (ghost) side, the inflow face and the interior cells
    ax.add_patch(Rectangle((-1.2, 0.0), 1.2, 1.15, facecolor=docs_style.ORANGE, alpha=0.12, lw=0))
    for k in range(4):
        ax.add_patch(Rectangle((0.35 * k, 0.0), 0.35, 1.15, fill=False, edgecolor=docs_style.INK, lw=0.6, alpha=0.6))
    ax.plot([0.0, 0.0], [0.0, 1.2], color=docs_style.INK, lw=2.5)
    ax.text(-0.6, 1.22, "ghost", ha="center", va="bottom", color=docs_style.ORANGE)
    ax.text(0.7, 1.22, "interior", ha="center", va="bottom")
    ax.text(0.0, -0.08, "inflow face", ha="center", va="top")
    arrow(ax, (1.45, 0.05), (1.45, 0.4), docs_style.INK, lw=1.0)
    ax.text(1.5, 0.22, "t", va="center")
    # Characteristics reaching the face, x = -λ (t_face - t), for inflow at a third of the sound speed
    u, c = 0.3, 0.8
    colors = {"imposed": docs_style.ORANGE, "interior": docs_style.BLUE}
    t_face, t_foot = 1.0, 0.16
    for name, speed, label in (("u", u, "u"), ("u+c", u + c, "u + c"), ("u-c", u - c, "u − c")):
        color = colors[spec["waves"][name]]
        foot = -0.9 * speed
        arrow(ax, (foot, t_foot), (0.0, t_face), color)
        ax.text(foot, t_foot - 0.03, "λ = " + label, color=color, fontsize=9, ha="center", va="top")
    ax.text(0.15, -0.2, "x, u along the inward normal; λ = u carries entropy, shear and species", fontsize=8.5, ha="center", va="top")

    tx = fig.add_axes([0.5, 0.0, 0.5, 1.0])
    tx.axis("off")
    y = 0.93
    tx.text(0.0, y, spec["title"], fontsize=13, fontweight="bold", va="top", family="monospace")
    y -= 0.13
    for heading, lines, color in (("imposed", spec["imposed"], docs_style.ORANGE),
                                  ("from the interior", spec["interior"], docs_style.BLUE),
                                  ("boundary state", spec["result"], docs_style.INK)):
        tx.text(0.0, y, heading, color=color, fontsize=11, fontweight="bold", va="top")
        y -= 0.085
        for line in lines:
            tx.text(0.04, y, line, fontsize=10, va="top")
            y -= 0.075
        y -= 0.03
    docs_style.save(fig, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir", nargs="?", default=os.path.join(os.path.dirname(__file__), "..", "docs", "images"))
    ap.add_argument("--only", nargs="*", choices=sorted(INLETS))
    args = ap.parse_args()
    docs_style.apply()
    for name in args.only or sorted(INLETS):
        draw(INLETS[name], os.path.join(args.output_dir, f"inlet_{name}.png"))


if __name__ == "__main__":
    np.seterr(all="raise")
    main()
