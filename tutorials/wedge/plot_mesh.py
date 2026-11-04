#!/usr/bin/env python3
"""The wedge tutorial's generated mesh and its boundary zones.

    python plot_mesh.py [input.toml] [--coarsen 4] [--out DIR]

Draws the `wedge` mesh of the input's [mesh] table (every COARSEN-th line, so
the cells stay visible) with the four boundary zones, as wedge_mesh.
"""
import argparse
import os
import sys
import tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from tutfig import label_box, render  # noqa: E402

sys.path.insert(0, os.path.join(HERE, "..", "..", "tools", "docs"))
import meshes  # noqa: E402

ZONES = {"left": "inflow (inlet_state)", "right": "outflow (outlet_extrapolation)", "top": "top (symmetry)",
         "bottom": "bottom: ramp (symmetry, a slip wall)"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("input", nargs="?", default="input.toml")
    ap.add_argument("--coarsen", type=int, default=4)
    ap.add_argument("--out", help="output directory (default: docs/images/tutorials)")
    args = ap.parse_args()
    with open(args.input, "rb") as f:
        mesh = meshes.generated(tomllib.load(f)["mesh"], coarsen=args.coarsen)

    def draw(t):
        fig, ax = plt.subplots(figsize=(8, 5.4))
        ax.add_collection(LineCollection(mesh.edges(), colors=t["mesh"], linewidths=0.4, alpha=0.6, rasterized=True))
        for (zone, text), color in zip(ZONES.items(), (t["blue"], t["plum"], t["teal"], t["flame"])):
            seg = mesh.zone_segments(zone)
            ax.add_collection(LineCollection(seg, colors=color, linewidths=3))
            c = seg.reshape(-1, 2).mean(0)
            off = {"left": (-0.06, 0, "right", "center", 90), "right": (0.06, 0, "left", "center", 90),
                   "top": (0, 0.05, "center", "bottom", 0), "bottom": (0.25, -0.12, "center", "top", 0)}[zone]
            ax.text(c[0] + off[0], c[1] + off[1], text, color=color, ha=off[2], va=off[3], rotation=off[4],
                    fontsize=11, bbox=label_box(t))
        ax.annotate("8° ramp from x = 0.5", (0.62, 0.02), (0.75, 0.32), color=t["ink"], fontsize=11,
                    arrowprops=dict(arrowstyle="-|>", color=t["ink2"]), bbox=label_box(t))
        ax.set_xlim(-0.25, 2.25)
        ax.set_ylim(-0.3, 1.65)
        ax.set_aspect("equal")
        ax.axis("off")
        return fig

    render(draw, args.out, "wedge_mesh")


if __name__ == "__main__":
    main()
