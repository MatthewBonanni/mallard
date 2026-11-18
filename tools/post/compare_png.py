#!/usr/bin/env python3
"""Side-by-side density plots of the last snapshot of several Mallard runs."""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import mallard  # noqa: E402

out, var, files = sys.argv[1], sys.argv[2], sys.argv[3:]
fig, axs = plt.subplots(1, len(files), figsize=(6 * len(files), 5.5), squeeze=False)
for ax, f in zip(axs[0], files):
    snapshot = mallard.open(f)
    g = snapshot.geometry()
    tris, tc = g.triangles()
    v = snapshot.read(var)[tc]
    tpc = ax.tripcolor(g.points[:, 0], g.points[:, 1], tris, facecolors=v, cmap="viridis")
    fig.colorbar(tpc, ax=ax)
    ax.set_aspect("equal")
    ax.set_title(f"{f.split('/')[-2]}  {var} [{v.min():.3f}, {v.max():.3f}]")
fig.savefig(out, dpi=90, bbox_inches="tight")
