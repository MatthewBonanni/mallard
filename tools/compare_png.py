#!/usr/bin/env python3
"""Side-by-side density plots of the last snapshot of several Mallard runs."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mallard_vtu import read_vtu

out, var, files = sys.argv[1], sys.argv[2], sys.argv[3:]
fig, axs = plt.subplots(1, len(files), figsize=(6 * len(files), 5.5), squeeze=False)
for ax, f in zip(axs[0], files):
    pts, tris, tc, d = read_vtu(f)
    v = d[var][tc]
    tpc = ax.tripcolor(pts[:, 0], pts[:, 1], tris, facecolors=v, cmap="viridis")
    fig.colorbar(tpc, ax=ax)
    ax.set_aspect("equal")
    ax.set_title(f"{f.split('/')[-2]}  {var} [{v.min():.3f}, {v.max():.3f}]")
fig.savefig(out, dpi=90, bbox_inches="tight")
