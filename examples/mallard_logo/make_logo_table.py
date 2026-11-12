#!/usr/bin/env python3
"""Write the Mallard logo as a CSV table for [[initialize.data]].

    make_logo_table.py OUTPUT.csv [--stride 2] [--preview logo.png]

Reads docs/images/mallard_light.png and classifies its pixels: ink (the black
frame and the wordmark), flame (the orange trail) and cream (the duck's white
halo). The columns ink, flame and cream are the fraction of each class in a
stride x stride block of pixels (0 elsewhere: the dark panel, the transparent
page); the input turns them into a density. The icon is mirrored so that its
duck lands on the duck of examples/mallard/make_mallard_mesh.py (bill tip at
the origin, flying in -x); the wordmark follows it, unmirrored, downstream.
Coordinates are pixel centers in the mesh frame, on a grid padded with zeros.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import median_filter

LOGO = Path(__file__).resolve().parents[2] / "docs" / "images" / "mallard_light.png"
PIXELS_PER_UNIT = 441.6         # the duck of make_mallard_mesh.py, scaled onto the logo's
WING_TIP_PIXEL = (195.0, 82.0)  # its control point (-0.1, 0.389), the raised wing's tip
WING_TIP = (-0.1, 0.389)
ICON_END = 807                  # a column between the icon and the wordmark
PAD = 2                         # blocks of zeros around the logo


def classes(rgba):
    """Fractions (ink, flame, cream) of every pixel."""
    rgb = rgba[..., :3].astype(float)
    alpha = rgba[..., 3].astype(float) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    ink = rgb.max(axis=-1) < 25
    flame = (r > 180) & (g < 200) & (b < 90)
    cream = (r > 200) & (g > 200) & (b > 140)
    return [median_filter(m.astype(float), size=3) * alpha for m in (ink, flame, cream)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--stride", type=int, default=2, help="Pixels per table node along each axis")
    ap.add_argument("--preview", help="Also save a plot of the density the example's input makes")
    args = ap.parse_args()

    rgba = np.asarray(Image.open(LOGO).convert("RGBA"))
    rgba = np.concatenate([rgba[:, ICON_END - 1::-1], rgba[:, ICON_END:]], axis=1)
    k = args.stride
    ny, nx = rgba.shape[0] // k, rgba.shape[1] // k
    fields = []
    for f in classes(rgba):
        f = f[:ny * k, :nx * k].reshape(ny, k, nx, k).mean(axis=(1, 3))
        fields.append(np.pad(f, PAD))

    s = PIXELS_PER_UNIT
    # Mirrored icon column c was column ICON_END - 1 - c; the mesh mirrors x -> 1 - x
    x0 = 1.0 - ((ICON_END - 1 - WING_TIP_PIXEL[0]) / s + WING_TIP[0])
    y0 = WING_TIP[1] + WING_TIP_PIXEL[1] / s - 0.1
    x = x0 + (np.arange(-PAD, nx + PAD) + 0.5) * k / s
    y = y0 - (np.arange(-PAD, ny + PAD) + 0.5) * k / s
    X, Y = np.meshgrid(x, y)
    table = np.column_stack([X.ravel(), Y.ravel()] + [f.ravel() for f in fields])
    with open(args.output, "w") as f:
        f.write(f"# Mallard logo ({LOGO.name}), {k}x{k} pixel blocks: fractions of ink, flame and cream\n")
        f.write("x,y,ink,flame,cream\n")
        np.savetxt(f, table, fmt=["%.5f", "%.5f", "%.3g", "%.3g", "%.3g"], delimiter=",")
    print(f"wrote {args.output}: {X.size} nodes, x in [{x[0]:.3f}, {x[-1]:.3f}], y in [{y[-1]:.3f}, {y[0]:.3f}]")

    if args.preview:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import sys
        sys.path.insert(0, str(LOGO.parents[2] / "examples" / "mallard"))
        from make_mallard_mesh import outline
        ink, flame, cream = fields
        rho = 1.0 + 2.0 * ink + 1.0 * cream - 0.75 * flame
        fig, ax = plt.subplots(figsize=(14, 3.5))
        ax.imshow(rho, extent=[x[0], x[-1], y[-1], y[0]], cmap="cividis", interpolation="nearest")
        duck = outline()
        ax.fill(duck[:, 0], duck[:, 1], color="#2a2f3a")
        ax.set_aspect("equal")
        fig.savefig(args.preview, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
