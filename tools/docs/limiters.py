"""The MUSCL limiters of docs/theory/finite_volume.md, from
src/numerics/face_reconstruction.cpp: the limiter value of one face against
the ratio of the neighbors' range to the face increment, and the limited
reconstruction of a profile with a smooth part and a jump.

    python tools/docs/limiters.py

The profile is the 1D form of MUSCL on uniform cells: the weighted
least-squares gradient over the two face neighbors is the central difference,
Delta_max and Delta_min come from the cell and its neighbors, the cell takes
the smallest limiter value of its two faces, and the Venkatakrishnan
threshold is (K h)^3 rho_i^2 with the default K = 5.
"""
import numpy as np
import matplotlib.pyplot as plt

import figstyle

OUT = figstyle.OUT.parent / "theory"
K = 5.0


def barth_jespersen(d_minus, d_max, d_min):
    if d_minus > 0:
        return min(1.0, d_max / d_minus)
    if d_minus < 0:
        return min(1.0, d_min / d_minus)
    return 1.0


def venkatakrishnan(d_minus, d_max, d_min, eps2):
    if d_minus == 0:
        return 1.0
    d_plus = d_max if d_minus > 0 else d_min
    return (d_plus**2 + eps2 + 2 * d_minus * d_plus) / (d_plus**2 + 2 * d_minus**2 + d_minus * d_plus + eps2)


def profile(x):
    return np.where(x < 0.62, 0.95 + 0.3 * np.sin(4 * np.pi * (x - 0.05)), 0.3)


def reconstruct(n=18):
    """Cell averages and, per limiter, the face values of the interior cells."""
    edges = np.linspace(0.0, 1.0, n + 1)
    h = edges[1] - edges[0]
    fine = np.linspace(0.0, 1.0, 200 * n + 1)
    avg = np.array([profile(fine[200 * i:200 * (i + 1) + 1]).mean() for i in range(n)])
    out = {}
    for name in ("none", "barth_jespersen", "venkatakrishnan"):
        faces = []
        for i in range(1, n - 1):
            grad = (avg[i + 1] - avg[i - 1]) / (2 * h)
            near = avg[i - 1:i + 2] - avg[i]
            phi = 1.0
            for side in (-1, 1):
                d = grad * side * h / 2
                if name == "barth_jespersen":
                    phi = min(phi, barth_jespersen(d, near.max(), near.min()))
                elif name == "venkatakrishnan":
                    phi = min(phi, venkatakrishnan(d, near.max(), near.min(), (K * h) ** 3 * avg[i] ** 2))
            faces.append((edges[i], avg[i] - phi * grad * h / 2, edges[i + 1], avg[i] + phi * grad * h / 2))
        out[name] = faces
    return edges, avg, out


STYLE = {"none": ("muted", (0, (3, 2)), 'limiter = "none"'),
         "barth_jespersen": ("blue", "-", 'limiter = "barth_jespersen"'),
         "venkatakrishnan": ("teal", "-", 'limiter = "venkatakrishnan" (default)')}
EPS_STYLES = ((0.0, "-"), (0.5, (0, (5, 2))), (2.0, (0, (1.5, 1.5))))


def limiters(P):
    fig = plt.figure(figsize=(10.6, 5.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.3], wspace=0.28, hspace=0.12)
    ax = fig.add_subplot(gs[:, 0])
    r = np.linspace(0.0, 5.0, 501)
    ax.axhline(1.0, color=P["mesh"], lw=0.8, zorder=0)
    ax.plot(r, [barth_jespersen(1.0, x, -x) for x in r], color=P["blue"], label="Barth-Jespersen")
    for eps2, ls in EPS_STYLES:
        ax.plot(r, [venkatakrishnan(1.0, x, -x, eps2) for x in r], color=P["teal"], ls=ls,
                label=rf"Venkatakrishnan, $\varepsilon^2/\Delta_-^2 = {eps2:g}$")
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1.2)
    ax.set_xlabel(r"$\Delta_+/\Delta_-$")
    ax.set_ylabel(r"limiter value $\phi_f$")
    ax.set_title("Limiter value of one face", fontsize=13)
    ax.legend(loc="lower right", fontsize=10.5, handlelength=2.2)

    edges, avg, faces = reconstruct()
    for row, name in enumerate(("barth_jespersen", "venkatakrishnan")):
        bx = fig.add_subplot(gs[row, 1])
        for i, a in enumerate(avg):
            bx.plot(edges[i:i + 2], [a, a], color=P["mesh"], lw=3.0, solid_capstyle="butt", zorder=1)
        for key in ("none", name):
            c, ls, _ = STYLE[key]
            for x0, y0, x1, y1 in faces[key]:
                bx.plot([x0, x1], [y0, y1], color=P[c], ls=ls, lw=1.6 if key == name else 1.2, zorder=2)
        figstyle.label(bx, 0.97, 0.93, STYLE[name][2], P, color=P[STYLE[name][0]], size=11, ha="right", va="top",
                       transform=bx.transAxes, family="monospace")
        bx.set_xlim(edges[1], edges[-2])
        bx.set_ylim(0.1, 1.75)
        bx.set_ylabel(r"$\rho$")
        if row == 0:
            bx.set_title("Linear reconstruction in 18 cells", fontsize=13)
            bx.tick_params(labelbottom=False)
            figstyle.label(bx, 0.97, 0.8, 'dashed: limiter = "none"', P, color=P["muted"], size=11, ha="right",
                           va="top", transform=bx.transAxes, family="monospace")
        else:
            bx.set_xlabel("x")
    return fig


def figures():
    figstyle.save("limiters", limiters, OUT)


if __name__ == "__main__":
    figures()
