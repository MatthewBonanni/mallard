"""Figures of the theory manual (docs/theory/), drawn into docs/images/theory/:
reconstruction stencils on an unstructured mesh, quadrature on a curved
face, the splitting sequences, characteristic boundaries, Riemann fans and
the exchange point of a modeled wall."""
import math

import numpy as np
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

import figstyle
from figstyle import bare, label
from schematics import arrow

OUT = figstyle.OUT.parent / "theory"


# ------------------------------------------------------------ stencils

def triangle_mesh(nx=10, ny=7, L=(4.5, 3.0), jitter=0.22, seed=3):
    """A jittered Delaunay triangulation of a rectangle (boundary nodes stay on it)."""
    rng = np.random.default_rng(seed)
    x, y = np.meshgrid(np.linspace(0, L[0], nx), np.linspace(0, L[1], ny))
    p = np.column_stack([x.ravel(), y.ravel()])
    h = np.array([L[0] / (nx - 1), L[1] / (ny - 1)])
    inner = (p[:, 0] > 0) & (p[:, 0] < L[0]) & (p[:, 1] > 0) & (p[:, 1] < L[1])
    p[inner] += jitter * h * rng.uniform(-1, 1, (inner.sum(), 2))
    edge_x = ((p[:, 1] == 0) | (p[:, 1] == L[1])) & (p[:, 0] > 0) & (p[:, 0] < L[0])
    p[edge_x, 0] += jitter * h[0] * rng.uniform(-1, 1, edge_x.sum())
    tri = Delaunay(p)
    return p, tri.simplices


def tri_poly(ax, pts, P, fill="none", edge=None, lw=0.8, alpha=1.0, ls="-", z=1):
    ax.add_patch(Polygon(pts, closed=True, facecolor=fill, edgecolor=edge or P["mesh"], lw=lw, alpha=alpha,
                         ls=ls, zorder=z))


def stencils(P):
    nodes, cells = triangle_mesh()
    cent = nodes[cells].mean(axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1), gridspec_kw=dict(wspace=0.06))

    # (a) MUSCL: face neighbors, and vertex neighbors (tetrahedra, viscous gradients)
    ax = axes[0]
    target = int(np.argmin(np.linalg.norm(cent - [2.25, 1.55], axis=1)))
    tset = set(cells[target])
    face_nb = [c for c in range(len(cells)) if c != target and len(tset & set(cells[c])) == 2]
    vert_nb = [c for c in range(len(cells)) if c != target and len(tset & set(cells[c])) == 1]
    for c in range(len(cells)):
        tri_poly(ax, nodes[cells[c]], P)
    for c in vert_nb:
        tri_poly(ax, nodes[cells[c]], P, fill=P["blue"], alpha=0.18, edge="none")
    for c in face_nb:
        tri_poly(ax, nodes[cells[c]], P, fill=P["teal"], alpha=0.55, edge="none")
    tri_poly(ax, nodes[cells[target]], P, fill=P["fg"], alpha=0.85, edge="none")
    for c in face_nb + vert_nb:
        ax.plot(*cent[c], ".", color=P["fg"], ms=3, zorder=4)
    ax.set_title("MUSCL and the viscous gradients", fontsize=12.5)
    label(ax, 2.25, -0.32, "face neighbors (MUSCL)", P, color=P["teal"], size=11.5, va="top")
    label(ax, 2.25, -0.68, "vertex neighbors (quadratic fit, tetrahedra)", P, color=P["blue"], size=11.5, va="top")
    bare(ax)
    ax.set_xlim(-0.1, 4.6)
    ax.set_ylim(-1.1, 3.1)

    # (b) TENO-E (order 4): central stencil with mirror images, one sector stencil
    ax = axes[1]
    target = int(np.argmin(np.linalg.norm(cent - [2.1, 0.45], axis=1)))
    c0 = cent[target]
    mirror = cent * [1, -1]
    cand = [(np.linalg.norm(cent[c] - c0), c, False) for c in range(len(cells)) if c != target]
    cand += [(np.linalg.norm(mirror[c] - c0), c, True) for c in range(len(cells))]
    cand.sort()
    central = cand[:18]  # 2 x 9 coefficients of a cubic
    for c in range(len(cells)):
        tri_poly(ax, nodes[cells[c]], P)
        if cent[c, 1] < 1.2:
            tri_poly(ax, nodes[cells[c]] * [1, -1], P, edge=P["mesh"], ls=(0, (2, 2)), lw=0.6)
    for _, c, m in central:
        pts = nodes[cells[c]] * ([1, -1] if m else [1, 1])
        tri_poly(ax, pts, P, fill=P["plum"], alpha=0.42 if not m else 0.22, edge="none")
    # sector of the face opposite the boundary: the cone from the centroid through its nodes
    tn = cells[target]
    faces = [(tn[i], tn[(i + 1) % 3]) for i in range(3)]
    face = max(faces, key=lambda f: nodes[list(f), 1].mean())
    a, b = nodes[face[0]] - c0, nodes[face[1]] - c0
    M = np.linalg.inv(np.column_stack([a, b]))
    sector = [(d, c) for d, c, m in cand if not m and np.all(M @ (cent[c] - c0) >= -1e-9)][:10]
    for _, c in sector:
        tri_poly(ax, nodes[cells[c]], P, edge=P["flame"], lw=1.8, z=3)
    for v in (a, b):
        e = c0 + v / np.linalg.norm(v) * 1.9
        ax.plot([c0[0], e[0]], [c0[1], e[1]], color=P["flame"], lw=1.0, ls="--", zorder=3)
    tri_poly(ax, nodes[cells[target]], P, fill=P["fg"], alpha=0.85, edge="none", z=4)
    ax.plot([-0.1, 4.6], [0, 0], color=P["fg"], lw=2.4, zorder=5)
    label(ax, 4.55, 0.1, "boundary", P, size=11, ha="right", va="bottom")
    ax.set_title("TENO-E, order 4", fontsize=12.5)
    label(ax, 2.25, -1.25, "central stencil, 2 × 9 cells (mirror images across the boundary paler)", P,
          color=P["plum"], size=11, va="top")
    label(ax, 2.25, -1.6, "sector stencil of one face (the 10 nearest cells in its cone)", P, color=P["flame"],
          size=11, va="top")
    bare(ax)
    ax.set_xlim(-0.1, 4.6)
    ax.set_ylim(-1.95, 3.1)
    axes[0].set_ylim(-1.95, 3.1)
    return fig


# ------------------------------------------------------------ curved face

def curved_quadrature(P):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9), gridspec_kw=dict(width_ratios=[1, 2.4], wspace=0.08))
    g = np.array([-math.sqrt(3 / 5), 0.0, math.sqrt(3 / 5)])
    ax = axes[0]
    ax.plot([-1, 1], [0, 0], color=P["line"], lw=1.6)
    ax.plot([-1, 1], [0, 0], "|", color=P["line"], ms=10)
    for s in g:
        ax.plot(s, 0, "o", color=P["flame"], ms=6, zorder=4)
    label(ax, -1, -0.18, "−1", P, size=11.5, va="top")
    label(ax, 1, -0.18, "1", P, size=11.5, va="top")
    label(ax, 0, 0.25, r"$\xi_q$, weights $\hat w_q$", P, size=12, color=P["flame"], va="bottom")
    label(ax, 0, -0.55, "reference element", P, size=11.5, color=P["muted"])
    bare(ax)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.2, 1.2)

    ax = axes[1]
    R, t0, t1 = 3.0, math.radians(62), math.radians(118)
    A = R * np.array([math.cos(t1), math.sin(t1)])
    B = R * np.array([math.cos(t0), math.sin(t0)])
    C = np.array([0.0, 1.35])
    tt = np.linspace(math.radians(50), math.radians(130), 200)
    ax.fill_between(R * np.cos(tt), R * np.sin(tt), 3.6, color=P["faint"], lw=0)
    ax.plot(R * np.cos(tt), R * np.sin(tt), color=P["muted"], lw=1.0)
    ts = np.linspace(t1, t0, 100)
    arc = R * np.column_stack([np.cos(ts), np.sin(ts)])
    ax.add_patch(Polygon(np.vstack([arc, C]), closed=True, facecolor=P["teal"], alpha=0.12, edgecolor="none"))
    ax.plot(arc[:, 0], arc[:, 1], color=P["teal"], lw=2.6)
    ax.plot([A[0], B[0]], [A[1], B[1]], color=P["muted"], lw=1.2, ls="--")
    for p in (A, B):
        ax.plot([p[0], C[0]], [p[1], C[1]], color=P["line"], lw=1.2)
    o = (A + B + C) / 3
    ax.plot(*o, "s", color=P["plum"], ms=5, zorder=5)
    label(ax, o[0] + 0.12, o[1] - 0.05, r"$\mathbf{o}$", P, color=P["plum"], ha="left", size=13)
    for s in g:
        t = 0.5 * (t1 + t0) - 0.5 * (t1 - t0) * s
        p = R * np.array([math.cos(t), math.sin(t)])
        ax.plot([o[0], p[0]], [o[1], p[1]], color=P["plum"], lw=0.9, ls=":")
        ax.plot(*p, "o", color=P["flame"], ms=6, zorder=5)
        arrow(ax, p, p + 0.5 * p / R, P, color=P["flame"], lw=1.4, ms=9)
    label(ax, B[0] + 0.15, B[1] - 0.05, r"curved face $\mathbf{x}(\xi)$", P, color=P["teal"], ha="left", size=12)
    label(ax, -0.15, 2.48, "chord", P, color=P["muted"], size=11, ha="right")
    label(ax, -0.6, 3.75, r"normals $\mathbf{n}_{f,q}$, areas $\mathbf{a}_q$ summing to $\int_f \mathbf{n}\,dS$",
          P, color=P["flame"], size=11.5, va="bottom")
    label(ax, 1.25, 1.55, "cones from $\\mathbf{o}$:\nvolume, centroid,\nmoments", P, color=P["plum"], size=11,
          ha="left")
    label(ax, -2.35, 3.3, "wall", P, color=P["muted"], size=11)
    bare(ax)
    ax.set_xlim(-2.8, 2.9)
    ax.set_ylim(1.1, 4.2)
    return fig


# ------------------------------------------------------------ splitting

def splitting_sequence(P):
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 4.2), gridspec_kw=dict(hspace=0.75))

    def bar(ax, x0, x1, y, text, color, note=None):
        ax.add_patch(plt.Rectangle((x0, y - 0.32), x1 - x0, 0.64, facecolor=color, alpha=0.28, edgecolor=color,
                                   lw=1.4))
        label(ax, 0.5 * (x0 + x1), y, text, P, size=11.5)
        if note:
            label(ax, -0.03, y, note, P, size=11.5, ha="right", color=P["muted"])

    ax = axes[0]
    bar(ax, 0.0, 0.5, 2, r"$\mathcal{R}$ over $\Delta t/2$", P["flame"], "1")
    bar(ax, 0.0, 1.0, 1, r"RK step on $\mathcal{T}(\mathbf{U})$ over $\Delta t$", P["teal"], "2")
    bar(ax, 0.5, 1.0, 0, r"$\mathcal{R}$ over $\Delta t/2$", P["flame"], "3")
    ax.set_title('Strang (coupling = "strang")', fontsize=12, loc="left")
    ax = axes[1]
    ax.plot(0, 2, "o", color=P["teal"], ms=8, clip_on=False)
    label(ax, 0.03, 2, r"$\mathbf{c} = -\mathcal{T}(\mathbf{U}^n)$", P, ha="left", size=11.5)
    label(ax, -0.03, 2, "1", P, size=11.5, ha="right", color=P["muted"])
    bar(ax, 0.0, 1.0, 1, r"$\mathcal{R}(\mathbf{U}) - \mathbf{c}$ over $\Delta t$ (stiff reactor)", P["flame"], "2")
    bar(ax, 0.5, 1.0, 0, r"RK on $\mathcal{T}(\mathbf{U}) + \mathbf{c}$", P["teal"], "3")
    ax.set_title('SIMPLER (coupling = "simpler")', fontsize=12, loc="left")
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 2.5)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xticks([0, 0.5, 1], [r"$t_n$", r"$t_n + \Delta t/2$", r"$t_{n+1}$"])
    return fig


# ------------------------------------------------------------ characteristic boundaries

def characteristics(P):
    """x-t diagrams of the waves at a subsonic outlet and inlet."""
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.9), gridspec_kw=dict(wspace=0.12))
    u, c = 0.35, 1.0
    for ax, side in zip(axes, ("outlet", "inlet")):
        xb = 1.0 if side == "outlet" else 0.0
        ax.fill_between([0, 1], 0, 1.05, color=P["teal"], alpha=0.08, lw=0)
        label(ax, 0.5, 0.03, "interior", P, size=11, color=P["muted"], va="bottom")
        arrow(ax, (-0.05 if side == "outlet" else -0.55, 0), (-0.05 if side == "outlet" else -0.55, 0.3), P,
              color=P["muted"], lw=1.2, ms=9)
        label(ax, (-0.05 if side == "outlet" else -0.55) + 0.03, 0.32, "t", P, size=11, color=P["muted"],
              ha="left", va="bottom")
        ax.plot([xb, xb], [0, 1.05], color=P["fg"], lw=3)
        tb = 0.85
        for speed, name in ((u - c, r"$u_n - c$"), (u, r"$u_n$"), (u + c, r"$u_n + c$")):
            inward = (speed < 0) if side == "outlet" else (speed > 0)
            color = P["flame"] if inward else P["teal"]
            dt = 0.6
            x0 = xb - speed * dt
            if inward:
                x0 = xb + speed * 0.5
                arrow(ax, (xb, tb - 0.5), (x0, tb), P, color=color, lw=2)
                label(ax, x0, tb + 0.04, name, P, size=11.5, va="bottom", color=color)
            else:
                arrow(ax, (x0, tb - dt), (xb, tb), P, color=color, lw=2)
                label(ax, x0, tb - dt - 0.04, name, P, size=11.5, va="top", color=color)
        if side == "outlet":
            ax.set_title("outlet_pressure + nscbc", fontsize=12, family="monospace")
            txt = ("outgoing $w^+ = p + Z u_n$ from the interior\n"
                   "incoming $w^- = p_f - Z u_{n,f}$:\n"
                   r"$dw^-/dt = -K(p_f - p_t) - \beta\,\mathcal{T}^-$")
            label(ax, 0.42, -0.12, txt, P, size=11, va="top")
            ax.set_xlim(-0.1, 1.6)
        else:
            ax.set_title("inlet_state + nscbc", fontsize=12, family="monospace")
            txt = ("outgoing $w^+ = p + Z u_n$ from the interior\n"
                   "incoming $w^-$ relaxed toward $u_{n,t}$;\n"
                   "$T$ and $\\mathbf{u}_t$ toward their targets")
            label(ax, 0.55, -0.12, txt, P, size=11, va="top")
            ax.set_xlim(-0.6, 1.1)
        label(ax, xb, 1.08, "boundary", P, size=11, va="bottom")
        bare(ax)
        ax.set_aspect("auto")
        ax.set_ylim(-0.75, 1.25)
    return fig


# ------------------------------------------------------------ Riemann fans

def wave_fan(P):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), gridspec_kw=dict(wspace=0.1))
    S = (-1.0, 0.25, 1.3)
    for ax, kind in zip(axes, ("HLL", "HLLC")):
        ax.plot([-1.6, 1.8], [0, 0], color=P["muted"], lw=1.0)
        speeds = (S[0], S[2]) if kind == "HLL" else S
        names = (r"$S_L$", r"$S_R$") if kind == "HLL" else (r"$S_L$", r"$S_*$", r"$S_R$")
        for s, n in zip(speeds, names):
            col = P["plum"] if n == r"$S_*$" else P["teal"]
            ax.plot([0, s], [0, 1], color=col, lw=2, ls="--" if n == r"$S_*$" else "-")
            label(ax, s * 1.08, 1.06, n, P, size=12, va="bottom", color=col)
        label(ax, -1.2, 0.35, r"$\mathbf{U}^L$", P, size=13)
        label(ax, 1.55, 0.35, r"$\mathbf{U}^R$", P, size=13)
        if kind == "HLL":
            label(ax, 0.15, 0.72, r"$\mathbf{U}^{\mathrm{HLL}}$", P, size=13)
        else:
            label(ax, -0.32, 0.78, r"$\mathbf{U}^{*L}$", P, size=13)
            label(ax, 0.72, 0.78, r"$\mathbf{U}^{*R}$", P, size=13)
        arrow(ax, (1.65, 0), (1.85, 0), P, color=P["muted"])
        label(ax, 1.9, 0, "x", P, ha="left", size=12, color=P["muted"])
        label(ax, 0, -0.08, "face", P, va="top", size=11, color=P["muted"])
        ax.set_title(kind, fontsize=12.5)
        bare(ax)
        ax.set_aspect("auto")
        ax.set_xlim(-1.7, 2.1)
        ax.set_ylim(-0.25, 1.25)
    return fig


# ------------------------------------------------------------ wall models

def _walk(nodes, cells, c, p):
    """Cells visited by the exchange-point walk of src/boundary/wall_exchange.cpp: from cell c across the
    face beyond which p lies farthest, until p is inside."""
    owner = {}
    for k, cell in enumerate(cells):
        for e in ((cell[0], cell[1]), (cell[1], cell[2]), (cell[2], cell[0])):
            owner.setdefault(frozenset(e), []).append(k)
    path = [c]
    while True:
        cell = cells[c]
        centroid = nodes[cell].mean(axis=0)
        worst, across = 1e-12, None
        for e in ((cell[0], cell[1]), (cell[1], cell[2]), (cell[2], cell[0])):
            pa, pb = nodes[e[0]], nodes[e[1]]
            n = np.array([pb[1] - pa[1], pa[0] - pb[0]])
            n /= np.linalg.norm(n)
            if np.dot(n, centroid - pa) > 0:
                n = -n
            d = np.dot(p - 0.5 * (pa + pb), n)
            if d > worst:
                worst, across = d, frozenset(e)
        if across is None:
            return path
        c = next(k for k in owner[across] if k != c)
        path.append(c)


def wall_exchange(P):
    """The exchange point of a modeled wall face (exchange_cells = 3) and the walk that locates it."""
    nodes, cells = triangle_mesh(nx=9, ny=5, L=(4.0, 2.0), seed=3)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    wall_faces = [c for c in range(len(cells)) if np.sum(nodes[cells[c], 1] == 0.0) == 2]
    wall_cell = min(wall_faces, key=lambda c: abs(nodes[cells[c]].mean(axis=0)[0] - 2.3))
    a, b = [v for v in cells[wall_cell] if nodes[v, 1] == 0.0]
    x_f = 0.5 * (nodes[a] + nodes[b])
    depth = nodes[cells[wall_cell]].mean(axis=0)[1]
    h = 3.0 * 2.0 * depth
    x_e = x_f + np.array([0.0, h])
    path = _walk(nodes, cells, wall_cell, x_e)
    for c in range(len(cells)):
        tri_poly(ax, nodes[cells[c]], P)
    for c in path[:-1]:
        tri_poly(ax, nodes[cells[c]], P, fill=P["teal"], alpha=0.14, edge="none")
    tri_poly(ax, nodes[cells[path[-1]]], P, fill=P["teal"], alpha=0.45, edge="none")
    ax.plot([0, 4.0], [0, 0], color=P["fg"], lw=3, zorder=3)
    for x in np.linspace(0.05, 4.0, 40):
        ax.plot([x, x - 0.08], [0, -0.1], color=P["muted"], lw=0.8)
    ax.plot([nodes[a, 0], nodes[b, 0]], [0, 0], color=P["flame"], lw=5, solid_capstyle="butt", zorder=4)
    cent = np.array([nodes[cells[c]].mean(axis=0) for c in path])
    ax.plot(cent[:, 0], cent[:, 1], color=P["teal"], lw=1.3, ls=":", marker="o", ms=3.5, zorder=4)
    arrow(ax, tuple(x_f), tuple(x_e), P, color=P["fg"], lw=1.4, ms=10, zorder=5)
    x_c = cent[-1]
    arrow(ax, tuple(x_c), tuple(x_e), P, color=P["blue"], lw=1.6, ms=9, zorder=5)
    ax.plot(*x_e, "o", color=P["flame"], ms=7, zorder=6)
    label(ax, x_f[0] + 0.07, 0.4 * h, "$h$", P, ha="left", size=13, box=True)
    label(ax, x_e[0] - 0.1, x_e[1] + 0.12, r"exchange point $\mathbf{x}_e = \mathbf{x}_f + h\,\mathbf{n}$", P,
          ha="right", va="bottom", size=11.5, box=True)
    label(ax, x_e[0] + 0.25, x_e[1] - 0.05,
          r"$\phi_e = \overline{\phi}_c + \nabla\phi_c \cdot (\mathbf{x}_e - \mathbf{x}_c)$", P, color=P["blue"],
          ha="left", size=11.5, box=True)
    label(ax, x_c[0] - 0.06, x_c[1] - 0.04, r"$\mathbf{x}_c$", P, color=P["blue"], ha="right", va="top", size=12)
    label(ax, x_f[0], -0.2, r"modeled wall face $f$, centroid $\mathbf{x}_f$", P, color=P["flame"], va="top",
          size=11.5)
    label(ax, cent[1][0] - 0.25, cent[1][1], "walk", P, color=P["teal"], ha="right", size=11.5, box=True)
    bare(ax)
    ax.set_xlim(-0.05, 4.05)
    ax.set_ylim(-0.5, 2.15)
    return fig


FIGURES = {
    "wall_exchange": wall_exchange,
    "stencils": stencils,
    "curved_quadrature": curved_quadrature,
    "splitting_sequence": splitting_sequence,
    "characteristics": characteristics,
    "wave_fan": wave_fan,
}


def figures(names=None):
    for name, draw in FIGURES.items():
        if not names or name in names:
            figstyle.save(name, draw, OUT)


if __name__ == "__main__":
    import sys
    figures(sys.argv[1:])
