"""Schematic figures of the user guide: mesh types, stretching, periodic
seams, curved faces, axisymmetry, boundary conditions, splitting, the
thickened flame, sponges, viscosity laws, cell averages and time-step limits.
Everything is computed from the formulas the guide states."""
import math

import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle, Circle
import matplotlib.pyplot as plt

import figstyle
from figstyle import bare, label
import meshes


def arrow(ax, a, b, P, color=None, lw=1.6, style="-|>", ms=12, **kw):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=ms, color=color or P["fg"], lw=lw,
                                 shrinkA=0, shrinkB=0, **kw))


def draw_cells(ax, mesh, P, lw=1.0, color=None, fill=None):
    for c in mesh.cells:
        ax.add_patch(Polygon(mesh.nodes[list(c)], closed=True, facecolor=fill or "none",
                             edgecolor=color or P["line"], lw=lw))


# ---------------------------------------------------------------- meshes

def mesh_types_2d(P):
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.5), gridspec_kw=dict(wspace=0.12))
    for ax, kind in zip(axes, ("cartesian", "cartesian_tri", "wedge")):
        L = (2.0, 1.5) if kind == "wedge" else (1.5, 1.0)
        m = meshes.generated(dict(type=kind, Nx=8 if kind == "wedge" else 6, Ny=6 if kind == "wedge" else 4,
                                  Lx=L[0], Ly=L[1]))
        draw_cells(ax, m, P, lw=0.9)
        bare(ax)
        ax.set_title(f'"{kind}"', fontsize=11.5, family="monospace", color=P["fg"])
    return fig


def _oblique(p):
    x, y, z = p
    return np.array([x + 0.42 * z, y + 0.30 * z])


def _cube_edges():
    v = [(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    return [(a, b) for a in v for b in v if a < b and sum(abs(np.subtract(a, b))) == 1]


def mesh_types_3d(P):
    corners = [(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    hidden = (0, 1, 0)  # nothing is hidden at this angle; draw all edges, interior ones lighter
    panels = {
        "cartesian": [],
        "cartesian_tet": [((0, 0, 0), (1, 1, 1)), ((0, 0, 0), (1, 1, 0)), ((0, 0, 0), (1, 0, 1)),
                          ((0, 0, 0), (0, 1, 1)), ((1, 0, 0), (1, 1, 1)), ((0, 1, 0), (1, 1, 1)),
                          ((0, 0, 1), (1, 1, 1))],
        "cartesian_prism": [((0, 0, 0), (1, 1, 0)), ((0, 0, 1), (1, 1, 1)), ((0, 0, 0), (0, 0, 1)),
                            ((1, 1, 0), (1, 1, 1))],
        "cartesian_pyramid": [((0.5, 0.5, 0.5), c) for c in corners],
    }
    del hidden
    fig, axes = plt.subplots(1, 4, figsize=(8.0, 2.5), gridspec_kw=dict(wspace=0.05))
    for ax, (kind, inner) in zip(axes, panels.items()):
        if kind == "cartesian_prism":
            inner = [((0, 0, 0), (1, 1, 0)), ((0, 0, 1), (1, 1, 1))]
            q = [_oblique(p) for p in ((0, 0, 0), (1, 1, 0), (1, 1, 1), (0, 0, 1))]
            ax.add_patch(Polygon(q, closed=True, facecolor=P["teal"], alpha=0.18, edgecolor="none"))
        if kind == "cartesian_tet":
            for f in (((0, 0, 0), (1, 1, 1), (1, 0, 0)), ((0, 0, 0), (1, 1, 1), (0, 1, 0))):
                ax.add_patch(Polygon([_oblique(p) for p in f], closed=True, facecolor=P["teal"], alpha=0.12,
                                     edgecolor="none"))
        if kind == "cartesian_pyramid":
            base = [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]
            ax.add_patch(Polygon([_oblique(p) for p in base], closed=True, facecolor=P["teal"], alpha=0.18,
                                 edgecolor="none"))
            ax.plot(*_oblique((0.5, 0.5, 0.5)), "o", color=P["flame"], ms=4)
        for a, b in _cube_edges():
            pa, pb = _oblique(a), _oblique(b)
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=P["line"], lw=1.3)
        for a, b in inner:
            pa, pb = _oblique(a), _oblique(b)
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=P["teal"], lw=1.1)
        bare(ax)
        ax.set_xlim(-0.1, 1.55)
        ax.set_ylim(-0.1, 1.4)
        n = {"cartesian": "1 hexahedron", "cartesian_tet": "6 tetrahedra", "cartesian_prism": "2 prisms",
             "cartesian_pyramid": "6 pyramids"}[kind]
        ax.set_title(f'"{kind}"\n{n}', fontsize=11, color=P["fg"], linespacing=1.3)
    return fig


def stretching(P):
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.4), gridspec_kw=dict(wspace=0.35))
    Ny = 24
    for ax, beta in zip(axes, (0.0, 1.7)):
        y = meshes.stretched(Ny, 2.0, beta)
        for v in y:
            ax.plot([0, 1.4], [v, v], color=P["line"], lw=0.9)
        for x in np.linspace(0, 1.4, 6):
            ax.plot([x, x], [0, 2], color=P["line"], lw=0.9)
        for v in (0, 2):
            ax.plot([0, 1.4], [v, v], color=P["fg"], lw=2.6)
        bare(ax)
        ax.set_title(f"stretching = [0, {beta:g}]", fontsize=12, family="monospace")
        r = (y[Ny // 2 + 1] - y[Ny // 2]) / (y[1] - y[0])
        label(ax, 0.7, -0.14, "uniform" if beta == 0 else f"center / wall cell height = {r:.1f}", P, size=11,
              color=P["muted"], va="top")
    return fig


def periodic(P):
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    m = meshes.generated(dict(type="cartesian", Nx=6, Ny=3, Lx=2.0, Ly=1.0))
    draw_cells(ax, m, P, lw=0.8, color=P["mesh"])
    ghost = meshes.generated(dict(type="cartesian", Nx=1, Ny=3, Lx=2 / 6, Ly=1.0))
    ghost.nodes[:, 0] += 2.0
    draw_cells(ax, ghost, P, lw=0.9, color=P["plum"])
    for k in range(3):
        ax.add_patch(Rectangle((2.0, k / 3), 1 / 3, 1 / 3, facecolor=P["plum"], alpha=0.10, edgecolor="none"))
        ax.add_patch(Rectangle((0.0, k / 3), 1 / 3, 1 / 3, facecolor=P["plum"], alpha=0.22, edgecolor="none"))
    ax.plot([0, 0], [0, 1], color=P["plum"], lw=3)
    ax.plot([2, 2], [0, 1], color=P["plum"], lw=3)
    label(ax, -0.06, 0.5, 'zone A\n"left"', P, ha="right")
    label(ax, 2.42, 0.5, "cells of A,\ntranslated", P, ha="left", color=P["plum"], size=12)
    label(ax, 1.86, 1.08, 'zone B "right"', P, ha="center", va="bottom")
    arrow(ax, (0.2, -0.12), (2.15, -0.12), P, color=P["muted"])
    label(ax, 1.17, -0.18, "translation  (B = A + t)", P, va="top", size=12, color=P["muted"])
    bare(ax)
    ax.set_xlim(-0.5, 3.4)
    ax.set_ylim(-0.4, 1.3)
    return fig


def rotational_periodic(P):
    sectors, n_t, n_r, r_in, r_out = 8, 6, 4, 1.0, 2.0
    alpha = 2 * math.pi / sectors
    dt = alpha / n_t
    radii = np.linspace(r_in, r_out, n_r + 1)
    turn = lambda x, y, a: (math.cos(a) * x - math.sin(a) * y, math.sin(a) * x + math.cos(a) * y)
    fig, ax = plt.subplots(figsize=(6.6, 5.4))

    def cells(t0, n, color, lw):
        for i in range(n + 1):
            t = t0 + i * dt
            ax.plot([r_in * math.cos(t), r_out * math.cos(t)], [r_in * math.sin(t), r_out * math.sin(t)],
                    color=color, lw=lw)
        for r in radii:
            t = t0 + np.linspace(0, n * dt, 4 * n + 1)
            ax.plot(r * np.cos(t), r * np.sin(t), color=color, lw=lw)

    cells(alpha, (sectors - 1) * n_t, P["mesh"], 0.5)
    cells(0.0, n_t, P["line"], 0.9)
    # The pair
    ax.plot([r_in, r_out], [0, 0], color=P["blue"], lw=3.2, solid_capstyle="round")
    ax.plot([r_in * math.cos(alpha), r_out * math.cos(alpha)], [r_in * math.sin(alpha), r_out * math.sin(alpha)],
            color=P["flame"], lw=3.2, solid_capstyle="round")
    label(ax, r_out + 0.08, -0.12, "zone A", P, color=P["blue"], ha="left", va="top")
    label(ax, r_out * math.cos(alpha) - 0.1, r_out * math.sin(alpha) + 0.32, "zone B = R(α) A", P,
          color=P["flame"], ha="left")
    # Axis and angle
    ax.plot(0, 0, marker="o", color=P["fg"], ms=5)
    label(ax, -0.08, -0.12, "axis", P, ha="right", va="top", size=12)
    for t in (0.0, alpha):
        ax.plot([0, r_in * math.cos(t)], [0, r_in * math.sin(t)], color=P["muted"], lw=0.9, ls=(0, (3, 3)))
    t = np.linspace(0, alpha, 30)
    ax.plot(0.45 * np.cos(t), 0.45 * np.sin(t), color=P["fg"], lw=1.0)
    label(ax, 0.52, 0.1, "α", P, ha="left", va="bottom")
    label(ax, -0.3, 0.45, "α = 360° / N\n(here N = 8)", P, size=11, color=P["muted"])
    arrow(ax, (2.25, 0.12), (2.25 * math.cos(alpha) + 0.05, 2.25 * math.sin(alpha) - 0.05), P, lw=1.0,
          connectionstyle="arc3,rad=0.38")
    label(ax, 2.45, 0.95, "R(α)", P, ha="left")
    # A cell next to B and its copy beyond A: the velocity turns with it
    r0, r1 = radii[2], radii[3]
    quad = lambda t0, t1: [(r0 * math.cos(t0), r0 * math.sin(t0)), (r0 * math.cos(t1), r0 * math.sin(t1)),
                           (r1 * math.cos(t1), r1 * math.sin(t1)), (r1 * math.cos(t0), r1 * math.sin(t0))]
    ax.add_patch(Polygon(quad(alpha - dt, alpha), closed=True, facecolor=P["teal"], alpha=0.3, edgecolor="none"))
    ax.add_patch(Polygon(quad(-dt, 0.0), closed=True, facecolor=P["teal"], alpha=0.12, edgecolor=P["teal"],
                         lw=1.2, ls=(0, (3, 2))))
    rc, tc = 0.5 * (r0 + r1), alpha - 0.5 * dt
    cx, cy = rc * math.cos(tc), rc * math.sin(tc)
    u = turn(0.25, 0.42, tc)
    arrow(ax, (cx, cy), (cx + u[0], cy + u[1]), P, color=P["teal"], lw=2.0)
    label(ax, cx + u[0] + 0.03, cy + u[1], "u", P, color=P["teal"], ha="left", va="bottom")
    ix, iy = turn(cx, cy, -alpha)
    v = turn(u[0], u[1], -alpha)
    arrow(ax, (ix, iy), (ix + v[0], iy + v[1]), P, color=P["teal"], lw=2.0)
    label(ax, ix - 0.05, iy - 0.3, "R(α)ᵀ u: the same cell\nseen from the cells next to A", P,
          color=P["teal"], ha="left", va="top", size=11)
    bare(ax)
    ax.set_xlim(-2.2, 3.1)
    ax.set_ylim(-2.15, 2.4)
    return fig


def curved(P):
    """Quarter of an O-grid around a circle: straight faces vs the curve, with quadrature points."""
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    R, n_t, radii = 1.0, 6, [1.0, 1.35, 1.8]
    th = np.linspace(0, math.pi / 2, n_t + 1)
    for r in radii[1:]:
        ax.plot(r * np.cos(th), r * np.sin(th), color=P["line"], lw=1.0)
    for t in th:
        ax.plot([radii[0] * math.cos(t), radii[-1] * math.cos(t)], [radii[0] * math.sin(t), radii[-1] * math.sin(t)],
                color=P["line"], lw=1.0)
    ax.plot(R * np.cos(th), R * np.sin(th), color=P["muted"], lw=1.2, ls="--")
    tt = np.linspace(0, math.pi / 2, 200)
    ax.fill_between(R * np.cos(tt), 0, R * np.sin(tt), color=P["faint"], lw=0)
    ax.plot(R * np.cos(tt), R * np.sin(tt), color=P["teal"], lw=2.6)
    g = np.array([-math.sqrt(3 / 5), 0.0, math.sqrt(3 / 5)])
    for k in range(n_t):
        a, b = th[k], th[k + 1]
        for s in g:
            t = 0.5 * (a + b) + 0.5 * (b - a) * s
            p = R * np.array([math.cos(t), math.sin(t)])
            ax.plot(*p, "o", color=P["flame"], ms=4.5, zorder=5)
            if k == 3:
                arrow(ax, p, p - 0.28 * p / R, P, color=P["flame"], lw=1.2, ms=8)
    label(ax, 0.25, 0.25, "body", P, color=P["muted"], size=12)
    label(ax, 1.05, 1.02, "curved face\n(on the circle)", P, color=P["teal"], ha="left", size=12)
    label(ax, 0.48, 0.67, "straight\nface", P, color=P["muted"], ha="right", size=11)
    label(ax, 1.9, 0.12, "quadrature points\nwith their own normals", P, color=P["flame"], ha="left", size=11)
    bare(ax)
    ax.set_xlim(-0.05, 2.9)
    ax.set_ylim(-0.05, 1.9)
    return fig


def axisymmetric(P):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    # revolved body: an ellipse for the circle of revolution at x = 2.6
    t = np.linspace(0, 2 * math.pi, 200)
    for x0 in (0.0, 3.2):
        ax.plot(x0 + 0.25 * np.cos(t), 1.0 * np.sin(t), color=P["mesh"], lw=1.0)
    ax.plot([0, 3.2], [1, 1], color=P["mesh"], lw=1.0)
    ax.plot([0, 3.2], [-1, -1], color=P["mesh"], lw=1.0)
    m = meshes.generated(dict(type="cartesian", Nx=8, Ny=3, Lx=3.2, Ly=1.0))
    ax.add_patch(Rectangle((0, 0), 3.2, 1.0, facecolor=P["teal"], alpha=0.12, edgecolor="none"))
    draw_cells(ax, m, P, lw=0.8)
    ax.plot([-0.5, 3.8], [0, 0], color=P["muted"], lw=1.2, ls="-.")
    arrow(ax, (3.55, 0), (3.85, 0), P, color=P["muted"])
    label(ax, 3.9, 0, "x (axis)", P, ha="left", size=12, color=P["muted"])
    arrow(ax, (-0.35, 0), (-0.35, 0.55), P, color=P["muted"])
    label(ax, -0.42, 0.32, "y = r", P, ha="right", size=12, color=P["muted"])
    ax.plot([0, 3.2], [0, 0], color=P["muted"], lw=2.6, ls=(0, (4, 2.5)))
    label(ax, 1.6, -0.12, "axis faces: symmetry", P, va="top", size=12)
    label(ax, 1.6, 1.12, "meridian half-plane: the 2D mesh", P, va="bottom", size=12, color=P["teal"])
    label(ax, 3.2, -1.12, "the body of revolution", P, color=P["muted"], size=11, va="top")
    bare(ax)
    ax.set_xlim(-1.0, 4.8)
    ax.set_ylim(-1.45, 1.35)
    return fig


# ------------------------------------------------------- boundary conditions

def boundary_state(P):
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    ax.add_patch(Rectangle((0, 0), 2, 2, facecolor=P["teal"], alpha=0.10, edgecolor=P["line"], lw=1.2))
    ax.add_patch(Rectangle((2, 0), 2, 2, facecolor="none", edgecolor=P["muted"], lw=1.2, ls="--"))
    ax.plot([2, 2], [-0.1, 2.1], color=P["fg"], lw=3)
    label(ax, 1, 1.25, "boundary cell", P, size=12)
    label(ax, 1, 0.75, "$U_L$\nreconstructed", P, size=12, color=P["teal"])
    label(ax, 3, 1.25, "exterior state", P, size=12, color=P["muted"])
    label(ax, 3, 0.75, "$U_R$\nfrom the condition", P, size=12, color=P["flame"])
    label(ax, 2, 2.25, r"face flux $F(U_L, U_R)$: the Riemann solver", P, size=12, va="bottom")
    arrow(ax, (2, -0.35), (2.55, -0.35), P, color=P["fg"])
    label(ax, 2.62, -0.35, "n", P, ha="left", size=12)
    bare(ax)
    ax.set_xlim(-0.2, 4.2)
    ax.set_ylim(-0.6, 2.6)
    return fig


def characteristic(P):
    """Waves at a subsonic characteristic outlet and inlet (1D view)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), gridspec_kw=dict(wspace=0.15))
    for ax, side in zip(axes, ("outlet", "inlet")):
        bx = 2.0 if side == "outlet" else 0.0
        inside = (0.0, 2.0)
        ax.add_patch(Rectangle((inside[0], 0), 2.0, 2.2, facecolor=P["teal"], alpha=0.08, edgecolor="none"))
        ax.plot([bx, bx], [0, 2.2], color=P["fg"], lw=3)
        lo, hi = (bx - 1.3, bx + 0.5) if side == "outlet" else (bx - 0.5, bx + 1.3)
        waves = [(r"$u$", 1.75, 1), (r"$u + c$", 1.15, 1), (r"$u - c$", 0.55, -1)]
        for text, y, sgn in waves:
            outgoing = (sgn == 1) if side == "outlet" else (sgn == -1)
            a, b = ((lo, y), (hi, y)) if sgn == 1 else ((hi, y), (lo, y))
            arrow(ax, a, b, P, color=P["teal"] if outgoing else P["flame"], lw=2)
            label(ax, 1.2 if side == "outlet" else 0.8, y + 0.2, text, P, size=12)
        if side == "outlet":
            note = "incoming acoustic wave\nrelaxes p toward the target:\n" + r"$K = \sigma\, c\,(1 - M^2) / L$"
            ax.set_title("outlet_pressure + nscbc", fontsize=12, family="monospace")
        else:
            note = "incoming waves relax\nu (and T) toward the targets;\nturbulence enters here"
            ax.set_title("inlet_state + nscbc", fontsize=12, family="monospace")
        label(ax, 1.0 if side == "outlet" else 1.0, -0.15, note, P, va="top", size=11, color=P["flame"])
        bare(ax)
        ax.set_xlim(-0.7 if side == "inlet" else -0.3, 2.7 if side == "outlet" else 2.3)
        ax.set_ylim(-1.1, 2.4)
    return fig


# ------------------------------------------------------------ chemistry, LES

def splitting(P):
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 3.2), gridspec_kw=dict(hspace=0.9))

    def bar(ax, x0, x1, y, text, color):
        ax.add_patch(Rectangle((x0, y - 0.3), x1 - x0, 0.6, facecolor=color, alpha=0.25, edgecolor=color, lw=1.4))
        label(ax, 0.5 * (x0 + x1), y, text, P, size=11.5)

    ax = axes[0]
    bar(ax, 0, 0.5, 0, r"chemistry $\Delta t/2$", P["flame"])
    bar(ax, 0.5, 1.5, 0, r"flow $\Delta t$ (all RK stages)", P["teal"])
    bar(ax, 1.5, 2.0, 0, r"chemistry $\Delta t/2$", P["flame"])
    ax.set_title('Strang (coupling = "strang", the default)', fontsize=12, loc="left")
    ax = axes[1]
    bar(ax, 0, 0.18, 0, r"$T(U^n)$", P["teal"])
    bar(ax, 0.18, 1.5, 0, r"chemistry + constant $T(U^n)$ over $\Delta t$", P["flame"])
    bar(ax, 1.5, 2.0, 0, r"flow: $T(U) - T(U^n)$", P["teal"])
    ax.set_title('SIMPLER (coupling = "simpler")', fontsize=12, loc="left")
    for ax in axes:
        ax.set_xlim(-0.02, 2.02)
        ax.set_ylim(-0.45, 0.45)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xticks([0, 1, 2], [r"$t^n$", r"$t^n + \Delta t/2$", r"$t^{n+1}$"])
    return fig


def tfles(P):
    """Laminar and thickened flame: progress variable and sensor."""
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    x = np.linspace(-6, 6, 600)
    dl, F = 1.0, 3.0

    def c(x, d):
        return 0.5 * (1 + np.tanh(2 * x / d))

    cl, ct = c(x, dl), c(x, F * dl)
    omega = np.minimum(1.0, ct * (1 - ct) / 0.0475)
    ax.fill_between(x, 0, omega, color=P["flame"], alpha=0.15, lw=0)
    ax.plot(x, omega, color=P["flame"], lw=1.6, label=r"sensor $\Omega$ (thickened flame)")
    ax.plot(x, cl, color=P["muted"], lw=1.8, ls="--", label=r"laminar flame, $\delta_L$")
    ax.plot(x, ct, color=P["teal"], lw=2.2, label=rf"thickened, $F\,\delta_L$ with $F = {F:g}$")
    ax.set_xlabel(r"distance across the flame / $\delta_L$")
    ax.set_ylabel(r"progress variable $c$")
    ax.set_ylim(-0.02, 1.12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=11, ncol=2)
    ax.set_xlim(-6, 6)
    return fig


def sponge(P):
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    x = np.linspace(0, 5, 400)
    s = np.where(x > 4, 20 * (x - 4) ** 2, 0)
    ax.fill_between(x, 0, s, color=P["plum"], alpha=0.18, lw=0)
    ax.plot(x, s, color=P["plum"], lw=2.2)
    ax.axvline(4, color=P["muted"], lw=1, ls=":")
    label(ax, 2.0, 12, "physical domain\n(strength = 0)", P, size=12, color=P["muted"])
    label(ax, 4.06, 16, "sponge\nlayer", P, size=12, ha="left", color=P["plum"])
    ax.set_xlabel("x")
    ax.set_ylabel("strength [1/s]")
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 21)
    return fig


def viscosity(P):
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    T = np.linspace(150, 2500, 300)
    Tr, S = 273.15, 110.4
    ax.plot(T, np.ones_like(T), color=P["muted"], lw=2, ls="--", label='"constant"')
    ax.plot(T, (T / Tr) ** 1.5 * (Tr + S) / (T + S), color=P["teal"], lw=2.2,
            label='"sutherland" (air: S = 110.4 K)')
    ax.plot(T, (T / Tr) ** 0.7, color=P["flame"], lw=2.2, label='"power_law", exponent 0.7')
    ax.set_xlabel("T [K]")
    ax.set_ylabel(r"$\mu\,/\,\mu$ at 273.15 K")
    ax.legend(loc="upper left")
    ax.set_xlim(150, 2500)
    ax.set_ylim(0, 4.2)
    return fig


def subdivisions(P):
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8), gridspec_kw=dict(wspace=0.25))
    tri = np.array([[0, 0], [1, 0], [0.35, 0.85]])
    for ax, n in zip(axes, (1, 4)):
        for i in range(n):
            for j in range(n - i):
                a = tri[0] + (tri[1] - tri[0]) * i / n + (tri[2] - tri[0]) * j / n
                b = a + (tri[1] - tri[0]) / n
                c = a + (tri[2] - tri[0]) / n
                ax.add_patch(Polygon([a, b, c], closed=True, facecolor="none", edgecolor=P["line"], lw=0.8))
                ax.plot(*((a + b + c) / 3), "o", color=P["flame"], ms=3)
                if j < n - i - 1:
                    d = b + (tri[2] - tri[0]) / n
                    ax.add_patch(Polygon([b, d, c], closed=True, facecolor="none", edgecolor=P["line"], lw=0.8))
                    ax.plot(*((b + c + d) / 3), "o", color=P["flame"], ms=3)
        ax.add_patch(Polygon(tri, closed=True, facecolor="none", edgecolor=P["fg"], lw=2))
        bare(ax)
        ax.set_title(f"n_subdivisions = {n}", fontsize=11.5, family="monospace")
    return fig


def cfl_limits(P):
    """Largest stable cfl (SSPRK3), from docs/numerics/overview.md."""
    cells = ["quads", "triangles", "hexahedra", "tetrahedra", "prisms"]
    data = {
        "first order": [1.26, 1.29, 1.27, 1.26, 1.27],
        "MUSCL": [1.26, 1.33, 1.27, 1.26, 1.33],
        "TENO 3": [1.33, 1.36, 1.32, 1.49, 1.35],
        "TENO 4": [1.32, 1.32, 1.33, 1.32, 1.33],
        "TENO 5": [1.38, 1.47, 1.47, 1.55, 1.43],
        "TENO 6": [1.33, 1.44, 1.33, 1.36, 1.32],
    }
    colors = [P["muted"], P["blue"], P["teal"], P["green"], P["flame"], P["plum"]]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for k, ((name, v), col) in enumerate(zip(data.items(), colors)):
        y = np.arange(len(cells)) + (k - 2.5) * 0.11
        ax.plot(v, y, "o", color=col, ms=6, label=name)
    ax.axvline(1.2, color=P["red"], lw=1.4, ls="--")
    ax.axvline(1.0, color=P["fg"], lw=1.0, ls=":")
    label(ax, 1.205, -0.45, "largest safe: 1.2", P, color=P["red"], size=11, ha="left")
    label(ax, 0.995, -0.45, "robust default: 1.0", P, size=11, ha="right")
    ax.set_yticks(range(len(cells)), cells)
    ax.invert_yaxis()
    ax.set_xlim(0.7, 1.6)
    ax.set_ylim(4.5, -0.75)
    ax.set_xlabel("largest stable cfl, SSPRK3 (smooth flow)")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=11, handletextpad=0.2)
    return fig


FIGURES = {
    "mesh_types_2d": mesh_types_2d,
    "mesh_types_3d": mesh_types_3d,
    "stretching": stretching,
    "periodic": periodic,
    "rotational_periodic": rotational_periodic,
    "curved_faces": curved,
    "axisymmetric": axisymmetric,
    "boundary_state": boundary_state,
    "characteristic": characteristic,
    "splitting": splitting,
    "tfles": tfles,
    "sponge": sponge,
    "viscosity": viscosity,
    "subdivisions": subdivisions,
    "cfl_limits": cfl_limits,
}


def figures(names=None):
    for name, draw in FIGURES.items():
        if not names or name in names:
            figstyle.save(name, draw)


if __name__ == "__main__":
    import sys
    figures(sys.argv[1:])
