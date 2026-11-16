"""Schematics of the inlet conditions: the waves at a subsonic inflow face,
which the condition imposes (flame) and which come from the interior (teal),
and how the boundary state follows from them; and the local coordinates of a
boundary's faces."""
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Rectangle
import matplotlib.pyplot as plt
import numpy as np

import figstyle
from figstyle import bare, label

INLETS = {
    "inlet_stagnation": dict(
        title="inlet_stagnation",
        imposed=[r"$p_0$, $T_0$", "flow direction", "composition (mixtures)"],
        interior=["outgoing Riemann invariant", r"$u - 2c / (\gamma_c - 1)$"],
        state=[r"speed $|u|$ from the invariant",
               r"$T$: $h(T) + |u|^2/2 = h(T_0)$",
               r"$p$: $s(T, p) = s(T_0, p_0)$"],
        waves={"u": "imposed", "u+c": "imposed", "u-c": "interior"},
    ),
    "inlet_mass_flow": dict(
        title="inlet_mass_flow",
        imposed=[r"mass flux $\rho u$ ($\dot m$ / area)", r"$T$, flow direction", "composition (mixtures)"],
        interior=["outgoing acoustic wave", r"$p - \rho c\, u$"],
        state=[r"$p$ from the wave with $u = \rho u / \rho$",
               r"$\rho = p / (R T)$",
               r"face flux = flux of this state: $\rho u$ exact"],
        waves={"u": "imposed", "u+c": "imposed", "u-c": "interior"},
    ),
}


def arrow(ax, a, b, color, lw=2.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0))


def inlet(P, spec):
    fig, (ax, tx) = plt.subplots(1, 2, figsize=(8.4, 3.3), gridspec_kw=dict(width_ratios=[1.0, 1.15], wspace=0.05))
    ax.add_patch(Rectangle((0.0, 0), 2.0, 2.2, facecolor=P["teal"], alpha=0.08, edgecolor="none"))
    ax.add_patch(Rectangle((-0.8, 0), 0.8, 2.2, facecolor=P["flame"], alpha=0.08, edgecolor="none"))
    ax.plot([0, 0], [0, 2.2], color=P["fg"], lw=3)
    label(ax, -0.4, 2.35, "ghost", P, size=11, color=P["flame"])
    label(ax, 1.0, 2.35, "interior", P, size=11, color=P["teal"])
    for text, y, sgn, key in ((r"$u$", 1.75, 1, "u"), (r"$u + c$", 1.15, 1, "u+c"), (r"$u - c$", 0.55, -1, "u-c")):
        color = P["flame"] if spec["waves"][key] == "imposed" else P["teal"]
        a, b = ((-0.6, y), (1.3, y)) if sgn == 1 else ((1.3, y), (-0.6, y))
        arrow(ax, a, b, color)
        label(ax, 0.8, y + 0.2, text, P, size=12)
    label(ax, 0.6, -0.12, r"$u$ along the inward normal; $u$ carries" + "\nentropy, shear and species",
          P, size=10, va="top", color=P["muted"])
    bare(ax)
    ax.set_xlim(-0.9, 2.1)
    ax.set_ylim(-0.75, 2.5)

    tx.axis("off")
    y = 0.97
    tx.text(0.02, y, spec["title"], fontsize=13, family="monospace", va="top", color=P["fg"])
    y -= 0.13
    for heading, lines, color in (("imposed", spec["imposed"], P["flame"]),
                                  ("from the interior", spec["interior"], P["teal"]),
                                  ("boundary state", spec["state"], P["fg"])):
        tx.text(0.02, y, heading, fontsize=12, fontweight="bold", va="top", color=color)
        y -= 0.095
        for line in lines:
            tx.text(0.07, y, line, fontsize=11.5, va="top", color=P["fg"])
            y -= 0.085
        y -= 0.03
    tx.set_xlim(0, 1)
    tx.set_ylim(0, 1)
    return fig


def boundary_frame(P):
    """Left: a 2D inlet above a wall, its frame (origin, inward normal, tangent) and a face's n, xi and s.
    Right: a 3D inlet seen along its normal, with xi, eta, r and theta."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.0, 4.0), gridspec_kw=dict(width_ratios=[1.15, 1.0], wspace=0.08))
    H = 2.0
    ax.add_patch(Rectangle((0.0, 0.0), 2.4, H, facecolor=P["teal"], alpha=0.07, edgecolor="none"))
    ax.add_patch(Rectangle((-0.2, -0.18), 2.6, 0.18, facecolor="none", edgecolor=P["muted"], hatch="///", lw=0))
    ax.plot([-0.2, 2.4], [0, 0], color=P["fg"], lw=2.5)
    label(ax, 2.05, -0.36, "wall", P, size=11, color=P["muted"])
    faces = np.linspace(0.0, H, 9)
    ax.plot([0, 0], [0, H], color=P["flame"], lw=3.5)
    ax.plot(np.zeros_like(faces), faces, "_", color=P["flame"], ms=10, mew=2)
    label(ax, -0.12, H + 0.16, "inlet faces", P, size=11, color=P["flame"], ha="left")
    origin = (0.0, 0.5 * H)
    ax.add_patch(Circle(origin, 0.05, color=P["fg"], zorder=5))
    label(ax, -0.14, origin[1] + 0.12, "origin", P, size=11, ha="right")
    arrow(ax, origin, (0.85, origin[1]), P["plum"])
    label(ax, 0.92, origin[1] + 0.02, "normal", P, size=11, color=P["plum"], ha="left")
    arrow(ax, (0.14, origin[1]), (0.14, origin[1] + 0.75), P["blue"])
    label(ax, 0.22, origin[1] + 0.62, r"tangent ($\xi$)", P, size=11, color=P["blue"], ha="left")
    yf = faces[2] + 0.5 * (faces[3] - faces[2])
    ax.add_patch(Circle((0.0, yf), 0.045, color=P["flame"], zorder=5))
    arrow(ax, (0.55, yf), (0.55, 0.0), P["teal"], lw=1.6)
    ax.plot([0.0, 0.55], [yf, yf], ls=":", color=P["muted"], lw=1.2)
    label(ax, 0.62, 0.5 * yf, r"$n$", P, size=13, color=P["teal"], ha="left")
    ax.annotate("", xy=(-0.42, yf), xytext=(-0.42, origin[1]),
                arrowprops=dict(arrowstyle="<->", color=P["blue"], lw=1.4, shrinkA=0, shrinkB=0))
    label(ax, -0.5, 0.5 * (yf + origin[1]), r"$\xi < 0$", P, size=12, color=P["blue"], ha="right")
    ax.annotate("", xy=(-0.95, yf), xytext=(-0.95, 0.0),
                arrowprops=dict(arrowstyle="<->", color=P["gold"], lw=1.4, shrinkA=0, shrinkB=0))
    label(ax, -1.03, 0.5 * yf, r"$s$ (2D)", P, size=12, color=P["gold"], ha="right")
    bare(ax)
    ax.set_xlim(-1.75, 2.45)
    ax.set_ylim(-0.5, H + 0.35)

    R = 1.0
    bx.add_patch(Circle((0, 0), R, facecolor=P["flame"], alpha=0.10, edgecolor=P["flame"], lw=2.5))
    arrow(bx, (0, 0), (1.35, 0), P["blue"])
    arrow(bx, (0, 0), (0, 1.35), P["blue"])
    label(bx, 1.45, 0.0, r"$\xi$", P, size=13, color=P["blue"], ha="left")
    label(bx, 0.0, 1.47, r"$\eta$", P, size=13, color=P["blue"])
    th = np.deg2rad(35.0)
    q = (0.72 * np.cos(th), 0.72 * np.sin(th))
    bx.plot([0, q[0]], [0, q[1]], color=P["fg"], lw=1.4)
    bx.add_patch(Circle(q, 0.04, color=P["flame"], zorder=5))
    label(bx, 0.5 * q[0] - 0.06, 0.5 * q[1] + 0.1, r"$r$", P, size=13)
    bx.add_patch(Arc((0, 0), 0.6, 0.6, theta1=0, theta2=35, color=P["fg"], lw=1.2))
    label(bx, 0.42, 0.11, r"$\theta$", P, size=12)
    bx.add_patch(Circle((0, 0), 0.05, color=P["fg"], zorder=5))
    label(bx, -0.08, -0.14, "origin", P, size=11, ha="right")
    label(bx, 0.0, -1.32, "3D inlet seen along its normal", P, size=11, color=P["muted"])
    bare(bx)
    bx.set_xlim(-1.4, 1.75)
    bx.set_ylim(-1.5, 1.65)
    return fig


def figures(names=None):
    for name, spec in INLETS.items():
        if not names or name in names:
            figstyle.save(name, lambda P, spec=spec: inlet(P, spec))
    if not names or "boundary_frame" in names:
        figstyle.save("boundary_frame", boundary_frame)


if __name__ == "__main__":
    import sys
    figures(sys.argv[1:])
