"""Schematics of the inlet conditions: the waves at a subsonic inflow face,
which the condition imposes (flame) and which come from the interior (teal),
and how the boundary state follows from them."""
from matplotlib.patches import FancyArrowPatch, Rectangle
import matplotlib.pyplot as plt

import figstyle
from figstyle import bare, label

INLETS = {
    "inlet_stagnation": dict(
        title="stagnation_inlet",
        imposed=[r"$p_0$, $T_0$", "flow direction", "composition (mixtures)"],
        interior=["outgoing Riemann invariant", r"$u - 2c / (\gamma_c - 1)$"],
        state=[r"speed $|u|$ from the invariant",
               r"$T$: $h(T) + |u|^2/2 = h(T_0)$",
               r"$p$: $s(T, p) = s(T_0, p_0)$"],
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


def figures(names=None):
    for name, spec in INLETS.items():
        if not names or name in names:
            figstyle.save(name, lambda P, spec=spec: inlet(P, spec))


if __name__ == "__main__":
    import sys
    figures(sys.argv[1:])
