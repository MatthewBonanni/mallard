"""Shared style of the documentation figures: each figure is drawn twice, for
the light and the dark theme of the website, as SVG with a transparent
background and real text.

A figure is a function draw(P) -> matplotlib Figure, where P is the theme's
palette; save(name, draw) writes docs/images/guide/<name>-light.svg and
<name>-dark.svg. The pages show them with

    ![alt](../images/guide/<name>-light.svg#gh-light-mode-only)
    ![alt](../images/guide/<name>-dark.svg#gh-dark-mode-only)

which GitHub and the website (build.sh maps the fragments) both honor.
"""
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "docs" / "images" / "guide"

LIGHT = dict(
    name="light",
    bg="#ffffff",
    fg="#1f1a1d",
    muted="#6b6166",
    line="#3b3338",
    mesh="#c9bfc4",
    faint="#ece6e9",
    panel="#f6f1f3",
    teal="#12655f",
    flame="#d9730d",
    plum="#7b3f6e",
    blue="#2f6db5",
    red="#c0392b",
    green="#3a7d2c",
    gold="#b58900",
    cmap="viridis",
)

DARK = dict(
    name="dark",
    bg="#171114",
    fg="#ece4e7",
    muted="#a99ea3",
    line="#cfc6ca",
    mesh="#4d4247",
    faint="#2a2126",
    panel="#211a1e",
    teal="#5fcfc1",
    flame="#f2a04d",
    plum="#d79bcb",
    blue="#7fb0ea",
    red="#f08070",
    green="#8fd17f",
    gold="#e3c45a",
    cmap="viridis",
)

THEMES = (LIGHT, DARK)

# Also write PNGs on the site's backgrounds into this directory, for review
PREVIEW = os.environ.get("MALLARD_FIGURE_PREVIEW")


def _rc(P):
    return {
        "font.family": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 13,
        "svg.fonttype": "none",
        "svg.hashsalt": "mallard",
        "text.color": P["fg"],
        "axes.labelcolor": P["fg"],
        "axes.edgecolor": P["muted"],
        "axes.facecolor": "none",
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": P["muted"],
        "ytick.color": P["muted"],
        "xtick.labelcolor": P["fg"],
        "ytick.labelcolor": P["fg"],
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.frameon": False,
        "legend.fontsize": 12,
        "figure.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.transparent": True,
        "lines.linewidth": 1.8,
        "hatch.color": P["muted"],
    }


def save(name, draw):
    """Draw the figure once per theme and write both SVGs."""
    OUT.mkdir(parents=True, exist_ok=True)
    for P in THEMES:
        with plt.rc_context(_rc(P)):
            fig = draw(P)
            path = OUT / f"{name}-{P['name']}.svg"
            fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.04, metadata={"Date": None}, dpi=150)
            if PREVIEW:
                bg = "#ffffff" if P["name"] == "light" else "#171114"
                fig.savefig(Path(PREVIEW) / f"{name}-{P['name']}.png", dpi=110, bbox_inches="tight",
                            pad_inches=0.1, facecolor=bg, transparent=False)
            plt.close(fig)
    print(f"wrote {name}-{{light,dark}}.svg")


def label(ax, x, y, text, P, color=None, size=13, box=False, **kw):
    """Text on a plot; box puts it on the page background (over a mesh); kw passes ha, va, rotation..."""
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    if box:
        kw["bbox"] = dict(boxstyle="round,pad=0.25", facecolor=P["bg"], edgecolor="none", alpha=0.9)
    return ax.text(x, y, text, color=color or P["fg"], fontsize=size, **kw)


def bare(ax):
    """An axis with equal aspect and nothing but the drawing."""
    ax.set_aspect("equal")
    ax.axis("off")
    return ax
