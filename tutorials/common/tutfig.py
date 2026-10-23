"""Figures of the tutorials, in the style of the user guide's (tools/docs/figstyle.py):
each figure is drawn twice, for the light and the dark theme, as SVG with a
transparent background, into docs/images/tutorials/<name>-light.svg and
<name>-dark.svg (or another directory with --out). The pages show them with

    ![alt](../images/tutorials/<name>-light.svg#gh-light-mode-only)
    ![alt](../images/tutorials/<name>-dark.svg#gh-dark-mode-only)

A figure is a function draw(t) -> matplotlib Figure, where t is the theme's
palette: the user guide's colors plus "series" (line colors in a fixed
order), "ink", "ink2", "grid", "axis", "surface" and "mesh". Meshes and
fields are rasterized inside the SVG (rasterized=True on their artists), so
files stay small. MALLARD_FIGURE_PREVIEW=dir also writes PNGs on the site's
backgrounds there, for review.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "docs"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from figstyle import THEMES, _rc  # noqa: E402

OUT = ROOT / "docs" / "images" / "tutorials"
PREVIEW = os.environ.get("MALLARD_FIGURE_PREVIEW")


def palette(P):
    return dict(P, series=[P["blue"], P["flame"], P["teal"], P["plum"], P["gold"], P["green"], P["red"]],
                ink=P["fg"], ink2=P["muted"], grid=P["faint"], axis=P["muted"], surface=P["bg"], mesh=P["line"])


def render(draw, out, name):
    """Draw the figure once per theme and write OUT/NAME-light.svg and OUT/NAME-dark.svg."""
    out = Path(out) if out else OUT
    out.mkdir(parents=True, exist_ok=True)
    for P in THEMES:
        rc = dict(_rc(P), **{"axes.grid": True, "grid.color": P["faint"], "grid.linewidth": 0.8,
                              "legend.labelcolor": P["fg"], "axes.titlecolor": P["fg"]})
        with plt.rc_context(rc):
            fig = draw(palette(P))
            fig.savefig(out / f"{name}-{P['name']}.svg", format="svg", bbox_inches="tight", pad_inches=0.04,
                        metadata={"Date": None}, dpi=150)
            if PREVIEW:
                Path(PREVIEW).mkdir(parents=True, exist_ok=True)
                bg = "#ffffff" if P["name"] == "light" else "#171114"
                fig.savefig(Path(PREVIEW) / f"{name}-{P['name']}.png", dpi=90, bbox_inches="tight",
                            pad_inches=0.1, facecolor=bg, transparent=False)
            plt.close(fig)
    print(f"wrote {out}/{name}-{{light,dark}}.svg")


def label_box(t):
    """Text box style for labels drawn over fields or meshes."""
    return {"boxstyle": "round,pad=0.25", "facecolor": t["surface"], "edgecolor": "none", "alpha": 0.9}
