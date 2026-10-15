"""Matplotlib style for documentation figures that read on light and dark
pages alike: transparent background, mid-gray text and axes, and colors of
middle lightness.

    import docs_style
    docs_style.apply()
    ...
    docs_style.save(fig, "docs/images/name.png")
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#8c8c8c"  # text, axes, ticks: 3.4:1 on white, 5.2:1 on #1e1e1e
BLUE = "#3d8fe0"
ORANGE = "#e8890c"
GREEN = "#2ea66b"
RED = "#e0524b"
PURPLE = "#9a6fe0"


def apply():
    plt.rcParams.update({
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.transparent": True,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "grid.color": INK,
        "grid.alpha": 0.25,
        "legend.frameon": False,
        "legend.labelcolor": INK,
        "font.size": 11,
        "axes.prop_cycle": matplotlib.cycler(color=[BLUE, ORANGE, GREEN, RED, PURPLE]),
    })


def save(fig, path, dpi=200):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)
