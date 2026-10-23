"""Dissipation rate of the Taylor-Green vortex continued on a finer mesh by a
remap (examples/taylor_green_3d, coarse.toml then remap.toml), against the
fine mesh from t = 0 (input.toml at the fine size), into
docs/images/guide/remap-tgv-{light,dark}.svg.

    python tools/docs/remap_tgv.py FINE.csv COARSE.csv REMAP.csv [--t-remap 3] [--labels ...]

The CSV files are the runs' [integrals] output.
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

import figstyle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_taylor_green import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("fine")
    ap.add_argument("coarse")
    ap.add_argument("remap")
    ap.add_argument("--t-remap", type=float, default=3.0)
    ap.add_argument("--mu", type=float, default=0.000625)
    ap.add_argument("--labels", nargs=3, default=["32³ from t = 0", "16³", "32³ remapped from 16³"])
    args = ap.parse_args()
    runs = [load(path, args.mu) for path in (args.fine, args.coarse, args.remap)]

    def draw(P):
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        for (t, _, eps, _), name, color, lw in zip(runs, args.labels, (P["blue"], P["muted"], P["flame"]),
                                                   (1.8, 1.4, 2.2)):
            ax.plot(t, eps, color=color, lw=lw, label=name)
        ax.axvline(args.t_remap, color=P["mesh"], lw=1, ls=":")
        figstyle.label(ax, args.t_remap + 0.15, 0.0015, "remap", P, ha="left", size=11, color=P["muted"])
        ax.set_xlabel("t")
        ax.set_ylabel("dissipation rate  $-dE/dt$")
        ax.set_xlim(0, max(r[0][-1] for r in runs))
        ax.set_ylim(0, None)
        ax.legend(loc="upper left")
        return fig

    figstyle.save("remap-tgv", draw)


if __name__ == "__main__":
    main()
