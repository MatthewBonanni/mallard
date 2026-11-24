"""Edges of the curtain of glass beads in examples/shock_curtain against the
schlieren edges of Wagner et al. (examples/shock_curtain/wagner_edges.csv),
drawn from shock_curtain.json:

    python tools/docs/shock_curtain.py --two-way RUN --one-way RUN   # rewrite shock_curtain.json
    python tools/docs/shock_curtain.py                               # redraw from shock_curtain.json

Each RUN is a run directory of the example's input (coupled two ways, or with
coupling = "one-way") holding the frames.npz that
examples/shock_curtain/animate_shock_curtain.py extract writes. The edges are
the positions that 1% and 99% of the beads have not passed, as displacements
from their initial positions, against the time since the shock reached the
curtain, every 10 us.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import figstyle

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "shock_curtain"
DATA = Path(__file__).resolve().parent / "shock_curtain.json"
NAME = "shock_curtain_edges"
T_END = 650.0  # us, the last time of the experiment's data
DT = 10.0  # us


def edges(run_dir):
    import numpy as np
    sys.path.insert(0, str(EXAMPLE))
    from animate_shock_curtain import edges_of

    t, e = edges_of(np.load(Path(run_dir) / "frames.npz"))
    ts = np.arange(0.0, T_END + DT / 2, DT)
    return {"t_us": ts.tolist(), "upstream_mm": np.round(np.interp(ts, t, e[:, 0]), 3).tolist(),
            "downstream_mm": np.round(np.interp(ts, t, e[:, 1]), 3).tolist()}


def experiment():
    with open(EXAMPLE / "wagner_edges.csv") as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in rows[0]}


def draw_figure(data):
    exp = experiment()

    def draw(P):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        for edge, color in (("upstream", P["blue"]), ("downstream", P["red"])):
            ax.fill_between(exp["t_us"], exp[edge + "_lo_mm"], exp[edge + "_hi_mm"], color=color, alpha=0.22, lw=0)
            ax.plot(exp["t_us"], exp[edge + "_mm"], color=color, lw=1.2, ls=(0, (4, 3)))
            ax.plot(data["two_way"]["t_us"], data["two_way"][edge + "_mm"], color=color, lw=2.2)
            ax.plot(data["one_way"]["t_us"], data["one_way"][edge + "_mm"], color=color, lw=1.4, ls=(0, (1, 2)))
        two, one = data["two_way"], data["one_way"]
        for y, text, color in ((exp["downstream_mm"][-1], "downstream", P["red"]),
                               (one["downstream_mm"][-1] + 0.5, "both edges", P["muted"]),
                               (two["downstream_mm"][-1], "downstream", P["red"]),
                               (exp["upstream_mm"][-1], "upstream", P["blue"])):
            ax.text(T_END + 8, y, text, color=color, fontsize=12, va="center", clip_on=False)
        handles = [plt.Line2D([], [], color=P["fg"], lw=1.2, ls=(0, (4, 3))),
                   plt.Line2D([], [], color=P["fg"], lw=2.2),
                   plt.Line2D([], [], color=P["fg"], lw=1.4, ls=(0, (1, 2)))]
        ax.legend(handles, ["experiment, median and range of 4 runs", "Mallard, two-way (this input)",
                            "Mallard, one-way (0.2 mm cells)"], loc="upper left", fontsize=11)
        ax.set_xlim(0, T_END)
        ax.set_ylim(0, 60)
        ax.set_xlabel("time after impact (µs)")
        ax.set_ylabel("edge displacement (mm)")
        fig.tight_layout()
        return fig
    return draw


def figures():
    figstyle.save(NAME, draw_figure(json.loads(DATA.read_text())))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--two-way", help="run directory of the example's input")
    ap.add_argument("--one-way", help="run directory of the input with coupling = \"one-way\"")
    args = ap.parse_args()
    if args.two_way or args.one_way:
        if not (args.two_way and args.one_way):
            sys.exit("give both --two-way and --one-way")
        DATA.write_text(json.dumps({"two_way": edges(args.two_way), "one_way": edges(args.one_way)}) + "\n")
    figures()


if __name__ == "__main__":
    main()
