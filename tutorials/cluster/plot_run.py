#!/usr/bin/env python3
"""The cluster tutorial's run: its final density with the ranks' pieces, and
its history over the jobs it took.

    python plot_run.py RUN_DIR [--out DIR]

RUN_DIR holds a run of input.toml, possibly stopped and resumed several times
(solut/dmr.pvd, mallard.jsonl). Prints one line per job of the run log and
writes the figure cluster_run.
"""
import argparse
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import matplotlib.pyplot as plt  # noqa: E402
import snapshots  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from tutfig import label_box, render  # noqa: E402


def load_pieces(pvtu):
    """Cells, centroids, density and owning rank of every piece of a .pvtu file."""
    polys, cents, rho, rank = [], [], [], []
    pieces = re.findall(r'Source="([^"]+)"', open(pvtu).read())
    for r, piece in enumerate(pieces):
        pts, conn, offs, a = snapshots.read(os.path.join(os.path.dirname(pvtu), piece))
        p, c = snapshots.cells(pts, conn, offs)
        polys += [q[:, :2] for q in p]
        cents.append(c[:, :2])
        rho.append(a["RHO"])
        rank.append(np.full(len(c), r))
    return polys, np.vstack(cents), np.concatenate(rho), np.concatenate(rank), len(pieces)


def load_jobs(path):
    """The jobs of a run log: their progress records, events and end record."""
    jobs = []
    for line in open(path):
        r = json.loads(line)
        if r["type"] == "start":
            jobs.append({"progress": [], "events": [], "end": None})
        elif r["type"] == "progress":
            jobs[-1]["progress"].append(r)
        elif r["type"] == "event":
            jobs[-1]["events"].append(r)
        elif r["type"] == "end":
            jobs[-1]["end"] = r
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("--out", help="output directory (default: docs/images/tutorials)")
    args = ap.parse_args()
    polys, c, rho, rank, n_ranks = load_pieces(snapshots.last(os.path.join(args.run, "solut", "dmr")))
    jobs = load_jobs(os.path.join(args.run, "mallard.jsonl"))
    for k, j in enumerate(jobs):
        e = j["end"] or {}
        print(f"job {k + 1}: steps {j['progress'][0]['step']} to {e.get('step', '?')}, "
              f"t = {j['progress'][0]['t']:.4g} to {e.get('t', float('nan')):.4g}: {e.get('reason', 'no end record')}")

    def draw(t):
        fig, (ax, bx) = plt.subplots(2, 1, figsize=(8.6, 5.8), gridspec_kw={"height_ratios": [1.1, 1]})
        pc = PolyCollection(polys, array=rho, cmap="viridis", edgecolors="none", antialiased=False, rasterized=True)
        ax.add_collection(pc)
        fig.colorbar(pc, ax=ax, shrink=0.9, label="density", pad=0.025)
        if n_ranks > 1:
            ax.tricontour(c[:, 0], c[:, 1], rank, levels=np.arange(n_ranks - 1) + 0.5, colors=[t["surface"]],
                          linewidths=1.4, linestyles="--")
        for r in range(n_ranks):
            m = c[rank == r].mean(0)
            ax.text(m[0], 0.85, f"rank {r}", color=t["ink"], ha="center", fontsize=10, bbox=label_box(t))
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(False)

        offset = 0.0
        for k, j in enumerate(jobs):
            col = t["series"][k % len(t["series"])]
            w = offset + np.array([p["wall"] for p in j["progress"]])
            tt = np.array([p["t"] for p in j["progress"]])
            end = j["end"]
            if end:
                w, tt = np.append(w, offset + end["wall"]), np.append(tt, end["t"])
            bx.plot(w, tt, "o-", ms=2.5, lw=1.4, color=col, label=f"job {k + 1}: {end['reason'] if end else 'running'}")
            starts = {p["step"]: offset + p["wall"] for p in j["progress"]}
            for e in j["events"]:
                if e["kind"] in ("checkpoint", "stop"):
                    x = offset + (end["wall"] if e["kind"] == "stop" and end else
                                  np.interp(e["step"], list(starts), list(starts.values())))
                    bx.plot(x, e["t"], "s" if e["kind"] == "checkpoint" else "X", ms=8, color=col,
                            mec=t["surface"])
                    bx.annotate(e["kind"], (x, e["t"]), xytext=(-6, 8), textcoords="offset points", ha="right",
                                color=t["ink"], fontsize=9)
            offset += end["wall"] if end else w[-1] - offset
        bx.set_xlabel("wall time of the jobs, setup included [s]")
        bx.set_ylabel("simulated time t")
        bx.legend(fontsize=9, loc="lower right")
        fig.tight_layout()
        return fig

    render(draw, args.out, "cluster_run")


if __name__ == "__main__":
    main()
