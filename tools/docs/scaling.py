"""Strong and weak scaling of the Taylor-Green vortex on GPUs
(examples/taylor_green_3d/scaling.toml), drawn from scaling.json:

    python tools/docs/scaling.py --strong RUN... --weak RUN... [--out tools/docs/scaling.json]
    python tools/docs/scaling.py            # redraw from scaling.json

Each RUN is a run directory (or its mallard.jsonl run log) of the inputs that
examples/taylor_green_3d/scaling.py writes. A time per step is the mean over an
interval between two progress records, from step --from on; runs of the same
series, GPU count and node count are repetitions, and the point is the median
of all their intervals (each run's median is kept too). Strong-scaling
efficiency is T_1 / (n T_n), weak-scaling efficiency T_1 / T_n. The GPU and
the Mallard commit come from the logs.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import figstyle

DATA = Path(__file__).resolve().parent / "scaling.json"
NAME = "scaling-taylor-green-3d"


def read_run(path, first_step):
    path = Path(path)
    log = path / "mallard.jsonl" if path.is_dir() else path
    start, intervals = None, []
    for line in log.read_text().splitlines():
        record = json.loads(line)
        if record["type"] == "start":
            start = record
        elif (record["type"] == "progress" and record["step"] >= first_step
              and record.get("wall_per_step") is not None):
            intervals.append(record["wall_per_step"])
            cells = record["cells"]
    if start is None or not intervals:
        sys.exit(f"{log}: no start record or no progress after step {first_step}")
    p = start["provenance"]
    return {"gpus": p["run"]["ranks"], "nodes": p["run"]["hosts"], "cells": cells,
            "intervals": intervals,
            "device": p["run"]["device"].split(" (")[0], "commit": p["mallard"]["git_sha"][:9]}


def collect(series_runs, first_step):
    points, devices, commits = [], set(), set()
    for series, paths in series_runs.items():
        by_layout = {}
        for path in paths:
            r = read_run(path, first_step)
            devices.add(r["device"])
            commits.add(r["commit"])
            by_layout.setdefault((r["gpus"], r["nodes"], r["cells"]), []).append(r["intervals"])
        for (gpus, nodes, cells), runs in sorted(by_layout.items()):
            points.append({"series": series, "gpus": gpus, "nodes": nodes, "cells": cells,
                           "ms_per_step": round(1e3 * statistics.median([t for run in runs for t in run]), 3),
                           "runs": [round(1e3 * statistics.median(run), 3) for run in runs]})
    return {"device": " / ".join(sorted(devices)), "commit": " / ".join(sorted(commits)),
            "from_step": first_step, "points": points}


def with_efficiency(data):
    for series in ("strong", "weak"):
        pts = [p for p in data["points"] if p["series"] == series]
        one = [p for p in pts if p["gpus"] == 1]
        if not one:
            continue
        t1 = one[0]["ms_per_step"]
        for p in pts:
            p["efficiency"] = t1 / (p["gpus"] * p["ms_per_step"]) if series == "strong" else t1 / p["ms_per_step"]
    return data


def draw_figure(data):
    pts = {s: sorted((p for p in data["points"] if p["series"] == s), key=lambda p: p["gpus"])
           for s in ("strong", "weak")}

    def draw(P):
        import matplotlib.pyplot as plt
        fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.0))
        gpus = sorted({p["gpus"] for s in pts.values() for p in s})
        s = pts["strong"]
        if s:
            t1 = s[0]["ms_per_step"]
            a.loglog(gpus, [t1 / g for g in gpus], ls=(0, (3, 3)), color=P["muted"], lw=1.2, label="ideal")
            a.loglog([p["gpus"] for p in s], [p["ms_per_step"] for p in s], "o-", color=P["teal"],
                     label=f"{round(s[0]['cells'] ** (1 / 3))}³ hexahedra")
            for p in s[1:]:
                a.annotate(f"{100 * p['efficiency']:.0f}%", (p["gpus"], p["ms_per_step"]), textcoords="offset points",
                           xytext=(6, 6), fontsize=11, color=P["fg"])
            a.set_yticks([p["ms_per_step"] for p in s])
            a.set_yticklabels([f"{p['ms_per_step']:.0f}" for p in s])
            a.set_title("Strong scaling")
            a.set_ylabel("time per step (ms)")
            a.legend(loc="lower left")
        w = pts["weak"]
        if w:
            for nodes, marker, fill in ((1, "o-", P["blue"]), (2, "s--", "none")):
                q = [p for p in w if p["nodes"] == nodes]
                if q:
                    b.plot([p["gpus"] for p in q], [100 * p["efficiency"] for p in q], marker, ms=7, color=P["blue"],
                           mfc=fill, lw=1.4, label="one node" if nodes == 1 else "two nodes")
            b.axhline(100, ls=(0, (3, 3)), color=P["muted"], lw=1.2)
            b.set_xscale("log", base=2)
            b.set_ylim(min(90.0, min(100 * p["efficiency"] for p in w) - 2), 101)
            n_per = round((w[0]["cells"] / w[0]["gpus"]) ** (1 / 3))
            b.set_title(f"Weak scaling, {n_per}³ hexahedra per GPU")
            b.set_ylabel("efficiency (%)")
            b.legend(loc="lower left")
        for ax in (a, b):
            ax.set_xscale("log", base=2)
            ax.set_xticks(gpus)
            ax.set_xticklabels([str(g) for g in gpus])
            ax.minorticks_off()
            ax.set_xlabel(f"GPUs ({data['device']})")
        fig.tight_layout()
        return fig
    return draw


def table(data):
    lines = ["| series | GPUs | nodes | cells | ms per step | efficiency |", "|---|---:|---:|---:|---:|---:|"]
    for p in data["points"]:
        eff = f"{100 * p['efficiency']:.1f}%" if "efficiency" in p else ""
        lines.append(f"| {p['series']} | {p['gpus']} | {p['nodes']} | {p['cells']:,} | {p['ms_per_step']:.2f} | {eff} |")
    return "\n".join(lines)


def figures():
    figstyle.save(NAME, draw_figure(with_efficiency(json.loads(DATA.read_text()))))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--strong", nargs="*", default=[], help="run directories or mallard.jsonl of the fixed box")
    ap.add_argument("--weak", nargs="*", default=[], help="run directories or mallard.jsonl, fixed cells per GPU")
    ap.add_argument("--from", dest="first_step", type=int, default=60, help="first step of the steady intervals")
    ap.add_argument("--out", type=Path, default=DATA)
    args = ap.parse_args()
    if args.strong or args.weak:
        data = collect({"strong": args.strong, "weak": args.weak}, args.first_step)
        args.out.write_text(json.dumps(data, indent=1) + "\n")
    data = with_efficiency(json.loads(args.out.read_text()))
    if args.out.resolve() == DATA:
        figures()
    print(table(data))


if __name__ == "__main__":
    main()
