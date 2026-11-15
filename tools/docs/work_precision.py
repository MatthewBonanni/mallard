"""Figures and tables of the work-precision study (tools/perf/work_precision.py): the error of MUSCL and TENO-E
orders 3 to 6 against cells and wall time, on CPU threads and on a GPU, from the measurements saved in
work_precision.json.

    python tools/docs/work_precision.py [--table]

Each case gets one figure, work_precision_<case>: error against cells (the longest sequence of any platform; the
platforms give the same errors), then against wall time on each platform.
--table prints, per case and platform, the cells and wall time each scheme needs to reach each target error
(interpolated in log-log between the meshes around it) and the cheapest scheme.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import figstyle

DATA = Path(__file__).resolve().parent / "work_precision.json"
SCHEMES = ("MUSCL", "TENO3", "TENO4", "TENO5", "TENO6")
LABELS = {"MUSCL": "MUSCL", "TENO3": "TENO-E 3", "TENO4": "TENO-E 4", "TENO5": "TENO-E 5", "TENO6": "TENO-E 6"}
COLORS = {"MUSCL": "gold", "TENO3": "blue", "TENO4": "green", "TENO5": "teal", "TENO6": "plum"}
CASES = {
    "vortex_2d_quad": dict(title="Isentropic vortex, 2D squares", norm="L2", targets=(1e-3, 1e-4, 1e-5, 1e-6)),
    "vortex_2d_tri": dict(title="Isentropic vortex, 2D irregular triangles", norm="L2",
                          targets=(1e-3, 1e-4, 1e-5, 1e-6)),
    "vortex_3d_hex": dict(title="Isentropic vortex, 3D hexahedra", norm="L2", targets=(1e-3, 1e-4, 1e-5)),
    "vortex_3d_tet": dict(title="Isentropic vortex, 3D tetrahedra", norm="L2", targets=(1e-3, 1e-4, 1e-5)),
    "shu_osher": dict(title="Shu-Osher problem", norm="L1", targets=(3e-2, 1e-2, 3e-3)),
}


def platform_name(p):
    if "cuda" in " ".join(p["backend"]).lower():
        return p["hardware"].replace("NVIDIA ", "")
    return f"{p['threads']} threads"


def series(platform, case, scheme, norm):
    runs = [r for r in platform["runs"] if r["case"] == case and r["scheme"] == scheme and "error" in r]
    return sorted(runs, key=lambda r: r["cells"])


def at_error(runs, key, target, norm):
    """The value of key at which the runs reach the target error, interpolated in log-log between the two runs
    around it, and whether the coarsest run is already below it (the value is then the coarsest run's: an upper
    bound); None when they do not reach it."""
    e = np.array([r["error"][norm] for r in runs])
    x = np.array([r[key] for r in runs], float)
    if len(e) and e[0] <= target:
        return float(x[0]), True
    for i in range(len(e) - 1):
        if e[i] >= target >= e[i + 1]:
            s = (np.log(target) - np.log(e[i])) / (np.log(e[i + 1]) - np.log(e[i]))
            return float(np.exp(np.log(x[i]) + s * (np.log(x[i + 1]) - np.log(x[i])))), False
    return None


def figure(data, case):
    spec = CASES[case]
    platforms = [p for p in data["platforms"] if any(r["case"] == case for r in p["runs"])]
    if not platforms:
        return

    def draw(P):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 1 + len(platforms), figsize=(4.4 * (1 + len(platforms)), 4.2), sharey=True)
        for scheme in SCHEMES:
            runs = max((series(p, case, scheme, spec["norm"]) for p in platforms), key=len)
            if runs:
                axes[0].loglog([r["cells"] for r in runs], [r["error"][spec["norm"]] for r in runs], "o-",
                               color=P[COLORS[scheme]], label=LABELS[scheme], ms=4.5)
            for ax, p in zip(axes[1:], platforms):
                runs = series(p, case, scheme, spec["norm"])
                if runs:
                    ax.loglog([r["wall"] for r in runs], [r["error"][spec["norm"]] for r in runs], "o-",
                              color=P[COLORS[scheme]], ms=4.5)
        axes[0].set(xlabel="cells", ylabel=f"${spec['norm'][0]}_{{{spec['norm'][1:]}}}$ error of $\\rho$")
        for ax, p in zip(axes[1:], platforms):
            ax.set(xlabel=f"wall time, s ({platform_name(p)})")
        from matplotlib.ticker import NullFormatter
        for ax in axes:
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
        axes[0].legend(loc="lower left", fontsize=10.5)
        fig.suptitle(spec["title"], fontsize=14)
        fig.tight_layout()
        return fig

    figstyle.save(f"work_precision_{case}", draw)


def table(data):
    for case, spec in CASES.items():
        for p in data["platforms"]:
            if not any(r["case"] == case for r in p["runs"]):
                continue
            print(f"\n{case} on {platform_name(p)} ({spec['norm']} error):")
            for scheme in SCHEMES:
                runs = series(p, case, scheme, spec["norm"])
                if len(runs) > 1:
                    n = np.array([r["n"] for r in runs], float)
                    e = np.array([r["error"][spec["norm"]] for r in runs])
                    order = -np.diff(np.log(e)) / np.diff(np.log(n))
                    print(f"  {scheme:6s} orders " + ", ".join(f"{o:.2f}" for o in order)
                          + "  errors " + ", ".join(f"{x:.2e}" for x in e)
                          + "  walls " + ", ".join(f"{r['wall']:.2f}" for r in runs))
            for target in spec["targets"]:
                row = []
                for scheme in SCHEMES:
                    runs = series(p, case, scheme, spec["norm"])
                    w, c = at_error(runs, "wall", target, spec["norm"]), at_error(runs, "cells", target, spec["norm"])
                    row.append((scheme, c, w))
                reached = [x for x in row if x[2] is not None]
                best = min(reached, key=lambda x: x[2][0])[0] if reached else "-"
                if not all(series(p, case, s, spec["norm"]) for s in SCHEMES):
                    best += " of the schemes measured"
                print(f"  error {target:g}: cheapest {best}; " + "; ".join(
                    f"{s} " + ("not reached" if w is None else
                               f"{'<= ' if w[1] else ''}{c[0]:,.0f} cells {w[0]:.3g} s") for s, c, w in row))


def figures():
    """The figures from the saved measurements (make_figures.py)."""
    data = json.loads(DATA.read_text())
    for case in CASES:
        figure(data, case)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--data", default=str(DATA))
    args = ap.parse_args()
    data = json.loads(Path(args.data).read_text())
    if args.table:
        table(data)
    if not args.no_figure:
        for case in CASES:
            figure(data, case)


if __name__ == "__main__":
    main()
