"""Figure of curved walls in axisymmetric runs (docs/design/axisymmetric.md):
inviscid flow over a sphere at Mach 0.3 (tools/dev/axisymmetric_sphere.py,
TENO3, HLLC, 32 L x 16 L quadrilaterals, from the potential flow), whose wall
entropy p / rho^gamma stays the free stream's in exact inviscid flow; its error
along the wall at L = 4 and its mean over the revolved wall against L, with
straight and with curved (P2) walls.

    python tools/docs/axisymmetric_curved.py [--compute --mallard build/src/mallard [--runs DIR]]

--compute runs the six cases (about 30 min on 2 threads) and writes
axisymmetric_curved.json next to this script; without it the figure reads
that file, so drawing it needs NumPy and Matplotlib only.
"""
import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import figstyle
from figstyle import label

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "axisymmetric_curved.json"
LEVELS = (1, 2, 4)
GEOMETRIES = ("straight", "curved")
MACH = 0.3


def compute(mallard, runs, threads=2):
    sys.path.insert(0, str(ROOT / "tools" / "dev"))
    import axisymmetric_sphere as sphere

    sphere.MACH = MACH
    args = argparse.Namespace(mallard=str(Path(mallard).resolve()), mach=MACH, t_stop=40.0, threads=threads)
    cases = []
    for geometry in GEOMETRIES:
        for level in LEVELS:
            case_dir = Path(runs) / f"TENO3_{geometry}_l{level}"
            r = sphere.run_case(str(case_dir), args, "TENO3", level, geometry)
            if r["failed"]:
                raise RuntimeError(f"{case_dir} failed: see its log.txt")
            wall = sphere.last_rows(str(case_dir / "probe_wall.csv"))
            angle = [math.degrees(math.atan2(sphere.column(w, "y"), -sphere.column(w, "x"))) for w in wall]
            error = [abs(sphere.column(w, "P") / sphere.column(w, "RHO") ** sphere.GAMMA
                         * sphere.RHO_INF ** sphere.GAMMA / sphere.P_INF - 1) for w in wall]
            order = np.argsort(angle)
            cases.append(dict(geometry=geometry, level=level, mean=r["entropy_error_mean"],
                              max=r["entropy_error_max"], angle=[angle[i] for i in order],
                              error=[error[i] for i in order]))
            print(f"{geometry} L = {level}: mean {r['entropy_error_mean']:.3e}, max {r['entropy_error_max']:.3e}")
    DATA.write_text(json.dumps(dict(mach=MACH, scheme="TENO3", riemann="HLLC", cases=cases), indent=1) + "\n")


def draw(P, d):
    color = dict(straight=P["flame"], curved=P["teal"])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.4, 4.0), gridspec_kw=dict(width_ratios=[1.35, 1.0], wspace=0.32))
    for c in d["cases"]:
        if c["level"] == 4:
            ax.semilogy(c["angle"], c["error"], color=color[c["geometry"]], label=f"{c['geometry']} walls")
    ax.set_xlim(0, 180)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlabel("angle from the front stagnation point [deg]")
    ax.set_ylabel(r"wall entropy error $|s / s_\infty - 1|$")
    ax.set_title("along the wall, 128 x 64 cells (L = 4)", fontsize=13)
    ax.legend(loc="lower right")
    for geometry in GEOMETRIES:
        rows = sorted((c for c in d["cases"] if c["geometry"] == geometry), key=lambda c: c["level"])
        L = np.array([c["level"] for c in rows], float)
        e = np.array([c["mean"] for c in rows])
        bx.loglog(L, e, "o-", color=color[geometry], label=f"{geometry} walls")
        rate = math.log(e[-2] / e[-1]) / math.log(L[-1] / L[-2])
        label(bx, L[-1] * 1.12, e[-1], f"order {rate:.1f}", P, color=color[geometry], size=12, ha="left")
    bx.set_xticks(LEVELS)
    bx.set_xticklabels([f"{32 * l} x {16 * l}" for l in LEVELS])
    bx.minorticks_off()
    bx.set_xlim(0.8, 7.5)
    bx.set_xlabel("cells (L = 1, 2, 4)")
    bx.set_ylabel("mean over the revolved wall")
    bx.set_title("against the mesh", fontsize=13)
    return fig


def figures():
    d = json.loads(DATA.read_text())
    figstyle.save("axisymmetric_curved_sphere", lambda P: draw(P, d))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--compute", action="store_true")
    ap.add_argument("--mallard")
    ap.add_argument("--runs")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    if args.compute:
        if not args.mallard:
            ap.error("--compute needs --mallard")
        if args.runs:
            compute(args.mallard, args.runs, args.threads)
        else:
            with tempfile.TemporaryDirectory() as runs:
                compute(args.mallard, runs, args.threads)
    figures()


if __name__ == "__main__":
    main()
