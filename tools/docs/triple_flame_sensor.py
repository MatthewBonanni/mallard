"""Figure of the thickened flame's reaction sensor on a triple flame
(docs/guide/les.md, sensor = "reaction"): the heat release, the sensor
TF_OMEGA with the sign of Takeno's FLAME_INDEX, and the thickening TF_F of a
short run of examples/triple_flame/tfles's mixtures and mesh.

    python tools/docs/triple_flame_sensor.py [--compute --mallard build/src/mallard --runs DIR]

--compute writes the run with examples/triple_flame/triple_flame.py (Cantera),
runs it (about 3 flame times, 72 x 120 cells, about 35 min on 2 threads) and
writes triple_flame_sensor.json next to this script from its last snapshot;
without it the figure reads that file, so drawing it needs NumPy and
Matplotlib only. The run is examples/triple_flame/tfles on a domain 40
delta_L wide and 24 delta_L long instead of 60 and 32, from the planar start,
with the snapshot fields of the figure; it is not the example's settled
flame, only long enough for the triple structure to form.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import figstyle

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tools")]
DATA = Path(__file__).resolve().parent / "triple_flame_sensor.json"
FIELDS = ["T", "HRR", "TF_OMEGA", "TF_F", "MIXTURE_FRACTION", "FLAME_INDEX"]
Z_ST = 0.30427

EXTRA = """
[output.mixture]
fuel = {X = {H2 = 0.5, N2 = 0.5}}
oxidizer = {X = {O2 = 1.0, N2 = 3.76}}

[les]
model = "vreman"
C = 1e-12

[les.combustion]
model = "tfles"
sensor = "reaction"
flame_table = "h2n2_air.csv"
fuel = {X = {H2 = 0.5, N2 = 0.5}}
oxidizer = {X = {O2 = 1.0, N2 = 3.76}}
efficiency = "none"
"""


def compute(mallard, runs):
    run = Path(runs).resolve() / "run"
    example = ROOT / "examples" / "triple_flame"
    subprocess.run([sys.executable, str(example / "triple_flame.py"), str(ROOT / "mechanisms" / "h2o2.yaml"), "ohmech",
                    str(run), "--mixing", "12", "--fuel", "H2:0.5,N2:0.5", "--u-in", "1.6", "--upstream", "12",
                    "--downstream", "12", "--width", "40", "--flame-times", "3", "--outputs", "6", "--cfl", "0.8",
                    "--cells", "3"], check=True)
    shutil.copy(example / "tfles" / "h2n2_air.csv", run)
    s = (run / "input.toml").read_text()
    s = re.sub(r'fields = \[[^\]]*\]', 'format = "vtu"\nfields = ["RHO", "U", ' + ", ".join(f'"{f}"' for f in FIELDS) + "]",
               s, count=1)
    (run / "input.toml").write_text(s + EXTRA)
    subprocess.run([mallard, "-i", "input.toml", "--kokkos-num-threads=2"], cwd=run, check=True,
                   stdout=subprocess.DEVNULL)
    extract(sorted((run / "solut").glob("flame_*.vtu"))[-1])


def extract(last):
    from mallard.vtu import read_vtu_cells

    pts, conn, offs, _, d = read_vtu_cells(str(last))
    starts = np.concatenate([[0], offs[:-1]])
    centers = np.array([pts[conn[a:b], :2].mean(axis=0) for a, b in zip(starts, offs)])
    xs, ys = np.unique(np.round(centers[:, 0], 9)), np.unique(np.round(centers[:, 1], 9))
    i = np.searchsorted(xs, np.round(centers[:, 0], 9))
    j = np.searchsorted(ys, np.round(centers[:, 1], 9))
    grids = {}
    for f in FIELDS:
        grids[f] = np.zeros((len(ys), len(xs)))
        grids[f][j, i] = d[f]
    # Rows of the flame (heat release above 1% of its peak) and 2.5 mm on either side
    dx = xs[1] - xs[0]
    rows = np.nonzero((grids["HRR"] > 0.01 * grids["HRR"].max()).any(axis=1))[0]
    margin = int(round(2.5e-3 / dx))
    lo, hi = max(rows[0] - margin, 0), min(rows[-1] + margin + 1, len(ys))
    out = dict(t=float(d["TIME"]), dx=float(dx), nx=len(xs), ny=int(hi - lo), y0=float(lo * dx))
    for f in FIELDS:
        out[f] = [[float(f"{v:.3g}") for v in row] for row in grids[f][lo:hi]]
    DATA.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(f"wrote {DATA.name} from {last.name} (t = {out['t']:.4g} s)")


def draw(P):
    d = json.loads(DATA.read_text())
    g = {f: np.array(d[f]) for f in FIELDS}
    dx = d["dx"] * 1e3
    y0 = d["y0"] * 1e3
    ext = (0, d["nx"] * dx, y0, y0 + d["ny"] * dx)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 5.6), layout="constrained", sharey=True)
    kw = dict(origin="lower", extent=ext, interpolation="nearest", aspect="equal")
    yc = y0 + (np.arange(d["ny"]) + 0.5) * dx
    xc = (np.arange(d["nx"]) + 0.5) * dx

    def zst(ax, color="#d8d8d8"):
        ax.contour(xc, yc, g["MIXTURE_FRACTION"], [Z_ST], colors=[color], linewidths=1.0, linestyles="--")

    def bar(im, ax, text):
        b = fig.colorbar(im, ax=ax, location="bottom", fraction=0.05, pad=0.02)
        b.set_label(text)
        b.outline.set_edgecolor(P["muted"])

    ax = axes[0]
    im = ax.imshow(g["HRR"] / 1e9, cmap="inferno", **kw)
    zst(ax)
    bar(im, ax, "HRR [GW/m³]")
    ax.set_title("HRR, Z_st (dashed)", fontsize=12)
    ax.set_ylabel("y [mm]")

    ax = axes[1]
    sign = np.where(g["HRR"] > 0.02 * g["HRR"].max(), np.sign(g["FLAME_INDEX"]), 0.0)
    cmap = ListedColormap([P["blue"], P["faint"], P["flame"]])
    im = ax.imshow(sign, cmap=cmap, vmin=-1.5, vmax=1.5, **kw)
    ax.contour(xc, yc, g["TF_OMEGA"], [0.5], colors=[P["fg"]], linewidths=1.8)
    zst(ax, P["muted"])
    ax.set_title("FLAME_INDEX sign, TF_OMEGA = 0.5 (line)", fontsize=12)
    b = fig.colorbar(im, ax=ax, location="bottom", fraction=0.05, pad=0.02, ticks=[-1, 0, 1])
    b.ax.set_xticklabels(["diffusion", "HRR < 2% of peak", "premixed"])
    b.outline.set_edgecolor(P["muted"])

    ax = axes[2]
    im = ax.imshow(g["TF_F"], cmap=P["cmap"], vmin=1.0, **kw)
    zst(ax)
    bar(im, ax, "TF_F")
    ax.set_title("TF_F", fontsize=12)
    for a in axes:
        a.set_xlabel("x [mm]")
    return fig


def figures():
    figstyle.save("triple_flame_sensor", draw)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--compute", action="store_true")
    ap.add_argument("--mallard", default=str(ROOT / "build" / "src" / "mallard"))
    ap.add_argument("--runs", default="triple_flame_sensor_runs")
    args = ap.parse_args()
    if args.compute:
        compute(args.mallard, args.runs)
    figures()


if __name__ == "__main__":
    main()
