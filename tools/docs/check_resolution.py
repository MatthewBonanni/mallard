"""Figure of the resolution report of `mallard check` per region (docs/guide/input_files.md#values-per-region):
a developing laminar channel flow checked against the DNS targets, with the dissipation of an earlier run's
statistics, in two regions, `entrance` and `downstream`.

    python tools/docs/check_resolution.py [--compute --mallard build/src/mallard]

--compute runs the earlier run (air at 20 m/s entering a 50 mm x 10 mm channel of 80 x 48 cells between
adiabatic walls, for 1.2 flow-through times, averaging STRAIN_RATE over the last third: about 4 min on 2
threads), checks the case from its restart file with --mesh-fields and --format json, and writes
check_resolution.json next to this script; without it the figure reads that file, so drawing it needs
NumPy and Matplotlib only.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

import figstyle

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "check_resolution.json"
REGIONS = {"entrance": (0.0, 0.0125), "downstream": (0.0125, 0.05)}

CASE = """[run]
t_stop = 3.0e-3
cfl = 0.9

[mesh]
type = "cartesian"
Nx = 80
Ny = 48
Lx = 0.05
Ly = 0.01
stretching = [0.0, 2.2]

[initialize]
type = "constant"
u = [20.0, 0.0]
p = "1 atm"
T = "300 K"

[[boundaries]]
name = "left"
type = "inlet_state"
u = [20.0, 0.0]
p = "1 atm"
T = "300 K"

[[boundaries]]
name = "right"
type = "outlet_pressure"
p = "1 atm"

[[boundaries]]
name = "bottom"
type = "wall_adiabatic"

[[boundaries]]
name = "top"
type = "wall_adiabatic"

[numerics.face_reconstruction]
type = "MUSCL"

[physics]
type = "navier_stokes"
gamma = 1.4
p_ref = "1 atm"
T_ref = "288.15 K"
rho_ref = 1.225
mu = 1.8e-5
"""

RUN = """
[statistics]
fields = ["STRAIN_RATE"]
products = ["STRAIN_RATE*STRAIN_RATE"]
t_start = 2.0e-3

[[write_data]]
prefix = "restart/channel"
format = "restart"
time_interval = 3.0e-3
"""


def check_case(restart):
    regions = "\n".join(f'{name} = {{type = "rectangle", center = [{(a + b) / 2}, 0.005], size = [{b - a}, 0.01]}}'
                        for name, (a, b) in REGIONS.items())
    return CASE + f"""
[regions]
{regions}

[check]
regions = [{", ".join(f'"{n}"' for n in REGIONS)}]
statistics = "{restart}"
"""


def compute(mallard):
    sys.path.insert(0, str(ROOT / "tools"))
    from mallard.vtu import read_vtu_cells

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "run.toml").write_text(CASE + RUN)
        subprocess.run([str(mallard), "-i", "run.toml", "--kokkos-num-threads=2"], cwd=work, check=True,
                       stdout=subprocess.DEVNULL)
        restart = sorted((work / "restart").glob("channel_*.restart"))[-1].relative_to(work)
        (work / "case.toml").write_text(check_case(restart))
        out = subprocess.run([str(mallard), "check", "case.toml", "--restart", str(restart), "--mesh-fields", "mesh",
                              "--format", "json", "--kokkos-num-threads=2"], cwd=work, capture_output=True, text=True)
        report = json.loads(out.stdout)
        pts, conn, offs, _, arrays = read_vtu_cells(str(work / "mesh_000000.vtu"))
    starts = np.concatenate([[0], offs[:-1]])
    centers = np.array([pts[conn[a:b], :2].mean(axis=0) for a, b in zip(starts, offs)])
    x, y = np.unique(np.round(pts[:, 0], 12)), np.unique(np.round(pts[:, 1], 12))
    order = np.lexsort((centers[:, 0], centers[:, 1]))
    shape = (len(y) - 1, len(x) - 1)
    rows = [dict(quantity=r["quantity"], value=r["value"], target=r["target"], status=r["status"],
                 region=r["region"]) for r in report["resolution"].values()
            if r["target"] and r["value"] is not None]
    DATA.write_text(json.dumps(dict(
        x=x.tolist(), y=y.tolist(),
        delta_over_eta=np.round(arrays["DELTA_OVER_ETA"][order].reshape(shape), 4).tolist(),
        y_plus_bottom=np.round(arrays["Y_PLUS"][order].reshape(shape)[0], 4).tolist(),
        rows=rows), indent=1) + "\n")


def draw(P):
    d = json.loads(DATA.read_text())
    x, y = np.array(d["x"]) * 1e3, np.array(d["y"]) * 1e3
    field = np.array(d["delta_over_eta"])
    fig = plt.figure(figsize=(10.8, 6.4))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], width_ratios=[1.0, 0.85], hspace=0.5, wspace=0.62)
    ax = fig.add_subplot(grid[0, :])
    X, Y = np.meshgrid(x, y)
    polys = [[(X[j, i], Y[j, i]), (X[j, i + 1], Y[j, i + 1]), (X[j + 1, i + 1], Y[j + 1, i + 1]),
              (X[j + 1, i], Y[j + 1, i])] for j in range(field.shape[0]) for i in range(field.shape[1])]
    norm = LogNorm(0.3, 30)
    pc = PolyCollection(polys, array=field.ravel(), cmap=P["cmap"], norm=norm, edgecolors="face", linewidths=0.2,
                        rasterized=False)
    ax.add_collection(pc)
    for name, (a, b) in REGIONS.items():
        ax.text((a + b) / 2 * 1e3, y[-1] + 0.35, name, ha="center", va="bottom", fontsize=12, color=P["fg"])
    ax.axvline(REGIONS["entrance"][1] * 1e3, color=P["bg"], lw=2.0, ls=(0, (4, 2)))
    xm, ym = 0.5 * (x[1:] + x[:-1]), 0.5 * (y[1:] + y[:-1])
    cs = ax.contour(xm, ym, field, levels=[2.0], colors=[P["bg"]], linewidths=1.2)
    ax.clabel(cs, fmt={2.0: "2"}, fontsize=9, inline_spacing=2, manual=[(30, ym[np.argmin(abs(field[:, 50] - 2))])])
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y[0], y[-1])
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    cb = fig.colorbar(pc, ax=ax, fraction=0.03, pad=0.015)
    cb.set_label("DELTA_OVER_ETA")
    cb.outline.set_visible(False)

    # First-cell y+ along the bottom wall
    bx = fig.add_subplot(grid[1, 0])
    xc = 0.5 * (x[1:] + x[:-1])
    bx.plot(xc, d["y_plus_bottom"], color=P["teal"], lw=2.0)
    bx.axhline(1.0, color=P["flame"], lw=1.4, ls=(0, (4, 2)))
    for name, (a, b) in REGIONS.items():
        bx.text((a + b) / 2 * 1e3, 3.9, name, ha="center", va="bottom", fontsize=11, color=P["muted"])
    bx.text(x[-1], 1.08, "DNS target", ha="right", va="bottom", color=P["flame"], fontsize=11)
    bx.axvline(REGIONS["entrance"][1] * 1e3, color=P["muted"], lw=1.0, ls=(0, (4, 2)))
    bx.set_xlim(x[0], x[-1])
    bx.set_ylim(0, 4.3)
    bx.set_xlabel("x [mm]")
    bx.set_ylabel("first-cell y+ (Y_PLUS)")

    # The checked rows: value over target, by status
    cx = fig.add_subplot(grid[1, 1])
    rows = [r for r in d["rows"] if r["quantity"].split(": ")[-1] in ("first-cell y+, largest", "cell size / eta, mean")]
    rank = {None: 0, **{name: k + 1 for k, name in enumerate(REGIONS)}}
    rows.sort(key=lambda r: (rank[r["region"]], r["quantity"].split(": ")[-1]))
    labels, ratio, colors, text = [], [], [], []
    for r in rows:
        target = float(r["target"].split()[1])
        short = {"first-cell y+, largest": "y+, largest", "cell size / eta, mean": "cell size / eta, mean"}
        labels.append(f"{r['region'] or 'domain'}: {short[r['quantity'].split(': ')[-1]]}")
        ratio.append(r["value"] / target)
        colors.append(P["teal"] if r["status"] == "ok" else P["flame"])
        text.append(f"{r['value']:.3g} ({r['target']})".replace(" (DNS)", ""))
    k = np.arange(len(rows))[::-1]
    cx.barh(k, ratio, color=colors, height=0.62)
    cx.axvline(1.0, color=P["fg"], lw=1.0)
    cx.set_xscale("log")
    cx.set_xlim(0.3, 40)
    cx.set_yticks(k, labels, fontsize=10.5)
    for kk, v, t in zip(k, ratio, text):
        cx.text(v * 1.12, kk, t, va="center", ha="left", fontsize=10, color=P["fg"])
    cx.set_xlabel("value / DNS target")
    cx.tick_params(axis="y", length=0)
    cx.spines["left"].set_visible(False)
    return fig


def figures():
    figstyle.save("check_resolution", draw)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--compute", action="store_true", help="run the case and its check into check_resolution.json")
    ap.add_argument("--mallard", help="2D Mallard executable, for --compute")
    args = ap.parse_args()
    if args.compute:
        if not args.mallard:
            ap.error("--compute needs --mallard")
        compute(Path(args.mallard).resolve())
    figures()


if __name__ == "__main__":
    main()
