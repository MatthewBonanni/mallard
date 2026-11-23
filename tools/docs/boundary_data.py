"""Figure of profiles from data at boundaries (docs/guide/boundary_conditions.md,
"Profiles from data"): one table on two inlets of different width, read at
the wall distance in meters (at = ["n_m"]) and at the normalized coordinate
along the inlet (at = ["u"]), the face values Mallard sets against the table's
rows.

    python tools/docs/boundary_data.py [--compute --mallard build/src/mallard]

--compute runs Mallard on a 2D box (one step each, a few seconds) whose left
side is a 30 mm inlet, a 20 mm wall and a 50 mm inlet, and writes the face
velocities of its boundary probe on that side to boundary_data.json next to
this script; without it the figure reads that file, so drawing needs NumPy and
Matplotlib only.
"""
import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt

import figstyle
from figstyle import bare, label

DATA = Path(__file__).resolve().parent / "boundary_data.json"
LX, LY, NX, NY = 0.04, 0.1, 16, 64
U0 = 10.0
INLETS = (("y < 0.03", 0.0, 0.03), ("y >= 0.05", 0.05, 0.1))
WALL = (0.03, 0.05)

# A Pohlhausen boundary layer 10 mm thick, every 1 mm (wall distance in mm), and
# a laminar channel profile across the inlet (0 to 1), every 0.1
BL = [(d, 2 * e - 2 * e ** 3 + e ** 4) for d in range(11) for e in [d / 10]]
CHANNEL = [(s, 4 * s * (1 - s)) for s in np.round(np.linspace(0, 1, 11), 2)]
CASES = {
    "n_m": dict(table=BL, header="d,u", data='at = ["n_m"]\nunits = { d = "mm" }'),
    "u": dict(table=CHANNEL, header="s,u", data='at = ["u"]'),
}

BASE = f"""
[run]
n_steps = 1
cfl = 0.5

[mesh]
type = "cartesian"
Nx = {NX}
Ny = {NY}
Lx = {LX}
Ly = {LY}

[initialize]
type = "constant"
u = [0.0, 0.0]
p = 1.0e5
T = 300.0
{{inlets}}
[[boundaries]]
name = "left"
type = "wall_adiabatic"
where = "y >= {WALL[0]} and y < {WALL[1]}"

[[boundaries]]
name = "right"
type = "outlet_pressure"
p = 1.0e5

[[boundaries]]
name = "bottom"
type = "wall_adiabatic"

[[boundaries]]
name = "top"
type = "wall_adiabatic"

[physics]
type = "navier_stokes"
gamma = 1.4
p_ref = 1.0e5
T_ref = 300.0
rho_ref = 1.1614
mu = 1.85e-5
Pr = 0.72

[[probes]]
name = "left"
boundary = "left"
fields = ["U"]
interval = 1
"""


def inlet(where, data):
    return f"""
[[boundaries]]
name = "left"
type = "inlet_state"
where = "{where}"
u = "{U0} * prof_u"
p = 1.0e5
T = 300.0

[[boundaries.data]]
name = "prof"
file = "profile.csv"
interpolation = "cubic"
{data}
"""


def compute(mallard):
    out = {"U0": U0, "inlets": [i[1:] for i in INLETS], "wall": WALL, "mesh": [LX, LY, NX, NY]}
    for key, case in CASES.items():
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "profile.csv").write_text(case["header"] + "\n" + "".join(f"{a:g},{b:.6g}\n" for a, b in case["table"]))
            (tmp / "input.toml").write_text(BASE.format(inlets="".join(inlet(w, case["data"]) for w, *_ in INLETS)))
            run = subprocess.run([mallard, "-i", "input.toml", "--kokkos-num-threads=2"], cwd=tmp,
                                 capture_output=True, text=True)
            if run.returncode:
                sys.exit(run.stdout + run.stderr)
            files = sorted(tmp.rglob("probe_left*.csv"))
            if not files:
                sys.exit(f"no probe output in {tmp}: " + ", ".join(str(p) for p in tmp.rglob("*")))
            rows = list(csv.DictReader(line for line in files[0].read_text().splitlines() if not line.startswith("#")))
            first = [r for r in rows if r.get("step", "0") == rows[0].get("step", "0")]
            faces = sorted((round(float(r["y"]), 8), round(float(r["U_X"]), 6)) for r in first)
        out[key] = dict(table=[[float(a), round(float(b), 6)] for a, b in case["table"]], faces=faces)
    DATA.write_text(json.dumps(out, indent=1) + "\n")


def table_points(d, key, lo, hi):
    """The table's rows placed on an inlet from lo to hi: (y, u) where the face coordinate equals a row's."""
    pts = []
    for c, u in d[key]["table"]:
        if key == "n_m":
            if c * 1e-3 <= 0.5 * (hi - lo):
                pts += [(lo + c * 1e-3, u), (hi - c * 1e-3, u)]
        else:
            pts.append((lo + c * (hi - lo), u))
    return pts


def draw(P, d):
    lx, ly, nx, ny = d["mesh"]
    scale = 0.024 / d["U0"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 7.0), gridspec_kw=dict(wspace=0.05))
    titles = {"n_m": 'at = ["n_m"]: d in mm,\nthe same 10 mm layers', "u": 'at = ["u"]: s from 0 to 1,\nstretched over each inlet'}
    for ax, key in zip(axes, ("n_m", "u")):
        bare(ax)
        xs = np.linspace(0, lx, nx + 1)
        ys = np.linspace(0, ly, ny + 1)
        grid = [[(x, 0), (x, ly)] for x in xs] + [[(0, y), (lx, y)] for y in ys]
        ax.add_collection(LineCollection(grid, colors=P["mesh"], lw=0.4, zorder=0))
        ax.plot([0, lx], [0, 0], color=P["line"], lw=2.5)
        ax.plot([0, lx], [ly, ly], color=P["line"], lw=2.5)
        ax.plot([0, 0], d["wall"], color=P["line"], lw=2.5)
        for lo, hi in d["inlets"]:
            ax.plot([0, 0], [lo, hi], color=P["flame"], lw=3)
            label(ax, -0.003, (lo + hi) / 2, f"inlet, {round((hi - lo) * 1e3)} mm", P, color=P["flame"],
                  size=11, rotation=90, ha="right")
        faces = np.array(d[key]["faces"])
        inside = np.array([any(lo <= y <= hi for lo, hi in d["inlets"]) for y in faces[:, 0]])
        f = faces[inside]
        ax.quiver(np.zeros(len(f)), f[:, 0], f[:, 1] * scale, np.zeros(len(f)), angles="xy", scale_units="xy",
                  scale=1, color=P["teal"], width=0.006, headwidth=3.5, headlength=4, zorder=3)
        for lo, hi in d["inlets"]:
            pts = np.array(table_points(d, key, lo, hi))
            ax.plot(pts[:, 1] * d["U0"] * scale, pts[:, 0], "o", ms=5, mfc=P["bg"], mec=P["plum"], mew=1.4, zorder=4)
        ax.set_title(titles[key], fontsize=12, color=P["fg"])
        ax.set_xlim(-0.012, lx + 0.001)
        ax.set_ylim(-0.003, ly + 0.003)
    handles = [plt.Line2D([], [], color=P["teal"], lw=2),
               plt.Line2D([], [], ls="none", marker="o", ms=5, mfc=P["bg"], mec=P["plum"], mew=1.4)]
    fig.legend(handles, [f"face values Mallard sets (arrow length: speed, {d['U0']:g} m/s in the core)",
                         "rows of the table"], loc="lower center", ncol=1, frameon=False, fontsize=11)
    fig.subplots_adjust(bottom=0.11)
    return fig


def figures():
    d = json.loads(DATA.read_text())
    figstyle.save("boundary_data", lambda P: draw(P, d))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--compute", action="store_true")
    ap.add_argument("--mallard", default="build/src/mallard")
    args = ap.parse_args()
    if args.compute:
        compute(args.mallard)
    figures()


if __name__ == "__main__":
    main()
