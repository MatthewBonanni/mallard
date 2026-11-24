"""Figures of `mallard post flame` (docs/guide/thermochemistry.md "Flame speed and
thickness"), from the stoichiometric H2/air flame of the premixed flame tutorial
(tutorials/premixed_flame/input.toml: 300 cells, 20 per thermal thickness, four
flame times from a tanh profile):

- post_flame_profile: the profile of the last snapshot with what the command
  reads from it: the fresh and burnt temperatures, the flame position x_f where
  T is midway between them, the steepest gradient and the thermal thickness it
  gives, and the consumption of H2 whose integral is S_c;
- post_flame_speeds: S_c and S_d of every snapshot against time, with the
  windows the summary averages and fits over, against Cantera's S_L.

    python tools/docs/post_flame.py [--compute --mallard build/src/mallard] [names...]

--compute runs the tutorial's input on a Cartesian strip of the same cells (its
mesh file needs Gmsh), with restart files at the snapshot times (post flame
reads restart files of any build, snapshots only as VTKHDF), measures it with
`mallard post flame` and writes post_flame.json next to this script (about 7 min
on 2 threads); without it the figures read that file, so drawing them needs
NumPy and Matplotlib only.
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import figstyle
from figstyle import label

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "post_flame.json"
S_L = 2.33108         # Cantera's flame speed, as in the tutorial
T_EQ = 2379.43        # Cantera's adiabatic equilibrium temperature of the fresh gas
X0, X1 = -0.8, 1.6    # the profile's window about x_f [mm]
SPECIES = ["H2", "O2", "H2O"]
CELLS, LENGTH = 300, 4.925194e-3  # the strip of tutorials/premixed_flame/strip.geo (mesh.sh)


# ------------------------------------------------------------------- compute
def tutorial_input():
    s = (ROOT / "tutorials" / "premixed_flame" / "input.toml").read_text()
    s = s.replace('type = "file"\nfilename = "strip.msh"',
                  f'type = "cartesian"\nNx = {CELLS}\nNy = 1\nLx = {LENGTH:.6e}\nLy = {10 * LENGTH / CELLS:.6e}')
    s = s.replace('name = "inlet"', 'name = "left"').replace('name = "outlet"', 'name = "right"')
    s = s.replace('name = "sides"\ntype = "symmetry"',
                  'name = "bottom"\ntype = "symmetry"\n\n[[boundaries]]\nname = "top"\ntype = "symmetry"')
    s = re.sub(r'mechanism = "[^"]*"', f'mechanism = "{ROOT / "mechanisms" / "h2o2.yaml"}"', s)
    interval = re.search(r"time_interval = ([0-9.e+-]+)", s).group(1)
    return s + f'\n[[write_data]]\nprefix = "./restart/flame"\nformat = "restart"\ntime_interval = {interval}\n'


def compute(mallard, runs=None):
    sys.path[:0] = [str(ROOT / "tools")]
    from mallard.vtu import read_vtu_cells

    mallard = str(Path(mallard).resolve())
    work = Path(runs) if runs else Path(tempfile.mkdtemp(prefix="post_flame_"))
    work.mkdir(parents=True, exist_ok=True)
    (work / "input.toml").write_text(tutorial_input())
    with open(work / "run.log", "w") as log:
        subprocess.run([mallard, "-i", "input.toml", "--kokkos-num-threads=2"], cwd=work, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    restarts = sorted(str(p.relative_to(work)) for p in (work / "restart").glob("flame_*.restart"))
    out = subprocess.run([mallard, "post", "flame", *restarts, "--reference-speed", str(S_L), "--format", "json"],
                         cwd=work, capture_output=True, text=True, check=True).stdout
    report = json.loads(out)
    last = sorted((work / "solut").glob("flame_*.vtu"))[-1]
    pts, conn, offs, _, cells = read_vtu_cells(str(last))
    starts = np.concatenate([[0], offs[:-1]])
    x = np.array([pts[conn[a:b], 0].mean() for a, b in zip(starts, offs)])
    order = np.argsort(x)
    profile = dict(x=x[order].tolist(), T=cells["T"][order].tolist(),
                   OMEGA_H2=cells["OMEGA_H2"][order].tolist(),
                   **{f"Y_{k}": cells[f"Y_{k}"][order].tolist() for k in SPECIES})
    rows = [dict(t=r["snapshot"]["t"], **{k: r[k] for k in ("x_f", "delta_T", "dTdx_max", "S_c", "S_d", "T_u", "T_b")})
            for r in report["snapshots"]]
    DATA.write_text(json.dumps(dict(profile=profile, rows=rows, summary=report["summary"], species=report["species"],
                                    t_profile=float(cells["TIME"]))) + "\n")
    print(f"wrote {DATA.name}")


# ------------------------------------------------------------------- figures
def load():
    d = json.loads(DATA.read_text())
    d["profile"] = {k: np.asarray(v) for k, v in d["profile"].items()}
    d["rows"] = {k: np.array([r[k] if r[k] is not None else np.nan for r in d["rows"]]) for k in d["rows"][0]}
    return d


def profile_figure(P, d):
    p, r = d["profile"], d["rows"]
    x, T = p["x"], p["T"]
    T_u, T_b, x_f = r["T_u"][-1], r["T_b"][-1], r["x_f"][-1]
    xi = (x - x_f) * 1e3
    g = np.diff(T) / np.diff(x)
    i = int(np.argmax(np.abs(g)))
    xg, Tg, slope = 0.5 * (x[i] + x[i + 1]), 0.5 * (T[i] + T[i + 1]), g[i]
    xa, xb = (xg + (T_u - Tg) / slope - x_f) * 1e3, (xg + (T_b - Tg) / slope - x_f) * 1e3

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ay = ax.twinx()
    ay.spines["right"].set_visible(True)
    w = -p["OMEGA_H2"]
    ay.fill_between(xi, 0, 0.9 * 0.3 * w / w.max(), color=P["flame"], alpha=0.16, lw=0)
    ki = int(np.argmax(w))
    label(ay, xi[ki] + 0.3, 0.12, "$-$OMEGA_H2: its\nintegral gives S_c", P, color=P["flame"], size=11, ha="left")
    colors = dict(H2=P["teal"], O2=P["blue"], H2O=P["plum"])
    for k in SPECIES:
        ay.plot(xi, p[f"Y_{k}"], color=colors[k], lw=1.4, alpha=0.9)
    ay.set_ylim(0, 0.3)
    ay.set_ylabel("mass fraction")
    label(ay, X0 + 0.05, p["Y_O2"][0] - 0.012, "Y_O2", P, color=colors["O2"], size=11, ha="left")
    label(ay, X0 + 0.05, p["Y_H2"][0] - 0.010, "Y_H2", P, color=colors["H2"], size=11, ha="left")
    label(ay, X1 - 0.05, np.interp(X1, xi, p["Y_H2O"]) - 0.012, "Y_H2O", P, color=colors["H2O"], size=11,
          ha="right")

    ax.plot(xi, T, color=P["fg"], lw=2.2, zorder=4)
    ax.set_zorder(ay.get_zorder() + 1)
    ax.patch.set_visible(False)
    for v, ls in ((T_u, "--"), (T_b, "--"), (T_EQ, ":")):
        ax.plot((X0, X1), (v, v), color=P["muted"], lw=1.0, ls=ls)
    label(ax, X1 - 0.05, T_u + 70, f"T_u = {T_u:.0f} K, at the fresh end", P, color=P["muted"], size=11, ha="right")
    label(ax, X0 + 0.05, T_b - 90, f"T_b = {T_b:.0f} K, the peak", P, color=P["muted"], size=11, ha="left")
    label(ax, X0 + 0.05, T_EQ + 75, f"equilibrium, {T_EQ:.0f} K (--burnt-temperature)", P, color=P["muted"],
          size=11, ha="left")
    ax.plot((xa, xb), (T_u, T_b), color=P["red"], lw=1.4)
    ax.plot((xg - x_f) * 1e3, Tg, "o", color=P["red"], ms=5, zorder=5)
    label(ax, (xg - x_f) * 1e3 - 0.05, Tg + 70, "max |dT/dx|", P, color=P["red"], size=11, ha="right")
    yb = 480
    ax.annotate("", (xa, yb), (xb, yb), arrowprops=dict(arrowstyle="<->", color=P["red"], lw=1.2, shrinkA=0, shrinkB=0))
    for xx in (xa, xb):
        ax.plot((xx, xx), (yb - 60, yb + 60), color=P["red"], lw=1.0)
    label(ax, xb + 0.05, yb, "delta_T = (T_b $-$ T_u) / max |dT/dx|", P, color=P["red"], size=11, ha="left")
    T_mid = 0.5 * (T_u + T_b)
    ax.plot((0, 0), (0, T_mid), color=P["fg"], lw=1.0, ls=(0, (2, 2)))
    ax.plot(0, T_mid, "s", color=P["fg"], ms=6, zorder=5)
    label(ax, -0.08, T_mid + 120, "x_f: T = (T_u + T_b) / 2", P, size=11, ha="right")
    ax.set_xlim(X0, X1)
    ax.set_ylim(0, 2600)
    ax.set_xlabel("x $-$ x_f [mm]")
    ax.set_ylabel("T [K]")
    return fig


def speeds_figure(P, d):
    r, s = d["rows"], d["summary"]
    t = r["t"] * 1e3
    fig, (top, ax) = plt.subplots(2, 1, figsize=(8.6, 5.4), sharex=True,
                                  gridspec_kw=dict(height_ratios=[1, 2.2], hspace=0.08))
    zoom = (2.15, 2.48)
    for a in (top, ax):
        a.plot(t, r["S_c"], "o-", color=P["flame"], ms=4, lw=1.6, label="S_c, consumption speed")
        a.plot(t, r["S_d"], "s-", color=P["teal"], ms=4, lw=1.4, label="S_d, displacement speed")
        a.axhline(S_L, color=P["muted"], lw=1.0, ls="--")
    top.axhspan(*zoom, color=P["muted"], alpha=0.12, lw=0)
    top.set_ylim(-1.2, 4.3)
    top.set_ylabel("speed [m/s]")
    label(top, t[-1], 3.2, "the start from a tanh profile (below: the shaded band)", P, color=P["muted"], size=11,
          ha="right")
    top.legend(loc="lower right", ncol=2)
    t_c, t_d = s["S_c_from"] * 1e3, s["S_d_from"] * 1e3
    ax.axvspan(t_c, t[-1], color=P["flame"], alpha=0.10, lw=0)
    ax.set_ylim(*zoom)
    label(ax, 0.01, S_L + 0.01, f"Cantera, S_L = {S_L} m/s", P, color=P["muted"], size=11, ha="left", va="bottom")
    for y, t0, color, text in ((2.395, t_c, P["flame"], f"S_c = {s['S_c']:.3f} m/s: the mean over the last third"),
                               (2.44, t_d, P["teal"], f"S_d = {s['S_d']:.3f} m/s: x_f fitted over the last half")):
        ax.plot((t0, t[-1]), (y, y), color=color, lw=2.4, solid_capstyle="butt")
        label(ax, t[-1], y + 0.008, text, P, color=color, size=11, ha="right", va="bottom")
    ax.set_xlim(0, t[-1] * 1.02)
    ax.set_xlabel("t [ms]")
    ax.set_ylabel("speed [m/s]")
    return fig


def figures(names=None):
    d = load()
    for name, draw in (("post_flame_profile", profile_figure), ("post_flame_speeds", speeds_figure)):
        if not names or name in names:
            figstyle.save(name, lambda P, draw=draw: draw(P, d))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--compute", action="store_true")
    ap.add_argument("--mallard", help="2D Mallard executable, for --compute")
    ap.add_argument("--runs", help="directory to run in (default: a temporary one)")
    ap.add_argument("names", nargs="*")
    args = ap.parse_args()
    if args.compute:
        if not args.mallard:
            ap.error("--compute needs --mallard")
        compute(args.mallard, args.runs)
    figures(args.names)


if __name__ == "__main__":
    main()
