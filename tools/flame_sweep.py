"""Equivalence-ratio sweep of 1D premixed flames against Cantera (V8).

    python tools/flame_sweep.py setup FULL_DIR RUNS_DIR [--fuel ch4] [--phi 0.6 ...]
        [--transport mix unity] [--cells 20] [--cfl 1.0] [--flame-times 5]
    python tools/flame_sweep.py report RUNS_DIR [--csv OUT.csv] [--png OUT.png]

setup writes one run per fuel, phi, transport model, resolution and CFL with
tools/flame_restart.py, from the full solutions of tools/flame_reference.py
(FULL_DIR/<case>_full.csv), into RUNS_DIR/<case>_c<cells>_cfl<cfl>, with ten
outputs per flame time delta_T / S_L.

report reads every run in RUNS_DIR and, against Cantera's flame speed S_L
(examples/premixed_flame/reference/flames.csv), gives the consumption speed
S_c averaged over the last flame time, its drift (the mean over the last half
flame time against the mean over the half before), the displacement speed S_d
(inflow speed minus the front's drift, fitted over the last flame time), and,
for the last output, the peak heat release rate against Cantera's and the
largest temperature difference (tools/plot_flame.py). With --png it plots the
flame speeds against phi and their errors for each resolution and CFL.
"""
import argparse
import csv
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from flame_reference import FUELS, MODELS, case_name  # noqa: E402
from flame_speed import analyze  # noqa: E402
from plot_flame import compare  # noqa: E402

REF_DIR = os.path.join(ROOT, "examples", "premixed_flame", "reference")
TRANSPORT = {"mix": "mixture_averaged", "unity": "unity_lewis"}
RUN = re.compile(r"(?P<case>(?P<fuel>\w+?)_phi(?P<phi>[\d.]+)_(?P<model>mix|unity))_c(?P<cells>[\d.]+)_cfl(?P<cfl>[\d.]+)$")


def references():
    with open(os.path.join(REF_DIR, "flames.csv")) as f:
        rows = csv.DictReader(line for line in f if not line.startswith("#"))
        return {case_name(r["fuel"], float(r["phi"]), r["transport"]): r for r in rows}


def setup(args):
    for phi in args.phi:
        for model in args.transport:
            case = case_name(args.fuel, phi, model)
            mech, phase = FUELS[args.fuel][:2]
            for cells in args.cells:
                for cfl in args.cfl:
                    run_dir = os.path.join(args.runs_dir, f"{case}_c{cells:g}_cfl{cfl:g}")
                    subprocess.run([sys.executable, os.path.join(HERE, "flame_restart.py"),
                                    os.path.join(args.full_dir, case + "_full.csv"), os.path.join(ROOT, mech), phase,
                                    f"{cells:g}", run_dir, "--cfl", f"{cfl:g}", "--transport", TRANSPORT[model],
                                    "--flame-times", f"{args.flame_times:g}",
                                    "--outputs", str(int(round(10 * args.flame_times)))], check=True)


def measure(run_dir, ref):
    S_L, tau = float(ref["S_L"]), float(ref["delta_T"]) / float(ref["S_L"])
    rows = analyze(run_dir)
    t, S_c, x_f, u_in = rows.T
    end = t[-1]
    last = t >= end - tau * (1 + 1e-6)
    late, early = t >= end - 0.5 * tau * (1 + 1e-6), last & (t < end - 0.5 * tau * (1 + 1e-6))
    S_d = u_in[last].mean() - np.polyfit(t[last], x_f[last], 1)[0]
    dT, hrr = compare(run_dir, os.path.join(REF_DIR, ref["case"] + ".csv"))
    return {"flame_times": end / tau, "S_c": S_c[last].mean(),
            "drift": 100 * (S_c[late].mean() / S_c[early].mean() - 1),
            "S_d": S_d, "err_c": 100 * (S_c[last].mean() / S_L - 1), "err_d": 100 * (S_d / S_L - 1),
            "dT_max": dT, "err_hrr": hrr}


def report(args):
    refs = references()
    results = []
    for name in sorted(os.listdir(args.runs_dir)):
        m = RUN.match(name)
        run_dir = os.path.join(args.runs_dir, name)
        if not m or not os.path.exists(os.path.join(run_dir, "solut", "flame.pvd")):
            continue
        ref = dict(refs[m["case"]], case=m["case"])
        r = dict(fuel=m["fuel"], phi=float(m["phi"]), transport=m["model"], cells=float(m["cells"]),
                 cfl=float(m["cfl"]), S_L=float(ref["S_L"]), **measure(run_dir, ref))
        results.append(r)
    keys = ["fuel", "phi", "transport", "cells", "cfl", "flame_times", "S_L", "S_c", "err_c", "drift", "S_d",
            "err_d", "err_hrr", "dT_max"]
    print(" ".join(f"{k:>11}" for k in keys))
    for r in results:
        print(" ".join(f"{r[k]:>11.5g}" if isinstance(r[k], float) else f"{r[k]:>11}" for k in keys))
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(keys)
            for r in results:
                w.writerow([f"{r[k]:.6g}" if isinstance(r[k], float) else r[k] for k in keys])
    if args.png:
        plot(results, refs, args.png)


def plot(results, refs, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fuel = results[0]["fuel"]
    colors = {"mix": "C0", "unity": "C3"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for model, label in MODELS.items():
        ref = sorted((float(r["phi"]), float(r["S_L"])) for r in refs.values()
                     if r["fuel"] == fuel and r["transport"] == model)
        ax1.plot(*zip(*ref), "-", color=colors[model], label=f"Cantera, {label}")
        base = sorted((r["phi"], r["S_c"]) for r in results
                      if r["transport"] == model and r["cells"] == 20 and r["cfl"] == 1.0)
        if base:
            ax1.plot(*zip(*base), "o", color=colors[model], mfc="none", label=f"Mallard, {label}")
    ax1.set_xlabel("equivalence ratio")
    ax1.set_ylabel("flame speed [m/s]")
    ax1.legend(fontsize=8)
    markers = {10: "v", 20: "o", 40: "^"}
    for model in MODELS:
        for (cells, cfl) in sorted({(r["cells"], r["cfl"]) for r in results if r["transport"] == model}):
            pts = sorted((r["phi"], r["err_c"]) for r in results
                         if r["transport"] == model and r["cells"] == cells and r["cfl"] == cfl)
            ax2.plot(*zip(*pts), ("-" if cfl == 1.0 else ":") + markers.get(int(cells), "s"),
                     color=colors[model], mfc="none" if cfl == 1.0 else colors[model], ms=5,
                     label=f"{model}, {cells:g} cells, CFL {cfl:g}")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_xlabel("equivalence ratio")
    ax2.set_ylabel("consumption speed error vs Cantera [%]")
    ax2.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(png, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup")
    s.add_argument("full_dir")
    s.add_argument("runs_dir")
    s.add_argument("--fuel", default="ch4", choices=list(FUELS))
    s.add_argument("--phi", type=float, nargs="+", default=[0.6, 0.8, 1.0, 1.2, 1.4])
    s.add_argument("--transport", nargs="+", default=list(MODELS), choices=list(MODELS))
    s.add_argument("--cells", type=float, nargs="+", default=[20.0])
    s.add_argument("--cfl", type=float, nargs="+", default=[1.0])
    s.add_argument("--flame-times", type=float, default=5.0)
    r = sub.add_parser("report")
    r.add_argument("runs_dir")
    r.add_argument("--csv")
    r.add_argument("--png")
    args = ap.parse_args()
    setup(args) if args.cmd == "setup" else report(args)


if __name__ == "__main__":
    main()
