"""Figures of the chemistry integrators (docs/design/chemistry.md, docs/guide/thermochemistry.md):

- chemistry_bdf: the largest ignition-delay error of the V1 (H2/air, h2o2) and V2 (CH4/air, GRI-3.0)
  constant-volume ignitions against Cantera, and the throughput of benchmarks/chemistry at dt = 1e-8 and 1e-6 s, of RODAS
  and BDF at each atol (rtol 1e-6);
- chemistry_explicit_step: what [chemistry] explicit_step does across freely propagating H2/air (h2o2) and
  CH4/air (GRI-3.0) flames at phi 1, 300 K and 1 atm: which cells keep the explicit step, which failed it and
  skip it while stiff; and the chemistry time per step of 2D wrinkled flames without and with it.

    python tools/docs/chemistry.py [--compute --mallard build/src/mallard [--threads 2] [--runs DIR]] [names...]

--compute runs mallard reactor (about 3 min) and the flames (about 10 min) and writes chemistry.json next to this
script; without it the figures read that file, so drawing them needs NumPy and Matplotlib only. The flames start
from Cantera's, made in DIR with

    python tools/setup/flame_reference.py --single h2 1.0 300 full    (and ch4, with GRI-3.0)
    python tools/setup/flame_restart.py full/h2_phi1_T300_mix_full.csv mechanisms/h2o2.yaml ohmech 20 flame_h2
    python tools/setup/flame_restart.py full/h2_phi1_T300_mix_full.csv mechanisms/h2o2.yaml ohmech 20 flame2d_h2 \
        --ny 64 --perturb 2
    (and flame_ch4, flame2d_ch4 from full/ch4_phi1_T300_mix_full.csv, mechanisms/gri30.yaml gri30)

Times are wall times on the machine that ran --compute (chemistry.json records its device).
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

import figstyle

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "chemistry.json"
sys.path.insert(0, str(ROOT / "tools"))

MECHANISMS = {"h2o2": ("H2/air, h2o2 (10 species)", "h2o2.toml"),
              "gri30": ("CH4/air, GRI-3.0 (53 species)", "gri30.toml")}
ATOLS = [1e-8, 1e-10, 1e-12, 1e-14, 1e-16]
INTEGRATORS = {"rosenbrock": "RODAS", "bdf": "BDF"}
BENCH_CELLS = {"h2o2": 16384, "gri30": 4096}
BENCH_REPEATS = 10
FLAMES = {"h2": "H2/air, h2o2", "ch4": "CH4/air, GRI-3.0"}
PROFILE_STEPS = {"h2": 2000, "ch4": 1000}
TIMING_STEPS = {"h2": 100, "ch4": 30}


# ------------------------------------------------------------------ mallard runs
def reactor_toml(mech, Y, T, p, end, interval, integrator, atol):
    comp = ", ".join(f'"{k}" = {v:.17g}' for k, v in Y.items() if v > 0)
    return f"""[physics]
gas = "mixture"
mechanism = "{ROOT / 'mechanisms' / (mech + '.yaml')}"

[reactor]
T = {T!r}
p = {p!r}
Y = {{ {comp} }}
end_time = {end!r}
output_interval = {interval!r}
output = "reactor.csv"

[chemistry]
integrator = "{integrator}"
rtol = 1.0e-6
atol = {atol!r}
"""


def ignition_errors(mallard, work, mech, integrator, atol):
    """Largest |tau / tau_Cantera - 1| over the 18 cases of test/data/chemistry/<mech>_ignition.csv, each run to 2
    tau in four calls (as splitting steps of tau / 2 would), and the sub-steps of all of them."""
    path = ROOT / "test" / "data" / "chemistry" / f"{mech}_ignition.csv"
    lines = [l for l in path.read_text().splitlines() if not l.startswith("#")]
    head = lines[0].split(",")
    worst, steps = 0.0, 0
    for line in lines[1:]:
        row = dict(zip(head, map(float, line.split(","))))
        tau = row["tau"]
        Y = {h[3:]: row[h] for h in head if h.startswith("Y0_")}
        (work / "r.toml").write_text(reactor_toml(mech, Y, row["T0"], row["p0"], 2 * tau, tau / 2, integrator, atol))
        out = subprocess.run([mallard, "reactor", "-i", "r.toml", "--kokkos-num-threads=1"], cwd=work,
                             capture_output=True, text=True, check=True).stdout
        t = float(re.search(r"Ignition delay\s+(\S+) s", out).group(1))
        steps += int(re.search(r"Sub-steps\s+(\d+) accepted", out).group(1))
        worst = max(worst, abs(t / tau - 1))
    return worst, steps


def throughput(mallard, work, mech, integrator, atol, threads):
    """benchmarks/chemistry/<mech>.toml with fewer cells and the best of more repeats, at each dt: {dt: (cells per second, mean sub-steps)}."""
    src = (ROOT / "benchmarks" / "chemistry" / MECHANISMS[mech][1]).read_text()
    src = src.replace("../../mechanisms", str(ROOT / "mechanisms"))
    src = re.sub(r"cells = \d+", f"cells = {BENCH_CELLS[mech]}", src)
    src = re.sub(r"atol = .*", f'atol = {atol!r}\nintegrator = "{integrator}"', src)
    src = re.sub(r"output = .*", 'output = "bench.csv"', src)
    src = re.sub(r"repeats = \d+", f"repeats = {BENCH_REPEATS}", src)
    (work / "b.toml").write_text(src)
    subprocess.run([mallard, "reactor", "-i", "b.toml", f"--kokkos-num-threads={threads}"], cwd=work,
                   capture_output=True, text=True, check=True)
    lines = (work / "bench.csv").read_text().splitlines()
    head = lines[0].split(",")
    rows = [dict(zip(head, l.split(","))) for l in lines[1:] if not l.startswith("#")]
    return {f"{float(r['dt']):g}": (float(r["cells_per_second"]), float(r["mean_substeps"])) for r in rows}


def flame(mallard, run, explicit, threads, steps, name):
    """Runs the flame of run/ (input.toml, flame.restart) for steps steps, with or without explicit steps, in
    run/name; returns the run log's start and end records and the final snapshot's cells."""
    from mallard.vtu import read_vtu_cells

    work = run / name
    work.mkdir(exist_ok=True)
    src = (run / "input.toml").read_text()
    src = re.sub(r"\[run\]\n[^\[]*", f"[run]\nn_steps = {steps}\ncfl = 1.0\n\n", src)
    src = re.sub(r'mechanism = ".*/mechanisms/', f'mechanism = "{ROOT}/mechanisms/', src)
    src = re.sub(r"\[\[write_data\]\]\n(?:.*\n)*?fields = .*\n", "", src)
    src = re.sub(r"\[chemistry\]\n", "[chemistry]\n" + ("explicit_step = true\n" if explicit else ""), src)
    src += f"""
[[write_data]]
prefix = "./solut/flame"
format = "vtu"
interval = {steps}
fields = ["T", "CHEM_H", "CHEM_COST"]
"""
    (work / "input.toml").write_text(src)
    (work / "flame.restart").unlink(missing_ok=True)
    (work / "flame.restart").symlink_to(run / "flame.restart")
    (work / "mallard.jsonl").unlink(missing_ok=True)
    subprocess.run([mallard, "-i", "input.toml", f"--kokkos-num-threads={threads}"], cwd=work,
                   capture_output=True, text=True, check=True)
    log = [json.loads(l) for l in (work / "mallard.jsonl").read_text().splitlines()]
    snap = max((work / "solut").glob("flame_*.vtu"))
    return log[0], log[-1], read_vtu_cells(str(snap))


def profile(mallard, run, threads, steps):
    """The cells of a 1D flame after steps steps with explicit steps, along x: T, CHEM_H, CHEM_COST, and the mean dt."""
    _, end, (pts, conn, offs, types, data) = flame(mallard, run, True, threads, steps, "explicit")
    x = np.array([pts[conn[o - 4:o], 0].mean() for o in offs])
    order = np.argsort(x)
    return dict(steps=steps, x=x[order].tolist(), T=data["T"][order].tolist(),
                chem_h=data["CHEM_H"][order].tolist(), cost=data["CHEM_COST"][order].tolist(),
                dt=end["t"] / end["steps"])


def timing(mallard, run, threads, steps, repeats):
    """Chemistry wall time per step of a flame without and with explicit steps, the best of repeats runs each."""
    out = {"rodas": [], "explicit": []}
    for _ in range(repeats):
        for k in out:
            start, end, (pts, conn, offs, types, data) = flame(mallard, run, k == "explicit", threads, steps, k)
            out[k].append(end["wall_chemistry"] / end["steps"])
    return dict(steps=steps, cells=len(offs), device=start["provenance"]["run"]["device"],
                **{k: min(v) for k, v in out.items()})


def compute(mallard, threads, runs, repeats):
    work = Path(runs) / "reactor"
    work.mkdir(parents=True, exist_ok=True)
    out = dict(threads=threads, bdf={}, profiles={}, timings={})
    for mech in MECHANISMS:
        for integ in INTEGRATORS:
            rows = []
            for atol in ATOLS:
                err, steps = ignition_errors(mallard, work, mech, integ, atol)
                rows.append(dict(atol=atol, error=err, substeps=steps,
                                 benchmark=throughput(mallard, work, mech, integ, atol, threads)))
                print(mech, integ, rows[-1], flush=True)
            out["bdf"][f"{mech}/{integ}"] = rows
    for fuel in FLAMES:
        out["profiles"][fuel] = profile(mallard, Path(runs) / f"flame_{fuel}", threads, PROFILE_STEPS[fuel])
        out["timings"][fuel] = timing(mallard, Path(runs) / f"flame2d_{fuel}", threads, TIMING_STEPS[fuel], repeats)
        print(fuel, out["timings"][fuel], flush=True)
    out["device"] = out["timings"]["h2"]["device"]
    DATA.write_text(json.dumps(out, indent=1) + "\n")


# ------------------------------------------------------------------ figures
def load():
    return json.loads(DATA.read_text())


def bdf(P, d):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.6), sharex=True, gridspec_kw=dict(wspace=0.25, hspace=0.12))
    colors = {"rosenbrock": P["teal"], "bdf": P["flame"]}
    for col, mech in enumerate(MECHANISMS):
        top, bottom = axes[0, col], axes[1, col]
        for integ, name in INTEGRATORS.items():
            rows = d["bdf"][f"{mech}/{integ}"]
            atol = np.array([r["atol"] for r in rows])
            top.plot(atol, [r["error"] for r in rows], marker="o", ms=5.5, color=colors[integ], label=name)
            for dt, ls, mfc in (("1e-08", "-", colors[integ]), ("1e-06", "--", "none")):
                bottom.plot(atol, [r["benchmark"][dt][0] for r in rows], marker="o", ms=5.5, ls=ls,
                            color=colors[integ], mfc=mfc, label=f"{name}, dt = {dt.replace('e-0', 'e-')} s")
        top.set_title(MECHANISMS[mech][0], fontsize=13)
        for ax in (top, bottom):
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, which="major", color=P["faint"], lw=0.8)
        bottom.set_xlabel(r"atol (rtol = $10^{-6}$)")
    axes[1, 0].invert_xaxis()
    axes[0, 0].set_ylabel("largest ignition-delay error")
    axes[1, 0].set_ylabel(f"throughput [cells/s]\n{d['threads']} CPU threads")
    axes[0, 0].legend(loc="upper right")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=10.5, bbox_to_anchor=(0.5, -0.04))
    return fig


CATEGORIES = (("kept", "explicit step kept", "teal"),
              ("failed", "explicit step failed: skipped while stiff", "flame"),
              ("rodas", "RODAS only (sub-steps shorter than the call)", "plum"),
              ("inactive", "no chemistry (inactive)", "mesh"))


def category(p):
    h, c = np.array(p["chem_h"]), np.array(p["cost"])
    return {"inactive": c == 0, "kept": (c > 0) & (h == 0), "failed": (c > 0) & (h < 0), "rodas": (c > 0) & (h > 0)}


def explicit_step(P, d):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11.0, 5.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.75, 1.0], wspace=0.32, hspace=0.55)
    for i, fuel in enumerate(FLAMES):
        ax = fig.add_subplot(gs[i, 0])
        p = d["profiles"][fuel]
        x, T = 1e3 * np.array(p["x"]), np.array(p["T"])
        cats = category(p)
        ax.plot(x, T, color=P["line"], lw=1.0, zorder=1)
        for key, text, color in CATEGORIES:
            m = cats[key]
            if m.any():
                ax.scatter(x[m], T[m], s=11, color=P[color], label=text, zorder=2, lw=0)
        ax.set_title(f"{FLAMES[fuel]}, 1D flame after {p['steps']:,} steps of {p['dt'] * 1e9:.0f} ns", fontsize=12)
        ax.set_ylabel("T [K]")
        ax.set_xlim(x[0], x[-1])
        if i == 1:
            ax.set_xlabel("x [mm]")
    handles = {}
    for a in fig.axes:
        for h, l in zip(*a.get_legend_handles_labels()):
            handles.setdefault(l, h)
    order = [t for _, t, _ in CATEGORIES if t in handles]
    fig.legend([handles[t] for t in order], order, loc="upper center", ncol=2, fontsize=10.5,
               bbox_to_anchor=(0.5, 1.07), markerscale=1.6)
    ax = fig.add_subplot(gs[:, 1])
    labels, xs = [], []
    for j, fuel in enumerate(FLAMES):
        t = d["timings"][fuel]
        for k, (key, name, color) in enumerate((("rodas", "RODAS", "muted"), ("explicit", "explicit steps", "teal"))):
            ax.bar(j + 0.38 * (k - 0.5), 1e3 * t[key], width=0.36, color=P[color],
                   label=name if j == 0 else None)
        ax.text(j, 1e3 * max(t["rodas"], t["explicit"]) * 1.04, f"speedup {t['rodas'] / t['explicit']:.2f}",
                ha="center", va="bottom", fontsize=11.5, color=P["fg"])
        labels.append(f"{FLAMES[fuel].split(',')[0]}\n{t['cells']:,} cells")
        xs.append(j)
    ax.set_yscale("log")
    ax.set_xticks(xs, labels)
    ax.set_ylabel(f"chemistry per step [ms], {d['threads']} CPU threads")
    ax.set_title("2D wrinkled flames", fontsize=12)
    ax.legend(loc="upper left", fontsize=10.5)
    ax.margins(y=0.2)
    return fig


def figures(names=None):
    d = load()
    for name, draw in (("chemistry_bdf", bdf), ("chemistry_explicit_step", explicit_step)):
        if not names or name in names:
            figstyle.save(name, lambda P, draw=draw: draw(P, d))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--compute", action="store_true")
    ap.add_argument("--mallard", default=str(ROOT / "build" / "src" / "mallard"))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--runs", default="/tmp/mallard-chemistry-figures")
    ap.add_argument("names", nargs="*")
    args = ap.parse_args()
    if args.compute:
        compute(str(Path(args.mallard).resolve()), args.threads, args.runs, args.repeats)
    figures(args.names)


if __name__ == "__main__":
    main()
