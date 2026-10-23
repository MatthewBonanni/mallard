"""The verification manual (docs/verification/index.md): its order-of-accuracy
tables and convergence plots, from the convergence tests of MallardTest.

    python tools/docs/verification.py --test build-2/test/MallardTest build-3/test/MallardTest
    python tools/docs/verification.py            # redraw from docs/verification/results.json

The tests check each order with verification::expect_order (test/verification.h),
which also writes the study to $MALLARD_VERIFICATION; this script runs the
convergence tests of the given 2D and 3D builds (double precision), including
the DISABLED_Verification instantiations that are too slow for every CI run,
keeps the results in docs/verification/results.json, rewrites the generated
blocks of the page and draws docs/images/verification/. It exits with status 1
if a study fails its test's bounds, unless KNOWN lists it with its issue.
"""
import argparse
import datetime
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import figstyle

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "verification" / "index.md"
RESULTS = ROOT / "docs" / "verification" / "results.json"
OUT = ROOT / "docs" / "images" / "verification"

# The tests that call verification::expect_order
FILTER = ":".join([
    "TimeIntegratorTest.*", "ChemistryReactorTest.Rodas*",
    "*/TENOOrder.*", "*/TENO3DOrder.*",
    "*PeriodicAdvection2D.*", "*PeriodicAdvection3D.*", "*SodMesh.TENOConverges*", "TENO3DSolver.AdvectedPulse*",
    "*ViscousMesh.StokesFirst*", "*ViscousMesh.ForcedChannel*", "*ForcedChannel3D.*",
    "AxisymmetricTest.Poiseuille*", "*AxisymmetricOrder.*",
    "CurvedGeometry.StraightWalls*", "*CurvedTENOOrder.*", "CurvedTENO.CellsReaching*", "Curved3DGeometry.Quadratic*",
    "RemapTest.SmoothPeriodic*", "*MixtureAdvectionOrder.*",
])

# Studies that fail their bounds for a known, filed reason: (section, method, cells) -> issue number
KNOWN = {}

SECTIONS = ["time", "reconstruction", "advection", "viscous", "boundaries", "axisymmetric", "curved", "remap",
            "mixtures"]
CELLS = ["quadrilaterals", "triangles", "hexahedra", "tetrahedra", "prisms", "pyramids", "mixed"]
SCHEMES = ["FO", "MUSCL", "TENO3", "TENO4", "TENO5", "TENO6"]


# ------------------------------------------------------------------ running

def list_tests(test):
    """The full names of the tests the filter selects."""
    out = subprocess.run([test, f"--gtest_filter={FILTER}", "--gtest_also_run_disabled_tests", "--gtest_list_tests"],
                         capture_output=True, text=True, check=True).stdout
    names, suite = [], ""
    for line in out.splitlines():
        if not line.startswith(" "):
            suite = line.strip()
        elif line.strip():
            names.append(suite + line.split()[0])
    return names


def run_one(test, name, threads):
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    env = dict(os.environ, MALLARD_VERIFICATION=path, OMP_NUM_THREADS=str(threads), OMP_PROC_BIND="false")
    proc = subprocess.run([test, f"--gtest_filter={name}", "--gtest_also_run_disabled_tests", "--gtest_brief=1"],
                          env=env, capture_output=True, text=True)
    lines = Path(path).read_text().splitlines()
    os.unlink(path)
    print(f"{'ok' if proc.returncode == 0 else 'FAILED'} {name}", flush=True)
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout[-3000:] + proc.stderr[-1000:])
    return [json.loads(line) for line in lines if line.strip()]


def run(tests, jobs, threads):
    """One process per test, jobs at a time."""
    tasks = [(t, name) for t in tests for name in list_tests(t)]
    tasks.sort(key=lambda a: (not a[1].startswith("DISABLED_"), -tests.index(a[0])))
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        parts = list(pool.map(lambda a: run_one(*a, threads), tasks))
    studies, seen = [], set()
    for s in (s for part in parts for s in part):
        key = (s["section"], s["method"], s["cells"], s["error"])
        if key not in seen:
            seen.add(key)
            studies.append(s)
    order = {name: i for i, name in enumerate(SECTIONS)}
    cell_order = {name: i for i, name in enumerate([""] + CELLS)}
    studies.sort(key=lambda s: (order.get(s["section"], 99), s["error"], cell_order.get(s["cells"], 99),
                                scheme_key(s["method"])))
    return studies


def scheme_key(method):
    m = re.match(r"(FO|MUSCL|TENO\d)", method)
    return (SCHEMES.index(m.group(1)) if m else 99, method)


def commit():
    try:
        git = ["git", "-C", str(ROOT)]
        sha = subprocess.run(git + ["rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(git + ["status", "--porcelain", "--", "src", "test"], capture_output=True, text=True,
                               check=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# ------------------------------------------------------------------ tables

def fmt_order(x):
    return "n/a" if x is None else f"{x:.2f}"


def bounds(s):
    lo, hi = s["min_order"], s["max_order"]
    if lo is None:
        return f"< {hi:g}"
    if hi is None:
        return f"> {lo:g}"
    return f"{lo:g} to {hi:g}"


def known(s):
    return KNOWN.get((s["section"], s["method"], s["cells"]))


def result(s):
    if s["pass"]:
        return "pass"
    issue = known(s)
    if issue:
        return f"**fail**, [#{issue}](https://github.com/MatthewBonanni/mallard/issues/{issue})"
    return "**FAIL**"


def section_table(studies):
    rows = ["| Method | Cells | Error | Resolutions | Errors | Design | Observed | Asserted | Result |",
            "|---|---|---|---|---|---|---|---|---|"]
    for s in studies:
        n = " → ".join(f"{x:g}" for x in s["n"])
        e = " → ".join(f"{x:.2e}" for x in s["errors"])
        rows.append(f"| {s['method']} | {s['cells'] or '–'} | {s['error']} | {n} | {e} | {s['design']:g} | "
                    f"{fmt_order(s['order'])} | {bounds(s)} | {result(s)} |")
    return "\n".join(rows)


def matrix(studies, section, error_filter=lambda e: True):
    """Observed orders by scheme (rows) and cell type (columns)."""
    cells = [c for c in CELLS if any(s["cells"] == c for s in studies if s["section"] == section)]
    rows = ["| | " + " | ".join(c.capitalize() for c in cells) + " |", "|---" * (len(cells) + 1) + "|"]
    for scheme in SCHEMES:
        line, any_value = [], False
        for c in cells:
            hits = [s for s in studies if s["section"] == section and s["method"] == scheme and s["cells"] == c
                    and error_filter(s["error"])]
            if hits:
                any_value = True
                s = hits[0]
                v = fmt_order(s["order"])
                line.append(v if s["pass"] else f"**{v}** ({result(s)})")
            else:
                line.append("–")
        if any_value:
            design = {"FO": 1, "MUSCL": 2}.get(scheme) or int(scheme[4:])
            rows.append(f"| {scheme} (design {design}) | " + " | ".join(line) + " |")
    return "\n".join(rows)


def blocks(data):
    studies = data["studies"]
    failed = [s for s in studies if not s["pass"]]
    unknown = [s for s in failed if not known(s)]
    summary = f"{len(studies)} convergence studies from commit `{data['commit']}` ({data['date']}): "
    if not failed:
        summary += "all within their tests' bounds."
    else:
        summary += f"{len(studies) - len(failed)} within their tests' bounds"
        if len(failed) > len(unknown):
            issues = sorted({known(s) for s in failed if known(s)})
            summary += f", {len(failed) - len(unknown)} short of them for known reasons (" + ", ".join(
                f"[#{i}](https://github.com/MatthewBonanni/mallard/issues/{i})" for i in issues) + ")"
        if unknown:
            summary += f", **{len(unknown)} failing**"
        summary += "."
    out = {
        "summary": summary,
        "matrix-advection": matrix(studies, "advection", lambda e: e == "max density error"),
        "matrix-reconstruction": matrix(studies, "reconstruction", lambda e: e.startswith("max face-value error")),
    }
    for section in SECTIONS:
        out[f"table-{section}"] = section_table([s for s in studies if s["section"] == section])
    return out


def write_page(data):
    text = PAGE.read_text()
    for name, body in blocks(data).items():
        pattern = re.compile(rf"(<!-- BEGIN GENERATED {name} -->\n).*?(<!-- END GENERATED {name} -->)", re.S)
        if not pattern.search(text):
            raise SystemExit(f"{PAGE}: no generated block {name}")
        text = pattern.sub(lambda m: m.group(1) + body + "\n" + m.group(2), text)
    PAGE.write_text(text)


# ------------------------------------------------------------------ figures

def scheme_colors(P):
    return dict(zip(SCHEMES, [P["muted"], P["blue"], P["teal"], P["gold"], P["red"], P["plum"]]))


def slope_marker(ax, n, err, order, P):
    """A reference slope of the given order through the coarse end of a study."""
    x0, x1 = n[0], n[-1]
    y0 = err[0] * 1.6
    ax.plot([x0, x1], [y0, y0 * (x0 / x1) ** order], ls=(0, (3, 3)), color=P["muted"], lw=1.0)


def convergence_grid(studies, cells, title_of=lambda c: c.capitalize()):
    def draw(P):
        import matplotlib.pyplot as plt
        cols = min(len(cells), 4)
        rows = math.ceil(len(cells) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.8 * rows), squeeze=False)
        colors = scheme_colors(P)
        for ax, c in zip(axes.flat, cells):
            for s in [s for s in studies if s["cells"] == c]:
                col = colors.get(re.match(r"\w+", s["method"]).group(0), P["fg"])
                slope_marker(ax, s["n"], s["errors"], s["design"], P)
                ax.loglog(s["n"], s["errors"], "o-", color=col, ms=4, label=s["method"])
            ax.set_title(title_of(c), fontsize=12)
            ax.set_xlabel("cells per side", fontsize=11)
            ax.minorticks_off()
            ax.set_xticks(sorted({x for s in studies if s["cells"] == c for x in s["n"]}))
            ax.set_xticklabels([f"{x:g}" for x in ax.get_xticks()])
        for ax in list(axes.flat)[len(cells):]:
            ax.axis("off")
        for ax in axes[:, 0]:
            ax.set_ylabel("error", fontsize=11)
        handles = {}
        for ax in axes.flat:
            for h, label in zip(*ax.get_legend_handles_labels()):
                handles.setdefault(label, h)
        labels = sorted(handles, key=scheme_key)
        fig.legend([handles[k] for k in labels], labels, loc="lower center", ncol=len(labels),
                   bbox_to_anchor=(0.5, -0.02 - 0.03 / rows))
        fig.tight_layout(rect=(0, 0.06 / rows, 1, 1))
        return fig
    return draw


def series_plot(studies, xlabel, label_of=lambda s: s["method"]):
    def draw(P):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        palette = [P["blue"], P["teal"], P["flame"], P["plum"], P["green"], P["gold"], P["red"], P["muted"]]
        for i, s in enumerate(studies):
            slope_marker(ax, s["n"], s["errors"], s["design"], P)
            ax.loglog(s["n"], s["errors"], "o-", color=palette[i % len(palette)], ms=4,
                      label=f"{label_of(s)} ({fmt_order(s['order'])})")
        ax.minorticks_off()
        ticks = sorted({x for s in studies for x in s["n"]})
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{x:g}" for x in ticks])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("error")
        ax.legend(fontsize=9.5, loc="center left", bbox_to_anchor=(1.02, 0.5))
        return fig
    return draw


def figures(data):
    studies = data["studies"]
    by = lambda section, cond=lambda s: True: [s for s in studies if s["section"] == section and cond(s)]
    adv = by("advection", lambda s: s["error"] == "max density error")
    rec = by("reconstruction", lambda s: s["error"].startswith("max face-value error") and "troubled" not in s["method"])
    out = {
        "advection": convergence_grid(adv, [c for c in CELLS if any(s["cells"] == c for s in adv)]),
        "reconstruction": convergence_grid(rec, [c for c in CELLS if any(s["cells"] == c for s in rec)]),
        "time": series_plot(by("time", lambda s: s["error"] == "max error, linear system"), "steps"),
        "viscous": series_plot(by("viscous"), "cells across the channel", lambda s: f"{s['method']}, {s['cells']}"),
        "curved": series_plot(by("curved", lambda s: "TENO" not in s["method"]), "cells around",
                              lambda s: f"{s['method']}, {s['cells']}"),
        "remap": series_plot(by("remap"), "source cells per side", lambda s: f"{s['method']}, {s['cells']}"),
    }
    for name, draw in out.items():
        figstyle.save(name, draw, OUT)


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--test", nargs="+", help="MallardTest executables (a 2D and a 3D double-precision build)")
    ap.add_argument("--jobs", type=int, default=4, help="test processes at a time")
    ap.add_argument("--threads", type=int, default=1, help="OpenMP threads per test process")
    args = ap.parse_args()
    if args.test:
        data = {"commit": commit(), "date": datetime.date.today().isoformat(),
                "studies": run(args.test, args.jobs, args.threads)}
        RESULTS.write_text(json.dumps(data, indent=1) + "\n")
    else:
        data = json.loads(RESULTS.read_text())
    write_page(data)
    figures(data)
    failed = [s for s in data["studies"] if not s["pass"]]
    for s in failed:
        print(f"{'known' if known(s) else 'FAIL'} {s['section']}: {s['method']} {s['cells']} {s['error']}: "
              f"order {fmt_order(s['order'])}, asserted {bounds(s)} ({s['test']})")
    print(f"{len(data['studies'])} studies, {len(failed)} failed ({sum(1 for s in failed if known(s))} known)")
    sys.exit(1 if any(not known(s) for s in failed) else 0)


if __name__ == "__main__":
    main()
