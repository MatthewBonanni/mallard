"""Convergence of the viscous fluxes with viscous_gradients = "least_squares"
(the default) and "reconstruction" (the TENO polynomials, docs/guide/numerics.md):
error against mesh size on every cell type, in 2D and 3D, against the design order.

    python tools/docs/viscous_gradients.py --test build-2/test/mallard-test2d build-3/test/mallard-test3d [--jobs 8]
    python tools/docs/viscous_gradients.py            # redraw from viscous_gradients.json

The studies are the DISABLED_Verification/ViscousGradientsStudy tests of
test/viscous_order_test.cpp (double precision builds), which write them through
verification::expect_order to $MALLARD_VERIFICATION:

- the viscous residual of a smooth manufactured state (u, T and rho varying) on
  periodic boxes, at its exact cell averages, against the exact cell average of
  div F_v: design order `order - 2` (face gradients of order `order - 1`, divided
  by h) with "reconstruction", 1 with least squares;
- the momentum of a decaying shear wave (an exact Navier-Stokes solution) at
  t = 0.025: design order `order - 1`, the face gradients', and 2 with least
  squares.

The reconstruction studies assert their design order (less 0.3); the least
squares ones are recorded as measured. The results go to viscous_gradients.json,
which make_figures.py draws without running anything.
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import figstyle
import verification

DATA = Path(__file__).resolve().parent / "viscous_gradients.json"
FILTER = "DISABLED_Verification/ViscousGradientsStudy.*"
CELLS = ["quadrilaterals", "triangles", "hexahedra", "prisms", "tetrahedra"]
METHODS = ["TENO5, least squares", "TENO4, reconstruction", "TENO5, reconstruction", "TENO6, reconstruction"]
FIGURES = {
    "viscous-gradients-residual": "mean viscous residual error, manufactured solution",
    "viscous-gradients-shear-wave": "mean momentum error, decaying shear wave",
}


def list_tests(test):
    out = subprocess.run([test, f"--gtest_filter={FILTER}", "--gtest_also_run_disabled_tests", "--gtest_list_tests"],
                         capture_output=True, text=True, check=True).stdout
    names, suite = [], ""
    for line in out.splitlines():
        if not line.startswith(" "):
            suite = line.strip()
        elif line.strip():
            names.append(suite + line.split()[0])
    return names


def run(tests, jobs, threads):
    tasks = [(t, name) for t in tests for name in list_tests(t)]
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        parts = list(pool.map(lambda a: verification.run_one(*a, threads), tasks))
    studies = [s for part in parts for s in part]
    studies.sort(key=lambda s: (s["error"], CELLS.index(s["cells"]), METHODS.index(s["method"])))
    return studies


def colors(P):
    return dict(zip(METHODS, [P["muted"], P["blue"], P["teal"], P["plum"]]))


def grid(studies):
    cells = [c for c in CELLS if any(s["cells"] == c for s in studies)]

    def draw(P):
        import matplotlib.pyplot as plt
        cols = min(len(cells), 3)
        rows = -(-len(cells) // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.3 * rows), squeeze=False)
        col = colors(P)
        for ax, c in zip(axes.flat, cells):
            for s in [s for s in studies if s["cells"] == c]:
                h = [1.0 / n for n in s["n"]]
                ls = "--" if "least squares" in s["method"] else "-"
                ax.loglog(h, s["errors"], "o", ls=ls, color=col[s["method"]], ms=4,
                          label=f"{verification.fmt_order(s['order'])} ({s['design']:g})")
                x0, x1 = h[0], h[-1]
                y0 = s["errors"][0] * 1.6
                ax.plot([x0, x1], [y0, y0 * (x1 / x0) ** s["design"]], ls=(0, (2, 3)), color=col[s["method"]],
                        lw=0.9, alpha=0.7)
            dim = "3D" if c in ("hexahedra", "prisms", "tetrahedra") else "2D"
            ax.set_title(f"{c.capitalize()} ({dim})", fontsize=12)
            ax.set_xlabel("h", fontsize=11)
            ax.minorticks_off()
            ticks = sorted({n for s in studies if s["cells"] == c for n in s["n"]})
            ax.set_xticks([1.0 / n for n in ticks])
            ax.set_xticklabels([f"1/{n:g}" for n in ticks])
            ax.legend(fontsize=9, loc="best", title="order (design)", title_fontsize=9, frameon=True,
                      facecolor=P["bg"], edgecolor="none", framealpha=0.85)
        for ax in list(axes.flat)[len(cells):]:
            ax.axis("off")
        for ax in axes[:, 0]:
            ax.set_ylabel("error", fontsize=11)
        handles = [plt.Line2D([], [], color=col[m], ls="--" if "least" in m else "-", marker="o", ms=4)
                   for m in METHODS]
        fig.legend(handles, METHODS, loc="lower center", ncol=len(METHODS), bbox_to_anchor=(0.5, -0.02 / rows))
        fig.tight_layout(rect=(0, 0.07 / rows, 1, 1))
        return fig
    return draw


def table(studies):
    """The measured orders, as the guide's table."""
    lines = []
    for error in FIGURES.values():
        lines.append(f"\n{error}: finest-pair order (design)\n")
        lines.append("| Cells | " + " | ".join(METHODS) + " |")
        lines.append("|---" * (len(METHODS) + 1) + "|")
        for c in CELLS:
            row = {s["method"]: s for s in studies if s["cells"] == c and s["error"] == error}
            if row:
                lines.append(f"| {c} | " + " | ".join(
                    f"{verification.fmt_order(row[m]['order'])} ({row[m]['design']:g})" if m in row else "-"
                    for m in METHODS) + " |")
    return "\n".join(lines)


def figures():
    studies = json.loads(DATA.read_text())["studies"]
    for name, error in FIGURES.items():
        figstyle.save(name, grid([s for s in studies if s["error"] == error]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--test", nargs="*", default=[], help="mallard-test2d and mallard-test3d (double precision)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    if args.test:
        studies = run(args.test, args.jobs, args.threads)
        DATA.write_text(json.dumps({"commit": verification.commit(), "studies": studies}, indent=1) + "\n")
    studies = json.loads(DATA.read_text())["studies"]
    figures()
    print(table(studies))
    short = [s for s in studies if not s["pass"]]
    for s in short:
        print(f"below design: {s['method']}, {s['cells']}, {s['error']}: {s['order']:.2f} (design {s['design']:g})")
    return 1 if any("reconstruction" in s["method"] for s in short) else 0


if __name__ == "__main__":
    sys.exit(main())
