"""Draw every figure of the user guide into docs/images/guide/ and of the theory
manual and TENO-E reference into docs/images/theory/, light and dark.

    python tools/docs/make_figures.py [--mallard build/src/mallard] [--mallard3d build-3d/src/mallard]

Schematics and domain diagrams need numpy, scipy and matplotlib only.
With --mallard (a 2D build; an MPI build for the partitions) the figures from
short runs and the sample run report are redrawn too (the Python package's with a
build with -DMallard_ENABLE_PYTHON=ON).
With --mallard3d the 3D region, probe and mesh-field figures are drawn too.
MALLARD_FIGURE_PREVIEW=dir also writes PNGs on the site's backgrounds there, for
review.

Each module's figures are drawn by one call below, in alphabetical order.
"""
import argparse
import sys

import domains
import effective_resolution
import geometry
import high_order_cost
import inlets
import les_models
import limiters
import mesh_fields
import probes
import python
import regions
import run_report
import runs
import schematics
import teno
import theory
import time_integrators


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mallard", help="2D Mallard executable, for the figures from runs")
    ap.add_argument("--mallard3d", help="3D Mallard executable, for the 3D region, probe and mesh-field figures")
    ap.add_argument("--mpirun", default="mpirun")
    args = ap.parse_args()
    domains.figures()
    effective_resolution.figures()
    geometry.figures(args.mallard, args.mallard3d)
    high_order_cost.figures()
    inlets.figures()
    les_models.figures()
    limiters.figures()
    schematics.figures()
    teno.figures()
    theory.figures()
    time_integrators.figures()
    if args.mallard:
        mesh_fields.figures(args.mallard, args.mallard3d)
        sys.argv = [sys.argv[0], "--mallard", args.mallard] + (["--mallard3d", args.mallard3d] if args.mallard3d else [])
        probes.main()
        python.figures(args.mallard)
        regions.figures(args.mallard, args.mallard3d)
        sys.argv = [sys.argv[0], "--mallard", args.mallard]
        run_report.main()
        sys.argv = [sys.argv[0], "--mallard", args.mallard, "--mpirun", args.mpirun]
        runs.main()


if __name__ == "__main__":
    main()
