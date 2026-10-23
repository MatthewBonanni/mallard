"""Draw every figure of the user guide into docs/images/guide/ and of the theory
manual and TENO-E reference into docs/images/theory/, light and dark.

    python tools/docs/make_figures.py [--mallard build/src/mallard] [--mallard3d build-3d/src/mallard]

Schematics and domain diagrams need numpy, scipy and matplotlib only; with
--mallard (a 2D build; an MPI build for the partitions) the figures from
short runs are redrawn too, and with --mallard3d the 3D region figure. MALLARD_FIGURE_PREVIEW=dir also writes PNGs on
the site's backgrounds there, for review.
"""
import argparse
import sys

import domains
import inlets
import regions
import runs
import schematics
import teno
import theory


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mallard", help="2D Mallard executable, for the figures from runs")
    ap.add_argument("--mallard3d", help="3D Mallard executable, for the 3D region figure")
    ap.add_argument("--mpirun", default="mpirun")
    args = ap.parse_args()
    schematics.figures()
    domains.figures()
    inlets.figures()
    theory.figures()
    teno.figures()
    if args.mallard:
        sys.argv = [sys.argv[0], "--mallard", args.mallard, "--mpirun", args.mpirun]
        runs.main()
        regions.figures(args.mallard, args.mallard3d)


if __name__ == "__main__":
    main()
