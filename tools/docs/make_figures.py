"""Draw every figure of the user guide into docs/images/guide/ and of the theory
manual into docs/images/theory/, light and dark.

    python tools/docs/make_figures.py [--mallard build/src/Mallard]

Schematics and domain diagrams need numpy, scipy and matplotlib only; with
--mallard (a 2D build; an MPI build for the partitions) the figures from
short runs are redrawn too. MALLARD_FIGURE_PREVIEW=dir also writes PNGs on
the site's backgrounds there, for review.
"""
import argparse
import sys

import domains
import probes
import runs
import schematics
import theory


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mallard", help="2D Mallard executable, for the figures from runs")
    ap.add_argument("--mallard3d", help="3D Mallard executable, for the 3D probe figures")
    ap.add_argument("--mpirun", default="mpirun")
    args = ap.parse_args()
    schematics.figures()
    domains.figures()
    theory.figures()
    if args.mallard:
        sys.argv = [sys.argv[0], "--mallard", args.mallard, "--mpirun", args.mpirun]
        runs.main()
        sys.argv = [sys.argv[0], "--mallard", args.mallard] + (["--mallard3d", args.mallard3d] if args.mallard3d else [])
        probes.main()


if __name__ == "__main__":
    main()
