#!/usr/bin/env python3
"""Repeat the flow of a restart file of a generated cartesian mesh along x.

    tile_restart.py IN.restart Nx_in COPIES OUT.restart

The generated 3D cartesian mesh numbers its cells with z fastest and x
slowest, so a mesh of COPIES * Nx_in cells along x (same Ny, Nz and spacing)
holds the old flow repeated COPIES times. Starts a spatially developing run
(e.g. a channel with synthetic inflow) from a developed periodic one. Keeps
the conservative variables (and species) only, at step 0 and time 0.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mallard import restart  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("restart")
    ap.add_argument("nx", type=int, help="cells along x of the input mesh")
    ap.add_argument("copies", type=int)
    ap.add_argument("out")
    args = ap.parse_args()
    r = restart.read(args.restart)
    keep = [n for n in r.names if n in ("RHO", "RHOU_X", "RHOU_Y", "RHOU_Z", "RHOE") or n.startswith("RHOY_")]
    if r.n_cells % args.nx:
        raise SystemExit(f"{r.n_cells} cells are not a multiple of Nx = {args.nx}")
    restart.write(args.out, {n: np.tile(r[n], args.copies) for n in keep}, real_size=r.real_size)
    print(f"{args.out}: {r.n_cells * args.copies} cells ({args.copies} x {args.nx} along x)")


if __name__ == "__main__":
    main()
