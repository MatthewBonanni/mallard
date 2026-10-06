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
import struct

import numpy as np


def read_restart(path):
    with open(path, "rb") as f:
        magic = f.read(16)
        version, real_size = struct.unpack("<II", f.read(8))
        n_cells, n_vars, _ = struct.unpack("<QQQ", f.read(24))
        f.read(8)
        if not magic.startswith(b"MALLARD-RESTART") or version not in (2, 3):
            raise SystemExit(f"{path}: expected a restart file of version 2 or 3")

        def name():
            (length,) = struct.unpack("<I", f.read(4))
            return f.read(length).decode()

        names = [name() for _ in range(n_vars)]
        if version == 3:
            (n_attributes,) = struct.unpack("<Q", f.read(8))
            for _ in range(n_attributes):
                name()
                f.read(8)
        dtype = "<f8" if real_size == 8 else "<f4"
        fields = np.fromfile(f, dtype=dtype, count=n_vars * n_cells).reshape(n_vars, n_cells)
    return dtype, real_size, {n: fields[v] for v, n in enumerate(names)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("restart")
    ap.add_argument("nx", type=int, help="cells along x of the input mesh")
    ap.add_argument("copies", type=int)
    ap.add_argument("out")
    args = ap.parse_args()
    dtype, real_size, fields = read_restart(args.restart)
    keep = [n for n in fields if n in ("RHO", "RHOU_X", "RHOU_Y", "RHOU_Z", "RHOE") or n.startswith("RHOY_")]
    n_cells = fields[keep[0]].size
    if n_cells % args.nx:
        raise SystemExit(f"{n_cells} cells are not a multiple of Nx = {args.nx}")
    with open(args.out, "wb") as f:
        f.write(b"MALLARD-RESTART\0")
        f.write(struct.pack("<II", 2, real_size))
        f.write(struct.pack("<QQQ", n_cells * args.copies, len(keep), 0))
        f.write(struct.pack("<d", 0.0))
        for name in keep:
            f.write(struct.pack("<I", len(name)) + name.encode())
        for name in keep:
            np.tile(fields[name], args.copies).astype(dtype).tofile(f)
    print(f"{args.out}: {n_cells * args.copies} cells ({args.copies} x {args.nx} along x)")


if __name__ == "__main__":
    main()
