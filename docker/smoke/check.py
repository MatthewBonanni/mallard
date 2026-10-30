#!/usr/bin/env python3
"""Checks the smoke-test runs against reference solutions.

    check.py TOOLS sod STEM [STEM...]           L1 density error against the exact Sod solution at t = 0.2
    check.py TOOLS reactor LOG                   `mallard reactor`'s ignition delay against Cantera's

TOOLS is the directory of Mallard's Python tools (share/mallard/tools). A STEM
PREFIX_NNNNNN names a 2D or 3D snapshot: its .vtkhdf file, or without HDF5 its
.vtu file (the .pvtu index of a distributed run).
"""
import os
import re
import sys

import numpy as np

SOD_L1_MAX = 6.0e-3  # TENO5: 2.6e-3 on 200 cells (2D), 4.1e-3 on 100 (3D)
IGNITION_DELAY_REF = 44.2e-6  # Cantera IdealGasReactor (examples/h2_ignition)
IGNITION_DELAY_RTOL = 0.02


def cells(stem):
    """Snapshot file, cell-center x and density of a snapshot."""
    from mallard.vtu import read_vtu_cells, snapshot
    path = snapshot(stem)
    pts, conn, offs, _, arrays = read_vtu_cells(path)
    starts = np.concatenate([[0], offs[:-1]])
    x = np.array([pts[conn[a:b], 0].mean() for a, b in zip(starts, offs)])
    return path, x, arrays["RHO"]


def check_sod(paths):
    from plot_sod import exact_sod
    ok = True
    for stem in paths:
        path, x, rho = cells(stem)
        err = np.mean(np.abs(rho - exact_sod(x, 0.2)))
        ok &= bool(err < SOD_L1_MAX)
        print(f"{path}: {x.size} cells, L1 density error {err:.2e} (limit {SOD_L1_MAX:.0e})")
    return ok


def check_reactor(path):
    m = re.search(r"Ignition delay\s+(\S+) s", open(path).read())
    delay = float(m.group(1))
    rel = abs(delay / IGNITION_DELAY_REF - 1)
    print(f"{path}: ignition delay {delay * 1e6:.1f} us, Cantera {IGNITION_DELAY_REF * 1e6:.1f} us "
          f"(difference {rel:.1%}, limit {IGNITION_DELAY_RTOL:.0%})")
    return rel < IGNITION_DELAY_RTOL


def main():
    sys.path[:0] = [sys.argv[1], os.path.join(sys.argv[1], "..", "examples", "sod")]
    kind, files = sys.argv[2], sys.argv[3:]
    ok = check_sod(files) if kind == "sod" else check_reactor(files[0])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
