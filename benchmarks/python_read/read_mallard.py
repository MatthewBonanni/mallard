"""The Python package's side of the read benchmark (benchmarks/README.md).

Usage: python read_mallard.py PATH TASK [NPY]

TASK `open` only opens PATH; `one` reads the geometry, RHO and U of the last snapshot; `series` reads RHO at every
snapshot, one at a time. Prints `READ <seconds from open> <checksum>` and saves the last RHO read to NPY.
"""
import sys
import time

import numpy as np

import mallard

path, task = sys.argv[1], sys.argv[2]
t0 = time.perf_counter()
f = mallard.open(path)
rho, check = np.zeros(0), 0.0
if task == "one":
    g = f.geometry()
    rho, u = f.read("RHO"), f.read("U")
    check = rho.sum() + u.sum() + g.points.sum()
elif task == "series":
    for k in range(len(f.snapshots)):
        rho = f.read("RHO", k)
        check += rho.sum()
print(f"READ {time.perf_counter() - t0:.4f} {check:.10e}")
if len(sys.argv) > 3:
    np.save(sys.argv[3], rho)
