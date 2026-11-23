"""ParaView's side of the read benchmark (benchmarks/README.md): what read_mallard.py does, with the reader
OpenDataFile picks. Only the arrays used are enabled, and they are taken from the reader's own output
(GetClientSideObject), without the copy of servermanager.Fetch.

Usage: pvbatch read_pvbatch.py PATH TASK [NPY]
"""
import sys
import time

import numpy as np
from paraview import simple
from vtkmodules.numpy_interface import dataset_adapter as dsa

path, task = sys.argv[1], sys.argv[2]
t0 = time.perf_counter()
r = simple.OpenDataFile(path)
rho, check = np.zeros(0), 0.0
if task != "open":
    for prop in ("CellArrayStatus", "CellArrays"):
        if r.GetProperty(prop) is not None:
            r.SetPropertyWithName(prop, ["RHO", "U"] if task == "one" else ["RHO"])
            break
    tv = r.TimestepValues
    times = [float(t) for t in tv] if hasattr(tv, "__len__") else [float(tv or 0.0)]


def output(t):
    r.UpdatePipeline(time=t)
    return dsa.WrapDataObject(r.GetClientSideObject().GetOutputDataObject(0))


def array(a):
    if isinstance(a, dsa.VTKCompositeDataArray):
        return np.concatenate([np.asarray(x) for x in a.Arrays])
    return np.asarray(a)


if task == "one":
    d = output(times[-1])
    rho, u = array(d.CellData["RHO"]), array(d.CellData["U"])
    check = rho.sum() + u.sum() + array(d.Points).sum()
elif task == "series":
    for t in times:
        rho = array(output(t).CellData["RHO"])
        check += rho.sum()
print(f"READ {time.perf_counter() - t0:.4f} {check:.10e}")
if len(sys.argv) > 3:
    np.save(sys.argv[3], rho)
