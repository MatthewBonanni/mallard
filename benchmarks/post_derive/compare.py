"""pvpython compare.py MALLARD.vtkhdf PARAVIEW.vtkhdf [MARGIN]: relative L2 and Linf differences per field."""
import sys

import numpy as np
from paraview import servermanager as sm
from paraview.simple import CellCenters, VTKHDFReader
from vtk.numpy_interface import dataset_adapter as dsa


def cells(path):
    r = VTKHDFReader(FileName=[path])
    r.UpdatePipeline()
    d = dsa.WrapDataObject(sm.Fetch(r))
    c = dsa.WrapDataObject(sm.Fetch(CellCenters(Input=r))).Points
    return {n: np.asarray(d.CellData[n]) for n in d.CellData.keys()}, np.asarray(c)


(a, xc), (b, _) = cells(sys.argv[1]), cells(sys.argv[2])
margin = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
lo, hi = xc.min(axis=0), xc.max(axis=0)
inner = np.all((xc >= lo + margin - 1e-12) & (xc <= hi - margin + 1e-12), axis=1)
print(f'{inner.sum()} of {len(inner)} cells, margin {margin}')
for name in ['MACH', 'Q_CRITERION', 'LAMBDA2', 'SCHLIEREN', 'VORTICITY']:
    x, y = a[name][inner], b[name][inner]
    if x.ndim == 1 and y.ndim == 2:
        y = y[:, 2]
    rel_l2 = np.linalg.norm(y - x) / np.linalg.norm(x)
    rel_inf = np.abs(y - x).max() / np.abs(x).max()
    print(f'{name:12s} rel L2 {rel_l2:.3e}  rel Linf {rel_inf:.3e}')
