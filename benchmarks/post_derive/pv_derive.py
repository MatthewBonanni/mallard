"""pvbatch equivalent of `mallard post derive FILE... --fields MACH,Q_CRITERION,LAMBDA2,SCHLIEREN,VORTICITY`
for a perfect gas of the given gamma (benchmarks/README.md).

Usage: pvbatch pv_derive.py GAMMA OUTDIR FILE.vtkhdf...
"""
import os
import sys

from paraview.simple import Calculator, Gradient, PassArrays, ProgrammableFilter, SaveData, VTKHDFReader, Delete

gamma = float(sys.argv[1])
outdir = sys.argv[2]
files = sys.argv[3:]
os.makedirs(outdir, exist_ok=True)

LAMBDA2 = """
import numpy as np
from vtk.numpy_interface import dataset_adapter as dsa
inp = dsa.WrapDataObject(self.GetInputDataObject(0, 0))
out = dsa.WrapDataObject(self.GetOutputDataObject(0))
g = np.asarray(inp.CellData['GRAD_U']).reshape(-1, 3, 3)
s = 0.5 * (g + g.transpose(0, 2, 1))
w = 0.5 * (g - g.transpose(0, 2, 1))
out.CellData.append(np.linalg.eigvalsh(s @ s + w @ w)[:, 1], 'LAMBDA2')
"""

for f in files:
    reader = VTKHDFReader(FileName=[f])
    reader.CellArrayStatus = ['RHO', 'U', 'P']
    gu = Gradient(Input=reader, ScalarArray=['CELLS', 'U'], ResultArrayName='GRAD_U',
                  ComputeQCriterion=1, QCriterionArrayName='Q_CRITERION',
                  ComputeVorticity=1, VorticityArrayName='VORTICITY')
    gr = Gradient(Input=gu, ScalarArray=['CELLS', 'RHO'], ResultArrayName='GRAD_RHO')
    mach = Calculator(Input=gr, AttributeType='Cell Data', ResultArrayName='MACH',
                      Function=f'mag(U)/sqrt({gamma}*P/RHO)')
    schl = Calculator(Input=mach, AttributeType='Cell Data', ResultArrayName='SCHLIEREN',
                      Function='mag(GRAD_RHO)')
    l2 = ProgrammableFilter(Input=schl, Script=LAMBDA2, CopyArrays=1)
    keep = PassArrays(Input=l2, CellDataArrays=['MACH', 'Q_CRITERION', 'LAMBDA2', 'SCHLIEREN',
                                                'VORTICITY'], PointDataArrays=[])
    SaveData(os.path.join(outdir, os.path.basename(f)), proxy=keep)
    for p in (keep, l2, schl, mach, gr, gu, reader):
        Delete(p)
