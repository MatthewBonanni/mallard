"""The figure of the Python package's guide page (docs/guide/python.md), drawn with the package itself from
the Sod example: the density of every snapshot along the tube, and its history at one cell.

    python tools/docs/python.py --mallard build/src/mallard

The build needs -DMallard_ENABLE_PYTHON=ON and HDF5; the package finds its library next to --mallard.
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import figstyle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import mallard  # noqa: E402


def sod(mallard_exe):
    work = Path(tempfile.mkdtemp())
    subprocess.run([mallard_exe, "new", "sod", str(work / "sod")], check=True, stdout=subprocess.DEVNULL)
    subprocess.run([mallard_exe, "-i", "input.toml", "--kokkos-num-threads=2"], cwd=work / "sod", check=True,
                   stdout=subprocess.DEVNULL)
    f = mallard.open(work / "sod" / "solut" / "sod.vtkhdf.series")
    centers = f.geometry().vertex_averages()
    row = np.flatnonzero(centers[:, 1] < centers[:, 1].min() + 1e-9)  # the bottom row of cells
    row = row[np.argsort(centers[row, 0])]
    rho = f.read_all("RHO")
    at = row[np.argmin(np.abs(centers[row, 0] - 0.8))]
    return centers[row, 0], f.times, rho[:, row], centers[at, 0], rho[:, at]

def draw(P, data):
    x, times, rho, x_at, history = data
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), gridspec_kw=dict(width_ratios=[1.5, 1], wspace=0.6))
    cmap = plt.get_cmap(P["cmap"])
    for k in range(len(times)):
        axes[0].plot(x, rho[k], color=cmap(k / max(len(times) - 1, 1)), lw=1.4)
    axes[0].axvline(x_at, color=P["muted"], lw=0.8, ls="--")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("RHO")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(times[0], times[-1]))
    fig.colorbar(sm, ax=axes[0], label="t", fraction=0.05, pad=0.02)
    axes[1].plot(times, history, "o-", color=P["teal"], ms=3.5)
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("RHO")
    axes[1].set_title(f"at x = {x_at:.3f}", fontsize=12, color=P["muted"])
    return fig

def figures(mallard_exe):
    """Draws the figure; a build without the library is skipped, with a message."""
    os.environ["MALLARD_EXECUTABLE"] = str(Path(mallard_exe).resolve())
    try:
        mallard._lib.libraries()
    except RuntimeError as e:
        print(f"python-sod skipped: {e}")
        return
    data = sod(os.environ["MALLARD_EXECUTABLE"])
    figstyle.save("python-sod", lambda P: draw(P, data))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mallard", required=True, help="Mallard executable of a build with -DMallard_ENABLE_PYTHON=ON")
    figures(ap.parse_args().mallard)

if __name__ == "__main__":
    main()
