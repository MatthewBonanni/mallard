"""Temperature and heat release profiles of a premixed-flame run against Cantera's (V8).

    python tools/plot_flame.py RUN_DIR REFERENCE.csv [PNG]

REFERENCE.csv is a profile from tools/flame_reference.py
(examples/premixed_flame/reference/<case>.csv). The last output of the run is
shifted so that its maximum of dT/dx sits at Cantera's; reports the largest
temperature difference over the reference window and the peak heat release
rates, and with PNG plots both profiles.
"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flame_speed import profile  # noqa: E402


def last_profile(run_dir):
    pvd = os.path.join(run_dir, "solut", "flame.pvd")
    last = re.findall(r'file="([^"]+)"', open(pvd).read())[-1]
    t, x, a = profile(os.path.join(os.path.dirname(pvd), last))
    T = a["T"]
    # Maximum of dT/dx, refined by a parabola through its neighbors
    g = np.gradient(T, x)
    i = int(np.argmax(g))
    den = g[i - 1] - 2 * g[i] + g[i + 1]
    x_f = x[i] + (0.5 * (g[i - 1] - g[i + 1]) / den * (x[1] - x[0]) if den != 0 else 0.0)
    return t, x - x_f, a


def compare(run_dir, ref_file):
    """Largest |T - T_Cantera| [K] and peak heat release error [%] of the run's last output."""
    ref = np.genfromtxt(ref_file, delimiter=",", names=True, skip_header=1)
    _, xs, a = last_profile(run_dir)
    dT = np.abs(np.interp(ref["x"], xs, a["T"]) - ref["T"]).max()
    return dT, 100 * (a["HRR"].max() / ref["hrr"].max() - 1)


def main():
    run_dir, ref_file = sys.argv[1], sys.argv[2]
    png = sys.argv[3] if len(sys.argv) > 3 else None
    ref = np.genfromtxt(ref_file, delimiter=",", names=True, skip_header=1)
    t, xs, a = last_profile(run_dir)
    T = a["T"]
    dT, hrr = compare(run_dir, ref_file)
    print(f"t = {t * 1e3:.4f} ms: max |T - T_Cantera| = {dT:.1f} K over [{ref['x'][0] * 1e3:.2f}, "
          f"{ref['x'][-1] * 1e3:.2f}] mm; peak HRR {a['HRR'].max():.4e} vs {ref['hrr'].max():.4e} W/m^3 "
          f"({hrr:+.1f}%)")
    if png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ref["x"] * 1e3, ref["T"], "k-", label="Cantera T")
        ax.plot(xs * 1e3, T, "o", ms=3, mfc="none", label="Mallard T")
        ax.set_xlim(ref["x"][0] * 1e3, ref["x"][-1] * 1e3)
        ax.set_xlabel("x - x_flame [mm]")
        ax.set_ylabel("T [K]")
        ax2 = ax.twinx()
        ax2.plot(ref["x"] * 1e3, ref["hrr"], "r-", label="Cantera HRR")
        ax2.plot(xs * 1e3, a["HRR"], "s", color="r", ms=3, mfc="none", label="Mallard HRR")
        ax2.set_ylabel("heat release rate [W/m^3]")
        fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88), fontsize=8)
        fig.tight_layout()
        fig.savefig(png, dpi=150)


if __name__ == "__main__":
    main()
