"""Reactive shock tube (examples/reactive_shock_tube) at several resolutions.

    python tools/plot_reactive_shock_tube.py RUN.pvd [RUN.pvd ...] [--png OUT.png]

For the outputs at 170 and 230 us of each run: the reaction front (the
rightmost cell above 1800 K), the peak temperature and pressure, and the
cell size; with --png, T and p at both times overlaid.
"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mallard_vtu import read_vtu_cells  # noqa: E402


def profile(path):
    pts, conn, offs, _, arrays = read_vtu_cells(path)
    starts = np.concatenate([[0], offs[:-1]])
    x = np.array([pts[conn[s:e], 0].mean() for s, e in zip(starts, offs)])
    order = np.argsort(x)
    return arrays["TIME"], x[order], arrays["P"][order], arrays["T"][order]


def at_times(pvd, times):
    base = os.path.dirname(pvd)
    entries = re.findall(r'timestep="([^"]+)" file="([^"]+)"', open(pvd).read())
    out = {}
    for t_want in times:
        t, f = min(entries, key=lambda e: abs(float(e[0]) - t_want))
        out[t_want] = profile(os.path.join(base, f))
    return out


def main():
    args = sys.argv[1:]
    png = None
    if "--png" in args:
        png = args[args.index("--png") + 1]
        args = args[:args.index("--png")]
    times = (170e-6, 230e-6)
    runs = {pvd: at_times(pvd, times) for pvd in args}
    print(f"{'run':>40} {'t [us]':>7} {'dx [um]':>8} {'front [mm]':>10} {'T_max [K]':>9} {'p_max [kPa]':>11}")
    for pvd, profiles in runs.items():
        for t_want, (t, x, p, T) in profiles.items():
            front = x[np.nonzero(T > 1800.0)[0][-1]] if (T > 1800.0).any() else float("nan")
            print(f"{pvd[-40:]:>40} {t * 1e6:7.1f} {(x[1] - x[0]) * 1e6:8.1f} {front * 1e3:10.3f} "
                  f"{T.max():9.1f} {p.max() / 1e3:11.1f}")
    if png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
        for pvd, profiles in runs.items():
            label = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(pvd))))
            for j, (t_want, (t, x, p, T)) in enumerate(profiles.items()):
                ax[0, j].plot(x * 100, T, lw=1, label=label)
                ax[1, j].plot(x * 100, p / 1e3, lw=1, label=label)
                ax[0, j].set_title(f"t = {t * 1e6:.0f} us")
        ax[0, 0].set_ylabel("T [K]")
        ax[1, 0].set_ylabel("p [kPa]")
        for a in ax[1]:
            a.set_xlabel("x [cm]")
        ax[0, 0].legend()
        fig.tight_layout()
        fig.savefig(png, dpi=120)


if __name__ == "__main__":
    main()
