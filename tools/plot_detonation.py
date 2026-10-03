"""Front speed and structure of a 1D detonation run (examples/detonation_1d).

    python tools/plot_detonation.py SERIES.pvd [D_CJ L_IND [PNG]]

For each output: the shock position (rightmost cell with p above twice the
initial minimum), the local front speed, the induction length (from the
shock to the maximum heat release rate behind it) and the peak pressure. With D_CJ
and the ZND induction length L_IND (tools/detonation_reference.py) it
reports the mean speed over the second half of the run and its error.
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
    return arrays["TIME"], x[order], arrays["P"][order], arrays["T"][order], arrays["HRR"][order]


def front(x, p, hrr, p0):
    i_s = np.nonzero(p > 2.0 * p0)[0]
    if i_s.size == 0:
        return None
    i_s = i_s[-1]
    window = (x > x[i_s] - 0.05) & (x <= x[i_s])
    i_max = np.nonzero(window)[0][np.argmax(hrr[window])]
    return x[i_s], x[i_s] - x[i_max], p[window].max()


def main():
    pvd = sys.argv[1]
    D_cj = float(sys.argv[2]) if len(sys.argv) > 2 else None
    L_ind = float(sys.argv[3]) if len(sys.argv) > 3 else None
    base = os.path.dirname(pvd)
    files = re.findall(r'file="([^"]+)"', open(pvd).read())
    rows = []
    p0 = None
    for f in files:
        t, x, p, T, hrr = profile(os.path.join(base, f))
        p0 = p.min() if p0 is None else p0
        r = front(x, p, hrr, p0)
        if r and x[-1] - r[0] > 0.01:
            rows.append((t, *r))
    rows = np.array(rows)
    print(f"{'t [us]':>8} {'x_s [mm]':>9} {'D [m/s]':>8} {'L_ind [mm]':>10} {'p_max [kPa]':>11}")
    for i, (t, xs, L, pm) in enumerate(rows):
        D = (xs - rows[i - 1, 1]) / (t - rows[i - 1, 0]) if i > 0 else float("nan")
        print(f"{t * 1e6:8.1f} {xs * 1e3:9.2f} {D:8.1f} {L * 1e3:10.3f} {pm / 1e3:11.1f}")
    half = rows[rows[:, 0] >= 0.5 * rows[-1, 0]]
    D_mean = np.polyfit(half[:, 0], half[:, 1], 1)[0]
    L_mean = half[:, 2].mean()
    print(f"mean over the second half: D = {D_mean:.1f} m/s, induction length {L_mean * 1e3:.3f} mm")
    if D_cj:
        print(f"D / D_CJ - 1 = {D_mean / D_cj - 1:+.4f}")
    if L_ind:
        print(f"L / L_ZND - 1 = {L_mean / (L_ind * 1e-3) - 1:+.4f}")
    if len(sys.argv) > 4:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t, x, p, T, _ = profile(os.path.join(base, files[-1]))
        fig, ax = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
        ax[0].plot(x * 1e3, p / 1e3)
        ax[0].set_ylabel("p [kPa]")
        ax[1].plot(x * 1e3, T)
        ax[1].set_ylabel("T [K]")
        ax[1].set_xlabel("x [mm]")
        fig.tight_layout()
        fig.savefig(sys.argv[4], dpi=120)


if __name__ == "__main__":
    main()
