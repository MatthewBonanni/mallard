#!/usr/bin/env python3
"""Soot foils and front speed of a 3D detonation in a duct (examples/detonation_3d).

    soot_foil_3d.py INPUT FRAME_DIR [--png foils.png] [--from-cell N] [--d-cj 1616.93]

Assembles the numerical soot foils (P_MAX next to the four walls) of a run of
tools/detonation_window.py from its foil chunks and last frame, and reports:
the mean front speed (a straight-line fit of the front position of the
frames against time, over their second half) against D_CJ; and for each
wall the dominant spacing of the tracks along x (cell length) and across
it (cell width), from the first peak of the foil's autocorrelation (log
P_MAX minus its mean along each line) along each direction, averaged over
the foil. --from-cell skips the start of the duct (default: the front's
position in the first frame). With --png, writes the four foils unrolled,
gray levels as tools/soot_foil.py (log P_MAX, 2nd to 99.5th percentile).
"""
import argparse
import glob
import os

import numpy as np

from animate_detonation_3d import WALL_LABELS, WALLS, foil_history, full_foils
from detonation_window import Case


def first_peak(signal, dx, min_lag):
    """Lag of the first maximum of the autocorrelation after it first drops below zero."""
    s = signal - signal.mean(axis=-1, keepdims=True)
    n = s.shape[-1]
    f = np.fft.rfft(s, 2 * n, axis=-1)
    ac = np.fft.irfft(f * np.conj(f), axis=-1)[..., :n].mean(axis=0)
    ac /= ac[0]
    neg = np.nonzero(ac < 0)[0]
    if neg.size == 0:
        return np.nan
    start = max(neg[0], min_lag)
    k = start + int(np.argmax(ac[start:n // 2])) if start < n // 2 else -1
    return k * dx if k > 0 and ac[k] > 0 else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("frame_dir")
    ap.add_argument("--png")
    ap.add_argument("--from-cell", type=int)
    ap.add_argument("--d-cj", type=float, default=1616.93)
    args = ap.parse_args()
    case = Case(args.input)
    files = sorted(glob.glob(os.path.join(args.frame_dir, "frame_*.npz")))
    t, x = [], []
    for f in files:
        with np.load(f) as z:
            t.append(float(z["t"]))
            x.append((int(z["shift_cells"]) + int(z["i_front"]) + 0.5) * float(z["dx"]))
    t, x = np.array(t), np.array(x)
    half = t >= t[0] + 0.5 * (t[-1] - t[0])
    speed = np.polyfit(t[half], x[half], 1)[0]
    print(f"front: {x[-1] - x[0]:.4f} m in {t[-1] - t[0]:.3e} s; mean speed over the second half "
          f"{speed:.1f} m/s = {speed / args.d_cj:.4f} D_CJ")
    with np.load(files[-1]) as z:
        last = {k: z[k] for k in z.files}
    foils = full_foils(foil_history(case), last)
    dx = case.dx
    i0 = args.from_cell if args.from_cell is not None else int(round(x[0] / dx))
    i1 = int(last["shift_cells"]) + int(last["i_front"]) - 10
    for w in WALLS:
        f = np.log(foils[w][i0:i1])
        length = first_peak(f.T, dx, int(0.01 / dx))
        width = first_peak(f, dx, 5)
        print(f"{WALL_LABELS[w]:6s}: track spacing along x {length * 100:.2f} cm, across {width * 100:.2f} cm")
    if args.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        img = [np.log(foils[w][i0:i1]).T for w in WALLS]
        lo, hi = np.percentile(np.concatenate([a.ravel() for a in img]), [2, 99.5])
        fig, axes = plt.subplots(4, 1, figsize=(16, 9), sharex=True)
        for ax, w, a in zip(axes, WALLS, img):
            ax.imshow(a, origin="lower", cmap="gray_r", vmin=lo, vmax=hi, aspect="equal",
                      extent=(0, a.shape[1] * dx * 100, 0, case.size[1] * 100))
            ax.set_ylabel(WALL_LABELS[w])
        axes[-1].set_xlabel("x from the start of the 3D run (cm)")
        fig.tight_layout()
        fig.savefig(args.png, dpi=150)


if __name__ == "__main__":
    main()
