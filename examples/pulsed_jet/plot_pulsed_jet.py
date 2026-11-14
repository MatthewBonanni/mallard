#!/usr/bin/env python3
"""The pulsed-jet example (examples/pulsed_jet): tracks the vortex rings'
cores (maxima of the azimuthal vorticity) through the snapshots, prints each
ring's translation speed, diameter and circulation against the slug model,
and the inlet cell's velocity against U(t).

    plot_pulsed_jet.py PREFIX PROBE [--png FILE] [--figure]

PREFIX is the snapshots' prefix (solut/jet), PROBE the inlet probe
(solut/probe_inlet.csv). --png draws the figure into FILE, --figure into
docs/images/guide/pulsed_jet-{light,dark}.svg.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../tools"))
from mallard.vtu import read_index, read_vtu_cells  # noqa: E402

U0, PERIOD, D = 0.2, 30.0, 1.0


def U(t):
    return U0 * (1.0 - np.tanh(4.0 * np.cos(2.0 * np.pi * t / PERIOD))) / 2.0


def grid(path):
    pts, conn, offs, _, a = read_vtu_cells(path)
    starts = np.concatenate([[0], offs[:-1]])
    cx = np.add.reduceat(pts[conn, 0], starts) / np.diff(np.concatenate([starts, [conn.size]]))
    cy = np.add.reduceat(pts[conn, 1], starts) / np.diff(np.concatenate([starts, [conn.size]]))
    order = np.lexsort((cx, cy))
    nx = int(np.sum(np.isclose(cy, cy.min())))
    shape = (cx.size // nx, nx)
    return cx[order].reshape(shape), cy[order].reshape(shape), a["VORTICITY"][order].reshape(shape), a["TIME"]


def cores(x, y, w, threshold):
    """The rings' cores: the strongest vorticity maximum of each ring between x = 1 and 9 D."""
    inner = w[1:-1, 1:-1]
    peak = np.ones_like(inner, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj:
                peak &= inner >= w[1 + di:w.shape[0] - 1 + di, 1 + dj:w.shape[1] - 1 + dj]
    peak &= (inner > threshold) & (x[1:-1, 1:-1] > 1.0) & (x[1:-1, 1:-1] < 9.0)
    i, j = np.nonzero(peak)
    kept = []  # the strongest maximum of each ring: none within 0.6 D of a stronger one
    for k in np.argsort(-inner[i, j]):
        xc, rc = x[1:-1, 1:-1][i[k], j[k]], y[1:-1, 1:-1][i[k], j[k]]
        if all(np.hypot(xc - a, rc - b) > 0.6 * D for a, b in kept):
            kept.append((xc, rc))
    return sorted(kept, reverse=True)


def track(files, threshold, dA):
    """Follows the rings by continuity: each core continues the ring whose position, extrapolated from
    the previous snapshots, is nearest, so a ring keeps its number after the rings ahead of it have
    left the window. Returns ring k (0: the first pulse's) -> [(t, x, r, circulation)]."""
    tracks = []
    for n, f in enumerate(files):
        x, y, w, t = grid(f)
        found = cores(x, y, w, threshold)
        predicted = {}
        for k, rows in enumerate(tracks):
            if n - rows[-1][4] > 2:
                continue
            t1, x1, r1 = rows[-1][:3]
            speed = (x1 - rows[-2][1]) / (t1 - rows[-2][0]) if len(rows) > 1 else 0.4 * U0
            predicted[k] = (x1 + speed * (t - t1), r1)
        pairs = sorted((np.hypot(xc - px, rc - pr), k, c) for k, (px, pr) in predicted.items()
                       for c, (xc, rc) in enumerate(found))
        owner = {}
        for dist, k, c in pairs:
            if dist < 0.5 * D and k not in owner.values() and c not in owner:
                owner[c] = k
        released = int(np.floor((t - 0.75 * PERIOD) / PERIOD)) + 1  # rings whose pulse has ended
        for c in range(len(found)):  # found runs downstream to upstream: a new ring is the furthest unclaimed core
            if c not in owner and len(tracks) < released:
                owner[c] = len(tracks)
                tracks.append([])
        for c, k in owner.items():
            xc, rc = found[c]
            box = (np.abs(x - xc) < 0.75 * D) & (w > 0.0)
            tracks[k].append((t, xc, rc, w[box].sum() * dA, n))
    return {k: [row[:4] for row in rows] for k, rows in enumerate(tracks)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("probe")
    ap.add_argument("--png")
    ap.add_argument("--figure", action="store_true")
    args = ap.parse_args()

    files = [f for _, f in read_index(args.prefix)]
    x, y, w, t = grid(files[-1])
    threshold = 0.25 * w[(x > 1.0) & (x < 9.0)].max()
    dA = (x[0, 1] - x[0, 0]) * (y[1, 0] - y[0, 0])
    tracks = track(files, threshold, dA)
    tt = np.linspace(0.0, PERIOD, 30001)
    slug = np.trapezoid(U(tt) ** 2 / 2.0, tt)
    stroke = np.trapezoid(U(tt), tt)
    print(f"stroke per pulse L / D = {stroke / D:.3f}; slug-model circulation U^2/2 dt = {slug:.4f}")
    fits = {}
    for k, rows in sorted(tracks.items()):
        r = np.array(rows)
        sel = (r[:, 0] >= (k + 0.75) * PERIOD + 5.0) & (r[:, 1] < 8.0 * D)  # formed, clear of the sponge
        if sel.sum() < 4:
            continue
        speed, x0 = np.polyfit(r[sel, 0], r[sel, 1], 1)
        fits[k] = (speed, x0, r[sel, 0])
        print(f"ring {k + 1}: speed {speed / U0:.3f} U0 over t = {r[sel, 0][0]:.0f}-{r[sel, 0][-1]:.0f}, "
              f"diameter {2 * r[sel, 2].mean() / D:.3f} D, circulation {r[sel, 3].mean():.4f} "
              f"({r[sel, 3].mean() / slug:.2f} x slug model)")

    probe = np.genfromtxt(args.probe, delimiter=",", names=True)
    tp, up = probe["t"], probe["U_X"]
    plateau = U(tp) > 0.95 * U0
    print(f"inlet cell: max |u - U(t)| = {np.abs(up - U(tp)).max() / U0:.4f} U0 over {tp.size} samples to "
          f"t = {tp[-1]:.1f}, {np.abs(up - U(tp))[plateau].max() / U0:.4f} U0 where U > 0.95 U0")
    for k in range(int((tp[-1] + 0.01 * PERIOD) // PERIOD)):  # U(t) vanishes at the period's end
        sel = (tp >= k * PERIOD) & (tp <= (k + 1) * PERIOD)
        print(f"pulse {k + 1}: stroke of the inlet cell {np.trapezoid(up[sel], tp[sel]) / D:.3f} D "
              f"(target {stroke / D:.3f})")
    if not (args.png or args.figure):
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def draw(P):
        fig = plt.figure(figsize=(12.5, 8.8))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.45, 1.0], hspace=0.3, wspace=0.28)
        ax = fig.add_subplot(gs[0, :])
        lim = 0.5 * w.max()
        xx, yy = x[0], y[:, 0]
        mesh = ax.pcolormesh(xx, np.concatenate([-yy[::-1], yy]), np.vstack([-w[::-1], w]), cmap="RdBu_r",
                             vmin=-lim, vmax=lim, shading="auto", rasterized=True)
        cb = fig.colorbar(mesh, ax=ax, fraction=0.03, pad=0.01)
        cb.outline.set_edgecolor(P["muted"])
        ax.set_aspect("equal")
        ax.set_xlim(0.0, 9.0)
        ax.set_ylim(-2.0, 2.0)
        ax.set_xlabel("x / D")
        ax.set_ylabel("r / D")
        ax.set_title(rf"azimuthal vorticity at $t = {t:.0f}$ ($t\,U_0/D = {t * U0:.0f}$)", fontsize=13)
        ax = fig.add_subplot(gs[1, 0])
        colors = (P["blue"], P["flame"], P["teal"], P["plum"], P["green"], P["red"])
        for k, rows in sorted(tracks.items()):
            r = np.array(rows)
            c = colors[k % len(colors)]
            ax.plot(r[:, 0] * U0 / D, r[:, 1] / D, "o", ms=3, mfc="none", color=c, label=f"ring {k + 1}")
            if k in fits:
                speed, x0, ts = fits[k]
                ax.plot(ts * U0 / D, (speed * ts + x0) / D, color=c, lw=1.0)
        ax.set_xlabel(r"$t\,U_0 / D$")
        ax.set_ylabel(r"core position $x / D$")
        if tracks:
            ax.legend(loc="upper left", fontsize=10)
        ax = fig.add_subplot(gs[1, 1])
        tf = np.linspace(0.0, tp[-1], 2000)
        ax.plot(tf * U0 / D, U(tf) / U0, color=P["fg"], lw=1.0, label=r"$U(t)$ (target)")
        ax.plot(tp[::5] * U0 / D, up[::5] / U0, "o", ms=2.5, mfc="none", color=P["blue"], label="inlet cell")
        ax.set_xlabel(r"$t\,U_0 / D$")
        ax.set_ylabel(r"$u / U_0$ on the axis")
        ax.set_ylim(-0.05, 1.25)
        ax.legend(loc="upper right", ncol=2, fontsize=10)
        return fig

    if args.png:
        P = dict(fg="#1f1a1d", muted="#6b6166", blue="#2f6db5", flame="#d9730d", teal="#12655f",
                 plum="#7b3f6e", green="#3a7d2c", red="#c0392b")
        draw(P).savefig(args.png, dpi=150, bbox_inches="tight")
        print(f"wrote {args.png}")
    if args.figure:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../tools/docs"))
        import figstyle
        figstyle.save("pulsed_jet", draw)


if __name__ == "__main__":
    main()
