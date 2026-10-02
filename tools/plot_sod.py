#!/usr/bin/env python3
"""Plot x-profiles of Mallard runs against the exact Sod solution."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mallard_vtu import read_vtu

def exact_sod(x, t, g=1.4):
    import scipy.optimize as so
    rl, ul, pl, rr, ur, pr = 1.0, 0.0, 1.0, 0.125, 0.0, 0.1
    al, ar = np.sqrt(g * pl / rl), np.sqrt(g * pr / rr)
    def f(p, rk, pk, ak):
        if p > pk:
            A, B = 2 / ((g + 1) * rk), (g - 1) / (g + 1) * pk
            return (p - pk) * np.sqrt(A / (p + B))
        return 2 * ak / (g - 1) * ((p / pk) ** ((g - 1) / (2 * g)) - 1)
    ps = so.brentq(lambda p: f(p, rl, pl, al) + f(p, rr, pr, ar) + ur - ul, 1e-6, 10)
    us = 0.5 * (ul + ur) + 0.5 * (f(ps, rr, pr, ar) - f(ps, rl, pl, al))
    rsl = rl * (ps / pl) ** (1 / g)
    rsr = rr * ((ps / pr) + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * ps / pr + 1)
    S = ur + ar * np.sqrt((g + 1) / (2 * g) * ps / pr + (g - 1) / (2 * g))
    ast = al * (ps / pl) ** ((g - 1) / (2 * g))
    out = []
    for xi in x:
        s = (xi - 0.5) / t
        if s < ul - al: out.append(rl)
        elif s < us - ast:
            c = 2 / (g + 1) + (g - 1) / ((g + 1) * al) * (ul - s)
            out.append(rl * c ** (2 / (g - 1)))
        elif s < us: out.append(rsl)
        elif s < S: out.append(rsr)
        else: out.append(rr)
    return np.array(out)

def main():
    out = sys.argv[1]
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.linspace(0, 1, 2000)
    ax.plot(xs, exact_sod(xs, 0.2), "k-", lw=1, label="exact")
    for f in sys.argv[2:]:
        pts, tris, tc, d = read_vtu(f)
        cen = np.array([pts[tris[tc == c]].mean(axis=(0, 1)) for c in range(len(d["RHO"]))])
        sel = cen[:, 1] < cen[:, 1].min() + 1e-9
        o = np.argsort(cen[sel, 0])
        ax.plot(cen[sel, 0][o], d["RHO"][sel][o], ".-", ms=3, lw=0.6, label=f.split("/")[-2])
    ax.legend()
    fig.savefig(out, dpi=100, bbox_inches="tight")


if __name__ == "__main__":
    main()
