"""Counterflow diffusion flame runs (V10) against Cantera.

    python tools/plot_counterflow.py OUT_PREFIX --run LABEL RUN_DIR [--run ...]
        [--reference examples/counterflow_diffusion/reference] [--history] [--still RADIUS]

Each RUN_DIR holds a run of tools/counterflow_setup.py; its symmetry-plane
output (solut/plane_*.vtu) gives the stagnation line: the cells next to the
axis, extrapolated to r = 0 with the cells one further out (f = f0 + b r^2).
On it: the peak temperature (parabola through the hottest cell and its
neighbors) and the local strain rates K_ox and K_f (the first maximum of
-du/dx going from each nozzle towards the flame) and the spread rate
V = v / r at the peak temperature, as tools/counterflow_reference.py
measures Cantera's.

Writes OUT_PREFIX_strain.png (peak T against K_ox and against V at the
flame: Cantera's sweep up to extinction, and the runs),
OUT_PREFIX_profiles.png (T, u and major species along the stagnation line against Cantera's flame of the same K_ox, from
reference/K<K_ox>.csv written by counterflow_reference.py --match, aligned
at the peak temperature), with --history OUT_PREFIX_history.png (peak T
and K_ox over time) and with --still OUT_PREFIX_still.png (the first run's
temperature on the symmetry plane, with streamlines), and prints the
comparison table.
"""
import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mallard_vtu import read_vtu_cells  # noqa: E402

SPECIES = ["H2", "O2", "H2O", "OH"]


def peak(x, T):
    """Peak of T through the parabola of its largest sample and neighbors."""
    i = int(np.clip(np.argmax(T), 1, T.size - 2))
    c = np.polyfit(x[i - 1:i + 2] - x[i], T[i - 1:i + 2], 2)
    if c[0] >= 0:
        return T.max(), x[np.argmax(T)]
    xm = -c[1] / (2 * c[0])
    return np.polyval(c, xm), x[i] + xm


def local_strains(z, u):
    """K_ox, K_f: the local strain rates ahead of the flame, the first maxima
    of -du/dz met going from each nozzle towards the flame (before the
    flame's dilatation reverses the gradient)."""
    g = -np.gradient(u, z)
    i = g.size - 1
    while i > 0 and g[i - 1] >= g[i]:
        i -= 1
    j = 0
    while j < g.size - 1 and g[j + 1] >= g[j]:
        j += 1
    return g[i], g[j]


def stagnation_line(path):
    """x and the fields on the axis from a z = 0 symmetry-plane VTU."""
    pts, conn, offs, _, arrays = read_vtu_cells(path)
    c = pts[conn.reshape(-1, 4)].mean(axis=1)
    x, y = np.round(c[:, 0], 12), np.round(c[:, 1], 12)
    ys = np.unique(y)
    rows = []
    for yy in ys[:2]:
        sel = np.nonzero(y == yy)[0]
        sel = sel[np.argsort(x[sel])]
        rows.append(sel)
    # cell centers of the plane's cells are at z = dz0 / 2, the first ones at y = dy0 / 2 = dz0 / 2
    z0 = ys[0]
    r2 = [ys[0] ** 2 + z0 ** 2, ys[1] ** 2 + z0 ** 2]
    w = r2[0] / (r2[1] - r2[0])
    out = {"x": x[rows[0]], "TIME": arrays["TIME"]}
    for name, v in arrays.items():
        if name == "TIME":
            continue
        if v.ndim == 2:
            v = v[:, 0]
        out[name] = v[rows[0]] - w * (v[rows[1]] - v[rows[0]])
    V = [arrays["U"][rows[k], 1] / ys[k] for k in range(2)]
    out["V"] = V[0] - w * (V[1] - V[0])
    return out


def run_series(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "solut", "plane_*.vtu")))
    if not files:
        raise SystemExit(f"{run_dir}: no solut/plane_*.vtu")
    return files


def measure(line):
    """Peak temperature, its position, K_ox, K_f, and the spread rate V = v / r there."""
    T_max, x_T = peak(line["x"], line["T"])
    K_ox, K_f = local_strains(line["x"], line["U"])
    return T_max, x_T, K_ox, K_f, np.interp(x_T, line["x"], line["V"])


def still(path, out, radius=None):
    """Temperature on the symmetry plane z = 0 mirrored about the axis, with
    streamlines and heat release contours."""
    from scipy.interpolate import RegularGridInterpolator
    pts, conn, offs, _, arrays = read_vtu_cells(path)
    c = pts[conn.reshape(-1, 4)].mean(axis=1)
    xs, ys = np.unique(np.round(c[:, 0], 12)), np.unique(np.round(c[:, 1], 12))
    i = np.searchsorted(xs, np.round(c[:, 0], 12))
    j = np.searchsorted(ys, np.round(c[:, 1], 12))
    grid = {}
    for name in ["T", "HRR"]:
        g = np.empty((xs.size, ys.size))
        g[i, j] = arrays[name]
        grid[name] = g
    for k, name in enumerate(["ux", "uy"]):
        g = np.empty((xs.size, ys.size))
        g[i, j] = arrays["U"][:, k]
        grid[name] = g
    # mirror to y < 0 (u_y odd) and resample on a uniform grid for streamplot
    ym = np.concatenate([-ys[::-1], ys])
    mirror = lambda g, sign=1.0: np.concatenate([sign * g[:, ::-1], g], axis=1)
    xu = np.linspace(xs[0], xs[-1], 400)
    yu = np.linspace(ym[0], ym[-1], 2 * int(400 * ys[-1] / xs[-1]))
    X, Y = np.meshgrid(xu, yu, indexing="ij")
    q = np.column_stack([X.ravel(), Y.ravel()])
    f = {n: RegularGridInterpolator((xs, ym), mirror(g, -1.0 if n == "uy" else 1.0), bounds_error=False,
                                    fill_value=None)(q).reshape(X.shape) for n, g in grid.items()}
    fig, ax = plt.subplots(figsize=(7.2, 7.2 * ym[-1] / xs[-1] + 0.6))
    im = ax.pcolormesh(Y.T * 1e3, X.T * 1e3, f["T"].T, cmap="inferno", shading="gouraud", rasterized=True)
    ax.streamplot(yu * 1e3, xu * 1e3, f["uy"], f["ux"], color="w", linewidth=0.5, density=1.4, arrowsize=0.6)
    ax.contour(Y.T * 1e3, X.T * 1e3, f["HRR"].T, levels=[0.1 * f["HRR"].max()], colors="c", linewidths=0.8)
    if radius:
        for xn in (xs[0], xs[-1]):
            ax.plot([-radius * 1e3, radius * 1e3], [xn * 1e3] * 2, "c-", lw=4, solid_capstyle="butt")
    ax.set_aspect("equal")
    ax.set_xlabel("r [mm] (symmetry plane, mirrored)")
    ax.set_ylabel("x [mm] (fuel nozzle at x = 0)")
    fig.colorbar(im, ax=ax, label="T [K]", shrink=0.8)
    ax.set_title(f"H$_2$/N$_2$ (bottom) against air (top), t = {arrays['TIME'] * 1e3:.2f} ms", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")


def load_reference(path):
    with open(path) as f:
        header = f.readline()
        names = f.readline().strip().split(",")
    data = np.loadtxt(path, delimiter=",", skiprows=2)
    return header, {n: data[:, i] for i, n in enumerate(names)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--run", nargs=2, action="append", metavar=("LABEL", "DIR"), required=True)
    ap.add_argument("--reference", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                        "examples", "counterflow_diffusion", "reference"))
    ap.add_argument("--history", action="store_true")
    ap.add_argument("--still", metavar="RADIUS", type=float,
                    help="also OUT_PREFIX_still.png: the first run's last plane, nozzles of this radius marked")
    args = ap.parse_args()
    if args.still:
        still(run_series(args.run[0][1])[-1], args.out + "_still.png", args.still)

    sweep = np.genfromtxt(os.path.join(args.reference, "strain.csv"), delimiter=",", names=True, skip_header=1)
    K_ref, T_ref, V_ref = sweep["K_ox"], sweep["T_max"], sweep["V_T"]
    matched = {}
    for path in glob.glob(os.path.join(args.reference, "K*.csv")):
        header, prof = load_reference(path)
        K = float(re.search(r"K_ox = ([0-9.eE+-]+)", header).group(1))
        matched[K] = prof

    results = []
    for label, run_dir in args.run:
        files = run_series(run_dir)
        line = stagnation_line(files[-1])
        T_max, x_T, K_ox, K_f, V_T = measure(line)
        hist = None
        if args.history:
            hist = np.array([(stagnation_line(f)["TIME"],) + measure(stagnation_line(f)) for f in files])
        T_cantera = np.interp(K_ox, K_ref, T_ref) if K_ox <= K_ref.max() else np.nan
        T_cantera_V = np.interp(V_T, V_ref, T_ref) if V_T <= V_ref.max() else np.nan
        results.append(dict(label=label, line=line, T_max=T_max, x_T=x_T, K_ox=K_ox, K_f=K_f, V_T=V_T, hist=hist,
                            T_cantera=T_cantera, T_cantera_V=T_cantera_V, t=line["TIME"]))

    print(f"{'run':<28s} {'t [ms]':>8s} {'K_ox':>8s} {'K_f':>8s} {'T_max':>8s} {'Cantera':>8s} {'error':>7s} "
          f"{'V_T':>8s} {'Cantera':>8s} {'error':>7s}")
    for r in results:
        err = (r["T_max"] - r["T_cantera"]) / r["T_cantera"]
        err_V = (r["T_max"] - r["T_cantera_V"]) / r["T_cantera_V"]
        print(f"{r['label']:<28s} {r['t'] * 1e3:8.3f} {r['K_ox']:8.1f} {r['K_f']:8.1f} {r['T_max']:8.1f} "
              f"{r['T_cantera']:8.1f} {100 * err:6.2f}% {r['V_T']:8.1f} {r['T_cantera_V']:8.1f} {100 * err_V:6.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), sharey=True)
    for ax, S_ref, key, label in [
            (axes[0], K_ref, "K_ox", r"local strain rate $K_{ox}$ = max $(-\partial u / \partial x)$ ahead of the flame [1/s]"),
            (axes[1], V_ref, "V_T", r"spread rate $V = v / r$ at the peak temperature [1/s]")]:
        ax.plot(S_ref, T_ref, "k-", lw=1.5, label="Cantera (axisymmetric similarity solution)")
        ax.plot(S_ref[-1], T_ref[-1], "kx", ms=9, mew=2, label=f"Cantera extinction ({S_ref[-1]:.0f} 1/s)")
        band = np.linspace(S_ref.min(), S_ref.max(), 200)
        T_band = np.interp(band, S_ref, T_ref)
        ax.fill_between(band, 0.98 * T_band, 1.02 * T_band, color="0.85", lw=0, label="±2%")
        for k, r in enumerate(results):
            ax.plot(r[key], r["T_max"], "os^D"[k // 10 % 4], color=f"C{k % 10}", ms=7,
                    label=f"Mallard, {r['label']}")
        ax.set_xscale("log")
        x0 = 0.5 * min(r[key] for r in results)
        ax.set_xlim(x0, 1.3 * S_ref[-1])
        T_lo = min(T_ref[-1], *(r["T_max"] for r in results))
        T_hi = max(np.interp(x0, S_ref, T_ref), *(r["T_max"] for r in results))
        ax.set_ylim(T_lo - 0.1 * (T_hi - T_lo), T_hi + 0.1 * (T_hi - T_lo))
        ax.set_xlabel(label)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("peak temperature [K]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="center left", bbox_to_anchor=(0.8, 0.5))
    fig.suptitle("H$_2$/N$_2$ (1:3) against air, 300 K, 1 atm: peak temperature against strain", fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.8, 1))
    fig.savefig(args.out + "_strain.png", dpi=150, bbox_inches="tight")

    with_ref = [r for r in results if matched]
    if with_ref:
        fig, axes = plt.subplots(len(with_ref), 2, figsize=(11, 3.4 * len(with_ref)), squeeze=False)
        for row, r in zip(axes, with_ref):
            K = min(matched, key=lambda k: abs(k - r["K_ox"]))
            ref = matched[K]
            z_T = peak(ref["z"], ref["T"])[1]
            x = (r["line"]["x"] - r["x_T"]) * 1e3
            zr = (ref["z"] - z_T) * 1e3
            ax = row[0]
            ax.plot(zr, ref["T"], "k-", lw=1.2, label=f"Cantera, $K_{{ox}}$ = {K:.0f} 1/s")
            ax.plot(x, r["line"]["T"], "C3o", ms=3, label=f"Mallard, $K_{{ox}}$ = {r['K_ox']:.0f} 1/s")
            ax.set_ylabel("T [K]")
            ax2 = ax.twinx()
            ax2.plot(zr, ref["u"], "k--", lw=1)
            ax2.plot(x, r["line"]["U"], "C0s", ms=2.5)
            ax2.set_ylabel("u [m/s] (dashed, squares)")
            ax.set_title(r["label"], fontsize=10)
            ax.legend(fontsize=8, loc="upper left")
            ax = row[1]
            for k, s in enumerate(SPECIES):
                scale = 10.0 if s == "OH" else 1.0
                lab = f"{s}" + (" x10" if scale != 1 else "")
                ax.plot(zr, scale * ref["Y_" + s], "-", color=f"C{k}", lw=1.2, label=lab)
                ax.plot(x, scale * r["line"]["Y_" + s], "o", color=f"C{k}", ms=2.5)
            ax.set_ylabel("mass fraction")
            ax.legend(fontsize=8, ncol=2)
            for a in row:
                a.set_xlim(-4, 4)
                a.set_xlabel("x - x(T$_{max}$) [mm]")
                a.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out + "_profiles.png", dpi=150)

    if args.history:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        for r in results:
            h = r["hist"]
            axes[0].plot(h[:, 0] * 1e3, h[:, 1], label=r["label"])
            axes[1].plot(h[:, 0] * 1e3, h[:, 3], label=r["label"])
        axes[0].set_ylabel("peak T [K]")
        axes[1].set_ylabel("K_ox [1/s]")
        for a in axes:
            a.set_xlabel("t [ms]")
            a.grid(True, alpha=0.3)
            a.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.out + "_history.png", dpi=150)


if __name__ == "__main__":
    main()
