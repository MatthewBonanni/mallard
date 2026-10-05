"""Propagation speed and structure of triple-flame runs (tools/triple_flame.py).

    python tools/triple_flame_analysis.py RUN_DIR [RUN_DIR ...] [--fuel H2:0.3,N2:0.7]
        [--oxidizer O2:1,N2:3.76] [--mechanism mechanisms/h2o2.yaml --phase ohmech]
        [--fit T0 T1] [--plot OUT.png] [--csv OUT.csv]

For each output of RUN_DIR/solut/flame.pvd:
  x_tip, y_tip  the leading edge: the most upstream point of the isotherm
                T_u + (T_b - T_u) / 2 of the planar flame at Z_st, interpolated
                along each row, refined by a parabola through the three rows
                around the minimum;
  Z_tip         the mixture fraction there, Z = (Y_N2 - Y_N2,ox) / (Y_N2,fuel -
                Y_N2,ox) (N2 does not react and diffuses with a Lewis number
                close to 1);
  D_TF          the local mixing thickness 1 / (dZ/dy) at Z_st over delta_L, on
                the column through the largest heat release within 2 delta_L
                of the leading edge (Ruetsch, Vervisch & Linan 1995 measure it
                through the maximum reaction rate);
  u_min         the smallest axial velocity on the row of the leading edge,
                upstream of it: the flow slows from u_in to about the local
                flame speed as the streamlines diverge ahead of the tip.
The propagation speed against the incoming flow, U_F = u_in - dx_tip/dt, is
fitted over --fit T0 T1 (in flame times delta_L / S_L; default the second half
of the run), and the local flame speed at the tip, S_tip = u_min - dx_tip/dt,
averaged there. Prints U_F / S_L, (U_F / S_L - 1) / (sqrt(rho_u / rho_b) - 1)
(the share of the heat-release limit of Ruetsch et al.'s scaling U_F / S_L ->
sqrt(rho_u / rho_b) for weak gradients), S_tip / S_L, U_F / S_tip (the
speed-up by the streamlines' divergence alone) and D_TF, and with --plot draws
them against Ruetsch et al.'s simulations (their Table I).
"""
import argparse
import os
import re
import sys

import cantera as ct
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mallard_vtu import read_quads  # noqa: E402

# Ruetsch, Vervisch & Linan (1995), Table I (beta = 8, Le = 1, Pr = 0.75): alpha = 1 - rho_b / rho_u, D_TF,
# U_F / S_L. At rho_u / rho_b = 4, U_F / S_L rises from 1.23 to 1.51 as D_TF goes from 8.3 to 34 (the ends of
# that series); their simulation IX has rho_u / rho_b = 5.
RUETSCH = [(0.75, 8.3, 1.23), (0.75, 34.0, 1.51), (0.8, 33.4, 1.61)]


def grid(path, names):
    t, c, arrays = read_quads(path, names)
    xs, ys = np.unique(np.round(c[:, 0], 12)), np.unique(np.round(c[:, 1], 12))
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    i = np.rint((c[:, 0] - xs[0]) / dx).astype(int)
    j = np.rint((c[:, 1] - ys[0]) / dy).astype(int)
    out = {}
    for name in names:
        a = arrays[name]
        f = np.empty((xs.size, ys.size) + a.shape[1:])
        f[i, j] = a
        out[name] = f
    return t, xs, ys, out


def stream_n2(spec, mechanism, phase):
    gas = ct.Solution(mechanism, phase)
    gas.TPX = 300.0, ct.one_atm, spec
    return gas["N2"].Y[0]


def run_info(run_dir, mechanism, phase):
    head = open(os.path.join(run_dir, "input.toml")).read()
    u_in = float(re.search(r'name = "left"\ntype = "upt"\nu = \[([^,]+),', head).group(1))
    d = np.loadtxt(os.path.join(run_dir, "planar_full.csv"), delimiter=",", skiprows=1)
    x, T = d[:, 0], d[:, 2]
    delta = (T[-1] - T[0]) / np.gradient(T, x).max()
    gas = ct.Solution(mechanism, phase)
    hrr = []
    for row in d:
        gas.TDY = row[2], row[3], row[4:]
        hrr.append(gas.heat_release_rate)
    return dict(u_in=u_in, S_L=d[0, 1], delta=delta, sigma=d[0, 3] / d[-1, 3], T_u=T[0], T_b=T[-1],
                hrr_max=max(hrr), mixing=float(re.search(r"delta_M = ([0-9.]+) delta_L", head).group(1)))


def leading_edge(xs, ys, T, T_mid):
    """Most upstream point of the isotherm T_mid, from row-wise crossings."""
    hot = T >= T_mid
    first = np.where(hot.any(axis=0), hot.argmax(axis=0), -1)
    xcross = np.full(ys.size, np.inf)
    for j in np.nonzero(first > 0)[0]:
        i = first[j]
        xcross[j] = xs[i - 1] + (T_mid - T[i - 1, j]) / (T[i, j] - T[i - 1, j]) * (xs[i] - xs[i - 1])
    j = int(np.argmin(xcross))
    if 0 < j < ys.size - 1 and np.all(np.isfinite(xcross[j - 1:j + 2])):
        a, b, c = xcross[j - 1:j + 2]
        den = a - 2 * b + c
        s = 0.5 * (a - c) / den if den > 0 else 0.0
        return b - 0.25 * (a - c) * s, ys[j] + s * (ys[1] - ys[0]), j
    return xcross[j], ys[j], j


def analyze(run_dir, info, Y_N2_F, Y_N2_O, Z_st):
    pvd = os.path.join(run_dir, "solut", "flame.pvd")
    files = re.findall(r'file="([^"]+)"', open(pvd).read())
    T_mid = info["T_u"] + 0.5 * (info["T_b"] - info["T_u"])
    rows = []
    for f in files:
        t, xs, ys, a = grid(os.path.join(run_dir, "solut", f), ["T", "HRR", "U", "Y_N2"])
        Z = (a["Y_N2"] - Y_N2_O) / (Y_N2_F - Y_N2_O)
        x_tip, y_tip, j = leading_edge(xs, ys, a["T"], T_mid)
        i_tip = int(np.searchsorted(xs, x_tip))
        Z_tip = Z[min(i_tip, xs.size - 1), j]
        near = (np.abs(xs[:, None] - x_tip) < 2 * info["delta"]) & (np.abs(ys[None, :] - y_tip) < 2 * info["delta"])
        h = np.where(near, a["HRR"], -np.inf)
        i_h, j_h = np.unravel_index(np.argmax(h), h.shape)
        col = Z[i_h]
        k = np.nonzero(np.diff(np.sign(col - Z_st)))[0]
        D_TF = np.nan
        if k.size:
            k = k[np.argmin(np.abs(k - j_h))]
            D_TF = (ys[1] - ys[0]) / abs(col[k + 1] - col[k]) / info["delta"]
        u_row = a["U"][:i_tip, j, 0]
        u_min = u_row.min() if u_row.size else np.nan
        rows.append((t, x_tip, y_tip, Z_tip, D_TF, u_min, a["HRR"].max() / info["hrr_max"]))
    return np.array(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--fuel", default="H2:0.3,N2:0.7")
    ap.add_argument("--oxidizer", default="O2:1,N2:3.76")
    ap.add_argument("--mechanism", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                                        "mechanisms", "h2o2.yaml"))
    ap.add_argument("--phase", default="ohmech")
    ap.add_argument("--fit", type=float, nargs=2, metavar=("T0", "T1"))
    ap.add_argument("--plot")
    ap.add_argument("--csv")
    args = ap.parse_args()
    gas = ct.Solution(args.mechanism, args.phase)
    gas.TP = 300.0, ct.one_atm
    gas.set_equivalence_ratio(1.0, args.fuel, args.oxidizer)
    Z_st = gas.mixture_fraction(args.fuel, args.oxidizer)
    Y_N2_F, Y_N2_O = stream_n2(args.fuel, args.mechanism, args.phase), stream_n2(args.oxidizer, args.mechanism,
                                                                                  args.phase)
    results = []
    for run in args.runs:
        info = run_info(run, args.mechanism, args.phase)
        rows = analyze(run, info, Y_N2_F, Y_N2_O, Z_st)
        S_L, tau = info["S_L"], info["delta"] / info["S_L"]
        t0, t1 = (np.array(args.fit) * tau) if args.fit else (0.5 * rows[-1, 0], rows[-1, 0])
        late = rows[(rows[:, 0] >= t0 - 1e-12) & (rows[:, 0] <= t1 + 1e-12)]
        v_tip, _ = np.polyfit(late[:, 0], late[:, 1], 1)
        U_F = info["u_in"] - v_tip
        S_tip = np.nanmean(late[:, 5]) - v_tip
        share = (U_F / S_L - 1) / (np.sqrt(info["sigma"]) - 1)
        D_TF = np.nanmean(late[:, 4])
        print(f"{run}: delta_M0 = {info['mixing']:g} delta_L, S_L = {S_L:.4f} m/s, delta_L = {info['delta'] * 1e3:.4f} mm, "
              f"rho_u/rho_b = {info['sigma']:.3f} (sqrt {np.sqrt(info['sigma']):.3f})")
        print(f"{'t/tau':>7} {'x_tip/dL':>9} {'y_tip/dL':>9} {'Z_tip':>7} {'D_TF':>7} {'u_min/SL':>9} {'HRRmax':>7}")
        for r in rows[:: max(1, len(rows) // 25)]:
            print(f"{r[0] / tau:7.2f} {r[1] / info['delta']:9.3f} {r[2] / info['delta']:9.3f} {r[3]:7.4f} {r[4]:7.2f} "
                  f"{r[5] / S_L:9.3f} {r[6]:7.3f}")
        print(f"  U_F / S_L = {U_F / S_L:.4f} (tip drift {v_tip / S_L:+.4f} S_L over t S_L / delta_L = "
              f"{t0 / tau:.2f}-{t1 / tau:.2f}, x_tip / delta_L = {late[0, 1] / info['delta']:.2f}-"
              f"{late[-1, 1] / info['delta']:.2f}), share of sqrt(rho_u/rho_b) - 1: {share:.3f}; "
              f"S_tip / S_L = {S_tip / S_L:.3f}, U_F / S_tip = {U_F / S_tip:.3f}; "
              f"D_TF = {D_TF:.2f}; Z_tip = {late[:, 3].mean():.4f} (Z_st = {Z_st:.4f})")
        results.append(dict(run=run, info=info, rows=rows, U_F=U_F / S_L, share=share, D_TF=D_TF,
                            u_min=np.nanmean(late[:, 5]) / S_L, S_tip=S_tip / S_L, Z_tip=late[:, 3].mean()))
    if args.csv:
        with open(args.csv, "w") as fh:
            fh.write("run,delta_M0,D_TF,U_F/S_L,share,u_min/S_L,S_tip/S_L,Z_tip,rho_u/rho_b\n")
            for r in results:
                fh.write(f"{r['run']},{r['info']['mixing']:g},{r['D_TF']:.3f},{r['U_F']:.4f},{r['share']:.4f},"
                         f"{r['u_min']:.4f},{r['S_tip']:.4f},{r['Z_tip']:.4f},{r['info']['sigma']:.4f}\n")
    if args.plot:
        plot(results, args.plot)


def plot(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8))
    colors = ["#e8710a", "#1a73e8", "#188038", "#a142f4"]
    ax = axs[0]
    for r, c in zip(results, colors):
        tau = r["info"]["delta"] / r["info"]["S_L"]
        rows = r["rows"]
        t = rows[:, 0] / tau
        v = np.gradient(rows[:, 1], rows[:, 0])
        k = max(3, len(t) // 15)
        v_s = np.convolve(v, np.ones(k) / k, mode="same")
        ax.plot(t[k:-k], (r["info"]["u_in"] - v_s[k:-k]) / r["info"]["S_L"], color=c,
                label=f"delta_M0 = {r['info']['mixing']:g} delta_L")
    if results:
        ax.axhline(np.sqrt(results[0]["info"]["sigma"]), color="#888", ls=":", lw=1)
    ax.set_xlabel("t S_L / delta_L")
    ax.set_ylabel("U_F / S_L")
    ax.set_title("Propagation speed against the inflow (smoothed)")
    ax.legend()
    ax = axs[1]
    for (alpha, D, U), m in zip(RUETSCH, ["o", "o", "^"]):
        ax.plot(D, U, m, mfc="none", mec="#555", ms=8)
    ax.plot([], [], "o", mfc="none", mec="#555", label="Ruetsch et al. 1995, rho_u/rho_b = 4")
    ax.plot([], [], "^", mfc="none", mec="#555", label="Ruetsch et al. 1995, rho_u/rho_b = 5")
    for s, ls in ((4.0, "--"), (5.0, "-.")):
        ax.axhline(np.sqrt(s), color="#999", ls=ls, lw=0.8)
    for r, c in zip(results, colors):
        ax.plot(r["D_TF"], r["U_F"], "s", color=c, ms=9,
                label=f"Mallard, delta_M0 = {r['info']['mixing']:g} (rho_u/rho_b = {r['info']['sigma']:.2f})")
    if results:
        ax.axhline(np.sqrt(results[0]["info"]["sigma"]), color=colors[0], ls=":", lw=1)
    ax.set_xlabel("D_TF = local mixing thickness / delta_L")
    ax.set_ylabel("U_F / S_L")
    ax.set_xlim(0, 40)
    ax.set_ylim(1, 2.6)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("U_F / S_L (lines: sqrt(rho_u / rho_b))")
    ax = axs[2]
    for (alpha, D, U), m in zip(RUETSCH, ["o", "o", "^"]):
        ax.plot(D, (U - 1) / (np.sqrt(1 / (1 - alpha)) - 1), m, mfc="none", mec="#555", ms=8)
    ax.plot([], [], "o", mfc="none", mec="#555", label="Ruetsch et al. 1995 (one-step, Le = 1)")
    for r, c in zip(results, colors):
        ax.plot(r["D_TF"], r["share"], "s", ms=9, color=c, label=f"Mallard, delta_M0 = {r['info']['mixing']:g} delta_L")
        ax.plot(r["D_TF"], (r["U_F"] / r["S_tip"] - 1) / (np.sqrt(r["info"]["sigma"]) - 1), "s", ms=9, mfc="none",
                color=c)
    ax.plot([], [], "s", ms=9, mfc="none", color="#555", label="Mallard, U_F / S_tip in place of U_F / S_L")
    ax.set_xlabel("D_TF = local mixing thickness / delta_L")
    ax.set_ylabel("(U_F / S_L - 1) / (sqrt(rho_u / rho_b) - 1)")
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 1.05)
    ax.axhline(1, color="#888", ls=":", lw=1)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("Share of the heat-release limit")
    fig.tight_layout()
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
