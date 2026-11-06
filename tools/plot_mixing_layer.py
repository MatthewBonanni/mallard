"""Laminar reacting mixing layer runs (V11) against the boundary-layer reference.

    python tools/plot_mixing_layer.py REFERENCE_DIR --run LABEL RUN_DIR [--run ...]
        [--out mixing_layer] [--history] [--average N]

REFERENCE_DIR holds ignition.csv and profiles.csv from
tools/mixing_layer_reference.py. Each run's last snapshot (solut/field_*,
or the mean of the last N with --average) is reduced column by column: the
peak temperature, peak Y_OH and peak heat release rate across the layer
against x, and the lift-off (ignition) lengths

  x_OH   where the peak Y_OH first reaches 2e-3
  x_T    where the peak temperature first exceeds the air's by 250 K
  x_HRR  where the peak heat release rate is largest

are printed against the reference's. Profiles across the layer at the
reference's stations are aligned at the stoichiometric mixture fraction
(Bilger's). Writes OUT_peaks.png, OUT_profiles.png, OUT_field.png and with
--history OUT_history.png (the lift-off lengths of every snapshot).
"""
import argparse
import glob
import os
import re

import cantera as ct
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mallard_vtu import read_quads
from mixing_layer_reference import AIR, MECH, PHASE, bilger

OH_LEVEL, DT_LEVEL = 2e-3, 250.0
SPECIES = ["H2", "O2", "H2O", "OH", "HO2", "H"]


def snapshots(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "solut", "field_*.pvtu")))
    return files or sorted(f for f in glob.glob(os.path.join(run_dir, "solut", "field_*.vtu"))
                           if not re.search(r"_p\d{4}\.vtu$", f))


def load(path, names):
    """Time, x and y of the cell centers, fields as (ny, nx) arrays of a tensor-product quad mesh."""
    t, c, arrays = read_quads(path, names)
    xs, ys = np.unique(np.round(c[:, 0], 12)), np.unique(np.round(c[:, 1], 12))
    i = np.searchsorted(xs, np.round(c[:, 0], 12))
    j = np.searchsorted(ys, np.round(c[:, 1], 12))
    out = {}
    for n in names:
        a = arrays[n]
        f = np.empty((ys.size, xs.size) + a.shape[1:])
        f[j, i] = a
        out[n] = f
    return float(np.ravel(t)[0]), xs, ys, out


def first_crossing(x, q, level):
    k = np.nonzero(q >= level)[0]
    if k.size == 0 or k[0] == 0:
        return np.nan
    k = k[0]
    return x[k - 1] + (level - q[k - 1]) * (x[k] - x[k - 1]) / (q[k] - q[k - 1])


def liftoff(x, T_max, OH_max, HRR_max, T_air):
    return (first_crossing(x, OH_max, OH_LEVEL), first_crossing(x, T_max, T_air + DT_LEVEL),
            x[np.argmax(HRR_max)])


def reference(ref_dir):
    with open(os.path.join(ref_dir, "ignition.csv")) as f:
        header = f.readline()
    T_air = float(re.search(r"air at ([0-9.]+) K", header).group(1))
    fuel = re.search(r"fuel (\S+) at", header).group(1)
    ign = np.genfromtxt(os.path.join(ref_dir, "ignition.csv"), delimiter=",", names=True, skip_header=1)
    prof = np.genfromtxt(os.path.join(ref_dir, "profiles.csv"), delimiter=",", names=True, skip_header=1)
    return T_air, fuel, ign, prof


def align(y, Z, Z_st):
    """y where Z crosses Z_st (Z falls from the fuel side to the air side)."""
    k = np.nonzero((Z[:-1] >= Z_st) & (Z[1:] < Z_st))[0][0]
    return y[k] + (Z_st - Z[k]) * (y[k + 1] - y[k]) / (Z[k + 1] - Z[k])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference")
    ap.add_argument("--run", nargs=2, action="append", metavar=("LABEL", "RUN_DIR"), required=True)
    ap.add_argument("--out", default="mixing_layer")
    ap.add_argument("--history", action="store_true")
    ap.add_argument("--average", type=int, default=1, help="average the last N snapshots")
    a = ap.parse_args()

    T_air, fuel, ign, prof = reference(a.reference)
    gas = ct.Solution(MECH, PHASE)
    gas.TPX = 300, ct.one_atm, fuel
    Y_fuel = gas.Y.copy()
    gas.TPX = T_air, ct.one_atm, AIR
    Y_air = gas.Y.copy()
    gas.set_equivalence_ratio(1.0, fuel, AIR)
    Z_st = float(bilger(gas, gas.Y, Y_fuel, Y_air))
    ref_lift = liftoff(ign["x"], ign["T_max"], ign["Y_OH_max"], ign["HRR_max"], T_air)
    print(f"Z_st = {Z_st:.4f}")
    print(f"{'':28s} x_OH [mm]  x_T [mm]  x_HRR [mm]  T at 25 mm [K]")
    T25 = np.interp(0.025, ign["x"], ign["T_max"])
    print(f"{'reference':28s} {ref_lift[0] * 1e3:9.3f} {ref_lift[1] * 1e3:9.3f} {ref_lift[2] * 1e3:10.3f}  {T25:.1f}")

    names = ["T", "HRR", "U"] + ["Y_" + s for s in gas.species_names]
    stations = np.unique(prof["x"])
    fig_p, ax_p = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    fig_s, ax_s = plt.subplots(len(stations), 3, figsize=(12, 2.6 * len(stations)), squeeze=False)
    ax_p[0].plot(ign["x"] * 1e3, ign["T_max"], "k-", lw=2.5, alpha=0.4, label="boundary layer (Cantera)")
    ax_p[1].semilogy(ign["x"] * 1e3, ign["Y_OH_max"], "k-", lw=2.5, alpha=0.4)
    ax_p[2].semilogy(ign["x"] * 1e3, ign["HRR_max"], "k-", lw=2.5, alpha=0.4)
    for row, xs_ in enumerate(stations):
        r = prof[prof["x"] == xs_]
        y0 = align(r["y"], r["Z"], Z_st)
        yy = (r["y"] - y0) * 1e3
        ax_s[row, 0].plot(yy, r["T"], "k-", lw=2.5, alpha=0.4)
        for s, c in zip(SPECIES[:3], ["C0", "C1", "C2"]):
            ax_s[row, 1].plot(yy, r["Y_" + s], "-", color=c, lw=2.5, alpha=0.4)
        for s, c in zip(SPECIES[3:], ["C3", "C4", "C5"]):
            ax_s[row, 2].plot(yy, r["Y_" + s], "-", color=c, lw=2.5, alpha=0.4)
    for n_run, (label, run_dir) in enumerate(a.run):
        files = snapshots(run_dir)
        if a.history:
            hist = []
            for f in files:
                t, xs, ys, d = load(f, ["T", "HRR", "Y_OH"])
                hist.append((t,) + liftoff(xs, d["T"].max(axis=0), d["Y_OH"].max(axis=0), d["HRR"].max(axis=0),
                                           T_air))
            hist = np.array(hist)
            if n_run == 0:
                fig_h, ax_h = plt.subplots(figsize=(7, 4))
            for k, (mark, nm) in enumerate(zip("os^", ["x_OH", "x_T", "x_HRR"])):
                ax_h.plot(hist[:, 0] * 1e3, hist[:, k + 1] * 1e3, mark + "-", color=f"C{n_run}", ms=3,
                          label=f"{label}: {nm}")
                ax_h.axhline(ref_lift[k] * 1e3, color="k", ls=":", lw=0.8)
        acc = None
        for f in files[-a.average:]:
            t, xs, ys, d = load(f, names)
            acc = d if acc is None else {k: acc[k] + d[k] for k in d}
        d = {k: v / min(a.average, len(files)) for k, v in acc.items()}
        T_max, OH_max, HRR_max = d["T"].max(axis=0), d["Y_OH"].max(axis=0), d["HRR"].max(axis=0)
        lift = liftoff(xs, T_max, OH_max, HRR_max, T_air)
        err = [(l - r) * 1e3 for l, r in zip(lift, ref_lift)]
        print(f"{label:28s} {lift[0] * 1e3:9.3f} {lift[1] * 1e3:9.3f} {lift[2] * 1e3:10.3f}  "
              f"{np.interp(0.025, xs, T_max):.1f}   (t = {t * 1e3:.3f} ms; differences "
              f"{err[0]:+.3f} {err[1]:+.3f} {err[2]:+.3f} mm)")
        style = dict(color=f"C{n_run}", lw=1.2, label=label)
        ax_p[0].plot(xs * 1e3, T_max, **style)
        ax_p[1].semilogy(xs * 1e3, OH_max, **style)
        ax_p[2].semilogy(xs * 1e3, HRR_max, **style)
        Ys = np.stack([d["Y_" + s] for s in gas.species_names], axis=-1)
        for row, xs_ in enumerate(stations):
            i = np.argmin(np.abs(xs - xs_))
            Z = bilger(gas, Ys[:, i], Y_fuel, Y_air)
            yy = (ys - align(ys, Z, Z_st)) * 1e3
            ls = dict(lw=1, ls="--" if n_run else "-")
            ax_s[row, 0].plot(yy, d["T"][:, i], color="k", **ls)
            for s, c in zip(SPECIES[:3], ["C0", "C1", "C2"]):
                ax_s[row, 1].plot(yy, d["Y_" + s][:, i], color=c, **ls)
            for s, c in zip(SPECIES[3:], ["C3", "C4", "C5"]):
                ax_s[row, 2].plot(yy, d["Y_" + s][:, i], color=c, **ls)
            ref_T = prof["T"][prof["x"] == xs_]
            print(f"    x = {xs_ * 1e3:4.1f} mm: peak T {d['T'][:, i].max():7.1f} K (reference {ref_T.max():7.1f}), "
                  f"peak Y_OH {d['Y_OH'][:, i].max():.3e} (reference {prof['Y_OH'][prof['x'] == xs_].max():.3e})")
        if n_run == 0:
            fig_f, ax_f = plt.subplots(figsize=(10, 4))
            im = ax_f.pcolormesh(xs * 1e3, ys * 1e3, d["T"], cmap="inferno", shading="nearest")
            ax_f.contour(xs * 1e3, ys * 1e3, d["HRR"], levels=np.array([0.05, 0.2, 0.5]) * d["HRR"].max(),
                         colors="w", linewidths=0.5)
            ax_f.axvline(ref_lift[0] * 1e3, color="c", ls="--", lw=1, label="reference x_OH")
            ax_f.axvline(lift[0] * 1e3, color="w", ls=":", lw=1, label=f"{label}: x_OH")
            band = np.nonzero(np.abs(d["T"] - d["T"][:, :1]).max(axis=1) > 5)[0]
            ax_f.set_ylim(ys[band[0]] * 1e3 - 0.5, ys[band[-1]] * 1e3 + 0.5)
            ax_f.set_xlabel("x [mm]")
            ax_f.set_ylabel("y [mm]")
            ax_f.legend(loc="lower right", fontsize=8)
            fig_f.colorbar(im, ax=ax_f, label="T [K]")
            fig_f.tight_layout()
            fig_f.savefig(a.out + "_field.png", dpi=150)
    for ax, lab in zip(ax_p, ["peak T [K]", "peak Y_OH", "peak HRR [W/m^3]"]):
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    ax_p[1].set_ylim(1e-6, None)
    ax_p[2].set_ylim(1e6, None)
    ax_p[-1].set_xlabel("x [mm]")
    ax_p[0].legend(fontsize=8)
    fig_p.tight_layout()
    fig_p.savefig(a.out + "_peaks.png", dpi=150)
    for row, xs_ in enumerate(stations):
        ax_s[row, 0].set_ylabel(f"x = {xs_ * 1e3:g} mm\nT [K]")
        ax_s[row, 1].set_ylabel("Y")
        ax_s[row, 2].set_ylabel("Y")
        for ax in ax_s[row]:
            ax.set_xlim(-2.5, 2.5)
            ax.grid(alpha=0.3)
    for s, c in zip(SPECIES, ["C0", "C1", "C2", "C3", "C4", "C5"]):
        ax_s[0, 1 if s in SPECIES[:3] else 2].plot([], [], color=c, label=s)
    ax_s[0, 1].legend(fontsize=7)
    ax_s[0, 2].legend(fontsize=7)
    ax_s[0, 0].plot([], [], "k-", lw=2.5, alpha=0.4, label="reference")
    ax_s[0, 0].plot([], [], "k-", lw=1, label=a.run[0][0])
    ax_s[0, 0].legend(fontsize=7)
    for ax in ax_s[-1]:
        ax.set_xlabel("y - y(Z_st) [mm]")
    fig_s.tight_layout()
    fig_s.savefig(a.out + "_profiles.png", dpi=120)
    if a.history:
        ax_h.set_xlabel("t [ms]")
        ax_h.set_ylabel("lift-off length [mm]")
        ax_h.legend(fontsize=7, ncol=2)
        ax_h.grid(alpha=0.3)
        fig_h.tight_layout()
        fig_h.savefig(a.out + "_history.png", dpi=150)


if __name__ == "__main__":
    main()
