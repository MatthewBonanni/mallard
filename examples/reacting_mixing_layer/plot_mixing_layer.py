"""Laminar reacting mixing layer runs (V11) against the boundary-layer reference.

    python examples/reacting_mixing_layer/plot_mixing_layer.py [--run NAME RUN_DIR ...] [--average N]
        [--history] [--reference DIR] [--results DIR] [--figure]

The reference (DIR, default reference/) holds ignition.csv and profiles.csv
from mixing_layer_reference.py. Each run's last snapshot (solut/field_*, or
the mean of the last N with --average) is reduced column by column to the
peak temperature, peak Y_OH and peak heat release rate across the layer
against x, written to results/NAME.csv, and to profiles across the layer at
the reference's stations, written to results/NAME_profiles.csv, aligned at
the stoichiometric mixture fraction (Bilger's). Every run in results/ is
compared with the reference through the lift-off (ignition) lengths

  x_OH   where the peak Y_OH first reaches 2e-3
  x_T    where the peak temperature first exceeds the air's by 250 K
  x_HRR  where the peak heat release rate is largest

and the peaks at the stations. --history prints the lift-off lengths of
every snapshot of each --run (whether it is steady). --figure draws
docs/images/guide/mixing_layer_{peaks,profiles}-{light,dark}.svg from the
reference and results/.
"""
import argparse
import glob
import os
import re
import sys

import cantera as ct
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../tools"))
sys.path.insert(0, os.path.join(HERE, "../../tools/docs"))
from mallard.vtu import read_quads, series  # noqa: E402
from mixing_layer_reference import AIR, MECH, PHASE, bilger  # noqa: E402

OH_LEVEL, DT_LEVEL = 2e-3, 250.0
SPECIES = ["H2", "O2", "H2O", "OH"]
RUNS = ["coarse", "base", "fine"]
TEX = {"H2": "H$_2$", "O2": "O$_2$", "H2O": "H$_2$O", "OH": "OH"}


def first_crossing(x, q, level):
    k = np.nonzero(q >= level)[0]
    if k.size == 0 or k[0] == 0:
        return np.nan
    k = k[0]
    return x[k - 1] + (level - q[k - 1]) * (x[k] - x[k - 1]) / (q[k] - q[k - 1])


def liftoff(x, T_max, OH_max, HRR_max, T_air):
    return (first_crossing(x, OH_max, OH_LEVEL), first_crossing(x, T_max, T_air + DT_LEVEL),
            x[np.argmax(HRR_max)])


def read_csv(path):
    with open(path) as f:
        header = f.readline()
    return header, np.genfromtxt(path, delimiter=",", names=True, skip_header=1)


class Streams:
    def __init__(self, ref_dir):
        header, self.ign = read_csv(os.path.join(ref_dir, "ignition.csv"))
        _, self.prof = read_csv(os.path.join(ref_dir, "profiles.csv"))
        self.T_air = float(re.search(r"air at ([0-9.]+) K", header).group(1))
        self.t_fuel = float(re.search(r"at ([0-9.]+) K and", header).group(1))
        self.fuel = re.search(r"fuel (\S+) at", header).group(1)
        self.gas = ct.Solution(MECH, PHASE)
        self.gas.TPX = self.t_fuel, ct.one_atm, self.fuel
        self.Y_fuel = self.gas.Y.copy()
        self.gas.TPX = self.T_air, ct.one_atm, AIR
        self.Y_air = self.gas.Y.copy()
        self.gas.set_equivalence_ratio(1.0, self.fuel, AIR)
        self.Z_st = float(bilger(self.gas, self.gas.Y, self.Y_fuel, self.Y_air))
        self.stations = np.unique(self.prof["x"])
        self.lift = liftoff(self.ign["x"], self.ign["T_max"], self.ign["Y_OH_max"], self.ign["HRR_max"], self.T_air)

    def align(self, y, Z):
        """y where Z crosses Z_st (Z falls from the fuel side to the air side)."""
        k = np.nonzero((Z[:-1] >= self.Z_st) & (Z[1:] < self.Z_st))[0][0]
        return y[k] + (self.Z_st - Z[k]) * (y[k + 1] - y[k]) / (Z[k + 1] - Z[k])


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


def reduce_run(st, name, run_dir, average, history, results):
    files = series(os.path.join(run_dir, "solut"), "field_[0-9]*")
    if not files:
        raise SystemExit(f"{run_dir}: no solut/field_*")
    if history:
        print(f"{name}: lift-off lengths of each snapshot [mm]")
        for f in files:
            t, xs, ys, d = load(f, ["T", "HRR", "Y_OH"])
            lift = liftoff(xs, d["T"].max(axis=0), d["Y_OH"].max(axis=0), d["HRR"].max(axis=0), st.T_air)
            print(f"  t = {t * 1e3:7.4f} ms: x_OH {lift[0] * 1e3:7.3f}  x_T {lift[1] * 1e3:7.3f}  "
                  f"x_HRR {lift[2] * 1e3:7.3f}")
    names = ["T", "HRR"] + ["Y_" + s for s in st.gas.species_names]
    used = files[-average:]
    peaks, profiles, grids = 0.0, {}, {}
    for f in used:
        t, xs, ys, d = load(f, names)
        peaks = peaks + np.column_stack([d["T"].max(axis=0), d["Y_OH"].max(axis=0), d["HRR"].max(axis=0)])
        Ys = np.stack([d["Y_" + s] for s in st.gas.species_names], axis=-1)
        for k, xs_ in enumerate(st.stations):
            i = int(np.argmin(np.abs(xs - xs_)))
            yy = ys - st.align(ys, bilger(st.gas, Ys[:, i], st.Y_fuel, st.Y_air))
            grid = grids.setdefault(k, yy[np.abs(yy) < 3e-3])
            q = np.column_stack([np.interp(grid, yy, d[n][:, i]) for n in ["T", "HRR"] + ["Y_" + s for s in SPECIES]])
            profiles[k] = profiles.get(k, 0.0) + q
    peaks /= len(used)
    os.makedirs(results, exist_ok=True)
    head = (f"Laminar mixing layer (V11), Mallard run {name}: {xs.size} x {ys.size} cells, "
            f"dx = {(xs[1] - xs[0]) * 1e6:.4g} um, smallest dy = {np.diff(ys).min() * 1e6:.4g} um; "
            f"t = {t:.6g} s, mean over the last {len(used)} snapshot(s)\n")
    np.savetxt(os.path.join(results, f"{name}.csv"), np.column_stack([xs, peaks]),
               delimiter=",", header=head + "x,T_max,Y_OH_max,HRR_max", comments="# ", fmt="%.7g")
    rows = [np.column_stack([np.full(grids[k].size, xs_), grids[k], profiles[k] / len(used)])
            for k, xs_ in enumerate(st.stations)]
    np.savetxt(os.path.join(results, f"{name}_profiles.csv"), np.vstack(rows), delimiter=",",
               header=head + "x,y_st,T,HRR," + ",".join("Y_" + s for s in SPECIES), comments="# ", fmt="%.7g")


def results_of(st, results):
    out = {}
    for name in RUNS + sorted(set(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(results, "*.csv"))
                                  if not p.endswith("_profiles.csv")) - set(RUNS)):
        path = os.path.join(results, f"{name}.csv")
        if os.path.exists(path):
            header, peaks = read_csv(path)
            out[name] = (header, peaks, read_csv(os.path.join(results, f"{name}_profiles.csv"))[1])
    return out


def compare(st, runs):
    ign = st.ign
    print(f"Z_st = {st.Z_st:.4f}")
    print(f"{'':12s} x_OH [mm]  x_T [mm]  x_HRR [mm]")
    print(f"{'reference':12s} {st.lift[0] * 1e3:9.3f} {st.lift[1] * 1e3:9.3f} {st.lift[2] * 1e3:10.3f}")
    for name, (header, p, prof) in runs.items():
        lift = liftoff(p["x"], p["T_max"], p["Y_OH_max"], p["HRR_max"], st.T_air)
        err = [(a - b) / b for a, b in zip(lift, st.lift)]
        print(f"{name:12s} {lift[0] * 1e3:9.3f} {lift[1] * 1e3:9.3f} {lift[2] * 1e3:10.3f}   "
              f"({100 * err[0]:+.1f}% {100 * err[1]:+.1f}% {100 * err[2]:+.1f}%)   {header[2:].strip()}")
        for xs_ in st.stations:
            i = int(np.argmin(np.abs(p["x"] - xs_)))
            ref = st.prof[st.prof["x"] == xs_]
            print(f"    x = {xs_ * 1e3:4.1f} mm: peak T {p['T_max'][i]:7.1f} K (reference {ref['T'].max():7.1f}, "
                  f"{p['T_max'][i] - ref['T'].max():+6.1f}), peak Y_OH {p['Y_OH_max'][i]:.3e} "
                  f"(reference {ref['Y_OH'].max():.3e}), peak HRR {p['HRR_max'][i]:.3e} "
                  f"(reference {np.interp(xs_, ign['x'], ign['HRR_max']):.3e})")


def figures(st, runs):
    import figstyle

    def colors(P):
        return {"coarse": P["blue"], "base": P["flame"], "fine": P["teal"]}

    def peaks(P):
        fig, axes = figstyle.plt.subplots(3, 1, figsize=(7.5, 8.0), sharex=True, gridspec_kw=dict(hspace=0.12))
        ign = st.ign
        for ax, key in zip(axes, ("T_max", "Y_OH_max", "HRR_max")):
            ax.plot(ign["x"] * 1e3, ign[key], color=P["muted"], lw=4.0, alpha=0.5,
                    label="boundary layer (Cantera)")
            for name in RUNS:
                if name in runs:
                    p = runs[name][1]
                    ax.plot(p["x"] * 1e3, p[key], color=colors(P)[name], lw=1.4, label=f"Mallard, {name}")
            ax.axvline(st.lift[0] * 1e3, color=P["muted"], lw=0.8, ls=":")
        axes[1].set_yscale("log")
        axes[2].set_yscale("log")
        axes[1].set_ylim(1e-6, 3e-2)
        axes[2].set_ylim(1e6, 3e10)
        axes[0].set_ylabel("peak T [K]")
        axes[1].set_ylabel(r"peak $Y_\mathrm{OH}$")
        axes[2].set_ylabel(r"peak HRR [W/m$^3$]")
        axes[2].set_xlabel("x [mm]")
        axes[2].set_xlim(0, ign["x"].max() * 1e3)
        axes[0].legend(loc="upper left")
        return fig

    def profiles(P):
        show = [x for x in st.stations if x >= 0.0149]
        fig, axes = figstyle.plt.subplots(len(show), 2, figsize=(10.0, 2.5 * len(show)), sharex=True,
                                          squeeze=False, gridspec_kw=dict(hspace=0.15, wspace=0.28))
        sc = {"H2": P["blue"], "O2": P["green"], "H2O": P["flame"], "OH": P["plum"]}
        base = "fine" if "fine" in runs else next(iter(runs))
        prof = runs[base][2]
        for row, xs_ in enumerate(show):
            r = st.prof[st.prof["x"] == xs_]
            yy = (r["y"] - st.align(r["y"], r["Z"])) * 1e3
            m = prof[np.abs(prof["x"] - xs_) < 1e-4]
            a, b = axes[row]
            a.plot(yy, r["T"], color=P["muted"], lw=4.0, alpha=0.5)
            a.plot(m["y_st"] * 1e3, m["T"], color=P["fg"], lw=1.3)
            for s in SPECIES:
                scale = 10.0 if s == "OH" else 1.0
                b.plot(yy, scale * r["Y_" + s], color=sc[s], lw=4.0, alpha=0.35)
                b.plot(m["y_st"] * 1e3, scale * m["Y_" + s], color=sc[s], lw=1.3,
                       label=TEX[s] + (" (x10)" if scale != 1.0 else ""))
            a.set_ylabel("T [K]")
            b.set_ylabel("Y")
            figstyle.label(a, 0.03, 0.88, f"x = {xs_ * 1e3:g} mm", P, transform=a.transAxes, ha="left", size=12)
        for ax in axes[-1]:
            ax.set_xlabel(r"y - y(Z$_\mathrm{st}$) [mm]")
            ax.set_xlim(-2.0, 2.0)
        axes[0, 0].plot([], [], color=P["muted"], lw=4.0, alpha=0.5, label="boundary layer (Cantera)")
        axes[0, 0].plot([], [], color=P["fg"], lw=1.3, label=f"Mallard, {base}")
        axes[0, 0].legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
        axes[0, 1].legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=4)
        return fig

    figstyle.save("mixing_layer_peaks", peaks)
    figstyle.save("mixing_layer_profiles", profiles)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs=2, action="append", default=[], metavar=("NAME", "RUN_DIR"))
    ap.add_argument("--average", type=int, default=1, help="average the last N snapshots")
    ap.add_argument("--history", action="store_true")
    ap.add_argument("--reference", default=os.path.join(HERE, "reference"))
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--figure", action="store_true")
    a = ap.parse_args()
    st = Streams(a.reference)
    for name, run_dir in a.run:
        reduce_run(st, name, run_dir, a.average, a.history, a.results)
    runs = results_of(st, a.results)
    compare(st, runs)
    if a.figure:
        figures(st, runs)


if __name__ == "__main__":
    main()
