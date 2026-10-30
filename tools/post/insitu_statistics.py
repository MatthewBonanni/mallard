#!/usr/bin/env python3
"""In-situ spectra, two-point correlations and phase averages ([statistics.spectra],
[statistics.phase]): read them, recompute them from snapshots, compare and plot.

    insitu_statistics.py spectra INPUT.toml RESTART [RESTART ...] -o reference.csv
    insitu_statistics.py compare A.csv B.csv
    insitu_statistics.py phase INPUT.toml PROBE.csv -o phase.npz
    insitu_statistics.py plot SPECTRA.csv [SPECTRA.csv ...] [--pair U_X,U_X] [--position 3] -o plot.png

spectra: the quantities of the run's [statistics.spectra] table (same fields,
products, directions, t_start; time-weighted like the solver: each sample
weighs the time since the previous one, or since t_start) computed with numpy
FFTs from restart files of a generated `cartesian` mesh taken at the sampling
steps, written in the solver's CSV format. Only the restarts after t_start
enter, in the order given. Variables RHO, U_X, U_Y, U_Z, P and T (perfect gas,
R = p_ref / (rho_ref T_ref)).

compare: the largest difference of every (quantity, a, b, direction) between
two spectra CSV files, relative to the largest magnitude of that quantity.

phase: phase-averages the fields of [statistics.phase] at the probe point
from the probe CSV (a [[probes]] series sampled at the same steps), with the
solver's phase rule, for comparison with the in-situ PHASE_ fields of the
cell nearest to the point.

plot: one-sided spectra and correlations of one pair, at one position, along
each direction (and the shell spectrum if present) of one or more CSV files.

A spectra CSV has a comment line (step, t, samples, weight) and the columns
quantity,a,b,direction,position,<inhomogeneous axes>,index,abscissa,value with
quantity one of mean, spectrum (one-sided, sum over n = 0..N/2 equal to the
covariance <a'b'>; abscissa k = 2 pi n / L), correlation (<a'(x) b'(x + r)>,
r = m L / N) and shell (|k| in shells of width dk = 2 pi / L_max).
"""
import argparse
import csv
import os
import struct
import sys
import tomllib

import numpy as np

AXES = "xyz"


def read_restart(path):
    """(fields by name, t, step, attributes, arrays) of an uncompressed double-precision restart file, versions 2-6."""
    with open(path, "rb") as f:
        if not f.read(16).startswith(b"MALLARD-RESTART"):
            raise SystemExit(f"{path}: not a Mallard restart file")
        version, real_size, n_cells, n_vars, step, t = struct.unpack("<IIQQQd", f.read(40))
        if version < 2 or version > 6 or real_size != 8:
            raise SystemExit(f"{path}: expected a double-precision restart file of version 2-6")

        def name():
            (length,) = struct.unpack("<I", f.read(4))
            return f.read(length).decode()
        names = [name() for _ in range(n_vars)]
        attributes, arrays = {}, {}
        if version >= 3:
            (n_attr,) = struct.unpack("<Q", f.read(8))
            for _ in range(n_attr):
                key = name()
                attributes[key] = struct.unpack("<d", f.read(8))[0]
        if version >= 4:
            f.read(12)  # face count and width
        if version >= 5:
            _, chunk_cells = struct.unpack("<IQ", f.read(12))
            if chunk_cells > 0:
                raise SystemExit(f"{path}: compressed restart files are not supported (write with compression = \"none\")")
        if version >= 6:
            (n_arrays,) = struct.unpack("<Q", f.read(8))
            for _ in range(n_arrays):
                key = name()
                (n,) = struct.unpack("<Q", f.read(8))
                arrays[key] = np.fromfile(f, dtype="<f8", count=n)
        data = np.fromfile(f, dtype="<f8", count=n_cells * n_vars).reshape(n_vars, n_cells)
    return dict(zip(names, data)), t, step, attributes, arrays


def si(value, key):
    """A plain number of the input; quantities with units ("5 mm") are not converted here."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise SystemExit(f"{key} = {value!r}: this script reads plain numbers in SI units; write it without a unit")


def variables(cons, names, phys, dim):
    """Cell values of the named variables from the conservatives of a perfect gas."""
    rho = cons["RHO"]
    u = [cons["RHOU_" + AXES[d].upper()] / rho for d in range(dim)]
    gamma = si(phys.get("gamma", 1.4), "physics.gamma")
    p = (gamma - 1.0) * (cons["RHOE"] - 0.5 * rho * sum(v * v for v in u))
    R = si(phys.get("p_ref", 1.0), "physics.p_ref") / (si(phys.get("rho_ref", 1.0), "physics.rho_ref") *
                                                       si(phys.get("T_ref", 1.0), "physics.T_ref"))
    known = {"RHO": rho, "P": p, "T": p / (rho * R)}
    for d in range(dim):
        known["U_" + AXES[d].upper()] = u[d]
    return [known[n] for n in names]


def parse_pairs(table):
    fields, pairs = [], []

    def add(name):
        if name not in fields:
            fields.append(name)
        return fields.index(name)
    for name in table.get("fields", []):
        a = add(name)
        pairs.append((a, a))
    for product in table.get("products", []):
        a, b = product.split("*")
        pairs.append((add(a), add(b)))
    return fields, pairs


class Box:
    """A generated cartesian mesh: sizes, lengths, cell centers along each axis."""

    def __init__(self, cfg):
        mesh = cfg["mesh"]
        self.dim = 3 if "Nz" in mesh else 2
        self.N = [mesh.get(k, 100) for k in ("Nx", "Ny", "Nz")[:self.dim]]
        self.L = [si(mesh.get(k, 1.0), "mesh." + k) for k in ("Lx", "Ly", "Lz")[:self.dim]]
        beta = list(mesh.get("stretching", [0.0] * self.dim)) + [0.0] * 3
        self.centers = []
        for d in range(self.dim):
            n, L, b = self.N[d], self.L[d], beta[d]
            if b > 0:
                nodes = L / 2 * (1 + np.tanh(b * (2 * np.arange(n + 1) / n - 1)) / np.tanh(b))
                self.centers.append(0.5 * (nodes[1:] + nodes[:-1]))
            else:
                self.centers.append(L * (np.arange(n) + 0.5) / n)


def spectra_from_restarts(cfg, files):
    """Rows of the solver's spectra CSV computed from restart snapshots."""
    stats = cfg["statistics"]
    table = stats["spectra"]
    box = Box(cfg)
    dim = box.dim
    fields, pairs = parse_pairs(table)
    sides = {"left": "x", "right": "x", "bottom": "y", "top": "y", "back": "z", "front": "z"}
    periodic = [sides[p["zones"][0]] for p in cfg.get("periodic", []) if p["zones"][0] in sides]
    H = sorted(AXES.index(a) for a in table.get("directions", periodic))
    I = [d for d in range(dim) if d not in H]
    t_start = si(table.get("t_start", stats.get("t_start", 0.0)), "statistics.t_start")
    n_modes = int(np.prod([box.N[d] for d in H]))
    acc = None
    weight, t_last, n_samples = 0.0, t_start, 0
    step = 0
    for path in files:
        cons, t, step, _, _ = read_restart(path)
        if t <= t_start:
            continue
        vals = variables(cons, fields, cfg.get("physics", {}), dim)
        hats = [np.fft.fftn(v.reshape(box.N), axes=H) / n_modes for v in vals]
        sample = {"mean": [], "spec": {}, "shell": {}}
        zero = tuple(0 if d in H else slice(None) for d in range(dim))
        sample["mean"] = [np.real(h[zero]) for h in hats]
        for p, (a, b) in enumerate(pairs):
            S = hats[a] * np.conj(hats[b])
            for h in H:
                others = tuple(d for d in H if d != h)
                kept = [d for d in range(dim) if d not in others]
                sample["spec"][(p, h)] = np.moveaxis(S.sum(axis=others) if others else S, kept.index(h), -1)
            if not I:
                k2 = sum((2 * np.pi / box.L[d] * np.fft.fftfreq(box.N[d], 1.0 / box.N[d])).reshape(
                    [-1 if e == d else 1 for e in range(dim)]) ** 2 for d in range(dim))
                dk = min(2 * np.pi / box.L[d] for d in range(dim))
                shell = np.floor(np.sqrt(k2) / dk + 0.5).astype(int).ravel()
                n_shells = int(np.floor(np.sqrt(sum((2 * np.pi / box.L[d] * (box.N[d] // 2)) ** 2
                                                    for d in range(dim))) / dk + 0.5)) + 1
                sample["shell"][p] = np.bincount(shell, weights=np.real(S).ravel(), minlength=n_shells)
        w = t - t_last
        t_last = t
        weight += w
        n_samples += 1
        a_w = w / weight
        if acc is None:
            acc = sample
            continue
        acc["mean"] = [m + a_w * (s - m) for m, s in zip(acc["mean"], sample["mean"])]
        for key in acc["spec"]:
            acc["spec"][key] = acc["spec"][key] + a_w * (sample["spec"][key] - acc["spec"][key])
        for key in acc["shell"]:
            acc["shell"][key] = acc["shell"][key] + a_w * (sample["shell"][key] - acc["shell"][key])
    if acc is None:
        raise SystemExit("no snapshot after t_start")

    rows = []
    shape_I = [box.N[d] for d in I]
    n_pos = int(np.prod(shape_I)) if I else 1
    for pos in range(n_pos):
        idx = np.unravel_index(pos, shape_I) if I else ()
        where = [box.centers[d][i] for d, i in zip(I, idx)]
        for f, name in enumerate(fields):
            rows.append(("mean", name, "", "", pos, *where, "", "", float(acc["mean"][f][idx])))
        for p, (a, b) in enumerate(pairs):
            for h in H:
                S = np.array(acc["spec"][(p, h)][idx] if I else acc["spec"][(p, h)], dtype=complex)
                S[0] -= acc["mean"][a][idx] * acc["mean"][b][idx]
                n = box.N[h]
                for k in range(n // 2 + 1):
                    e = S[k].real + (S[n - k].real if 0 < k and n - k != k else 0.0)
                    rows.append(("spectrum", fields[a], fields[b], AXES[h], pos, *where, k,
                                 2 * np.pi * k / box.L[h], e))
                R = np.real(np.fft.fft(S))  # sum_k S(k) exp(-2 pi i k m / n)
                for m in range(n):
                    rows.append(("correlation", fields[a], fields[b], AXES[h], pos, *where, m,
                                 box.L[h] * m / n, R[m]))
    for p, (a, b) in enumerate(pairs):
        if p not in acc["shell"]:
            continue
        E = acc["shell"][p].copy()
        E[0] -= acc["mean"][a] * acc["mean"][b]
        dk = min(2 * np.pi / box.L[d] for d in range(dim))
        for s, e in enumerate(E):
            rows.append(("shell", fields[a], fields[b], "k", 0, s, s * dk, e))
    header = ["quantity", "a", "b", "direction", "position"] + [AXES[d] for d in I] + ["index", "abscissa", "value"]
    meta = f"# step = {step}, t = {t_last!r}, samples = {n_samples}, weight = {weight!r}"
    return header, rows, meta


def read_spectra(path):
    """Rows of a spectra CSV as a dict (quantity, a, b, direction) -> (positions, index, abscissa, value) arrays,
    and "units", the value's units (files of Mallard 0.9 on; empty for older files and recomputed ones)."""
    with open(path) as f:
        lines = [line for line in f if not line.startswith("#")]
    reader = csv.reader(lines)
    header = next(reader)
    col = {name: i for i, name in enumerate(header)}
    i_index, i_abscissa, i_value = col["index"], col["abscissa"], col["value"]
    out, units = {}, {}
    for row in reader:
        key = (row[0], row[1], row[2], row[3])
        out.setdefault(key, []).append((int(row[4]), int(row[i_index]) if row[i_index] else -1,
                                        float(row[i_abscissa]) if row[i_abscissa] else np.nan, float(row[i_value]),
                                        [float(v) for v in row[5:i_index]]))
        units[key] = row[col["units"]] if "units" in col else ""
    result = {}
    for key, values in out.items():
        result[key] = {"position": np.array([v[0] for v in values]), "index": np.array([v[1] for v in values]),
                       "abscissa": np.array([v[2] for v in values]), "value": np.array([v[3] for v in values]),
                       "coordinates": np.array([v[4] for v in values]), "units": units[key]}
    return header, result


def compare(a_path, b_path):
    _, a = read_spectra(a_path)
    _, b = read_spectra(b_path)
    worst = 0.0
    for key in sorted(a):
        if key not in b:
            print(f"{key}: missing from {b_path}")
            continue
        va, vb = a[key]["value"], b[key]["value"]
        if va.shape != vb.shape:
            print(f"{key}: {va.size} rows against {vb.size}")
            continue
        scale = max(np.abs(va).max(), np.abs(vb).max(), 1e-300)
        d = np.abs(va - vb).max() / scale
        worst = max(worst, d)
        print(f"{','.join(k for k in key if k):32s} rows {va.size:7d}  max |diff| / max |value| = {d:.2e}")
    print(f"worst: {worst:.2e}")
    return worst


def phase_bins(cfg, probe_csv):
    """Phase-binned means of the probe variables with the solver's phase rule (period or trigger)."""
    stats = cfg["statistics"]
    table = stats["phase"]
    bins = int(table["bins"])
    t_start = si(table.get("t_start", stats.get("t_start", 0.0)), "statistics.t_start")
    fields = table["fields"]
    data = np.genfromtxt(probe_csv, delimiter=",", names=True)
    # The solver samples only after steps; a run's first probe row is its initial state
    data = data[data["step"] > data["step"][0]]
    t = data["t"]
    weight = np.zeros(bins)
    mean = np.zeros((bins, len(fields)))
    t_last = t_start
    crossings = []
    v_prev = t_prev = None
    level = si(table.get("level", 0.0), "statistics.phase.level")
    for i in range(len(t)):
        use = t[i] > t_start
        if "probe" in table:
            v = data[table["variable"]][i]
            if v_prev is not None and v_prev < level <= v:
                crossings.append(t_prev + (level - v_prev) / (v - v_prev) * (t[i] - t_prev))
            v_prev, t_prev = v, t[i]
            if len(crossings) < 2:
                use = False
            else:
                phase = (t[i] - crossings[-1]) / (crossings[-1] - crossings[-2])
        else:
            x = (t[i] - si(table.get("t_reference", 0.0), "statistics.phase.t_reference")) / si(table["period"], "statistics.phase.period")
            phase = x - np.floor(x)
        if not use:
            if t[i] > t_start:
                t_last = t[i]
            continue
        b = min(int(np.floor(phase * bins)), bins - 1)
        w = t[i] - t_last
        t_last = t[i]
        weight[b] += w
        for j, name in enumerate(fields):
            mean[b, j] += w / weight[b] * (data[name][i] - mean[b, j])
    return fields, weight, mean, np.array(crossings)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("spectra")
    s.add_argument("input")
    s.add_argument("restart", nargs="+")
    s.add_argument("-o", "--output", required=True)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--tol", type=float, default=None, help="exit 1 if the worst relative difference exceeds this")
    ph = sub.add_parser("phase")
    ph.add_argument("input")
    ph.add_argument("probe")
    ph.add_argument("-o", "--output", required=True)
    pl = sub.add_parser("plot")
    pl.add_argument("csv", nargs="+")
    pl.add_argument("--pair", default=None, help="A,B (default: the first pair)")
    pl.add_argument("--position", type=int, default=0)
    pl.add_argument("--label", nargs="*")
    pl.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    if args.cmd == "spectra":
        with open(args.input, "rb") as f:
            cfg = tomllib.load(f)
        header, rows, meta = spectra_from_restarts(cfg, args.restart)
        with open(args.output, "w", newline="") as f:
            f.write(meta + "\n")
            w = csv.writer(f)
            w.writerow(header)
            for row in rows:
                w.writerow([repr(float(v)) if isinstance(v, (float, np.floating)) else v for v in row])
    elif args.cmd == "compare":
        worst = compare(args.a, args.b)
        if args.tol is not None and worst > args.tol:
            sys.exit(1)
    elif args.cmd == "phase":
        with open(args.input, "rb") as f:
            cfg = tomllib.load(f)
        fields, weight, mean, crossings = phase_bins(cfg, args.probe)
        np.savez(args.output, fields=np.array(fields), weight=weight, mean=mean, crossings=crossings)
        periods = np.diff(crossings)
        if periods.size:
            print(f"{crossings.size} crossings, period {periods.mean():.6g} +- {periods.std():.2g}")
        for b in range(len(weight)):
            print(f"bin {b:3d} weight {weight[b]:.6g} " + " ".join(f"{n}={v:.6g}" for n, v in zip(fields, mean[b])))
    else:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        data = [read_spectra(p)[1] for p in args.csv]
        labels = args.label or [os.path.basename(p) for p in args.csv]
        keys = [k for k in data[0] if k[0] == "spectrum"]
        pair = args.pair.split(",") if args.pair else [keys[0][1], keys[0][2]]
        dirs = sorted({k[3] for k in keys if [k[1], k[2]] == pair})
        has_shell = ("shell", pair[0], pair[1], "k") in data[0]
        fig, axes = plt.subplots(2, len(dirs) + has_shell, figsize=(4.5 * (len(dirs) + has_shell), 7), squeeze=False)
        for col, d in enumerate(dirs):
            for dat, lab in zip(data, labels):
                for row, q in enumerate(("spectrum", "correlation")):
                    r = dat[("spectrum" if q == "spectrum" else "correlation", pair[0], pair[1], d)]
                    sel = r["position"] == args.position
                    x, y = r["abscissa"][sel], r["value"][sel]
                    ax = axes[row, col]
                    if q == "spectrum":
                        ax.loglog(x[1:], np.abs(y[1:]), label=lab)
                        ax.set_xlabel(f"k_{d}")
                        ax.set_ylabel(f"E_{{{pair[0]},{pair[1]}}}")
                    else:
                        ax.plot(x[: len(x) // 2 + 1], y[: len(x) // 2 + 1] / y[0], label=lab)
                        ax.set_xlabel(f"r_{d}")
                        ax.set_ylabel("R / R(0)")
        if has_shell:
            for dat, lab in zip(data, labels):
                r = dat[("shell", pair[0], pair[1], "k")]
                axes[0, -1].loglog(r["abscissa"][1:], np.abs(r["value"][1:]), label=lab)
            axes[0, -1].set_xlabel("|k|")
            axes[0, -1].set_ylabel("shell spectrum")
            axes[1, -1].axis("off")
        axes[0, 0].legend()
        fig.tight_layout()
        fig.savefig(args.output, dpi=130)


if __name__ == "__main__":
    main()
