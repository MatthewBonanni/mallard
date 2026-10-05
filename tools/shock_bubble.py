#!/usr/bin/env python3
"""Shock / light-bubble interaction (examples/shock_bubble_3d): frames and comparison with Haas & Sturtevant.

    shock_bubble.py init INPUT OUT.restart --d D --xb XB [--gap 0.1] [--delta-cells 3] [--ms 1.25] [--air 0.28]
    shock_bubble.py frame INPUT RESTART OUT.npz
    shock_bubble.py frames INPUT RESTART_DIR FRAME_DIR [--keep-last] [--watch SECONDS]
    shock_bubble.py compare INPUT probe_axis.csv FRAME_DIR [--png xt.png] [--csv features.csv]

The run is a quarter of a square shock tube (symmetry planes y = 0 and z = 0)
on a generated cartesian box, with the bubble centered at (XB, 0, 0). `init`
writes its initial state (needs Cantera): air (N2/O2 0.767/0.233 by mass)
at T0, p0, a bubble of diameter D of helium with a mass fraction AIR of air,
at rest at T0, p0, with a tanh profile DELTA_CELLS cells thick, and a shock
of Mach MS (Rankine-Hugoniot of the thermally perfect air) GAP diameters
ahead of it moving to +x, with the post-shock air behind it; cell averages
over 2 x 2 x 2 points. It prints the post-shock state, which the input's
`upt` inflow on the left must match, and writes the geometry to bubble.json
next to INPUT for the other commands.

`frame` reduces a restart file to what the renders need: the He mass
fraction and the vorticity magnitude in the box around the helium
(Y_HE > 0.002, float16; vorticity in units of u_p / D, u_p the inflow
velocity behind the shock), the density gradient magnitude (numerical
schlieren, in units of the ambient density over D) and pressure (kPa) on the
symmetry plane z = 0, and the vortex ring's
core (largest vorticity on that plane, off the axis). `frames` does this for
every restart file in RESTART_DIR as it appears (deleting each, except the
newest with --keep-last), e.g. next to a running job with --watch.

`compare` turns the axis probe (Y_HE, P along y = z = 0) and the frames into
the x-t diagram of the experiment, scaled to its bubble (4.5 cm): upstream
and downstream interfaces (Y_HE = 0.5 crossings on the axis), refracted and
transmitted shocks (largest pressure gradients inside and behind the
helium), the incident shock (in the air off the bubble's shadow, y = z =
LY), the vortex ring and jet, with velocities fitted over the windows of
Haas & Sturtevant (1987, J. Fluid Mech. 181, 41-76, table 2, sphere,
Ms = 1.25). Velocities need no scaling (same gases and shock strength);
times and positions scale with the diameter.
"""
import argparse
import csv
import json
import glob
import os
import re
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detonation_window import Case, molecular_weights, open_restart, GAS_CONSTANT  # noqa: E402

D_EXPERIMENT = 0.045
# Haas & Sturtevant (1987), table 2, helium sphere, Ms = 1.25: velocities in m/s (10% error unless noted),
# times in ms from the shock's arrival at the bubble
HS_SPHERE = {"V_s": 420, "V_R": 960, "V_T": 365, "V_ui": 190, "V_uf": 125, "V_di": 145, "V_df": 165, "V_j": 335,
             "V_v": 165, "t_j": 0.10, "t_v": 0.30}


def geometry(case):
    """Bubble center XB, diameter D, shock start XS and shock speed W, from the bubble.json of `init`."""
    with open(case.path("bubble.json")) as f:
        b = json.load(f)
    return b["xb"], b["d"], b["xs"], b["w"]


def init(case, out, d, xb, gap, delta_cells, ms, air, T0, p0):
    """Initial state: the bubble, the incident shock GAP * D upstream of it, and the post-shock air behind."""
    import cantera as ct
    gas = ct.Solution(case.mechanism)
    air_y = {"N2": 0.767, "O2": 0.233}
    gas.TPY = T0, p0, air_y
    rho1, h1, c1 = gas.density, gas.enthalpy_mass, gas.sound_speed
    w = ms * c1

    def jump(r2):
        u2 = w * rho1 / r2
        p2 = p0 + rho1 * w ** 2 - r2 * u2 ** 2
        gas.TPY = T0, p0, air_y
        gas.TPY = p2 / (r2 * ct.gas_constant / gas.mean_molecular_weight), p2, air_y
        return gas.enthalpy_mass + 0.5 * u2 ** 2 - (h1 + 0.5 * w ** 2), p2, u2

    lo, hi = 1.01 * rho1, 3 * rho1
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if jump(lo)[0] * jump(mid)[0] <= 0:
            hi = mid
        else:
            lo = mid
    r2 = 0.5 * (lo + hi)
    _, p2, u2 = jump(r2)
    T2 = gas.T
    up = w - u2
    nx, ny, nz = case.shape
    dx = case.dx
    xs = round((xb - 0.5 * d - gap * d) / dx) * dx  # on a cell face
    delta = delta_cells * dx
    names = [s for s in gas.species_names]
    # Species internal energies at the two temperatures, J/kg
    e = {}
    for T in (T0, T2):
        gas.TP = T, p0
        e[T] = gas.standard_int_energies_RT * ct.gas_constant * T / gas.molecular_weights
    wts = gas.molecular_weights
    k_he = names.index("HE")
    y_air = np.array([air_y.get(s, 0.0) for s in names])
    n_cells = nx * ny * nz
    var_names = ["RHO", "RHOU_X", "RHOU_Y", "RHOU_Z", "RHOE"] + ["RHOY_" + s for s in names] + ["T_SEED"]
    sub = (np.arange(2) + 0.5) / 2
    with open(out, "wb") as f:
        f.write(b"MALLARD-RESTART\0")
        f.write(struct.pack("<IIQQQd", 2, 8, n_cells, len(var_names), 0, 0.0))
        for name in var_names:
            f.write(struct.pack("<I", len(name)) + name.encode())
        y = (np.arange(ny)[:, None, None, None] + sub[None, :, None, None]) * dx
        z = (np.arange(nz)[None, None, :, None] + sub[None, None, None, :]) * dx
        cols = {v: np.empty((nx, ny, nz)) for v in var_names}
        for i in range(nx):
            acc = {v: np.zeros((ny, nz)) for v in var_names}
            for sx in sub:
                x = (i + sx) * dx
                r = np.sqrt((x - xb) ** 2 + y ** 2 + z ** 2)  # (ny, 2, nz, 2)
                y_he = (1 - air) * 0.5 * (1 - np.tanh((r - 0.5 * d) / delta))
                Y = y_he[..., None] * np.eye(len(names))[k_he] + (1 - y_he)[..., None] * y_air
                behind = x < xs
                T, p, u = (T2, p2, up) if behind else (T0, p0, 0.0)
                rho = p / (ct.gas_constant * T * (Y / wts).sum(-1))
                e_mix = (Y * e[T]).sum(-1)
                parts = {"RHO": rho, "RHOU_X": rho * u, "RHOU_Y": 0 * rho, "RHOU_Z": 0 * rho,
                         "RHOE": rho * (e_mix + 0.5 * u * u), "T_SEED": T + 0 * rho}
                for k, s in enumerate(names):
                    parts["RHOY_" + s] = rho * Y[..., k]
                for v in var_names:
                    acc[v] += parts[v].mean(axis=(1, 3)) / 2
            for v in var_names:
                cols[v][i] = acc[v]
        for v in var_names:
            cols[v].astype("<f8").tofile(f)
    with open(case.path("bubble.json"), "w") as f:
        json.dump({"xb": xb, "d": d, "xs": xs, "w": w, "ms": ms, "u_p": up, "p2": p2, "T2": T2, "rho2": r2,
                   "T0": T0, "p0": p0, "air_in_bubble": air, "delta": delta}, f, indent=1)
    print(f"wrote {out}: {nx} x {ny} x {nz} cells; shock at x = {xs:.6g} (W = {w:.2f} m/s); behind it "
          f"u = {up:.4f} m/s, p = {p2:.2f} Pa, T = {T2:.4f} K (the left boundary's upt state)")


def fields(case, restart):
    h, data = open_restart(restart, case.shape)
    n = h["names"]
    rho = np.asarray(data[n.index("RHO")])
    u = [np.asarray(data[n.index("RHOU_" + c)]) / rho for c in "XYZ"]
    y_he = np.asarray(data[n.index("RHOY_HE")]) / rho
    weights = molecular_weights(case.mechanism)
    rho_over_w = sum(np.asarray(data[n.index(v)]) / weights[v[5:]] for v in n if v.startswith("RHOY_"))
    p = rho_over_w * GAS_CONSTANT * np.asarray(data[n.index("T_SEED")])
    return h, rho, u, y_he, p


def vorticity(u, dx):
    """|curl u| by central differences (one-sided at the box faces)."""
    dux = [np.gradient(c, dx, axis=a) for c in u for a in range(3)]  # du_i/dx_j at 3 i + j
    wx = dux[7] - dux[5]
    wy = dux[2] - dux[6]
    wz = dux[3] - dux[1]
    return np.sqrt(wx * wx + wy * wy + wz * wz)


def frame(case, restart, out):
    h, rho, u, y_he, p = fields(case, restart)
    dx = case.dx
    w = vorticity(u, dx)
    mask = y_he > 0.002
    if mask.any():
        idx = [np.nonzero(mask.any(axis=tuple(a for a in range(3) if a != k)))[0] for k in range(3)]
        lo = [max(0, int(i.min()) - 4) for i in idx]
        hi = [min(case.shape[k], int(idx[k].max()) + 5) for k in range(3)]
    else:
        lo, hi = [0, 0, 0], list(case.shape)
    box = tuple(slice(lo[k], hi[k]) for k in range(3))
    grad = np.gradient(rho[:, :, 0], dx)
    schlieren = np.hypot(*grad)
    plane_w = w[:, :, 0]
    j0 = 3  # skip the axis
    i, j = np.unravel_index(np.argmax(plane_w[:, j0:]), plane_w[:, j0:].shape)
    _, d, _, _ = geometry(case)
    u_ref = max(abs(float(b["u"][0])) for b in case.toml["boundaries"] if b.get("type") == "upt")
    rho_ref = float(rho[-1, -1, -1])
    np.savez(out, t=h["t"], step=h["step"], dx=dx, lo=np.array(lo), d=d, u_ref=u_ref, rho_ref=rho_ref,
             y_he=y_he[box].astype(np.float16), vort=(w[box] * d / u_ref).astype(np.float16),
             schlieren=(schlieren * d / rho_ref).astype(np.float16),
             p_plane=(p[:, :, 0] / 1000).astype(np.float16), y_he_plane=y_he[:, :, 0].astype(np.float16),
             ring=np.array([(i + 0.5) * dx, (j + j0 + 0.5) * dx]), vort_max=w.max())
    print(f"{out}: t = {h['t']:.6e}, helium box {lo}..{hi}", flush=True)


def frames(case, restart_dir, frame_dir, keep_last, watch):
    os.makedirs(frame_dir, exist_ok=True)
    done = set()
    while True:
        files = sorted(glob.glob(os.path.join(restart_dir, "*.restart")))
        ready = files[:-1] if watch else files
        for f in ready:
            if f in done:
                continue
            step = int(re.search(r"_(\d+)\.restart$", f).group(1))
            frame(case, f, os.path.join(frame_dir, f"frame_{step:09d}.npz"))
            done.add(f)
            if not (keep_last and f == files[-1]):
                os.remove(f)
        if not watch:
            return
        time.sleep(watch)


def read_probe(path):
    """{t: (x, Y_HE, P)} of the axis probe, rows grouped by time step."""
    rows = {}
    with open(path) as f:
        r = csv.reader(f)
        header = next(r)
        ix, iy, ip = header.index("x"), header.index("Y_HE"), header.index("P")
        for row in r:
            if row[0] == "step":
                continue
            rows.setdefault(float(row[1]), []).append((float(row[ix]), float(row[iy]), float(row[ip])))
    out = {}
    for t, v in sorted(rows.items()):
        a = np.array(sorted(v))
        out[t] = a
    return out


def crossings(x, y, level):
    s = np.nonzero(np.diff(np.sign(y - level)))[0]
    return x[s] + (level - y[s]) * (x[s + 1] - x[s]) / (y[s + 1] - y[s])


def fit(t, x, window):
    sel = (t >= window[0]) & (t <= window[1]) & np.isfinite(x)
    if sel.sum() < 3:
        return np.nan
    return np.polyfit(t[sel], x[sel], 1)[0]


def compare(case, probe_path, frame_dir, png, out_csv):
    xb, d, xs, ws = geometry(case)
    scale = D_EXPERIMENT / d
    probe = read_probe(probe_path)
    t_hit = (xb - d / 2 - xs) / ws
    ts, ui, di, shock_in, shock_out = [], [], [], [], []
    for t, a in probe.items():
        x, y, p = a[:, 0], a[:, 1], a[:, 2]
        c = crossings(x, y, 0.5 * y.max()) if y.max() > 0.1 else np.array([])
        ts.append((t - t_hit) * scale)
        ui.append((c.min() - xb) * scale if c.size else np.nan)
        di.append((c.max() - xb) * scale if c.size else np.nan)
        dp = np.gradient(p, x)
        helium = y > 0.25
        shock_in.append((x[helium][np.argmax(np.abs(dp[helium]))] - xb) * scale if helium.any() else np.nan)
        behind = x > (c.max() if c.size else xb + d / 2)
        shock_out.append((x[behind][np.argmax(np.abs(dp[behind]))] - xb) * scale if behind.any() else np.nan)
    ts, ui, di = np.array(ts), np.array(ui), np.array(di)
    shock_in, shock_out = np.array(shock_in), np.array(shock_out)
    ring_t, ring_x = [], []
    for f in sorted(glob.glob(os.path.join(frame_dir, "frame_*.npz"))):
        with np.load(f) as z:
            ring_t.append((float(z["t"]) - t_hit) * scale)
            ring_x.append((float(z["ring"][0]) - xb) * scale)
    ring_t, ring_x = np.array(ring_t), np.array(ring_x)
    ms = 1e-3
    # Windows (experiment time, s): early interface motion before the jet, late motion, ring after it forms
    result = {
        "V_ui": fit(ts, ui, (0.01 * ms, 0.06 * ms)),
        "V_uf": fit(ts, ui, (0.15 * ms, 0.30 * ms)),
        "V_di": fit(ts, di, (0.03 * ms, 0.10 * ms)),
        "V_df": fit(ts, di, (0.15 * ms, 0.40 * ms)),
        "V_R": fit(ts, shock_in, (0.0, 0.03 * ms)),
        "V_T": fit(ts, shock_out, (0.08 * ms, 0.15 * ms)),
        "V_v": fit(ring_t, ring_x, (0.40 * ms, ring_t.max() if ring_t.size else 0)),
    }
    # The jet: the upstream interface on the axis overtakes the downstream one
    pierce = np.nonzero(np.isnan(ui) & (ts > 0.02 * ms))[0]
    result["t_j"] = ts[pierce[0]] / ms if pierce.size else np.nan
    for k, v in result.items():
        ref = HS_SPHERE.get(k)
        print(f"{k:5s} Mallard {v:8.1f}   Haas & Sturtevant {ref}")
    if out_csv:
        with open(out_csv, "w") as f:
            f.write("feature,mallard,haas_sturtevant\n")
            for k, v in result.items():
                f.write(f"{k},{v:.6g},{HS_SPHERE.get(k, '')}\n")
    if png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(ts / ms, ui * 1e3, label="upstream interface / jet (axis)")
        ax.plot(ts / ms, di * 1e3, label="downstream interface (axis)")
        if ring_t.size:
            ax.plot(ring_t / ms, ring_x * 1e3, "o", ms=3, label="vortex ring core")
        for key, x0, t0, label in (("V_ui", -22.5, 0.0, "H&S V_ui"), ("V_v", None, HS_SPHERE["t_v"], "H&S V_v")):
            pass
        ax.set_xlabel("t (ms, experiment scale)")
        ax.set_ylabel("x - x_bubble (mm, experiment scale)")
        ax.legend()
        fig.savefig(png, dpi=150)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("frame")
    p.add_argument("input")
    p.add_argument("restart")
    p.add_argument("out")
    p = sub.add_parser("frames")
    p.add_argument("input")
    p.add_argument("restart_dir")
    p.add_argument("frame_dir")
    p.add_argument("--keep-last", action="store_true")
    p.add_argument("--watch", type=float, default=0)
    p = sub.add_parser("compare")
    p.add_argument("input")
    p.add_argument("probe")
    p.add_argument("frame_dir")
    p.add_argument("--png")
    p.add_argument("--csv")
    p = sub.add_parser("init")
    p.add_argument("input")
    p.add_argument("out")
    p.add_argument("--d", type=float, required=True, help="bubble diameter [m]")
    p.add_argument("--xb", type=float, required=True, help="bubble center x [m] (center on y = z = 0)")
    p.add_argument("--gap", type=float, default=0.1, help="shock start ahead of the bubble, in diameters")
    p.add_argument("--delta-cells", type=float, default=3.0, help="interface tanh thickness in cells")
    p.add_argument("--ms", type=float, default=1.25)
    p.add_argument("--air", type=float, default=0.28, help="mass fraction of air in the bubble")
    p.add_argument("--T0", type=float, default=295.0)
    p.add_argument("--p0", type=float, default=101325.0)
    args = ap.parse_args()
    case = Case(args.input)
    if args.cmd == "init":
        init(case, args.out, args.d, args.xb, args.gap, args.delta_cells, args.ms, args.air, args.T0, args.p0)
    elif args.cmd == "frame":
        frame(case, args.restart, args.out)
    elif args.cmd == "frames":
        frames(case, args.restart_dir, args.frame_dir, args.keep_last, args.watch)
    else:
        compare(case, args.probe, args.frame_dir, args.png, args.csv)


if __name__ == "__main__":
    main()
