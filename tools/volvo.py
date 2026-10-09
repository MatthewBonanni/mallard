#!/usr/bin/env python3
"""Analysis of the Volvo bluff-body LES (examples/volvo_bluff_body).

    volvo.py probes [--x0 -0.04 --x1 0.36 --dx 2e-3 --dy 2e-3 --z 0.04 --interval 1800 --vars U_X,U_Y]
        print [[probes]] lines spanning a mid-span plane (append them to the input)
    volvo.py profiles STATS.(p)vtu --u-bulk U [--d 0.04] --out profiles.npz
        time and span averages of [statistics] at the stations, the centerline, the recirculation length
    volvo.py plot --exp DIR --out PNG [--kind cold|reacting] RUN.npz[:label] ...
        mean and RMS axial velocity at the stations and on the centerline against the experiment
    volvo.py budget integrals.csv --t0 T0 [--t1 T1]
        time averages of the kinetic-energy budget's dissipation terms
    volvo.py slice RUN_DIR --out slice.npz
        frames of the mid-span probe plane from probe_slice_*.csv
    volvo.py animate slice.npz --out MP4 [--field vorticity|T]

Distances are in units of D = 0.04 m from the base of the triangle (x) and
the channel's mid-plane (y); velocities over U_bulk.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

D = 0.04
STATIONS = [0.375, 0.95, 1.53, 3.75, 9.4]
T_STATIONS = [3.75, 8.75, 13.75]


def cmd_probes(a):
    ys = np.arange(-0.06 + a.dy / 2, 0.06, a.dy)
    n = int(round((a.x1 - a.x0) / a.dx)) + 1
    variables = ", ".join(f'"{v}"' for v in a.vars.split(","))
    for k, y in enumerate(ys):
        print(f'\n[[probes]]\nname = "slice_{k:03d}"\nvariables = [{variables}]\n'
              f"start = [{a.x0}, {y:.6g}, {a.z}]\nend = [{a.x1}, {y:.6g}, {a.z}]\nn_points = {n}\ninterval = {a.interval}")


def read_cells(path, names):
    """Cell centroids and fields of a 3D VTU, or of all pieces of a PVTU."""
    from mallard_vtu import read_vtu_cells
    if path.endswith(".pvtu"):
        pieces = re.findall(r'Source="([^"]+)"', open(path).read())
        parts = [read_cells(os.path.join(os.path.dirname(path), p), names) for p in pieces]
        return (parts[0][0], np.concatenate([p[1] for p in parts]),
                {n: np.concatenate([p[2][n] for p in parts]) for n in names})
    pts, conn, offs, _, arrays = read_vtu_cells(path)
    start = np.concatenate([[0], offs[:-1]])
    count = offs - start
    c = np.add.reduceat(pts[conn], start, axis=0) / count[:, None]
    return arrays.get("TIME"), c, {n: arrays[n] for n in names}


def span_average(c, fields):
    """Average over z of an extruded mesh: cells with the same (x, y)."""
    key = np.round(c[:, :2] / 1e-7).astype(np.int64)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    cnt = np.bincount(inv)
    xy = np.column_stack([np.bincount(inv, c[:, 0]), np.bincount(inv, c[:, 1])]) / cnt[:, None]
    out = {}
    for n, f in fields.items():
        f = f if f.ndim == 1 else f[:, 0]
        out[n] = np.bincount(inv, f) / cnt
    return xy, out


def cmd_profiles(a):
    from scipy.interpolate import LinearNDInterpolator
    names = ["MEAN_U_X", "MEAN_U_Y", "COV_U_X_U_X", "COV_U_Y_U_Y", "COV_U_X_U_Y"]
    extra = [n for n in ("MEAN_T",) if a.temperature]
    t, c, f = read_cells(a.stats, names + extra)
    for n in list(f):
        if f[n].ndim > 1:
            f[n] = f[n][:, 0]
    xy, avg = span_average(c, f)
    U = a.u_bulk
    q = {"U_mean": avg["MEAN_U_X"] / U, "V_mean": avg["MEAN_U_Y"] / U,
         "U_rms": np.sqrt(np.maximum(avg["COV_U_X_U_X"], 0)) / U,
         "V_rms": np.sqrt(np.maximum(avg["COV_U_Y_U_Y"], 0)) / U, "uv_mean": avg["COV_U_X_U_Y"] / U ** 2}
    if a.temperature:
        q["T_mean"] = avg["MEAN_T"]
    xd, yd = xy[:, 0] / a.d, xy[:, 1] / a.d
    interp = LinearNDInterpolator(np.column_stack([xd, yd]), np.column_stack([q[k] for k in q]))
    keys = list(q)
    y = np.linspace(-1.5, 1.5, 241)
    out = {"keys": np.array(keys), "y": y, "stations": np.array(STATIONS + T_STATIONS)}
    for s in STATIONS + T_STATIONS:
        out[f"x{s}"] = interp(np.full_like(y, s), y)
    x = np.linspace(0.0, 16.0, 1601)
    cl = interp(x, np.zeros_like(x))
    out["x"], out["centerline"] = x, cl
    u = cl[:, keys.index("U_mean")]
    i = np.where((u[:-1] < 0) & (u[1:] >= 0) & (x[:-1] > 0.05))[0]
    lr = x[i[-1]] - u[i[-1]] * (x[i[-1] + 1] - x[i[-1]]) / (u[i[-1] + 1] - u[i[-1]]) if len(i) else np.nan
    out["L_r"] = lr
    out["u_min"], out["x_u_min"] = np.nanmin(u), x[np.nanargmin(u)]
    np.savez(a.out, **out)
    print(f"{a.stats}: t = {t}, recirculation length L_r / D = {lr:.3f}, "
          f"min U/U_b on the centerline {np.nanmin(u):.3f} at x/D = {x[np.nanargmin(u)]:.2f}")


def read_exp(path):
    import csv
    rows = [r for r in csv.reader(l for l in open(path) if not l.startswith("#"))]
    head = rows[0]
    return head, rows[1:]


def exp_profiles(path):
    head, rows = read_exp(path)
    out = {}
    for r in rows:
        d = dict(zip(head, r))
        out.setdefault((d["quantity"], float(d["x_over_D"])), []).append((float(d["y_over_D"]), float(d["value"])))
    return {k: np.array(sorted(v)) for k, v in out.items()}


def exp_centerline(path):
    head, rows = read_exp(path)
    out = {}
    for r in rows:
        d = dict(zip(head, r))
        out.setdefault(d["quantity"], []).append((float(d["x_over_D"]), float(d["value"])))
    return {k: np.array(sorted(v)) for k, v in out.items()}


def errors(run, exp, quantities=("U_mean", "U_rms")):
    """Mean absolute difference from the experiment at its points, per quantity and station (units of U_b)."""
    keys = list(run["keys"])
    res = {}
    for qn in quantities:
        for s in STATIONS:
            pts = exp.get((qn, s))
            if pts is None:
                continue
            sim = np.interp(pts[:, 0], run["y"], run[f"x{s}"][:, keys.index(qn)])
            res[(qn, s)] = float(np.nanmean(np.abs(sim - pts[:, 1])))
    return res


def cmd_plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pref = "nonreacting" if a.kind == "cold" else "reacting"
    exp = exp_profiles(os.path.join(a.exp, f"{pref}_profiles.csv"))
    expc = exp_centerline(os.path.join(a.exp, f"{pref}_centerline.csv"))
    runs = []
    for spec in a.runs:
        path, _, label = spec.partition(":")
        runs.append((np.load(path), label or os.path.basename(path)))
    temp = a.kind == "reacting" and "T_mean" in list(runs[0][0]["keys"])
    nrow = 3 if temp else 2
    fig = plt.figure(figsize=(15, 4.2 * nrow))
    gs = fig.add_gridspec(nrow, 6)
    colors = ["C0", "C3", "C2", "C1", "C4", "C5"]
    for row, qn in enumerate(["U_mean", "U_rms"]):
        for k, s in enumerate(STATIONS):
            ax = fig.add_subplot(gs[row, k])
            pts = exp.get((qn, s))
            if pts is not None:
                ax.plot(pts[:, 1], pts[:, 0], "ko", ms=3, label="experiment")
            for (r, lab), col in zip(runs, colors):
                ax.plot(r[f"x{s}"][:, list(r["keys"]).index(qn)], r["y"], color=col, lw=1.2, label=lab)
            ax.set_title(f"x/D = {s}", fontsize=9)
            ax.set_xlabel("U/U_b" if qn == "U_mean" else "u'/U_b")
            ax.set_ylim(-1.5, 1.5)
            if k == 0:
                ax.set_ylabel("y/D")
            ax.grid(alpha=0.3)
        ax = fig.add_subplot(gs[row, 5])
        if qn == "U_mean":
            e = expc.get("U_mean")
            if e is not None:
                ax.plot(e[:, 0], e[:, 1], "ko", ms=3, label="experiment")
            for (r, lab), col in zip(runs, colors):
                ax.plot(r["x"], r["centerline"][:, list(r["keys"]).index("U_mean")], color=col, lw=1.2,
                        label=f"{lab}: L_r/D = {float(r['L_r']):.2f}")
            ax.axhline(0, color="gray", lw=0.5)
            ax.set_xlabel("x/D")
            ax.set_ylabel("U/U_b on y = 0")
            ax.set_xlim(0, 12)
            ax.legend(fontsize=7)
        else:
            e = expc.get("fluct_sqrt_Urms2_plus_Vrms2")
            if e is not None:
                ax.plot(e[:, 0], e[:, 1], "ko", ms=3)
            for (r, lab), col in zip(runs, colors):
                keys = list(r["keys"])
                cl = r["centerline"]
                ax.plot(r["x"], np.hypot(cl[:, keys.index("U_rms")], cl[:, keys.index("V_rms")]), color=col, lw=1.2)
            ax.set_xlabel("x/D")
            ax.set_ylabel("sqrt(u'^2 + v'^2)/U_b on y = 0")
            ax.set_xlim(0, 12)
        ax.grid(alpha=0.3)
    if temp:
        expt = exp_profiles(os.path.join(a.exp, "reacting_temperature.csv"))
        for k, s in enumerate(T_STATIONS):
            ax = fig.add_subplot(gs[2, k])
            pts = expt.get(("T_mean", s)) or next((v for (qq, ss), v in expt.items() if ss == s), None)
            if pts is not None:
                ax.plot(pts[:, 1], pts[:, 0], "ko", ms=3)
            for (r, lab), col in zip(runs, colors):
                ax.plot(r[f"x{s}"][:, list(r["keys"]).index("T_mean")], r["y"], color=col, lw=1.2)
            ax.set_title(f"x/D = {s}", fontsize=9)
            ax.set_xlabel("T (K)")
            ax.set_ylim(-1.5, 1.5)
            ax.grid(alpha=0.3)
    fig.axes[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    for r, lab in runs:
        err = errors(r, exp)
        txt = ", ".join(f"{qn[0]}{'' if qn == 'U_mean' else 'rms'}@{s}: {v:.3f}" for (qn, s), v in err.items())
        mean_u = np.mean([v for (qn, s), v in err.items() if qn == "U_mean"])
        mean_r = np.mean([v for (qn, s), v in err.items() if qn == "U_rms"])
        print(f"{lab}: L_r/D {float(r['L_r']):.3f}; mean |error| / U_b: U {mean_u:.3f}, u' {mean_r:.3f}; {txt}")
    if "U_mean" in expc:
        e = expc["U_mean"]
        i = np.where((e[:-1, 1] < 0) & (e[1:, 1] >= 0))[0][-1]
        lr = e[i, 0] - e[i, 1] * (e[i + 1, 0] - e[i, 0]) / (e[i + 1, 1] - e[i, 1])
        print(f"experiment: L_r/D {lr:.3f}")


def cmd_budget(a):
    import csv
    rows = list(csv.DictReader(open(a.integrals)))
    t = np.array([float(r["t"]) for r in rows])
    sel = (t >= a.t0) & (t <= (a.t1 if a.t1 is not None else np.inf))
    get = lambda k: np.array([float(r[k]) for r in rows])[sel]
    eps_sgs, eps_mol, eps_num = -get("ke_rate_sgs"), -get("ke_rate_viscous"), get("eps_numerical")
    ts = t[sel]
    avg = lambda f: np.trapezoid(f, ts) / (ts[-1] - ts[0])
    s, m, n = avg(eps_sgs), avg(eps_mol), avg(eps_num)
    tot = s + m + n
    print(f"{a.integrals}: t = {ts[0]:.4g}-{ts[-1]:.4g} s ({sel.sum()} rows): eps_sgs {s:.4g} W, eps_mol {m:.4g} W, "
          f"eps_num {n:.4g} W; shares numerical / SGS / molecular {100 * n / tot:.1f} / {100 * s / tot:.1f} / "
          f"{100 * m / tot:.1f}%; eps_num / eps_sgs = {n / s if s > 0 else float('nan'):.3f}")


def cmd_slice(a):
    files = sorted(glob.glob(os.path.join(a.run, "probe_slice_*.csv")))
    ys, frames, xs, times, names = [], [], None, None, None
    for f in files:
        with open(f) as fh:
            head = fh.readline().strip().split(",")
        d = np.loadtxt(f, delimiter=",", skiprows=1)
        steps = np.unique(d[:, 0])
        npt = int(d[:, 2].max()) + 1
        nt = len(steps)
        d = d[: nt * npt].reshape(nt, npt, -1)
        if xs is None:
            xs, times, names = d[0, :, 3], d[:, 0, 1], head[6:]
        nt = min(nt, len(times))
        ys.append(d[0, 0, 4])
        frames.append(d[:nt, :, 6:])
    nt = min(f.shape[0] for f in frames)
    data = np.stack([f[:nt] for f in frames], axis=1)  # (t, y, x, var)
    np.savez_compressed(a.out, x=xs, y=np.array(ys), t=times[:nt], names=np.array(names), data=data.astype(np.float32))
    print(f"{a.out}: {nt} frames, {len(ys)} x {len(xs)} points, {names}")


def cmd_animate(a):
    import matplotlib
    matplotlib.use("Agg")
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt
    s = np.load(a.slice)
    x, y, t, names, data = s["x"] / D, s["y"] / D, s["t"], list(s["names"]), s["data"]
    inside = (x[None, :] > -np.sqrt(3) / 2) & (x[None, :] < 0) & (np.abs(y[:, None]) < (x[None, :] + np.sqrt(3) / 2) / np.sqrt(3))
    if a.field == "vorticity":
        u, v = data[..., names.index("U_X")], data[..., names.index("U_Y")]
        f = (np.gradient(v, x * D, axis=2) - np.gradient(u, y * D, axis=1))
        cmap, lim, lab = "RdBu_r", (-a.vmax, a.vmax), "vorticity omega_z (1/s), mid-span"
    else:
        f = data[..., names.index("T")]
        cmap, lim, lab = "inferno", (a.vmin, a.vmax), "temperature (K), mid-span"
    w = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=8)
    start = np.searchsorted(t, a.t0)
    for k in range(start, len(t), a.every):
        fig, ax = plt.subplots(figsize=(12, 4))
        im = ax.pcolormesh(x, y, np.ma.masked_where(inside, f[k]), cmap=cmap, vmin=lim[0], vmax=lim[1], shading="auto")
        ax.fill([-np.sqrt(3) / 2, 0, 0], [0, 0.5, -0.5], color="0.5")
        ax.set_aspect("equal")
        ax.set_xlabel("x/D")
        ax.set_ylabel("y/D")
        ax.set_title(f"{a.title}  t = {1e3 * t[k]:.1f} ms")
        fig.colorbar(im, ax=ax, label=lab, shrink=0.8)
        fig.tight_layout()
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        h, wd = img.shape[0] // 2 * 2, img.shape[1] // 2 * 2
        w.append_data(img[:h, :wd])
        plt.close(fig)
    w.close()
    print(f"wrote {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probes")
    p.add_argument("--x0", type=float, default=-0.04)
    p.add_argument("--x1", type=float, default=0.36)
    p.add_argument("--dx", type=float, default=2e-3)
    p.add_argument("--dy", type=float, default=2e-3)
    p.add_argument("--z", type=float, default=0.04)
    p.add_argument("--interval", type=int, default=1800)
    p.add_argument("--vars", default="U_X,U_Y")
    p = sub.add_parser("profiles")
    p.add_argument("stats")
    p.add_argument("--u-bulk", type=float, required=True)
    p.add_argument("--d", type=float, default=D)
    p.add_argument("--temperature", action="store_true")
    p.add_argument("--out", required=True)
    p = sub.add_parser("plot")
    p.add_argument("runs", nargs="+")
    p.add_argument("--exp", required=True)
    p.add_argument("--kind", default="cold", choices=["cold", "reacting"])
    p.add_argument("--out", required=True)
    p = sub.add_parser("budget")
    p.add_argument("integrals")
    p.add_argument("--t0", type=float, default=0.0)
    p.add_argument("--t1", type=float)
    p = sub.add_parser("slice")
    p.add_argument("run")
    p.add_argument("--out", required=True)
    p = sub.add_parser("animate")
    p.add_argument("slice")
    p.add_argument("--out", required=True)
    p.add_argument("--field", default="vorticity", choices=["vorticity", "T"])
    p.add_argument("--vmin", type=float, default=288.0)
    p.add_argument("--vmax", type=float, default=8000.0)
    p.add_argument("--t0", type=float, default=0.0)
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--title", default="")
    a = ap.parse_args()
    {"probes": cmd_probes, "profiles": cmd_profiles, "plot": cmd_plot, "budget": cmd_budget, "slice": cmd_slice,
     "animate": cmd_animate}[a.cmd](a)


if __name__ == "__main__":
    main()
