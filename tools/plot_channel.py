#!/usr/bin/env python3
"""Turbulent channel statistics against the DNS of Moser, Kim & Mansour (1999).

    python tools/plot_channel.py input.toml profile.csv forces_bottom.csv forces_top.csv \\
        --t-start 160 [--half profile_half.csv] [--out channel.png]

profile.csv is tools/plane_average.py's average over x and z of a snapshot
with the [statistics] fields of examples/channel_retau180 (MEAN_RHO, MEAN_U_X,
COV_U_X_U_X, COV_U_Y_U_Y, COV_U_Z_U_Z, COV_U_X_U_Y); the two halves of the
channel are folded together. The wall shear stress tau_w is the time average
of the viscous x-force on both walls (the forces CSVs, which may concatenate
several runs) from --t-start on, the same window as the statistics, and sets
u_tau = sqrt(tau_w / rho_w) and nu_w = mu / rho_w at the walls. --half adds a
profile averaged over the first part of the window, to show convergence.

Reference data (--retau 180, 395 or 590: Re_tau = 178.12, 392.24 or 587.19; Phys. Fluids 11, 943, doi:10.1063/1.869966;
method of Kim, Moin & Moser, J. Fluid Mech. 177, 133, doi:10.1017/S0022112087000892):
chan<case>.means and chan<case>.reystress from the Turbulence Mapping data archive,
read from --mkm (downloaded there if missing).
"""
import argparse
import os
import sys
import tomllib
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from channel_init import grid, nodes  # noqa: E402

MKM_URL = "https://turbulence.oden.utexas.edu/data/MKM/chan{0}/profiles/"
MKM_RE_TAU = {180: 178.12, 395: 392.24, 590: 587.19}


def mkm(directory, case=180):
    os.makedirs(directory, exist_ok=True)
    data = {}
    for kind in ("means", "reystress"):
        name = f"chan{case}.{kind}"
        path = os.path.join(directory, name)
        if not os.path.exists(path):
            urllib.request.urlretrieve(MKM_URL.format(case) + name, path)
        data[kind] = np.loadtxt(path, comments="#")
    m, r = data["means"], data["reystress"]
    return {"y": m[:, 0], "yp": m[:, 1], "U": m[:, 2], "uu": r[:, 2], "vv": r[:, 3], "ww": r[:, 4],
            "uv": r[:, 5], "Re_tau": MKM_RE_TAU[case], "Ub": np.trapezoid(m[:, 2], m[:, 0])}


def read_profile(path, h):
    """Columns of a plane_average CSV, folded about the center: y is the distance to the nearer wall over h."""
    d = np.genfromtxt(path, delimiter=",", names=True)
    y = d["y"] / h
    lower = y < 1
    out = {"y": np.where(lower, y, 2 - y)}
    sign_v = np.where(lower, 1.0, -1.0)  # v and uv change sign under y -> 2h - y
    out["U"] = d["MEAN_U_X"]
    out["rho"] = d["MEAN_RHO"] if "MEAN_RHO" in d.dtype.names else np.ones_like(y)
    out["T"] = d["MEAN_T"] if "MEAN_T" in d.dtype.names else np.ones_like(y)
    out["uu"], out["vv"], out["ww"] = d["COV_U_X_U_X"], d["COV_U_Y_U_Y"], d["COV_U_Z_U_Z"]
    out["uv"] = d["COV_U_X_U_Y"] * sign_v
    order = np.argsort(out["y"], kind="stable")
    folded = {k: v[order] for k, v in out.items()}
    # Average the mirror cells of the two halves
    _, idx = np.unique(np.round(folded["y"], 10), return_inverse=True)
    return {k: np.bincount(idx, v) / np.bincount(idx) for k, v in folded.items()}


def read_forces(paths):
    """Time and the sum of the viscous x-forces on the walls (one CSV per wall)."""
    out = []
    for p in paths:
        # step, t, F_pressure (3), F_viscous (3); runs started from a restart file write no header
        with open(p) as f:
            header = not f.readline()[0].isdigit()
        d = np.loadtxt(p, delimiter=",", skiprows=int(header), ndmin=2)
        t, f = d[:, 1], d[:, 5]
        _, keep = np.unique(t, return_index=True)  # runs restarted from a restart repeat rows
        out.append((t[keep], f[keep]))
    t = out[0][0]
    f = sum(np.interp(t, ti, fi) for ti, fi in out)
    return t, f


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("input")
    ap.add_argument("profile")
    ap.add_argument("forces", nargs=2, help="forces CSVs of the bottom and top walls")
    ap.add_argument("--t-start", type=float, required=True, help="start of the averaging window")
    ap.add_argument("--t-end", type=float, help="end of the averaging window (the profile's time; default: last force)")
    ap.add_argument("--half", help="profile over the first part of the window")
    ap.add_argument("--half-label", default="first part of the window", help="legend of --half")
    ap.add_argument("--mkm", default=os.path.expanduser("~/.cache/mallard/mkm"))
    ap.add_argument("--retau", type=int, default=180, choices=sorted(MKM_RE_TAU), help="MKM case to compare with")
    ap.add_argument("--out", default="channel.png")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(args.input, "rb") as f:
        inp = tomllib.load(f)
    mu = float(inp["physics"]["mu"])
    Lx, Ly, Lz = (float(inp["mesh"][k]) for k in ("Lx", "Ly", "Lz"))
    h = Ly / 2
    m_b = float(np.linalg.norm(inp["source"]["mass_flow"]))  # rho_b U_b
    ref = mkm(args.mkm, args.retau)

    t, F = read_forces(args.forces)
    tau_t = F / (2 * Lx * Lz)  # both walls
    win = (t >= args.t_start) & (t <= (args.t_end if args.t_end is not None else np.inf))
    tau_w = np.trapezoid(tau_t[win], t[win]) / (t[win][-1] - t[win][0])
    prof = read_profile(args.profile, h)
    rho_w = prof["rho"][0]
    u_tau = np.sqrt(tau_w / rho_w)
    nu_w = mu / rho_w
    Re_tau = u_tau * h / nu_w
    raw = np.genfromtxt(args.profile, delimiter=",", names=True)
    _, _, (_, y_centers, _) = grid(inp)
    widths = np.diff(nodes(len(y_centers), Ly, float(inp["mesh"].get("stretching", [0, 0, 0])[1])))
    if len(raw["y"]) != len(widths):
        raise SystemExit(f"{args.profile}: {len(raw['y'])} rows, the mesh has {len(widths)} cells in y")
    rho_b = (raw["MEAN_RHO"] * widths).sum() / Ly
    U_b = m_b / rho_b
    Cf = 2 * tau_w / (rho_b * U_b**2)
    Cf_mkm = 2 / ref["Ub"]**2
    Re_b = 2 * h * rho_b * U_b / mu
    Cf_dean = 0.073 * Re_b**-0.25
    window = (t[win][-1] - args.t_start) * u_tau / h

    def wall(p):
        return {"yp": p["y"] * h * u_tau / nu_w, "U": p["U"] / u_tau,
                "u": np.sqrt(p["uu"]) / u_tau, "v": np.sqrt(p["vv"]) / u_tau,
                "w": np.sqrt(p["ww"]) / u_tau, "uv": p["uv"] / u_tau**2}

    s = wall(prof)
    r = {"yp": ref["yp"], "U": ref["U"], "u": np.sqrt(ref["uu"]), "v": np.sqrt(ref["vv"]),
         "w": np.sqrt(ref["ww"]), "uv": ref["uv"]}

    def at(p, key, yp):
        return np.interp(yp, p["yp"], p[key])

    lines = [
        f"averaging window: t = {args.t_start:g} to {t[win][-1]:g} h/U_b = {window:.1f} h/u_tau",
        f"Re_tau = {Re_tau:.1f} (MKM {ref['Re_tau']:.1f}), u_tau / U_b = {u_tau / U_b:.5f} (MKM {1 / ref['Ub']:.5f})",
        f"Cf = {Cf:.5f} (MKM {Cf_mkm:.5f}, {100 * (Cf / Cf_mkm - 1):+.1f}%; Dean {Cf_dean:.5f}), Re_b = {Re_b:.0f}",
        f"U_b+ = {U_b / u_tau:.2f} (MKM {ref['Ub']:.2f}), U_c+ = {s['U'][-1]:.2f} (MKM {ref['U'][-1]:.2f})",
    ]
    for key, label in (("u", "u_rms+"), ("v", "v_rms+"), ("w", "w_rms+"), ("uv", "-uv+")):
        sign = -1 if key == "uv" else 1
        i, j = np.argmax(sign * s[key]), np.argmax(sign * r[key])
        lines.append(f"peak {label} = {sign * s[key][i]:.3f} at y+ = {s['yp'][i]:.1f} "
                     f"(MKM {sign * r[key][j]:.3f} at {r['yp'][j]:.1f})")
    for yp in (5, 10, 30, 100):
        lines.append(f"U+({yp}) = {at(s, 'U', yp):.2f} (MKM {at(r, 'U', yp):.2f})")
    lines.append(f"wall: rho_w = {rho_w:.4f}, T from {prof['T'][0]:.4f} (wall) to {prof['T'][-1]:.4f} (center)")
    print("\n".join(lines))

    half = wall(read_profile(args.half, h)) if args.half else None
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    a = ax[0, 0]
    a.semilogx(r["yp"][1:], r["U"][1:], "k-", lw=1.2, label="MKM 1999")
    a.semilogx(s["yp"], s["U"], "o", ms=3.5, mfc="none", color="C0", label="Mallard")
    if half:
        a.semilogx(half["yp"], half["U"], "-", color="C1", lw=0.8, label=args.half_label)
    yy = np.logspace(0, 2.4)
    a.semilogx(yy[yy < 12], yy[yy < 12], ":", color="0.5", lw=0.8)
    a.semilogx(yy[yy > 20], np.log(yy[yy > 20]) / 0.41 + 5.2, ":", color="0.5", lw=0.8)
    a.set(xlabel="$y^+$", ylabel="$U^+$", xlim=(0.3, 1.1 * ref["Re_tau"]), title="Mean velocity")
    a.legend(frameon=False)
    for a, key, label in ((ax[0, 1], "u", "$u_{rms}^+$"), (ax[0, 2], "v", "$v_{rms}^+$"),
                          (ax[1, 0], "w", "$w_{rms}^+$"), (ax[1, 1], "uv", "$-\\overline{u'v'}^+$")):
        sign = -1 if key == "uv" else 1
        a.plot(r["yp"], sign * r[key], "k-", lw=1.2, label="MKM 1999")
        a.plot(s["yp"], sign * s[key], "o", ms=3.5, mfc="none", color="C0", label="Mallard")
        if half:
            a.plot(half["yp"], sign * half[key], "-", color="C1", lw=0.8, label=args.half_label)
        a.set(xlabel="$y^+$", ylabel=label, xlim=(0, ref["Re_tau"]))
    ax[1, 1].plot(r["yp"], 1 - r["yp"] / ref["Re_tau"], ":", color="0.5", lw=0.8, label="total stress $1 - y/h$")
    ax[1, 1].legend(frameon=False)
    a = ax[1, 2]
    Re_t = np.sqrt(tau_t / rho_w) * h / nu_w
    a.plot(t * u_tau / h, Re_t, color="C0", lw=0.6)
    a.axvspan(args.t_start * u_tau / h, t[win][-1] * u_tau / h, color="C0", alpha=0.08, label="averaging window")
    a.axhline(ref["Re_tau"], color="k", lw=1.0, label="MKM")
    a.set(xlabel="$t u_\\tau / h$", ylabel="$Re_\\tau(t)$ from the wall shear", title="Wall shear history",
          ylim=(0.8 * ref["Re_tau"], 1.2 * ref["Re_tau"]))
    a.legend(frameon=False)
    fig.suptitle(f"Channel flow, $Re_\\tau$ = {Re_tau:.1f}: Mallard vs Moser, Kim & Mansour (1999)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
