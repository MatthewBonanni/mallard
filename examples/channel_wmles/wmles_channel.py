#!/usr/bin/env python3
"""Wall-modeled LES of channel flow (examples/channel_wmles): initial state and analysis.

    wmles_channel.py init input.toml OUT.restart [--seed 1] [--perturbation 0.1] [--T-center TC]
    wmles_channel.py analyze input.toml forces_bottom.csv forces_top.csv --t-start T \\
        --reference {lm2000,lm5200,tl-m1.7,tl-m3.0} [--profile profile.csv] [--out wmles.png]

The input uses the generated `cartesian` mesh with walls at y = 0 and y = 2h,
`[source] mass_flow = [1, 0, 0]` (bulk mass flux rho_b U_b = 1, mean density 1)
and isothermal walls at T_w.

init: the turbulent-like profile and perturbations of channel_init.py, with a
temperature T = T_w + (T_c - T_w) U / U_c and the density p / (R T) at the
uniform pressure that makes the mean density 1. T_c defaults to the estimate
T_w (1 + Pr^(1/3) (gamma - 1) M_b^2 / 2) of a channel whose walls take the
heat of the dissipation. The equilibrium temperature comes out of the run.

analyze: time averages over t >= --t-start of the wall shear stress, wall
pressure and wall heat flux into the fluid from the force monitors (both walls;
the column Q of modeled walls), and from them, with rho_w = p_w / (R T_w),
u_tau = sqrt(tau_w / rho_w):

- Re_tau = rho_w u_tau h / mu_w, Cf = 2 tau_w / (rho_b U_b^2), and
  B_q = q_w / (cp T_w rho_w u_tau);
- with --profile (tools/post/plane_average.py's average over x and z of the
  [statistics] snapshot), U+ against y+ and the centerline temperature, with
  the semi-local Re_tau* = sqrt(rho_c tau_w) h / mu_c.

References: Lee & Moser (2015), J. Fluid Mech. 774, 395,
doi:10.1017/jfm.2015.268, Re_tau = 1995 and 5186 (mean profiles from the
Oden Institute's data archive, cached in --cache); Trettel & Larsson (2016),
Phys. Fluids 28, 026102, doi:10.1063/1.4942022, the cases M_b = 1.7 and 3.0 at
Re_tau* ~ 590 as tabulated by Griffin, Fu & Moin (2023), J. Fluid Mech. 970,
A36, doi:10.1017/jfm.2023.627, table 1 (Re_b, Re_tau, Re_tau*, -B_q). For
mu ~ T^0.75 the centerline temperature of those DNS is
T_c / T_w = (Re_tau / Re_tau*)^0.8.
"""
import argparse
import os
import sys
import tomllib
import urllib.request

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../tools/setup"))
from channel_init import fresh, grid, nodes, write_restart  # noqa: E402

LM_URL = "https://turbulence.oden.utexas.edu/channel2015/data/LM_Channel_{0}_mean_prof.dat"
REFERENCES = {
    "lm2000": {"name": "Lee & Moser 2015", "case": 2000, "Re_tau": 1994.756, "u_tau": 4.58794e-2, "Re_b": 86957},
    "lm5200": {"name": "Lee & Moser 2015", "case": 5200, "Re_tau": 5185.897, "u_tau": 4.14872e-2, "Re_b": 250000},
    "tl-m1.7": {"name": "Trettel & Larsson 2016", "M_b": 1.699, "Re_b": 15490, "Re_tau": 963.6, "Re_tau_star": 590.8,
                "B_q": -0.04942},
    "tl-m3.0": {"name": "Trettel & Larsson 2016", "M_b": 2.997, "Re_b": 23980, "Re_tau": 1842.0,
                "Re_tau_star": 589.7, "B_q": -0.1150},
}


def gas(inp):
    ph = inp["physics"]
    gamma = float(ph["gamma"])
    R = float(ph["p_ref"]) / (float(ph["rho_ref"]) * float(ph["T_ref"]))
    mu_ref = float(ph["mu"])
    if ph.get("viscosity_model", "constant") == "power_law":
        T0, n = float(ph["T_mu_ref"]), float(ph["viscosity_exponent"])
        mu = lambda T: mu_ref * (np.asarray(T) / T0) ** n  # noqa: E731
    else:
        mu = lambda T: mu_ref + 0 * np.asarray(T)  # noqa: E731
    return gamma, R, mu, float(ph.get("Pr", 0.72))


def wall_temperature(inp):
    for b in inp["boundaries"]:
        if b["name"] == "bottom":
            return float(b["T"])
    raise SystemExit("input: no bottom wall")


def init(args):
    with open(args.input, "rb") as f:
        inp = tomllib.load(f)
    n, L, centers = grid(inp)
    fields = fresh(inp, n, L, centers, args.seed, args.perturbation)
    gamma, R, _, Pr = gas(inp)
    T_w = wall_temperature(inp)
    u, v, w = fields["RHOU_X"], fields["RHOU_Y"], fields["RHOU_Z"]
    M_b = 1.0 / np.sqrt(gamma * R * T_w)
    T_c = args.T_center or T_w * (1 + Pr ** (1 / 3) * (gamma - 1) / 2 * M_b**2)
    U_mean = u.mean(axis=(0, 2), keepdims=True)
    T = T_w + (T_c - T_w) * np.clip(U_mean / U_mean.max(), 0, 1) + 0 * u
    dy = np.diff(nodes(n[1], L[1], float(inp["mesh"].get("stretching", [0, 0, 0])[1])))
    weights = np.broadcast_to(dy[None, :, None], u.shape)
    p = R / ((weights / T).sum() / weights.sum())
    rho = p / (R * T)
    scale = (weights * rho * u).sum() / weights.sum()
    u, v, w = u / scale, v / scale, w / scale
    E = p / (gamma - 1) + 0.5 * rho * (u**2 + v**2 + w**2)
    write_restart(args.out, 0, 0.0, {"RHO": rho, "RHOU_X": rho * u, "RHOU_Y": rho * v, "RHOU_Z": rho * w, "RHOE": E})
    print(f"{args.out}: {np.prod(n)} cells, M_b = {M_b:.3f}, T_c / T_w = {T_c / T_w:.3f}, p = {p:.5g}")


def read_forces(path):
    with open(path) as f:
        header = not f.readline()[0].isdigit()
    d = np.loadtxt(path, delimiter=",", skiprows=int(header), ndmin=2)
    _, keep = np.unique(d[:, 1], return_index=True)
    return d[keep]


def lee_moser(cache, case):
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, f"LM_Channel_{case}_mean_prof.dat")
    if not os.path.exists(path):
        urllib.request.urlretrieve(LM_URL.format(case), path)
    d = np.loadtxt(path, comments="%")
    return d[:, 1], d[:, 2]


def analyze(args):
    with open(args.input, "rb") as f:
        inp = tomllib.load(f)
    gamma, R, mu, Pr = gas(inp)
    cp = gamma * R / (gamma - 1)
    T_w = wall_temperature(inp)
    Lx, Ly, Lz = (float(inp["mesh"][k]) for k in ("Lx", "Ly", "Lz"))
    h, A = Ly / 2, Lx * Lz
    ref = REFERENCES[args.reference]
    bottom, top = read_forces(args.forces[0]), read_forces(args.forces[1])
    t = bottom[:, 1]
    top = np.array([np.interp(t, top[:, 1], top[:, j]) for j in range(top.shape[1])]).T
    tau_t = (bottom[:, 5] + top[:, 5]) / (2 * A)  # the flow drags both walls along +x
    p_t = (top[:, 3] - bottom[:, 3]) / (2 * A)
    q_t = (bottom[:, 8] + top[:, 8]) / (2 * A) if bottom.shape[1] > 8 else np.zeros_like(t)
    win = t >= args.t_start
    if win.sum() < 2:
        raise SystemExit("analyze: fewer than two samples after --t-start")

    def mean(x):
        return np.trapezoid(x[win], t[win]) / (t[win][-1] - t[win][0])

    tau_w, p_w, q_w = mean(tau_t), mean(p_t), mean(q_t)
    rho_w = p_w / (R * T_w)
    mu_w = float(mu(T_w))
    u_tau = np.sqrt(tau_w / rho_w)
    Re_tau = rho_w * u_tau * h / mu_w
    m_b = float(np.linalg.norm(inp["source"]["mass_flow"]))
    rho_b = 1.0
    prof = None
    if args.profile:
        prof = np.genfromtxt(args.profile, delimiter=",", names=True)
        dy = np.diff(nodes(len(prof["y"]), Ly, float(inp["mesh"].get("stretching", [0, 0, 0])[1])))
        rho_b = (prof["MEAN_RHO"] * dy).sum() / Ly
    U_b = m_b / rho_b
    Cf = 2 * tau_w / (rho_b * U_b**2)
    Re_b = rho_b * U_b * h / mu_w
    B_q = q_w / (cp * T_w * rho_w * u_tau)
    blocks = int(args.blocks)
    edges = np.linspace(t[win][0], t[win][-1], blocks + 1)
    Re_blocks = [np.sqrt(rho_w * np.mean(tau_t[(t >= a) & (t <= b)])) * h / mu_w for a, b in zip(edges[:-1], edges[1:])]
    window = (t[win][-1] - t[win][0]) * u_tau / h
    lines = [f"window t = {t[win][0]:g}-{t[win][-1]:g} h/U_b = {window:.0f} h/u_tau; Re_b (h, mu_w) = {Re_b:.0f}, "
             f"rho_w / rho_b = {rho_w / rho_b:.4f}"]
    if "Re_tau_star" in ref:
        lines.append(f"Re_tau = {Re_tau:.1f} ({ref['name']} {ref['Re_tau']:.1f}, {100 * (Re_tau / ref['Re_tau'] - 1):+.2f}%; "
                     f"Re_tau^2 {100 * ((Re_tau / ref['Re_tau'])**2 - 1):+.2f}%), blocks {np.std(Re_blocks):.1f} rms")
        lines.append(f"B_q = {B_q:.5f} ({ref['B_q']:.5f}, {100 * (B_q / ref['B_q'] - 1):+.1f}%)")
    else:
        Cf_ref = 2 * ref["u_tau"] ** 2
        lines.append(f"Re_tau = {Re_tau:.1f} ({ref['name']} {ref['Re_tau']:.1f}, {100 * (Re_tau / ref['Re_tau'] - 1):+.2f}%), "
                     f"blocks {np.std(Re_blocks):.1f} rms")
        lines.append(f"Cf = {Cf:.6f} ({Cf_ref:.6f}, {100 * (Cf / Cf_ref - 1):+.2f}%)")
    out = {"Re_tau": Re_tau, "Cf": Cf, "B_q": B_q}
    if prof is not None:
        y = np.where(prof["y"] < h, prof["y"], Ly - prof["y"]) / h
        order = np.argsort(y)
        y, U, T, rho = y[order], prof["MEAN_U_X"][order], prof["MEAN_T"][order], prof["MEAN_RHO"][order]
        # fold the two halves
        yu, idx = np.unique(np.round(y, 10), return_inverse=True)
        U, T, rho = (np.bincount(idx, a) / np.bincount(idx) for a in (U, T, rho))
        y = yu
        T_c, rho_c = T[-1], rho[-1]
        Re_star = np.sqrt(rho_c * tau_w) * h / float(mu(T_c))
        lines.append(f"T_c / T_w = {T_c / T_w:.4f}, Re_tau* = {Re_star:.1f}")
        if "Re_tau_star" in ref:
            Tc_ref = (ref["Re_tau"] / ref["Re_tau_star"]) ** 0.8
            lines.append(f"  DNS: T_c / T_w = {Tc_ref:.4f} ({100 * (T_c / T_w / Tc_ref - 1):+.2f}%), "
                         f"Re_tau* = {ref['Re_tau_star']:.1f} ({100 * (Re_star / ref['Re_tau_star'] - 1):+.2f}%)")
        yp, Up = y * Re_tau, U / u_tau
        out.update(yp=yp, Up=Up, T=T / T_w, y=y)
        if "case" in ref:
            ry, rU = lee_moser(args.cache, ref["case"])
            errs = []
            for f in (0.1, 0.2, 0.3, 0.5, 0.8):
                Us, Ur = np.interp(f * Re_tau, yp, Up), np.interp(f * ref["Re_tau"], ry, rU)
                errs.append(f"{f:g}: {Us:.2f} ({Ur:.2f})")
            lines.append("U+ at y/h = " + ", ".join(errs))
            out.update(ref_yp=ry, ref_U=rU)
    print("\n".join(lines))
    if args.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        if prof is not None:
            if "ref_yp" in out:
                ax[0].semilogx(out["ref_yp"][1:], out["ref_U"][1:], "k-", lw=1.2, label=ref["name"])
            ax[0].semilogx(out["yp"], out["Up"], "o", mfc="none", label="Mallard WMLES")
            ax[0].set(xlabel="$y^+$", ylabel="$U^+$")
            ax[0].legend(frameon=False)
        ax[1].plot(t * u_tau / h, np.sqrt(rho_w * tau_t) * h / mu_w, lw=0.6)
        ax[1].axhline(ref["Re_tau"], color="k", lw=1.0, label=ref["name"])
        ax[1].axvline(args.t_start * u_tau / h, color="0.6", ls=":")
        ax[1].set(xlabel="$t u_\\tau / h$", ylabel="$Re_\\tau(t)$")
        ax[1].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(args.out, dpi=150)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    a = sub.add_parser("init")
    a.add_argument("input")
    a.add_argument("out")
    a.add_argument("--seed", type=int, default=1)
    a.add_argument("--perturbation", type=float, default=0.1)
    a.add_argument("--T-center", type=float)
    a = sub.add_parser("analyze")
    a.add_argument("input")
    a.add_argument("forces", nargs=2, help="forces CSVs of the bottom and top walls")
    a.add_argument("--t-start", type=float, required=True)
    a.add_argument("--reference", required=True, choices=sorted(REFERENCES))
    a.add_argument("--profile")
    a.add_argument("--blocks", type=int, default=4, help="blocks of the window for the spread of Re_tau")
    a.add_argument("--cache", default=os.path.expanduser("~/.cache/mallard/lee_moser"))
    a.add_argument("--out")
    args = ap.parse_args()
    init(args) if args.command == "init" else analyze(args)


if __name__ == "__main__":
    main()
