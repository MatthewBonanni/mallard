"""Heat release and ignition-front analysis of stratified autoignition runs (examples/autoignition_2d).

    python tools/autoignition_analysis.py REFERENCE_DIR DEFLAGRATION.csv --run LABEL RUN_DIR [--run ...]
        [--ratio 1.5] [--out DIR] [--every 1]

REFERENCE_DIR holds tools/autoignition_reference.py's output (homogeneous.csv,
ignition_delay.csv and multizone_<run>.csv, <run> the basename of RUN_DIR);
DEFLAGRATION.csv the deflagration speeds S_L(T_u, p) (columns T_u, p_atm,
S_L, delta) from tools/autoignition_deflagration.py. Each RUN_DIR has the
initial state autoignition.restart and the output series
solut/autoignition.pvd (with RHO, P, T, HRR, Y_H2O, OMEGA_H2O, X_H2O, D_H2O).

For every output the domain-mean heat release rate and pressure are taken,
and each cell is classified by the ignition-front diagnostic of Chen et al.
(2006): the density-weighted displacement speed of the H2O isolines,

    S_d* = (omega_H2O + div(rho D_H2O (W_H2O / W) grad X_H2O)) / (rho_u |grad Y_H2O|),

against the deflagration speed S_L of the end gas (the mean temperature,
pressure and density rho_u of the least reacted tenth of the cells, or of
all cells with less than 10% of the burnt H2O; none once that tenth has more
than half of it).
A front whose reaction is carried by diffusion moves at S_d* ~ S_L
(deflagration); a spontaneous ignition front, where neighboring cells ignite
one after the other by their own chemistry, moves much faster. Heat released
where S_d* < --ratio S_L counts as deflagrative (the diffusion term is the
mixture-averaged flux with which Mallard transports the species, without the
correction velocity).

The a priori regime criterion of Sankaran et al. (2005) compares S_L with the
spontaneous-propagation speed S_sp = 1 / |grad tau_ig| of the initial field:
beta = S_L |dtau/dT| |grad T| > 1 predicts deflagrations; the share of the
mass with beta > 1 is printed for each run, of the initial field and of the
field at 10% heat release (turbulent mixing has then weakened the gradients).

Writes OUT_DIR/<run>_history.csv (t, p, mean HRR, deflagrative share of the
HRR, T_u, S_L) and the figures autoignition_hrr.png (mean HRR against
t / tau_0 with the multizone and homogeneous references), autoignition_
timing.png (HRR peak time and width and the deflagrative share against T')
and autoignition_speed.png (HRR-weighted distribution of S_d* / S_L), and
prints the table of results (with the HRR-weighted median of |S_d*| / S_L).
"""
import argparse
import os
import re
import struct
import sys

import cantera as ct
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mallard_vtu import grid_fields  # noqa: E402

FIELDS = ["RHO", "P", "T", "HRR", "Y_H2O", "OMEGA_H2O", "X_H2O", "D_H2O"]
W_H2O = 18.01528e-3


def read_restart(path):
    with open(path, "rb") as f:
        raw = f.read()
    _, _, n, nv, _, _ = struct.unpack_from("<IIQQQd", raw, 16)
    off = 16 + struct.calcsize("<IIQQQd")
    names = []
    for _ in range(nv):
        (ln,) = struct.unpack_from("<I", raw, off)
        names.append(raw[off + 4:off + 4 + ln].decode())
        off += 4 + ln
    data = np.frombuffer(raw, "<f8", n * nv, off).reshape(nv, n)
    return dict(zip(names, data))


class DeflagrationSpeed:
    """S_L(T_u, p): log-linear in T_u at each tabulated pressure, linear in p between them."""

    def __init__(self, path):
        d = np.genfromtxt(path, delimiter=",", names=True)
        self.fits = {}
        p_atm = np.round(d["p_atm"])
        for p in np.unique(p_atm):
            s = p_atm == p
            self.fits[p] = np.polyfit(d["T_u"][s], np.log(d["S_L"][s]), 1 if s.sum() < 4 else 2)

    def __call__(self, T, p_atm):
        ps = sorted(self.fits)
        vals = np.array([np.exp(np.polyval(self.fits[q], T)) for q in ps])
        if len(ps) == 1:
            return vals[0]
        return np.array([np.interp(pp, ps, v) for pp, v in zip(np.broadcast_to(p_atm, np.shape(T)).ravel(),
                                                               vals.reshape(len(ps), -1).T)]).reshape(np.shape(T))


def series(run_dir):
    pvd = open(os.path.join(run_dir, "solut", "autoignition.pvd")).read()
    return [os.path.join(run_dir, "solut", f) for f in re.findall(r'file="([^"]+)"', pvd)]


def ddx(f, h, axis):
    return (np.roll(f, -1, axis) - np.roll(f, 1, axis)) / (2 * h)


def analyze_snapshot(path, Y_b, S_L_of, ratio, bins):
    t, xs, ys, f = grid_fields(path, FIELDS)
    h = xs[1] - xs[0]
    rho, p, T, hrr = f["RHO"], f["P"], f["T"], f["HRR"]
    Y, X, D, w = f["Y_H2O"], f["X_H2O"], f["D_H2O"], f["OMEGA_H2O"]
    W = rho * ct.gas_constant * 1e-3 * T / p
    a = rho * D * W_H2O / W
    div = ddx(a * ddx(X, h, 0), h, 0) + ddx(a * ddx(X, h, 1), h, 1)
    grad = np.hypot(ddx(Y, h, 0), ddx(Y, h, 1))
    c = Y / Y_b
    # End gas: the least reacted tenth of the cells (at least those below 10% of the burnt H2O)
    c_u = max(0.1, np.percentile(c, 10))
    unburnt = c <= c_u
    row = dict(t=t, p=p.mean(), hrr=hrr.mean(), defl=np.nan, T_u=np.nan, S_L=np.nan, T_rms=T.std())
    hist = np.zeros(len(bins) - 1)
    if c_u < 0.5 and hrr.sum() > 0:
        rho_u = rho[unburnt].mean()
        T_u, p_u = T[unburnt].mean(), p[unburnt].mean() / ct.one_atm
        S_L = float(S_L_of(T_u, p_u))
        S_d = (w + div) / (rho_u * np.maximum(grad, 1e-300))
        q = np.maximum(hrr, 0.0)
        row.update(defl=q[S_d < ratio * S_L].sum() / q.sum(), T_u=T_u, S_L=S_L)
        r = np.log10(np.clip(np.abs(S_d) / S_L, 10 ** bins[0], 10 ** bins[-1]))
        hist, _ = np.histogram(r, bins=bins, weights=q)
    return row, hist


def timing(t, hrr):
    """Peak time, full width at half maximum, and the 10-90% heat release times."""
    i = np.argmax(hrr)
    above = np.nonzero(hrr >= 0.5 * hrr[i])[0]
    q = np.concatenate([[0.0], np.cumsum(0.5 * (hrr[1:] + hrr[:-1]) * np.diff(t))])
    q /= q[-1]
    return t[i], t[above[-1]] - t[above[0]], np.interp(0.1, q, t), np.interp(0.9, q, t), hrr[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("deflagration")
    ap.add_argument("--run", nargs=2, action="append", metavar=("LABEL", "RUN_DIR"), required=True)
    ap.add_argument("--ratio", type=float, default=1.5)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--out", default=".")
    ap.add_argument("--mechanism", default=os.path.join(os.path.dirname(__file__), "..", "mechanisms", "h2o2.yaml"))
    ap.add_argument("--phase", default="ohmech")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    hom = np.genfromtxt(os.path.join(args.reference, "homogeneous.csv"), delimiter=",", names=True)
    tau0 = hom["t"][np.argmax(np.gradient(hom["T"], hom["t"]))]
    tig = np.genfromtxt(os.path.join(args.reference, "ignition_delay.csv"), delimiter=",", names=True)
    S_L_of = DeflagrationSpeed(args.deflagration)
    bins = np.linspace(-2, 3, 101)

    results = []
    for label, run_dir in args.run:
        run = os.path.basename(os.path.normpath(run_dir))
        init = read_restart(os.path.join(run_dir, "autoignition.restart"))
        n = int(round(np.sqrt(init["RHO"].size)))
        T0 = init["T_SEED"].reshape(n, n)
        rho0 = init["RHO"].reshape(n, n)
        gas = ct.Solution(args.mechanism, args.phase)
        Y0 = np.array([init["RHOY_" + s] for s in gas.species_names]).mean(axis=1) / rho0.mean()
        gas.TDY = T0.mean(), rho0.mean(), Y0 / Y0.sum()
        p0 = gas.P
        gas.equilibrate("UV")
        Y_b = gas["H2O"].Y[0]

        # A priori: Sankaran et al.'s beta of the initial field
        with open(os.path.join(run_dir, "input.toml")) as fh:
            L = float(re.search(r"Lx = ([0-9.eE+-]+)", fh.read()).group(1))
        h = L / n

        def beta_share(T, rho, p_atm):
            beta = S_L_of(T, p_atm) * np.abs(np.interp(T, tig["T"], tig["dtau_dT"])) * \
                np.hypot(ddx(T, h, 0), ddx(T, h, 1))
            return (rho * (beta > 1)).sum() / rho.sum()

        rows, hist = [], np.zeros(len(bins) - 1)
        files = series(run_dir)[::args.every]
        for path in files:
            row, hh = analyze_snapshot(path, Y_b, S_L_of, args.ratio, bins)
            rows.append(row)
            hist += hh
        hist_t = np.array([r["t"] for r in rows])
        hrr = np.array([r["hrr"] for r in rows])
        defl = np.array([r["defl"] for r in rows])
        np.savetxt(os.path.join(args.out, f"{run}_history.csv"),
                   np.array([[r[k] for k in ("t", "p", "hrr", "defl", "T_u", "S_L", "T_rms")] for r in rows]),
                   delimiter=",", header="t,p,HRR,deflagrative_share,T_u,S_L,T_rms", comments="")
        ok = np.isfinite(defl)
        defl_total = np.trapezoid((hrr * defl)[ok], hist_t[ok]) / np.trapezoid(hrr, hist_t)
        mz = np.loadtxt(os.path.join(args.reference, f"multizone_{run}.csv"), delimiter=",", skiprows=1)
        dns = timing(hist_t, hrr)
        # The same criterion on the field as ignition starts (10% of the heat released)
        _, _, _, f10 = grid_fields(files[int(np.argmin(np.abs(hist_t - dns[2])))], ["T", "RHO", "P"])
        results.append(dict(label=label, run=run, t_rms=T0.std(), t=hist_t, hrr=hrr, defl=defl, hist=hist,
                            beta_share_10=beta_share(f10["T"], f10["RHO"], f10["P"].mean() / ct.one_atm),
                            t_rms_10=f10["T"].std(), dns=dns, mz=timing(mz[:, 0], mz[:, 2]), mz_t=mz[:, 0], mz_hrr=mz[:, 2],
                            defl_total=defl_total, beta_share=beta_share(T0, rho0, p0 / ct.one_atm), p_end=rows[-1]["p"],
                            complete=hrr[-1] < 0.02 * hrr.max()))

    def median_ratio(hist):
        """HRR-weighted median of |S_d*| / S_L."""
        q = np.cumsum(hist) / hist.sum()
        return 10 ** np.interp(0.5, q, bins[1:])

    hom_timing = timing(hom["t"], hom["HRR"])
    print(f"homogeneous: tau_0 = {tau0 * 1e3:.4f} ms, HRR peak {hom_timing[4]:.3e} W/m^3, FWHM {hom_timing[1] * 1e6:.1f} us")
    print(f"{'run':>14} {'T_rms':>6} | {'t_peak/tau0':>11} {'FWHM/tau0':>9} {'t10/tau0':>8} {'t90/tau0':>8} {'peak/hom':>8}"
          f" | {'MZ t_peak':>9} {'MZ FWHM':>8} {'MZ t10':>7} {'MZ t90':>7} {'MZ peak':>8} | {'defl':>5} {'S_d/S_L':>7} {'beta>1':>6} {'at t10':>6} {'T_rms(t10)':>10}")
    for r in results:
        d, m = r["dns"], r["mz"]
        print(f"{r['label']:>14} {r['t_rms']:6.2f} | {d[0] / tau0:11.4f} {d[1] / tau0:9.4f} {d[2] / tau0:8.4f} {d[3] / tau0:8.4f}"
              f" {d[4] / hom_timing[4]:8.4f} | {m[0] / tau0:9.4f} {m[1] / tau0:8.4f} {m[2] / tau0:7.4f} {m[3] / tau0:7.4f}"
              f" {m[4] / hom_timing[4]:8.4f} | {r['defl_total']:5.3f} {median_ratio(r['hist']):7.2f} {r['beta_share']:6.3f} {r['beta_share_10']:6.3f} {r['t_rms_10']:10.2f}"
              f"{'' if r['complete'] else '  (run ends before the heat release does)'}")

    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(results)))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hom["t"] / tau0, hom["HRR"] / hom_timing[4], color="0.6", lw=1, ls=":", label="homogeneous (0D)")
    for r, col in zip(results, colors):
        ax.plot(r["t"] / tau0, r["hrr"] / hom_timing[4], color=col, lw=2, label=f"{r['label']} DNS")
        ax.plot(r["mz_t"] / tau0, r["mz_hrr"] / hom_timing[4], color=col, lw=1, ls="--")
    ax.plot([], [], color="k", lw=1, ls="--", label="multizone (no transport)")
    ax.set_xlim(0.4, 1.6)
    ax.set_ylim(0, 1.05 * max(r["hrr"].max() for r in results) / hom_timing[4])
    ax.set_xlabel(r"$t / \tau_0$")
    ax.set_ylabel(r"mean HRR / homogeneous peak")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(r"H$_2$/air, $\phi$ = 0.1, 1070 K, 41 atm")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "autoignition_hrr.png"), dpi=150)

    Tr = np.array([r["t_rms"] for r in results])
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    axs[0].plot(Tr, [r["dns"][0] / tau0 for r in results], "o-", label="DNS")
    axs[0].plot(Tr, [r["mz"][0] / tau0 for r in results], "s--", label="multizone")
    axs[0].set_ylabel(r"HRR peak time / $\tau_0$")
    axs[1].plot(Tr, [r["dns"][1] / tau0 for r in results], "o-", label="DNS FWHM")
    axs[1].plot(Tr, [r["mz"][1] / tau0 for r in results], "s--", label="multizone FWHM")
    axs[1].plot(Tr, [(r["dns"][3] - r["dns"][2]) / tau0 for r in results], "o:", label="DNS 10-90% heat release")
    axs[1].plot(Tr, [(r["mz"][3] - r["mz"][2]) / tau0 for r in results], "s:", label="multizone 10-90%")
    axs[1].set_ylabel(r"burn duration / $\tau_0$")
    axs[2].plot(Tr, [r["defl_total"] for r in results], "o-", label=rf"DNS: $S_d^* < {args.ratio:g}\,S_L$")
    axs[2].plot(Tr, [r["beta_share"] for r in results], "s--", label=r"$\beta > 1$, initial field (mass)")
    axs[2].plot(Tr, [r["beta_share_10"] for r in results], "^:", label=r"$\beta > 1$ at 10% heat release")
    axs[2].set_ylabel("deflagrative share of heat release")
    axs[2].set_ylim(0, 1)
    for a in axs:
        a.set_xlabel(r"$T'$ [K]")
        a.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "autoignition_timing.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    centers = 0.5 * (bins[1:] + bins[:-1])
    for r, col in zip(results, colors):
        pdf = r["hist"] / (r["hist"].sum() * (bins[1] - bins[0]))
        ax.plot(centers, pdf, color=col, lw=2, label=r["label"])
    ax.axvline(np.log10(args.ratio), color="k", lw=1, ls="--")
    ax.set_xlabel(r"$\log_{10}(|S_d^*| / S_L)$")
    ax.set_ylabel("HRR-weighted PDF")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "autoignition_speed.png"), dpi=150)


if __name__ == "__main__":
    main()
