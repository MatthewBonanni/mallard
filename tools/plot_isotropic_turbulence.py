#!/usr/bin/env python3
"""Decaying isotropic turbulence runs against the filtered DNS references.

    plot_isotropic_turbulence.py RUN_DIR [RUN_DIR ...] [--label NAME ...]
        [--ref-dir examples/isotropic_turbulence/reference] [-o hit]

Each RUN_DIR holds a run of examples/isotropic_turbulence with its
integrals.csv and, from tools/isotropic_turbulence_stats.py, stats.csv and
spectra/. Writes
  OUT_reference.png   statistics of each run filtered to 64^3 as the
                      references were: sixth-order differences against
                      Johnsen et al. (2010), spectral derivatives against
                      Subramaniam et al. (2019);
  OUT_convergence.png the unfiltered statistics of each run (the DNS
                      quantities): kinetic energy, enstrophy, dilatation,
                      density and temperature variances, Mt and k_max eta;
  OUT_spectra.png     velocity, vorticity, dilatation and density spectra at
                      t / tau = 2 and 4 against Subramaniam et al.'s;
and prints the differences from the references at their sample times.
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from isotropic_turbulence_restart import parameters

BOX = (2 * np.pi) ** 3
COLORS = ["#2f9e44", "#3b82f6", "#e8590c", "#9c36b5", "#c92a2a", "#868e96"]


def load_csv(path, **kwargs):
    """CSV with a header row, after any leading '#' comment lines."""
    with open(path) as f:
        skip = sum(1 for line in f if line.startswith("#"))
    return np.genfromtxt(path, delimiter=",", names=True, skip_header=skip, **kwargs)


def load_integrals(path, par):
    """Normalized histories from [integrals], keeping the latest rows where a
    resumed run overlaps the previous one."""
    d = load_csv(path)
    keep = np.ones(len(d), bool)
    t = d["t"]
    for i in range(1, len(t)):
        if t[i] <= t[i - 1]:
            keep[:i] &= t[:i] < t[i]
    d = d[keep]
    u0, lam0, mt = par["u_rms"], par["lambda"], par["mt"]
    T_mean = d["temperature"] / BOX
    return {
        "t": d["t"] / par["tau"],
        "kinetic_energy": d["velocity_squared"] / BOX / (3 * u0 ** 2),
        "enstrophy": d["vorticity_squared"] / BOX / (u0 ** 2 / lam0 ** 2),
        "dilatation_variance": d["dilatation_squared"] / BOX / (u0 ** 2 / lam0 ** 2),
        "temperature_variance": (d["temperature_squared"] / BOX - T_mean ** 2)
        / ((par["gamma"] - 1) * par["T0"] * mt ** 2) ** 2,
        "density_variance": (d["density_squared"] / BOX - 1.0) / mt ** 4,
    }


def load_stats(path):
    d = load_csv(path)
    return d[np.argsort(d["t"])]


def load_spectra(run, t_over_tau):
    files = sorted(glob.glob(os.path.join(run, "spectra", "spectra_*.npz")))
    if not files:
        return None
    best = min(files, key=lambda f: abs(float(np.load(f)["t_over_tau"]) - t_over_tau))
    s = np.load(best)
    return s if abs(float(s["t_over_tau"]) - t_over_tau) < 0.03 else None


def style(ax, xlabel=r"$t/\tau$"):
    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.3)


LABELS = {
    "kinetic_energy": r"$\langle u_iu_i\rangle/(3u_{rms,0}^2)$",
    "enstrophy": r"$\langle\omega_i\omega_i\rangle\,\lambda_0^2/u_{rms,0}^2$",
    "dilatation_variance": r"$\langle\theta^2\rangle\,\lambda_0^2/u_{rms,0}^2$",
    "temperature_variance": r"$\langle T'^2\rangle/((\gamma-1)T_0M_{t,0}^2)^2$",
    "density_variance": r"$\langle\rho'^2\rangle/(\rho_0^2 M_{t,0}^4)$",
}


def report(name, t_ref, v_ref, t, v, label):
    """Mean and largest relative difference at the reference times (t/tau >= 0.2)."""
    m = (t_ref >= 0.19) & (t_ref <= t.max() + 1e-6)
    if not m.any():
        return
    err = np.interp(t_ref[m], t, v) / v_ref[m] - 1
    k = int(np.argmax(np.abs(err)))
    print(f"  {label:>12s} {name:22s} mean {100 * np.mean(err):+6.1f}%  max {100 * err[k]:+6.1f}% "
          f"(t/tau = {t_ref[m][k]:.1f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--label", nargs="*")
    ap.add_argument("--ref-dir", default=os.path.join(os.path.dirname(__file__), "..", "examples",
                                                       "isotropic_turbulence", "reference"))
    ap.add_argument("-o", "--output", default="hit")
    args = ap.parse_args()
    labels = args.label or [os.path.basename(os.path.normpath(r)) for r in args.runs]
    par = parameters()
    johnsen = load_csv(os.path.join(args.ref_dir, "johnsen2010_filtered256.csv"))
    wchr = load_csv(os.path.join(args.ref_dir, "subramaniam2019_filtered512.csv"))
    stats = [load_stats(os.path.join(r, "stats.csv")) if os.path.exists(os.path.join(r, "stats.csv")) else None
             for r in args.runs]
    integrals = [load_integrals(os.path.join(r, "integrals.csv"), par) for r in args.runs]

    # Filtered statistics against the references
    names = ["kinetic_energy", "enstrophy", "dilatation_variance", "temperature_variance"]
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
    print("Filtered to 64^3, sixth-order differences, against Johnsen et al. (2010):")
    for j, name in enumerate(names):
        ax = axes[0, j]
        ax.plot(johnsen["t_over_tau"], johnsen[name], "o", mfc="none", mec="k", ms=6,
                label="Johnsen et al. 2010, $256^3$ filtered")
        for s, lab, c in zip(stats, labels, COLORS):
            if s is None:
                continue
            ax.plot(s["t_over_tau"], s[f"{name}_fd6"], color=c, lw=1.8, label=lab)
            report(name, johnsen["t_over_tau"], johnsen[name], s["t_over_tau"], s[f"{name}_fd6"], lab)
        ax.set_ylabel(LABELS[name])
        style(ax)
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_title("filtered to $64^3$, 6th-order differences", loc="left", fontsize=10)
    print("Filtered to 64^3, spectral derivatives, against Subramaniam et al. (2019):")
    for j, name in enumerate(["kinetic_energy", "enstrophy", "dilatation_variance", "density_variance"]):
        ax = axes[1, j]
        if name in wchr.dtype.names:
            ax.plot(wchr["t_over_tau"], wchr[name], "k-", lw=2.5, alpha=0.35,
                    label="Subramaniam et al. 2019, $512^3$ filtered")
        for s, lab, c in zip(stats, labels, COLORS):
            if s is None:
                continue
            ax.plot(s["t_over_tau"], s[f"{name}_spec"], color=c, lw=1.8, label=lab)
            if name in wchr.dtype.names:
                sel = np.arange(4, len(wchr), 4)
                report(name, wchr["t_over_tau"][sel], wchr[name][sel], s["t_over_tau"], s[f"{name}_spec"], lab)
        ax.set_ylabel(LABELS[name])
        style(ax)
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].set_title("filtered to $64^3$, spectral derivatives", loc="left", fontsize=10)
    fig.suptitle(r"Decaying isotropic turbulence, $M_{t,0} = 0.6$, $Re_{\lambda,0} = 100$: filtered statistics")
    fig.tight_layout()
    fig.savefig(args.output + "_reference.png", dpi=130)

    # Unfiltered statistics: grid convergence of the DNS
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
    panels = names + ["density_variance"]
    for j, name in enumerate(panels):
        ax = axes.flat[j]
        for d, s, lab, c in zip(integrals, stats, labels, COLORS):
            ax.plot(d["t"], d[name], color=c, lw=1.8, label=f"{lab} [integrals]")
            if s is not None:
                ax.plot(s["t_over_tau"][::4], s[f"{name}_full"][::4], "o", color=c, ms=3.5, mfc="none",
                        label=f"{lab} spectral")
        ax.set_ylabel(LABELS[name])
        style(ax)
    axes.flat[0].legend(fontsize=7)
    for ax, (key, lab) in zip(axes.flat[5:], [("Mt", r"$M_t$"), ("Re_lambda", r"$Re_\lambda$"),
                                               ("kmax_eta", r"$k_{max}\eta$")]):
        for s, l, c in zip(stats, labels, COLORS):
            if s is not None:
                ax.plot(s["t_over_tau"], s[key], color=c, lw=1.8, label=l)
        ax.set_ylabel(lab)
        style(ax)
    axes.flat[7].axhline(1.0, color="k", lw=0.8, ls=":")
    axes.flat[7].legend(fontsize=8)
    fig.suptitle("Unfiltered statistics on each grid (lines: [integrals] with TENO gradients of the cell "
                 "averages; circles: spectral derivatives of the deconvolved fields)", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.output + "_convergence.png", dpi=130)

    # Spectra. Subramaniam et al. do not state the normalization of their
    # spectra, so each reference curve is scaled by the one factor that best
    # fits the finest run over 3 <= k <= 24 (in log space): the shapes are compared
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
    print("Spectra: factor dividing the reference, and RMS log10 deviation of its shape from the finest run:")
    for i, tt in enumerate((2, 4)):
        path = os.path.join(args.ref_dir, f"subramaniam2019_spectra_t{tt}.csv")
        ref = load_csv(path)
        q = load_csv(path, dtype=None, encoding=None)["quantity"]
        spectra = [load_spectra(r, tt) for r in args.runs]
        finest = [s for s in spectra if s is not None][-1]
        for j, name in enumerate(["velocity", "vorticity", "dilatation", "density"]):
            ax = axes[i, j]
            m = q == name
            k_ref, E_ref = ref["k"][m], ref["E"][m]
            fit = (k_ref >= 3) & (k_ref <= 24)
            d = np.log10(E_ref[fit]) - np.log10(np.interp(k_ref[fit], finest["k"], finest[name]))
            scale = 10 ** d.mean()
            print(f"  t/tau = {tt} {name:10s} {scale:5.2f}  {np.sqrt(np.mean((d - d.mean()) ** 2)):.3f}")
            ax.loglog(k_ref, E_ref / scale, "k*-", lw=2, alpha=0.4, ms=7,
                      label="Subramaniam et al. 2019, $512^3$ filtered (scaled)")
            for s, lab, c in zip(spectra, labels, COLORS):
                if s is not None:
                    ax.loglog(s["k"][1:], s[name][1:], color=c, lw=1.6, label=lab)
            ax.set_title(f"{name}, $t/\\tau = {tt}$ (reference / {scale:.2f})", fontsize=10)
            ax.set_xlim(1, None)
            style(ax, "$k$")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(r"Shell spectra, $E(k) = \sum_{|\mathbf{k}| \approx k} |\hat f|^2/2$", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.output + "_spectra.png", dpi=130)
    print(f"wrote {args.output}_reference.png, {args.output}_convergence.png, {args.output}_spectra.png")


if __name__ == "__main__":
    main()
