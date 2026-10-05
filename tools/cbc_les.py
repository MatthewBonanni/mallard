#!/usr/bin/env python3
"""Decaying isotropic turbulence of Comte-Bellot & Corrsin (J. Fluid Mech. 48, 1971) for LES.

    cbc_les.py init N OUT.restart [--seed 1] [--mt 0.1]
    cbc_les.py spectra RESTART [RESTART ...] -o spectra.csv
    cbc_les.py compare spectra.csv [--budget integrals.csv] [--plot out.png]

Grid turbulence behind a mesh of M = 5.08 cm at U0 = 10 m/s, measured at
t U0 / M = 42, 98 and 171 (examples/cbc_les/reference/cbc_spectra.csv, their
table 3 in SI units). The LES box is periodic with side L = 11 M (mesh
type "cartesian", cells numbered (i * N + j) * N + k, x slowest), and t = 0
is station 1, so stations 2 and 3 are at t = 56 M / U0 = 0.28448 s and
129 M / U0 = 0.65532 s.

init: a random solenoidal velocity whose shell spectrum is the measured
spectrum of station 1 (log-log interpolated) for every mode below the grid
cutoff k_c = pi N / L and zero above (sharp spectral filter), random phases
from the seed, independent of N for the modes two grids share; cells take
the exact averages of that field. Uniform density 1.2 kg/m^3 and pressure
from a fictitious sound speed c = sqrt(<u_i u_i>) / Mt (Mt = 0.1 by default,
so compressibility stays at the 1% level while the Reynolds number is the
experiment's: nu = 1.5e-5 m^2/s). Prints the [physics] values.

spectra: shell spectra E(k) of each snapshot (cell averages turned into point
values by dividing each mode by its cell-average factor), one row per shell
and time.

compare: the LES spectra at stations 2 and 3 against the measurements for
k <= 2/3 k_c, as the RMS of log10(E_LES / E_CBC) and the ratio of the
resolved energy up to k_c to the measured energy over the same band; with
--budget, the time-averaged dissipation budget over stations 1-3 from an
[integrals] file written with budget = true.
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isotropic_turbulence_restart import read_restart, write_restart  # noqa: E402

M = 0.0508
U0 = 10.0
L = 11.0 * M
NU = 1.5e-5
RHO = 1.2
GAMMA = 1.4
STATIONS = {42: 0.0, 98: 56 * M / U0, 171: 129 * M / U0}
REFERENCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "cbc_les", "reference",
                         "cbc_spectra.csv")
K_DRAW = 64  # modes |n_i| <= K_DRAW are drawn, independently of the mesh


def reference():
    """Measured k (1/m) and E(k) (m^3/s^2) at the three stations."""
    data = np.loadtxt(REFERENCE, delimiter=",", skiprows=1)
    return data[:, 0], {42: data[:, 1], 98: data[:, 2], 171: data[:, 3]}


def interpolate(k, k_ref, e_ref):
    """Log-log interpolation of a measured spectrum, extrapolated as k^4 below its range and with its last slope above."""
    lk, le = np.log(k_ref), np.log(e_ref)
    out = np.zeros_like(k)
    pos = k > 0
    lq = np.log(k[pos])
    val = np.interp(lq, lk, le)
    lo = lq < lk[0]
    val[lo] = le[0] + 4.0 * (lq[lo] - lk[0])
    hi = lq > lk[-1]
    slope = (le[-1] - le[-2]) / (lk[-1] - lk[-2])
    val[hi] = le[-1] + slope * (lq[hi] - lk[-1])
    out[pos] = np.exp(val)
    return out


def integer_waves(n):
    k = np.fft.fftfreq(n, 1.0 / n)
    return k[:, None, None], k[None, :, None], k[None, None, :]


def draw_modes(seed, k_ref, e_ref):
    """Solenoidal Hermitian coefficients on |n_i| <= K_DRAW with shell spectrum E(k) of the reference."""
    rng = np.random.default_rng(seed)
    m = 2 * K_DRAW + 1
    n1 = np.arange(-K_DRAW, K_DRAW + 1, dtype=float)
    nx, ny, nz = np.meshgrid(n1, n1, n1, indexing="ij")
    nn = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    g = rng.standard_normal((3, m, m, m)) + 1j * rng.standard_normal((3, m, m, m))
    nv = np.stack([nx, ny, nz])
    with np.errstate(invalid="ignore", divide="ignore"):
        g -= nv * (nv * g).sum(axis=0) / nn ** 2
        g /= np.sqrt((np.abs(g) ** 2).sum(axis=0))
    g[:, K_DRAW, K_DRAW, K_DRAW] = 0.0
    flipped = np.conj(g[:, ::-1, ::-1, ::-1])
    upper = (nx > 0) | ((nx == 0) & (ny > 0)) | ((nx == 0) & (ny == 0) & (nz > 0))
    g = np.where(upper, g, flipped)
    # Energy per mode: E(k) dk spread over the shell area 4 pi n^2 (in units of the integer lattice)
    dk = 2 * np.pi / L
    k = nn * dk
    e = interpolate(k, k_ref, e_ref)
    with np.errstate(invalid="ignore", divide="ignore"):
        amplitude = np.sqrt(2.0 * e * dk / (4 * np.pi * nn ** 2))
    amplitude[K_DRAW, K_DRAW, K_DRAW] = 0.0
    return g * amplitude, nn


def init(args):
    n = args.n
    k_ref, e_ref = reference()
    modes, nn = draw_modes(args.seed, k_ref, e_ref[42])
    keep = n // 2 - 1
    cutoff = n / 2.0
    idx = np.arange(-keep, keep + 1)
    sel = (idx + K_DRAW)
    sub = modes[:, sel[:, None, None], sel[None, :, None], sel[None, None, :]]
    sub = np.where(nn[sel[:, None, None], sel[None, :, None], sel[None, None, :]] <= cutoff, sub, 0.0)
    u_hat = np.zeros((3, n, n, n), dtype=complex)
    u_hat[:, (idx % n)[:, None, None], (idx % n)[None, :, None], (idx % n)[None, None, :]] = sub * n ** 3
    kx, ky, kz = integer_waves(n)
    avg = np.sinc(kx / n) * np.sinc(ky / n) * np.sinc(kz / n)
    u = np.real(np.fft.ifftn(u_hat * avg, axes=(1, 2, 3)))
    uu = (u ** 2).sum(axis=0).mean()
    c = np.sqrt(uu) / args.mt
    p = RHO * c * c / GAMMA
    fields = {"RHO": np.full(n ** 3, RHO)}
    for d, name in enumerate(["RHOU_X", "RHOU_Y", "RHOU_Z"]):
        fields[name] = RHO * u[d].ravel()
    fields["RHOE"] = p / (GAMMA - 1.0) + 0.5 * RHO * (u ** 2).sum(axis=0).ravel()
    write_restart(args.out, fields)
    T = 300.0
    print(f"wrote {args.out}: {n}^3 cells of side {L / n:.6g} m, k_c = {np.pi * n / L:.6g} 1/m, seed {args.seed}")
    print(f"u_rms = {np.sqrt(uu / 3):.6g} m/s, c = {c:.6g} m/s")
    print(f"[physics] gamma = {GAMMA}, p_ref = {p:.16g}, T_ref = {T}, rho_ref = {RHO}, mu = {RHO * NU:.16g}, "
          f"Pr = 0.71")
    print(f"stations: t = {STATIONS[98]:.16g} (98), {STATIONS[171]:.16g} (171)")


def rescale(args):
    """The velocity of a snapshot with each shell rescaled to the spectrum of station 1, at t = 0."""
    fields, _, _ = read_restart(args.input)
    n = round(len(fields["RHO"]) ** (1 / 3))
    u = np.stack([fields[f"RHOU_{x}"] / fields["RHO"] for x in "XYZ"]).reshape(3, n, n, n)
    kx, ky, kz = integer_waves(n)
    avg = np.sinc(kx / n) * np.sinc(ky / n) * np.sinc(kz / n)
    u_hat = np.fft.fftn(u, axes=(1, 2, 3)) / avg
    nn = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    # Remove the dilatational part, then scale each unit-width shell to the measured energy
    with np.errstate(invalid="ignore", divide="ignore"):
        kv = np.stack(np.broadcast_arrays(kx, ky, kz)).astype(float)
        u_hat -= kv * (kv * u_hat).sum(axis=0) / nn ** 2
    u_hat[:, 0, 0, 0] = 0.0
    shell = np.rint(nn).astype(int)
    energy = np.bincount(shell.ravel(), 0.5 * (np.abs(u_hat / n ** 3) ** 2).sum(axis=0).ravel())
    k_ref, e_ref = reference()
    dk = 2 * np.pi / L
    target = interpolate(np.arange(len(energy)) * dk, k_ref, e_ref[42]) * dk
    with np.errstate(invalid="ignore", divide="ignore"):
        factor = np.where((energy > 0) & (np.arange(len(energy)) <= n / 2), np.sqrt(target / energy), 0.0)
    u_hat *= factor[shell]
    u = np.real(np.fft.ifftn(u_hat * avg, axes=(1, 2, 3)))
    p = (GAMMA - 1.0) * np.mean(fields["RHOE"] - 0.5 * (fields["RHOU_X"] ** 2 + fields["RHOU_Y"] ** 2 +
                                                          fields["RHOU_Z"] ** 2) / fields["RHO"])
    out = {"RHO": np.full(n ** 3, RHO)}
    for d, name in enumerate(["RHOU_X", "RHOU_Y", "RHOU_Z"]):
        out[name] = RHO * u[d].ravel()
    out["RHOE"] = p / (GAMMA - 1.0) + 0.5 * RHO * (u ** 2).sum(axis=0).ravel()
    write_restart(args.output, out)
    print(f"wrote {args.output}: u_rms = {np.sqrt((u ** 2).sum(axis=0).mean() / 3):.6g} m/s, p = {p:.10g}")


def spectrum(path):
    fields, t, step = read_restart(path)
    n = round(len(fields["RHO"]) ** (1 / 3))
    u = np.stack([fields[f"RHOU_{x}"] / fields["RHO"] for x in "XYZ"]).reshape(3, n, n, n)
    kx, ky, kz = integer_waves(n)
    avg = np.sinc(kx / n) * np.sinc(ky / n) * np.sinc(kz / n)
    u_hat = np.fft.fftn(u, axes=(1, 2, 3)) / n ** 3 / avg
    energy = 0.5 * (np.abs(u_hat) ** 2).sum(axis=0)
    shell = np.rint(np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)).astype(int)
    dk = 2 * np.pi / L
    e = np.bincount(shell.ravel(), energy.ravel()) / dk
    k = np.arange(len(e)) * dk
    return t, step, n, k[1:n // 2], e[1:n // 2]


def spectra(args):
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "step", "n", "k", "E"])
        seen = set()
        for path in args.restarts:
            t, step, n, k, e = spectrum(path)
            if step in seen:
                continue
            seen.add(step)
            for ki, ei in zip(k, e):
                w.writerow([f"{t:.10g}", step, n, f"{ki:.8g}", f"{ei:.8g}"])


def compare(args):
    k_ref, e_ref = reference()
    data = np.loadtxt(args.spectra, delimiter=",", skiprows=1)
    rows = []
    for station in (98, 171):
        t_s = STATIONS[station]
        times = np.unique(data[:, 0])
        t = times[np.argmin(np.abs(times - t_s))]
        sel = data[:, 0] == t
        n = int(data[sel, 2][0])
        k, e = data[sel, 3], data[sel, 4]
        kc = np.pi * n / L
        band = (k <= 2.0 / 3.0 * kc) & (k >= k_ref[0])
        e_meas = interpolate(k, k_ref, e_ref[station])
        rms = np.sqrt(np.mean(np.log10(e[band] / e_meas[band]) ** 2))
        bias = np.mean(np.log10(e[band] / e_meas[band]))
        inband = (k <= kc) & (k >= k_ref[0])
        ratio = np.trapezoid(e[inband], k[inband]) / np.trapezoid(e_meas[inband], k[inband])
        worst = np.max(np.abs(e[band] / e_meas[band] - 1.0))
        rows.append((station, t, n, rms, bias, worst, ratio))
        print(f"station {station} (t = {t:.5g} s, {n}^3): log10 RMS {rms:.3f}, mean bias {bias:+.3f}, "
              f"worst ratio deviation {worst:.2f} for k <= 2/3 k_c; resolved energy / measured up to k_c {ratio:.3f}")
    if args.budget:
        b = np.genfromtxt(args.budget, delimiter=",", names=True)
        sel = (b["t"] >= 0.0) & (b["t"] <= STATIONS[171] * 1.0001)
        t = b["t"][sel]
        eps_sgs = -b["ke_rate_sgs"][sel]
        eps_mol = -b["ke_rate_viscous"][sel]
        eps_num = b["eps_numerical"][sel]
        pi = b["pressure_dilatation"][sel]
        mean = lambda x: np.trapezoid(x, t) / (t[-1] - t[0])  # noqa: E731
        total = mean(eps_sgs) + mean(eps_mol) + mean(eps_num)
        print(f"budget over t = 0 to station 3: eps_sgs {mean(eps_sgs) / total:.3f}, eps_mol {mean(eps_mol) / total:.3f}, "
              f"eps_num {mean(eps_num) / total:.3f} of the total; eps_num / eps_sgs = "
              + (f"{mean(eps_num) / mean(eps_sgs):.3f}" if mean(eps_sgs) > 0 else "n/a (no model)")
              + f"; |Pi| / total = {mean(np.abs(pi)) / total:.3f}")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 4))
        for station, color in zip((42, 98, 171), ("C0", "C1", "C2")):
            ax.loglog(k_ref, e_ref[station], "o", color=color, mfc="none", label=f"CBC tU0/M = {station}")
            times = np.unique(data[:, 0])
            t = times[np.argmin(np.abs(times - STATIONS[station]))]
            sel = data[:, 0] == t
            ax.loglog(data[sel, 3], data[sel, 4], "-", color=color)
        ax.set_xlabel("k (1/m)")
        ax.set_ylabel("E(k) (m^3/s^2)")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("init")
    a.add_argument("n", type=int)
    a.add_argument("out")
    a.add_argument("--seed", type=int, default=1)
    a.add_argument("--mt", type=float, default=0.1)
    a = sub.add_parser("rescale")
    a.add_argument("input")
    a.add_argument("output")
    a = sub.add_parser("spectra")
    a.add_argument("restarts", nargs="+")
    a.add_argument("-o", "--output", required=True)
    a = sub.add_parser("compare")
    a.add_argument("spectra")
    a.add_argument("--budget")
    a.add_argument("--plot")
    args = ap.parse_args()
    {"init": init, "rescale": rescale, "spectra": spectra, "compare": compare}[args.cmd](args)


if __name__ == "__main__":
    main()
