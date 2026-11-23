"""Figures of tabulated chemistry (docs/guide/flamelet.md, docs/design/flamelet.md,
docs/design/table_format.md), from a sample FPV table built the way the design
notes' section 2 says a generator builds one:

- flamelet_s_curve: the S-curve of the steady counterflow flamelets (peak
  temperature and source of C against the stoichiometric scalar dissipation
  rate: the stable, unstable and extinguished branches) and the temperature
  of flamelets of each branch against the mixture fraction;
- flamelet_progress: the table at the stoichiometric Z against Lambda, its
  nodes from the stable and the unstable branch;
- flamelet_table: the table's fields at zero variance over (Z, Lambda), with
  the Lambda of the extinction flamelet and of the last unstable one;
- flamelet_lookup: the FPV map from a transported C to Lambda, and the
  multilinear lookup in a cell of the (Z, Sz, Lambda) grid;
- flamelet_variance: the fields of the most burning flamelet against Z at
  several normalized variances Sz (beta-PDF integration);
- flamelet_tfles_sensor: the thickened flame's progress sensor
  (ThickenedFlame::progress_sensor, src/physics/tfles.h) against the source of C
  across the table's Lambda axis.

    python tools/docs/flamelet.py [--compute] [names...]

--compute solves the flamelets with Cantera (about 1 min) and writes
flamelet.json next to this script; without it the figures read that file, so
drawing them needs NumPy, SciPy and Matplotlib only. The flamelets: H2/N2
(1:1 by mole) against air, both at 300 K and 1 atm, mechanisms/h2o2.yaml, unity
Lewis number (as Mallard's transport of the flamelet gas), progress variable
C = Y_H2O. Cantera's two-point flame control walks the stable branch from a
low strain to extinction and down the unstable branch; the mixing solution
closes the table at C = 0. Mallard generates no tables: this script writes no
table file, only the figures' data.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import betainc

import figstyle
from figstyle import label

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "flamelet.json"
FUEL, OXIDIZER, T_IN, P = "H2:0.5,N2:0.5", "O2:0.21,N2:0.79", 300.0, 101325.0
SPECIES = ["H2O", "OH"]
# Storage grid in Z: fine around Z_st (0.304) and in the lean side, where the flamelets vary
Z_GRID = np.unique(np.concatenate([np.linspace(0, 0.1, 11), np.linspace(0.1, 0.6, 67), np.linspace(0.6, 1, 17)]))
LAMBDA = np.linspace(0, 1, 101)


# ------------------------------------------------------------------- Cantera
def compute():
    import cantera as ct

    gas = ct.Solution(str(ROOT / "mechanisms" / "h2o2.yaml"), "ohmech")
    gas.transport_model = "unity-Lewis-number"
    gas.TPX = T_IN, P, FUEL
    Yf, rf, hf = gas.Y.copy(), gas.density, gas.enthalpy_mass
    gas.TPX = T_IN, P, OXIDIZER
    Yo, ro, ho = gas.Y.copy(), gas.density, gas.enthalpy_mass
    gas.set_equivalence_ratio(1.0, FUEL, OXIDIZER)
    Z_st = gas.mixture_fraction(Yf, Yo, "mass", "Bilger")
    idx = {s: gas.species_index(s) for s in SPECIES}
    W_C = gas.molecular_weights[idx["H2O"]]

    def record(states, Z, chi_st=None):
        order = np.argsort(Z)
        Zs = np.maximum.accumulate(Z[order])
        Zs, keep = np.unique(Zs, return_index=True)
        out = {}
        for name, values in states.items():
            out[name] = np.interp(Z_GRID, Zs, np.asarray(values)[order][keep]).tolist()
        out["T_max"] = float(np.max(states["T"]))
        out["chi_st"] = chi_st
        return out

    def C_st(fl):
        return float(np.interp(Z_st, Z_GRID, fl["Y_H2O"]))

    def flamelet(f):
        Z, rates = [], f.net_production_rates
        for j in range(len(f.grid)):
            gas.TPY = f.T[j], P, f.Y[:, j]
            Z.append(gas.mixture_fraction(Yf, Yo, "mass", "Bilger"))
        Z = np.array(Z)
        D = f.thermal_conductivity / (f.density * f.cp_mass)
        chi = 2 * D * np.gradient(Z, f.grid) ** 2
        order = np.argsort(Z)
        chi_st = float(np.interp(Z_st, Z[order], chi[order]))
        states = dict(T=f.T, rho=f.density, omega_C=W_C * rates[idx["H2O"]], HRR=f.heat_release_rate,
                      **{f"Y_{s}": f.Y[idx[s]] for s in SPECIES})
        return record(states, Z, chi_st)

    # The mixing solution: enthalpy and composition linear in Z (unity Lewis number), frozen
    mixing = {k: [] for k in ["T", "rho", "omega_C", "HRR"] + [f"Y_{s}" for s in SPECIES]}
    for Z in Z_GRID:
        gas.HPY = Z * hf + (1 - Z) * ho, P, Z * Yf + (1 - Z) * Yo
        mixing["T"].append(gas.T)
        mixing["rho"].append(gas.density)
        mixing["omega_C"].append(0.0)
        mixing["HRR"].append(0.0)
        for s in SPECIES:
            mixing[f"Y_{s}"].append(gas.Y[idx[s]])
    mixing = record(mixing, Z_GRID)

    f = ct.CounterflowDiffusionFlame(gas, width=0.1)
    f.P = P
    f.fuel_inlet.X, f.fuel_inlet.T = FUEL, T_IN
    f.oxidizer_inlet.X, f.oxidizer_inlet.T = OXIDIZER, T_IN
    f.fuel_inlet.mdot = 0.01
    f.oxidizer_inlet.mdot = 0.01 * np.sqrt(ro / rf)
    f.set_refine_criteria(ratio=3, slope=0.06, curve=0.12, prune=0.02)
    f.solve(loglevel=0, auto=True)
    flamelets = [flamelet(f)]
    f.two_point_control_enabled = True
    # March the control temperatures down (then, where that stalls, up), keeping each solution whose C_st
    # falls: the stable branch to extinction and the unstable branch until the flamelets stop changing
    sign, dT = 1.0, 10.0
    while len(flamelets) < 400:
        saved = f.to_array()
        Tc = f.T.min() + 0.95 * (f.T.max() - f.T.min())
        f.set_left_control_point(Tc)
        f.set_right_control_point(Tc)
        f.left_control_point_temperature -= sign * dT
        f.right_control_point_temperature -= sign * dT
        try:
            f.solve(loglevel=0)
            candidate = flamelet(f)
            ok = C_st(candidate) < C_st(flamelets[-1]) - 1e-6
        except ct.CanteraError:
            ok = False
        if ok:
            flamelets.append(candidate)
            dT = min(1.3 * dT, 20.0)
            print(f"T_max {candidate['T_max']:7.1f} K  chi_st {candidate['chi_st']:9.3g} 1/s  C_st {C_st(candidate):.4f}",
                  flush=True)
        else:
            f.from_array(saved)
            dT /= 2
            if dT < 0.01:
                if sign < 0:
                    break
                sign, dT = -1.0, 5.0
    # Thinned: a flamelet where C_st or chi_st has changed enough since the last one kept, the extinction point
    # and the last
    kept = [flamelets[0]]
    ext = max(flamelets, key=lambda fl: fl["chi_st"])
    for fl in flamelets[1:]:
        if (fl is ext or fl is flamelets[-1] or C_st(fl) < C_st(kept[-1]) - 0.0025
                or abs(np.log(fl["chi_st"] / kept[-1]["chi_st"])) > 0.35):
            kept.append(fl)
    short = lambda v: [float(f"{x:.4g}") for x in v] if isinstance(v, list) else float(f"{v:.6g}")  # noqa: E731
    DATA.write_text(json.dumps(dict(
        Z=short(Z_GRID.tolist()), Z_st=Z_st, fuel=FUEL, oxidizer=OXIDIZER, T=T_IN, p=P,
        mixing={k: short(v) for k, v in mixing.items() if v is not None},
        flamelets=[{k: short(v) for k, v in fl.items()} for fl in kept]), separators=(",", ":")) + "\n")
    print(f"wrote {DATA.name}: {len(kept)} of {len(flamelets)} flamelets")


# ------------------------------------------------------------------- table
def load():
    d = json.loads(DATA.read_text())
    d["Z"] = np.array(d["Z"])
    for fl in d["flamelets"] + [d["mixing"]]:
        for k, v in fl.items():
            if isinstance(v, list):
                fl[k] = np.array(v)
    i_st = lambda fl: np.interp(d["Z_st"], d["Z"], fl["Y_H2O"])  # noqa: E731
    d["mixing"]["C_st"] = 0.0
    for fl in d["flamelets"]:
        fl["C_st"] = i_st(fl)
    # Ordered by C_st, the flamelet parameter, from the mixing solution to the most burning flamelet
    d["by_C_st"] = [d["mixing"]] + sorted(d["flamelets"], key=lambda fl: fl["C_st"])
    return d


def beta_weights(z, mean, Sz):
    """Weights w with sum(w * phi) = the beta-PDF mean of phi piecewise linear on the nodes z (design section 2)."""
    if Sz <= 0:
        w = np.zeros_like(z)
        i = min(max(np.searchsorted(z, mean) - 1, 0), len(z) - 2)
        t = (mean - z[i]) / (z[i + 1] - z[i])
        w[i], w[i + 1] = 1 - t, t
        return w
    if mean <= 0 or mean >= 1 or Sz >= 1:
        w = np.zeros_like(z)
        w[0], w[-1] = 1 - mean, mean
        return w
    a, b = mean * (1 / Sz - 1), (1 - mean) * (1 / Sz - 1)
    F = betainc(a, b, z)
    G = mean * betainc(a + 1, b, z)
    dF, dG, h = np.diff(F), np.diff(G), np.diff(z)
    w = np.zeros_like(z)
    w[:-1] += (z[1:] * dF - dG) / h
    w[1:] += (dG - z[:-1] * dF) / h
    return w


def table(d, fields, Sz=0.0):
    """The table at one Sz: {field: array (n_Z, n_Lambda)}, C linear in Lambda at each Z."""
    z, flm = d["Z"], d["by_C_st"]
    W = np.array([beta_weights(z, m, Sz) for m in z])
    integrated = {f: np.array([W @ fl[f] for fl in flm]) for f in set(fields) | {"Y_H2O"}}
    out = {f: np.zeros((len(z), len(LAMBDA))) for f in fields}
    out["C"] = np.zeros((len(z), len(LAMBDA)))
    for i in range(len(z)):
        C = np.maximum.accumulate(integrated["Y_H2O"][:, i])
        C, keep = np.unique(C, return_index=True)
        Cl = C[0] + LAMBDA * (C[-1] - C[0])
        out["C"][i] = Cl
        for f in fields:
            out[f][i] = np.interp(Cl, C, integrated[f][keep, i]) if len(C) > 1 else integrated[f][0, i]
    return out


# ------------------------------------------------------------------- figures
def branches(d):
    """The flamelets in continuation order, the extinction index, and the Lambda(Z) of a flamelet at Sz = 0."""
    flm = d["flamelets"]
    ext = int(np.argmax([fl["chi_st"] for fl in flm]))
    C_max = np.max([fl["Y_H2O"] for fl in d["by_C_st"]], axis=0)

    def lam(fl):
        return np.divide(fl["Y_H2O"], C_max, out=np.zeros_like(C_max), where=C_max > 0)
    return flm, ext, lam


def s_curve(P, d):
    import matplotlib.pyplot as plt

    flm, ext, _ = branches(d)
    chi = np.array([fl["chi_st"] for fl in flm])
    T = np.array([fl["T_max"] for fl in flm])
    w = np.array([np.interp(d["Z_st"], d["Z"], fl["omega_C"]) for fl in flm])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw=dict(wspace=0.3))
    lo, hi = 0.5 * chi.min(), 3 * chi.max()
    floor = 10 ** np.floor(np.log10(max(w.min(), 1e-6)))
    for ax, y, ylabel, base in ((axes[0], T, r"peak temperature $T_{max}$ (K)", d["T"]),
                                (axes[1], w, r"$\dot\omega_C$ at $Z_{st}$ (kg/m³/s)", None)):
        ax.plot(chi[:ext + 1], y[:ext + 1], color=P["flame"], lw=2.2, label="stable (burning)")
        ax.plot(chi[ext:], y[ext:], color=P["plum"], lw=2.2, ls="--", label="unstable")
        base = floor if base is None else base
        ax.plot([lo, hi], [base, base], color=P["muted"], lw=2.2, ls=":", label="extinguished (mixing)")
        ax.plot(chi[ext], y[ext], "o", color=P["fg"], ms=6)
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        ax.set_xlabel(r"$\chi_{st}$ (1/s)")
        ax.set_ylabel(ylabel)
    axes[0].set_ylim(0, None)
    axes[1].set_yscale("log")
    axes[1].set_ylim(floor, 3 * w.max())
    label(axes[1], lo * 1.3, 1.4 * floor, "0 (extinguished)", P, size=10.5, ha="left", va="bottom", color=P["muted"])
    label(axes[0], chi[ext] * 0.8, T[ext] + 70, "extinction", P, size=12, ha="right")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0.02, 0.17), fontsize=11)
    axes[0].set_title("S-curve: peak temperature", loc="left")
    axes[1].set_title("S-curve: source of C", loc="left")

    c = axes[2]
    for k in np.linspace(0, ext, 5).round().astype(int):
        c.plot(d["Z"], flm[k]["T"], color=P["flame"], lw=1.5)
    for k in np.linspace(ext, len(flm) - 1, 5).round().astype(int)[1:]:
        c.plot(d["Z"], flm[k]["T"], color=P["plum"], lw=1.5, ls="--")
    c.plot(d["Z"], d["mixing"]["T"], color=P["muted"], lw=1.8, ls=":")
    c.axvline(d["Z_st"], color=P["muted"], ls=":", lw=1.0)
    label(c, d["Z_st"] + 0.015, 2000, r"$Z_{st}$", P, size=12, ha="left")
    c.set_xlim(0, 1)
    c.set_ylim(0, 2100)
    c.set_xlabel("mixture fraction Z")
    c.set_ylabel("T (K)")
    c.set_title("Flamelets of each branch", loc="left")
    return fig


def progress(P, d):
    """The table at Z_st and Sz = 0 against Lambda, with the branch each node comes from."""
    import matplotlib.pyplot as plt

    flm, ext, lam = branches(d)
    t = table(d, ["T", "omega_C", "Y_OH"])
    Zs = d["Z_st"]
    at = {f: np.array([np.interp(Zs, d["Z"], t[f][:, k]) for k in range(len(LAMBDA))]) for f in ("T", "omega_C", "Y_OH")}
    node = np.array([np.interp(Zs, d["Z"], lam(fl)) for fl in flm])
    lam_ext, lam_low = node[ext], node.min()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharex=True, gridspec_kw=dict(wspace=0.3))
    panels = (("T", r"$T_0$ (K)", 1.0), ("omega_C", r"$\dot\omega_C$ (kg/m³/s)", 1.0), ("Y_OH", r"$Y_{OH}$", 1.0))
    for ax, (f, ylabel, scale) in zip(axes, panels):
        ax.axvspan(0, lam_low, color=P["faint"], lw=0)
        ax.axvspan(lam_low, lam_ext, color=P["plum"], alpha=0.10, lw=0)
        ax.axvspan(lam_ext, 1, color=P["flame"], alpha=0.10, lw=0)
        ax.plot(LAMBDA, scale * at[f], color=P["fg"], lw=2)
        y = np.array([np.interp(Zs, d["Z"], fl[f]) for fl in flm])
        ax.plot(node[:ext + 1], scale * y[:ext + 1], "o", color=P["flame"], ms=4)
        ax.plot(node[ext:], scale * y[ext:], "o", color=P["plum"], ms=4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, None)
        ax.set_xlabel(r"$\Lambda$ (C_NORM)")
        ax.set_ylabel(ylabel)
    top = axes[0].get_ylim()[1]
    label(axes[0], 0.5 * lam_low, 0.9 * top, "no steady\nflamelet:\ninterpolated", P, size=10.5, color=P["muted"])
    label(axes[0], 0.5 * (lam_low + lam_ext), 0.9 * top, "unstable", P, size=11.5, color=P["plum"])
    label(axes[0], 0.5 * (lam_ext + 1), 0.08 * top, "stable", P, size=11.5, color=P["flame"])
    axes[0].set_title(r"Temperature at $Z_{st}$", loc="left")
    axes[1].set_title(r"Source of C at $Z_{st}$", loc="left")
    axes[2].set_title(r"OH at $Z_{st}$", loc="left")
    return fig


FIELDS = [("T", r"$T_0$ (K)", 1.0), ("rho", r"$\rho_0$ (kg/m³)", 1.0),
          ("omega_C", r"$\dot\omega_C$ (kg/m³/s)", 1.0), ("HRR", "HRR (GW/m³)", 1e-9),
          ("C", r"$C = Y_{H2O}$", 1.0), ("Y_OH", r"$Y_{OH}$", 1.0)]


def table_fields(P, d):
    import matplotlib.pyplot as plt

    t = table(d, [f for f, _, _ in FIELDS if f != "C"])
    flm, ext, lam = branches(d)
    lam_ext, lam_low = lam(flm[ext]), lam(flm[-1])
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), sharex=True, sharey=True,
                             gridspec_kw=dict(hspace=0.32, wspace=0.12))
    for ax, (f, title, scale) in zip(axes.flat, FIELDS):
        m = ax.pcolormesh(d["Z"], LAMBDA, scale * t[f].T, cmap=P["cmap"], shading="gouraud", rasterized=True)
        ax.axvline(d["Z_st"], color="white", ls=":", lw=1.0, alpha=0.8)
        inner = (d["Z"] > 0.01) & (d["Z"] < 0.99)
        ax.plot(d["Z"][inner], lam_ext[inner], color="white", lw=1.2, ls="--", alpha=0.9)
        ax.plot(d["Z"][inner], lam_low[inner], color="white", lw=1.2, ls=":", alpha=0.9)
        ax.set_title(title, loc="left", fontsize=13)
        ax.set_xlim(0, 0.8)
        cb = fig.colorbar(m, ax=ax, pad=0.02, fraction=0.06)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=10)
    for ax in axes[1]:
        ax.set_xlabel("Z")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\Lambda$")
    i = int(np.searchsorted(d["Z"], 0.6))
    label(axes[0, 0], 0.6, 0.5 * (lam_ext[i] + 1), "stable", P, size=11, color="white")
    label(axes[0, 0], 0.6, 0.5 * (lam_ext[i] + lam_low[i]), "unstable", P, size=11, color="white")
    label(axes[0, 0], 0.6, 0.5 * lam_low[i], "interpolated", P, size=11, color="white")
    return fig


def lookup(P, d):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401

    fig = plt.figure(figsize=(12, 4.6))
    a = fig.add_axes([0.06, 0.13, 0.5, 0.76])
    colors = [P["flame"], P["plum"], P["blue"]]
    for Sz, c in zip((0.0, 0.1, 0.4), colors):
        t = table(d, [], Sz)
        a.plot(d["Z"], t["C"][:, -1], color=c, label=rf"$C_{{max}}$, $S_z$ = {Sz:g}")
        a.plot(d["Z"], t["C"][:, 0], color=c, ls=":", lw=1.2)
    t = table(d, [], 0.0)
    Zq = 0.42
    C_max = float(np.interp(Zq, d["Z"], t["C"][:, -1]))
    C_min = float(np.interp(Zq, d["Z"], t["C"][:, 0]))
    Cq = C_min + 0.6 * (C_max - C_min)
    a.annotate("", (Zq, C_max), (Zq, C_min), arrowprops=dict(arrowstyle="<->", color=P["fg"], lw=1.2))
    a.plot(Zq, Cq, "o", color=P["fg"], ms=6)
    label(a, Zq + 0.02, Cq, r"transported $(Z, C)$:  $\Lambda = \frac{C - C_{min}}{C_{max} - C_{min}} = 0.6$",
          P, size=12, ha="left")
    label(a, 0.9, 0.012, r"$C_{min}$ (mixing, every $S_z$)", P, size=11, ha="right", color=P["muted"])
    a.set_xlim(0, 1)
    a.set_ylim(0, None)
    a.set_xlabel("Z")
    a.set_ylabel(r"C = $Y_{H2O}$")
    a.legend(loc="upper right")
    a.set_title(r"The FPV map: bounds of C at $(Z, S_z)$", loc="left")

    b = fig.add_axes([0.62, 0.05, 0.36, 0.9], projection="3d")
    b.set_proj_type("ortho")
    b.view_init(elev=22, azim=-58)
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float)
    for p in corners:
        for q in corners:
            if np.sum(np.abs(p - q)) == 1 and tuple(p) < tuple(q):
                b.plot(*zip(p, q), color=P["muted"], lw=1.2)
    b.scatter(*corners.T, color=P["teal"], s=36, depthshade=False)
    q = np.array([0.45, 0.35, 0.6])
    b.scatter(*q, color=P["flame"], s=50, depthshade=False)
    for p in corners:
        b.plot(*zip(p, q), color=P["flame"], lw=0.6, alpha=0.45)
    b.text(0.5, -0.18, -0.05, "Z", color=P["fg"], ha="center")
    b.text(1.1, 0.5, -0.1, r"$S_z$", color=P["fg"])
    b.text(-0.12, 0, 0.5, r"$\Lambda$", color=P["fg"], ha="right")
    b.text(q[0] + 0.06, q[1], q[2] + 0.06, "query", color=P["flame"])
    b.set_axis_off()
    b.set_box_aspect((1, 1, 1))
    fig.text(0.62, 0.9, r"Multilinear lookup: $2^3$ corners", fontsize=14, color=P["fg"])
    return fig


def variance(P, d):
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw=dict(wspace=0.28))
    cmap = plt.get_cmap(P["cmap"])
    levels = (0.0, 0.01, 0.05, 0.2, 0.6)
    k = int(np.argmin(np.abs(LAMBDA - 0.7)))
    for n, Sz in enumerate(levels):
        t = table(d, ["T", "omega_C"], Sz)
        c = cmap(0.9 * n / (len(levels) - 1))
        a.plot(d["Z"], t["T"][:, -1], color=c, label=rf"$S_z$ = {Sz:g}")
        b.plot(d["Z"], t["omega_C"][:, k], color=c)
    for ax in (a, b):
        ax.axvline(d["Z_st"], color=P["muted"], ls=":", lw=1.2)
        ax.set_xlim(0, 1)
        ax.set_xlabel(r"mean mixture fraction $\tilde Z$")
    a.set_ylabel(r"$T_0$ (K)")
    b.set_ylabel(r"$\dot\omega_C$ (kg/m³/s)")
    a.legend(loc="upper right")
    a.set_title(r"Temperature at $\Lambda = 1$", loc="left")
    b.set_title(r"Source of C at $\Lambda = 0.7$", loc="left")
    return fig


def tfles_sensor(P, d):
    import matplotlib.pyplot as plt

    C_SENSOR = 0.05
    omega = np.minimum(1.0, LAMBDA * (1 - LAMBDA) / (C_SENSOR * (1 - C_SENSOR)))
    t = table(d, ["omega_C"])
    fig, a = plt.subplots(figsize=(7.6, 4.2))
    a.fill_between(LAMBDA, omega, color=P["teal"], alpha=0.12, lw=0)
    a.plot(LAMBDA, omega, color=P["teal"], lw=2.2, label=r"sensor $\Omega(\Lambda)$")
    for Z, c in ((0.15, P["blue"]), (d["Z_st"], P["flame"]), (0.5, P["plum"])):
        w = np.array([np.interp(Z, d["Z"], t["omega_C"][:, k]) for k in range(len(LAMBDA))])
        name = r"$Z_{st}$" if Z == d["Z_st"] else f"Z = {Z:g}"
        a.plot(LAMBDA, w / w.max(), color=c, lw=1.6, ls="--", label=rf"$\dot\omega_C / \max$, {name}")
    a.set_xlim(0, 1)
    a.set_ylim(0, 1.08)
    a.set_xlabel(r"normalized progress variable $\Lambda$ (C_NORM)")
    a.legend(loc="center left", bbox_to_anchor=(0.07, 0.55), fontsize=11)
    a.set_title(r"Thickened-flame sensor against the table's source of C ($S_z = 0$)", loc="left")
    return fig


FIGURES = {
    "flamelet_s_curve": s_curve,
    "flamelet_progress": progress,
    "flamelet_table": table_fields,
    "flamelet_lookup": lookup,
    "flamelet_variance": variance,
    "flamelet_tfles_sensor": tfles_sensor,
}


def figures(names=None):
    d = load()
    for name, draw in FIGURES.items():
        if not names or name in names:
            figstyle.save(name, lambda P, draw=draw: draw(P, d))


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--compute" in args:
        args.remove("--compute")
        compute()
    figures(args)
