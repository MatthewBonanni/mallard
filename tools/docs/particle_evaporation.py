"""Figures of evaporating droplets, in the particles pages of the user guide
(docs/images/guide/) and the theory manual (docs/images/theory/).

    python tools/docs/particle_evaporation.py [--mallard build/src/mallard] [names...]

Water droplets in air of mechanisms/h2o2.yaml at 1 atm. From the
formulas of src/particles/particle_evaporation.h: the surface's vapor mass
fraction and Spalding's transfer number of water, and Abramzon and Sirignano's
film corrections (droplet_transfer_numbers). From short Mallard runs (a 2D build,
up to 3 min on 2 CPU threads): an isolated droplet at rest under
the d^2 law and Abramzon and Sirignano's model against the wet-bulb temperatures
and d^2 rates of its film, computed here with Cantera's thermodynamics and
mixture-averaged transport of the same mechanism, as test/particles_evaporation_test.cpp
does (droplet_evaporation); and the vapor of two-way coupled droplets in a
sheared stream of hot air (droplet_vapor_field). Needs Cantera and the restart
reader (tools/mallard/restart.py).
"""
import argparse
import csv
import functools
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

import figstyle
from figstyle import bare

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from mallard import restart  # noqa: E402
from mallard.vtu import outlines, read_vtu_cells, snapshot  # noqa: E402

THEORY = figstyle.OUT.parent / "theory"
MECHANISM = ROOT / "mechanisms" / "h2o2.yaml"

# Water, as in the guide and test/particles_evaporation_test.cpp; air at 1000 K and 1 atm
RHO_L, CP_L, L_B, T_B, P_B, T_C, P_C = 958.0, 4216.0, 2.257e6, 373.15, 101325.0, 647.1, 22.064e6
ANTOINE = (10.19621, 1730.63, -39.724)
P_GAS, T_GAS = 101325.0, 1000.0
AIR = {"O2": 0.21, "N2": 0.79}
LIQUID = (f'[particles.liquid]\nspecies = "H2O"\ndensity = {RHO_L}\nspecific_heat = {CP_L}\nlatent_heat = {L_B}\n'
          f'boiling_temperature = {T_B}\ncritical_temperature = {T_C}\ncritical_pressure = {P_C}\n')


def run(binary, toml, workdir):
    (Path(workdir) / "input.toml").write_text(toml)
    subprocess.run([str(binary), "-i", "input.toml", "--kokkos-num-threads=2"], cwd=workdir, check=True,
                   stdout=subprocess.DEVNULL)


def film_correction(B):
    """F(B) = (1 + B)^0.7 ln(1 + B) / B (particles::film_correction)."""
    B = np.asarray(B, dtype=float)
    return np.where(np.abs(B) < 1e-6, 1.0 + 0.2 * B, (1.0 + B) ** 0.7 * np.log1p(B) / np.where(B == 0, 1, B))


def log_ratio(B):
    """ln(1 + B) / B (particles::log_ratio)."""
    B = np.asarray(B, dtype=float)
    return np.where(np.abs(B) < 1e-6, 1.0 - 0.5 * B, np.log1p(B) / np.where(B == 0, 1, B))


class Water:
    """Water vapor in the air of the mechanism: saturation, surface and film by the 1/3 rule (Cantera)."""

    def __init__(self):
        import cantera as ct
        self.gas = ct.Solution(str(MECHANISM), transport_model="mixture-averaged")
        self.gas.TPX = T_GAS, P_GAS, AIR
        self.Y_air = self.gas.Y.copy()
        self.k_v = self.gas.species_index("H2O")
        self.W_v = self.gas.molecular_weights[self.k_v]
        self.W_air = self.gas.mean_molecular_weight
        self.R_v = ct.gas_constant / self.W_v
        self.vapor = self.gas.species("H2O").thermo

    def saturation_pressure(self, T, antoine=False):
        if antoine:
            A, B, C = ANTOINE
            return 10.0 ** (A - B / (T + C))
        return P_B * np.exp(L_B / self.R_v * (1.0 / T_B - 1.0 / T))

    def surface_fraction(self, Ts, antoine=False):
        X_s = np.minimum(self.saturation_pressure(Ts, antoine) / P_GAS, 1.0)
        return X_s * self.W_v / (X_s * self.W_v + (1.0 - X_s) * self.W_air)

    def h_vapor(self, T):
        return self.vapor.h(T) / self.W_v

    def latent(self, T):
        """Kirchhoff's latent heat L(T) = h_v(T) - h_l(T)."""
        return self.h_vapor(T) - (self.h_vapor(T_B) - L_B) - CP_L * (T - T_B)

    def film(self, Ts):
        """rho_f, D_f, k_f, cp_f, cp_v(T_f) and B_M of the film of a surface at Ts in dry air."""
        Y_s = float(self.surface_fraction(Ts))
        Yv_f = 2.0 / 3.0 * Y_s
        Y = self.Y_air * (1.0 - Yv_f)
        Y[self.k_v] = Yv_f
        T_f = Ts + (T_GAS - Ts) / 3.0
        self.gas.TPY = T_f, P_GAS, Y
        cp_v = self.vapor.cp(T_f) / self.W_v
        return (self.gas.density, self.gas.mix_diff_coeffs[self.k_v], self.gas.thermal_conductivity, self.gas.cp_mass,
                cp_v, Y_s / (1.0 - Y_s))


def bisect(f, a, b):
    fa = f(a)
    for _ in range(100):
        m = 0.5 * (a + b)
        if (f(m) > 0) == (fa > 0):
            a = m
        else:
            b = m
    return 0.5 * (a + b)


@functools.cache
def wet_bulb(model):
    """(T_wb, K) of a droplet at rest: where the heat into the liquid vanishes, and its rate of fall of d^2."""
    w = Water()
    if model == "d2_law":
        def balance(T):
            rho, D, k, cp, cp_v, B_M = w.film(T)
            return k * (T_GAS - T) - rho * D * math.log1p(B_M) * w.latent(T)
    else:
        def balance(T):
            rho, D, k, cp, cp_v, B_M = w.film(T)
            Le = k / (rho * cp * D)
            B_T = (1.0 + B_M) ** (cp_v / cp / Le) - 1.0
            return cp_v * (T_GAS - T) / w.latent(T) - B_T
    T = bisect(balance, 280.0, T_B - 1.0)
    rho, D, _, _, _, B_M = w.film(T)
    return float(T), float(8.0 * rho * D * math.log1p(B_M) / RHO_L)


# ------------------------------------------------------------ formulas

def droplet_transfer_numbers(P):
    w = Water()
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 4.0), gridspec_kw=dict(wspace=0.3))
    Ts = np.linspace(273.15, T_B - 0.2, 300)
    for antoine, ls, name in [(False, "-", "Clausius-Clapeyron"), (True, "--", "Antoine")]:
        Y_s = w.surface_fraction(Ts, antoine)
        a.semilogy(Ts, Y_s / (1.0 - Y_s), color=P["blue"], ls=ls, label=f"$B_M$, {name}")
        a.semilogy(Ts, Y_s, color=P["teal"], ls=ls, label=f"$Y_s$, {name}")
    for model, color in [("d2_law", P["flame"]), ("abramzon_sirignano", P["plum"])]:
        T_wb = wet_bulb(model)[0]
        a.axvline(T_wb, color=color, lw=0.9, ls=":")
        right = model == "d2_law"
        a.text(T_wb + (1.0 if right else -1.0), 1.6e-3, ("$d^2$ law" if right else "A–S") + f"\nwet bulb\n{T_wb:.1f} K",
               color=color, fontsize=10, ha="left" if right else "right")
    a.set_xlabel("surface temperature $T_s$ (K)")
    a.set_ylabel("water vapor at the surface, dry air")
    a.set_ylim(1e-3, 300)
    a.set_title("water, 1 atm (wet bulb: in air at 1000 K)", fontsize=12)
    a.legend(fontsize=9.5, loc="upper left")
    B = np.logspace(-2, np.log10(20), 300)
    b.semilogx(B, film_correction(B), color=P["blue"], label=r"film thickening $F(B)$")
    b.semilogx(B, log_ratio(B), color=P["flame"], label=r"blowing $\ln(1 + B_T) / B_T$")
    Sh0 = 2.0 + 0.6 * math.sqrt(100.0) * 0.7 ** (1.0 / 3.0)
    b.semilogx(B, (2.0 + (Sh0 - 2.0) / film_correction(B)) / Sh0, color=P["teal"], ls="--",
               label=r"$Sh^* / Sh_0$ at $Re_p$ = 100, $Sc$ = 0.7")
    b.axhline(1, color=P["muted"], lw=0.8, ls=":")
    b.set_xlabel("transfer number $B$")
    b.set_ylim(0, 1.75)
    b.set_title("Abramzon & Sirignano's film corrections", fontsize=12)
    b.legend(fontsize=9.5, loc="upper left")
    return fig


# ------------------------------------------------------------ runs

def quiescent(d0, T0, model, dt, n_steps, interval):
    """One droplet at rest at 300 K or its wet-bulb temperature in a periodic box of air at rest (one-way)."""
    return f"""[run]
n_steps = {n_steps}
dt = {dt!r}
[mesh]
type = "cartesian"
Nx = 8
Ny = 8
Lx = 1.0
Ly = 1.0
[[periodic]]
zones = ["left", "right"]
[[periodic]]
zones = ["bottom", "top"]
[initialize]
p = {P_GAS}
T = {T_GAS}
u = [0.0, 0.0]
X = {{ O2 = 0.21, N2 = 0.79 }}
[numerics]
preset = "first_order"
[physics]
type = "navier_stokes"
gas = "mixture"
mechanism = "{MECHANISM}"
transport = "mixture_averaged"
[output]
check_interval = 1000000
[particles]
type = "droplet"
diameter = {d0!r}
temperature = {T0!r}
evaporation = "{model}"
velocity = [0.0, 0.0]
points = [[0.43, 0.56]]
[[particles.probes]]
name = "drop"
fields = ["D_P", "T_P"]
interval = {interval}
""" + LIQUID


D0 = 50e-6


@functools.cache
def history(binary, model, T0):
    """(t, d, T_p) of a 50 µm droplet at rest released at T0, until it is gone."""
    t_life = D0 ** 2 / wet_bulb(model)[1]
    n_steps = 600
    work = Path(tempfile.mkdtemp())
    run(binary, quiescent(D0, T0, model, 1.2 * t_life / n_steps, n_steps, 3), work)
    return probe_rows(work / "particle_probe_drop.csv", D0, T0)


def probe_rows(path, d0, T0):
    """(t, d, T_p) of a probe's samples while its droplet lives, from the release."""
    with open(path) as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
    names = {name: k for k, name in enumerate(rows[0])}
    col = lambda *keys: next(names[k] for k in keys if k in names)  # noqa: E731 (COUNT, MEAN_<field> since #1466)
    n, d, T = col("COUNT", "count"), col("MEAN_D_P", "D_P mean"), col("MEAN_T_P", "T_P mean")
    live = [r for r in rows[1:] if int(r[n]) > 0]
    out = np.array([[float(r[names["t"]]), float(r[d]), float(r[T])] for r in live])
    return np.vstack([[0.0, d0, T0], out]).T


def droplet_evaporation(P, args):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 4.0), gridspec_kw=dict(wspace=0.28))
    for model, color, name in [("d2_law", P["flame"], "$d^2$ law"), ("abramzon_sirignano", P["blue"],
                                                                     "Abramzon & Sirignano")]:
        T_wb, K = wet_bulb(model)
        t, d, _ = history(args.mallard, model, T_wb)
        ms = 1e3
        a.plot(t * ms, (d / D0) ** 2, "o", color=color, ms=3.5, mfc="none", markevery=6,
               label=f"{name}, from $T_{{wb}}$")
        tt = np.linspace(0, D0 ** 2 / K, 50)
        a.plot(tt * ms, 1 - K * tt / D0 ** 2, color=color, lw=1.2)
        t, d, Tp = history(args.mallard, model, 300.0)
        a.plot(t * ms, (d / D0) ** 2, color=color, ls="--", lw=1.4, label=f"{name}, from 300 K")
        b.plot(t * ms, Tp, color=color, lw=1.6, label=name)
        b.axhline(T_wb, color=color, lw=0.9, ls=":")
        b.text(t[-1] * ms, T_wb + 0.8, f"wet bulb {T_wb:.1f} K", color=color, fontsize=10, ha="right", va="bottom")
    a.set_xlabel("t (ms)")
    a.set_ylabel(r"$(d / d_0)^2$")
    a.set_ylim(0, 1.05)
    a.set_xlim(left=0)
    a.text(0.98, 0.97, "lines: $d^2 = d_0^2 - K t$ at the wet bulb\nmarkers, dashed: Mallard", transform=a.transAxes,
           ha="right", va="top", fontsize=10, color=P["muted"])
    a.legend(fontsize=9.5, loc="lower left", handlelength=1.6)
    a.set_title(r"diameter, 50 µm water droplet at rest", fontsize=12)
    b.set_xlabel("t (ms)")
    b.set_ylabel("droplet temperature $T_p$ (K)")
    b.set_xlim(left=0)
    b.set_title("released at 300 K in air at 1000 K", fontsize=12)
    b.legend(fontsize=10, loc="center right")
    return fig


VAPOR_BOX, VAPOR_N, VAPOR_T, VAPOR_D = 0.01, 64, 1e-3, 20e-6


def vortex_cloud():
    """Two-way coupled droplets released at rest in a periodic sheared stream of air at 1000 K."""
    L = VAPOR_BOX
    return f"""[run]
t_stop = {VAPOR_T!r}
cfl = 0.6
[mesh]
type = "cartesian"
Nx = {VAPOR_N}
Ny = {VAPOR_N}
Lx = {L}
Ly = {L}
[[periodic]]
zones = ["left", "right"]
[[periodic]]
zones = ["bottom", "top"]
[initialize]
type = "analytical"
p = "{P_GAS}"
T = "{T_GAS}"
u = ["10 + 4 * sin(2 * pi * y / {L})", "0.0"]
X = {{ O2 = 0.21, N2 = 0.79 }}
[numerics]
riemann_solver = "HLLC"
time_integrator = "SSPRK3"
[numerics.face_reconstruction]
type = "MUSCL"
[physics]
type = "navier_stokes"
gas = "mixture"
mechanism = "{MECHANISM}"
transport = "mixture_averaged"
[output]
check_interval = 1000000
[particles]
type = "droplet"
diameter = {VAPOR_D!r}
temperature = 300.0
evaporation = "abramzon_sirignano"
coupling = "two-way"
weight = 3e3
deposition_width = 2e-4
velocity = [0.0, 0.0]
[particles.lattice]
lo = [{0.1 * L}, {0.07 * L}]
hi = [{0.55 * L}, {0.93 * L}]
n = [4, 7]
[[write_data]]
prefix = "vapor"
format = "vtu"
fields = ["Y_H2O", "PARTICLE_SOURCE_RHOY_H2O"]
time_interval = {VAPOR_T!r}
[[write_data]]
prefix = "drops"
format = "restart"
time_interval = {VAPOR_T!r}
""" + LIQUID


@functools.cache
def vapor_run(binary):
    work = Path(tempfile.mkdtemp())
    run(binary, vortex_cloud(), work)
    stems = sorted(p.with_suffix("") for p in work.glob("vapor_*.*") if p.suffix in (".vtu", ".pvtu"))
    pts, conn, offs, types, arrays = read_vtu_cells(snapshot(str(stems[-1])))
    r = restart.read(str(sorted(work.glob("drops_*.restart"))[-1]))
    x = np.asarray(r.arrays["PARTICLE_X"], dtype=float).reshape(-1, 2)
    d = (6.0 * np.asarray(r.arrays["PARTICLE_M"], dtype=float) / (math.pi * RHO_L)) ** (1.0 / 3.0)
    return cell_polygons(pts, conn, offs, types), arrays, x, d


def cell_polygons(pts, conn, offs, types):
    polys = [None] * len(offs)
    for cells, ids in outlines(conn, offs, types):
        for c, row in zip(cells, ids):
            polys[c] = pts[row][:, :2]
    return polys


def droplet_vapor_field(P, args):
    polys, arrays, x, d = vapor_run(args.mallard)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.9), gridspec_kw=dict(wspace=0.08))
    mm = 1e3
    panels = [("Y_H2O", "water vapor mass fraction $Y_{H_2O}$", 1.0),
              ("PARTICLE_SOURCE_RHOY_H2O", "vapor source (kg/m³/s)", 1.0)]
    for ax, (name, title, scale) in zip(axes, panels):
        values = np.asarray(arrays[name], dtype=float) * scale
        pc = PolyCollection([p * mm for p in polys], array=values, cmap=P["cmap"], edgecolors="face", linewidths=0.2,
                            rasterized=True)
        ax.add_collection(pc)
        ax.scatter(x[:, 0] * mm, x[:, 1] * mm, s=(d / VAPOR_D) ** 2 * 18, facecolors="none",
                   edgecolors=P["bg"] if name == "Y_H2O" else P["fg"], linewidths=0.8)
        bare(ax)
        ax.set_xlim(0, VAPOR_BOX * mm)
        ax.set_ylim(0, VAPOR_BOX * mm)
        cb = fig.colorbar(pc, ax=ax, orientation="horizontal", fraction=0.05, pad=0.04)
        cb.set_label(title, fontsize=11)
        cb.ax.tick_params(labelsize=10)
        cb.outline.set_visible(False)
    fig.suptitle(f"{VAPOR_D * 1e6:g} µm water droplets released at rest into air at 1000 K flowing right, "
                 f"t = {VAPOR_T * 1e3:g} ms (circles: droplets, area ∝ d²)", fontsize=12, y=0.98)
    return fig


GUIDE_RUNS = {"droplet_evaporation": droplet_evaporation, "droplet_vapor_field": droplet_vapor_field}
THEORY_FIGURES = {"droplet_transfer_numbers": droplet_transfer_numbers}


def figures(mallard=None, names=None):
    args = argparse.Namespace(mallard=str(Path(mallard).resolve()) if mallard else None)
    want = lambda name: not names or name in names  # noqa: E731
    for name, draw in THEORY_FIGURES.items():
        if want(name):
            figstyle.save(name, draw, THEORY)
    if args.mallard:
        for name, draw in GUIDE_RUNS.items():
            if want(name):
                figstyle.save(name, lambda P, draw=draw: draw(P, args))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mallard", help="2D Mallard executable, for the figures from runs")
    ap.add_argument("names", nargs="*")
    args = ap.parse_args()
    figures(args.mallard, args.names)


if __name__ == "__main__":
    main()
