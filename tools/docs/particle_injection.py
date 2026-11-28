"""Figures of particle injectors, in the particles page of the user guide
(docs/images/guide/).

    python tools/docs/particle_injection.py --mallard build/src/mallard [names...]

From short Mallard runs (a 2D build, about 2 min in all on 2 CPU threads):

- particle_injection: a spray of a point injector with a cone of velocities
  and a Rosin-Rammler size distribution by mass into air at rest, its parcels
  colored by diameter, and the diameters it drew against the distribution's
  density by number and by mass;
- particle_injection_sites: where each kind of injector puts its particles: a
  boundary zone (the inner circle of an annulus, along its inward normal), a
  filled region (a disk and a tilted rectangle, uniformly in area) and points
  with and without a cone;
- particle_size_distributions: the diameters drawn from each of the five
  distributions against its density, by number and by mass;
- particle_injection_rate: the parcels of a rate of 0.7 parcels per step,
  with a start and an end within steps, against rate times time, with the
  fraction of a parcel carried over, and the mass of a mass_flow injector
  against mass_flow times time.

The parcels are read from restart files (tools/mallard/restart.py).
"""
import argparse
import functools
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import figstyle
from figstyle import label

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from mallard import restart  # noqa: E402

# Rosin-Rammler by mass: size X, spread q, clipped to [D_MIN, D_MAX]
X, Q, D_MIN, D_MAX = 40e-6, 3.0, 5e-6, 150e-6
LX, LY, NOZZLE, SPEED, CONE, T_END = 0.2, 0.1, (0.004, 0.05), 30.0, 12.0, 4e-3

GAS = """[initialize]
type = "constant"
u = [0.0, 0.0]
p = 1.0e5
T = 300.0
[numerics]
preset = "first_order"
[physics]
type = "navier_stokes"
gamma = 1.4
p_ref = 1.0e5
T_ref = 300.0
rho_ref = 1.1614401858304297
mu = 1.8e-5
Pr = 0.72
"""


def box(nx, ny, lx, ly):
    return f"""[mesh]
type = "cartesian"
Nx = {nx}
Ny = {ny}
Lx = {lx}
Ly = {ly}
[[periodic]]
zones = ["left", "right"]
[[periodic]]
zones = ["bottom", "top"]
"""


def restarts(prefix, every):
    return f"""[[write_data]]
prefix = "{prefix}"
format = "restart"
{every}
"""


def spray():
    return f"""[run]
t_stop = {T_END}
cfl = 0.8
{box(40, 20, LX, LY)}{GAS}[particles]
type = "solid"
density = 800.0
seed = 1
[[particles.injectors]]
name = "spray"
point = [{NOZZLE[0]}, {NOZZLE[1]}]
rate = 1e6
speed = {SPEED}
direction = [1.0, 0.0]
cone_angle = {CONE}
diameter = {{distribution = "rosin_rammler", size = {X}, spread = {Q}, basis = "mass", min = {D_MIN}, max = {D_MAX}}}
{restarts("end", f"time_interval = {T_END}")}"""


def run(binary, toml, prefix="end"):
    """The restart files of a run of toml, in order of time."""
    work = Path(tempfile.mkdtemp())
    (work / "input.toml").write_text(toml)
    subprocess.run([str(binary), "-i", "input.toml", "--kokkos-num-threads=2"], cwd=work, check=True,
                   stdout=subprocess.DEVNULL)
    return [restart.read(str(p)) for p in sorted(work.glob(f"{prefix}_*.restart"))]


def parcels(r):
    """Positions, diameters and weights of a restart file's parcels."""
    x = np.asarray(r.arrays["PARTICLE_X"], float).reshape(-1, 2)
    return x, np.asarray(r.arrays["PARTICLE_D"], float), np.asarray(r.arrays["PARTICLE_W"], float)


@functools.cache
def spray_run(binary):
    x, d, _ = parcels(run(binary, spray())[-1])
    return x, d


def rosin_rammler_mass(d):
    """The density by mass of the Rosin-Rammler distribution, f = q/X (d/X)^(q-1) exp(-(d/X)^q)."""
    return Q / X * (d / X) ** (Q - 1) * np.exp(-(d / X) ** Q)


def particle_injection(P, args):
    x, d = spray_run(args.mallard)
    fig = plt.figure(figsize=(11.0, 4.0))
    a = fig.add_axes([0.05, 0.12, 0.52, 0.8])
    cax = fig.add_axes([0.585, 0.22, 0.01, 0.6])
    b = fig.add_axes([0.71, 0.15, 0.28, 0.75])
    order = np.argsort(-d)
    sc = a.scatter(x[order, 0] * 1e3, x[order, 1] * 1e3, c=d[order] * 1e6, s=4, cmap=P["cmap"],
                   norm=LogNorm(D_MIN * 1e6, 80), linewidths=0)
    reach = 75.0
    for s in (-1, 1):
        a.plot([NOZZLE[0] * 1e3, NOZZLE[0] * 1e3 + reach], [NOZZLE[1] * 1e3, NOZZLE[1] * 1e3 +
               s * reach * np.tan(np.radians(CONE))], color=P["muted"], lw=0.8, ls=":")
    label(a, NOZZLE[0] * 1e3 + 40, NOZZLE[1] * 1e3 + 14, f"cone half angle {CONE:g}°", P, color=P["muted"],
          size=10, ha="left")
    a.set_aspect("equal")
    a.set_xlim(0, 80)
    a.set_ylim(NOZZLE[1] * 1e3 - 20, NOZZLE[1] * 1e3 + 20)
    a.set_xlabel("x (mm)")
    a.set_ylabel("y (mm)")
    a.set_title(f"spray at {SPEED:g} m/s into air at rest, t = {T_END * 1e3:g} ms", fontsize=12)
    cb = fig.colorbar(sc, cax=cax)
    cb.set_ticks([5, 10, 20, 40, 80], labels=["5", "10", "20", "40", "80"])
    cb.minorticks_off()
    cb.set_label("diameter (µm)", fontsize=11)
    cb.ax.tick_params(labelsize=10)
    cb.outline.set_visible(False)
    edges = np.geomspace(D_MIN, D_MAX, 31)
    dd = np.geomspace(D_MIN, D_MAX, 400)
    f_mass = rosin_rammler_mass(dd)
    f_number = f_mass / dd ** 3
    # Densities in ln d, normalized over the clipped support
    for f, weights, color, name in [(f_number, np.ones_like(d), P["blue"], "by number"),
                                    (f_mass, d ** 3, P["flame"], "by mass")]:
        g = f * dd
        g /= np.trapezoid(g, np.log(dd))
        h, _ = np.histogram(d, edges, weights=weights)
        h = h / (h.sum() * np.diff(np.log(edges)))
        b.stairs(h, edges * 1e6, color=color, lw=1.2, alpha=0.9)
        b.semilogx(dd * 1e6, g, color=color, lw=1.6, ls="--", label=name)
    b.set_xlabel("diameter (µm)")
    b.set_ylabel(r"density in $\ln d$")
    b.set_xlim(D_MIN * 1e6, D_MAX * 1e6)
    b.set_ylim(0, None)
    b.legend(fontsize=10, loc="upper left", title_fontsize=9.5,
             title=f"steps: {len(d)} parcels drawn\ndashed: Rosin-Rammler,\n" + rf"$X$ = {X * 1e6:g} µm, $q$ = {Q:g}")
    return fig


# Where particles enter: a zone, a filled region, points
R_IN, R_OUT, ZONE_SPEED, ZONE_T = 0.01, 0.05, 5.0, 6e-3
FILL = dict(lx=0.06, ly=0.04, nx=12, ny=8, disk=(0.018, 0.02, 0.011), rect=(0.042, 0.02, 0.026, 0.012, 30.0))
JET_SPEED, JET_T, JET_CONE, JETS = 10.0, 5e-3, 15.0, ((0.002, 0.028), (0.002, 0.008))
PARTICLES = """[particles]
type = "solid"
density = 1000.0
diameter = 100e-6
seed = 1
"""


def zone_input():
    return f"""[run]
t_stop = {ZONE_T}
cfl = 0.8
[mesh]
type = "annulus"
r_inner = {R_IN}
r_outer = {R_OUT}
Ntheta = 48
Nr = 8
[[boundaries]]
name = "inner"
type = "symmetry"
[[boundaries]]
name = "outer"
type = "symmetry"
{GAS}{PARTICLES}[[particles.injectors]]
name = "ring"
region = "inner"
rate = 3e5
speed = {ZONE_SPEED}
{restarts("end", f"time_interval = {ZONE_T}")}"""


def fill_input():
    xd, yd, rd = FILL["disk"]
    xr, yr, a, b, angle = FILL["rect"]
    disk = f'{{type = "disk", center = [{xd}, {yd}], radius = {rd}}}'
    rect = f'{{type = "rectangle", center = [{xr}, {yr}], size = [{a}, {b}], rotate = {angle}, pivot = [{xr}, {yr}]}}'
    return f"""[run]
t_stop = 1e-4
cfl = 0.8
{box(FILL["nx"], FILL["ny"], FILL["lx"], FILL["ly"])}{GAS}{PARTICLES}[[particles.injectors]]
name = "cloud"
region = {{type = "union", regions = [{disk}, {rect}]}}
rate = 2e7
velocity = [0.0, 0.0]
{restarts("end", "time_interval = 1e-4")}"""


def jets_input():
    jets = "".join(f"""[[particles.injectors]]
name = "{name}"
point = [{p[0]}, {p[1]}]
rate = 1e5
speed = {JET_SPEED}
direction = [1.0, 0.0]
cone_angle = {cone}
""" for name, p, cone in [("cone", JETS[0], JET_CONE), ("straight", JETS[1], 0.0)])
    return f"""[run]
t_stop = {JET_T}
cfl = 0.8
{box(30, 20, 0.06, 0.04)}{GAS}{PARTICLES}{jets}{restarts("end", f"time_interval = {JET_T}")}"""


@functools.cache
def site_runs(binary):
    return tuple(parcels(run(binary, toml)[-1])[0] for toml in (zone_input(), fill_input(), jets_input()))


def particle_injection_sites(P, args):
    ring, cloud, jets = site_runs(args.mallard)
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13.0, 4.4), gridspec_kw=dict(width_ratios=[1.0, 1.25, 1.25]))
    th = np.linspace(0, 2 * np.pi, 400)
    for r in (R_IN, R_OUT):
        a.plot(r * 1e3 * np.cos(th), r * 1e3 * np.sin(th), color=P["line"], lw=1.6 if r == R_IN else 1.0)
    a.scatter(ring[:, 0] * 1e3, ring[:, 1] * 1e3, s=1.5, color=P["teal"], linewidths=0)
    for t in np.radians([20, 140, 260]):
        a.annotate("", (R_OUT * 1e3 * 0.9 * np.cos(t), R_OUT * 1e3 * 0.9 * np.sin(t)),
                   (R_IN * 1e3 * np.cos(t), R_IN * 1e3 * np.sin(t)),
                   arrowprops=dict(arrowstyle="-|>", color=P["flame"], lw=1.4))
    label(a, 0, 0, 'zone\n"inner"', P, size=10)
    a.set_title('a boundary zone: region = "inner"', fontsize=12)
    lim = R_OUT * 1e3 * 1.02
    a.set_xlim(-lim, lim)
    a.set_ylim(-lim, lim)

    lx, ly, nx, ny = (FILL[k] for k in ("lx", "ly", "nx", "ny"))
    for i in range(nx + 1):
        b.axvline(i * lx / nx * 1e3, color=P["mesh"], lw=0.6, zorder=0)
    for j in range(ny + 1):
        b.axhline(j * ly / ny * 1e3, color=P["mesh"], lw=0.6, zorder=0)
    xd, yd, rd = FILL["disk"]
    b.plot((xd + rd * np.cos(th)) * 1e3, (yd + rd * np.sin(th)) * 1e3, color=P["line"], lw=1.2)
    xr, yr, ra, rb, angle = FILL["rect"]
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]) * [ra / 2, rb / 2]
    rot = np.radians(angle)
    corners = corners @ np.array([[np.cos(rot), np.sin(rot)], [-np.sin(rot), np.cos(rot)]]) + [xr, yr]
    b.plot(corners[:, 0] * 1e3, corners[:, 1] * 1e3, color=P["line"], lw=1.2)
    b.scatter(cloud[:, 0] * 1e3, cloud[:, 1] * 1e3, s=1.5, color=P["teal"], linewidths=0)
    b.set_title("a filled region: a disk and a rectangle", fontsize=12)
    b.set_xlim(0, lx * 1e3)
    b.set_ylim(0, ly * 1e3)

    for p, name in [(JETS[0], f"cone_angle = {JET_CONE:g}"), (JETS[1], "cone_angle = 0")]:
        c.plot(p[0] * 1e3, p[1] * 1e3, "o", color=P["flame"], ms=5)
        label(c, p[0] * 1e3 + 1, p[1] * 1e3 + (9.5 if p is JETS[0] else 2.5), name, P, size=10, ha="left")
    reach = 42.0
    for s in (-1, 1):
        c.plot([JETS[0][0] * 1e3, JETS[0][0] * 1e3 + reach], [JETS[0][1] * 1e3, JETS[0][1] * 1e3 + s * reach *
               np.tan(np.radians(JET_CONE))], color=P["muted"], lw=0.8, ls=":")
    c.scatter(jets[:, 0] * 1e3, jets[:, 1] * 1e3, s=2, color=P["teal"], linewidths=0)
    c.set_title("points, with and without a cone", fontsize=12)
    c.set_xlim(0, 60)
    c.set_ylim(0, 40)
    for ax in (a, b, c):
        ax.set_aspect("equal")
        ax.set_xlabel("x (mm)")
    a.set_ylabel("y (mm)")
    fig.tight_layout(w_pad=1.5)
    return fig


# The five size distributions, each drawn by an injector of its own at a point
DISTRIBUTIONS = [
    ("rosin_rammler", "mass", dict(size=40e-6, spread=3.0, min=5e-6, max=100e-6)),
    ("log_normal", "number", dict(median=20e-6, sigma=0.5)),
    ("nukiyama_tanasawa", "number", dict(size=8e-6, spread=1.0, exponent=2.0)),
    ("uniform", "number", dict(min=10e-6, max=50e-6)),
    ("table", "mass", dict(diameters=[10e-6, 30e-6, 60e-6, 90e-6], pdf=[0.0, 1.0, 1.0, 0.0])),
]
DRAWS = 20000


def toml_value(v):
    return "[" + ", ".join(f"{x:g}" for x in v) + "]" if isinstance(v, list) else f"{v:g}"


def distributions_input():
    injectors = "".join(f"""[[particles.injectors]]
name = "{kind}"
point = [{0.01 + 0.02 * k:g}, 0.01]
rate = {DRAWS / 1e-4:g}
velocity = [0.0, 0.0]
diameter = {{distribution = "{kind}", basis = "{basis}", {", ".join(f"{key} = {toml_value(v)}" for key, v in params.items())}}}
""" for k, (kind, basis, params) in enumerate(DISTRIBUTIONS))
    return f"""[run]
t_stop = 1e-4
dt = 1e-4
{box(10, 4, 0.1, 0.02)}{GAS}[particles]
type = "solid"
density = 1000.0
seed = 1
{injectors}{restarts("end", "interval = 1")}"""


def density(kind, params, d):
    """The density of a distribution in its basis, unnormalized."""
    if kind == "rosin_rammler":
        X, q = params["size"], params["spread"]
        return q / X * (d / X) ** (q - 1) * np.exp(-(d / X) ** q)
    if kind == "log_normal":
        return np.exp(-0.5 * (np.log(d / params["median"]) / params["sigma"]) ** 2) / d
    if kind == "nukiyama_tanasawa":
        X = params["size"]
        return (d / X) ** params["exponent"] * np.exp(-(d / X) ** params["spread"])
    if kind == "uniform":
        return ((d >= params["min"]) & (d <= params["max"])).astype(float)
    return np.interp(d, params["diameters"], params["pdf"], left=0.0, right=0.0)


@functools.cache
def distribution_runs(binary):
    x, d, _ = parcels(run(binary, distributions_input())[-1])
    which = np.rint((x[:, 0] - 0.01) / 0.02).astype(int)
    return tuple(d[which == k] for k in range(len(DISTRIBUTIONS)))


def particle_size_distributions(P, args):
    drawn = distribution_runs(args.mallard)
    fig, axes = plt.subplots(1, 5, figsize=(15.0, 3.6))
    for ax, (kind, basis, params), d in zip(axes, DISTRIBUTIONS, drawn):
        s = np.sort(d)
        lo, hi = 0.0, s[np.searchsorted(np.cumsum(s ** 3) / np.sum(s ** 3), 0.995)]
        if "min" in params:
            lo, hi = params["min"], params["max"]
        elif kind == "table":
            lo, hi = params["diameters"][0], params["diameters"][-1]
        dd = np.linspace(max(lo, 1e-3 * hi), hi, 600)
        f = density(kind, params, dd)
        f_number = f / dd ** 3 if basis == "mass" else f
        edges = np.linspace(lo, hi, 31)
        for g, weights, color, name in [(f_number, np.ones_like(d), P["blue"], "number"),
                                        (f_number * dd ** 3, d ** 3, P["flame"], "mass")]:
            g = g / np.trapezoid(g, dd * 1e6)
            h, _ = np.histogram(d, edges, weights=weights)
            h = h / (h.sum() * np.diff(edges * 1e6))
            ax.stairs(h, edges * 1e6, color=color, lw=1.1, alpha=0.9)
            given = name == basis
            ax.plot(dd * 1e6, g, color=color, lw=2.0 if given else 1.4, ls="-" if given else "--",
                    label=f"by {name}" + (" (given)" if given else ""))
        ax.set_title(f'"{kind}"\nbasis = "{basis}"', fontsize=11.5)
        ax.set_xlabel("diameter (µm)")
        ax.set_xlim(lo * 1e6, hi * 1e6)
        ax.set_ylim(0, None)
        ax.set_yticks([])
        ax.legend(fontsize=9.5, loc="upper right", handlelength=1.6)
    axes[0].set_ylabel("density in d")
    fig.tight_layout(w_pad=1.0)
    return fig


# Rate accounting: rate * dt = 0.7 parcels per step, from a start to an end within steps
DT, RATE, START, END, T_STOP = 1e-3, 700.0, 3.5e-3, 26.5e-3, 30e-3
MDOT_D, MDOT_RHO, MDOT_RATE = (20e-6, 40e-6), 1000.0, 2e4
# mass_flow for MDOT_RATE parcels per second: the mean of d^3 of the uniform distribution is (a + b)(a^2 + b^2)/4
MDOT = MDOT_RATE * MDOT_RHO * np.pi / 6 * (MDOT_D[0] + MDOT_D[1]) * (MDOT_D[0] ** 2 + MDOT_D[1] ** 2) / 4


def rate_input():
    return f"""[run]
t_stop = {T_STOP}
dt = {DT}
{box(4, 4, 0.04, 0.04)}{GAS}[particles]
type = "solid"
density = {MDOT_RHO}
weight = 1.0
seed = 1
[[particles.injectors]]
name = "by_rate"
point = [0.01, 0.02]
rate = {RATE}
start = {START}
end = {END}
velocity = [0.0, 0.0]
diameter = 20e-6
[[particles.injectors]]
name = "by_mass"
point = [0.03, 0.02]
mass_flow = {MDOT:.6g}
velocity = [0.0, 0.0]
diameter = {{distribution = "uniform", basis = "number", min = {MDOT_D[0]}, max = {MDOT_D[1]}}}
{restarts("step", "interval = 1")}"""


@functools.cache
def rate_run(binary):
    t, n, carry, mass = [0.0], [0], [0.0], [0.0]
    for r in run(binary, rate_input(), "step"):
        x, d, w = parcels(r)
        right = x[:, 0] > 0.02
        t.append(r.t)
        n.append(int(np.sum(~right)))
        carry.append(float(r.arrays["PARTICLE_INJECTOR_by_rate"][1]))
        mass.append(float(np.sum(w[right] * MDOT_RHO * np.pi / 6 * d[right] ** 3)))
    return tuple(np.array(v) for v in (t, n, carry, mass))


def particle_injection_rate(P, args):
    t, n, carry, mass = rate_run(args.mallard)
    fig = plt.figure(figsize=(12.0, 4.6))
    a = fig.add_axes([0.06, 0.38, 0.42, 0.54])
    a2 = fig.add_axes([0.06, 0.12, 0.42, 0.18], sharex=a)
    b = fig.add_axes([0.58, 0.12, 0.40, 0.80])
    ms = t * 1e3
    tt = np.linspace(0, T_STOP, 601)
    a.plot(tt * 1e3, RATE * (np.clip(tt, START, END) - START), color=P["flame"], ls="--",
           label=r"rate $\times$ time injecting")
    a.step(ms, n, where="post", color=P["blue"], lw=1.6, label="parcels injected")
    a.plot(ms, n, "o", color=P["blue"], ms=3.5)
    for s, name in [(START, "start"), (END, "end")]:
        for ax in (a, a2):
            ax.axvline(s * 1e3, color=P["muted"], lw=0.8, ls=":")
        label(a, s * 1e3 + 0.4, 17.5, name, P, color=P["muted"], size=10, ha="left")
    a.set_ylabel("parcels")
    a.set_ylim(0, 18.5)
    a.legend(fontsize=10, loc="lower right")
    a.set_title(f"rate = {RATE:g} with dt = {DT:g}: {RATE * DT:g} parcels per step", fontsize=12)
    plt.setp(a.get_xticklabels(), visible=False)
    a2.step(ms, carry, where="post", color=P["teal"], lw=1.4)
    a2.set_ylim(0, 1)
    a2.set_yticks([0, 1])
    a2.set_ylabel("carried", fontsize=11)
    a2.set_xlabel("t (ms)")
    a2.set_xlim(0, T_STOP * 1e3)
    b.plot(tt * 1e3, MDOT * tt * 1e9, color=P["flame"], ls="--", label=r"mass_flow $\times$ t")
    b.step(ms, mass * 1e9, where="post", color=P["blue"], lw=1.6, label="mass of the parcels injected")
    b.plot(ms, mass * 1e9, "o", color=P["blue"], ms=3.0)
    b.set_xlabel("t (ms)")
    b.set_ylabel("mass (µg per m of depth)")
    b.set_xlim(0, T_STOP * 1e3)
    b.set_ylim(0, None)
    b.legend(fontsize=10, loc="upper left")
    b.set_title(f"mass_flow, diameters uniform in [{MDOT_D[0] * 1e6:g}, {MDOT_D[1] * 1e6:g}] µm", fontsize=12)
    return fig


GUIDE_RUNS = {
    "particle_injection": particle_injection,
    "particle_injection_rate": particle_injection_rate,
    "particle_injection_sites": particle_injection_sites,
    "particle_size_distributions": particle_size_distributions,
}


def figures(mallard=None, names=None):
    if not mallard:
        return
    args = argparse.Namespace(mallard=str(Path(mallard).resolve()))
    for name, draw in GUIDE_RUNS.items():
        if not names or name in names:
            figstyle.save(name, lambda P, draw=draw: draw(P, args))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mallard", required=True, help="2D Mallard executable")
    ap.add_argument("names", nargs="*")
    args = ap.parse_args()
    figures(args.mallard, args.names)


if __name__ == "__main__":
    main()
