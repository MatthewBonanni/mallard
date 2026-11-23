"""Figures of particle injectors, in the particles page of the user guide
(docs/images/guide/).

    python tools/docs/particle_injection.py --mallard build/src/mallard [names...]

From a short Mallard run (a 2D build, about 1 min on 2 CPU threads): a
spray of a point injector with a cone of velocities and a Rosin-Rammler size
distribution by mass into air at rest, its parcels colored by diameter, and the
diameters it drew against the distribution's density by number and by mass
(particle_injection). The parcels are read from a restart file
(tools/mallard/restart.py).
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


def spray():
    return f"""[run]
t_stop = {T_END}
cfl = 0.8
[mesh]
type = "cartesian"
Nx = 40
Ny = 20
Lx = {LX}
Ly = {LY}
[[periodic]]
zones = ["left", "right"]
[[periodic]]
zones = ["bottom", "top"]
[initialize]
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
[particles]
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
[[write_data]]
prefix = "end"
format = "restart"
time_interval = {T_END}
"""


@functools.cache
def spray_run(binary):
    """Positions and diameters of the parcels at the end of the run."""
    work = Path(tempfile.mkdtemp())
    (work / "input.toml").write_text(spray())
    subprocess.run([str(binary), "-i", "input.toml", "--kokkos-num-threads=2"], cwd=work, check=True,
                   stdout=subprocess.DEVNULL)
    r = restart.read(str(sorted(work.glob("end_*.restart"))[-1]))
    return np.asarray(r.arrays["PARTICLE_X"], float).reshape(-1, 2), np.asarray(r.arrays["PARTICLE_D"], float)


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


GUIDE_RUNS = {"particle_injection": particle_injection}


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
