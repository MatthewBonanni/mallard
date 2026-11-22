"""Figures of the particles pages: the user guide's (docs/images/guide/) and the
theory manual's (docs/images/theory/).

    python tools/docs/particles.py [--mallard build/src/mallard] [names...]

Exact curves: the response of a particle to a step and to an oscillation of the
gas (particle_response), the drag coefficient of the drag laws (particle_drag;
Singh et al.'s from singh_drag_factor, the formula of src/particles/particle_forces.h)
and a step of a particle across a face (particle_start_cell). From short Mallard
runs (a 2D build, about a minute each on two CPU threads): particles of several
Stokes numbers spiraling out of solid-body rotation against their exact spirals
(particle_stokes_spirals), the terminal velocities of the settling box against
Stokes' law and the Schiller-Naumann and Singh correlations (particle_terminal_velocity),
and the particles of examples/particle_bounce against their exact trajectories
(particle_bounce). The trajectories are read from restart files
(tools/mallard/restart.py), which carry the particles.
"""
import argparse
import csv
import functools
import math
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import figstyle
from figstyle import bare, label
from schematics import arrow

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "examples" / "particle_bounce"),
                str(ROOT / "examples" / "particle_centrifuge"), str(ROOT / "examples" / "particle_settling")]
from mallard import restart  # noqa: E402
from check_particle_bounce import trajectory, walls_of  # noqa: E402
from check_particle_centrifuge import spiral  # noqa: E402
from check_particle_settling import terminal_velocity  # noqa: E402

THEORY = figstyle.OUT.parent / "theory"


def run(binary, toml, workdir, name="input.toml"):
    (Path(workdir) / name).write_text(toml)
    subprocess.run([str(binary), "-i", name, "--kokkos-num-threads=2"], cwd=workdir, check=True,
                   stdout=subprocess.DEVNULL)


def snapshots(workdir, prefix):
    """(t, {id: (x, y)}) of every restart file of a run, in time order."""
    out = []
    for path in sorted(Path(workdir).glob(f"{prefix}_*.restart")):
        r = restart.read(str(path))
        ids = np.asarray(r.arrays.get("PARTICLE_ID", []), dtype=np.int64)
        if ids.size == 0:
            out.append((r.t, {}))
            continue
        xs = np.asarray(r.arrays["PARTICLE_X"], dtype=float).reshape(len(ids), -1)
        out.append((r.t, {int(i): x[:2] for i, x in zip(ids, xs)}))
    return out


def schiller_naumann(re):
    re = np.asarray(re, dtype=float)
    return np.where(re < 1000, 1.0 + 0.15 * re ** 0.687, 0.44 * re / 24.0)


def singh_drag_factor(Re, M, Kn, Tp_T, gamma):
    """C_D Re / 24 of Singh et al. (2022, Appendix C): particles::singh_drag_factor of
    src/particles/particle_forces.h, term by term."""
    omega, delta0, alpha0, A1, A2, A3, eta, alpha_hoc, Cd_hypersonic = 0.74, 9.4, 0.356, 2.514, 0.8, 0.55, 1.8, 1.27, 0.9
    C0 = 24.0 / delta0 ** 2
    gm1 = gamma - 1.0
    exponent = gamma / gm1
    Us_U, Ms2, Ts_T, alpha, pressure = 1.0, M * M, 1.0, 1.0, 0.0
    if M > 1.0:
        M2 = M * M
        Us_U = (2.0 + gm1 * M2) / ((gamma + 1.0) * M2)
        Ms2 = (gm1 * M2 + 2.0) / (2.0 * gamma * M2 - gm1)
        Ts_T = (gm1 * M2 + 2.0) * (2.0 * gamma * M2 - gm1) / ((gamma + 1.0) ** 2 * M2)
        alpha = 1.0 / (alpha0 * M + 1.0 - alpha0)
        C1 = (Cd_hypersonic - C0 * (1.0 + gm1 ** 2 / (4.0 * gamma)) ** exponent) / (
            1.0 - gm1 / ((gamma + 1.0) * alpha0 * M))
        pressure = C1 * (1.0 - alpha * Us_U)
    theta_s = (1.0 + 0.5 * gm1 * Ms2) ** exponent
    p = (gamma + 1.0) / (2.0 * gamma) - gm1 * omega / gamma
    Re_ratio = (alpha * alpha * Ts_T) ** omega * theta_s ** -p
    root = math.sqrt(Re) + delta0 * math.sqrt(Re_ratio)
    continuum = pressure * Re + C0 * theta_s * root * root
    Mw = M ** (2.0 * omega - 1.0)
    wall = (1.0 + Tp_T / Ts_T) ** omega
    Wr_T = Kn * Mw * math.sqrt(2.0 / (math.pi * gamma)) * wall
    Br = Kn * math.sqrt(2.0 / (math.pi * gamma)) * wall * (Mw + 1.0)
    f_Kn = 1.0 / ((1.0 + Kn * (A1 + A2 * math.exp(-A3 / Kn))) * (1.0 + alpha_hoc * Wr_T))
    s = M * math.sqrt(0.5 * gamma)
    s2 = s * s
    if s < 0.3:
        g = (16.0 / 3.0 + s2 * (16.0 / 15.0 + s2 * (-8.0 / 105.0 + s2 * (8.0 / 945.0 - s2 * 2.0 / 2079.0)))) / math.sqrt(
            math.pi)
    else:
        g = (1.0 + 2.0 * s2) * math.exp(-s2) / (s2 * math.sqrt(math.pi)) + \
            (4.0 * s2 * s2 + 4.0 * s2 - 1.0) * math.erf(s) / (2.0 * s2 * s)
    free_molecular = (g + 2.0 / 3.0 * math.sqrt(math.pi * Tp_T)) * math.sqrt(math.pi) / Kn
    B = Br ** eta
    return (continuum * f_Kn + free_molecular * B) / (24.0 * (1.0 + B))


def knudsen(Re, M, gamma=1.4):
    """The Knudsen number of a slip Reynolds and Mach number, Kn = sqrt(gamma pi / 2) M / Re."""
    return math.sqrt(gamma * math.pi / 2.0) * M / Re


# ------------------------------------------------------------ exact curves

def particle_response(P):
    """A step of the gas velocity, Stokes and Schiller-Naumann drag, and an oscillation of the gas."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw=dict(wspace=0.32))
    t = np.linspace(0, 5, 400)
    a.plot(t, 1 - np.exp(-t), color=P["blue"], label="Stokes drag")
    for re0, color in [(20, P["teal"]), (200, P["flame"])]:
        # dv/dt = (1 - v) f(Re0 (1 - v)) in units of U and tau_St
        v, vs, h = 0.0, [0.0], t[1] - t[0]
        rate = lambda v: (1 - v) * float(schiller_naumann(re0 * abs(1 - v)))
        for _ in t[1:]:
            k1 = rate(v); k2 = rate(v + 0.5 * h * k1); k3 = rate(v + 0.5 * h * k2); k4 = rate(v + h * k3)
            v += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            vs.append(v)
        a.plot(t, vs, color=color, label=f"Schiller-Naumann, $Re_0$ = {re0}")
    a.axhline(1, color=P["muted"], lw=0.8, ls=":")
    a.set_xlabel(r"$t / \tau_{St}$")
    a.set_ylabel(r"$v / U$")
    a.set_title("released at rest in a stream $U$", fontsize=12)
    a.legend(fontsize=10, loc="lower right")
    w = np.logspace(-2, 2, 300)
    b.semilogx(w, 1 / np.sqrt(1 + w ** 2), color=P["blue"], label="amplitude ratio")
    b.semilogx(w, np.arctan(w) / (np.pi / 2), color=P["flame"], label="phase lag / 90°")
    b.axvline(1, color=P["muted"], lw=0.8, ls=":")
    label(b, 1.15, 0.12, r"$St = \omega\tau = 1$", P, color=P["muted"], ha="left", size=11)
    b.set_xlabel(r"$\omega \tau$")
    b.set_ylim(0, 1.05)
    b.set_title(r"in gas oscillating as $\sin \omega t$ (Stokes drag)", fontsize=12)
    b.legend(fontsize=10, loc="center left")
    return fig


def particle_drag(P):
    re = np.logspace(-2, 4, 400)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.loglog(re, 24 / re, color=P["blue"], ls="--", label="Stokes, $24 / Re_p$")
    ax.loglog(re, 24 / re * schiller_naumann(re), color=P["teal"], label="Schiller-Naumann")
    for M, color, ls in [(1e-4, P["flame"], "-"), (0.5, P["plum"], "-."), (2.0, P["gold"], ":")]:
        cd = [24 / r * singh_drag_factor(r, M, knudsen(r, M), 1.0, 1.4) for r in re]
        ax.loglog(re, cd, color=color, ls=ls, label=f"Singh et al., $M$ = {M:g}")
    ax.axvline(1000, color=P["muted"], lw=0.8, ls=":")
    label(ax, 1150, 30, "$C_D = 0.44$\nabove $Re_p$ = 1000", P, color=P["muted"], ha="left", size=11)
    ax.set_xlabel("particle Reynolds number $Re_p$")
    ax.set_ylabel("drag coefficient $C_D$")
    ax.set_ylim(0.1, 3000)
    ax.legend(fontsize=10)
    return fig


def particle_start_cell(P):
    """Stages of one step crossing a face, all reading the start cell's polynomial."""
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    cells = {"i": [(0, 0), (2, -0.2), (2.2, 2), (0.1, 2.1)], "j": [(2, -0.2), (4.1, 0.1), (4, 2.2), (2.2, 2)]}
    ax.add_patch(Polygon(cells["i"], closed=True, facecolor=P["teal"], alpha=0.16, edgecolor=P["line"], lw=1.2))
    ax.add_patch(Polygon(cells["j"], closed=True, facecolor="none", edgecolor=P["line"], lw=1.2))
    label(ax, 0.55, 1.75, "cell $i_n$", P, color=P["teal"], size=12)
    label(ax, 3.55, 1.75, "cell $j$", P, size=12)
    s = np.linspace(0, 1, 50)
    path = np.column_stack([1.0 + 1.9 * s, 0.7 + 0.55 * np.sin(np.pi * s * 0.8)])
    ax.plot(path[:, 0], path[:, 1], color=P["muted"], lw=1.2, ls=(0, (4, 3)))
    stages = [(0.0, r"$\mathbf{x}^n$"), (0.5, "stages"), (1.0, r"$\mathbf{x}^{n+1}$")]
    for k, (f, text) in enumerate(stages):
        p = path[int(round(f * (len(path) - 1)))]
        ax.plot(*p, "o", color=P["flame"] if k == 2 else P["blue"], ms=8, zorder=5)
    p_half = path[len(path) // 2]
    ax.plot(*path[int(0.75 * (len(path) - 1))], "o", color=P["blue"], ms=8, zorder=5)
    label(ax, path[0, 0] - 0.1, path[0, 1] - 0.25, r"$\mathbf{x}_p^n$", P, color=P["blue"], ha="right", size=12)
    label(ax, path[-1, 0] + 0.1, path[-1, 1] - 0.25, r"$\mathbf{x}_p^{n+1}$", P, color=P["flame"], ha="left", size=12)
    label(ax, p_half[0], p_half[1] + 0.35, "stage positions", P, color=P["blue"], size=11)
    for f in (0.5, 0.75, 1.0):
        q = path[int(round(f * (len(path) - 1)))]
        arrow(ax, (0.75, 1.15), (q[0] - 0.08, q[1] + 0.08), P, color=P["teal"], lw=1.0, ms=9, alpha=0.8,
              connectionstyle="arc3,rad=0.25")
    label(ax, 0.55, 1.15, r"$\mathbf{W}_{i_n}(\mathbf{x})$", P, color=P["teal"], size=12, box=True)
    label(ax, 3.1, -0.55, "relocated into $j$ at the end of the step", P, color=P["flame"], size=11)
    bare(ax)
    ax.set_xlim(-0.2, 4.4)
    ax.set_ylim(-0.8, 2.4)
    return fig


# ------------------------------------------------------------ runs

CENTRIFUGE_STOKES = [0.0, 0.25, 1.0, 4.0]


@functools.cache
def centrifuge_run(binary, stokes):
    """The particle_centrifuge flow on 64 x 16 cells, one particle at r = 0.3, to t = 20, restarts every 1."""
    work = Path(tempfile.mkdtemp())
    cfg = (ROOT / "examples" / "particle_centrifuge" / "input.toml").read_text()
    cfg = cfg.replace("Ntheta = 128", "Ntheta = 64").replace("Nr = 32", "Nr = 16").replace("t_stop = 15.0",
                                                                                          "t_stop = 20.0")
    start = cfg.index("[particles]")
    end = cfg.index("[output]")
    omega, mu, density = 0.2, 1e-3, 1000.0
    if stokes == 0:
        particles = '[particles]\ntype = "tracer"\npoints = [[0.3, 0.0]]\n'
    else:
        d = math.sqrt(18 * mu * (stokes / omega) / density)
        particles = (f'[particles]\ntype = "solid"\ndiameter = {d!r}\ndensity = {density}\ndrag = "stokes"\n'
                     'points = [[0.3, 0.0]]\n[[particles.walls]]\nzones = ["outer"]\ninteraction = "stick"\n')
    cfg = cfg[:start] + particles + "\n" + cfg[end:]
    cfg += '\n[[write_data]]\nprefix = "spiral"\nformat = "restart"\ntime_interval = 1.0\n'
    run(binary, cfg, work)
    return snapshots(work, "spiral")


def particle_stokes_spirals(P, args):
    omega = 0.2
    colors = [P["muted"], P["blue"], P["teal"], P["flame"]]
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    theta = np.linspace(0, 2 * np.pi, 400)
    for r in (0.2, 1.0):
        ax.plot(r * np.cos(theta), r * np.sin(theta), color=P["line"], lw=1.2)
    for st, color in zip(CENTRIFUGE_STOKES, colors):
        snaps = centrifuge_run(args.mallard, st)
        tau = st / omega
        t = np.linspace(0, 20, 800)
        z = np.array([0.3 * np.exp(1j * omega * s) if st == 0 else spiral(0.3, omega, tau, s) for s in t])
        inside = np.abs(z) <= 1.0
        last = np.argmin(inside) if not inside.all() else len(z)
        ax.plot(z.real[:last], z.imag[:last], color=color, lw=1.4,
                label="tracer" if st == 0 else rf"$\Omega\tau$ = {st:g}")
        pts = np.array([s[1][0] for s in snaps if 0 in s[1]])
        if len(pts):
            ax.plot(pts[:, 0], pts[:, 1], "o", color=color, ms=4)
    ax.plot([0.3], [0.0], "o", color=P["fg"], ms=5)
    label(ax, 0.32, -0.07, "release", P, ha="left", size=11)
    label(ax, 0.0, 1.07, "outer wall (stick)", P, color=P["muted"], size=11)
    bare(ax)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.15, 1.15)
    ax.legend(fontsize=10.5, loc="lower right", bbox_to_anchor=(1.12, -0.06))
    return fig


SETTLING_DIAMETERS = [10e-6, 30e-6, 100e-6, 200e-6]


SINGH_DIAMETERS = [1e-6, 3e-6, 10e-6, 100e-6]


@functools.cache
def settling_velocity(binary, d, rho_p, drag="schiller_naumann", periods=6):
    """The settling velocity of the particle_settling box with particles of diameter d, after periods tau."""
    work = Path(tempfile.mkdtemp())
    cfg = (ROOT / "examples" / "particle_settling" / "input.toml").read_text()
    phys = tomllib.loads(cfg)
    rho = 1.1614401858304297
    v_t, tau, _ = terminal_velocity(d, rho_p, rho, phys["physics"]["mu"], 9.81, "schiller_naumann")
    cfg = cfg.replace("diameter = 100e-6", f"diameter = {d!r}").replace("density = 1000.0", f"density = {rho_p}")
    cfg = cfg.replace('drag = "schiller_naumann"', f'drag = "{drag}"')
    cfg = cfg.replace("t_stop = 0.25", f"t_stop = {periods * tau!r}").replace("Lx = 0.04", "Lx = 0.16").replace(
        "Ly = 0.04", "Ly = 0.16")
    run(binary, cfg, work)
    with open(Path(work) / "droplets.csv") as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
    return -float(rows[1][rows[0].index("v_y")]), v_t


def singh_terminal_velocity(d, rho_p, rho, mu, g, c, gamma=1.4):
    """|v_t| = tau_St g (1 - rho / rho_p) / f with Singh et al.'s f at the slip v_t, by fixed-point iteration."""
    tau_st, g_eff = rho_p * d * d / (18.0 * mu), g * (1.0 - rho / rho_p)
    Kn = mu / (rho * c * d) * math.sqrt(0.5 * math.pi * gamma)
    v = tau_st * g_eff
    for _ in range(500):
        v = tau_st * g_eff / singh_drag_factor(rho * v * d / mu, v / c, Kn, 1.0, gamma)
    return v


def particle_terminal_velocity(P, args):
    mu, rho, g = 1.8e-5, 1.1614401858304297, 9.81
    c = math.sqrt(1.4 * 1e5 / rho)
    d = np.logspace(np.log10(1e-6), np.log10(1e-3), 200)
    fig, (ax, rx) = plt.subplots(1, 2, figsize=(11.2, 4.4), gridspec_kw=dict(wspace=0.28))
    for rho_p, name, color in [(1000.0, "water", P["blue"]), (2500.0, "glass", P["flame"])]:
        stokes = rho_p * d ** 2 / (18 * mu) * g * (1 - rho / rho_p)
        sn = np.array([terminal_velocity(x, rho_p, rho, mu, g, "schiller_naumann")[0] for x in d])
        singh = np.array([singh_terminal_velocity(x, rho_p, rho, mu, g, c) for x in d])
        ax.loglog(d * 1e6, stokes, color=color, ls="--", lw=1.2)
        ax.loglog(d * 1e6, sn, color=color, label=f"{name} ({rho_p:g} kg/m³)")
        ax.loglog(d * 1e6, singh, color=color, ls=":", lw=1.6)
        rx.semilogx(d * 1e6, stokes / sn, color=color, ls="--", lw=1.2)
        rx.semilogx(d * 1e6, singh / sn, color=color, ls=":", lw=1.6, label=f"{name}, Singh et al.")
        fill = "none" if rho_p > 1000 else color
        if args.mallard:
            for x in SETTLING_DIAMETERS:
                v, _ = settling_velocity(args.mallard, x, rho_p)
                ax.loglog([x * 1e6], [v], "o", color=color, ms=6, mfc=fill)
            if rho_p == 1000.0:
                for x in SINGH_DIAMETERS:
                    v, _ = settling_velocity(args.mallard, x, rho_p, "singh", 15)
                    v_sn = terminal_velocity(x, rho_p, rho, mu, g, "schiller_naumann")[0]
                    rx.semilogx([x * 1e6], [v / v_sn], "o", color=color, ms=6, mfc=fill)
    label(ax, 900, 2e-5, "dashed: Stokes' law\ndotted: Singh et al.", P, color=P["muted"], ha="right", size=11)
    ax.set_xlabel("diameter (µm)")
    ax.set_ylabel("terminal velocity (m/s)")
    ax.set_ylim(1e-5, 20)
    ax.legend(fontsize=11, loc="upper left")
    rx.axhline(1, color=P["muted"], lw=0.8)
    label(rx, 1.1, 1.22, "dashed: Stokes' law", P, color=P["muted"], ha="left", size=11)
    rx.set_xlabel("diameter (µm)")
    rx.set_ylabel("ratio to the Schiller-Naumann value")
    rx.set_ylim(0.8, 1.25)
    rx.legend(fontsize=10.5, loc="lower right")
    return fig


@functools.cache
def bounce_run(binary):
    work = Path(tempfile.mkdtemp())
    run(binary, (ROOT / "examples" / "particle_bounce" / "input.toml").read_text() +
        '\n[[write_data]]\nprefix = "bounce"\nformat = "restart"\ntime_interval = 0.05\n', work)
    return snapshots(work, "bounce")


def particle_bounce(P, args):
    cfg = tomllib.loads((ROOT / "examples" / "particle_bounce" / "input.toml").read_text())
    snaps = bounce_run(args.mallard)
    p, ph = cfg["particles"], cfg["physics"]
    tau = p["density"] * p["diameter"] ** 2 / (18 * ph["mu"])
    g_eff = [x * (1 - 1 / p["density"]) for x in p["gravity"]]
    box = [(0.0, 1.0), (0.0, 1.0)]
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.add_patch(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)], closed=True, facecolor="none", edgecolor=P["line"],
                         lw=1.4))
    ax.plot([0, 1], [0, 0], color=P["flame"], lw=3)
    colors = [P["blue"], P["teal"], P["plum"]]
    times = np.linspace(0, cfg["run"]["t_stop"], 1200)
    for k, seed in enumerate(p["points"]):
        samples = {"times": list(times), "points": []}
        trajectory(seed, p["velocity"], cfg["run"]["t_stop"], tau, g_eff, box, walls_of(cfg), samples)
        exact = np.array(samples["points"])
        ax.plot(exact[:, 0], exact[:, 1], color=colors[k], lw=1.2)
        pts = np.array([s[1][k] for s in snaps if k in s[1]])
        ax.plot(pts[:, 0], pts[:, 1], "o", color=colors[k], ms=3)
    label(ax, 0.5, -0.06, "floor: $e_n$ = 0.8, $e_t$ = 0.9", P, color=P["flame"], size=11)
    label(ax, 1.03, 0.5, "elastic", P, color=P["muted"], size=11, rotation=90)
    label(ax, 0.5, 1.04, "ceiling: stick", P, color=P["muted"], size=11)
    bare(ax)
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.12, 1.1)
    return fig


GUIDE = {"particle_response": particle_response}
GUIDE_RUNS = {"particle_bounce": particle_bounce, "particle_stokes_spirals": particle_stokes_spirals}
THEORY_FIGURES = {"particle_drag": particle_drag, "particle_start_cell": particle_start_cell}


def figures(mallard=None, names=None):
    args = argparse.Namespace(mallard=str(Path(mallard).resolve()) if mallard else None)
    want = lambda name: not names or name in names
    for name, draw in GUIDE.items():
        if want(name):
            figstyle.save(name, draw)
    for name, draw in THEORY_FIGURES.items():
        if want(name):
            figstyle.save(name, draw, THEORY)
    if want("particle_terminal_velocity"):
        figstyle.save("particle_terminal_velocity", lambda P: particle_terminal_velocity(P, args))
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
