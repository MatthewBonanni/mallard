"""A premixed flame-vortex interaction run (Poinsot, Veynante & Candel 1991).

    python tools/flame_vortex.py FULL.csv MECHANISM PHASE RUN_DIR --r R --u U
        [--cells 12] [--cfl 0.4] [--flame-times T] [--outputs 100] [--transport mixture_averaged]

A planar flame in its frame (tools/flame_restart.py: the fresh gas enters on
the left at the flame speed S_L, cells of delta / --cells, delta the thermal
thickness of FULL.csv's Cantera flame) and, upstream of it, a pair of
counter-rotating vortices whose self-induced jet drives it into the flame.
Only the half y > 0 is computed: the axis of the pair is a symmetry plane
(bottom), and so is the top of the channel.

Each vortex has the Gaussian stream function psi = +-C exp(-|x - x_v|^2 / 2 R_c^2)
(its velocity peaks at R_c from its center and decays like a Gaussian, so the
pair has no far field); the centers are 2 R_c apart. As in Poinsot et al., the
pair's length scale is r = vortex diameter + distance between the centers =
4 R_c, and its velocity scale u' the largest velocity the pair induces (on the
axis between the vortices). --r and --u give r / delta and u' / S_L. The
pressure of each vortex is its cyclostrophic deficit -rho C^2 / (2 R_c^2)
exp(-|x - x_v|^2 / R_c^2), at the fresh gas's temperature.

The channel is max(2 r, 8 delta) high; the vortices start 3 R_c + delta
upstream of the flame (the maximum of dT/dx) and 3 R_c + 2 delta from the
inlet, and the flame has 2 r + 6 delta of burnt gas behind it. --flame-times
sets the run's length in delta / S_L (default: the time for the pair to reach
the flame at 0.3 u' + S_L, plus 3 flame times). Writes RUN_DIR/input.toml,
RUN_DIR/flame.restart and a copy of FULL.csv as RUN_DIR/planar_full.csv (the
reference of tools/flame_vortex_analysis.py); outputs RUN_DIR/solut/flame.pvd
(T, HRR, velocity, density, H2 and O2).
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from interpolate_restart import read_restart, write_restart  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def pair_velocity(x, y, x_v, a, R_c, C, H):
    """Velocity of the pair (vortices at y = +-a) and its images in the top plane y = H."""
    u = np.zeros_like(x)
    v = np.zeros_like(x)
    for y_v, s in ((a, 1.0), (-a, -1.0), (2 * H - a, -1.0), (2 * H + a, 1.0)):
        g = s * C * np.exp(-((x - x_v) ** 2 + (y - y_v) ** 2) / (2 * R_c**2))
        u += -g * (y - y_v) / R_c**2
        v += g * (x - x_v) / R_c**2
    return u, v


def pair_pressure_deficit(x, y, x_v, a, R_c, C, H, rho):
    dp = np.zeros_like(x)
    for y_v in (a, -a, 2 * H - a, 2 * H + a):
        dp -= rho * C**2 / (2 * R_c**2) * np.exp(-((x - x_v) ** 2 + (y - y_v) ** 2) / R_c**2)
    return dp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("full")
    ap.add_argument("mechanism")
    ap.add_argument("phase")
    ap.add_argument("run_dir")
    ap.add_argument("--r", type=float, required=True, help="pair size r over the thermal thickness")
    ap.add_argument("--u", type=float, required=True, help="largest pair velocity u' over S_L")
    ap.add_argument("--cells", type=float, default=12)
    ap.add_argument("--cfl", type=float, default=0.4)
    ap.add_argument("--flame-times", type=float)
    ap.add_argument("--outputs", type=int, default=100)
    ap.add_argument("--transport", default="mixture_averaged")
    args = ap.parse_args()

    data = np.loadtxt(args.full, delimiter=",", skiprows=1)
    x, u, T = data[:, 0], data[:, 1], data[:, 2]
    delta = (T[-1] - T[0]) / np.gradient(T, x).max()
    S_L, rho_u, T_u = u[0], data[0, 3], T[0]
    tau = delta / S_L
    r = args.r * delta
    R_c = r / 4
    a = R_c
    H_cells = int(np.ceil(max(2 * r, 8 * delta) / delta * args.cells))
    upstream = (6 * R_c + 3 * delta) / delta
    downstream = (2 * r + 6 * delta) / delta
    u_pair = args.u * S_L
    t_stop = args.flame_times * tau if args.flame_times else (3 * R_c + delta) / (0.3 * u_pair + S_L) + 3 * tau

    cmd = [sys.executable, os.path.join(HERE, "flame_restart.py"), args.full, args.mechanism, args.phase,
           str(args.cells), args.run_dir, "--cfl", str(args.cfl), "--transport", args.transport,
           "--ny", str(H_cells), "--upstream", f"{upstream:.6f}", "--downstream", f"{downstream:.6f}",
           "--t-stop", f"{t_stop:.9e}", "--outputs", str(args.outputs)]
    subprocess.run(cmd, check=True)
    shutil.copyfile(args.full, os.path.join(args.run_dir, "planar_full.csv"))

    # Channel of symmetry planes instead of a periodic one, and the outputs the analysis reads
    path = os.path.join(args.run_dir, "input.toml")
    text = open(path).read()
    text = text.replace('periodic = ["y"]\n', "")
    text = text.replace('name = "right"\ntype = "p_out"\np = 101325.0\n',
                        'name = "right"\ntype = "p_out"\np = 101325.0\n\n[[boundaries]]\nname = "bottom"\n'
                        'type = "symmetry"\n\n[[boundaries]]\nname = "top"\ntype = "symmetry"\n')
    text = re.sub(r"variables = \[.*\]", 'variables = ["RHO", "U", "P", "T", "HRR", "Y_H2", "Y_O2"]', text)
    dx = delta / args.cells
    nx = int(re.search(r"Nx = (\d+)", text).group(1))
    ny = H_cells
    H = ny * dx
    x_f = upstream * delta
    x_v = x_f - 3 * R_c - delta
    text = text.replace("# Premixed flame (V8)", f"# Flame-vortex interaction: r / delta = {args.r:g}, u' / S_L = {args.u:g} "
                        f"(r = {r * 1e3:.4f} mm, u' = {u_pair:.4f} m/s), pair at x = {x_v * 1e3:.4f} mm.\n# Planar flame",
                        1)
    open(path, "w").write(text)

    # Unit pair: its largest velocity, to scale C
    s = np.linspace(-4 * R_c, 4 * R_c, 801)
    X, Y = np.meshgrid(s, s, indexing="ij")
    uu, vv = pair_velocity(X, Y, 0.0, a, R_c, 1.0, 1e3 * R_c)
    C = u_pair / np.sqrt(uu**2 + vv**2).max()

    restart = read_restart(os.path.join(args.run_dir, "flame.restart"))
    f = restart["fields"].copy()
    names = restart["names"]
    idx = {n: i for i, n in enumerate(names)}
    xc = ((np.arange(nx) + 0.5) * dx)[:, None] * np.ones((1, ny))
    yc = np.ones((nx, 1)) * ((np.arange(ny) + 0.5) * dx)[None, :]
    xc, yc = xc.reshape(-1), yc.reshape(-1)  # cells numbered with y fastest
    du, dv = pair_velocity(xc, yc, x_v, a, R_c, C, H)
    p0 = 101325.0
    factor = 1.0 + pair_pressure_deficit(xc, yc, x_v, a, R_c, C, H, rho_u) / p0
    rho = f[idx["RHO"]]
    ux, uy = f[idx["RHOU_X"]] / rho, f[idx["RHOU_Y"]] / rho
    rho_e = f[idx["RHOE"]] - 0.5 * rho * (ux**2 + uy**2)
    rho_new = rho * factor
    ux, uy = ux + du, uy + dv
    for n in names:
        if n == "RHO" or n.startswith("RHOY_"):
            f[idx[n]] *= factor
    f[idx["RHOU_X"]] = rho_new * ux
    f[idx["RHOU_Y"]] = rho_new * uy
    f[idx["RHOE"]] = factor * rho_e + 0.5 * rho_new * (ux**2 + uy**2)
    write_restart(os.path.join(args.run_dir, "flame.restart"), restart, f)
    print(f"{args.run_dir}: r / delta = {args.r:g}, u' / S_L = {args.u:g}: R_c = {R_c * 1e3:.4f} mm, "
          f"u' = {u_pair:.3f} m/s, vortex Reynolds number u' r / nu_u = "
          f"{u_pair * r * rho_u / ct_viscosity(args, T_u, data):.0f}, channel {nx} x {ny}, "
          f"t_stop = {t_stop / tau:.2f} flame times ({t_stop * 1e3:.4f} ms)")


def ct_viscosity(args, T_u, data):
    import cantera as ct
    gas = ct.Solution(args.mechanism, args.phase)
    gas.TPY = T_u, 101325.0, data[0, 4:]
    return gas.viscosity


if __name__ == "__main__":
    main()
