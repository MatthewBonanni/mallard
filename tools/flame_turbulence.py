#!/usr/bin/env python3
"""Premixed flame in decaying turbulence: DNS reference and LES (examples/flame_turbulence).

    flame_turbulence.py hit OUT.restart [--n 128] [--length L] [--urms U] [--lt LT] [--seed 1]
    flame_turbulence.py init FULL.csv MECHANISM PHASE HIT.restart OUT.restart
        [--ratio R] [--thicken F] [--n 128] [--nx 224] [--length L] ...
    flame_turbulence.py laminar FULL.csv MECHANISM PHASE      # prints --q-l and --t-b for analyze
    flame_turbulence.py analyze RUN_DIR --q-l Q --t-b T [--out series.csv] [--pdf-times T1,T2] [--filter R1,R2]
    flame_turbulence.py figure LABEL=RUN_DIR/series.csv ... [--pdf LABEL=pdf.csv ...] --out fig.png

Geometry (all runs; cells numbered (i * N + j) * N + k, x slowest): a box of
NX x N x N cubic cells of the DNS width h = delta_L / 10, periodic in y and z,
with a pressure outlet on the left (burnt gas) and a slip wall on the right
(fresh gas). The coarse runs use R^3 DNS cells per cell (R = 2, 4, 8, 16), so
their initial state is the exact box filter of the DNS's.

hit: a solenoidal velocity field of the Passot-Pouquet spectrum
E(k) = 16 sqrt(2 / pi) u'^2 / k_e (k / k_e)^4 exp(-2 (k / k_e)^2) (u' the rms
of one component, k_e = sqrt(2 pi) / l_t with l_t its longitudinal integral
scale) on the periodic cube of N cells and side L, random directions and
phases from the seed, written as cell averages for a single-gas pre-run with
the fresh mixture's density, pressure and viscosity (examples/flame_turbulence/hit.toml).

init: the planar laminar flame of FULL.csv (tools/flame_reference.py, x
relative to the maximum of dT/dx) at x_f = --burnt delta_L from the outlet,
in the frame of the fresh gas (velocity u_Cantera(x) - S_L, so the fresh gas
is at rest and the burnt gas leaves through the outlet), plus the developed
velocity of HIT.restart (the pre-run's end, on the DNS mesh) times a window
w(x) that rises from 0 to 1 between --gap delta_L ahead of the flame and one
l_t further, and falls back to 0 over the last --margin delta_L before the
wall. --thicken F stretches the laminar profile by F about x_f (the TFLES
flame's thickness); the turbulence is unchanged. Every DNS cell is the
average of the profile over 4 points along x; with --ratio R the cells are
averaged in R^3 blocks.

analyze: per snapshot of RUN_DIR/solut (VTU pieces, read by cell centroid,
so any rank count), the consumption speed S_c / S_L = int HRR dV / (Q_L L^2)
(Q_L = int HRR dx of the laminar flame of FULL.csv, so a planar laminar flame
gives 1; TFLES writes its modeled rate), the resolved flame surface
int |grad c| dV / L^2 of the progress variable c = (T - T_u) / (T_b - T_u),
the efficiency-weighted surface int E |grad c| dV / L^2 (TFLES), the mean
flame position (x where <c> = 0.5) and the brush thickness
1 / max d<c>/dx of the plane-averaged progress variable; with --pdf-times
the PDF of c over the cells with 0.02 < c < 0.98, of the run and (--filter)
of the run box-filtered in R^3 blocks.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isotropic_turbulence_restart import read_restart, write_restart  # noqa: E402

# Fresh H2/air at phi = 1, 300 K, 1 atm (mechanisms/h2o2.yaml, Cantera 3.2, mixture-averaged)
RHO_U = 0.8494721085515865
MU_U = 1.8346476839430128e-05
GAMMA_U = 1.4008750762916216
P0 = 101325.0
T_U = 300.0


def passot_pouquet(n, length, urms, lt, seed):
    """Cell averages (3, n, n, n) of a solenoidal field with the Passot-Pouquet spectrum, x slowest."""
    rng = np.random.default_rng(seed)
    k1 = np.fft.fftfreq(n, 1.0 / n)
    kx, ky, kz = np.meshgrid(k1, k1, k1, indexing="ij")
    nn = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    g = rng.standard_normal((3, n, n, n)) + 1j * rng.standard_normal((3, n, n, n))
    kv = np.stack([kx, ky, kz])
    with np.errstate(invalid="ignore", divide="ignore"):
        g -= kv * (kv * g).sum(axis=0) / nn ** 2
        g /= np.sqrt((np.abs(g) ** 2).sum(axis=0))
    g[:, 0, 0, 0] = 0.0
    dk = 2 * np.pi / length
    k = nn * dk
    ke = np.sqrt(2 * np.pi) / lt
    e = 16.0 * np.sqrt(2.0 / np.pi) * urms ** 2 / ke * (k / ke) ** 4 * np.exp(-2.0 * (k / ke) ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        amp = np.sqrt(e * dk / (4 * np.pi * nn ** 2))
    amp[0, 0, 0] = 0.0
    amp[nn >= n / 2] = 0.0
    avg = np.sinc(kx / n) * np.sinc(ky / n) * np.sinc(kz / n)
    # Real part of the inverse transform of a non-Hermitian set: half the energy, so scale by sqrt(2)
    u = np.sqrt(2.0) * np.real(np.fft.ifftn(g * amp * avg * n ** 3, axes=(1, 2, 3)))
    return u


def hit(args):
    n, length = args.n, args.length
    u = passot_pouquet(n, length, args.urms, args.lt, args.seed)
    # Exact rms of the drawn field (one component), then rescale to --urms
    u *= args.urms / np.sqrt((u ** 2).mean(axis=(1, 2, 3)).mean())
    u -= u.mean(axis=(1, 2, 3), keepdims=True)
    fields = {"RHO": np.full(n ** 3, RHO_U)}
    for d, name in enumerate(["RHOU_X", "RHOU_Y", "RHOU_Z"]):
        fields[name] = RHO_U * u[d].ravel()
    fields["RHOE"] = P0 / (GAMMA_U - 1.0) + 0.5 * RHO_U * (u ** 2).sum(axis=0).ravel()
    write_restart(args.out, fields)
    nu = MU_U / RHO_U
    eps = 0.5 * args.urms ** 3 / args.lt
    print(f"{args.out}: {n}^3, L = {length * 1e3:.4f} mm, u' = {args.urms:.3f} m/s, l_t = {args.lt * 1e3:.3f} mm, "
          f"Re_t = {args.urms * args.lt / nu:.1f}, eta(eps = u'^3 / (2 l_t)) = {(nu ** 3 / eps) ** 0.25 * 1e6:.2f} um")


def laminar_profile(path):
    with open(path) as f:
        names = f.readline().strip().split(",")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    x, u, T, rho = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    species = [s[2:] for s in names[4:]]
    dTdx = np.gradient(T, x)
    i_f = np.argmax(dTdx)
    delta = (T[-1] - T[0]) / dTdx[i_f]
    return {"x": x - x[i_f], "u": u, "T": T, "rho": rho, "Y": data[:, 4:], "species": species,
            "delta": delta, "S_L": u[0]}


def block_average(a, r):
    """Average of a (nx, n, n) array over r^3 blocks."""
    if r == 1:
        return a
    nx, ny, nz = a.shape
    return a.reshape(nx // r, r, ny // r, r, nz // r, r).mean(axis=(1, 3, 5))


def init(args):
    import cantera as ct
    prof = laminar_profile(args.full)
    gas = ct.Solution(args.mechanism, args.phase)
    assert gas.species_names == prof["species"], "species differ between the flame and the mechanism"
    delta, S_L = prof["delta"], prof["S_L"]
    h = delta / 10.0
    n, nx, r = args.n, args.nx, args.ratio
    if n % r or nx % r:
        raise SystemExit("--ratio must divide --n and --nx")
    fields, _, _ = read_restart(args.hit)
    if fields["RHO"].size != n ** 3:
        raise SystemExit(f"{args.hit}: expected {n}^3 cells")
    ut = np.stack([(fields[f"RHOU_{c}"] / fields["RHO"]).reshape(n, n, n) for c in "XYZ"])
    ut -= ut.mean(axis=(1, 2, 3), keepdims=True)

    # 1D profile, cell averages over 4 points per DNS cell, stretched by F about x_f
    F = args.thicken
    x_f = args.burnt * delta
    n_sub = 4
    xs = ((np.arange(nx)[:, None] + (np.arange(n_sub)[None, :] + 0.5) / n_sub) * h).ravel()
    xi = (x_f - xs) / F  # Cantera's fresh side (x < 0) to the right of x_f

    def p1(q):
        return np.interp(xi, prof["x"], q).reshape(nx, n_sub)

    e = np.empty(prof["x"].size)
    for i in range(e.size):
        gas.TPY = prof["T"][i], P0, prof["Y"][i]
        e[i] = gas.int_energy_mass
    rho = p1(prof["rho"])
    u1 = S_L - p1(prof["u"])  # fresh gas at rest, burnt gas leaving to the left
    ns = len(prof["species"])
    col = {"RHO": rho.mean(axis=1), "RHOU_X": (rho * u1).mean(axis=1),
           "E1": (rho * (p1(e) + 0.5 * u1 * u1)).mean(axis=1), "T": p1(prof["T"]).mean(axis=1)}
    rhoY = np.stack([(rho * p1(prof["Y"][:, k])).mean(axis=1) for k in range(ns)])

    # Turbulence window along x (DNS cell centers), the periodic HIT box tiled along x from its start
    xc = (np.arange(nx) + 0.5) * h
    x_a = x_f + args.gap * delta
    x_b = nx * h - args.margin * delta
    ramp = args.lt
    w = (0.5 * (1 + np.tanh((xc - x_a - ramp) / (0.25 * ramp)))) * (0.5 * (1 - np.tanh((xc - x_b + ramp) / (0.25 * ramp))))
    w[xc < x_a] = 0.0
    w[xc > x_b] = 0.0
    i0 = int(round(x_a / h))
    idx = (np.arange(nx) - i0) % n

    # The 3D DNS-mesh field, then block averages: conservatives are cell averages, so averaging is exact
    rho3 = np.broadcast_to(col["RHO"][:, None, None], (nx, n, n))
    out = {"RHO": block_average(rho3, r)}
    ke = np.zeros((nx, n, n))
    for d, c in enumerate("XYZ"):
        ud = w[:, None, None] * ut[d][idx]
        if d == 0:
            mom = col["RHOU_X"][:, None, None] + rho3 * ud
        else:
            mom = rho3 * ud
        # Kinetic energy of the turbulence on top of the 1D energy (cross term with the 1D velocity included)
        u1c = col["RHOU_X"] / col["RHO"] if d == 0 else 0.0
        ke += rho3 * (ud * ud + (2.0 * ud * (u1c[:, None, None] if d == 0 else 0.0)))
        out[f"RHOU_{c}"] = block_average(mom, r)
    out["RHOE"] = block_average(col["E1"][:, None, None] + 0.5 * ke, r)
    for k, s in enumerate(prof["species"]):
        out[f"RHOY_{s}"] = block_average(np.broadcast_to(rhoY[k][:, None, None], (nx, n, n)), r)
    out["T_SEED"] = block_average(np.broadcast_to(col["T"][:, None, None], (nx, n, n)), r)
    write_restart(args.out, {k: np.ascontiguousarray(v).ravel() for k, v in out.items()})
    urms = np.sqrt((ut ** 2).mean())
    print(f"{args.out}: {nx // r} x {n // r} x {n // r} cells of {h * r * 1e6:.3f} um (DNS h = delta_L / 10 = "
          f"{h * 1e6:.3f} um), Lx = {nx * h * 1e3:.4f} mm, L = {n * h * 1e3:.4f} mm, delta_L = {delta * 1e3:.4f} mm, "
          f"S_L = {S_L:.5f} m/s, x_f = {x_f * 1e3:.4f} mm, F = {F:g}, u' = {urms:.3f} m/s")


# ---------------------------------------------------------------- analysis

def read_snapshot(paths, h):
    """Cell fields of the VTU pieces of one snapshot on the (i, j, k) lattice of spacing h."""
    from mallard_vtu import read_vtu_cells
    parts = [read_vtu_cells(p) for p in paths]
    t = parts[0][4]["TIME"] if "TIME" in parts[0][4] else np.nan
    cents, vals = [], {}
    for pts, conn, offs, types, arrays in parts:
        starts = np.concatenate([[0], offs[:-1]])
        cnt = offs - starts
        csum = np.zeros((len(offs), 3))
        for q in range(int(cnt.max())):
            csum += pts[conn[starts + np.minimum(q, cnt - 1)]]
        cents.append(csum / cnt[:, None])
        for kname, v in arrays.items():
            if kname == "TIME":
                continue
            vals.setdefault(kname, []).append(v)
    c = np.concatenate(cents)
    ijk = np.floor(c / h).astype(int)
    shape = tuple(ijk.max(axis=0) + 1)
    out = {}
    for kname, v in vals.items():
        a = np.concatenate(v)
        if a.ndim > 1:
            continue
        g = np.empty(shape)
        g[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = a
        out[kname] = g
    return float(np.ravel(t)[0]), out


def surface(c, h):
    """|grad c| per cell, central differences (periodic in y, z; one-sided at the x ends)."""
    gx = np.gradient(c, h, axis=0)
    gy = (np.roll(c, -1, axis=1) - np.roll(c, 1, axis=1)) / (2 * h)
    gz = (np.roll(c, -1, axis=2) - np.roll(c, 1, axis=2)) / (2 * h)
    return np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)


def laminar_heat_release(full, mechanism, phase):
    import cantera as ct
    prof = laminar_profile(full)
    gas = ct.Solution(mechanism, phase)
    hrr = np.empty(prof["x"].size)
    for i in range(hrr.size):
        gas.TPY = prof["T"][i], P0, prof["Y"][i]
        hrr[i] = gas.heat_release_rate
    return np.trapezoid(hrr, prof["x"]), prof


def snapshots(run):
    """Snapshot groups of RUN/solut: {index: [piece paths]}."""
    groups = {}
    for p in sorted(glob.glob(os.path.join(run, "solut", "*.vtu"))):
        m = re.search(r"_(\d+)(?:_\d+)?\.vtu$", os.path.basename(p))
        if m:
            groups.setdefault(int(m.group(1)), []).append(p)
    return groups


def laminar(args):
    q_l, prof = laminar_heat_release(args.full, args.mechanism, args.phase)
    print(f"--q-l {q_l:.8e} --t-b {prof['T'][-1]:.6f} --y-u {prof['Y'][0][prof['species'].index('H2')]:.8e} "
          f"--s-l {prof['S_L']:.8f}")


def analyze(args):
    import tomllib
    q_l, T_b = args.q_l, args.t_b
    with open(os.path.join(args.run, "input.toml"), "rb") as f:
        inp = tomllib.load(f)
    m = inp["mesh"]
    h = m["Ly"] / m["Ny"]
    area = m["Ly"] * m["Lz"]
    pdf_times = [float(s) for s in args.pdf_times.split(",")] if args.pdf_times else []
    filters = [int(s) for s in args.filter.split(",")] if args.filter else []
    bins = np.linspace(0.0, 1.0, 51)
    rows, pdf_rows = [], []
    groups = snapshots(args.run)
    for idx in sorted(groups):
        t, f = read_snapshot(groups[idx], h)
        V = h ** 3
        c = np.clip(1.0 - f["Y_H2"] / args.y_u, 0.0, 1.0)
        theta = np.clip((f["T"] - T_U) / (T_b - T_U), 0.0, 1.0)
        g = surface(c, h)
        cbar = c.mean(axis=(1, 2))
        x = (np.arange(c.shape[0]) + 0.5) * h
        i = np.argmax(cbar < 0.5) if cbar[0] >= 0.5 else 0
        x50 = np.interp(0.5, [cbar[i], cbar[i - 1]], [x[i], x[i - 1]]) if i > 0 else np.nan
        brush = 1.0 / np.max(np.abs(np.gradient(cbar, h)))
        row = {"t": t, "S_c": -f["OMEGA_H2"].sum() * V / (RHO_U * args.y_u * args.s_l * area),
               "S_hrr": f["HRR"].sum() * V / (q_l * area), "A_res": g.sum() * V / area,
               "A_E": (g * f["TF_E"]).sum() * V / area if "TF_E" in f else np.nan,
               "x_flame": x50, "brush": brush, "T_max": f["T"].max(),
               "E_mean": (f["TF_E"] * g).sum() / g.sum() if "TF_E" in f else 1.0,
               "F_mean": (f["TF_F"] * g).sum() / g.sum() if "TF_F" in f else 1.0}
        rows.append(row)
        print(", ".join(f"{k} {v:.5g}" for k, v in row.items()), flush=True)
        for tp in pdf_times:
            if abs(t - tp) > 1e-3 * max(tp, 1e-12) + 1e-9:
                continue
            for r in [1] + filters:
                cr = block_average(theta[: theta.shape[0] // r * r], r) if r > 1 else theta
                sel = (cr > 0.02) & (cr < 0.98)
                hist, _ = np.histogram(cr[sel], bins=bins, density=True)
                pdf_rows.append((t, r, hist))
    out = args.out or os.path.join(args.run, "series.csv")
    keys = list(rows[0])
    np.savetxt(out, np.array([[r[k] for k in keys] for r in rows]), delimiter=",", header=",".join(keys), comments="")
    if pdf_rows:
        mid = 0.5 * (bins[1:] + bins[:-1])
        with open(os.path.join(os.path.dirname(out), "pdf.csv"), "w") as fo:
            fo.write("t,filter," + ",".join(f"{v:.3f}" for v in mid) + "\n")
            for t, r, hist in pdf_rows:
                fo.write(f"{t:.6e},{r}," + ",".join(f"{v:.6g}" for v in hist) + "\n")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("hit")
    a.add_argument("out")
    a.add_argument("--n", type=int, default=128)
    a.add_argument("--length", type=float, required=True)
    a.add_argument("--urms", type=float, required=True)
    a.add_argument("--lt", type=float, required=True)
    a.add_argument("--seed", type=int, default=1)
    a = sub.add_parser("init")
    for name in ("full", "mechanism", "phase", "hit", "out"):
        a.add_argument(name)
    a.add_argument("--ratio", type=int, default=1)
    a.add_argument("--thicken", type=float, default=1.0)
    a.add_argument("--n", type=int, default=128)
    a.add_argument("--nx", type=int, default=224)
    a.add_argument("--burnt", type=float, default=8.0)
    a.add_argument("--gap", type=float, default=1.5)
    a.add_argument("--margin", type=float, default=0.2)
    a.add_argument("--lt", type=float, required=True, help="window ramp length (the integral scale)")
    a = sub.add_parser("laminar")
    for name in ("full", "mechanism", "phase"):
        a.add_argument(name)
    a = sub.add_parser("analyze")
    a.add_argument("run")
    a.add_argument("--q-l", type=float, required=True)
    a.add_argument("--t-b", type=float, required=True)
    a.add_argument("--y-u", type=float, required=True)
    a.add_argument("--s-l", type=float, required=True)
    a.add_argument("--out")
    a.add_argument("--pdf-times")
    a.add_argument("--filter")
    args = ap.parse_args()
    {"hit": hit, "init": init, "laminar": laminar, "analyze": analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
