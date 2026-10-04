"""A counterflow diffusion flame run for Mallard (V10, 3D build).

    python tools/counterflow_setup.py PROFILE.csv RUN_DIR [--cells-per-width 30]
        [--radius 5e-3] [--width 7.5e-3] [--dr-axis D] [--growth 1.08]
        [--strain-times 12] [--outputs 60] [--cfl 0.8] [--restarts 10]

PROFILE.csv is Cantera's solution at the run's nozzle velocities from
tools/counterflow_reference.py --profile (z from the fuel nozzle, u, spread
rate V, T, rho, Y), whose header gives the velocities.

The run is a quarter of two opposed round jets: x from the fuel nozzle
(x = 0) to the oxidizer nozzle (x = L, Cantera's gap), y and z from the
symmetry planes (y = 0, z = 0) to pressure outlets at W (--width). On the
nozzle planes, the jets (r < --radius) enter as plug flows at the nozzle
velocities, surrounded by N2 coflows at the same velocity (the N2 curtains of
experimental burners), so the whole of each nozzle plane is an inlet and the
products leave sideways. The mesh is a tensor product of hexahedra written as
a Gmsh file: along x, cells of FWHM / --cells-per-width (FWHM: the width of
the temperature profile at half its rise) through the flame and mixing
layer, growing by --growth per cell towards the nozzles; across, cells of
--dr-axis (default 4 axial cells) out to r = 2 mm, then growing by --growth.

Each cell starts from Cantera's solution: axial velocity u(x), radial
velocity r V(x), T(x), Y(x) at 1 atm, averaged over 16 points along x. The
run lasts --strain-times / K_ox. Writes RUN_DIR/counterflow.msh,
RUN_DIR/counterflow.restart and RUN_DIR/input.toml; the run writes the
symmetry plane z = 0 (solut/plane.pvd, its cells' values) for
tools/plot_counterflow.py, and restart files (restart/) every
t_stop / --restarts.
"""
import argparse
import os
import re
import struct

import cantera as ct
import numpy as np

P = ct.one_atm
FUEL_X = {"H2": 0.25, "N2": 0.75}
OX_X = {"O2": 0.21, "N2": 0.79}


def graded(x0, x1, h_fine, growth, h_max):
    """Points from x0 to x1 with cells growing from h_fine at x0 by growth
    (capped at h_max), the last one stretched to land on x1."""
    pts = [x0]
    h = h_fine
    sign = 1.0 if x1 > x0 else -1.0
    while abs(x1 - pts[-1]) > 1.5 * h:
        pts.append(pts[-1] + sign * h)
        h = min(h * growth, h_max)
    pts.append(x1)
    return np.array(pts)


def axial_points(L, a, b, h, growth, h_max):
    n = max(int(np.ceil((b - a) / h)), 1)
    fine = np.linspace(a, b, n + 1)
    h = (b - a) / n
    left = graded(a, 0.0, h * growth, growth, h_max)[::-1]
    right = graded(b, L, h * growth, growth, h_max)
    return np.concatenate([left[:-1], fine, right[1:]])


def write_gmsh(path, xs, ys, zs):
    nx, ny, nz = xs.size - 1, ys.size - 1, zs.size - 1
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    node = np.arange(xs.size * ys.size * zs.size).reshape(xs.size, ys.size, zs.size) + 1

    zones = ["fuel", "oxidizer", "sym_y", "sym_z", "side_y", "side_z"]
    with open(path, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n7\n")
        for k, name in enumerate(zones):
            f.write(f'2 {k + 1} "{name}"\n')
        f.write('3 7 "fluid"\n$EndPhysicalNames\n$Nodes\n')
        f.write(f"{node.size}\n")
        coords = np.column_stack([node.ravel(order="C"), X.ravel(), Y.ravel(), Z.ravel()])
        np.savetxt(f, coords, fmt=["%d", "%.12e", "%.12e", "%.12e"])
        f.write("$EndNodes\n$Elements\n")
        quads = []
        # boundary quadrilaterals: (physical tag, 4 nodes)
        for k, (axis, idx) in enumerate([(0, 0), (0, nx), (1, 0), (2, 0), (1, ny), (2, nz)]):
            sl = [slice(None)] * 3
            sl[axis] = idx
            face = node[tuple(sl)]
            a = face[:-1, :-1].ravel()
            b = face[1:, :-1].ravel()
            c = face[1:, 1:].ravel()
            d = face[:-1, 1:].ravel()
            quads.append(np.column_stack([np.full(a.size, k + 1), a, b, c, d]))
        quads = np.concatenate(quads)
        # hexahedra with x fastest: cell (i, j, k) is number i + nx (j + ny k)
        I, J, K = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
        I, J, K = (A.ravel(order="F") for A in (I, J, K))
        hexes = np.column_stack([node[I, J, K], node[I + 1, J, K], node[I + 1, J + 1, K], node[I, J + 1, K],
                                 node[I, J, K + 1], node[I + 1, J, K + 1], node[I + 1, J + 1, K + 1],
                                 node[I, J + 1, K + 1]])
        f.write(f"{len(quads) + len(hexes)}\n")
        ids = np.arange(1, len(quads) + 1)
        np.savetxt(f, np.column_stack([ids, np.full(ids.size, 3), np.full(ids.size, 2), quads[:, 0], quads[:, 0],
                                       quads[:, 1:]]), fmt="%d")
        ids = np.arange(len(quads) + 1, len(quads) + len(hexes) + 1)
        np.savetxt(f, np.column_stack([ids, np.full(ids.size, 5), np.full(ids.size, 2), np.full(ids.size, 7),
                                       np.full(ids.size, 7), hexes]), fmt="%d")
        f.write("$EndElements\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("run_dir")
    ap.add_argument("--mechanism", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                                        "mechanisms", "h2o2.yaml"))
    ap.add_argument("--phase", default="ohmech")
    ap.add_argument("--cells-per-width", type=float, default=30.0)
    ap.add_argument("--radius", type=float, default=5e-3)
    ap.add_argument("--width", type=float, default=7.5e-3)
    ap.add_argument("--dr-axis", type=float)
    ap.add_argument("--growth", type=float, default=1.08)
    ap.add_argument("--strain-times", type=float, default=12.0)
    ap.add_argument("--outputs", type=int, default=60)
    ap.add_argument("--restarts", type=int, default=10)
    ap.add_argument("--cfl", type=float, default=0.8)
    args = ap.parse_args()

    with open(args.profile) as f:
        header = f.readline()
        names = f.readline().strip().split(",")
    U_o = float(re.search(r"U_o = ([0-9.eE+-]+)", header).group(1))
    U_f = float(re.search(r"U_f = ([0-9.eE+-]+)", header).group(1))
    K_ox = float(re.search(r"K_ox = ([0-9.eE+-]+)", header).group(1))
    data = np.loadtxt(args.profile, delimiter=",", skiprows=2)
    z, u, V, T, rho = (data[:, i] for i in range(5))
    species = [n[2:] for n in names[5:]]
    Y = data[:, 5:]
    gas = ct.Solution(args.mechanism, args.phase)
    assert gas.species_names == species, "species differ between the profile and the mechanism"
    L = z[-1]

    T_in = T.min()
    hot = z[T > T_in + 0.5 * (T.max() - T_in)]
    fwhm = hot[-1] - hot[0]
    h = fwhm / args.cells_per_width
    # Fine cells wherever T or the fuel and oxygen differ from the inlets by 1% of their range
    k_h2, k_o2 = species.index("H2"), species.index("O2")
    active = ((T - T_in) > 0.01 * (T.max() - T_in)) | \
             ((Y[0, k_h2] - Y[:, k_h2]) > 0.01 * Y[0, k_h2]) & ((Y[-1, k_o2] - Y[:, k_o2]) > 0.01 * Y[-1, k_o2])
    a, b = z[active].min(), z[active].max()
    a, b = max(a - 0.5 * fwhm, 0.1 * L), min(b + 0.5 * fwhm, 0.9 * L)
    xs = axial_points(L, a, b, h, args.growth, 20 * h)
    dr = args.dr_axis or 4 * h
    r_fine = min(2e-3, 0.5 * args.width)
    n_fine = int(np.ceil(r_fine / dr))
    rs = np.concatenate([np.linspace(0.0, n_fine * dr, n_fine + 1)[:-1],
                         graded(n_fine * dr, args.width, dr * args.growth, args.growth, 1e9)])
    nx, nr = xs.size - 1, rs.size - 1
    n = nx * nr * nr
    os.makedirs(args.run_dir, exist_ok=True)
    write_gmsh(os.path.join(args.run_dir, "counterflow.msh"), xs, rs, rs)

    e = np.empty(z.size)
    for i in range(z.size):
        gas.TPY = T[i], P, Y[i]
        e[i] = gas.int_energy_mass
    n_sub = 16
    xq = xs[:-1, None] + (xs[1:] - xs[:-1])[:, None] * (np.arange(n_sub) + 0.5)[None, :] / n_sub
    prof = lambda q: np.interp(xq, z, q)
    r_q, u_q, V_q, e_q = prof(rho), prof(u), prof(V), prof(e)
    yc = 0.5 * (rs[:-1] + rs[1:])
    ns = len(species)
    rho_c = r_q.mean(axis=1)
    rhou_c = (r_q * u_q).mean(axis=1)
    rhoV_c = (r_q * V_q).mean(axis=1)
    rhoY_c = np.stack([(r_q * prof(Y[:, k])).mean(axis=1) for k in range(ns)])
    T_c = prof(T).mean(axis=1)
    # cells numbered i + nx (j + ny k)
    i_idx = np.tile(np.arange(nx), nr * nr)
    j_idx = np.tile(np.repeat(np.arange(nr), nx), nr)
    k_idx = np.repeat(np.arange(nr), nx * nr)
    y_c, z_c = yc[j_idx], yc[k_idx]
    rho_all = rho_c[i_idx]
    rhou_x = rhou_c[i_idx]
    rhou_y = rhoV_c[i_idx] * y_c
    rhou_z = rhoV_c[i_idx] * z_c
    # rho E = rho e + |rho u|^2 / (2 rho), e averaged along x
    rhoE = (r_q * e_q).mean(axis=1)[i_idx] + 0.5 * (rhou_x ** 2 + rhou_y ** 2 + rhou_z ** 2) / rho_all
    fields_names = ["RHO", "RHOU_X", "RHOU_Y", "RHOU_Z", "RHOE"] + ["RHOY_" + s for s in species] + ["T_SEED"]
    fields = [rho_all, rhou_x, rhou_y, rhou_z, rhoE] + [rhoY_c[k][i_idx] for k in range(ns)] + [T_c[i_idx]]
    with open(os.path.join(args.run_dir, "counterflow.restart"), "wb") as f:
        f.write(b"MALLARD-RESTART\0")
        f.write(struct.pack("<IIQQQd", 2, 8, n, len(fields_names), 0, 0.0))
        for name in fields_names:
            f.write(struct.pack("<I", len(name)) + name.encode())
        for field in fields:
            f.write(np.asarray(field, dtype="<f8").tobytes())

    t_stop = args.strain_times / K_ox
    mech_path = os.path.relpath(os.path.abspath(args.mechanism), os.path.abspath(args.run_dir))
    fmt = lambda X: ", ".join(f'"{s}" = {v}' for s, v in X.items())
    R2 = f"{args.radius ** 2:.10g}"
    jet = f"y * y + z * z < {R2}"
    coflow = f"y * y + z * z >= {R2}"
    with open(os.path.join(args.run_dir, "input.toml"), "w") as f:
        f.write(f"""# Counterflow diffusion flame (V10): {os.path.basename(args.profile)}, U_o = {U_o:.6g} m/s,
# U_f = {U_f:.6g} m/s, Cantera K_ox = {K_ox:.6g} 1/s; {nx} x {nr} x {nr} = {n} hexahedra,
# {h * 1e6:.2f} um along x through the flame ({args.cells_per_width:g} per FWHM of {fwhm * 1e3:.4f} mm),
# {dr * 1e6:.1f} um across near the axis; nozzle radius {args.radius * 1e3:g} mm, domain width
# {args.width * 1e3:g} mm, gap {L * 1e3:g} mm; t_stop = {args.strain_times:g} / K_ox

[run]
t_stop = {t_stop:.10g}
cfl = {args.cfl}

[mesh]
type = "file"
filename = "counterflow.msh"

[initialize]
type = "restart"
file = "counterflow.restart"

[[boundaries]]
name = "fuel"
where = "{jet}"
type = "upt"
u = [{U_f:.10g}, 0.0, 0.0]
p = {P}
T = {T_in:.10g}
X = {{ {fmt(FUEL_X)} }}

[[boundaries]]
name = "fuel"
where = "{coflow}"
type = "upt"
u = [{U_f:.10g}, 0.0, 0.0]
p = {P}
T = {T_in:.10g}
X = {{ "N2" = 1.0 }}

[[boundaries]]
name = "oxidizer"
where = "{jet}"
type = "upt"
u = [{-U_o:.10g}, 0.0, 0.0]
p = {P}
T = {T_in:.10g}
X = {{ {fmt(OX_X)} }}

[[boundaries]]
name = "oxidizer"
where = "{coflow}"
type = "upt"
u = [{-U_o:.10g}, 0.0, 0.0]
p = {P}
T = {T_in:.10g}
X = {{ "N2" = 1.0 }}

[[boundaries]]
name = "sym_y"
type = "symmetry"

[[boundaries]]
name = "sym_z"
type = "symmetry"

[[boundaries]]
name = "side_y"
type = "p_out"
p = {P}

[[boundaries]]
name = "side_z"
type = "p_out"
p = {P}

[numerics]
riemann_solver = "HLLC"
time_integrator = "SSPRK3"

[numerics.face_reconstruction]
type = "MUSCL"

[physics]
type = "navier_stokes"
gas = "mixture"
mechanism = "{mech_path}"
phase = "{args.phase}"
transport = "mixture_averaged"

[chemistry]
rtol = 1.0e-6
atol = 1.0e-10

[output]
check_interval = 2000

[[write_data]]
prefix = "./solut/plane"
format = "vtu"
geometry = "sym_z"
time_interval = {t_stop / args.outputs:.10g}
variables = ["RHO", "U", "P", "T", "HRR", "Y_*"]

[[write_data]]
prefix = "./solut/volume"
format = "vtu"
time_interval = {t_stop:.10g}
variables = ["U", "T", "HRR", "Y_H2", "Y_O2", "Y_OH", "Y_H2O"]

[[write_data]]
prefix = "./restart/counterflow"
format = "restart"
time_interval = {t_stop / args.restarts:.10g}
""")
    print(f"{args.run_dir}: {nx} x {nr} x {nr} = {n} cells, dx = {h * 1e6:.2f} um over [{a * 1e3:.3f}, {b * 1e3:.3f}] mm "
          f"(FWHM {fwhm * 1e3:.4f} mm), dr = {dr * 1e6:.1f} um, t_stop = {t_stop:.4e} s")


if __name__ == "__main__":
    main()
