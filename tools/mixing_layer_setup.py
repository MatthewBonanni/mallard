"""A laminar reacting mixing layer run for Mallard (V11, 2D build).

    python tools/mixing_layer_setup.py FIELD.npz RUN_DIR [--dy 15e-6] [--dx 50e-6]
        [--length 0.03] [--y-min -8e-3] [--y-max 12e-3] [--growth 1.08]
        [--flow-throughs 1] [--outputs 20] [--cfl 0.8]

FIELD.npz is the boundary-layer solution from
tools/mixing_layer_reference.py --field, which also gives the streams. Fuel
enters through the inlet (x = 0) below y = 0, hot air above it (two upt
conditions, a step at the trailing edge of a splitter plate of zero
thickness); the outlet (x = --length) and the sides (y = --y-min, --y-max)
are pressure outlets at the streams' pressure, through which the free streams
also entrain.

The mesh is a tensor product of quadrilaterals written as a Gmsh file:
uniform cells of --dx along x; across, cells of --dy through the layer (where
it differs from the free streams at x = --length, plus 0.5 mm on each side),
growing by --growth per cell outside it (up to 20 --dy). Each cell starts
from the boundary-layer solution at its center (v = 0). The run lasts
--flow-throughs times length / (slower stream's velocity). Writes
RUN_DIR/mixing_layer.msh, RUN_DIR/mixing_layer.restart and RUN_DIR/input.toml;
the run writes the whole field (solut/field.pvd) every t_stop / --outputs for
tools/plot_mixing_layer.py, and restart files at the same times.
"""
import argparse
import os
import struct

import cantera as ct
import numpy as np


def graded(y0, y1, h, growth, h_max):
    """Points from y0 to y1 with cells growing from h by growth (capped at
    h_max), the last one stretched to land on y1."""
    pts = [y0]
    sign = 1.0 if y1 > y0 else -1.0
    while abs(y1 - pts[-1]) > 1.5 * h:
        pts.append(pts[-1] + sign * h)
        h = min(h * growth, h_max)
    pts.append(y1)
    return np.array(pts)


def write_gmsh(path, xs, ys):
    nx, ny = xs.size - 1, ys.size - 1
    node = np.arange(xs.size * ys.size).reshape(ys.size, xs.size) + 1  # node[j, i]
    X, Y = np.meshgrid(xs, ys)
    zones = ["inlet", "outlet", "bottom", "top"]
    lines = [np.column_stack([node[:-1, 0], node[1:, 0]]), np.column_stack([node[:-1, -1], node[1:, -1]]),
             np.column_stack([node[0, :-1], node[0, 1:]]), np.column_stack([node[-1, :-1], node[-1, 1:]])]
    with open(path, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n5\n")
        for k, name in enumerate(zones):
            f.write(f'1 {k + 1} "{name}"\n')
        f.write('2 5 "fluid"\n$EndPhysicalNames\n$Nodes\n')
        f.write(f"{node.size}\n")
        np.savetxt(f, np.column_stack([node.ravel(), X.ravel(), Y.ravel(), np.zeros(node.size)]),
                   fmt=["%d", "%.12e", "%.12e", "%.1f"])
        f.write("$EndNodes\n$Elements\n")
        n_lines = sum(len(l) for l in lines)
        f.write(f"{n_lines + nx * ny}\n")
        i = 1
        for k, l in enumerate(lines):
            ids = np.arange(i, i + len(l))
            np.savetxt(f, np.column_stack([ids, np.full(ids.size, 1), np.full(ids.size, 2), np.full(ids.size, k + 1),
                                           np.full(ids.size, k + 1), l]), fmt="%d")
            i += len(l)
        # cell (i, j) is number i + nx j, counterclockwise
        quads = np.column_stack([node[:-1, :-1].ravel(), node[:-1, 1:].ravel(), node[1:, 1:].ravel(),
                                 node[1:, :-1].ravel()])
        ids = np.arange(i, i + len(quads))
        np.savetxt(f, np.column_stack([ids, np.full(ids.size, 3), np.full(ids.size, 2), np.full(ids.size, 5),
                                       np.full(ids.size, 5), quads]), fmt="%d")
        f.write("$EndElements\n")


def column(field, x):
    """The boundary-layer solution at x: y and the fields along the eta grid."""
    xr = field["x"]
    if x <= xr[0]:
        k, w = 0, 0.0
    else:
        k = min(int(np.searchsorted(xr, x)) - 1, xr.size - 2)
        w = min((x - xr[k]) / (xr[k + 1] - xr[k]), 1.0)
    mix = lambda a: (1 - w) * a[k] + w * a[k + 1]
    return mix(field["y"]), mix(field["u"]), mix(field["T"]), mix(field["Y"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("field")
    ap.add_argument("run_dir")
    ap.add_argument("--mechanism", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                                        "mechanisms", "h2o2.yaml"))
    ap.add_argument("--phase", default="ohmech")
    ap.add_argument("--dx", type=float, default=50e-6)
    ap.add_argument("--dy", type=float, default=15e-6)
    ap.add_argument("--length", type=float, default=0.03)
    ap.add_argument("--y-min", type=float, default=-8e-3)
    ap.add_argument("--y-max", type=float, default=12e-3)
    ap.add_argument("--growth", type=float, default=1.08)
    ap.add_argument("--flow-throughs", type=float, default=1.0)
    ap.add_argument("--outputs", type=int, default=20)
    ap.add_argument("--cfl", type=float, default=0.8)
    a = ap.parse_args()

    fd = dict(np.load(a.field))
    species = [str(s) for s in fd["species"]]
    gas = ct.Solution(a.mechanism, a.phase)
    assert gas.species_names == species, "species differ between the field and the mechanism"
    p, u_f, u_a, t_f, t_a = (float(fd[k]) for k in ("p", "u_fuel", "u_air", "t_fuel", "t_air"))
    fuel = str(fd["fuel"])

    # The layer at the outlet: where u, T or a reactant differ from the free streams by 0.5% of their jumps
    y, u, T, Y = column(fd, a.length)
    k_h2, k_o2 = species.index("H2"), species.index("O2")
    dev = np.zeros(y.size, dtype=bool)
    for q in (u, T, Y[:, k_h2], Y[:, k_o2]):
        dev |= np.minimum(np.abs(q - q[0]), np.abs(q - q[-1])) > 5e-3 * np.abs(q[-1] - q[0])
    ya, yb = y[dev].min() - 5e-4, y[dev].max() + 5e-4
    na, nb = int(np.ceil(-ya / a.dy)), int(np.ceil(yb / a.dy))
    core = np.arange(-na, nb + 1) * a.dy
    ys = np.concatenate([graded(core[0], a.y_min, a.dy * a.growth, a.growth, 20 * a.dy)[::-1][:-1], core,
                         graded(core[-1], a.y_max, a.dy * a.growth, a.growth, 20 * a.dy)[1:]])
    nx = int(round(a.length / a.dx))
    xs = np.linspace(0.0, a.length, nx + 1)
    ny = ys.size - 1
    n = nx * ny
    os.makedirs(a.run_dir, exist_ok=True)
    write_gmsh(os.path.join(a.run_dir, "mixing_layer.msh"), xs, ys)

    xc, yc = 0.5 * (xs[1:] + xs[:-1]), 0.5 * (ys[1:] + ys[:-1])
    uc, Tc = np.empty((ny, nx)), np.empty((ny, nx))
    Yc = np.empty((ny, nx, len(species)))
    for i, x in enumerate(xc):
        y, u, T, Y = column(fd, x)
        uc[:, i] = np.interp(yc, y, u)
        Tc[:, i] = np.interp(yc, y, T)
        for k in range(len(species)):
            Yc[:, i, k] = np.interp(yc, y, Y[:, k])
    Yc = np.clip(Yc, 0, None)
    Yc /= Yc.sum(axis=2, keepdims=True)
    states = ct.SolutionArray(gas, n)
    states.TPY = Tc.ravel(), p, Yc.reshape(n, -1)
    rho = states.density
    rhou = rho * uc.ravel()
    rhoE = rho * states.int_energy_mass + 0.5 * rhou * uc.ravel()
    names = ["RHO", "RHOU_X", "RHOU_Y", "RHOE"] + ["RHOY_" + s for s in species] + ["T_SEED"]
    fields = [rho, rhou, np.zeros(n), rhoE] + [rho * Yc.reshape(n, -1)[:, k] for k in range(len(species))] + \
             [Tc.ravel()]
    with open(os.path.join(a.run_dir, "mixing_layer.restart"), "wb") as f:
        f.write(b"MALLARD-RESTART\0")
        f.write(struct.pack("<IIQQQd", 2, 8, n, len(names), 0, 0.0))
        for name in names:
            f.write(struct.pack("<I", len(name)) + name.encode())
        for q in fields:
            f.write(np.asarray(q, dtype="<f8").tobytes())

    t_stop = a.flow_throughs * a.length / min(u_f, u_a)
    mech_path = os.path.relpath(os.path.abspath(a.mechanism), os.path.abspath(a.run_dir))
    X = lambda s: ", ".join(f'"{k}" = {v}' for k, v in (kv.split(":") for kv in s.split(",")))
    with open(os.path.join(a.run_dir, "input.toml"), "w") as f:
        f.write(f"""# Laminar reacting mixing layer (V11) from {os.path.basename(a.field)}: fuel {fuel} at {t_f:g} K and
# {u_f:g} m/s (y < 0), air at {t_a:g} K and {u_a:g} m/s (y > 0), p = {p:g} Pa; {nx} x {ny} = {n} quadrilaterals,
# dx = {a.dx * 1e6:g} um, dy = {a.dy * 1e6:g} um over y = [{core[0] * 1e3:.3f}, {core[-1] * 1e3:.3f}] mm, growing by
# {a.growth:g} to [{a.y_min * 1e3:g}, {a.y_max * 1e3:g}] mm; length {a.length * 1e3:g} mm; t_stop = {a.flow_throughs:g} flow-through(s)
# of the slower stream

[run]
t_stop = {t_stop:.10g}
cfl = {a.cfl}

[mesh]
type = "file"
filename = "mixing_layer.msh"

[initialize]
type = "restart"
file = "mixing_layer.restart"

[[boundaries]]
name = "inlet"
where = "y < 0"
type = "upt"
u = [{u_f:.10g}, 0.0]
p = {p:.10g}
T = {t_f:.10g}
X = {{ {X(fuel)} }}

[[boundaries]]
name = "inlet"
where = "y >= 0"
type = "upt"
u = [{u_a:.10g}, 0.0]
p = {p:.10g}
T = {t_a:.10g}
X = {{ "O2" = 0.21, "N2" = 0.79 }}

[[boundaries]]
name = "outlet"
type = "p_out"
p = {p:.10g}

[[boundaries]]
name = "bottom"
type = "p_out"
p = {p:.10g}

[[boundaries]]
name = "top"
type = "p_out"
p = {p:.10g}

[numerics]
riemann_solver = "HLLC"
time_integrator = "SSPRK3"

[numerics.face_reconstruction]
type = "MUSCL"

[physics]
type = "navier_stokes"
gas = "mixture"
mechanism = "{mech_path}"
phase = "{a.phase}"
transport = "mixture_averaged"

[chemistry]
rtol = 1.0e-6
atol = 1.0e-10

[output]
check_interval = 2000

[[write_data]]
prefix = "./solut/field"
format = "vtu"
time_interval = {t_stop / a.outputs:.10g}
variables = ["RHO", "U", "P", "T", "HRR", "Y_*"]

[[write_data]]
prefix = "./restart/mixing_layer"
format = "restart"
time_interval = {t_stop / a.outputs:.10g}
""")
    print(f"{a.run_dir}: {nx} x {ny} = {n} cells, dx = {a.dx * 1e6:g} um, dy = {a.dy * 1e6:g} um over "
          f"[{core[0] * 1e3:.3f}, {core[-1] * 1e3:.3f}] mm, t_stop = {t_stop:.4e} s")


if __name__ == "__main__":
    main()
