#!/usr/bin/env python3
"""Initial state of the turbulent channel (examples/channel_retau180) as a restart file.

    channel_init.py input.toml OUT.restart [--seed 1] [--perturbation 0.15]

Reads the box, the cell counts, the wall-normal stretching and the gas from
the input, which must use the generated `cartesian` mesh with walls at y = 0
and y = Ly = 2h, and holds the bulk velocity at 1 (`[source] mass_flow =
[1, 0, 0]` with density 1).

The state is a turbulent-like mean profile U = (8/7) (1 - |y/h - 1|)^(1/7)
(bulk velocity 1) plus a random, three-dimensional, divergence-free
perturbation u' = curl(g A) of RMS --perturbation (in units of U_b): A is a
sum of random Fourier modes (streamwise wavelengths down to Lx / 8, spanwise
down to Lz / 12) and g = (1 - (y/h - 1)^2)^2 makes u' vanish at the walls.
It breaks down to turbulence within about 1 h/u_tau. Density 1 and uniform
pressure p_ref.

Cells are numbered as Mallard numbers the generated box: (i * Ny + j) * Nz + k
for cell i, j, k along x, y, z.
"""
import argparse
import struct
import tomllib

import numpy as np

NAMES = ["RHO", "RHOU_X", "RHOU_Y", "RHOU_Z", "RHOE"]


def nodes(n, L, beta):
    """Node coordinates of a generated mesh direction (see [mesh] stretching)."""
    j = np.arange(n + 1)
    if beta == 0:
        return L * j / n
    x = 0.5 * L * (1 + np.tanh(beta * (2 * j / n - 1)) / np.tanh(beta))
    x[0], x[-1] = 0.0, L
    return x


def grid(inp, n=None):
    m = inp["mesh"]
    if m.get("type") != "cartesian":
        raise SystemExit("channel_init.py: the input must use the generated cartesian mesh")
    n = n or (m["Nx"], m["Ny"], m["Nz"])
    L = (float(m["Lx"]), float(m["Ly"]), float(m["Lz"]))
    beta = [float(b) for b in m.get("stretching", [0, 0, 0])]
    edges = [nodes(n[d], L[d], beta[d]) for d in range(3)]
    centers = [0.5 * (e[1:] + e[:-1]) for e in edges]
    return n, L, centers


def read_restart(path):
    with open(path, "rb") as f:
        magic = f.read(16)
        version, real_size = struct.unpack("<II", f.read(8))
        n_cells, n_vars, step = struct.unpack("<QQQ", f.read(24))
        (t,) = struct.unpack("<d", f.read(8))
        if not magic.startswith(b"MALLARD-RESTART") or version not in (2, 3):
            raise SystemExit(f"{path}: expected a restart file of version 2 or 3")

        def name():
            (length,) = struct.unpack("<I", f.read(4))
            return f.read(length).decode()

        names = [name() for _ in range(n_vars)]
        if version == 3:  # named scalar attributes (statistics), not needed here
            (n_attributes,) = struct.unpack("<Q", f.read(8))
            for _ in range(n_attributes):
                name()
                f.read(8)
        dtype = "<f8" if real_size == 8 else "<f4"
        fields = np.fromfile(f, dtype=dtype, count=n_vars * n_cells).reshape(n_vars, n_cells)
    return step, t, {name: fields[v].astype(float) for v, name in enumerate(names)}


def write_restart(path, step, t, fields):
    n_cells = fields[NAMES[0]].size
    with open(path, "wb") as f:
        f.write(b"MALLARD-RESTART\0")
        f.write(struct.pack("<II", 2, 8))
        f.write(struct.pack("<QQQ", n_cells, len(NAMES), step))
        f.write(struct.pack("<d", t))
        for name in NAMES:
            f.write(struct.pack("<I", len(name)) + name.encode())
        np.stack([fields[name].ravel() for name in NAMES]).astype("<f8").tofile(f)


def curl(A, centers):
    """Curl of a vector field at the cell centers: periodic central differences in x and z."""
    x, y, z = centers
    dx, dz = x[1] - x[0], z[1] - z[0]

    def d(f, axis):
        if axis == 1:
            return np.gradient(f, y, axis=1)
        step = dx if axis == 0 else dz
        return (np.roll(f, -1, axis) - np.roll(f, 1, axis)) / (2 * step)

    return np.stack([d(A[2], 1) - d(A[1], 2), d(A[0], 2) - d(A[2], 0), d(A[1], 0) - d(A[0], 1)])


def fresh(inp, n, L, centers, seed, perturbation):
    ph = inp["physics"]
    gamma, p0 = float(ph["gamma"]), float(ph["p_ref"])
    x, y, z = np.meshgrid(*centers, indexing="ij")
    Lx, Ly, Lz = L
    h = 0.5 * Ly
    eta = y / h - 1
    s = 1 - np.abs(eta)                  # wall distance / h
    rng = np.random.default_rng(seed)

    u = 8 / 7 * s ** (1 / 7)
    # Random divergence-free perturbation u' = curl(g A): A is a sum of random
    # Fourier modes (streamwise wavelengths down to Lx / 8, spanwise down to
    # Lz / 12, three wall-normal shapes) and g = (1 - eta^2)^2 brings u' and
    # its normal derivative of v to zero at the walls
    g = (1 - eta**2) ** 2
    A = np.zeros((3,) + u.shape)
    for c in range(3):
        for _ in range(48):
            kx, kz, m = rng.integers(0, 9), rng.integers(1, 13), rng.integers(1, 4)
            alpha, beta = 2 * np.pi * kx / Lx, 2 * np.pi * kz / Lz
            phase, shift = rng.uniform(0, 2 * np.pi, 2)
            amp = rng.standard_normal() / np.hypot(alpha, beta)
            A[c] += amp * np.cos(alpha * x + beta * z + phase) * np.cos(0.5 * m * np.pi * eta + shift)
    A *= g
    up = curl(A, centers)
    up *= perturbation / np.sqrt((up**2).sum(0).mean())
    u = u + up[0]
    v, w = up[1], up[2]
    # Bulk velocity exactly 1 (volume average over the stretched cells)
    dy = np.diff(nodes(n[1], Ly, float(inp["mesh"].get("stretching", [0, 0, 0])[1])))
    u /= (u.mean(axis=(0, 2)) * dy).sum() / Ly
    rho = np.ones_like(u)
    E = p0 / (gamma - 1) + 0.5 * (u**2 + v**2 + w**2)
    return {"RHO": rho, "RHOU_X": u, "RHOU_Y": v, "RHOU_Z": w, "RHOE": E}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("input")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--perturbation", type=float, default=0.15, help="RMS of the perturbation velocity / U_b")
    args = ap.parse_args()
    with open(args.input, "rb") as f:
        inp = tomllib.load(f)
    n, L, centers = grid(inp)
    write_restart(args.out, 0, 0.0, fresh(inp, n, L, centers, args.seed, args.perturbation))
    print(f"{args.out}: {np.prod(n)} cells ({n[0]} x {n[1]} x {n[2]})")


if __name__ == "__main__":
    main()
