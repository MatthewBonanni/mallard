#!/usr/bin/env python3
"""Write a Gmsh 2.2 mesh of hexahedra for the Volvo bluff-body combustor.

    make_volvo_mesh.py OUTPUT.msh [--ds 1.25e-3] [--nz 64] [--lz 0.08] [--x-in -0.2] [--x-out 0.68]

Geometry (Sjunnesson et al. 1992; the conditions of the AIAA Model Validation
for Propulsion workshop): a channel of height H = 0.12 m with an equilateral
triangular bluff body of edge D = 0.04 m, apex upstream, its base at x = 0
and centered at y = 0; spanwise periodic depth `lz` (the rig is 0.24 m wide;
LES commonly use 1-2 D). The planar mesh has four blocks, so the triangle's
three faces are mesh lines:

- upstream of the apex: x in [x_in, -a] (a = D sqrt(3) / 2), the full height;
- above and below the slanted faces: x in [-a, 0], mesh lines that leave the
  face along its normal (so the wall cells are orthogonal) and turn vertical
  toward the wall and toward the base corner;
- the wake: x in [0, x_out], the full height.

`ds` is the cell size of the shear layers and of the near wake (x < 5 D):
the edges of the base and the separation corners get `ds / 2`, the walls
`ds / 2.5`; downstream of x = 5 D the cells grow to `4 ds` at x = 10 D and
`8 ds` at the outlet, upstream of the body to `2 ds` at the inlet (synthetic turbulence enters there). The
extrusion is uniform: `nz` cells over `lz`.

Physical surfaces: "inlet", "outlet", "top", "bottom" (channel walls),
"bluffbody", "back" (z = 0) and "front" (z = lz).
"""
import argparse

import numpy as np

D = 0.04
H = 0.12
A = D * np.sqrt(3.0) / 2.0


def graded(points, spacings):
    """Node positions from points[0] to points[-1] with the spacing interpolated
    linearly between the control points; the count is rounded and the nodes
    rescaled to hit the end exactly."""
    points = np.asarray(points, float)
    spacings = np.asarray(spacings, float)
    xs = np.linspace(points[0], points[-1], 20001)
    s = np.interp(xs, points, spacings)
    # number of cells to reach x: integral of dx / s
    k = np.concatenate([[0.0], np.cumsum(np.diff(xs) / (0.5 * (s[1:] + s[:-1])))])
    n = max(1, int(round(k[-1])))
    return np.interp(np.linspace(0.0, k[-1], n + 1), k, xs)


def normalized(points, spacings):
    x = graded(points, spacings)
    return (x - x[0]) / (x[-1] - x[0])


def normal_line(p0, p1, m0, m1, fractions):
    """Points at the arc-length fractions of the cubic Hermite curve from p0 (tangent m0) to p1 (tangent m1)."""
    L = np.linalg.norm(p1 - p0)
    tau = np.linspace(0.0, 1.0, 4001)[:, None]
    h00, h10 = 2 * tau ** 3 - 3 * tau ** 2 + 1, tau ** 3 - 2 * tau ** 2 + tau
    h01, h11 = -2 * tau ** 3 + 3 * tau ** 2, tau ** 3 - tau ** 2
    c = h00 * p0 + h10 * L * m0 + h01 * p1 + h11 * L * m1
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
    f = np.asarray(fractions) * arc[-1]
    return np.column_stack([np.interp(f, arc, c[:, 0]), np.interp(f, arc, c[:, 1])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--ds", type=float, default=1.25e-3)
    ap.add_argument("--nz", type=int, default=64)
    ap.add_argument("--lz", type=float, default=0.08)
    ap.add_argument("--x-in", type=float, default=-0.2)
    ap.add_argument("--x-out", type=float, default=0.68)
    args = ap.parse_args()
    ds = args.ds

    # wall-normal distribution from the triangle's corner (0) to the wall (1)
    t = normalized([0.0, 0.25 * D, 0.5 * D, D], [0.5 * ds, ds, 1.2 * ds, ds / 2.5])
    # along the slanted faces, apex (0) to corner (1)
    s = normalized([0.0, 0.5 * D, D], [ds, ds, 0.5 * ds])
    # along the base, -D/2 to D/2
    yb = graded([-D / 2, 0.0, D / 2], [0.5 * ds, ds, 0.5 * ds])
    # streamwise
    x_up = graded([args.x_in, -A], [2 * ds, ds])
    x_wake = graded([0.0, 0.01, 5 * D, 10 * D, args.x_out], [0.5 * ds, ds, 1.2 * ds, 4 * ds, 8 * ds])

    # Planar node coordinates of each block as (ni, nj, 2) arrays, j along y
    front = []
    for sign in (1.0, -1.0):
        face = np.column_stack([-A + A * s, sign * (D / 2) * s])
        top = np.column_stack([face[:, 0], np.full(len(s), sign * H / 2)])
        normal = np.array([-0.5, sign * np.sqrt(3.0) / 2])
        lines = []
        for i in range(len(s)):
            w = s[i] ** 4
            m0 = (1 - w) * normal + w * np.array([0.0, sign])
            lines.append(normal_line(face[i], top[i], m0 / np.linalg.norm(m0), np.array([0.0, sign]), t))
        front.append(np.array(lines))
    blocks = []
    right = np.concatenate([front[1][0][::-1], front[0][0][1:]])             # upstream block's right edge
    left = np.column_stack([np.full(len(right), args.x_in), np.concatenate([-(H / 2) * t[::-1], (H / 2) * t[1:]])])
    # Cartesian over the upstream half (the inlet's cells are boxes), blending into the curved edge
    xi = np.clip((x_up - x_up[0]) / (x_up[-1] - x_up[0]) * 2.0 - 1.0, 0.0, 1.0) ** 2
    X = x_up[:, None] + xi[:, None] * (right[None, :, 0] + A)
    Y = (1 - xi)[:, None] * left[None, :, 1] + xi[:, None] * right[None, :, 1]
    blocks.append(np.stack([X, Y], axis=-1))
    blocks += front
    y_wake = np.concatenate([-(D / 2 + (H - D) / 2 * t[::-1]), yb[1:-1], D / 2 + (H - D) / 2 * t])
    X, Y = np.meshgrid(x_wake, y_wake, indexing="ij")
    blocks.append(np.stack([X, Y], axis=-1))

    # Merge nodes shared by blocks
    pts, quads = [], []
    offset = 0
    for b in blocks:
        ni, nj, _ = b.shape
        pts.append(b.reshape(-1, 2))
        idx = offset + np.arange(ni * nj).reshape(ni, nj)
        q = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], axis=-1).reshape(-1, 4)
        quads.append(q)
        offset += ni * nj
    pts = np.concatenate(pts)
    quads = np.concatenate(quads)
    key = np.round(pts / 1e-8).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    order = np.argsort(first)
    renum = np.empty_like(order)
    renum[order] = np.arange(len(order))
    nodes2 = pts[first[order]]
    quads = renum[inverse[quads]]
    # counter-clockwise
    p = nodes2[quads]
    area = 0.5 * np.sum(p[:, :, 0] * np.roll(p[:, :, 1], -1, axis=1) - np.roll(p[:, :, 0], -1, axis=1) * p[:, :, 1], axis=1)
    quads[area < 0] = quads[area < 0][:, ::-1]
    area = np.abs(area)
    assert area.min() > 0

    # boundary edges: edges of one quad
    edges = np.concatenate([quads[:, [0, 1]], quads[:, [1, 2]], quads[:, [2, 3]], quads[:, [3, 0]]])
    srt = np.sort(edges, axis=1)
    _, inv, cnt = np.unique(srt, axis=0, return_inverse=True, return_counts=True)
    bedges = edges[cnt[inv.ravel()] == 1]
    mid = 0.5 * (nodes2[bedges[:, 0]] + nodes2[bedges[:, 1]])
    eps = 1e-9
    zone = np.full(len(bedges), -1)
    zone[np.abs(mid[:, 0] - args.x_in) < eps] = 1
    zone[np.abs(mid[:, 0] - args.x_out) < eps] = 2
    zone[np.abs(mid[:, 1] - H / 2) < eps] = 3
    zone[np.abs(mid[:, 1] + H / 2) < eps] = 4
    body = (mid[:, 0] > -A - eps) & (mid[:, 0] < eps) & (np.abs(mid[:, 1]) < D / 2 + eps)
    zone[body] = 5
    assert (zone > 0).all(), "unclassified boundary edge"

    nz, lz = args.nz, args.lz
    n2 = len(nodes2)
    z = np.linspace(0.0, lz, nz + 1)
    names = ["inlet", "outlet", "top", "bottom", "bluffbody", "back", "front"]
    n_hex = len(quads) * nz
    with open(args.output, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n%d\n" % len(names))
        for k, nme in enumerate(names):
            f.write('2 %d "%s"\n' % (k + 1, nme))
        f.write("$EndPhysicalNames\n$Nodes\n%d\n" % (n2 * (nz + 1)))
        ids = np.arange(n2)
        for k in range(nz + 1):
            arr = np.column_stack([ids + k * n2 + 1, nodes2[:, 0], nodes2[:, 1], np.full(n2, z[k])])
            np.savetxt(f, arr, fmt=["%d", "%.12g", "%.12g", "%.12g"])
        f.write("$EndNodes\n")
        n_bnd = len(bedges) * nz + 2 * len(quads)
        f.write("$Elements\n%d\n" % (n_bnd + n_hex))
        eid = 1
        # side boundaries: quads (a, b, b', a') with outward normal for an edge of a ccw quad
        for k in range(nz):
            a = bedges[:, 0] + k * n2 + 1
            b = bedges[:, 1] + k * n2 + 1
            arr = np.column_stack([np.arange(eid, eid + len(a)), np.full(len(a), 3), np.full(len(a), 2), zone, zone,
                                   a, a + n2, b + n2, b])
            np.savetxt(f, arr, fmt="%d")
            eid += len(a)
        q = quads + 1
        arr = np.column_stack([np.arange(eid, eid + len(q)), np.full(len(q), 3), np.full(len(q), 2), np.full(len(q), 6),
                               np.full(len(q), 6), q[:, ::-1]])
        np.savetxt(f, arr, fmt="%d")
        eid += len(q)
        qt = q + nz * n2
        arr = np.column_stack([np.arange(eid, eid + len(q)), np.full(len(q), 3), np.full(len(q), 2), np.full(len(q), 7),
                               np.full(len(q), 7), qt])
        np.savetxt(f, arr, fmt="%d")
        eid += len(q)
        for k in range(nz):
            qa = q + k * n2
            arr = np.column_stack([np.arange(eid, eid + len(q)), np.full(len(q), 5), np.full(len(q), 2),
                                   np.zeros(len(q), int), np.ones(len(q), int), qa, qa + n2])
            np.savetxt(f, arr, fmt="%d")
            eid += len(q)
        f.write("$EndElements\n")

    h2d = np.sqrt(area)
    print(f"wrote {args.output}: {len(quads)} quads x {nz} = {n_hex} hexahedra, {n2 * (nz + 1)} nodes")
    print(f"  wake x cells {len(x_wake) - 1}, upstream {len(x_up) - 1}, slanted faces {len(s) - 1}, "
          f"base {len(yb) - 1}, corner to wall {len(t) - 1}; dz = {lz / nz:.4g} m")
    print(f"  planar cell size sqrt(A): min {h2d.min():.3g} m, max {h2d.max():.3g} m")


if __name__ == "__main__":
    main()
