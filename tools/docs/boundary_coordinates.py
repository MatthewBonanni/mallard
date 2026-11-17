"""Figures of the local coordinates of boundary faces (docs/guide/boundary_conditions.md):
the inlet of a duct in its 3D mesh with u, v and n, u and v over seven patches
(four-sided, disk and ring), and u along a curved 2D inlet.

    python tools/docs/boundary_coordinates.py [--mallard build/src/mallard] [--mallard3d build-3d/src/mallard]
                                              [names...]

The coordinates come from a NumPy port of src/boundary/patch_coordinates.cpp and
boundary_coordinates.cpp. With --mallard and --mallard3d, Mallard reads the same
meshes and the patch and the range of every coordinate in its log must match the
port's.
"""
import argparse
import functools
import heapq
import itertools
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

import figstyle
from figstyle import bare, label


# Meshes: nodes (N, dim), cells (node lists), zones {name: [face node lists]}

def _extrude(base, quads, length, layers):
    """A 2D mesh of quadrilaterals in the (y, z) plane extruded along +x into hexahedra; zones by position."""
    m = len(base)
    xs = np.linspace(0.0, length, layers + 1)
    nodes = np.vstack([np.column_stack([np.full(m, x), base]) for x in xs])
    cells = [[v + k * m for v in q] + [v + (k + 1) * m for v in q] for k in range(layers) for q in quads]
    sides = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
    count = {}
    for c in cells:
        for s in sides:
            face = tuple(c[i] for i in s)
            key = tuple(sorted(face))
            count[key] = (count[key][0] + 1, face) if key in count else (1, face)
    zones = {"inlet": [], "outlet": [], "wall": []}
    for n, face in count.values():
        if n == 1:
            x = nodes[list(face), 0]
            zones["inlet" if np.all(x == 0.0) else "outlet" if np.all(x == length) else "wall"].append(list(face))
    return nodes, cells, zones


def _structured(Y, Z):
    """Nodes and quadrilaterals of a structured (i, j) grid of the (y, z) plane."""
    ni, nj = Y.shape
    quads = [[i * nj + j, (i + 1) * nj + j, (i + 1) * nj + j + 1, i * nj + j + 1]
             for i in range(ni - 1) for j in range(nj - 1)]
    return np.column_stack([Y.ravel(), Z.ravel()]), quads


def _stretched(n, beta=1.6):
    """n + 1 points on [0, 1], finer at both ends."""
    return 0.5 * (1.0 + np.tanh(beta * np.linspace(-1.0, 1.0, n + 1)) / np.tanh(beta))


def _merge(parts):
    """Join (nodes, quads) blocks, merging coincident nodes."""
    nodes, quads, index = [], [], {}
    for pts, qs in parts:
        ids = []
        for p in pts:
            key = tuple(np.round(p, 12))
            if key not in index:
                index[key] = len(nodes)
                nodes.append(p)
            ids.append(index[key])
        quads += [[ids[v] for v in q] for q in qs]
    return np.array(nodes), quads


def duct_mesh(width=2.0, height=1.0, ny=16, nz=8, length=0.5, layers=2):
    Y, Z = np.meshgrid(width * _stretched(ny), height * _stretched(nz), indexing="ij")
    return _extrude(*_structured(Y, Z), length, layers)


def pipe_mesh(radius=1.0, m=8, k=6, length=0.5, layers=2):
    """An O-grid of the disk: a square core and four blocks out to the circle."""
    a = 0.42 * radius
    s = np.linspace(-1.0, 1.0, m + 1)
    parts = [_structured(*np.meshgrid(a * s, a * s, indexing="ij"))]
    t = np.linspace(0.0, 1.0, k + 1) ** 0.8
    for q in range(4):
        c, sn = np.cos(q * np.pi / 2), np.sin(q * np.pi / 2)
        rot = np.array([[c, -sn], [sn, c]])
        inner = np.column_stack([np.full(m + 1, a), a * s])
        phi = np.pi / 4 * s
        outer = radius * np.column_stack([np.cos(phi), np.sin(phi)])
        P = inner[:, None, :] * (1 - t[None, :, None]) + outer[:, None, :] * t[None, :, None]
        P = P @ rot.T
        parts.append(_structured(P[..., 0].T, P[..., 1].T))
    return _extrude(*_merge(parts), length, layers)


def annulus_mesh(inner=0.45, outer=1.0, nr=8, nt=48, length=0.5, layers=2):
    r = inner + (outer - inner) * _stretched(nr, 1.2)
    t = np.linspace(0.0, 2 * np.pi, nt + 1)
    R, T = np.meshgrid(r, t, indexing="ij")
    pts, quads = _structured(R * np.cos(T), R * np.sin(T))
    return _extrude(*_merge([(pts, quads)]), length, layers)


def half_ring_mesh(inner=1.0, outer=2.0, angle=180.0, nr=6, nt=36):
    """2D: half a ring; inlet on the outer arc, outlet on the inner one, walls on the straight sides."""
    r = inner + (outer - inner) * np.linspace(0, 1, nr + 1)
    t = np.deg2rad(angle) * np.linspace(0, 1, nt + 1)
    R, T = np.meshgrid(r, t, indexing="ij")
    nodes, quads = _structured(R * np.cos(T), R * np.sin(T))
    nj = nt + 1
    zones = {"inlet": [[nr * nj + j + 1, nr * nj + j] for j in range(nt)],
             "outlet": [[j, j + 1] for j in range(nt)],
             "wall": [[(i + 1) * nj, i * nj] for i in range(nr)]
             + [[i * nj + nt, (i + 1) * nj + nt] for i in range(nr)]}
    return nodes, quads, zones


def write_gmsh(path, nodes, cells, zones):
    dim = nodes.shape[1]
    types = {2: 1, 3: 2, 4: 3, 8: 5}
    names = list(zones)
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", str(len(names) + 1)]
    lines += [f'{dim - 1} {i + 1} "{n}"' for i, n in enumerate(names)] + [f'{dim} {len(names) + 1} "fluid"']
    lines += ["$EndPhysicalNames", "$Nodes", str(len(nodes))]
    lines += [f"{i + 1} " + " ".join(f"{x:.17g}" for x in (*p, 0.0)[:3]) for i, p in enumerate(nodes)]
    elements = [(i + 1, f) for i, n in enumerate(names) for f in zones[n]] + [(len(names) + 1, c) for c in cells]
    lines += ["$EndNodes", "$Elements", str(len(elements))]
    lines += [f"{k + 1} {types[len(e)]} 2 {tag} {tag} " + " ".join(str(v + 1) for v in e)
              for k, (tag, e) in enumerate(elements)]
    lines.append("$EndElements")
    path.write_text("\n".join(lines) + "\n")


# The port of src/boundary/boundary_coordinates.cpp and patch_coordinates.cpp

def face_geometry(nodes, face):
    """Center and outward area vector of a face, as Mallard computes them (a triangle fan in 3D)."""
    p = nodes[face]
    if p.shape[1] == 2:
        d = p[1] - p[0]
        return 0.5 * (p[0] + p[1]), np.array([d[1], -d[0]])
    c = p.mean(axis=0)
    a = np.cross(p - c, np.roll(p, -1, axis=0) - c)
    A = 0.5 * a.sum(axis=0)
    w = 0.5 * a @ A / np.linalg.norm(A)
    x = (w[:, None] * (c + p + np.roll(p, -1, axis=0)) / 3.0).sum(axis=0) / w.sum()
    return x, A


def _inward(nodes, cells, faces):
    """Each face's center and area vector, turned into the domain (the side of its cell)."""
    owner = {}
    for c in cells:
        cs = set(c)
        for i, f in enumerate(faces):
            if set(f) <= cs:
                owner[i] = nodes[c].mean(axis=0)
    centers, areas = [], []
    for i, f in enumerate(faces):
        x, A = face_geometry(nodes, f)
        if A @ (owner[i] - x) < 0:
            A = -A
        centers.append(x)
        areas.append(A)
    return np.array(centers), np.array(areas)


def _closest_on_segment(x, a, b):
    ab = b - a
    t = np.clip(((x - a) * ab).sum(-1) / (ab * ab).sum(-1), 0.0, 1.0)
    return a + t[..., None] * ab


def _closest_on_triangles(x, a, b, c):
    """Closest points to the points x (P, 1, 3) on the triangles (a, b, c) (T, 3): (P, T, 3)."""
    n = np.cross(b - a, c - a)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    q = x - ((x - a) * n).sum(-1, keepdims=True) * n
    inside = np.ones(q.shape[:-1], bool)
    for u, v in ((a, b), (b, c), (c, a)):
        inside &= (np.cross(v - u, q - u) * n).sum(-1) >= 0
    d = np.where(inside, np.linalg.norm(x - q, axis=-1), np.inf)
    for u, v in ((a, b), (b, c), (c, a)):
        e = _closest_on_segment(x, u, v)
        de = np.linalg.norm(x - e, axis=-1)
        better = de < d
        q[better], d[better] = e[better], de[better]
    return q


def wall_closest(nodes, walls, x):
    """The points of the wall faces nearest to the points x (P, dim): segments in 2D, triangles fanned from each
    face's lowest node in 3D."""
    x = np.atleast_2d(x)
    if nodes.shape[1] == 2:
        pieces = (nodes[[f[0] for f in walls]], nodes[[f[1] for f in walls]])
        closest = _closest_on_segment
    else:
        tris = []
        for f in walls:
            v = nodes[f]
            k = min(range(len(f)), key=lambda i: tuple(v[i]))
            v = np.roll(v, -k, axis=0)
            tris += [(v[0], v[j], v[j + 1]) for j in range(1, len(f) - 1)]
        pieces = tuple(np.array(t) for t in zip(*tris))
        closest = _closest_on_triangles
    out = np.empty_like(x)
    for i in range(0, len(x), 500):
        chunk = x[i:i + 500, None, :]
        pts = closest(chunk, *pieces)
        k = np.argmin(np.linalg.norm(pts - chunk, axis=-1), axis=1)
        out[i:i + 500] = pts[np.arange(len(k)), k]
    return out


def wall_distance(nodes, walls, x):
    return np.linalg.norm(wall_closest(nodes, walls, x) - np.atleast_2d(x), axis=-1)


def _order(nodes, faces):
    """The patch's nodes in the order of their coordinates, and its faces in those."""
    used = np.unique(np.concatenate([np.asarray(f) for f in faces]))
    X = nodes[used]
    order = np.lexsort(X.T[::-1])
    remap = {int(used[o]): i for i, o in enumerate(order)}
    return X[order], [[remap[int(v)] for v in f] for f in faces]


class Patch:
    """The port of src/boundary/patch_coordinates.cpp: u, v, u_m, v_m of the faces of a patch."""

    CORNER = np.pi / 6

    def __init__(self, nodes, faces, inward, corners=None):
        self.x, polygons = _order(nodes, faces)
        x = self.x
        tri, first, sides = [], [0], {}
        for p, n in zip(polygons, inward):
            newell = sum(np.cross(x[p[k]], x[p[(k + 1) % len(p)]]) for k in range(len(p)))
            if newell @ n < 0:
                p = p[::-1]
            k = p.index(min(p))
            p = p[k:] + p[:k]
            for a, b in zip(p, p[1:] + p[:1]):
                sides.setdefault((min(a, b), max(a, b)), []).append((a, b))
            tri += [[p[0], p[k], p[k + 1]] for k in range(1, len(p) - 1)]
            first.append(len(tri))
        self.tri, self.first = np.array(tri), np.array(first)
        c = np.cross(x[self.tri[:, 1]] - x[self.tri[:, 0]], x[self.tri[:, 2]] - x[self.tri[:, 0]])
        self.area = 0.5 * np.linalg.norm(c, axis=1)
        self.normal = c / (2 * self.area[:, None])
        self.centroid = x[self.tri].mean(axis=1)
        self.center = (self.area[:, None] * self.centroid).sum(0) / self.area.sum()
        self.mean_normal = 0.5 * c.sum(0)
        nxt = {a: b for pairs in sides.values() if len(pairs) == 1 for a, b in pairs}
        self.loops, seen = [], set()
        for i in sorted(nxt):
            if i in seen:
                continue
            loop, j = [], i
            while j not in seen:
                seen.add(j)
                loop.append(j)
                j = nxt[j]
            self.loops.append(loop)
        n = len(x)
        rows, cols, vals = [], [], []
        for t, a in zip(self.tri, self.area):
            for k in range(3):
                i, j, m = t[k], t[(k + 1) % 3], t[(k + 2) % 3]
                w = 0.25 * (x[j] - x[i]) @ (x[m] - x[i]) / a
                rows += [j, m, j, m]
                cols += [m, j, j, m]
                vals += [-w, -w, w, w]
        self.K = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
        if len(self.loops) == 2:
            self.shape = "ring"
            self._ring()
        else:
            loop = self.loops[0]
            found = []
            for k in range(len(loop)):
                a, b = x[loop[k]] - x[loop[k - 1]], x[loop[(k + 1) % len(loop)]] - x[loop[k]]
                if np.arctan2(np.linalg.norm(np.cross(a, b)), a @ b) > self.CORNER:
                    found.append(k)
            if corners is None and len(found) not in (0, 4):
                raise ValueError(f"{len(found)} corners")
            if (corners is None and not found) or (corners is not None and len(corners) == 0):
                self.shape = "disk"
                self._disk()
            else:
                self.shape = "four-sided"
                self._four_sided(found, corners)

    def solve(self, fixed, values, load=None):
        free = ~fixed
        rhs = -(self.K[free][:, fixed] @ values[fixed])
        if load is not None:
            rhs = rhs + load[free]
        out = values.copy()
        out[free] = spsolve(self.K[free][:, free].tocsc(), rhs)
        return out

    def at_faces(self, g):
        per = self.area * g[self.tri].mean(axis=1)
        return np.add.reduceat(per, self.first[:-1]) / np.add.reduceat(self.area, self.first[:-1])

    def face_centers(self):
        per = self.area[:, None] * self.centroid
        return np.add.reduceat(per, self.first[:-1]) / np.add.reduceat(self.area, self.first[:-1])[:, None]

    @staticmethod
    def reference(normal):
        axis = np.zeros(3)
        axis[(int(np.argmax(np.abs(normal))) + 1) % 3] = 1.0
        first = axis - (axis @ normal) * normal
        first /= np.linalg.norm(first)
        second = np.cross(normal, first)
        return first, second if second[np.argmax(np.abs(second))] >= 0 else -second

    def chain_center(self, chain, closed):
        p = self.x[chain + chain[:1]] if closed else self.x[chain]
        e = np.linalg.norm(np.diff(p, axis=0), axis=1)
        return (e[:, None] * 0.5 * (p[1:] + p[:-1])).sum(0) / e.sum()

    def loop_length(self, loop):
        return np.linalg.norm(np.diff(self.x[loop + loop[:1]], axis=0), axis=1).sum()

    def distance(self, d):
        """Fast marching from the nodes where d is finite."""
        x, around = self.x, {}
        for t in self.tri:
            for i in t:
                around.setdefault(int(i), []).append(t)
        heap = [(v, i) for i, v in enumerate(d) if np.isfinite(v)]
        heapq.heapify(heap)
        while heap:
            da, a = heapq.heappop(heap)
            if da > d[a]:
                continue
            for t in around[a]:
                for c in t:
                    if c == a:
                        continue
                    b = next(int(v) for v in t if v != a and v != c)
                    candidate = self._front(x[a], x[b], x[c], da, d[b])
                    if candidate < d[c] - 1e-13 * candidate:
                        d[c] = candidate
                        heapq.heappush(heap, (candidate, int(c)))
        return d

    @staticmethod
    def _front(A, B, C, da, db):
        ca, cb = A - C, B - C
        edge = min(da + np.linalg.norm(ca), db + np.linalg.norm(cb))
        if not (np.isfinite(da) and np.isfinite(db)):
            return edge
        g00, g01, g11 = ca @ ca, ca @ cb, cb @ cb
        det = g00 * g11 - g01 * g01
        if not det > 1e-28 * g00 * g11:
            return edge
        q00, q01, q11 = g11 / det, -g01 / det, g00 / det
        s11, s1d = q00 + 2 * q01 + q11, (q00 + q01) * da + (q01 + q11) * db
        disc = s1d * s1d - s11 * (q00 * da * da + 2 * q01 * da * db + q11 * db * db - 1.0)
        if not (disc >= 0 and s11 > 0):
            return edge
        t = (s1d + np.sqrt(disc)) / s11
        w0, w1 = q00 * (da - t) + q01 * (db - t), q01 * (da - t) + q11 * (db - t)
        return min(edge, t) if w0 <= 0 and w1 <= 0 and t >= max(da, db) else edge

    def conjugate(self, g, pole=None):
        """The harmonic conjugate of g at the triangles' centroids, up to its period."""
        x, tri = self.x, self.tri
        edge_tris = {}
        for t, nodes in enumerate(tri):
            for k in range(3):
                edge_tris.setdefault(frozenset((int(nodes[(k + 1) % 3]), int(nodes[(k + 2) % 3]))), []).append(t)
        at_edge, at_tri = {}, np.full(len(tri), np.nan)

        def place(t, spread=True):
            n = tri[t]
            gradient = sum(g[n[k]] / (2 * self.area[t]) * np.cross(self.normal[t], x[n[(k + 2) % 3]] - x[n[(k + 1) % 3]])
                           for k in range(3))
            turned = np.cross(self.normal[t], gradient)
            keys = [frozenset((int(n[(k + 1) % 3]), int(n[(k + 2) % 3]))) for k in range(3)]
            offset = [0.5 * turned @ (self.centroid[t] - x[n[k]]) for k in range(3)]
            known = next((k for k in range(3) if keys[k] in at_edge), None)
            at_tri[t] = 0.0 if known is None else at_edge[keys[known]] - offset[known]
            if spread:
                for k in range(3):
                    at_edge.setdefault(keys[k], at_tri[t] + offset[k])
            return keys

        start = 1 if pole == 0 else 0
        queue, seen = [start], {start}
        while queue:
            t = queue.pop(0)
            n = tri[t]
            for k in range(3):
                for other in edge_tris[frozenset((int(n[(k + 1) % 3]), int(n[(k + 2) % 3])))]:
                    if other != pole and other not in seen:
                        seen.add(other)
                        queue.append(other)
            place(t)
        if pole is not None:
            place(pole, False)
        return at_tri

    def around(self, at_tri, period, normal):
        first, second = self.reference(normal)
        base = at_tri[self.first[:-1]]
        d = at_tri - np.repeat(base, np.diff(self.first))
        d -= period * np.round(d / period)
        area = np.add.reduceat(self.area, self.first[:-1])
        raw = (base + np.add.reduceat(self.area * d, self.first[:-1]) / area) / period
        off = self.face_centers() - self.center
        theta = np.arctan2(off @ second, off @ first)
        z = [(area * np.exp(1j * (s * 2 * np.pi * raw - theta))).sum() for s in (1, -1)]
        s = 0 if abs(z[0]) >= abs(z[1]) else 1
        u = (raw if s == 0 else -raw) - np.angle(z[s]) / (2 * np.pi)
        return u - np.floor(u)

    def _four_sided(self, found, corners):
        x, loop = self.x, self.loops[0]
        n = len(loop)

        def side(a, b, step):
            out = [loop[a]]
            while a != b:
                a = (a + step) % n
                out.append(loop[a])
            return out

        if corners is not None:
            q = [int(np.argmin(np.linalg.norm(x[loop] - c, axis=1))) for c in corners]
            forward = (q[1] - q[0]) % n < (q[2] - q[0]) % n < (q[3] - q[0]) % n
            sides = [side(q[i], q[(i + 1) % 4], 1 if forward else n - 1) for i in range(4)]
        else:
            first, second = self.reference(self.mean_normal / np.linalg.norm(self.mean_normal))
            segment = [side(found[i], found[(i + 1) % 4], 1) for i in range(4)]
            center = [self.chain_center(s, False) for s in segment]
            unit = lambda v: v / np.linalg.norm(v)  # noqa: E731
            best = -np.inf
            for start in range(4):
                for forward in (True, False):
                    order = [(start + i) % 4 if forward else (start + 7 - i) % 4 for i in range(4)]
                    score = (unit(center[order[1]] - center[order[3]]) @ first
                             + unit(center[order[2]] - center[order[0]]) @ second)
                    if score > best + 1e-12:
                        best = score
                        sides = [segment[o] if forward else segment[o][::-1] for o in order]
        fixed = np.zeros(len(x), bool)
        u, v = np.zeros(len(x)), np.zeros(len(x))
        from_u, from_v = np.full(len(x), np.inf), np.full(len(x), np.inf)
        for i, nodes in enumerate(sides):
            s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(x[nodes], axis=0), axis=1))])
            s /= s[-1]
            fixed[nodes] = True
            u[nodes] = (s, 1.0, 1.0 - s, 0.0)[i]
            v[nodes] = (0.0, s, 1.0, 1.0 - s)[i]
            if i == 3:
                from_u[nodes] = 0.0
            if i == 0:
                from_v[nodes] = 0.0
        self.nodal = dict(u=self.solve(fixed, u), v=self.solve(fixed, v))
        self.corners = x[[s[0] for s in sides]]
        self.faces = dict(u=np.clip(self.at_faces(self.nodal["u"]), 0, 1), v=np.clip(self.at_faces(self.nodal["v"]), 0, 1),
                          u_m=self.at_faces(self.distance(from_u)), v_m=self.at_faces(self.distance(from_v)))

    def _arc_lengths(self, from_v, last, length_first, length_last):
        across = from_v[last].mean()
        part = np.minimum(self.faces["v_m"] / across, 1.0)
        self.faces["u_m"] = self.faces["u"] * (length_first + (length_last - length_first) * part)

    def _ring(self):
        x = self.x
        rim = [self.loop_length(loop) for loop in self.loops]
        center = [self.chain_center(loop, True) for loop in self.loops]
        axis = self.mean_normal / np.linalg.norm(self.mean_normal)
        if not np.linalg.norm(self.mean_normal) > 0.1 * self.area.sum():
            axis = (center[1] - center[0]) / np.linalg.norm(center[1] - center[0])
        k = int(np.argmax(np.abs(axis)))
        inner = 0 if center[0][k] <= center[1][k] else 1
        if abs(rim[0] - rim[1]) > 1e-6 * (rim[0] + rim[1]):
            inner = 0 if rim[0] < rim[1] else 1
        first, last = self.loops[inner], self.loops[1 - inner]
        fixed = np.zeros(len(x), bool)
        fixed[first + last] = True
        v, from_v = np.zeros(len(x)), np.full(len(x), np.inf)
        v[last] = 1.0
        from_v[first] = 0.0
        v = self.solve(fixed, v)
        from_v = self.distance(from_v)
        period = (self.K @ v)[last].sum()
        self.nodal = dict(v=v)
        self.faces = dict(u=self.around(self.conjugate(v), period, axis), v=np.clip(self.at_faces(v), 0, 1),
                          v_m=self.at_faces(from_v))
        self._arc_lengths(from_v, last, rim[inner], rim[1 - inner])

    def _disk(self):
        x, tri = self.x, self.tri
        best = (np.inf, 0, None, None)
        for t, n in enumerate(tri):
            # The triangle's point nearest to the center: these patches are flat, so it is the center or on an edge
            w = np.linalg.lstsq(np.vstack([x[n].T, np.ones(3)]), np.append(self.center, 1.0), rcond=None)[0]
            w = np.clip(w, 0.0, None)
            w /= w.sum()
            d = np.linalg.norm(w @ x[n] - self.center)
            if d < best[0] - 1e-12:
                best = (d, t, w, w @ x[n])
        _, pole, weights, center = best
        fixed = np.zeros(len(x), bool)
        fixed[self.loops[0]] = True
        load, from_v = np.zeros(len(x)), np.full(len(x), np.inf)
        load[tri[pole]] = weights
        from_v[tri[pole]] = np.linalg.norm(x[tri[pole]] - center, axis=1)
        g = self.solve(fixed, np.zeros(len(x)), load)
        u = self.around(self.conjugate(g, pole), 1.0, self.mean_normal / np.linalg.norm(self.mean_normal))
        v = np.exp(-2 * np.pi * g)
        from_v = self.distance(from_v)
        self.nodal = dict(v=v)
        self.pole = center
        self.faces = dict(u=u, v=np.clip(self.at_faces(v), 0, 1), v_m=self.at_faces(from_v))
        self._arc_lengths(from_v, self.loops[0], 0.0, self.loop_length(self.loops[0]))


def arc_length(nodes, faces):
    """u_m at each face's center of a 2D boundary, and its length."""
    at = {}
    for k, f in enumerate(faces):
        for v in f:
            at.setdefault(v, []).append(k)
    ends = [v for v, ks in at.items() if len(ks) == 1]
    axis = int(np.argmax(np.abs(nodes[ends[1]] - nodes[ends[0]])))
    v = min(ends, key=lambda v: nodes[v][axis])
    s, length, seen = np.zeros(len(faces)), 0.0, set()
    for _ in faces:
        k = next(k for k in at[v] if k not in seen)
        seen.add(k)
        w = faces[k][0] if faces[k][1] == v else faces[k][1]
        h = np.linalg.norm(nodes[w] - nodes[v])
        s[k] = length + 0.5 * h
        length += h
        v = w
    return s, length


def turned(mesh, angles=(0.5, 0.3, 0.2), shift=(0.3, -0.2, 0.7)):
    """The mesh rotated about z, y and x by angles (rad) and shifted: a patch on no axis."""
    nodes, cells, zones = mesh
    c, s = np.cos(angles), np.sin(angles)
    rz = np.array([[c[0], -s[0], 0], [s[0], c[0], 0], [0, 0, 1]])
    ry = np.array([[c[1], 0, s[1]], [0, 1, 0], [-s[1], 0, c[1]]])
    rx = np.array([[1, 0, 0], [0, c[2], -s[2]], [0, s[2], c[2]]])
    return nodes @ (rx @ ry @ rz).T + shift, cells, zones


def sheared_mesh():
    """A duct whose section is a parallelogram."""
    nodes, cells, zones = duct_mesh(width=1.6, height=1.0, ny=12, nz=8)
    nodes[:, 1] += 0.6 * nodes[:, 2]
    return nodes, cells, zones


def warped_mesh():
    """A duct with curved sides whose inlet is a saddle rather than a plane."""
    nodes, cells, zones = duct_mesh(width=1.0, height=1.0, ny=14, nz=10, length=0.6)
    p, q = nodes[:, 1].copy(), nodes[:, 2].copy()
    nodes[:, 1] = 1.8 * p + 0.22 * np.sin(np.pi * q)
    nodes[:, 2] = q + 0.18 * np.sin(np.pi * p) * (2 * q - 1)
    inlet = nodes[:, 0] == 0.0
    nodes[inlet, 0] = -0.5 * (p[inlet] - 0.5) * (q[inlet] - 0.5) - 0.25 * p[inlet] ** 2
    return nodes, cells, zones


def cone_mesh():
    """A pipe that widens along x: its wall is the side of a cone."""
    nodes, cells, zones = pipe_mesh(m=6, k=4, length=1.6, layers=10)
    nodes[:, 1:] *= (0.5 + 0.5 * nodes[:, :1] / 1.6)
    return nodes, cells, zones


# name: (mesh, the patch's zone, the zones n measures from)
CASES = {"duct": (duct_mesh, "inlet", "wall"),
         "parallelogram": (sheared_mesh, "inlet", "wall"),
         "warped": (warped_mesh, "inlet", "wall"),
         "pipe": (pipe_mesh, "inlet", "wall"),
         "pipe_turned": (lambda: turned(pipe_mesh()), "inlet", "wall"),
         "annulus": (annulus_mesh, "inlet", "wall"),
         "cone_side": (cone_mesh, "wall", "inlet"),
         "pipe_side": (lambda: pipe_mesh(m=6, k=4, length=1.6, layers=10), "wall", "inlet"),
         "duct_long": (lambda: duct_mesh(width=1.6, height=1.0, ny=10, nz=6, length=2.4, layers=8), "inlet", "wall"),
         "half_ring": (half_ring_mesh, "inlet", "wall")}


@functools.cache
def case(name):
    """The mesh of a case and the local coordinates of its patch's faces."""
    make, zone, wall = CASES[name]
    nodes, cells, zones = make()
    faces = zones[zone]
    centers, areas = _inward(nodes, cells, faces)
    n_m = wall_distance(nodes, zones[wall], centers)
    used = np.unique(np.concatenate([np.asarray(f) for f in faces]))
    c = dict(n_m=n_m, n=n_m / max(n_m.max(), wall_distance(nodes, zones[wall], nodes[used]).max()))
    patch = None
    if nodes.shape[1] == 3:
        patch = Patch(nodes, faces, areas)
        c.update(patch.faces)
        if patch.shape != "four-sided":
            c.update(r=c["v"], r_m=c["v_m"], theta=2 * np.pi * c["u"])
    else:
        c["u_m"], length = arc_length(nodes, faces)
        c["u"] = c["u_m"] / length
    return dict(nodes=nodes, cells=cells, zones=zones, faces=faces, wall=wall, zone=zone, centers=centers, coords=c,
                patch=patch)


# The check against Mallard

def check(binary, name):
    """Mallard's patch and coordinate ranges of the case (from its log) must match the port's."""
    data = case(name)
    dim = data["nodes"].shape[1]
    used = list(data["coords"])
    zero = ", ".join(["0.0"] * dim)
    others = "".join(f'[[boundaries]]\nname = "{z}"\ntype = "symmetry"\n' for z in data["zones"] if z != data["zone"])
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        write_gmsh(work / "mesh.msh", data["nodes"], data["cells"], data["zones"])
        (work / "input.toml").write_text(f"""[run]
n_steps = 1
cfl = 0.5
[mesh]
type = "file"
filename = "mesh.msh"
[initialize]
type = "constant"
u = [{zero}]
p = 1.0
T = 1.0
[[boundaries]]
name = "{data['zone']}"
type = "inlet_state"
u = "0.1 + 0 * ({' + '.join(used)})"
p = 1.0
T = 1.0
walls = ["{data['wall']}"]
{others}[numerics.face_reconstruction]
type = "MUSCL"
[physics]
type = "euler"
gamma = 1.4
p_ref = 1.0
T_ref = 1.0
rho_ref = 1.0
""")
        log = subprocess.run([str(Path(binary).resolve()), "check", "input.toml", "--kokkos-num-threads=2"], cwd=work,
                             capture_output=True, text=True, check=True).stdout
    lines = log.splitlines()
    start = next((i for i, line in enumerate(lines) if "local coordinates:" in line), None)
    if start is None:
        raise RuntimeError(f"{name}: no local coordinates in Mallard's log:\n{log}")
    indent = len(lines[start + 1]) - len(lines[start + 1].lstrip())
    wrapped = [lines[start]] + list(itertools.takewhile(lambda line: line[:indent].isspace(), lines[start + 1:]))
    text = " ".join(" ".join(wrapped).split()).split("local coordinates: ")[1]
    if dim == 3 and not text.startswith(data["patch"].shape.replace("four-sided", "four-sided patch")):
        raise RuntimeError(f"{name}: Mallard's patch is {text.split(',')[0]}, the port's {data['patch'].shape}")
    for key in used:
        lo, hi = (float(v) for v in re.search(rf"(?:^|; ){key} (\S+) to (\S+?)(?:;|$)", text).groups())
        mine = data["coords"][key]
        if not np.allclose([lo, hi], [mine.min(), mine.max()], rtol=2e-5, atol=2e-6):
            raise RuntimeError(f"{name}: Mallard's {key} {lo} to {hi}, the port's {mine.min()} to {mine.max()}")
    print(f"{name}: the port matches Mallard ({text.split(';')[0].split(', (u, v)')[0].split(', v = 0')[0]}: "
          + ", ".join(used) + ")")


# Figures

def _isolines(x, tri, values, levels, wrap=False):
    """Segments where the piecewise linear function with the nodal (or per-triangle) values crosses the levels."""
    segments = []
    for t, n in enumerate(tri):
        g = values[t] if values.ndim == 2 else values[n]
        if wrap:
            g = g - np.round(g - g[0])
        for level in levels:
            for shift in ((-1.0, 0.0, 1.0) if wrap else (0.0,)):
                pts = []
                for a, b in ((0, 1), (1, 2), (2, 0)):
                    da, db = g[a] - level - shift, g[b] - level - shift
                    if (da < 0) != (db < 0):
                        pts.append(x[n[a]] + da / (da - db) * (x[n[b]] - x[n[a]]))
                if len(pts) == 2:
                    segments.append(pts)
    return segments


def _nodal_u(patch):
    """u at the three nodes of each triangle of a ring or a disk (it wraps), from the faces' u and theta of the nodes."""
    first, second = patch.reference(patch.mean_normal / np.linalg.norm(patch.mean_normal)
                                    if np.linalg.norm(patch.mean_normal) > 0.1 * patch.area.sum() else _axis(patch))
    off = patch.x - patch.center
    theta = np.arctan2(off @ second, off @ first) / (2 * np.pi)
    return (theta - np.floor(theta))[patch.tri]


def _axis(patch):
    c = [patch.chain_center(loop, True) for loop in patch.loops]
    return (c[1] - c[0]) / np.linalg.norm(c[1] - c[0])


def _draw_patch(ax, P, data, key, flat, levels):
    """The faces of a case's patch in the color of `key`, and its lines; flat patches seen along their normal."""
    patch, cmap = data["patch"], plt.get_cmap(P["cmap"])
    values = data["coords"][key]
    polys3 = [data["nodes"][f] for f in data["faces"]]
    if key == "u" and patch.shape != "four-sided":
        lines = _isolines(patch.x, patch.tri, _nodal_u(patch), levels, wrap=True)
    else:
        lines = _isolines(patch.x, patch.tri, patch.nodal[key], levels)
    colors = cmap(plt.Normalize(0, 1)(values))
    if flat is not None:
        to2 = lambda p: np.column_stack([np.asarray(p) @ flat[0], np.asarray(p) @ flat[1]])  # noqa: E731
        ax.add_collection(PolyCollection([to2(p) for p in polys3], facecolors=colors, edgecolors=P["bg"], linewidths=0.3))
        ax.add_collection(LineCollection([to2(s) for s in lines], colors=P["fg"], linewidths=0.9))
        bare(ax)
        ax.autoscale_view()
    else:
        ax.add_collection3d(Poly3DCollection(polys3, facecolors=colors, edgecolors=P["bg"], linewidths=0.3))
        ax.add_collection3d(Line3DCollection(lines, colors=P["fg"], linewidths=0.9))
        lo, hi = patch.x.min(0), patch.x.max(0)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi - lo, zoom=1.35)
        ax.set_axis_off()


def boundary_patch(P):
    """A duct's inlet in its mesh, colored by u with the lines of u and v, and one face's coordinates."""
    data = case("duct_long")
    nodes, zones, c, patch = data["nodes"], data["zones"], data["coords"], data["patch"]
    L, W, H = nodes.max(0)
    fig = plt.figure(figsize=(10.4, 5.2))
    ax = fig.add_axes([-0.02, -0.05, 0.70, 1.1], projection="3d", computed_zorder=False)
    back = [nodes[f] for f in zones["wall"] if nodes[f][:, 2].max() == 0.0 or nodes[f][:, 1].min() == W]
    ax.add_collection3d(Poly3DCollection(back, facecolors=P["faint"], edgecolors=P["mesh"], linewidths=0.7))
    cmap = plt.get_cmap(P["cmap"])
    ax.add_collection3d(Poly3DCollection([nodes[f] for f in zones["inlet"]], facecolors=cmap(c["u"]), alpha=0.9,
                                         edgecolors=P["bg"], linewidths=0.4))
    levels = np.arange(0.25, 1.0, 0.25)
    for key, color in (("u", P["fg"]), ("v", P["fg"])):
        ax.add_collection3d(Line3DCollection(_isolines(patch.x, patch.tri, patch.nodal[key], levels), colors=color,
                                             linewidths=1.1))
    for (a, b), text in zip(((0, 1), (0, 3)), (r"$u$", r"$v$")):
        p, q = patch.corners[a], patch.corners[b]
        out = -0.12 * (patch.corners[3 if b == 1 else 1] - p) / np.linalg.norm(patch.corners[3 if b == 1 else 1] - p)
        ax.quiver(*(p + out), *(q - p), color=P["blue"], lw=2.0, arrow_length_ratio=0.08)
        ax.text(*(0.5 * (p + q) + 2.2 * out), text, color=P["blue"], fontsize=14, ha="center", va="center")
    for corner, text in zip(patch.corners, ("(0, 0)", "(1, 0)", "(1, 1)", "(0, 1)")):
        away = corner - patch.center
        ax.text(*(corner + 0.16 * away / np.linalg.norm(away)), text, color=P["muted"], fontsize=10.5, ha="center",
                va="center")
    target = int(np.argmin(np.linalg.norm(data["centers"] - [0.0, 0.62 * W, 0.7 * H], axis=1)))
    ax.add_collection3d(Poly3DCollection([nodes[zones["inlet"][target]]], facecolors=P["flame"], edgecolors=P["fg"],
                                         linewidths=1.0))
    x = data["centers"][target]
    wall = wall_closest(nodes, zones["wall"], x)[0]
    ax.plot(*np.array([x, wall]).T, color=P["teal"], lw=2.6)
    ax.text(*(0.5 * (x + wall) + [0, -0.07, 0.03]), r"$n_m$", color=P["teal"], fontsize=13, ha="left")
    ax.text(0.7 * L, 0.45 * W, 0.0, "wall", color=P["muted"], fontsize=12, ha="center")
    ax.text(0.0, 0.5 * W, -0.5 * H, "inlet, colored by $u$", color=P["fg"], fontsize=12, ha="center", va="top")
    ax.set_xlim(-0.1, L)
    ax.set_ylim(-0.2, W + 0.1)
    ax.set_zlim(-0.2, H + 0.1)
    ax.set_box_aspect((L + 0.1, W + 0.3, H + 0.3), zoom=0.98)
    ax.view_init(elev=20, azim=-140)
    ax.set_axis_off()

    tx = fig.add_axes([0.67, 0.06, 0.33, 0.88])
    tx.axis("off")
    rows = [("the patch: four-sided", None, None),
            (r"$u$", "0 on one side, 1 on the opposite one", P["blue"]),
            (r"$v$", "0 on a third side, 1 on the fourth", P["blue"]),
            (r"$n$", "0 on the walls, 1 farthest from them", P["teal"]),
            (r"…$\_$m", "the same in meters", P["muted"]),
            ("the highlighted face", None, None)]
    rows += [(f"${k}$ = {c[k][target]:.2f}", f"${k}_m$ = {c[k + '_m'][target]:.2f} m", P["teal"] if k == "n" else P["blue"])
             for k in ("u", "v", "n")]
    y = 1.0
    for head, text, color in rows:
        if text is None:
            y -= 0.05 if y < 1 else 0
            tx.text(0.0, y, head, fontsize=12.5, fontweight="bold", va="top", color=P["fg"])
            y -= 0.09
            continue
        tx.text(0.03, y, head, fontsize=12, va="top", color=color)
        tx.text(0.36, y, text, fontsize=11, va="top", color=P["muted"])
        y -= 0.085
    tx.set_xlim(0, 1)
    tx.set_ylim(0, 1)
    return fig


# name, title, the two directions a flat patch is seen in (None: in 3D), the view of a 3D one
SURFACES = (("duct", "rectangle", ((0, 1, 0), (0, 0, 1)), None),
            ("parallelogram", "parallelogram", ((0, 1, 0), (0, 0, 1)), None),
            ("warped", "warped, curved sides", None, (18, -155)),
            ("pipe", "circle", ((0, 1, 0), (0, 0, 1)), None),
            ("annulus", "annulus", ((0, 1, 0), (0, 0, 1)), None),
            ("cone_side", "side of a cone", None, (18, -60)),
            ("pipe_side", "side of a pipe", None, (18, -60)))


def boundary_coordinates(P):
    """u (top) and v (bottom) over seven patches: the face values and contour lines."""
    fig = plt.figure(figsize=(14.0, 4.9))
    levels = np.arange(0.125, 1.0, 0.125)
    for col, (name, title, flat, view) in enumerate(SURFACES):
        data = case(name)
        for row, key in enumerate(("u", "v")):
            box = [0.035 + col * 0.138, 0.50 - row * 0.46, 0.13, 0.40]
            if flat is None:
                ax = fig.add_axes(box, projection="3d", computed_zorder=False)
                ax.view_init(elev=view[0], azim=view[1])
            else:
                ax = fig.add_axes(box)
            _draw_patch(ax, P, data, key, None if flat is None else np.array(flat, float), levels)
            if row == 0:
                fig.text(box[0] + 0.5 * box[2], 0.94, title, ha="center", va="bottom", fontsize=12, color=P["fg"])
                fig.text(box[0] + 0.5 * box[2], 0.905, data["patch"].shape, ha="center", va="bottom", fontsize=10,
                         color=P["muted"])
        if col == 0:
            for row, key in enumerate(("u", "v")):
                fig.text(0.012, 0.70 - row * 0.46, f"${key}$", ha="center", va="center", fontsize=15, color=P["fg"])
    fig.text(0.5, 0.012, "each face in the color of its value, 0 to 1; lines every 1/8", ha="center", fontsize=10.5,
             color=P["muted"])
    return fig


def boundary_arc_length(P):
    """A curved 2D inlet: u along it and a face's n, and both along the inlet."""
    data = case("half_ring")
    nodes, zones, c = data["nodes"], data["zones"], data["coords"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.4, 4.2), gridspec_kw=dict(width_ratios=[1.25, 1.0], wspace=0.22))
    ax.add_collection(PolyCollection([nodes[q] for q in data["cells"]], facecolors=P["faint"], edgecolors=P["mesh"],
                                     linewidths=0.5))
    for f in zones["wall"]:
        ax.plot(*nodes[f].T, color=P["fg"], lw=2.6, solid_capstyle="butt")
    cmap = plt.get_cmap(P["cmap"])
    for k, f in enumerate(zones["inlet"]):
        ax.plot(*nodes[f].T, color=cmap(c["u"][k]), lw=7, solid_capstyle="butt")
    k = int(np.argmin(np.abs(np.arctan2(data["centers"][:, 1], data["centers"][:, 0]) - np.deg2rad(55))))
    x = data["centers"][k]
    wall = wall_closest(nodes, zones["wall"], x)[0]
    ax.plot(*np.array([x, wall]).T, color=P["teal"], lw=2)
    label(ax, *(0.5 * (x + wall) + [0.14, 0.0]), r"$n_m$", P, size=13, color=P["teal"])
    start = data["centers"][np.argmin(c["u"])]
    a = np.linspace(np.arctan2(start[1], start[0]), np.arctan2(x[1], x[0]), 80)
    ax.plot(2.22 * np.cos(a), 2.22 * np.sin(a), color=P["gold"], lw=1.8)
    ax.plot(2.22 * np.cos(a[-1]), 2.22 * np.sin(a[-1]), "o", color=P["gold"], ms=4)
    label(ax, 2.42 * np.cos(a.mean()), 2.42 * np.sin(a.mean()), r"$u_m$", P, size=13, color=P["gold"])
    ax.plot(*x, "o", color=P["fg"], ms=5, zorder=5)
    label(ax, 2.2 * np.cos(a[0]) + 0.1, 2.2 * np.sin(a[0]) - 0.18, "$u$ = 0", P, size=11, color=P["fg"], ha="left")
    label(ax, -2.3, -0.18, "$u$ = 1", P, size=11, color=P["fg"], ha="right")
    label(ax, -1.5, -0.15, "wall", P, size=11, color=P["muted"], va="top")
    label(ax, 1.5, -0.15, "wall", P, size=11, color=P["muted"], va="top")
    label(ax, 2.2, 1.25, "inlet,\ncolored\nby $u$", P, size=11, color=P["fg"], ha="left")
    bare(ax)
    ax.set_xlim(-2.95, 2.95)
    ax.set_ylim(-0.45, 2.55)

    order = np.argsort(c["u"])
    for key, color in (("u", P["gold"]), ("n", P["teal"])):
        bx.plot(c["u_m"][order], c[key][order], "o-", ms=3.5, lw=1.6, color=color, label=f"${key}$")
    bx.set_xlabel(r"$u_m$ of the inlet face [m]")
    bx.set_ylabel("coordinate")
    bx.legend(loc="upper left")
    return fig


def figures(mallard=None, mallard3d=None, names=None):
    if mallard3d:
        for name in CASES:
            if name != "half_ring":
                check(mallard3d, name)
    if mallard:
        check(mallard, "half_ring")
    for name, draw in (("boundary_arc_length", boundary_arc_length), ("boundary_coordinates", boundary_coordinates),
                       ("boundary_patch", boundary_patch)):
        if not names or name in names:
            figstyle.save(name, draw)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mallard", help="2D Mallard executable, to check the 2D coordinates")
    ap.add_argument("--mallard3d", help="3D Mallard executable, to check the 3D coordinates")
    ap.add_argument("names", nargs="*")
    ns = ap.parse_args()
    figures(ns.mallard, ns.mallard3d, ns.names)


if __name__ == "__main__":
    sys.exit(main())
