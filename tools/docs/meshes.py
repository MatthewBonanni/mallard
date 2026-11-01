"""2D meshes for the documentation figures: Mallard's generated meshes
(src/mesh/mesh_block.cpp: cartesian, cartesian_tri, wedge, with stretching)
and ASCII Gmsh 2.2 files, as nodes, cells and named boundary faces."""
import math
import re

import numpy as np


class Mesh2D:
    """nodes (n, 2); cells: list of node-index tuples; faces: {zone: [(a, b), ...]}."""

    def __init__(self, nodes, cells, faces):
        self.nodes = np.asarray(nodes, float)
        self.cells = cells
        self.faces = faces

    def edges(self):
        """Unique edges of all cells, as an (m, 2, 2) array of segments."""
        seen = set()
        for c in self.cells:
            for k in range(len(c)):
                a, b = c[k], c[(k + 1) % len(c)]
                seen.add((min(a, b), max(a, b)))
        e = np.array(sorted(seen))
        return self.nodes[e]

    def zone_segments(self, zone):
        f = np.array(self.faces.get(zone, []), int).reshape(-1, 2)
        return self.nodes[f]


def stretched(n, L, beta):
    j = np.arange(n + 1)
    if beta <= 0:
        return L * j / n
    s = 2.0 * j / n - 1.0
    x = 0.5 * L * (1.0 + np.tanh(beta * s) / math.tanh(beta))
    x[0], x[-1] = 0.0, L
    return x


def generated(mesh, coarsen=1, periodic=()):
    """A generated 2D mesh from a [mesh] table and the case's [[periodic]] entries; coarsen divides Nx and Ny
    (for drawing)."""
    kind = mesh.get("type", "file")
    nx = max(1, round(mesh.get("Nx", 100) / coarsen))
    ny = max(1, round(mesh.get("Ny", 100) / coarsen))
    Lx, Ly = float(mesh.get("Lx", 1.0)), float(mesh.get("Ly", 1.0))
    beta = list(mesh.get("stretching", [0.0, 0.0])) + [0.0, 0.0]
    xs, ys = stretched(nx, Lx, beta[0]), stretched(ny, Ly, beta[1])
    nodes = []
    for i in range(nx + 1):
        for j in range(ny + 1):
            x, y = xs[i], ys[j]
            if kind == "wedge" and x > 0.5:
                yb = (x - 0.5) * math.tan(math.radians(8.0))
                y = (y / Ly) * (Ly - yb) + yb
            nodes.append((x, y))
    nid = lambda i, j: i * (ny + 1) + j  # noqa: E731
    cells = []
    for i in range(nx):
        for j in range(ny):
            tr, tl, bl, br = nid(i + 1, j + 1), nid(i, j + 1), nid(i, j), nid(i + 1, j)
            if kind == "cartesian_tri":
                cells += [(br, tr, bl), (tl, bl, tr)]
            else:
                cells.append((tr, tl, bl, br))
    faces = {
        "bottom": [(nid(i, 0), nid(i + 1, 0)) for i in range(nx)],
        "top": [(nid(i, ny), nid(i + 1, ny)) for i in range(nx)],
        "left": [(nid(0, j), nid(0, j + 1)) for j in range(ny)],
        "right": [(nid(nx, j), nid(nx, j + 1)) for j in range(ny)],
    }
    for pair in periodic:
        for z in pair["zones"]:
            if z in faces:
                faces[z + "@periodic"] = faces.pop(z)
    return Mesh2D(nodes, cells, faces)


def read_gmsh22(path, plane_zone=None):
    """Read an ASCII Gmsh 2.2 file. 2D: triangles and quads with physical lines.
    3D (plane_zone given): the quads of that physical surface are the cells,
    and the edges other zones' quads share with it are their faces."""
    text = open(path).read()
    names = {}
    m = re.search(r"\$PhysicalNames\n(\d+)\n(.*?)\$EndPhysicalNames", text, re.S)
    if m:
        for line in m.group(2).strip().splitlines():
            dim, tag, name = line.split(maxsplit=2)
            names[int(tag)] = name.strip('"')
    m = re.search(r"\$Nodes\n(\d+)\n(.*?)\$EndNodes", text, re.S)
    ids, xyz = [], []
    for line in m.group(2).strip().splitlines():
        p = line.split()
        ids.append(int(p[0]))
        xyz.append([float(v) for v in p[1:4]])
    index = {g: k for k, g in enumerate(ids)}
    xyz = np.array(xyz)
    m = re.search(r"\$Elements\n(\d+)\n(.*?)\$EndElements", text, re.S)
    n_nodes = {1: 2, 2: 3, 3: 4, 8: 3, 9: 6, 10: 9, 26: 4, 21: 10, 16: 8}
    cells, faces, quads_by_zone = [], {}, {}
    for line in m.group(2).strip().splitlines():
        p = [int(v) for v in line.split()]
        etype, ntags = p[1], p[2]
        tags = p[3:3 + ntags]
        nodes = [index[g] for g in p[3 + ntags:]]
        zone = names.get(tags[0], f"physical_{tags[0]}") if tags else "unassigned"
        if plane_zone is None:
            if etype in (1, 8, 26):
                faces.setdefault(zone, []).append((nodes[0], nodes[1]))
            elif etype in (2, 9, 21):
                cells.append(tuple(nodes[:3]))
            elif etype in (3, 10, 16):
                cells.append(tuple(nodes[:4]))
        elif etype in (3, 10, 16):
            quads_by_zone.setdefault(zone, []).append(tuple(nodes[:4]))
    if plane_zone is None:
        return Mesh2D(xyz[:, :2], cells, faces)
    cells = quads_by_zone.pop(plane_zone)
    edge_set = set()
    for c in cells:
        for k in range(4):
            a, b = c[k], c[(k + 1) % 4]
            edge_set.add((min(a, b), max(a, b)))
    for zone, quads in quads_by_zone.items():
        for q in quads:
            for k in range(4):
                a, b = q[k], q[(k + 1) % 4]
                if (min(a, b), max(a, b)) in edge_set:
                    faces.setdefault(zone, []).append((a, b))
    return Mesh2D(xyz[:, :2], cells, faces)


def exprtk_to_python(expr):
    """Translate the subset of exprtk the example inputs use (ternaries,
    comparisons, ^, and/or) into a Python expression."""
    s = expr.replace("\\\n", " ").replace("\n", " ").strip()
    s = s.replace("^", "**").replace("&&", " and ").replace("||", " or ")
    return _ternary(s)


def _ternary(s):
    depth, q = 0, -1
    for k, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "?" and depth == 0:
            q = k
            break
    if q < 0:
        s = s.strip()
        if s.startswith("(") and _matching(s, 0) == len(s) - 1:
            return "(" + _ternary(s[1:-1]) + ")"
        return re.sub(r"\(([^()]*\?[^()]*)\)", lambda m: "(" + _ternary(m.group(1)) + ")", s)
    depth, nested, c = 0, 0, -1
    for k in range(q + 1, len(s)):
        ch = s[k]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "?" and depth == 0:
            nested += 1
        elif ch == ":" and depth == 0:
            if nested == 0:
                c = k
                break
            nested -= 1
    cond, a, b = s[:q], s[q + 1:c], s[c + 1:]
    return f"(({_ternary(a)}) if ({_ternary(cond)}) else ({_ternary(b)}))"


def _matching(s, i):
    depth = 0
    for k in range(i, len(s)):
        depth += {"(": 1, ")": -1}.get(s[k], 0)
        if depth == 0:
            return k
    return -1


def evaluate(expr, x, y, z=0.0, t=0.0):
    env = {n: getattr(np, n) for n in ("sin", "cos", "tan", "exp", "sqrt", "tanh", "log", "abs", "pi")}
    env.update(x=x, y=y, z=z, t=t, pi=math.pi)
    f = np.vectorize(lambda x, y: eval(exprtk_to_python(str(expr)), env, {"x": x, "y": y, "z": z, "t": t}))
    return f(x, y)
