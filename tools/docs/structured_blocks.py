"""Figure of the cell numbering (docs/guide/performance.md, docs/design/structured_blocks.md):
the structured blocks that `[mesh] cell_order = "auto"` finds in a 2D mesh of quadrilaterals and
triangles, periodic in x, and the order their cells are numbered in.

    python tools/docs/structured_blocks.py [--mallard build/src/mallard]

The blocks come from a NumPy port of src/mesh/structured_blocks.cpp (2D). With --mallard, Mallard
checks the same mesh and the number of blocks and their share of the cells in its log's cell order
must match the port's.
"""
import argparse
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt

import figstyle
from figstyle import bare

H = 0.25                 # cell size
NX, NY = 24, 8           # columns and rows of the node grid's cells
MIN_EXTENT = 3           # 2 layers + 1: MUSCL reads one vertex layer


# ------------------------------------------------------------------ the mesh
def mesh():
    """Nodes, cells (counterclockwise node lists, in file order) of the 6 x 2 strip.

    Quadrilaterals in columns 0-5 and 18-23 (numbered row by row), which the periodic boundary at
    x = 0, 6 joins; a block of quadrilaterals in columns 8-15 of the upper half numbered column by
    column, and a short column of quadrilaterals beside it; triangles elsewhere.
    """
    nid = lambda i, j: j * (NX + 1) + i  # noqa: E731
    X, Y = np.meshgrid(np.arange(NX + 1) * H, np.arange(NY + 1) * H)
    nodes = np.column_stack([X.ravel(), Y.ravel()])
    quad = lambda i, j: [nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)]  # noqa: E731
    cells = [quad(i, j) for j in range(NY) for i in range(6)]
    cells += [quad(i, j) for j in range(NY) for i in range(18, NX)]
    cells += [quad(i, j) for i in range(8, 16) for j in range(4, NY)]
    cells += [quad(i, j) for j in range(4, 6) for i in range(16, 17)]
    taken = {(i, j) for i in range(6) for j in range(NY)} | {(i, j) for i in range(18, NX) for j in range(NY)}
    taken |= {(i, j) for i in range(8, 16) for j in range(4, NY)} | {(i, j) for i in range(16, 17) for j in range(4, 6)}
    for j in range(NY):
        for i in range(NX):
            if (i, j) in taken:
                continue
            q = quad(i, j)
            if (i + j) % 2:
                cells += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
            else:
                cells += [[q[0], q[1], q[3]], [q[1], q[2], q[3]]]
    return nodes, cells


def boundary_zones(nodes, cells):
    edges = {}
    for c in cells:
        for a, b in zip(c, c[1:] + c[:1]):
            k = (min(a, b), max(a, b))
            edges[k] = (edges[k][0] + 1, (a, b)) if k in edges else (1, (a, b))
    zones = {"left": [], "right": [], "wall": []}
    for count, (a, b) in edges.values():
        if count > 1:
            continue
        x = nodes[[a, b], 0]
        zones["left" if np.all(x < 1e-9) else "right" if np.all(x > NX * H - 1e-9) else "wall"].append((a, b))
    return zones


def node_keys(nodes):
    """Periodic images of a node share its key: nodes at x = 6 take the key of their partner at x = 0."""
    key = np.arange(len(nodes))
    right = np.flatnonzero(nodes[:, 0] > NX * H - 1e-9)
    key[right] = right - NX
    return key


def write_gmsh(path, nodes, cells, zones):
    names = list(zones)
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", str(len(names) + 1)]
    lines += [f'1 {k + 1} "{n}"' for k, n in enumerate(names)] + [f'2 {len(names) + 1} "fluid"', "$EndPhysicalNames"]
    lines += ["$Nodes", str(len(nodes))] + [f"{i + 1} {x:.17g} {y:.17g} 0" for i, (x, y) in enumerate(nodes)]
    elements = [(1, k + 1, f) for k, n in enumerate(names) for f in zones[n]]
    elements += [({3: 2, 4: 3}[len(c)], len(names) + 1, c) for c in cells]
    lines += ["$EndNodes", "$Elements", str(len(elements))]
    lines += [f"{e + 1} {t} 2 {p} {p} " + " ".join(str(v + 1) for v in nodes_) for e, (t, p, nodes_) in enumerate(elements)]
    path.write_text("\n".join(lines + ["$EndElements"]) + "\n")


# ------------------------------------------- port of structured_blocks (2D)
CORNER_BITS = (0b00, 0b01, 0b11, 0b10)   # local (i, j) bits of the quadrilateral's nodes
CORNER_OF = (0, 1, 3, 2)


def face_key(c, side):
    axis, bit = divmod(side, 2)
    return tuple(sorted(c[CORNER_OF[b]] for b in range(4) if ((b >> axis) & 1) == bit))


def frame_across(c, fc, o, o_side, g, up):
    axis, grows = [0, 0], [False, False]
    axis[g] = o_side // 2
    grows[g] = (o_side % 2 == 0) == up
    bits_a = (1 if fc[1][g] == up else 0) << fc[0][g]
    h = 1 - g
    node_a, node_b = c[CORNER_OF[bits_a]], c[CORNER_OF[bits_a | (1 << fc[0][h])]]
    oa = next((CORNER_BITS[k] for k in range(4) if o[k] == node_a), None)
    ob = next((CORNER_BITS[k] for k in range(4) if o[k] == node_b), None)
    if oa is None or ob is None or bin(oa ^ ob).count("1") != 1:
        return None
    axis[h] = (oa ^ ob).bit_length() - 1
    if axis[h] == axis[g]:
        return None
    grows[h] = ((ob & (oa ^ ob)) != 0) == fc[1][h]
    return tuple(axis), tuple(grows)


def structured_blocks(cells, key, min_extent):
    """Blocks [(n, cells i fastest)] in the order of their lowest cell, as src/mesh/structured_blocks.cpp."""
    n_cells = len(cells)
    keyed = [[int(key[v]) for v in c] for c in cells]
    faces = {}
    for c in range(n_cells):
        if len(cells[c]) == 4:
            for s in range(4):
                faces.setdefault(face_key(keyed[c], s), []).append((c, s))
    across = [[None] * 4 for _ in range(n_cells)]
    for entries in faces.values():
        if len(entries) == 2 and entries[0][0] != entries[1][0]:
            (a, sa), (b, sb) = entries
            seam = face_key(cells[a], sa) != face_key(cells[b], sb)
            across[a][sa], across[b][sb] = (b, sb, seam), (a, sa, seam)
    labeled, used = [False] * n_cells, [0] * n_cells
    frame, coord, member = [None] * n_cells, [None] * n_cells, [None] * n_cells
    boxes = []
    for seed in range(n_cells):
        if len(cells[seed]) != 4 or labeled[seed]:
            continue
        at, members, neighbor, deferred = {}, [seed], [[None] * 4], []
        labeled[seed], member[seed], frame[seed], coord[seed] = True, 0, ((0, 1), (True, True)), (0, 0)
        at[(0, 0)] = seed

        def visit(q, g, up, defer):
            c = members[q]
            side = 2 * frame[c][0][g] + (1 if frame[c][1][g] == (up == 1) else 0)
            if across[c][side] is None:
                return
            o, o_side, seam = across[c][side]
            if seam and defer and not labeled[o]:
                deferred.append((q, g, up))
                return
            fo = frame_across(keyed[c], frame[c], keyed[o], o_side, g, up == 1)
            if fo is None:
                return
            x = list(coord[c])
            x[g] += 1 if up else -1
            x = tuple(x)
            if not labeled[o]:
                if x in at:
                    return
                at[x] = o
                labeled[o], frame[o], coord[o], member[o] = True, fo, x, len(members)
                members.append(o)
                neighbor.append([None] * 4)
            elif member[o] is None or coord[o] != x or frame[o] != fo:
                return
            neighbor[q][2 * g + up] = o

        q = d = 0
        while True:
            while q < len(members):
                for g in range(2):
                    for up in range(2):
                        visit(q, g, up, True)
                q += 1
            if d == len(deferred):
                break
            visit(*deferred[d], False)
            d += 1
        gap, flip = [None, None], [False, False]
        for g in range(2):
            for up in range(2):
                o = neighbor[0][2 * g + up]
                if o is None or (gap[g] is not None and abs(o - seed) >= gap[g]):
                    continue
                gap[g], flip[g] = abs(o - seed), (o > seed) != (up == 1)
        axes = sorted(range(2), key=lambda a: np.inf if gap[a] is None else gap[a])

        def index(c):
            return tuple(-coord[c][axes[a]] if flip[axes[a]] else coord[c][axes[a]] for a in range(2))

        def nxt(c, a):
            return neighbor[member[c]][2 * axes[a] + (0 if flip[axes[a]] else 1)]

        members.sort(key=lambda c: index(c)[::-1])
        for start in members:
            if used[start]:
                continue
            n, box = [1, 1], [start]
            used[start] = 2
            for a in range(2):
                layer = box
                while True:
                    ahead = [nxt(c, a) for c in layer]
                    ok = all(c is not None and not used[c] for c in ahead)
                    stride = 1
                    for b in range(a):
                        if not ok:
                            break
                        for k in range(len(ahead)):
                            if (k // stride) % n[b] + 1 < n[b] and nxt(ahead[k], b) != ahead[k + stride]:
                                ok = False
                                break
                        stride *= n[b]
                    if not ok:
                        break
                    for c in ahead:
                        used[c] = 2
                    box = box + ahead
                    layer = ahead
                    n[a] += 1
            for c in box:
                used[c] = 1
            if min(n) >= min_extent:
                boxes.append((tuple(n), box))
        for c in members:
            member[c] = None
    boxes.sort(key=lambda b: min(b[1]))
    return boxes


# ------------------------------------------------------------------- check
def mallard_blocks(mallard, nodes, cells):
    """Mallard's count of blocks and their share of the cells, from the cell order of its check."""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        write_gmsh(work / "mesh.msh", nodes, cells, boundary_zones(nodes, cells))
        (work / "input.toml").write_text(f"""[run]
n_steps = 1
cfl = 0.5
[mesh]
type = "file"
filename = "mesh.msh"
[[periodic]]
zones = ["left", "right"]
translation = [{NX * H}, 0.0]
[initialize]
type = "constant"
u = [0.0, 0.0]
p = 1.0
T = 1.0
[[boundaries]]
name = "wall"
type = "symmetry"
[numerics.face_reconstruction]
type = "MUSCL"
[physics]
type = "euler"
gamma = 1.4
p_ref = 1.0
T_ref = 1.0
rho_ref = 1.0
""")
        log = subprocess.run([str(mallard), "check", "input.toml", "--kokkos-num-threads=2"], cwd=work,
                             capture_output=True, text=True, check=True).stdout
    m = re.search(r"structured blocks first \((\d+)\s+blocks?,\s+([\d.]+)%\s+of\s+the\s+cells\)", log)
    if not m:
        raise RuntimeError("no structured blocks in Mallard's cell order:\n" + log)
    return int(m.group(1)), float(m.group(2))


# ------------------------------------------------------------------ figure
def numbering(P):
    nodes, cells = mesh()
    blocks = structured_blocks(cells, node_keys(nodes), MIN_EXTENT)
    in_block = {c for _, box in blocks for c in box}
    width = NX * H
    fig, ax = plt.subplots(figsize=(10.4, 4.4))
    colors = [P["teal"], P["flame"]]
    face = [P["faint"]] * len(cells)
    for (_, box), color in zip(blocks, colors):
        for c in box:
            face[c] = color
    ax.add_collection(PolyCollection([nodes[c] for c in cells], facecolors=face, edgecolors=P["bg"],
                                     linewidths=1.2))
    centers = np.array([nodes[c].mean(axis=0) for c in cells])
    arrow = dict(arrowstyle="-|>", color=P["bg"], lw=1.0, mutation_scale=8, shrinkA=0, shrinkB=0)
    for _, box in blocks:
        for a, b in zip(box, box[1:]):
            pa, pb = centers[a], centers[b]
            if abs(pb[1] - pa[1]) > 1.5 * H or (abs(pb[0] - pa[0]) > 1.5 * H and abs(pb[0] - pa[0]) < width / 2):
                continue  # the next row or column starts
            if abs(pb[0] - pa[0]) > width / 2:  # across the periodic boundary: out at x = 6, in at x = 0
                ax.add_patch(FancyArrowPatch(pa, (width + 0.06, pa[1]), **arrow, clip_on=False))
                ax.add_patch(FancyArrowPatch((-0.06, pb[1]), pb, **arrow, clip_on=False))
            else:
                ax.add_patch(FancyArrowPatch(pa, pb, **arrow))
    first = 0
    for (n, box), color in zip(blocks, colors):
        ax.plot(*centers[box[0]], "o", color=P["bg"], ms=6, zorder=5)
        ax.annotate(f"cell {first}", centers[box[0]], xytext=(0, -16), textcoords="offset points",
                    ha="center", va="top", fontsize=10, color=P["fg"],
                    bbox=dict(boxstyle="round,pad=0.15", facecolor=P["bg"], edgecolor="none"))
        first += len(box)
    for x in (0.0, width):
        ax.plot([x, x], [0, NY * H], color=P["plum"], lw=2.2, ls=(0, (5, 3)))
        ax.text(x, NY * H + 0.1, "periodic", ha="center", va="bottom", color=P["plum"], fontsize=11)
    (n0, b0), (n1, b1) = blocks
    ax.text(width / 2, -0.42, f"block 1: {n0[0]} x {n0[1]} cells across the periodic boundary, "
            f"i along x: cells 0-{len(b0) - 1}", ha="center", va="top", fontsize=11, color=P["teal"])
    ax.text(width / 2, NY * H + 0.1, f"block 2: {n1[0]} x {n1[1]} cells, i along y: cells "
            f"{len(b0)}-{len(b0) + len(b1) - 1}", ha="center", va="bottom", fontsize=11, color=P["flame"])
    ax.text(width / 2, 0.52, f"cells {len(in_block)}-{len(cells) - 1}: the rest\n(triangles, and quadrilaterals\n"
            f"fewer than {MIN_EXTENT} cells long),\nin the curve's or the input order",
            ha="center", va="center", fontsize=10, color=P["fg"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor=P["bg"], edgecolor="none", alpha=0.9))
    ax.set_xlim(-0.2, width + 0.2)
    ax.set_ylim(-0.75, NY * H + 0.45)
    ax.set_aspect("equal")
    bare(ax)
    return fig


def figures(mallard=None):
    if mallard:
        nodes, cells = mesh()
        blocks = structured_blocks(cells, node_keys(nodes), MIN_EXTENT)
        share = 100.0 * sum(len(b) for _, b in blocks) / len(cells)
        count, mallard_share = mallard_blocks(mallard, nodes, cells)
        if count != len(blocks) or abs(mallard_share - share) > 0.05:
            raise RuntimeError(f"the port finds {len(blocks)} blocks ({share:.1f}% of the cells), "
                               f"Mallard {count} ({mallard_share:.1f}%)")
    figstyle.save("structured_blocks", numbering)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mallard", help="2D Mallard executable, to check the port's blocks against")
    figures(ap.parse_args().mallard)


if __name__ == "__main__":
    main()
