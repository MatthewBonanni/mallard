"""Figures of the TENO-E implementation reference (docs/numerics/teno_e.md),
drawn into docs/images/theory/: the Lebesgue-bound growth of a central
stencil on a mixed mesh, and the stencil selection at a discontinuity.

Both come from a small Python model of Mallard's 2D setup and selection
(src/numerics/teno.cpp): the zero-mean monomial basis in the cell-scaled
frame, least-squares fits to exact cell averages, the Lebesgue constant at
the face Gauss points, the cone test of the sector stencils and the
smoothness indicators with the adaptive cutoff. Mirror images are left out
(the target cells are interior).
"""
import math

import numpy as np
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

import figstyle
from figstyle import bare, label

OUT = figstyle.OUT.parent / "theory"
TOL = 1e-10


# ------------------------------------------------------------------- mesh

def mixed_mesh(nx=13, L=6.0, H=4.2, rows=(0.14, 0.2, 0.28, 0.4), jitter=0.2, seed=5):
    """Stretched quadrilaterals along the bottom under a jittered triangulation."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(0.0, L, nx)
    ys = np.concatenate([[0.0], np.cumsum(rows)])
    nodes = [(x, y) for y in ys for x in xs]
    cells = []
    for j in range(len(rows)):
        for i in range(nx - 1):
            a = j * nx + i
            cells.append([a, a + 1, a + nx + 1, a + nx])
    y0 = ys[-1]
    base = len(nodes) - nx
    dx = xs[1] - xs[0]
    ny = int(round((H - y0) / dx))
    tri_pts = [nodes[base + i] for i in range(nx)]
    for j in range(1, ny + 1):
        y = y0 + (H - y0) * j / ny
        shift = 0.5 * dx if j % 2 else 0.0
        row = [0.0] + [x + shift for x in xs[:-1] if 0.0 < x + shift < L] + [L]
        for x in row:
            p = np.array([x, y])
            if 0.0 < x < L and j < ny:
                p = p + jitter * dx * rng.uniform(-1, 1, 2)
            tri_pts.append(tuple(p))
    tri_pts = np.array(tri_pts)
    tri = Delaunay(tri_pts)
    index = {}
    for k, p in enumerate(tri_pts):
        if k < nx:
            index[k] = base + k
        else:
            index[k] = len(nodes)
            nodes.append(tuple(p))
    for s in tri.simplices:
        p = tri_pts[s]
        area = 0.5 * ((p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1]))
        if abs(area) < 1e-9:
            continue
        c = [index[k] for k in s]
        cells.append(c if area > 0 else c[::-1])
    return np.array(nodes), cells


def triangle_mesh(nx=11, ny=9, L=(5.0, 4.0), jitter=0.2, seed=2):
    rng = np.random.default_rng(seed)
    x, y = np.meshgrid(np.linspace(0, L[0], nx), np.linspace(0, L[1], ny))
    p = np.column_stack([x.ravel(), y.ravel()])
    inner = (p[:, 0] > 0) & (p[:, 0] < L[0]) & (p[:, 1] > 0) & (p[:, 1] < L[1])
    p[inner] += jitter * (L[0] / (nx - 1)) * rng.uniform(-1, 1, (inner.sum(), 2))
    cells = []
    for s in Delaunay(p).simplices:
        q = p[s]
        area = (q[1, 0] - q[0, 0]) * (q[2, 1] - q[0, 1]) - (q[2, 0] - q[0, 0]) * (q[1, 1] - q[0, 1])
        cells.append(list(s) if area > 0 else list(s[::-1]))
    return p, cells


# -------------------------------------------------------------- the model

def exponents(r):
    return [(d - b, b) for d in range(1, r + 1) for b in range(d + 1)]


GL = np.polynomial.legendre.leggauss(8)


def triangle_rule(a, b, c):
    """Collapsed Gauss rule on a triangle: points and weights (exact to degree 15)."""
    x, w = (GL[0] + 1) / 2, GL[1] / 2
    u, v = np.meshgrid(x, x, indexing="ij")
    wu, wv = np.meshgrid(w, w, indexing="ij")
    s, t = u * (1 - v), u * v
    J = abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
    pts = a + np.outer(s.ravel(), b - a) + np.outer(t.ravel(), c - a)
    return pts, (wu * wv * u).ravel() * J


def polygon_rule(poly):
    pts, wts = [], []
    for k in range(1, len(poly) - 1):
        p, w = triangle_rule(poly[0], poly[k], poly[k + 1])
        pts.append(p)
        wts.append(w)
    return np.vstack(pts), np.concatenate(wts)


def monomials(xi, r):
    return np.column_stack([xi[:, 0] ** a * xi[:, 1] ** b for a, b in exponents(r)])


class Mesh:
    def __init__(self, nodes, cells):
        self.nodes, self.cells = nodes, cells
        self.polys = [nodes[c] for c in cells]
        self.rules = [polygon_rule(p) for p in self.polys]
        self.area = np.array([w.sum() for _, w in self.rules])
        self.cent = np.array([(x * w[:, None]).sum(0) / w.sum() for x, w in self.rules])
        by_node = {}
        for k, c in enumerate(cells):
            for n in c:
                by_node.setdefault(n, set()).add(k)
        self.vertex_nb = [sorted(set().union(*(by_node[n] for n in c)) - {k}) for k, c in enumerate(cells)]

    def means(self, c, x0, h, r):
        x, w = self.rules[c]
        return (monomials((x - x0) / h, r) * w[:, None]).sum(0) / w.sum()

    def layers(self, i, n_min):
        seen, order, frontier = {i}, [], [i]
        while frontier and len(order) <= n_min:
            nxt = []
            for c in frontier:
                for d in self.vertex_nb[c]:
                    if d not in seen:
                        seen.add(d)
                        nxt.append(d)
            order += nxt
            frontier = nxt
        return order


def pseudo_inverse(A, max_condition=1e8):
    scale = np.linalg.norm(A, axis=0)
    if np.any(scale == 0):
        return None
    Q, R = np.linalg.qr(A / scale)
    d = np.abs(np.diag(R))
    if d.min() * max_condition < d.max():
        return None
    return np.linalg.solve(R, Q.T) / scale[:, None]


class Cell:
    """Mallard's 2D setup of one cell (TENO::compute_stencils_and_matrices)."""

    def __init__(self, mesh, i, order, bound=10.0, factor=2.0, n_sector=10, scan=False):
        self.mesh, self.i, self.r = mesh, i, order - 1
        r = self.r
        self.nk = len(exponents(r))
        self.x0, self.h = mesh.cent[i], math.sqrt(mesh.area[i])
        self.mean0 = mesh.means(i, self.x0, self.h, r)
        poly = mesh.polys[i]
        g = np.polynomial.legendre.leggauss(max(1, min(4, (order + 1) // 2)))[0]
        fq = np.array([a + (s + 1) / 2 * (b - a) for a, b in zip(poly, np.roll(poly, -1, 0)) for s in g])
        self.psi_faces = monomials((fq - self.x0) / self.h, r) - self.mean0
        ns, ns_max = math.ceil(factor * self.nk), math.ceil(3.5 * self.nk)
        cand = mesh.layers(i, ns_max)
        d2 = np.array([np.sum((mesh.cent[c] - self.x0) ** 2) for c in cand])
        o = np.argsort(d2, kind="stable")
        self.cand, self.d2 = [cand[k] for k in o], d2[o]
        self.tried = []
        self.n_used = None
        best = np.inf
        for n in range(ns, min(ns_max, len(self.cand)) + 1):
            if n < len(self.cand) and abs(self.d2[n] - self.d2[n - 1]) < TOL * self.h ** 2:
                continue
            P = self.fit(self.cand[:n], r)
            if P is None:
                continue
            lam = self.lebesgue(P)
            self.tried.append((n, lam))
            if lam < best:
                best, self.n_used, self.P = lam, n, P
            if lam <= bound and not scan:
                break
        self.n_nominal = ns
        # Sector stencils: the nearest n_sector entries in the cone of each face
        self.sectors = []
        for a, b in zip(poly, np.roll(poly, -1, 0)):
            M = np.linalg.inv(np.column_stack([a - self.x0, b - self.x0]))
            inside = [c for c in self.cand if np.all(M @ (mesh.cent[c] - self.x0) >= -TOL)]
            if len(inside) < n_sector:
                self.sectors.append(None)
                continue
            S = inside[:n_sector]
            self.sectors.append((S, self.fit(S, 2)))
        self.S = self.si_matrix()

    def fit(self, stencil, deg):
        n = len(exponents(deg))
        A = np.array([self.mesh.means(c, self.x0, self.h, deg)[:n] for c in stencil]) - self.mean0[:n]
        return pseudo_inverse(A)

    def lebesgue(self, P):
        c = self.psi_faces @ P
        return np.max(np.abs(1 - c.sum(1)) + np.abs(c).sum(1))

    def si_matrix(self):
        x, w = self.mesh.rules[self.i]
        xi = (x - self.x0) / self.h
        w = w / self.h ** 2
        ex = exponents(self.r)
        S = np.zeros((self.nk, self.nk))
        for bp in range(self.r + 1):
            for bq in range(self.r + 1 - bp):
                if bp + bq == 0:
                    continue
                D = np.column_stack([
                    math.perm(a, bp) * math.perm(b, bq) * xi[:, 0] ** max(a - bp, 0) * xi[:, 1] ** max(b - bq, 0)
                    if a >= bp and b >= bq else np.zeros(len(w)) for a, b in ex])
                S += (D * w[:, None]).T @ D
        return S

    def select(self, ubar):
        """Shares gamma_K / sum and gamma_k / sum_k of the central and sector polynomials of averages ubar."""
        u0 = ubar[self.i]
        aK = self.P @ (ubar[self.cand[:self.n_used]] - u0)
        betas = [aK @ self.S @ aK + 1e-12]
        for s in self.sectors:
            if s is None:
                continue
            a = s[1] @ (ubar[s[0]] - u0)
            betas.append(a @ self.S[:5, :5] @ a + 1e-12)
        b = np.array(betas)
        g = (b.min() / b) ** 6
        return g[0] / g.sum(), g[1:] / g[1:].sum()


def adaptive_CT(sigma, lo=1e-3, hi=1e-2):
    m = min(1.0, max(0.0, (sigma - lo) / (hi - lo)))
    g = (1 - m) ** 2 * (1 + 2 * m)
    return 10.0 ** -math.floor(10 - 4 * (1 - g))


def clipped_area(poly, n, d):
    """Area of the part of a polygon with x . n > d (Sutherland-Hodgman on one plane)."""
    out = []
    for a, b in zip(poly, np.roll(poly, -1, 0)):
        fa, fb = a @ n - d, b @ n - d
        if fa >= 0:
            out.append(a)
        if fa * fb < 0:
            out.append(a + fa / (fa - fb) * (b - a))
    if len(out) < 3:
        return 0.0
    p = np.array(out)
    return 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))


# ---------------------------------------------------------------- figures

def draw_mesh(ax, mesh, P):
    for poly in mesh.polys:
        ax.add_patch(Polygon(poly, closed=True, facecolor="none", edgecolor=P["mesh"], lw=0.7))


def lebesgue(P):
    mesh = Mesh(*mixed_mesh(rows=(0.08, 0.12, 0.18, 0.27, 0.4)))
    target = int(np.argmin(np.linalg.norm(mesh.cent - [3.75, 0.51], axis=1)))
    cell = Cell(mesh, target, order=4, scan=True)
    n, lam = np.array(cell.tried).T
    used = int(n[np.argmax(lam <= 10.0)])
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), gridspec_kw=dict(width_ratios=[1.3, 1], wspace=0.25))
    ax = axes[0]
    draw_mesh(ax, mesh, P)
    for k, c in enumerate(cell.cand[:used]):
        ax.add_patch(Polygon(mesh.polys[c], closed=True, facecolor=P["teal"] if k < cell.n_nominal else P["flame"],
                             alpha=0.45 if k < cell.n_nominal else 0.8, edgecolor="none"))
    ax.add_patch(Polygon(mesh.polys[target], closed=True, facecolor=P["fg"], edgecolor=P["fg"]))
    c0 = mesh.cent[target]
    bare(ax)
    ax.set_xlim(c0[0] - 1.9, c0[0] + 1.9)
    ax.set_ylim(-0.05, 2.35)
    ax.set_title(f"order 4: the {cell.n_nominal} nearest cells, plus {used - cell.n_nominal}", fontsize=12.5)
    ax = axes[1]
    ax.semilogy(n, lam, "o-", color=P["teal"], ms=4)
    ax.semilogy([used], [lam[n == used][0]], "o", color=P["flame"], ms=8, zorder=3)
    ax.axhline(10, color=P["flame"], lw=1.2, ls="--")
    ax.axhline(4, color=P["plum"], lw=1.2, ls=":")
    label(ax, n[-1], 10 * 1.12, "2D bound", P, color=P["flame"], size=11, ha="right", va="bottom")
    label(ax, n[-1], 4 * 1.12, "3D bound", P, color=P["plum"], size=11, ha="right", va="bottom")
    ax.set_xlabel("stencil size $n$")
    ax.set_ylabel(r"Lebesgue constant $\Lambda$")
    ax.set_yticks([2, 4, 10])
    ax.yaxis.set_major_formatter(plt.ScalarFormatter())
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    return fig


def selection(P):
    mesh = Mesh(*triangle_mesh())
    target = int(np.argmin(np.linalg.norm(mesh.cent - [2.5, 2.0], axis=1)))
    cell = Cell(mesh, target, order=5)
    c0, h = cell.x0, cell.h
    nrm = np.array([math.cos(0.35), math.sin(0.35)])
    ds = np.linspace(-6.0, 6.0, 241) * h
    shares, CTs, sigmas = [], [], []
    for d in ds:
        frac = np.array([clipped_area(p, nrm, c0 @ nrm + d) for p in mesh.polys]) / mesh.area
        rho = 1.0 + 0.5 * frac
        g = np.abs(rho[cell.cand[:cell.n_used]] - rho[target]) / rho[target]
        C_T = adaptive_CT(g.var())
        share, _ = cell.select(rho)
        sigmas.append(g.var())
        shares.append(share)
        CTs.append(C_T)
    shares, CTs, sigmas = np.array(shares), np.array(CTs), np.array(sigmas)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), gridspec_kw=dict(width_ratios=[1, 1.35], wspace=0.25))
    ax = axes[0]
    draw_mesh(ax, mesh, P)
    for c in cell.cand[:cell.n_used]:
        ax.add_patch(Polygon(mesh.polys[c], closed=True, facecolor=P["teal"], alpha=0.25, edgecolor="none"))
    for a, b in zip(mesh.polys[target], np.roll(mesh.polys[target], -1, 0)):
        for v in (a, b):
            e = c0 + (v - c0) / np.linalg.norm(v - c0) * 1.6
            ax.plot([c0[0], e[0]], [c0[1], e[1]], color=P["flame"], lw=0.9, ls="--")
    ax.add_patch(Polygon(mesh.polys[target], closed=True, facecolor=P["fg"], edgecolor=P["fg"]))
    t = np.array([-nrm[1], nrm[0]])
    for d, ls in ((-1.5 * h, ":"), (0.0, "-"), (1.5 * h, ":")):
        p = c0 + d * nrm
        ax.plot(*np.array([p - 2.2 * t, p + 2.2 * t]).T, color=P["red"], lw=1.4, ls=ls)
    bare(ax)
    ax.set_xlim(c0[0] - 1.9, c0[0] + 1.9)
    ax.set_ylim(c0[1] - 1.9, c0[1] + 1.9)
    ax.set_title("density step at distance $d$", fontsize=12.5)
    ax = axes[1]
    x = ds / h
    troubled = sigmas >= 1e-3
    central = troubled & (shares >= CTs)
    sector = troubled & ~central
    ax.fill_between(x, 1e-30, 10, where=central, color=P["teal"], alpha=0.3, step="mid", lw=0, zorder=0)
    ax.fill_between(x, 1e-30, 10, where=sector, color=P["flame"], alpha=0.14, step="mid", lw=0, zorder=0)
    ax.semilogy(x, np.where(troubled, shares, np.nan), color=P["teal"], label=r"central share $\gamma_K/(\gamma_K + \Sigma_k\gamma_k)$")
    ax.semilogy(x, np.where(troubled, CTs, np.nan), color=P["muted"], ls="--", lw=1.3, label="$C_T$")
    ax.set_ylim(1e-17, 3)
    ax.set_xlim(x[0], x[-1])
    ax.set_xlabel("$d/h$")
    for xs, text in ((-4.8, "smooth"), (4.8, "smooth")):
        label(ax, xs, 1e-8, text, P, color=P["muted"], size=11)
    label(ax, -1.6, 3e-16, "sectors", P, color=P["flame"], size=11)
    label(ax, 1.6, 3e-16, "sectors", P, color=P["flame"], size=11)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=11)
    ax.yaxis.set_major_locator(plt.LogLocator(numticks=5))
    return fig


FIGURES = {"teno_lebesgue": lebesgue, "teno_selection": selection}


def figures(names=None):
    for name, draw in FIGURES.items():
        if not names or name in names:
            figstyle.save(name, draw, OUT)


if __name__ == "__main__":
    import sys
    figures(sys.argv[1:])
