"""Domain diagrams of example cases: the mesh as the input builds it, with
each boundary zone colored by the kind of its condition and labeled with
its zone name and [[boundaries]] type."""
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
import matplotlib.pyplot as plt

import figstyle
import meshes

ROOT = Path(__file__).resolve().parents[2]

KIND = {
    "wall_adiabatic": "wall", "wall_isothermal": "wall", "wall_heat_flux": "wall",
    "symmetry": "symmetry",
    "upt": "inflow", "dirichlet": "inflow", "nscbc_inlet": "inflow",
    "p_out": "outflow", "p_out_average": "outflow", "nscbc_outlet": "outflow", "extrapolation": "outflow",
    "farfield": "farfield",
    "periodic": "periodic",
}


def kind_style(kind, P):
    return {
        "wall": dict(color=P["fg"], lw=3.2, ls="-"),
        "symmetry": dict(color=P["muted"], lw=2.6, ls=(0, (4, 2.5))),
        "inflow": dict(color=P["blue"], lw=3.2, ls="-"),
        "outflow": dict(color=P["flame"], lw=3.2, ls="-"),
        "farfield": dict(color=P["teal"], lw=3.2, ls="-"),
        "periodic": dict(color=P["plum"], lw=2.6, ls=(0, (1.2, 1.8))),
    }[kind]


def legend(ax, kinds, P, **kw):
    handles = [Line2D([], [], **kind_style(k, P), label=k) for k in kinds]
    kw.setdefault("loc", "upper center")
    kw.setdefault("bbox_to_anchor", (0.5, -0.02))
    kw.setdefault("ncol", len(kinds))
    ax.legend(handles=handles, handlelength=2.2, columnspacing=1.2, **kw)


def entries(case):
    """(zone, type, where) of each [[boundaries]] entry, plus periodic seams."""
    out = [(b["name"], b["type"], b.get("where")) for b in case.get("boundaries", [])]
    return out


def segments_for(mesh, zone, where):
    seg = mesh.zone_segments(zone)
    if where is None or len(seg) == 0:
        return seg
    mid = seg.mean(axis=1)
    keep = np.array([bool(meshes.evaluate(where, x, y)) for x, y in mid])
    return seg[keep]


def outward(mesh, seg):
    """Unit outward normal of a set of boundary segments (relative to the mesh center)."""
    d = seg[:, 1] - seg[:, 0]
    n = np.stack([d[:, 1], -d[:, 0]], axis=1)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    c = mesh.nodes.mean(axis=0)
    mid = seg.mean(axis=1)
    sign = np.sign(np.einsum("ij,ij->i", n, mid - c))
    sign[sign == 0] = 1
    return n * sign[:, None]


def mesh_patch(segments, P, lw, view=None):
    """All mesh edges as one SVG path (one element instead of one per edge)."""
    if view is not None:
        (x0, x1), (y0, y1) = view
        mid = segments.mean(axis=1)
        segments = segments[(mid[:, 0] > x0) & (mid[:, 0] < x1) & (mid[:, 1] > y0) & (mid[:, 1] < y1)]
    verts = np.round(segments.reshape(-1, 2), 6)
    codes = np.tile([MplPath.MOVETO, MplPath.LINETO], len(segments))
    return PathPatch(MplPath(verts, codes), facecolor="none", edgecolor=P["mesh"], lw=lw, zorder=1)


def draw_2d(ax, mesh, case, P, labels=None, mesh_lw=0.35, label_size=12, skip_labels=(), view=None):
    """Mesh plus colored zones; labels maps (zone, where) -> (x, y, ha, va) or None to place automatically."""
    ax.add_patch(mesh_patch(mesh.edges(), P, mesh_lw, view))
    kinds = []
    lo, hi = mesh.nodes.min(axis=0), mesh.nodes.max(axis=0)
    size = (hi - lo).max()
    for zone in [z for z in mesh.faces if z.endswith("@periodic")]:
        seg = mesh.zone_segments(zone)
        ax.add_collection(LineCollection(seg, zorder=3, capstyle="butt", **_lc(kind_style("periodic", P))))
        kinds.append("periodic")
    for zone, btype, where in entries(case):
        seg = segments_for(mesh, zone, where)
        if len(seg) == 0:
            continue
        kind = KIND[btype]
        kinds.append(kind)
        ax.add_collection(LineCollection(seg, zorder=3, capstyle="butt", **_lc(kind_style(kind, P))))
        if (zone, where) in skip_labels:
            continue
        text = f"{zone}\n{btype}" if zone != btype else btype
        if labels and (zone, where) in labels:
            x, y, ha, va = labels[(zone, where)]
        else:
            mid = seg.mean(axis=1)
            k = np.argmin(np.linalg.norm(mid - mid.mean(axis=0), axis=1))
            n = outward(mesh, seg)[k]
            x, y = mid[k] + n * 0.045 * size
            ha = "left" if n[0] > 0.5 else "right" if n[0] < -0.5 else "center"
            va = "bottom" if n[1] > 0.5 else "top" if n[1] < -0.5 else "center"
        figstyle.label(ax, x, y, text, P, size=label_size, ha=ha, va=va, linespacing=1.15)
    figstyle.bare(ax)
    pad = 0.02 * size
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    return list(dict.fromkeys(kinds))


def _lc(style):
    return dict(colors=style["color"], linewidths=style["lw"], linestyles=style["ls"])


def load(example, file="input.toml"):
    return tomllib.load(open(ROOT / "examples" / example / file, "rb"))


def save_generated(name, example, coarsen, title, file="input.toml", figsize=(6.4, 3.4), labels=None,
                   note=None, extra=None, legend_kw=None):
    case = load(example, file)

    def draw(P):
        mesh = meshes.generated(case["mesh"], coarsen, case.get("periodic", []))
        fig, ax = plt.subplots(figsize=figsize)
        kinds = draw_2d(ax, mesh, case, P, labels=labels)
        if extra:
            extra(ax, P, case)
        legend(ax, kinds, P, **(legend_kw or {}))
        if title:
            ax.set_title(title, color=P["fg"], pad=26)
        if note:
            ax.text(0.5, -0.2, note, transform=ax.transAxes, ha="center", va="top", color=P["muted"], fontsize=11)
        return fig

    figstyle.save(name, draw)


def make_cylinder_mesh(args):
    tmp = Path(tempfile.mkdtemp()) / "mesh.msh"
    subprocess.run([sys.executable, str(ROOT / "tools" / "mesh" / "make_cylinder_mesh.py"), str(tmp), *args], check=True,
                   stdout=subprocess.DEVNULL)
    return meshes.read_gmsh22(tmp)


AXIS_OF_SIDE = {"left": "x", "right": "x", "bottom": "y", "top": "y", "back": "z", "front": "z"}


def box_3d(name, example, file="input.toml", grid=(16, 12), sponge_from=None, notes=()):
    """A generated 3D box seen from the inlet side: faces filled by the kind of their condition."""
    case = load(example, file)
    m = case["mesh"]
    L = np.array([m.get("Lx", 1.0), m.get("Ly", 1.0), m.get("Lz", 1.0)], float)
    s = 3.2 / L.max()
    L = L * s
    a, b = -0.55, 0.38

    def proj(x, y, z):
        return np.array([x + a * z, y + b * z])

    types = {e[0]: e[1] for e in entries(case)}
    turbulent = {b["name"] for b in case.get("boundaries", []) if "turbulence" in b}
    periodic_dirs = []
    for pair in case.get("periodic", []):
        for z in pair["zones"]:
            types[z] = "periodic"
        periodic_dirs.append(AXIS_OF_SIDE[pair["zones"][0]])
    X, Y, Z = L
    faces = {  # zone: corners, visible
        "back": ([(0, 0, 0), (X, 0, 0), (X, Y, 0), (0, Y, 0)], True),
        "left": ([(0, 0, 0), (0, Y, 0), (0, Y, Z), (0, 0, Z)], True),
        "top": ([(0, Y, 0), (X, Y, 0), (X, Y, Z), (0, Y, Z)], True),
        "right": ([(X, 0, 0), (X, Y, 0), (X, Y, Z), (X, 0, Z)], False),
        "bottom": ([(0, 0, 0), (X, 0, 0), (X, 0, Z), (0, 0, Z)], False),
        "front": ([(0, 0, Z), (X, 0, Z), (X, Y, Z), (0, Y, Z)], False),
    }
    beta = list(m.get("stretching", [0, 0, 0]))

    def draw(P):
        fig, ax = plt.subplots(figsize=(7.6, 3.4))
        kinds = []
        for zone, (corners, visible) in faces.items():
            kind = KIND.get(types.get(zone, ""), None)
            if kind is None or kind == "periodic":
                continue
            kinds.append(kind)
            st = kind_style(kind, P)
            q = np.array([proj(*c) for c in corners])
            ax.add_patch(plt.Polygon(q, closed=True, facecolor=st["color"], alpha=0.28, edgecolor=st["color"],
                                     lw=2.4, zorder=3))
        hidden = (0.0, 0.0, Z)
        for p0, p1 in _box_edges(X, Y, Z):
            back = hidden in (p0, p1)
            ax.plot(*np.array([proj(*p0), proj(*p1)]).T, color=P["line"], lw=1.0 if back else 1.6,
                    ls=":" if back else "-", zorder=2)
        if periodic_dirs:
            kinds.append("periodic")
            figstyle.label(ax, *proj(0.5 * X, Y, 0.5 * Z), "periodic in " + " and ".join(periodic_dirs), P,
                           color=P["plum"], size=12, zorder=5)
        xs = meshes.stretched(grid[0], X, beta[0] if beta else 0)
        ys = meshes.stretched(grid[1], Y, beta[1] if len(beta) > 1 else 0)
        for x in xs[1:-1]:
            ax.plot(*np.array([proj(x, 0, 0), proj(x, Y, 0)]).T, color=P["mesh"], lw=0.6, zorder=2)
        for y in ys[1:-1]:
            ax.plot(*np.array([proj(0, y, 0), proj(X, y, 0)]).T, color=P["mesh"], lw=0.6, zorder=2)
        if sponge_from is not None:
            x0 = sponge_from * s
            q = np.array([proj(x0, 0, 0), proj(X, 0, 0), proj(X, Y, 0), proj(x0, Y, 0)])
            ax.add_patch(plt.Polygon(q, closed=True, facecolor=P["plum"], alpha=0.22, edgecolor=P["plum"],
                                     lw=1, ls="--", zorder=2))
            figstyle.label(ax, *proj(0.5 * (x0 + X), 0.5 * Y, 0), "sponge", P, color=P["plum"], size=11, box=True,
                           zorder=4)
        for zone, (corners, visible) in faces.items():
            if zone not in types or types[zone] == "periodic":
                continue
            c = np.mean([proj(*p) for p in corners], axis=0)
            out = {"back": (0, -0.55), "left": (-0.5, 0), "top": (0, 0.45), "right": (0.75, 0),
                   "bottom": (0, -0.3), "front": (0.3, 0.6)}[zone]
            if zone == "back":
                c = proj(0.5 * X, 0, 0)
            if zone == "bottom":
                c = proj(0.62 * X, 0, 0.5 * Z)
            if zone == "front":
                c = proj(0.75 * X, Y, Z)
            ha = "right" if out[0] < 0 else "left" if out[0] > 0 else "center"
            text = f"{zone}\n{types[zone]}" + ("\n+ turbulence" if zone in turbulent else "")
            figstyle.label(ax, c[0] + out[0], c[1] + out[1], text, P, size=11, ha=ha, linespacing=1.15)
        for t in notes:
            figstyle.label(ax, *t[0], t[1], P, size=11, color=P["muted"])
        figstyle.bare(ax)
        ax.autoscale_view()
        lo = np.min([proj(*c) for f in faces.values() for c in f[0]], axis=0)
        hi = np.max([proj(*c) for f in faces.values() for c in f[0]], axis=0)
        ax.set_xlim(lo[0] - 1.4, hi[0] + 1.5)
        ax.set_ylim(lo[1] - 0.45, hi[1] + 0.2)
        legend(ax, list(dict.fromkeys(kinds)), P, bbox_to_anchor=(0.5, 0.0))
        return fig

    figstyle.save(name, draw)


def _box_edges(X, Y, Z):
    v = [(x, y, z) for x in (0.0, X) for y in (0.0, Y) for z in (0.0, Z)]
    return [(p, q) for i, p in enumerate(v) for q in v[i + 1:] if sum(a != b for a, b in zip(p, q)) == 1]


def figures():
    box_3d("domain_turbulent_inflow_duct", "turbulent_inflow_duct", grid=(24, 8))

    # Double Mach reflection: one zone split between two conditions by `where`
    def dmr_extra(ax, P, case):
        ax.annotate("x = 1/6", xy=(1 / 6, 0), xytext=(0.75, 0.2), color=P["fg"], fontsize=12, va="center",
                    arrowprops=dict(arrowstyle="-", color=P["muted"], lw=1))
        x = np.array([1 / 6, 1 / 6 + 1 / np.sqrt(3)])
        ax.plot(x, np.sqrt(3) * (x - 1 / 6), color=P["red"], lw=1.8, zorder=4)
        figstyle.label(ax, 0.62, 0.55, "shock\n(initial)", P, color=P["red"], size=11, ha="left")

    save_generated(
        "domain_double_mach", "double_mach", 40, None, figsize=(7.2, 2.9),
        labels={
            ("bottom", "x < 1/6"): (0.05, -0.07, "center", "top"),
            ("bottom", "x >= 1/6"): (2.2, -0.07, "center", "top"),
            ("top", None): (2.0, 1.07, "center", "bottom"),
            ("left", None): (-0.07, 0.5, "right", "center"),
            ("right", None): (4.07, 0.5, "left", "center"),
        },
        extra=dmr_extra, legend_kw=dict(bbox_to_anchor=(0.5, -0.2)))

    save_generated(
        "domain_wedge", "wedge", 8, None, figsize=(5.6, 4.2),
        labels={("left", None): (-0.06, 0.75, "right", "center"), ("right", None): (2.06, 0.85, "left", "center"),
                ("top", None): (1.0, 1.56, "center", "bottom"), ("bottom", None): (1.2, 0.0, "center", "top")},
        legend_kw=dict(bbox_to_anchor=(0.5, -0.08)))

    save_generated(
        "domain_viscous_shock_tube", "viscous_shock_tube", 40, None, figsize=(5.6, 3.4),
        legend_kw=dict(bbox_to_anchor=(0.5, -0.12)))

    def flame2d_extra(ax, P, case):
        figstyle.label(ax, 0.0061, 0.0039, "periodic in y", P, color=P["plum"], size=12)

    save_generated(
        "domain_flame_2d", "flame_2d", 12, None, figsize=(5.6, 4.0),
        extra=flame2d_extra, legend_kw=dict(bbox_to_anchor=(0.5, -0.1)))

    # Cylinder: O-grid of the example (generated 4x coarser for the drawing)
    cyl = load("cylinder")

    def draw_cyl(P):
        mesh = make_cylinder_mesh(["--n-theta", "64", "--n-r", "24", "--r-far", "25", "--dr0", "0.04"])
        fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.0), gridspec_kw=dict(width_ratios=[1, 1], wspace=0.08))
        kinds = draw_2d(axes[0], mesh, cyl, P, mesh_lw=0.3,
                        labels={("farfield", None): (0, -27, "center", "top")}, skip_labels={("cylinder", None)})
        draw_2d(axes[1], mesh, cyl, P, mesh_lw=0.45, skip_labels={("farfield", None), ("cylinder", None)},
                view=((-1.7, 1.7), (-1.7, 1.7)))
        axes[1].set_xlim(-1.6, 1.6)
        axes[1].set_ylim(-1.6, 1.6)
        figstyle.label(axes[1], 0, -0.62, "cylinder\nwall_adiabatic", P, size=11, va="top", box=True)
        for ax, t in zip(axes, ("radius 25 diameters", "near the cylinder (diameter 1)")):
            ax.set_title(t, fontsize=12, color=P["muted"])
        legend(axes[0], kinds, P, bbox_to_anchor=(1.04, -0.14))
        return fig

    figstyle.save("domain_cylinder", draw_cyl)

    sphere = load("sphere_axisymmetric")

    def draw_sphere(P):
        mesh = make_cylinder_mesh(["--half", "--n-theta", "40", "--n-r", "24", "--r-far", "25", "--dr0", "0.04"])
        fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), gridspec_kw=dict(wspace=0.08))
        kinds = draw_2d(axes[0], mesh, sphere, P, mesh_lw=0.3,
                        labels={("farfield", None): (0, 26.5, "center", "bottom"),
                                ("axis", None): (-14, -1.5, "center", "top")},
                        skip_labels={("cylinder", None)})
        draw_2d(axes[1], mesh, sphere, P, mesh_lw=0.45, skip_labels={("farfield", None)},
                labels={("cylinder", None): (0, 1.15, "center", "bottom"), ("axis", None): (-1.15, -0.08, "center", "top")},
                view=((-1.6, 1.6), (-0.1, 1.6)))
        axes[1].set_xlim(-1.5, 1.5)
        axes[1].set_ylim(-0.25, 1.4)
        legend(axes[0], kinds, P, bbox_to_anchor=(1.04, -0.3))
        return fig

    figstyle.save("domain_sphere_axisymmetric", draw_sphere)


if __name__ == "__main__":
    figures()
