"""Figure of region snapshots of lower dimension from the polynomials ([[write_data]]
region, high_order): a cut line and a wall of a TENO order 5 run on a coarse
mesh, written with one value per element (high_order = false) and as Lagrange
curves (the default), against the initial field.

    python tools/docs/surface_high_order.py --mallard build/src/mallard
"""
import argparse
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import figstyle

N = 6
ORDER = 5
POINT = np.array([0.5, 0.3])
NORMAL = np.array([0.3, 1.0])
CENTER = (0.45, 0.0)
WIDTH = 0.08
# Small enough that no cell is troubled (TENO_SIGMA < troubled_threshold) on this mesh: troubled cells take the
# linear least-squares values, and their elements would be straight
AMPLITUDE = 0.1


def exact(x, y):
    return 1.0 + AMPLITUDE * np.exp(-((x - CENTER[0]) ** 2 + (y - CENTER[1]) ** 2) / WIDTH)


def run(mallard, work):
    rho = f"1.0 + {AMPLITUDE} * exp(-((x - {CENTER[0]})^2 + (y - {CENTER[1]})^2) / {WIDTH})"
    sides = "".join(f'[[boundaries]]\nname = "{b}"\ntype = "outlet_extrapolation"\n' for b in ("left", "right", "top"))
    writers = "".join(f'[[write_data]]\nprefix = "out/{name}"\nformat = "vtu"\ninterval = 1\n{keys}'
                      'regions = ["cut", "bottom"]\nfields = ["RHO"]\n'
                      for name, keys in (("lagrange", ""), ("element", "high_order = false\n")))
    (work / "input.toml").write_text(
        f'[run]\nn_steps = 1\ncfl = 0.2\n[mesh]\ntype = "cartesian"\nNx = {N}\nNy = {N}\nLx = 1.0\nLy = 1.0\n'
        f'[initialize]\ntype = "analytical"\nrho = "{rho}"\nu = ["0.0", "0.0"]\np = "1.0"\n{sides}'
        '[[boundaries]]\nname = "bottom"\ntype = "wall_adiabatic"\n'
        '[physics]\ntype = "euler"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n'
        f'[numerics.face_reconstruction]\ntype = "TENO"\norder = {ORDER}\n'
        f'[regions]\ncut = {{type = "cut_plane", point = [{POINT[0]}, {POINT[1]}], normal = [{NORMAL[0]}, {NORMAL[1]}]}}\n'
        f'{writers}[output]\nrun_log = ""\n')
    subprocess.run([mallard, "-i", "input.toml", "--kokkos-num-threads=2"], cwd=work, check=True,
                   stdout=subprocess.DEVNULL)
    out = work / "out"
    return {region: (read_vtk(out / f"element_{region}_000000.vtp"), read_vtk(out / f"lagrange_{region}_000000.vtu"))
            for region in ("cut", "bottom")}


def read_vtk(path):
    """The appended arrays of a raw little-endian VTK XML file, by name."""
    raw = path.read_bytes()
    head = raw[:raw.index(b"<AppendedData")].decode()
    base = raw.index(b"_", raw.index(b"<AppendedData")) + 1
    arrays = {}
    for tag in head.split("<DataArray")[1:]:
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', tag.split(">")[0]))
        if "offset" not in attrs:
            continue
        at = base + int(attrs["offset"])
        size = int(np.frombuffer(raw[at:at + 8], "<u8")[0])
        dtype = {"Float64": "<f8", "Int64": "<i8", "UInt8": "u1"}[attrs["type"]]
        arrays[attrs.get("Name", "Points")] = np.frombuffer(raw[at + 8:at + 8 + size], dtype)
    arrays["Points"] = arrays["Points"].reshape(-1, 3)[:, :2]
    return arrays


def line(region):
    """Origin and unit direction of the region's line, the parameter of the plots."""
    if region == "bottom":
        return np.zeros(2), np.array([1.0, 0.0])
    t = np.array([NORMAL[1], -NORMAL[0]]) / np.hypot(*NORMAL)
    return POINT - POINT @ t * t, t


def curves(view, origin, t, n_fine=40):
    """Each Lagrange curve (VTK node order: ends, then the interior nodes) as its polynomial on a fine lattice."""
    s = (view["Points"] - origin) @ t
    start, out = 0, []
    for end in view["offsets"]:
        sn, fn = s[view["connectivity"][start:end]], view["RHO"][view["connectivity"][start:end]]
        coef = np.polyfit(sn - sn[0], fn, len(sn) - 1)
        fine = np.linspace(sn[0], sn[1], n_fine)
        out.append((fine, np.polyval(coef, fine - sn[0]), sn, fn))
        start = end
    return out


def steps(elements, lagrange, origin, t):
    """One value per element, drawn flat across the element: cut segments carry cell data; a 2D zone's polyline
    holds its face-centroid values at its points, each drawn across the face (a Lagrange curve's ends) about it."""
    if "connectivity" in elements and len(elements["offsets"]) > 1:
        s = (elements["Points"] - origin) @ t
        ends = s[elements["connectivity"]].reshape(-1, 2)
        return [(e, v) for e, v in zip(ends, elements["RHO"])]
    centroids = (elements["Points"] - origin) @ t
    faces = [np.sort(c[2][:2]) for c in curves(lagrange, origin, t)]
    return [(next(f for f in faces if f[0] <= c <= f[1]), v) for c, v in zip(centroids, elements["RHO"])]


def draw(P, data):
    fig = plt.figure(figsize=(11.6, 3.9))
    grid = fig.add_gridspec(1, 3, width_ratios=[0.8, 1.15, 1.15], wspace=0.3)
    ax = fig.add_subplot(grid[0])
    figstyle.bare(ax)
    xs = np.linspace(0, 1, 200)
    X, Y = np.meshgrid(xs, xs)
    ax.contourf(X, Y, exact(X, Y), levels=14, cmap=P["cmap"]).set_rasterized(True)
    for k in range(N + 1):
        ax.plot([k / N, k / N], [0, 1], color=P["bg"], lw=0.7)
        ax.plot([0, 1], [k / N, k / N], color=P["bg"], lw=0.7)
    origin, t = line("cut")
    ends = [c[2][:2] for c in curves(data["cut"][1], origin, t)]
    a, b = min(e.min() for e in ends), max(e.max() for e in ends)
    ax.plot(*np.array([origin + a * t, origin + b * t]).T, color=P["flame"], lw=2.4)
    ax.plot([0, 1], [0, 0], color=P["plum"], lw=4, solid_capstyle="butt")
    figstyle.label(ax, *(origin + (a + 0.8 * (b - a)) * t + [0, 0.08]), "cut", P, color=P["flame"], box=True)
    figstyle.label(ax, 0.5, 0.07, "wall", P, color=P["plum"], box=True)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.04, 1.02)
    ax.set_aspect("equal")
    ax.set_title(f"RHO, TENO order {ORDER}, {N} x {N} cells", fontsize=12)

    for k, (region, title) in enumerate((("cut", "Along the cut"), ("bottom", "Along the wall"))):
        ax = fig.add_subplot(grid[k + 1])
        elements, lagrange = data[region]
        origin, t = line(region)
        cs = curves(lagrange, origin, t)
        s = np.linspace(min(c[0].min() for c in cs), max(c[0].max() for c in cs), 400)
        x = origin + np.outer(s, t)
        ax.plot(s, exact(x[:, 0], x[:, 1]), color=P["muted"], lw=1.2, ls=(0, (4, 3)), label="initial field")
        for j, (e, v) in enumerate(steps(elements, lagrange, origin, t)):
            ax.plot(e, [v, v], color=P["flame"], lw=2.2, solid_capstyle="butt",
                    label="one value per element" if j == 0 else None)
        for j, (fine, f, sn, fn) in enumerate(cs):
            ax.plot(fine, f, color=P["teal"], lw=1.8, label=f"Lagrange curves, degree {ORDER - 1}" if j == 0 else None)
            ax.plot(sn, fn, "o", ms=3, color=P["teal"])
        for e in {round(v, 12) for c in cs for v in c[2][:2]}:
            ax.axvline(e, color=P["faint"], lw=0.8, zorder=0)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("x" if region == "bottom" else "distance along the cut")
        if k == 0:
            ax.set_ylabel("RHO")
            handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=11, loc="lower center", ncol=3, bbox_to_anchor=(0.6, -0.1))
    return fig


def figures(mallard):
    with tempfile.TemporaryDirectory() as tmp:
        data = run(str(Path(mallard).resolve()), Path(tmp))
    figstyle.save("surface-high-order", lambda P: draw(P, data))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mallard", required=True, help="2D Mallard executable")
    figures(ap.parse_args().mallard)


if __name__ == "__main__":
    main()
