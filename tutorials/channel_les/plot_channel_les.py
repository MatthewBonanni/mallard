#!/usr/bin/env python3
"""Figures of the channel LES tutorial.

    python plot_channel_les.py mesh channel.msh [--out DIR]
    python plot_channel_les.py results RUN_DIR [--t-start 100] [--retau 180] [--out DIR]

mesh: the box with its boundary zones and the cells of a cross-section
(channel_geometry, channel_mesh).
results: RUN_DIR holds a run of input.toml. Averages the last statistics
snapshot (stats/channel.pvd) over x and z, folds the two halves, takes the
wall shear from the viscous x-forces on both walls averaged from --t-start
on, and compares with the DNS of Moser, Kim & Mansour (1999), downloaded on
first use: prints Re_tau, Cf and the profile peaks and draws
channel_history (Re_tau from the wall shear, and the shares of the kinetic
energy dissipation), channel_field (streamwise velocity on a plane near the
wall and across the channel) and channel_profiles (mean velocity and
Reynolds stresses in wall units).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
import matplotlib.pyplot as plt  # noqa: E402
import snapshots  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from plane_average import plane_average, read_cells  # noqa: E402
from plot_channel import MKM_RE_TAU, mkm, read_forces, read_profile  # noqa: E402
from tutfig import label_box, render  # noqa: E402

MU = 3.5714285714285714e-4
LX, LZ, H = 2 * np.pi, np.pi, 1.0


def mesh(args):
    from gmsh_mesh import read_msh
    xyz, cells, zones = read_msh(args.path)
    side = [xyz[f] for f in zones["left"]]  # the x = 0 face: y-z cells

    def geometry(t):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(projection="3d")
        s = t["series"]
        colors = {"bottom": s[1], "top": s[1], "left": s[0], "right": s[0], "back": s[2], "front": s[2]}
        alpha = {"bottom": 0.9, "top": 0.15, "left": 0.35, "right": 0.15, "back": 0.35, "front": 0.15}
        for name in ("bottom", "back", "left", "top", "right", "front"):
            quads = [xyz[f][:, [0, 2, 1]] for f in zones[name]]
            ax.add_collection3d(Poly3DCollection(quads, facecolor=colors[name], edgecolor=t["mesh"],
                                                 linewidth=0.05, alpha=alpha[name], rasterized=True))
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=s[1], label='"bottom" (y = 0), "top" (y = 2h): walls'),
                           Patch(color=s[0], label='"left", "right": periodic in x'),
                           Patch(color=s[2], label='"back", "front": periodic in z')],
                  loc="upper left", bbox_to_anchor=(0.0, 1.02), fontsize=11)
        ax.quiver(1.0, LZ / 2, 1.0, 1.5, 0, 0, color=t["ink"], arrow_length_ratio=0.25)
        ax.text(1.8, LZ / 2, 1.2, "U_b", color=t["ink"])
        ax.set_xlim(0, LX)
        ax.set_ylim(0, LZ)
        ax.set_zlim(0, 2)
        ax.set_box_aspect((LX, LZ, 2))
        ax.set_xlabel("x (streamwise)")
        ax.set_ylabel("z (spanwise)")
        ax.set_zlabel("y")
        ax.view_init(elev=18, azim=-62)
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.set_pane_color((0, 0, 0, 0))
        return fig

    def cross(t):
        fig, ax = plt.subplots(figsize=(9, 4.6))
        ax.add_collection(PolyCollection([q[:, [2, 1]] for q in side], facecolors="none", edgecolors=t["mesh"],
                                         linewidths=0.4, rasterized=True))
        ax.plot([0, LZ], [0, 0], color=t["series"][1], lw=3, label='"bottom", "top": walls')
        ax.plot([0, LZ], [2, 2], color=t["series"][1], lw=3)
        y = np.unique(np.round(xyz[:, 1], 12))
        ax.set_xlim(-0.05, LZ + 0.05)
        ax.set_ylim(-0.08, 2.08)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xlabel("z")
        ax.set_ylabel("y")
        n_y = len(y) - 1
        ax.set_title(f"Cross-section x = 0: {n_y} cells across, first {y[1]:.4f} h from the wall", fontsize=12)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
        return fig

    render(geometry, args.out, "channel_geometry")
    render(cross, args.out, "channel_mesh")


def results(args):
    run = args.path
    ref = mkm(os.path.expanduser("~/.cache/mallard/mkm"), args.retau)
    stats = snapshots.last(os.path.join(run, "stats", "channel.pvd"))
    centers, volumes, fields = read_cells(stats)
    kept, coords, counts, columns = plane_average(centers, volumes, fields, "xz")
    profile = os.path.join(run, "profile.csv")
    names = list(columns)
    np.savetxt(profile, np.column_stack([coords[:, 0], counts] + [columns[k] for k in names]), delimiter=",",
               header=",".join(["y", "cells"] + names), comments="")
    t, F = read_forces([os.path.join(run, "forces_bottom.csv"), os.path.join(run, "forces_top.csv")])
    tau = F / (2 * LX * LZ)
    win = t >= args.t_start
    tau_w = np.trapezoid(tau[win], t[win]) / (t[win][-1] - t[win][0])
    prof = read_profile(profile, H)
    rho_w = prof["rho"][0]
    u_tau = np.sqrt(tau_w / rho_w)
    nu_w = MU / rho_w
    re_tau = u_tau * H / nu_w
    cf, cf_mkm = 2 * tau_w, 2 / ref["Ub"] ** 2
    s = {"yp": prof["y"] * H * u_tau / nu_w, "U": prof["U"] / u_tau, "u": np.sqrt(prof["uu"]) / u_tau,
         "v": np.sqrt(prof["vv"]) / u_tau, "w": np.sqrt(prof["ww"]) / u_tau, "uv": -prof["uv"] / u_tau**2}
    r = {"yp": ref["yp"], "U": ref["U"], "u": np.sqrt(ref["uu"]), "v": np.sqrt(ref["vv"]), "w": np.sqrt(ref["ww"]),
         "uv": -ref["uv"]}
    window = (t[win][-1] - args.t_start) * u_tau / H
    print(f"averaging window t = {args.t_start:g} to {t[win][-1]:g} h/U_b ({window:.1f} h/u_tau)")
    print(f"Re_tau = {re_tau:.1f} (MKM {ref['Re_tau']:.1f}, {100 * (re_tau / ref['Re_tau'] - 1):+.1f}%), "
          f"Cf = {cf:.5f} (MKM {cf_mkm:.5f}, {100 * (cf / cf_mkm - 1):+.1f}%)")
    for key, label in (("u", "u_rms+"), ("v", "v_rms+"), ("w", "w_rms+"), ("uv", "-u'v'+")):
        i, j = np.argmax(s[key]), np.argmax(r[key])
        print(f"peak {label} {s[key][i]:.3f} at y+ = {s['yp'][i]:.1f} (MKM {r[key][j]:.3f} at {r['yp'][j]:.1f})")
    for yp in (10, 30, 100):
        print(f"U+({yp}) = {np.interp(yp, s['yp'], s['U']):.2f} (MKM {np.interp(yp, r['yp'], r['U']):.2f})")

    integrals = np.genfromtxt(os.path.join(run, "integrals.csv"), delimiter=",", names=True)
    ti = integrals["t"]
    shares = None
    if "eps_numerical" in integrals.dtype.names:
        sel = ti >= args.t_start
        parts = np.array([-integrals["ke_rate_viscous"][sel].mean(), -integrals["ke_rate_sgs"][sel].mean(),
                          integrals["eps_numerical"][sel].mean()])
        shares = 100 * parts / parts.sum()
        print(f"dissipation: molecular {shares[0]:.1f}%, SGS {shares[1]:.1f}%, numerical {shares[2]:.1f}%")

    def history(th):
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [2.2, 1]})
        ax.plot(t, np.sqrt(tau / rho_w) * H / nu_w, color=th["series"][0], lw=0.8)
        ax.axhline(ref["Re_tau"], color=th["ink"], ls="--", lw=1.2, label=f"MKM, {ref['Re_tau']:.0f}")
        ax.axvspan(args.t_start, t[-1], color=th["faint"], zorder=-1)
        ax.text(0.5 * (args.t_start + t[-1]), 1.25 * ref["Re_tau"], "statistics", ha="center", color=th["ink2"])
        ax.set_ylim(0.6 * ref["Re_tau"], 1.35 * ref["Re_tau"])
        ax.set_xlabel("t U_b / h")
        ax.set_ylabel("Re_τ from the wall shear")
        ax.legend(loc="lower right")
        if shares is not None:
            labels = ["molecular", "SGS model", "numerical"]
            bx.barh(labels, shares, color=[th["series"][0], th["series"][1], th["series"][3]])
            for k, v in enumerate(shares):
                bx.text(v + 1, k, f"{v:.1f}%", va="center", color=th["ink"], fontsize=11)
            bx.set_xlim(0, 110)
            bx.invert_yaxis()
            bx.set_xlabel("share of the dissipation (%)")
            bx.grid(False)
        fig.tight_layout()
        return fig

    pts, conn, offs, a = snapshots.read(snapshots.last(os.path.join(run, "solut", "flow.pvd")))
    polys, c = snapshots.cells(pts, conn, offs)
    ux = a["U"][:, 0]
    y_cells = np.unique(np.round(c[:, 1], 10))
    y15 = y_cells[np.argmin(np.abs(y_cells * u_tau / nu_w - 15))]
    near = np.isclose(c[:, 1], y15, rtol=0, atol=1e-9)
    z_cells = np.unique(np.round(c[:, 2], 10))
    zmid = z_cells[len(z_cells) // 2]
    cut = np.isclose(c[:, 2], zmid, rtol=0, atol=1e-9)

    def field(th):
        fig, (ax, bx) = plt.subplots(2, 1, figsize=(10, 7.2), gridspec_kw={"height_ratios": [1, 0.7]})

        def face(sel, i, j):
            out = []
            for k in np.where(sel)[0]:
                q = polys[k]
                lo = q[:, 3 - i - j] <= np.median(q[:, 3 - i - j])  # the cell's face on one side of the plane
                f = q[lo][:, [i, j]]
                ctr = f.mean(0)
                out.append(f[np.argsort(np.arctan2(f[:, 1] - ctr[1], f[:, 0] - ctr[0]))])
            return out
        pa = PolyCollection(face(near, 0, 2), array=ux[near], cmap=th["cmap"], edgecolors="none", linewidths=0,
                            antialiased=False, rasterized=True)
        ax.add_collection(pa)
        ax.set_xlim(0, LX)
        ax.set_ylim(0, LZ)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_title(f"Streaks: u on the plane y⁺ ≈ {y15 * u_tau / nu_w:.0f}")
        pb = PolyCollection(face(cut, 0, 1), array=ux[cut], cmap=th["cmap"], edgecolors="none", linewidths=0,
                            antialiased=False, rasterized=True)
        bx.add_collection(pb)
        bx.set_xlim(0, LX)
        bx.set_ylim(0, 2)
        bx.set_aspect("equal")
        bx.grid(False)
        bx.set_xlabel("x")
        bx.set_ylabel("y")
        bx.set_title("u across the channel, z = πh/2")
        for p in (pa, pb):
            p.set_clim(0, 1.3)
        fig.subplots_adjust(hspace=0.4)
        cb = fig.colorbar(pb, ax=[ax, bx], shrink=0.6)
        cb.set_label("u / U_b")
        cb.outline.set_edgecolor(th["axis"])
        return fig

    def profiles(th):
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2))
        ax.semilogx(r["yp"][1:], r["U"][1:], color=th["ink"], lw=1.4, label="DNS, Moser, Kim & Mansour")
        ax.semilogx(s["yp"], s["U"], "o", ms=4, mfc="none", color=th["series"][0], label="LES, Mallard")
        yy = np.logspace(0, 2.3)
        ax.semilogx(yy[yy < 12], yy[yy < 12], ":", color=th["muted"], lw=1)
        ax.semilogx(yy[yy > 25], np.log(yy[yy > 25]) / 0.41 + 5.2, ":", color=th["muted"], lw=1)
        ax.text(2, 4.5, "U⁺ = y⁺", color=th["ink2"], fontsize=10)
        ax.text(60, 12.5, "log law", color=th["ink2"], fontsize=10)
        ax.set_xlim(0.3, 1.1 * ref["Re_tau"])
        ax.set_xlabel("y⁺")
        ax.set_ylabel("U⁺")
        ax.legend(loc="upper left", fontsize=10)
        ax.set_title("Mean velocity")
        for k, (key, label) in enumerate((("u", "u_rms"), ("v", "v_rms"), ("w", "w_rms"), ("uv", "−u'v'"))):
            col = th["series"][k]
            bx.plot(r["yp"], r[key], color=col, lw=1.4)
            bx.plot(s["yp"], s[key], "o", ms=3.5, mfc="none", color=col, label=label + "⁺")
        bx.set_xlim(0, ref["Re_tau"])
        bx.set_xlabel("y⁺")
        bx.set_title("Resolved Reynolds stresses (lines: DNS)")
        bx.legend(fontsize=10)
        fig.tight_layout()
        return fig

    render(history, args.out, "channel_history")
    render(field, args.out, "channel_field")
    render(profiles, args.out, "channel_profiles")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=["mesh", "results"])
    ap.add_argument("path")
    ap.add_argument("--t-start", type=float, default=100.0, help="start of the averaging window (h/U_b)")
    ap.add_argument("--retau", type=int, default=180, choices=sorted(MKM_RE_TAU))
    ap.add_argument("--out", help="output directory (default: docs/images/tutorials)")
    args = ap.parse_args()
    mesh(args) if args.what == "mesh" else results(args)


if __name__ == "__main__":
    main()
