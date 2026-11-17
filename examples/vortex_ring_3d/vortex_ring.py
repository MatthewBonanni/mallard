#!/usr/bin/env python3
"""The 3D vortex ring example (examples/vortex_ring_3d): its initial table,
and the azimuthal (Widnall) instability measured against theory.

    vortex_ring.py table INPUT ring.csv
    vortex_ring.py modes INPUT RESTART... --csv modes.csv [--planes planes.npz]
    vortex_ring.py compare INPUT modes.csv [--planes planes.npz] [--window T1 T2] [--png FILE] [--figure]

`table` writes the axisymmetric ring that `[[initialize.data]]` of INPUT
reads: a Gaussian core of azimuthal vorticity
    omega = Gamma / (pi a^2) exp(-((x - x0)^2 + (r - R)^2) / a^2)
with R, a, Gamma and x0 from the constants below. The Stokes streamfunction
solves E^2 psi = -r omega, periodic in x over the box length (the ring's
coaxial images) and set at the outer radius to the field of the ring and its
images as filaments; u_x = psi_r / r, u_r = -psi_x / r. The pressure solves
the Poisson equation lap p = -rho0 (grad u : grad u^T), so the ring starts in
equilibrium. Columns x, r, ux, q = u_r / r (finite on the axis), p, on a grid
of spacing a / 16.

`modes` reads restart files of INPUT's run and samples the vorticity in a
window around the core in each of 128 meridian half-planes. The norm of each
azimuthal Fourier mode n of the three vorticity components over the window,
relative to that of the mean azimuthal vorticity, measures the instability
(columns W1..W24); the centroid of the azimuthal vorticity (above a fifth of
its peak) in each half-plane gives the core's radial and axial displacement
(A1..A24, their Fourier amplitudes over R). Also the ring radius R, the core
radius a (the second moment of the azimuthally averaged vorticity, as in
Hattori et al. 2019), the circulation and the axial position. One row per
file; --planes also keeps |omega| on the ring's plane for the figure.

`compare` fits the growth rate of each mode n (of W_n) over WINDOW, where
the leading mode grows exponentially, and compares the leading mode number
and its growth rate with the theory of the elliptic instability of a Gaussian ring
(Blanco-Rodriguez & Le Dizes 2016, corrigendum 2017):
    sigma 2 pi R^2 / Gamma = 0.5171 ln(8 / eps) - 0.9285 - 12.5 / (Re eps^2)
with eps = a / R and Re = Gamma / (2 pi nu), at k a = n eps = 2.33 (the
principal mode (-1, 1, 1); Hattori et al. 2019, figure 8). The viscous term
is read off the Re = 5000 curve of the corrigendum's figure 8 (12.5 +- 1 over
eps = 0.18-0.25). --png draws the figure into FILE, --figure into
docs/images/guide/vortex_ring_3d-{light,dark}.svg.
"""
import argparse
import csv
import os
import sys
import tomllib

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../tools"))

N_THETA = 128
K_PRINCIPAL = 2.33
WINDOW = (30.0, 70.0)            # fit window of the full-size run (time units of the input)
SNAPSHOTS = (0.0, 6.0, 8.5)      # times of the figure's top row, t Gamma / 2 pi R^2

# The ring, in units of its radius R, the ambient density rho0 and sound speed c0 (p0 = 1 / gamma)
R = 1.0
A = 0.2                # Gaussian core radius: eps = a / R = 0.2
MACH = 0.3             # peak swirl velocity of the core over c0
RE = 5000.0            # Gamma / (2 pi nu)
GAMMA_GAS = 1.4
V_FACTOR = 0.6382      # peak swirl of a Lamb-Oseen core, over Gamma / (2 pi a)


def ring():
    circulation = 2 * np.pi * A * MACH / V_FACTOR
    # Self-induced speed of a thin Gaussian ring (Saffman 1970): the input's background flow cancels it
    speed = circulation / (4 * np.pi * R) * (np.log(8 * R / A) - 0.558)
    return dict(R=R, a=A, circulation=circulation, speed=speed, nu=circulation / (2 * np.pi * RE),
                p0=1.0 / GAMMA_GAS, time=2 * np.pi * R ** 2 / circulation)


def params(path):
    """Box of INPUT, and the ring centered in it."""
    with open(path, "rb") as f:
        m = tomllib.load(f)["mesh"]
    P = dict(Nx=m["Nx"], Ny=m["Ny"], Nz=m["Nz"], Lx=m["Lx"], Ly=m["Ly"], Lz=m["Lz"], **ring())
    P.update(x0=P["Lx"] / 2, yc=P["Ly"] / 2, zc=P["Lz"] / 2)
    return P


def solve_axisymmetric(rhs, h, nr, outer, laplace_sign):
    """Solve f_xx + f_rr + laplace_sign f_r / r = rhs on x periodic (rows), r_j = j h, j = 0..nr.

    laplace_sign = -1: Stokes operator E^2 (f = 0 on the axis); +1: Laplacian (f_r = 0 on the axis).
    f(r = nr h) = outer (one value per x). FFT in x, then a tridiagonal solve in r for each wavenumber."""
    from scipy.linalg import solve_banded
    nx = rhs.shape[0]
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=h)
    # Spectrally exact in x would make the operator stiff at grid scale; use the second-order symbol.
    k2 = (2 - 2 * np.cos(kx * h)) / h ** 2
    rhs_hat = np.fft.fft(rhs, axis=0)
    out_hat = np.fft.fft(outer)
    r = np.arange(nr + 1) * h
    f_hat = np.zeros((nx, nr + 1), dtype=complex)
    f_hat[:, nr] = out_hat
    # unknowns j = j0..nr-1
    j0 = 1 if laplace_sign < 0 else 0
    js = np.arange(j0, nr)
    n = len(js)
    for i in range(nx):
        ab = np.zeros((3, n), dtype=complex)
        b = rhs_hat[i, js].astype(complex)
        rj = r[js]
        lower = np.ones(n) / h ** 2
        upper = np.ones(n) / h ** 2
        diag = -2 / h ** 2 - k2[i] * np.ones(n)
        if laplace_sign < 0:
            lower = lower + 1 / (2 * h * rj)
            upper = upper - 1 / (2 * h * rj)
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                lower = lower - np.where(rj > 0, 1 / (2 * h * rj), 0)
                upper = upper + np.where(rj > 0, 1 / (2 * h * rj), 0)
            # r = 0: f_rr + f_r / r -> 2 f_rr, with f_-1 = f_1
            diag[0] = -4 / h ** 2 - k2[i]
            upper[0] = 4 / h ** 2
        b[-1] -= upper[-1] * f_hat[i, nr]
        ab[0, 1:] = upper[:-1]
        ab[1] = diag
        ab[2, :-1] = lower[1:]
        f_hat[i, js] = solve_banded((1, 1), ab, b)
    return np.real(np.fft.ifft(f_hat, axis=0))


def filament_psi(x, r, x0, R, gamma, lx, images=8):
    """Stokes streamfunction of a ring filament (circulation gamma, radius R at x0) and its coaxial images."""
    from scipy.special import ellipe, ellipk
    psi = np.zeros_like(x, dtype=float)
    for k in range(-images, images + 1):
        dx = x - x0 - k * lx
        r1 = np.sqrt(dx ** 2 + (r - R) ** 2)
        r2 = np.sqrt(dx ** 2 + (r + R) ** 2)
        lam = (r2 - r1) / (r2 + r1)
        psi += gamma / (2 * np.pi) * (r1 + r2) * (ellipk(lam ** 2) - ellipe(lam ** 2))
    return psi


def table(args):
    P = params(args.input)
    a, G, x0, rho0, p0 = P["a"], P["circulation"], P["x0"], 1.0, P["p0"]
    lx = P["Lx"]
    h = a / 16
    nx = int(round(lx / h))
    h = lx / nx
    rmax = 1.05 * np.hypot(P["Ly"], P["Lz"]) / 2
    nr = int(np.ceil(rmax / h))
    x = np.arange(nx) * h
    r = np.arange(nr + 1) * h
    X, Rr = np.meshgrid(x, r, indexing="ij")
    dxp = (X - x0 + lx / 2) % lx - lx / 2
    omega = G / (np.pi * a ** 2) * np.exp(-(dxp ** 2 + (Rr - R) ** 2) / a ** 2)
    outer = filament_psi(x, np.full(nx, r[-1]), x0, R, G, lx)
    psi = solve_axisymmetric(-Rr * omega, h, nr, outer, -1)

    def ddx(f):
        return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2 * h)

    def ddr(f):
        g = np.gradient(f, h, axis=1, edge_order=2)
        return g

    psi_r = ddr(psi)
    ux = np.empty_like(psi)
    ux[:, 1:] = psi_r[:, 1:] / Rr[:, 1:]
    ux[:, 0] = (16 * psi[:, 1] - psi[:, 2]) / (6 * h ** 2)
    q = np.empty_like(psi)  # u_r / r = -psi_x / r^2
    psi_x = ddx(psi)
    q[:, 1:] = -psi_x[:, 1:] / Rr[:, 1:] ** 2
    q[:, 0] = -ddx(ux[:, 0]) / 2
    ur = q * Rr
    s = ddx(ux) ** 2 + ddr(ur) ** 2 + q ** 2 + 2 * ddr(ux) * ddx(ur)
    p = p0 + solve_axisymmetric(-rho0 * s, h, nr, np.zeros(nx), +1)

    # Circulation and peak swirl as checks
    circ = omega.sum() * h * h
    print(f"table: {nx + 1} x {nr + 1} nodes, h = {h:.5f} (a / {a / h:.1f}); circulation {circ:.6f} "
          f"(target {G:.6f}); u_x on the axis at the ring {ux[int(round(x0 / h)), 0]:.5f}; "
          f"p_min - p0 = {p.min() - p0:.5f}")
    with open(args.output, "w") as f:
        f.write("# Gaussian vortex ring for examples/vortex_ring_3d (vortex_ring.py table)\n")
        f.write("x,r,ux,q,p\n")
        idx = list(range(nx)) + [0]
        for ii, i in enumerate(idx):
            xi = x[i] if ii < nx else lx
            for j in range(nr + 1):
                f.write(f"{xi:.6f},{r[j]:.6f},{ux[i, j]:.10e},{q[i, j]:.10e},{p[i, j]:.10e}\n")
    print(f"wrote {args.output}")


def load_velocity(path, P):
    """Time and the velocity (3, Nz, Ny, Nx) of a restart file of INPUT's cartesian box.

    A generated box numbers its cells with z fastest, then y, then x."""
    from mallard.restart import read
    rs = read(path)
    shape = (P["Nx"], P["Ny"], P["Nz"])
    rho = rs["RHO"].reshape(shape)
    u = np.stack([np.ascontiguousarray((rs[f"RHOU_{c}"].reshape(shape) / rho).T) for c in "XYZ"])
    return rs.t, u.astype(np.float32)


def vorticity(u, h):
    """Curl of u (components x, y, z on axes 2, 1, 0) by fourth-order periodic central differences."""
    def d(f, axis):
        return (8 * (np.roll(f, -1, axis) - np.roll(f, 1, axis)) - (np.roll(f, -2, axis) - np.roll(f, 2, axis))) / (12 * h)
    ux, uy, uz = u
    return np.stack([d(uz, 1) - d(uy, 0), d(ux, 0) - d(uz, 2), d(uy, 2) - d(ux, 1)])


def sample(f, P, x, y, z):
    """f (Nz, Ny, Nx; spline_filter'ed) at points: periodic cubic spline through the cell values."""
    from scipy.ndimage import map_coordinates
    h = P["Lx"] / P["Nx"]
    coords = np.array([np.ravel(z) / h - 0.5, np.ravel(y) / h - 0.5, np.ravel(x) / h - 0.5])
    return map_coordinates(f, coords, order=3, mode="grid-wrap", prefilter=False).reshape(np.shape(x))


def core(w, P, xm, rm, half=0.6, ds=0.01):
    """Core centroid in each of N_THETA meridian half-planes, and the azimuthal average's moments."""
    theta = 2 * np.pi * np.arange(N_THETA) / N_THETA
    s = np.arange(-half, half + ds / 2, ds)
    X, Rr = np.meshgrid(xm + s, rm + s, indexing="ij")
    X3 = np.broadcast_to(X, (N_THETA,) + X.shape)
    R3 = Rr[None] * np.ones((N_THETA, 1, 1))
    c, sn = np.cos(theta)[:, None, None], np.sin(theta)[:, None, None]
    Y3, Z3 = P["yc"] + R3 * c, P["zc"] + R3 * sn
    wx, wy, wz = (sample(w[k], P, X3, Y3, Z3) for k in range(3))
    wt = -wy * sn + wz * c
    wr = wy * c + wz * sn
    # Norm of each azimuthal mode of the vorticity in the window, over that of the mean azimuthal vorticity
    spec = sum(np.abs(np.fft.rfft(f, axis=0)) ** 2 for f in (wx, wr, wt)).sum(axis=(1, 2))
    vort = np.sqrt(spec / (np.abs(wt.sum(axis=0)) ** 2).sum())
    xc, rc = np.empty(N_THETA), np.empty(N_THETA)
    for j in range(N_THETA):
        m = np.where(wt[j] > 0.2 * wt[j].max(), wt[j], 0.0)
        xc[j], rc[j] = (m * X).sum() / m.sum(), (m * Rr).sum() / m.sum()
    avg = wt.mean(axis=0)
    pos = np.maximum(avg, 0.0)
    circ = pos.sum() * ds * ds
    xbar, rbar = (pos * X).sum() / pos.sum(), (pos * Rr).sum() / pos.sum()
    a2 = (pos * ((X - xbar) ** 2 + (Rr - rbar) ** 2)).sum() / pos.sum()
    return theta, xc, rc, dict(circulation=circ, x=xbar, R=rbar, a=np.sqrt(a2), peak=avg.max(), vort=vort)


def plane(w, P, x, n=240):
    """|omega| on the plane x (the ring's plane), on an n x n grid over the box's cross-section."""
    y = (np.arange(n) + 0.5) * P["Ly"] / n
    z = (np.arange(n) + 0.5) * P["Lz"] / n
    Y, Z = np.meshgrid(y, z, indexing="xy")
    X = np.full_like(Y, x)
    mag = np.sqrt(sum(sample(w[k], P, X, Y, Z) ** 2 for k in range(3)))
    return mag.astype(np.float32)


def modes(args):
    P = params(args.input)
    h = P["Lx"] / P["Nx"]
    nmax = 24
    rows, planes = [], {}
    xm, rm = P["x0"], P["R"]
    for path in args.restarts:
        t, u = load_velocity(path, P)
        from scipy.ndimage import spline_filter
        w = np.stack([spline_filter(c, order=3, mode="grid-wrap") for c in vorticity(u, h)])
        del u
        # Two passes: the core's mean position first, then the centroids around it
        for _ in range(2):
            theta, xc, rc, m = core(w, P, xm, rm)
            xm, rm = m["x"], m["R"]
        dr = np.fft.rfft(rc - rc.mean()) * 2 / N_THETA
        dx = np.fft.rfft(xc - xc.mean()) * 2 / N_THETA
        amp = np.sqrt(np.abs(dr) ** 2 + np.abs(dx) ** 2)[:nmax + 1]
        row = dict(t=t, x=m["x"], R=m["R"], a=m["a"], circulation=m["circulation"], peak=m["peak"])
        row.update({f"A{n}": amp[n] for n in range(1, nmax + 1)})
        row.update({f"W{n}": m["vort"][n] for n in range(1, nmax + 1)})
        rows.append(row)
        planes[f"t{t:.3f}"] = plane(w, P, m["x"])
        planes[f"core{t:.3f}"] = np.stack([xc, rc]).astype(np.float32)
        lead = int(np.argmax(amp[2:])) + 2
        print(f"t = {t:8.3f}  R = {m['R']:.4f}  a = {m['a']:.4f}  Gamma = {m['circulation']:.4f}  "
              f"leading n = {lead} ({amp[lead]:.2e})", flush=True)
    with open(args.csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        for row in rows:
            wr.writerow({k: f"{v:.8g}" for k, v in row.items()})
    if args.planes:
        np.savez_compressed(args.planes, **planes)
    print(f"wrote {args.csv}")


def theory(eps, re):
    """Growth rate of the principal elliptic mode of a Gaussian ring, times 2 pi R^2 / Gamma."""
    return 0.5171 * np.log(8 / eps) - 0.9285 - 12.5 / (re * eps ** 2)


def read_modes(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    cols = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    nmax = max(int(k[1:]) for k in cols if k.startswith("W"))
    amp = np.array([cols[f"W{n}"] for n in range(1, nmax + 1)]).T  # (time, n - 1)
    return cols, amp


def fit(args):
    P = params(args.input)
    cols, amp = read_modes(args.modes)
    t = cols["t"]
    t1, t2 = args.window
    win = (t >= t1) & (t <= t2)
    R, a, circ = cols["R"][win].mean(), cols["a"][win].mean(), cols["circulation"][win].mean()
    scale = 2 * np.pi * R ** 2 / circ
    ns = np.arange(1, amp.shape[1] + 1)
    sig = np.array([np.polyfit(t[win], np.log(amp[win, k]), 1)[0] for k in range(amp.shape[1])]) * scale
    eps, re = a / R, circ / (2 * np.pi * P["nu"])
    return dict(t=t, amp=amp, ns=ns, sig=sig, eps=eps, re=re, scale=scale, R=R, a=a, circ=circ, win=win,
                theory=theory(eps, re), n_theory=K_PRINCIPAL / eps, cols=cols, P=P)


def compare(args):
    F = fit(args)
    ns, sig = F["ns"], F["sig"]
    lead = int(ns[1:][np.argmax(sig[1:])])
    big = int(ns[1:][np.argmax(F["amp"][F["win"]][-1, 1:])])
    print(f"window t = {args.window[0]:g}-{args.window[1]:g}: R = {F['R']:.4f}, a = {F['a']:.4f} "
          f"(eps = {F['eps']:.4f}), Gamma = {F['circ']:.4f} (Re = {F['re']:.0f}), 2 pi R^2 / Gamma = {F['scale']:.3f}")
    for n in range(max(2, lead - 4), min(len(ns), lead + 4) + 1):
        print(f"  n = {n:2d}  n eps = {n * F['eps']:.3f}  sigma 2 pi R^2 / Gamma = {sig[n - 1]:.3f}")
    print(f"fastest-growing n = {lead} ({sig[lead - 1]:.3f}); largest at the window's end n = {big}")
    print(f"theory (principal elliptic mode, Gaussian core): n = 2.33 / eps = {F['n_theory']:.2f}, "
          f"sigma 2 pi R^2 / Gamma = {F['theory']:.3f} (inviscid {theory(F['eps'], np.inf):.3f}); "
          f"measured / theory = {sig[lead - 1] / F['theory']:.3f}")
    if args.png or args.figure:
        draw_figure(args, F)


def draw_figure(args, F):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    planes = np.load(args.planes) if args.planes else None

    def draw(P):
        fig = plt.figure(figsize=(12, 8.2))
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.95], hspace=0.38, wspace=0.32)
        T = F["scale"]
        if planes is not None:
            keys = sorted((float(k[1:]), k) for k in planes.files if k.startswith("t"))
            times = np.array([k[0] for k in keys])
            for i, ts in enumerate(args.snapshots):
                ax = fig.add_subplot(gs[0, i])
                k = keys[int(np.argmin(np.abs(times - ts * T)))]
                img = planes[k[1]]
                L = F["P"]["Ly"]
                vmax = np.percentile(img, 99.9)
                ax.imshow(img, origin="lower", extent=[-L / 2, L / 2, -L / 2, L / 2], cmap="magma" if P["name"] == "dark" else "magma_r",
                          vmin=0, vmax=vmax, interpolation="bilinear")
                ax.set_xlim(-1.6, 1.6)
                ax.set_ylim(-1.6, 1.6)
                ax.set_aspect("equal")
                ax.set_xlabel(r"$y / R$")
                if i == 0:
                    ax.set_ylabel(r"$z / R$")
                ax.set_title(rf"$|\omega|$ in the ring's plane, $t\,\Gamma / 2\pi R^2 = {k[0] / T:.1f}$", fontsize=12)
        t = F["t"] / T
        ax = fig.add_subplot(gs[1, :2])
        lead = int(F["ns"][1:][np.argmax(F["sig"][1:])])
        colors = (P["green"], P["flame"], P["blue"], P["teal"], P["plum"])
        win = F["win"]
        for c, n in zip(colors, range(lead - 2, lead + 3)):
            ax.semilogy(t, F["amp"][:, n - 1], color=c, lw=1.4, label=f"n = {n}")
            tw = t[win]
            k0 = np.polyfit(F["t"][win], np.log(F["amp"][win, n - 1]), 1)
            ax.semilogy(tw, np.exp(np.polyval(k0, F["t"][win])), color=c, lw=3.5, alpha=0.35)
        tw = t[win]
        k0 = np.polyfit(F["t"][win], np.log(F["amp"][win, lead - 1]), 1)
        mid = 0.5 * (tw[0] + tw[-1])
        a_mid = np.exp(np.polyval(k0, mid * T))
        tt = np.array([tw[0] - 1.0, tw[-1] + 1.0])
        ax.semilogy(tt, a_mid * np.exp(F["theory"] * (tt - mid)), "--", color=P["fg"], lw=1.2,
                    label=rf"theory, $\sigma\,2\pi R^2/\Gamma = {F['theory']:.2f}$")
        ax.set_xlabel(r"$t\,\Gamma / 2\pi R^2$")
        ax.set_ylabel(r"vorticity of mode $n$ in the core, $\|\omega_n\| / \|\bar\omega_\theta\|$")
        ax.legend(loc="upper left", fontsize=10, ncol=2)
        ax = fig.add_subplot(gs[1, 2])
        ns = F["ns"][1:]
        ax.plot(ns, F["sig"][1:], "o", color=P["blue"], ms=5, label="Mallard")
        ax.axhline(F["theory"], color=P["fg"], ls="--", lw=1.2, label="theory, maximum")
        ax.axvline(F["n_theory"], color=P["flame"], ls=":", lw=1.4, label=rf"$n = 2.33\,R/a$")
        ax.set_xlim(1, ns[-1] + 0.5)
        ax.set_ylim(min(-0.2, F["sig"][1:].min() - 0.05), 1.5 * max(F["theory"], F["sig"].max()))
        ax.set_xlabel("azimuthal mode $n$")
        ax.set_ylabel(r"growth rate $\sigma\,2\pi R^2 / \Gamma$")
        ax.legend(loc="upper left", fontsize=9)
        return fig

    if args.png:
        P = dict(name="light", fg="#1f1a1d", muted="#6b6166", blue="#2f6db5", flame="#d9730d", teal="#12655f",
                 plum="#7b3f6e", green="#3a7d2c", red="#c0392b")
        draw(P).savefig(args.png, dpi=150, bbox_inches="tight")
        print(f"wrote {args.png}")
    if args.figure:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../tools/docs"))
        import figstyle
        figstyle.save("vortex_ring_3d", draw)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table")
    t.add_argument("input")
    t.add_argument("output")
    m = sub.add_parser("modes")
    m.add_argument("input")
    m.add_argument("restarts", nargs="+")
    m.add_argument("--csv", required=True)
    m.add_argument("--planes", help="npz of |omega| on the ring's plane and the core centroids, for the figure")
    c = sub.add_parser("compare")
    c.add_argument("input")
    c.add_argument("modes")
    c.add_argument("--window", type=float, nargs=2, default=WINDOW, help="fit window in time units of INPUT")
    c.add_argument("--planes", help="npz from modes --planes, for the figure's top row")
    c.add_argument("--snapshots", type=float, nargs=3, default=SNAPSHOTS, help="times of the top row, t Gamma / 2 pi R^2")
    c.add_argument("--png")
    c.add_argument("--figure", action="store_true")
    args = ap.parse_args()
    {"table": table, "modes": modes, "compare": compare}[args.cmd](args)


if __name__ == "__main__":
    main()
