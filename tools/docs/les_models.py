"""Figures of the LES theory (docs/theory/les.md) and the hybrid flux
(docs/theory/fluxes.md), from the formulas of src/physics/les.h,
src/physics/les.cpp and src/numerics/flux_functor.h:

- sgs_models: the operator D(g) of each subgrid-scale model for canonical
  velocity gradients, and its decay toward a no-slip wall;
- filter_width: Scotti's correction of the filter width for anisotropic cells;
- hybrid_sensor: where the Ducros sensor makes the hybrid flux upwind.

    python tools/docs/les_models.py

The near-wall gradient is that of u = a1 y + a2 y^2, v = b2 y^2,
w = c1 y + c2 y^2, with x and z derivatives of the coefficients chosen so the
flow is divergence free at first order in y (the expansion of Nicoud et al.
2011).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import figstyle

OUT = figstyle.OUT.parent / "theory"


def strain_squared(g):
    s = 0.5 * (g + g.T)
    return float(np.sum(s * s))


def smagorinsky(g):
    return np.sqrt(2.0 * strain_squared(g))


def wale(g):
    g2 = g @ g
    sd = 0.5 * (g2 + g2.T) - np.eye(3) * np.trace(g2) / 3.0
    sd = float(np.sum(sd * sd))
    ss = strain_squared(g)
    den = ss**2.5 + sd**1.25
    return sd**1.5 / den if den > 0 else 0.0


def vreman(g):
    b = g @ g.T
    aa = float(np.sum(g * g))
    B = b[0, 0] * b[1, 1] - b[0, 1]**2 + b[0, 0] * b[2, 2] - b[0, 2]**2 + b[1, 1] * b[2, 2] - b[1, 2]**2
    return np.sqrt(max(B, 0.0) / aa) if aa > 0 else 0.0


def sigma(g):
    s = np.linalg.svd(g, compute_uv=False)
    if not s[0] > 0:
        return 0.0
    return max(s[2] * (s[0] - s[1]) * (s[1] - s[2]) / s[0]**2, 0.0)


MODELS = {"smagorinsky": smagorinsky, "wale": wale, "vreman": vreman, "sigma": sigma}
COLORS = {"smagorinsky": "flame", "wale": "blue", "vreman": "plum", "sigma": "teal"}

CASES = {
    "pure shear": [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
    "solid rotation": [[0, -1, 0], [1, 0, 0], [0, 0, 0]],
    "plane strain": [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
    "a general 2D gradient": [[0.3, 1.1, 0], [-0.4, -0.3, 0], [0, 0, 0]],
    "axisymmetric strain": [[-1, 0, 0], [0, 0.5, 0], [0, 0, 0.5]],
    "isotropic expansion": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "random 3D gradients": None,
}


def near_wall_gradient(y, seed=4):
    rng = np.random.default_rng(seed)
    a1, a2, c1, c2, a1x, a1z, a2x, c1x, c1z, b2x, b2z = rng.uniform(-1, 1, 11)
    b2 = -0.5 * (a1x + c1z)
    return np.array([
        [a1x * y + a2x * y**2, a1 + 2 * a2 * y, a1z * y],
        [b2x * y**2, 2 * b2 * y, b2z * y**2],
        [c1x * y, c1 + 2 * c2 * y, c1z * y],
    ])


def random_gradients(n=2000, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        g = rng.normal(size=(3, 3))
        out.append(g - np.eye(3) * np.trace(g) / 3.0)
    return out


def sgs_models(P):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.8, 4.6), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.35))
    names = list(MODELS)
    table = np.zeros((len(CASES), len(names)))
    for i, (case, g) in enumerate(CASES.items()):
        gs = random_gradients() if g is None else [np.array(g, dtype=float)]
        for j, name in enumerate(names):
            table[i, j] = np.mean([MODELS[name](x) / np.sqrt(np.sum(x * x)) for x in gs])
    cmap = LinearSegmentedColormap.from_list("t", [P["faint"], P["teal"]])
    ax.imshow(table, cmap=cmap, vmin=0, vmax=1.6, aspect="auto")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table[i, j]
            text, color = ("0", P["muted"]) if v < 1e-12 else (f"{v:.2f}", P["bg"] if v > 0.9 else P["fg"])
            ax.text(j, i, text, ha="center", va="center", fontsize=11.5, color=color)
    ax.set_xticks(range(len(names)), [f'"{n}"' for n in names], fontsize=11, family="monospace")
    ax.set_yticks(range(len(CASES)), list(CASES), fontsize=11.5)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(r"$\mathcal{D}(\mathbf{g})\,/\,|\mathbf{g}|$", fontsize=13, pad=30)

    y = np.logspace(-3, -1, 41)
    for name, f in MODELS.items():
        d = np.array([f(near_wall_gradient(v)) for v in y])
        bx.loglog(y, d, color=P[COLORS[name]])
        k = {"smagorinsky": 4, "vreman": 4, "wale": 30, "sigma": 30}[name]
        dy = {"smagorinsky": 2.2, "vreman": 0.3, "wale": 5.0, "sigma": 0.08}[name]
        figstyle.label(bx, y[k] * (1.15 if name == "sigma" else 1.0), d[k] * dy, f'"{name}"', P, color=P[COLORS[name]], size=11,
                       ha="right" if name == "wale" else "left", family="monospace")
        if name != "sigma":
            bx.text(y[-1] * 1.12, d[-1], {"smagorinsky": r"$y^0$", "vreman": r"$y^1$", "wale": r"$y^3$"}[name],
                    color=P["muted"], fontsize=11, va="center" if name != "wale" else "top")
    bx.set_ylim(1e-9, 20)
    bx.set_xlim(1e-3, 0.18)
    bx.set_xlabel("distance from the wall, y")
    bx.set_ylabel(r"$\mathcal{D}$")
    bx.set_title("Toward a no-slip wall", fontsize=13)
    return fig


def scotti(h1, h2, h3):
    l1, l2 = np.log(h1 / h3), np.log(h2 / h3)
    return np.cosh(np.sqrt(4.0 / 27.0 * (l1 * l1 - l1 * l2 + l2 * l2)))


def filter_width(P):
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    a = np.logspace(0, 2, 401)
    X, Y = np.meshgrid(a, a)
    F = scotti(1.0, X, X * Y)
    levels = [1.0, 1.1, 1.25, 1.5, 2, 3, 5, F.max()]
    cmap = LinearSegmentedColormap.from_list("t", [P["faint"], P["teal"]])
    cf = ax.contourf(X, Y, F, levels=levels, cmap=cmap)
    ax.contour(X, Y, F, levels=levels[1:-1], colors=[P["bg"]], linewidths=0.8)
    cb = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.03)
    cb.outline.set_visible(False)
    cb.set_ticks(levels[:-1], labels=[f"{v:g}" for v in levels[:-1]])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$h_2/h_1$")
    ax.set_ylabel(r"$h_3/h_2$")
    ax.set_title(r"$\Delta/\Delta_V$ with $h_1 \leq h_2 \leq h_3$", fontsize=13)
    figstyle.label(ax, 1.08, 30, "pencils\n" + r"$h_1 = h_2$", P, size=10.5, ha="left", box=True)
    figstyle.label(ax, 30, 1.13, r"pancakes, $h_2 = h_3$", P, size=10.5, box=True)
    figstyle.label(ax, 1.15, 1.15, "cube", P, size=10.5, ha="left", va="bottom")
    ax.plot([1], [1], marker="o", color=P["muted"], ms=4, clip_on=False)
    return fig


def hybrid_sensor(P):
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    x = np.linspace(-1.0, 1.0, 2)
    for theta, ls, lw in ((0.3, (0, (5, 2)), 1.1), (0.65, "-", 1.8), (0.9, (0, (1.5, 1.5)), 1.1)):
        slope = np.sqrt((1 - theta) / theta)
        if theta == 0.65:
            ax.fill_between([-1.0, 0.0], [slope, 0.0], 0.0, color=P["flame"], alpha=0.22, lw=0)
        ax.plot([-1.0, 0.0], [slope, 0.0], color=P["flame"], ls=ls, lw=lw,
                label=f"sensor_threshold = {theta:g}" + (" (default)" if theta == 0.65 else ""))
    figstyle.label(ax, -0.55, 0.12, r"$\phi = 1$: Riemann flux", P, size=12, box=True)
    figstyle.label(ax, 0.5, 0.85, r"$\phi = \phi_\mathrm{floor}$:" + "\nKEEP central flux", P, size=12)
    leg = fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=1, fontsize=10.5, handlelength=2.2)
    for t in leg.get_texts():
        t.set_family("monospace")
    figstyle.label(ax, -0.3, 1.2, "vortices, turbulence", P, color=P["muted"], size=10.5)
    figstyle.label(ax, -0.97, 0.02, "shocks", P, color=P["muted"], size=10.5, ha="left", va="bottom")
    ax.axvline(0, color=P["mesh"], lw=0.8, zorder=0)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(0, 1.3)
    ax.set_xlabel(r"$\nabla\cdot\mathbf{u}$")
    ax.set_ylabel(r"$|\nabla\times\mathbf{u}|$")
    ax.set_xticks([-1, 0, 1], ["compression", "0", "expansion"])
    ax.set_yticks([0])
    ax.set_title("Upwind fraction of a cell", fontsize=13)
    return fig


FIGURES = {"sgs_models": sgs_models, "filter_width": filter_width, "hybrid_sensor": hybrid_sensor}


def figures(names=None):
    for name, draw in FIGURES.items():
        if not names or name in names:
            figstyle.save(name, draw, OUT)


if __name__ == "__main__":
    import sys
    figures(sys.argv[1:])
