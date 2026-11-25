"""Figures of the library of named liquids, in the particles guide and the design
notes (docs/images/guide/).

    python tools/docs/liquid_library.py [--compute]

With --compute (needs the CoolProp Python package, of the version the build
fetches, CMakeLists.txt), the saturated-liquid curves of the library's fluids
and a resolution study of their tables are computed from CoolProp and saved in
liquid_library.json; without it (make_figures.py) the saved data are redrawn.
The study mirrors src/particles/liquid_library.cpp: cubic Hermite tables on the
knots of particles::Curve, with CoolProp's values and central-difference slopes
at the knots, from the triple point to 0.999 of the critical temperature; their
largest relative error at 8001 temperatures against the knots, on the stretched
grid of the tables (a = 0.05) and on a uniform one.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import figstyle

DATA = Path(__file__).with_suffix(".json")

CURVES = ("water", "methanol", "ethanol", "ammonia", "n-heptane", "n-dodecane", "oxygen")
COOLPROP = {"water": "Water", "methanol": "Methanol", "ethanol": "Ethanol", "ammonia": "Ammonia",
            "n-heptane": "n-Heptane", "n-dodecane": "n-Dodecane", "oxygen": "Oxygen"}
COLORS = {"water": "blue", "methanol": "teal", "ethanol": "green", "ammonia": "plum", "n-heptane": "flame",
          "n-dodecane": "red", "oxygen": "gold"}
STUDY = ("water", "n-dodecane")
PROPERTIES = ("density", "latent heat", "saturation pressure", "viscosity", "conductivity", "surface tension")
LOGARITHMS = ("saturation pressure", "viscosity")
PROPERTY_COLORS = dict(zip(PROPERTIES, ("blue", "flame", "teal", "plum", "red", "green")))
KNOTS = [2 ** m + 1 for m in range(3, 14)]
MARGIN = 1e-3


class Fluid:
    """CoolProp's saturated liquid of one fluid, its tables' range as liquid_coolprop.cpp finds it."""

    def __init__(self, name):
        import CoolProp

        self.state = CoolProp.AbstractState("HEOS", COOLPROP[name])
        self.T_triple = self.state.keyed_output(CoolProp.iT_triple)
        self.T_critical = self.state.T_critical()
        limit = self.T_critical
        if not self.exists(self.T_critical * (1 - MARGIN)):
            lo, hi = self.T_triple, self.T_critical
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if self.exists(mid) else (lo, mid)
            limit = lo
        self.T_max = limit * (1 - MARGIN)

    def exists(self, T):
        try:
            self.values(np.array([T]))
            return True
        except ValueError:
            return False

    def values(self, T):
        """The properties at temperatures T, the logarithms of the saturation pressure and viscosity."""
        import CoolProp

        out = {p: np.empty(len(T)) for p in PROPERTIES}
        for i, t in enumerate(T):
            self.state.update(CoolProp.QT_INPUTS, 1.0, t)
            h_vapor = self.state.hmass()
            self.state.update(CoolProp.QT_INPUTS, 0.0, t)
            out["density"][i] = self.state.rhomass()
            out["latent heat"][i] = h_vapor - self.state.hmass()
            out["saturation pressure"][i] = np.log(self.state.p())
            out["viscosity"][i] = np.log(self.state.viscosity())
            out["conductivity"][i] = self.state.conductivity()
            out["surface tension"][i] = self.state.surface_tension()
        return out

    def specific_heat(self, T):
        import CoolProp

        out = np.empty(len(T))
        for i, t in enumerate(T):
            self.state.update(CoolProp.QT_INPUTS, 0.0, t)
            out[i] = self.state.cpmass()
        return out


def knots(lo, hi, a, n):
    """particles::Curve::at at the n knots, and ds/dx there."""
    u = 1.0 - np.linspace(0.0, 1.0, n)
    s = hi - (hi - lo) * u * (u + a) / (1 + a)
    s[0], s[-1] = lo, hi
    return s, (hi - lo) * (2 * u + a) / (1 + a)


def coordinate(T, lo, hi, a):
    r = np.clip((hi - T) / (hi - lo), 0.0, 1.0)
    return 1.0 - 2.0 * (1 + a) * r / (a + np.sqrt(a * a + 4 * (1 + a) * r))


def hermite(x, y, d, n):
    """particles::Curve::evaluate, the slopes dy/dx at the knots."""
    scaled = x * (n - 1)
    k = np.minimum(scaled.astype(int), n - 2)
    t, h = scaled - k, 1.0 / (n - 1)
    return ((2 * t ** 3 - 3 * t ** 2 + 1) * y[k] + (t ** 3 - 2 * t ** 2 + t) * d[k] * h
            + (3 * t ** 2 - 2 * t ** 3) * y[k + 1] + (t ** 3 - t ** 2) * d[k + 1] * h)


def study(fluid, a):
    lo, hi = fluid.T_triple, fluid.T_max
    T = np.linspace(lo, hi, 8001)
    exact = fluid.values(T)
    errors = {p: [] for p in PROPERTIES}
    for n in KNOTS:
        s, stretch = knots(lo, hi, a, n)
        y, minus, plus = fluid.values(s), fluid.values(s * (1 - 1e-6)), fluid.values(s * (1 + 1e-6))
        x = coordinate(T, lo, hi, a)
        for p in PROPERTIES:
            d = (plus[p] - minus[p]) / (2e-6 * s) * stretch
            v = hermite(x, y[p], d, n)
            e = np.abs(v - exact[p]) if p in LOGARITHMS else np.abs(v - exact[p]) / np.abs(exact[p])
            errors[p].append(float(f"{e.max():.3g}"))
    return errors


def compute():
    import CoolProp

    data = {"coolprop": CoolProp.__version__, "knots": KNOTS, "curves": {}, "study": {}}
    for name in CURVES:
        f = Fluid(name)
        # Crowded toward the critical point, where c_p diverges
        T = f.T_max - (f.T_max - f.T_triple) * np.linspace(1.0, 0.0, 200) ** 2
        v = f.values(T)
        rounded = lambda x: [float(f"{y:.5g}") for y in x]  # noqa: E731
        data["curves"][name] = dict(T=rounded(T), T_critical=f.T_critical, specific_heat=rounded(f.specific_heat(T)),
                                    **{p: rounded(np.exp(v[p]) if p in LOGARITHMS else v[p]) for p in PROPERTIES})
        print(f"{name}: {f.T_triple:.2f} to {f.T_max:.2f} K")
    for name in STUDY:
        f = Fluid(name)
        data["study"][name] = {"stretched": study(f, 0.05), "uniform": study(f, 1e9)}
        for p in PROPERTIES:
            print(f"{name} {p}: " + " ".join(f"{e:.1e}" for e in data["study"][name]["stretched"][p]))
    DATA.write_text(json.dumps(data) + "\n")
    return data


def figure(data):
    import matplotlib.pyplot as plt

    panels = (("density", r"density, kg/m$^3$", False), ("specific_heat", r"$c_p$, kJ/(kg K)", True),
              ("latent heat", "latent heat, MJ/kg", False), ("saturation pressure", "saturation pressure, Pa", True),
              ("viscosity", "viscosity, Pa s", True), ("surface tension", "surface tension, mN/m", False))
    scale = {"specific_heat": 1e-3, "latent heat": 1e-6, "surface tension": 1e3}

    def properties(P):
        fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2), sharex=True)
        for ax, (key, title, log) in zip(axes.flat, panels):
            for name, c in data["curves"].items():
                T = np.array(c["T"])
                ax.plot(T / c["T_critical"], np.array(c[key]) * scale.get(key, 1.0), color=P[COLORS[name]],
                        label=name, lw=1.5)
            ax.set_yscale("log" if log else "linear")
            ax.set_title(title, fontsize=12.5)
        for ax in axes[1]:
            ax.set_xlabel(r"$T / T_c$")
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=11)
        return fig

    def resolution(P):
        fig, axes = plt.subplots(1, len(STUDY), figsize=(11.5, 4.5), sharey=True)
        n = np.array(data["knots"])
        for ax, name in zip(axes, STUDY):
            for p in PROPERTIES:
                color = P[PROPERTY_COLORS[p]]
                ax.loglog(n, data["study"][name]["stretched"][p], "o-", color=color, ms=3.5, label=p)
                ax.loglog(n, data["study"][name]["uniform"][p], "--", color=color, lw=1.2)
            ax.axhline(1e-6, color=P["muted"], lw=1.0, ls=":")
            ax.set(xlabel="knots", title=f"{name}: stretched (solid) and uniform (dashed) knots")
            ax.title.set_fontsize(12)
        axes[0].set_ylabel("largest relative error")
        axes[0].legend(fontsize=10, loc="lower left")
        fig.tight_layout()
        return fig

    figstyle.save("liquid_library_properties", properties)
    figstyle.save("liquid_library_resolution", resolution)


def figures():
    """The figures from the saved data (make_figures.py)."""
    figure(json.loads(DATA.read_text()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--compute", action="store_true", help="recompute the data from CoolProp")
    args = ap.parse_args()
    figure(compute() if args.compute else json.loads(DATA.read_text()))


if __name__ == "__main__":
    main()
