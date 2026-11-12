"""Stability regions of the time integrators, from their coefficients in
src/numerics/time_integrator.cpp, and their measured temporal convergence.

    python tools/docs/time_integrators.py

Each method is applied to dy/dt = z y, the same stage updates as the C++
(SSPRK3's Shu-Osher stages, RK4, the 3S* stages of SSPRK54 and RDPK3Sp35, the
2N stages of CK45), with z a polynomial variable: one step is y_{n+1} = R(z) y_n,
and the method is stable where |R(z)| <= 1. The convergence panel plots the
measured errors of the temporal convergence table in docs/numerics/overview.md
(isentropic vortex, 40 x 40 quads, TENO5, to t = 2).
"""
import re
from pathlib import Path

import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt

import figstyle

SOURCE = Path(__file__).resolve().parents[2] / "src" / "numerics" / "time_integrator.cpp"
OUT = figstyle.OUT.parent / "theory"
Z = Polynomial([0.0, 1.0])
ONE = Polynomial([1.0])


def _block(text, start, end):
    i = text.index(start)
    return text[i:text.index(end, i)]


def _numbers(block, key):
    """The values of `.key = coefficients({...})` in a constructor's tableau."""
    m = re.search(rf"\.{key} = coefficients\(\{{([^}}]*)\}}\)", block)
    return [float(v) for v in m.group(1).replace("\n", " ").split(",") if v.strip()]


def _fractions(block, name):
    m = re.search(rf"double {name}\[5\] = \{{([^}}]*)\}};", block)
    out = []
    for v in m.group(1).split(","):
        a, _, b = v.partition("/")
        out.append(float(a) / float(b or 1))
    return out


def tableaus():
    text = SOURCE.read_text()
    low_storage = {}
    for name, end in (("SSPRK54", "RDPK3Sp35::RDPK3Sp35()"), ("RDPK3Sp35", "CK45::CK45()")):
        block = _block(text, f"{name}::{name}()", end)
        low_storage[name] = {k: _numbers(block, k) for k in ("beta", "gamma1", "gamma2", "delta")}
    block = _block(text, "void CK45::take_step", "const State U")
    return low_storage, dict(A=_fractions(block, "A"), B=_fractions(block, "B"))


def r_3s(t):
    """LowStorage3S::take_step on y' = z y (stage_3s)."""
    y0 = ONE
    y = y0 + t["beta"][0] * Z * y0
    s = 0 * ONE
    for i in range(1, len(t["beta"])):
        d = y - y0
        s = s + t["delta"][i] * d
        y = y0 + t["gamma1"][i] * d + t["gamma2"][i] * s + t["beta"][i] * Z * y
    return y


def r_2n(A, B):
    """CK45::take_step on y' = z y (stage_2n)."""
    y, d = ONE, 0 * ONE
    for a, b in zip(A, B):
        d = a * d + Z * y
        y = y + b * d
    return y


def r_ssprk3():
    u1 = ONE + Z
    u2 = 0.75 * ONE + 0.25 * (u1 + Z * u1)
    return ONE / 3 + 2 / 3 * (u2 + Z * u2)


def r_rk4():
    k1 = ONE
    k2 = ONE + 0.5 * Z * k1
    k3 = ONE + 0.5 * Z * k2
    k4 = ONE + Z * k3
    return ONE + Z / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def polynomials():
    low_storage, ck = tableaus()
    return {
        "FE": ONE + Z,
        "SSPRK3": r_ssprk3(),
        "RK4": r_rk4(),
        "SSPRK54": r_3s(low_storage["SSPRK54"]),
        "CK45": r_2n(ck["A"], ck["B"]),
        "RDPK3Sp35": r_3s(low_storage["RDPK3Sp35"]),
    }


def real_extent(R):
    """Largest x with |R(-s)| <= 1 for all 0 <= s <= x."""
    s = np.linspace(0, 8, 160001)
    bad = np.abs(R(-s)) > 1 + 1e-12
    return s[np.argmax(bad)] if bad.any() else s[-1]


def imag_extent(R):
    s = np.linspace(0, 6, 120001)
    bad = np.abs(R(1j * s)) > 1 + 1e-12
    return s[np.argmax(bad[1:]) + 1] if bad[1:].any() else s[-1]


COLORS = {"FE": "muted", "SSPRK3": "teal", "RK4": "blue", "SSPRK54": "flame", "CK45": "plum", "RDPK3Sp35": "green"}
STYLES = {"FE": (0, (2, 2)), "RK4": (0, (5, 2))}

# docs/numerics/overview.md, temporal convergence: max density difference from RK4 at cfl 0.02
CFL = (1.0, 0.5, 0.25)
ERRORS = {
    "SSPRK3": (1.03e-4, 1.23e-5, 1.51e-6),
    "RK4": (4.59e-6, 2.95e-7, 2.50e-8),
    "SSPRK54": (2.27e-6, 1.47e-7, 1.56e-8),
    "CK45": (1.76e-6, 1.06e-7, 9.02e-9),
    "RDPK3Sp35": (2.21e-5, 2.75e-6, 3.43e-7),
}


def stability(P):
    R = polynomials()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.4, 4.3), gridspec_kw=dict(width_ratios=[1.45, 1], wspace=0.3))
    x, y = np.meshgrid(np.linspace(-6, 1.2, 721), np.linspace(-4.2, 4.2, 841))
    z = x + 1j * y
    ax.axhline(0, color=P["mesh"], lw=0.8, zorder=0)
    ax.axvline(0, color=P["mesh"], lw=0.8, zorder=0)
    for name, r in R.items():
        c = P[COLORS[name]]
        ax.contourf(x, y, np.abs(r(z)), levels=[0, 1], colors=[c], alpha=0.07)
        ax.contour(x, y, np.abs(r(z)), levels=[1], colors=[c], linewidths=1.8,
                   linestyles=[STYLES.get(name, "-")])
        ax.plot([], [], color=c, ls=STYLES.get(name, "-"), label=f"{name} ({real_extent(r):.2f})")
    ax.set_aspect("equal")
    ax.set_xlim(-6, 1.2)
    ax.set_ylim(-4.2, 4.2)
    ax.set_xlabel(r"Re $\lambda\Delta t$")
    ax.set_ylabel(r"Im $\lambda\Delta t$")
    ax.set_title("Stability regions, $|R(\\lambda\\Delta t)| \\leq 1$", fontsize=13)
    for name, e in ERRORS.items():
        c = P[COLORS[name]]
        bx.loglog(CFL, e, marker="o", color=c, ms=5, ls=STYLES.get(name, "-"), label=name)
    for order, e0, y_text in ((3, 1.3e-4, 2.2e-4), (4, 1.2e-6, 4.5e-7)):
        c = np.array([0.25, 1.0])
        bx.loglog(c, e0 * c**order, color=P["muted"], lw=1.0, ls=":")
        bx.text(0.6, y_text * 0.6**order, f"order {order}", color=P["muted"], fontsize=10.5, ha="center",
                va="bottom" if order == 3 else "top")
    bx.set_xlabel("cfl")
    bx.set_ylabel("density error")
    bx.set_xticks(sorted(CFL), ["0.25", "0.5", "1"])
    bx.minorticks_off()
    bx.set_title("Time error, isentropic vortex", fontsize=13)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3,
               fontsize=11, handlelength=2.2, columnspacing=2.0, title="real-axis extent of the stability region",
               title_fontsize=11)
    fig.legends[0].get_title().set_color(P["muted"])
    return fig


def figures():
    figstyle.save("time_integrators", stability, OUT)


if __name__ == "__main__":
    for name, r in polynomials().items():
        print(f"{name:10s} real {real_extent(r):.3f}  imag {imag_extent(r):.3f}  coefficients "
              + " ".join(f"{c:.6g}" for c in r.coef))
    figures()
