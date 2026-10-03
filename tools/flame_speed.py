"""Flame speed of a 1D premixed flame run (tools/flame_restart.py, V8).

    python tools/flame_speed.py RUN_DIR [S_REF]

For each output of RUN_DIR/solut/flame.pvd:
  S_c  consumption speed -int W_k omega_k dx / (rho_u (Y_k,u - Y_k,b)) of the
       deficient reactant (H2 or CH4 when lean, O2 when rich), with the fresh
       state at the inlet and the burnt one at the outlet: in a steady flame it
       equals the flame speed, in any frame;
  x_f  the position of the mean of the inlet and maximum temperatures.
Reports S_c averaged over the last third of the outputs, the displacement
speed u_u - dx_f/dt fitted over the last half, and with S_REF their errors.
"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mallard_vtu import read_vtu_cells  # noqa: E402


def profile(path):
    pts, conn, offs, _, arrays = read_vtu_cells(path)
    starts = np.concatenate([[0], offs[:-1]])
    x = np.array([pts[conn[s:e], 0].mean() for s, e in zip(starts, offs)])
    order = np.argsort(x)
    return arrays["TIME"], x[order], {k: v[order] for k, v in arrays.items() if np.ndim(v) == 1 and v.size == x.size}


def deficient(a):
    fuel = "H2" if "Y_CH4" not in a else "CH4"
    stoich_O2_per_fuel = 0.5 if fuel == "H2" else 2.0
    W = {"H2": 2.016, "CH4": 16.043, "O2": 31.998}
    n_fuel, n_O2 = a["Y_" + fuel][0] / W[fuel], a["Y_O2"][0] / W["O2"]
    return fuel if n_O2 >= stoich_O2_per_fuel * n_fuel * (1 - 1e-9) else "O2"


def analyze(run_dir):
    pvd = os.path.join(run_dir, "solut", "flame.pvd")
    files = re.findall(r'file="([^"]+)"', open(pvd).read())
    rows = []
    for f in files:
        t, x, a = profile(os.path.join(os.path.dirname(pvd), f))
        k = deficient(a)
        dx = np.diff(np.concatenate([[0.0], 0.5 * (x[1:] + x[:-1]), [x[-1] + 0.5 * (x[-1] - x[-2])]]))
        consumed = -(a["OMEGA_" + k] * dx).sum()
        S_c = consumed / (a["RHO"][0] * (a["Y_" + k][0] - a["Y_" + k][-1]))
        T = a["T"]
        T_mid = 0.5 * (T[0] + T.max())
        i = np.nonzero(T > T_mid)[0][0]
        x_f = x[i - 1] + (T_mid - T[i - 1]) / (T[i] - T[i - 1]) * (x[i] - x[i - 1])
        rows.append((t, S_c, x_f, a["U_X"][0]))
    return np.array(rows)


def main():
    rows = analyze(sys.argv[1])
    S_ref = float(sys.argv[2]) if len(sys.argv) > 2 else None
    print(f"{'t [ms]':>8} {'S_c [m/s]':>10} {'x_f [mm]':>9} {'u_in [m/s]':>10}")
    for t, S_c, x_f, u in rows:
        print(f"{t * 1e3:8.4f} {S_c:10.5f} {x_f * 1e3:9.4f} {u:10.5f}")
    late = rows[rows[:, 0] >= rows[-1, 0] * 2.0 / 3.0]
    S_c = late[:, 1].mean()
    half = rows[rows[:, 0] >= 0.5 * rows[-1, 0]]
    v_f = np.polyfit(half[:, 0], half[:, 2], 1)[0]
    S_d = half[:, 3].mean() - v_f
    print(f"consumption speed {S_c:.5f} m/s (spread {late[:, 1].max() - late[:, 1].min():.2e}), "
          f"displacement speed {S_d:.5f} m/s (front drift {v_f:.4f} m/s)")
    if S_ref:
        print(f"errors vs {S_ref:.5f}: consumption {100 * (S_c / S_ref - 1):+.3f}%, "
              f"displacement {100 * (S_d / S_ref - 1):+.3f}%")


if __name__ == "__main__":
    main()
