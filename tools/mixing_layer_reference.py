"""Laminar reacting mixing layer (V11) from the boundary-layer equations, with Cantera.

    python tools/mixing_layer_reference.py OUT_DIR [--t-air 1150 --u-air 100 --u-fuel 50 ...]

Two parallel streams meet at x = 0 behind a splitter plate of zero
thickness: fuel (H2/N2, 300 K) below, hot air above, both at 1 atm. Hot air
autoignites the mixture that forms in the layer, and the ignition kernel
turns into a diffusion flame trailing downstream: a lifted flame,
stabilized by autoignition rather than propagation as long as the flow is
much faster than the premixed flames of the layer's mixtures.

The steady, planar, constant-pressure boundary-layer equations (no
streamwise diffusion) are marched in x from the step, with detailed
chemistry (mechanisms/h2o2.yaml) and Cantera's mixture-averaged transport,
the same model as Mallard's `transport = "mixture_averaged"` (diffusion
fluxes -rho D_km (W_k / W) grad X_k with a correction velocity, enthalpy
transport by them, viscous heating). In von Mises form (stream function
psi, d psi / dy = rho u) with the similarity variable eta = psi / sqrt(x)
and tau = ln x, the equations are parabolic in tau on a fixed eta grid:

  d phi / d tau = (eta / 2) d phi / d eta + d/d eta (C d phi / d eta) + x S

with C = rho u mu for u, rho u lambda / cp for T and the species fluxes
rho^2 u (W_k / W) D_km d X_k / d eta; S is the chemical source over rho u.
Without chemistry the solution is self-similar (Lock's mixing layer: tau
drops out), so the march starts at x = 1 nm from the step and the layer
is resolved by the same number of points at every x. The method of lines
(second-order differences, scipy's BDF with a sparse Jacobian) integrates
the stiff chemistry with the transport. y is recovered from
y = sqrt(x) int d eta / (rho u), from the dividing streamline psi = 0 (the
boundary-layer equations leave the layer's lateral position open).

Writes to OUT_DIR:
  ignition.csv   along x: peak T, peak Y_OH, peak heat release rate and the
                 layer-integrated heat release, with their positions
  profiles.csv   profiles at the stations (--stations): y, eta, u, T,
                 Bilger's mixture fraction Z, Y_k, heat release
  field.npz      with --field: the whole solution at the points of
                 ignition.csv (x, y, u, T, Y), the initial state of
                 Mallard's runs (tools/mixing_layer_setup.py)
"""
import argparse
import os
import time

import cantera as ct
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import kron, diags, csr_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MECH, PHASE = os.path.join(ROOT, "mechanisms", "h2o2.yaml"), "ohmech"
AIR = "O2:0.21,N2:0.79"


def bilger(gas, Y, Y_fuel, Y_air):
    """Bilger's mixture fraction of mass fractions Y (rows), 0 in air, 1 in fuel."""
    def beta(Y):
        w = {}
        for e in ("H", "O"):
            k = gas.element_index(e)
            w[e] = sum(gas.n_atoms(s, e) * Y[..., i] / gas.molecular_weights[i]
                       for i, s in enumerate(gas.species_names)) * gas.atomic_weight(k)
        return w["H"] / (2 * gas.atomic_weight("H")) - w["O"] / gas.atomic_weight("O")
    b, bf, ba = beta(Y), beta(Y_fuel), beta(Y_air)
    return (b - ba) / (bf - ba)


class MixingLayer:
    def __init__(self, gas, fuel, t_fuel, u_fuel, t_air, u_air, p, n, eta_width):
        self.gas, self.p = gas, p
        self.K = gas.n_species
        self.W = gas.molecular_weights
        gas.TPX = t_fuel, p, fuel
        self.Y_fuel, rho_f, mu_f, D_f = gas.Y.copy(), gas.density, gas.viscosity, gas.mix_diff_coeffs.max()
        alpha_f = gas.thermal_conductivity / gas.cp_mass
        gas.TPX = t_air, p, AIR
        self.Y_air, rho_a, mu_a, D_a = gas.Y.copy(), gas.density, gas.viscosity, gas.mix_diff_coeffs.max()
        alpha_a = gas.thermal_conductivity / gas.cp_mass
        # Widths of each side in eta: the largest diffusivity of the free stream
        w_f = np.sqrt(rho_f * u_fuel * max(mu_f, alpha_f, rho_f * D_f))
        w_a = np.sqrt(rho_a * u_air * max(mu_a, alpha_a, rho_a * D_a))
        self.eta = np.linspace(-eta_width * w_f, eta_width * w_a, n)
        self.h = self.eta[1] - self.eta[0]
        self.n = n
        self.nv = 2 + self.K
        fuel_side = self.eta < 0
        self.u0 = np.where(fuel_side, u_fuel, u_air)
        self.T0 = np.where(fuel_side, t_fuel, t_air)
        self.Y0 = np.where(fuel_side[:, None], self.Y_fuel, self.Y_air)
        self.states = ct.SolutionArray(gas, n)
        self.n_rhs = 0

    def unpack(self, z):
        v = z.reshape(self.n - 2, self.nv)
        u = np.concatenate(([self.u0[0]], v[:, 0], [self.u0[-1]]))
        T = np.concatenate(([self.T0[0]], v[:, 1], [self.T0[-1]]))
        Y = np.vstack((self.Y0[:1], v[:, 2:], self.Y0[-1:]))
        return u, T, Y

    def pack(self, u, T, Y):
        return np.column_stack((u[1:-1], T[1:-1], Y[1:-1])).ravel()

    def properties(self, T, Y):
        s = self.states
        s.TPY = T, self.p, np.clip(Y, 0, None)
        return s

    def rhs(self, tau, z):
        self.n_rhs += 1
        x = np.exp(tau)
        u, T, Y = self.unpack(z)
        s = self.properties(T, Y)
        rho, mu, lam, cp = s.density, s.viscosity, s.thermal_conductivity, s.cp_mass
        Wbar = s.mean_molecular_weight
        D = s.mix_diff_coeffs
        X = s.X
        h = self.h
        mean = lambda a: 0.5 * (a[1:] + a[:-1])
        ru = rho * u
        eta_i = self.eta[1:-1, None]
        # Momentum
        Fu = mean(ru * mu) * np.diff(u) / h
        du = 0.5 * self.eta[1:-1] * (u[2:] - u[:-2]) / (2 * h) + np.diff(Fu) / h
        # Species fluxes at faces, with the correction velocity
        coef = (rho * ru)[:, None] * (self.W / Wbar[:, None]) * D
        J = -mean(coef) * np.diff(X, axis=0) / h
        J -= mean(Y) * J.sum(axis=1, keepdims=True)
        wdot = s.net_production_rates  # kmol/m^3/s
        src = (wdot * self.W) / ru[:, None]
        dY = eta_i / 2 * (Y[2:] - Y[:-2]) / (2 * h) - np.diff(J, axis=0) / h + x * src[1:-1]
        # Energy
        Fq = mean(ru * lam) * np.diff(T) / h
        dTdeta = (T[2:] - T[:-2]) / (2 * h)
        cpk = s.partial_molar_cp / self.W  # J/kg/K
        Jn = mean(J)  # fluxes at the interior nodes
        enth = -np.sum(cpk[1:-1] * Jn, axis=1) * dTdeta
        visc = (mu * ru)[1:-1] * ((u[2:] - u[:-2]) / (2 * h)) ** 2
        hrr = -np.sum(s.partial_molar_enthalpies * wdot, axis=1) / ru
        dT = 0.5 * self.eta[1:-1] * dTdeta + (np.diff(Fq) / h + enth + visc + x * hrr[1:-1]) / cp[1:-1]
        return np.column_stack((du, dT, dY)).ravel()

    def sparsity(self):
        m = self.n - 2
        band = diags([1.0, 1.0, 1.0], [-1, 0, 1], shape=(m, m))
        return csr_matrix(kron(band, np.ones((self.nv, self.nv))))

    def fields(self, x, z):
        """Profiles at x: y from the dividing streamline, u, T, Z, Y, heat release rate."""
        u, T, Y = self.unpack(z)
        s = self.properties(T, Y)
        rho = s.density
        g = 1 / (rho * u)
        y = np.sqrt(x) * np.concatenate(([0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(self.eta))))
        y -= np.interp(0.0, self.eta, y)
        hrr = -np.sum(s.partial_molar_enthalpies * s.net_production_rates, axis=1)
        Z = bilger(self.gas, Y, self.Y_fuel, self.Y_air)
        return dict(y=y, eta=self.eta, u=u, T=T, rho=rho, Z=Z, Y=Y, HRR=hrr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--fuel", default="H2:1,N2:1")
    ap.add_argument("--t-fuel", type=float, default=300.0)
    ap.add_argument("--u-fuel", type=float, default=50.0)
    ap.add_argument("--t-air", type=float, default=1150.0)
    ap.add_argument("--u-air", type=float, default=100.0)
    ap.add_argument("--p", type=float, default=ct.one_atm)
    ap.add_argument("--n", type=int, default=801, help="points across the layer")
    ap.add_argument("--eta-width", type=float, default=10.0,
                    help="half-widths of the eta grid, in each free stream's diffusion width")
    ap.add_argument("--x-max", type=float, default=0.03)
    ap.add_argument("--n-x", type=int, default=600, help="points of ignition.csv, log-spaced from 0.1 mm")
    ap.add_argument("--stations", type=float, nargs="*", default=[0.005, 0.01, 0.015, 0.02, 0.025, 0.03])
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--field", action="store_true", help="also write field.npz")
    a = ap.parse_args()

    gas = ct.Solution(MECH, PHASE)
    gas.transport_model = "mixture-averaged"
    ml = MixingLayer(gas, a.fuel, a.t_fuel, a.u_fuel, a.t_air, a.u_air, a.p, a.n, a.eta_width)
    x_out = np.unique(np.concatenate((np.geomspace(1e-4, a.x_max, a.n_x), a.stations)))
    z0 = ml.pack(ml.u0, ml.T0, ml.Y0)
    t0 = time.time()
    sol = solve_ivp(ml.rhs, (np.log(1e-9), np.log(a.x_max)), z0, method="BDF", t_eval=np.log(x_out),
                    jac_sparsity=ml.sparsity(), rtol=a.rtol, atol=1e-12)
    if not sol.success:
        raise SystemExit(sol.message)
    print(f"{sol.t.size} outputs, {ml.n_rhs} right-hand sides, {time.time() - t0:.0f} s")

    os.makedirs(a.out_dir, exist_ok=True)
    i_oh = gas.species_index("OH")
    header = (f"Laminar mixing layer (V11), boundary-layer equations: fuel {a.fuel} at {a.t_fuel} K and "
              f"{a.u_fuel} m/s, air at {a.t_air} K and {a.u_air} m/s, p = {a.p} Pa, {a.n} points, "
              f"eta width {a.eta_width}; rtol {a.rtol}\n")
    rows, field = [], {"y": [], "u": [], "T": [], "Y": []}
    for k, tau in enumerate(sol.t):
        x = np.exp(tau)
        f = ml.fields(x, sol.y[:, k])
        for name in field:
            field[name].append(f[name])
        hrr_int = np.trapezoid(f["HRR"], f["y"])
        iT, io, ih = np.argmax(f["T"]), np.argmax(f["Y"][:, i_oh]), np.argmax(f["HRR"])
        rows.append([x, f["T"][iT], f["y"][iT], f["Z"][iT], f["Y"][io, i_oh], f["y"][io], f["HRR"][ih],
                     f["y"][ih], hrr_int])
    np.savetxt(os.path.join(a.out_dir, "ignition.csv"), np.array(rows), delimiter=",",
               header=header + "x,T_max,y_T_max,Z_T_max,Y_OH_max,y_OH_max,HRR_max,y_HRR_max,HRR_integral",
               comments="# ", fmt="%.8g")
    if a.field:
        np.savez_compressed(os.path.join(a.out_dir, "field.npz"), x=np.exp(sol.t),
                            **{k: np.array(v) for k, v in field.items()}, species=gas.species_names,
                            fuel=a.fuel, t_fuel=a.t_fuel, u_fuel=a.u_fuel, t_air=a.t_air, u_air=a.u_air, p=a.p)
    prof = []
    for xs in a.stations:
        k = int(np.argmin(np.abs(np.exp(sol.t) - xs)))
        f = ml.fields(xs, sol.y[:, k])
        prof.append(np.column_stack((np.full(ml.n, xs), f["y"], f["eta"], f["u"], f["T"], f["rho"], f["Z"],
                                     f["HRR"], f["Y"])))
    names = ",".join("Y_" + s for s in gas.species_names)
    np.savetxt(os.path.join(a.out_dir, "profiles.csv"), np.vstack(prof), delimiter=",",
               header=header + "x,y,eta,u,T,rho,Z,HRR," + names, comments="# ", fmt="%.8g")


if __name__ == "__main__":
    main()
