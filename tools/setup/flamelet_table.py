"""Flamelet/progress-variable (FPV) tables from Cantera counterflow flamelets.

    python tools/setup/flamelet_table.py TABLE.toml [-o OUT.mtab]
    python tools/setup/flamelet_table.py TABLE.toml --reference U_OX OUT.csv

Solves the S-curve of steady counterflow diffusion flamelets with Cantera's
CounterflowDiffusionFlame (the axisymmetric stagnation-flow similarity solution):
one flamelet at the oxidizer velocity [flamelets] U_ox, then two-point flame
control (Cantera's diffusion_flame_continuation sample; FPVgen's continuation
logic) lowers the temperature at two control points by
[flamelets] temperature_step at a time, which walks the stable branch to
extinction and the unstable branch below it down to T_max = [flamelets] T_min.
Each flamelet is mapped to the Bilger mixture fraction Z and the progress
variable C = sum_k w_k Y_k ([progress_variable] weights), and labeled by
Lambda = C at the stoichiometric Z (Ihme & Pitsch, Combust. Flame 155, 2008),
which is monotonic along the S-curve. Together with the mixing solution
(Lambda = 0) they give the laminar table phi(Z, Lambda), which is integrated
over a beta PDF of Z at each mean and normalized variance S = Zvar / (Z (1 - Z))
(exactly, for phi piecewise linear in Z, through the incomplete beta function)
with a delta PDF of Lambda. The integrated table is then tabulated against the
normalized mean progress variable c = (C - C(Lambda = 0)) / (C(Lambda_max) - C(Lambda = 0))
in place of Lambda, so that the solver finds its node from the transported
(Z, Zvar, C) without a search. Writes Mallard's table format (.mtab,
docs/design/flamelet.md): a JSON header with axes, fields, units and provenance,
then the values in float64.

--reference U_OX OUT.csv writes one flamelet of the same streams at that
oxidizer velocity instead (z, Z, C, T, omega_C and the table's species): a
verification reference that is not one of the table's flamelets.
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
import tomllib

import numpy as np

FORMAT = "mallard-table"
VERSION = 1
MAGIC = b"mallard-table 1\n"

# Fields every table holds, with their units; species mass fractions follow as Y_<name>.
BASE_FIELDS = [
    ("C", "1"),             # progress variable
    ("omega_C", "kg/m^3/s"),  # its source term
    ("T", "K"),
    ("rho", "kg/m^3"),
    ("W", "kg/kmol"),       # mean molar mass
    ("cp", "J/kg/K"),
    ("e", "J/kg"),          # internal energy, with the species' formation energies
    ("mu", "Pa s"),
    ("kappa", "W/m/K"),     # thermal conductivity
    ("heat_release", "W/m^3"),
]

DEFAULTS = {
    "transport": "unity-Lewis-number",
    "width": 0.01,
    "U_ox": 0.1,
    "temperature_step": 10.0,
    "T_min": 500.0,
    "max_flamelets": 1000,
    "refine": {"ratio": 3.0, "slope": 0.05, "curve": 0.1, "prune": 0.01},
    "Z_points": 101, "Z_cluster": 4.0,
    "S_points": 9,
    "c_points": 41,
    "lambda_points": 101,
    "integration_points": 2001,
}


class ConfigError(ValueError):
    pass


def _take(table, key, default, where, kind=None):
    if key not in table:
        if default is None:
            raise ConfigError(f"{where}: missing key '{key}'")
        return default
    v = table.pop(key)
    if kind is not None and not isinstance(v, kind):
        raise ConfigError(f"{where}.{key}: expected {kind.__name__ if isinstance(kind, type) else kind}, got {v!r}")
    return v


def _check_empty(table, where):
    if table:
        raise ConfigError(f"{where}: unknown key(s) {', '.join(sorted(table))}")


def load_config(path):
    """The table input, with defaults; unknown keys are errors, as in the solver's input."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    text = open(path, encoding="utf-8").read()
    base = os.path.dirname(os.path.abspath(path))
    cfg = {"input_text": text, "input_path": os.path.abspath(path)}
    num = (int, float)
    cfg["output"] = os.path.join(base, _take(raw, "output", os.path.splitext(os.path.basename(path))[0] + ".mtab",
                                             "table input", str))
    mech = _take(raw, "mechanism", None, "table input", dict)
    cfg["mechanism"] = os.path.join(base, _take(mech, "file", None, "[mechanism]", str))
    cfg["phase"] = _take(mech, "phase", "", "[mechanism]", str)
    cfg["transport"] = _take(mech, "transport", DEFAULTS["transport"], "[mechanism]", str)
    if cfg["transport"] not in ("unity-Lewis-number", "mixture-averaged"):
        raise ConfigError("[mechanism].transport: 'unity-Lewis-number' (default) or 'mixture-averaged'")
    _check_empty(mech, "[mechanism]")

    streams = _take(raw, "streams", None, "table input", dict)
    cfg["pressure"] = float(_take(streams, "pressure", None, "[streams]", num))
    for name in ("fuel", "oxidizer"):
        s = _take(streams, name, None, "[streams]", dict)
        cfg[name] = {"T": float(_take(s, "T", None, f"[streams.{name}]", num)),
                     "X": dict(_take(s, "X", None, f"[streams.{name}]", dict))}
        _check_empty(s, f"[streams.{name}]")
    _check_empty(streams, "[streams]")

    fl = _take(raw, "flamelets", {}, "table input", dict)
    cfg["width"] = float(_take(fl, "width", DEFAULTS["width"], "[flamelets]", num))
    cfg["U_ox"] = float(_take(fl, "U_ox", DEFAULTS["U_ox"], "[flamelets]", num))
    cfg["temperature_step"] = float(_take(fl, "temperature_step", DEFAULTS["temperature_step"], "[flamelets]", num))
    cfg["T_min"] = float(_take(fl, "T_min", DEFAULTS["T_min"], "[flamelets]", num))
    cfg["max_flamelets"] = int(_take(fl, "max_flamelets", DEFAULTS["max_flamelets"], "[flamelets]", int))
    refine = dict(DEFAULTS["refine"])
    refine.update(_take(fl, "refine", {}, "[flamelets]", dict))
    cfg["refine"] = refine
    _check_empty(fl, "[flamelets]")

    pv = _take(raw, "progress_variable", None, "table input", dict)
    if not pv or not all(isinstance(v, num) for v in pv.values()):
        raise ConfigError("[progress_variable]: weights of species mass fractions, e.g. { H2O = 1.0 }")
    cfg["progress_variable"] = {k: float(v) for k, v in pv.items()}

    tab = _take(raw, "table", {}, "table input", dict)
    for key in ("Z_points", "S_points", "c_points", "lambda_points", "integration_points"):
        cfg[key] = int(_take(tab, key, DEFAULTS[key], "[table]", int))
        if cfg[key] < 2:
            raise ConfigError(f"[table].{key}: at least 2")
    cfg["Z_cluster"] = float(_take(tab, "Z_cluster", DEFAULTS["Z_cluster"], "[table]", num))
    cfg["species"] = list(_take(tab, "species", [], "[table]", list))
    _check_empty(tab, "[table]")
    _check_empty(raw, "table input")
    return cfg


def make_gas(cfg):
    import cantera as ct
    transport = "mixture-averaged" if cfg["transport"] == "mixture-averaged" else "unity-Lewis-number"
    gas = ct.Solution(cfg["mechanism"], cfg["phase"], transport_model=transport)
    for sp in list(cfg["progress_variable"]) + cfg["species"]:
        if sp not in gas.species_names:
            raise ConfigError(f"species '{sp}' is not in {cfg['mechanism']}")
    return gas


def stoichiometric_Z(gas, cfg):
    gas.set_equivalence_ratio(1.0, cfg["fuel"]["X"], cfg["oxidizer"]["X"])
    return float(gas.mixture_fraction(cfg["fuel"]["X"], cfg["oxidizer"]["X"], basis="mole", element="Bilger"))


def make_flame(cfg):
    import cantera as ct
    gas = make_gas(cfg)
    gas.TP = cfg["oxidizer"]["T"], cfg["pressure"]
    f = ct.CounterflowDiffusionFlame(gas, width=cfg["width"])
    f.fuel_inlet.X, f.fuel_inlet.T = cfg["fuel"]["X"], cfg["fuel"]["T"]
    f.oxidizer_inlet.X, f.oxidizer_inlet.T = cfg["oxidizer"]["X"], cfg["oxidizer"]["T"]
    f.set_refine_criteria(**cfg["refine"])
    f.set_max_grid_points(f.flame, 5000)
    return f


def set_velocity(f, cfg, U_ox):
    """Momentum-balanced jets (rho_f U_f^2 = rho_o U_o^2) at oxidizer velocity U_ox."""
    gas = f.gas
    gas.TPX = cfg["fuel"]["T"], cfg["pressure"], cfg["fuel"]["X"]
    rho_f = gas.density
    gas.TPX = cfg["oxidizer"]["T"], cfg["pressure"], cfg["oxidizer"]["X"]
    rho_o = gas.density
    f.fuel_inlet.mdot = rho_f * U_ox * np.sqrt(rho_o / rho_f)
    f.oxidizer_inlet.mdot = rho_o * U_ox


def flamelet_fields(f, cfg):
    """A flamelet's fields per grid point: Z, then BASE_FIELDS and Y_<species>."""
    arr = f.to_array()
    gas = f.gas
    Z = np.empty(arr.T.size)
    for i in range(Z.size):
        gas.TPY = arr.T[i], cfg["pressure"], arr.Y[i]
        Z[i] = gas.mixture_fraction(cfg["fuel"]["X"], cfg["oxidizer"]["X"], basis="mole", element="Bilger")
    idx = {k: gas.species_index(k) for k in cfg["progress_variable"]}
    Y = arr.Y
    C = sum(w * Y[:, idx[k]] for k, w in cfg["progress_variable"].items())
    wdot = arr.net_production_rates  # kmol/m^3/s
    omega_C = sum(w * gas.molecular_weights[idx[k]] * wdot[:, idx[k]] for k, w in cfg["progress_variable"].items())
    out = {
        "z": np.asarray(f.grid), "Z": Z, "C": C, "omega_C": omega_C, "T": np.asarray(arr.T),
        "rho": np.asarray(arr.density), "W": np.asarray(arr.mean_molecular_weight), "cp": np.asarray(arr.cp_mass),
        "e": np.asarray(arr.int_energy_mass), "mu": np.asarray(arr.viscosity),
        "kappa": np.asarray(arr.thermal_conductivity), "heat_release": np.asarray(f.heat_release_rate),
    }
    for sp in cfg["species"]:
        out["Y_" + sp] = Y[:, gas.species_index(sp)]
    return out


def mixing_state(gas, cfg, Z):
    """Fields of the frozen adiabatic mixture of the streams at mixture fraction Z (Lambda = 0)."""
    gas.TPX = cfg["fuel"]["T"], cfg["pressure"], cfg["fuel"]["X"]
    Yf, hf = gas.Y.copy(), gas.enthalpy_mass
    gas.TPX = cfg["oxidizer"]["T"], cfg["pressure"], cfg["oxidizer"]["X"]
    Yo, ho = gas.Y.copy(), gas.enthalpy_mass
    rows = []
    for z in Z:
        gas.HPY = z * hf + (1 - z) * ho, cfg["pressure"], z * Yf + (1 - z) * Yo
        rows.append(state_row(gas, cfg))
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def state_row(gas, cfg):
    row = {"C": sum(w * gas.Y[gas.species_index(k)] for k, w in cfg["progress_variable"].items()),
           "omega_C": sum(w * gas.molecular_weights[gas.species_index(k)] * gas.net_production_rates[gas.species_index(k)]
                          for k, w in cfg["progress_variable"].items()),
           "T": gas.T, "rho": gas.density, "W": gas.mean_molecular_weight, "cp": gas.cp_mass,
           "e": gas.int_energy_mass, "mu": gas.viscosity, "kappa": gas.thermal_conductivity,
           "heat_release": -float(np.dot(gas.net_production_rates, gas.partial_molar_enthalpies))}
    for sp in cfg["species"]:
        row["Y_" + sp] = gas.Y[gas.species_index(sp)]
    return row


def s_curve(cfg, log=print):
    """The steady flamelets from U_ox along the S-curve: stable branch, extinction, unstable branch."""
    import cantera as ct
    f = make_flame(cfg)
    set_velocity(f, cfg, cfg["U_ox"])
    f.solve(loglevel=0, auto=True)
    if f.extinct():
        raise RuntimeError(f"no burning flamelet at U_ox = {cfg['U_ox']} m/s: lower [flamelets] U_ox")
    flamelets = [flamelet_fields(f, cfg)]
    log(f"flamelet 0: T_max = {np.max(f.T):.1f} K, {f.grid.size} points")
    f.two_point_control_enabled = True
    step = cfg["temperature_step"]
    last = f.to_array()
    while len(flamelets) < cfg["max_flamelets"]:
        T_ctrl = np.min(f.T) + 0.95 * (np.max(f.T) - np.min(f.T))
        f.set_left_control_point(T_ctrl)
        f.set_right_control_point(T_ctrl)
        f.left_control_point_temperature -= step
        f.right_control_point_temperature -= step
        f.clear_stats()
        try:
            f.solve(loglevel=0)
        except ct.CanteraError:
            # Near the turning point the step can be too large: retry from the last flamelet with half of it
            f.from_array(last)
            step *= 0.5
            if step < 0.01 * cfg["temperature_step"]:
                log(f"two-point control stopped at T_max = {np.max(f.T):.1f} K (step {step:.3g} K)")
                break
            continue
        last = f.to_array()
        step = min(cfg["temperature_step"], 1.5 * step)
        T_max = float(np.max(f.T))
        flamelets.append(flamelet_fields(f, cfg))
        if len(flamelets) % 10 == 0:
            log(f"flamelet {len(flamelets) - 1}: T_max = {T_max:.1f} K, {f.grid.size} points")
        if T_max < cfg["T_min"]:
            break
    return flamelets


def z_axis(n, Z_st, cluster):
    """n nodes on [0, 1], denser near Z_st: uniform in a coordinate stretched by sinh about Z_st."""
    if cluster <= 0:
        return np.linspace(0.0, 1.0, n)
    s = np.linspace(0.0, 1.0, n)
    a = np.arcsinh(-Z_st * cluster)
    b = np.arcsinh((1.0 - Z_st) * cluster)
    Z = Z_st + np.sinh(a + (b - a) * s) / cluster
    Z[0], Z[-1] = 0.0, 1.0
    return Z


def s_axis(n):
    """Normalized variance nodes on [0, 1], quadratic so that small variances are resolved."""
    return np.linspace(0.0, 1.0, n) ** 2


def on_z(fl, Z, names, mixing):
    """A flamelet's fields interpolated to Z (sorted by its own Z), closed by the pure streams
    (mixing at Z = 0 and 1) where its nozzles' compositions differ from them."""
    order = np.argsort(fl["Z"], kind="stable")
    z = np.clip(fl["Z"][order], 0.0, 1.0)
    lo, hi = int(z[0] > 0.0), int(z[-1] < 1.0)
    z = np.concatenate([[0.0]] * lo + [z] + [[1.0]] * hi)
    cols = []
    for n in names:
        v = fl[n][order]
        cols.append(np.interp(Z, z, np.concatenate([[mixing[n][0]]] * lo + [v] + [[mixing[n][-1]]] * hi)))
    return np.stack(cols, axis=-1)


def laminar_table(flamelets, mixing, Z, Z_st, n_lambda, names, log=print):
    """phi(Z, Lambda) on a uniform Lambda axis from 0 (mixing) to the most burning flamelet."""
    on = [on_z(fl, Z, names, mixing) for fl in flamelets]
    iC = names.index("C")
    mix = np.stack([mixing[n] for n in names], axis=-1)
    i_st = np.searchsorted(Z, Z_st)
    lam = np.array([float(np.interp(Z_st, Z, t[:, iC])) - float(np.interp(Z_st, Z, mix[:, iC])) for t in on])
    order = np.argsort(lam, kind="stable")
    lam, on = lam[order], [on[i] for i in order]
    keep = np.concatenate([[True], np.diff(lam) > 1e-12 * max(lam[-1], 1e-300)])
    lam, on = lam[keep], [t for t, k in zip(on, keep) if k]
    lam = np.concatenate([[0.0], lam])
    stack = np.stack([mix] + on, axis=1)  # (Z, flamelet, field)
    grid = np.linspace(0.0, lam[-1], n_lambda)
    out = np.empty((Z.size, n_lambda, len(names)))
    for i in range(Z.size):
        for k in range(len(names)):
            out[i, :, k] = np.interp(grid, lam, stack[i, :, k])
    log(f"laminar table: {len(on)} flamelets, Lambda_max = {lam[-1]:.5g} (Z_st = {Z_st:.5g}, node {i_st})")
    return grid, out


def beta_weights(z, Zm, S):
    """Weights w[i] with sum_i w[i] phi(z[i]) = the mean of phi over a beta PDF of mean Zm and
    normalized variance S, exactly for phi piecewise linear on the nodes z (sorted, z[0] = 0, z[-1] = 1).
    S = 0: phi(Zm); S = 1: the two deltas at 0 and 1."""
    from scipy.special import betainc
    n = z.size
    w = np.zeros(n)
    if Zm <= 0.0 or Zm >= 1.0 or S <= 1e-10:
        Zc = min(max(Zm, 0.0), 1.0)
        j = min(max(int(np.searchsorted(z, Zc, side="right")) - 1, 0), n - 2)
        t = (Zc - z[j]) / (z[j + 1] - z[j])
        w[j] += 1.0 - t
        w[j + 1] += t
        return w
    if S >= 1.0 - 1e-10:
        w[0], w[-1] = 1.0 - Zm, Zm
        return w
    g = 1.0 / S - 1.0
    a, b = Zm * g, (1.0 - Zm) * g
    F0 = betainc(a, b, z)          # P(Z <= z)
    F1 = Zm * betainc(a + 1.0, b, z)  # integral of Z P(Z) up to z
    dF0, dF1 = np.diff(F0), np.diff(F1)
    h = np.diff(z)
    # phi on [z_i, z_i+1] = phi_i (z_i+1 - Z) / h + phi_i+1 (Z - z_i) / h
    left = (z[1:] * dF0 - dF1) / h
    right = (dF1 - z[:-1] * dF0) / h
    w[:-1] += left
    w[1:] += right
    return w


def integrate(Zf, lam_table, Zm_axis, S_axis):
    """Beta-PDF means over Z of the laminar table (Zf, Lambda, field) at each (Zm, S): (Zm, S, Lambda, field)."""
    out = np.empty((Zm_axis.size, S_axis.size) + lam_table.shape[1:])
    for i, Zm in enumerate(Zm_axis):
        for j, S in enumerate(S_axis):
            out[i, j] = np.tensordot(beta_weights(Zf, Zm, S), lam_table, axes=(0, 0))
    return out


def to_progress(table, iC, c_axis):
    """(Zm, S, Lambda, field) -> (Zm, S, c, field) with c the normalized mean progress variable."""
    nZ, nS, nL, nF = table.shape
    out = np.empty((nZ, nS, c_axis.size, nF))
    nonmonotonic = 0
    scale = max(float(np.max(np.abs(table[..., iC]))), 1e-300)
    for i in range(nZ):
        for j in range(nS):
            C = table[i, j, :, iC]
            Cm = np.maximum.accumulate(C)
            span = Cm[-1] - Cm[0]
            if span <= 1e-9 * scale:
                out[i, j] = table[i, j, -1][None, :]
                continue
            if np.any(Cm - C > 1e-6 * span):
                nonmonotonic += 1
            c = (Cm - Cm[0]) / span
            for k in range(nF):
                out[i, j, :, k] = np.interp(c_axis, c, table[i, j, :, k])
            out[i, j, :, iC] = Cm[0] + c_axis * span  # C itself is exactly linear in c
    return out, nonmonotonic


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_table(path, header, data):
    """Mallard's table format: MAGIC, one line of JSON (the header), then the values as little-endian float64,
    node-major with the last axis fastest and a node's fields contiguous."""
    shape = tuple(len(a["values"]) for a in header["axes"]) + (len(header["fields"]),)
    if data.shape != shape:
        raise ValueError(f"data shape {data.shape} does not match the header's {shape}")
    if not np.all(np.isfinite(data)):
        raise ValueError("the table holds non-finite values")
    text = json.dumps(header, separators=(",", ":"), allow_nan=False)
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(text.encode("utf-8") + b"\n")
        f.write(np.ascontiguousarray(data, dtype="<f8").tobytes())


def read_table(path):
    """(header, data) of a .mtab file; data has the axes' shape plus the fields."""
    with open(path, "rb") as f:
        if f.readline() != MAGIC:
            raise ValueError(f"{path} is not a Mallard table (version {VERSION})")
        header = json.loads(f.readline())
        data = np.frombuffer(f.read(), dtype="<f8")
    shape = tuple(len(a["values"]) for a in header["axes"]) + (len(header["fields"]),)
    return header, data.reshape(shape)


def build(cfg, log=print):
    import cantera as ct
    gas = make_gas(cfg)
    Z_st = stoichiometric_Z(gas, cfg)
    names = [n for n, _ in BASE_FIELDS] + ["Y_" + s for s in cfg["species"]]
    units = dict(BASE_FIELDS) | {"Y_" + s: "1" for s in cfg["species"]}
    flamelets = s_curve(cfg, log)
    Zf = z_axis(cfg["integration_points"], Z_st, cfg["Z_cluster"])
    mixing = mixing_state(gas, cfg, Zf)
    lam, lam_table = laminar_table(flamelets, mixing, Zf, Z_st, cfg["lambda_points"], names, log)
    Zm = z_axis(cfg["Z_points"], Z_st, cfg["Z_cluster"])
    S = s_axis(cfg["S_points"])
    c = np.linspace(0.0, 1.0, cfg["c_points"])
    integrated = integrate(Zf, lam_table, Zm, S)
    data, nonmonotonic = to_progress(integrated, names.index("C"), c)
    if nonmonotonic:
        log(f"warning: C is not monotonic in Lambda at {nonmonotonic} of {Zm.size * S.size} (Z, S) nodes; "
            "the table follows its running maximum there")
    T_max = [float(np.max(fl["T"])) for fl in flamelets]
    header = {
        "format": FORMAT, "version": VERSION, "model": "fpv",
        "axes": [{"name": "Z", "values": Zm.tolist()},
                 {"name": "S", "values": S.tolist()},
                 {"name": "c", "values": c.tolist()}],
        "fields": [{"name": n, "units": units[n]} for n in names],
        "progress_variable": cfg["progress_variable"],
        "Z_st": Z_st,
        "pressure": cfg["pressure"],
        "streams": {"fuel": cfg["fuel"], "oxidizer": cfg["oxidizer"]},
        "flamelets": {"type": "counterflow", "count": len(flamelets), "transport": cfg["transport"],
                      "T_max_first": T_max[0], "T_max_last": T_max[-1], "Lambda_max": float(lam[-1])},
        "provenance": {
            "tool": "tools/setup/flamelet_table.py",
            "cantera": ct.__version__,
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "mechanism": {"file": os.path.basename(cfg["mechanism"]), "phase": gas.name,
                          "sha256": sha256(cfg["mechanism"])},
            "input": cfg["input_text"],
        },
    }
    return header, data


def reference(cfg, U_ox, out, log=print):
    """One flamelet at U_ox, continued from U_ox = [flamelets] U_ox: z, Z, C, T, omega_C, Y_<species>."""
    import cantera as ct
    f = make_flame(cfg)
    U = min(cfg["U_ox"], U_ox)
    set_velocity(f, cfg, U)
    f.solve(loglevel=0, auto=True)
    while U < U_ox:
        U = min(1.5 * U, U_ox)
        set_velocity(f, cfg, U)
        f.solve(loglevel=0)
    if f.extinct():
        raise RuntimeError(f"extinct at U_ox = {U_ox} m/s")
    fl = flamelet_fields(f, cfg)
    names = ["z", "Z", "C", "T", "omega_C"] + ["Y_" + s for s in cfg["species"]]
    with open(out, "w") as fh:
        fh.write(f"# Cantera {ct.__version__} CounterflowDiffusionFlame, {os.path.basename(cfg['mechanism'])}, "
                 f"{cfg['transport']}; U_ox = {U_ox:g} m/s, T_max = {np.max(fl['T']):.6g} K, "
                 f"{fl['z'].size} points; streams of {os.path.basename(cfg['input_path'])}\n")
        fh.write(",".join(names) + "\n")
        np.savetxt(fh, np.column_stack([fl[n] for n in names]), delimiter=",", fmt="%.10e")
    log(f"{out}: U_ox = {U_ox:g} m/s, T_max = {np.max(fl['T']):.1f} K, {fl['z'].size} points")


def main(argv=None):
    ap = argparse.ArgumentParser(description="FPV tables from Cantera counterflow flamelets")
    ap.add_argument("input", help="table input (TOML)")
    ap.add_argument("-o", "--output", help="table file (default: the input's output key)")
    ap.add_argument("--reference", nargs=2, metavar=("U_OX", "OUT"),
                    help="write one flamelet at oxidizer velocity U_OX to OUT (CSV) instead of the table")
    args = ap.parse_args(argv)
    try:
        cfg = load_config(args.input)
    except ConfigError as e:
        sys.exit(f"{args.input}: {e}")
    if args.reference:
        reference(cfg, float(args.reference[0]), args.reference[1])
        return
    header, data = build(cfg)
    out = args.output or cfg["output"]
    write_table(out, header, data)
    print(f"{out}: {' x '.join(str(len(a['values'])) for a in header['axes'])} nodes, {len(header['fields'])} fields, "
          f"{os.path.getsize(out) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
