"""Writes the inputs of the read benchmark (benchmarks/README.md): OUT/<case>/serial.toml (VTKHDF and XDMF)
and OUT/<case>/mpi.toml (VTU, run on several ranks for PVTU), 5 snapshots of RHO, U and P each.

Usage: python make_cases.py OUT
"""
import os
import sys

CASES = {
    "tut2d": dict(type="cartesian_tri", N=(480, 120), L=(4.0, 1.0)),  # the cluster tutorial's mesh
    "big2d": dict(type="cartesian", N=(1000, 1000), L=(1.0, 1.0)),
    "big3d": dict(type="cartesian", N=(100, 100, 100), L=(1.0, 1.0, 1.0)),
    "huge3d": dict(type="cartesian", N=(200, 200, 200), L=(1.0, 1.0, 1.0)),
}


def toml(c, formats):
    dim = len(c["N"])
    mesh = "\n".join(f"N{a} = {n}\nL{a} = {length}" for a, n, length in zip("xyz", c["N"], c["L"]))
    u = '["0.5 + 0.1 * y", "0.1 * x"' + (', "0.05 * z"]' if dim == 3 else "]")
    rho = "1.0 + 0.3 * x + 0.2 * y * y" + (" + 0.1 * z" if dim == 3 else "")
    zones = ["left", "right", "top", "bottom"] + (["front", "back"] if dim == 3 else [])
    boundaries = "\n".join(f'[[boundaries]]\nname = "{z}"\ntype = "outlet_extrapolation"\n' for z in zones)
    writes = "\n".join(f'[[write_data]]\nprefix = "{f}"\nformat = "{f.replace("pvtu", "vtu")}"\n'
                       f'fields = ["RHO", "U", "P"]\ninterval = 1\n' for f in formats)
    return f"""[run]
n_steps = 4
cfl = 0.2

[mesh]
type = "{c['type']}"
{mesh}

[initialize]
type = "analytical"
rho = "{rho}"
u = {u}
p = "1.0 + 0.1 * x * y"

{boundaries}
[numerics]
riemann_solver = "HLLC"
time_integrator = "SSPRK3"

[numerics.face_reconstruction]
type = "MUSCL"

[physics]
type = "euler"
gamma = 1.4
p_ref = 1.0
T_ref = 1.0
rho_ref = 1.0

[output]
check_interval = 1000

{writes}"""


for name, c in CASES.items():
    d = os.path.join(sys.argv[1], name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "serial.toml"), "w") as f:
        f.write(toml(c, ["vtkhdf", "xdmf"]))
    with open(os.path.join(d, "mpi.toml"), "w") as f:
        f.write(toml(c, ["pvtu"]))
