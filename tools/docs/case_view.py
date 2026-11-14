"""The case view figures of the user guide: the images `mallard check --view` writes, light and dark,
of a 2D duct and a 3D channel periodic across its span.

    python tools/docs/case_view.py --mallard build/src/mallard [--mallard3d build-3d/src/mallard]

Mallard draws them itself; this script writes the cases, runs the check and copies its images to
docs/images/guide/case-view-{light,dark}.svg and case-view-3d-{light,dark}.svg.
"""
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from figstyle import OUT

PHYSICS = """[physics]
type = "navier_stokes"
gamma = 1.4
p_ref = "1 atm"
T_ref = "288.15 K"
rho_ref = 1.225
mu = 1.8e-5
"""

DUCT = """[run]
t_stop = 0.01
cfl = 0.8

[mesh]
type = "cartesian"
Nx = 40
Ny = 20
Lx = 2.0
Ly = 1.0

[initialize]
type = "constant"
u = [10.0, 0.0]
p = "1 atm"
T = "300 K"

[[boundaries]]
name = "left"
type = "inlet_mass_flow"
mdot = 0.2
T = "300 K"

[[boundaries]]
name = "right"
type = "outlet_pressure"
p = "1 atm"

[[boundaries]]
name = "bottom"
type = "wall_isothermal"
T = "350 K"

[[boundaries]]
name = "top"
type = "wall_adiabatic"

[regions]
heater = {type = "disk", center = [0.6, 0.4], radius = 0.15}

[[sources]]
region = "heater"
rhoE = 1e5

[[sponges]]
strength = 100.0
region = {type = "rectangle", center = [1.85, 0.5], size = [0.3, 1.0]}
u = [10.0, 0.0]
p = "1 atm"
T = "300 K"

[[probes]]
name = "wake"
fields = ["P"]
type = "line"
start = [1.1, 0.1]
end = [1.1, 0.9]
n_points = 9

[[probes]]
name = "tap"
fields = ["P"]
point = [0.3, 0.8]

[[probes]]
name = "ring"
fields = ["P"]
region = {type = "circle", center = [1.4, 0.5], radius = 0.12}
""" + PHYSICS

CHANNEL = """[run]
t_stop = 0.01
cfl = 0.8

[mesh]
type = "cartesian"
Nx = 24
Ny = 12
Nz = 6
Lx = 2.0
Ly = 1.0
Lz = 0.5

[[periodic]]
zones = ["back", "front"]

[initialize]
type = "constant"
u = [10.0, 0.0, 0.0]
p = "1 atm"
T = "300 K"

[[boundaries]]
name = "left"
type = "inlet_state"
u = [10.0, 0.0, 0.0]
p = "1 atm"
T = "300 K"

[[boundaries]]
name = "right"
type = "outlet_pressure"
p = "1 atm"

[[boundaries]]
name = "bottom"
type = "wall_adiabatic"

[[boundaries]]
name = "top"
type = "symmetry"

[regions]
heater = {type = "sphere", center = [0.6, 0.4, 0.25], radius = 0.15}

[[sources]]
region = "heater"
rhoE = 1e5

[[probes]]
name = "wake"
fields = ["P"]
type = "line"
start = [1.2, 0.1, 0.25]
end = [1.2, 0.9, 0.25]
n_points = 9

[[probes]]
name = "section"
fields = ["P"]
region = {type = "disk", center = [1.6, 0.5, 0.25], normal = [1, 0, 0], radius = 0.2}
""" + PHYSICS


def view(mallard, toml, name):
    """Run mallard check --view on the case and copy its two images to docs/images/guide/<name>-*.svg."""
    work = Path(tempfile.mkdtemp())
    (work / "case.toml").write_text(toml)
    subprocess.run([str(mallard), "check", "case.toml", "--view", "view", "--kokkos-num-threads=2"], cwd=work,
                   check=True, stdout=subprocess.DEVNULL)
    OUT.mkdir(parents=True, exist_ok=True)
    for theme in ("light", "dark"):
        shutil.copyfile(work / "view" / f"case-{theme}.svg", OUT / f"{name}-{theme}.svg")
    shutil.rmtree(work)
    print(f"wrote {name}-{{light,dark}}.svg")


def figures(mallard, mallard3d=None):
    view(Path(mallard).resolve(), DUCT, "case-view")
    if mallard3d:
        view(Path(mallard3d).resolve(), CHANNEL, "case-view-3d")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mallard", required=True, help="2D Mallard executable")
    ap.add_argument("--mallard3d", help="3D Mallard executable")
    args = ap.parse_args()
    figures(args.mallard, args.mallard3d)


if __name__ == "__main__":
    main()
