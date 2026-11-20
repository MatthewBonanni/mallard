"""The example images of `mallard post render` in the output guide, light and dark: the density and
numerical schlieren of the 2D Riemann problem on 256^2 quadrilaterals, and a slice of a 3D blast.

    python tools/docs/render.py --mallard build/src/mallard [--mallard3d build-3d/src/mallard]

Mallard draws them itself; this script runs the cases, derives SCHLIEREN with mallard post derive and
writes docs/images/guide/render-{density,schlieren,slice}-{light,dark}.png with mallard post render.
"""
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from figstyle import OUT

ROOT = Path(__file__).resolve().parents[2]
THREADS = "--kokkos-num-threads=8"


def riemann():
    """examples/riemann_2d_quads on 256^2 cells, writing only the final state."""
    text = (ROOT / "examples" / "riemann_2d_quads" / "input.toml").read_text()
    text = text.replace("Nx = 400", "Nx = 256").replace("Ny = 400", "Ny = 256")
    head, _, _ = text.partition("[[write_data]]")
    return head + '[[write_data]]\nprefix = "solut/riemann"\ntime_interval = 0.8\nfields = ["RHO", "U_X", "U_Y", "P"]\n'


BLAST = """[run]
t_stop = 0.12
cfl = 0.8

[mesh]
type = "cartesian"
Nx = 64
Ny = 48
Nz = 32
Lx = 2.0
Ly = 1.5
Lz = 1.0

[initialize]
type = "analytical"
rho = "(x - 0.7)^2 + (y - 0.75)^2 + (z - 0.5)^2 < 0.04 ? 2.0 : 1.0"
u = [0.0, 0.0, 0.0]
p = "(x - 0.7)^2 + (y - 0.75)^2 + (z - 0.5)^2 < 0.04 ? 10.0 : 1.0"

[[periodic]]
zones = ["left", "right"]

[[periodic]]
zones = ["bottom", "top"]

[[periodic]]
zones = ["back", "front"]

[numerics]
riemann_solver = "HLLC"
time_integrator = "SSPRK3"

[numerics.face_reconstruction]
type = "MUSCL"

[physics]
type = "euler"
units = "nondimensional"
gamma = 1.4
p_ref = 1.0
T_ref = 1.0
rho_ref = 1.0

[[write_data]]
prefix = "solut/blast"
time_interval = 0.12
fields = ["RHO", "U", "P"]
"""


def run(cmd, cwd):
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True, stdout=subprocess.DEVNULL)


def render(mallard, cwd, name, *args):
    for theme in ("light", "dark"):
        out = OUT / f"render-{name}-{theme}.png"
        run([mallard, "post", "render", *args, "--latest", "--size", "960", "--theme", theme, "-o", out], cwd)
    print(f"wrote render-{name}-{{light,dark}}.png")


def figures(mallard, mallard3d=None):
    OUT.mkdir(parents=True, exist_ok=True)
    mallard = Path(mallard).resolve()
    work = Path(tempfile.mkdtemp())
    (work / "input.toml").write_text(riemann())
    run([mallard, "-i", "input.toml", THREADS], work)
    run([mallard, "post", "derive", "solut/riemann.vtkhdf", "--latest", "--fields", "SCHLIEREN", THREADS], work)
    render(mallard, work, "density", "solut/riemann.vtkhdf", "--field", "RHO",
           "--title", "2D Riemann problem, configuration 3, 256 x 256 cells")
    render(mallard, work, "schlieren", "derived/derived.vtkhdf", "--field", "SCHLIEREN",
           "--title", "Numerical schlieren, from mallard post derive")
    shutil.rmtree(work)
    if mallard3d:
        mallard3d = Path(mallard3d).resolve()
        work = Path(tempfile.mkdtemp())
        (work / "input.toml").write_text(BLAST)
        run([mallard3d, "-i", "input.toml", THREADS], work)
        render(mallard3d, work, "slice", "solut/blast.vtkhdf", "--field", "P", "--slice", "z=0.5", "--mesh",
               "--title", "3D blast in a box: pressure on the slice z = 0.5")
        shutil.rmtree(work)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mallard", required=True, help="2D Mallard executable")
    ap.add_argument("--mallard3d", help="3D Mallard executable")
    args = ap.parse_args()
    figures(args.mallard, args.mallard3d)


if __name__ == "__main__":
    main()
