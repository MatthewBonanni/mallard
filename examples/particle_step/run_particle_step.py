"""Runs of particle_step: the flow (flow.toml) to developed shedding, then input.toml from its restart file at
several Stokes numbers, each in its own directory st_<St>/, then the frames of each for the animation.

    python examples/particle_step/run_particle_step.py [--mallard PATH] [--stokes 0 1 4] [--args ...]

Run it in examples/particle_step after generating the mesh (see flow.toml). St = 0 runs tracers; otherwise
the diameter is d = sqrt(18 mu tau / rho_p) for tau = St tau_f, tau_f = 5 H / U0. The particles do not act on
the gas, so every run has the same flow, and only the tracers write its snapshots. Then
animate_particle_step.py render OUT.mp4 st_* and
animate_particle_step.py profiles st_*.
"""
import argparse
import glob
import math
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MU, RHO_P, TAU_F = 1.0e-4, 2000.0, 5.0 / 0.3
D_ST1 = "3.872983346207417e-3"


def particles_input(text, st):
    if st == 0:
        text = re.sub(r'^type = "solid"\n', 'type = "tracer"\n', text, count=1, flags=re.M)
        text = re.sub(r"^# St = 1:.*\n^density = .*\n^diameter = .*\n", "", text, count=1, flags=re.M)
        return re.sub(r"^# Peak particle Reynolds number.*\n(^.*\S.*\n)+\n", "", text, count=1, flags=re.M)
    text = re.sub(r'^\[\[write_data\]\]\nprefix = "./frames/step"\n(^.*\S.*\n)+', "", text, count=1, flags=re.M)
    d = math.sqrt(18 * MU * st * TAU_F / RHO_P)
    text = re.sub(r"^# St = 1:.*$", f"# St = {st:g}: tau = {st:g} x 5 H / U0 = {st * TAU_F:.4g}, "
                  "d = sqrt(18 mu tau / rho_p)", text, count=1, flags=re.M)
    return text.replace(D_ST1, repr(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mallard", default=os.path.join(HERE, "..", "..", "build", "src", "mallard"))
    ap.add_argument("--stokes", type=float, nargs="+", default=[0.0, 1.0, 4.0])
    ap.add_argument("--args", nargs=argparse.REMAINDER, default=[], help="more arguments of every run")
    args = ap.parse_args()
    mallard = os.path.abspath(args.mallard)
    if len(glob.glob("solut/flow_*.restart")) < 2:
        subprocess.run([mallard, "-i", "flow.toml", *args.args], check=True)
    flow = sorted(glob.glob(os.path.abspath("solut/flow_*.restart")))[-1]
    text = open("input.toml").read()
    for st in args.stokes:
        d = f"st_{st:g}"
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "input.toml"), "w").write(particles_input(text, st))
        if not os.path.exists(os.path.join(d, "step.msh")):
            os.symlink(os.path.abspath("step.msh"), os.path.join(d, "step.msh"))
        print(f"St = {st:g} in {d}/", flush=True)
        with open(os.path.join(d, "run.log"), "w") as log:
            subprocess.run([mallard, "-i", "input.toml", "--restart", flow, *args.args], cwd=d, check=True,
                           stdout=log, stderr=subprocess.STDOUT)
        subprocess.run([sys.executable, os.path.join(HERE, "animate_particle_step.py"), "extract", d], check=True)


if __name__ == "__main__":
    main()
