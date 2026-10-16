# Premixed flame

A flat premixed flame burns into its fresh mixture at the laminar flame
speed S_L, set by the balance of chemistry and molecular transport. This
tutorial computes freely propagating hydrogen-air flames with detailed
chemistry and mixture-averaged transport, first at stoichiometry and then
from lean to rich, and compares their flame speeds and structure with
Cantera's one-dimensional `FreeFlame` solver, which uses the same mechanism
and transport model.

You will use: gas mixtures from a Cantera mechanism, finite-rate chemistry,
initial conditions as expressions of x, an inflow with a composition, a
pressure outlet, and a Python script that writes Mallard input files. Files:
[`tutorials/premixed_flame/`](../../tutorials/premixed_flame). Setup:
[Tutorials](index.md).

## 1. Geometry and the reference flame

Mallard solves the flame in its own frame: fresh H2/air at 300 K and 1 atm
enters a strip on the left at the flame speed and burnt gas leaves on the
right, so a flame at S_L stays in place. The strip runs from 6 thermal
thicknesses δ upstream of the flame to 9 downstream, where
δ = (T_b − T_u) / max dT/dx is the thermal thickness of Cantera's flame.

[`flame_setup.py`](../../tutorials/premixed_flame/flame_setup.py) computes
Cantera's flame and writes the run around it:

```bash
mkdir -p run/premixed_flame && cd run/premixed_flame
python3 $MALLARD/tutorials/premixed_flame/flame_setup.py 1.0 .
```

```text
phi = 1: S_L = 2.33237 m/s, T_b = 2384.3 K, delta = 0.3301 mm; 300 cells over 4.9521 mm, t_stop = 0.5662 ms
```

It writes `reference.csv` (Cantera's profiles of velocity, temperature, heat
release and the H2, O2, H2O and OH mass fractions), `input.toml` and
`mesh.sh`. The input for φ = 1 is also in the repository,
[`tutorials/premixed_flame/input.toml`](../../tutorials/premixed_flame/input.toml).

## 2. Mesh

[`strip.geo`](../../tutorials/premixed_flame/strip.geo) is a rectangle of
`n` quadrilaterals along x and one across, ten times longer across than along
so the faces across the strip do not limit the time step, with the boundary
zones `inlet`, `outlet` and `sides`. `mesh.sh` runs it with the length and
cell count of this flame, 20 cells per δ:

```bash
sh mesh.sh    # gmsh .../strip.geo -2 -setnumber L 4.952096e-03 -setnumber n 300 -o strip.msh
```

![The strip of 300 cells with its boundary zones, and the initial temperature against Cantera's flame](../images/tutorials/premixed_flame_setup-light.svg#gh-light-mode-only)
![The strip of 300 cells with its boundary zones, and the initial temperature against Cantera's flame](../images/tutorials/premixed_flame_setup-dark.svg#gh-dark-mode-only)

## 3. Input file

| Table | In this case |
|---|---|
| `[run]` | Four flame times δ / S_L, at CFL 1 |
| `[mesh]` | `strip.msh` |
| `[initialize]` | `analytical`: temperature, velocity and mole fractions blend from the fresh to the burnt gas of Cantera's flame along c(x) = (1 + tanh(2 (x − x_f) / δ)) / 2, with N2 as the `balance` species; the mole fractions are a subtable, `[initialize.X]`, one expression per species |
| `[[boundaries]]` | `inlet`: `upt` at S_L, 300 K, 1 atm and the fresh composition `X`; `outlet`: `p_out` at 1 atm; `sides`: `symmetry`, which makes the strip one-dimensional |
| `[numerics]` | HLLC, SSPRK3, MUSCL |
| `[physics]` | `navier_stokes`, `gas = "mixture"`, `mechanism = "../../mechanisms/h2o2.yaml"` (9 species, 29 reactions; the path is relative to the run directory), `transport = "mixture_averaged"` |
| `[chemistry]` | Strang splitting with the Rosenbrock integrator, `rtol = 1e-6`, `atol = 1e-10` |
| `[[write_data]]` | Every 0.1 flame time: density, velocity, pressure, temperature, heat release, mass fractions (`Y_*`) and production rates (`OMEGA_*`) |

The tanh profile is only a start: the flame builds its own structure within
a flame time. Starting from Cantera's solution instead (as
`examples/premixed_flame` does, through a restart file) shortens that
transient but needs a restart file written by a script; expressions keep
this input self-contained.

## 4. Run

The strip is small, so one thread is enough (73,000 steps, about 15 minutes
on one core; the chemistry takes 60% of the time):

```bash
$MALLARD/build/src/Mallard -i input.toml --kokkos-num-threads=1
```

On two MPI ranks each takes half the strip, with the same results bit for
bit:

```bash
mpirun -n 2 $MALLARD/build/src/Mallard -i input.toml
```

The progress rows add, for mixtures, the minimum and maximum temperature,
and with chemistry the share of cells whose reactors were integrated in the
last half step and the most sub-steps any cell took.

## 5. Monitor

In a steady flame the fuel consumed per unit area, divided by the density
and the fuel mass fraction of the fresh gas, is the flame speed, whatever the
frame. This **consumption speed** needs only one snapshot, so it can follow
the run:

```bash
python3 $MALLARD/tools/flame_speed.py . 2.33237
```

prints it for every snapshot written so far (and the displacement speed, from
the motion of the flame). `plot_flame.py` draws it:

```bash
python3 $MALLARD/tutorials/premixed_flame/plot_flame.py results . --out .
```

![Consumption speed over time: from the tanh start it overshoots by 2% and settles on Cantera's after two flame times](../images/tutorials/premixed_flame_history-light.svg#gh-light-mode-only)
![Consumption speed over time: from the tanh start it overshoots by 2% and settles on Cantera's after two flame times](../images/tutorials/premixed_flame_history-dark.svg#gh-dark-mode-only)

The flame settles after about two flame times; the comparison averages over
the last one.

## 6. Post-process and compare

`plot_flame.py results` also prints the flame speed over the last flame time
and plots the final profiles against Cantera's, aligned at the maximum of
dT/dx:

```text
phi = 1: consumption speed 2.3296 m/s over the last flame time, Cantera 2.3324 (-0.12%)
```

![Temperature, heat release, and the H2, O2, H2O and OH mass fractions across the flame against Cantera's](../images/tutorials/premixed_flame_profiles-light.svg#gh-light-mode-only)
![Temperature, heat release, and the H2, O2, H2O and OH mass fractions across the flame against Cantera's](../images/tutorials/premixed_flame_profiles-dark.svg#gh-dark-mode-only)

### From lean to rich

`flame_setup.py` writes the same run for any equivalence ratio. Lean flames
are slower and thicker, and their transient from the tanh start is longer:
give them more flame times.

```bash
cd ..
for phi in 0.6 0.8 1.2 1.4; do
  times=$(case $phi in 0.6) echo 10;; 0.8) echo 8;; *) echo 4;; esac)
  python3 $MALLARD/tutorials/premixed_flame/flame_setup.py $phi phi$phi --flame-times $times
  (cd phi$phi && sh mesh.sh && $MALLARD/build/src/Mallard -i input.toml --kokkos-num-threads=1 > run.log)
done
python3 $MALLARD/tutorials/premixed_flame/plot_flame.py results premixed_flame phi0.6 phi0.8 phi1.2 phi1.4 --out .
```

(the runs are independent: start them side by side on a multi-core machine).
With several runs, `plot_flame.py results` compares their flame speeds:

![Flame speed against equivalence ratio, Mallard against Cantera, and the error](../images/tutorials/premixed_flame_speed-light.svg#gh-light-mode-only)
![Flame speed against equivalence ratio, Mallard against Cantera, and the error](../images/tutorials/premixed_flame_speed-dark.svg#gh-dark-mode-only)

| φ | Cantera S_L (m/s) | Mallard S_c (m/s) | Error |
|---|---|---|---|
| 0.6 | 0.8083 | 0.8265 | +2.3% (still falling: run stopped at 4.6 of 10 flame times) |
| 0.8 | 1.6568 | 1.6602 | +0.21% |
| 1.0 | 2.3324 | 2.3296 | −0.12% |
| 1.2 | 2.7814 | 2.7832 | +0.07% |
| 1.4 | 3.0346 | 3.0308 | −0.13% |

## 7. Things to try

- **Resolution.** `--cells-per-delta 10` or `40` in `flame_setup.py`, and
  compare the flame speeds.
- **Transport.** `transport = "unity_lewis"` in `[physics]` against Cantera's
  `unity-Lewis-number` model (change `transport_model` in
  `flame_setup.py`): the structure of the lean flames changes most, since
  hydrogen diffuses much faster than heat.
- **Soret effect.** `soret = true` in `[physics]` adds thermal diffusion of
  the light species (Cantera: `flame.soret_enabled = True`).
- **Two dimensions.** `examples/flame_2d` wrinkles a lean flame in a
  periodic channel and shows what differential diffusion does to it.
