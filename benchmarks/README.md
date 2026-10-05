# Benchmarks

Performance cases with recorded baselines (see `docs/design/chemistry.md`,
milestone 10 as implemented, for the numbers and the hardware).

## `chemistry/`: chemistry throughput

`MallardReactor -i <case>.toml` samples states along a constant-volume
ignition (fresh, igniting and burnt gas), replicates them over many cells and
advances them with the solver's chemistry kernels over splitting steps of
1e-8 s (detonations) and 1e-6 s (flames); it prints cells per second and a
histogram of the sub-steps per cell (bins 1, 2, 3-4, 5-8, ...) and writes a
CSV.

| Case | Mechanism | Species | Reactions |
|---|---|---|---|
| `h2o2.toml` | H2/air, h2o2 | 10 | 29 |
| `gri30.toml` | CH4/air, GRI-Mech 3.0 | 53 | 325 |
| `ndodecane.toml` | n-dodecane/air, Wang et al. (2014) | 100 | 432 |
| `nhexane.toml` | n-hexane/air, NUIG (Zhang et al. 2015); copy it first with `python tools/cantera_mechanisms.py` | 1268 | 5336 |

`[benchmark]` keys: `cells`, `samples` (trajectory states), `dt` (list),
`repeats` (the best is reported, after a warm-up call), `igniting_fraction`
(e.g. 0.01: one igniting cell in a hundred, the others fresh or burnt, to
measure the imbalance between cells), and the execution: `lanes` (vector
lanes per cell, 1 for one thread per cell, 0 automatic), `threads` (team
threads per cell for every cell; 0, the default: one, with several for the
cells whose last call took 16 or more sub-steps), `shared` (-1 automatic, 0: work memory in global memory)
and `bin_by_cost` (GPUs: order the queue by the cells' last cost, most
expensive first; default true, ignored on CPUs).
`[chemistry] sparse = true/false` chooses the LU (by default sparse from 30 species
when it pays: GRI-3.0 and larger). V3 with the large mechanism: `python tools/v3_check.py
<build>/src/MallardReactor` against `chemistry/nhexane_ignition.csv`.

## `perf/`: performance set

A fixed set for tracking performance across commits, defined in
`perf/suite.json`, with baselines per hardware in `perf/baselines.csv`
(cells per second; higher is better):

| Benchmark | Build | Case |
|---|---|---|
| `teno5_2d` | 2D | 2D Riemann problem (configuration 3) on 1024 x 1024 quads, TENO5, 300 steps |
| `muscl_2d` | 2D | the same with MUSCL |
| `teno5_3d` | 3D | Taylor-Green vortex (Navier-Stokes) on 96^3 hexes, TENO5, 100 steps |
| `h2o2@dt=...`, `gri30@dt=...` | 2D | `chemistry/h2o2.toml` and `chemistry/gri30.toml` at each `dt` |

The solver cases report the throughput over the time stepping (the run's
summary), the chemistry cases `MallardReactor`'s cells per second.

```bash
python tools/perf_suite.py run --build-2d build --build-3d build3d --out perf.json --csv perf.csv
python tools/perf_suite.py compare perf.json            # exit 1 on a regression
python tools/perf_suite.py update-baselines perf.json   # record new baselines
```

`run` runs each case once to warm up, then `--repeats` times (default 3),
and records every run, the best and the spread ((best - worst) / best).
MPI builds need a launcher, e.g. `--launcher "mpirun -n 1"`. The hardware is
the GPU name from `nvidia-smi`, else the CPU model (`--hardware` overrides it).
`compare` reports each benchmark's change from its baseline for that hardware
and fails when one is slower by more than both `--threshold` (default 5%) and
its spread. Benchmarks without a baseline are reported as `new`. After a
deliberate performance change, update the baselines in the same pull request.

## `solver/`: full solver

`reactive_shock_tube_2d.toml`: the reactive shock tube (V6) on 960,000 cells
for 400 steps; the run's summary gives the time per cell and step and the
chemistry's share.
