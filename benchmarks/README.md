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
| `flame_3d` | 3D | reacting LES in the Volvo combustor's configuration without the bluff body: premixed propane-air (two-step mechanism, mixture-averaged transport, TFLES, Sigma), 96 x 48 x 48 hexes, MUSCL, hybrid flux, Strang-split chemistry, 100 steps |
| `air_3d` | 3D | the same flow of air (single gas): the reference of the mixture's cost per cell and step |
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
the GPU name from `nvidia-smi` or `amd-smi`, else the CPU model (`--hardware` overrides it).
`compare` reports each benchmark's change from its baseline for that hardware
and fails when one is slower by more than both `--threshold` (default 5%) and
its spread. Benchmarks without a baseline are reported as `new`. After a
deliberate performance change, update the baselines in the same pull request.

## `solver/`: full solver

`reactive_shock_tube_2d.toml`: the reactive shock tube (V6) on 960,000 cells
for 400 steps; the run's summary gives the time per cell and step and the
chemistry's share.

## AMD MI355X against the A100

One AMD Instinct MI355X (ROCm 10.0, Kokkos HIP, gfx950) against one
A100-SXM4-80GB (CUDA), same commit, double precision, OpenMP host backend.
The 2D Riemann cases are `examples/riemann_2d_quads` on 1000 x 1000 quads
for 500 steps, the 3D explosion `examples/explosion_3d` on 100^3 hexahedra
for 100 steps; times per step from the run's summary. Chemistry: cells per
second at dt = 1e-8 / 1e-6 s, at the defaults (a cell per thread for h2o2, a
warp per cell with wide teams for the others; a warp is 32 lanes on the
A100 and a 64-lane wavefront on the MI355X).

| Case | A100 | MI355X | Speedup |
|---|---|---|---|
| 2D Riemann, TENO5, 1M quads | 18.4 ms | 5.36 ms | 3.4x |
| 2D Riemann, MUSCL, 1M quads | 3.86 ms | 1.54 ms | 2.5x |
| 3D explosion, TENO5, 1M hexahedra | 392 ms | 193 ms | 2.0x |
| `solver/reactive_shock_tube_2d.toml` | 67.9 ms | 26.3 ms | 2.6x |
| `perf/` `teno5_3d` (cells/s) | 2.64M | 5.35M | 2.0x |
| h2o2 | 6.59M / 2.63M | 12.7M / 3.59M | 1.9x / 1.4x |
| GRI-3.0 | 530k / 535k | 786k / 750k | 1.5x / 1.4x |
| n-dodecane | 310k / 49.0k | 409k / 54.2k | 1.3x / 1.1x |

Two MI355X with GPU-aware MPI (Open MPI over UCX with ROCm) run the 2D TENO5
case in 4.03 ms per step (1.33x one GPU; the A100 pair gives 1.7x at this
size).

Notes:

- HIP compiles a kernel without launch bounds for 1024-thread blocks, which
  caps it at 128 registers per thread. The register-heavy kernels (TENO,
  MUSCL gradients and limiter, fluxes, species reconstruction, one-thread
  chemistry) launch with 256-thread bounds on HIP (`HeavyRange`), which
  allows 512: 2D TENO5 went from 20.6 to 5.36 ms per step, the 3D case from
  241 to 193 ms, h2o2 from 9.2M to 12.7M cells per second.
- The chemistry teams ask HIP for 2 waves per SIMD (`MIN_TEAMS`): 1 was
  6-27% slower for GRI-3.0 and n-dodecane, 4 no faster than 2. 64-lane
  teams beat 32-lane ones by 23-24% (GRI-3.0) and 26% / 70% (n-dodecane).
- The n-hexane case was not run on the MI355X.
- With automatic NUMA balancing on, the ROCm driver stalled the GPU for
  seconds at a time (TENO's setup took minutes); the runs above bind each
  process to its GPU's NUMA node (see the README).
