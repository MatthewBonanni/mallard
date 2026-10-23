# Output and diagnostics

A run writes snapshots of the solution (`[[write_data]]`), restart files, and
diagnostics: wall forces, domain integrals, running statistics and probes.
The log on the terminal reports progress and timing.

## Snapshots: `[[write_data]]`

```toml
[[write_data]]
prefix = "solut/flow"         # directories are created
format = "vtu"                # vtu, hdf5 or restart
time_interval = 0.01          # or interval = 100 (steps)
variables = ["RHO", "U", "P", "T", "VORTICITY"]

[[write_data]]                # several entries write independently
prefix = "solut/wall"
format = "vtu"
interval = 50
geometry = "cylinder"         # the faces of one boundary zone
variables = ["P", "T"]
```

*Used in:* [`sod`](https://matthewbonanni.github.io/mallard/docs/examples/#sod), [`shu_osher`](https://matthewbonanni.github.io/mallard/docs/examples/#shu-osher), [`riemann_2d`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d), [`riemann_2d_quads`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d-quads), [`double_mach`](https://matthewbonanni.github.io/mallard/docs/examples/#double-mach) and 39 more.

| Key | Default | Meaning |
|---|---|---|
| `prefix` | required | Output path prefix |
| `format` | required | `vtu` (ParaView, with a `.pvd` time series), `hdf5` (with XDMF), or `restart` |
| `interval` / `time_interval` | one required | Every this many steps, or this much simulated time (the step is shortened to land on each output time) |
| `variables` | required (`vtu`, `hdf5`) | Variables to write, below; `Y_*` selects every variable with that prefix |
| `geometry` | `"all"` | `vtu`: a boundary zone name writes that zone's faces with the values of their cells (wall pressure, heat flux) |
| `compression` | `"none"` | `"zlib"`: lossless compression of `vtu`, `hdf5` and `restart` files (below) |
| `relative_error` | none (lossless) | `vtu`, `hdf5` with `compression = "zlib"`: lossy, each cell value within this relative error |
| `chunk_cells` | 65536 | Compressed `restart` files: cells per chunk |

**Variables:**

| Group | Names |
|---|---|
| Conservative | `RHO`, `RHOU_X`, `RHOU_Y`, `RHOU_Z`, `RHOE`, `RHOU` (vector) |
| Primitive | `U_X`, `U_Y`, `U_Z`, `U` (vector), `P`, `T`, `H` |
| Derived | `CFL`, `VORTICITY`, `Q` (Q-criterion) |
| TENO | `TENO_SIGMA` (troubled-cell indicator) |
| Mixtures | `Y_<s>`, `X_<s>`, `RHOY_<s>`; viscous: `MU`, `LAMBDA`, `D_<s>` (and `DT_<s>` with `soret`) |
| Chemistry, radiation | `HRR`, `OMEGA_<s>`, `CHEM_COST`, `QRAD` |
| LES | `MU_T`; TFLES: `TF_F`, `TF_E`, `TF_OMEGA`; PaSR: `PASR_KAPPA` |
| Detonations | `P_MAX`, the largest pressure each cell has seen (numerical soot foil) |
| Statistics | `MEAN_<A>`, `COV_<A>_<B>`, `MEAN_*`, `COV_*` |

**VTU or HDF5.** VTU is the simple choice: one file per snapshot (and per rank
in parallel, assembled by a `.pvtu`). `format = "hdf5"` (builds with
`-DMallard_ENABLE_HDF5=ON`) writes one file per snapshot from all ranks with
collective I/O, plus the mesh once, and XDMF indexes that ParaView opens. It
pays off for large runs on parallel file systems; the files do not depend on
the rank count. Read them in Python with `tools/mallard/h5.py`.

**Asynchronous and compressed output.** Writing large snapshots can take longer than many steps. By default (`[output] async = true`) each snapshot is copied out and written on a background thread while the run goes on, with the same files byte for byte as `async = false`; `io_threads` copies and compresses on several threads (GPU runs leave the host cores idle):

```toml
[output]
io_threads = 8              # async = true is the default

[[write_data]]
prefix = "solut/flow"
format = "vtu"
interval = 500
variables = ["RHO", "U", "P", "T"]
compression = "zlib"       # lossless; readable by any ParaView
relative_error = 1e-4       # optional: lossy, |x' - x| <= 1e-4 |x| per value
```

Compressed VTU uses VTK's own zlib compressor and compressed HDF5 the standard shuffle and deflate filters, so ParaView and h5py read them unchanged. VTU and restart compression needs a build with `-DMallard_ENABLE_ZLIB=ON`. `relative_error` rounds each value to the fewest mantissa bits within the bound (the mesh stays exact), which compresses smooth fields several times further; restart files are always lossless. On 4.2M quads on an A100, a VTU snapshot of `RHO`, `U`, `P`, `T` is 474 MB, 164 MB with `zlib` and 59 MB with `relative_error = 1e-4`.

Pitfalls: the run's summary reports the time spent writing; if each file takes longer to write than the steps between files, the run waits for the previous file anyway (write less often, or compress with more `io_threads`). Errors writing a file surface at the next file or at the end of the run, which waits for every file; a run killed outright (e.g. `kill -9`) can leave the file being written incomplete, while the files before it are complete. HDF5 output on several ranks stays synchronous (parallel HDF5 calls MPI only from the main thread).

**Restart files** (`format = "restart"`) are described in
[Initialization and restarts](initialization.md#restarts).

## The log

```toml
[output]
check_interval = 100   # a progress row every 100 steps
```

*Used in:* [`sod`](https://matthewbonanni.github.io/mallard/docs/examples/#sod), [`shu_osher`](https://matthewbonanni.github.io/mallard/docs/examples/#shu-osher), [`riemann_2d`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d), [`riemann_2d_quads`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d-quads), [`double_mach`](https://matthewbonanni.github.io/mallard/docs/examples/#double-mach) and 39 more.

Rank 0 prints a header (version, build, device), the setup time of each
stage, summaries of mesh, physics, numerics, boundaries and output, then one
progress row every `[output] check_interval` steps (default 1):

| Column | Meaning |
|---|---|
| `step`, `t`, `dt` | Step, time and time step |
| `done` | Fraction of the run, by whichever stop condition comes first |
| `wall/step`, `cells/s`, `ETA` | Time-stepping wall time per step, cell updates per second, time remaining |
| `min rho`, `min p`, `max Ma` | Positivity and Mach number at a glance |
| `troubled` | TENO: share of cells selecting stencils (shocks) |

Chemistry adds the share of cells advanced and the largest sub-step count.
Written files appear as rows with their step and time. The run ends with a
summary of wall time (setup, time stepping, diagnostics, output) and
throughput. The same information, in machine-readable form, goes to the
[run log](#the-run-log). Errors go to stderr from any rank. Colors are off when the output
is not a terminal or `NO_COLOR` is set (`CLICOLOR_FORCE=1` forces them, e.g.
under `mpirun`).

## The run log

Next to the human log, rank 0 writes `mallard.jsonl` (`[output] run_log`):
one JSON object per line ([JSON Lines](https://jsonlines.org)), flushed as
it is written, so scripts and dashboards can follow a running case:

```python
import json
records = [json.loads(line) for line in open("mallard.jsonl")]
progress = [r for r in records if r["type"] == "progress"]
```

A run starting from a restart file appends to the log, so a case continued
several times keeps one history, each part led by its own `start` record.
Each record has a `type`:

| `type` | When | Fields |
|---|---|---|
| `start` | After setup | `schema` (`"mallard-run-log"`), `version` (1), `provenance` (below), `placement` (one object per rank, as the [rank report](parallel.md#the-rank-report) finds it: `rank`, `host`, `threads`, `cpus` (the CPU set before Kokkos starts, e.g. `"0-15"`; `null` where unknown), `cpu_numa`, `omp_bind`, and on GPUs `gpu` (PCI address), `gpu_uuid`, `gpu_index`, `gpu_numa`; `null` where unknown or on CPUs), `setup_phases` (seconds of each setup phase: `max` and `mean` over ranks) |
| `progress` | Every progress row | as below |
| `event` | A file written, or a request acted on (the rows between progress rows) | `step`, `t`, `kind` (`vtu`, `hdf5`, `restart`, ...; `checkpoint`, `stop`, `set` and `diverged` from [run control](running.md)), `text` |
| `warning`, `error` | A warning, or the error that ended the run | `message` |
| `end` | The end of the run | `reason`, `step`, `t`, `steps` (this run's), `wall` and its parts `wall_setup`, `wall_stepping`, `wall_diagnostics`, `wall_output` (and `wall_chemistry`), `rejected_steps` (with error control), `cells_per_second` |

Progress records:

| Field | Meaning |
|---|---|
| `step`, `t`, `dt` | Step, time, time step |
| `cfl` | `dt` over the stable time step at CFL 1 |
| `progress` | Fraction of the run, by the first stop condition to be reached |
| `wall` | Seconds since the solver started |
| `wall_per_step`, `cells_per_second` | Time stepping since the previous progress record; `null` before any step |
| `eta` | Seconds remaining, `null` until known |
| `cells`, `ranks` | Global cell count and rank count |
| `memory_host`, `memory_host_peak` | Resident memory of the largest rank, and its peak, in bytes |
| `memory_device`, `memory_device_total` | GPU memory in use (by all processes) on the fullest device, and its size; `null` without a GPU |
| `monitors` | The progress row's diagnostics: `min_rho`, `min_p`, `max_mach`; TENO: `troubled_cells` (fraction); mixtures: `min_T`, `max_T`, `max_sum_Y_error`; chemistry: `reacting_cells` (fraction), `max_chemistry_cost` |
| `integrals` | The latest `[integrals]` (`step`, `t` and the columns of `integrals.csv`) |
| `forces_<zone>` | The latest `[[forces]]` of each zone (`step`, `t` and the columns of its file) |
| `health` | The [run health](#run-health) at this row (`[output] health`) |

Numbers are in SI units (seconds, bytes) and read back exactly; unknown
values are `null`. Within a `version`, records only gain fields; renaming,
removing or changing the meaning of one increments it.

## Provenance

Every output file records where it came from: the provenance block, one
JSON object, is written into restart files (a trailer after the data, also of
compressed files, which older readers skip), VTU, `.pvtu`, `.pvd`, XDMF and probe `.vtp` files (an XML comment
`<!-- provenance {...} -->` after the XML declaration), HDF5 files, probe files included (the
string attribute `provenance` of the root group; a continued run's probe file holds the latest run's), CSV files of forces,
integrals and probes (a line `# provenance {...}` after the header, and
another where a continued run starts appending), and the run log's `start`
record.

| Field | Contents |
|---|---|
| `schema`, `version` | `"mallard-provenance"`, 1 |
| `mallard` | `version` (`git describe`), `git_sha`, `git_dirty` (uncommitted changes in the build); `null` outside a git checkout. Taken at build time |
| `dependencies` | Versions of `compiler`, `kokkos`, `mpi` (the library's own string and the standard), `hdf5`, `cgns`, `zlib`, `nccl`, `gpu_runtime` and `gpu_driver` (CUDA or HIP); `null` for those not built in |
| `build` | `dim`, `precision`, `build_type`, `compiler`, `cxx_flags`, `kokkos` (version), `backend` and `host_backend` (Kokkos execution spaces), `mpi`, `gpu_aware_mpi`, `nccl`, `hdf5`, `kaminpar`, `species_layout` |
| `run` | `start_time` (UTC, ISO 8601), `ranks`, `host`, `device` (rank 0's), `directory`, `hosts` (the number of hosts; where each rank ran is the run log's `placement`) |
| `input` | `file`, `sha256` of its bytes, and its `text` |
| `mesh` | `type`; for mesh files `file` and the `sha256` of its bytes (of a directory: of each file's relative path and bytes, by path; skipped with `[output] hash_mesh = false`) |
| `preset` | The `[numerics] preset` (`null` without one): `name`, and per key it covers its `key`, resolved `value` and `from_preset` (`false` where the input overrides it) |
| `setup` | The resolved setup, defaults included: each section of the log's setup summary (`Mesh`, `Physics`, `Numerics`, `Boundaries`, `Output`) as `[key, value]` pairs |

Check that a file belongs to a given input with `sha256sum input.toml`; read
the block in Python:

```python
import json, h5py
json.loads(h5py.File("solut/flow_000100.h5")["/"].attrs["provenance"][0])
```

or, from any other file, `provenance(path)` of `tools/mallard/vtu.py`.

## Run health

At every progress row Mallard also checks that the run is healthy, and
writes what it finds into the progress record's `health` (and a short
**Health** section at the end of the human log). Everything is computed at
progress rows only, from the state of that step. Between rows the ledger
adds a reduction over the boundary faces (and one over the cells where
there are volume sources) per stage, chemistry two reductions of the
element totals per call and a sub-step count per integrated cell, and one
step per row is timed phase by phase (below); none of these waits for the
device. `[output] health = false` turns
it all off.

| Field | Contents |
|---|---|
| `margins` | The smallest `rho`, `p`, `T` and, for mixtures, mass fraction (`Y_<species>`) over the domain: `value`, cell centroid `x` and `step` |
| `margins_run` | The smallest of each at any progress row so far |
| `activity` | Fractions of cells, over the domain (`all`) and the least and most active rank (`rank_min`, `rank_max`): `troubled_cells` (TENO's troubled-cell indicator), `limited_cells` (MUSCL cells whose limiter is below one in some variable), `upwind_fraction` (the hybrid flux's mean upwind fraction) |
| `chemistry` | Since the previous row: `substeps` and `rejected_substeps` of the Rosenbrock integration, `failures`, `reacting_cells` (fraction of the last call), `wall` (seconds of the slowest rank) and `imbalance` (the slowest rank's chemistry time over the ranks' mean) |
| `timing` | Seconds of each phase of the last sampled step, `all` (mean), `rank_min`, `rank_max` over ranks: `halo`, `reconstruction`, `convective_flux`, `viscous_flux`, `sum_and_sources`, `update` (the time integrator's vector updates), `chemistry`, `time_step` (the CFL time step and its global minimum), `checks`, `ledger`; and per step since the previous row, `output_sync` (the run's own output time) and `output_async` (time of the background writers) |
| `repairs`, `repairs_run` | Each [numerical repair](#numerical-repairs), since the previous row and since the start: `activations`, `cells` (distinct owned cells) and `change`, the integrated change of each quantity it changes |
| `balance`, `balance_run` | The [conservation ledger](#conservation-ledger), since the previous row and since the first row |
| `element_drift` | Mixtures with chemistry: the largest change of a rank's element mass over one chemistry call, beyond the recorded repairs, relative to the rank's mass (round-off) |

The sampled step is the step before each progress row: it runs kernel by
kernel (no CUDA graph) with a fence after each phase, so its phases add up
to its wall time but the halo exchange does not overlap the computation
there.

Mallard warns once per kind when the density or pressure falls below
10<sup>-3</sup> of its domain mean, when a mass fraction is below
-10<sup>-6</sup>, and when a checked balance has an unexplained relative
change above 10<sup>5</sup> machine epsilons.

### Conservation ledger

The ledger is an independent account of the domain totals of mass,
momentum, total energy and, for mixtures, each element's mass. It does not
read the solver's face fluxes or residual: at each progress row it
integrates the totals from the conservative variables, and compares their
change with

- `boundary`: the transfers through the boundary faces, which it evaluates
  itself at every stage from the faces' states (the interior reconstruction
  and the boundary condition's exterior state), normals and areas, with the
  convective and viscous point fluxes of the scheme (wall models included),
- `sources`: gravity, `[source]` fields, the mass flow forcing and the
  sponges, re-evaluated from each stage's state,
- `repairs`: the recorded changes of the [numerical repairs](#numerical-repairs),

each stage weighted by the time integrator's weights b<sub>i</sub> and
committed with its time step once the step is accepted. The fields of each
quantity are `total`, `change`, `boundary`, `sources`, `repairs`,
`residual` (the change none of these explain), `relative` (residual over
the larger of the totals and their absolute integral; for momentum,
sum |&rho;u| + (2 &rho; &rho;E)<sup>1/2</sup>), `checked` and, where the
ledger cannot close a quantity, a `note` saying why. A conservative setup
closes at round-off, in a periodic box as with inflow, outflow and walls;
a missing factor or a wrong sign in the update shows as a residual many
orders larger.

Not closed (`checked = false`): boundary transfers of gas mixtures (their
mass and element balances close where the boundaries are walls, symmetry
planes or periodic), the energy with double flux (not conservative) or
radiation, the radial momentum of axisymmetric runs, element masses with
sponges, and every quantity with `coupling = "simpler"`.

### Numerical repairs

Every place that changes the state outside the discretized equations is
audited:

| Site | What it does | Recorded change |
|---|---|---|
| `composition_clip` | Chemistry: negative mass fractions are clipped to zero and the composition renormalized, before and after a cell's integration (the energy stays exact) | `species_<name>` and `element_<symbol>` masses |
| `energy_reset` | Double flux: after each step, &rho;E is reset to the true equation of state at the pressure of the frozen one | `energy` |

The temperature inversion (Newton from the cell's previous temperature,
safeguarded by bisection within the thermodynamic fits' range) never
changes the state, and a failed chemistry integration ends the run with an
error, so neither is a repair.

## Wall forces: `[[forces]]`

```toml
[[forces]]
zone = "cylinder"
interval = 10        # default 1
# file = "forces_cylinder.csv"
```

*Used in:* [`sphere_mach3`](https://matthewbonanni.github.io/mallard/docs/examples/#sphere-mach3), [`sphere_axisymmetric`](https://matthewbonanni.github.io/mallard/docs/examples/#sphere-axisymmetric), [`cylinder`](https://matthewbonanni.github.io/mallard/docs/examples/#cylinder), [`sphere_re300`](https://matthewbonanni.github.io/mallard/docs/examples/#sphere-re300), [`channel_retau180`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-retau180) and 2 more.

Writes `step, t` and the pressure and viscous force components of the fluid
on the zone, per unit depth in 2D (per radian in axisymmetric runs). Divide by
the dynamic pressure and a reference length (or area) for drag and lift
coefficients. On walls with a [wall model](les.md#wall-models-wall_model)
the viscous force is the modeled stress, and a last column `Q` gives the
modeled heat flux into the fluid.

## Domain integrals: `[integrals]`

```toml
[integrals]
interval = 1
budget = true        # kinetic-energy budget (LES), default false
```

*Used in:* [`taylor_green_3d`](https://matthewbonanni.github.io/mallard/docs/examples/#taylor-green-3d), [`isotropic_turbulence`](https://matthewbonanni.github.io/mallard/docs/examples/#isotropic-turbulence), [`channel_retau180`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-retau180), [`forced_isotropic_turbulence`](https://matthewbonanni.github.io/mallard/docs/examples/#forced-isotropic-turbulence), [`cbc_les`](https://matthewbonanni.github.io/mallard/docs/examples/#cbc-les) and 4 more.

Writes `integrals.csv` with the volume integrals of the kinetic energy,
enstrophy, squared dilatation and pressure dilatation, and of `|u|^2`,
`|omega|^2`, `rho^2`, `T` and `T^2` (from which mean squares and variances
follow). For decaying turbulence (the Taylor–Green
vortex) the dissipation rate is `-dE/dt`. With TENO, velocity gradients come
from the reconstruction polynomials (order-consistent: within 1% of spectral
derivatives of the same field). `budget = true` adds the
[kinetic-energy budget](les.md#low-dissipation-flux-and-the-budget):
molecular, SGS and numerical dissipation.

## Statistics

```toml
[statistics]
fields = ["RHO", "U_X", "U_Y", "U_Z", "P", "T"]
products = ["U_X*U_X", "U_Y*U_Y", "U_Z*U_Z", "U_X*U_Y"]
t_start = 40.0
interval = 10
```

*Used in:* [`cylinder`](https://matthewbonanni.github.io/mallard/docs/examples/#cylinder), [`channel_retau180`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-retau180), [`channel_inflow_retau180`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-inflow-retau180), [`turbulent_inflow_duct`](https://matthewbonanni.github.io/mallard/docs/examples/#turbulent-inflow-duct), [`channel_les`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-les) and 2 more.

`[statistics]` keeps running time averages and covariances per cell, on the
device, without communication: bitwise the same on any number of ranks, and
carried across restarts.

| Key | Default | Meaning |
|---|---|---|
| `fields` | none | `RHO`, `U_X`, `U_Y`, `U_Z`, `P`, `T`, `H`, `Y_<s>` |
| `products` | none | Pairs `"A*B"` whose covariance to keep |
| `t_start` | 0 | Start averaging after this time |
| `interval` | 1 | Sample every this many steps (weighted by the time since the last sample) |
| `reset` | `false` | Start afresh on a restart |

Write them with `variables = ["MEAN_*", "COV_*"]`, and average further over
homogeneous directions with `tools/post/plane_average.py` (e.g. over x and z of a
channel, giving profiles in y; its CSV also feeds
[synthetic inflow](boundary_conditions.md#synthetic-turbulence)).

### Spectra and two-point correlations: `[statistics.spectra]`

```toml
[statistics]
t_start = 120.0
interval = 20

[statistics.spectra]
fields = ["U_X", "U_Y", "U_Z"]   # autospectra and autocorrelations
products = ["U_X*U_Y"]           # cross spectra and correlations
# directions = ["x", "z"]        # default: the mesh's periodic directions
# average = false                # per-sample files instead of time averages
```

*Used in:* the in-situ statistics tests.

On a generated `cartesian` mesh, spectra and two-point correlations along the
homogeneous (periodic, unstretched) directions, at each position of the
others: per y of a channel, or shell spectra of a periodic box. Computed by
FFTs on the device, bitwise the same on any number of ranks.

| Key | Default | Meaning |
|---|---|---|
| `fields`, `products` | none | Variables (as `[statistics]`) and pairs `"A*B"` |
| `directions` | periodic directions | Homogeneous directions |
| `average` | `true` | Time averages, written to `file` with every snapshot and carried by restart files; `false`: one `<file stem>_<step>.csv` per sample |
| `file` | `spectra.csv` | Output CSV (`quantity, a, b, direction, position, <coordinates>, index, abscissa, value`) |
| `interval`, `t_start` | `[statistics]`'s | Sampling |

The file holds plane means, one-sided spectra (sums, not densities: they
add up to the covariance `<a'b'>`), correlations `<a'(x) b'(x + r)>` and, in
a periodic box, shell spectra. `tools/post/insitu_statistics.py` plots them
(`plot`) and recomputes them from restart snapshots (`spectra`, `compare`).

- Values are cell averages: at the grid cutoff a spectrum is lower than one
  of point values by the cell-average factor (sinc^2(k h / 2) per direction,
  0.41 at k = N/2), as `examples/isotropic_turbulence/isotropic_turbulence_stats.py --no-deconvolve`.
- A sample costs about a third of a step of the Re_tau 180 channel (11 ms
  on an A100): sample every 10-20 steps, not every step.
- Correlations are periodic: `R(L - r) = R(-r)`; use `r <= L/2`.

### Phase averages: `[statistics.phase]`

```toml
[statistics.phase]
fields = ["U_X", "U_Y", "P"]
bins = 12
probe = [3.0, 0.0]               # or: period = 30.3 (and t_reference)
variable = "U_Y"
level = 0.0
```

*Used in:* [`cylinder`](https://matthewbonanni.github.io/mallard/docs/examples/#cylinder).

Per-cell means over the samples whose phase in a periodic cycle falls in each
of `bins` bins, output as `PHASE_<bin>_<field>` (`variables = ["PHASE_*"]`).
The phase comes from a fixed `period`, or from the upward crossings of a
level by a variable at a probe point.

| Key | Default | Meaning |
|---|---|---|
| `fields` | none | Variables to average |
| `bins` | none | Phase bins |
| `period`, `t_reference` | none, 0 | Fixed period, phase 0 at `t_reference` |
| `probe`, `variable`, `level` | none, none, 0 | Trigger point, variable and level |
| `interval`, `t_start` | `[statistics]`'s | Sampling |

- The trigger needs two crossings before it bins anything, and is sampled
  only every `interval` steps: keep `interval` well below a period / `bins`.
- A cycle longer than the previous one puts its tail in the last bin.
- `examples/cylinder` phase-averages
  its vortex street this way.

## Probes: `[[probes]]`

A probe samples the solution on a geometry every `interval` steps (or
`time_interval` of simulation time) and writes the samples, reductions over
them, or both. The files are the same bit for bit on any number of ranks.

```toml
[[probes]]
name = "centerline"
variables = ["RHO", "P"]
start = [0.0, 0.01]
end = [1.0, 0.01]
n_points = 200
interval = 10
```

*Used in:* [`cylinder`](https://matthewbonanni.github.io/mallard/docs/examples/#cylinder), [`channel_inflow_retau180`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-inflow-retau180), [`turbulent_inflow_duct`](https://matthewbonanni.github.io/mallard/docs/examples/#turbulent-inflow-duct), [`flame_outflow`](https://matthewbonanni.github.io/mallard/docs/examples/#flame-outflow), [`shock_bubble_3d`](https://matthewbonanni.github.io/mallard/docs/examples/#shock-bubble-3d).

```toml
# Wake survey: high-order point values on an arc, a VTK series for ParaView,
# and the arc's mean and RMS in probe_wake_reductions.csv
[[probes]]
name = "wake"
variables = ["U", "P"]
sampling = "point"
shape = "arc"
center = [0.0, 0.0]
radius = 5.0
angles = [-45.0, 45.0]
n_points = 91
time_interval = 1.0
format = "vtp"
reductions = ["mean", "rms"]
```

| Key | Default | Meaning |
|---|---|---|
| `name` | required | Names the files |
| `variables` | required | `[statistics]` fields; `U` and `Y_*` expand to all components and species |
| `shape` | from the keys | `point`, `points` (`points` or `points_file`), `line`, `polyline`, `circle`, `arc`, `curve` (`x`, `y`, `z` of `s`) |
| `sampling` | `cell` | `cell`: the average of the cell holding each sample; `point`: the reconstruction's value at the sample |
| `interval` / `time_interval` | 1 step | Output times; `time_interval` shortens the step to land on them |
| `format` | `csv` | `csv`, `hdf5` (one extensible file) or `vtp` (one file per output and a `.pvd` series) |
| `reductions` | none | `integral`, `mean`, `favre`, `min`, `max`, `rms` along the geometry |
| `samples` | `true` | `false` writes only the reductions |
| `translate`, `rotate`, `scale` | none | Move the geometry (scale, then rotate about the origin, then translate) |

With TENO, `sampling = "point"` reads each cell's reconstruction polynomial at
the sample, so a probe converges at the scheme's order (fifth order for
`order = 5`) instead of the first order of cell averages at points off the
centroids. The [Sod comparison](numerics.md#reconstruction) on the Numerics
page was read from a line probe. The full key list is in the [input reference](../input.md#probes).

Pitfalls:

- Point values fall back to the linear least-squares reconstruction in cells
  the TENO indicator flags troubled (shocks, and coarse meshes of smooth
  fields), and to the cell average outside the domain.
- Mass fractions are cell averages in both modes; `T` and `H` follow the
  reconstructed `p / rho` at the cell's composition.
- CSV is for small probes: it repeats the coordinates every output. Use `hdf5`
  or `vtp` for thousands of samples.
- Reductions of point sets weigh each point 1 (`integral` is a sum); curves
  integrate with three Gauss points per element between samples.

## Post-processing

The Python scripts in `tools/` (numpy, matplotlib; see
[Tools](https://matthewbonanni.github.io/mallard/docs/tools/)) read Mallard's output directly:

| Script | Use |
|---|---|
| `mallard/vtu.py`, `mallard/h5.py` | Readers for VTU and HDF5 output, for your own scripts (`from mallard.vtu import read_vtu_cells` with `tools/` on `sys.path`) |
| `post/plot_vtu.py`, `post/compare_png.py` | A field of one or several snapshots |
| `post/animate.py` | MP4/GIF animation of a 2D series with numerical schlieren |
| `post/plane_average.py` | Averages over homogeneous directions |
| `post/insitu_statistics.py` | Plots `[statistics.spectra]` output; recomputes it, or phase-bins a probe series, to check the in-situ values |
| `plot_<case>.py` | Each validation case against its reference |

ParaView opens `.pvd` (VTU) and `.xmf` (HDF5) series directly.

## File formats and compatibility

Every file Mallard writes carries a format version. **The promise:** a release
reads the restart and mesh files of every earlier release of the same major
version. In 0.x that means every restart file since 0.1 and every HDF5 mesh since
0.3. A file from a *newer*
Mallard is rejected with a message that gives its version and the versions this
build reads. It is never misread.

| File | Version, where | Versions (written from) | Read by |
|---|---|---|---|
| Restart (`format = "restart"`) | `uint32` after `MALLARD-RESTART` | 1 (0.1), 2: named variables (0.4), 3: `[statistics]` weights (0.6), 4: characteristic boundary states (0.7), 5: `compression = "zlib"` (0.9), 6: `[statistics.spectra]` and `[statistics.phase]` state (0.9) | Mallard: all |
| HDF5 mesh (`mallard mesh convert`) | root attribute `version` | 1 (0.3) | Mallard: all |
| HDF5 output (`format = "hdf5"`) | root attribute `version` of both files | 1 (0.6) | ParaView, `tools/mallard/h5.py` |
| VTU output | field data `MALLARD_FORMAT_VERSION` | none before 0.9, 1 (0.9) | ParaView, `tools/mallard/vtu.py` |
| Run log (`[output] run_log`) | `version` of each `start` record | 1 (0.9) | Your scripts (records only gain fields within a version) |
| Provenance block (every output file) | its `version` | 1 (0.9) | `tools/mallard/vtu.py` (`provenance()`); in restart files a trailer that every restart reader skips, so it changes no restart version |
| Probe output (`[[probes]]` `format = "hdf5"`, `"vtp"`) | root attribute `version` (HDF5), field data `MALLARD_FORMAT_VERSION` (VTP) | 1 (0.9) | Mallard (restarts append to HDF5), ParaView |
| TENO cache (`cache_file`) | magic `MALLARD-TENO-<n>` | - | Rebuilt automatically when it does not match |

- Restart files are written with the **lowest** version that holds their
  contents, so a run without statistics, characteristic boundaries or
  compression still writes version 2, which every release since 0.4 reads.
- A restart file must also match the build's dimension and precision
  (`Mallard_DIM`, `Mallard_USE_DOUBLE`); the message says which differs.
- Output files gain arrays and attributes within a version. A version changes
  only when existing content changes meaning or layout. The Python tools refuse
  versions newer than they know; update `tools/` with Mallard.
- A change to a format adds a version and keeps the reader of every older one,
  with a fixture in `test/data/formats/` that a test reads (see its
  [README](../../test/data/formats/README.md)).

## See also

- [Input reference: `[[write_data]]`](../input.md#write_data), [`[output]`](../input.md#output), [`[[forces]]`](../input.md#forces), [`[integrals]`](../input.md#integrals), [`[statistics]`](../input.md#statistics), [`[statistics.spectra]`](../input.md#statisticsspectra), [`[statistics.phase]`](../input.md#statisticsphase), [`[[probes]]`](../input.md#probes)
