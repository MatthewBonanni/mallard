# Output

A run writes snapshots of the solution, restart files, and the diagnostics it
is asked for: probes, wall forces, domain integrals and statistics. It reports
its progress on the terminal and in a machine-readable run record.

## What a run writes

| Output | File(s) | Turned on by | Details |
|---|---|---|---|
| Solution snapshots | `PREFIX_<n>.vtkhdf` (`<n>`: the step, or with `time_interval` the snapshot's index) and `PREFIX.vtkhdf.series`; VTU: `.vtu`, `.pvtu`, `.pvd` | `[[write_data]]` | [Solution snapshots](#solution-snapshots-write_data) |
| Restart files | `PREFIX_<step>.restart` | `[[write_data]]` with `format = "restart"` | [Initialization and restarts](initialization.md#restarts) |
| Terminal output | standard output (errors: standard error) | always; `[output] check_interval` sets the progress rows | [Terminal output](#terminal-output) |
| Run record | `mallard.jsonl` | on by default; `[output] run_log` | [Run record](#run-record-mallardjsonl) |
| Run health | the run record's `health` and the terminal output's **Health** section | on by default; `[output] health` | [Run health](#run-health) |
| Probes | `probe_<name>.csv` (or `.h5`, `.vtkhdf`, `.pvd`), `probe_<name>_reductions.csv` | `[[probes]]` | [Probes and monitors](probes.md#probes-probes) |
| Wall forces | `forces_<zone>.csv` | `[[forces]]` | [Wall forces](probes.md#wall-forces-forces) |
| Domain integrals | `integrals.csv` | `[integrals]` | [Domain integrals](probes.md#domain-integrals-integrals) |
| Time and phase averages | `MEAN_*`, `COV_*` and `PHASE_*` fields of the snapshots and restart files | `[statistics]`, `[statistics.phase]` | [Statistics](statistics.md) |
| Spectra and two-point correlations | `spectra.csv` | `[statistics.spectra]` | [Spectra](statistics.md#spectra-and-two-point-correlations-statisticsspectra) |

Every file records its [units](#units), its [provenance](#provenance) and a
[format version](#formats-and-compatibility).

## Solution snapshots: `[[write_data]]`

```toml
[[write_data]]
prefix = "solut/flow"         # directories are created
time_interval = 0.01          # or interval = 100 (steps)
variables = ["RHO", "U", "P", "T", "VORTICITY"]

[[write_data]]                # several entries write independently
prefix = "solut/wall"
interval = 50
region = "cylinder"           # a boundary zone, a slice, a volume: any region (below)
variables = ["P", "T"]
```

*Used in:* [`sod`](https://matthewbonanni.github.io/mallard/docs/examples/#sod), [`shu_osher`](https://matthewbonanni.github.io/mallard/docs/examples/#shu-osher), [`riemann_2d`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d), [`riemann_2d_quads`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d-quads), [`double_mach`](https://matthewbonanni.github.io/mallard/docs/examples/#double-mach) and 46 more.

| Key | Default | Meaning |
|---|---|---|
| `prefix` | required | Output path prefix |
| `format` | `"vtkhdf"` | `vtkhdf` (ParaView 5.12+, with a `.vtkhdf.series` index), `vtu` (ParaView, with a `.pvd` time series), `restart`, or the deprecated `hdf5` (with XDMF). Builds without HDF5 write `vtu`, with a warning |
| `series` | `false` | `vtkhdf`: `true` keeps every snapshot in one file, `PREFIX.vtkhdf` |
| `interval` / `time_interval` | one required | Every this many steps, or this much simulated time (the step is shortened to land on each output time) |
| `variables` | required (all but `restart`) | Variables to write, below; `Y_*` selects every variable with that prefix |
| `region` | none (the whole mesh) | `vtkhdf`, `vtu`: write only this [region](geometry.md#regions), inline or by name: a filled region's cells, or a boundary zone, surface, curve, points or cut plane ([below](#snapshots-of-regions)). Replaces `geometry` (deprecated; `mallard input upgrade` rewrites it) |
| `regions` | none | Instead of `region`: a list of regions, each written as its own series, `PREFIX_<name>` (inline ones: `PREFIX_<place in the list>`) |
| `compression` | `"none"` | `"zlib"`: lossless compression of `vtkhdf`, `vtu`, `hdf5` and `restart` files (below) |
| `relative_error` | none (lossless) | `vtkhdf`, `vtu`, `hdf5` with `compression = "zlib"`: lossy, each cell value within this relative error |
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
| [Statistics](statistics.md) | `MEAN_<A>`, `COV_<A>_<B>`, `MEAN_*`, `COV_*`; `PHASE_<bin>_<A>`, `PHASE_*` |

**VTKHDF, the default.** `format = "vtkhdf"` writes VTK's HDF5 format, which
ParaView 5.12 and later open directly: open `PREFIX.vtkhdf.series` for the time
series. Every rank writes its part of one file per snapshot with collective I/O
(parallel HDF5 on several ranks), each array in global cell order, so the files
are the same on any number of ranks. Cells touching a curved wall are written as
quadratic Lagrange cells, so ParaView shows the curved geometry. With `series = true` the mesh is stored
once and every snapshot goes into `PREFIX.vtkhdf`; a restarted run drops the
snapshots after its restart time and continues the file. Read the files in
Python with `tools/mallard/vtkhdf.py` (or `read_vtu_cells`, which reads every
format), or with h5py alone.

Builds without HDF5 write VTU instead (and say so): one file per snapshot, and
per rank in parallel, assembled by a `.pvtu`, with a `.pvd` series. Set
`format = "vtu"` to choose it explicitly. `format = "hdf5"` (a mesh file, one
HDF5 file of fields per snapshot and XDMF indexes) is deprecated and will be
removed in 0.10.0; `mallard input upgrade` rewrites it to `"vtkhdf"`.

**Asynchronous and compressed output.** Writing large snapshots can take longer than many steps. By default (`[output] async = true`) each snapshot is copied out and written on a background thread while the run goes on, with the same files byte for byte as `async = false`; `io_threads` copies and compresses on several threads (GPU runs leave the host cores idle):

```toml
[output]
io_threads = 8              # async = true is the default

[[write_data]]
prefix = "solut/flow"
interval = 500
variables = ["RHO", "U", "P", "T"]
compression = "zlib"       # lossless; readable by any ParaView
relative_error = 1e-4       # optional: lossy, |x' - x| <= 1e-4 |x| per value
```

Compressed VTU uses VTK's own zlib compressor and compressed VTKHDF and HDF5 the standard shuffle and deflate filters, so ParaView and h5py read them unchanged. VTU and restart compression needs a build with `-DMallard_ENABLE_ZLIB=ON`. `relative_error` rounds each value to the fewest mantissa bits within the bound (the mesh stays exact), which compresses smooth fields several times further; restart files are always lossless. On 4.2M quads on an A100, a VTU snapshot of `RHO`, `U`, `P`, `T` is 474 MB, 164 MB with `zlib` and 59 MB with `relative_error = 1e-4`.

Pitfalls: the run's summary reports the time spent writing; if each file takes longer to write than the steps between files, the run waits for the previous file anyway (write less often, or compress with more `io_threads`). Errors writing a file surface at the next file or at the end of the run, which waits for every file; a run killed outright (e.g. `kill -9`) can leave the file being written incomplete, while the files before it are complete. VTKHDF and HDF5 output on several ranks stays synchronous (parallel HDF5 calls MPI only from the main thread).

**Restart files** (`format = "restart"`) are described in
[Initialization and restarts](initialization.md#restarts).

### Snapshots of regions

Large runs need the whole field rarely and a few planes, a near field or the
walls often. `region` writes just that, at its own interval, with the same keys
as a whole-mesh snapshot:

```toml
[regions]
near_field = {type = "sphere", center = [0, 0, 0], radius = 0.05}
x1 = {type = "cut_plane", point = [0.01, 0, 0], normal = [1, 0, 0]}
x2 = {type = "cut_plane", point = [0.02, 0, 0], normal = [1, 0, 0]}
x3 = {type = "cut_plane", point = [0.04, 0, 0], normal = [1, 0, 0]}

[[write_data]]                # the cells of a volume
prefix = "solut/near"
time_interval = 1e-4
region = "near_field"
variables = ["RHO", "U", "P"]

[[write_data]]                # three slices: solut/x_x1.vtkhdf, solut/x_x2.vtkhdf, solut/x_x3.vtkhdf
prefix = "solut/x"
time_interval = 2e-5
regions = ["x1", "x2", "x3"]
variables = ["U", "P", "MASS_FLUX"]

[[write_data]]                # a boundary zone, by its name
prefix = "solut/wall"
interval = 50
region = "cylinder"
variables = ["P", "T"]
```

![A disk's cells with their REGION_FRACTION, and their pressure with three slices across the field, on a triangle mesh](../images/guide/region_snapshots-light.svg#gh-light-mode-only)
![A disk's cells with their REGION_FRACTION, and their pressure with three slices across the field, on a triangle mesh](../images/guide/region_snapshots-dark.svg#gh-dark-mode-only)

What a snapshot holds depends on the region's dimension:

| Region | Written | Values |
|---|---|---|
| A volume (an area in 2D): any region of [`[regions]`](geometry.md#regions), e.g. `sphere`, `box`, booleans | The cells it covers (cell fraction above 0), as a mesh of its own: only their nodes, curved cells as quadratic ones | The cells' fields, and `REGION_FRACTION`, each cell's [volume fraction](geometry.md#regions-on-a-mesh) of the region |
| `"domain"` | The whole mesh, as without `region` | The cells' fields |
| A boundary zone, by its name | Its faces: polygons (3D), a polyline (2D) | Point values at the face centroids, as a [boundary probe](probes.md#boundary-data) with `sampling = "point"` writes them, wall quantities included; other cell fields (`P_MAX`, `HRR`) from each face's cell |
| A surface, curve or points of [`[regions]`](geometry.md#surfaces-and-curves-on-a-mesh) (`filled = false`, `boundary`, flat disks, circles, ...) | Its elements on the mesh: polygons (3D), segments (2D), points | Point values at each element's center, with its normal and measure; other cell fields from the element's cell |
| A cut plane or other [probe type as a region](geometry.md#probe-types-as-regions) | The exact cut of the mesh, or the resampled surface, curve or points | Point values of the reconstruction, as a [probe](probes.md) of it with `sampling = "point"`; fluxes (`MASS_FLUX`, ...) through surfaces and 2D curves; other cell fields from the cell of each sample |

A volume is written by every rank in global order, like the whole mesh;
`compression`, `relative_error` and `series` apply as they do there. A boundary
zone, cut plane, surface, curve or points is written as a probe writes it, by
the same code: `vtkhdf` writes one PolyData file `PREFIX.vtkhdf` of every
snapshot (`series` is always on), `vtu` one VTK PolyData file per snapshot,
`PREFIX_<step>.vtp`, and the `PREFIX.pvd` series. Its `variables` are sampled
as [`[[probes]]`](probes.md) sample them (`U` writes `U_X`, `U_Y`, `U_Z`);
any other variable of a snapshot (`HRR`, `P_MAX`, `MEAN_P`, ...) takes the value
of the cell holding each sample. Either kind is
the same on any number of ranks, takes [units](#units) and descriptions, and
continues its series on restart. Every region is also written at the end of the
run, as the whole mesh is.

Pitfalls:

- A slice is sampled where the plane cuts each cell, from the reconstruction:
  the values are those of a [probe](probes.md#cut-planes-and-surfaces) of the plane, not
  a cell average. Across a shock, use a volume region around it instead.
- On a surface, curve or points, `RHO`, the primitives and `Y_<s>` are point
  values, while cell fields such as `HRR`, `VORTICITY` or `MEAN_*` are the
  averages of the cells the samples lie in: first order across the cell.
- A volume region much smaller than a cell can cover no cell, which is an error.
- A boundary zone's samples lie at the face centroids, on the wall: `P` is the
  reconstruction's wall pressure, not the first cell's average. Cell fields
  such as `P_MAX` are still the cell's.
- Regions are written as `vtkhdf` or `vtu` only, not as the deprecated `hdf5`
  (XDMF) format.
- `compression` on a region of lower dimension, alone in its entry, is an
  unused key: those files are small and written uncompressed.

## Run progress

### Terminal output

```toml
[output]
check_interval = 100   # a progress row every 100 steps
```

*Used in:* [`sod`](https://matthewbonanni.github.io/mallard/docs/examples/#sod), [`shu_osher`](https://matthewbonanni.github.io/mallard/docs/examples/#shu-osher), [`riemann_2d`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d), [`riemann_2d_quads`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d-quads), [`double_mach`](https://matthewbonanni.github.io/mallard/docs/examples/#double-mach) and 46 more.

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
[run record](#run-record-mallardjsonl). Errors go to stderr from any rank. Colors are off when the output
is not a terminal or `NO_COLOR` is set (`CLICOLOR_FORCE=1` forces them, e.g.
under `mpirun`).

### Run record: `mallard.jsonl`

Next to the terminal output, rank 0 writes `mallard.jsonl` (`[output] run_log`):
one JSON object per line ([JSON Lines](https://jsonlines.org)), flushed as
it is written, so scripts and dashboards can follow a running case:

```python
import json
records = [json.loads(line) for line in open("mallard.jsonl")]
progress = [r for r in records if r["type"] == "progress"]
```

A run starting from a restart file appends to the record, so a case continued
several times keeps one history, each part led by its own `start` record.
Each record has a `type`:

| `type` | When | Fields |
|---|---|---|
| `start` | After setup | `schema` (`"mallard-run-log"`), `version` (1), `provenance` (below), `placement` (one object per rank, as the [rank report](parallel.md#the-rank-report) finds it: `rank`, `host`, `threads`, `cpus` (the CPU set before Kokkos starts, e.g. `"0-15"`; `null` where unknown), `cpu_numa`, `omp_bind`, and on GPUs `gpu` (PCI address), `gpu_uuid`, `gpu_index`, `gpu_numa`; `null` where unknown or on CPUs), `setup_phases` (seconds of each setup phase: `max` and `mean` over ranks) |
| `progress` | Every progress row | as below |
| `event` | A file written, or a request acted on (the rows between progress rows) | `step`, `t`, `kind` (`vtkhdf`, `surface`, `vtu`, `hdf5`, `restart`, ...; `checkpoint`, `stop`, `set` and `diverged` from [run control](running.md)), `text` |
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

## Run health

At every progress row Mallard also checks that the run is healthy, and
writes what it finds into the progress record's `health` (and a short
**Health** section at the end of the terminal output). Everything is computed at
progress rows only, from the state of that step. Between rows the ledger
adds a reduction over the boundary faces (and one over the cells where
there are volume sources) per stage, chemistry two reductions of the
element totals per call and a sub-step count per integrated cell; none of
these waits for the device. `[output] health = false` turns
it all off.

| Field | Contents |
|---|---|
| `margins` | The smallest `rho`, `p`, `T` and, for mixtures, mass fraction (`Y_<species>`) over the domain: `value`, cell centroid `x` and `step` |
| `margins_run` | The smallest of each at any progress row so far |
| `activity` | Fractions of cells, over the domain (`all`) and the least and most active rank (`rank_min`, `rank_max`): `troubled_cells` (TENO's troubled-cell indicator), `limited_cells` (MUSCL cells whose limiter is below one in some variable), `upwind_fraction` (the hybrid flux's mean upwind fraction) |
| `chemistry` | Since the previous row: `substeps` and `rejected_substeps` of the Rosenbrock integration, `failures`, `reacting_cells` (fraction of the last call), `wall` (seconds of the slowest rank) and `imbalance` (the slowest rank's chemistry time over the ranks' mean) |
| `timing` | Per step since the previous row, `all` (mean), `rank_min`, `rank_max` over ranks: `output_sync` (the run's own output time) and `output_async` (time of the background writers). The step's own wall time and throughput are the row's `wall_per_step` and `cells_per_second` |
| `repairs`, `repairs_run` | Each [numerical repair](#numerical-repairs), since the previous row and since the start: `activations`, `cells` (distinct owned cells) and `change`, the integrated change of each quantity it changes |
| `balance`, `balance_run` | The [conservation ledger](#conservation-ledger), since the previous row and since the first row |
| `element_drift` | Mixtures with chemistry: the largest change of a rank's element mass over one chemistry call, beyond the recorded repairs, relative to the rank's mass (round-off) |

A breakdown of the step by phase is a profiling question, not a health check:
timing phases needs a fence after each one, which a CUDA-graph step does not
have; use a profiler (Nsight Systems, Kokkos Tools) for it.

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

## In every output file

### Units

Every value Mallard writes is in SI units (the mesh-quality angle of
`mallard check` excepted, in degrees), and every output file says which,
as metadata in the way its format's tools expect; field names never change.
Restart files are the exception: they hold the conservative variables, in SI
units by definition, for Mallard to read back.

| File | Where the units are |
|---|---|
| VTKHDF, HDF5 (snapshots, their mesh file, probes) | CF-convention string attributes of each dataset: `units` ([UDUNITS](https://docs.unidata.ucar.edu/udunits/current/) syntax, `1` for dimensionless) and `long_name`; `standard_name` only where a CF standard name fits, which no field does yet (`air_pressure` and the like are for atmospheres) |
| XDMF (`.xmf`) | `<Information Name="units" Value="..."/>` and `<Information Name="long_name" .../>` in each `<Attribute>` |
| VTU, VTP (each piece of a PVTU) | Field data: the string arrays `MALLARD_FIELD_NAMES`, `MALLARD_FIELD_UNITS` and `MALLARD_FIELD_LONG_NAMES`, entry *i* of each for the same array (VTK has no units attribute) |
| CSV (forces, integrals, probes) | A line `# units: ...` after the header, one entry per column |
| Spectra CSV (`[statistics.spectra]`) | A last column `units`: the units of each row's `value` |

```python
import h5py
f = h5py.File("solut/flow_000100.h5")
f["fields/P"].attrs["units"], f["fields/P"].attrs["long_name"]   # ('kg m-1 s-2', 'pressure')

import sys; sys.path.insert(0, "tools")
from mallard.vtu import units, long_names        # any output file: .vtkhdf, .vtu, .pvtu, .vtp, .h5, .xmf, .csv
units("solut/flow_000100.vtkhdf")["U"]             # 'm s-1'
from mallard.tables import read_csv               # CSV columns and their units
columns, column_units = read_csv("forces_cylinder.csv")
```

These are the attributes xarray and netCDF tools look for, readable as
they are. Units are written in SI base units (`kg m-1 s-2`, not `Pa`), so that
derived fields compose: `MEAN_<A>` and `PHASE_<bin>_<A>` have the units of
`A`, `COV_<A>_<B>` their product (`COV_U_X_U_Y`: `m2 s-2`), and `SOURCE_<A>`
those of `A` per second.

| Field | Units | Field | Units |
|---|---|---|---|
| `RHO`, `RHOY_<s>` | `kg m-3` | `MU`, `MU_T` | `kg m-1 s-1` |
| `RHOU` | `kg m-2 s-1` | `LAMBDA` | `kg m s-3 K-1` |
| `RHOE`, `P`, `P_MAX` | `kg m-1 s-2` | `D_<s>` | `m2 s-1` |
| `U` | `m s-1` | `DT_<s>` | `kg m-1 s-1` |
| `T`, `T_SEED` | `K` | `HRR`, `QRAD` | `kg m-1 s-3` (W/m³) |
| `H` | `m2 s-2` (J/kg) | `OMEGA_<s>` | `kg m-3 s-1` |
| `VORTICITY` | `s-1` | `CHEM_H` | `s` |
| `Q` | `s-2` | `CFL`, `TENO_SIGMA`, `Y_<s>`, `X_<s>`, `CHEM_COST`, `TF_F`, `TF_E`, `TF_OMEGA`, `PASR_KAPPA` | `1` |
| `NON_ORTHOGONALITY` (`mallard check`) | `degree` | `SKEWNESS`, `ASPECT_RATIO`, `VOLUME_RATIO` | `1` |
| `MASS_FLUX`, `SPECIES_FLUX_<s>` (probes) | `kg m-2 s-1` | `ENTHALPY_FLUX`, `Q_W` | `kg s-3` (W/m²) |
| `MOMENTUM_FLUX`, `TRACTION_PRESSURE`, `TRACTION_VISCOUS`, `TAU_W` | `kg m-1 s-2` | `U_TAU` | `m s-1` |
| `CF`, `CP`, `Y_PLUS` | `1` | `REGION_FRACTION` (regions) | `1` |

CSV columns: `t` in `s`, coordinates, `s` and probe `length` in `m`, `step`,
`point`, `face`, `cell`, `count` and the normal `nx`, `ny`, `nz` `1`; the
`area` of a sample of a boundary, cut plane or surface, and its reductions'
`area`, in `m2` (in 2D, where these are curves, `m`, the reductions' column
`length`); forces in `N` (`kg m s-2`), per unit depth in 2D
(`kg s-2`), and the wall model's `Q` in `W`; domain integrals carry the
volume's `m3` (per unit depth in 2D: `m2`); probe reductions have their
variable's units, `integral` times those of the measure (`m` along a curve,
`m2` over a 3D boundary, cut plane or surface: `integral_MASS_FLUX` in `kg s-1`; over a region its
`volume`, `m3`, or `area`, `m2`, per unit depth in 2D). In axisymmetric runs,
forces, integrals and region probes are per radian of the revolved domain, and their units
say so: `kg m s-2 rad-1`, `kg m2 s-2 rad-1`; so are boundary probes' `area`
(`m2 rad-1`), region probes' `volume` (`m3 rad-1`) and `integral` reductions. Spectra rows: a `mean` has its
field's units, `spectrum`, `correlation` and `shell` (sums, not densities)
the product of their pair's (`U_X*U_Y`: `m2 s-2`); the `abscissa` is a
wavenumber in `m-1` (spectra, shells) or a separation in `m` (correlations).
Files written before Mallard 0.9 have no units; the readers return them empty.

### Provenance

Every output file records where it came from: the provenance block, one
JSON object, is written into restart files (a trailer after the data, also of
compressed files, which older readers skip), VTU, `.pvtu`, `.pvd`, XDMF and probe `.vtp` files (an XML comment
`<!-- provenance {...} -->` after the XML declaration), VTKHDF and HDF5 files, probe files included (the
string attribute `provenance` of the root group; a continued run's probe file holds the latest run's), CSV files of forces,
integrals and probes (a line `# provenance {...}` after the header and its
[units line](#units), and another where a continued run starts appending), and
the run record's `start` record.

| Field | Contents |
|---|---|
| `schema`, `version` | `"mallard-provenance"`, 1 |
| `mallard` | `version` (`git describe`), `git_sha`, `git_dirty` (uncommitted changes in the build); `null` outside a git checkout. Taken at build time |
| `dependencies` | Versions of `compiler`, `kokkos`, `mpi` (the library's own string and the standard), `hdf5`, `cgns`, `zlib`, `nccl`, `gpu_runtime` and `gpu_driver` (CUDA or HIP); `null` for those not built in |
| `build` | `dim`, `precision`, `build_type`, `compiler`, `cxx_flags`, `kokkos` (version), `backend` and `host_backend` (Kokkos execution spaces), `mpi`, `gpu_aware_mpi`, `nccl`, `hdf5`, `kaminpar`, `species_layout` |
| `run` | `start_time` (UTC, ISO 8601), `ranks`, `host`, `device` (rank 0's), `directory`, `hosts` (the number of hosts; where each rank ran is the run record's `placement`) |
| `input` | `file`, `sha256` of its bytes, and its `text` |
| `mesh` | `type`; for mesh files `file` and the `sha256` of its bytes (of a directory: of each file's relative path and bytes, by path; skipped with `[output] hash_mesh = false`) |
| `preset` | The `[numerics] preset` (`null` without one): `name`, and per key it covers its `key`, resolved `value` and `from_preset` (`false` where the input overrides it) |
| `setup` | The resolved setup, defaults included: each section of the terminal output's setup summary (`Mesh`, `Physics`, `Numerics`, `Boundaries`, `Output`) as `[key, value]` pairs |

Check that a file belongs to a given input with `sha256sum input.toml`; read
the block in Python:

```python
import json, h5py
json.loads(h5py.File("solut/flow_000100.h5")["/"].attrs["provenance"][0])
```

or, from any other file, `provenance(path)` of `tools/mallard/vtu.py`.

### Formats and compatibility

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
| VTKHDF output (`format = "vtkhdf"`) | root attribute `version` (VTK's own in `/VTKHDF`'s `Version`: 2.0) | 1 (0.9) | ParaView 5.12+, `tools/mallard/vtkhdf.py` |
| HDF5 output (`format = "hdf5"`, deprecated) | root attribute `version` of both files | 1 (0.6) | ParaView, `tools/mallard/h5.py` |
| VTU output | field data `MALLARD_FORMAT_VERSION` | none before 0.9, 1 (0.9) | ParaView, `tools/mallard/vtu.py` |
| Run record (`mallard.jsonl`, `[output] run_log`) | `version` of each `start` record | 1 (0.9) | Your scripts (records only gain fields within a version) |
| Provenance block (every output file) | its `version` | 1 (0.9) | `tools/mallard/vtu.py` (`provenance()`); in restart files a trailer that every restart reader skips, so it changes no restart version |
| Probe output (`[[probes]]` `format = "hdf5"`, `"vtkhdf"`, `"vtp"`) | root attribute `version` (HDF5, VTKHDF), field data `MALLARD_FORMAT_VERSION` (VTP) | 1 (0.9) | Mallard (restarts append to HDF5), ParaView |
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

## Post-processing

The Python scripts in `tools/` (numpy, matplotlib; see
[Tools](https://matthewbonanni.github.io/mallard/docs/tools/)) read Mallard's output directly:

| Script | Use |
|---|---|
| `mallard/vtu.py`, `mallard/vtkhdf.py`, `mallard/h5.py`, `mallard/tables.py` | Readers for VTKHDF, VTU, HDF5 and CSV output and the [units](#units) of each file, for your own scripts (`from mallard.vtu import read_vtu_cells` with `tools/` on `sys.path`) |
| `post/plot_vtu.py`, `post/compare_png.py` | A field of one or several snapshots |
| `post/animate.py` | MP4/GIF animation of a 2D series with numerical schlieren |
| `post/plane_average.py` | Averages over homogeneous directions |
| `post/insitu_statistics.py` | Plots `[statistics.spectra]` output; recomputes it, or phase-bins a probe series, to check the in-situ values |
| `plot_<case>.py` | Each validation case against its reference |

ParaView opens `.vtkhdf.series` and `.vtkhdf` (VTKHDF), `.pvd` (VTU) and `.xmf` (HDF5) series directly. `read_vtu_cells` reads any snapshot file, and `mallard.vtu.snapshot("solut/flow_000100")` finds it whichever format the build wrote.

## See also

- [Input reference: `[[write_data]]`](../input.md#write_data), [`[output]`](../input.md#output)
- [Probes and monitors](probes.md): probes, wall forces, domain integrals
- [Statistics](statistics.md): time averages, spectra and correlations, phase averages
- [Initialization and restarts](initialization.md): restart files and their format
- [Running](running.md): run control of a running case
