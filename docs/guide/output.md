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
| Run report | `report.html` | on by default; `[output] report` | [Run report](#run-report-reporthtml) |
| Divergence report | `divergence_report.txt`, `.json`, `divergence_report_neighborhood_<step>.vtkhdf` | a run that diverges; `[output] divergence_report` | [The divergence report](running.md#the-divergence-report) |
| Run health | the run record's `health` and the terminal output's **Health** section | on by default; `[output] health` | [Run health](#run-health) |
| Probes | `probe_<name>.csv` (or `.h5`, `.vtkhdf`, `.pvd`), `probe_<name>_reductions.csv` | `[[probes]]` | [Probes and monitors](probes.md#probes-probes) |
| Wall forces | `forces_<zone>.csv` | `[[forces]]` | [Wall forces](probes.md#wall-forces-forces) |
| Domain integrals | `integrals.csv` | `[integrals]` | [Domain integrals](probes.md#domain-integrals-integrals) |
| Time and phase averages | `MEAN_*`, `COV_*` and `PHASE_*` fields of the snapshots and restart files; probe columns `MEAN_<variable>` with `average = true` | `[statistics]`, `[statistics.phase]`, `[[probes]]` | [Statistics](statistics.md), [Probes](probes.md#time-averages) |
| Mesh and stencil fields | `PREFIX_000000.vtkhdf`, `PREFIX_stencils.h5`; in a run, fields of the snapshots and a stencil file | `mallard check --mesh-fields`; `[[write_data]]` fields, `[output] write_stencils` | [Mesh and stencil fields](#mesh-and-stencil-fields) |
| Spectra and two-point correlations | `spectra.csv` | `[statistics.spectra]` | [Spectra](statistics.md#spectra-and-two-point-correlations-statisticsspectra) |
| Time spectra of probes | `probe_<name>_spectra.csv`, `probe_<name>_reductions_spectra.csv` | `[[probes]]` with `spectra` | [Time spectra](probes.md#time-spectra) |

Every file records its [units](#units), the [run block](#run-block) of the run
that wrote it, each snapshot's [snapshot metadata](#snapshot-metadata) (step,
time, why it was written, by which run) and a
[format version](#formats-and-compatibility). File names are only names:
Mallard's readers take step, time and run from what the files record.

## Solution snapshots: `[[write_data]]`

```toml
[[write_data]]
prefix = "solut/flow"         # directories are created
time_interval = 0.01          # or interval = 100 (steps)
fields = ["RHO", "U", "P", "T", "VORTICITY"]

[[write_data]]                # several entries write independently
prefix = "solut/wall"
interval = 50
region = "cylinder"           # a boundary zone, a slice, a volume: any region (below)
fields = ["P", "T"]
```

*Used in:* [`sod`](https://matthewbonanni.github.io/mallard/docs/examples/#sod), [`shu_osher`](https://matthewbonanni.github.io/mallard/docs/examples/#shu-osher), [`riemann_2d`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d), [`riemann_2d_quads`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d-quads), [`double_mach`](https://matthewbonanni.github.io/mallard/docs/examples/#double-mach) and 48 more.

| Key | Default | Meaning |
|---|---|---|
| `prefix` | required | Output path prefix |
| `format` | `"vtkhdf"` | `vtkhdf` (ParaView 5.12+, with a `.vtkhdf.series` index), `vtu` (ParaView, with a `.pvd` time series), `xdmf` (XDMF indexes over HDF5 files) or `restart`; see [which format](#which-format). Builds without HDF5 write `vtu`, with a warning |
| `series` | `false` | `vtkhdf`: `true` keeps every snapshot in one file, `PREFIX.vtkhdf` |
| `interval` / `time_interval` | one required | Every this many steps, or this much simulated time (the step is shortened to land on each output time) |
| `fields` | required (all but `restart`) | Fields to write, below (`mallard list fields input.toml` lists those of a case); `Y_*` selects every field with that prefix |
| `region` | none (the whole mesh) | `vtkhdf`, `vtu`: write only this [region](geometry.md#regions), inline or by name: a filled region's cells, or a boundary zone, surface, curve, points or cut plane ([below](#snapshots-of-regions)). Replaces `geometry` (removed in 0.10.0; `mallard input upgrade` rewrites it) |
| `regions` | none | Instead of `region`: a list of regions, each written as its own series, `PREFIX_<name>` (inline ones: `PREFIX_<place in the list>`) |
| `compression` | `"none"` | `"zlib"`: lossless compression of `vtkhdf`, `vtu`, `xdmf` and `restart` files (below) |
| `relative_error` | none (lossless) | `vtkhdf`, `vtu`, `xdmf` with `compression = "zlib"`: lossy, each cell value within this relative error |
| `chunk_cells` | 65536 | Compressed `restart` files: cells per chunk |

**Fields:** `mallard list fields input.toml` lists exactly those of a case, and
`mallard list fields` all of them with their units and what each needs
([Command line](cli.md#mallard-list-fields)).

| Group | Names |
|---|---|
| Conservative | `RHO`, `RHOU_X`, `RHOU_Y`, `RHOU_Z`, `RHOE`, `RHOU` (vector) |
| Primitive | `U_X`, `U_Y`, `U_Z`, `U` (vector), `P`, `T`, `H` |
| Derived | `CFL`, `VORTICITY`, `MACH`, `Q_CRITERION`, `LAMBDA2`, `SCHLIEREN` ([below](#derived-fields)) |
| TENO | `TENO_SIGMA` (troubled-cell indicator) |
| Mixtures | `Y_<s>`, `X_<s>`, `RHOY_<s>`; viscous: `MU`, `LAMBDA`, `D_<s>` (and `DT_<s>` with `soret`) |
| Chemistry, radiation | `HRR`, `OMEGA_<s>`, `CHEM_COST`, `QRAD` |
| LES | `MU_T`; TFLES: `TF_F`, `TF_E`, `TF_OMEGA`; PaSR: `PASR_KAPPA` |
| Detonations | `P_MAX`, the largest pressure each cell has seen (numerical soot foil) |
| [Mesh and stencils](#mesh-and-stencil-fields) | `NON_ORTHOGONALITY`, `SKEWNESS`, `ASPECT_RATIO`, `VOLUME_RATIO`; `STENCIL_SIZE`, `STENCIL_SECTORS`, `STENCIL_DEGREE`, `STENCIL_CONDITION`, `STENCIL_WALL`, `STENCIL_SEAM`, `STENCIL_PARTITION` |
| [Statistics](statistics.md) | `MEAN_<A>`, `COV_<A>_<B>`, `MEAN_*`, `COV_*`; `PHASE_<bin>_<A>`, `PHASE_*` |

### Derived fields

Fields that show the flow's structure, computed from the state being written
(with the same cell gradients as `VORTICITY` and the `[integrals]` monitor:
the TENO reconstruction polynomials, else least squares), so they cost nothing
in steps that write none:

```toml
[[write_data]]
prefix = "solut/jet"
interval = 500
fields = ["RHO", "U", "P", "MACH", "Q_CRITERION", "LAMBDA2", "SCHLIEREN"]
```

| Name | Units | Definition |
|---|---|---|
| `MACH` | `1` | \|u\| / c, with c = √(γ p / ρ) and, for mixtures, each cell's frozen γ |
| `Q_CRITERION` | `s-2` | Q = (‖Ω‖² − ‖S‖²) / 2 ([Hunt, Wray & Moin 1988](../references.md#hunt-wray-moin-1988)): positive where rotation dominates strain, i.e. in vortices. Before 0.10.0 it was called `Q`, which is now an error that names `Q_CRITERION` |
| `LAMBDA2` | `s-2` | λ₂, the middle eigenvalue of S² + Ω² ([Jeong & Hussain 1995](../references.md#jeong-hussain-1995)): negative in vortex cores |
| `SCHLIEREN` | `kg m-4` | \|∇ρ\|, a numerical schlieren image of shocks, contact surfaces and mixing layers |

S and Ω are the symmetric and antisymmetric parts of the velocity
gradient ∂u_i/∂x_j. A 2D run treats its flow as a 3D one
with no third velocity component and no variation along z, so that Q and
λ₂ mean what they do in 3D; axisymmetric runs add the hoop strain
u_r / r to the gradient. In 2D, λ₂ = −Q for an incompressible flow,
so `LAMBDA2` adds little to `Q_CRITERION` there.

Pitfalls: both vortex criteria are squares of gradients, so their magnitude
scales with the flow's rate squared; pick isosurface levels relative to a
reference rate (e.g. Q = 0.1 (U/D)²) rather than absolute values. On coarse
meshes the gradients, and hence the criteria, are smoothed. `SCHLIEREN` spans
orders of magnitude between shocks and the rest of the flow; a logarithmic
color map or exp(−k \|∇ρ\| / max \|∇ρ\|) in
ParaView's calculator shows both.

### Which format

| `format` | Opens in | Files |
|---|---|---|
| `vtkhdf` (default) | ParaView 5.12+, PyVista, h5py | One file per snapshot, or one for the series (`series = true`) |
| `xdmf` | VisIt, meshio, ParaView of any version ([open by a path with a directory](#xdmf-in-paraview)); h5py | The mesh once, one HDF5 file of fields per snapshot, and XDMF indexes |
| `vtu` | Anything with a VTK XML reader (e.g. Tecplot); builds without HDF5 | One file per rank per snapshot, with a `.pvtu` and a `.pvd` series |
| `restart` | Mallard, to continue or remap a run | One file per snapshot |

`vtkhdf` and `xdmf` need a build with HDF5 (parallel HDF5 on several ranks).
Before 0.10.0, `xdmf` was called `format = "hdf5"`, which is now an error since
VTKHDF is HDF5 too; `mallard input upgrade` rewrites it to `"vtkhdf"`.

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

**Restarted runs continue the series.** A run resumed from a restart file
numbers its next snapshot after the snapshots up to the restart time, which the
restart file counts, and keeps those entries of the `.vtkhdf.series`, `.pvd` or
`.xmf` index; this holds for every `time_interval` writer, regions included
(`interval` writers are numbered by step). Resuming from an earlier restart file
than the last one rewinds: the entries after the restart time are dropped and
their snapshots are overwritten as the run reaches them again. If the index is
missing, the run says so, still never overwrites an earlier snapshot, and lists
only the snapshots it writes. Restart files older than this count number from
the index, as before.

Builds without HDF5 write VTU instead (and say so): one file per snapshot, and
per rank in parallel, assembled by a `.pvtu`, with a `.pvd` series. Set
`format = "vtu"` to choose it explicitly. `format = "xdmf"` writes the mesh
once (`PREFIX_mesh.h5`), one HDF5 file of fields per snapshot and XDMF
indexes (`PREFIX.xmf` for the series), all in global cell order; read it in
Python with `tools/mallard/h5.py`.

<a id="xdmf-in-paraview"></a>**XDMF in ParaView.** The `.xmf` files name their
HDF5 files relative to themselves, as XDMF allows. ParaView's XDMF reader
(ParaView 6.0.1, Xdmf3) gets this wrong when it is given a bare file name, as in
`pvpython script.py flow.xmf` or `OpenDataFile("flow.xmf")`: it fails to open
`flow.xmfflow_mesh.h5` and aborts with a misleading error about the number of
dimensions. Give it a path with a directory in it: `./flow.xmf`,
`solut/flow.xmf` or an absolute path. The ParaView GUI passes absolute paths, so
files opened from it are not affected. VTKHDF does not have this problem.
Mallard does not write absolute paths into the `.xmf` files, because then a run
directory could no longer be moved or copied.

**Asynchronous and compressed output.** Writing large snapshots can take longer than many steps. By default (`[output] async = true`) each snapshot is copied out and written on a background thread while the run goes on, with the same files byte for byte as `async = false`; `io_threads` copies and compresses on several threads (GPU runs leave the host cores idle):

```toml
[output]
io_threads = 8              # async = true is the default

[[write_data]]
prefix = "solut/flow"
interval = 500
fields = ["RHO", "U", "P", "T"]
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
fields = ["RHO", "U", "P"]

[[write_data]]                # three slices: solut/x_x1.vtkhdf, solut/x_x2.vtkhdf, solut/x_x3.vtkhdf
prefix = "solut/x"
time_interval = 2e-5
regions = ["x1", "x2", "x3"]
fields = ["U", "P", "MASS_FLUX"]

[[write_data]]                # a boundary zone, by its name
prefix = "solut/wall"
interval = 50
region = "cylinder"
fields = ["P", "T"]
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
`PREFIX_<step>.vtp`, and the `PREFIX.pvd` series. Its `fields` are sampled
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
- Regions are written as `vtkhdf` or `vtu` only, not as `xdmf`.
- `compression` on a region of lower dimension, alone in its entry, is an
  unused key: those files are small and written uncompressed.

## Mesh and stencil fields

Where a mesh is poor, and which cells each cell's reconstruction reads, as
fields to view in ParaView and a file of every stencil. Use them when a run
diverges or loses accuracy near a cell, before a long run on a new mesh, and to
see what a wall, a periodic seam or a partition does to the stencils.

```sh
mallard check input.toml --mesh-fields mesh                    # mesh_000000.vtkhdf: quality and stencil fields
mallard check input.toml --mesh-fields mesh --cells 120,4031   # and those cells' stencils: fields and mesh_stencils.h5
mallard check input.toml --mesh-fields mesh --region '{type = "disk", center = [0.5, 0], radius = 0.05}'
mallard check input.toml --mesh-fields mesh --stencils all --weights   # every cell's stencils, with their weights
mallard info mesh.msh --mesh-fields mesh                       # the quality fields of a mesh file alone
```

`--mesh-fields PREFIX` writes `PREFIX_000000.vtkhdf` (`.vtu` in builds without
HDF5), one value per cell of each field below, the
[resolution](input_files.md#resolution) fields of the case, and with `--cells`
or `--region` a field `STENCIL_<id>` per selected cell (the first 64): 1 on the
cells of its central stencil only, 2 of a sector stencil only, 3 of both, 4 on
the cell itself. `--cells` takes global cell ids (from 0, as `mallard check`
reports cells), `--region` a [region](geometry.md#regions), inline or by name,
whose cells to select. With one of `--stencils all`, `--cells` or `--region`, it
also writes those cells' stencils to `PREFIX_stencils.h5` (builds with HDF5).
The stencils are those the case's run builds: its reconstruction, after the
partition, with the mirror images of its boundaries and the reduced degree of
`curved_wall_degree`. A run writes the same: the fields with `[[write_data]]`,
and every cell's stencils with `[output] write_stencils`:

```toml
[output]
write_stencils = "stencils.h5"   # every cell's stencils, once, at the setup

[[write_data]]
prefix = "solut/flow"
interval = 100
fields = ["RHO", "P", "NON_ORTHOGONALITY", "STENCIL_CONDITION"]
```

| Field | Units | Meaning |
|---|---|---|
| `NON_ORTHOGONALITY` | `degree` | The largest over the cell's faces ([Mesh quality](meshes.md#mesh-quality)) |
| `SKEWNESS`, `VOLUME_RATIO` | `1` | The largest over the cell's faces |
| `ASPECT_RATIO` | `1` | The cell's |
| `STENCIL_SIZE` | `1` | Members of the central stencil: the cell itself, its mirror and periodic images included |
| `STENCIL_SECTORS` | `1` | Sector stencils kept (TENO; 0 otherwise) |
| `STENCIL_DEGREE` | `1` | Degree of the central reconstruction polynomial: the scheme's, lower where `curved_wall_degree` reduces it; 1 with MUSCL, 0 first order |
| `STENCIL_CONDITION` | `1` | 2-norm condition number of the central least-squares fit (TENO) or of the gradient fit (MUSCL): how much the fit amplifies rounding and noise in the cell values. 0 at first order, which fits nothing |
| `STENCIL_WALL`, `STENCIL_SEAM`, `STENCIL_PARTITION` | `1` | 1 where a stencil reaches a wall, a periodic image, or a cell of another rank |

![Non-orthogonality, the condition number of the central fit, and the central stencil of one triangle next to a symmetry boundary (with its five mirror images, hatched) on a mesh of quadrilaterals and triangles, from mallard check --mesh-fields](../images/guide/mesh_fields-light.svg#gh-light-mode-only)
![Non-orthogonality, the condition number of the central fit, and the central stencil of one triangle next to a symmetry boundary (with its five mirror images, hatched) on a mesh of quadrilaterals and triangles, from mallard check --mesh-fields](../images/guide/mesh_fields-dark.svg#gh-dark-mode-only)

![Non-orthogonality on the surface cells of a mesh of hexahedra and prisms, and the central stencil of one cell](../images/guide/mesh_fields_3d-light.svg#gh-light-mode-only)
![Non-orthogonality on the surface cells of a mesh of hexahedra and prisms, and the central stencil of one cell](../images/guide/mesh_fields_3d-dark.svg#gh-dark-mode-only)

**The stencil file** (HDF5) lists, for each selected cell in global order, its
stencils' members, compressed-row style: target `t`'s members are rows
`offsets[t]` to `offsets[t + 1]` of `/stencils/members`.

| Dataset | Per | Holds |
|---|---|---|
| `/stencils/cells` | target | Global id of the target cell |
| `/stencils/offsets` | target, plus one | Its first member row |
| `/stencils/degree`, `condition` | target | As the fields `STENCIL_DEGREE` and `STENCIL_CONDITION` |
| `/stencils/sizes` | target × (1 + faces) | Cells of the central stencil, then of the sector stencil of each face (0 for faces it lacks) |
| `/stencils/reach` | target | Bits: 1 a wall, 2 a periodic image |
| `/stencils/members/cell` | member | Global id of the member cell |
| `/stencils/members/stencil` | member | 0 the central stencil, 1 + k the sector stencil of face k (TENO) |
| `/stencils/members/role` | member | Bits: 1 a mirror image across a boundary face (MUSCL: the boundary's ghost or image value), 2 a periodic image |
| `/stencils/members/face_cell`, `face` | member | The boundary face of a mirror image: the global id of its cell (-1 for none) and its place among that cell's faces (255 for none) |
| `/stencils/members/weights` | member × coefficients | With `--weights`: the member's row of the central or sector fit's pseudo-inverse (TENO), its contribution to each polynomial coefficient (a sector fit's quadratic ones, then zeros), or its weight in each gradient component (MUSCL), each times the member's value minus the cell's (a mirror's: its ghost or image value). First order has none |
| `/partition/rank`, `member_rank` | target, member | The rank that owns each target and member |

Root attributes: `format` (`"mallard-stencils"`), `version` (1), `dimension`,
`scheme` (`TENO`, `MUSCL` or `first_order`), `n_cells` of the mesh,
`weights_width` (coefficients per weight row, 0 without `--weights`) and the
[run block](#run-block) (`mallard_run`). Every dataset but `/partition` is the
same whatever the number of ranks. `tools/mallard/stencils.py` reads it:

```python
import sys; sys.path.insert(0, "tools")
from mallard.stencils import MIRROR, read_stencils
s = read_stencils("mesh_stencils.h5")
st = s.stencil(120)                          # cell 120's members, by name
st["cell"][st["stencil"] == 0]               # its central stencil
st["cell"][(st["role"] & MIRROR) != 0]       # the cells mirrored across a boundary
s.sectors(120)                               # {face: its sector stencil's cells}
```

Pitfalls: a member's `cell` is the cell its value comes from, so a mirror
image or a periodic image has the id of a cell already in the stencil, or far
from it; `role` and `face_cell` say which. The file of every cell is large:
about 1.5 kB per cell in 2D at fifth order, 8 kB with `--weights`, and more in 3D; select cells
with `--cells` or `--region` on large meshes. `STENCIL_PARTITION` and `/partition`
show the partition of the check's or run's ranks: `mpirun -n 8 mallard check
input.toml --mesh-fields mesh` shows that of 8 ranks.

## Run progress

### Terminal output

```toml
[output]
check_interval = 100   # a progress row every 100 steps
```

*Used in:* [`sod`](https://matthewbonanni.github.io/mallard/docs/examples/#sod), [`shu_osher`](https://matthewbonanni.github.io/mallard/docs/examples/#shu-osher), [`riemann_2d`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d), [`riemann_2d_quads`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d-quads), [`double_mach`](https://matthewbonanni.github.io/mallard/docs/examples/#double-mach) and 48 more.

Rank 0 prints a header (version, build, device), the setup time of each
stage, summaries of mesh, physics (with the case's units, `physics.units`),
numerics, boundaries and output, then one progress row every
`[output] check_interval` steps (default 1). A note above the table says what
the columns are, and under the column names a row gives their units:

```text
       step           t        dt   done  wall/step  throughput        ETA     min rho       min p   max Ma  troubled  written
          -           s         s      %         ms     cells/s    h:mm:ss      kg/m^3          Pa        -         %  files
          0  0.0000e+00  8.78e-04    0.0          -           -          -   1.400e+00   1.000e+00    1.827      0.00  1 vtu, 1 restart
         10  8.6159e-03  8.53e-04   16.7       37.9    1.90e+05    0:00:02   1.399e+00   9.991e-01    1.881      5.72  2 vtu
         20  1.7029e-02  8.28e-04   33.3       34.2    2.10e+05    0:00:02   1.399e+00   9.990e-01    1.882      5.89  3 vtu
         30  2.4894e-02  8.08e-04   50.0       40.5    1.78e+05    0:00:02   1.399e+00   9.989e-01    1.878      6.19  2 vtu, 1 restart
```

| Column | Units | Meaning |
|---|---|---|
| `step`, `t`, `dt` | -, s, s | Step, simulated time and time step |
| `done` | % | Of the run, by whichever stop condition comes first |
| `wall/step`, `throughput`, `ETA` | ms, cells/s, h:mm:ss | Time-stepping wall time per step and cell updates per second since the previous row, wall time to the stop |
| `min rho`, `min p`, `max Ma` | kg/m^3, Pa, - | Positivity and Mach number at a glance, over the cells |
| `troubled` | % | TENO: share of cells selecting stencils (shocks) |
| `min T`, `max T`, `sum Y-1` | K, K, - | Mixtures: temperature range, and the largest \|sum of the mass fractions - 1\| |
| `reacting`, `chem/cell` | %, steps | Chemistry: share of cells advanced, most integrator steps of a cell |
| `written` | files | Files written since the previous row, by kind (`2 vtu, 1 restart`) |

Every column keeps one unit, so a column reads straight down. In a
nondimensional case (`physics.units = "nondimensional"`, [Units](input_files.md#units))
the units row shows `-` for the simulated quantities. The files themselves
are named in the [run record](#run-record-mallardjsonl) and the summary;
`mallard -i input.toml --verbose` adds a line for each between the rows.
Requests acted on during the run (a checkpoint, a stop, a setting changed,
[Run control](running.md)) and a divergence print a remark between the rows:

```text
  >> step 339, t = 0.0610224: stop: stop requested by mallard ctl
```

The run ends with a **Summary**: why and where it stopped, a table of wall
time by phase (setup, time stepping and its chemistry, diagnostics, output),
the throughput, the bytes written and every file kind written, then the
[run health](#run-health). The same information, in machine-readable form,
goes to the run record. Errors go to stderr from any rank. Colors are off when
the output is not a terminal or `NO_COLOR` is set (`CLICOLOR_FORCE=1` forces
them, e.g. under `mpirun`).

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
several times keeps one history, each part led by its own `start` record, which
names the restart file. Resuming from an earlier restart file than the last one
(a rewind) keeps the records after the restart step, and the `start` record says
which step the record had reached.
Each record has a `type`:

| `type` | When | Fields |
|---|---|---|
| `start` | After setup | `schema` (`"mallard-run-log"`), `version` (2), `run_id` (the run's, as in its [run block](#run-block)), `restart_of` (`run_id`, `file` and `step` of the restart file the run started from, `run_id` `null` for a file of a run before 0.10; `null` for a fresh start), `provenance` (the [run block](#run-block); its key keeps its name), `placement` (one object per rank, as the [rank report](parallel.md#the-rank-report) finds it: `rank`, `host`, `threads`, `cpus` (the CPU set before Kokkos starts, e.g. `"0-15"`; `null` where unknown), `cpu_numa`, `omp_bind`, and on GPUs `gpu` (PCI address), `gpu_uuid`, `gpu_index`, `gpu_numa`; `null` where unknown or on CPUs), `setup_phases` (seconds of each setup phase: `max` and `mean` over ranks), `units` (the units of the progress records' quantities, as the CSV files' units lines write them, `"1"` for none: `t`, `dt`, `cfl`, `wall`, ..., and per object `monitors`, `margins` (of `health.margins`), `integrals`, `forces_<zone>` and by probe `probes`, a column each; bytes `B`; in a nondimensional case `"1"` for every physical quantity), `check` (the [check at setup](#the-check-at-setup)); runs started from a restart file: `restart` (`file`, `step`, `t`, and on a rewind `rewound_from`, the last step recorded before it) |
| `progress` | Every progress row | as below |
| `output` | A snapshot completely written (also in the background): the run's output index | `step`, `t`, `kind` (`vtkhdf`, `vtu`, `xdmf`, `restart`; probes: `vtp`, `vtkhdf`; `spectra`), `path`, `reason` (as the [snapshot metadata](#snapshot-metadata)'s), `writer` (the `[[write_data]]` prefix, probe or output that wrote it), `run_id`, `complete` (`true`), `bytes` (of its files), `files` (every file of it, each rank's piece too: `path`, `bytes`, `sha256`; no `sha256` for a [file appended to](#complete-files)) |
| `event` | A request acted on (the rows between progress rows) | `step`, `t`, `kind` (`checkpoint`, `stop`, `set` and `diverged` from [run control](running.md); `trigger` from [triggers](running.md#triggers), with also `name`, `firing` (1 for the first) and `values`, the values its condition read), `text` |
| `divergence` | A run that diverged, before its `error` record | the [divergence report](running.md#the-divergence-report): `step`, `t`, `invalid_cells`, `failed` (cells by the quantity that failed first), `box` (their extent), `last_valid` (`step`, `t`, `file`, or `null`), `causes` (most likely first: `kind` (`time_step`, `mesh`, `boundary`, `initial_state`, `chemistry`, `numerics`), `title`, `score`, `evidence`, `fix`), `cell`, `neighbors`, `others` (`id`, `rank`, `x`, `failed`, `rho`, `speed`, `p`, `T`, `cfl`, `quality`, `repaired`, `troubled`), `time_step` (`integrator`, `limit`, `dt`, `dt_cfl1`, `cfl`, `cells_over_limit`), `zones` (`name`, `type`, `near`, and where compared `inadmissible`, `ratio`, `jump`), `history` (the last progress rows), `snapshot`, `units` (of `t`, `dt`, `x`, `rho`, `speed`, `p` and `T`, as the CSV files' units lines write them; `"1"` for each in a nondimensional case) |
| `warning`, `error` | A warning, or the error that ended the run | `code` (stable: [Warnings and errors](running.md#warnings-and-errors)), `message`, `docs` (the page and anchor that explain the code) |
| `end` | The end of the run, also on a stop signal, divergence or an error (on rank 0): a run without one was killed (`SIGKILL`, a lost node) or is still running | `outcome` (`finished`, `stopped`, `signal`, `wall_time_limit`, `diverged`, `error`), `cause` (`finished`: `n_steps` or `t_stop`; `stopped`: `stop_file`, `mallard_ctl` or `trigger`; `signal`: `SIGTERM`, `SIGINT` or `SIGUSR2`), `trigger` (the [trigger](running.md#triggers) that stopped the run, by name), `reason` (in words; the error's message), `step`, `t`, `last_snapshot` and `last_restart` (the last of each recorded complete: `step`, `t`, `path`; none if the run wrote none), `steps` (this run's), `wall` and its parts `wall_setup`, `wall_stepping`, `wall_diagnostics`, `wall_output` (and `wall_chemistry`), `rejected_steps` (with error control), `cells_per_second` |

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
| `probes` | By probe, the latest of its [monitored values](probes.md#in-the-run-record-and-report) (`step`, `t` of its last write, and its reductions, then a one-point probe's values, as in its files): the first 64 over every probe, in the order of `[[probes]]` |
| `health` | The [run health](#run-health) at this row (`[output] health`) |

Numbers are in SI units (seconds, bytes) and read back exactly; unknown
values are `null`. Within a `version`, records only gain fields; renaming,
removing or changing the meaning of one increments it. Version 2 (0.10) added
`run_id`, `restart_of` and the `output` records, which replace version 1's
`event` records of written files; `--restart latest` and `mallard info DIR`
find a run's files through them.

#### Complete files

A file is written under the name `NAME.partial` and gets its name `NAME`
once complete, as restart files always were: snapshots (each rank's piece
too), their indexes (`.pvd`, `.vtkhdf.series`, `.xmf`), XDMF data and mesh
files, probe snapshots and spectra. A file under its own name is therefore
never cut short by a job killed while writing it, and a `NAME.partial` left
behind is one whose write did not finish: `mallard info`, `mallard diff` and
`mallard post extract` refuse it and say so, and `--restart latest` never
picks it.

The `output` record of a snapshot is written once all its files are
complete, with each file's size and SHA-256, so a snapshot without a record
was not completely written. One thread hashes the files after their write, on
the CPU's SHA instructions where it has them (over 1 GB/s); a run that writes
faster than that waits for it, and the run summary says how long.
`mallard info DIR` checks every recorded file against its record (its size;
`--verify` also its SHA-256) and reports one that is missing or differs.

Files appended to in place cannot be renamed: a series in one file
(`series = true`), a probe's HDF5 or VTKHDF series and the CSV files of
probes, forces and integrals. A job killed while appending can leave such a
file damaged (HDF5 files are not safe against it) or ending in a partial
row; the step's missing `output` record says that step was not completely
written. Their records give the size after each step and no SHA-256, which
would rehash the whole file at every step.

### Run report: `report.html`

Every run also writes `report.html`, one page to open after a run (or during
one) in any browser: the first thing to look at.
[A sample report](../images/guide/run_report_sample.html) of a short shock
run shows every section (download it and open it in a browser; GitHub shows its
source; `tools/docs/run_report.py` builds it). It shows

- the **outcome**: how and where the run ended (finished, stopped, a signal,
  the wall-time limit, diverged, an error), its last snapshot and restart file,
  or that it has no end record yet (still running, or killed);
- the **run**: its run id, the run it restarted from, the Mallard version and
  build, where and on how many ranks it ran, the input file and mesh with their
  SHA-256, and each run of a record continued by restarts;
- the **configuration**: physics, mesh, numerics, time step and stop
  conditions, the boundaries, and the whole resolved input;
- the [**check at setup**](#the-check-at-setup): mesh quality, the CFL number
  against the time integrator's limit, the resolution estimates, the memory
  after setup, and the check's warnings;
- its **health**: the lowest density, pressure and temperature and where they
  occurred, the numerics' activity, the numerical repairs, the conservation
  ledger, and the warnings and errors;
- plots of the **monitors** over the run: the time step and CFL number, the
  progress row's minima and maxima, throughput and memory, the `[integrals]`,
  each `[[forces]]` zone, each probe's reductions and one-point values, and the
  conservation ledger, with a table of each
  quantity's last, least and largest value;
- the **cost**: wall time by phase, throughput, memory;
- the **output index**: what each writer wrote, from the run record;
- the [divergence report](running.md#the-divergence-report) of a run that
  diverged: its likely causes with their evidence and fix.

```toml
[output]
report = "report.html"   # the default; "" writes none
```

The page is built from the run's records alone, the ones
[`mallard.jsonl`](#run-record-mallardjsonl) holds, never from the terminal
output or by reading output files back. Rank 0 keeps them in memory as the run
writes them and writes the page at the first progress row, after each restart
file and at the end, also of a run that stops, diverges or fails; a run killed
outright keeps the page of its last restart file, the point it restarts from.
It is written as `report.html.partial` and renamed once complete. A run
started from a restart file reads its run record once, at setup, so the page
covers the whole record. `[output] run_log = ""` does not turn the page off.

`mallard info RUN_DIR` prints the page's outcome, health, check at setup, cost and warnings
as text, from the same records, each section with its status (ok, WARN,
ERROR) as `mallard check` gives them. `mallard info RUN_DIR --html` builds the same page from `RUN_DIR/mallard.jsonl`
into `RUN_DIR/report.html`, byte for byte what the run would have written: for
a run that is still going, was killed, or ran with `report = ""`
([Command line](cli.md#mallard-info)).

The page is self-contained: no network, no files beside it, so it can be
mailed or archived with the run. It follows the browser's light or dark
setting and prints. The plots are drawn in the browser by a small script
inlined in the page; without JavaScript, every table, the last values
included, still shows. Its size does not grow with the run: each plotted
quantity keeps at most 512 points, a long run's progress rows combined in
pairs as they come, each point the last value of its rows with a band between
their least and largest, so a spike is never lost. At most 48 quantities are
plotted (the others are in the table), and the warnings and events shown are
the first and last 20 (all are in `mallard.jsonl`). A page is typically a few
hundred kB, under 2 MB whatever the run's length, and costs milliseconds to
write.

#### The check at setup

At the end of setup a run measures, as [`mallard check`](input_files.md#checking-an-input)
does and with the same code, the mesh quality and the resolution of the case's
physical scales, and records them with the time step's stability limit and the
memory after setup in the `start` record's `check`:

| Member | Holds |
|---|---|
| `status`, `diagnostics` | As in [the check's JSON](input_files.md#the-report-as-json): `ok` or `warning`, and the warnings with their codes (`W-QUALITY-SKEWNESS`, `W-CFL-LIMIT`, `W-WALL-Y-PLUS`, ...) |
| `mesh_quality`, `resolution` | As in the check's JSON, with the 3 worst cells of each metric |
| `time_step` | `integrator`, `cfl_limit` (its stability limit for smooth flow, `null` if none is known), `set_by` (`CFL` or `fixed`), and `cfl` (`run.cfl`) when set by CFL |
| `memory` | Bytes of the rank that uses the most, after setup: `host`, `host_peak`, `device`, `device_total` (`null` where unknown) |

It costs one pass over the cells and faces, timed as the setup phase "mesh
quality and resolution"; the run does not repeat the check's memory and cost
estimates or its partitions for other rank counts. The report and
`mallard info RUN_DIR` show it in the check's terms and statuses, with the
largest CFL number of the progress rows against the same limit. The check's
warnings are shown there, not as warnings of the run.

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
| `activity` | Fractions of cells, over the domain (`all`) and the least and most active rank (`rank_min`, `rank_max`): `troubled_cells` (TENO's troubled-cell indicator), `limited_cells` (MUSCL cells whose smallest limiter over the variables is below 0.9, where the limiter cuts the reconstructed gradient by more than a tenth: shocks and steep gradients; smooth limiters such as Venkatakrishnan's sit just below 1 almost everywhere in smooth flow, which this leaves out), `upwind_fraction` (the hybrid flux's mean upwind fraction) |
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

Every value Mallard writes is in SI units (the mesh-quality angle
`NON_ORTHOGONALITY` excepted, in degrees), and every output file says which,
as metadata in the way its format's tools expect; field names never change.
Restart files are the exception: they hold the conservative variables, in SI
units by definition, for Mallard to read back.

| File | Where the units are |
|---|---|
| VTKHDF, HDF5 (snapshots, their mesh file, probes) | CF-convention string attributes of each dataset: `units` ([UDUNITS](https://docs.unidata.ucar.edu/udunits/current/) syntax, `1` for dimensionless) and `long_name`; `standard_name` only where a CF standard name fits, which no field does yet (`air_pressure` and the like are for atmospheres) |
| XDMF (`.xmf`) | `<Information Name="units" Value="..."/>` and `<Information Name="long_name" .../>` in each `<Attribute>` |
| VTU, VTP (each piece of a PVTU) | Field data: the string arrays `MALLARD_FIELD_NAMES`, `MALLARD_FIELD_UNITS` and `MALLARD_FIELD_LONG_NAMES`, entry *i* of each for the same array (VTK has no units attribute) |
| CSV (forces, integrals, probes, `mallard reactor` and its chemistry benchmark) | A line `# units: ...` after the header, one entry per column (`1` for a dimensionless or text column) |
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
| `Q_CRITERION`, `LAMBDA2` | `s-2` | `CFL`, `MACH`, `TENO_SIGMA`, `Y_<s>`, `X_<s>`, `CHEM_COST`, `TF_F`, `TF_E`, `TF_OMEGA`, `PASR_KAPPA` | `1` |
| `NON_ORTHOGONALITY` | `degree` | `SKEWNESS`, `ASPECT_RATIO`, `VOLUME_RATIO`, `STENCIL_*`, `DELTA_OVER_ETA`, `CELLS_PER_DELTA_L` (`mallard check`) | `1` |
| `MASS_FLUX`, `SPECIES_FLUX_<s>`, `SPECIES_DIFFUSIVE_FLUX_<s>`, `SPECIES_TOTAL_FLUX_<s>` (probes) | `kg m-2 s-1` | `ENTHALPY_FLUX`, `ENERGY_DIFFUSIVE_FLUX`, `ENERGY_TOTAL_FLUX`, `Q_W` | `kg s-3` (W/m²) |
| `MOMENTUM_FLUX`, `TRACTION_PRESSURE`, `TRACTION_VISCOUS`, `TAU_W` | `kg m-1 s-2` | `U_TAU` | `m s-1` |
| `CF`, `CP`, `Y_PLUS` | `1` | `REGION_FRACTION` (regions) | `1` |
| `SCHLIEREN` | `kg m-4` | | |
| `WALL_MODEL_U` | `m s-1` | `WALL_MODEL_T`, `WALL_MODEL_H` | `K`, `m` |

CSV columns: `t` in `s`, coordinates, `s` and probe `length` in `m`, `step`,
`point`, `face`, `cell`, `count` and the normal `nx`, `ny`, `nz` `1`; the
`area` of a sample of a boundary, cut plane or surface, and its reductions'
`area`, in `m2` (in 2D, where these are curves, `m`, the reductions' column
`length`); forces in `N` (`kg m s-2`), per unit depth in 2D
(`kg s-2`), and the wall model's `Q` in `W`; domain integrals carry the
volume's `m3` (per unit depth in 2D: `m2`); probe reductions have their
variable's units, `integral` times those of the measure (`m` along a curve,
`m2` over a 3D boundary, cut plane or surface: `integral_MASS_FLUX` in `kg s-1`; `moment` those of
`integral` times `m`; over a region its
`volume`, `m3`, or `area`, `m2`, per unit depth in 2D). In axisymmetric runs,
forces, integrals and region probes are per radian of the revolved domain, and their units
say so: `kg m s-2 rad-1`, `kg m2 s-2 rad-1`; so are boundary probes' `area`
(`m2 rad-1`), region probes' `volume` (`m3 rad-1`) and `integral` reductions. Spectra rows: a `mean` has its
field's units, `spectrum`, `correlation` and `shell` (sums, not densities)
the product of their pair's (`U_X*U_Y`: `m2 s-2`); the `abscissa` is a
wavenumber in `m-1` (spectra, shells) or a separation in `m` (correlations).
Files written before Mallard 0.9 have no units; the readers return them empty.

### Run block

Every output file records the run that wrote it: the run block, one JSON
object, the same in every file of that run (where [snapshot
metadata](#snapshot-metadata) is each snapshot's own). It is written into
restart files (a trailer after the data, also of compressed files), VTU,
`.pvtu`, `.pvd`, XDMF and probe `.vtp` files (an XML comment `<!-- mallard-run
{...} -->` after the XML declaration), VTKHDF and HDF5 files, probe files
included (the string attribute `mallard_run` of the root group; a continued
run's file holds the latest run's), mesh quality files and the case view, CSV
files of forces, integrals, probes, region reductions, spectra and the
reactor (a line `# mallard-run {...}` after the header and its [units
line](#units), and another where a continued run starts appending: the rows
after it are that run's), and the run record's `start` record.

| Field | Contents |
|---|---|
| `schema`, `version` | `"mallard-run"`, 3 |
| `mallard` | `version` (`git describe`), `git_sha`, `git_dirty` (uncommitted changes in the build); `null` outside a git checkout. Taken at build time |
| `dependencies` | Versions of `compiler`, `kokkos`, `mpi` (the library's own string and the standard), `hdf5`, `cgns`, `zlib`, `nccl`, `gpu_runtime` and `gpu_driver` (CUDA or HIP); `null` for those not built in |
| `build` | `dim`, `precision`, `build_type`, `compiler`, `cxx_flags`, `kokkos` (version), `backend` and `host_backend` (Kokkos execution spaces), `mpi`, `gpu_aware_mpi`, `nccl`, `hdf5`, `hdf5_parallel` (HDF5 built with MPI), `kaminpar`, `species_layout` |
| `run` | `run_id` (a random UUID, new for every run, restarts included), `start_time` (UTC, ISO 8601), `ranks`, `host`, `device` (rank 0's), `directory`, `hosts` (the number of hosts; where each rank ran is the run record's `placement`), `restart_of` (as the run record's: `run_id`, `file`, `step`; `null` for a fresh start) |
| `input` | `file`, `sha256` of its bytes, its `text`, and `resolved`: the input with every key explicit, as the run used it ([the resolved input](input_files.md#the-resolved-input), what `mallard input resolve` writes) |
| `mesh` | `type`; for mesh files `file` and the `sha256` of its bytes (of a directory: of each file's relative path and bytes, by path; skipped with `[output] hash_mesh = false`) |
| `files` | Each other file the run read, once (absent if none): `kind` (`mechanism`, `table`, `inflow_profile`, `probe_points`, `restart`, `remap_source`, `remap_mesh`, and `snapshot` for each snapshot `mallard post extract` read, once per step of a file of several), `path` as given, `resolved`, `bytes`, and `sha256` of its bytes. A restart file, remap source or snapshot is not hashed again: it records the `run_id`, `step` and `t` of the run that wrote it, and the `sha256` from that run's [output record](#run-record-mallardjsonl) of the file (`sha256_from`). Where there is none (a copied or moved file, a run before 0.10, a changed file) and for a remap source mesh, `sha256` is `null`, with the file's `modified` time and a `note` saying why |
| `preset` | The `[numerics] preset` (`null` without one): `name`, `default` (`true` when the input names no preset and gets the default, `high_order`), and per key it covers its `key`, its `value` in the resolved input (as TOML: `"\"HLLC\""`, `0.5`) and `from_preset` (`false` where the input overrides it) |

Before 0.10 the block was named the provenance block: schema
`"mallard-provenance"`, the comment `<!-- provenance {...} -->`, the attribute
`provenance` and the CSV line `# provenance {...}`, without `run_id` and
`restart_of`. Every reader (`mallard info`, the post-processing commands,
`tools/mallard`) reads those names too, and says so. Version 2 (0.9.1) added
`input.resolved`; version 1 (0.9) had none, its preset values were plain text
(`"HLLC"`), and a `setup` object held the terminal output's setup summary as
`[key, value]` pairs per section; `resolved_input()` of `tools/mallard/vtu.py`
returns `None` for such files.

Check that a file belongs to a given input with `sha256sum input.toml`; read
the block in Python with `run_block(path)` of `tools/mallard/vtu.py` (any
file; `run_blocks(path)` gives each run of a continued CSV file).
`mallard info FILE` summarizes the block, and `mallard info --input FILE`
prints the input that produced the file, ready to run again
([Command line](cli.md#mallard-info)).

### Snapshot metadata

Every snapshot records which snapshot it is, in its format's own slot and in
a snapshot block, one JSON object per snapshot:

| Format | Step and time | Snapshot block |
|---|---|---|
| VTU, probe VTP | field data `TIME` (s) and `STEP` | XML comment `<!-- mallard-snapshot {...} -->` |
| PVTU | its pieces' | XML comment, as VTU |
| PVD index | `timestep` and `step` of each `DataSet` | its members' |
| VTKHDF, one snapshot per file | field data `TIME` and `STEP` | dataset `mallard_snapshot` of the root group |
| VTKHDF, `series = true`, and probe VTKHDF | `Steps/Values`, and field data `TIME` and `STEP` per step | dataset `mallard_snapshot`, one row per step |
| `.vtkhdf.series` index | `time` and `step` of each entry | its members' |
| XDMF | `<Time>` and `<Information Name="step">` of each grid; root attributes `time` and `step` of each `.h5` | dataset `mallard_snapshot` of each `.h5` |
| Restart | header `step` and `t` | trailer before the run block's |
| Probe HDF5 | datasets `time` and `step` | root attribute `mallard_runs`: each run's `run_id`, `restart_of`, `first_row`, `first_step` |
| CSV: forces, integrals, probes, reductions | columns `step` and `t` | each run's `# mallard-run` line precedes its rows |
| Spectra CSV | in its block | line `# mallard-snapshot {...}` |
| Reactor CSV | columns `step` and `t` | `# mallard-run` line |

The block's fields:

| Field | Contents |
|---|---|
| `schema`, `version` | `"mallard-snapshot"`, 1 |
| `run_id`, `restart_of` | The run that wrote it, as its [run block](#run-block)'s: a continued series holds the snapshots of each run that wrote to it |
| `step`, `time` | Step; simulated time in s |
| `written_at` | UTC, ISO 8601 |
| `reason` | `interval` (its `interval` or `time_interval`), `start` (the initial state), `end` (the run's end), `stop` (a stop request or signal), `checkpoint` (a [run control](running.md) request), `divergence` (the last valid state, written as the run diverged), `trigger` (a [trigger](running.md#triggers)'s `write`) |
| `format`, `format_version`, `solver_precision` | The file's format and its [version](#formats-and-compatibility); the precision the solver ran in (`double`, `single`) |
| `statistics` | Snapshots of time-averaged fields (`MEAN_`, `COV_`, `PHASE_`) and spectra: `average_start`, `average_end` (s), `samples`; absent otherwise |
| `fields` | Per field: `name`, `units`, `long_name`, `stored` (`float64`, `float32`: as written, which a single-precision build or output makes `float32`) |
| `units` | The units of the block's own numbers (`time`: `s`) |

Files written before 0.10 have no snapshot block. Readers then take step and
time from the format's own slots, and say so: VTU, VTKHDF and the indexes had
no step, so their snapshots are numbered by their position in the file or
index, from 0; their run is that of the file's run block, and their reason
and stored precision are unknown. `snapshot_blocks(path)` of
`tools/mallard/vtu.py` reads the blocks of any file (`[]` for older files),
and `mallard info FILE` prints the last one under **This snapshot**.

### Formats and compatibility

Every file Mallard writes carries a format version. **The promise:** a release
reads the restart and mesh files of every earlier release of the same major
version. In 0.x that means every restart file since 0.1 and every HDF5 mesh since
0.3. A file from a *newer*
Mallard is rejected with a message that gives its version and the versions this
build reads. It is never misread.

| File | Version, where | Versions (written from) | Read by |
|---|---|---|---|
| Restart (`format = "restart"`) | `uint32` after `MALLARD-RESTART` | 1 (0.1), 2: named variables (0.4), 3: `[statistics]` weights (0.6), 4: characteristic boundary states (0.7), 5: `compression = "zlib"` (0.9), 6: `[statistics.spectra]` and `[statistics.phase]` state (0.9), 7: the [snapshot block](#snapshot-metadata) trailer (0.10) | Mallard: all; `tools/mallard/restart.py`: 2 on |
| HDF5 mesh (`mallard mesh convert`) | root attribute `version` | 1 (0.3) | Mallard: all |
| VTKHDF output (`format = "vtkhdf"`) | root attribute `version` (VTK's own in `/VTKHDF`'s `Version`: 2.0) | 1 (0.9) | ParaView 5.12+, `tools/mallard/vtkhdf.py` |
| XDMF output (`format = "xdmf"`; `"hdf5"` before 0.10) | root attribute `version` of both files | 1 (0.6) | ParaView, `tools/mallard/h5.py` |
| VTU output | field data `MALLARD_FORMAT_VERSION` | none before 0.9, 1 (0.9) | ParaView, `tools/mallard/vtu.py` |
| Run record (`mallard.jsonl`, `[output] run_log`) | `version` of each `start` record | 1 (0.9), 2: `run_id`, `restart_of`, `output` records (0.10) | Your scripts (records only gain fields within a version) |
| [Run block](#run-block) (every output file) | its `version` | 3: `mallard-run`, `run_id`, `restart_of` (0.10); as the provenance block: 2 (0.9.1), 1 (0.9) | `mallard info`, `tools/mallard/vtu.py` (`run_block()`, `resolved_input()`, every version) |
| [Snapshot block](#snapshot-metadata) (every snapshot) | its `version` | 1 (0.10) | `mallard info`, `tools/mallard/vtu.py` (`snapshot_blocks()`) |
| Probe output (`[[probes]]` `format = "hdf5"`, `"vtkhdf"`, `"vtp"`) | root attribute `version` (HDF5, VTKHDF), field data `MALLARD_FORMAT_VERSION` (VTP) | 1 (0.9) | Mallard (restarts append to HDF5), ParaView |
| TENO cache (`cache_file`) | magic `MALLARD-TENO-<n>` | - | Rebuilt automatically when it does not match |

- Restart files are written with version 7 since 0.10, whose snapshot block
  trailer earlier releases would take for a truncated file: they reject it
  with their newer-version message. (Until 0.9 they were written with the
  lowest version that held their contents.)
- A restart file must also match the build's dimension and precision
  (`Mallard_DIM`, `Mallard_USE_DOUBLE`); the message says which differs.
- Output files gain arrays and attributes within a version. A version changes
  only when existing content changes meaning or layout. The Python tools refuse
  versions newer than they know; update `tools/` with Mallard.
- A change to a format adds a version and keeps the reader of every older one,
  with a fixture in `test/data/formats/` that a test reads (see its
  [README](../../test/data/formats/README.md)).

## Post-processing

`mallard info` tells what any of these files holds, without an input file:
its kind, format version, mesh, times, fields with their units, ranges and
stored precision, its run block (**Run**, the same for every file of the run)
and the last snapshot's block (**This snapshot**); `mallard info solut` does so for each file a run wrote, one table
row each ([Command line](cli.md#mallard-info)). Every number comes in a table
column or a labeled line that names its units (`t [s]`, `min [m]`, a field's
units, `dimensionless`); a `-` is something the file does not record:

```
$ mallard info solut/flow.vtkhdf.series
== solut/flow.vtkhdf.series ===============================================================
  Kind               snapshot
  Format             VTKHDF series index (VTKHDF snapshots), version 1
  Values             2D, double precision
  Simulated time     3 snapshots, t = 0 to 0.002 s

== Mesh ===================================================================================
  cells  points
  12     20
...
== Snapshots ================================================================== 3 snapshots
  #  t [s]  file
  0  0      flow_000000.vtkhdf
  1  0.001  flow_000001.vtkhdf
  2  0.002  flow_000002.vtkhdf

== Fields ================================================= ranges at t = 0.002 s, the last
  field  units       components  min      max      description
  P      kg m-1 s-2  1           1        1        pressure
  RHO    kg m-3      1           1.03804  1.26232  density
  U      m s-1       3           0.3      0.3      velocity
  min and max of fields with several components are of their magnitude.
```

`--format json` prints the same as one JSON object, for scripts.

`mallard post extract` evaluates probes, region integrals and snapshots of
chosen fields on the state of restart files and snapshots after the run, as
the run would have written them ([Probes of existing output](probes.md#probes-of-existing-output)).

`mallard post convergence` compares runs of a case at several resolutions:
the observed order, the extrapolated value and the grid convergence index of
their forces, integrals and probes ([Grid convergence](convergence.md)).

### Comparing two solutions: `mallard diff`

`mallard diff A B` tells whether two files hold the same solution and, if not,
how far apart and where: for a regression check in CI, two settings of a case,
a restart that should reproduce a run, or a serial run against an MPI one
([Command line](cli.md#mallard-diff)). A and B are snapshot, restart, probe or
CSV files, of any format each; it compares every field both hold (`--fields`
picks some), at the last snapshot of each, or at the one of a step (`--step N`) or
nearest a time (`--time T`) as the files record them:

```
$ mallard diff serial/solut/restart_000020.restart mpi/solut/restart_000020.restart
== Diff ================================================================================= ok
  Result: ok, identical, the same state bit for bit (exit status 0)
  A: serial/solut/restart_000020.restart (restart)
  B: mpi/solut/restart_000020.restart (restart)
  Compared: cell by cell, in the mesh's global cell order (768 cells)
  Tolerance: 0 (values must be equal); on max |B - A| / max |A| per field
...
== Fields =============================================================================== ok
  |B - A| per cell, in the field's units: L1, L2, Linf are its mean, root mean square and
  max over the cells, not weighted by volume; differing: values not bitwise equal.
  status  field   units       differing cells  L1  L2  Linf
  ok      RHO     kg m-3      0 of 768         0   0   0
  ok      RHOU_X  kg m-2 s-1  0 of 768         0   0   0
...
```

- **Per field**, the `Fields` table gives the cells whose values are not
  bitwise equal and the L1, L2 and Linf norms of B - A (mean, root mean square
  and max over the cells, not weighted by volume) in the field's units;
  `Relative differences` divides each by the same norm of A; `Largest
  differences` names the cell of the max, its coordinates and both values.
  Vectors are compared by the magnitude of the difference; a NaN in one file
  only is an infinite difference.
- **Restart files** are also compared in the rest of their state, bit for bit:
  scalar attributes, named arrays (statistics weights) and the face records of
  characteristic boundaries. The [run block](#run-block) and snapshot blocks
  are not compared: they record when and where each run ran.
- **Tolerance and exit status.** A field passes if its max |B - A| / max |A|
  is at most `--tol` (default 0: any difference in value fails), and so does t,
  relative to the larger |t|; `--tol 1e-12,P=1e-6` sets one field apart. Steps
  that differ, or fields only one file holds, are warnings. As in
  [`mallard check`](cli.md#mallard-check), the status column, each section's
  heading and the first and last lines read `ok` (identical), `WARN` (within
  tolerance) or `ERROR` (beyond it); the exit status is 0, 0 and 1, and 3 when the files cannot be
  compared, so `mallard diff a b --tol 1e-10 || exit 1` gates a CI job.
  `--format json` prints the same as one object, its statuses `ok`, `warning`
  and `error`.
- **Cells are matched by their global index**, which restart files and VTKHDF
  or XDMF snapshots of the whole mesh keep whatever the ranks that wrote them;
  a restart file against a snapshot compares the fields both hold (RHO). VTU
  pieces and region snapshots are in their own order: `mallard diff` refuses
  to match them against those files, and compares two of the same layout value
  by value.
- **Different meshes** are refused, unless `--remap`: B's cell values are
  remapped onto A's mesh by the conservative remap of
  [`[initialize] type = "remap"`](initialization.md) (degree 2, limited) and
  compared there. The meshes, and the cell coordinates of the largest
  differences, come from the input each file's run block records (a mesh file
  relative to the run's directory, checked against its SHA-256), or from
  `--mesh-a` and `--mesh-b`.

```
$ mallard diff coarse/solut/restart_000020.restart fine/solut/restart_000020.restart --remap --tol 1e-2
...
== Remap ============================================================================== WARN
  Conservative, degree 2, limited. Remapped results differ from A by the remap's own error
  too, so a remap is never identical. A cells outside B's mesh take the nearest B cell's
  value.
  B cells  A cells  integrals kept to  A cells partly covered  A cells outside B
  3,072    768      1.2e-15            0                       0
...
== Relative differences =============================================================== WARN
  A norm of B - A over the same norm of A; a field passes if its Linf is at most its
  tolerance.
  status  field   units          L1           L2           Linf         tolerance on Linf
  WARN    RHO     dimensionless  0.000213803  0.000262748  0.000440052  0.01
...
```

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
- [Grid convergence](convergence.md): discretization error from runs at several resolutions
- [Statistics](statistics.md): time averages, spectra and correlations, phase averages
- [Initialization and restarts](initialization.md): restart files and their format
- [Running](running.md): run control of a running case
