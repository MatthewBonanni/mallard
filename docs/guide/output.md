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

| Key | Default | Meaning |
|---|---|---|
| `prefix` | required | Output path prefix |
| `format` | required | `vtu` (ParaView, with a `.pvd` time series), `hdf5` (with XDMF), or `restart` |
| `interval` / `time_interval` | one required | Every this many steps, or this much simulated time (the step is shortened to land on each output time) |
| `variables` | required (`vtu`, `hdf5`) | Variables to write, below; `Y_*` selects every variable with that prefix |
| `geometry` | `"all"` | `vtu`: a boundary zone name writes that zone's faces with the values of their cells (wall pressure, heat flux) |

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
the rank count. Read them in Python with `tools/mallard_h5.py`.

**Restart files** (`format = "restart"`) are described in
[Initialization and restarts](initialization.md#restarts).

## The log

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
throughput. Errors go to stderr from any rank. Colors are off when the output
is not a terminal or `NO_COLOR` is set (`CLICOLOR_FORCE=1` forces them, e.g.
under `mpirun`).

## Wall forces: `[[forces]]`

```toml
[[forces]]
zone = "cylinder"
interval = 10        # default 1
# file = "forces_cylinder.csv"
```

Writes `step, t` and the pressure and viscous force components of the fluid
on the zone, per unit depth in 2D (per radian in axisymmetric runs). Divide by
the dynamic pressure and a reference length (or area) for drag and lift
coefficients.

## Domain integrals: `[integrals]`

```toml
[integrals]
interval = 1
budget = true        # kinetic-energy budget (LES), default false
```

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

`[statistics]` keeps running time averages and covariances per cell, on the
device, without communication: bitwise the same on any number of ranks, and
carried across restarts.

```toml
[statistics]
fields = ["RHO", "U_X", "U_Y", "U_Z", "P", "T"]
products = ["U_X*U_X", "U_Y*U_Y", "U_Z*U_Z", "U_X*U_Y"]
t_start = 40.0
interval = 10
```

| Key | Default | Meaning |
|---|---|---|
| `fields` | none | `RHO`, `U_X`, `U_Y`, `U_Z`, `P`, `T`, `H`, `Y_<s>` |
| `products` | none | Pairs `"A*B"` whose covariance to keep |
| `t_start` | 0 | Start averaging after this time |
| `interval` | 1 | Sample every this many steps (weighted by the time since the last sample) |
| `reset` | `false` | Start afresh on a restart |

Write them with `variables = ["MEAN_*", "COV_*"]`, and average further over
homogeneous directions with `tools/plane_average.py` (e.g. over x and z of a
channel, giving profiles in y; its CSV also feeds
[synthetic inflow](boundary_conditions.md#synthetic-turbulence)).

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

![Probe shapes over a triangle mesh, colored by their point values of the pressure](../images/guide/probe_shapes-light.svg#gh-light-mode-only)
![Probe shapes over a triangle mesh, colored by their point values of the pressure](../images/guide/probe_shapes-dark.svg#gh-dark-mode-only)

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
| `scale`, `rotate`, `pivot`, `translate` | none | Move the geometry (scale, then rotate, about `pivot`, then translate) |

### Boundary data

`boundary = "<zone>"` (a name or a list) or `boundary_type = "wall"` (every
wall) turns each face of the zones into a sample: the state the boundary
condition sets on the face, wall quantities and fluxes. In 2D the faces form
ordered polylines with their arc length `s`; in 3D, polygons for ParaView.

```toml
[[probes]]
name = "cylinder"
boundary = "cylinder"
variables = ["P", "CP", "CF", "Q_W", "Y_PLUS"]
reference = {rho = 1.0, u = [0.2, 0.0], p = 0.7142857142857143}
time_interval = 1.0

[[probes]]
name = "inflow"
boundary = ["inlet", "outlet"]
samples = false
variables = ["MASS_FLUX", "MOMENTUM_FLUX", "ENTHALPY_FLUX"]
reductions = ["integral"]
```

![Boundary probes on the curved walls of an annulus: each wall a closed polyline traversed with the fluid on its left, and the inner wall's pressure against arc length, point values against cell averages and the exact value](../images/guide/probe_walls-light.svg#gh-light-mode-only)
![Boundary probes on the curved walls of an annulus: each wall a closed polyline traversed with the fluid on its left, and the inner wall's pressure against arc length, point values against cell averages and the exact value](../images/guide/probe_walls-dark.svg#gh-dark-mode-only)

In 3D the faces are polygons; ParaView colors the `.vtp` series by any variable:

![Heat flux into an isothermal wall under a sheared warm blob, from a 3D boundary probe](../images/guide/probe_wall_3d-light.svg#gh-light-mode-only)
![Heat flux into an isothermal wall under a sheared warm blob, from a 3D boundary probe](../images/guide/probe_wall_3d-dark.svg#gh-dark-mode-only)

| Variable | Meaning |
|---|---|
| `MASS_FLUX`, `MOMENTUM_FLUX`, `ENTHALPY_FLUX`, `SPECIES_FLUX_*` | Convective fluxes through the face, positive out of the domain |
| `TRACTION_PRESSURE`, `TRACTION_VISCOUS` | Force per unit area on the boundary; their `integral` is the `[[forces]]` force |
| `TAU_W`, `TAU_W_X`, ... | Wall shear stress (magnitude, components) |
| `CF`, `CP` | Skin friction and pressure coefficients of `reference` |
| `Q_W` | Heat flux into the fluid |
| `U_TAU`, `Y_PLUS` | Friction velocity and first-cell `y+` |

With TENO, `sampling = "point"` reads each cell's reconstruction polynomial at
the sample, so a probe converges at the scheme's order (fifth order for
`order = 5`) instead of the first order of cell averages at points off the
centroids. The full key list is in the [input reference](../input.md#probes).

Pitfalls:

- Point values fall back to the linear least-squares reconstruction in cells
  the TENO indicator flags troubled (shocks, and coarse meshes of smooth
  fields), and to the cell average outside the domain.
- Mass fractions are cell averages in both modes; `T` and `H` follow the
  reconstructed `p / rho` at the cell's composition.
- CSV is for small probes: it repeats the coordinates every output. Use `hdf5`
  or `vtp` for thousands of samples.
- Wall shear and heat flux use the viscous flux's one-sided wall gradients:
  second-order shear on smooth meshes, first-order heat flux at isothermal
  walls. `CF` is signed along `reference.u`.
- Fluxes of boundary probes are those of the face states, not the scheme's
  numerical fluxes: they agree as the mesh is refined.
- Reductions of point sets weigh each point 1 (`integral` is a sum); curves
  integrate with three Gauss points per element between samples.

## Post-processing

The Python scripts in `tools/` (numpy, matplotlib; see
[Tools](https://matthewbonanni.github.io/mallard/docs/tools/)) read Mallard's output directly:

| Script | Use |
|---|---|
| `mallard_vtu.py`, `mallard_h5.py` | Readers for VTU and HDF5 output, for your own scripts |
| `plot_vtu.py`, `compare_png.py` | A field of one or several snapshots |
| `animate.py` | MP4/GIF animation of a 2D series with numerical schlieren |
| `plane_average.py` | Averages over homogeneous directions |
| `plot_<case>.py` | Each validation case against its reference |

ParaView opens `.pvd` (VTU) and `.xmf` (HDF5) series directly.

## See also

- [Input reference: `[[write_data]]`](../input.md#write_data), [`[output]`](../input.md#output), [`[[forces]]`](../input.md#forces), [`[integrals]`](../input.md#integrals), [`[statistics]`](../input.md#statistics), [`[[probes]]`](../input.md#probes)
