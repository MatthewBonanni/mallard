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

### Spectra and two-point correlations: `[statistics.spectra]`

On a generated `cartesian` mesh, spectra and two-point correlations along the
homogeneous (periodic, unstretched) directions, at each position of the
others: per y of a channel, or shell spectra of a periodic box. Computed by
FFTs on the device, bitwise the same on any number of ranks.

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

| Key | Default | Meaning |
|---|---|---|
| `fields`, `products` | none | Variables (as `[statistics]`) and pairs `"A*B"` |
| `directions` | periodic directions | Homogeneous directions |
| `average` | `true` | Time averages, written to `file` with every snapshot and carried by restart files; `false`: one `<file stem>_<step>.csv` per sample |
| `file` | `spectra.csv` | Output CSV (`quantity, a, b, direction, position, <coordinates>, index, abscissa, value`) |
| `interval`, `t_start` | `[statistics]`'s | Sampling |

The file holds plane means, one-sided spectra (sums, not densities: they
add up to the covariance `<a'b'>`), correlations `<a'(x) b'(x + r)>` and, in
a periodic box, shell spectra. `tools/insitu_statistics.py` plots them
(`plot`) and recomputes them from restart snapshots (`spectra`, `compare`).

- Values are cell averages: at the grid cutoff a spectrum is lower than one
  of point values by the cell-average factor (sinc^2(k h / 2) per direction,
  0.41 at k = N/2), as `tools/isotropic_turbulence_stats.py --no-deconvolve`.
- A sample costs about a third of a step of the Re_tau 180 channel (11 ms
  on an A100): sample every 10-20 steps, not every step.
- Correlations are periodic: `R(L - r) = R(-r)`; use `r <= L/2`.

### Phase averages: `[statistics.phase]`

Per-cell means over the samples whose phase in a periodic cycle falls in each
of `bins` bins, output as `PHASE_<bin>_<field>` (`variables = ["PHASE_*"]`).
The phase comes from a fixed `period`, or from the upward crossings of a
level by a variable at a probe point:

```toml
[statistics.phase]
fields = ["U_X", "U_Y", "P"]
bins = 12
probe = [3.0, 0.0]               # or: period = 30.3 (and t_reference)
variable = "U_Y"
level = 0.0
```

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

```toml
[[probes]]
name = "centerline"
variables = ["RHO", "P"]
start = [0.0, 0.01]
end = [1.0, 0.01]
n_points = 200
interval = 10
```

Point (`point = [x, y]`) or line (`start`, `end`, `n_points`) probes append
`step, t, point, x, y, <variables>` to `probe_<name>.csv`, at full precision,
from the cell containing each point. The [Sod comparison](numerics.md#reconstruction)
on the Numerics page was read from such a line probe. Variables are those of
`[statistics]`.

## Post-processing

The Python scripts in `tools/` (numpy, matplotlib; see
[Tools](https://matthewbonanni.github.io/mallard/docs/tools/)) read Mallard's output directly:

| Script | Use |
|---|---|
| `mallard_vtu.py`, `mallard_h5.py` | Readers for VTU and HDF5 output, for your own scripts |
| `plot_vtu.py`, `compare_png.py` | A field of one or several snapshots |
| `animate.py` | MP4/GIF animation of a 2D series with numerical schlieren |
| `plane_average.py` | Averages over homogeneous directions |
| `insitu_statistics.py` | Plots `[statistics.spectra]` output; recomputes it, or phase-bins a probe series, to check the in-situ values |
| `plot_<case>.py` | Each validation case against its reference |

ParaView opens `.pvd` (VTU) and `.xmf` (HDF5) series directly.

## See also

- [Input reference: `[[write_data]]`](../input.md#write_data), [`[output]`](../input.md#output), [`[[forces]]`](../input.md#forces), [`[integrals]`](../input.md#integrals), [`[statistics]`](../input.md#statistics), [`[statistics.spectra]`](../input.md#statisticsspectra), [`[statistics.phase]`](../input.md#statisticsphase), [`[[probes]]`](../input.md#probes)
