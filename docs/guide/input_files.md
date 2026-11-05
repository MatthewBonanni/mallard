# Input files

Everything about a run is in one [TOML](https://toml.io) file, read by

```sh
mallard -i input.toml [Kokkos options]
```

Kokkos options such as `--kokkos-num-threads=8` (CPU threads) or
`--kokkos-device-id=1` (which GPU) follow the input file. `mallard --version`
prints the version and `mallard --help` the usage. Errors in the input stop
the run before it starts, with the file name, the key and, for options, the
accepted values.

The quickest way to a working input is a template: `mallard new --list`
shows them, and `mallard new cylinder` creates a directory with the input,
the scripts that generate its mesh, run it and plot the results, and a README
([First simulation](../tutorials/first_simulation.md#7-templates)). The input is the
example's, with a comment above each table that links to its keys.

## Anatomy of an input

Each table configures one part of the problem. This is the complete input of
the Sod shock tube (`examples/sod/input.toml`), slightly shortened:

```toml
[run]                           # when to stop, how large the steps are
t_stop = 0.2
cfl = 1.0

[mesh]                          # a generated 200 x 4 strip
type = "cartesian"
Nx = 200
Ny = 4
Lx = 1.0
Ly = 0.02

[physics]                       # Euler equations, calorically perfect gas
type = "euler"
gamma = 1.4
p_ref = 1.0
T_ref = 1.0
rho_ref = 1.0

[initialize]                    # the state at t = 0, as expressions in x, y, z
type = "analytical"
rho = "x < 0.5 ? 1.0 : 0.125"
u = ["0.0", "0.0"]
p = "x < 0.5 ? 1.0 : 0.1"

[[boundaries]]                  # one entry per boundary zone (here all four sides)
name = "left"
type = "symmetry"
# ... right, top, bottom alike

[numerics]
riemann_solver = "HLLC"

[numerics.face_reconstruction]  # fifth-order TENO-E
type = "TENO"
order = 5

[[write_data]]                  # snapshots for ParaView (VTKHDF)
prefix = "solut/sod"
time_interval = 0.02
variables = ["RHO", "U_X", "P", "T"]
```

| Table | Required | Topic |
|---|---|---|
| `[run]` | yes | [Time stepping](numerics.md#time-step-and-stopping): `cfl` or `dt`, and a stop condition |
| `[mesh]`, `[[mesh.curved]]`, `[[periodic]]` | yes | [Meshes](meshes.md) |
| `[physics]` | yes | [Physics](physics.md); gas mixtures: [Thermochemistry](thermochemistry.md) |
| `[initialize]` | yes | [Initialization and restarts](initialization.md) |
| `[[boundaries]]` | unless the mesh is fully periodic | [Boundary conditions](boundary_conditions.md) |
| `[numerics]` and its subtables | no | [Numerics](numerics.md) |
| `[chemistry]`, `[radiation]` | no | [Thermochemistry](thermochemistry.md) |
| `[les]`, `[les.combustion]` | no | [Large-eddy simulation](les.md) |
| `[source]`, `[[sponges]]` | no | [Sources and sponges](sources.md) |
| `[regions]` | no | [Named regions](geometry.md#named-regions), used by `[[sources]]`, `[[sponges]]` and `[[probes]]` |
| `[[write_data]]`, `[output]` | no | [Output](output.md) |
| `[[probes]]`, `[[forces]]`, `[integrals]` | no | [Probes and monitors](probes.md) |
| `[statistics]` and its subtables | no | [Statistics](statistics.md) |
| `[parallel]` | no | [Running in parallel](parallel.md) |

Regions that several tables use (inline, or named once in `[regions]`), and
probe points and curves, are on [Geometry](geometry.md).

Double brackets (`[[boundaries]]`) are TOML arrays of tables: repeat them, one
entry each. Every key, with its default, is in the
[input reference](../input.md).
Tables, arrays and inline tables nest at most 64 levels deep (a key of a
top-level table is at level 2); deeper input is an error, not a crash.

## Numbers, vectors and expressions

- **Numbers.** Integers are accepted wherever a real number is expected
  (`cfl = 1` works).
- **Vectors** have one component per spatial dimension: `u = [1.0, 0.0]` in
  the 2D solver, `u = [1.0, 0.0, 0.0]` in the 3D one (the dimension is fixed
  when Mallard is built, see [Installation](https://matthewbonanni.github.io/mallard/docs/installation/)).
- **Expressions.** Initial states, boundary states, sources, sponges, regions
  and probe curves can be strings of `x`, `y`, `z` (0 in 2D) and, where noted,
  the time `t`: `"x < 0.5 ? 1.0 : 0.125"`, `"exp(-((x - 1)^2 + y^2) / 0.01)"`.
  One language, with local variables, conditionals and loops, serves every
  section: see [Expressions](expressions.md).
- **Units.** Plain numbers are in SI units (Pa, K, m, s, kg), and any
  dimensional number can instead be written with its unit, as a string:
  `p = "1 atm"`. See [Units](#units).

## Units

Mallard computes in SI units. A number without a unit is SI, so existing
inputs mean what they always did; a quoted number with a unit is converted to
SI when it is read:

```toml
[run]
dt = "0.5 us"
t_stop = "2 ms"
t_wall_stop = "11.5 h"

[mesh]
type = "cartesian"
Lx = "5 cm"
Ly = "20 mm"

[initialize]
u = ["10 m/s", 0.0]       # units and plain numbers mix freely
p = "1 atm"
T = "25 degC"
X = { H2 = 2.0, O2 = 1.0, N2 = 3.76 }

[[boundaries]]
name = "left"
type = "wall_heat_flux"
q = "50 kW/m^2"

[[sponges]]
strength = "20 1/s"       # a constant, or an expression in SI: "x > 0.04 ? 2e4 * (x - 0.04) : 0"
u = [0.0, 0.0]
p = "101.325 kPa"
T = "300 K"

[source]
gravity = [0.0, "-9.81 m/s^2"]
```

Each key has a dimension (pressure, temperature, length, velocity, ...),
listed in the [input reference](../input.md#units), and a unit of another
dimension stops the run before it starts:

```text
error: input.toml: line 14: boundaries.p: "300 K" has the dimension of temperature (K). Expected pressure (kg m^-1 s^-2), e.g. in Pa, kPa, MPa, bar, atm, psi; a number without a unit is in SI. See docs/guide/input_files.md#units.
```

**Writing units.** A number, an optional space, then the unit:

| Form | Examples |
|---|---|
| A unit, with an SI prefix (`Y Z E P T G M k h da d c m u n p f a y z`; `u` or `µ` for micro) | `"5 mm"`, `"5mm"`, `"3 MPa"`, `"2 us"`, `"1 kmol"` |
| Products: a space, `*` or `·` | `"1.8e-5 Pa s"`, `"2 N*m"` |
| Powers: `^`, `**` or `²` `³`; negative powers too | `"9.81 m/s^2"`, `"1 g/cm³"`, `"9.81 m s^-2"` |
| Quotients: `/` divides by the next unit or a parenthesized group | `"8.314 J/(mol K)"`, `"20 1/s"`, `"1.2 kg/m^3"` |

`"J/kg K"` is ambiguous (is K in the numerator?) and an error: write
`"J/(kg K)"`.

| Dimension | Units |
|---|---|
| Length | `m`, `in`, `ft` (and `mm`, `cm`, `km`, `um`, ...) |
| Time | `s`, `min`, `h` (and `ms`, `us`, `ns`, ...) |
| Mass | `g`, `lb` (and `kg`, `mg`, ...) |
| Temperature | `K`, `degR`; absolute temperatures also `degC` (or `°C`), `degF` (or `°F`) |
| Amount of substance | `mol` (and `kmol`, ...) |
| Angle | `rad`, `deg` (or `°`) |
| Pressure | `Pa`, `bar`, `atm`, `psi`, `Torr`, `mmHg` (and `kPa`, `MPa`, `mbar`, ...) |
| Energy, power, force, frequency | `J`, `cal`, `W`, `N`, `lbf`, `Hz` (and `kJ`, `kcal`, `kW`, `kHz`, ...) |
| Volume | `L` (and `mL`) |

**Temperatures.** `degC` and `degF` are absolute temperatures: `"20 degC"`
is 293.15 K. They stand alone (`"20 degC/s"` is an error; use K) and are an
error for the one key that is a temperature difference, `sutherland_S`.

**Keys documented in other units.** Two keys keep plain numbers in their own
units, and take units like any other: `[mesh.transform] rotate` (degrees:
`rotate = 90` or `"1.5708 rad"`) and `[chemistry] C_reg` (kmol/m^3, as
Cantera; `"1e-9 mol/m^3"` is the same as `1e-12`).
`[mesh.transform] scale` is the length of one unit of the mesh file:
`scale = "1 mm"` for a mesh drawn in millimeters (the same as `0.001`).

**Expressions are in SI units.** Where a key takes an expression, a quoted
quantity with a unit is still a single number (`"20 1/s"`), but the
expressions themselves (`"x < 0.05 ? 1e5 : 2e4"`) are in SI units, `x`, `y`,
`z` in meters and `t` in seconds, and contain no unit symbols. A unit symbol
in an expression that fails to parse is pointed out in the error.

**Nondimensional runs.** Single gases are scale-free: the reference state
(`p_ref`, `T_ref`, `rho_ref`) sets the gas constant, so nondimensional inputs
(`p_ref = T_ref = rho_ref = 1`) work as plain numbers. Units then make no
sense; write them only in dimensional inputs. Gas mixtures read real
thermochemistry and are always dimensional.

## Expressions

One language, kexpr, serves every key that takes an expression: its
variables, operators, functions, statements and loops are on the
[Expressions](expressions.md) page.

## Inputs from older releases

Inputs keep working across releases. When a key is renamed, moved or
removed, the change goes through two steps:

1. **Deprecated** (at least one minor release): the old key is still read,
   and the run prints a warning that names the line, the new key and the
   release that will remove the old one.
2. **Removed**: the run stops with the same message.

`mallard check` lists both kinds, deprecated keys as warnings and removed
ones as errors.

In both steps, one command brings a file up to date:

```sh
mallard input upgrade old.toml > new.toml     # changes listed on stderr
mallard input upgrade --check old.toml        # exit status 1 if anything is out of date
```

It changes only the lines of those keys. A renamed key keeps its value and its
comment, a moved key goes to the end of its new table, a removed key is
commented out with a note, and a renamed option changes in place, also inside
inline tables. A file with nothing to change comes back unchanged, byte for
byte. When a renamed, moved or removed key is written as a dotted key
(`run.cfl = 1`) or in an inline table, the file is reformatted instead: its
comments are kept, its layout is not. Some renames depend on the case's
dimension, which the command reads from the mesh as a run does (a relative
mesh path starts from the working directory, so run it where you run the
case); when the mesh is not at hand, state it: `mallard input upgrade --dim 3 old.toml`. The command is tested on the input
files of every release (`tools/dev/check_upgrade_input.py`).

Keys and options changed so far:

| Old | New | Deprecated | Removed |
|---|---|---|---|
| `[numerics.face_reconstruction]` `basis_order = p` | `order = p + 1` (order of accuracy) | 0.2.0 | 0.2.0 |
| `basis_type`, `max_stencil_size_factor`, `quadrature_type`, `quadrature_order`, `quadrature_type_cell`, `quadrature_order_cell`, `quadrature_type_face`, `quadrature_order_face` (same table) | none: TENO-E sets them | 0.2.0 | 0.2.0 |
| `[physics]` `p_min`, `p_max` | none: pressures are not clipped | 0.2.0 | 0.2.0 |
| `[[write_data]]` `format = "tecplot"` | `"vtu"` | 0.2.0 | 0.2.0 |
| `[[write_data]]` `format = "hdf5"` (XDMF) | `"vtkhdf"` (the default), or `"xdmf"` for the same XDMF and HDF5 files ([which format](output.md#which-format)) | 0.9.0 | 0.10.0 |
| `[[boundaries]]` `type = "upt"`, `"p_out"`, `"extrapolation"` | `"inlet_state"`, `"outlet_pressure"`, `"outlet_extrapolation"` ([boundary conditions](boundary_conditions.md#inputs-from-older-releases)) | 0.9.0 | 0.10.0 |
| `[[boundaries]]` `type = "p_out_average"` | `"outlet_pressure"` with `average = true` | 0.9.0 | 0.10.0 |
| `[[boundaries]]` `type = "nscbc_inlet"`, `"nscbc_outlet"` with `L`, `sigma`, `beta`, `sigma_T`, `sigma_t` | `"inlet_state"`, `"outlet_pressure"` with `nscbc = { L, sigma, ... }` | 0.9.0 | 0.10.0 |
| `[[mesh.curved]]` `shape` | `type`: the zone lies on a [surface region](geometry.md#regions); `"sphere"` and `"cylinder"` (3D) get `filled = false`, their surface | 0.9.0 | 0.10.0 |
| `[[write_data]]` `geometry = "<zone>"` | `region = "<zone>"`, and `geometry = "all"` becomes `region = "domain"` ([any region](output.md#snapshots-of-regions)) | 0.9.0 | 0.10.0 |
| `[mesh]` `periodic = ["x", "z"]` | `[[periodic]]` `zones = ["left", "right"]` and `zones = ["back", "front"]` (`y`: `bottom`/`top`, `theta`: `start`/`end`); `[initialize.mesh]` `periodic` likewise becomes `[[initialize.periodic]]` | 0.9.0 | 0.10.0 |

Changes before 0.9.0 were not deprecated first. Since 0.9.0, every rename
of a key or option that a release shipped appears in this table and in `src/common/input_upgrade.cpp`, which is the
source of truth. A test fails if a key is removed without a deprecation
period.

## Unknown and unused keys

Every key in the file must be one the run reads. A misspelled key, a key in
the wrong table, or one the chosen settings do not use (for example
`[numerics.face_reconstruction] order` with `type = "MUSCL"`) stops the run,
listed with its line and the closest key Mallard looked for:

```text
error: input.toml: keys that are unknown or unused with these settings (mallard check lists them all):
  line 8: mesh.Nyy (did you mean mesh.Ny?)
  line 11: mesh.t_stop (did you mean run.t_stop?)
```

When another error stops the setup first, the keys with a suggestion are
listed under it as possible misspellings.

## Error messages

An error in the input names the file, the line and the key, says what is
wrong, lists the accepted values or the closest one, says what to do, and
links the docs that explain the key; `mallard check` reports it the same way:

```text
error: input.toml: line 6: mesh.Nx: is 16 (a float), but must be a non-negative integer. Write it without a decimal point: 16. See docs/input.md#mesh.
error: input.toml: line 20: boundaries[name = "inlet"].type: "wall" is not accepted (did you mean "wall_adiabatic", "wall_heat_flux" or "wall_isothermal"?). Accepted values: ... See docs/input.md#boundaries.
```

A value of the wrong type is always an error, never replaced by the default.
Errors during the run name the step, the time and the cells or faces involved
(see [Divergence](running.md#divergence)).

## Checking an input

`mallard check input.toml` runs the whole setup of the run (mesh,
partition, boundaries, numerics, initial solution) without time stepping and
without writing any file, then reports on it; the exit status is 1 if there
is an error. Run it under `mpirun -n N` to check a run on N ranks.

```text
$ mallard check input.toml --ranks 4 --throughput 5e7 --quality quality
...
== Check ==================================================================================
  Input keys         all used
  Boundaries         every boundary face has a condition (5 conditions)
  non-orthogonality  max 27.7, mean 4.87
                     worst 27.7 at (1.222, 0.3962), 27.7 at (1.209, 0.389)
  skewness           max 0.363, mean 0.0545
                     worst 0.363 at (0.8478, 0.4729), 0.363 at (0.8561, 0.4827)
  aspect ratio       max 1.95, mean 1.13
  volume ratio       max 2.26, mean 1.18
                     worst 2.26 at (0.8694, 0.488), 2.26 at (0.8561, 0.4827)
  Time step          0.0001416 (from run.cfl), CFL 0.25 at the initial state
  CFL-limiting       cell 9,273 at (0.468, 0.05182)
                     2 cells within 10% of it, in (0.4556, 0.05182) to (0.468, 0.05629)
  Memory             72.2MB per rank after setup (peak 72.2MB), 9,385 cells, 1 rank
                     on 4 ranks: about 26.7MB per rank (peak 26.7MB), largest part 2,347 + 1,119 halo cells
  Throughput         50.0M cells/s per rank
  Cost               7060 steps per unit simulated time
                     per unit time: 331 ms wall on 4 ranks, 0.000368 rank-hours
                     per flow-through (0.4375): 145 ms wall on 4 ranks, 0.000161 rank-hours
                     to t_stop = 0.75: 248 ms wall on 4 ranks, 0.000276 rank-hours
  Result             ok
```

(`examples/mallard` on a coarse mesh of 9k triangles, on a CPU.) The report covers:

| Line | What it is |
|---|---|
| Input keys | Unknown or unused keys, with suggestions (errors) |
| Boundaries | Zones without a condition are an error, named with their face count |
| Mesh quality | Largest and mean non-orthogonality, skewness, aspect ratio and volume ratio, and where the worst cells are ([Meshes](meshes.md#mesh-quality)) |
| Time step | The stable step at the initial state (or `run.dt` and its CFL number), the cell that sets it and the region of the cells within 10% of it |
| Resolution | Cell sizes, and the physical scales the case sets against them: Kolmogorov and Taylor lengths, wall units, cells across the flame, each with what it is estimated from, or why it is not ([Resolution](#resolution)) |
| Memory | Kokkos allocations per rank (device memory on GPUs) after setup and at their peak. With `--ranks N` in a serial check, scaled to the owned and halo cells of the largest part the run's partitioner would make. CUDA's own context, a few hundred MB per GPU, is not counted |
| Cost | With `--throughput R` (cells/s per rank, the `Throughput` of a run's summary divided by its ranks) or `--steps N` (timed over N steps after an untimed one): steps, wall time and GPU-hours (rank-hours on CPUs) per unit simulated time, per flow-through time (domain length along the mean initial velocity over its magnitude) and to `t_stop`, assuming the throughput per rank holds on N ranks |
| Warnings | A CFL number above the integrator's limit for smooth flow (1.2 for SSPRK3, 1.35 for RK4), TFLES thickening `n_res Delta / delta_L` above 5, poor cells, a mesh short of the [resolution](#resolution) its regime needs |

### Resolution

The report compares the mesh with the scales the case sets, so that an
under-resolved DNS or wall-resolved LES shows before the run. These are
estimates: each line says what it assumes, a scale the case does not set is
reported as not estimated, and a mesh short of its target is a warning, not an
error. For `examples/channel_retau180` on 24 × 32 × 16 cells, started from a
laminar profile:

```text
  Cell size          V^(1/3) from 0.129 to 0.249, mean 0.218
  Walls              u_tau 0.065 from Dean's channel correlation at the forcing's Re_b = U_b 2h / nu = 5.6e+03 (h = volume / wall area = 1)
                     first-cell y+ up to 1.43 at (0.2618, 1.992, 0.1309), mean 1.43; x+ up to 95.2, z+ up to 47.6
  Kolmogorov         eta 0.0102 from the mean epsilon 0.00422 (the forcing's power, u_tau^2 U_b / h) and nu 0.000357
                     cells of 21.4 eta on average, up to 24.5 at (8.639, 1.057, 1.44)
warning: first cells at y+ up to 1.43 (at (0.2618, 1.992, 0.1309); u_tau from Dean's channel correlation): DNS needs y+ <= 1 at the first cell centers: place them at most 0.0055 from the wall, or model the walls (wall_model)
warning: wall cells up to x+ 95.2 and z+ 47.6 (along the flow and across it): DNS needs x+ <= 15 and z+ <= 8: refine the mesh along the walls
warning: DNS: cells of 21.4 eta on average (eta 0.0102 from the forcing's power, u_tau^2 U_b / h): resolving the dissipation needs about 2 eta or less: refine to cells of 0.0204, or model the small scales with LES ([les])
```

| Scale | Estimated from |
|---|---|
| Cell size | `V^(1/d)` of every cell |
| Friction velocity `u_tau` (walls of viscous cases) | A restart's (or remap's) first-cell velocity, by the linear law or the log law; else, with `source.mass_flow`, Dean's channel correlation `C_f = 0.073 Re_b^(-1/4)` at the bulk velocity `U_b` and `h` = volume / wall area (the half-height of a channel); else a flat plate's mean skin friction (laminar, or turbulent above `Re_L = 5e5`) over the walls' length along the mean flow at the largest initial speed |
| y+, x+, z+ | The first cell center's distance to the wall, and the wall face's extents along the flow and across it, in wall units `nu / u_tau` |
| Wall model's exchange point | `exchange_distance`, or `exchange_cells` wall-cell heights, in wall units |
| Dissipation `epsilon` | With `source.mass_flow`, the forcing's power `u_tau^2 U_b / h`; else the viscous dissipation of the initial (or restart) velocity field, mass-weighted over the domain |
| Kolmogorov length `eta` | `(nu^3 / epsilon)^(1/4)` with the domain's mean `nu` |
| Taylor microscale `lambda` | `(15 nu u'^2 / epsilon)^(1/2)`, `u'` the velocity's fluctuation about its mean (from the field, not the forcing) |
| Flame thickness | `les.combustion.delta_L` (TFLES): cells across it. Otherwise not estimated |

The targets the warnings use, by regime (LES when `[les]` sets a model, DNS
otherwise; walls with a `wall_model` are wall-modeled):

| Regime | Target |
|---|---|
| DNS | Cells of at most about 2 `eta` on average; first cells at `y+ <= 1`, wall cells at `x+ <= 15`, `z+ <= 8` (about Kim, Moin & Moser, 1987) |
| Wall-resolved LES | First cells at `y+ <= 1`, wall cells at `x+ <= 150`, `z+ <= 40` (the coarse ends of Georgiadis, Rizzetta & Fureby, 2010) |
| Wall-modeled LES | Exchange points in the log layer, `y+ >= 30` |
| TFLES | Thickening `F = n_res Delta / delta_L` at most 5 |

`--quality PREFIX` also writes the estimates' fields with the mesh-quality
ones: `Y_PLUS` on wall cells, `DELTA_OVER_ETA` (cell size over the mean
Kolmogorov length) and `CELLS_PER_DELTA_L` (TFLES), to see in ParaView where
the mesh is short. The mean `eta` stands for the whole domain, so
`DELTA_OVER_ETA` is pessimistic where the flow dissipates less than its mean
(far from walls, in a free stream).

### The key catalog

`--catalog keys.json` also writes, as JSON, every key the setup looked up for
this input: whether the input set it, the type the setup reads it as, its
physical dimension (see [units](#units)), its default (as TOML) or that it is
required, and the values it accepts, by path
(`numerics.riemann_solver`, `boundaries.type`). It is written even when the
setup stops on an error, with the keys read up to there. The setup's accessors
(`find_value`, `find_value_or`, `find_real_or` and `find_option` in
`src/common/input.h`) record these, so the catalog is what the code does.
`tools/dev/input_catalog.py build` merges the catalogs of every example and
tutorial, in 2D and 3D, into
[`docs/input-keys.json`](../input-keys.json), which the website's input builder
reads; CI checks that it is up to date, and `tools/dev/input_catalog.py docs`
that the defaults this reference states match it.

## The resolved input

An input sets the keys that matter for its case; the run uses many more, at
their defaults or as a [numerics preset](numerics.md#presets) fills them.
`mallard input resolve` writes the input as the run uses it:

```sh
mallard input resolve input.toml > full.toml
```

`full.toml` has every key the setup reads, each with the value the run takes:

- defaults and the keys of a preset written out, renamed keys under their new
  names ([inputs from older releases](#inputs-from-older-releases));
- quantities with a unit as plain numbers in the key's units (`t_stop = "1 min"`
  becomes `t_stop = 60.0`), and numbers written so they read back exactly;
- keys sorted, comments dropped.

Keys the setup does not read are left out, and so are tables the input leaves
out, since a table's presence can turn a feature on (`[integrals]`,
`[statistics]`). A key whose
absence is a choice of its own (one of alternative keys such as `X` and `Y`, a
`[[write_data]]` `format` the build picks) stays absent too. `mallard -i
full.toml` runs bitwise the same as `input.toml`, and resolves to `full.toml`
again, so comparing two resolved inputs (`diff`) shows every setting in which
two runs differ.

Every output file's [provenance](output.md#provenance) holds the same text,
`input.resolved`, next to the input file's own text: the results of a run carry
the input that reproduces them. The resolved input, the key catalog above and
the preset's keys in the provenance are all made from the same record, the
setup's accessors, so they cannot disagree.

## The other commands

Every command's options, examples and exit status: [Command line](cli.md).

| Command | Purpose | Guide |
|---|---|---|
| `mallard input upgrade old.toml` | Rewrites renamed, moved and removed keys of an input from an older release | [Inputs from older releases](#inputs-from-older-releases) |
| `mallard input resolve input.toml` | The input as the run uses it, every key explicit | [The resolved input](#the-resolved-input) |
| `mallard reactor -i input.toml` | 0D constant-volume reactor with the solver's chemistry; chemistry benchmarks | [Thermochemistry](thermochemistry.md#0d-reactors-mallard-reactor) |
| `mallard mesh convert IN OUT.h5` | Gmsh file, or the mesh of an input file, to Mallard's HDF5 mesh format (builds with HDF5) | [Meshes](meshes.md#hdf5-meshes-for-large-runs) |
| `mallard test` | The unit and regression tests: `mallard-test2d` and `mallard-test3d` (`build/test/mallard-test2d`), from a build or an install tree; `--dim 2\|3` runs one | [Installation](https://matthewbonanni.github.io/mallard/docs/installation/) |

The Python scripts in `tools/` prepare meshes and restart files and post-process
output; [Tools](https://matthewbonanni.github.io/mallard/docs/tools/) lists them.
