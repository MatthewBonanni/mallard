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

The quickest way to a working input is a template: `mallard list templates`
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
fields = ["RHO", "U_X", "P", "T"]
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
error E-INPUT: input.toml: line 14: boundaries.p: "300 K" has the dimension of temperature (K). Expected pressure (kg m^-1 s^-2), e.g. in Pa, kPa, MPa, bar, atm, psi; a number without a unit is in SI. See docs/guide/input_files.md#units.
  docs/guide/input_files.md#e-input
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
(`p_ref = T_ref = rho_ref = 1`) work as plain numbers. Say so with
`units = "nondimensional"` in `[physics]`: the terminal output then states it
and labels no number with an SI unit (the progress table's units row shows
`-`), instead of calling a time of 0.2 "0.2 s". Units in the input make no
sense then; write them only in dimensional inputs. Gas mixtures read real
thermochemistry and are always dimensional, so `units = "nondimensional"` is
an error with `gas = "mixture"`.

```toml
[physics]
type = "euler"
units = "nondimensional"
gamma = 1.4
p_ref = 1.0
T_ref = 1.0
rho_ref = 1.0
```

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
| `[[write_data]]` `variables`, `[[probes]]` `variables` | `fields`, as `[statistics]` names them ([`mallard list fields`](cli.md#mallard-list-fields)) | 0.10.0 | 0.11.0 |
| `[numerics]` `preset = "robust"`, `"accurate"` | `"second_order"`, `"high_order"`: presets are named by their order ([presets](numerics.md#presets)) | 0.10.0 | 0.11.0 |

A changed default is not a deprecated key, and `mallard input upgrade`
leaves the file as it is: it only prints a note, which does not change the
exit status of `--check`. In 0.10.0 the default without a `[numerics] preset`
or `[numerics.face_reconstruction]` became the `high_order` preset, not first
order; add `[numerics.face_reconstruction]` `type = "FO"` to reproduce older
first-order results. Before 1.0, these rules keep old inputs running, not
their results identical.

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
error E-KEY-UNUSED: input.toml: keys that are unknown or unused with these settings (mallard check lists them all):
  line 8: mesh.Nyy (did you mean mesh.Ny?)
  line 11: mesh.t_stop (did you mean run.t_stop?)
  docs/guide/input_files.md#e-key-unused
```

When another error stops the setup first, the keys with a suggestion are
listed under it as possible misspellings.

## Error messages

An error in the input names the file, the line and the key, says what is
wrong, lists the accepted values or the closest one, says what to do, and
links the docs that explain the key; `mallard check` reports it the same way:

```text
error E-INPUT: input.toml: line 6: mesh.Nx: is 16 (a float), but must be a non-negative integer. Write it without a decimal point: 16. See docs/input.md#mesh.
  docs/guide/input_files.md#e-input
error E-INPUT: input.toml: line 20: boundaries[name = "inlet"].type: "wall" is not accepted (did you mean "wall_adiabatic", "wall_heat_flux" or "wall_isothermal"?). Accepted values: ... See docs/input.md#boundaries.
  docs/guide/input_files.md#e-input
```

A value of the wrong type is always an error, never replaced by the default.
Errors during the run name the step, the time and the cells or faces involved
(see [Divergence](running.md#divergence)).

## Checking an input

`mallard check input.toml` runs the whole setup of the run (mesh,
partition, boundaries, numerics, initial solution) without time stepping and
without writing any file, then reports on it; the exit status is 1 if there
is an error. Run it under `mpirun -n N` (or `srun`) to check a run on N ranks.

`mallard check` without an input checks the environment only: what a run
would run on, with what to fix. Under the launcher of a real job it checks
that job's ranks, nodes and devices before the run waits in the queue again:

```sh
mpirun -n 8 mallard check                # the environment of 8 ranks
mallard check --format json | python3 -c "import json, sys; print(json.load(sys.stdin)['environment']['build'])"
```

```text
$ mallard check input.toml --ranks 4 --throughput 5e7 --memory a100-80 --mesh-fields mesh --view view
...
== Check ================================================================================ ok
  Result: ok, 0 errors and 0 warnings (exit status 0)
  Input: input.toml, set up on 1 rank
  Units: SI (m, s, kg, K) wherever a column or a value names none; bytes with decimal
  prefixes (1 GB = 10^9 B)

== Input ================================================================================ ok
  status  check                   result
  ok      setup                   completed: mesh, partition, boundaries, numerics, initial state
  ok      unknown or unused keys  none: every key is used
  ok      deprecated keys         none

== Boundaries =========================================================================== ok
  Every boundary face has a condition: 5 zones (periodic zones are paired, not listed).
  status  zone    condition             faces  details
  ok      left    dirichlet             240    -
  ok      right   outlet_extrapolation  240    -
  ok      bottom  dirichlet             40     where x < 1/6
  ok      bottom  symmetry              920    where x >= 1/6
  ok      top     dirichlet             960    -

== Regions ============================================================================== ok
  The mesh's own (the domain, its boundary and cell zones) and every region the case uses;
  cells: the cells each touches. A region of the case that touches no cell is an error.
  status  region  kind    measure  units  cells    made from
  ok      domain  filled  4        m^2    460,800  the mesh's
  ok      bottom  curve   4        m      960      the mesh's
  ok      left    curve   1        m      240      the mesh's
  ok      right   curve   1        m      240      the mesh's
  ok      top     curve   4        m      960      the mesh's
  Case view (every region and zone, for ParaView): view/case.vtm

== Mesh quality ========================================================================= ok
  Over every cell; a cell above the limit is questionable.
  status  metric             units          max       mean      limit  cells over limit
  ok      non-orthogonality  deg            26.6      26.6      70     0
  ok      skewness           dimensionless  1.43e-13  2.85e-14  0.5    0
  ok      aspect ratio       dimensionless  1.58      1.58      1000   0
  ok      volume ratio       dimensionless  1         1         10     0

  Worst cells of the metrics at or above their notable value (non-orthogonality 10 deg,
  skewness 0.05, aspect ratio 10, volume ratio 2), worst first:
  metric             value  units  cell (global id)  x [m]   y [m]
  non-orthogonality  26.6   deg    59,999            0.5181  0.9986
  non-orthogonality  26.6   deg    67,199            0.5806  0.9986
  non-orthogonality  26.6   deg    74,399            0.6431  0.9986
  non-orthogonality  26.6   deg    81,599            0.7056  0.9986
  non-orthogonality  26.6   deg    88,799            0.7681  0.9986
  Quality fields: quality_000000.vtu

== Time step ============================================================================ ok
  At the initial state, t = 0 s.
  status  quantity                      value      units          target  basis
          time step dt                  0.0001098  s                      run.cfl times the time step at CFL 1
  ok      CFL number                    1          dimensionless  <= 1.2  SSPRK3's stability limit for smooth flow
          time step at CFL 1            0.0001098  s                      the smallest over the cells
          CFL-limiting cell             57,397     global id              at (0.4972, 0.5778) m
          cells within 10% of its step  52,499     cells                  in (0.001389, 0.001389) m to (0.7431, 0.9986)
                                                                          m

== Resolution =========================================================================== ok
  Estimates from the initial state, against the targets of the case's regime (DNS, or LES
  with [les]); value -: not estimated, for the reason its basis gives. See
  docs/guide/input_files.md#resolution.
  status  quantity                       value    units  basis
          cell size V^(1/2), smallest    0.00295  m      over every cell
          cell size V^(1/2), mean        0.00295  m      volume-weighted
          cell size V^(1/2), largest     0.00295  m      over every cell, at (3.803, 0.2597) m
          Kolmogorov length, wall units  -               not estimated: inviscid

== Memory =============================================================================== ok
  Kokkos allocations in host memory per rank, of the rank with the largest peak (with
  --ranks from a serial check, scaled to the largest part the partitioner makes); peak:
  during the setup.
  status  run         ranks  owned cells  halo cells  after setup  peak     peak of the device
          this check  1      460,800      0           3.96 GB      3.96 GB  -
          --ranks 4   4      115,200      7,680       1.06 GB      1.06 GB  -

  By category, of the same rank; per cell: per owned and halo cell of the rank (460,800);
  categories below 0.05% of the peak are left out.
  category               at the setup's peak  per cell  after setup  per cell  proportional to the cells
  solution and stages    95.8 MB              208 B     95.8 MB      208 B     yes
  faces and gradients    658 MB               1.43 kB   658 MB       1.43 kB   yes
  reconstruction tables  3.07 GB              6.66 kB   3.07 GB      6.66 kB   yes
  mesh and geometry      127 MB               275 B     127 MB       275 B     yes
  boundaries and models  5.62 MB              12 B      5.62 MB      12 B      yes
  setup                  3.43 MB              -         0 B          -         no
  total                  3.96 GB              -         3.96 GB      -

== GPU sizing =========================================================================== ok
  The largest rank's predicted peak (the larger of the setup's and the steady state's, plus
  the memory outside Kokkos) against the memory per GPU, for --memory or this build's GPU.
  status  quantity                            value      units                 basis
          memory per GPU                      85.1       GB                    A100-80GB
          kept free                           5          % of the memory       fragmentation and allocations during the
                                                                               run
          outside Kokkos per GPU              1.60       GB                    typical: CUDA's context and kernels'
                                                                               local memory on an A100
  ok      most cells per GPU                  9,226,381  owned and halo cells  8.59 kB per cell + 3.43 MB, at the
                                                                               setup's peak
  ok      fewest GPUs                         1          GPUs                  largest part 460,800 owned + 0 halo
                                                                               cells, predicted peak 5.56 GB (from the
                                                                               partitioner)
          most GPUs with halo < 10% of owned  4          GPUs                  doubling from the fewest
          wall time per unit simulated time   21.0       s                     on 4 GPUs at 50.0M cells/s each

== Cost ===================================================================================
  On 4 ranks at 50.0M cells/s per rank (--throughput), at the initial time step; rank-hours:
  ranks times wall hours.
  period                   simulated time [s]  steps   wall time  rank-hours
  per unit simulated time  1                   9,109   21.0 s     0.0233
  per flow-through time    1.135               10,340  23.8 s     0.0265
  to run.t_stop            0.2                 1,822   4.20 s     0.00466

== Diagnostics ========================================================================== ok
  None.

  Result: ok, 0 errors and 0 warnings (exit status 0)
```

(`examples/double_mach`: 460,800 quadrilaterals with TENO5, checked on a CPU in a build without HDF5 (hence `.vtm` and `.vtu`) and sized for an A100-80GB. The setup's own sections come before the report and are left out here.)

The report opens and closes with the result, and the exit status is 1 if there is an error. Each section's heading ends with its status, the worst of its rows. Each row of a table starts with its status: `ok` (checked and fine), `WARN`, `ERROR`, or blank (information, nothing to check). The status column and the headings alone show every problem. Every number states its units: in its column's header (`x [m]`), in the `units` column of a table that has one (where every value is bare: `21.0` with `s` beside it), or else in the value itself (`3.96 GB`, `21.0 s`), never two of these. Dimensionless numbers say `dimensionless`, or `wall units` when they are relative to `nu / u_tau`. Numbers are SI, and bytes use decimal prefixes (1 GB = 10<sup>9</sup> B). Each section says what its numbers are of: which rank, which state, against which limit.

| Section | What it is |
|---|---|
| Check | The result (errors, warnings, exit status), the input, the ranks it was set up on, the units |
| Input | Whether the setup completed, and how many keys are unknown or unused (errors) or deprecated (warnings): each is a diagnostic |
| Boundaries | Every zone with its condition, its faces (over every rank) and how the condition varies or what models it. A zone without a condition stops the setup (`E-INPUT`) |
| Regions | Every region the case uses, and the mesh's own (the domain and its zones): kind (filled, surface, curve, points), measure with its units, the cells it touches, and how it is made (the mesh's, cell fractions, exact, or extracted at a resolution). A region that touches no cell is an error |
| Mesh quality | The largest and mean non-orthogonality, skewness, aspect ratio and volume ratio over every cell, against their limits ([Meshes](meshes.md#mesh-quality)). For a metric at or above its notable value, its 5 worst cells with their global ids and centers |
| Time step | At the initial state (its time `t`): the time step and what sets it (`run.cfl`, `run.dt`, or the sponges), its CFL number against the integrator's limit for smooth flow (1.2 for SSPRK3, 1.35 for RK4), the time step at CFL 1, the cell that sets it, and the box of the cells within 10% of its time step |
| Resolution | Cell sizes, and the physical scales the case sets against them. Kolmogorov and Taylor lengths, wall units, cells across the flame, and TFLES's thickening, each with its basis and its regime's target ([Resolution](#resolution)) |
| Memory | Kokkos allocations per rank (device memory on GPUs) of the rank with the largest peak: after setup, once the data only the setup reads is freed (what the time steps hold), and at the peak (the setup's, and the timed steps' with `--steps`), with that rank's owned and halo cells. With `--ranks N` in a serial check, a second row is scaled to the owned and halo cells of the largest part the run's partitioner would make. On a GPU, the peak as a share of the device's memory. Then the same rank by category, at the setup's peak and after setup (at the steps' peak with `--steps`), in total and per owned and halo cell; the setup's workspace and work memory sized by the device are not proportional to the cells. On GPUs, also the device memory outside Kokkos: CUDA's context and the local memory of kernels with large stacks, 1.5 GB (2D) to 2.1 GB (3D) on an A100 |
| GPU sizing | With `--memory SIZE` or `--memory GPU` (by default the GPU's own memory in GPU builds; `Rank sizing` in CPU builds): the memory, the 5% kept free and the memory outside Kokkos per GPU, the most owned and halo cells per GPU, the fewest GPUs the mesh fits on (from the parts the run's partitioner makes, in a serial check) with the largest part's predicted peak, up to how many GPUs the halo stays below 10% of the owned cells, and the wall time per unit simulated time there. The peak is the larger of the setup's and the steady state's. The memory outside Kokkos is measured on the GPU with `--steps` (the step kernels reserve their local memory at their first launch), else it is the typical 1.6 GB (2D) or 2.2 GB (3D). On an A100 the prediction is within -1.0% to +3.3% of the measured peak, extrapolating CPU checks of 4 to 16 times fewer cells ([performance](../design/performance.md#8-memory-sizing-in-mallard-check-550)) |
| Cost | With `--throughput R` (cells/s per rank, the `Throughput` of a run's summary divided by its ranks) or `--steps N` (timed over N steps after an untimed one): simulated time, steps, wall time and GPU-hours (rank-hours on CPUs) per unit simulated time, per flow-through time (domain length along the mean initial velocity over its magnitude), to `run.t_stop` and for `run.n_steps`. It assumes the throughput per rank and the initial time step hold on N ranks |
| Environment | What the run runs on, with or without an input: the build (dimension, precision, build type, compiler, from the same `build` block as every output file's [run block](output.md#run-block)) and the solvers installed with it; Kokkos and its execution spaces; MPI (the library, ranks, hosts, the most ranks on one host); GPU-aware MPI (built or not, whether the MPI library says it reads device memory, whether this run uses it); NCCL (built or not, whether halo exchanges use it: `[parallel] halo_exchange`, on several ranks); HDF5 (serial or parallel), CGNS, zlib and KaMinPar. Then a row per rank: host, device (GPU and memory, else the CPU), the GPU's index among those it sees, its CPUs, threads and whether OpenMP binds them. The placement checks are the run's own (`Ranks` at the start of a run on several ranks) |
| Diagnostics | Every error, then every warning: its code, what is wrong with its numbers and the fix, then the anchor below that explains it |

### Diagnostics

Each error and warning has a stable code that scripts can match. Errors (`E-`) make the exit status 1; warnings (`W-`) do not. A run reports the same codes, and those of its own warnings and errors ([Warnings and errors](running.md#warnings-and-errors)).

| Code | What it means | What to do |
|---|---|---|
| <a id="e-input"></a>`E-INPUT` | The setup stopped on an invalid input: a syntax error, a value of the wrong type or dimension, a missing key, a zone without a condition. The message names the key and its line, what it accepts, and the docs of the key | Fix the key; the report covers only what the setup got to |
| <a id="e-setup"></a>`E-SETUP` | The setup stopped on another error: a file it cannot read, a mesh it cannot partition, memory it cannot allocate | The message says what failed |
| <a id="e-key-unused"></a>`E-KEY-UNUSED` | A key that no setting reads: misspelled, in the wrong table, or of an option not chosen. The message gives its line and the closest key the setup looked up | Fix or remove it. A run rejects the same keys |
| <a id="w-key-deprecated"></a>`W-KEY-DEPRECATED` | A key renamed or replaced in a later release, still read for now | `mallard input upgrade input.toml` rewrites it ([Inputs from older releases](#inputs-from-older-releases)) |
| <a id="e-catalog-write"></a>`E-CATALOG-WRITE` | `--catalog FILE` cannot be written | Check the path and its permissions |
| <a id="e-ranks"></a>`E-RANKS` | `--ranks N` in a check run under `mpirun` on another number of ranks | Check serially (memory is then scaled to N ranks) or on N ranks |
| <a id="e-memory-device"></a>`E-MEMORY-DEVICE` | The estimated peak per rank exceeds the device's memory | Run on more ranks or GPUs, or on a coarser mesh; GPU sizing says how many fit |
| <a id="e-region-empty"></a>`E-REGION-EMPTY` | A region the case uses overlaps the domain's bounding box but touches no cell: too thin, or in a hole of the mesh | Enlarge or move it; `--view` shows it with the mesh's zones ([Regions on a mesh](geometry.md#regions-on-a-mesh)) |
| <a id="e-region-outside"></a>`E-REGION-OUTSIDE` | A region the case uses lies outside the domain. The message gives both extents | Move it inside: often a unit or a sign ([Regions on a mesh](geometry.md#regions-on-a-mesh)) |
| <a id="w-quality-non-orthogonality"></a>`W-QUALITY-NON-ORTHOGONALITY`, <a id="w-quality-skewness"></a>`W-QUALITY-SKEWNESS`, <a id="w-quality-aspect-ratio"></a>`W-QUALITY-ASPECT-RATIO`, <a id="w-quality-volume-ratio"></a>`W-QUALITY-VOLUME-RATIO` | Cells above the metric's limit (70 deg, 0.5, 1000, 10). Such cells limit the time step and the accuracy, and can make a run diverge | Improve the mesh where they are: `--mesh-fields PREFIX` writes the metrics as fields ([Mesh quality](meshes.md#mesh-quality)) |
| <a id="e-mesh-fields"></a>`E-MESH-FIELDS` | `--mesh-fields` could not write its files, or `--cells` or `--region` selects no cell: an id beyond the mesh's cells, a region that touches none | Check the path, the ids (global, from 0) and the region ([Mesh and stencil fields](output.md#mesh-and-stencil-fields)) |
| <a id="w-cfl-limit"></a>`W-CFL-LIMIT` | The CFL number at the initial state is above the time integrator's stability limit for smooth flow (1.2 for SSPRK3, 1.35 for RK4) | Lower `run.cfl`, or `run.dt` |
| <a id="w-tfles-thickening"></a>`W-TFLES-THICKENING` | TFLES thickens the flame by more than 5 (`F = n_res Delta / delta_L`). Beyond that the flame's response to turbulence is unreliable | Refine the mesh where the flame goes, or lower `les.combustion.n_res` ([LES](les.md#thickened-flames-lescombustion)) |
| <a id="w-wall-y-plus"></a>`W-WALL-Y-PLUS` | First cells of walls without a wall model above y+ 1, which DNS and wall-resolved LES need. The message gives the first-cell distance to aim for | Refine toward the walls, or model them (`wall_model`) ([Resolution](#resolution)) |
| <a id="w-wall-spacing"></a>`W-WALL-SPACING` | Wall cells longer along the flow (x+) or wider across it (z+) than the regime needs: 15 and 8 for DNS, 150 and 40 for wall-resolved LES | Refine along the walls ([Resolution](#resolution)) |
| <a id="w-wall-model-exchange"></a>`W-WALL-MODEL-EXCHANGE` | A wall model's exchange points reach below the log layer (y+ 30) that its law assumes | Raise `exchange_distance` or `exchange_cells` |
| <a id="w-dns-eta"></a>`W-DNS-ETA` | A case without `[les]` (DNS) whose mean cell is more than 2 Kolmogorov lengths. The dissipation is not resolved | Refine to the cell size the message gives, or model the small scales (`[les]`) ([Resolution](#resolution)) |
| <a id="w-sizing-fixed"></a>`W-SIZING-FIXED` | With `--memory` (or a GPU's own), even the memory that does not grow with the cells does not fit in the memory per rank | Size for a larger memory |
| <a id="w-sizing-parts"></a>`W-SIZING-PARTS` | No partition into a few thousand parts fits in the memory per rank | Size for a larger memory, or coarsen the mesh |
| <a id="e-env-gpu-aware-mpi"></a>`E-ENV-GPU-AWARE-MPI` | The build passes device buffers to MPI (`-DMallard_GPU_AWARE_MPI=ON`), but the MPI library says it cannot read device memory (Open MPI's `MPIX_Query_cuda_support`, `MPIX_Query_rocm_support`): the run would crash or read garbage | Load an MPI built with CUDA or ROCm support, or rebuild with `-DMallard_GPU_AWARE_MPI=OFF` ([MPI](parallel.md#mpi)) |
| <a id="w-env-host-staging"></a>`W-ENV-HOST-STAGING` | A GPU run on several ranks whose halo exchanges copy device buffers to host memory and back: neither NCCL (`[parallel] halo_exchange = "nccl"`) nor GPU-aware MPI | Use NCCL, or rebuild with `-DMallard_ENABLE_NCCL=ON` or `-DMallard_GPU_AWARE_MPI=ON` |
| <a id="w-env-placement"></a>`W-ENV-PLACEMENT` | Ranks placed so that they waste hardware: several ranks on a GPU, more ranks than GPUs on a node, an idle GPU while others are shared, ranks on CPUs away from their GPU's NUMA node; on CPUs, more threads than cores on a node, or pinned threads of ranks that share cores | One rank per GPU, each bound to the cores next to it ([Running on clusters](parallel.md#running-on-clusters)) |
| <a id="w-env-thread-binding"></a>`W-ENV-THREAD-BINDING` | OpenMP builds: ranks with several threads but neither `OMP_PROC_BIND` nor `OMP_PLACES` set, so their threads may migrate between cores | Set `OMP_PROC_BIND=spread` and `OMP_PLACES=threads` |

### The report as JSON

`--format json` prints the same report as one JSON object on stdout, and nothing else; the setup's sections are left out. The exit status is the same. Its `version` is 1: a version gains members, and changes only when an existing member changes meaning. Numbers are SI (bytes, seconds, meters) and read back exactly; a number that is not estimated is `null`, and a section that was not reached is `null`.

| Member | Holds |
|---|---|
| `version`, `status` (`ok`, `warning`, `error`), `errors`, `warnings`, `input_file` (`null` without an input), `setup_done`, `dim` | The result |
| `diagnostics` | `[{severity, code, message, docs}]`, in the order found |
| `unused_keys` | `[{key, line, suggestion}]` (`null` where unknown) |
| `boundaries` | `[{zone, condition, faces, details, turbulence}]` |
| `regions`, `case_view` | `[{name, of_mesh, dimension, measure, per_radian, cells, exact, resolution}]`; the case view's file |
| `mesh_quality`, `mesh_fields_file`, `stencil_file` | By field name (`NON_ORTHOGONALITY`, ...): `{name, max, mean, units, limit, above_limit, worst: [{value, cell, x}]}`; the files of `--mesh-fields` |
| `time_step` | `t`, `dt`, `dt_cfl1`, `cfl`, `cfl_limit`, `integrator`, `set_by` (`CFL`, `fixed`, `sponge`), `limiting_cell`, `limiting_x`, `region_cells`, `region_lo`, `region_hi` |
| `resolution` | By id (`cell_size_min`, `u_tau`, `y_plus_max`, `eta`, `delta_over_eta_mean`, `tfles_thickening_max`, ...): `{quantity, value, units, target, status, basis, x}` |
| `memory` | Bytes of the largest rank: `bytes`, `bytes_peak`, `local_cells`, `local_owned`, the `target_*` of `--ranks`, `device_capacity`, `outside_kokkos`, and by category `at_setup_peak` and `steady` |
| `sizing` | `name`, `gpu`, `memory`, `margin`, `outside_kokkos`, `bytes_per_cell`, `fixed_bytes`, `max_local_cells`, `min_ranks`, `min_owned`, `min_halo`, `min_peak`, `halo_ranks`, ... |
| `environment` | `build` and `dependencies` (the run block's), `solvers` (the dimensions installed), `ranks`, `hosts`, `ranks_per_host`, `gpu_aware_mpi` (`library`: what the MPI library says, `null` if it does not; `used`), `nccl` (`used`), `placement`: `[{rank, host, device, threads, cpus, omp_bind, gpu, gpu_index, gpu_numa, gpus_visible}]` |
| `cost` | `cells`, `ranks`, `throughput` (cells/s per rank), `wall_per_time` (wall seconds per unit simulated time), `flow_through_time`, `t_stop`, `n_steps` |

```sh
mallard check input.toml --format json | python3 -c "import json, sys; r = json.load(sys.stdin); print(r['status'], [d['code'] for d in r['diagnostics']])"
```

### Resolution

The report compares the mesh with the scales the case sets, so that an
under-resolved DNS or wall-resolved LES shows before the run. These are
estimates: each line says what it assumes, a scale the case does not set is
reported as not estimated, and a mesh short of its target is a warning, not an
error. For `examples/channel_retau180` on 24 × 32 × 16 cells, started from a
laminar profile:

```text
== Resolution ========================================================================= WARN
  Estimates from the initial state, against the targets of the case's regime (DNS, or LES
  with [les]); value -: not estimated, for the reason its basis gives. See
  docs/guide/input_files.md#resolution.
  status  quantity                      value     units          target       basis
          cell size V^(1/3), smallest   0.129     m                           over every cell
          cell size V^(1/3), mean       0.218     m                           volume-weighted
          cell size V^(1/3), largest    0.249     m                           over every cell, at (8.639, 1.057, 1.44) m
          friction velocity u_tau       0.065     m/s                         root mean square over the walls' area,
                                                                              from Dean's channel correlation at the
                                                                              forcing's Re_b = U_b 2h / nu = 5.6e+03 (h
                                                                              = volume / wall area = 1)
  WARN    first-cell y+, largest        1.43      wall units     <= 1 (DNS)   first cell centers of the walls without a
                                                                              wall model, at (0.2618, 1.992, 0.1309) m
          first-cell y+, mean           1.43      wall units                  area-weighted over every wall
  WARN    wall-cell x+, largest         95.2      wall units     <= 15 (DNS)  wall faces' extent along the flow
  WARN    wall-cell z+, largest         47.6      wall units     <= 8 (DNS)   wall faces' extent across the flow
          dissipation epsilon, mean     0.00422   m^2/s^3                     the forcing's power, u_tau^2 U_b / h
          kinematic viscosity nu, mean  0.000357  m^2/s                       the mean mu over the mean rho
          Kolmogorov length eta         0.0102    m                           (nu^3 / epsilon)^(1/4)
  WARN    cell size / eta, mean         21.4      dimensionless  <= 2 (DNS)   the volume-weighted mean cell size over
                                                                              eta
          cell size / eta, largest      24.5      dimensionless               the largest cell size over eta, at (8.639,
                                                                              1.057, 1.44) m
...
== Diagnostics ======================================================================== WARN
  status  code            message, then the docs that explain it
  WARN    W-WALL-Y-PLUS   first cells at y+ up to 1.43 (at (0.2618, 1.992, 0.1309) m; u_tau from Dean's channel
                          correlation): DNS needs y+ <= 1 at the first cell centers: place them at most 0.0055 m from
                          the wall, or model the walls (wall_model)
                          docs/guide/input_files.md#w-wall-y-plus
  WARN    W-WALL-SPACING  wall cells up to x+ 95.2 and z+ 47.6 (along the flow and across it): DNS needs x+ <= 15 and z+
                          <= 8: refine the mesh along the walls
                          docs/guide/input_files.md#w-wall-spacing
  WARN    W-DNS-ETA       DNS: cells of 21.4 eta on average (eta 0.0102 m from the forcing's power, u_tau^2 U_b / h):
                          resolving the dissipation needs about 2 eta or less: refine to cells of 0.0204 m, or model the
                          small scales with LES ([les])
                          docs/guide/input_files.md#w-dns-eta
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

`--mesh-fields PREFIX` also writes the estimates' fields with the mesh-quality
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
that every key this reference documents is in it and that the defaults it states
match it.

`mallard list keys` shows that catalog as tables, one per input table, with
each key's type, units, default and accepted values; `mallard list keys
input.toml` shows the catalog of one case (what `--catalog` writes, with
whether the input sets each key), and a section keeps one table or key path:

```bash
mallard list keys numerics                  # every [numerics] key there is
mallard list keys write_data input.toml     # the [[write_data]] keys this case reads
```

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

Every output file's [run block](output.md#run-block) holds the same text,
`input.resolved`, next to the input file's own text: the results of a run carry
the input that reproduces them. The resolved input, the key catalog above and
the preset's keys in the run block are all made from the same record, the
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
