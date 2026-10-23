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

[[write_data]]                  # VTU snapshots for ParaView
prefix = "solut/sod"
format = "vtu"
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
| `[[write_data]]`, `[output]`, `[[forces]]`, `[integrals]`, `[statistics]`, `[[probes]]` | no | [Output and diagnostics](output.md) |
| `[parallel]` | no | [Running in parallel](parallel.md) |

Double brackets (`[[boundaries]]`) are TOML arrays of tables: repeat them, one
entry each. Every key, with its default, is in the
[input reference](../input.md).

## Numbers, vectors and expressions

- **Numbers.** Integers are accepted wherever a real number is expected
  (`cfl = 1` works).
- **Vectors** have one component per spatial dimension: `u = [1.0, 0.0]` in
  the 2D solver, `u = [1.0, 0.0, 0.0]` in the 3D one (the dimension is fixed
  when Mallard is built, see [Installation](https://matthewbonanni.github.io/mallard/docs/installation/)).
- **Expressions.** Initial states, boundary states of `dirichlet`, `upt` and
  `nscbc_inlet`, sources and sponges can be strings in
  [exprtk](https://www.partow.net/programming/exprtk/) syntax of `x`, `y`, `z`
  (0 in 2D) and, where noted, the time `t`: `"x < 0.5 ? 1.0 : 0.125"`,
  `"exp(-((x - 1)^2 + y^2) / 0.01)"`. exprtk also has local variables
  (`var r := sqrt(x^2 + y^2); ...`), `and`, `or`, `if`, the usual functions
  and `pi`. Multi-line strings use TOML's `"""` with a trailing `\`.
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
mallard upgrade-input old.toml > new.toml     # changes listed on stderr
mallard upgrade-input --check old.toml        # exit status 1 if anything is out of date
```

It changes only the lines of those keys. A renamed key keeps its value and its
comment, a moved key goes to the end of its new table, and a removed key is
commented out with a note. A file with nothing to change comes back
unchanged, byte for byte. When a changed key is written as a dotted key
(`run.cfl = 1`) or in an inline table, the file is reformatted instead: its
comments are kept, its layout is not. The command is tested on the input
files of every release (`tools/check_upgrade_input.py`).

Keys and options changed so far:

| Old | New | Deprecated | Removed |
|---|---|---|---|
| `[numerics.face_reconstruction]` `basis_order = p` | `order = p + 1` (order of accuracy) | 0.2.0 | 0.2.0 |
| `basis_type`, `max_stencil_size_factor`, `quadrature_type`, `quadrature_order`, `quadrature_type_cell`, `quadrature_order_cell`, `quadrature_type_face`, `quadrature_order_face` (same table) | none: TENO-E sets them | 0.2.0 | 0.2.0 |
| `[physics]` `p_min`, `p_max` | none: pressures are not clipped | 0.2.0 | 0.2.0 |
| `[[write_data]]` `format = "tecplot"` | `"vtu"` | 0.2.0 | 0.2.0 |

Changes before 0.9.0 were not deprecated first. Since 0.9.0, every rename
appears in this table and in `src/common/input_upgrade.cpp`, which is the
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

## Checking an input

`mallard check input.toml` runs the whole setup of the run (mesh,
partition, boundaries, numerics, initial solution) without time stepping and
without writing any file, then reports on it; the exit status is 1 if there
is an error. Run it under `mpirun -n N` to check a run on N ranks.

```text
$ mallard check input.toml --ranks 4 --throughput 5e7 --quality-vtu quality
...
== Check ==================================================================================
  Input keys         all used
  Boundaries         every boundary face has a condition (5 conditions)
  non-orthogonality  max 23.4, mean 4.66
                     worst 23.4 at (1.428, 0.09977), 23.4 at (1.448, 0.09361)
  skewness           max 0.374, mean 0.0489
                     worst 0.374 at (1.025, -0.6632), 0.374 at (1.022, -0.6803)
  aspect ratio       max 1.96, mean 1.12
  volume ratio       max 2.21, mean 1.16
                     worst 2.21 at (1.095, 0.8271), 2.21 at (1.099, 0.8141)
  Time step          8.771e-05 (from run.cfl), CFL 0.25 at the initial state
  CFL-limiting       cell 13,165 at (0.2704, -0.07489)
                     4 cells within 10% of it, in (0.2617, -0.07529) to (0.9787, 0.5039)
  Memory             133MB per rank after setup (peak 133MB), 17,297 cells, 1 rank
                     on 4 ranks: about 46.6MB per rank (peak 46.6MB), largest part 4,325 + 1,729 halo cells
  Throughput         50.0M cells/s per rank
  Cost               1.14e+04 steps per unit simulated time
                     per unit time: 986 ms wall on 4 ranks, 0.0011 rank-hours
                     per flow-through (0.4375): 431 ms wall on 4 ranks, 0.000479 rank-hours
                     to t_stop = 0.75: 740 ms wall on 4 ranks, 0.000822 rank-hours
  Result             ok
```

(`examples/mallard` on a coarse mesh of 17k triangles, on a CPU.) The report covers:

| Line | What it is |
|---|---|
| Input keys | Unknown or unused keys, with suggestions (errors) |
| Boundaries | Zones without a condition are an error, named with their face count |
| Mesh quality | Largest and mean non-orthogonality, skewness, aspect ratio and volume ratio, and where the worst cells are ([Meshes](meshes.md#mesh-quality)) |
| Time step | The stable step at the initial state (or `run.dt` and its CFL number), the cell that sets it and the region of the cells within 10% of it |
| Memory | Kokkos allocations per rank (device memory on GPUs) after setup and at their peak. With `--ranks N` in a serial check, scaled to the owned and halo cells of the largest part the run's partitioner would make. CUDA's own context, a few hundred MB per GPU, is not counted |
| Cost | With `--throughput R` (cells/s per rank, the `Throughput` of a run's summary divided by its ranks) or `--steps N` (timed over N steps after an untimed one): steps, wall time and GPU-hours (rank-hours on CPUs) per unit simulated time, per flow-through time (domain length along the mean initial velocity over its magnitude) and to `t_stop`, assuming the throughput per rank holds on N ranks |
| Warnings | A CFL number above the integrator's limit for smooth flow (1.2 for SSPRK3, 1.35 for RK4), TFLES thickening `n_res Delta / delta_L` above 5, poor cells |

### The key catalog

`--catalog keys.json` also writes, as JSON, every key the setup looked up for
this input: whether the input set it, the type the setup reads it as, its
physical dimension (see [units](#units)), its default (as TOML) or that it is
required, and the values it accepts, by path
(`numerics.riemann_solver`, `boundaries.type`). It is written even when the
setup stops on an error, with the keys read up to there. The setup's accessors
(`find_value`, `find_value_or`, `find_real_or` and `find_option` in
`src/common/input.h`) record these, so the catalog is what the code does.
`tools/input_catalog.py build` merges the catalogs of every example and
tutorial, in 2D and 3D, into
[`docs/input-keys.json`](../input-keys.json), which the website's input builder
reads; CI checks that it is up to date, and `tools/input_catalog.py docs`
that the defaults this reference states match it.

## The other commands

| Command | Purpose | Guide |
|---|---|---|
| `mallard upgrade-input old.toml` | Rewrites renamed, moved and removed keys of an input from an older release | [Inputs from older releases](#inputs-from-older-releases) |
| `mallard reactor -i input.toml` | 0D constant-volume reactor with the solver's chemistry; chemistry benchmarks | [Thermochemistry](thermochemistry.md#0d-reactors-mallard-reactor) |
| `mallard mesh convert IN OUT.h5` | Gmsh file, or the mesh of an input file, to Mallard's HDF5 mesh format (builds with HDF5) | [Meshes](meshes.md#hdf5-meshes-for-large-runs) |
| `mallard test` | The unit and regression tests: `mallard-test` (`build/test/mallard-test`), from a build or an install tree | [Installation](https://matthewbonanni.github.io/mallard/docs/installation/) |

The Python scripts in `tools/` prepare meshes and restart files and post-process
output; [Tools](https://matthewbonanni.github.io/mallard/docs/tools/) lists them.
