# Input files

Everything about a run is in one [TOML](https://toml.io) file, read by

```sh
Mallard -i input.toml [Kokkos options]
```

Kokkos options such as `--kokkos-num-threads=8` (CPU threads) or
`--kokkos-device-id=1` (which GPU) follow the input file. `Mallard --version`
prints the version and `Mallard --help` the usage. Errors in the input stop
the run before it starts, with the file name, the key and, for options, the
accepted values.

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
- **Units.** Single gases are scale-free: the reference state
  (`p_ref`, `T_ref`, `rho_ref`) sets the gas constant, so nondimensional
  inputs work as they are (`p_ref = T_ref = rho_ref = 1`). Gas mixtures read
  real thermochemistry and are in SI units (Pa, K, m, s, kg).

## The other executables

| Executable | Purpose | Guide |
|---|---|---|
| `MallardReactor -i input.toml` | 0D constant-volume reactor with the solver's chemistry; chemistry benchmarks | [Thermochemistry](thermochemistry.md#0d-reactors-mallardreactor) |
| `mallard-mesh-convert IN OUT.h5` | Gmsh file, or the mesh of an input file, to Mallard's HDF5 mesh format (builds with HDF5) | [Meshes](meshes.md#hdf5-meshes-for-large-runs) |
| `MallardTest` | The unit and regression tests (`build/test/MallardTest`) | [Installation](https://matthewbonanni.github.io/mallard/docs/installation/) |

The Python scripts in `tools/` prepare meshes and restart files and post-process
output; [Tools](https://matthewbonanni.github.io/mallard/docs/tools/) lists them.
