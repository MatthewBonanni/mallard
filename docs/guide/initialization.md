# Initialization and restarts

`[initialize]` sets the state at the start of a run: uniform, from
expressions, or from a restart file written by an earlier run or by a script.
Regions and tabulated data (profiles, point clouds) layer over any of these.

## Uniform state

```toml
[initialize]
type = "constant"     # the default
u = [0.2, 0.0]
p = 0.7142857142857143
T = 1.0
# X = { O2 = 1.0, N2 = 3.76 }   gas mixtures: the composition
```

*Used in:* [`wedge`](https://matthewbonanni.github.io/mallard/docs/examples/#wedge), [`sphere_mach3`](https://matthewbonanni.github.io/mallard/docs/examples/#sphere-mach3), [`mallard`](https://matthewbonanni.github.io/mallard/docs/examples/#mallard), [`poiseuille_pipe`](https://matthewbonanni.github.io/mallard/docs/examples/#poiseuille-pipe), [`source_regions`](https://matthewbonanni.github.io/mallard/docs/examples/#source-regions) and 2 more.

## Expressions

```toml
# examples/riemann_2d: the four quadrants of configuration 3
[initialize]
type = "analytical"
rho = """\
    x >= 0.8 ? (y >= 0.8 ? 1.5 : 0.532258064516129) \
             : (y >= 0.8 ? 0.532258064516129 : 0.137992831541219)"""
u = ["x >= 0.8 ? 0.0 : 1.206045378311055",
     "y >= 0.8 ? 0.0 : 1.206045378311055"]
p = """\
    x >= 0.8 ? (y >= 0.8 ? 1.5 : 0.3) \
             : (y >= 0.8 ? 0.3 : 0.029032258064516)"""
```

*Used in:* [`sod`](https://matthewbonanni.github.io/mallard/docs/examples/#sod), [`shu_osher`](https://matthewbonanni.github.io/mallard/docs/examples/#shu-osher), [`riemann_2d`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d), [`riemann_2d_quads`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d-quads), [`double_mach`](https://matthewbonanni.github.io/mallard/docs/examples/#double-mach) and 16 more.

| Key | Default | Meaning |
|---|---|---|
| `type` | `"constant"` | `constant`, `analytical`, `restart`, or `remap` (a restart file of another mesh) |
| `u` | required | `constant`: numbers; `analytical`: one expression per component |
| `p`, `T` (`constant`) | required | Pressure and temperature |
| `rho`, `p`, `T` (`analytical`) | two of the three | Expressions of x, y, z |
| `X` or `Y` | required for mixtures | Composition by species: numbers, or expressions (`analytical`) |
| `balance` | none | Mixtures, `analytical`: the species that takes `1 - sum` of the others |
| `n_subdivisions` | 4 (2D), 1 (3D) | Resolution of the cell averages of `analytical` states and [regions](#regions) |
| `file` | required (`restart`) | Restart file to start from |

Analytical states are integrated over each cell, not sampled at its center,
so discontinuities that cut a cell give the right average (and high-order runs
start from true cell averages). In 2D each cell's triangles are split into
`n_subdivisions`² sub-triangles; in 3D each tetrahedron of a cell into
`n_subdivisions`³ sub-tetrahedra with a degree-5 rule. Raise it for sharp
features smaller than a cell. A cell where the state is the same at every
point takes it exactly. The integration runs on the device (the GPU in GPU
builds), one cell per thread.

![A triangle with one and with sixteen sub-triangles for the cell average of an analytical state](../images/guide/subdivisions-light.svg#gh-light-mode-only)
![A triangle with one and with sixteen sub-triangles for the cell average of an analytical state](../images/guide/subdivisions-dark.svg#gh-dark-mode-only)

Expressions may use local variables, conditionals and loops
([Expressions](input_files.md#expressions)), and read
[tabulated data](#tabulated-data-and-point-clouds). Pitfall:
TOML requires the expressions to be strings, also for constants: `u =
["0.0", "0.0"]` with `analytical`.

## Regions

```toml
# A hot kernel with a smooth edge in air at rest, and a tilted slab of hydrogen
[initialize]
type = "constant"
u = [0.0, 0.0]
p = "1 atm"
T = 300
X = { O2 = 1.0, N2 = 3.76 }

[[initialize.regions]]
name = "kernel"
region = { type = "disk", center = [0.0, 0.0], radius = "2 mm", edge = "0.5 mm" }
T = 2000              # T alone: the pressure, velocity and composition stay

[[initialize.regions]]
name = "fuel"
region = { type = "rectangle", center = ["6 mm", 0.0], size = ["2 mm", "8 mm"],
           pivot = ["6 mm", 0.0], rotate = 30 }
X = { H2 = 1.0 }
```

*Used in:* [`shock_bubble_2d`](https://matthewbonanni.github.io/mallard/docs/examples/#shock-bubble-2d).

Each `[[initialize.regions]]` entry sets some of `u`, `rho`, `p`, `T` and the
composition inside a region, and leaves the rest to the state below it. The
shapes (primitives, booleans, transforms, smooth edges) are those of every
region in Mallard: see [Geometry: regions](geometry.md#regions). Values are
numbers, quantities with units, or expressions of `x`, `y`, `z` and
[tabulated data](#tabulated-data-and-point-clouds).

- **Layers.** The base state of `[initialize]` comes first, then the regions
  by `priority` (lowest first) and, at equal priority, in the order listed:
  where regions overlap, the later one wins.
- **Thermodynamic state.** A region sets at most two of `rho`, `p`, `T`. One
  alone replaces that quantity and keeps the pressure of the state below
  (its temperature where the state below has no pressure): `T` alone makes a
  hot spot, `rho` alone a light or heavy bubble, at the same pressure.
- **Composition.** `X` or `Y` (and `balance`, as in `[initialize]`) replace
  the whole composition below.
- **Smooth edges.** Across a region's `edge`, `u`, `p`, `T` and the mass
  fractions blend linearly from the region's values to those below.
- **Cells.** With `cells = "volume_fraction"` (the default) a cell holds the
  average of the layered state over the cell: a cut cell mixes the states by
  the volume of each part, and the integrals of mass, momentum and energy are
  those of the shapes. With `cells = "center"` a cell whose centroid is
  inside takes the region's state whole: no mixed cells (e.g. compositions
  that must not mix), but staircase edges whose volume depends on the mesh.

Cells entirely inside or outside every region take the state exactly. Cells
that a sharp surface cuts are subdivided further (four levels in 2D, three in
3D, below `n_subdivisions`) and the finest pieces are cut along the surface:
exactly along flat surfaces (boxes, half-spaces, polygons, and booleans and
transforms of them: the integrals are exact to rounding), along chords of
curved ones (second order in the piece size: a disk 10 cells across loses
2e-6 of its area, a sphere 5 cells across 1e-3 of its volume).

| Key | Default | Meaning |
|---|---|---|
| `region` | required | The [region](geometry.md#regions), e.g. `{type = "sphere", center = [0, 0, 0], radius = 0.1}`, with an optional smooth `edge` |
| `u`, `rho`, `p`, `T` | the state below | Values inside the region: numbers, quantities with units or expressions |
| `X` or `Y`, `balance` | the composition below | Mixtures: the composition inside, as in `[initialize]` |
| `priority` | 0 | Higher priorities go on top; equal ones in the order listed |
| `cells` | `"volume_fraction"` | `volume_fraction`: cell averages; `center`: whole cells by their centroids |
| `name` | `initialize.regions[i]` | Name in the run log |

**Over a restart or a remap.** With `type = "restart"` or `"remap"`, regions
change the file's state: ignite a converged cold flow, add a bubble to a
turbulent field. Cells that no region reaches keep the file's values bit for
bit; the step and time continue from the file.

Pitfalls:

- A region's values must give a physical state with the state below: `rho`
  alone over a state given by `rho` and `T` changes the pressure.
- With `cells = "center"` and a smooth edge, the edge's weight at the
  centroid holds for the whole cell.

## Tabulated data and point clouds

```toml
# examples/shock_bubble_2d: a bubble from its radial density profile
[initialize]
type = "analytical"
rho = "bubble_rho"
u = ["0", "0"]
p = "1"

[[initialize.data]]
name = "bubble"
file = "bubble.csv"          # r,rho
interpolation = "cubic"
at = ["hypot(x - 0.6, y)"]   # the distance from the bubble's center
```

*Used in:* [`shock_bubble_2d`](https://matthewbonanni.github.io/mallard/docs/examples/#shock-bubble-2d).

Each `[[initialize.data]]` entry reads a CSV file (a header of column names,
then rows of numbers; lines starting with `#` are comments) and gives every
column other than the coordinates to the expressions of `[initialize]` and
its regions as a variable `<name>_<column>`: above, `bubble_rho`.

- **Tables** (`type = "table"`, the default): values on a grid of one to
  three coordinates, every combination of their distinct values once, rows in
  any order, e.g. a 1D flame `x, T, u, Y_H2, ...` or a 2D field on `(x, r)`.
  `interpolation = "linear"` is exact for linear data; `"cubic"` (Lagrange on
  the 4 nearest nodes of each axis, non-uniform grids too) for cubic data.
  Outside the table, the values at its ends.
- **Point clouds** (`type = "points"`): scattered points, e.g. the cell
  centers of a solution from another code. A point takes the value of the
  nearest data point (ties to the earlier row), or with `neighbors = k` the
  average of the k nearest weighted by inverse distance squared (exact at
  the data points). Solutions of Mallard on another mesh remap conservatively
  with `type = "remap"` instead.
- **Where a point reads them.** `at` gives one expression of `x`, `y`, `z`
  per coordinate: an axis (`"x - 0.005"`, `"0.6 * x + 0.8 * y"`), a radius
  (`"hypot(y, z)"`), any mapping; without it, coordinates named `x`, `y`, `z`
  are read where they are. For one coordinate, `region` instead reads it at
  the signed distance to a [region](geometry.md#regions) (negative inside):
  a profile normal to any surface.

| Key | Default | Meaning |
|---|---|---|
| `name` | required | Prefix of the variables: `<name>_<column>` (characters other than letters, digits and `_` become `_`) |
| `file` | required | CSV file, relative to where Mallard runs |
| `type` | `"table"` | `table` (on a grid, interpolated) or `points` (scattered) |
| `coordinates` | first column (`table`); `x`, `y` (`z`) (`points`) | Names of the coordinate columns |
| `at` | the coordinates, if named `x`, `y`, `z` | One expression of `x`, `y`, `z` per coordinate |
| `region` | none | Instead of `at`, one coordinate: the signed distance to this region |
| `interpolation` | `"linear"` | Tables: `linear` or `cubic` |
| `neighbors` | 1 | Points: how many nearest points to weight (1-16) |
| `units` | SI | Units of columns, e.g. `{x = "mm", T = "degC"}` |

Every point of the cell integration reads the data, so cell averages are
those of the interpolated field. Pitfalls:

- Cubic interpolation overshoots at fronts sharper than the table's spacing;
  negative mass fractions are clipped and the composition renormalized, but
  temperatures are not: use `linear` there.
- An entry that no expression reads is an error; so are rows that do not
  form a grid.

## Restarts

```toml
[initialize]
type = "restart"
file = "solut/restart_010000.restart"

[[write_data]]                 # and keep writing them
prefix = "solut/restart"
format = "restart"
interval = 10000
```

*Used in:* [`isotropic_turbulence`](https://matthewbonanni.github.io/mallard/docs/examples/#isotropic-turbulence), [`channel_retau180`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-retau180), [`channel_inflow_retau180`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-inflow-retau180), [`cbc_les`](https://matthewbonanni.github.io/mallard/docs/examples/#cbc-les), [`channel_les`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-les) and 13 more.

`format = "restart"` in `[[write_data]]` writes the conservative variables
(and whatever else the run needs to continue: chemistry sub-steps, the soot
foil `P_MAX`, `[statistics]` averages, characteristic boundary states) in
global cell order. A restarted run continues the uninterrupted one **bit for
bit**, also on a different number of MPI ranks, or serially from a parallel
run. The step and time continue from the file; `t_stop` and `n_steps` refer to
the total.

- `[statistics] reset = true` starts the averages afresh instead of
  continuing them.
- Probes and force monitors append to their files on a restart.
- Restart files of every earlier release are read, back to 0.1, and files from a newer Mallard are rejected clearly ([compatibility](output.md#file-formats-and-compatibility)).
- `compression = "zlib"` writes compressed restart files (version 5, losslessly: they restart bit for bit), about a third smaller for smooth flows; reading them needs a build with `Mallard_ENABLE_ZLIB`, and the Python tools in `tools/` read only uncompressed ones.

### Initial states from scripts

Many examples start from a restart file that a script in `tools/` computes:
a Cantera flame, a ZND detonation, a turbulent field. These are the patterns
to copy for your own cases:

| Script | Initial state |
|---|---|
| `flame_restart.py` | Premixed flame in its own frame from a Cantera solution (1D, or a wrinkled 2D front) |
| `znd_restart.py` | ZND detonation profile, with pockets of fresh gas to trigger cells |
| `channel_init.py` | Turbulent channel: mean profile plus a divergence-free perturbation |
| `isotropic_turbulence_restart.py` | Isotropic turbulence with a given spectrum |
| `autoignition_restart.py` | Stratified temperature and velocity fields |
| `interpolate_restart.py` | A restart file interpolated onto another Gmsh mesh (nearest cells; `type = "remap"` below is conservative and high order) |
| `tile_restart.py` | A restart file of a periodic box tiled into a larger box |

A restart file is binary, little-endian: the 16 bytes `MALLARD-RESTART\0`,
then `uint32` version (2; [all versions](output.md#file-formats-and-compatibility)) and real size (8), `uint64` cell count, variable
count and step, `float64` time, each variable name as a `uint32` length and its
characters, and finally the values, variable after variable, each over all
cells in the mesh's global order. The variables are `RHO`, `RHOU_X`, `RHOU_Y`
(`RHOU_Z`), `RHOE` and, for gas mixtures, `RHOY_<species>` for every species
and `T_SEED` (a temperature guess); `channel_init.py` has a short writer.

## Remap from another mesh

```toml
[initialize]
type = "remap"
file = "coarse/run_000100.restart"   # written on another mesh

[initialize.mesh]                   # that mesh: the source run's [mesh] table
type = "cartesian"
Nx = 100
Ny = 100
```

*Used in:* [`riemann_2d_quads`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d-quads), [`taylor_green_3d`](https://matthewbonanni.github.io/mallard/docs/examples/#taylor-green-3d).

`type = "remap"` starts from a restart file written on a **different mesh**: a
finer one (grid convergence, spinning up an LES from a coarse run), another
cell type, any partition and rank count on either side. Give the source run's
mesh as `[initialize.mesh]`:

```toml
# examples/riemann_2d_quads/remap.toml: 10,000 quads at t = 0.5 onto 131,072 triangles
[initialize]
type = "remap"
file = "coarse/riemann_000218.restart"
degree = 3              # of the source polynomials, 0-5 (default 2)
# limiter = "none"      # for smooth fields; default "bounds"

[initialize.mesh]       # the source run's [mesh] table
type = "cartesian"
Nx = 100
Ny = 100
```

| Key | Default | Meaning |
|---|---|---|
| `file` | required | Restart file of the source run |
| `[initialize.mesh]` | required | The source run's `[mesh]` table |
| `[[initialize.periodic]]` | the run's `[[periodic]]` | The source mesh's periodic zone pairs |
| `degree` | 2 | Degree of each source cell's least-squares polynomial |
| `limiter` | `"bounds"` | `bounds`: no new extrema, no negative partial densities; `none`: full order |

Each source cell's polynomial is integrated **exactly** over its intersection
with every target cell, so the integrals of all conserved variables are kept
to rounding and polynomial fields up to `degree` are remapped exactly. Every
variable of the file is remapped: the flow, species, temperature seeds and
`[statistics]` averages. The run continues from the file's step and time with
fresh output series; the log reports how closely the integrals were kept.

![A target triangle cut into pieces by the source quadrilaterals it overlaps](../images/guide/remap-supermesh-light.svg#gh-light-mode-only)
![A target triangle cut into pieces by the source quadrilaterals it overlaps](../images/guide/remap-supermesh-dark.svg#gh-dark-mode-only)

![Density on quadrilaterals and remapped to triangles, and along a line](../images/guide/remap-fields-light.svg#gh-light-mode-only)
![Density on quadrilaterals and remapped to triangles, and along a line](../images/guide/remap-fields-dark.svg#gh-dark-mode-only)

Pitfalls:

- `[initialize.mesh]` must be the mesh the restart file was written on (its
  cell count is checked); a mesh file's `transform` applies to it as usual.
- The source mesh may be periodic by translation only; a source with a
  rotational pair (`[[periodic]] angle`, or an `annulus` periodic in `theta`)
  is an error.
- Where the meshes cover different domains (a curved wall discretized
  differently, a shorter box), target cells average over the covered part and
  cells outside take the nearest source cell; the log counts them, and
  conservation then holds only over the common part.
- The default `bounds` limiter flattens smooth extrema slightly; use
  `limiter = "none"` for smooth flows such as the Taylor-Green vortex.
- Restarting on the **same** mesh is `type = "restart"`, which is bitwise exact.

Method, accuracy and cost: [docs/design/remap.md](../design/remap.md).

## See also

- [Input reference: `[initialize]`](../input.md#initialize)
- [Output and diagnostics](output.md): writing restart files and snapshots
