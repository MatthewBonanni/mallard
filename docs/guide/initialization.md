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

*Used in:* [`wedge`](https://matthewbonanni.github.io/mallard/docs/examples/#wedge), [`sphere_mach3`](https://matthewbonanni.github.io/mallard/docs/examples/#sphere-mach3), [`ellipsoid`](https://matthewbonanni.github.io/mallard/docs/examples/#ellipsoid), [`mallard`](https://matthewbonanni.github.io/mallard/docs/examples/#mallard), [`heated_duct`](https://matthewbonanni.github.io/mallard/docs/examples/#heated-duct) and 9 more.

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

*Used in:* [`sod`](https://matthewbonanni.github.io/mallard/docs/examples/#sod), [`shu_osher`](https://matthewbonanni.github.io/mallard/docs/examples/#shu-osher), [`riemann_2d`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d), [`riemann_2d_quads`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d-quads), [`double_mach`](https://matthewbonanni.github.io/mallard/docs/examples/#double-mach) and 19 more.

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
([Expressions](expressions.md)), and read
[tabulated data](#tabulated-data-and-point-clouds); a comparison such as
`hypot(x, y) < 0.2` draws a region ([Geometry](geometry.md#who-uses-what)). Pitfall:
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
region.type = "rectangle"
region.center = ["6 mm", 0.0]
region.size = ["2 mm", "8 mm"]
region.pivot = ["6 mm", 0.0]
region.rotate = 30
X = { H2 = 1.0 }
```

*Used in:* [`shock_bubble_2d`](https://matthewbonanni.github.io/mallard/docs/examples/#shock-bubble-2d), [`turbulent_hot_spot_2d`](https://matthewbonanni.github.io/mallard/docs/examples/#turbulent-hot-spot-2d), [`mallard_logo`](https://matthewbonanni.github.io/mallard/docs/examples/#mallard-logo).

Each `[[initialize.regions]]` entry sets some of `u`, `rho`, `p`, `T` and the
composition inside a region, and leaves the rest to the state below it. The
regions (primitives, booleans, transforms, smooth edges) are those used
everywhere in Mallard: see [Geometry: regions](geometry.md#regions). Values are
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
  those of the regions. With `cells = "center"` a cell whose centroid is
  inside takes the region's state whole: no mixed cells (e.g. compositions
  that must not mix), but staircase edges whose volume depends on the mesh.

![Density of an initial state built in layers on a triangle mesh: a tabulated radial profile, a rotated rectangle over it, and a smooth-edged disk of higher priority on top](../images/guide/initial-layers-light.svg#gh-light-mode-only)
![Density of an initial state built in layers on a triangle mesh: a tabulated radial profile, a rotated rectangle over it, and a smooth-edged disk of higher priority on top](../images/guide/initial-layers-dark.svg#gh-dark-mode-only)

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
| `name` | `initialize.regions[i]` | Name in the terminal output |

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

*Used in:* [`shock_bubble_2d`](https://matthewbonanni.github.io/mallard/docs/examples/#shock-bubble-2d), [`mallard_logo`](https://matthewbonanni.github.io/mallard/docs/examples/#mallard-logo).

Each `[[initialize.data]]` entry reads a CSV file (a header of column names,
then rows of numbers; lines starting with `#` are comments) and gives every
column other than the coordinates to the expressions of `[initialize]` and
its regions as a variable `<name>_<column>`: above, `bubble_rho`.

- **Tables** (`type = "table"`, the default): values on a grid of one to
  three coordinates, every combination of their distinct values once, rows in
  any order, e.g. a 1D flame `x, T, u, Y_H2, ...` or a 2D field on `(x, r)`.
  `interpolation = "linear"` is exact for linear data; `"cubic"` (Lagrange on
  the 4 nearest nodes of each axis, non-uniform grids too) for cubic data.
  Outside the table, the values at its ends. An image is a 2D table: one row
  per pixel (`x, y` and a value per color class), as `mallard_logo` paints
  the Mallard logo in gas (`examples/mallard_logo/make_logo_table.py`); pad it
  with a border of background pixels, since the edge values extend outward.
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

## Synthetic turbulence

A divergence-free random velocity with a prescribed energy spectrum E(k),
added to any initial state or inside a region: decaying turbulence, a
turbulent hot spot, fluctuations over a restart.

```toml
# examples/turbulent_hot_spot_2d, on the periodic box [0, 2 pi]^2
[initialize]
type = "constant"
u = [0.0, 0.0]
p = 0.7142857142857143
T = 0.7142857142857143
n_subdivisions = 1   # smooth fields: no need for the 2D default, 4

[initialize.turbulence]
spectrum = "passot_pouquet"
intensity = 0.1      # u', the rms of each component
length_scale = 0.5   # the integral length scale

[[initialize.regions]]
name = "hot spot"
region = {type = "disk", center = [3.141592653589793, 3.141592653589793], radius = 1.0, edge = 0.3}
T = 2.142857142857143
turbulence = {spectrum = "von_karman_pao", intensity = 0.2, length_scale = 0.2, seed = 2}
```

The field is a sum of Fourier modes `a cos(k · x + φ)`, each with `a`
normal to `k`, so it is divergence-free at every point
([Kraichnan 1970](../references.md#kraichnan-1970)). The wavevectors are
those of a periodic box (`period`, by default the mesh's bounding box), so
the field is periodic over it, grouped into shells of width
dk = 2π / (shortest period) centered on `j dk`, up to `k_max`. Each shell
holds the spectrum's energy between its edges, split evenly among at most
`modes_per_shell` of its wavevectors; the shells' total is (d / 2) u'² in d
dimensions, so u' is the rms of each velocity component. Shells of
negligible energy (below 10⁻¹² of the total, as in the exponential tails)
are left out. Like the other initial states, cells take the average of the
field over their quadrature points.

| `spectrum` | E(k), up to a constant |
|---|---|
| `von_karman_pao` (default) | (k / k_e)⁴ / (1 + (k / k_e)²)^(17/6) · exp(-2 (k η)²) ([Bailly & Juvé 1999](../references.md#bailly-juve-1999)) |
| `passot_pouquet` | (k / k_e)⁴ exp(-2 (k / k_e)²) ([Passot & Pouquet 1987](../references.md#passot-pouquet-1987)) |
| `table` | Linear between the rows of a CSV `file` of columns `k` (1/m) and `E` (m³/s²), zero outside |

`k_e` follows from `length_scale`, the integral length scale
L = C ∫E/k dk / ∫E dk with C = 3π/4 in 3D and 2 in 2D
([Pope 2000](../references.md#pope-2000), §6.5): for Passot–Pouquet,
k_e = √(2π) / L in 3D. `kolmogorov_scale` (η, default none) adds the
dissipation range of von Kármán–Pao. A table gives E itself, and `intensity`,
if given, scales it.

**Determinism.** Each shell keeps its wavevectors of smallest hash of `seed`
and the wavevector, and their directions and phases are hashes of the same,
computed on the host: any rank count gives the same field bit for bit. A
wavevector's direction and phase do not depend on `k_max`,
`modes_per_shell` or the other shells, and the same table in the base state
and in a region gives the same field.

**Layering.** In `[initialize]` the field adds to the base velocity of any
`type`, a restart or remap included (every cell then changes). In
`[[initialize.regions]]` it adds to the region's `u`, or to the velocity
below when the region sets none, and blends through the region's `edge`
like any other value, so it fades out over the edge.

The run log's *Initial state* lists the modes and shells, and the share of
the spectrum's energy the shells hold. Pitfalls:

- Cost is cells × quadrature points × modes, about 2.5 × 10⁸ modes times
  points per second per CPU thread: 28 points per quadrilateral times
  `n_subdivisions`² in 2D, 336 per hexahedron times `n_subdivisions`³ in 3D, and up to
  `modes_per_shell` modes in each of up to `k_max / dk` shells. Smooth edges
  of regions refine the cells along them. For large runs on CPUs, lower
  `modes_per_shell` or `k_max`, or in 2D `n_subdivisions`.
- The field is periodic over `period`, not over a region: the largest eddies
  are a period long, and wavenumbers below dk / 2 get no energy.
- A spectrum whose energy lies mostly beyond `k_max` (L close to the cell
  size) is cut there, and the shells left carry all of u'²: the log's share
  says how much was cut.
- With the default `k_max` (π over the mean cell size), the smallest eddies
  span two cells; they decay at once.
- The velocity alone changes: density and pressure stay as given, not in
  equilibrium with the eddies, so compressible runs start with weak acoustic
  waves.
- Not available on axisymmetric meshes.
- `tools/setup/isotropic_turbulence_restart.py` writes the case of Johnsen
  et al. as a restart file instead: every mode of the box, cell-averaged
  exactly by FFT, for hexahedral boxes only.

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
- A restart file is written as `<name>.restart.partial` (in the background with `[output] async`) and renamed to `<name>.restart` once every rank has written its part and it is flushed to storage, so a `.restart` file is always complete. A `.partial` file is what a run killed while writing left behind; resume from the newest `.restart` file instead, as `--restart latest` does.
- A restart on a mesh file whose contents differ from those the file was written on (the hashes in the files' [run blocks](output.md#run-block)) warns, even with as many cells; the log's *Initial state* says when the input differs from the one that wrote the file.

### Resuming from the newest restart

```sh
mallard -i input.toml --restart latest
```

`--restart latest` resumes the run from the newest restart file it wrote, or
starts it from `[initialize]` if there is none, so the same command starts a
run and continues it after every stop, as the [job scripts](parallel.md#job-scripts)
do. `mallard check input.toml --restart latest` checks the run it would start.

- It looks at the restart files of the input's `[[write_data]]` writers with
  `format = "restart"` (not those of a region), or of `restart`, where
  restarts on request go, if there is none: those the run record's [output
  index](output.md#run-record-mallardjsonl) lists for each writer. For a
  writer it lists none of (a run record before 0.10, or none), it looks at
  the files `<prefix>_*.restart` instead, and says so. `.partial` files are
  ignored. Step and time are those the files record, never their names.
- Newest is the file of the run that started last (the start time in the
  files' [run blocks](output.md#run-block)), then of the highest step, then of the first writer: after
  a rewind, it resumes from the rewound run's files, not from higher steps
  the abandoned run left behind. Modification times do not count.
- A file that does not read completely (cut short by a full disk or a copy)
  is skipped with a warning naming it. If no file reads completely, the run
  stops with an error instead of starting afresh: name a file with `--restart
  FILE`, or delete them.
- The log says which file it picked, under *Initial state*, or that there was
  none. With MPI, rank 0 decides for every rank.
- A file named `latest` is given as `./latest`.
- Restart files of every earlier release are read, back to 0.1, and files from a newer Mallard are rejected clearly ([compatibility](output.md#formats-and-compatibility)).
- `compression = "zlib"` writes compressed restart files (version 5, losslessly: they restart bit for bit), about a third smaller for smooth flows; reading them needs a build with `Mallard_ENABLE_ZLIB`; the Python tools in `tools/` read them too.

### Initial states from scripts

Many examples start from a restart file that a script in `tools/` computes:
a Cantera flame, a ZND detonation, a turbulent field. These are the patterns
to copy for your own cases:

| Script | Initial state |
|---|---|
| `flame_restart.py` | Premixed flame in its own frame from a Cantera solution (1D, or a wrinkled 2D front) |
| `znd_restart.py` | ZND detonation profile, with pockets of fresh gas to trigger cells |
| `channel_init.py` | Turbulent channel: mean profile plus a divergence-free perturbation |
| `isotropic_turbulence_restart.py` | Isotropic turbulence of the Johnsen et al. case on a periodic box, every mode of the box (any spectrum, any mesh, any region: [synthetic turbulence](#synthetic-turbulence) without a script) |
| `autoignition_restart.py` | Stratified temperature and velocity fields |
| `interpolate_restart.py` | A restart file interpolated onto another Gmsh mesh (nearest cells; `type = "remap"` below is conservative and high order) |
| `tile_restart.py` | A restart file of a periodic box tiled into a larger box |

A restart file is binary, little-endian: the 16 bytes `MALLARD-RESTART\0`,
then `uint32` version (2; [all versions](output.md#formats-and-compatibility)) and real size (8), `uint64` cell count, variable
count and step, `float64` time, each variable name as a `uint32` length and its
characters, and finally the values, variable after variable, each over all
cells in the mesh's global order. The variables are `RHO`, `RHOU_X`, `RHOU_Y`
(`RHOU_Z`), `RHOE` and, for gas mixtures, `RHOY_<species>` for every species
and `T_SEED` (a temperature guess). `tools/mallard/restart.py` reads every version from 2 on (`read`) and writes them (`write`), as the scripts above do.

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
fresh output series; the terminal output reports how closely the integrals were kept.

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
  cells outside take the nearest source cell; the terminal output counts them, and
  conservation then holds only over the common part.
- The default `bounds` limiter flattens smooth extrema slightly; use
  `limiter = "none"` for smooth flows such as the Taylor-Green vortex.
- Restarting on the **same** mesh is `type = "restart"`, which is bitwise exact.

Method, accuracy and cost: [docs/design/remap.md](../design/remap.md).

## See also

- [Input reference: `[initialize]`](../input.md#initialize)
- [Geometry](geometry.md): regions, and how cells average them
- [Output and diagnostics](output.md): writing restart files and snapshots
