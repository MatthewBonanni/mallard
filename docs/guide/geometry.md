# Geometry

Several features take a piece of space as input: where a source acts, where a
probe samples or integrates, where a sponge damps, which faces are curved. This
page explains the regions they share and how Mallard puts them on a mesh. Every key, with its default, is in the
[input reference](../input.md#regions).

## Who uses what

| Feature | Geometry | Guide |
|---|---|---|
| `[[sources]]` `region` | a [region](#regions) or its [name](#named-regions) | [Sources](sources.md#regions) |
| `[[probes]]` `region` | a [region](#regions) or its [name](#named-regions): integrals, means, extremes | [Output](output.md#region-probes) |
| `[[probes]]` `shape` | [points or a curve](#point-sets-and-curves); [boundary zones](#boundary-zones) and [surfaces](#surfaces) soon | [Output](output.md#probes-probes) |
| `[[sponges]]` `region` | a [region](#regions) or its [name](#named-regions), times `strength` | [Sponge layers](sources.md#sponge-layers) |
| `[[sponges]]` `strength` | an expression of `x`, `y`, `z` | [Sponge layers](sources.md#sponge-layers) |
| `[statistics.phase]` `probe` | a point | [Output](output.md#phase-averages-statisticsphase) |
| `[initialize]` `type = "analytical"` | expressions of `x`, `y`, `z`, averaged over each cell | [Initialization](initialization.md#expressions) |
| `[[initialize.regions]]` `region` | a [region](#regions) or its [name](#named-regions); cut cells averaged exactly for flat surfaces | [Initialization](initialization.md#regions) |
| `[[initialize.data]]` `region` | a [region](#regions) or its [name](#named-regions): its signed distance is a table's coordinate | [Initialization](initialization.md#tabulated-data-and-point-clouds) |
| `[[mesh.curved]]` | `circle`, `sphere` or `cylinder` of its own keys (to move onto regions: [#407](https://github.com/MatthewBonanni/mallard/issues/407)) | [Curved boundaries](meshes.md#curved-boundaries) |

Analytical initial states take expressions; a comparison such as
`"hypot(x, y) < 0.2 ? 2.0 : 1.0"` draws a sharp region there, which the initial
state averages over each cell. `[[initialize.regions]]` takes the region itself,
and also layers it over a restart.

## Coordinates and units

- **Coordinates** are those of the mesh as the run sees it, after
  [`[mesh.transform]`](meshes.md#transform-on-read).
- **2D** runs use the `z = 0` slice of each shape: a sphere is a disk, a
  cylinder along z a disk, a cylinder along x a strip. Points have two
  components; a third, `z`, may be given (default 0), and the shape is still
  cut at `z = 0`.
- **Lengths** take units: `radius = "5 mm"`, `center = ["10 cm", 0, 0]`.
- **Angles** are plain numbers in degrees (or quantities, `"0.5 rad"`).
- **Directions** (`axis`, `normal`, `axes`) and `scale` are plain numbers; a
  direction need not be normalized.

## Regions

```toml
[[sources]]
rhoE = "1 MW/m^3"
region = {type = "sphere", center = [0, 0, 0], radius = "2 mm"}
```

A region is a table: a primitive, or a boolean of regions, each with
optional transforms, nested to any depth, or the [name](#named-regions) of one.
The regions below fill space (volumes; areas in 2D runs). Points, curves and
surfaces are to join them as regions of lower dimension, each `type` with one
fixed dimension, and each feature checks that a region has the dimension it
needs. Each has a **signed distance** φ,
negative inside, that never exceeds the distance to its surface. Mallard uses it
to skip cells far from the surface and to give [smooth edges](#smooth-edges)
their width.

![Signed distance of a disk with a slot cut out, and its smooth-edged indicator](../images/guide/geometry-signed-distance-light.svg#gh-light-mode-only)
![Signed distance of a disk with a slot cut out, and its smooth-edged indicator](../images/guide/geometry-signed-distance-dark.svg#gh-dark-mode-only)

### Named regions

```toml
[regions]
burner = {type = "cylinder", center = [0, 0, 0], axis = [0, 0, 1], radius = 0.02, length = 0.05, edge = 0.004}

[[sources]]
name = "heat release"
total = true
rhoE = "5 kW"
region = "burner"

[[probes]]
name = "burner"
variables = ["RHO", "T"]
region = "burner"
reductions = ["integral", "mean", "max"]
```

`[regions]` defines regions once, by name; any `region` key, and any part of a
`union`, `intersection` or `difference`, takes the name instead of a table. The
source and the probe above act on the same cells with the same fractions: the
probe's `integral_RHO` is the mass the source heats. Mallard reads each named
region, and computes its fractions on the mesh, once.

```toml
[regions]
duct = {type = "cylinder", center = [0, 0, 0], axis = [1, 0, 0], radius = 0.05, length = 0.4}
plug = {type = "box", center = [0.15, 0, 0], size = [0.1, 0.2, 0.2]}
fluid = {type = "difference", regions = ["duct", "plug"]}
```

Pitfalls: a name that is not defined is an error that lists the defined ones;
regions that name each other in a loop are an error naming the loop; a named
region no feature uses is an unused key (an error, like any other); and `edge`
belongs to the outermost region, so a named region with `edge` cannot be a part
of a boolean.

### Primitives

![The 3D region primitives with their keys: sphere, ellipsoid, cylinder, annular cylinder, cone, box, torus, half-space and an implicit superellipsoid](../images/guide/geometry-primitives-3d-light.svg#gh-light-mode-only)
![The 3D region primitives with their keys: sphere, ellipsoid, cylinder, annular cylinder, cone, box, torus, half-space and an implicit superellipsoid](../images/guide/geometry-primitives-3d-dark.svg#gh-dark-mode-only)

| `type` | Example |
|---|---|
| `sphere` (2D: `disk`, `circle`) | `{type = "sphere", center = [0, 0, 0], radius = 1}` |
| `ellipsoid` (`ellipse`) | `{type = "ellipsoid", center = [0, 0, 0], semi_axes = [1, 0.5, 0.4]}` |
| `cylinder` | `{type = "cylinder", center = [0, 0, 0], axis = [0, 0, 1], radius = 0.5, length = 2}` (no `length`: infinite) |
| `cylinder`, annular | the same with `inner_radius = 0.3`: a pipe wall |
| `annulus` (2D) | `{type = "annulus", center = [0, 0], radius = 1, inner_radius = 0.6}` |
| `cone` (`frustum`) | `{type = "cone", base = [0, 0, 0], axis = [0, 0, 1], length = 1.5, base_radius = 0.8, top_radius = 0.3}` |
| `box` (`rectangle`) | `{type = "box", center = [0, 0, 0], size = [2, 1, 0.5]}` |
| `torus` | `{type = "torus", center = [0, 0, 0], axis = [0, 0, 1], major_radius = 1, minor_radius = 0.25}` |
| `half_space` | `{type = "half_space", point = [1, 0, 0], normal = [1, 0, 0]}`: the side `x < 1` |
| `polygon` (2D) | `{type = "polygon", vertices = [[0, 0], [1, 0], [0.5, 1]]}` |
| `implicit` | `{type = "implicit", f = "x^4 + y^4 + z^4 - 0.41"}`: where `f < 0` |

In 2D, each shape is the `z = 0` slice; Mallard computes the cell volume
fractions below on a coarse triangle mesh, with the exact outline drawn over them:

![Cell volume fractions of the 2D primitives on a coarse triangle mesh, with the exact outline](../images/guide/region-primitives-light.svg#gh-light-mode-only)
![Cell volume fractions of the 2D primitives on a coarse triangle mesh, with the exact outline](../images/guide/region-primitives-dark.svg#gh-dark-mode-only)

### Booleans

```toml
region = {type = "difference",
          a = {type = "disk", center = [0.42, 0.5], radius = 0.27},
          b = {type = "rectangle", center = [0.64, 0.5], size = [0.5, 0.26], rotate = 25, pivot = [0.64, 0.5]}}
```

`union` and `intersection` take `regions = [...]`; `difference` takes `a` and
`b`, or `regions` (the first minus the others). Booleans nest; a region
whose evaluation needs a stack deeper than 16 is an error.

![A disk A and a rotated rectangle B, then their union, intersection and difference](../images/guide/geometry-csg-light.svg#gh-light-mode-only)
![A disk A and a rotated rectangle B, then their union, intersection and difference](../images/guide/geometry-csg-dark.svg#gh-dark-mode-only)

![Cell volume fractions of a difference, a union, an intersection and a smooth-edged disk](../images/guide/region-booleans-light.svg#gh-light-mode-only)
![Cell volume fractions of a difference, a union, an intersection and a smooth-edged disk](../images/guide/region-booleans-dark.svg#gh-dark-mode-only)

### Transforms

Any region, primitive or boolean, may be scaled, rotated and translated, in
that order. `scale` and `rotate` act about `pivot` (default the origin).

![An L shape translated, rotated about the origin, rotated about a pivot, scaled, and all of these in order](../images/guide/geometry-transforms-light.svg#gh-light-mode-only)
![An L shape translated, rotated about the origin, rotated about a pivot, scaled, and all of these in order](../images/guide/geometry-transforms-dark.svg#gh-dark-mode-only)

| Key | 2D | 3D |
|---|---|---|
| `scale` | a factor, or `[s_x, s_y]` | a factor, or `[s_x, s_y, s_z]` |
| `rotate` | degrees, counterclockwise | `{axis = [...], angle = degrees}` (right-handed), or `{axes = [[...], [...], [...]]}`: the shape's x, y, z axes in world coordinates |
| `pivot` | center of `scale` and `rotate` | same |
| `translate` | a displacement | same |

Pitfall: without `pivot`, `rotate` turns the shape about the origin, which
also moves it. To turn a shape in place, give its center as `pivot`.

**Boxes not aligned with the grid.** Rotate a box about its center: Mallard
computes the fraction of each cell that the rotated box covers, so it needs no
grid lines along its sides. A test of cell centroids would make a staircase
and miss part of the area:

```toml
region = {type = "rectangle", center = [0.5, 0.5], size = [0.66, 0.28], rotate = 30, pivot = [0.5, 0.5]}
```

![A rotated rectangle on an 18 by 18 Cartesian mesh: cells whose centroid is inside, Mallard's volume fractions, and the fractions with a smooth edge](../images/guide/geometry-rotated-box-light.svg#gh-light-mode-only)
![A rotated rectangle on an 18 by 18 Cartesian mesh: cells whose centroid is inside, Mallard's volume fractions, and the fractions with a smooth edge](../images/guide/geometry-rotated-box-dark.svg#gh-dark-mode-only)

In 3D, rotate about an axis, or give the shape's own axes:

```toml
region = {type = "box", center = [1, 0, 0], size = [0.4, 0.2, 0.2],
          rotate = {axis = [0, 0, 1], angle = 30}, pivot = [1, 0, 0]}
region = {type = "box", center = [0, 0, 0], size = [1, 0.5, 0.2],
          rotate = {axes = [[0.6, 0.8, 0], [-0.8, 0.6, 0], [0, 0, 1]]}}
```

**Annular cylinders** have an `inner_radius`; tilt them with `axis`:

```toml
region = {type = "cylinder", center = [0.5, 0.5, 0.5], axis = [1, 1, 0.6],
          radius = 0.33, inner_radius = 0.18, length = 0.5}
```

![Volume fractions in a tetrahedral mesh of a tilted annular cylinder, a rotated box and a drilled sphere, cut open](../images/guide/regions-3d-light.svg#gh-light-mode-only)
![Volume fractions in a tetrahedral mesh of a tilted annular cylinder, a rotated box and a drilled sphere, cut open](../images/guide/regions-3d-dark.svg#gh-dark-mode-only)

### Smooth edges

`edge` (on the outermost region only) replaces the sharp surface by a band of
that width, across which the indicator falls from 1 to 0 as a smoothstep.
A source that switches on gradually excites fewer grid-scale waves. The band
keeps the volume of a flat surface and changes that of a curved one slightly
(`pi R edge^2 / 5` for a sphere).

```toml
region = {type = "sphere", center = [0, 0, 0], radius = 2e-3, edge = 5e-4}
```

![Volume fraction against radius for a sharp and a smooth-edged disk, with the smoothstep](../images/guide/region-smooth-edge-light.svg#gh-light-mode-only)
![Volume fraction against radius for a sharp and a smooth-edged disk, with the smoothstep](../images/guide/region-smooth-edge-dark.svg#gh-dark-mode-only)

### Implicit regions

```toml
# A five-lobed blob
region = {type = "implicit", f = "hypot(x - 0.5, y - 0.5) - 0.28 - 0.08 * cos(5 * atan2(y - 0.5, x - 0.5))"}
```

Any shape `f(x, y, z) < 0`, in the [expression syntax](../input.md#regions).
Implicit regions combine and transform like the others (`f` reads the shape's
local coordinates).

Pitfalls:

- `edge` is in units of `f`. It is a width in meters only when `f` is a
  distance, such as `"sqrt(x^2 + y^2) - 1"`.
- `f` gives no bound on the distance to its surface, so Mallard subdivides a
  cell unless `f` keeps its sign across it. A thin feature that falls between
  the sample points of a coarse cell can be missed: refine the mesh, or write
  `f` as a distance.

### Regions on a mesh

Each cell gets its **volume fraction**: the average of the region's indicator
over the cell. Mallard splits each cell into triangles (2D) or tetrahedra (3D)
and subdivides them where the surface may cross them (where `|φ|` is less than
their size), up to 8 levels in 2D and 5 in 3D. Cells entirely inside get
exactly 1, cells outside 0. A region much smaller than a cell can fall
between the samples of the deepest level; a region that then covers no part of
any cell is an error.

![Three triangles cut by a circle, with no subdivision, two levels and six levels, and their fractions](../images/guide/geometry-subdivision-light.svg#gh-light-mode-only)
![Three triangles cut by a circle, with no subdivision, two levels and six levels, and their fractions](../images/guide/geometry-subdivision-dark.svg#gh-dark-mode-only)

The same region on a stretched O-grid around a cylinder (a shell around the
wall and a wake box tilted 12 degrees) shows which cells it selects and by how
much. Write `SOURCE_RHOE` from a source of `rhoE = 1` to see this field for
your own region and mesh:

```toml
region = {type = "union", regions = [
    {type = "annulus", center = [0, 0], radius = 0.85, inner_radius = 0.5},
    {type = "rectangle", center = [2.4, 0], size = [3.4, 0.9], rotate = 12, pivot = [0.7, 0]}]}
```

![Volume fractions of a shell and a tilted wake box on the O-grid around a cylinder](../images/guide/geometry-mesh-region-light.svg#gh-light-mode-only)
![Volume fractions of a shell and a tilted wake box on the O-grid around a cylinder](../images/guide/geometry-mesh-region-dark.svg#gh-dark-mode-only)

In axisymmetric runs the fractions are weighted by the radius, so they are true
volume fractions of the revolved cells.

## Point sets and curves

```toml
[[probes]]
name = "wake"
variables = ["U", "P"]
shape = "arc"
center = [0.0, 0.0]
radius = 5.0
angles = [-45.0, 45.0]
n_points = 91
```

Probes sample along a point set (`point`, `points`, `points_file`) or a curve
(`line`, `polyline`, `circle`, `arc`, `curve`). Keys:
[`[[probes]]`](../input.md#probes).

![The 2D probe shapes with the samples Mallard places on them; a ring marks each one's first sample](../images/guide/geometry-probes-light.svg#gh-light-mode-only)
![The 2D probe shapes with the samples Mallard places on them; a ring marks each one's first sample](../images/guide/geometry-probes-dark.svg#gh-dark-mode-only)

![A tilted circle, a helix and a line in a 3D domain, with their samples](../images/guide/geometry-probes-3d-light.svg#gh-light-mode-only)
![A tilted circle, a helix and a line in a 3D domain, with their samples](../images/guide/geometry-probes-3d-dark.svg#gh-dark-mode-only)

### Where the samples go

| `shape` | Samples |
|---|---|
| `point`, `points`, `points_file` | the points as given, in order |
| `line` | `n_points` evenly from `start` to `end`, both included |
| `polyline` | `n_points` evenly on each segment, ends included; shared vertices once |
| `circle` | `n_points` evenly, the first along `reference` (3D) or +x (2D), not repeated at the end |
| `arc` | `n_points` evenly from the first angle to the second, both included |
| `curve` | `n_points` evenly in the parameter `s` over `s_range`: denser where the curve moves slowly in `s` |

Transforms of probes are `scale`, then `rotate`, then `translate`, all about
the origin (probes have no `pivot`): the scaled and rotated circle above is
centered at the origin, then translated. A sample outside the domain takes the
nearest cell's average.

Curve reductions (`mean`, `integral`, ...) integrate along the curve with three
Gauss points between consecutive samples; point sets weigh each sample 1.

### Cell averages or point values

`sampling = "cell"` (the default) returns the average of the cell holding each
sample: first-order accurate at a point off the cell's centroid.
`sampling = "point"` evaluates the reconstruction at the sample, which with
TENO converges at the scheme's order:

![Error of probe values against cell size: cell averages first order, MUSCL point values second, TENO orders 3 and 5 at their order](../images/guide/geometry-sampling-light.svg#gh-light-mode-only)
![Error of probe values against cell size: cell averages first order, MUSCL point values second, TENO orders 3 and 5 at their order](../images/guide/geometry-sampling-dark.svg#gh-dark-mode-only)

Use `point` to compare with references or to measure orders of accuracy, and
`cell` for robust values across shocks. Details and pitfalls:
[Output](output.md#probes-probes).

## Boundary zones

> **Coming soon** ([#466](https://github.com/MatthewBonanni/mallard/pull/466)):
> probes that sample the faces of a boundary zone.

<!-- Boundary-zone probes: fill in when #466 merges (keys, sample placement, a figure). -->

## Surfaces

> **Coming soon** ([#395](https://github.com/MatthewBonanni/mallard/pull/395)):
> cut planes and resampled surfaces.

<!-- Cut planes and resampled surfaces: fill in when #395 merges (keys, a figure of a cut plane). -->

## Reading probe output

`format = "csv"` (default) writes one row per sample and output, after the
header and a `# provenance` line:

```python
import numpy as np

d = np.genfromtxt("probe_wake.csv", delimiter=",", names=True)  # skips the provenance line
last = d[d["step"] == d["step"].max()]
print(last["x"], last["y"], last["P"])
```

`format = "hdf5"` writes one file, its arrays growing by a row per output:

```python
import h5py

with h5py.File("probe_wake.h5") as f:
    xyz = f["coordinates"][...]   # (samples, dimension)
    s = f["arc_length"][...]      # curves only: distance along the curve
    t = f["time"][...]            # (outputs,)
    p = f["fields/P"][...]        # (outputs, samples); U expands to fields/U_X, U_Y (, U_Z)
```

`format = "vtkhdf"` writes one VTKHDF PolyData file holding every output, and
`format = "vtp"` one PolyData file per output and a `.pvd` series: open the
`.vtkhdf` (or `.pvd`) in ParaView, then plot along the curve with *Plot Data* (by
`arc_length`) or over time with *Plot Data Over Time*. Reductions go to
`probe_<name>_reductions.csv`. Format details: [`[[probes]]`](../input.md#probes).

## See also

- [Input reference: regions](../input.md#regions), [`[regions]`](../input.md#named-regions-regions), [`[[sources]]`](../input.md#sources), [`[[sponges]]`](../input.md#sponges), [`[[probes]]`](../input.md#probes), [`[mesh]`](../input.md#mesh)
- [Sources and sponges](sources.md), [Output and diagnostics](output.md#probes-probes), [Meshes](meshes.md#curved-boundaries)
- Figures: [`tools/docs/geometry.py`](../../tools/docs/geometry.py), [`tools/docs/regions.py`](../../tools/docs/regions.py)
