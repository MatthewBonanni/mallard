# Geometry

Several features take a piece of space as input: where a source acts, where a
probe samples or integrates, where a sponge damps, which faces are curved. This
page explains the regions they share and how Mallard puts them on a mesh. Every key, with its default, is in the
[input reference](../input.md#regions).

## Who uses what

| Feature | Geometry | Guide |
|---|---|---|
| `[[sources]]` `region` | a filled [region](#regions), its [name](#named-regions), or a [region of the mesh](#regions-of-the-mesh) (`domain`, a cell zone); it may [move](#moving-regions) or [follow the solution](#regions-defined-by-the-solution), and expressions read the index of [copies](#arrays-of-copies) as `copy` | [Sources](sources.md#regions) |
| `[[probes]]` `region` | a [region](#regions) of any dimension, its [name](#named-regions), or a [region of the mesh](#regions-of-the-mesh): integrals, means, extremes over its cells (filled regions) or over a [surface or curve](#surfaces-and-curves-on-a-mesh); filled regions may [move](#moving-regions); regions of any dimension may [follow the solution](#regions-defined-by-the-solution); `copies = "each"` reduces each [copy](#arrays-of-copies) | [Probes](probes.md#region-probes) |
| `[[probes]]` `type` | [points or a curve](#points-and-curves); [boundary zones](#boundary-zones) and [surfaces](#surfaces) soon | [Probes](probes.md#probes-probes) |
| `[[write_data]]` `region`, `regions` | a [region](#regions) of any dimension or its [name](#named-regions): a filled region's cells, a surface or curve at its elements, a boundary zone, `domain`, or a [probe type as a region](#probe-types-as-regions); filled regions may [move](#moving-regions), and regions of any dimension may [follow the solution](#regions-defined-by-the-solution) | [Output](output.md#snapshots-of-regions) |
| `[[sponges]]` `region` | a filled [region](#regions), its [name](#named-regions), or a [region of the mesh](#regions-of-the-mesh), times `strength`; it may [move](#moving-regions) or [follow the solution](#regions-defined-by-the-solution) | [Sponge layers](sources.md#sponge-layers) |
| `[[sponges]]` `strength` | an expression of `x`, `y`, `z` | [Sponge layers](sources.md#sponge-layers) |
| `mallard post extract` `--region` | what a `[[probes]]` `region` takes, sampled on the state of existing output; [regions defined by the solution](#regions-defined-by-the-solution) are placed on that state | [Probes](probes.md#probes-of-existing-output) |
| `mallard check` `--mesh-fields` `--region` | a [region](#regions) or its [name](#named-regions): the cells whose stencils to write | [Mesh and stencil fields](output.md#mesh-and-stencil-fields) |
| `mallard check` (`--view DIR`) | every region used and the mesh's own: dimension, measure, cells; a ParaView file of them; regions defined by the solution are listed, placed during the run | [Checking regions](#checking-regions) |
| `[statistics.phase]` `probe` | a point | [Statistics](statistics.md#phase-averages-statisticsphase) |
| `[initialize]` `type = "analytical"` | expressions of `x`, `y`, `z`, averaged over each cell | [Initialization](initialization.md#expressions) |
| `[[initialize.regions]]` `region` | a [region](#regions) or its [name](#named-regions); cut cells averaged exactly for flat surfaces; a [moving region](#moving-regions) where it is at `t = 0` | [Initialization](initialization.md#regions) |
| `[[initialize.data]]` `region` | a [region](#regions) or its [name](#named-regions): its signed distance is a table's coordinate | [Initialization](initialization.md#tabulated-data-and-point-clouds) |
| `[[mesh.curved]]` | a surface (2D: a curve), e.g. `{type = "sphere", ..., filled = false}` or a `circle` | [Curved boundaries](meshes.md#curved-boundaries) |

Every region may be [repeated](#arrays-of-copies), and wraps across
[periodic boundaries](#periodic-boundaries). Filled regions are volumes (areas in 2D runs). Surfaces and curves, such as
boundary zones and the [boundaries](#boundaries-and-interiors) of regions, are
regions too; each feature states the dimension it takes, and a region of
another dimension is an error naming both.

Sponges and analytical initial states take expressions; a comparison
such as `"hypot(x, y) < 0.2 ? 2.0 : 1.0"` draws a sharp region there, which the
initial state averages over each cell. `[[initialize.regions]]` takes the
region itself, and also layers it over a restart.

## Coordinates and units

- **Coordinates** are those of the mesh as the run sees it, after
  [`[mesh.transform]`](meshes.md#transform-on-read).
- **2D** runs use the `z = 0` slice of each region: a sphere is a disk, a
  cylinder along z a disk, a cylinder along x a strip. Points have two
  components; a third, `z`, may be given (default 0), and the region is still
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

A region is a table: a primitive, a boolean of regions, or the
[boundary or interior](#boundaries-and-interiors) of a region, each with
optional transforms, nested to any depth; or the [name](#named-regions) of
one, or of a [region of the mesh](#regions-of-the-mesh); or a
[region defined by the solution](#regions-defined-by-the-solution). Each `type` has one
dimension, the same wherever it is used: filled (a volume; an area in 2D), a
surface, a curve. A filled region of primitives has a **signed distance** φ,
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
fields = ["RHO", "T"]
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

Pitfalls: a name that is not defined is an error that lists the defined ones
and the [regions of the mesh](#regions-of-the-mesh);
regions that name each other in a loop are an error naming the loop; a named
region no feature uses is an unused key (an error, like any other); and `edge`
belongs to the outermost region, so a named region with `edge` cannot be a part
of a boolean.

### Primitives

![The 3D region primitives with their keys: sphere, ellipsoid, cylinder, annular cylinder, cone, box, torus, half-space and an implicit superellipsoid](../images/guide/geometry-primitives-3d-light.svg#gh-light-mode-only)
![The 3D region primitives with their keys: sphere, ellipsoid, cylinder, annular cylinder, cone, box, torus, half-space and an implicit superellipsoid](../images/guide/geometry-primitives-3d-dark.svg#gh-dark-mode-only)

| `type` | Example |
|---|---|
| `sphere` (2D: also `disk`) | `{type = "sphere", center = [0, 0, 0], radius = 1}` |
| `ellipsoid` | `{type = "ellipsoid", center = [0, 0, 0], semi_axes = [1, 0.5, 0.4]}` |
| `cylinder` | `{type = "cylinder", center = [0, 0, 0], axis = [0, 0, 1], radius = 0.5, length = 2}` (no `length`: infinite) |
| `cylinder`, annular | the same with `inner_radius = 0.3`: a pipe wall |
| `annulus` (2D) | `{type = "annulus", center = [0, 0], radius = 1, inner_radius = 0.6}`: a ring |
| `cone` (`frustum`) | `{type = "cone", base = [0, 0, 0], axis = [0, 0, 1], length = 1.5, base_radius = 0.8, top_radius = 0.3}` |
| `box` (3D) | `{type = "box", center = [0, 0, 0], size = [2, 1, 0.5]}` |
| `rectangle` (2D) | `{type = "rectangle", center = [0, 0], size = [2, 1]}` |
| `torus` | `{type = "torus", center = [0, 0, 0], axis = [0, 0, 1], major_radius = 1, minor_radius = 0.25}` |
| `half_space` | `{type = "half_space", point = [1, 0, 0], normal = [1, 0, 0]}`: the side `x < 1` |
| `polygon` (2D) | `{type = "polygon", vertices = [[0, 0], [1, 0], [0.5, 1]]}` |
| `implicit` | `{type = "implicit", f = "x^4 + y^4 + z^4 - 0.41"}`: where `f < 0` |

Each type has one dimension, the same in every run: `circle` and `ellipse`
are curves; `disk`, `annulus`, `rectangle` and `polygon` are flat (areas in 2D,
surfaces in 3D); `box` is a solid, so 3D only. Sources need filled regions, so
3D runs use `sphere`, `ellipsoid`, `box` and `cylinder` (hollow with
`inner_radius`), and 2D runs `rectangle` for a rectangle; in 2D the `z = 0`
slice of an `ellipsoid` is the area inside an ellipse. Older inputs with the
former names (`circle` and `ellipse` for the filled regions, `box` in 2D,
`disk`, `annulus` and `rectangle` in 3D) are rewritten by
[`mallard input upgrade`](input_files.md#inputs-from-older-releases).

Curves and flat surfaces of 3D runs lie in the plane through `center` normal
to `normal`; `reference` (default: the axis of x, y, z least aligned with
`normal`) gives the plane's first axis, along which a `rectangle`'s first size
and an `ellipse`'s first semi-axis lie:

| `type` | Example | Dimension |
|---|---|---|
| `circle` | 2D: `{type = "circle", center = [0, 0], radius = 1}`; 3D: `{type = "circle", center = [0, 0, 0], normal = [0, 0, 1], radius = 1}` | curve |
| `ellipse` | 2D: `{type = "ellipse", center = [0, 0], semi_axes = [1, 0.5]}`; 3D: with `normal` | curve |
| `disk`, `annulus` (3D) | `{type = "disk", center = [0, 0, 0], normal = [1, 0, 0], radius = 0.5}`; `annulus` with `inner_radius` | surface |
| `rectangle` (3D) | `{type = "rectangle", center = [0, 0, 0], normal = [0, 0, 1], size = [2, 1], reference = [1, 1, 0]}` | surface |
| `polygon` (3D) | `{type = "polygon", vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 1]]}`: its normal by the right-hand rule | surface |

In 2D, each region is the `z = 0` slice; Mallard computes the cell volume
fractions below on a coarse triangle mesh, with the exact outline drawn over them:

![Cell volume fractions of the 2D primitives on a coarse triangle mesh, with the exact outline](../images/guide/region-primitives-light.svg#gh-light-mode-only)
![Cell volume fractions of the 2D primitives on a coarse triangle mesh, with the exact outline](../images/guide/region-primitives-dark.svg#gh-dark-mode-only)

### Regions of the mesh

```toml
# Total mass, and the temperature away from the heater
[[probes]]
name = "total"
fields = ["RHO"]
region = "domain"

[[probes]]
name = "away"
fields = ["T"]
region = {type = "difference", regions = ["domain", {type = "sphere", center = [0, 0, 0], radius = 0.05}]}
```

Every mesh brings regions of its own, by name, without a `[regions]` entry:

- `domain`: every cell, the filled region of the whole domain;
- each **boundary zone** (`inlet`, `wall`, `left`, ...): its faces, a surface
  (a curve in 2D) facing out of the domain;
- each **cell zone** of a mesh file: its cells, a filled region. Cell zones are
  Gmsh physical groups of cells (physical surfaces in 2D, physical volumes in
  3D), Fluent cell zones, CGNS families of element sections and OpenFOAM
  `cellZones`. Generated meshes have none.

They work wherever a region does: as a `region`, as a part of a boolean and as
the `of` of [`boundary` and `interior`](#boundaries-and-interiors). Their cells
are entirely inside or outside (fractions 0 or 1), so a boolean of one with
primitives keeps the primitives' exact fractions where it keeps them:
`domain` minus a sphere is exactly one minus the sphere's fractions.

On a square mesh with a hole (boundary zones `outer` and `hole`, cell zones
`left_half` and `right_half`), the domain minus a disk, the cell zone
`left_half` inside a circle, and the cells inside the zone `outer`:

![Volume fractions of the domain minus a disk, of a cell zone inside a circle, and of the interior of the outer boundary zone, on a square mesh with a hole](../images/guide/region-mesh-regions-light.svg#gh-light-mode-only)
![Volume fractions of the domain minus a disk, of a cell zone inside a circle, and of the interior of the outer boundary zone, on a square mesh with a hole](../images/guide/region-mesh-regions-dark.svg#gh-dark-mode-only)

Pitfalls:

- A name in `[regions]` that is also a boundary zone, a cell zone or `domain`
  is an error naming both, and so is a mesh whose cell zone has the name of a
  boundary zone or `domain`: rename one of them in the mesh file.
- Regions of the mesh do not move: transforms of a boolean that holds one, and
  `edge`, are errors.
- They need no use: unlike named regions, an unused one is not an unused key.

### Booleans

```toml
region.type = "difference"
region.a = {type = "disk", center = [0.42, 0.5], radius = 0.27}
region.b = {type = "rectangle", center = [0.64, 0.5], size = [0.5, 0.26], rotate = 25, pivot = [0.64, 0.5]}
```

`union` and `intersection` take `regions = [...]`; `difference` takes `a` and
`b`, or `regions` (the first minus the others). Booleans nest; a region
whose evaluation needs a stack deeper than 16 is an error.

![A disk A and a rotated rectangle B, then their union, intersection and difference](../images/guide/geometry-csg-light.svg#gh-light-mode-only)
![A disk A and a rotated rectangle B, then their union, intersection and difference](../images/guide/geometry-csg-dark.svg#gh-dark-mode-only)

![Cell volume fractions of a difference, a union, an intersection and a smooth-edged disk](../images/guide/region-booleans-light.svg#gh-light-mode-only)
![Cell volume fractions of a difference, a union, an intersection and a smooth-edged disk](../images/guide/region-booleans-dark.svg#gh-dark-mode-only)

### Boundaries and interiors

```toml
# A sphere, as the region its surface encloses
region = {type = "interior", of = {type = "sphere", center = [0, 0, 0], radius = 1, filled = false}}
```

```toml
# The fluid enclosed by the boundary zones "outer" and "inlet" of a mesh
region = {type = "interior", of = {type = "union", regions = ["outer", "inlet"]}}
```

`{type = "boundary", of = ...}` is the boundary of a region, one dimension
lower; `{type = "interior", of = ...}` the filled region a closed surface (a
closed curve in 2D) encloses, one dimension higher. They are inverses: the
interior of a region's boundary is the region. `filled = false` on a filled
region or a flat surface is its boundary: `{type = "sphere", ..., filled = false}`
is the sphere's surface, and in 2D the same table, or a `disk` with
`filled = false`, is the circle, `{type = "circle", ...}`.

| `of` | `boundary` | `interior` |
|---|---|---|
| a filled region | its closed surface (2D: curve), facing out | error: already filled |
| a closed surface (2D: closed curve) | error: it has none | the filled region inside |
| a flat surface (3D: `disk`, `rectangle`, ...) | its edge curve | error: not closed |
| a closed curve in 3D (`circle`, the edge of a disk) | error: it has none | error: interior takes a surface |
| boundary zones | their edges (2D: end points) if open | the cells inside if closed |

Boundaries of filled booleans are boundaries too: the boundary of `domain`
minus a sphere is the domain's boundary zones and the sphere's surface, facing
into the sphere. `filled = false` on a curve is an error that points to
`{type = "boundary", of = ...}`, which takes regions by name. The interior of
boundary zones is found by winding numbers: each connected part of the mesh
is inside or outside them as a whole, so the interior of the zones around a
domain is the domain, and that of a body's wall holds no cell.

### Clipping across dimensions

```toml
# The part of the wall zone inside a box
region = {type = "intersection", regions = ["wall", {type = "box", center = [0.5, 0, 0], size = [0.2, 1, 1]}]}
```

```toml
# A plane's section of a duct, outside a centerbody
region = {type = "difference", regions = [{type = "half_space", point = [0.3, 0, 0], normal = [1, 0, 0], filled = false}, "centerbody"]}
```

`intersection` and `difference` of one surface or curve with filled regions
clip it and keep its dimension: an `intersection` keeps the part inside every
filled region, a `difference` (the surface or curve first) the part outside
the others. Clipping keeps the surface's orientation; the boundary of a clipped
surface is where the clipping regions cut it. Two surfaces or curves combine
only by `union`, and a `difference` that subtracts a surface from a filled
region is an error (it would remove nothing).

### Orientation

Surfaces (2D: curves) face one way, which sets the sign of what is measured
through them:

| Region | Faces |
|---|---|
| a closed surface: the boundary of a filled region, `circle` and `ellipse` in 2D | out of the region it encloses |
| a boundary zone, `domain`'s boundary | out of the domain |
| a flat surface (3D `disk`, `annulus`, `rectangle`, `polygon`) | along `normal` (`polygon`: by the right-hand rule along its vertices) |
| the part of a boundary that a `difference` subtracts | into the subtracted region (out of the result) |
| a 2D curve | to the right of the direction of travel |
| a 3D curve (`circle`, `ellipse`, the edge of a surface) | runs counterclockwise about its `normal`, or about the normal of the surface it bounds (right-hand rule) |

`flip = true` on any surface or curve reverses it. Points have no orientation.

As `mallard check --view` writes them: a circle and the boundary of a
rectangle minus a disk (the disk's part facing into the disk), and the part of
a boundary zone in a box with a flipped ellipse, on a mesh with a hole:

![Normals of an exact circle, of the extracted boundary of a rectangle minus a disk, of a clipped boundary zone and of a flipped ellipse](../images/guide/region-orientation-light.svg#gh-light-mode-only)
![Normals of an exact circle, of the extracted boundary of a rectangle minus a disk, of a clipped boundary zone and of a flipped ellipse](../images/guide/region-orientation-dark.svg#gh-dark-mode-only)

### Surfaces and curves on a mesh

```toml
# The area of a sphere's surface and the mean pressure on it
[[probes]]
name = "shell"
fields = ["P"]
region = {type = "sphere", center = [0, 0, 0], radius = 0.1, filled = false}
reductions = ["integral", "mean"]
```

A surface, curve or set of points is a set of elements in the cells it
crosses: polygons, segments or points, each with its measure (area, length,
or 1 for a point) and orientation. They are made in one of two ways, and
`mallard check` reports which:

- **Exact**: the surfaces of primitives (`sphere`, `ellipsoid`, `cylinder`,
  `cone`, `box`, `torus`; in 2D their circles, ellipses and polygons) and the
  flat surfaces and curves of 3D runs follow their own parameters, cut into
  patches about half a cell wide and integrated with Gauss points, so an
  unclipped surface has its exact area to round-off; planes (`half_space`) are
  cut from the cells exactly; boundary zones and the faces around regions of
  the mesh are the mesh's faces.
- **Extracted**: the boundaries of booleans and of `implicit` regions are cut
  from the cells as the zero set of the signed distance, linear on each cell's
  triangles (tetrahedra) subdivided `resolution` times where the surface may
  cross them (default 3 in 2D, 2 in 3D): second-order accurate in the
  subdivided size.

Exact surfaces follow these parameters `(u, v)`, with `x_u x x_v` facing
out; a probe resampling a surface region (`[[probes]]` with `region` and
`n_points`) takes `n_points = [n_u, n_v]` nodes along each patch, and a parameter that closes a turn
does not repeat its first node:

| Surface | `u` | `v` |
|---|---|---|
| `sphere`, `ellipsoid` | polar angle, from the local +z pole | angle around local z, from +x |
| `cylinder`, `cone` (side) | angle around the axis | along the axis, from the base |
| their ends | radius (top), angle (bottom) | angle (top), radius (bottom) |
| `box` faces | the face's next axis | the one after (cyclic x, y, z) |
| `torus` | angle around the axis | angle around the tube |
| `disk`, `annulus` (3D) | radius, from `inner_radius` | angle from `reference` |
| `rectangle` (3D) | along `reference` | along `normal x reference` |
| `polygon` (3D) | each triangle of its ear clipping | |

Clipping regions cut exact surfaces where their own surface crosses them:
along lines of the surface's parameters, the Gauss points fill the kept
intervals between the clipping region's roots, found to round-off, so a clipped
area or flux converges at high order (in the tests, errors fall by over 8 per
halving of the cells, to round-off); patches where the clipping boundary is not
a graph over one parameter, or that several clipping regions cross, are
subdivided. Extracted elements and mesh faces are cut along their edges at the
roots of the clipping region's distance. Only the parts inside the domain
count. In axisymmetric runs measures are those of the r-z plane, not of the
revolved surface.

On a mesh with [curved boundaries](meshes.md#curved-boundaries), every kind
follows the curved walls:

- Exact surfaces' Gauss points are located in the cells' curved shapes, so
  the slivers between a straight face and the wall count, and points beyond
  the wall do not. A patch the wall crosses is halved until its kept part ends
  within 2^-30 (curves) or 2^-6 (surfaces) of the patch at the wall.
- Boundary zones, and the faces around regions of the mesh where they lie on
  a curved wall, are the curved faces, with Gauss points, normals and areas
  of the curved geometry; clipped, their pieces are cut on the curved face.
- Extracted surfaces and planes are cut from the curved cells, split into
  cones from the cell's vertex average to each face, each curved face in 8
  (3D: 4 x 4) flat pieces.

![The closed surface of a disk's part inside a curved wall, as mallard check --view writes it: the disk's edge in the domain and the wall inside the disk with their outward normals, and the net mass outflow of a uniform expansion against 0.2 times the area inside](../images/guide/region-control-surface-light.svg#gh-light-mode-only)
![The closed surface of a disk's part inside a curved wall, as mallard check --view writes it: the disk's edge in the domain and the wall inside the disk with their outward normals, and the net mass outflow of a uniform expansion against 0.2 times the area inside](../images/guide/region-control-surface-dark.svg#gh-dark-mode-only)

### Checking regions

`mallard check` lists every region a case uses, and the mesh's own, with its
kind, measure and cells; surfaces and curves say whether they are exact
or extracted. Filled regions of the case also get their own volume and
surface area (2D: area and perimeter), independent of the mesh, against those
of their part in the domain, and the share outside it, as
[region probes](probes.md#region-probes) record them. A region that covers no
cell is an error (`E-REGION-OUTSIDE` or `E-REGION-EMPTY`) naming it with its
extent and the domain's. [Regions defined by the solution](#regions-defined-by-the-solution)
are listed with `-` for measure and cells, made from `the solution, during
the run`, and `"solution": true` in `--json`; the case view leaves them out:

```
== Regions =========================================================================== ERROR
  The mesh's own (the domain, its boundary and cell zones) and every region the case uses;
  cells: the cells each touches. A region of the case that touches no cell is an error.
  status  region                        kind    measure  units  cells  made from
  ok      domain                        filled  2        m^2    128    the mesh's
  ok      bottom                        curve   2        m      16     the mesh's
  ok      left                          curve   1        m      8      the mesh's
  ok      right                         curve   1        m      8      the mesh's
  ok      top                           curve   2        m      16     the mesh's
  ERROR   sponges[0].region             filled  0        m^2    0      cell fractions
  ok      probes[name = "ring"].region  curve   1.5708   m      13     exact
  Filled regions: their part in the domain (its surface: theirs inside the domain and the
  domain's boundary inside them) and their own measures, independent of the mesh ("-" for
  regions of the mesh or unbounded ones), exact or within a relative tolerance; outside: the
  share outside the domain.
  status  region             quantity   in domain  own  units  tolerance  outside
  ok      sponges[0].region  area       0          2    m^2    exact      100%
  ok      sponges[0].region  perimeter  0          6    m      exact
...
  ERROR   E-REGION-OUTSIDE  region sponges[0].region lies outside the domain: it spans (4.5, 4) to (5.5, 6) m, the
                            domain (0, 0) to (2, 1) m: move or resize it (--view shows both)
                            docs/guide/input_files.md#e-region-outside
```

`mallard check input.toml --view view` also writes `view/case.vtkhdf`, the
case view: one block per region (filled regions as their cells with
`fraction`; surfaces, curves and points with `normal` or, for 3D curves,
`tangent`, and their `area`, `length` or `count`) and per boundary zone. Open
it in ParaView (5.12 or later) and pick blocks in the Multiblock Inspector;
builds without HDF5 write `view/case.vtm`. More of the case (conditions,
probes, injectors) is to join it
([#546](https://github.com/MatthewBonanni/mallard/issues/546)).

### Transforms

Any region, primitive or boolean, may be scaled, rotated and translated, in
that order. `scale` and `rotate` act about `pivot` (default the origin).

![An L shape translated, rotated about the origin, rotated about a pivot, scaled, and all of these in order](../images/guide/geometry-transforms-light.svg#gh-light-mode-only)
![An L shape translated, rotated about the origin, rotated about a pivot, scaled, and all of these in order](../images/guide/geometry-transforms-dark.svg#gh-dark-mode-only)

| Key | 2D | 3D |
|---|---|---|
| `scale` | a factor, or `[s_x, s_y]` | a factor, or `[s_x, s_y, s_z]` |
| `rotate` | degrees, counterclockwise | `{axis = [...], angle = degrees}` (right-handed), or `{axes = [[...], [...], [...]]}`: the region's x, y, z axes in world coordinates |
| `pivot` | center of `scale` and `rotate` | same |
| `translate` | a displacement | same |

Pitfall: without `pivot`, `rotate` turns the region about the origin, which
also moves it. To turn a region in place, give its center as `pivot`.

**Boxes not aligned with the grid.** Rotate a rectangle (3D: a box) about its center: Mallard
computes the fraction of each cell that it covers, so it needs no
grid lines along its sides. A test of cell centroids would make a staircase
and miss part of the area:

```toml
region = {type = "rectangle", center = [0.5, 0.5], size = [0.66, 0.28], rotate = 30, pivot = [0.5, 0.5]}
```

![A rotated rectangle on an 18 by 18 Cartesian mesh: cells whose centroid is inside, Mallard's volume fractions, and the fractions with a smooth edge](../images/guide/geometry-rotated-box-light.svg#gh-light-mode-only)
![A rotated rectangle on an 18 by 18 Cartesian mesh: cells whose centroid is inside, Mallard's volume fractions, and the fractions with a smooth edge](../images/guide/geometry-rotated-box-dark.svg#gh-dark-mode-only)

In 3D, rotate about an axis, or give the region's own axes:

```toml
region = {type = "box", center = [1, 0, 0], size = [0.4, 0.2, 0.2], rotate = {axis = [0, 0, 1], angle = 30}, pivot = [1, 0, 0]}
```

```toml
region = {type = "box", center = [0, 0, 0], size = [1, 0.5, 0.2], rotate = {axes = [[0.6, 0.8, 0], [-0.8, 0.6, 0], [0, 0, 1]]}}
```

**Annular cylinders** have an `inner_radius`; tilt them with `axis`:

```toml
region = {type = "cylinder", center = [0.5, 0.5, 0.5], axis = [1, 1, 0.6], radius = 0.33, inner_radius = 0.18, length = 0.5}
```

![Volume fractions in a tetrahedral mesh of a tilted annular cylinder, a rotated box and a drilled sphere, cut open](../images/guide/regions-3d-light.svg#gh-light-mode-only)
![Volume fractions in a tetrahedral mesh of a tilted annular cylinder, a rotated box and a drilled sphere, cut open](../images/guide/regions-3d-dark.svg#gh-dark-mode-only)

### Smooth edges

`edge` (on the outermost region only, a filled region of primitives) replaces the sharp surface by a band of
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

Any region `f(x, y, z) < 0`, in the [expression syntax](../input.md#regions).
Implicit regions combine and transform like the others (`f` reads the region's
local coordinates).

Pitfalls:

- `edge` is in units of `f`. It is a width in meters only when `f` is a
  distance, such as `"sqrt(x^2 + y^2) - 1"`.
- `f` gives no bound on the distance to its surface, so Mallard subdivides a
  cell unless `f` keeps its sign across it. A thin feature that falls between
  the sample points of a coarse cell can be missed: refine the mesh, or write
  `f` as a distance.

### Arrays of copies

```toml
[regions]
# Eight injectors on a ring, and a 4 by 3 grid of heaters
injectors = {type = "disk", center = [0.8, 0], radius = 0.05, repeat = {count = 8, rotate = 45}}
heaters = {type = "rectangle", center = [0.1, 0.1], size = [0.05, 0.02], repeat = [
    {count = 4, translate = [0.2, 0]},
    {count = 3, translate = [0, 0.25]}]}
```

`repeat` on any region (a primitive, a boolean, a named region) makes it the
union of `count` copies: copy `j` is the region translated `j` times by
`translate`, or rotated `j` times by `rotate` about `pivot` (default the
origin; 3D: `rotate = {axis = [...], angle = degrees}`). A list of them
repeats the copies of the levels before it, so arrays of arrays compose:
the 4 by 3 grid above has 12 copies, numbered with the first level listed
fastest (copy `k` is column `k % 4`, row `k / 4`). `repeat` applies after the
region's own transforms. Regions of the mesh do not repeat.

The copies act as one region, or one at a time:

- **Sources** read the copy's index as `copy` in their expressions: each copy
  then has its own term, and `total = true` spreads its total over that copy
  alone. `rho = "1e-3 * (1 + copy)"` gives copy `k` a mass of `(k + 1)` g/s.
- **Probes** with `copies = "each"` write one reductions file per copy,
  `<name>_copy<k>` (and `reductions_file` gets `_copy<k>` before its extension).
- **Measures** of disjoint copies are their sum in closed form; overlapping
  copies are measured as their union.

### Moving regions

```toml
# A heater that moves with the flow, and a sponge that grows
[[sources]]
rhoE = "1 MW/m^3"
region = {type = "disk", center = ["0.2 + 3 * t", 0.5], radius = "1 cm"}

[[sponges]]
strength = 50
u = [0, 0]
p = 101325
T = 300
region = {type = "disk", center = [1, 0.5], radius = "0.1 + 0.05 * t"}
```

Centers, points, sizes (`radius`, `inner_radius`, `length`, `size`, a cone's
radii, a torus' radii), the transforms (`translate`, `rotate` angles, `scale`)
and an implicit region's `f` may be expressions of `t`, the time in seconds:
the region moves. At every stage Mallard places it where it is at the stage's
time and recomputes its cell fractions on the device, curved cells on their
curved shapes, so a captured CUDA graph of the step follows it with no host
work. A total of a moving region (`total = true`) is divided by the integral
of its fractions at each stage, so it stays exact as the region enters or
leaves the domain.

Who follows a moving region:

- `[[sources]]` and `[[sponges]]`: at every stage. A sponge's `strength` is
  that of the cells, times the fraction the region has there at the stage;
  its time step limit takes the fraction as 1.
- `[[probes]]` of filled regions: at every write. The reductions file's
  measure column (`area`, `volume`) is the region's measure in the domain at
  that time; the file's `measures.region` is the region's own while it only
  translates and rotates, and `null` once it changes shape (`scale` or sizes
  of `t`, an implicit `f` of `t`).
- `mallard check`, `[[initialize.regions]]` and `[[initialize.data]]` take the
  region where it is at `t = 0`.

A [snapshot](output.md#snapshots-of-regions) of a moving filled region
(`[[write_data]]`) holds the cells it covers at each write. Surfaces, curves
and points that move, and averaged probes of them, are errors for now
([#747](https://github.com/MatthewBonanni/mallard/issues/747)).

### Regions defined by the solution

```toml
# The burnt gas, and the flame as the isosurface T = 1500 K
[regions]
burnt = {type = "where", condition = "T > 1500"}

[[probes]]
name = "burnt"
region = "burnt"
fields = ["RHO", "T"]
reductions = ["integral", "mean"]

[[probes]]
name = "flame"
region = {type = "isosurface", field = "T", value = "1500 K"}
fields = ["MASS_FLUX", "P"]
reductions = ["integral", "mean"]
```

A `where` region is filled where its `condition` holds on the solution: a
comparison (`<`, `<=`, `>`, `>=`) of expressions in `x`, `y`, `z`, `t` and the
solution's `RHO`, `U_X`, `U_Y`, `U_Z`, `P`, `T` and `Y_<species>` (SI units),
or several joined by `and`, `or`, `not` and parentheses
(`"T > 1500 and not (Y_OH < 1e-3)"`). An `isosurface` (2D: a curve) is where
`field`, one of those variables, equals `value`, given in the field's units;
it faces toward increasing `field`, and `flip = true` reverses it. It is the
[boundary](#boundaries-and-interiors) of `{type = "where", condition =
"<field> < <value>"}`, and the boundary of any `where` region is its own
isosurface, facing out of it.

Each cell's **level** is linear: the value of the condition's level function
(for `a < b`, `a - b`; `and` the larger of two, `or` the smaller, `not` its
negative; the condition holds where it is negative) at the cell's centroid,
with its gradient fit by least squares to the values of the cells sharing a
node with it, across periodic boundaries too. A cell's volume fraction is the exact part of its
straight triangles (tetrahedra) where that linear level is negative, and an
isosurface's elements are the level's zero set on them. Both are exact for a
linear field and converge at second order otherwise, smoothly as the region
moves through the cells. Booleans with regions of shapes or of the mesh
combine the cell fractions (`intersection` their product, `union` the
larger, `difference` the first times one minus the second), exact where a
cell is wholly inside or outside one of them.

![Volume fractions of where T > 1.5 over two hot spots on a coarse triangle mesh, with the exact isotherm; and the relative errors of a where region's area and of its isosurface's length on refined triangle meshes, both converging at second order](../images/guide/region-solution-light.svg#gh-light-mode-only)
![Volume fractions of where T > 1.5 over two hot spots on a coarse triangle mesh, with the exact isotherm; and the relative errors of a where region's area and of its isosurface's length on refined triangle meshes, both converging at second order](../images/guide/region-solution-dark.svg#gh-dark-mode-only)

Who follows the solution, and when:

- `[[sources]]` and `[[sponges]]`: at every stage, on the device, as a captured
  CUDA graph of the step does with no host work. A total (`total = true`) is
  divided by the integral of the fractions at the stage, and is zero while the
  region is empty.
- `[[probes]]`: at each write, a `where` region's fractions and an
  isosurface's elements, extracted anew, and at each sample of a probe with
  `average = true`. The measure column of the reductions file is the region's
  at that time. The run-health timing row `regions` is
  the time this takes.
- `[[write_data]]`: at each write, a `where` region's cells (those of positive
  fraction, with `REGION_FRACTION`) and an isosurface's elements, so each
  [snapshot](output.md#snapshots-of-regions) is a mesh of its own.
- `mallard post extract`: the same regions on the state of each restart file
  or snapshot it reads, in its probes and `--snapshot`
  ([Probes of existing output](probes.md#probes-of-existing-output)), e.g.
  `--region '{type = "isosurface", field = "T", value = 1500}'`.
- `mallard check` lists them, with no measure or cells: the solution places
  them during the run.

They take no transforms, `repeat` or `edge`. Probes of `n_points` on
isosurfaces, whose elements have no parameters to place a grid of samples on,
`[[initialize.regions]]` and `[[initialize.data]]` are errors.

### Periodic boundaries

On a mesh with [periodic boundaries](meshes.md#periodic-boundaries), a region
that crosses one wraps: its part beyond the boundary enters across the other
side, as the region's images one period away (translational, rotational or
both), so it covers its whole measure in the domain. There is no key; a region
that crosses no periodic boundary is unchanged. Copies wrap each on their own,
and a moving region wraps while it stays within one period of where it is at
`t = 0`. `mallard check` and the run log
note `wrapped across periodic boundaries (N images)`.

![Volume fractions of eight copies of a disk about a pivot on a triangle mesh, and of a disk across the periodic boundary of a box periodic in x](../images/guide/region-arrays-light.svg#gh-light-mode-only)
![Volume fractions of eight copies of a disk about a pivot on a triangle mesh, and of a disk across the periodic boundary of a box periodic in x](../images/guide/region-arrays-dark.svg#gh-dark-mode-only)

![The same in 3D on tetrahedral meshes, cut open: six copies of a sphere about the z axis, and a sphere across a periodic boundary](../images/guide/region-arrays-3d-light.svg#gh-light-mode-only)
![The same in 3D on tetrahedral meshes, cut open: six copies of a sphere about the z axis, and a sphere across a periodic boundary](../images/guide/region-arrays-3d-dark.svg#gh-dark-mode-only)

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
volume fractions of the revolved cells. Cells at a
[curved wall](meshes.md#curved-boundaries) are sampled in their curved shapes,
as cones from the cell's vertex average to each face, subdivided the same way,
so the fractions are of the curved cells and `sum f V` counts the slivers
between the straight faces and the wall.

### Probe types as regions

```toml
[regions]
mid_plane = {type = "cut_plane", point = [0.5, 0, 0], normal = [1, 0, 0]}
wake_line = {type = "line", start = [1, -0.5, 0], end = [1, 0.5, 0], n_points = 101}
```

A table whose `type` is one of the [probe types](#points-and-curves) `point`,
`points`, `line`, `polyline`, `arc`, `curve`, `cut_plane` or `surface` is a
region of lower dimension for `[[write_data]]` and `[[probes]]`: it takes that
probe type's keys, below and in [`[[probes]]`](../input.md#probes), and its
samples go where the probe puts them. It cannot be a part of a boolean, takes no
`edge`, and transforms as probes do (`scale`, `rotate`, `translate`, about the
origin).

## Points and curves

```toml
[[probes]]
name = "wake"
fields = ["U", "P"]
type = "arc"
center = [0.0, 0.0]
radius = 5.0
angles = [-45.0, 45.0]
n_points = 91
```

Probes sample at points (`point`, `points`, `points_file`) or along a curve
(`line`, `polyline`, `circle`, `arc`, `curve`). Keys:
[`[[probes]]`](../input.md#probes).

![The 2D probe types with the samples Mallard places on them; a ring marks each one's first sample](../images/guide/geometry-probes-light.svg#gh-light-mode-only)
![The 2D probe types with the samples Mallard places on them; a ring marks each one's first sample](../images/guide/geometry-probes-dark.svg#gh-dark-mode-only)

![A tilted circle, a helix and a line in a 3D domain, with their samples](../images/guide/geometry-probes-3d-light.svg#gh-light-mode-only)
![A tilted circle, a helix and a line in a 3D domain, with their samples](../images/guide/geometry-probes-3d-dark.svg#gh-dark-mode-only)

### Where the samples go

| `type` | Samples |
|---|---|
| `point`, `points`, `points_file` | the points as given, in order |
| `line` | `n_points` evenly from `start` to `end`, both included |
| `polyline` | `n_points` evenly on each segment, ends included; shared vertices once |
| `circle` | `n_points` evenly, the first along `reference` (3D) or +x (2D), not repeated at the end |
| `arc` | `n_points` evenly from the first angle to the second, both included |
| `curve` | `n_points` evenly in the parameter `s` over `s_range`: denser where the curve moves slowly in `s` |

Transforms of probes are `scale`, then `rotate`, then `translate`, all about
the origin (probes have no `pivot`): the scaled and rotated circle above is
centered at the origin, then translated. A `point` or `points` probe with a
point outside the domain is an error, and curves are clipped to the domain:
[Samples outside the domain](probes.md#samples-outside-the-domain).

Curve reductions (`mean`, `integral`, ...) integrate along the curve with three
Gauss points between consecutive samples; points weigh each sample 1.

### Cell averages or point values

`sampling = "cell"` (the default) returns the average of the cell holding each
sample: first-order accurate at a point off the cell's centroid.
`sampling = "point"` evaluates the reconstruction at the sample, which with
TENO converges at the scheme's order:

![Error of probe values against cell size: cell averages first order, MUSCL point values second, TENO orders 3 and 5 at their order](../images/guide/geometry-sampling-light.svg#gh-light-mode-only)
![Error of probe values against cell size: cell averages first order, MUSCL point values second, TENO orders 3 and 5 at their order](../images/guide/geometry-sampling-dark.svg#gh-dark-mode-only)

Use `point` to compare with references or to measure orders of accuracy, and
`cell` for robust values across shocks. Details and pitfalls:
[Probes and monitors](probes.md#accuracy-and-pitfalls).

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
header, its units line and a `# mallard-run` line:

```python
import numpy as np

d = np.genfromtxt("probe_wake.csv", delimiter=",", names=True)  # skips the comment lines
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

- [Input reference: regions](../input.md#regions), [regions of the mesh](../input.md#regions-of-the-mesh), [`[regions]`](../input.md#named-regions-regions), [`[[sources]]`](../input.md#sources), [`[[sponges]]`](../input.md#sponges), [`[[probes]]`](../input.md#probes), [`[mesh]`](../input.md#mesh)
- [Sources and sponges](sources.md), [Probes and monitors](probes.md), [Meshes](meshes.md#curved-boundaries)
- Figures: [`tools/docs/geometry.py`](../../tools/docs/geometry.py), [`tools/docs/regions.py`](../../tools/docs/regions.py)
