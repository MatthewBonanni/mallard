# Meshes

Mallard is a cell-centered finite volume solver for unstructured meshes:
triangles and quadrilaterals in 2D; tetrahedra, hexahedra, prisms and
pyramids in 3D, in any mix. The mesh comes from `[mesh]`: generated boxes for
simple domains, Gmsh files for anything else, and Mallard's HDF5 mesh format
for large distributed runs. Each boundary face belongs to a named **zone**,
which `[[boundaries]]` assigns a condition to.

The spatial dimension is a build option (`-DMallard_DIM=2`, the default, or
`3`): one solver per dimension, `mallard2d` and `mallard3d`, with the same input
keys. `mallard` runs the one a case needs, by its mesh: a mesh file's own
dimension, and for a generated mesh 3D when `[mesh]` sets `Nz` or `Lz` (or
`periodic` has `z`, or the type has 3D cells). `mallard --dim 2|3 -i input.toml`
states it.

## Generated meshes

```toml
[mesh]
type = "cartesian"    # quadrilaterals
Nx = 200              # cells along x
Ny = 100              # cells along y
Lx = 2.0              # the domain is [0, Lx] x [0, Ly]
Ly = 1.0
```

*Used in:* [`sod`](https://matthewbonanni.github.io/mallard/docs/examples/#sod), [`shu_osher`](https://matthewbonanni.github.io/mallard/docs/examples/#shu-osher), [`riemann_2d`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d), [`riemann_2d_quads`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d-quads), [`double_mach`](https://matthewbonanni.github.io/mallard/docs/examples/#double-mach) and 32 more.

![The three generated 2D meshes: quadrilaterals, triangles split along the bottom-left to top-right diagonal, and the 8 degree compression ramp](../images/guide/mesh_types_2d-light.svg#gh-light-mode-only)
![The three generated 2D meshes: quadrilaterals, triangles split along the bottom-left to top-right diagonal, and the 8 degree compression ramp](../images/guide/mesh_types_2d-dark.svg#gh-dark-mode-only)

| `type` | Cells |
|---|---|
| `cartesian` | Quadrilaterals (2D) or hexahedra (3D) |
| `cartesian_tri` | Each quadrilateral split into two triangles along its bottom-left to top-right diagonal |
| `wedge` | Quadrilaterals over an 8 degree compression ramp starting at x = 0.5 (the `wedge` example) |
| `cartesian_tet`, `cartesian_prism`, `cartesian_pyramid`, `cartesian_mixed` | 3D only, see below |
| `annulus` | The annulus `r_inner <= r <= r_outer` (default 1, 2) about the z axis, or its sector of `angle` degrees (default 360), of `Ntheta x Nr` quadrilaterals (default 64 x 16); in 3D extruded over `[0, Lz]` into `Nz` layers of hexahedra. Zones `inner`, `outer`, for a sector `start` and `end`, in 3D `back` and `front` |

The zones of a generated mesh are `left` (x = 0), `right` (x = Lx), `bottom`
(y = 0), `top` (y = Ly) and, in 3D, `back` (z = 0) and `front` (z = Lz).

In 3D, the box `[0, Lx] x [0, Ly] x [0, Lz]` is cut into `Nx x Ny x Nz`
blocks, each filled with cells of the chosen type (`cartesian_mixed`:
hexahedra, pyramids and prisms in successive thirds of x):

![The cells of one block of each generated 3D mesh type](../images/guide/mesh_types_3d-light.svg#gh-light-mode-only)
![The cells of one block of each generated 3D mesh type](../images/guide/mesh_types_3d-dark.svg#gh-dark-mode-only)

| Key | Default | Meaning |
|---|---|---|
| `type` | `"file"` | Mesh type, above |
| `Nx`, `Ny`, `Nz` | 100 | Cells (2D) or blocks (3D) along each axis |
| `Lx`, `Ly`, `Lz` | 1.0 | Domain size |
| `stretching` | zeros | One factor per axis clustering nodes toward both ends |
| `periodic` | none | Directions joined periodically, e.g. `["x", "z"]`; `annulus`: `"theta"`, `"z"` |

### Stretching

```toml
[mesh]
type = "cartesian"
Nx = 64
Ny = 64
Lx = 1.0
Ly = 2.0
stretching = [0.0, 1.7]   # cluster cells toward y = 0 and y = Ly
```

*Used in:* [`channel_retau180`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-retau180), [`channel_inflow_retau180`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-inflow-retau180), [`channel_les`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-les).

`stretching = [beta_x, beta_y]` (three factors in 3D) clusters nodes toward
both ends of each axis with a nonzero factor: node `j` of `N` sits at
`L/2 (1 + tanh(beta (2j/N - 1)) / tanh(beta))`. Wall-bounded flows use it to
resolve both walls of a channel:

```toml
[mesh]
type = "cartesian"
Nx = 192
Ny = 96
Nz = 128
Lx = 12.566370614359172
Ly = 2.0
Lz = 4.1887902047863905
stretching = [0.0, 1.7, 0.0]   # cells at the walls ~8x thinner than at the center
periodic = ["x", "z"]
```

![Uniform and stretched node spacing across a channel](../images/guide/stretching-light.svg#gh-light-mode-only)
![Uniform and stretched node spacing across a channel](../images/guide/stretching-dark.svg#gh-dark-mode-only)

## Periodic boundaries

```toml
[mesh]
type = "cartesian"
Nx = 64
Ny = 64
periodic = ["x"]     # left and right joined
```

*Used in:* [`rotational_sector`](https://matthewbonanni.github.io/mallard/docs/examples/#rotational-sector), [`poiseuille_pipe`](https://matthewbonanni.github.io/mallard/docs/examples/#poiseuille-pipe), [`taylor_green_3d`](https://matthewbonanni.github.io/mallard/docs/examples/#taylor-green-3d), [`isotropic_turbulence`](https://matthewbonanni.github.io/mallard/docs/examples/#isotropic-turbulence), [`channel_retau180`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-retau180) and 10 more.

Periodic boundaries join two zones into interior faces: every scheme sees the
seam as the inside of the domain, and the zones disappear from
`[[boundaries]]` (a fully periodic box needs no `[[boundaries]]` at all).

![A periodic seam: the cells of zone A are translated next to zone B, so the faces of the seam become interior faces](../images/guide/periodic-light.svg#gh-light-mode-only)
![A periodic seam: the cells of zone A are translated next to zone B, so the faces of the seam become interior faces](../images/guide/periodic-dark.svg#gh-dark-mode-only)

- **Generated meshes:** `periodic = ["x", "y"]` in `[mesh]` joins
  `left`/`right`, `bottom`/`top` and (3D, `"z"`) `back`/`front`. Not for
  `wedge`.
- **Mesh files:** pair zones with `[[periodic]]`; zone B must be zone A
  translated node for node (to 1e-6 of the zones' shortest edge), as Gmsh's
  `Periodic` constraint makes it. A node without a partner is reported with
  its coordinates.

```toml
[mesh]
type = "file"
filename = "box.msh"

[[periodic]]
zones = ["left", "right"]
translation = [1.0, 0.0]     # B = A + translation
```

Each periodic direction needs at least 3 cells, and each zone joins at most
one pair. Periodic meshes work in distributed runs, and with every
reconstruction ([design order across seams](https://matthewbonanni.github.io/mallard/validation/#periodic-seams)).

### Rotational periodicity

Sectors of annular combustors or blade rows are periodic through a rotation:
zone B is zone A turned by `angle` degrees about an axis. Give `angle`
instead of `translation` (with `center`, a point of the axis, and in 3D
`axis`); the generated `annulus` mesh takes `periodic = ["theta"]`.
Velocity, momentum and gradients turn across the seam; density, pressure,
temperature, species and scalars carry over unchanged.

![A sector of an annulus: zone B is zone A turned by alpha about the axis, and the velocity of a cell next to B, seen from the cells next to A, turns with it](../images/guide/rotational_periodic-light.svg#gh-light-mode-only)
![A sector of an annulus: zone B is zone A turned by alpha about the axis, and the velocity of a cell next to B, seen from the cells next to A, turns with it](../images/guide/rotational_periodic-dark.svg#gh-dark-mode-only)

| Key | Default | Meaning |
|---|---|---|
| `angle` | none | Degrees from A to B, right-handed about `axis` (counterclockwise in 2D) |
| `center` | origin | A point of the axis (2D: the center of rotation) |
| `axis` | `[0, 0, 1]` | (3D) The axis direction |

```toml
[[periodic]]
zones = ["side_a", "side_b"]   # a 30 degree sector about the z axis
angle = 30.0
```

Pitfalls:

- One rotation per mesh; in 3D it combines with translations along its axis
  only (an annular sector periodic in z), in 2D with none.
- The sector must not reach the axis, and needs at least 3 cells around.
- MUSCL limits each Cartesian velocity component and TENO-E weighs Cartesian
  derivatives, so a sector reproduces the full annulus to rounding only for
  multiples of 90 degrees; other angles agree to the truncation error.
- Transmissive boundary images and the transverse terms of characteristic
  boundaries do not reach across the rotation; axisymmetric runs take none.

![The full annulus and a quarter run with rotational periodicity agree to rounding](../images/guide/rotational_sector-light.svg#gh-light-mode-only)
![The full annulus and a quarter run with rotational periodicity agree to rounding](../images/guide/rotational_sector-dark.svg#gh-dark-mode-only)

The quarter of `examples/rotational_sector` against its full annulus: the
largest difference over all cells is that of the annulus with its own copies.

## Mesh files (Gmsh)

```toml
[mesh]
type = "file"
filename = "cylinder.msh"
```

*Used in:* [`sphere_mach3`](https://matthewbonanni.github.io/mallard/docs/examples/#sphere-mach3), [`mallard`](https://matthewbonanni.github.io/mallard/docs/examples/#mallard), [`nozzle`](https://matthewbonanni.github.io/mallard/docs/examples/#nozzle), [`sphere_axisymmetric`](https://matthewbonanni.github.io/mallard/docs/examples/#sphere-axisymmetric), [`cylinder`](https://matthewbonanni.github.io/mallard/docs/examples/#cylinder) and 3 more.

Mallard reads ASCII Gmsh meshes, format 2.2 or 4.1, of linear triangles and
quadrilaterals (2D) or tetrahedra, pyramids, prisms and hexahedra (3D).

- Each **physical curve** (2D) or **physical surface** (3D) becomes a zone
  named after it (`physical_<tag>` if it has no name). Boundary faces in no
  physical group form the zone `unassigned`.
- Elements of other dimensions (points, and curves in 3D) are ignored.
- Second- and third-order elements are read by their corners; high-order
  boundary elements also [curve their faces](#curved-boundaries).

Several examples build their meshes with scripts in `tools/` (numpy only,
except where noted):

| Script | Mesh |
|---|---|
| `make_cylinder_mesh.py` | O-grid of quadrilaterals around a cylinder; `--half` for the axisymmetric sphere; `--order 2` or `3` for curved high-order boundary elements |
| `make_annulus_mesh.py` | Quadrilaterals between two circles |
| `make_mallard_mesh.py` | Triangles around a flying mallard (needs the `gmsh` Python module) |
| `make_sphere_mesh.py`, `make_sphere_re300_mesh.py` | Tetrahedra, and prism layers plus tetrahedra, around a sphere (need `gmsh`) |
| `make_volvo_mesh.py` | Hexahedra of the Volvo bluff-body combustor |

![The O-grid of the cylinder example (drawn coarser than it runs), with its zones](../images/guide/domain_cylinder-light.svg#gh-light-mode-only)
![The O-grid of the cylinder example (drawn coarser than it runs), with its zones](../images/guide/domain_cylinder-dark.svg#gh-dark-mode-only)

*The `cylinder` example: an O-grid from `make_cylinder_mesh.py` (drawn here
four times coarser than the example's 384 x 128 cells), with the zones
`cylinder` and `farfield`.*

## Other mesh formats

```toml
[mesh]
type = "file"
filename = "wing.cgns"    # or a Fluent .msh/.cas, or an OpenFOAM case directory
```

*Used in:* the mesh-import tests (`test/data/meshes/`).

| Format | `filename` | Zones | Build |
|---|---|---|---|
| CGNS (unstructured, HDF5-based) | `*.cgns` | `BC_t` by family name (Gmsh's physical names), else by BC name; boundary sections in no BC by section name | `-DMallard_ENABLE_CGNS=ON` |
| ANSYS Fluent mesh or case, ASCII or binary | `*.msh` starting with `(`, `*.cas` | Non-interior face zones, by name | always |
| OpenFOAM polyMesh, ASCII or binary | the case directory (or its `constant/polyMesh`) | Patches (2D: all but `empty`) | always |

Every format gives the same mesh as the Gmsh file it was exported from (the
tests check cells, faces and zones), so a case can switch formats without
renaming its `[[boundaries]]`.

![One mesh read from Gmsh, CGNS, Fluent and OpenFOAM, with its boundary zones](../images/guide/mesh_import-light.svg#gh-light-mode-only)
![One mesh read from Gmsh, CGNS, Fluent and OpenFOAM, with its boundary zones](../images/guide/mesh_import-dark.svg#gh-dark-mode-only)

Pitfalls:

- CGNS: one base, unstructured zones only; several zones are joined only
  through `Abutting1to1` vertex connectivity (as Gmsh writes partitioned
  meshes); polyhedral `NGON_n`/`NFACE_n` sections are rejected. Higher-order
  elements are read by their corners, and high-order boundary elements curve
  faces as in Gmsh files.
- Fluent: cells must be triangles and quadrilaterals (2D) or tetrahedra,
  pyramids, prisms and hexahedra (3D); polyhedral meshes and adapted meshes
  (cell and face trees) are rejected. Periodic face zones stay boundary
  zones: pair them with `[[periodic]]`.
- OpenFOAM: the 2D build reads a mesh one cell thick in z between `empty`
  patches (OpenFOAM's 2D convention); compressed (`.gz`) and decomposed
  (`processor*`) meshes must be decompressed or reconstructed first. Cells
  must be of Mallard's types, as for Fluent.
- `mallard` runs the solver of the mesh's dimension, which must be installed
  (built with the matching `-DMallard_DIM`); `mallard2d` and `mallard3d` run
  directly reject a mesh of the other dimension.

## Transform on read

```toml
[mesh.transform]
scale = 0.001                            # mm to m; or one factor per axis
rotate = 90                              # 2D: degrees counterclockwise
# rotate = {axis = [0, 0, 1], angle = 90}  3D: right-handed about the axis
translate = [0.1, 0.0]
```

*Used in:* the mesh-import tests.

`[mesh.transform]` maps every node of a mesh file, `x -> R (S x) + t`: scale,
then rotate about the origin, then translate.

![A mesh as read and with a transform; + marks the origin](../images/guide/mesh_transform-light.svg#gh-light-mode-only)
![A mesh as read and with a transform; + marks the origin](../images/guide/mesh_transform-dark.svg#gh-dark-mode-only)

Pitfalls:

- Everything else in the input (`[[periodic]]` translations,
  `[[mesh.curved]]` shapes, probes, initial conditions) is in the
  transformed coordinates.
- Mesh files only: generated meshes reject it.
- `mallard mesh convert input.toml mesh.h5` writes the transformed mesh; do
  not transform it again when running from `mesh.h5`.

## HDF5 meshes for large runs

```toml
[mesh]
type = "file"
filename = "mesh.h5"   # from mallard mesh convert
```

*Used in:* any example, after `mallard mesh convert input.toml mesh.h5`.

Every rank of a distributed run reads a Gmsh or other mesh file whole,
keeping only its share. Large meshes should be converted once to Mallard's HDF5 mesh format, of
which each rank reads only its own part (builds with
`-DMallard_ENABLE_HDF5=ON`):

```sh
mallard mesh convert mesh.msh mesh.h5                # from any mesh file
mpirun -n 8 mallard mesh convert input.toml mesh.h5  # the mesh of an input, written in parallel
```

and then `filename = "mesh.h5"`. Generated meshes are always produced per
rank, so no rank ever holds a whole generated or HDF5 mesh. The file layout is
in the [input reference](../input.md#mesh). Later releases keep reading these
files ([compatibility](output.md#formats-and-compatibility)).

## Curved boundaries

```toml
[[mesh.curved]]
zone = "cylinder"
shape = "circle"        # 2D; "sphere" or "cylinder" in 3D
center = [0.0, 0.0]
radius = 0.5
# axis = [0.0, 0.0, 1.0]  for shape = "cylinder"
```

*Used in:* the curved-boundary validation (`tools/dev/curved_convergence.py`).

Boundary faces can follow the true geometry instead of straight edges or flat
facets. Curved faces get quadrature points on the curve with their own normals,
and the cells next to them their exact volumes and centroids, so TENO-E keeps
orders 3 to 6 at curved walls
([convergence](https://matthewbonanni.github.io/mallard/validation/#curved-boundaries)).

![Straight faces of a coarse mesh around a circle, and the curved faces on the circle with their quadrature points and normals](../images/guide/curved_faces-light.svg#gh-light-mode-only)
![Straight faces of a coarse mesh around a circle, and the curved faces on the circle with their quadrature points and normals](../images/guide/curved_faces-dark.svg#gh-dark-mode-only)

The geometry comes from either

- **high-order boundary elements** of a Gmsh file (`line3`, `line4` in 2D;
  `triangle6`, `triangle9`, `triangle10`, `quadrangle8`, `quadrangle9`,
  `quadrangle16` in 3D), e.g. `make_cylinder_mesh.py --order 3`; or
- **an analytic shape** that a zone is projected onto, on any mesh.

| Key | Default | Meaning |
|---|---|---|
| `[mesh] curved_geometry` | `true` | `false` ignores high-order nodes and `[[mesh.curved]]` |
| `zone` | required | Boundary zone to curve |
| `shape` | required | `circle` (2D), `sphere` or `cylinder` (3D) |
| `center`, `radius` | required | The shape (for a cylinder, a point of its axis) |
| `axis` | required for `cylinder` | Direction of the cylinder's axis |

These shapes are curved boundaries' own; [#407](https://github.com/MatthewBonanni/mallard/issues/407)
plans to take them from the [region engine](geometry.md#regions) instead.

Pitfalls:

- Zone nodes farther than 5% of an edge length from the shape are an error:
  the mesh must already approximate it.
- Not with `[physics] axisymmetric = true` yet.
- At curved walls TENO-E uses one-sided stencils of limited degree
  (`curved_wall_degree`, default 3), so orders 5 and 6 are fourth order there;
  see [Numerics](numerics.md#teno-e-options).

## Mesh quality

`mallard check input.toml` ([Checking an input](input_files.md#checking-an-input))
measures every cell of the mesh, taking for each the worst of its faces:

| Metric | Definition | Warning above |
|---|---|---|
| Non-orthogonality | Angle in degrees between the face normal and the line joining the centroids of the two cells (cell to face centroid at a boundary) | 70 |
| Skewness | Distance from the face centroid to where that line crosses the face, over the distance between the centroids | 0.5 |
| Volume ratio | Larger over smaller volume of the two cells | 10 |
| Aspect ratio | Largest over smallest distance from the cell centroid to its face centroids | 1000 |

It reports the largest and mean value of each and where the worst cells are;
`--quality PREFIX` writes the four fields to `PREFIX_000000.vtkhdf`
(`PREFIX_000000.vtu` in builds without HDF5) for ParaView, and
`tools/mesh/plot_mesh_quality.py PREFIX_000000.vtkhdf out.png` draws a
2D mesh's with the worst cells circled.

![Non-orthogonality and skewness of a coarse triangle mesh around the mallard (examples/mallard), the six worst cells of each circled](../images/guide/mesh_quality-light.svg#gh-light-mode-only)
![Non-orthogonality and skewness of a coarse triangle mesh around the mallard (examples/mallard), the six worst cells of each circled](../images/guide/mesh_quality-dark.svg#gh-dark-mode-only)

## See also

- [Input reference: `[mesh]`](../input.md#mesh), [`[[periodic]]`](../input.md#periodic)
- [Boundary conditions](boundary_conditions.md): what to put on each zone
- [Running in parallel](parallel.md): partitions and halos
- [Geometry](geometry.md): shapes shared by sources, probes and other features
- Design notes: [periodic boundaries](../design/periodic.md),
  [curved boundaries](../design/curved_boundaries.md)
