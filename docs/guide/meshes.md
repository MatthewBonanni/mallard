# Meshes

Mallard is a cell-centered finite volume solver for unstructured meshes:
triangles and quadrilaterals in 2D; tetrahedra, hexahedra, prisms and
pyramids in 3D, in any mix. The mesh comes from `[mesh]`: generated boxes for
simple domains, Gmsh files for anything else, and Mallard's HDF5 mesh format
for large distributed runs. Each boundary face belongs to a named **zone**,
which `[[boundaries]]` assigns a condition to.

The spatial dimension is a build option (`-DMallard_DIM=2`, the default, or
`3`): one executable per dimension, the same input keys for both.

## Generated meshes

```toml
[mesh]
type = "cartesian"    # quadrilaterals
Nx = 200              # cells along x
Ny = 100              # cells along y
Lx = 2.0              # the domain is [0, Lx] x [0, Ly]
Ly = 1.0
```

![The three generated 2D meshes: quadrilaterals, triangles split along the bottom-left to top-right diagonal, and the 8 degree compression ramp](../images/guide/mesh_types_2d-light.svg#gh-light-mode-only)
![The three generated 2D meshes: quadrilaterals, triangles split along the bottom-left to top-right diagonal, and the 8 degree compression ramp](../images/guide/mesh_types_2d-dark.svg#gh-dark-mode-only)

| `type` | Cells |
|---|---|
| `cartesian` | Quadrilaterals (2D) or hexahedra (3D) |
| `cartesian_tri` | Each quadrilateral split into two triangles along its bottom-left to top-right diagonal |
| `wedge` | Quadrilaterals over an 8 degree compression ramp starting at x = 0.5 (the `wedge` example) |
| `cartesian_tet`, `cartesian_prism`, `cartesian_pyramid`, `cartesian_mixed` | 3D only, see below |

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
| `periodic` | none | Directions joined periodically, e.g. `["x", "z"]` |

### Stretching

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

## Mesh files (Gmsh)

```toml
[mesh]
type = "file"
filename = "cylinder.msh"
```

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

## HDF5 meshes for large runs

Every rank of a distributed run reads a Gmsh file whole, keeping only its
share. Large meshes should be converted once to Mallard's HDF5 mesh format, of
which each rank reads only its own part (builds with
`-DMallard_ENABLE_HDF5=ON`):

```sh
mallard-mesh-convert mesh.msh mesh.h5                # from Gmsh
mpirun -n 8 mallard-mesh-convert input.toml mesh.h5  # the mesh of an input, written in parallel
```

and then `filename = "mesh.h5"`. Generated meshes are always produced per
rank, so no rank ever holds a whole generated or HDF5 mesh. The file layout is
in the [input reference](../input.md#mesh).

## Curved boundaries

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
- **an analytic shape** that a zone is projected onto, on any mesh:

```toml
[[mesh.curved]]
zone = "cylinder"
shape = "circle"        # 2D; "sphere" or "cylinder" in 3D
center = [0.0, 0.0]
radius = 0.5
# axis = [0.0, 0.0, 1.0]  for shape = "cylinder"
```

| Key | Default | Meaning |
|---|---|---|
| `[mesh] curved_geometry` | `true` | `false` ignores high-order nodes and `[[mesh.curved]]` |
| `zone` | required | Boundary zone to curve |
| `shape` | required | `circle` (2D), `sphere` or `cylinder` (3D) |
| `center`, `radius` | required | The shape (for a cylinder, a point of its axis) |
| `axis` | required for `cylinder` | Direction of the cylinder's axis |

Pitfalls:

- Zone nodes farther than 5% of an edge length from the shape are an error:
  the mesh must already approximate it.
- Not with `[physics] axisymmetric = true` yet.
- At curved walls TENO-E uses one-sided stencils of limited degree
  (`curved_wall_degree`, default 3), so orders 5 and 6 are fourth order there;
  see [Numerics](numerics.md#teno-e-options).

## See also

- [Input reference: `[mesh]`](../input.md#mesh), [`[[periodic]]`](../input.md#periodic)
- [Boundary conditions](boundary_conditions.md): what to put on each zone
- [Running in parallel](parallel.md): partitions and halos
- Design notes: [periodic boundaries](../design/periodic.md),
  [curved boundaries](../design/curved_boundaries.md)
