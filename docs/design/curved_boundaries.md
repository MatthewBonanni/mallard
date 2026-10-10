# Design: curved high-order boundaries

Status: in progress (#162).

A straight-sided cell next to a curved wall misplaces the wall by
`O(h^2 / R)`. That error enters the face quadrature points, the face normals,
the cell volume and centroid, and every moment that TENO-E fits its
polynomials with, so orders 3 to 6 drop to about second order at cylinders,
spheres and airfoils (Bassi & Rebay, J. Comput. Phys. 138, 1997).

## Goals

- Faces on curved walls follow the true geometry to the order of the scheme:
  face quadrature points, per-point normals and Jacobians, and the volume,
  centroid and moments of every cell touching them.
- Straight-sided meshes stay bitwise identical to main: nothing below runs
  without curved data.
- Any rank count gives bitwise identical geometry, and restarts are exact.
- The TENO setup stays on the device; only the few curved cells are handled on
  the host.

## Geometry sources

1. **High-order Gmsh boundary elements** (`line3`, `line4` in 2D; `triangle6`,
   `triangle9`, `triangle10`, `quadrangle8`, `quadrangle9`, `quadrangle16` in
   3D) in the physical groups. Cells may be of any order; only their corners
   are kept, so interior faces are straight unless they share a curved edge.
   `mallard-mesh-convert` keeps the high-order boundary nodes in the HDF5 file.
2. **Analytic projection** for meshes without high-order nodes (generated, P1
   Gmsh, HDF5): `[[mesh.curved]]` names a zone and a shape (`circle` in 2D;
   `sphere` or `cylinder` with an axis in 3D). A point of the straight face is
   projected radially onto the shape, minus the linear interpolant of the
   corners' own projection offsets, so faces still meet at the mesh nodes.

`[mesh] curved = false` ignores both, for straight-versus-curved comparisons on
the same file.

## Representation

Every curved face is a map from its reference element, built from **curved
edges** plus a correction for the face interior:

- 2D: the face is its edge, `x(s)`, `s` in `[-1, 1]`.
- Boundary faces carry their own map: Lagrange interpolation of the Gmsh
  nodes, or the analytic projection.
- In 3D a curved boundary face curves its edges, and every face sharing such
  an edge (the side faces of a prism layer, tetrahedra touching the wall
  along an edge, symmetry planes meeting the sphere) is blended from its edges:
  Szabo-Babuska blending on triangles (`x = linear + sum_e lambda_i lambda_j
  q_e(t)`), a Coons patch on quadrilaterals. For polynomial edges both are
  polynomial and reproduce the Lagrange element of that order.

The curved edges and boundary faces of the whole mesh are a small global table
(surface only) that every rank holds, keyed by sorted global node ids. Each
rank evaluates its local faces from it, with the same operations, so the
geometry does not depend on the partition.

## Derived geometry

- **Face quadrature.** The flat rule (Gauss on edges, Dunavant on triangles,
  Gauss tensor on quadrilaterals) is mapped through the face map: point
  `x(xi_q)`, area vector `w_q (x_u x x_v)(xi_q)`. The area vectors are shifted
  by a common vector so that they sum exactly to the face's area vector
  `int n dS = 1/2 closed-integral x x dx`, which is exact for its (curved)
  edges. Every cell then still closes, so free streams are preserved to round
  off. Per point the solver keeps a unit normal and a weight.
- **Face normals and areas** (`face_normals`, `face_area`) remain the area
  vector `int n dS` and its length: unchanged in 2D (the chord), and exact for
  curved edges in 3D. `face_coords` stays the flat centroid.
- **Cells** with a curved face integrate over cones from their vertex average
  to each face (`x = o + tau (x_f(xi) - o)`): triangles and tetrahedra for
  straight faces, as before, and tensor Gauss rules over the curved ones. The
  signed cones are exact for polynomials even where a cell is not star-shaped.
  The volume, centroid and TENO moments of curved cells come from this rule;
  the device-side 3D setup overwrites their rows of the moment table.
- **Initialization** averages use the same rule.

## Boundary conditions and mirror images

Wall, symmetry and far-field ghost states use the normal at each quadrature
point. TENO-E mirror images across a curved face: TBD by the convergence
study (mirroring across the chord plane is consistent only to `O(h^2)` for a
wall that is not a symmetry plane of the flow).

## Alternatives considered

- **Fully isoparametric cells** (curved interior faces from the Gmsh cell
  nodes). Interior curvature does not change the accuracy of a finite volume
  scheme, which only needs a consistent partition of the true domain, and
  would need per-point normals on every face.
- **Curved quadrature points only** (straight volumes and moments). The
  `O(h^2)` volume and centroid errors of the boundary cells still cap the
  order.
- **Normal correction only** (Krivodonova & Berger, J. Comput. Phys. 211,
  2006: straight faces with the exact wall normal at the quadrature points).
  Cheap and enough for slip walls in DG, but the cell moments of a finite
  volume reconstruction stay wrong.
- **CAD projection** (OpenCASCADE). A heavy dependency; Gmsh already projects
  its high-order nodes onto the CAD model.

## Validation

- Unit tests: geometry (areas and volumes of curved annulus and sphere cells
  against exact values, closure, rank independence), reconstruction order on
  curved cells.
- Inviscid steady vortex between two concentric circular walls (exact
  solution): convergence of orders 3 to 6, straight versus curved walls.
- Cylinder at Re = 100 and the Mach 3 sphere: drag and standoff, straight
  versus curved.
