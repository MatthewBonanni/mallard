# Design: curved high-order boundaries

Status: implemented (#162), 2D and 3D, serial and distributed, CPU and GPU.

A straight-sided cell next to a curved wall misplaces the wall by
`O(h^2 / R)`. That error enters the face quadrature points, the face normals,
the cell volume and centroid, and every moment that TENO-E fits its
polynomials with. At slip walls the polygonal wall also turns the flow at
every node. In the vortex below, orders 3 to 6 with straight walls do not
converge at all (Bassi & Rebay, J. Comput. Phys. 138, 1997, show the same for
DG).

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
   3D) in the physical groups. Cells may be of order 1 to 3; only their
   corners are kept, so interior faces are straight unless they share a
   curved edge. High-order edges whose nodes lie on their chord stay straight
   (Gmsh writes planes at the mesh's order too). `mallard-mesh-convert` keeps
   the high-order boundary faces in the HDF5 file (`boundary/high_order`).
2. **Analytic projection** for meshes without high-order nodes (generated, P1
   Gmsh, HDF5): `[[mesh.curved]]` names a zone and a shape (`circle` in 2D;
   `sphere` or `cylinder` with an axis in 3D). A point of the straight face is
   projected radially onto the shape, minus the linear interpolant of the
   corners' own projection offsets, so faces still meet at the mesh nodes.

`[mesh] curved_geometry = false` ignores both, for straight-versus-curved
comparisons on the same file.

## Representation

Every curved face is a map from its reference element, built from **curved
edges** plus a correction for the face interior:

- 2D: the face is its edge, `x(s)`, `s` in `[-1, 1]`.
- Boundary faces carry their own map: the analytic projection, or edge
  blending plus the interior nodes of the Gmsh element (`triangle10`,
  `quadrangle9`, `quadrangle16`), which reproduces its Lagrange map.
- In 3D a curved boundary face curves its edges, and every face sharing such
  an edge (the side faces of a prism layer, tetrahedra touching the wall
  along an edge, symmetry planes meeting the sphere) is blended from its edges:
  Szabo-Babuska blending on triangles (`x = linear + sum_e lambda_i lambda_j
  q_e(t)`, `t = (1 + lambda_j - lambda_i) / 2`), a Coons patch on
  quadrilaterals. For polynomial edges both are polynomial and reproduce the
  Lagrange element of that order.

The curved boundary faces of the whole mesh form a small table (surface only)
that every rank holds, keyed by sorted global node ids: Gmsh files are read
whole by every rank anyway, and distributed runs allgather it from their
blocks. Each rank evaluates its local faces from it, with the same
operations, so the geometry does not depend on the partition.

## Derived geometry

- **Face quadrature.** The flat rule (Gauss on edges, Dunavant on triangles,
  Gauss tensor on quadrilaterals) is mapped through the face map: point
  `x(xi_q)`, area vector `w_q (x_u x x_v)(xi_q)`. The area vectors are shifted
  by a common vector so that they sum exactly to the face's area vector
  `int n dS = 1/2 closed-integral x x dx`, which is exact for its (curved)
  edges. Every cell then still closes and gas at rest stays at rest to round
  off (without the shift, `|u|` reaches 5e-3 in 20 steps). Per point the
  solver keeps a unit normal and a weight (`face_quad_normals`,
  `face_quad_weights`, also in 2D on curved meshes); the flux kernels and the
  boundary conditions use the point's normal.
- **Face normals and areas** (`face_normals`, `face_area`) remain the area
  vector `int n dS` and its length: unchanged in 2D (the chord), and exact for
  curved edges in 3D. `face_coords` stays the flat centroid.
- **Cells** with a curved face integrate over cones from their vertex average
  to each face (`x = o + tau (x_f(xi) - o)`): the tetrahedra of
  `cell_tetrahedra` for straight faces, and tensor Gauss rules over the curved
  ones. The signed cones are exact for polynomials even where a cell is not
  star-shaped. The volume, centroid, TENO moments (2D on the host; 3D: the
  rows of the device's moment table) and initial averages of curved cells
  come from this rule.

## TENO at curved walls

- **No mirror images across curved walls and symmetry faces.** A reflection
  across the face's plane matches a curved wall only to `O(h^2)`; in the
  vortex below, mirrored stencils do not converge (orders 3-6, errors 1e-2 to
  1e-1). In 3D every face of a curved wall would also be a mirror plane of its
  own, beyond the setup's 32 planes per stencil search. Stencils are one-sided
  there (`curved_mirrors = false`). Curved far fields and inflows keep their
  mirror images: one-sided far fields made the order-5 cylinder's drag noisy
  (high-frequency rms 2.5e-3 against 6e-5).
- **Degree cap near curved walls.** One-sided central fits of degree 4 and 5
  grow in time at curved walls: in the vortex, the density error of orders 5
  and 6 grows exponentially from `t ~ 2` to `6` at every resolution, with
  Lebesgue bounds of 4 or 2, wall-normal-stretched (deeper) stencils, or degree
  4 alike. Cells whose central stencil reaches a curved wall (one of its first
  `ns` candidates has a one-sided wall face) fit degree
  `curved_wall_degree = 3` from that degree's stencil size: stable, with the
  error drifting linearly in time as at orders 3 and 4. Orders 5 and 6 are
  then fourth order at the walls and of their order elsewhere. Design order
  at the walls needs a boundary-constrained reconstruction (the wall
  condition imposed at the wall's quadrature points), left for later.

## Validation

### Steady vortex between circular walls

`tools/curved_convergence.py`: the irrotational vortex `u_theta = M / r` in
the annulus `1 <= r <= 1.384` (`M` at the inner wall), an exact steady Euler
solution, with slip walls (`symmetry`) on both circles and no other boundary.
Quadrilateral O-grids of `n_r x 16 n_r` cells (triangles: each split in two),
HLLC, SSPRK3, CFL 0.4, from the exact cell averages to `t = 12`. Density
error against the exact averages at `t = 12`, A100.

Subsonic (`M = 0.5`), quadrilaterals:

| Order | Walls | L1, n_r = 8 / 16 / 32 / 64 | L1 rate (last) | Linf, n_r = 64 | Linf rate |
|---|---|---|---|---|---|
| 3 | straight | 1.8e-2 / 9.0e-3 / 4.4e-3 / 2.5e-5 | erratic | 5.9e-4 | - |
| 3 | curved | 3.9e-5 / 3.5e-6 / 3.4e-7 / 4.2e-8 | 3.03 | 3.6e-7 | 2.86 |
| 4 | straight | 1.0e-1 / 6.9e-2 / 7.9e-2 / 7.8e-2 | 0.0 | 6.2e-1 | - |
| 4 | curved | 1.3e-5 / 1.1e-6 / 1.0e-8 / 3.5e-10 | 4.89 | 8.2e-9 | 3.99 |
| 5 | straight | 6.6e-2 / 7.2e-2 / 6.7e-2 / 4.8e-2 | 0.5 | 5.2e-1 | - |
| 5 | curved | 1.3e-5 / 2.8e-7 / 9.5e-9 / 3.2e-10 | 4.90 | 7.4e-9 | 3.95 |
| 6 | straight | 1.1e-1 / 7.3e-2 / 4.3e-2 / 2.0e-2 | 1.1 | 3.8e-1 | - |
| 6 | curved | 1.3e-5 / 1.6e-6 / 1.2e-8 / 3.8e-10 | 4.92 | 7.3e-9 | 3.65 |

Triangles, curved walls (straight walls do not converge, L1 2e-2 to 6e-2 at
`n_r = 64`): L1 rates 3.45, 4.38, 4.91, 4.91 for orders 3-6 (Linf 2.95,
3.94, 3.94, 3.96), L1 at `n_r = 64` 9.4e-9, 1.8e-10, 1.0e-10, 9.9e-11.

Supersonic (`M = 2.25`, quadrilaterals): straight walls shed waves at every
node and do not converge (L1 0.05 to 0.26). Curved: L1 rates 3.64, 4.86, 5.00,
5.57 for orders 3-6 (Linf 2.80, 3.99, 4.07, 4.18), L1 at `n_r = 64` 7.7e-6,
1.5e-7, 1.0e-7, 1.0e-7.

P3 walls from the Gmsh file (`line4`) match the projected walls to three
digits at every level. With mirror images across the curved walls
(`curved_mirrors = true`), orders 3-6 do not converge (L1 1e-2 to 1e-1).

### Cylinder at Re = 100

`examples/cylinder` (Mach 0.2), t = 60-160, walls and far field curved:

| Mesh | Order | Walls | St | mean Cd | Cl' |
|---|---|---|---|---|---|
| 128 x 64 | 3 | straight | 0.1633 | 1.3671 | 0.335 |
| 128 x 64 | 3 | curved | 0.1633 | 1.3643 | 0.337 |
| 128 x 64 | 5 | straight | 0.1647 | 1.3698 | 0.335 |
| 128 x 64 | 5 | curved | 0.1647 | 1.3656 | 0.333 |
| 384 x 128 | 3 | straight | 0.1648 | 1.3637 | 0.334 |
| 384 x 128 | 3 | curved | 0.1648 | 1.3629 | 0.335 |
| 384 x 128 | 5 | straight | (noise) | 1.349 | 2.76 |
| 384 x 128 | 5 | curved | 0.1648 | 1.3614 | 0.331 |

At Re = 100 the viscous layer is resolved by many cells, so curving the wall
moves the drag by 0.1-0.3% (toward the incompressible 1.33-1.35). On the fine
mesh, order 5 with straight walls is unstable (drag fluctuations of +-2.9);
with curved walls it runs, with residual drag noise of rms 7e-4 from the far
field (the instability noted in `docs/numerics/teno_e.md`, also there with
mirrored far fields).

### Mach 3 sphere

`examples/sphere_mach3` (Euler, Mach 3, `make_sphere_mesh.py` resolution,
810k tetrahedra) in a quarter box instead of the quarter cylinder, whose
faceted lateral far field exceeds the 3D setup's 32 mirror planes, at t = 3:

| Scheme | Walls | Delta / R (Billig 0.205) | Cd (pressure) |
|---|---|---|---|
| MUSCL | straight | 0.2211 | 0.9716 |
| MUSCL | curved | 0.2212 | 0.9720 |
| TENO5, bound-preserving | curved | 0.2213 | 0.9790 |

With 0.02 D cells on the wall the facets sit 1e-4 D inside the sphere, so the
standoff does not move. TENO5 with straight walls cannot run here: the
sphere's facets are mirror planes of their own, beyond the setup's limit.

## Alternatives considered

- **Fully isoparametric cells** (curved interior faces from the Gmsh cell
  nodes). Interior curvature does not change the accuracy of a finite volume
  scheme, which only needs a consistent partition of the true domain, and
  would need per-point normals on every face.
- **Curved quadrature points only** (straight volumes and moments). The
  `O(h^2)` volume and centroid errors of the boundary cells still cap the
  order (the reconstruction test's straight-geometry case: second order).
- **Normal correction only** (Krivodonova & Berger, J. Comput. Phys. 211,
  2006: straight faces with the exact wall normal at the quadrature points).
  Cheap and enough for slip walls in DG, but the cell moments of a finite
  volume reconstruction stay wrong.
- **CAD projection** (OpenCASCADE). A heavy dependency; Gmsh already projects
  its high-order nodes onto the CAD model.

## Limitations

- Orders 5 and 6 are fourth order at curved walls (degree cap, above).
- Axisymmetric runs reject curved boundaries.
- An edge shared by two different analytic shapes takes the first shape's
  curve; the other face's own projection then differs along that edge (the
  closure shift keeps every cell closed).
- Main's 3D TENO setup accepts at most 32 mirror planes per stencil search:
  faceted curved far fields (e.g. the quarter-cylinder domain of
  `examples/sphere_mach3`, or a straight-walled sphere) exceed it. Curved
  walls take no mirror images, so they do not.
