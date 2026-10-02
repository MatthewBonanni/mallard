# Design: periodic boundaries

Status: generated meshes, serial and distributed (Gmsh periodic zones to follow). Motivated by the canonical Taylor-Green vortex (the full
box `[-pi L, pi L]^3`) and decaying isotropic turbulence, which need fully
periodic boxes, and by 2D cases (isentropic vortex, Kelvin-Helmholtz).

## Goals

- Periodic pairs of boundary zones become ordinary interior faces: every
  scheme (FO, MUSCL, TENO, viscous fluxes) sees the same stencils across a
  periodic seam as in the interior, at its design order.
- Non-periodic meshes are bitwise unchanged.
- No ghost cells and no copies of the solution: periodicity lives entirely in
  the connectivity plus a translation per cell pair.
- Works in 2D and 3D, on any cell type, for one, two or three periodic
  directions (edges and corners of a fully periodic box included).

## Node identification

A periodic pair `(A, B, T)` says that zone `B` is zone `A` translated by `T`.
For each pair, the nodes of `A` translated by `T` are matched to the nodes of
`B` geometrically, through a hash grid, with a tolerance of `1e-6 h` (`h` the
shortest edge of the zones' faces). Every node needs exactly one match, or the
mesh is rejected.

A union-find over the matches, carrying lattice offsets, assigns each node

- a **key**, the lowest node id of its periodic class, and
- an **int8 lattice offset** `L` with `x_n = x_key + sum_j L_j T_j`.

Edges and corners of a box periodic in several directions fall out of the
union-find: a corner node of a 3D box joins the class of the 7 other corners
through three pairs. A matching that would put one node at two different
offsets from its key (e.g. a pair whose zones touch) is rejected.

## Faces and cells

`init_from_connectivity` keys faces by their sorted node *keys* instead of node
ids (2D and 3D alike), so a face of `A` and its image in `B` become one
interior face. The face keeps the nodes of its first cell (cell 0), so its
centroid, normal and quadrature points are in cell 0's frame. Its **shift**,
the translation that brings cell 1 next to cell 0, is `(L(a) - L(b)) . T` for a
node `a` of cell 0 and the node `b` of cell 1 with the same key. Zones `A` and
`B` are dropped.

Every periodic direction needs at least 3 cells: with fewer, a cell would touch
itself, or another cell through two different translations, which neither the
face nor the vertex-neighbor storage can express. Such meshes are rejected with
a clear error.

## Storage

- `shifts`: a small table of translation vectors; row 0 is zero.
- `face_shift`: a `uint8` row index per face (cell 1 seen from cell 0 is at
  `x_c1 + shift`; the face seen from cell 1 is at `x_f - shift`).
- `cells_of_cell_shift`: a `uint8` row index per vertex-neighbor entry. Vertex
  neighbors are found through shared node keys and become (cell, shift) pairs.

On a non-periodic mesh every index is 0, and kernels add the zero row as
`(x_c1 + 0) - x_c0` or `(x_f - 0) - x_c1`, which round exactly like the
unshifted expressions: results are bitwise unchanged.

## Consumers

| Where | Change |
|---|---|
| Face and vertex LSQ gradients | neighbor offsets add the shift |
| Limiter, MUSCL extrapolation | face centroid seen from cell 1 subtracts the shift |
| Viscous flux | centroid difference adds the shift |
| TENO reconstruction | face-point basis subtracts the shift on cell-1 sides |
| TENO setup | stencil search walks (cell, lattice offset) pairs, so stencils wrap across seams at any depth and may hold one cell at several offsets; monomial means, mirror lines/planes, point-in-cell tests and sector cones use translated nodes and centroids in each cell's own frame |
| Transmissive image search | also searches vertex neighbors across seams |

The device needs no lattice information for TENO: a stencil entry's value is
its cell's state; the translation only enters the precomputed pseudo-inverses.

Known approximation: under gravity, a TENO mirror image across a boundary face
that is not parallel to a periodic translation, taken from a cell reached
through a different lattice offset than the face, uses the untranslated
cell-to-face distance for the hydrostatic ghost pressure.

## Input

- Generated meshes: `[mesh] periodic = ["x", "y", "z"]` pairs `left`/`right`,
  `bottom`/`top` and `back`/`front`.
- Gmsh meshes (later): `[[periodic]]` entries naming zone `A`, zone `B` and the
  translation.
- `[[boundaries]]` is optional, since a fully periodic box has no boundary
  faces; naming a periodic zone there is an error.

## Distributed runs

- `DistributedMesh` finds the node classes first: every rank gathers the faces
  and node coordinates of the periodic zones and runs the serial matching on
  that surface, so all ranks agree on every key and offset.
- Faces are hashed by their node keys, so seam faces pair into interior faces
  (and dual-graph edges, which the graph partitioner sees); boundary faces of
  periodic zones are not posted.
- The node directory that grows halo layers is keyed by node keys, so halos
  extend across seams.
- Each local mesh gets its nodes' offsets and keys (the first local node of
  each class), and computes its own shifts: stencils, and so results, match the
  serial ones.

Known limit: gathering the periodic zones puts O(N^(2/3)) data (the paired
surfaces) on every rank. That is fine up to about 1e8-1e9 cells but breaks the
"no rank holds global data" rule of `mpi.md`. The scalable alternative follows
the face matching: send each zone node, snapped to the matching tolerance and
translated into zone A's frame, to a rank chosen by hashing its snapped
coordinates; pair the matches there, and resolve the classes (at most 8 nodes,
at a box corner) with a few rounds of key propagation to the nodes' owners.

## Testing

- Non-periodic meshes: bitwise-identical solutions with and without the change.
- Free-stream preservation and discrete conservation on periodic boxes (2D
  quadrilaterals and triangles, 3D hexahedra, tetrahedra and mixed meshes).
- One-period advection of a smooth profile at design order for MUSCL and
  TENO3-6, with a time error kept below the spatial one.
- The 2D isentropic vortex across the seam.
- Distributed periodic runs matching serial ones on 1-4 ranks, including
  partitions that cut the seam.

## Pull requests

1. Generated-mesh periodicity and the serial schemes (this design, TENO
   included).
2. The distributed path.
3. Gmsh periodic zone pairs.
