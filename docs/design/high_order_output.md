# High-order output

How a run writes what its high-order reconstruction knows inside each cell,
not only the cell averages. Issue
[#1448](https://github.com/MatthewBonanni/mallard/issues/1448). User docs:
[The high-order view](../guide/output.md#high-order-view).

## What is written

1. **A companion view, the solution files unchanged.** Each whole-mesh
   `[[write_data]]` entry in `vtkhdf` or `vtu` (default on, `high_order =
   false` opts out) writes `PREFIX_high_order...` beside its own files, at the
   same steps, in the same format and series layout. The solution files stay
   cell averages byte for byte, so restarts, `mallard post`, statistics and
   every reader of cell values are untouched, and a user who wants neither
   size nor view deletes or skips the companion. Boundary, surface, region and
   point writers have no view: their samples are already point values or
   averages over something else.
2. **Lagrange cells at the reconstruction's own order.** A TENO run of order
   `r` has central polynomials of degree `p = r - 1`; each owned cell is one
   VTK Lagrange cell of order `p` whose nodes hold those polynomials. No order
   switching: order 3 writes quadratic cells, order 6 quintic ones. MUSCL and
   first order have no polynomial beyond the average and write no view.
3. **Point values where the reconstruction has them.** Nodes of smooth cells
   with admissible values take the central polynomial of the conservative
   variables, the derived fields (`U`, `P`, `T`, `H`, `MACH`, ...) from those
   point values; troubled cells, and cells where the polynomial leaves the
   admissible set, take their averages at every node, exactly as the probes'
   `point_value` sampling does. Fields without a polynomial (indicators,
   sources, statistics, `CFL`, ...) are written as cell data of the view, so
   one file shows them all. The view's snapshot block names node fields
   `point_value` with the reconstruction and its fallback.
4. **Not a new solution format.** Nothing reads the view back; restarts stay
   cell averages. `mallard post derive` writes it for runs written with it
   off (from their restart files' averages, rebuilding the polynomials).

## Cells and nodes

| Mesh cell | VTK cell | Nodes of order p | Exact for |
|---|---|---|---|
| triangle | `VTK_LAGRANGE_TRIANGLE` (69) | (p+1)(p+2)/2 | P_p (affine) |
| quadrilateral | `VTK_LAGRANGE_QUADRILATERAL` (70) | (p+1)^2 | P_p: x(xi) bilinear, P_p∘x ⊂ Q_p |
| tetrahedron | `VTK_LAGRANGE_TETRAHEDRON` (71) | (p+1)(p+2)(p+3)/6 | P_p (affine) |
| hexahedron | `VTK_LAGRANGE_HEXAHEDRON` (72) | (p+1)^3 | P_p: trilinear map, P_p∘x ⊂ Q_p |
| prism | `VTK_LAGRANGE_WEDGE` (73) | (p+1)^2(p+2)/2 | P_p: map in P_1(xi,eta)⊗P_1(zeta) |
| pyramid | `VTK_LAGRANGE_HEXAHEDRON`, top face collapsed to the apex | (p+1)^3 | P_p: the degenerate trilinear map |

- **Exact interpolation.** Each polynomial is evaluated at the cell's nodes
  and VTK interpolates between them in the cell's own space: affine simplices
  span P_p in physical coordinates; on quadrilaterals, hexahedra and prisms
  the bi-/trilinear geometry composed with P_p lies in the Lagrange space
  (Q_p, or P_p⊗Q_p), so the rendered field is the polynomial everywhere in
  the cell, not only at nodes. Curved cells (`curved` boundaries) take a
  quadratic geometry, the existing `quadratic_nodes`, and are exact at nodes.
- **Pyramids as collapsed hexahedra.** VTK defines `VTK_LAGRANGE_PYRAMID`
  (74) but 9.x has no cell class for it: readers drop it. A hexahedron with
  its top four corners at the apex has a degenerate trilinear map whose
  composition with P_p is in Q_p, so it is exact, and every VTK reader draws
  it. Its Jacobian vanishes at the apex, which rendering (forward map)
  does not mind, but VTK's point location (Newton on the inverse map) does:
  with VTK 9.7.1 a probe of a pyramid mesh found 59 of 300 interior points,
  within 2e-4 of the polynomial, against all of them within 4e-8 on prisms
  and tetrahedra. Splitting each pyramid into two Lagrange tetrahedra would be
  exact and locatable where the base is planar, at the cost of view cells
  that are not mesh cells; left for when a reader needs it.
- **Orderings.** `lagrange::lattice` numbers nodes as `vtkLagrangeTriangle`
  ... `vtkLagrangeWedge` do: vertices, edges in VTK's edge order, faces in
  its face order (tetrahedron faces (0,1,3), (2,3,1), (0,3,2), (0,2,1) with
  recursive triangle interiors; hexahedron faces x=0, x=1, y=0, y=1, z=0, z=1,
  first free axis fastest; wedge triangle faces and body row-major, quad
  faces edge direction fastest), then the body. All five shapes at orders 1
  to 6 were compared with VTK 9.7.1's parametric coordinates; the tests keep
  the orders where each rule first differs from its alternatives.

## Writers

- **VTKHDF.** Its own `UnstructuredGrid` with `Points` per node (nodes are
  not shared between cells: neighbors' polynomials differ on their common
  face, and the discontinuity is what the view shows), `Connectivity` the
  identity, `PointData` for node fields and `CellData` for the rest. Ranks
  gather nodes in global cell order, as cell data already are, so the file
  is the same on any number of ranks. Series: per-snapshot files with the
  geometry externally linked to `PREFIX_high_order_mesh.vtkhdf`, exactly as
  the solution's.
- **VTU / PVD.** Each rank writes its piece's nodes and `PPointData` in the
  `.pvtu`; the `.pvd` lists the view's steps.
- **XDMF and restart files** write no view: XDMF entries keep cell values, restart files the averages the solver restarts from.

## Cost

- **Size.** Per snapshot the view writes `N_nodes x (3 + components)` reals
  plus the cells, against `N_cells x components` for the solution: about
  `(p+1)^d` times the solution for tensor cells, `(p+1)(p+2)/2` for
  triangles. `mallard check` reports each writer's projected bytes per
  snapshot, both, and warns (`W-OUTPUT-SIZE`) above 10 GB naming
  `high_order = false`.
- **Memory.** The nodes' geometry and basis tables are built once at setup;
  values for the requested fields live in one pinned host array the writers
  read, and evaluation runs in chunks of 2^18 nodes, so the device scratch is
  bounded whatever the mesh.
- **Device path.** Node coordinates come from small basis tables times each
  cell's geometry nodes (kernel), the point values from TENO's
  `point_values` on the chunk (the same kernel as the probes), and the
  fields from those (kernel), writing straight into the pinned array: no
  host work, no allocation and no synchronization per snapshot besides the
  writers' existing fence. The step graph stages it with the other output
  arrays, inside the conditional node of a due output (whenever any writer
  has a view: one flag fewer in the graph than a flag per view).
- **Measured** (Apple M4 Pro, 2 threads, 256^2 periodic quadrilaterals,
  TENO5, VTU of `RHO`, `U`, `P` every 10 steps, 3 alternated runs each): the
  output phase takes 50 ms per snapshot with the view (evaluation, copy and
  queueing; the files are written in the background) against 2.6 ms without,
  out of 100 ms per step; the files are 18 times larger (613 MB against 34 MB
  for 5 snapshots); setup adds 35 ms.

## Determinism

The view of a snapshot is a pure function of the cell averages, the mesh and
the reconstruction: within a build it is bitwise the same on any number of
ranks (VTKHDF in global cell order; VTU pieces per rank, as the solution).
The polynomials need the halo current: the snapshot exchanges it once when a
view is due.
