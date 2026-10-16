# Design: solution remap between meshes

Status: implemented (#323), 2D and 3D, serial and distributed.
`[initialize] type = "remap"` starts a run from a restart file written on
another mesh: a finer one for a grid-convergence study or to spin up an LES
from a coarse run, a different cell type or topology, any partition and any
rank count on either side.

## Goals

- **Conservative**: the integral of every conserved variable over the target
  mesh equals the source's to rounding, when the meshes cover the same domain.
- **High order**: the target cell averages of a field the source represents
  exactly as a polynomial of degree `p` (globally) are exact to rounding.
- **Robust** at shocks and contacts: no new extrema, no negative densities.
- **Any rank count, any partition**, and results that do not depend on them,
  bit for bit.
- Every variable of the restart file: the flow, the species of a mixture,
  temperature seeds and `[statistics]` averages.

## Method

The remap integrates a polynomial reconstruction of the source exactly over a
**supermesh**, the intersections of the source and target cells:

```text
ū_T = (sum over source cells S of ∫_{T ∩ S} u_S dV) / (sum over S of |T ∩ S|)
```

where `u_S` is source cell `S`'s polynomial. Since the pieces `T ∩ S` of a
source cell tile it, the target integrals sum to the source's: conservation
holds by construction, not by a correction. Since `u_S` is integrated exactly,
a field the polynomials reproduce is remapped exactly.

![A target triangle cut into pieces by the source quadrilaterals](../images/guide/remap-supermesh-light.svg#gh-light-mode-only)
![A target triangle cut into pieces by the source quadrilaterals](../images/guide/remap-supermesh-dark.svg#gh-dark-mode-only)

### Source polynomials

Each source cell gets a least-squares polynomial of degree `p` (`degree`,
default 2, at most 5) in the monomials of `((x - x_c) / h_c)` that TENO uses
(`teno::exponents`), with zero-mean basis functions so the cell's average is
kept exactly:

```text
u_c(x) = ū_c + sum_l a_l (φ_l(x) - mean_c φ_l)
```

The coefficients fit the averages of the cell's vertex neighbors in rings,
inverse-distance weighted, through each neighbor's own moments translated to
`c`'s frame (so every average is matched exactly, not at a centroid). A cell
uses the fewest rings with 1.5 times as many cells as unknowns, more rings while
the fit is rank deficient (Householder QR with a relative pivot tolerance of
`1e-8`), then lower degrees; cells near boundaries keep their order with
one-sided stencils. Across periodic seams the neighbors are translated as in
the TENO stencils (docs/design/periodic.md).

The fit is unlimited: it is TENO's central polynomial, without the weighting
of the candidate stencils, which buys nothing for a one-time projection.
Instead, `limiter = "bounds"` (the default) scales each polynomial's
non-constant part so that its values at the cell's vertices and quadrature
points stay within the averages of the cell and its vertex neighbors
(Barth-Jespersen applied to the full polynomial). Scaling keeps the average,
so conservation is unaffected. Density and the species' partial densities
share one factor, so the sum of `rho Y_k` remains `rho` exactly (the remap is
linear). `limiter = "none"` keeps the full order everywhere, for smooth fields.

TENO's own polynomials were not reused: they live on the device on the
target run's halo and boundary data, are built from primitive variables at
face quadrature points, and their stencil search is tuned for face values.
The remap fits conservative variables on the source mesh, on the host, once.

### Exact intersection

Cells are convex polytopes or the tetrahedra that define them:

- a convex cell with planar faces (every cell of generated meshes, most cells
  of mesh files) is the intersection of the half-spaces of its faces, and is
  tiled by triangles (2D) or tetrahedra fanned from its vertex average;
- any other cell is the mesh's own decomposition (`cell_tetrahedra` in 3D,
  the vertex-average fan in 2D), each simplex a convex piece.

`T ∩ S` is the target's simplices clipped by every half-space of each convex
piece of `S`. Clipping a tetrahedron by a plane leaves one tetrahedron or a
prism of three, a triangle by a line one or two triangles, so the result is a
list of simplices with no topology to maintain and no tolerances: a vertex on
the plane gives a degenerate simplex of zero volume, which integrates to
nothing. The polynomials are integrated on these simplices with conical
products of Gauss-Legendre rules exact to degree `p` (`p + 1` in axisymmetric
runs, which weight every integral by `r = y` as `cell_measure` does).

### Distributed search

Neither mesh is gathered. The source mesh is read and partitioned like any
mesh (`DistributedMesh`, multi-jagged), with enough halo layers for the
stencils, and each rank reads its source cells from the restart file by global
id. Each rank builds a bounding-box tree (`BoxTree`, `src/mesh/box_tree.h`)
over its owned source cells and publishes a coarse cover of up to 16 boxes,
the tree's nodes at depth 4. A target cell goes, as its node coordinates, to
every rank whose cover it overlaps (with positive volume); that rank
intersects it with the candidates its tree returns and sends back, per source
cell met, the measure and the integrals of every variable. The target rank
sums them in increasing source global id, so the result does not depend on
either partition: `MPITest.RemapDoesNotDependOnTheRankCount` (2D, also a
reacting mixture) and `MPI3DTest.RemapDoesNotDependOnTheRankCount` compare
the remapped runs on 1 to 4 ranks bitwise with the same run on one.

Halo cells of the target are remapped like owned ones, with the same bits.

### Domains that differ

Where the meshes cover different domains (a curved wall discretized
differently, a shorter source box), a target cell's average is over the part
the source covers; cells the source misses entirely take the average of the
source cell with the nearest centroid (found through the same trees, ties to
the lower id). The run log reports how many cells were partly or wholly
uncovered. Conservation then holds only over the common part. Curved
boundary geometry (`apply_curved`) is not used: cells are their straight
sided polytopes.

## Accuracy and cost

Tests (`test/remap_test.cpp`, both dimensions):

| Property | Test | Result |
|---|---|---|
| Polynomial exactness | random polynomials of degree `p` = 1-4 (2D), 1-3 (3D), quads to triangles, hexahedra to tetrahedra of unrelated sizes, no limiter | max error below 1e-14 (bound 1e-11); degree `p + 1` is not reproduced |
| Conservation | a discontinuous field, degree 3, with and without the limiter, quads to triangles, prisms to tetrahedra | integrals equal to 3e-15 of their magnitude (bound 1e-14) |
| Round trip | coarse to a nested 2x finer mesh and back, limited | the coarse averages to 1e-15 |
| Convergence | a smooth periodic field across seams, sources at `h` and `h/2` | 2D: orders 3.0, 3.2, 4.7 for `p` = 1, 2, 3; 3D: 3.3, 3.5 for `p` = 1, 2 |
| Bounds | the discontinuous field, limited | within the source's range |

Cost: a one-time host computation at startup, dominated by the
intersections. 10,000 quadrilaterals to 131,072 triangles at degree 3 (the
example below) take 0.6 s on 4 CPU threads, including the source mesh and its
polynomials; the integrals agree to 7e-14 relative there, the rounding of
summing 131,072 cells.

## Example

`examples/riemann_2d_quads/coarse.toml` runs the 2D Riemann problem on 100 x
100 quadrilaterals to t = 0.5; `remap.toml` continues it on 131,072 triangles
(degree 3, bounds limiter), keeping the shocks without new extrema. The
figure (from `tools/docs/runs.py`) remaps a smooth bump and a contact from 256
quadrilaterals to 2,738 triangles: inside each source cell the target cells
follow the polynomial's slopes rather than a constant.

![Density on quadrilaterals and remapped to triangles, and along a line](../images/guide/remap-fields-light.svg#gh-light-mode-only)
![Density on quadrilaterals and remapped to triangles, and along a line](../images/guide/remap-fields-dark.svg#gh-dark-mode-only)


`examples/taylor_green_3d/coarse.toml` runs the Taylor-Green vortex on 16^3
hexahedra to t = 6, and `remap.toml` continues it on 32^3 (degree 3, no
limiter; the remap takes 0.7 s on 6 threads). The coarse run is far too
dissipative by then; on the finer mesh the dissipation drops at once to the
fine run's level and evolves with it, carrying the energy the coarse run had
already lost (12%): the peak is 13% lower and later than that of the 32^3 run
from t = 0, and the dissipation stays within 4-8% of it from t = 9. A remap
keeps what the coarse run computed, errors included; it does not undo them.
Figure: `tools/docs/remap_tgv.py`.

![Dissipation rate of the Taylor-Green vortex: 16^3, 32^3 and 32^3 remapped from 16^3 at t = 6](../images/guide/remap-tgv-light.svg#gh-light-mode-only)
![Dissipation rate of the Taylor-Green vortex: 16^3, 32^3 and 32^3 remapped from 16^3 at t = 6](../images/guide/remap-tgv-dark.svg#gh-dark-mode-only)

## Shared pieces

`src/mesh/box_tree.h` (overlap, point-location candidates and nearest queries,
covers for distributed searches) and `src/numerics/overlap.h` (exact cell
intersections, clipping by half-spaces, simplex quadrature of any degree) are
general: probes (#318) and initialization from regions and data (#324) can
use them for point location, cut planes and exact cell averages.
