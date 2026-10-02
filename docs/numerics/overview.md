# Numerical methods

Mallard solves the 2D compressible Euler or Navier-Stokes equations with a
cell-centered finite volume method on unstructured meshes of triangles and
quadrilaterals.

## Discretization

For each cell, `dU/dt = -(1/V) sum_faces F(U_L, U_R) . n A + viscous and source terms`.
Each face is integrated with Gauss-Legendre points (one for first order and
MUSCL, `ceil((order + 1) / 2)` for TENO). At each point the convective flux is
an approximate Riemann solver applied to the reconstructed left and right
states. Each face's convective and viscous fluxes are stored once and every
cell sums its faces in a fixed order, without atomics, so results are bitwise
independent of the thread count and scheduling.
Time integration is explicit (SSPRK3 by default). The time step comes
from a per-cell spectral radius,
`dt_i = V / (sum_f (|u_n| + a) A_f + 4 nu_eff sum_f A_f^2 / V)`.

## Reconstruction

- **First order**: cell averages.
- **MUSCL**:
  - Variables: W = [rho, u, v, p].
  - Gradients: weighted least squares over face neighbors, with boundary ghost states placed at the mirror image of the cell centroid.
  - Limiter: Barth-Jespersen or Venkatakrishnan.
  - Faces whose density or pressure would be non-positive fall back to first order.
- **TENO-E** (Liang, Shyy & Fu 2025; see `teno_e.md`):
  - Least squares in a cell-scaled frame: `xi = (x - x_c) / sqrt(V)`, with a zero-mean monomial basis.
  - Stencils:
    - One large central stencil, about 2x the number of coefficients. It grows until full rank and well conditioned, and never splits equidistant candidates.
    - One degree-2 sector stencil per face.
    - Mirror images of cells across straight boundary segments, carrying the boundary condition's ghost state.
  - Smooth cells, as judged by a density-jump variance indicator, use the high-order polynomial on the conservative variables.
  - Troubled cells select stencils per characteristic variable at each face, using an adaptive cutoff.

## Riemann solvers

| Solver | Description |
|---|---|
| Rusanov | |
| HLL | Einfeldt wave speeds |
| HLLC | Toro, with the same Einfeldt wave speeds |
| Roe | Harten entropy fix |
| RHLL | Nishikawa & Kitamura's rotated hybrid: HLL along the velocity-difference direction, Roe across it. Carbuncle-free. |

## Boundary conditions

Every boundary condition is imposed weakly through an exterior state passed to
the Riemann solver.

- **Symmetry and slip walls**: reflect the normal velocity.
- **No-slip walls** (viscous): reflect the full velocity relative to the wall velocity, and use one-sided wall gradients for the viscous flux.
- **Transmissive boundaries**: take the exterior state from an *image face*, the interior face reached by translating the boundary face inward by the depth of the boundary cell. This is exactly what an interior face sees for a solution that does not vary normal to the boundary.
  - A zero-gradient copy of the boundary cell is not used, because at inflow boundaries it feeds the cell back to itself.
  - On triangles, where boundary-cell centroids are offset from the face, the copy creates an O(1) mass imbalance at every moving shock.

## Viscous fluxes

- Face gradients of velocity and temperature average the vertex-neighbor least-squares cell gradients. They are then corrected along the face normal so that their component along the line between the cell centroids matches the direct difference.
- Stress follows the Stokes hypothesis; heat flux uses a constant Prandtl number.

## Known limitations

- The scheme is not exactly well balanced: hydrostatic states carry small spurious velocities, which vanish at second order under refinement (wall ghosts continue the hydrostatic pressure gradient).
- TENO's k-exact least squares with 2x oversampling is noticeably more dissipative for under-resolved smooth waves than compact structured stencils. `stencil_factor = 1.5` helps, at some cost in robustness at discontinuities.
