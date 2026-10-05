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
Time integration is explicit (SSPRK3 by default; [Shu & Osher 1988](../references.md#shu-osher-1988)). The time step comes
from a per-cell spectral radius ([Blazek 2015](../references.md#blazek-2015)),
`dt_i = V / (sum_f (|u_n| + a) A_f + 4 nu_eff sum_f A_f^2 / V)`.

## Reconstruction

- **First order**: cell averages.
- **MUSCL** ([van Leer 1979](../references.md#van-leer-1979)):
  - Variables: W = [rho, u, v, p].
  - Gradients: weighted least squares ([Mavriplis 2003](../references.md#mavriplis-2003)) over face neighbors, with boundary ghost states placed at the mirror image of the cell centroid; on tetrahedra, over vertex neighbors. A tetrahedron's four face neighbors make weakly limited MUSCL unstable: a 1% acoustic pulse grows without bound in a box of generated tetrahedra with symmetry walls, and in a Delaunay tetrahedral mesh. Vertex neighbors are not used everywhere because their one-sided stencils at boundaries break the exact one-dimensionality of flows along aligned quadrilaterals.
  - Limiter: Barth-Jespersen ([Barth & Jespersen 1989](../references.md#barth-jespersen-1989)) or Venkatakrishnan ([Venkatakrishnan 1995](../references.md#venkatakrishnan-1995)).
  - Faces whose density or pressure would be non-positive fall back to first order.
- **TENO-E** ([Liang, Shyy & Fu 2025](../references.md#liang-shyy-fu-2025), extending TENO, [Fu, Hu & Adams 2016](../references.md#fu-hu-adams-2016); see [TENO-E details](teno_e.md)):
  - k-exact least squares ([Barth & Frederickson 1990](../references.md#barth-frederickson-1990)) in a cell-scaled frame: `xi = (x - x_c) / sqrt(V)`, with a zero-mean monomial basis.
  - Stencils:
    - One large central stencil, about 2x the number of coefficients. It grows until full rank and well conditioned, and never splits equidistant candidates.
    - One degree-2 sector stencil per face.
    - Mirror images of cells across straight boundary segments, carrying the boundary condition's ghost state.
  - Smooth cells, as judged by a density-jump variance indicator, use the high-order polynomial on the conservative variables.
  - Troubled cells select stencils per characteristic variable at each face, using an adaptive cutoff.

## Riemann solvers

| Solver | Description |
|---|---|
| Rusanov | [Rusanov 1962](../references.md#rusanov-1962) |
| HLL | [Harten, Lax & van Leer 1983](../references.md#harten-lax-van-leer-1983), with Einfeldt wave speeds ([Einfeldt 1988](../references.md#einfeldt-1988); [Einfeldt et al. 1991](../references.md#einfeldt-1991)) |
| HLLC | [Toro, Spruce & Speares 1994](../references.md#toro-spruce-speares-1994), with the same Einfeldt wave speeds |
| Roe | [Roe 1981](../references.md#roe-1981), with Harten's entropy fix ([Harten 1983](../references.md#harten-1983)) |
| RHLL | [Nishikawa & Kitamura 2008](../references.md#nishikawa-kitamura-2008), a rotated hybrid: HLL along the velocity-difference direction, Roe across it. Carbuncle-free. |

All five also solve gas mixtures, on each side's frozen `cp / cv` and energy
offset ([chemistry design](../design/chemistry.md#riemann-solvers)). The
mixture Roe solver ([Glaister 1988](../references.md#glaister-1988);
[Shuen, Liou & van Leer 1990](../references.md#shuen-liou-van-leer-1990)) lets the
contact, shear and composition waves, which all move with the flow, carry
the whole jump of the conservative variables left by the two acoustic waves.
That makes it exact at contacts between different gases for any averaged
sound speed, and it reduces to the single-gas Roe solver for one gas.

### Low-Mach correction

Upwind fluxes damp the jump of the reconstructed velocity across a face at the
sound speed. As the Mach number falls, that dissipation does not vanish
relative to the flow scales ([Guillard & Viozat 1999](../references.md#guillard-viozat-1999);
[Rieper 2011](../references.md#rieper-2011)). At M = 0.1 it dominates under-resolved
vortical flows. On the Taylor-Green vortex at 64^3 (full-box equivalent),
TENO5 reaches its dissipation peak two time units early, and the resolved
enstrophy is a third lower.

Mallard follows [Thornber et al. (2008)](../references.md#thornber-2008). Before the
Riemann solver, the velocity jump across each interior face is scaled by
`z = min(1, max(M_L, M_R, M_cut))`, and its mean is kept. Supersonic faces
(`z = 1`) are untouched, and density, pressure and the Riemann solver itself
are unchanged.

The cutoff `M_cut` (`[numerics] low_mach_cutoff`, default 0.1) keeps some
acoustic damping in gas nearly at rest, like the cutoff Mach number of
preconditioned all-speed schemes ([Weiss & Smith 1995](../references.md#weiss-smith-1995)). Without
it, spurious acoustic velocities of a hydrostatic atmosphere are not damped at
all and do not converge under refinement. With it, they converge at second
order, about four times larger than without the correction.

`z` uses lab-frame velocities, so the scheme is not Galilean invariant. A
shock moving into gas at rest sees `z < 1` on its upstream faces; Sod and
Shu-Osher are unaffected, within 10% in L1 and without overshoots.

## Boundary conditions

Every boundary condition is imposed weakly through an exterior state passed to
the Riemann solver.

- **Symmetry and slip walls**: reflect the normal velocity.
- **No-slip walls** (viscous): reflect the full velocity relative to the wall velocity, and use one-sided wall gradients for the viscous flux.
- **Periodic boundaries**: no ghost states; the paired zones are joined into interior faces whose second cell is translated next to the first (see `docs/design/periodic.md`), so every scheme treats the seam as interior.
- **Characteristic far field**: the outgoing Riemann invariant comes from the interior and the incoming one from the free stream ([Blazek 2015](../references.md#blazek-2015)).
- **Transmissive boundaries**: take the exterior state from an *image face*, the interior face reached by translating the boundary face inward by the depth of the boundary cell. This is exactly what an interior face sees for a solution that does not vary normal to the boundary.
  - A zero-gradient copy of the boundary cell is not used, because at inflow boundaries it feeds the cell back to itself.
  - On triangles, where boundary-cell centroids are offset from the face, the copy creates an O(1) mass imbalance at every moving shock.

## Viscous fluxes

- Cell gradients come from a weighted least-squares quadratic fit ([Barth & Frederickson 1990](../references.md#barth-frederickson-1990)) over the vertex neighbors and the cell's boundary ghosts. A linear fit is only first-order accurate on one-sided boundary stencils and on triangles, where its errors cancel only in the interior of regular meshes.
- Face gradients of velocity and temperature average the two cell gradients. They are then corrected along the face normal so that their component along the line between the cell centroids matches the direct difference ([Diskin et al. 2010](../references.md#diskin-2010)).
- Transmissive faces take the face values and gradients of their image face, as the convective flux does.
- Stress follows the Stokes hypothesis; heat flux uses a constant Prandtl number. Viscosity is constant or follows Sutherland's law ([Sutherland 1893](../references.md#sutherland-1893)).

## Axisymmetric flows

With `[physics] axisymmetric = true`, cells are integrated over their solids
of revolution about the x axis (per radian): r-weighted cell averages,
revolved volumes and face areas, and the radial momentum source
`int (p - tau_thetatheta) dA`, made high order from TENO's polynomials. TENO's
least squares fit r-weighted averages, and second-order operators work at
r-weighted centroids. See [the design notes](../design/axisymmetric.md).

## Known limitations

- The scheme is not exactly well balanced: hydrostatic states carry small spurious velocities, which vanish at second order under refinement (wall ghosts continue the hydrostatic pressure gradient).
- RHLL does not keep a 1D problem on a 2D mesh exactly 1D. Where the velocity jump across a face is small but above its fallback threshold, the rotation direction tilts by the round-off transverse velocity divided by that jump. On the Sod strip (800 x 4 quads, TENO5), the rows start out identical to 1e-16, still agree to 1e-12 at t = 0.04, and differ by up to about 1e-5 later on, at levels that depend on the time-step history. HLL and HLLC stay at round-off.
- TENO's k-exact least squares with 2x oversampling is noticeably more dissipative for under-resolved smooth waves than compact structured stencils. `stencil_factor = 1.5` helps, at some cost in robustness at discontinuities.

All sources, with where Mallard uses them: [References](../references.md).
