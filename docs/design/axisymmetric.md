# Axisymmetric (r-z) flows

Status: implemented (issue #157).

`[physics] axisymmetric = true` turns the 2D solver into one for flows
symmetric about the x axis without swirl: x is the axial coordinate z and y
the radius $r \ge 0$. One 2D mesh of the meridian half-plane then stands for the
3D body of revolution, so round jets, counterflow and Bunsen flames, blunt
bodies and spherical blasts run at 2D cost.

## Goals

- Conservative and well balanced: a quiescent gas at uniform pressure stays at
  rest to round-off on any mesh, with any reconstruction.
- Mass, axial momentum, energy and species are conserved discretely (per
  radian of the revolved domain).
- Every scheme keeps its planar machinery: FO, MUSCL, TENO-E, all Riemann
  solvers, viscous and mixture transport, chemistry, MPI, restarts.
- Planar runs are bitwise unchanged.
- 2D builds only; 3D builds reject the option.

## Formulation: area-weighted (revolved finite volumes)

Mallard integrates the 3D conservation laws over the cell's solid of
revolution, per radian. With `dA` the planar area element and `dl` the
planar length element,

    d/dt int_cell U r dA + sum_faces int_face F(U) . n r dl = int_cell S r dA + int_cell H dA,

where `H` is the geometric source of the radial momentum alone,

    H = [0, 0, p - tau_thetatheta, 0, (0 for every species)].

So the unknown of a cell is its r-weighted average `U_c = int U r dA / V_c`,
with the revolved volume `V_c = int r dA`, and a face flux is weighted by the
revolved area `int_face r dl`. This is the area-weighted (or "volume")
formulation, as opposed to the source-term formulation, which keeps planar
volumes and areas and adds `-(1/r) [rho v, rho u v, rho v^2, (E + p) v, ...]`
as a source.

Why area-weighted:

- Conservation: axial momentum, mass, energy and species have no source at
  all, so they are conserved exactly in the revolved geometry. The
  source-term form conserves nothing in 3D, and its sources are singular as
  `1 / r` at the axis.
- Axis: faces on the axis have zero revolved area. Nothing crosses the axis,
  without any special flux treatment; the axis boundary condition only feeds
  reconstruction stencils and gradients (see below).
- Well balance: the one remaining source, `p` in the radial momentum, is
  integrated over the planar cell area, `int p dA`. For uniform `p` the face
  pressure fluxes sum to `p sum_f n_r int_f r dl = p int_cell div(r e_r) dA =
  p A_c`, exactly the source. The identity is exact for straight-sided cells
  as long as `int_f r dl` is integrated exactly (any Gauss rule with one
  point or more: `r` is linear along a face), so a quiescent state is
  preserved to round-off.
- Species and energy need nothing: they are scalars, with no geometric source
  in this form.

## Geometry

`Mesh::make_axisymmetric()` (called by the solver after the mesh is built,
on every rank) keeps the planar geometry (areas, lengths, normals, which
reconstruction, limiters and stencil searches use) and adds

- `cell_measure(c) = int_c r dA` (the revolved volume per radian), exact for
  polygons, used wherever the solver uses a volume as a volume: the RHS
  divide, the time step, integrals, sources, cell averages of initial
  conditions;
- `face_measure(f) = r_f L_f`, with `r_f` the face midpoint radius (exact),
  used by the one-point viscous fluxes and the time step;
- revolved centroids as `cell_coords`: `x_c = int x r dA / int r dA`. A linear
  function equals its r-weighted cell average at the revolved centroid, so
  second-order schemes (MUSCL, least-squares gradients, viscous face
  gradients and wall distances) stay second order up to the axis. The planar
  centroid would be off by `O(h^2 / r)`, `O(h)` next to the axis.

In planar runs `cell_measure` and `face_measure` are the same views as
`cell_volume` and `face_area`.

Convective fluxes are integrated with their Gauss points weighted by `r_q`:
`int_f F . n r dl = L_f / 2 sum_q w_q r_q F_q`, a per-face weight table
`(face, q)` that the 2D flux kernels use when it is non-empty (3D already
has one). The species upwind fluxes use the same weights.

All integral quantities are per radian: forces (`[[forces]]`, multiply by
`2 pi` for the body force; only the axial component is meaningful),
`[integrals]`, conserved totals.

## Sources

Per owned cell, added to the radial momentum before dividing by `V_c`:

$$
S_c = \int_c \left(p - \tau_{\theta\theta}\right)\mathrm{d}A, \qquad
\tau_{\theta\theta} = \mu\left(\frac{2u_r}{r} - \frac{2}{3}\nabla\cdot\mathbf{u}\right), \qquad
\nabla\cdot\mathbf{u} = \frac{\partial u_x}{\partial x} + \frac{\partial u_r}{\partial r} + \frac{u_r}{r}.
$$

- FO and MUSCL, and TENO's troubled cells: `S_c = (p_c - tau_thetatheta,c) A_c`
  from the cell values at the revolved centroid and the cell gradients the
  viscous flux uses (second order).
- TENO, smooth cells: the central polynomial is evaluated at the points of a
  collapsed Gauss rule on a fan of triangles, exact to at least the degree of
  the reconstruction, and `p - tau_thetatheta` (velocity derivatives from the
  polynomial, `mu` the cell's) is integrated over the planar cell. If the
  polynomial gives a non-positive density or pressure at a point, the cell
  keeps the cell-value source.

Both are exactly well balanced: for uniform pressure every polynomial
coefficient vanishes, so `S_c = p sum_q w_q = p A_c` to round-off, and
`A_c` is what the revolved face areas close against.

The viscous face tractions use the full 3D divergence (`+ u_r / r` at the
face; at axis faces, whose flux vanishes anyway, `du_r/dr`). The `-2 mu u_r /
r^2` terms of the non-conservative form of the radial momentum equation are
exactly this hoop stress together with the revolved face areas; they are not
added separately. Gravity and expression sources are per unit volume and scale
with `V_c`.

The time step adds the hoop-stress decay rate `nu_eff / r_c^2` to the viscous
spectral radius (comparable to the diffusive limit only in the first cells
off the axis).

## Viscous gradients near the axis

Cell values are r-weighted averages, whose second moments differ from cell
to cell near the axis even on uniform meshes (`<r^2> - r_c^2` is `h^2 / 18`
in the first row of quads, `h^2 / 12` far away). Planar second-order
operators that treat the averages as values at the centers then leave an
O(1) truncation error in the first rows: in Hagen-Poiseuille flow the
x-momentum residual of the first row was 14% of the forcing. In axisymmetric
runs:

- The quadratic least-squares fit of the cell gradients fits the averages:
  its quadratic monomials carry the r-weighted second central moments of
  each neighbor (`Mesh::cell_covariance`; mirrored for mirror ghosts, zero for
  prescribed face states).
- Face values and gradients are interpolated to the face, at the fraction of
  the line between the two centers where the face center projects, instead
  of averaged (the revolved centers are not symmetric about the faces).
- The jump of the two averages in the face correction loses its part due to
  the cells' different spreads along that line, `q_ss (var_1 - var_0) / 2`,
  with `q_ss` from the two cell gradients.

With these, the Poiseuille residual next to the axis is below 1e-5 of the forcing.

## Axis boundary

The axis (faces on `y = 0`) must be a `symmetry` boundary; the solver rejects
anything else on the axis and any node below it. Ghost states mirror across
the axis (`u_r -> -u_r`), which is exactly the reflection symmetry of an
axisymmetric field in the meridian plane: scalars are even in r and `u_r` odd.
Convective and viscous fluxes through axis faces are zero through their zero
revolved area.

## Reconstruction

Reconstruction stays in the planar `(x, r)` coordinates. Cell averages are
r-weighted, so the k-exact least-squares conditions of TENO-E are r-weighted
too: the moments of the basis over each stencil entry are

    <phi_l>_j = int_j phi_l |r| dA / int_j |r| dA.

`|r|` makes the mirror image of a cell across the axis carry the mirror of its
r-weighted average, so stencils of axis cells are the full mirrored ones. With
these moments the reconstruction is exact for polynomials of degree k in
`(x, r)` (k-exactness in the r-weighted sense), so face values keep their
design order, up to the axis. The smoothness indicators stay planar (they
measure oscillation, not averages). MUSCL uses the revolved centroids. The
TENO cache key includes the flag.

Accuracy: with the high-order source, smooth axisymmetric Euler flows keep
TENO's design order up to the axis (the truncation error of a manufactured
solution converges at order 3.1, 4.5, 5.4 for TENO3-5 on quadrilaterals).
Viscous fluxes are second order, as in planar runs.

## Curved boundaries

A body of revolution with a curved meridian (a sphere, a nozzle contour) has
curved boundary faces ([curved boundaries](curved_boundaries.md), #725).
`make_axisymmetric` runs after `apply_curved`, so every revolved quantity of
the cells and faces that touch a curved face comes from the curved geometry:

- **Cells.** `cell_measure`, the revolved centroid and the r-weighted
  covariance of a curved cell are integrated over its curved rule
  (`curved::Geometry::cell_rule`, cubic, the cones from its vertex average)
  times `r`. Its planar area and centroid are `apply_curved`'s.
- **Faces.** `face_measure` of a curved face is `int r dS` along the curve
  (16-point Gauss), for the viscous fluxes, inflows and probe surfaces.
- **Flux points.** At the Gauss points of a curved face, the area vectors of
  the planar rule (`face_rule`, which sum to the chord's) are multiplied by
  `r_q` and shifted by a common vector, in proportion to the Gauss weights, so
  that they sum to the face's revolved area vector
  `R_f = int r n dS = ((y_1^2 - y_0^2) / 2, -int y dx)`, the first component
  exact and the second by 16-point Gauss. Their lengths and directions are the
  points' flux weights (`flux_weights`) and unit normals (`flux_normals`, the
  face reconstruction's elsewhere). Summed over a cell, the `R_f` give
  `(0, A_c)` with `A_c` the curved planar area, exactly for polynomial
  (Gmsh) curves and to round-off for projected ones; with the cell-value
  source `p A_c`, a uniform state then stays uniform to round-off, at rest or
  moving. The shift is the r-weighted Gauss rule's error, `O(h^(2 n_q))` of
  the face; a per-point normal is needed because one point (first order, or
  MUSCL) cannot match both components with a weight.
- **Sources.** The cell-value source multiplies by `A_c`. TENO's high-order
  source evaluates the polynomial at the curved cell's own rule (degree
  `k + 1`, `TENO::curved_rule`), whose weights sum to `A_c`, instead of the
  fan of triangles.
- **Reconstruction.** TENO's moments of curved cells are r-weighted as those
  of straight cells; MUSCL uses the revolved centroids.
- **Initial averages, regions and probes.** Analytical initial conditions are
  averaged over the curved rules times `r`. Region volume fractions and
  measures over curved cells already integrate `r` over the curved cones
  (`sampler::cone`, #724), point and cut-cell probes weight by `r` and
  `cell_measure`, and boundary probes use `face_measure`.

Each of these is needed: without the r-weighted flux points, gas at rest or
in uniform motion between curved walls changes by 0.1-0.5 in 20 steps on a
3 x 9 shell; without the curved measures, TENO's curved source rules, the
r-weighted initial averages or the r-weighted moments of curved cells, the
TENO4 residual of the shell state below converges at order 1.9-2.0 instead of
3.1.

## Validation

Unit tests (`test/axisymmetric_test.cpp`, `MPITest.AxisymmetricRunsMatchSerial`):

- Geometry: Pappus's theorem for every cell, and the closure `sum_f n_r
  int_f r dl = A_c` that well balance rests on.
- Quiescent gas, quads and triangles, FO/MUSCL/TENO3/TENO5, Euler (with a
  density field) and Navier-Stokes, and gas mixtures: velocities stay at
  round-off.
- Conservation of mass, axial momentum and energy in a periodic pipe.
- Hagen-Poiseuille: residual of the exact solution second order, below 1e-5
  of the forcing next to the axis.
- Manufactured smooth state: truncation error at design order for TENO3-5
  on quadrilaterals (3.1, 4.5, 5.4) and triangles, axis cells included; with
  the cell-value source instead, 2.5-2.9 on quadrilaterals and 1.1-1.6 on
  triangles.
- Distributed runs match serial ones bitwise on 1-4 ranks.
- Curved walls (the meridian half-plane of a spherical shell, `1 <= R <= 2`,
  quadrilaterals and triangles, walls projected or P2): exact revolved
  volumes, first moments and sphere areas; uniform flow and gas at rest stay
  uniform to round-off for FO, MUSCL, TENO3 and TENO5; boundary probes
  measure `2 R^2` per radian; the residual of a smooth state tangent to the
  spheres converges at order 2.7 (TENO3) and 3.1 (TENO4), from 2.7e-3 to
  4.3e-4 and 7.2e-4 to 8.1e-5 between 8 x 24 and 16 x 48 cells, below a
  tenth of the straight walls' error.

Examples:

- `poiseuille_pipe`: the steady profile converges at second order (mean
  error 1.3e-3, 3.2e-4, 8.0e-5 of U for 16, 32, 64 cells across the radius).
- `noh_axisymmetric`: shock radius within 1% of t / 3 in every direction,
  post-shock density 62-63 of 64 (up to 74 in the first cells along the axis).
- `sedov_axisymmetric`: shock radius 0.6% ahead of the exact one at t = 0.8,
  density peak 4.66-4.76 in every direction.
- `sphere_axisymmetric`: Re = 100 and 150, Cd within 1.2% of Clift, Grace &
  Weber at Mach 0.2, separation angle within 0.2 degrees, wake length 0.885 and
  1.21 D (Johnson & Patel: 0.88, about 1.2).
- Sphere with curved walls (`tools/dev/axisymmetric_sphere.py`, Euler,
  TENO3, 32 L x 16 L quadrilaterals, P2 sphere). At Mach 0.3 (HLLC, from the
  potential flow), the wall entropy error (mean over the revolved wall)
  converges at third order with curved walls, 3.8e-3, 6.4e-4, 9.2e-5 at
  L = 1, 2, 4 (rates 2.6 and 2.8), and at first order with straight ones,
  2.8e-3, 1.1e-3, 4.4e-4 (1.4 and 1.3). With HLL, which has no contact wave
  and so diffuses entropy at every face, both stall near second order
  (7.1e-3, 1.9e-3, 4.6e-4 curved). TENO4 on these O-grids builds up entropy
  errors of 5-50% with straight or curved walls, planar or axisymmetric,
  with HLL or HLLC. At Mach 3 (HLL, t = 6; HLLC grows a carbuncle on the
  axis) the standoff is 0.2305 R at L = 4 for both walls (Billig 0.205, an
  experimental fit); the wall entropy stays 0.5-1.5% above the normal
  shock's with curved walls, where straight walls dip 0.6% below it, which
  inviscid flow cannot do. The stagnation pressure is 2.7% below Rayleigh's
  at L = 4 for both walls, an error of the axisymmetric runs only (the
  planar cylinder on the same mesh: 0.24% below), made where the captured
  shock crosses the axis row: there the cells behind it hold 2.7% more
  entropy than the normal shock's (L = 2), relaxing to 1.5% at the wall
  (`p_0` scales as entropy to the power -2.5), where the planar cylinder's
  stay within 1% of it.

  ![Wall entropy error of inviscid flow over a sphere at Mach 0.3, TENO3: along the wall at 128 x 64 cells, and its mean against the mesh, converging at order 1.3 with straight walls and 2.8 with curved ones](../images/guide/axisymmetric_curved_sphere-light.svg#gh-light-mode-only)
  ![Wall entropy error of inviscid flow over a sphere at Mach 0.3, TENO3: along the wall at 128 x 64 cells, and its mean against the mesh, converging at order 1.3 with straight walls and 2.8 with curved ones](../images/guide/axisymmetric_curved_sphere-dark.svg#gh-dark-mode-only)

  *The Mach 0.3 runs (`python tools/docs/axisymmetric_curved.py`): at L = 4
  curved walls keep the error 1.5 to 6.6 times below the straight walls'
  along the whole wall: 1.7 near the front stagnation point, 5 to 6.6 past
  the equator.*
- Counterflow diffusion flame (H2/N2 against air, 1 atm, 10 mm gap, the
  finite-jet setup of the 3D counterflow validation on the meridian plane, 20
  cells per flame width, MUSCL, mixture-averaged transport, started from
  Cantera's flame and run for $4/K_\mathrm{ox}$): the peak temperature is within 0.4%
  ($U_o$ = 2 m/s) and 0.7% (4 m/s) of Cantera's axisymmetric similarity solution
  at the same local strain rate $K_\mathrm{ox}$ measured on the axis, as in the 3D runs.
