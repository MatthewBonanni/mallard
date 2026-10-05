# Design: axisymmetric (r-z) flows

Status: implemented (issue #157).

`[physics] axisymmetric = true` turns the 2D solver into one for flows
symmetric about the x axis without swirl: x is the axial coordinate z and y
the radius r >= 0. One 2D mesh of the meridian half-plane then stands for the
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

    S_c = int_c (p - tau_thetatheta) dA,
    tau_thetatheta = mu (2 u_r / r - 2/3 div u),  div u = du_x/dx + du_r/dr + u_r / r.

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
- Counterflow diffusion flame (H2/N2 against air, 1 atm, 10 mm gap, the
  finite-jet setup of the 3D counterflow validation on the meridian plane, 20
  cells per flame width, MUSCL, mixture-averaged transport, started from
  Cantera's flame and run for 4 / K_ox): the peak temperature is within 0.4%
  (U_o = 2 m/s) and 0.7% (4 m/s) of Cantera's axisymmetric similarity solution
  at the same local strain rate K_ox measured on the axis, as in the 3D runs.
