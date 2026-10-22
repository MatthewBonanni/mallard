# Design: wall-modeled large-eddy simulation

Status: stages 1 and 2 implemented (#256); stage 3 (non-equilibrium) and the
optional comparisons are planned. Recommendations are marked **Decision**;
the alternatives considered are listed with each.

Mallard's LES ([les.md](les.md)) resolves the near-wall cycle: the first cell
is at `y+ ~ 1`, and the cell count of a wall-bounded flow grows like
`Re_tau^2` (Choi & Moin 2012). Wall-modeled LES (WMLES) resolves only the
outer layer: a model supplies the wall shear stress and heat flux from the
LES state at some height above the wall, and the mesh scales with the
boundary-layer thickness, not with the viscous length. This document adds
WMLES for compressible flows with heat transfer on Mallard's unstructured
meshes, with physics-based models only (no machine-learned models:
maintainer decision).

## Goals

- **Compressible, with heat transfer.** Isothermal walls carry a modeled heat
  flux; the models hold at supersonic bulk Mach numbers with strong wall
  cooling.
- **Unstructured, mixed meshes**: the exchange location is a point, located in
  any cell type, not "the n-th cell" of a structured wall-normal line.
- **GPUs**: the models are per-face device functions without host round
  trips, captured in the step's CUDA graph.
- **Determinism**: results bitwise independent of the rank count; a
  restarted run reproduces an uninterrupted one bitwise.
- **No cost and no change without a wall model**: runs without one give
  byte-identical results.
- Works with the explicit LES (Sigma with Scotti's width, the hybrid flux,
  the dissipation budget) and with TENO and MUSCL near walls.

## Non-goals

- Machine-learned wall models.
- RANS and hybrid RANS/LES (DES, IDDES), whose "wall model" is a RANS layer
  resolved by the mesh.
- Roughness, transpiration, chemically reacting walls.

## Summary of decisions

| # | Decision | Main alternatives | Why |
|---|---|---|---|
| 1 | Wall-stress model: the model's `tau_w` and `q_w` replace the viscous flux of the wall face; the LES sees a no-penetration (slip) wall elsewhere | Slip-velocity models (Bose & Moin 2014); keeping the no-slip ghost | The standard WMLES coupling (Larsson et al. 2016; Bose & Park 2018); one flux, no change to the convective scheme |
| 2 | Exchange point a distance `h` along each wall face's inward normal (Kawai & Larsson 2012), at least about 3 cells off the wall | The first cell's value; the n-th cell of a wall-normal line | Removes the log-layer mismatch; a point is defined on any mesh |
| 3 | State at the point by linear reconstruction in the cell containing it: the cell average plus the LSQ gradient times the offset | The containing cell's average at its own height; trilinear interpolation of vertex values | Exact for linear fields on every cell type; uses gradients the RHS already has; one cell per point |
| 4 | Containing cells found at setup by a walk through faces; the halo deepened until every point's cell (and its gradient stencil) is local | Point-to-point messages from the owner of the point to the owner of the face | Reuses the halo machinery that already makes stencils rank-independent; no new communication in the time loop |
| 5 | Stage 1: the equilibrium ODE model (coupled thin-layer momentum and energy, mixing length with damping in semi-local units) | Algebraic laws of the wall (Reichardt, Spalding) | The classical compressible baseline (Kawai & Larsson 2012), with heat flux |
| 6 | **Stage 2, default: Griffin, Fu & Moin (2023)**, the inverse of their velocity transformation plus an algebraic temperature-velocity relation | The ODE model; flux-controlled compressible law of the wall | 2-4 times smaller `tau_w` and `q_w` errors than the ODE model in their channel and boundary-layer tests, most at strong heat transfer; marching, no tridiagonal solve |
| 7 | Edge state for that relation from a second point, `edge_distance` along the normal | Boundary-layer edge detection (Griffin, Fu & Moin 2021a) | Exact for channels (the centerline) and simple; edge detection is a later addition for developing boundary layers |
| 8 | Instantaneous inputs, evaluated at every stage | Time-filtered inputs (Yang et al. 2017) | Stateless: restarts are exact without new restart fields |
| 9 | Single gases first; gas mixtures after isothermal mixture walls (#295) | | The models need `cp`, `mu(T)`, `Pr`, and the heat flux path of mixture walls |

## 1. Coupling to the LES (decision 1)

A wall face with a wall model takes its momentum and energy flux from the
model:

```
F_momentum = -tau_w t,      F_energy = u_wall . F_momentum + q_w
```

with `t` the unit vector of the velocity at the exchange point relative to
the wall, tangential to the face, `tau_w >= 0` the modeled wall shear stress,
and `q_w` the modeled heat flux into the fluid (zero on adiabatic walls,
`q` on heat-flux walls with the ODE model). There is no normal viscous
stress and no SGS flux on the face. The convective flux of the wall face is
unchanged: its mirror ghost has no mass flux, and the Riemann solvers and
the hybrid flux return the pressure force.

The ghost state seen by reconstruction and by the gradients is that of a
slip wall (mirrored normal velocity, interior temperature): with the
no-slip ghost the boundary cell's gradient would hold the jump from the wall
to a first cell that is in the log layer (`u_1 / (Delta / 2)`), which feeds
the eddy viscosity and the MUSCL and TENO slopes of the wall cells with a
shear the LES does not resolve. With the slip ghost the boundary cell's
wall-normal gradient comes from the interior side only. The wall-resolved
path is unchanged.

Alternatives: slip-velocity models (Bose & Moin 2014; Bae et al. 2019)
impose a Robin condition on the velocity instead of a flux; they need a
dynamic procedure with test filtering (les.md, section 2.5, on why that is
hard on Mallard's meshes) and are kept as the optional comparison of stage 4.

## 2. Exchange location (decisions 2-4)

Kawai & Larsson (2012) showed that the "log-layer mismatch" of WMLES (a
15% error in `Cf` when the model is fed the first cell's value) comes from
the under-resolution of the first few cells: the LES state there is
contaminated by numerical errors and by the SGS model's inaccuracy at the
grid scale. Taking the model's input from a height `h` of several cells
(`h >= 3 Delta`, and `h` within the inner layer, `h <= 0.1-0.3 delta`)
removes it.

- **Exchange point.** For wall face `f` with centroid `x_f` and inward unit
  normal `n`, `x_e = x_f + h n`. `h` is `exchange_distance` (absolute), or
  `exchange_cells` times the wall cell's height `2 (x_f - x_c) . n`, which
  adapts to meshes whose near-wall spacing varies along the wall.
- **Interpolation.** `q(x_e) = q_c + grad q_c . (x_e - x_c)` in the cell
  `c` containing `x_e`, for `q = rho, u, p` and their LSQ gradients (the
  viscous gradients of the RHS); `T = p / (rho R)`. It is exact for linear
  fields on any cell type, continuous in `h` within a cell, and needs no
  vertex values. The containing cell's value at its own height (`h_f = (x_c
  - x_f) . n`, without interpolation) is the cheaper option for meshes whose
  near-wall cells are aligned with the wall; it is equivalent to the
  interpolation at the cell's centroid and is what structured codes do.
- **Location.** At setup, a walk from the wall cell through the face across
  which `x_e` lies most outside (the face with the largest `(x_e - x_face) .
  n_out`), translated across periodic seams, until `x_e` is inside the cell
  (the same test as the transmissive boundary's image search). The walk is
  on the local mesh (owned and halo cells). A point outside the domain (a
  thin region, a concave corner) is an input error naming the face.
- **MPI.** The point's cell must be local with a complete gradient stencil:
  owned, or in a halo layer below the deepest. When it is not, the halo is
  deepened on every rank and the setup after the mesh repeats, as for TENO's
  stencils ([mpi.md](mpi.md), section 6). Then the interpolated state, and
  hence the wall flux, is bitwise the same on any number of ranks. The cost
  is a deeper halo where a partition boundary runs parallel to a wall,
  within `h` of it; the built-in partitioners cut boxes into slabs and
  columns, which in a channel run along the wall-normal direction.
  Exchanging only the needed cells (trimmed halo) is a possible later
  optimization.

## 3. Equilibrium ODE model (stage 1, decision 5)

Thin-layer, equilibrium (no convection, no pressure gradient) momentum and
energy equations along the wall-normal coordinate `y` from the wall to `h`
(Kawai & Larsson 2012; Larsson et al. 2016):

```
d/dy [ (mu + mu_t) du/dy ] = 0
d/dy [ (mu + mu_t) u du/dy + cp (mu / Pr + mu_t / Pr_t) dT/dy ] = 0
mu_t = kappa rho y sqrt(tau_w / rho) [1 - exp(-y* / A+)]^2,   y* = y sqrt(rho tau_w) / mu
```

with `kappa = 0.41`, `A+ = 17`, `Pr_t = 0.9`, the gas's `mu(T)` and `Pr`, and
`p` constant across the layer (`rho = p / (R T)`). Boundary conditions:
`u = 0`, and `T = T_w` (isothermal), `-k dT/dy = q` (heat-flux, adiabatic
`q = 0`) at the wall; `u = U_e`, `T = T_e` at `y = h`, with `U_e` the
tangential speed relative to the wall at the exchange point.

- **Discretization.** 64 nodes, geometrically stretched (ratio 1.1, first
  spacing `2.2e-4 h`, so `y1+ < 1` for `h+` up to 4500). Both equations say
  a flux is constant across the layer, `(mu + mu_t) du/dy = tau_w` and
  `k_eff dT/dy + tau_w u = -q_w`, so for given `mu_t` and properties `u`
  and `T` follow by quadrature (Simpson's rule on each interval, with
  midpoint values) and the boundary values fix `tau_w` and `q_w` (or `T_w`
  on heat-flux walls). `mu_t`, the properties and `tau_w` are iterated to a
  fixed point, accelerated by secant steps on `tau_w`, until `tau_w`
  changes by less than `1e-10` relatively: 3-5 iterations from `h+ = 1` to
  `10^4` (at most 200).

## 4. Inverse velocity transformation model (stage 2, decision 6)

Griffin, Fu & Moin (2021b) proposed the velocity transformation

```
S_t+ = S_eq+ / (1 + S_eq+ - S_TL+),   S_eq+ = (1 / mu+) dU+/dy*,   S_TL+ = mu+ dU+/dy+
```

(`U+ = U / u_tau`, `y+` in wall units, `y*` in semi-local units, `mu+ = mu /
mu_w`, `rho+ = rho / rho_w`) that maps compressible profiles with and
without heat transfer onto the incompressible law `dU_t+/dy* = S_t+`.
Griffin, Fu & Moin (2023) invert it: given the incompressible law (the
constant-property version of the ODE model of section 3, `S_t+ = 1 / (1 +
kappa y* [1 - exp(-y* / A+)]^2)`, `kappa = 0.41`, `A+ = 17`) and a
temperature-velocity relation, the compressible profile follows by
integrating

```
dU+/dy* = [ 1 / (mu+ S_t+) - 1 / mu+ + sqrt(rho+) (1 + y+/(2 rho+) drho+/dy+ - y+/mu+ dmu+/dy+) ]^-1
```

from the wall. The temperature-velocity relation is the generalized Reynolds
analogy of Zhang et al. (2014), with its quadratic coefficient set by the
exchange point so that it passes through `(U_m, T_m)` exactly:

```
T = T_w + s Pr (T_r - T_w) (U / U_e) (1 - U / U_m) + (U / U_m)^2 (T_m - T_w)
T_r = T_e + r U_e^2 / (2 cp),   r = Pr^(1/3),   s = 1.14
q_w = s tau_w cp (T_w - T_r) / U_e
```

(`q_w` is `-k_w dT/dy` at the wall of this profile). Unknown: `tau_w`.

- **Discretization.** On the grid of section 3, from node `j` to `j + 1`
  the transformation is applied in difference form, `S_eq+ = dU+ / (mu+
  dy*)`, `S_TL+ = mu+ dU+ / dy+` with `mu+` the mean of the two nodes and
  `S_t+` averaged over the interval in `y*` (Simpson's rule): solving for
  the increment gives `dU+ = S_t+ / [(1 - S_t+) / (mu+ dy*) + S_t+ mu+ /
  dy+]`. `dy*` depends on the temperature (hence on `U`) at node `j + 1`,
  so each step predicts `U_(j+1)` from the previous increment and corrects
  it three times. The differences of `y*`
  carry the density and viscosity derivatives of the differential form
  exactly. With constant properties `dU+ = S_t+ dy+`.
- **Shooting.** `U(h)` increases with `tau_w`; the root of `log U(h;
  tau_w) = log U_m`, nearly linear in `log tau_w`, is found by secant steps
  on `log tau_w` from the larger of the laminar and log-law estimates, to
  `1e-9`: 5-7 profile evaluations.
- **Accuracy** (unit tests): with constant properties both models agree
  with the integral of the incompressible law to `1e-5` from `h+ = 1` to
  `10^4`; with variable properties each agrees with an RK4 solution of its
  differential equations on a fine grid to `3e-5` (`tau_w`, `q_w`).
- **Inputs.** `U_m`, `T_m`, `p` at the exchange point; `T_w`; `U_e`, `T_e` at
  the edge (section 5); the gas's `mu(T)`, `cp`, `Pr`. Adiabatic walls take
  `T_w = T_r` (`q_w = 0`); heat-flux walls are not supported by this model
  (the ODE model is).
- **Why the default.** In Griffin et al.'s WMLES of the channels of Trettel
  & Larsson (2016) (`M_b` = 0.7-4, `-B_q` up to 0.19), the errors of the ODE
  model in `tau_w` and `q_w` grow with the heat transfer while theirs stay
  small and independent of it; a priori, over 48 channels, pipes and
  boundary layers, they are within a few percent. It needs no tridiagonal
  solve, and its incompressible limit is the ODE model's (tested).

## 5. Edge state (decision 7)

The relation of section 4 needs the edge (or centerline) velocity and
temperature. Griffin et al. find the boundary-layer edge with their general
method (Griffin, Fu & Moin 2021a), which integrates the vorticity along the
wall normal. Mallard samples a second point, `x_f + edge_distance n`, like
the exchange point: in channels and pipes `edge_distance` is the half-height
(the centerline), in boundary layers an estimate of `delta`. `U_e` is
clipped from below at `U_m`. Edge detection along normals is planned with
stage 3, when developing and separating boundary layers need it.

## 6. Determinism, restarts and the time loop (decision 8)

- The wall flux of a face is a function of the state and gradients of two
  local cells (exchange, edge), computed with fixed iteration rules, so it
  is the same on any number of ranks when those cells' values are (section
  2). No reduction enters.
- Inputs are instantaneous and the models keep no state between stages, so
  restarts are exact without new restart variables. Time filtering of the
  inputs (Yang et al. 2017), helpful for very coarse meshes, would add a
  per-face restart field and is deferred.
- The models run at every RHS evaluation, after the viscous gradients, as
  one kernel over the wall faces; the force monitors evaluate them from the
  current state (after a halo exchange) and report the modeled stress.

## 7. Implementation

- Input, per `[[boundaries]]` wall entry: a `wall_model` table with `model`
  (`"gfm"`, the default, or `"equilibrium"`), `exchange_distance` or
  `exchange_cells`, `edge_distance` (GFM, isothermal and adiabatic walls),
  and optional constants `kappa`, `A_plus`, `Pr_t` (ODE), `s` (GFM).
  Requires `[physics] type = "navier_stokes"` and a single gas.
- `wall_model.h`: the models as device functions on plain values (unit
  tested on the host); `WallModelFunctor` per wall face; the face index and
  modeled flux in `WallModelFlux`, read by `ViscousFluxFunctor` and the force
  monitors.
- `BoundaryCondition::modeled` switches the wall's ghost to slip and its
  viscous flux to the model's.
- Output: `TAU_W` and `Q_W` are not cell fields; the `[[forces]]` files of
  modeled zones add a column `Q`, the heat flux into the fluid integrated
  over the zone, after the viscous force.

## 8. Tests

- **Models, incompressible limit**: at constant properties both models give
  the `tau_w` of the integral of `(1 + kappa y+ D) dU+/dy+ = 1` (evaluated
  independently by quadrature) within 0.5%, from `h+ = 1` to `10^4`.
- **Laminar limit**: Couette flow with heat transfer at `h+ << 1`: the ODE
  model's `tau_w = mu U / h` and `q_w = -(k (T_e - T_w) + mu U^2 / 2) / h`.
- **GFM relation**: the profile passes through `(U_m, T_m)`; `q_w` equals
  the Reynolds analogy; adiabatic walls give `q_w = 0` and `T_w = T_r`.
- **Exchange points**: on hexahedra, tetrahedra, prisms, pyramids and the
  mixed box, the located cell contains the point and a linear field is
  reconstructed exactly at it; periodic seams are crossed.
- **Solver**: a channel with modeled walls and the laminar-limit model gives
  the exact wall stress of the steady laminar solution; the wall flux
  balances the mass-flow force in a statistically steady channel.
- **MPI and restarts**: 1, 2 and 3 ranks bitwise identical with both models;
  a restarted run matches the uninterrupted one bitwise.

## 9. Validation

Coarse WMLES of channels (GPU budget: four A100s), Sigma with Scotti's
width, hybrid flux, MUSCL without limiter, `exchange_distance = 0.3 delta`,
uniform hexahedra of `delta / 12` (the resolution of Griffin et al.), box
`2 pi x 2 x pi`, mass flow held, isothermal walls. Each case runs with both
models.

1. **Incompressible-like channel**, `M_b = 0.2`, at `Re_tau` = 2000 and 5200
   against Lee & Moser (2015): `Cf` and `U+(y)`. Bulk Reynolds numbers of
   their runs (`2 delta U_b / nu` = 86,957 and 250,000).
2. **Compressible isothermal channels**, `M_b = U_b / c_w` = 1.7 and 3.0
   (`Re_tau*` = 590) of Trettel & Larsson (2016), the a posteriori cases of
   Griffin et al. (2023), with `mu ~ T^0.75`, `Pr = 0.7`, the body force's
   work as the only energy source: `Re_tau` (hence `tau_w`), `B_q = q_w /
   (cp T_w rho_w u_tau)` and the centerline temperature from `Re_tau*` (for
   `mu ~ T^0.75`, `T_c / T_w = (Re_tau / Re_tau*)^0.8`), against their
   table 1.

Results: section 10.

## 10. Results

Reported with the validation examples (stage 2's validation PR).

## 11. Stages

1. **Equilibrium ODE model** with the off-wall exchange point, the coupling
   (section 1), tests. Implemented.
2. **GFM model** (default), edge sampling: implemented; channel validation
   (`examples/channel_wmles`) follows.
3. **Non-equilibrium**: the pressure-gradient (and convection) terms of the
   thin-layer equations, `d/dy[(mu + mu_t) du/dy] = dp/dx` (+ the
   integrated convective terms, Park & Moin 2014 style) for adverse pressure
   gradients and separation, with the pressure-gradient extension of the
   inverse-transformation model of Griffin et al. (CTR briefs); edge
   detection along normals (Griffin, Fu & Moin 2021a). Validation: a
   separating case (NASA wall-mounted hump, Greenblatt et al. 2006, or the
   Boeing Gaussian bump), compressible boundary layers with synthetic inflow
   (`[boundaries.turbulence]`) at `M_e` = 2-6. Optionally time-filtered
   inputs. Next PR.
4. **Optional comparisons**: slip-wall models (Bose & Moin 2014; Bae et al.
   2019); the flux-controlled compressible law of the wall (arXiv:2512.04688)
   as a cross-check.
5. **Gas mixtures**: after #295 (isothermal mixture walls): the models with
   the mixture's `cp`, `mu` and `Pr` frozen at the exchange point's
   composition, zero species flux at the wall.

## References

DOIs checked against Crossref.

- Bae, Lozano-Durán, Bose & Moin 2019, *J. Fluid Mech.* 859, 400. [doi:10.1017/jfm.2018.838](https://doi.org/10.1017/jfm.2018.838)
- Bose & Moin 2014, *Phys. Fluids* 26, 015104. [doi:10.1063/1.4849535](https://doi.org/10.1063/1.4849535)
- Bose & Park 2018, *Annu. Rev. Fluid Mech.* 50, 535. [doi:10.1146/annurev-fluid-122316-045241](https://doi.org/10.1146/annurev-fluid-122316-045241)
- Cabot & Moin 2000, *Flow Turbul. Combust.* 63, 269. [doi:10.1023/A:1009958917113](https://doi.org/10.1023/A:1009958917113)
- Choi & Moin 2012, *Phys. Fluids* 24, 011702. [doi:10.1063/1.3676783](https://doi.org/10.1063/1.3676783)
- Greenblatt, Paschal, Yao, Harris, Schaeffler & Washburn 2006, *AIAA J.* 44, 2820. [doi:10.2514/1.13817](https://doi.org/10.2514/1.13817)
- Griffin, Fu & Moin 2021a, *Phys. Rev. Fluids* 6, 024608. [doi:10.1103/PhysRevFluids.6.024608](https://doi.org/10.1103/PhysRevFluids.6.024608)
- Griffin, Fu & Moin 2021b, *Proc. Natl. Acad. Sci.* 118, e2111144118. [doi:10.1073/pnas.2111144118](https://doi.org/10.1073/pnas.2111144118)
- Griffin, Fu & Moin 2023, *J. Fluid Mech.* 970, A36 (arXiv:2307.04958). [doi:10.1017/jfm.2023.627](https://doi.org/10.1017/jfm.2023.627)
- Kawai & Larsson 2012, *Phys. Fluids* 24, 015105. [doi:10.1063/1.3678331](https://doi.org/10.1063/1.3678331)
- Larsson, Kawai, Bodart & Bermejo-Moreno 2016, *Mech. Eng. Rev.* 3, 15-00418. [doi:10.1299/mer.15-00418](https://doi.org/10.1299/mer.15-00418)
- Lee & Moser 2015, *J. Fluid Mech.* 774, 395. [doi:10.1017/jfm.2015.268](https://doi.org/10.1017/jfm.2015.268)
- Park & Moin 2014, *Phys. Fluids* 26, 015108. [doi:10.1063/1.4861069](https://doi.org/10.1063/1.4861069)
- Trettel & Larsson 2016, *Phys. Fluids* 28, 026102. [doi:10.1063/1.4942022](https://doi.org/10.1063/1.4942022)
- van Driest 1956, *J. Aeronaut. Sci.* 23, 1007. [doi:10.2514/8.3713](https://doi.org/10.2514/8.3713)
- Wang & Moin 2002, *Phys. Fluids* 14, 2043. [doi:10.1063/1.1476668](https://doi.org/10.1063/1.1476668)
- Yang, Park & Moin 2017, *Phys. Rev. Fluids* 2, 104601. [doi:10.1103/PhysRevFluids.2.104601](https://doi.org/10.1103/PhysRevFluids.2.104601)
- Zhang, Bi, Hussain & She 2014, *J. Fluid Mech.* 739, 392. [doi:10.1017/jfm.2013.620](https://doi.org/10.1017/jfm.2013.620)
