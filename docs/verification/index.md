# Verification manual

Observed orders of accuracy of Mallard's discretizations, measured against
exact and manufactured solutions: every reconstruction on every cell type,
the Euler and Navier-Stokes operators, the time integrators, boundary
conditions, curved boundaries, remapping and gas mixtures. The methods are
described in the [theory manual](../theory/index.md); stability limits and
other measurements are in [Numerical methods](../numerics/overview.md).

<!-- BEGIN GENERATED summary -->
<!-- END GENERATED summary -->

## How the tables are made

Each study is a test of `MallardTest` that computes an error on two
resolutions $n_1 < n_2$ (cells per side, or time steps) and checks the
observed order

$$
\begin{equation}
p = \frac{\log(e_1 / e_2)}{\log(n_2 / n_1)}
\label{eq:observed-order}
\end{equation}
$$

against bounds of its own: usually a lower bound a little below the design
order, which a pre-asymptotic error may exceed. The tests check their orders
through one function, `verification::expect_order`
([`test/verification.h`](../../test/verification.h)), which also records
each study; [`tools/docs/verification.py`](../../tools/docs/verification.py)
runs the tests and writes the tables, the plots and
[`results.json`](results.json). A row passes when the test's bounds hold,
so a failing row is a failing test.

Most studies run in every CI build. The rest of the matrix of reconstructions
and cell types, too slow for every run, is instantiated as
`DISABLED_Verification/...` and runs only here. All run on the CPU in double
precision, on the meshes of a few seconds' run.

**Regenerating** (on every release, and with any change to a scheme): build
`MallardTest` in 2D and 3D, then

```bash
python tools/docs/verification.py --test build-2/test/MallardTest build-3/test/MallardTest
```

(about 10 minutes on 4 cores; `--jobs` sets the number of test processes).
It exits with status 1 if a study fails. Without `--test` it redraws the
page and plots from `results.json`. The *Verification* workflow
(`.github/workflows/verification.yml`, dispatched by hand) does the same on
GitHub's runners and uploads the regenerated files.

## Reconstruction on every cell type

Errors of the face values reconstructed from the exact cell averages of a
smooth field: the operator alone, without time integration. In 2D, the
maximum over interior face quadrature points; in 3D, over all faces, with the
mirrored stencils of symmetry walls. Observed orders:

<!-- BEGIN GENERATED matrix-reconstruction -->
<!-- END GENERATED matrix-reconstruction -->

![Reconstruction convergence by cell type](../images/verification/reconstruction-light.svg#gh-light-mode-only)
![Reconstruction convergence by cell type](../images/verification/reconstruction-dark.svg#gh-dark-mode-only)

Dashed lines show the design slope of each study.

<!-- BEGIN GENERATED table-reconstruction -->
<!-- END GENERATED table-reconstruction -->

## Euler equations: advection of a smooth wave

A density wave advected through a periodic box by a uniform flow at constant
pressure, an exact solution of the Euler equations, for a quarter period with
RK4 at $\Delta t \sim h^{\max(1, p/4)}$, so that the time error stays below
the spatial one; the error is the largest difference from the exact cell
averages. The full solver runs: reconstruction, HLLC fluxes, the face
quadrature and the time integrator. Prisms and pyramids are periodic in two
directions only; along the third the wave is uniform and bounded by symmetry
walls. Observed orders:

<!-- BEGIN GENERATED matrix-advection -->
<!-- END GENERATED matrix-advection -->

![Advection convergence by cell type](../images/verification/advection-light.svg#gh-light-mode-only)
![Advection convergence by cell type](../images/verification/advection-dark.svg#gh-dark-mode-only)

The pulse rows advect a Gaussian with SSPRK3 at fixed CFL, whose time error
limits the observed order to 3.

<!-- BEGIN GENERATED table-advection -->
<!-- END GENERATED table-advection -->

## Navier-Stokes equations: viscous operators

Stokes' first problem (an impulsively started isothermal wall, with the erfc
profile as exact solution) and a channel driven by a body force between walls
at rest, $u = 0.05\,(y - y^3)$, whose ends are transmissive. Both use MUSCL
with the Venkatakrishnan limiter and the second-order viscous gradients.

![Viscous convergence](../images/verification/viscous-light.svg#gh-light-mode-only)
![Viscous convergence](../images/verification/viscous-dark.svg#gh-dark-mode-only)

<!-- BEGIN GENERATED table-viscous -->
<!-- END GENERATED table-viscous -->

## Time integrators

The linear system $\mathrm{d}U_i/\mathrm{d}t = \lambda_i U_i$ to $t = 1$ checks
each method's order; $\mathrm{d}U/\mathrm{d}t = \cos t$ checks that the stages
are evaluated at the right times. RODAS, the stiff integrator of the
chemistry, runs the Van der Pol oscillator.

![Time integrator convergence](../images/verification/time-light.svg#gh-light-mode-only)
![Time integrator convergence](../images/verification/time-dark.svg#gh-dark-mode-only)

<!-- BEGIN GENERATED table-time -->
<!-- END GENERATED table-time -->

## Boundary conditions

The TENO face values next to symmetry walls, where the stencils include the
mirror images of the cells. The viscous studies above verify isothermal
no-slip walls, moving walls and transmissive ends; the axisymmetric studies
below the axis.

<!-- BEGIN GENERATED table-boundaries -->
<!-- END GENERATED table-boundaries -->

## Axisymmetric flows

The spatial operator at the exact $r$-weighted averages of a manufactured
state, up to and including the cells on the axis (a residual, one order below
the reconstruction's), and the viscous residual of Poiseuille pipe flow.

<!-- BEGIN GENERATED table-axisymmetric -->
<!-- END GENERATED table-axisymmetric -->

## Curved boundaries

The area of an annulus and the volume of a spherical shell meshed with
straight and with curved walls, and the TENO wall values from the exact
averages over the curved cells. Seen as straight-sided cells, the same
averages give second order at best; by default orders 5 and 6 fit degree 3
in the cells whose stencils reach a curved wall.

![Curved geometry convergence](../images/verification/curved-light.svg#gh-light-mode-only)
![Curved geometry convergence](../images/verification/curved-dark.svg#gh-dark-mode-only)

<!-- BEGIN GENERATED table-curved -->
<!-- END GENERATED table-curved -->

## Remapping

Conservative remapping of a smooth periodic field between unrelated meshes,
by the overlap integrals of piecewise polynomials of each degree, unlimited:
the target cell averages converge at the degree plus one.

![Remap convergence](../images/verification/remap-light.svg#gh-light-mode-only)
![Remap convergence](../images/verification/remap-dark.svg#gh-dark-mode-only)

<!-- BEGIN GENERATED table-remap -->
<!-- END GENERATED table-remap -->

## Gas mixtures

A smooth composition and temperature wave of H<sub>2</sub>, O<sub>2</sub>
and N<sub>2</sub> advected at uniform velocity and pressure, thermally
perfect: both the flow variables and the species reach the scheme's order.

<!-- BEGIN GENERATED table-mixtures -->
<!-- END GENERATED table-mixtures -->

## Not covered

- The reacting source terms in space (chemistry is verified as an ODE
  integrator, above, and against Cantera in the
  [validation cases](https://matthewbonanni.github.io/mallard/validation/)).
- Mixture-averaged transport, LES models and characteristic boundaries,
  which are tested for exactness and consistency, not for an order.
- Discontinuous solutions, whose errors converge at order 1 or below
  regardless of the scheme; see the validation cases.
