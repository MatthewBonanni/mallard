# Verification manual

Observed orders of accuracy of Mallard's discretizations, measured against
exact and manufactured solutions: every reconstruction on every cell type,
the Euler and Navier-Stokes operators, the time integrators, boundary
conditions, curved boundaries, remapping and gas mixtures. The methods are
described in the [theory manual](../theory/index.md); stability limits and
other measurements are in [Numerical methods](../numerics/overview.md).

<!-- BEGIN GENERATED summary -->
149 convergence studies from commit `52ed962` (2026-10-22): 147 within their tests' bounds, 2 short of them for known reasons ([#420](https://github.com/MatthewBonanni/mallard/issues/420)).
<!-- END GENERATED summary -->

## How the tables are made

Each study is a test of `mallard-test` that computes an error on two
resolutions $n_1 < n_2$ (cells per side, or time steps) and checks the
observed order

$$
\begin{equation}
p = \frac{\log(e_1 / e_2)}{\log(n_2 / n_1)}
\label{eq:observed-order}
\end{equation}
$$

against bounds of its own: usually a lower bound a little below the design
order (0.25 below in 2D, 0.5 on the coarser 3D meshes), which a
pre-asymptotic error may exceed. The tests check their orders
through one function, `verification::expect_order`
([`test/verification.h`](../../test/verification.h)), which also records
each study; [`tools/docs/verification.py`](../../tools/docs/verification.py)
runs the tests and writes the tables, the plots and
[`results.json`](results.json). A row passes when the test's bounds hold,
so a failing row is a failing test; a known deficiency links its issue.

Most studies run in every CI build. The rest of the matrix of reconstructions
and cell types, too slow for every run, is instantiated as
`DISABLED_Verification/...` and runs only here. All run on the CPU in double
precision, on the meshes of a few seconds' run.

**Regenerating** (on every release, and with any change to a scheme): build
`mallard-test` in 2D and 3D, then

```bash
python tools/docs/verification.py --test build-2/test/mallard-test build-3/test/mallard-test
```

(about 10 minutes on 4 cores; `--jobs` sets the number of test processes).
It exits with status 1 if a study fails, other than the known deficiencies
listed in the script. Without `--test` it redraws the
page and plots from `results.json`. The *Verification* workflow
(`.github/workflows/verification.yml`, dispatched by hand) does the same on
GitHub's runners and uploads the regenerated files.

## Reconstruction on every cell type

Errors of the face values reconstructed from the exact cell averages of a
smooth field: the operator alone, without time integration. In 2D, the
maximum over interior face quadrature points; in 3D, over all faces, with the
mirrored stencils of symmetry walls. Observed orders (TENO6 on mixed meshes
is left out, at a minute per resolution; their cell types are covered one by
one):

<!-- BEGIN GENERATED matrix-reconstruction -->
| | Quadrilaterals | Triangles | Hexahedra | Tetrahedra | Prisms | Pyramids | Mixed |
|---|---|---|---|---|---|---|---|
| TENO3 (design 3) | 2.89 | 2.92 | 2.72 | 2.68 | 2.81 | 2.88 | 2.54 |
| TENO4 (design 4) | 4.08 | 3.97 | 3.77 | 3.55 | 3.55 | 3.78 | 3.56 |
| TENO5 (design 5) | 4.90 | 4.98 | 4.93 | 4.73 | 5.07 | 4.84 | 5.04 |
| TENO6 (design 6) | 6.05 | 5.95 | 5.73 | 5.58 | 5.58 | 5.60 | – |
<!-- END GENERATED matrix-reconstruction -->

![Reconstruction convergence by cell type](../images/verification/reconstruction-light.svg#gh-light-mode-only)
![Reconstruction convergence by cell type](../images/verification/reconstruction-dark.svg#gh-dark-mode-only)

Dashed lines show the design slope of each study.

<!-- BEGIN GENERATED table-reconstruction -->
| Method | Cells | Error | Resolutions | Errors | Design | Observed | Asserted | Result |
|---|---|---|---|---|---|---|---|---|
| TENO3 | hexahedra | max face-value error | 6 → 12 | 2.89e-02 → 4.37e-03 | 3 | 2.72 | > 2.5 | pass |
| TENO4 | hexahedra | max face-value error | 6 → 12 | 1.12e-02 → 8.22e-04 | 4 | 3.77 | > 3.5 | pass |
| TENO5 | hexahedra | max face-value error | 6 → 12 | 1.34e-02 → 4.39e-04 | 5 | 4.93 | > 4.5 | pass |
| TENO5, troubled | hexahedra | max face-value error | 6 → 12 | 1.34e-02 → 4.39e-04 | 5 | 4.93 | > 4.5 | pass |
| TENO6 | hexahedra | max face-value error | 6 → 12 | 7.84e-03 → 1.48e-04 | 6 | 5.73 | > 5.5 | pass |
| TENO3 | tetrahedra | max face-value error | 4 → 8 | 5.18e-02 → 8.09e-03 | 3 | 2.68 | > 2.5 | pass |
| TENO4 | tetrahedra | max face-value error | 4 → 8 | 2.58e-02 → 2.20e-03 | 4 | 3.55 | > 3.5 | pass |
| TENO4, troubled | tetrahedra | max face-value error | 4 → 8 | 2.58e-02 → 2.20e-03 | 4 | 3.55 | > 3.5 | pass |
| TENO5 | tetrahedra | max face-value error | 4 → 8 | 1.80e-02 → 6.77e-04 | 5 | 4.73 | > 4.5 | pass |
| TENO6 | tetrahedra | max face-value error | 4 → 8 | 7.30e-03 → 1.52e-04 | 6 | 5.58 | > 5.5 | pass |
| TENO3 | prisms | max face-value error | 6 → 12 | 1.87e-02 → 2.66e-03 | 3 | 2.81 | > 2.5 | pass |
| TENO4 | prisms | max face-value error | 6 → 12 | 9.43e-03 → 8.07e-04 | 4 | 3.55 | > 3.5 | pass |
| TENO5 | prisms | max face-value error | 6 → 12 | 5.91e-03 → 1.76e-04 | 5 | 5.07 | > 4.5 | pass |
| TENO6 | prisms | max face-value error | 6 → 12 | 5.20e-03 → 1.09e-04 | 6 | 5.58 | > 5.5 | pass |
| TENO3 | pyramids | max face-value error | 6 → 12 | 1.37e-02 → 1.86e-03 | 3 | 2.88 | > 2.5 | pass |
| TENO4 | pyramids | max face-value error | 6 → 12 | 3.39e-03 → 2.46e-04 | 4 | 3.78 | > 3.5 | pass |
| TENO5 | pyramids | max face-value error | 6 → 12 | 1.70e-03 → 5.93e-05 | 5 | 4.84 | > 4.5 | pass |
| TENO6 | pyramids | max face-value error | 6 → 12 | 4.40e-04 → 9.03e-06 | 6 | 5.60 | > 5.5 | pass |
| TENO3 | mixed | max face-value error | 6 → 12 | 2.29e-02 → 3.95e-03 | 3 | 2.54 | > 2.5 | pass |
| TENO4 | mixed | max face-value error | 6 → 12 | 9.69e-03 → 8.22e-04 | 4 | 3.56 | > 3.5 | pass |
| TENO5 | mixed | max face-value error | 6 → 12 | 1.19e-02 → 3.61e-04 | 5 | 5.04 | > 4.5 | pass |
| TENO3 | quadrilaterals | max face-value error, interior | 16 → 32 | 1.86e-03 → 2.50e-04 | 3 | 2.89 | > 2.8 | pass |
| TENO4 | quadrilaterals | max face-value error, interior | 16 → 32 | 1.91e-04 → 1.13e-05 | 4 | 4.08 | > 3.8 | pass |
| TENO5 | quadrilaterals | max face-value error, interior | 16 → 32 | 1.32e-04 → 4.42e-06 | 5 | 4.90 | > 4.8 | pass |
| TENO6 | quadrilaterals | max face-value error, interior | 16 → 32 | 2.39e-05 → 3.61e-07 | 6 | 6.05 | > 5.8 | pass |
| TENO3 | triangles | max face-value error, interior | 16 → 32 | 6.77e-04 → 8.92e-05 | 3 | 2.92 | > 2.8 | pass |
| TENO4 | triangles | max face-value error, interior | 16 → 32 | 1.26e-04 → 7.99e-06 | 4 | 3.97 | > 3.8 | pass |
| TENO5 | triangles | max face-value error, interior | 16 → 32 | 4.81e-05 → 1.52e-06 | 5 | 4.98 | > 4.8 | pass |
| TENO6 | triangles | max face-value error, interior | 16 → 32 | 4.53e-06 → 7.33e-08 | 6 | 5.95 | > 5.8 | pass |
<!-- END GENERATED table-reconstruction -->

## Euler equations: advection of a smooth wave

A density wave advected through a periodic box by a uniform flow at constant
pressure, an exact solution of the Euler equations, for a quarter period with
RK4 at $\Delta t = h^{\max(1, p/4)}/4$, so that the time error stays below
the spatial one (in 3D, a quarter of that on the thinner cells of the types
other than hexahedra, for stability); the error is the
largest difference from the exact cell averages. The full solver runs:
reconstruction, HLLC fluxes, the face quadrature and the time integrator.
Prisms and pyramids are periodic in two directions only; along the third the
wave is uniform and bounded by symmetry walls. In 3D the coarse mesh has 16
cells per side at first order, 12 for MUSCL, TENO3 and TENO5, and 8 for TENO4
and TENO6; TENO5 and TENO6 run on hexahedra only, at several minutes per
resolution on the other cells. Observed orders:

<!-- BEGIN GENERATED matrix-advection -->
| | Quadrilaterals | Triangles | Hexahedra | Tetrahedra | Prisms | Pyramids | Mixed |
|---|---|---|---|---|---|---|---|
| FO (design 1) | 0.80 | 0.87 | 0.75 | 0.92 | 0.89 | 0.89 | 0.82 |
| MUSCL (design 2) | 2.20 | 2.07 | 2.28 | 2.31 | 2.02 | 1.90 | 2.08 |
| TENO3 (design 3) | 2.90 | 2.98 | 2.79 | 2.94 | 2.92 | 2.98 | 2.64 |
| TENO4 (design 4) | 4.41 | 4.05 | 4.26 | 4.07 | 4.20 | 4.18 | 3.75 |
| TENO5 (design 5) | 4.87 | 4.96 | 4.73 | – | – | – | – |
| TENO6 (design 6) | 6.40 | 5.97 | 6.07 | – | – | – | – |
<!-- END GENERATED matrix-advection -->

![Advection convergence by cell type](../images/verification/advection-light.svg#gh-light-mode-only)
![Advection convergence by cell type](../images/verification/advection-dark.svg#gh-dark-mode-only)

The pulse rows advect a Gaussian with SSPRK3 at fixed CFL, whose time error
limits the observed order to 3.

<!-- BEGIN GENERATED table-advection -->
| Method | Cells | Error | Resolutions | Errors | Design | Observed | Asserted | Result |
|---|---|---|---|---|---|---|---|---|
| TENO5, SSPRK3 | quadrilaterals | L1 density error, pulse | 32 → 64 | 2.97e-04 → 1.33e-05 | 3 | 4.48 | > 2.7 | pass |
| TENO5, SSPRK3 | triangles | L1 density error, pulse | 32 → 64 | 6.89e-05 → 3.46e-06 | 3 | 4.32 | > 2.7 | pass |
| TENO5, SSPRK3 | hexahedra | L1 density error, pulse | 12 → 24 | 1.48e-03 → 1.13e-04 | 3 | 3.71 | > 3 | pass |
| FO | quadrilaterals | max density error | 16 → 32 | 7.12e-02 → 4.09e-02 | 1 | 0.80 | > 0.75 | pass |
| MUSCL | quadrilaterals | max density error | 16 → 32 | 5.01e-03 → 1.09e-03 | 2 | 2.20 | > 1.75 | pass |
| TENO3 | quadrilaterals | max density error | 16 → 32 | 7.77e-03 → 1.04e-03 | 3 | 2.90 | > 2.75 | pass |
| TENO4 | quadrilaterals | max density error | 16 → 32 | 6.11e-04 → 2.87e-05 | 4 | 4.41 | > 3.75 | pass |
| TENO5 | quadrilaterals | max density error | 16 → 32 | 9.39e-04 → 3.21e-05 | 5 | 4.87 | > 4.75 | pass |
| TENO6 | quadrilaterals | max density error | 16 → 32 | 1.02e-04 → 1.21e-06 | 6 | 6.40 | > 5.75 | pass |
| FO | triangles | max density error | 16 → 32 | 4.60e-02 → 2.52e-02 | 1 | 0.87 | > 0.75 | pass |
| MUSCL | triangles | max density error | 16 → 32 | 3.07e-03 → 7.28e-04 | 2 | 2.07 | > 1.75 | pass |
| TENO3 | triangles | max density error | 16 → 32 | 1.59e-03 → 2.01e-04 | 3 | 2.98 | > 2.75 | pass |
| TENO4 | triangles | max density error | 16 → 32 | 1.52e-04 → 9.16e-06 | 4 | 4.05 | > 3.75 | pass |
| TENO5 | triangles | max density error | 16 → 32 | 2.65e-04 → 8.51e-06 | 5 | 4.96 | > 4.75 | pass |
| TENO6 | triangles | max density error | 16 → 32 | 9.54e-06 → 1.52e-07 | 6 | 5.97 | > 5.75 | pass |
| FO | hexahedra | max density error | 16 → 32 | 7.81e-02 → 4.65e-02 | 1 | 0.75 | > 0.5 | pass |
| MUSCL | hexahedra | max density error | 8 → 16 | 2.65e-02 → 5.46e-03 | 2 | 2.28 | > 1.75 | pass |
| TENO3 | hexahedra | max density error | 12 → 24 | 1.86e-02 → 2.68e-03 | 3 | 2.79 | > 2.5 | pass |
| TENO4 | hexahedra | max density error | 8 → 16 | 1.33e-02 → 6.92e-04 | 4 | 4.26 | > 3.75 | pass |
| TENO5 | hexahedra | max density error | 12 → 24 | 5.02e-03 → 1.89e-04 | 5 | 4.73 | > 4.5 | pass |
| TENO6 | hexahedra | max density error | 8 → 16 | 7.78e-03 → 1.16e-04 | 6 | 6.07 | > 5.5 | pass |
| FO | tetrahedra | max density error | 16 → 32 | 4.25e-02 → 2.25e-02 | 1 | 0.92 | > 0.5 | pass |
| MUSCL | tetrahedra | max density error | 12 → 24 | 9.78e-03 → 1.97e-03 | 2 | 2.31 | > 1.5 | pass |
| TENO3 | tetrahedra | max density error | 12 → 24 | 3.46e-03 → 4.49e-04 | 3 | 2.94 | > 2.5 | pass |
| TENO4 | tetrahedra | max density error | 8 → 16 | 1.66e-03 → 9.91e-05 | 4 | 4.07 | > 3.5 | pass |
| FO | prisms | max density error | 16 → 32 | 4.07e-02 → 2.20e-02 | 1 | 0.89 | > 0.5 | pass |
| MUSCL | prisms | max density error | 12 → 24 | 4.43e-03 → 1.10e-03 | 2 | 2.02 | > 1.5 | pass |
| TENO3 | prisms | max density error | 12 → 24 | 4.75e-03 → 6.26e-04 | 3 | 2.92 | > 2.5 | pass |
| TENO4 | prisms | max density error | 8 → 16 | 2.98e-03 → 1.63e-04 | 4 | 4.20 | > 3.5 | pass |
| FO | pyramids | max density error | 16 → 32 | 4.23e-02 → 2.28e-02 | 1 | 0.89 | > 0.5 | pass |
| MUSCL | pyramids | max density error | 12 → 24 | 6.75e-03 → 1.80e-03 | 2 | 1.90 | > 1.5 | pass |
| TENO3 | pyramids | max density error | 12 → 24 | 3.21e-03 → 4.07e-04 | 3 | 2.98 | > 2.5 | pass |
| TENO4 | pyramids | max density error | 8 → 16 | 9.54e-04 → 5.26e-05 | 4 | 4.18 | > 3.5 | pass |
| FO | mixed | max density error | 16 → 32 | 7.39e-02 → 4.20e-02 | 1 | 0.82 | > 0.5 | pass |
| MUSCL | mixed | max density error | 12 → 24 | 1.44e-02 → 3.39e-03 | 2 | 2.08 | > 1.5 | pass |
| TENO3 | mixed | max density error | 12 → 24 | 1.42e-02 → 2.28e-03 | 3 | 2.64 | > 2.5 | pass |
| TENO4 | mixed | max density error | 8 → 16 | 9.82e-03 → 7.30e-04 | 4 | 3.75 | > 3.5 | pass |
<!-- END GENERATED table-advection -->

## Navier-Stokes equations: viscous operators

Stokes' first problem (an impulsively started isothermal wall, with the erfc
profile as exact solution) and a channel driven by a body force between
isothermal walls at rest, $u = 0.05\,(y - y^3)$: in 2D a strip with
transmissive ends, in 3D a column of 3 x 3 cells with periodic sides (walls
normal to $y$ on prisms, which are not periodic in $y$, else to $z$). All use
MUSCL with the Venkatakrishnan limiter and the second-order viscous gradients.

![Viscous convergence](../images/verification/viscous-light.svg#gh-light-mode-only)
![Viscous convergence](../images/verification/viscous-dark.svg#gh-dark-mode-only)

<!-- BEGIN GENERATED table-viscous -->
| Method | Cells | Error | Resolutions | Errors | Design | Observed | Asserted | Result |
|---|---|---|---|---|---|---|---|---|
| Forced channel, transmissive ends | quadrilaterals | max velocity error | 8 → 16 | 5.49e-04 → 1.42e-04 | 2 | 1.95 | > 1.8 | pass |
| Stokes' first problem | quadrilaterals | max velocity error | 32 → 64 | 7.92e-05 → 1.98e-05 | 2 | 2.00 | > 1.8 | pass |
| Forced channel, transmissive ends | triangles | max velocity error | 8 → 16 | 2.41e-04 → 6.33e-05 | 2 | 1.93 | > 1.8 | pass |
| Stokes' first problem | triangles | max velocity error | 32 → 64 | 3.60e-05 → 8.04e-06 | 2 | 2.16 | > 1.8 | pass |
| Forced channel | hexahedra | max velocity error | 8 → 16 | 5.49e-04 → 1.42e-04 | 2 | 1.95 | > 1.8 | pass |
| Forced channel | tetrahedra | max velocity error | 8 → 16 | 1.60e-04 → 4.25e-05 | 2 | 1.91 | > 1.8 | pass |
| Forced channel | prisms | max velocity error | 8 → 16 | 2.42e-04 → 6.27e-05 | 2 | 1.95 | > 1.8 | pass |
| Forced channel | pyramids | max velocity error | 8 → 16 | 2.64e-04 → 6.86e-05 | 2 | 1.95 | > 1.8 | pass |
| Forced channel | mixed | max velocity error | 8 → 16 | 5.24e-04 → 1.36e-04 | 2 | 1.94 | > 1.8 | pass |
<!-- END GENERATED table-viscous -->

## Time integrators

The linear system $\mathrm{d}U_i/\mathrm{d}t = \lambda_i U_i$ to $t = 1$ checks
each method's order; $\mathrm{d}U/\mathrm{d}t = \cos t$ checks that the stages
are evaluated at the right times. RODAS, the stiff integrator of the
chemistry, runs the Van der Pol oscillator.

![Time integrator convergence](../images/verification/time-light.svg#gh-light-mode-only)
![Time integrator convergence](../images/verification/time-dark.svg#gh-dark-mode-only)

<!-- BEGIN GENERATED table-time -->
| Method | Cells | Error | Resolutions | Errors | Design | Observed | Asserted | Result |
|---|---|---|---|---|---|---|---|---|
| RODAS (chemistry) | – | error, Van der Pol oscillator | 32 → 64 | 3.86e-09 → 2.39e-10 | 4 | 4.01 | 3.8 to 4.3 | pass |
| CK45 | – | error, time-dependent right-hand side | 10 → 20 | 3.69e-10 → 2.17e-11 | 4 | 4.09 | > 3.85 | pass |
| RDPK3Sp35 | – | error, time-dependent right-hand side | 10 → 20 | 5.23e-09 → 3.25e-10 | 3 | 4.01 | > 2.85 | pass |
| RK4 | – | error, time-dependent right-hand side | 10 → 20 | 2.92e-08 → 1.83e-09 | 4 | 4.00 | > 3.85 | pass |
| SSPRK3 | – | error, time-dependent right-hand side | 10 → 20 | 2.92e-08 → 1.83e-09 | 3 | 4.00 | > 2.85 | pass |
| SSPRK54 | – | error, time-dependent right-hand side | 10 → 20 | 1.69e-08 → 1.06e-09 | 4 | 4.00 | > 3.85 | pass |
| RDPK3Sp35 error estimate | – | estimate after one step | 10 → 20 | 2.67e-06 → 3.41e-07 | 3 | 2.97 | 2.9 to 3.1 | pass |
| RODAS error estimate | – | local error estimate, Van der Pol oscillator | 32 → 64 | 5.98e-07 → 4.07e-08 | 4 | 3.88 | 3.8 to 4.3 | pass |
| CK45 | – | max error, linear system | 20 → 40 | 9.56e-08 → 5.81e-09 | 4 | 4.04 | 3.9 to 4.1 | pass |
| FE | – | max error, linear system | 20 → 40 | 1.38e-02 → 6.82e-03 | 1 | 1.01 | 0.9 to 1.1 | pass |
| RDPK3Sp35 | – | max error, linear system | 20 → 40 | 2.81e-06 → 3.43e-07 | 3 | 3.03 | 2.9 to 3.1 | pass |
| RK4 | – | max error, linear system | 20 → 40 | 2.45e-07 → 1.47e-08 | 4 | 4.06 | 3.9 to 4.1 | pass |
| SSPRK3 | – | max error, linear system | 20 → 40 | 1.22e-05 → 1.47e-06 | 3 | 3.06 | 2.9 to 3.1 | pass |
| SSPRK54 | – | max error, linear system | 20 → 40 | 1.11e-07 → 6.73e-09 | 4 | 4.05 | 3.9 to 4.1 | pass |
<!-- END GENERATED table-time -->

## Boundary conditions

The TENO face values next to symmetry walls, where the stencils include the
mirror images of the cells, and the 3D forced channel of the previous section
with transmissive sides (2 x 2 cells) in place of periodic ones. On
tetrahedra and pyramids the transmissive sides drop the channel to first
order ([#420](https://github.com/MatthewBonanni/mallard/issues/420)); with
periodic sides the same cells converge at second order. The viscous studies
also verify isothermal no-slip and moving walls; the axisymmetric studies
below, the axis.

<!-- BEGIN GENERATED table-boundaries -->
| Method | Cells | Error | Resolutions | Errors | Design | Observed | Asserted | Result |
|---|---|---|---|---|---|---|---|---|
| TENO3 | quadrilaterals | max face-value error, symmetry walls | 32 → 64 | 3.65e-04 → 4.60e-05 | 3 | 2.99 | > 2.6 | pass |
| TENO4 | quadrilaterals | max face-value error, symmetry walls | 32 → 64 | 2.69e-05 → 1.73e-06 | 4 | 3.96 | > 3.6 | pass |
| TENO5 | quadrilaterals | max face-value error, symmetry walls | 32 → 64 | 6.57e-06 → 2.08e-07 | 5 | 4.98 | > 4.6 | pass |
| TENO6 | quadrilaterals | max face-value error, symmetry walls | 32 → 64 | 7.36e-07 → 1.19e-08 | 6 | 5.96 | > 5.6 | pass |
| TENO3 | triangles | max face-value error, symmetry walls | 32 → 64 | 1.28e-04 → 1.62e-05 | 3 | 2.98 | > 2.6 | pass |
| TENO4 | triangles | max face-value error, symmetry walls | 32 → 64 | 1.22e-05 → 7.74e-07 | 4 | 3.97 | > 3.6 | pass |
| TENO5 | triangles | max face-value error, symmetry walls | 32 → 64 | 2.09e-06 → 6.69e-08 | 5 | 4.96 | > 4.6 | pass |
| TENO6 | triangles | max face-value error, symmetry walls | 32 → 64 | 1.26e-07 → 2.02e-09 | 6 | 5.97 | > 5.6 | pass |
| Forced channel, transmissive sides | hexahedra | max velocity error | 8 → 16 | 5.49e-04 → 1.42e-04 | 2 | 1.95 | > 1.8 | pass |
| Forced channel, transmissive sides | tetrahedra | max velocity error | 8 → 16 | 2.18e-04 → 7.26e-05 | 2 | 1.59 | > 1.8 | **fail**, [#420](https://github.com/MatthewBonanni/mallard/issues/420) |
| Forced channel, transmissive sides | prisms | max velocity error | 8 → 16 | 2.41e-04 → 6.09e-05 | 2 | 1.98 | > 1.8 | pass |
| Forced channel, transmissive sides | pyramids | max velocity error | 8 → 16 | 9.05e-03 → 4.60e-03 | 2 | 0.98 | > 1.8 | **fail**, [#420](https://github.com/MatthewBonanni/mallard/issues/420) |
| Forced channel, transmissive sides | mixed | max velocity error | 8 → 16 | 1.00e-03 → 2.03e-04 | 2 | 2.30 | > 1.8 | pass |
<!-- END GENERATED table-boundaries -->

## Axisymmetric flows

The spatial operator at the exact $r$-weighted averages of a manufactured
state, up to and including the cells on the axis (a residual, one order below
the reconstruction's), and the viscous residual of Poiseuille pipe flow.

<!-- BEGIN GENERATED table-axisymmetric -->
| Method | Cells | Error | Resolutions | Errors | Design | Observed | Asserted | Result |
|---|---|---|---|---|---|---|---|---|
| Poiseuille pipe flow | quadrilaterals | mean axial momentum residual | 8 → 16 | 6.10e-07 → 1.28e-07 | 2 | 2.26 | > 1.8 | pass |
| TENO3 | quadrilaterals | mean residual error, manufactured solution | 16 → 32 | 2.92e-02 → 3.42e-03 | 2 | 3.09 | > 1.7 | pass |
| TENO4 | quadrilaterals | mean residual error, manufactured solution | 16 → 32 | 5.40e-03 → 2.45e-04 | 3 | 4.46 | > 2.7 | pass |
| TENO5 | quadrilaterals | mean residual error, manufactured solution | 16 → 32 | 5.06e-03 → 1.22e-04 | 4 | 5.37 | > 3.7 | pass |
| TENO3 | triangles | mean residual error, manufactured solution | 16 → 32 | 2.33e-02 → 4.83e-03 | 2 | 2.27 | > 1.7 | pass |
| TENO4 | triangles | mean residual error, manufactured solution | 16 → 32 | 7.91e-03 → 7.38e-04 | 3 | 3.42 | > 2.7 | pass |
| TENO5 | triangles | mean residual error, manufactured solution | 16 → 32 | 8.73e-03 → 5.94e-05 | 4 | 7.20 | > 3.7 | pass |
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
| Method | Cells | Error | Resolutions | Errors | Design | Observed | Asserted | Result |
|---|---|---|---|---|---|---|---|---|
| P2 walls | quadrilaterals | annulus area error | 16 → 32 | 1.42e-04 → 8.90e-06 | 4 | 4.00 | > 3.8 | pass |
| P3 walls | quadrilaterals | annulus area error | 16 → 32 | 2.09e-05 → 1.32e-06 | 4 | 3.99 | > 3.8 | pass |
| straight walls | quadrilaterals | annulus area error | 32 → 64 | 1.84e-02 → 4.62e-03 | 2 | 2.00 | 1.95 to 2.05 | pass |
| TENO3, curved walls | quadrilaterals | max wall-value error | 4 → 8 | 3.27e-03 → 4.34e-04 | 3 | 2.91 | > 2.5 | pass |
| TENO3, straight-sided cells | quadrilaterals | max wall-value error | 4 → 8 | 2.09e-02 → 9.35e-03 | 2 | 1.16 | < 2.3 | pass |
| TENO4, curved walls | quadrilaterals | max wall-value error | 4 → 8 | 1.37e-03 → 7.22e-05 | 4 | 4.24 | > 3.5 | pass |
| TENO4, straight-sided cells | quadrilaterals | max wall-value error | 4 → 8 | 2.30e-02 → 1.26e-02 | 2 | 0.87 | < 2.3 | pass |
| TENO5, curved walls | quadrilaterals | max wall-value error | 4 → 8 | 7.00e-04 → 9.72e-06 | 5 | 6.17 | > 4.5 | pass |
| TENO5, curved walls, wall degree 3 | quadrilaterals | max wall-value error | 4 → 8 | 1.42e-03 → 7.52e-05 | 4 | 4.23 | 3.5 to 4.5 | pass |
| TENO5, curved walls, wall degree 4 | quadrilaterals | max wall-value error | 4 → 8 | 7.00e-04 → 9.72e-06 | 5 | 6.17 | > 4.5 | pass |
| TENO5, straight-sided cells | quadrilaterals | max wall-value error | 4 → 8 | 4.90e-02 → 1.82e-02 | 2 | 1.43 | < 2.3 | pass |
| TENO6, curved walls | quadrilaterals | max wall-value error | 4 → 8 | 2.13e-03 → 1.43e-05 | 6 | 7.22 | > 5.5 | pass |
| TENO6, straight-sided cells | quadrilaterals | max wall-value error | 4 → 8 | 5.42e-02 → 1.25e-02 | 2 | 2.12 | < 2.3 | pass |
| TENO3, curved walls | triangles | max wall-value error | 4 → 8 | 2.47e-03 → 3.30e-04 | 3 | 2.91 | > 2.5 | pass |
| TENO3, straight-sided cells | triangles | max wall-value error | 4 → 8 | 3.25e-02 → 1.50e-02 | 2 | 1.11 | < 2.3 | pass |
| TENO4, curved walls | triangles | max wall-value error | 4 → 8 | 5.68e-04 → 4.27e-05 | 4 | 3.73 | > 3.5 | pass |
| TENO4, straight-sided cells | triangles | max wall-value error | 4 → 8 | 3.39e-02 → 1.62e-02 | 2 | 1.06 | < 2.3 | pass |
| TENO5, curved walls | triangles | max wall-value error | 4 → 8 | 1.91e-04 → 7.51e-06 | 5 | 4.67 | > 4.5 | pass |
| TENO5, straight-sided cells | triangles | max wall-value error | 4 → 8 | 3.42e-02 → 1.63e-02 | 2 | 1.07 | < 2.3 | pass |
| TENO6, curved walls | triangles | max wall-value error | 4 → 8 | 2.81e-04 → 1.40e-06 | 6 | 7.65 | > 5.5 | pass |
| TENO6, straight-sided cells | triangles | max wall-value error | 4 → 8 | 4.62e-02 → 2.35e-02 | 2 | 0.97 | < 2.3 | pass |
| quadratic walls | hexahedra | spherical shell volume error | 4 → 8 | 9.63e-03 → 6.07e-04 | 4 | 3.99 | > 3.7 | pass |
| straight walls | hexahedra | spherical shell volume error | 4 → 8 | 6.48e-01 → 1.66e-01 | 2 | 1.96 | 1.85 to 2.15 | pass |
| quadratic walls | prisms | spherical shell volume error | 4 → 8 | 4.63e-03 → 2.91e-04 | 4 | 3.99 | > 3.7 | pass |
| straight walls | prisms | spherical shell volume error | 4 → 8 | 6.48e-01 → 1.66e-01 | 2 | 1.96 | 1.85 to 2.15 | pass |
<!-- END GENERATED table-curved -->

## Remapping

Conservative remapping of a smooth periodic field between unrelated meshes,
by the overlap integrals of piecewise polynomials of each degree, unlimited:
the target cell averages converge at the degree plus one.

![Remap convergence](../images/verification/remap-light.svg#gh-light-mode-only)
![Remap convergence](../images/verification/remap-dark.svg#gh-dark-mode-only)

<!-- BEGIN GENERATED table-remap -->
| Method | Cells | Error | Resolutions | Errors | Design | Observed | Asserted | Result |
|---|---|---|---|---|---|---|---|---|
| degree 1 | quadrilaterals | max error of the target averages | 10 → 20 | 5.56e-02 → 6.97e-03 | 2 | 3.00 | > 1.6 | pass |
| degree 2 | quadrilaterals | max error of the target averages | 10 → 20 | 3.95e-02 → 4.30e-03 | 3 | 3.20 | > 2.6 | pass |
| degree 3 | quadrilaterals | max error of the target averages | 10 → 20 | 1.73e-02 → 6.53e-04 | 4 | 4.73 | > 3.6 | pass |
| degree 1 | hexahedra | max error of the target averages | 5 → 10 | 1.98e-01 → 2.04e-02 | 2 | 3.27 | > 1.6 | pass |
| degree 2 | hexahedra | max error of the target averages | 5 → 10 | 1.85e-01 → 1.64e-02 | 3 | 3.50 | > 2.6 | pass |
<!-- END GENERATED table-remap -->

## Gas mixtures

A smooth wave of H<sub>2</sub> in N<sub>2</sub> and of temperature advected
at uniform velocity and pressure, thermally perfect: both the flow variables
and the species reach the scheme's order.

<!-- BEGIN GENERATED table-mixtures -->
| Method | Cells | Error | Resolutions | Errors | Design | Observed | Asserted | Result |
|---|---|---|---|---|---|---|---|---|
| MUSCL | quadrilaterals | L1 H2 partial density error | 16 → 32 | 8.15e-06 → 1.86e-06 | 2 | 2.13 | > 1.8 | pass |
| TENO3 | quadrilaterals | L1 H2 partial density error | 32 → 64 | 1.91e-06 → 2.80e-07 | 3 | 2.77 | > 2.6 | pass |
| TENO5 | triangles | L1 H2 partial density error | 16 → 32 | 5.37e-05 → 4.27e-07 | 5 | 6.98 | > 4 | pass |
| MUSCL | hexahedra | L1 H2 partial density error | 8 → 16 | 3.54e-05 → 8.15e-06 | 2 | 2.12 | > 1.8 | pass |
| TENO3 | hexahedra | L1 H2 partial density error | 8 → 16 | 1.27e-04 → 1.14e-05 | 3 | 3.47 | > 2.4 | pass |
| MUSCL | quadrilaterals | L1 density error | 16 → 32 | 2.63e-04 → 6.44e-05 | 2 | 2.03 | > 1.8 | pass |
| TENO3 | quadrilaterals | L1 density error | 32 → 64 | 5.51e-05 → 7.02e-06 | 3 | 2.97 | > 2.6 | pass |
| TENO5 | triangles | L1 density error | 16 → 32 | 7.28e-05 → 2.55e-06 | 5 | 4.83 | > 4 | pass |
| MUSCL | hexahedra | L1 density error | 8 → 16 | 1.17e-03 → 2.63e-04 | 2 | 2.15 | > 1.8 | pass |
| TENO3 | hexahedra | L1 density error | 8 → 16 | 1.50e-03 → 2.49e-04 | 3 | 2.60 | > 2.4 | pass |
<!-- END GENERATED table-mixtures -->

## Not covered

- The reacting source terms in space (chemistry is verified as an ODE
  integrator, above, and against Cantera in the
  [validation cases](https://matthewbonanni.github.io/mallard/validation/)).
- Mixture-averaged transport, LES models and characteristic boundaries,
  which are tested for exactness and consistency, not for an order.
- Discontinuous solutions, whose errors converge at order 1 or below
  regardless of the scheme; see the validation cases.
