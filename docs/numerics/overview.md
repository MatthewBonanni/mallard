# Numerical methods

Mallard solves the compressible Euler or Navier-Stokes equations, for a single
gas or a reacting mixture, with a cell-centered finite volume method on
unstructured meshes: triangles and quadrilaterals in 2D (planar or
axisymmetric), tetrahedra, hexahedra, prisms and pyramids in 3D. This page
describes the methods and measures them; the [theory manual](../theory/index.md)
gives every equation as implemented, the [verification manual](../verification/index.md)
the observed orders of accuracy of each scheme on each cell type, and the
[numerics guide](../guide/numerics.md) is about choosing among them.

## Discretization

For each cell $i$ of volume $V_i$,

$$
\begin{equation}
\frac{\mathrm{d} U_i}{\mathrm{d} t} = -\frac{1}{V_i} \sum_{f \in \partial i} A_f \sum_{q} w_q\,
F\big(U_L(x_q), U_R(x_q)\big) \cdot n_f + \text{viscous and source terms},
\label{eq:semi-discrete}
\end{equation}
$$

where the inner sum runs over the quadrature points $x_q$ (weights $w_q$) of face $f$.
Each face is integrated with Gauss-Legendre points (one for first order and
MUSCL, `floor((order + 1) / 2)` per edge for TENO). At each point the convective flux is
an approximate Riemann solver applied to the reconstructed left and right
states. Each face's convective and viscous fluxes are stored once and every
cell sums its faces in a fixed order, without atomics, so results are bitwise
independent of the thread count and scheduling.

## Time step

Time integration is explicit (SSPRK3 by default; [Shu & Osher 1988](../references.md#shu-osher-1988)). The time step comes
from per-cell spectral radii ([Blazek 2015](../references.md#blazek-2015), eqs. 6.20-6.21 with C = 4):

$$
\begin{align}
\Delta t &= \mathrm{cfl} \, \min_i \Delta t_i, \label{eq:dt} \\
\Delta t_i &= \frac{2 V_i}{\sum_f \left(|u_n| + a\right) A_f + 4 \nu_\mathrm{eff} \sum_f A_f^2 / V_i}, \label{eq:dt-cell}
\end{align}
$$

where $|u_n| + a$ is the larger of the two cells' on each face and
$\nu_\mathrm{eff} = \max(4/3, \gamma/\mathrm{Pr})\, \mu / \rho$ (for gas mixtures, the largest of
$\tfrac{4}{3} \mu / \rho$, $\lambda / (\rho c_v)$ and the species' diffusion coefficients;
axisymmetric viscous runs add $\nu_\mathrm{eff} V_i / r^2$ to the denominator of $\eqref{eq:dt-cell}$).
Half the sum over faces stands for the sum over coordinate directions of the
projected areas, so this is the usual unstructured CFL number: on a uniform
grid of spacing h it gives `dt = cfl * h / (|u| + |v| + 2a)` in 2D and
`cfl * h / (|u| + |v| + |w| + 3a)` in 3D. (Mallard 0.5 and earlier omitted the
factor 2: their inputs give the same time step with half their `cfl`.)

Largest stable `cfl` for smooth inviscid flow, SSPRK3 / RK4, with HLLC and
default options, on periodic boxes of generated meshes (40 x 40 quads or
twice as many triangles; 20 x 20 x 6 blocks in 3D, one hexahedron, six
tetrahedra or two prisms each): the largest `cfl`, bisected to 0.02, that
still damps a 1% Gaussian acoustic pulse in gas at rest over 12 acoustic
crossings of the box (500-1000 steps at the limit; the growth past it is
abrupt), and keeps the isentropic vortex, convected diagonally at Mach 1.2,
within its initial deviation over ten periods. The pulse sets every limit;
the vortex tolerates 15-45% more.

| Reconstruction | Quads | Triangles | Hexahedra | Tetrahedra | Prisms |
|---|---|---|---|---|---|
| First order | 1.26 / 1.41 | 1.29 / 1.39 | 1.27 / 1.41 | 1.26 / 1.39 | 1.27 / 1.39 |
| MUSCL | 1.26 / 1.41 | 1.33 / 1.49 | 1.27 / 1.41 | 1.26 / 1.39 | 1.33 / 1.47 |
| TENO 3 | 1.33 / 1.43 | 1.36 / 1.52 | 1.32 / 1.47 | 1.49 / 1.67 | 1.35 / 1.49 |
| TENO 4 | 1.32 / 1.47 | 1.32 / 1.47 | 1.33 / 1.47 | 1.32 / 1.46 | 1.33 / 1.46 |
| TENO 5 | 1.38 / 1.53 | 1.47 / 1.59 | 1.47 / 1.64 | 1.55 / 1.71 | 1.43 / 1.58 |
| TENO 6 | 1.33 / 1.46 | 1.44 / 1.59 | 1.33 / 1.47 | 1.36 / 1.50 | 1.32 / 1.47 |

The first-order limits are the integrator's own: on quads and hexahedra the
odd-even (checkerboard) acoustic mode has the eigenvalue `-2 cfl` in this
normalization, so it leaves the stability region on the negative real axis at
`cfl` = 2.51 / 2 = 1.26 (SSPRK3) and 2.79 / 2 = 1.39 (RK4), and forward Euler
stops at 1; the other cells come within 3% of that. Higher reconstruction
orders are stable to about the same `cfl`, up to 25% more. The low-Mach correction
does not change the limits (within 0.03 with `low_mach_cutoff = 1`), and the
viscous term never binds first: acoustic pulses at cell Reynolds numbers below
one (first order and TENO5) are stable to `cfl` = 2.5 on quads and 3.2 on
triangles.

Accuracy: the vortex's density error after one period on the 40 x 40 grid
stays within 10% of its small-`cfl` value up to

| Reconstruction | Quads | Triangles |
|---|---|---|
| First order | 1.51 / 1.64 | 1.60 / 1.76 |
| MUSCL | 1.55 / 1.82 | 1.66 / 1.90 |
| TENO 3 | 1.60 / 1.77 | 1.93 / 2.11 |
| TENO 4 | 1.63 / 1.83 | 1.45 / 2.03 |
| TENO 5 | 1.76 / 2.00 | 1.84 / 2.18 |
| TENO 6 | 1.50 / 1.96 | 1.40 / 2.17 |

above the stability limits, as in 3D (1.6-2.6 on hexahedra and tetrahedra, up
to TENO5). Errors that are smaller relative to the time error (finer meshes,
higher orders) lower these values, TENO4 and TENO6 with SSPRK3 first.

So `cfl` = 1.2 (SSPRK3) or 1.35 (RK4) is the largest safe value on any of
these cell types for smooth flow; distorted or stretched cells and strong
shocks call for some margin, and 1.0 with SSPRK3 is a good default. Shocks do
not lower the limit much: on the 200-cell Sod and Shu-Osher problems (TENO5,
SSPRK3) the density L1 error is unchanged (within 1%) from `cfl` = 0.25 to 1.0,
Shu-Osher's then grows by 4% at 1.2 and 11% at 1.4, and the double Mach
reflection (on 240 x 60 blocks) runs at 1.2. Bound-preserving TENO keeps
density and pressure positive only at smaller steps in the most extreme
cases: the Noh problem (`examples/noh_axisymmetric`) needs `cfl` = 0.25 and
goes to negative pressure at 0.5, while the Sedov blasts run at 1.0. With chemistry the splitting
error can bind first: the 1D CJ detonation of `examples/detonation_1d` keeps
its ZND induction length within 5% up to `cfl` = 0.35 but not at 0.5 (-5.5%) or
1.0 (-25%), while the premixed flame's speed is the same to 0.001% at 0.2 and
1.0. SIMPLER balanced splitting (`[chemistry] coupling = "simpler"`) keeps the
mean induction length within 5% at `cfl` = 1.0 (-0.9% and +3.2% at 10 and
20 cells per induction length), with a front that oscillates more from output
to output
([chemistry.md](../design/chemistry.md#simpler-balanced-splitting-option)).

## Time integrators

| `time_integrator` | Method | Order | Stages | Registers (state-sized) |
|---|---|---|---|---|
| `FE` | Forward Euler | 1 | 1 | 2 |
| `SSPRK3` | [Shu & Osher 1988](../references.md#shu-osher-1988) | 3 | 3 | 3 |
| `RK4` | Classical | 4 | 4 | 6 |
| `SSPRK54` | [Kraaijevanger 1991](../references.md#kraaijevanger-1991); [Spiteri & Ruuth 2002](../references.md#spiteri-ruuth-2002), SSP coefficient 1.508 | 4 | 5 | 4 |
| `CK45` | [Carpenter & Kennedy 1994](../references.md#carpenter-kennedy-1994), 2N storage | 4 | 5 | 3 |
| `RDPK3Sp35` | [Ranocha et al. 2022](../references.md#ranocha-2022), embedded 2nd order | 3 | 5 | 5 |

`SSPRK54` and `RDPK3Sp35` share one implementation of Ketcheson's 3S* form
([Ketcheson 2008](../references.md#ketcheson-2008), [2010](../references.md#ketcheson-2010)),
written for the deviations from U<sup>n</sup>:
`S += delta_i (Y_{i-1} - U^n)`, `Y_i = U^n + gamma1_i (Y_{i-1} - U^n) + gamma2_i S + beta_i dt L(Y_{i-1})`.
A steady state then stays bitwise steady (rounded weights summing to 1 - 2^-54
are what changed SSPRK3's carbuncle behavior in 0.7, [below](#stationary-shocks-on-cell-faces)),
and one fused kernel per stage updates every register. For SSPRK54 the
coefficients come from Spiteri and Ruuth's Shu–Osher form re-solved to 50
digits (the fourth-order conditions hold to 1e-50, and their 15-digit values
agree to 1e-15); its equal ratios alpha/beta = 1.508 make delta_4 and gamma3
vanish, so U<sup>n</sup>, the solution and `S = alpha_52 (Y_2 - U^n)` suffice.
`CK45` is the 2N form `D = A_i D + dt L(U)`, `U += B_i D`, with the RHS kept
apart from D. None changes the existing integrators: restart files of FE,
SSPRK3 and RK4 runs are byte-identical to those of the previous main. All run inside the step's
CUDA graph (dt read on the device), with SIMPLER and Strang splitting, and
take bitwise the same steps on any number of ranks.

**Stability limits** (the setup of the [table above](#time-step): 1% acoustic
pulse over 12 crossings and the isentropic vortex over ten periods, on 40 x 40
quads or twice as many triangles, HLLC; SSPRK54 / CK45 / RDPK3Sp35):

| Reconstruction | Pulse, quads | Pulse, triangles | Vortex, quads | Vortex, triangles |
|---|---|---|---|---|
| First order | 2.69 / 2.40 / 2.50 | 2.79 / 2.36 / 2.55 | 2.83 / 2.82 / 3.36 | 3.28 / 3.17 / 3.59 |
| MUSCL | 2.69 / 2.40 / 2.50 | 2.77 / 2.57 / 2.69 | **2.31 / 2.36** / 2.86 | 2.65 / 2.69 / 3.31 |
| TENO 3 | 2.73 / 2.41 / 2.58 | 2.90 / 2.58 / 2.95 | 3.44 / 3.01 / 3.22 | > 3.6 |
| TENO 4 | 2.81 / 2.45 / 2.61 | 2.85 / 2.50 / 2.66 | 3.12 / 3.17 / 3.48 | 3.20 / 3.16 / > 3.6 |
| TENO 5 | 2.94 / 2.57 / 2.73 | 3.07 / 2.79 / 2.84 | 3.24 / 3.16 / 3.56 | 3.56 / 3.28 / > 3.6 |
| TENO 6 | 2.81 / 2.45 / 2.66 | 3.05 / 2.67 / 2.83 | 3.10 / 2.93 / 3.58 | 3.56 / 3.24 / > 3.6 |

First order matches the stability polynomials' real-axis extents: 5.33 / 4.66
/ 4.93, i.e. `cfl` 2.67 / 2.33 / 2.46 for the checkerboard mode (the pulse
grows slowly just past them, so the measured values sit up to 3% above).
Unlike SSPRK3 and RK4, the vortex binds first with MUSCL (its limiter is
active on the vortex). The vortex's error after one period stays within 10%
of its small-`cfl` value up to at least the pulse limit, except TENO6 with
RDPK3Sp35 (2.40 quads, 2.27 triangles) and TENO4 on triangles with RDPK3Sp35
(2.34), and TENO6 on triangles with SSPRK54 (2.96). The safe value on these cells is 2.2 for all three,
against 1.2 (SSPRK3) and 1.35 (RK4).

**Temporal convergence** (vortex, 40 x 40 quads, TENO5, to t = 2; maximum
density difference from RK4 at `cfl` 0.02 on the same mesh):

| `cfl` | SSPRK3 | RK4 | SSPRK54 | CK45 | RDPK3Sp35 |
|---|---|---|---|---|---|
| 1.0 | 1.03e-4 | 4.59e-6 | 2.27e-6 | 1.76e-6 | 2.21e-5 |
| 0.5 | 1.23e-5 | 2.95e-7 | 1.47e-7 | 1.06e-7 | 2.75e-6 |
| 0.25 | 1.51e-6 | 2.50e-8 | 1.56e-8 | 9.02e-9 | 3.43e-7 |
| Order (1 to 0.5) | 3.06 | 3.96 | 3.95 | 4.05 | 3.01 |

(Below about 1e-8 the fourth-order methods flatten out at TENO's level of
round-off in the nonlinear weights.)

**Cost per unit simulated time** on one A100 (2D Riemann problem on 1024 x
1024 quads, 300 steps, CUDA graphs): the time per step, and that time divided
by each method's smallest limit above (SSPRK3 / RK4 from the
[table above](#time-step)), relative to SSPRK3:

| | SSPRK3 | RK4 | SSPRK54 | CK45 | RDPK3Sp35 |
|---|---|---|---|---|---|
| MUSCL, ms per step | 3.62 | 4.78 | 5.70 | 5.69 | 5.96 |
| MUSCL, per unit time | 1 | 1.18 | 0.86 | 0.84 | 0.83 |
| TENO5, ms per step | 19.6 | 26.0 | 32.4 | 32.3 | 32.5 |
| TENO5, per unit time | 1 | 1.20 | 0.78 | 0.89 | 0.84 |

A stage of the low-storage methods costs about the same as one of SSPRK3's
(MUSCL: 1.14 ms against 1.21 ms). With shocks the SSP coefficient is what
counts: run to t = 0.3 with MUSCL, SSPRK54 at `cfl` 1.5 (966 steps, 5.4 s)
gives a density 3.2e-4 (relative L1) from SSPRK3 at 0.25, like SSPRK3 at 1.0
and 1.2 (2.5e-4 and 5.2e-4; 1449 and 1208 steps, 5.3 s and 4.4 s); SSPRK54 at
2.5 takes 591 steps (3.4 s) for 2.6e-3, and CK45 at 2.2 659 steps (3.8 s) for
6.7e-4.

### Error-controlled steps

`[numerics.error_control]` with `RDPK3Sp35` sizes the steps by the embedded
error estimate, as in [Ranocha et al. 2022](../references.md#ranocha-2022):
with `w` the largest over owned cells and conservative variables of
`|e| / (abs_tol + rel_tol max(|U^n|, |U^{n+1}|))` and `eps = 1 / w`,
`dt_{n+1} = kappa(eps_{n+1}^{b1/k} eps_n^{b2/k} eps_{n-1}^{b3/k}) dt_n`
(`k` = 3, `beta` = (0.64, -0.31, 0.04)), with the limiter
`kappa(x) = 1 + atan(x - 1)` of [Söderlind & Wang 2006](../references.md#soderlind-wang-2006);
a step is accepted if `kappa >= 0.81` and otherwise restored to U<sup>n</sup>
and retried. The maximum norm, unlike Ranocha et al.'s weighted L2 norm, is
independent of the rank and thread counts, so steps stay bitwise
reproducible. The controller runs on the device in both the captured step and
the kernel-by-kernel path (a rejected graph is simply relaunched); restart
files carry its step proposal and error history, so restarts are exact. It
costs one reduction per step: on the Riemann problem above, 3% more per
step than RDPK3Sp35 at a fixed `cfl` with MUSCL, and none measurable with
TENO5.

Tolerance 1e-4 on the isentropic vortex (40 x 40 quads, MUSCL) runs at
`cfl` about 3 with no rejections, a third of SSPRK3's steps at 1.0. A first
step at `cfl` 20 is rejected and retried until the error fits. On the
Riemann problem the shocks keep the error high: 1861 steps to t = 0.3 (5
rejected), an average `cfl` of 0.78, at an error of 1.3e-4 from the
reference above.

Not available with `[chemistry]` (a rejection would have to undo the
chemistry substeps) or `double_flux`.

**Relaxation Runge–Kutta** ([Ranocha et al. 2020](../references.md#ranocha-2020)),
which rescales each step so that the total entropy changes as the spatial
discretization prescribes, is deferred until Mallard has entropy-stable
fluxes (#288): with the current upwind fluxes there is no entropy estimate
for it to preserve.

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
| RHLL | [Nishikawa & Kitamura 2008](../references.md#nishikawa-kitamura-2008), a rotated hybrid: HLL along the velocity-difference direction, Roe across it. Blended toward HLL along the face normal by the pressure jump across the face, with the rotated flux's weight min(p_l / p_r, p_r / p_l)^3 (the pressure weight of AUSMPW+, [Kim, Kim & Rho 2001](../references.md#kim-kim-rho-2001)): on tetrahedra the rotation otherwise turns HLL toward the shock normal on every face that crosses a shock, which grows a carbuncle on the stagnation line of blunt bodies with MUSCL or TENO (#80). Contacts and shear layers carry no pressure jump and keep the rotated flux. Carbuncle-free, except at a shock at rest on cell faces ([below](#stationary-shocks-on-cell-faces)). |

All five also solve gas mixtures, on each side's frozen `cp / cv` and energy
offset ([chemistry design](../design/chemistry.md#riemann-solvers)). The
mixture Roe solver ([Glaister 1988](../references.md#glaister-1988);
[Shuen, Liou & van Leer 1990](../references.md#shuen-liou-van-leer-1990)) lets the
contact, shear and composition waves, which all move with the flow, carry
the whole jump of the conservative variables left by the two acoustic waves.
That makes it exact at contacts between different gases for any averaged
sound speed, and it reduces to the single-gas Roe solver for one gas.

### Stationary shocks on cell faces

The Roe average of the two sides of a normal shock at rest is exactly sonic.
On a face that holds such a shock, the Einfeldt left wave speed
`min(u_L - a_L, u_Roe - a_Roe)` is therefore zero up to round-off, and its
sign selects the flux:

- If it is positive or zero, HLL is the upwind flux `F_L`.
- If it is negative, HLL averages the two sides and passes the post-shock
  perturbations into the upstream cells, with a gain of the full shock jump.

The first case is stable and the second is not. So whether HLL, and RHLL
(which uses HLL across the shock), stay free of the carbuncle on such a shock
depends only on how the time integrator rounds:

- From 3000 to 10000 steps of a Mach 6 shock at rest on the faces of a
  40 x 40 grid (the setup of `carbuncle_growth` in `test/solver_test.cpp`
  with no drift), both grow an O(0.1-1) carbuncle with FE, RK4 and SSPRK3.
  This holds with or without FMA contraction and at CFL 0.18-0.3.
- Until 0.7.0 the SSPRK3 last stage was `2/3 U2 + 1/3 U`. Its rounded weights
  sum to 1 - 2^-54, which nudged the state down by an ulp in a fraction of
  the cells every step and happened to keep that wave speed positive.

A shock that moves, or that sits inside a cell, never puts the sonic speed on
a face. For example, the same shock drifting at 0.03, as in
`SolverValidation.RotatedHybridRiemannSolverIsCarbuncleFree`:

- RHLL keeps the transverse velocity below 1e-5 for 10000 steps.
- HLL damps it.
- Roe grows it to O(1).

RHLL is marginal on slower shocks: drifting at 0.003 to 0.03 in either
direction, its transverse velocity reaches up to 1e-2 within 20000 steps.

This is a known limitation. None of the following wave-speed changes was
worth adopting (#244); all were measured on the setup above:

- **Davis bounds**, `S_l = min(u_L - a_L, u_Roe - a_Roe, u_R - a_R)` and
  `S_r = max(u_R + a_R, u_Roe + a_Roe, u_L + a_L)`:
  - They remove the sign dependence. At rest on faces, HLL and RHLL decay to
    1e-11-1e-14 by 10000 steps with FE, RK4 and SSPRK3, for a density bump or
    a dip behind the shock.
  - They spread every slow, strong shock over two cells instead of one. The
    L1 density error at the shock grows 1.7-1.9 times for the Mach 6 shock
    moving at ±0.5 (first order), and 2 times at rest (MUSCL). On weaker or
    faster shocks the cost is small: L1 errors grow by 0.1-0.4% for Sod,
    0.1-0.6% for Shu-Osher, and 0.6-1.4% for the double Mach reflection.
  - In RHLL the smeared shock grows a carbuncle: 2e-2 to 7e-2 for drifts of
    0.003 to 0.03, against 1e-5 for the drift of 0.03 with Einfeldt speeds.
    RHLL would fail its carbuncle test.
  - HLLC grows a carbuncle on these shocks with either bounds.
- **Snapping a near-zero `S_l` to 0** (a deterministic tie-break, here
  `|S_l| < 1e-10 a_Roe`): it keeps the shock at rest at 1e-14 when a density
  bump behind it pushes `S_l` up. A dip of the same size pushes `S_l` down
  instead, and the carbuncle grows to 0.2.
- **Smoothing `min(S_l, 0)` near zero** (a sonic entropy fix of width 1e-3 to
  0.3 `a_Roe`): the shock at rest still grows a carbuncle, to 0.08-0.25 with
  RHLL and up to 0.2 with HLL.

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

The remaining upwind dissipation still matters for wall-bounded turbulence
at DNS resolution. In the Re_tau = 180 channel (`examples/channel_retau180`;
dx+ = 12, dz+ = 6, MUSCL) HLLC removes about 7% of the kinetic-energy
dissipation and the wall shear comes out 7% low; Roe is the same, the
cutoff 0.01 gains 1%, and without the correction it is 23% low. Halving dx
and dz brings HLLC to within 1% of the reference; the `hybrid` convective
flux does so on the original mesh (0.9% numerical dissipation, wall shear 2%
high).

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
- **Characteristic boundaries** (`nscbc_outlet`, `nscbc_inlet`): outgoing waves come from the interior and the incoming acoustic wave from the face's own pressure and normal velocity, advanced each step by the LODI relations of [Poinsot & Lele (1992)](../references.md#poinsot-lele-1992) with relaxation toward a target and the transverse terms of [Lodato, Domingo & Vervisch (2008)](../references.md#lodato-2008); see `docs/design/nscbc.md`. An inlet's target velocity is per face and may carry synthetic turbulence: the digital filter of [Klein et al. (2003)](../references.md#klein-2003) with the Reynolds stress of [Lund et al. (1998)](../references.md#lund-1998), entering through the incoming wave as [Guézennec & Poinsot (2009)](../references.md#guezennec-poinsot-2009) propose; see `docs/design/synthetic_inflow.md`. MUSCL boundary cells reconstruct without the ghosts across them: the zero normal gradient of a ghost leaves a jump at the cell's inner face, which the low-Mach correction turns into a reflected acoustic wave (15 to 34% of a normal pulse through `extrapolation`, with MUSCL or TENO).

## Viscous fluxes

- Cell gradients come from a weighted least-squares quadratic fit ([Barth & Frederickson 1990](../references.md#barth-frederickson-1990)) over the vertex neighbors and the cell's boundary ghosts. A linear fit is only first-order accurate on one-sided boundary stencils and on triangles, where its errors cancel only in the interior of regular meshes.
- Face gradients of velocity and temperature average the two cell gradients. They are then corrected along the face normal so that their component along the line between the cell centroids matches the direct difference ([Diskin et al. 2010](../references.md#diskin-2010)).
- Transmissive faces take the face values and gradients of their image face, as the convective flux does.
- Stress follows the Stokes hypothesis; heat flux uses a constant Prandtl number. Viscosity is constant, follows Sutherland's law ([Sutherland 1893](../references.md#sutherland-1893)) or a power law.

## Axisymmetric flows

With `[physics] axisymmetric = true`, cells are integrated over their solids
of revolution about the x axis (per radian): r-weighted cell averages,
revolved volumes and face areas, and the radial momentum source
`int (p - tau_thetatheta) dA`, made high order from TENO's polynomials. TENO's
least squares fit r-weighted averages, and second-order operators work at
r-weighted centroids. See [the design notes](../design/axisymmetric.md).

## Known limitations

- The scheme is not exactly well balanced: hydrostatic states carry small spurious velocities, which vanish at second order under refinement (wall ghosts continue the hydrostatic pressure gradient).
- RHLL does not keep a 1D problem on a 2D mesh exactly 1D. Where the velocity jump across a face is small but above its fallback threshold, the rotation direction tilts by the round-off transverse velocity divided by that jump. On the Sod strip (800 x 4 quads, TENO5), the rows start out identical to 1e-16, still agree to 1e-12 at t = 0.04, and differ by up to about 1e-5 later on, at levels that depend on the time-step history. HLL and HLLC stay at round-off. In gas mixtures this costs the mass fractions' positivity with TENO5: on the multicomponent shock tube (4 rows) N2 reaches -2e-10 in three cells of the driver gas next to the diaphragm, where HLLC, Roe and MUSCL stay within [0, 1]. The species' bounds are enforced at face points only, which keeps them only for fluxes that do not draw a cell's species beyond its content.
- TENO's k-exact least squares with 2x oversampling is noticeably more dissipative for under-resolved smooth waves than compact structured stencils. `stencil_factor = 1.5` helps, at some cost in robustness at discontinuities.

All sources, with where Mallard uses them: [References](../references.md).
