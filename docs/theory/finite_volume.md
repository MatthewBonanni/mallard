# Finite-volume discretization and reconstruction

Mallard discretizes eq. (3) of [chapter 1](governing_equations.md#conservation-laws) with a cell-centered finite-volume method on
unstructured meshes: triangles and quadrilaterals in 2D, tetrahedra,
hexahedra, prisms and pyramids in 3D, with optionally curved boundary
faces. This chapter covers the semi-discrete equations, face quadrature,
curved geometry, the reconstruction of face states (first order, MUSCL,
TENO-E), the low-Mach correction of those states and the cell gradients of
the viscous terms. The numerical fluxes are in [chapter 7](fluxes.md) and
the time integration in [chapter 8](time_integration.md). Usage:
[Numerics](../guide/numerics.md); measured stability and accuracy limits:
[Numerical methods](../numerics/overview.md).

## Semi-discrete equations

The unknowns are the cell averages
$\overline{\mathbf{U}}_i = \lvert\Omega_i\rvert^{-1}\int_{\Omega_i}\mathbf{U}\,\mathrm{d}V$.
Integrating eq. (3) of [chapter 1](governing_equations.md#conservation-laws) over cell $i$,

$$
\begin{equation}
\frac{\mathrm{d}\overline{\mathbf{U}}_i}{\mathrm{d}t}
= -\frac{1}{\lvert\Omega_i\rvert}\sum_{f\in\partial\Omega_i} s_{i,f}\left(\mathbf{F}_f - \mathbf{F}^v_f\right) + \overline{\mathbf{S}}_i,
\label{eq:semi-discrete-equations-1}
\end{equation}
$$

where $s_{i,f} = +1$ if $i$ is the face's first cell (the normal
$\mathbf{n}_f$ points out of $i$) and $-1$ otherwise, and $\mathbf{F}_f$,
$\mathbf{F}^v_f$ are the convective and viscous fluxes integrated over the
face:

$$
\begin{equation}
\mathbf{F}_f = \frac{A_f}{2}\sum_{q} w_{f,q}\,\hat{\mathbf{F}}\left(\mathbf{U}^L_{f,q}, \mathbf{U}^R_{f,q}; \mathbf{n}_{f,q}\right),
\label{eq:semi-discrete-equations-2}
\end{equation}
$$

with face weights normalized to $\sum_q w_{f,q} = 2$ (Gauss-Legendre on
$[-1, 1]$ in 2D; 3D rules rescaled to that sum). The viscous flux uses one
point per face ([chapter 7](fluxes.md#viscous-fluxes)). Each face's flux is
computed and stored once, and every cell sums its faces in a fixed order
without atomic operations (`FaceFluxSumFunctor`,
`src/numerics/flux_functor.h`), so the result is bitwise independent of
thread count and scheduling. Boundary faces use the exterior state of the
boundary condition as $\mathbf{U}^R$
([chapter 9](boundaries_sources.md#weak-imposition)). The source term is
evaluated at the cell average.

## Face quadrature

On straight faces the quadrature points and weights are
(`src/numerics/quadrature.cpp`, `face_reconstruction.cpp`):

| Reconstruction | 2D edges | 3D triangles | 3D quadrilaterals |
|---|---|---|---|
| First order, MUSCL | 1 point (midpoint) | centroid | centroid |
| TENO-E, order 3 or 4 | Gauss-Legendre, 2 points | Dunavant rule of degree 4 | tensor Gauss-Legendre, 2 × 2 points, bilinear map |
| TENO-E, order 5 or 6 | Gauss-Legendre, 3 points | Dunavant rule of degree 5 | tensor Gauss-Legendre, 3 × 3 points, bilinear map |

so that the reconstructed polynomial, of degree one less than the order, is integrated exactly
over straight faces ([Dunavant 1985](../references.md#dunavant-1985)). Each
face has one unit normal $\mathbf{n}_f$, except curved faces, which have one
per point.

## Curved boundaries

A straight cell next to a curved wall misplaces it by $O(h^2/R)$, which
caps the order of the scheme. With high-order Gmsh boundary elements or an
analytic shape on a boundary zone (`[[mesh.curved]]`: circle, sphere or
cylinder), boundary faces follow the curved geometry
(`src/mesh/curved.cpp`; [design notes](../design/curved_boundaries.md)).

**Face maps.** A curved face is a map $\mathbf{x}(\boldsymbol{\xi})$ from its
flat reference element. Analytic shapes project a point of the straight face
radially onto the shape, minus the linear interpolant of the corners'
projection offsets, so that faces still meet at the mesh nodes. Gmsh
elements of order 2 or 3 give Lagrange edges, plus interior nodes. In 3D,
every face sharing an edge with a curved face is blended from its edges:
on triangles $\mathbf{x} = \mathbf{x}_\mathrm{lin} + \sum_e \lambda_i\lambda_j\,\mathbf{q}_e(t)$,
$t = (1 + \lambda_j - \lambda_i)/2$, with $\lambda$ the barycentric
coordinates and $\mathbf{q}_e$ the edge's deviation from its chord; on
quadrilaterals a Coons patch.

**Face quadrature.** The flat rule $(\boldsymbol{\xi}_q, \hat w_q)$ is mapped
through the face map. Each point gets the area vector
$\mathbf{a}_q = \hat w_q\,(\partial_u\mathbf{x}\times\partial_v\mathbf{x})(\boldsymbol{\xi}_q)$
(in 2D, the rotated tangent), and all area vectors are shifted by one common
vector so that they sum exactly to the face's area vector,

$$
\begin{equation}
\sum_q \mathbf{a}_q = \int_f \mathbf{n}\,\mathrm{d}S = \frac12\oint_{\partial f}\mathbf{x}\times\mathrm{d}\mathbf{x},
\label{eq:curved-boundaries}
\end{equation}
$$

which is exact for the face's curved edges. Every cell then still closes,
$\sum_f s_{i,f}\int_f\mathbf{n}\,\mathrm{d}S = 0$, and gas at rest stays at
rest to round-off. Each point carries the unit normal
$\mathbf{n}_{f,q} = \mathbf{a}_q/\lvert\mathbf{a}_q\rvert$ and the weight
$w_{f,q} = 2\lvert\mathbf{a}_q\rvert/A_f$ in eq. $\eqref{eq:semi-discrete-equations-2}$, where $A_f$ is the
length of the face's area vector.

**Cells.** Cells with a curved face are integrated over signed cones from
their vertex average $\mathbf{o}$ to each face,
$\mathbf{x} = \mathbf{o} + \tau\left(\mathbf{x}_f(\boldsymbol{\xi}) - \mathbf{o}\right)$,
$\tau\in[0, 1]$, with tensor Gauss rules over the curved faces. These rules
give the volume, the centroid, the TENO-E moments and the initial cell
averages of curved cells, exactly for polynomials.

![Quadrature on a curved boundary face](../images/theory/curved_quadrature-light.svg#gh-light-mode-only)
![Quadrature on a curved boundary face](../images/theory/curved_quadrature-dark.svg#gh-dark-mode-only)

## Reconstruction

The face states $\mathbf{U}^L$, $\mathbf{U}^R$ at each quadrature point
come from polynomials reconstructed from the cell averages around each
cell (`[numerics.face_reconstruction] type`). A single gas reconstructs the
variables named below; a gas mixture reconstructs
$\mathbf{W} = [\rho, \mathbf{u}, p]$ and its scalars separately
([below](#gas-mixtures)). Boundary faces reconstruct only the interior
side.

![Reconstruction stencils on an unstructured mesh](../images/theory/stencils-light.svg#gh-light-mode-only)
![Reconstruction stencils on an unstructured mesh](../images/theory/stencils-dark.svg#gh-dark-mode-only)

### First order

$\mathbf{U}^L_{f,q} = \overline{\mathbf{U}}_i$: piecewise constant.

### MUSCL

MUSCL ([van Leer 1979](../references.md#van-leer-1979);
`src/numerics/face_reconstruction.cpp`) reconstructs the primitive
variables $\mathbf{W} = [\rho, \mathbf{u}, p]$ linearly, with one point per
face at its centroid $\mathbf{x}_f$:

$$
\begin{equation}
\mathbf{W}^L_f = \mathbf{W}_i + \boldsymbol{\phi}_i\circ\left(\nabla\mathbf{W}_i\cdot(\mathbf{x}_f - \mathbf{x}_i)\right),
\label{eq:muscl-1}
\end{equation}
$$

with one limiter value per variable and cell. A face side whose density or
pressure would be non-positive takes the cell value instead.

**Gradients.** Weighted least squares
([Mavriplis 2003](../references.md#mavriplis-2003)): with neighbor offsets
$\Delta\mathbf{x}_j = \mathbf{x}_j - \mathbf{x}_i$ and weights
$w_j = \lvert\Delta\mathbf{x}_j\rvert^{-2}$, $\nabla W_i$ minimizes
$\sum_j w_j\left(W_j - W_i - \nabla W_i\cdot\Delta\mathbf{x}_j\right)^2$,

$$
\begin{equation}
\left(\sum_j w_j\,\Delta\mathbf{x}_j\Delta\mathbf{x}_j^T\right)\nabla W_i = \sum_j w_j\,\Delta\mathbf{x}_j\,(W_j - W_i).
\label{eq:muscl-2}
\end{equation}
$$

The neighbors are the face neighbors. A boundary face contributes its ghost
state ([chapter 9](boundaries_sources.md#weak-imposition)) at the mirror
image of the cell centroid, $\Delta\mathbf{x} = 2 d\,\mathbf{n}_f$ with $d$
the centroid's distance to the face; prescribed states (`dirichlet`, `upt`,
`farfield`) sit at the face centroid instead. Across `extrapolation` faces
with an image face ([chapter 9](boundaries_sources.md#transmissive-boundaries))
the neighbor is the image cell, moved outward by the translation that takes
the image face onto the boundary face. Tetrahedra, whose four face
neighbors make weakly limited MUSCL unstable, and cells on characteristic
boundaries, whose zero-gradient ghosts would halve outgoing waves, use the
linear fit over their vertex neighbors instead (the same fit as eq. $\eqref{eq:muscl-2}$
over the vertex-neighbor stencil of [below](#gradients-for-the-viscous-terms)).

**Limiters.** With $\Delta_{-} = \nabla W_i\cdot(\mathbf{x}_f - \mathbf{x}_i)$
at each face centroid and $\Delta_{\max}$, $\Delta_{\min}$ the largest and
smallest $W_j - W_i$ over the cell and its face neighbors (ghosts
included), $\phi_i = \min_f\phi_f$ with
([Barth & Jespersen 1989](../references.md#barth-jespersen-1989);
[Venkatakrishnan 1995](../references.md#venkatakrishnan-1995))

$$
\begin{equation}
\phi_f^\mathrm{BJ} =
\begin{cases}
\min\left(1, \Delta_{\max}/\Delta_{-}\right) & \Delta_{-} > 0, \\
\min\left(1, \Delta_{\min}/\Delta_{-}\right) & \Delta_{-} < 0, \\
1 & \Delta_{-} = 0,
\end{cases}
\qquad
\phi_f^\mathrm{V} = \frac{\Delta_{+}^2 + \varepsilon^2 + 2\Delta_{-}\Delta_{+}}{\Delta_{+}^2 + 2\Delta_{-}^2 + \Delta_{-}\Delta_{+} + \varepsilon^2},
\label{eq:muscl-3}
\end{equation}
$$

where $\Delta_{+} = \Delta_{\max}$ if $\Delta_{-} > 0$ and $\Delta_{\min}$
otherwise, and the Venkatakrishnan threshold (the default limiter) is

$$
\begin{equation}
\varepsilon^2 = (K h_i)^3\,s^2, \qquad h_i = \lvert\Omega_i\rvert^{1/d},
\label{eq:muscl-4}
\end{equation}
$$

with $K$ = `venkatakrishnan_K` (default 5) and the scale $s$ of the
variable: $\rho_i$ for density, $p_i$ for pressure, the sound speed $a_i$
for the velocity components. Limiting the velocity components one by one
makes limited runs depend on the orientation of the axes
([#459](https://github.com/MatthewBonanni/mallard/issues/459)); without a
limiter, MUSCL does not.

### TENO-E

`type = "TENO"` with `order` $k = 3,\dots,6$ is the TENO-E scheme of
[Liang, Shyy & Fu 2025](../references.md#liang-shyy-fu-2025), built on TENO
([Fu, Hu & Adams 2016](../references.md#fu-hu-adams-2016)) and its
unstructured finite-volume form
([Ji, Liang & Fu 2022](../references.md#ji-liang-fu-2022)), extended to 3D
cells (`src/numerics/teno.cpp`, `teno_setup.cpp`). Implementation details
and measured stability cases: [TENO-E details](../numerics/teno_e.md).

**Polynomials.** In the cell-scaled frame $\boldsymbol{\xi} = (\mathbf{x} - \mathbf{x}_i)/h_i$,
$h_i = \lvert\Omega_i\rvert^{1/d}$, a polynomial of degree $r = k - 1$ uses
the zero-mean monomial basis

$$
\begin{equation}
\psi_l(\boldsymbol{\xi}) = \phi_l(\boldsymbol{\xi}) - \frac{1}{\lvert\Omega_i\rvert}\int_{\Omega_i}\phi_l\,\mathrm{d}V, \qquad
u(\boldsymbol{\xi}) = \overline{u}_i + \sum_{l=1}^{N_r} a_l\,\psi_l(\boldsymbol{\xi}),
\label{eq:teno-e-1}
\end{equation}
$$

with $\phi_l$ the $N_r$ monomials of degrees 1 to $r$, so that the
polynomial keeps the cell average exactly. On a stencil $\mathcal{S}$ of
cells $s$, the coefficients are the least-squares solution
([Barth & Frederickson 1990](../references.md#barth-frederickson-1990)) of

$$
\begin{equation}
\sum_l A_{sl}\,a_l = \overline{u}_s - \overline{u}_i, \qquad
A_{sl} = \frac{1}{\lvert\Omega_s\rvert}\int_{\Omega_s}\psi_l\,\mathrm{d}V, \qquad
\mathbf{a} = \mathbf{P}\left(\overline{\mathbf{u}}_\mathcal{S} - \overline{u}_i\right),
\label{eq:teno-e-2}
\end{equation}
$$

whose pseudo-inverse $\mathbf{P}$ is computed once, at start-up, by
Householder QR after column equilibration; a stencil whose equilibrated
$\mathbf{R}$ factor has a diagonal ratio above `max_condition` ($10^8$) is
rejected. Cell averages of monomials are integrated exactly (polygon or
tetrahedral decompositions, or the cone rules of curved cells; in 3D from
central moments shifted binomially). Axisymmetric runs fit r-weighted
averages.

**Stencils.** Each cell has

- one *central stencil* of about $\lceil 2N_r\rceil$ cells
    (`stencil_factor`, default 2), from vertex-neighbor layers sorted by
    centroid distance ([Tsoutsanis, Titarev & Drikakis 2011](../references.md#tsoutsanis-2011)),
    in 3D measured in a metric of the local mesh spacing where the cells are
    strongly anisotropic. A group of equidistant candidates is never split, so
    mirror-symmetric meshes give mirror-symmetric reconstructions;
- one degree-2 *sector stencil* per face, of the nearest cells whose
    centroid direction lies in the cone spanned by the face's vertices from
    the cell centroid (default 10 cells in 2D, 18 in 3D);
- across boundary faces, *mirror images* of interior cells across the
    face's line or plane, whose state is the boundary condition's ghost state
    of that cell, so boundary cells get centered stencils instead of
    one-sided extrapolation. Curved walls and curved symmetry faces take none
    (one-sided stencils), since a reflection matches a curved wall only to
    $O(h^2)$.

**Lebesgue-bound selection.** Full rank is not enough: a stencil that
resolves one direction only through small centroid offsets has full rank
and amplifies cell averages. The central stencil therefore grows, from
its nominal size up to $\lceil 3.5N_r\rceil$ cells (3D: up to $3.5N_r + 64$),
until the Lebesgue constant of its reconstruction at the cell's face
quadrature points,

$$
\begin{equation}
\Lambda = \max_q\left(\Bigl\lvert 1 - \sum_s c_{qs}\Bigr\rvert + \sum_s\lvert c_{qs}\rvert\right), \qquad
c_{qs} = \sum_l \psi_l(\boldsymbol{\xi}_q)\,P_{ls},
\label{eq:teno-e-3}
\end{equation}
$$

the largest factor by which the face values $u_q = \overline{u}_i + \sum_s c_{qs}(\overline{u}_s - \overline{u}_i)$
can amplify the cell averages, is at most 10 in 2D and 4 in 3D. If no size
reaches the bound, 2D takes the stencil with the smallest $\Lambda$, and 3D
the smallest stencil within 1.25 times the smallest $\Lambda$ found. The
lower 3D bound removes growing modes of the linearized scheme on
near-degenerate stencils ([TENO-E details](../numerics/teno_e.md#3d-stability)).
Cells whose central stencil reaches a curved wall fit at most degree
`curved_wall_degree` (3).

**Smooth and troubled cells.** Each cell's troubled-cell measure is the
variance of the relative density jumps over its central stencil (mixtures:
the larger of that and the variance of the relative molar-mass jumps):

$$
\begin{equation}
\sigma_i = \operatorname{var}_{s\in\mathcal{S}_K}\left(\frac{\lvert\rho_s - \rho_i\rvert}{\rho_i}\right).
\label{eq:teno-e-4}
\end{equation}
$$

Cells with $\sigma_i < \sigma_L$ (`troubled_threshold`, $10^{-3}$) use the
central polynomial on the conservative variables of a single gas (on
$\mathbf{W}$ for mixtures). Troubled cells select stencils per face and per
characteristic variable: the variables are projected on the left
eigenvectors of the Euler flux Jacobian in the face's normal direction at
the mean of the two cells' states ([Toro 2009](../references.md#toro-2009)),
and for each characteristic variable the smoothness indicators of the
central polynomial $K$ and of the sector polynomials $k$,

$$
\begin{equation}
\beta = \sum_{1\le\lvert\boldsymbol{\alpha}\rvert\le r}\frac{\lvert\boldsymbol{\alpha}\rvert!}{\boldsymbol{\alpha}!}\int_{\Omega_i}\left(D^{\boldsymbol{\alpha}}u\right)^2\mathrm{d}\boldsymbol{\xi}
\quad\text{(in the scaled frame)}, \qquad
\gamma_m = \left(\frac{\min_n(\beta_n + \epsilon)}{\beta_m + \epsilon}\right)^{6}, \quad \epsilon = 10^{-12},
\label{eq:teno-e-5}
\end{equation}
$$

decide (the multinomial weights make each degree's sum the squared norm of
that derivative tensor, so the indicators do not depend on the orientation of
the axes): the central polynomial is used if
$\gamma_K/(\gamma_K + \sum_k\gamma_k) \ge C_T$; otherwise every sector
stencil with $\gamma_k/\sum_{k'}\gamma_{k'} \ge C_T$ is kept, with equal
weights (the renormalization over the sector stencils alone guarantees a
survivor). The cutoff adapts to the measure
([Liang, Shyy & Fu 2025](../references.md#liang-shyy-fu-2025)):

$$
\begin{equation}
m = \min\left(1, \max\left(0, \frac{\sigma_i - \sigma_L}{\sigma_U - \sigma_L}\right)\right), \quad
g = (1 - m)^2(1 + 2m), \quad C_T = 10^{-\lfloor 10 - 4(1 - g)\rfloor},
\label{eq:teno-e-6}
\end{equation}
$$

from $10^{-10}$ at $\sigma_L$ to $10^{-6}$ above $\sigma_U$
(`troubled_upper`, $10^{-2}$), unless `C_T` fixes it. The face values are
projected back with the right eigenvectors.

**Admissibility and bounds.** A cell whose face values have a non-positive
density or pressure anywhere falls back to first order. With
`bound_preserving = true`, troubled cells first scale their deviation from
the cell average by the largest $\theta\in[0, 1]$ that keeps every face value
$\overline{v}_i + \theta(v_q - \overline{v}_i)$ within $[m, M]$, for $v = \rho, p$
and the extrema $m$, $M$ of the cell and its face neighbors
([Zhang & Shu 2010](../references.md#zhang-shu-2010)).

Admissible face values do not make the update admissible: a troubled cell's
faces take different stencils, so its face values are not those of one
polynomial with the cell's average, and their fluxes can draw more internal
energy from the cell than it holds (in flows where the internal energy is a
small part of the total, Mach 10 and above). With `bound_preserving = true`
in planar single-gas runs, the convective fluxes are then limited
([Hu, Adams & Shu 2013](../references.md#hu-adams-shu-2013)). With the
closure $\sum_f A_f\mathbf{n}_f = 0$, the forward-Euler update of cell $i$
with face rates $\mathbf{R}_f$ (the flux into the cell, integrated over the
face) is a convex combination of partial states,

$$
\begin{equation}
\mathbf{U}_i^{n+1} = \sum_f \mu_f\,\mathbf{U}_i^f, \quad
\mathbf{U}_i^f = \mathbf{U}_i + \frac{\kappa_i}{A_f\alpha_f}\left(\mathbf{R}_f + A_f\,\mathbf{F}(\mathbf{U}_i)\cdot\mathbf{n}_f\right), \quad
\mu_f = \frac{A_f\alpha_f}{\sum_g A_g\alpha_g}, \quad
\kappa_i = \frac{\Delta t}{\lvert\Omega_i\rvert}\sum_g A_g\alpha_g,
\label{eq:positivity-limiter}
\end{equation}
$$

with $\alpha_f$ the larger $\lvert u_n\rvert + a$ of the face's two cell
averages (or the cell and its ghost), as in the time step. For the local
Lax-Friedrichs flux of the cell averages, $\mathbf{R}_f^{LF}$, every partial
state is admissible when $\kappa_i \le 1$, that is $\mathrm{CFL} \le 0.5$ at
the start of the step. Each face takes
$\mathbf{R}_f = \mathbf{R}_f^{LF} + \theta_f(\mathbf{R}_f^{HO} - \mathbf{R}_f^{LF})$
with the largest $\theta_f \in [0, 1]$ that keeps the partial states of both
its cells above $10^{-10}$ of their Lax-Friedrichs density and pressure:
linear in $\theta_f$ for the density, and a bound for the pressure, which is
concave in $\theta_f$. Faces that need no limiting keep their flux exactly,
and the scheme stays conservative. SSP Runge-Kutta stages are convex
combinations of such steps.

### Gas mixtures

Mixtures reconstruct $\mathbf{W} = [\rho, \mathbf{u}, p]$ with the scheme
above (TENO-E with primitive eigenvectors for any equation of state), and
the cell scalars $[Y_1,\dots,Y_{N_s}, \gamma, e_0]$ linearly with weights
shared by all of them (`src/numerics/scalar_reconstruction.cpp`,
`teno_scalars.h`), so that mass fractions summing to one sum to one at every
face point:

- MUSCL and first order: eq. $\eqref{eq:muscl-2}$ and one limiter value per cell for all
    scalars, the smallest of eq. $\eqref{eq:muscl-3}$ over them (thresholds scaled by 1 for
    $Y_k$, by $\gamma$, and by $\lvert e_0\rvert + p/(\rho(\gamma - 1))$ for
    $e_0$), further reduced so that every face value keeps $0 \le Y_k \le 1$
    and $\gamma - 1$ above half its cell value;
- TENO-E: the central-stencil weights in smooth cells, and in troubled
    cells, on each face, the stencils selected for the entropy (contact)
    characteristic field; a factor $\theta$ per cell then keeps
    $Y_k\in[0, 1]$, $\gamma - 1$ above half its cell value and, in troubled
    cells, every scalar within its neighbors' range.

The face $(\gamma, e_0)$ pairs enter the Riemann solvers
([chapter 7](fluxes.md#gas-mixtures)), and the face mass fractions the
species fluxes ([chapter 7](fluxes.md#species-fluxes)).

## Low-Mach correction

Upwind fluxes damp the jump of the reconstructed velocity across a face at
the sound speed, a dissipation that does not vanish relative to the flow as
$M\to 0$ ([Guillard & Viozat 1999](../references.md#guillard-viozat-1999);
[Rieper 2011](../references.md#rieper-2011)). Before the Riemann solver,
Mallard scales that jump at every face point, keeping its mean
([Thornber et al. 2008](../references.md#thornber-2008);
`low_mach_correction`, `src/numerics/flux_functor.h`):

$$
\begin{equation}
\mathbf{u}^{L,R} \leftarrow \frac{\mathbf{u}^L + \mathbf{u}^R}{2} \pm z\,\frac{\mathbf{u}^L - \mathbf{u}^R}{2}, \qquad
z = \min\left(1, \max\left(M^L, M^R, M_\mathrm{cut}\right)\right),
\label{eq:low-mach-correction}
\end{equation}
$$

with $M^{L,R} = \lvert\mathbf{u}^{L,R}\rvert/a^{L,R}$ from each side's own
$\gamma$, and $M_\mathrm{cut}$ = `[numerics] low_mach_cutoff` (default 0.1;
1 disables the correction). Density, pressure and the Riemann solver are
unchanged, and supersonic faces ($z = 1$) are untouched. It applies at
interior faces and at characteristic boundaries. Its motivation and
measured effects: [Numerical methods](../numerics/overview.md#low-mach-correction).

## Gradients for the viscous terms

The viscous fluxes, the SGS models and the hybrid flux's shock sensor use
cell gradients from a weighted least-squares fit of a *quadratic* over the
cell's vertex neighbors and the ghost states of its own boundary faces
(`LSQVertexGradientFunctor`, `src/numerics/gradient.h`;
[Barth & Frederickson 1990](../references.md#barth-frederickson-1990)). With
offsets $\Delta\mathbf{x}_j$ scaled by the inverse distance of the first
point, weights $w_j = \lvert\Delta\mathbf{x}_j\rvert^{-2}$ and the
$d + d(d+1)/2$ linear and quadratic monomials $\boldsymbol{\varphi}(\Delta\mathbf{x})$,

$$
\begin{equation}
\nabla W_i = \sum_j \mathbf{c}_j\,(W_j - W_i), \qquad
\mathbf{c}_j = w_j\left[\left(\sum_m w_m\,\boldsymbol{\varphi}_m\boldsymbol{\varphi}_m^T\right)^{-1}\boldsymbol{\varphi}_j\right]_{1..d},
\label{eq:gradients-for-the-viscous-terms}
\end{equation}
$$

the first $d$ components of the least-squares solution, precomputed per
cell as fixed weights. A linear fit is only first-order accurate on the
one-sided stencils of boundary cells and on triangles, where its errors
cancel only in the interior of regular meshes; stencils that cannot
determine the quadratic (fewer than $d + d(d+1)/2$ points, or a Cholesky
pivot below $10^{-10}$ of its diagonal) use the linear fit. Ghosts sit at
the mirror image of the centroid as for MUSCL. Across `extrapolation`
boundaries with image faces the stencil continues instead, as across a
periodic seam: each boundary node maps to the matching node of its image
face, and the cells around that node, moved back by the same translation,
that land outside the domain join the stencil of every cell sharing the
node (chaining the maps of two or three boundaries at edges and corners).
Boundary cells then have the stencil of an interior cell in a flow that does
not vary normal to the boundary; one-sided stencils carry O(h) errors on
tetrahedra that the interior's do not, which made a channel with
extrapolation sides first order and drove a cross flow. Axisymmetric runs fit
r-weighted averages through the cells' second moments. Single gases
differentiate $\mathbf{W}$, with
$\nabla T = (\nabla p - R T\nabla\rho)/(\rho R)$; mixtures differentiate
$[\mathbf{u}, T, X_1,\dots,X_{N_s}]$ directly, with the wall velocity at
viscous walls and the prescribed $T$ and $X_k$ at `upt` boundaries
(the cell's own values elsewhere). The face gradients built from these are
in [chapter 7](fluxes.md#viscous-fluxes).

## Axisymmetric cells

In axisymmetric runs ([chapter 1](governing_equations.md#axisymmetric-flows)),
each cell keeps its planar geometry for reconstruction stencils and
limiters, and adds the revolved measures
(`Mesh::make_axisymmetric`; [design notes](../design/axisymmetric.md)):
the revolved volume $\int_\Omega r\,\mathrm{d}A$ (exact for polygons), used
wherever a volume divides or weights; the revolved face area $r_f L_f$ for
the one-point viscous fluxes and the time step; the revolved centroid
$\int_\Omega\mathbf{x}\,r\,\mathrm{d}A/\int_\Omega r\,\mathrm{d}A$, at which a
linear function equals its r-weighted average; and convective face weights
$w_{f,q}\,r_q$ in eq. $\eqref{eq:semi-discrete-equations-2}$. The radial momentum source $\int_\Omega(p - \tau_{\theta\theta})\,\mathrm{d}A$
of eq. (9) of [chapter 1](governing_equations.md#axisymmetric-flows) uses the cell values (first order, MUSCL, troubled TENO cells)
or, in smooth TENO cells, the central polynomial at a collapsed Gauss rule
exact to its degree; both are exactly well balanced for uniform pressure.
