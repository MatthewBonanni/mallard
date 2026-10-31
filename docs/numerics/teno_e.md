# TENO-E details

Implementation reference for the `TENO` reconstruction
(`[numerics.face_reconstruction] type = "TENO"`): what Mallard computes, in
which order, with which constants, and the measurements behind its choices.
The method itself is summarized in the theory manual
([TENO-E](../theory/finite_volume.md#teno-e)), whose notation this page
follows; its options and their defaults are in the user guide
([Numerics](../guide/numerics.md#teno-e-options)).

TENO-E ([Liang, Shyy & Fu 2025](../references.md#liang-shyy-fu-2025)) is the
unstructured finite-volume TENO scheme of
[Ji, Liang & Fu 2022](../references.md#ji-liang-fu-2022), built on TENO
([Fu, Hu & Adams 2016](../references.md#fu-hu-adams-2016)), with an adaptive
cutoff. Mallard runs the same algorithm on
triangles and quadrilaterals in 2D and on tetrahedra, hexahedra, prisms and
pyramids in 3D (at most 4 and 6 faces per cell), any mix of them, at orders
$k = 3,\dots,6$, with mirror-image stencils at boundaries, curved boundary
faces, and its own stencil selection by a Lebesgue bound.

| File | Contents |
|---|---|
| `src/numerics/teno.h` | Monomials, Euler eigenvectors, the adaptive cutoff, packed stencil storage |
| `src/numerics/teno.cpp` | Options, the 2D setup, the cache, and the reconstruction kernels |
| `src/numerics/teno_setup.h`, `teno_setup.cpp` | The 3D setup on the device: stencil search, ranking metric, moments, fits |
| `src/numerics/teno_scalars.h` | Mixture scalars reconstructed with the flow's stencil choices |
| `src/numerics/face_reconstruction.h`, `face_reconstruction.cpp` | Face quadrature, and the interface shared with first order and MUSCL |

## Polynomials

A reconstruction of order $k$ fits polynomials of degree $r = k - 1$ in the
cell-scaled frame of the target cell $i$,

$$
\begin{equation}
\boldsymbol{\xi} = \frac{\mathbf{x} - \mathbf{x}_i}{h_i}, \qquad h_i = \lvert\Omega_i\rvert^{1/d},
\label{eq:frame}
\end{equation}
$$

with $\mathbf{x}_i$ the centroid. The basis is the $N_r$ monomials
$\phi_l = \xi^a\eta^b$ (3D: $\xi^a\eta^b\zeta^c$) of total degree 1 to $r$,

$$
\begin{equation}
N_r = \binom{r + d}{d} - 1,
\label{eq:dofs}
\end{equation}
$$

ordered by degree, then by decreasing power of $\xi$, then of $\eta$
(`teno::exponents`), each minus its mean over the target cell:

$$
\begin{equation}
\psi_l = \phi_l - \overline{\phi}_{l,i}, \qquad
\overline{\phi}_{l,c} = \frac{1}{\lvert\Omega_c\rvert}\int_{\Omega_c}\phi_l\,\mathrm{d}V, \qquad
u(\boldsymbol{\xi}) = \overline{u}_i + \sum_{l=1}^{N_r} a_l\,\psi_l(\boldsymbol{\xi}).
\label{eq:basis}
\end{equation}
$$

$N_r$ is 5, 9, 14 and 20 in 2D and 9, 19, 34 and 55 in 3D for $r = 2$ to
5; the sector polynomials always have $r = 2$, so $N_2$ coefficients.

**Fits.** On a stencil $\mathcal{S}$ of $m \ge N_r$ entries $s$ (cells, or
mirror images of cells), the coefficients solve

$$
\begin{equation}
\min_{\mathbf{a}}\sum_{s\in\mathcal{S}}\Bigl(\sum_l A_{sl}\,a_l - (\overline{u}_s - \overline{u}_i)\Bigr)^2, \qquad
A_{sl} = \overline{\phi}_{l,s} - \overline{\phi}_{l,i}, \qquad
a_l = \sum_s P_{ls}\,(\overline{u}_s - \overline{u}_i),
\label{eq:fit}
\end{equation}
$$

with the means of every entry taken in the frame $\eqref{eq:frame}$ of the
target cell ([Barth & Frederickson 1990](../references.md#barth-frederickson-1990)).
The pseudo-inverse $\mathbf{P} = \mathbf{R}^{-1}\mathbf{Q}^T$ ($N_r\times m$)
comes from Householder QR after scaling each column of $\mathbf{A}$ to unit
length (`pseudo_inverse`). A fit is rejected, and the stencil with it, if a
column is zero, if a diagonal entry of $\mathbf{R}$ is smaller than its
largest diagonal entry over `max_condition` ($10^8$), or, in 3D, if a column
is shorter than $10^{-10}$ times the longest before scaling: a column of
round-off, such as $\xi\eta$ over centroids that all lie on axis planes,
would pass the rank test once scaled. Only $\mathbf{P}$ is stored; at run
time each polynomial costs one product of $\mathbf{P}$ with the stencil's
cell averages.

**Cell averages of monomials.** In 2D, polygons are split into triangles
from their first vertex and integrated with a collapsed Gauss rule of
$r + 2$ points per direction, exact to degree $2r + 2$ (curved cells: the cone
rules of the [theory manual](../theory/finite_volume.md#curved-boundaries),
exact to degree $2r$). In 3D, each cell $c$ first gets its central moments
up to degree $2r$ in its own frame,

$$
\begin{equation}
\mu_c^{\boldsymbol{\kappa}} = \frac{1}{\lvert\Omega_c\rvert}\int_{\Omega_c}
\left(\frac{\mathbf{x} - \mathbf{x}_c}{h_c}\right)^{\boldsymbol{\kappa}}\mathrm{d}V,
\label{eq:moments}
\end{equation}
$$

integrated once over the cell's tetrahedral decomposition (cones from the
vertex average to each face, quadrilateral faces split at their centroid;
the decomposition that defines the mesh geometry). The means of the target's
monomials over a stencil cell then follow by the binomial theorem, with
$s_c = h_c/h_i$ and $\boldsymbol{\delta} = (\mathbf{x}_c - \mathbf{x}_i)/h_i$:

$$
\begin{equation}
\overline{\boldsymbol{\xi}^{\boldsymbol{\alpha}}}_c = \sum_{\boldsymbol{\kappa}\le\boldsymbol{\alpha}}
\binom{\boldsymbol{\alpha}}{\boldsymbol{\kappa}}\,\boldsymbol{\delta}^{\boldsymbol{\alpha} - \boldsymbol{\kappa}}\,
s_c^{\lvert\boldsymbol{\kappa}\rvert}\,\mu_c^{\boldsymbol{\kappa}}
\label{eq:binomial}
\end{equation}
$$

(multi-index notation, $\binom{\boldsymbol{\alpha}}{\boldsymbol{\kappa}} = \prod_j\binom{\alpha_j}{\kappa_j}$).
A mirror image across a plane with unit normal $\mathbf{n}$ has the moments
of the cell transformed by the reflection $\mathbf{Q} = \mathbf{I} - 2\mathbf{n}\mathbf{n}^T$,
degree by degree, and enters eq. $\eqref{eq:binomial}$ at the mirrored centroid.

## Stencils

![Lebesgue-bound growth of the central stencil of a stretched quadrilateral under triangles: the 18 nearest cells give a Lebesgue constant of 13, one more cell gives 5.5](../images/theory/teno_lebesgue-light.svg#gh-light-mode-only)
![Lebesgue-bound growth of the central stencil of a stretched quadrilateral under triangles: the 18 nearest cells give a Lebesgue constant of 13, one more cell gives 5.5](../images/theory/teno_lebesgue-dark.svg#gh-dark-mode-only)

*Central stencil of a quadrilateral (black) of aspect ratio 2.6 below a
triangulated region, at order 4 ($N_3 = 9$). The 18 nearest cells (teal)
take only the first row of triangles and fit the vertical direction from
small centroid offsets: $\Lambda = 13$. One more cell (orange) brings
$\Lambda$ to 5.5, within the 2D bound; the 3D bound would take 21 cells
($\Lambda = 2.2$). Computed with the 2D setup's algorithm in
`tools/docs/teno.py`.*

### Candidates

The candidates of a cell are gathered breadth-first by vertex-neighbor
layers (`Layers`; across periodic boundaries at their lattice translation),
a layer at a time while at most $n_\mathrm{max}$ cells have been gathered,
for at most 64 layers; the sector stencils search up to $8n_s$ cells within
6 layers. Every gathered cell is a candidate, and so is its mirror image
across each straight boundary line (2D) or plane (3D) that a gathered cell
touches, except:

- images that land inside a gathered cell (non-convex boundaries);
- in 2D, images of cells whose foot on the line lies beyond the line's
    gathered faces, of cells behind the line, and, for a target cell behind
    the line, all of its images (#382). A facet of a straight-sided curved
    wall, extended, passes inside the next columns of cells, by about a cell
    height for the thin wall cells of an airfoil mesh; reflections across it
    put the wall-normal profile there at the wrong distance from the wall. At
    a closed trailing edge thinner than its wall cells, the cells on one side
    mirrored across the other side land next to the cells on the first. Both
    gave an odd-even mode in the wall pressure of the NACA 0012 tutorial mesh
    at order 3;
- boundaries between MPI ranks (`PARTITION` faces), which are not boundaries
    of the domain;
- curved wall and symmetry faces, which take no images unless
    `curved_mirrors = true`: a reflection across their plane matches the
    curved surface only to $O(h^2)$.

Coplanar boundary faces form one plane; each image takes the ghost state of
the boundary condition (applied to its cell's state) from the face of that
plane nearest to it, so one plane may carry several conditions. A 3D search
may meet at most 255 distinct planes (#392; a faceted curved boundary with
mirror images has one plane per facet); a cell beyond that stops the setup
with an error that names it. Counting only interior cells toward
$n_\mathrm{max}$ matters near walls: counting images too stops the search
before the cells that are nearer than the farther images, and leaves
order-6 stencils rank deficient.

Candidates are sorted by distance from $\mathbf{x}_i$ to their centroid (or
mirrored centroid). A stencil never splits a group of candidates whose
squared distances differ by less than $10^{-10}h_i^2$, so the stencils do
not depend on the cell numbering and mirror-symmetric meshes give
mirror-symmetric reconstructions.

**Ranking metric.** On strongly anisotropic cells, the distance is
measured in the metric of the local mesh spacing,
$\lVert\mathbf{d}\rVert_\mathbf{M}^2 = \mathbf{d}^T\mathbf{M}\,\mathbf{d}$
(`teno_setup::ranking_metric`). With $\mathbf{C} = \sum_j \mathbf{d}_j\mathbf{d}_j^T$
the second moment of the offsets $\mathbf{d}_j$ of the cell's vertex
neighbors, completed at its boundary faces by the mirror images of the
neighbors at most as far from the face as the cell, and $\lambda_{\max}$,
$\lambda_{\min}$ its extreme eigenvalues,

$$
\begin{equation}
\mathbf{M} =
\begin{cases}
(\det\mathbf{C})^{1/d}\,\mathbf{C}^{-1} & \sqrt{\lambda_{\max}/\lambda_{\min}} > 2.5, \\
\mathbf{I} & \text{otherwise},
\end{cases}
\label{eq:metric}
\end{equation}
$$

a metric of unit determinant whose unit length is the spacing in every
direction. The reasons and the cut-off of 2.5 are in
[3D stability](#3d-stability). 2D meshes need it too (#382): the wall cells
of the NACA 0012 tutorial mesh (aspect ratios of 10 to 30) took whole
wall-normal columns by physical distance, and orders 3 to 6 settled into an
odd-even wall-pressure mode, which at order 4 neither the metric nor the
mirror-image rules above remove alone.

### Central stencil

The central stencil starts from $n_0 = \lceil f N_r\rceil$ entries
($f$ = `stencil_factor`, default 2) and grows one entry at a time, skipping
sizes that split a tie, up to

$$
\begin{equation}
n_\mathrm{max} = \lceil 3.5 N_r\rceil \ \text{(2D)}, \qquad
n_\mathrm{max} = \lceil 3.5 N_r\rceil + 64 \ \text{(3D)},
\label{eq:nmax}
\end{equation}
$$

since the equidistant shells of 3D lattices are large. Each full-rank size
is scored by the Lebesgue constant of its reconstruction at the cell's own
face quadrature points $\boldsymbol{\xi}_q$ (`lebesgue_constant`):

$$
\begin{equation}
\Lambda = \max_q\left(\Bigl\lvert 1 - \sum_s c_{qs}\Bigr\rvert + \sum_s\lvert c_{qs}\rvert\right), \qquad
c_{qs} = \sum_l \psi_l(\boldsymbol{\xi}_q)\,P_{ls},
\label{eq:lebesgue}
\end{equation}
$$

the largest factor by which the face values
$u_q = (1 - \sum_s c_{qs})\,\overline{u}_i + \sum_s c_{qs}\,\overline{u}_s$
can amplify the cell averages. The first size with $\Lambda \le \Lambda_\mathrm{max}$
is taken, with

$$
\begin{equation}
\Lambda_\mathrm{max} = 10 \ \text{(2D)}, \qquad \Lambda_\mathrm{max} = 4 \ \text{(3D)}.
\label{eq:bound}
\end{equation}
$$

If no size reaches it, 2D takes the size with the smallest $\Lambda$ and 3D
the smallest size within 1.25 times the smallest $\Lambda$ found (the best
alone is often the largest stencil, and up to twice as inaccurate). Full
rank is not enough: a stencil that resolves one direction only through small
centroid offsets has full rank and amplifies the cell averages, and with them
the truncation error. The 2 × $N_r$ nearest cells of a jittered hexahedral or
prism mesh span only about three cell layers per direction, so odd degrees
(3 and 5) see the extra layer they need only through such offsets:
Lebesgue constants of 20 to 150 and $O(1)$ errors at order 4. Why the 3D
bound is 4 rather than 10: [3D stability](#3d-stability).

**Curved walls.** A stencil without mirror images at a curved wall is
one-sided there, and one-sided fits of degree 4 and 5 grow in time whatever
the bound or stencil depth. Cells whose first $n_0$ candidates include a
cell with a curved wall or symmetry face (without `curved_mirrors`) fit at
most degree `curved_wall_degree` (default 3), from
$\lceil f N_{r'}\rceil$ entries for that degree $r'$; their coefficients above
it are zero, so the run-time kernels are unchanged.

If no size has full rank, the setup stops ("could not build a full-rank
large stencil"), unless the search was cut off by the halo of an MPI
partition, in which case the cell is redone once the halo is deep enough.

### Sector stencils

Each face of the cell has one sector stencil of degree 2: the nearest
candidates (in the same order as the central stencil's) whose offset
$\mathbf{d}$ from $\mathbf{x}_i$ lies in the cone spanned from $\mathbf{x}_i$
by the face's vertices $\mathbf{v}_j$. In 2D that is
$\mathbf{d} = \alpha(\mathbf{v}_1 - \mathbf{x}_i) + \beta(\mathbf{v}_2 - \mathbf{x}_i)$
with $\alpha, \beta \ge -10^{-10}$; in 3D the same test on the triangles of a
fan from the face's first vertex. The stencil has $n_s$ entries
(`small_stencil_size`: 10 in 2D, $2N_2 = 18$ in 3D), up to $2n_s$ so as not to
split a tie. A face whose cone holds fewer than $n_s$ candidates, whose tie
cannot be resolved within $2n_s$, or whose fit is rejected has no sector
stencil; the log counts these ("sector stencils cut by boundaries"), which
occur at corners of the domain. A troubled cell then chooses among the
stencils it has.

## Smoothness indicators

The indicator of a polynomial with coefficients $\mathbf{a}$ is a quadratic
form in them,

$$
\begin{equation}
\beta = \sum_{1\le\lvert\boldsymbol{\alpha}\rvert\le r}\frac{\lvert\boldsymbol{\alpha}\rvert!}{\boldsymbol{\alpha}!}\int_{\Omega_i}\left(D^{\boldsymbol{\alpha}}u\right)^2\mathrm{d}\boldsymbol{\xi}
= \mathbf{a}^T\mathbf{S}\,\mathbf{a}, \qquad
S_{lm} = \sum_{1\le\lvert\boldsymbol{\alpha}\rvert\le r}\frac{\lvert\boldsymbol{\alpha}\rvert!}{\boldsymbol{\alpha}!}\int_{\Omega_i} D^{\boldsymbol{\alpha}}\phi_l\,D^{\boldsymbol{\alpha}}\phi_m\,\mathrm{d}\boldsymbol{\xi},
\label{eq:si}
\end{equation}
$$

with derivatives and the integral over the target cell in the scaled frame,
so that $\beta$ does not depend on the cell size. The multinomial weight
$\lvert\boldsymbol{\alpha}\rvert!/\boldsymbol{\alpha}!$ counts every ordering
of the derivatives, so each degree's sum is the squared Frobenius norm of
that derivative tensor and $\beta$ does not depend on the orientation of the
axes (the unweighted sum of Jiang & Shu's indicators does: a rotation by 30
degrees changed the solution at $10^{-3}$ after one step, #436). The sector polynomials use
the leading $N_2\times N_2$ block of the same $\mathbf{S}$. 2D stores the
upper triangle of $\mathbf{S}$ per cell, integrated with the rule of the cell
averages. 3D stores the cell's central moments $\eqref{eq:moments}$ instead
(up to degree $2r$) and forms the entries of $\mathbf{S}$ from them in each
troubled pass: every entry is a fixed sum of integer weights times moments
(`teno::SmoothnessTerms`), summed in the setup's order without fused
multiply-adds, so it equals the entry the setup would have stored.

## Reconstruction

Each step, every reconstructed cell runs the smooth pass; the troubled cells
it queues then run the troubled passes, in rounds of at most
`troubled_capacity` of the cells (default 0.5 in 3D, 1 in 2D) to bound
their scratch memory.

### Troubled-cell measure

$$
\begin{equation}
\sigma_i = \frac{1}{n}\sum_{s\in\mathcal{S}_K}\left(g_s - \overline{g}\right)^2, \qquad
g_s = \frac{\lvert\rho_s - \rho_i\rvert}{\rho_i}, \qquad
\overline{g} = \frac{1}{n}\sum_{s\in\mathcal{S}_K} g_s,
\label{eq:sigma}
\end{equation}
$$

the variance over the $n$ entries of the central stencil $\mathcal{S}_K$
(mirror images at their ghost states), accumulated in the same pass as the
central coefficients. Gas mixtures take the larger of $\sigma_i$ and the
same variance of the relative molar-mass jumps, which shows species
interfaces at constant density. The cell is troubled if
$\sigma_i \ge \sigma_L$ (`troubled_threshold`, $10^{-3}$); output
`TENO_SIGMA` shows $\sigma_i$.

### Smooth cells

The central polynomial of the conservative variables (of
$\mathbf{W} = [\rho, \mathbf{u}, p]$ for mixtures) gives the face values
directly,

$$
\begin{equation}
\mathbf{U}_q = \overline{\mathbf{U}}_i + \sum_l \mathbf{a}_l\,\psi_l(\boldsymbol{\xi}_q),
\label{eq:smooth}
\end{equation}
$$

with no smoothness indicators and no characteristic projection.

### Troubled cells

Stencils are selected per face and per characteristic variable. On face
$f$, the variables are projected on the left eigenvectors $\mathbf{L}$ of the
Euler flux Jacobian in the face's unit normal, evaluated at the arithmetic
mean of the two cells' primitive states (the cell's own at boundary faces;
[Toro 2009](../references.md#toro-2009)). Single gases use the
conservative-variable eigenvectors with the gas's $\gamma$; mixtures the
primitive-variable ones, valid for any equation of state, with the mean of
the two cells' $\gamma$ in the sound speed. The variables are ordered
$u_n - a$, $u_n$ (entropy), $u_n + a$, then one shear wave per tangent.
In 3D the two shear waves select together, on the sum of their indicators
($\beta_m$ below), since the tangents are an arbitrary basis of the face's
plane (`tangent_basis` projects out the coordinate axis least aligned with
the normal); selecting them one by one made the solution depend on the
orientation of the axes and on the cell numbering, which flips normals (#436).
`characteristic = false` uses the identity instead; its momentum
components then select their stencils separately, so that, unlike the
characteristic default, the solution depends on the orientation of the axes.

For each characteristic variable $v$, with the central coefficients
$\mathbf{c}_K = \mathbf{L}_v\mathbf{a}_K$ and the sector ones
$\mathbf{c}_k = \mathbf{L}_v\mathbf{a}_k$,

$$
\begin{equation}
\beta_m = \mathbf{c}_m^T\mathbf{S}\,\mathbf{c}_m + \epsilon, \qquad
\gamma_m = \left(\frac{\min_n\beta_n}{\beta_m}\right)^6, \qquad \epsilon = 10^{-12},
\label{eq:gamma}
\end{equation}
$$

over $m \in \{K, 1, \dots\}$, normalized by the smallest $\beta$ so that no
weight overflows, even in single precision. Then

$$
\begin{equation}
v_q =
\begin{cases}
v_i + \mathbf{c}_K\cdot\boldsymbol{\psi}(\boldsymbol{\xi}_q) & \dfrac{\gamma_K}{\gamma_K + \sum_k\gamma_k} \ge C_T, \\[2ex]
v_i + \dfrac{1}{\lvert\mathcal{K}\rvert}\displaystyle\sum_{k\in\mathcal{K}}\mathbf{c}_k\cdot\boldsymbol{\psi}(\boldsymbol{\xi}_q),
\quad \mathcal{K} = \Bigl\{k : \dfrac{\gamma_k}{\sum_{k'}\gamma_{k'}} \ge C_T\Bigr\} & \text{otherwise},
\end{cases}
\label{eq:select}
\end{equation}
$$

where $\boldsymbol{\psi}$ holds the first $N_2$ basis functions for the
sectors. The central polynomial is also taken when the cell has no sector
stencil. Renormalizing over the sectors alone in the second case
([Ji, Liang & Fu 2022](../references.md#ji-liang-fu-2022)) keeps at least
the smoothest sector, so $\mathcal{K}$ is never empty. The face values
$\mathbf{U}_q = \mathbf{R}\,\mathbf{v}_q$ return through the right
eigenvectors.

**Adaptive cutoff.** Unless `C_T` fixes it,
([Liang, Shyy & Fu 2025](../references.md#liang-shyy-fu-2025); `teno::adaptive_CT`)

$$
\begin{equation}
m = \min\left(1, \max\left(0, \frac{\sigma_i - \sigma_L}{\sigma_U - \sigma_L}\right)\right), \qquad
g = (1 - m)^2(1 + 2m), \qquad
C_T = 10^{-\lfloor 10 - 4(1 - g)\rfloor},
\label{eq:ct}
\end{equation}
$$

with $\sigma_U$ = `troubled_upper` ($10^{-2}$): $C_T = 10^{-9}$ just above
$\sigma_L$, stepping to $10^{-6}$ at and above $\sigma_U$. A larger $C_T$
rejects the central polynomial more readily.

![Stencil selection for a density step at distance d from a triangle's centroid: smooth beyond about three cell sizes, sector stencils in the troubled band, the central polynomial again where the step crosses the cell](../images/theory/teno_selection-light.svg#gh-light-mode-only)
![Stencil selection for a density step at distance d from a triangle's centroid: smooth beyond about three cell sizes, sector stencils in the troubled band, the central polynomial again where the step crosses the cell](../images/theory/teno_selection-dark.svg#gh-dark-mode-only)

*A density step ($\rho$ from 1 to 1.5) at signed distance $d$ from the
centroid of a triangle (black) at order 5; $h = h_i$. Beyond about $3h$ the
step lies outside the central stencil (teal) and the cell is smooth. Once
the step enters the stencil, $\sigma_i \ge \sigma_L$ and the central share
falls far below $C_T$: the sectors on the step's far side are smooth, and
their polynomials are used (orange band). Where the step crosses the cell
itself, every candidate sees it and the central polynomial is the least
oscillatory (teal bands). From `tools/docs/teno.py`, which models the 2D
setup and eqs. $\eqref{eq:sigma}$ to $\eqref{eq:ct}$ for a scalar.*

### Admissibility and bounds

A cell any of whose face values has a non-positive density or pressure, or
a non-finite pressure, falls back to first order on all its faces (smooth
and troubled cells alike). With `bound_preserving = true`, a troubled cell
first scales the deviation of all its face values from the cell average by
one factor
([Zhang & Shu 2010](../references.md#zhang-shu-2010)),

$$
\begin{equation}
\mathbf{U}_q \leftarrow \overline{\mathbf{U}}_i + \theta\left(\mathbf{U}_q - \overline{\mathbf{U}}_i\right), \qquad
\theta = \max\left(0, \min_{q,\,v}\theta_{q,v}\right), \qquad
\theta_{q,v} =
\begin{cases}
\dfrac{M_v - v_i}{v_q - v_i} & v_q > v_i, \\[2ex]
\dfrac{m_v - v_i}{v_q - v_i} & v_q < v_i, \\[1ex]
1 & v_q = v_i,
\end{cases}
\label{eq:bound-preserving}
\end{equation}
$$

over the face points $q$ and $v \in \{\rho, p\}$, where $m_v$ and $M_v$ are
the extrema of $v$ over the cell and its face neighbors (boundary faces add
none). The scaled values are checked again, and an inadmissible cell still
falls back to first order.

Admissible face values do not make the cell update admissible, since a
troubled cell's face values come from different stencils and do not average
to the cell's mean. In planar single-gas runs `bound_preserving = true` also
limits the convective fluxes so that every forward-Euler stage keeps each
cell's density and pressure positive at $\mathrm{CFL} \le 0.5$
([Hu, Adams & Shu 2013](../references.md#hu-adams-shu-2013); details in the
[theory manual](../theory/finite_volume.md#reconstruction)).

### Face quadrature

The face points $\boldsymbol{\xi}_q$, set once
(`FaceReconstruction::init_face_quadrature_3d` and `TENO::init`), are exact
for the degree-$r$ polynomial over straight faces:

- 2D edges: Gauss-Legendre, 2 points for orders 3 and 4, 3 for orders 5
    and 6;
- 3D triangles: the Dunavant rule ([Dunavant 1985](../references.md#dunavant-1985))
    of degree 4 for orders 3 and 4, of degree 5 for orders 5 and 6;
- 3D quadrilaterals: tensor Gauss-Legendre, $2\times 2$ or $3\times 3$
    points, mapped bilinearly;
- curved faces: the flat rule mapped onto the face
    ([theory manual](../theory/finite_volume.md#curved-boundaries)).

The same points enter the Lebesgue constant $\eqref{eq:lebesgue}$.

## Setup and storage

**2D** runs the setup on the host, a cell at a time on host threads.
**3D** runs it on the device (`teno_setup::Setup3DHandle`): the geometry
(scales $h_c$, moments, bounding boxes, ranking metrics) once, then the
stencils, fits and Lebesgue constants of batches of cells sized to the free
device memory, a team of threads per cell on GPUs. Cells whose search
outgrows their scratch are redone on the host with more room. Every table is
computed with the same sequence of floating-point operations on the device
and on the host, so it is independent of the batch, the thread count and
where it ran. The log's "TENO setup" row gives the execution space and the
time of each phase.

The tables are packed without padding to the largest stencil: per cell, the
scale $h_i$, the means $\overline{\phi}_{l,i}$, the 2D matrix $\mathbf{S}$ or
the 3D moments, and per stencil entry its cell, its mirror face (or none)
and its column of $\mathbf{P}$. On GPUs, 32 consecutive cells form a slice
whose entries are interleaved, so that a warp reads consecutive words.
`single_precision_tables` (3D, single gas) keeps $\mathbf{P}$, and the moments' values, in single
precision on the device.

**Cache.** `cache_file` writes the tables after the setup and reads them back
on later runs (format version 11; one file per MPI rank). The file is used
only if its key matches: the options that change the tables, the precision,
the rank count and rank, and a hash of the mesh (nodes, connectivity,
boundary conditions, partition, curved geometry, periodic shifts,
axisymmetry).

## 3D stability

**Lebesgue bound.** The 3D bound is 4, not 10, for linear stability
(#126, #229). On lattices the equidistant-shell rule skips the stencil sizes
in between, so stencils within 10 were stable there. Where the lattice's ties
are broken, the first size within 10 can be a near-degenerate stencil. Ties
break at symmetry walls of tilings that are not mirror-symmetric (prisms
split along a diagonal, Kuhn tetrahedra): the mirror images make the wall a
seam where the split flips. They also break at such seams in the interior
and on jittered meshes. For example, prisms of aspect ratio 2 next to a side
wall take 38 entries with $\Lambda = 8.7$, against 50 entries and 3.4 in the
interior. The face values of such a stencil put up to 73% of their weight on
cells across the face (32% on the lattice), so the outflow value follows the
downstream cell. The linearized operator (acoustics at rest, HLLC) then has
growing modes; the largest real part of its eigenvalues, in units of the
sound speed over the box size, with the bound of 10 and of 4:

| Case | Bound 10 | Bound 4 |
|---|---|---|
| prisms, aspect ratio 2, 6 × 6 × 4, walls / side walls only, order 4 | 1.9 / 2.5 | < 1e-8 |
| same, order 6 | 0.50 / 2.5 | < 1e-8 |
| periodic prisms with a seam (12 × 4 × 8, aspect ratio 2, order 4) | 3.5 | < 1e-7 |
| jittered prisms (8³, jitter 0.1, periodic), order 6 | 3.4 | 1.4e-7 |
| jittered prisms, aspect ratio 1.5, side walls, order 6 | 0.32 | 7e-6 |
| Kuhn tetrahedra, aspect ratio 25, side walls, order 4 / 6 | 1.1 / 0.27 | 8e-8 / 0.022 |
| mixed tilings (4³), walls, order 4 / 6 | 0.19 / 0.08 | 8e-4 / 1e-7 |

Making only the near-degenerate cells first order removes these modes, and
in the walled boxes the energy is produced at their wall faces. Bounds of 5
to 7 do not remove them. Order 6 on tetrahedra and pyramids reaches only
$\Lambda = 4.3$ to 5, hence the slack of 1.25 above. That leaves one slower
mode, at order 6 on 25:1 Kuhn tetrahedra with side walls ($+0.022$; $+10^{-7}$
with the best stencil).

**Ranking metric.** Physical distance on thin cells takes the whole
wall-normal column first and resolves the other directions only through
small centroid offsets. That fit has full rank and Lebesgue constants within
the bound, but it amplifies grid-scale vortical modes: the linearized
operator (acoustics at rest, exact upwind flux) has real eigenvalues up to
$+13$ (sound speed over box size) on thin boxes, where every eigenvalue
should have a non-positive real part. The metric $\eqref{eq:metric}$ removes
them. Its cut-off of 2.5 lies above the spacing ratios
$\sqrt{\lambda_{\max}/\lambda_{\min}}$ that cell shapes alone produce on
regular tilings, which keep physical distance and so their stencils:

| Mesh (6³ to 10³ blocks, interior cells) | Spacing ratio |
|---|---|
| hexahedra | 1 |
| Kuhn tetrahedra (1 / 2 / 3 neighbor layers) | 2.13 / 2.04 / 2.02 |
| prisms (split along the xy diagonal) | 1.73 |
| pyramids | 1.57 |
| Kuhn tetrahedra stretched 2 / 6.25 / 25 : 1 | 3.15 / 9.1 / 36 |
| prisms stretched 2 / 6.25 / 25 : 1 | 2.25 / 7.0 / 28 |

A metric everywhere also stabilizes the thin cases, but it ranks the
stencils of the regular Kuhn tiling along its common cube diagonal and makes
them up to 8 times less accurate in the interior at order 5.

**Measured orders** (maximum error at the face quadrature points, symmetry
walls): hexahedra from $16^3$ to $24^3$, 3.83 and 4.85 for orders 4 and 5
(3 from $12^3$ to $16^3$: 2.87); Kuhn tetrahedra from $8^3$ to $12^3$: 2.92,
3.90 and 4.84 for orders 3, 4 and 5. Lowering the 3D bound from 10 to 4 kept
the rate from $9^3$ to $12^3$ for every cell type and order (hexahedra, Kuhn
tetrahedra, prisms, pyramids, mixed, jittered). Order-6 errors at $n = 12$
changed by +10% (hexahedra, walls only), +22% (prisms), +11% (tetrahedra),
−18% (pyramids) and −63% (jittered).

## Known limitations

- **Order 5 and up on curved boundaries.** With straight (polygonal) walls,
    the fine cylinder O-grid (`examples/cylinder`, 384 × 128 cells,
    stretched outer cells) blows up at order 5 (drag fluctuations of ±2.9
    at $Re = 100$), and the steady vortex between circular walls does not
    converge at any order ([curved boundaries](../design/curved_boundaries.md)).
    With curved walls (high-order Gmsh faces or `[[mesh.curved]]`) and the
    degree cap above, orders 5 and 6 are stable: fifth order in $L_1$ and
    fourth at the walls. The fine cylinder at order 5 keeps a small drag
    noise (rms $7\times 10^{-4}$) from the outermost ring's odd-even mode
    along the far field, which every far-field condition tried (`inlet_state`,
    `outlet_pressure`, `farfield`) shows, mirrored or not.

- **Thin cells.** Before the ranking metric, a 1% pressure pulse at rest in
    a 16 × 16 × 4 box of `cartesian_prism` cells of
    $0.0625\times 0.0625\times 0.01$ (symmetry walls) reached Mach 0.8 by
    $t = 2$ at order 3, and `cartesian_tet` cells of the same size Mach 1.2.
    The Lebesgue bound cured that order-3 prism case, but tetrahedra (order
    3, aspect ratio 1.5 and up) and prisms (order 4 with walls all around,
    aspect ratio 3 and up) kept growing modes until the metric. On the
    `examples/sphere_re300` mesh (`--scale 0.6`, 258k cells, prism layers of
    aspect ratio about 7), TENO3 once reached Mach 0.9 by $t = 1$ with a free
    stream at 0.2; it now stays at or below the potential-flow start's 0.30
    to $t = 10$ (6741 steps), at 2.1 times MUSCL's cost per step. A possible
    improvement: a metric from the second moment of the node positions
    within a Euclidean ball. The nodes of the Kuhn tiling form the cubic
    lattice, so they read isotropic and the cut-off could be lower; but on a
    ball of twice the cell's radius, stretched meshes read only 1.6 to 2.0 at
    6.25:1, since a ball holds the same density in every direction, so its
    radius would need to follow the spacing.

- **Pyramid lattices at orders 4 and 6.** The regular `cartesian_pyramid`
    lattice is linearly unstable at order 4 on $6^3$ periodic boxes even with
    the bound of 10 (largest real part $+2\times 10^{-3}$), and the bound of 4
    makes pure pyramid lattices worse:

    | Case | Bound 10 | Bound 4 |
    |---|---|---|
    | order 4, periodic, 4³ / 6³ | 7e-9 / 2e-3 | 8e-3 / 2e-2 |
    | order 4, walls, 4³ | 3.3e-3 | 1.1e-2 |
    | order 6, periodic, 4³ | 8e-9 | 1.6e-3 |
    | order 6, walls, 4³ | 4.7e-3 | 8.5e-6 |

    Order 4 on that lattice is stable only from 85 stencil entries
    ($\Lambda \approx 3.5$; the bound of 4 takes 57). Mixed meshes with
    pyramids improve with the bound (table above). Pure pyramid lattices are
    rare in practice; prefer orders 3 or 5 there.
