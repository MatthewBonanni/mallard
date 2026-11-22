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
| `src/numerics/teno.cpp` | Options, the cache, and the reconstruction kernels |
| `src/numerics/teno_setup.h`, `teno_setup.cpp` | The setup on the device: stencil search, ranking metric, moments, fits |
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
- images of cells whose foot on the line or plane lies on none of its
    gathered faces, of cells behind it, and, for a target cell behind it, all
    of its images (#382, #443). A facet of a straight-sided curved wall,
    extended, passes inside the next columns of cells, by about a cell height
    for the thin wall cells of an airfoil mesh; reflections across it put the
    wall-normal profile there at the wrong distance from the wall. At a closed
    trailing edge thinner than its wall cells, the cells on one side mirrored
    across the other side land next to the cells on the first. Both gave an
    odd-even mode in the wall pressure of the NACA 0012 tutorial mesh at
    order 3;
- in 2D, images deeper behind the line than the body under their foot: the
    path from the foot to the image crosses another boundary face, so the
    image lands past the far side of the body, among the flow there (#976).
    Near a closed trailing edge the body is thinner than the stencils, and
    the cells on one side mirrored across their own side land beyond the
    other side, where cells of the other side and of the wake are also in
    the stencil. On the NACA 0012 tutorial mesh this gave an odd-even mode
    in the wall pressure at order 4 from x = 0.9 to the trailing edge,
    largest at x = 0.975 (6% of the dynamic pressure at t = 20, against 1%
    for order 5 anywhere on the wall). The setup finds, for each mirror face,
    the parts of the other boundary faces in the strip behind it
    (`boundary_behind_faces`), whose least depth over the foot bounds the
    body; it searches every boundary face of the rank's mesh, so the far side
    must lie within its halo;
- boundaries between MPI ranks (`PARTITION` faces), which are not boundaries
    of the domain;
- curved wall and symmetry faces, which take no images unless
    `curved_mirrors = true`: a reflection across their plane matches the
    curved surface only to $O(h^2)$.

Coplanar boundary faces form one plane; each image takes the ghost state of
the boundary condition (applied to its cell's state) from the face of that
plane nearest to it (in 3D, nearest among the faces under its cell's foot), so
one plane may carry several conditions. A 3D search
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

**Pyramids at odd degrees.** At orders 4 and 6 ($r = 3$, 5), a pyramid
starts from
$n_0 = \max\left(\lceil f N_r\rceil, \lceil 2.5\,N_{r+1}\rceil\right)$
entries (`PYRAMID_ODD_FACTOR`), 85 and 208 instead of 38 and 110, and so does
a pyramid fitting the odd `curved_wall_degree` 3 at a curved wall. Smaller
stencils within the bound give pyramid lattices growing modes
([3D stability](#3d-stability)). Other cells, and pyramids at orders 3 and 5,
start from $\lceil f N_r\rceil$.

**Curved walls.** A stencil without mirror images at a curved wall is
one-sided there, and one-sided fits of degree 4 and 5 grow in time whatever
the bound or stencil depth. Cells whose first $n_0$ candidates include a
cell with a curved wall or symmetry face (without `curved_mirrors`) fit at
most degree `curved_wall_degree` (default 3), from
$\lceil f N_{r'}\rceil$ entries for that degree $r'$; their coefficients above
it are zero, so the run-time kernels are unchanged. A stencil that the
search grows past $n_0$ to such a cell is searched again (#1202): in 2D at
that degree; in 3D below that cell's position at the full degree, and only if
no size there fits, at the wall degree, so the full degree is kept where it
can be. No other stencil changes.

**Meshes one cell thick.** A cell whose central-stencil candidates (mirror
images aside) all lie level with it along a coordinate axis, as on a 2D strip
or a 3D slab or pencil one cell thick, has no data across that axis: its
mirror images only repeat its own column, and every monomial in that
coordinate makes the fit rank deficient. Its central and sector fits (`flat_axes`,
`fit_columns`) take only the monomials of the other axes, in a polynomial of
the same degree (so the design order holds along the strip), from
$\lceil f N'\rceil$ entries for their number $N'$; the coefficients of the
dropped monomials are zero, so the run-time kernels are unchanged. A strip
not aligned with an axis is not detected and still fails as below.

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
stencil; the terminal output counts these ("sector stencils cut by boundaries"), which
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
`troubled_capacity` of the cells (default 0.25 in 3D, 1 in 2D) to bound
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

**Order 4: faces that look away from a discontinuity.** At order 4 the
central polynomial is also rejected on a face whose own sector has the
smallest indicator of all the candidates (ties to round-off included, so
that mirror-image faces decide alike) while another sector is cut
($\gamma_k/\sum_{k'}\gamma_{k'} < C_T$). The downstream face of the cell a
shock is crossing is such a face. The cubic's indicator, an integral over the
cell, stays within the cut-off there (7.6 to 9.9 times the downstream
sector's for the $u_n + a$ wave of a Mach 3 shock), but at that face the cubic
undershoots the state ahead (a density of 0.68 and a pressure of 0.29 against
1.17 and 1.02 ahead), and the Riemann problem with the cell ahead then drains
it (#977). Lowest values over every step on 200 × 4 quadrilaterals
(`examples/sod`, `examples/shu_osher`):

| | Order 4 without the rule | Order 4 | Order 5 |
|---|---|---|---|
| Sod, density (0.125 ahead) | 0.1236 | 0.1249 | 0.1249 |
| Shu–Osher, pressure (1 ahead) | 0.9528 | 0.9976 | 0.9988 |

The rule does not act in smooth flow: every convergence study of the
[verification manual](../verification/index.md) is unchanged to the last
bit, and so is a convected density wave of 90% amplitude at 16 to 64 cells
per wavelength, on quadrilaterals and triangles, with 44 to 100% of the cells
troubled. Orders 3, 5 and 6 do not have the rule. The [order-4 study](../design/teno4_robustness.md#shocks-undershoots)
compares the orders' undershoots, and gives what the rule changes in order 4's
errors on shocks and the one undershoot it leaves.

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
none), widened by $10^{-3}\,|v_i|$ (`teno::BOUND_SLACK`). A smooth extremum
overshoots its neighbors' averages; where a quantity barely varies, such as
the pressure of a low-Mach flame (ripples of $10^{-4}\,p$), that overshoot is
small against the quantity but not against its range, and without the
widening $\theta$ fell to 0 in such cells: first order in the flame made it
4-6% too fast. The overshoot at a shock is a fraction of the jump, which the
widening leaves limited. The scaled values are checked again, and an
inadmissible cell still falls back to first order.

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

The setup runs on the device (`teno_setup::SetupHandle`): the geometry
(scales $h_c$ and ranking metrics; the rules of curved cells in 2D; moments
and bounding boxes in 3D) once, then the stencils, fits, Lebesgue constants
and, in 2D, smoothness-indicator matrices of batches of cells sized to the
free device memory: a thread per cell in 2D, a team of threads per cell on
GPUs in 3D. In 2D each cell's scratch is sized from the stencil sizes (twice
the cells the searches gather). On CUDA a batch is one wave of the kernel's
threads (as many as the GPU holds at once with the kernel's registers: 41,472
on an A100), within a quarter of the free memory; on other GPUs it takes as
many cells as fit what 16,384 cells take with twice that scratch. Meshes of
at least 16,384 cells share one batch size, so that the setup's workspace
does not grow with the mesh. Cells whose search
outgrows their scratch are redone on the host with more room. Every table is
computed with the same sequence of floating-point operations on the device
and on the host, so it is independent of the batch, the thread count and
where it ran. The terminal output's "TENO setup" row gives the execution space and the
time of each phase.

The tables are packed without padding to the largest stencil: per cell, the
scale $h_i$, the means $\overline{\phi}_{l,i}$, the 2D matrix $\mathbf{S}$ or
the 3D moments, and per stencil entry its cell and its column of
$\mathbf{P}$. A mirror image's entry instead points into a list of the
rank's images, which holds the cell and the mirror face. On GPUs, 32 consecutive cells form a slice
whose entries are interleaved, so that a warp reads consecutive words.
`single_precision_tables` (3D, single gas) keeps $\mathbf{P}$, and the moments' values, in single
precision on the device.

**Shared tables.** The cells of one stencil shape share one table of
pseudo-inverses (`teno_sharing.h`). The slots of a
stencil are ranked by their columns of $\mathbf{P}$ on a grid of $2^{-30}$ of
its largest entry, within the central stencil and within each sector, so that
equal stencils of a lattice, whose equidistant entries the search ranks
differently from cell to cell, list their entries in the same order. Two cells
have the same shape if their tables are equal on that grid in that order (on a
lattice they differ by about $5\times 10^{-14}$); the shape's table is that of
its cell with the lowest global id, agreed over the ranks at setup, so a cell's
table does not depend on the partition, the batches, the thread count or the
slice width. A shape whose entries straddle a grid line becomes two shapes,
which costs memory only. The tables of the shapes are packed like the cells'
stencils, in blocks of 32 interleaved tables on GPUs, and each cell reaches its
own through `pinv_index` (block and lane); on a mesh with a shape per cell this
is the cells' own packing. A cell keeps its own slots' cells and mirror faces.
The device setup ranks, hashes and compares the tables where it computes them;
only the hashes, and the tables that the shapes keep, reach the host, which
matches the hashes in cell order.
A caller that knows its stencil shapes, such as a structured block's offset
pattern, can key a shape itself (`TableStore::add_keyed`) instead of comparing
tables; such shapes are agreed over the ranks by their key in the same way.
The basis means of a cell and its moments (3D) or smoothness-indicator matrix
(2D) depend on its own geometry only, in its scaled frame, so they are equal
for the cells of a lattice and for every cell of the same shape and
orientation, whatever its size. They form one row per cell shape, in double precision, shared the same way
(equal on the grid, the lowest global id's, agreed over the ranks); each cell
reaches its row through `cell_shape`. The scale $h$ stays per cell.

**Cache.** `cache_file` writes the tables after the setup and reads them back
on later runs (format version 25; one file per MPI rank). The file is used
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

**Pyramids at odd degrees** (#236). On the `cartesian_pyramid` lattice the
odd degrees grow with every stencil within the Lebesgue bound below a size
found empirically, about $2.5\,N_{r+1}$ entries. The face jump of the
leading error of an odd-degree fit (of the even degree $r + 1$, which cancels
between cells that are point images of each other through the face) does not
explain it: the face-averaged jump grows with the stable sizes, and the
`cartesian_tri` lattice, whose faces are all point-symmetric, grows as well.
What decides is the sign of the scheme's lowest even term (see odd degrees on
other lattices, below). On the infinite lattice, the largest real part of the
eigenvalues of the linearized operator (acoustics at rest, HLLC; sound speeds
over the block size), by the size of the central stencil:

| Order 4: entries | 57 | 65 | 77 | 85 | 89 |
|---|---|---|---|---|---|
| $\Lambda$ | 3.95 | 3.96 | 3.69 | 3.52 | 3.54 |
| largest real part | 6.8e-3 | 8.8e-3 | 9.2e-4 | < 1e-8 | < 1e-8 |

| Order 6: entries | 145 | 165 | 205 | 209 |
|---|---|---|---|---|
| $\Lambda$ | 5.20 | 5.09 | 4.44 | 4.42 |
| largest real part | 2.1e-3 | 1.1e-3 | < 1e-8 | < 1e-8 |

(von Neumann analysis: the Jacobian of the right-hand side in the variables
of one block's six cells, by finite differences on a periodic $10^3$ box,
Fourier-transformed over the blocks at $12^3$ wavevectors.) The 85 entries of
order 4 are 2.5 $N_4$, hence the start from $2.5\,N_{r+1}$
([central stencil](#central-stencil)). At order 6 a start from $2.0\,N_6$ (166)
leads the search to 185 entries, which leave slow modes: $2.0\times10^{-7}$ on
the lattice and $6.8\times10^{-7}$ (sound speed over the box size) on the
walled 4³ box, against $3\times10^{-9}$ from 209.
The Lebesgue constants of the stable and the unstable stencils overlap (Kuhn
tetrahedra at order 4 still grow with 58 entries, $\Lambda = 3.84$; see below). A
finer scan of the wavevectors confirms 85 and 209 entries (largest real part
below $4\times10^{-9}$). With the pyramids' start from $2.5\,N_{r+1}$, the largest
real part on whole boxes (sound speed over the box size):

| Case | From $\lceil f N_r\rceil$ | From $2.5\,N_{r+1}$ |
|---|---|---|
| pyramids, order 4, periodic, 4³ / 6³ | 1.6e-2 / 4.1e-2 | < 1e-8 |
| pyramids, order 4, walls, 4³ / 6³ | 2.3e-2 / – | 5e-9 / 9e-9 |
| pyramids, order 6, walls, 4³ | 1.7e-5 | 3e-9 |
| pyramids, aspect ratio 6.25, 6 × 6 × 4, side walls, periodic in z, order 4 / 6 | 0.20 / 4e-8 | 4e-8 / 4e-8 |
| mixed tilings (4³), walls, order 4 / 6 | 1.6e-3 / 4e-9 | 5e-9 / 3e-9 |
| mixed tilings (4³), periodic, order 4 / 6 | 3e-9 / 3e-9 | 3e-9 / 3e-9 |

These units agree with the lattice's: 6.8e-3 per block on 6³ blocks is
4.1e-2 per box. A pyramid's stencils hold 85 entries instead of 57 at order 4
and 209 instead of 145 at order 6. A pyramid needs at least 85 or 208 cells
and mirror images within reach of its search, against 38 and 110 for other
cells; on smaller meshes setup stops with "could not build a full-rank large
stencil (mesh too small for this order?)".

**Odd degrees on other lattices** (#1136). Scanning the Bloch symbol finely
near $k = 0$ and near the grid scale, rather than at the box's own
wavevectors, shows slow growing modes on other lattices at the even orders.
Largest real part (sound speeds over the block size), by central stencil
size; the defaults are the first entries:

| Lattice | Order | Entries: largest real part |
|---|---|---|
| Kuhn tetrahedra | 4 | 46: 1.3e-5; 58: 5.0e-7; 66: 5.2e-7; 78 and 86: < 2e-9 |
| `cartesian_tri` (squares split in two) | 4 | 18: 1.25e-5; 24 to 30: < 2e-9 |
| squares split into four triangles at their centers | 4 | 18: 1.0e-6; 23: 1.6e-7; 27: < 1e-9 |
| hexahedra | 6 | 122: 5.25e-5 |
| squares split into four triangles | 6 | 42: 2.8e-8 |

Hexahedra at order 4, prisms at orders 4 and 6, Kuhn tetrahedra at order 6,
pyramids with the start above, and 2D quadrilaterals and `cartesian_tri` at
order 6 stay below $4\times10^{-9}$.

The order-4 modes are long acoustic waves. The eigenvalue's imaginary part is
the exact $2\pi\lvert k\rvert$, and its real part grows as $(kh)^6$ up to a
cut-off of 0.06 to 0.13 cycles per block, beyond which the mode is damped.
The scheme's lowest even term, $h^5\partial^6$, is then anti-dissipative. An
odd-degree fit's even error of degree $r + 1$ cancels in the face jump
only between point images, but the `cartesian_tri` lattice, where it cancels
at every face, grows as much as the Kuhn lattice: the sign of that term is a
sum over the whole lattice, which no check of one face or one stencil
decides. Larger stencils shrink both the growth and the cut-off before the
term changes sign, so the band of growing waves moves toward $k = 0$.

The hexahedral mode at order 6 is a vortical mode: transverse velocity
alternating cell by cell along one axis ($k = 1/2$ there) and slowly varying
across it, growing as the square of the transverse wavenumber up to about
0.07 cycles per block. HLLC at rest does not damp shear, so only the
reconstruction can.

Both bands are narrow: the analysis behind the
[pyramid tables](#3d-stability) sampled $12^3$ wavevectors, multiples of
$1/12$, which fall outside them, and a test of
$\max\lVert\exp(5000\,S)\rVert$ sees $1.3\times10^{-5}$ only as a 6% rise.
Detecting them takes the eigenvalues at wavevectors interpolated between the
box's (the Jacobian's columns reach only a few blocks), down to about $0.02$
cycles per block. The growth is slow: $1.3\times10^{-5}$ e-folds once in
$8\times10^4\,h/c$, about 800 acoustic crossings of a 100-cell domain, and
the hexahedral order-6 mode ($5.25\times10^{-5}$) in about 190.

The growth comes from the Riemann solver's jump term alone: with the left
and right states averaged before HLLC (a central flux, in a scratch build),
the long waves are neutral on all three lattices (below $5\times10^{-9}$).
The jump of the unweighted degree-3 fits is slightly anti-dissipative for
them. Its sign is erratic in the stencil size, not a threshold as on
pyramids (order 4, largest real part):

| Entries | `cartesian_tri` | squares split in four |
|---|---|---|
| 14, 16 | < 1e-9 | 1.75 |
| 18 (default) | 1.25e-5 | 1.0e-6 |
| 19 or 20 | 1.0e-5 (20) | 4.7e-4 (19) |
| 23 or 24 | < 1e-9 (24) | 1.6e-7 (23) |
| 26 to 31 | < 1e-9 | < 1e-9 (27, 31) |

Larger starts, like the pyramids', therefore stabilize these lattices only
from about 27 entries on triangles and 78 on Kuhn tetrahedra, and they cost
accuracy. Maximum density error of the
[advection study](../verification/index.md) on the finer mesh, against the
default: triangles +46% from 24 entries and +136% from 27; Kuhn tetrahedra
+11% (58 entries), +24% (66) and +37% (78). Weighting the fit by the inverse
centroid distance lowers the triangles' errors (advection −16%) and the
growth to $1.5\times10^{-7}$ and $5\times10^{-8}$, but does not remove it,
and on Kuhn tetrahedra it leaves $2.4\times10^{-7}$. Neither is adopted,
and neither was tried on the hexahedral mode. A criterion tied to the sign of
the jump's dissipation, covering every cell shape, is the planned fix (#1268).

With SSPRK3 the time integrator damps the order-4 modes at `cfl = 1`: the
amplification factor of the full step leaves no growth on either triangle
lattice (it appears below `cfl` of about 0.8 on `cartesian_tri` and between
0.6 and 0.7 on squares split in four) and $5.5\times10^{-7}$ on Kuhn tetrahedra. The tests
`TENOLattice` (2D) and `TENO3DKuhnLattice` record the semi-discrete rates
above and fail when they change.

**Graded hexahedra** (#1232). Orders 4 and 6 grow on hexahedra graded along
one axis from 5% cell-to-cell growth, the default order 5 and order 3 do not.
On a periodic box of 24 cells along x with `stretching = [β, 0, 0]` and cells
of the mean spacing across, the fastest mode is a sound wave along the grading,
uniform across, of 4 to 6 cells per wavelength. Largest real part of the
eigenvalues of the linearized operator (acoustics at rest, HLLC; sound speeds
over the mean cell size), the column as the block of a Bloch analysis:

| β (largest cell-to-cell ratio) | 0.6 (1.05) | 0.9 (1.11) | 1.5 (1.25) |
|---|---|---|---|
| order 3, order 5 | < 1e-9 | < 1e-9 | < 1e-9 |
| order 4 | 7.5e-7 | 5.7e-2 | 5.1e-2 |
| order 6 | 5.7e-2 | 4.7e-2 | 5.1e-5 |

The cause is the ranking by physical distance. On the uniform box every cell
takes the same 56 entries, symmetric along x (5, 13, 20, 13 and 5 cells at
x offsets −2 to 2). Grading splits each equidistant shell of the lattice by
the side and the direction of its members: at β = 1.5 neighbors take 39 to 79
entries, mostly three layers along x plus single cells two or three layers
away on the finer side only, and the size jumps between neighbors (39, 43, 63
entries in a row). An odd degree's own dissipation does not cover the
mismatch between the face values of such neighbors. Ranking experiments (a
scratch build, both orders, β = 0.6 / 0.9 / 1.5):

| Ranking | Order 4 | Order 6 |
|---|---|---|
| physical distance | 7.5e-7 / 5.7e-2 / 5.1e-2 | 5.7e-2 / 4.7e-2 / 5.1e-5 |
| no cut within 1% of the cut distance | < 1e-9 / 3.9e-2 / 8.1e-3 | 5.7e-2 / 4.1e-2 / 4.1e-5 |
| offsets scaled per axis by $2s_i/(s_i + s_j)$ | < 1e-9 / 4.1e-2 / 3.3e-3 | 3.2e-2 / 4.2e-2 / 1.1e-6 |
| both | < 1e-9 / < 1e-9 / 1.4e-9 | < 1e-9 / < 1e-9 / 3.8e-7 |
| both, with $s$ the cube root of the volume | < 1e-9 / 4.0e-2 / 4.6e-2 | 5.9e-3 / 4.1e-2 / 1.1e-2 |

with $s_i$ and $s_j$ the widths of the target cell and of the candidate
along each axis, a distance in which a geometric grading leaves every shell
whole. Neither part removes the growth alone, and a scalar size does not:
the ranking has to undo the grading direction by direction and close the
shells it nearly restores. None of it is adopted:

- Widths per axis exist only for hexahedra aligned with the axes. The
    tensor spacing of the vertex neighbors (the [ranking metric's](#candidates)
    second moment) differs between the cells of the Kuhn and pyramid tilings,
    so it changes their stencils: the Kuhn rate of #1136 rises from 1.20e-5 to
    1.48e-5, and pyramids at order 4 grow again. A spacing summed over the
    centroids around each node of the cell keeps every regular tiling's
    stencils, but it needs a 2% closure, still grows at β = 1.8 (2.1e-2 at
    order 4, 1.0e-3 at order 6), and fails to set up order 6 on walled
    $8^3$ and $16^3$ boxes.
- Accuracy is not kept everywhere. Order 4's face-value error on walled
    graded boxes changes by −3% to +7%; applied to every degree it raises
    order 5's by 35% to 100% at `stretching = [1, 1, 1]`.
- Each key needs the spacing of every candidate, which a partition's
    outermost halo layer lacks.

`TENO3DGradedLattice` records the rates of order 4 at β = 0.9 and 1.5 and of
order 6 at β = 0.6 and 0.9, and keeps orders 3 and 5 below 1e-7 at β = 1.5;
a fix gives orders 4 and 6 that bound.

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
    fourth at the walls. On the fine cylinder with curved walls and far
    field, order 5's drag is as smooth as order 3's
    ([curved boundaries](../design/curved_boundaries.md#cylinder-at-re-100)).

- **Order 4.** `high_order` stays at order 5 because order 4 fails two
    robustness checks that order 5 passes
    ([order-4 study](../design/teno4_robustness.md)): it amplifies an
    outgoing acoustic wave on a radially stretched O-grid (#975) and grows
    acoustic modes on graded hexahedra from 5% cell-to-cell grading, as
    order 6 does ([graded hexahedra](#3d-stability), #1232). Its undershoots
    ahead of plane shocks, once about 10 times order 5's (#977), are down to
    order 5's on quads ([troubled cells](#troubled-cells)); where shocks start
    from a corner it still undershoots. The odd-even wall
    pressure it left at the closed trailing edge of the NACA 0012 tutorial
    mesh (#976) came from mirror images past the far side of the body, which
    the setup now rejects ([candidates](#candidates)).

- **Thin cells** (#427). Before the ranking metric, a 1% pressure pulse at rest in
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

- **Slow growing modes at the even orders on lattices.** The pyramid rule
    above is tied to the cell type. With their default stencils, Kuhn
    tetrahedra and `cartesian_tri` triangles at order 4 grow at about
    $1.3\times 10^{-5}$ sound speeds over the block size, squares split into
    four triangles at $10^{-6}$, and hexahedra at order 6 at
    $5.25\times10^{-5}$ (#1136;
    [3D stability](#3d-stability)). The cause is the Riemann solver's jump
    term; larger stencils remove the order-4 modes only at 37% to 136% more
    error, so the defaults stay, and SSPRK3 at `cfl = 1` damps them on
    triangles.
