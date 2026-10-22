# What's new in Mallard's numerics

Most of Mallard's numerics are published methods, implemented as their
papers describe them and cited on the page that describes each one
([References](../references.md)). This page says which parts are not: what
Mallard adds to the literature, what it combines in a way we did not find
published, and what it only implements. It is meant to be conservative:
where we were not sure, an item is classed lower.

**Classes.**

- **Original**: we found no published method that does this (the searches
  are listed in the audit behind this page).
- **Original extension**: a published method, extended in a specific way
  that we did not find published; the row says exactly what is new.
- **Adapted**: a published method, with engineering changes that are
  Mallard's own but not a new method.
- **Standard**: a published method, implemented as published.

"Not found" means not found by the searches made (Google Scholar, arXiv,
Crossref and journal pages, October 2026); it is not proof that no one has
done it. Corrections are welcome as issues.

## Summary

| Feature | Class | Closest prior work | What Mallard adds | Evidence |
|---|---|---|---|---|
| Central-stencil size chosen by a Lebesgue-constant bound | Original | Lebesgue constants rating spectral-volume partitions (Wang 2002; Chen 2005); eigenvalue-based reconstruction stabilization (Zangeneh & Ollivier-Gooch 2019) | A per-cell, a priori rule: grow the least-squares stencil until the reconstruction's Lebesgue constant at the face points is at most 10 (2D) or 4 (3D) | [Lebesgue bound](#lebesgue-bound) |
| Stencil ranking in a mesh-spacing metric on thin 3D cells | Original extension | Local tangential-normal frames for least squares on anisotropic meshes (Jalali & Ollivier-Gooch 2013, 2017) | Candidates ranked by the unit-determinant inverse of the second moment of the vertex-neighbor offsets, only where that metric's ratio exceeds 2.5 | [Ranking metric](#ranking-metric) |
| TENO-E on 3D hexahedra, prisms, pyramids and mixed meshes, orders 3 to 6 | Original extension | TENO on unstructured meshes (Ji, Liang & Fu 2022: 2D, one 3D tetrahedral test); TENO-E (Liang, Shyy & Fu 2025); WENO on 3D mixed elements (Tsoutsanis et al. 2011) | The TENO-E algorithm on every 3D cell type without splitting cells, with the two items above, mirror stencils and curved walls; stability measured on the linearized operator | [TENO-E in 3D](#teno-e-in-3d) |
| Species reconstructed with the flow's contact-field stencil choice | Original extension | Mass-flux upwinding of species (Larrouturou 1991); primitive-variable reconstruction at material interfaces (Johnsen & Colonius 2006) | Every scalar takes, per face, the stencil TENO chose for the entropy/contact field (one byte), so all species share linear weights and one bound-preserving factor | [Species stencils](#species-stencils) |
| Double flux with TENO on unstructured meshes | Original extension (narrow) | Double flux (Abgrall & Karni 2001; Billet & Abgrall 2003); with WENO on structured grids (Houim & Kuo 2011); in DG on unstructured meshes (Lv & Ihme 2014; Johnson & Kercher 2020) | The combination with high-order TENO finite volumes on mixed meshes, and wave speeds from each side's own frozen pair so both fluxes carry the same mass flux | [Double flux](#double-flux) |
| SIMPLER splitting with an explicit integrator: stability polynomial, and the instability of non-dissipative fluxes | Original (analysis) | SIMPLER (Wu, Ma & Ihme 2019) and balanced splitting (Speth et al. 2013), analysed with exact substeps | The composite polynomial with SSPRK3, which exceeds 1 everywhere on the imaginary axis, so a central flux is unstable at any step | [SIMPLER stability](#simpler-stability) |
| RHLL blended toward HLL by a pressure-jump weight | Original extension (minor) | Rotated hybrid (Nishikawa & Kitamura 2008); the weight of AUSMPW+ (Kim, Kim & Rho 2001); pressure-based shock sensing in RoeM (Kim, Kim & Rho 2003); Rotated-RoeM with a Mach-number weight (Choi et al. 2024) | The blend of the rotated flux with face-normal HLL by `min(p_L/p_R, p_R/p_L)^3`, which removes the stagnation-line carbuncle on tetrahedra | [RHLL blend](#rhll-blend) |
| Mirror-image stencil entries at straight boundaries | Adapted | Mirrored ghost cells in high-order unstructured stencils (Yang, Pan & Xu 2023) | Coplanar faces as one plane with several conditions, moments reflected degree by degree, images not counted toward the stencil size | [TENO-E details](teno_e.md#candidates) |
| Curved boundaries for k-exact reconstruction | Adapted | Ollivier-Gooch & Van Altena 2002; Krivodonova & Berger 2006 | Quadrature area vectors shifted to sum exactly to the face's; degree cap 3 next to curved walls | [Curved boundaries](../design/curved_boundaries.md) |
| Hybrid KEEP / Riemann flux with a Ducros sensor | Standard | Kuya, Totani & Kawai 2018; Ducros et al. 1999; Bhagatwala & Lele 2009; Ma, Lv & Ihme 2017 | Upwind floor and the rule for wall faces | [LES design](../design/les.md#43-low-dissipation-convective-flux-decision-7) |
| Low-Mach correction | Adapted | Thornber et al. 2008; cutoff as Weiss & Smith 1995 | Cutoff Mach number | [Overview](overview.md#low-mach-correction) |
| Dynamic thickened-flame efficiency | Adapted | Charlette et al. 2002 (II); Wang, Boileau & Veynante 2011; Volpiani et al. 2016 | Test filter as repeated vertex-neighbor averages sized by their covariance; exact global sums | [LES design](../design/les.md#dynamic-efficiency-294) |
| Conservative remap on an exact supermesh | Adapted | Farrell & Maddison 2011; Menon & Schmidt 2011 | Intersections as lists of clipped simplices; results bitwise independent of both partitions | [Remap design](../design/remap.md) |
| Results bitwise independent of rank and thread counts | Adapted | Order-invariant integer sums (Hallberg & Adcroft 2014); reproducible summation (Demmel & Nguyen 2015) | Applied throughout the solver: per-face fluxes gathered in a fixed order, fixed-point global sums, partition-independent setup and restarts | [Performance](../design/performance.md) |
| Wall-model exchange point located by a walk | Standard | Lawson 1977; Devillers, Pion & Teillaud 2002; exchange height of Kawai & Larsson 2012 | | [Wall models](../theory/wall_models.md#exchange-point) |

The TENO-E setup on the GPU, its incremental rebuild and its cache are
implementation, not numerics, and are not counted here
([performance](../design/performance.md#4-teno-setup-on-the-device)).

## Original and extended items

### Lebesgue bound

Mallard fits each cell's central polynomial by least squares over a stencil
of nearby cells (Barth & Frederickson 1990), as TENO on unstructured meshes
does with a fixed size of about twice the number of coefficients
(Ji, Liang & Fu 2022). A fixed size is not enough: a stencil can have full
rank yet resolve one direction only through small centroid offsets, and then
amplify the cell averages, and the truncation error with them. Mallard
grows the stencil one entry at a time and takes the first size whose
Lebesgue constant at the cell's face quadrature points,

$$
\begin{equation}
\Lambda = \max_q\left(\Bigl\lvert 1 - \sum_s c_{qs}\Bigr\rvert + \sum_s\lvert c_{qs}\rvert\right),
\label{eq:lebesgue}
\end{equation}
$$

is at most 10 (2D) or 4 (3D), where $c_{qs}$ is the weight of cell $s$'s
average in the face value at point $q$
([TENO-E details](teno_e.md#central-stencil)). In 3D, if no size reaches
the bound, it takes the smallest within 1.25 times the best found.

Lebesgue constants have rated reconstruction operators before, to choose the
control-volume partitions of the spectral volume method (Wang 2002;
Chen 2005), and eigenvalue analysis has been used to modify reconstructions
until the discrete operator is stable (Zangeneh & Ollivier-Gooch 2019). We
did not find the Lebesgue constant used to choose the size of a
reconstruction stencil cell by cell. The bounds themselves are empirical.

**Evidence.** The largest real part of the eigenvalues of the linearized
operator (acoustics at rest, HLLC), with the bound of 10 and of 4:
1.9 to 3.5 against below $10^{-7}$ on prism meshes with walls, seams or
jitter, and 1.1 against $8\times 10^{-8}$ on 25:1 Kuhn tetrahedra at order 4.
Bounds of 5 to 7 do not remove the growing modes. Lowering the bound kept
the measured convergence rates of every cell type and order
([3D stability](teno_e.md#3d-stability)). Known exception: pure pyramid
lattices at order 4 are less stable with the bound of 4.

### Ranking metric

On thin cells, ranking candidates by physical distance takes the whole
wall-normal column first. Mallard ranks them instead in the metric of the
local mesh spacing,

$$
\begin{equation}
\lVert\mathbf{d}\rVert_\mathbf{M}^2 = \mathbf{d}^T\mathbf{M}\,\mathbf{d}, \qquad
\mathbf{M} = (\det\mathbf{C})^{1/3}\,\mathbf{C}^{-1}, \qquad
\mathbf{C} = \sum_j \mathbf{d}_j\mathbf{d}_j^T,
\label{eq:metric}
\end{equation}
$$

with $\mathbf{d}_j$ the offsets of the cell's vertex neighbors (mirrored at
boundary faces), on cells where
$\sqrt{\lambda_{\max}/\lambda_{\min}}$ of $\mathbf{C}$ exceeds 2.5, a value
above what cell shapes alone give on regular tilings
([ranking metric](teno_e.md#3d-stability)). Least squares on anisotropic
meshes has been improved by fitting in a local tangential-normal frame
(Jalali & Ollivier-Gooch 2013, 2017); we did not find a metric of this kind
used to rank stencil candidates.

**Evidence.** Without the metric, thin boxes have eigenvalues of the
linearized operator up to $+13$; with it, none above zero. On the
`examples/sphere_re300` mesh (prism layers of aspect ratio about 7), TENO3
once reached Mach 0.9 at $t = 1$; it now stays at or below 0.30 to $t = 10$.

### TENO-E in 3D

TENO on unstructured meshes was introduced in 2D, with one 3D test on
tetrahedra, other cell types split into triangles or tetrahedra
(Ji, Liang & Fu 2022), and TENO-E adds the adaptive dissipation that Mallard
uses (Liang, Shyy & Fu 2025). Mallard runs TENO-E on tetrahedra, hexahedra,
prisms, pyramids and mixed meshes without splitting the cells, at orders
3 to 6, with cell moments from one tetrahedral decomposition per cell and
the stencil rules above. WENO-type schemes on 3D mixed elements are
established (Tsoutsanis, Titarev & Drikakis 2011); what is new is TENO-E
there, with its stability measured and the stencil rules it needed.

**Evidence.** Measured orders on symmetry-walled boxes: 3.83 and 4.85 for
orders 4 and 5 on hexahedra, 2.92, 3.90 and 4.84 for orders 3 to 5 on Kuhn
tetrahedra ([measured orders](teno_e.md#3d-stability)); polynomial exactness
on prisms, pyramids and mixed meshes in the test suite.

### Species stencils

In a gas mixture, TENO selects stencils for the flow variables only. Each
species then takes, on each face, the stencil that TENO chose for the
entropy/contact characteristic field there, the wave that carries
composition, stored as one byte per face side
([chemistry design](../design/chemistry.md#reconstruction)). All species
are then linear in the same weights, so their face values sum to one, and
one Zhang–Shu factor per cell keeps them in $[0, 1]$. Species reconstruction
costs only dot products with weights already computed. The species flux is
Larrouturou's (1991). We did not find the stencil choice of one
characteristic field reused for passive scalars.

### Double flux

Double flux (Abgrall & Karni 2001; Billet & Abgrall 2003) keeps pressure and
velocity uniform across contacts between gases of different $\gamma$. It has
been combined with WENO on structured grids (Houim & Kuo 2011) and with DG
on unstructured meshes (Lv & Ihme 2014; Johnson & Kercher 2020). Mallard
combines it with TENO-E on unstructured mixed meshes, for all five Riemann
solvers. Both of a face's fluxes estimate their wave speeds from each side's
own frozen $(\gamma, e_0)$: with each flux's own pair, HLL, Rusanov and RHLL
gave the two fluxes different mass fluxes at a contact and perturbed the
pressure by 0.1%. Contacts keep $p$ and $\mathbf{u}$ uniform to $10^{-12}$
with MUSCL and TENO5 ([chemistry design](../design/chemistry.md#reconstruction)).

### SIMPLER stability

SIMPLER balanced splitting (Wu, Ma & Ihme 2019) gives every cell the flow's
tendency during the reaction substep, then integrates the flow over half a
step. With an explicit integrator for the flow, the step is a new
Runge-Kutta method; with SSPRK3 its stability polynomial is

$$
\begin{equation}
P(z) = 1 + z + \frac{z^2}{2} + \frac{z^3}{8} + \frac{z^4}{48}, \qquad
\lvert P(iy)\rvert^2 = 1 + \frac{y^4}{24} - \frac{y^6}{192} + \frac{y^8}{2304} > 1 \quad (y \ne 0),
\label{eq:simpler}
\end{equation}
$$

so a non-dissipative operator, such as the central flux of the hybrid
scheme, is unstable at any time step, while upwind operators gain a larger
real-axis limit ([chemistry design](../design/chemistry.md#simpler-balanced-splitting-option)).
The published analyses of balanced splitting (Speth et al. 2013;
Wu, Ma & Ihme 2019) treat substeps solved exactly for dissipative
operators. The derivation is elementary; we did not find it published.

**Evidence.** A 2D Taylor-Green vortex grows exponentially with SIMPLER and
the hybrid flux (10 times per 50 steps with MUSCL) and stays bounded with
Strang splitting or the Riemann flux. Mallard rejects the combination.

### RHLL blend

Nishikawa and Kitamura's rotated hybrid applies HLL along the velocity jump
and Roe across it. On tetrahedra this turns HLL toward the shock normal on
every face that crosses a shock and grows a carbuncle on the stagnation line
of blunt bodies. Mallard blends the rotated flux with face-normal HLL,

$$
\begin{equation}
\hat{\mathbf{F}} = w\,\hat{\mathbf{F}}_\mathrm{rot} + (1 - w)\,\hat{\mathbf{F}}_\mathrm{HLL}(\mathbf{n}), \qquad
w = \min\left(\frac{p^L}{p^R}, \frac{p^R}{p^L}\right)^3,
\label{eq:rhll}
\end{equation}
$$

with the pressure weight of AUSMPW+ (Kim, Kim & Rho 2001); contacts and shear
layers keep the rotated flux ([theory](../theory/fluxes.md#rhll)).
Pressure-based shock sensing already controls the dissipation of Roe-type
fluxes (RoeM, Kim, Kim & Rho 2003), and the rotated hybrid has been
combined with RoeM through a Mach-number weight (Choi et al. 2024); the blend
with the rotated hybrid is the only new part, and it is small.

## References

- R. Abgrall and S. Karni, *J. Comput. Phys.* 169, 594–623 (2001). [doi:10.1006/jcph.2000.6685](https://doi.org/10.1006/jcph.2000.6685)
- T. J. Barth and P. O. Frederickson, AIAA Paper 90-0013 (1990). [doi:10.2514/6.1990-13](https://doi.org/10.2514/6.1990-13)
- A. Bhagatwala and S. K. Lele, *J. Comput. Phys.* 228, 4965–4969 (2009). [doi:10.1016/j.jcp.2009.04.009](https://doi.org/10.1016/j.jcp.2009.04.009)
- G. Billet and R. Abgrall, *Comput. Fluids* 32, 1473–1495 (2003). [doi:10.1016/S0045-7930(03)00004-5](https://doi.org/10.1016/S0045-7930%2803%2900004-5)
- F. Charlette, C. Meneveau and D. Veynante, *Combust. Flame* 131, 181–197 (2002). [doi:10.1016/S0010-2180(02)00401-7](https://doi.org/10.1016/S0010-2180%2802%2900401-7)
- Q.-Y. Chen, *J. Sci. Comput.* 29, 299–319 (2006; online 2005). [doi:10.1007/s10915-005-9009-0](https://doi.org/10.1007/s10915-005-9009-0)
- S. Choi, D. Kim, J. Park and J. Park, *J. Comput. Phys.* 505, 112913 (2024). [doi:10.1016/j.jcp.2024.112913](https://doi.org/10.1016/j.jcp.2024.112913)
- J. Demmel and H. D. Nguyen, *IEEE Trans. Comput.* 64, 2060–2070 (2015). [doi:10.1109/TC.2014.2345391](https://doi.org/10.1109/TC.2014.2345391)
- O. Devillers, S. Pion and M. Teillaud, *Int. J. Found. Comput. Sci.* 13, 181–199 (2002). [doi:10.1142/S0129054102001047](https://doi.org/10.1142/S0129054102001047)
- F. Ducros et al., *J. Comput. Phys.* 152, 517–549 (1999). [doi:10.1006/jcph.1999.6238](https://doi.org/10.1006/jcph.1999.6238)
- P. E. Farrell and J. R. Maddison, *Comput. Methods Appl. Mech. Eng.* 200, 89–100 (2011). [doi:10.1016/j.cma.2010.07.015](https://doi.org/10.1016/j.cma.2010.07.015)
- R. Hallberg and A. Adcroft, *Parallel Comput.* 40, 140–143 (2014). [doi:10.1016/j.parco.2014.04.007](https://doi.org/10.1016/j.parco.2014.04.007)
- R. W. Houim and K. K. Kuo, *J. Comput. Phys.* 230, 8527–8553 (2011). [doi:10.1016/j.jcp.2011.07.031](https://doi.org/10.1016/j.jcp.2011.07.031)
- A. Jalali and C. Ollivier-Gooch, AIAA Paper 2013-2565 (2013). [doi:10.2514/6.2013-2565](https://doi.org/10.2514/6.2013-2565)
- A. Jalali and C. Ollivier-Gooch, *Comput. Fluids* 143, 32–47 (2017). [doi:10.1016/j.compfluid.2016.11.004](https://doi.org/10.1016/j.compfluid.2016.11.004)
- F. Ji, T. Liang and L. Fu, *J. Sci. Comput.* 92, 61 (2022). [doi:10.1007/s10915-022-01925-5](https://doi.org/10.1007/s10915-022-01925-5)
- R. F. Johnson and A. D. Kercher, *J. Comput. Phys.* 423, 109826 (2020). [doi:10.1016/j.jcp.2020.109826](https://doi.org/10.1016/j.jcp.2020.109826)
- E. Johnsen and T. Colonius, *J. Comput. Phys.* 219, 715–732 (2006). [doi:10.1016/j.jcp.2006.04.018](https://doi.org/10.1016/j.jcp.2006.04.018)
- S. Kawai and J. Larsson, *Phys. Fluids* 24, 015105 (2012). [doi:10.1063/1.3678331](https://doi.org/10.1063/1.3678331)
- K. H. Kim, C. Kim and O.-H. Rho, *J. Comput. Phys.* 174, 38–80 (2001). [doi:10.1006/jcph.2001.6873](https://doi.org/10.1006/jcph.2001.6873)
- S.-s. Kim, C. Kim, O.-H. Rho and S. K. Hong, *J. Comput. Phys.* 185, 342–374 (2003). [doi:10.1016/S0021-9991(02)00037-2](https://doi.org/10.1016/S0021-9991%2802%2900037-2)
- L. Krivodonova and M. Berger, *J. Comput. Phys.* 211, 492–512 (2006). [doi:10.1016/j.jcp.2005.05.029](https://doi.org/10.1016/j.jcp.2005.05.029)
- Y. Kuya, K. Totani and S. Kawai, *J. Comput. Phys.* 375, 823–853 (2018). [doi:10.1016/j.jcp.2018.08.058](https://doi.org/10.1016/j.jcp.2018.08.058)
- B. Larrouturou, *J. Comput. Phys.* 95, 59–84 (1991). [doi:10.1016/0021-9991(91)90253-H](https://doi.org/10.1016/0021-9991%2891%2990253-H)
- C. L. Lawson, in *Mathematical Software III*, 161–194 (1977). [doi:10.1016/B978-0-12-587260-7.50011-X](https://doi.org/10.1016/B978-0-12-587260-7.50011-X)
- T. Liang, W. Shyy and L. Fu, *J. Sci. Comput.* 104, 1 (2025). [doi:10.1007/s10915-025-02918-w](https://doi.org/10.1007/s10915-025-02918-w)
- Y. Lv and M. Ihme, *J. Comput. Phys.* 270, 105–137 (2014). [doi:10.1016/j.jcp.2014.03.029](https://doi.org/10.1016/j.jcp.2014.03.029)
- P. C. Ma, Y. Lv and M. Ihme, *J. Comput. Phys.* 340, 330–357 (2017). [doi:10.1016/j.jcp.2017.03.022](https://doi.org/10.1016/j.jcp.2017.03.022)
- S. Menon and D. P. Schmidt, *Comput. Methods Appl. Mech. Eng.* 200, 2797–2804 (2011). [doi:10.1016/j.cma.2011.04.025](https://doi.org/10.1016/j.cma.2011.04.025)
- H. Nishikawa and K. Kitamura, *J. Comput. Phys.* 227, 2560–2581 (2008). [doi:10.1016/j.jcp.2007.11.003](https://doi.org/10.1016/j.jcp.2007.11.003)
- C. Ollivier-Gooch and M. Van Altena, *J. Comput. Phys.* 181, 729–752 (2002). [doi:10.1006/jcph.2002.7159](https://doi.org/10.1006/jcph.2002.7159)
- M. A. Speth, W. H. Green, S. MacNamara and G. Strang, *SIAM J. Numer. Anal.* 51, 3084–3105 (2013). [doi:10.1137/120878641](https://doi.org/10.1137/120878641)
- B. Thornber et al., *J. Comput. Phys.* 227, 4873–4894 (2008). [doi:10.1016/j.jcp.2008.01.036](https://doi.org/10.1016/j.jcp.2008.01.036)
- P. Tsoutsanis, V. A. Titarev and D. Drikakis, *J. Comput. Phys.* 230, 1585–1601 (2011). [doi:10.1016/j.jcp.2010.11.023](https://doi.org/10.1016/j.jcp.2010.11.023)
- P. S. Volpiani, T. Schmitt and D. Veynante, *Combust. Flame* 174, 166–178 (2016). [doi:10.1016/j.combustflame.2016.08.007](https://doi.org/10.1016/j.combustflame.2016.08.007)
- G. Wang, M. Boileau and D. Veynante, *Combust. Flame* 158, 2199–2213 (2011). [doi:10.1016/j.combustflame.2011.04.008](https://doi.org/10.1016/j.combustflame.2011.04.008)
- Z. J. Wang, *J. Comput. Phys.* 178, 210–251 (2002). [doi:10.1006/jcph.2002.7041](https://doi.org/10.1006/jcph.2002.7041)
- J. M. Weiss and W. A. Smith, *AIAA J.* 33, 2050–2057 (1995). [doi:10.2514/3.12946](https://doi.org/10.2514/3.12946)
- H. Wu, P. C. Ma and M. Ihme, *Comput. Phys. Commun.* 243, 81–96 (2019). [doi:10.1016/j.cpc.2019.04.016](https://doi.org/10.1016/j.cpc.2019.04.016)
- L. Yang, L. Pan and K. Xu, *Comput. Fluids* 255, 105834 (2023). [doi:10.1016/j.compfluid.2023.105834](https://doi.org/10.1016/j.compfluid.2023.105834)
- R. Zangeneh and C. Ollivier-Gooch, *J. Comput. Phys.* 393, 375–405 (2019). [doi:10.1016/j.jcp.2019.05.002](https://doi.org/10.1016/j.jcp.2019.05.002)
