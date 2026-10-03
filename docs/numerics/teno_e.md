# TENO-E on unstructured triangles (implementation reference)

Primary source ([Liang, Shyy & Fu 2025](../references.md#liang-shyy-fu-2025)): Liang, Shyy, Fu, "Efficient Arbitrary-High-Order TENO Schemes with Local
Adaptive Dissipation for Compressible Flow Simulation on Unstructured Meshes",
J. Sci. Comput. 104:1 (2025), doi:10.1007/s10915-025-02918-w.
Predecessor: Ji, Liang, Fu, J. Sci. Comput. 92:61 (2022), arXiv:2105.02127 ([Ji, Liang & Fu 2022](../references.md#ji-liang-fu-2022)).
TENO itself: [Fu, Hu & Adams 2016](../references.md#fu-hu-adams-2016).

## Stencils
- Large central stencil S_K: degree r, ~2x the number of non-constant DOFs, grown by
  neighbour layers and sorted by centroid distance (Tsoutsanis NCB; [Tsoutsanis, Titarev & Drikakis 2011](../references.md#tsoutsanis-2011)).
- K = 3 small directional stencils (degree 2), one per face sector of the triangle.

## k-exact constrained least squares
- Target-local reference frame: affine map of the target triangle (x = x1 + J xi).
- Zero-mean basis on target: psi_l = phi_l - mean_{V0}(phi_l), monomials of degree 1..r.
- A_sl = int_{V_s} psi_l, b_s = |V_s| (U_s - U_0); a = pinv(A) b, pinv precomputed.

## Smoothness indicator
SI_k = sum_{1<=|beta|<=r} int_{V0'} (D^beta P_k)^2 (reference coords) = a^T M a.

## Troubled-cell indicator (density only)
gamma_k = |rho_k - rho_i| / |rho_i| over large-stencil neighbours,
sigma_i = variance of gamma_k, troubled if sigma_i >= 1e-3.

## Hybrid reconstruction
- Smooth: linear degree-r reconstruction of conserved variables (no SI, no weights).
- Troubled: characteristic variables (face-normal eigenvectors at a face-average state);
  upsilon_k = (SI_k + 1e-12)^-6, chi_k = upsilon_k / sum, delta_k = chi_k >= C_T.
  If delta_K: use P_K. Else equal weights over surviving small stencils
  (2022 variant: renormalize chi over small stencils only, guarantees a survivor).
- Adaptive C_T: m = (min(sigma_U, sigma) - sigma_L)/(sigma_U - sigma_L),
  g = (1-m)^2 (1+2m), psi = 10 - 4 (1 - g), C_T = 10^-floor(psi);
  sigma_L = 1e-3, sigma_U = 1e-2 (C_T from 1e-10 smooth-ish to 1e-6 at shocks).

## Flux / time integration
HLL (Davis/Einfeldt speeds) or HLLC, Gauss points per edge, SSPRK3, CFL 0.4.

## Test cases in paper
2D Riemann config 8 (t=0.25) and 16 (t=0.2) on [-0.5,0.5]^2; DMR; sin^2 density advection
for accuracy (expect design order on uniform triangles).

## 3D (Mallard extension)
- Same algorithm on tetrahedra, hexahedra, prisms and pyramids: trivariate monomials
  (scaled by h = V^(1/3)), one sector stencil per face (entries whose direction from the
  target centroid lies in the cone spanned by the face's vertices; 18 entries by
  default), mirror images across planar boundary faces.
- Cell averages of the basis and the SI matrix come from central moments of each cell,
  integrated once over its tetrahedral decomposition (the one defining the mesh
  geometry); stencil entries follow by binomial shifts, mirrored entries are integrated
  directly.
- Face quadrature: Dunavant rules on triangles, Gauss rules mapped bilinearly on
  quadrilaterals, exact to the reconstruction order.
- Equidistant shells are large on 3D lattices, so the large stencil may grow up to
  3.5 x DOFs + 64 entries to avoid splitting one; columns of round-off (e.g. no xy
  information when all centroids lie on axis planes) are rejected as rank deficient.
- Full rank is not enough: the large stencil keeps growing until the Lebesgue
  constant of its reconstruction at the cell's face quadrature points,
  max_q |1 - sum_s c_qs| + sum_s |c_qs|, is at most 10 (else the smallest one found
  is used; 2D likewise). The 2 x DOFs nearest cells of a jittered hex/prism mesh
  span only about three cell layers per direction, so odd degrees (3 and 5) see the
  extra layer they need only through small centroid offsets: full rank, Lebesgue
  constants of 20 to 150, and O(1) errors at order 4 (the generated mixed and
  tetrahedral meshes had some up to 270 and 54). Lattice-aligned meshes include
  the next layer through the equidistant-shell rule and stay at 2-5.
- Candidates are gathered by vertex-neighbor layers until there are enough interior
  cells; counting mirror images too would stop the search before the cells that
  are nearer than the farther images (rank-deficient order-6 stencils near walls).
- Candidates (central and sector stencils) are ranked by distance in a metric of
  the local mesh spacing, |d|_M^2 = d^T M d. M is the inverse of the second moment
  of the offsets of the cell's vertex neighbors, scaled to unit determinant. At
  boundary faces the neighbors are completed by the mirror images of the cells
  at most as far from the face as the cell. M is the identity unless the spacing
  ratio sqrt(lambda_max / lambda_min) of that second moment exceeds 2.5.
  Physical distance on thin cells takes the whole wall-normal column first and
  resolves the other directions only through small centroid offsets. That fit has
  full rank and Lebesgue constants within the bound, but it amplifies grid-scale
  vortical modes: the linearized operator (acoustics at rest, exact upwind
  flux) has real eigenvalues up to +13 (in units of the sound speed over the box
  size) on the thin boxes below, where every eigenvalue should have
  Re <= 0. The cut-off of 2.5 lies above the spacing ratios that cell shapes
  alone produce on regular tilings, which keep physical distance and so their
  stencils:

  | Mesh (6^3 to 10^3 blocks, interior cells) | spacing ratio |
  |---|---|
  | hexahedra | 1 |
  | Kuhn tetrahedra (1 / 2 / 3 neighbor layers) | 2.13 / 2.04 / 2.02 |
  | prisms (split along the xy diagonal) | 1.73 |
  | pyramids | 1.57 |
  | Kuhn tetrahedra stretched 2 / 6.25 / 25 : 1 | 3.15 / 9.1 / 36 |
  | prisms stretched 2 / 6.25 / 25 : 1 | 2.25 / 7.0 / 28 |

  A metric everywhere also stabilizes the thin cases, but it ranks stencils of
  the regular Kuhn tiling along its common cube diagonal and makes them up to
  8 times less accurate in the interior at order 5.
- Measured orders (max error at face quadrature points, symmetry walls): hexahedra
  16 -> 24: 3.83 and 4.85 for orders 4 and 5 (12 -> 16: 2.87 for order 3); Kuhn
  tetrahedra 8 -> 12: 2.92, 3.90, 4.84 for orders 3, 4, 5.

## Known limitations in Mallard

- Order 5 and up on curved, polygonal boundaries: on the cylinder O-grid
  (`examples/cylinder`, fine variant with 384 x 128 cells and stretched outer
  cells) the outermost ring develops a growing odd-even mode along the far
  field with every far-field condition tried (`upt`, `p_out`, `farfield`).
  Order 3 and MUSCL are stable there. Dropping the mirror images across the
  curved boundary makes it worse (noise from t = 1), so the high-degree fit
  in the stretched boundary cells is the more likely cause; reducing the
  order near boundaries is the next thing to try. Use `order = 3` near
  curved boundaries until this is resolved.

- Thin cells, before stencils were ranked in the spacing metric: a 1% pressure
  pulse at rest in a 16 x 16 x 4 box of `cartesian_prism` cells
  0.0625 x 0.0625 x 0.01 (symmetry walls) reached Mach 0.8 by t = 2 at order 3;
  `cartesian_tet` cells of the same size reached Mach 1.2. The Lebesgue bound
  cured that order-3 prism case, but tetrahedra (order 3, aspect ratio 1.5 and up)
  and prisms (order 4 with walls all around, aspect ratio 3 and up) kept growing
  modes. On the `examples/sphere_re300` mesh (`--scale 0.6`, 258k cells, prism
  layers of aspect ratio about 7), TENO3 previously reached Mach 0.9 by t = 1
  with a free stream at 0.2. It now stays at the potential-flow start's 0.30
  or below to t = 10 (6741 steps), at 2.1 times MUSCL's cost per step. Remaining:
  - Prisms of aspect ratio 2 at order 4 in a box with walls all around (spacing
    ratio 2.25, below the cut-off) keep a growing mode (eigenvalue +2.8) as before.
    With the metric applied, the same box is stable.
  - At aspect ratio 25, order 4 prisms with walls all around have one slow mode
    (eigenvalue +0.02, against spectral radius 350), with or without the metric.
  - Follow-up: a metric from the second moment of the node positions within a
    Euclidean ball may serve better than the edge-connected neighbors. The nodes
    of the Kuhn tiling form the cubic lattice, so they read isotropic, and the
    cut-off could then be lower. On a ball of twice the cell's radius, though,
    stretched meshes read only 1.6-2.0 at 6.25:1, because a ball holds the same
    density in every direction. The radius would need to follow the spacing.
