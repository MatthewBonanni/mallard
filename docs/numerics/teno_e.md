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
  max_q |1 - sum_s c_qs| + sum_s |c_qs|, is at most 4 in 3D (10 in 2D). The
  2 x DOFs nearest cells of a jittered hex/prism mesh span only about three cell
  layers per direction, so odd degrees (3 and 5) see the extra layer they need
  only through small centroid offsets: full rank, Lebesgue constants of 20 to
  150, and O(1) errors at order 4.
- The 3D bound is 4, not 10, for linear stability (#126, #229). On lattices the
  equidistant-shell rule skips the stencil sizes in between, so stencils within
  10 were stable there. Where the lattice's ties are broken, the first size
  within 10 can be a near-degenerate stencil. Ties break at symmetry walls of
  tilings that are not mirror-symmetric (prisms split along a diagonal, Kuhn
  tetrahedra): the mirror images make the wall a seam where the split flips. They
  also break at such seams in the interior and on jittered meshes. For example,
  aspect-ratio-2 prisms next to a side wall take 38 entries with a Lebesgue
  constant of 8.7, against 50 entries and 3.4 in the interior. The face values of
  such a stencil put up to 73% of their weight on cells across the face (32% on
  the lattice), so the outflow value follows the downstream cell. The linearized
  operator (acoustics at rest, HLLC) then has growing modes:

  | Case | max Re lambda (c/L), bound 10 -> 4 |
  |---|---|
  | prisms, aspect ratio 2, 6x6x4, walls / side walls only, order 4 | 1.9 / 2.5 -> < 1e-8 |
  | same, order 6 | 0.50 / 2.5 -> < 1e-8 |
  | periodic prisms with a seam (12x4x8, aspect ratio 2, order 4) | 3.5 -> < 1e-7 |
  | jittered prisms (8^3, jitter 0.1, periodic), order 6 | 3.4 -> 1.4e-7 |
  | jittered prisms, aspect ratio 1.5, side walls, order 6 | 0.32 -> 7e-6 |
  | Kuhn tetrahedra, aspect ratio 25, side walls, order 4 / 6 | 1.1 / 0.27 -> 8e-8 / 0.022 |
  | mixed tilings (4^3), walls, order 4 / 6 | 0.19 / 0.08 -> 8e-4 / 1e-7 |

  Making only the near-degenerate cells first order removes these modes, and in
  the walled boxes the energy is produced at their wall faces. Bounds of 5 to 7
  do not remove them. Order 6 on tetrahedra and pyramids reaches only 4.3 to 5;
  there the stencil is the smallest within 1.25 times the best Lebesgue constant
  found, since the best alone is often the largest stencil and up to twice as
  inaccurate. That leaves one slower mode, at order 6 on 25:1 Kuhn tetrahedra
  with side walls (+0.022; +1e-7 with the best stencil).
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
  tetrahedra 8 -> 12: 2.92, 3.90, 4.84 for orders 3, 4, 5. Lowering the 3D bound
  from 10 to 4 kept the rate 9 -> 12 for every cell type and order (hexahedra,
  Kuhn tetrahedra, prisms, pyramids, mixed, jittered). Order 6 errors changed by
  +10% (hexahedra, walls only), +22% (prisms), +11% (tetrahedra), -18%
  (pyramids) and -63% (jittered) at n = 12.

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
  - Prisms of aspect ratio 2 next to walls grew until the 3D Lebesgue bound
    went from 10 to 4 (see Stencils above).
  - Follow-up: a metric from the second moment of the node positions within a
    Euclidean ball may serve better than the edge-connected neighbors. The nodes
    of the Kuhn tiling form the cubic lattice, so they read isotropic, and the
    cut-off could then be lower. On a ball of twice the cell's radius, though,
    stretched meshes read only 1.6-2.0 at 6.25:1, because a ball holds the same
    density in every direction. The radius would need to follow the spacing.

- Pyramid tilings at orders 4 and 6: the regular `cartesian_pyramid` lattice is
  linearly unstable at order 4 on 6^3 periodic boxes even with the old bound
  (max Re lambda +2e-3 c/L), and the 3D bound of 4 makes pure pyramid lattices
  worse:

  | Case | bound 10 -> 4 |
  |---|---|
  | order 4, periodic, 4^3 / 6^3 | 7e-9 / 2e-3 -> 8e-3 / 2e-2 |
  | order 4, walls, 4^3 | 3.3e-3 -> 1.1e-2 |
  | order 6, periodic, 4^3 | 8e-9 -> 1.6e-3 |
  | order 6, walls, 4^3 | 4.7e-3 -> 8.5e-6 |

  Order 4 on that lattice is stable only from 85 stencil entries (Lebesgue
  constant about 3.5; the bound of 4 takes 57). Mixed meshes with pyramids
  improve with the bound (table above). Pure pyramid lattices are rare in
  practice; prefer orders 3 or 5 there.
