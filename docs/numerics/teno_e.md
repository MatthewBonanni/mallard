# TENO-E on unstructured triangles (implementation reference)

Primary source: Liang, Shyy, Fu, "Efficient Arbitrary-High-Order TENO Schemes with Local
Adaptive Dissipation for Compressible Flow Simulation on Unstructured Meshes",
J. Sci. Comput. 104:1 (2025), doi:10.1007/s10915-025-02918-w.
Predecessor: Ji, Liang, Fu, J. Sci. Comput. 92:61 (2022), arXiv:2105.02127.

## Stencils
- Large central stencil S_K: degree r, ~2x the number of non-constant DOFs, grown by
  neighbour layers and sorted by centroid distance (Tsoutsanis NCB).
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
