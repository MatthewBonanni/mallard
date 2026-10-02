# References

The numerical methods Mallard implements and the reference data it is validated against, grouped by topic. Each entry says where Mallard uses it. Techniques with no entry here (the transmissive image faces, the mirror ghost cells of TENO-E stencils, the deterministic face-flux accumulation, the stencil-driven halo depth) are Mallard's own and are described in [Numerical methods](numerics/overview.md). The design notes ([distributed memory](design/mpi.md), [chemistry](design/chemistry.md)) cite their own sources for work not yet implemented.

## Reconstruction

- <a id="van-leer-1979"></a>B. van Leer, Towards the ultimate conservative difference scheme. V. A second-order sequel to Godunov's method, *J. Comput. Phys.* 32, 101–136 (1979). [doi:10.1016/0021-9991(79)90145-1](https://doi.org/10.1016/0021-9991%2879%2990145-1)
  Used in: MUSCL reconstruction (`type = "MUSCL"`; `src/numerics/face_reconstruction.cpp`).
- <a id="barth-jespersen-1989"></a>T. J. Barth and D. C. Jespersen, The design and application of upwind schemes on unstructured meshes, AIAA Paper 89-0366, 27th Aerospace Sciences Meeting (1989). [doi:10.2514/6.1989-366](https://doi.org/10.2514/6.1989-366)
  Used in: the `barth_jespersen` MUSCL limiter.
- <a id="venkatakrishnan-1995"></a>V. Venkatakrishnan, Convergence to steady state solutions of the Euler equations on unstructured grids with limiters, *J. Comput. Phys.* 118, 120–130 (1995). [doi:10.1006/jcph.1995.1084](https://doi.org/10.1006/jcph.1995.1084)
  Used in: the `venkatakrishnan` MUSCL limiter (the default) and its threshold `(K h)³` (`venkatakrishnan_K`).
- <a id="mavriplis-2003"></a>D. J. Mavriplis, Revisiting the least-squares procedure for gradient reconstruction on unstructured meshes, AIAA Paper 2003-3986, 16th AIAA Computational Fluid Dynamics Conference (2003). [doi:10.2514/6.2003-3986](https://doi.org/10.2514/6.2003-3986)
  Used in: the inverse-distance-weighted least-squares gradients of MUSCL and of the viscous terms (`src/numerics/gradient.h`).
- <a id="barth-frederickson-1990"></a>T. J. Barth and P. O. Frederickson, Higher order solution of the Euler equations on unstructured grids using quadratic reconstruction, AIAA Paper 90-0013, 28th Aerospace Sciences Meeting (1990). [doi:10.2514/6.1990-13](https://doi.org/10.2514/6.1990-13)
  Used in: k-exact least-squares reconstruction (TENO-E polynomials; the quadratic fit of the viscous cell gradients).
- <a id="fu-hu-adams-2016"></a>L. Fu, X. Y. Hu and N. A. Adams, A family of high-order targeted ENO schemes for compressible-fluid simulations, *J. Comput. Phys.* 305, 333–359 (2016). [doi:10.1016/j.jcp.2015.10.037](https://doi.org/10.1016/j.jcp.2015.10.037)
  Used in: the TENO stencil selection with a sharp cutoff C<sub>T</sub>, on which TENO-E builds.
- <a id="ji-liang-fu-2022"></a>F. Ji, T. Liang and L. Fu, A class of new high-order finite-volume TENO schemes for hyperbolic conservation laws with unstructured meshes, *J. Sci. Comput.* 92, 61 (2022). [doi:10.1007/s10915-022-01925-5](https://doi.org/10.1007/s10915-022-01925-5)
  Used in: TENO on unstructured meshes; the fallback that renormalizes the small-stencil weights ([TENO-E details](numerics/teno_e.md)).
- <a id="liang-shyy-fu-2025"></a>T. Liang, W. Shyy and L. Fu, Efficient arbitrary-high-order TENO schemes with local adaptive dissipation for compressible flow simulation on unstructured meshes, *J. Sci. Comput.* 104, 1 (2025). [doi:10.1007/s10915-025-02918-w](https://doi.org/10.1007/s10915-025-02918-w)
  Used in: TENO-E, the `TENO` reconstruction of orders 3 to 6: large central and sector stencils, the density-based troubled-cell indicator, characteristic-wise selection with the adaptive cutoff (`src/numerics/teno.cpp`, `teno.h`).
- <a id="tsoutsanis-2011"></a>P. Tsoutsanis, V. A. Titarev and D. Drikakis, WENO schemes on arbitrary mixed-element unstructured meshes in three space dimensions, *J. Comput. Phys.* 230, 1585–1601 (2011). [doi:10.1016/j.jcp.2010.11.023](https://doi.org/10.1016/j.jcp.2010.11.023)
  Used in: building the TENO-E central stencil from neighbor layers sorted by centroid distance.
- <a id="zhang-shu-2010"></a>X. Zhang and C.-W. Shu, On maximum-principle-satisfying high order schemes for scalar conservation laws, *J. Comput. Phys.* 229, 3091–3120 (2010). [doi:10.1016/j.jcp.2009.12.030](https://doi.org/10.1016/j.jcp.2009.12.030)
  Used in: the scaling limiter of `bound_preserving = true`, which scales troubled-cell polynomials toward the cell average to keep density and pressure within the neighbors' range.

## Riemann solvers

- <a id="toro-2009"></a>E. F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction*, 3rd ed., Springer (2009). [doi:10.1007/b79761](https://doi.org/10.1007/b79761)
  Used in: the exact Riemann solver of the tests (`test/exact_riemann.h`), the HLLC star states (Toro's equations 10.37–10.39), the Euler eigenvectors of the characteristic decomposition and the Roe solver, and the spherical explosion test (§17.1.3).
- <a id="rusanov-1962"></a>V. V. Rusanov, The calculation of the interaction of non-stationary shock waves and obstacles, *USSR Comput. Math. Math. Phys.* 1, 304–320 (1962). [doi:10.1016/0041-5553(62)90062-9](https://doi.org/10.1016/0041-5553%2862%2990062-9)
  Used in: `riemann_solver = "Rusanov"`.
- <a id="harten-lax-van-leer-1983"></a>A. Harten, P. D. Lax and B. van Leer, On upstream differencing and Godunov-type schemes for hyperbolic conservation laws, *SIAM Rev.* 25, 35–61 (1983). [doi:10.1137/1025002](https://doi.org/10.1137/1025002)
  Used in: `riemann_solver = "HLL"`, and the HLL part of RHLL.
- <a id="einfeldt-1988"></a>B. Einfeldt, On Godunov-type methods for gas dynamics, *SIAM J. Numer. Anal.* 25, 294–318 (1988). [doi:10.1137/0725021](https://doi.org/10.1137/0725021)
- <a id="einfeldt-1991"></a>B. Einfeldt, C.-D. Munz, P. L. Roe and B. Sjögreen, On Godunov-type methods near low densities, *J. Comput. Phys.* 92, 273–295 (1991). [doi:10.1016/0021-9991(91)90211-3](https://doi.org/10.1016/0021-9991%2891%2990211-3)
  Used in: the wave-speed estimates (from Roe averages) of HLL and HLLC.
- <a id="toro-spruce-speares-1994"></a>E. F. Toro, M. Spruce and W. Speares, Restoration of the contact surface in the HLL-Riemann solver, *Shock Waves* 4, 25–34 (1994). [doi:10.1007/BF01414629](https://doi.org/10.1007/BF01414629)
  Used in: `riemann_solver = "HLLC"` (the default).
- <a id="roe-1981"></a>P. L. Roe, Approximate Riemann solvers, parameter vectors, and difference schemes, *J. Comput. Phys.* 43, 357–372 (1981). [doi:10.1016/0021-9991(81)90128-5](https://doi.org/10.1016/0021-9991%2881%2990128-5)
  Used in: `riemann_solver = "Roe"`, the Roe part of RHLL, and the Roe averages of the wave-speed estimates.
- <a id="harten-1983"></a>A. Harten, High resolution schemes for hyperbolic conservation laws, *J. Comput. Phys.* 49, 357–393 (1983). [doi:10.1016/0021-9991(83)90136-5](https://doi.org/10.1016/0021-9991%2883%2990136-5)
  Used in: the entropy fix of the Roe solver on the acoustic waves.
- <a id="nishikawa-kitamura-2008"></a>H. Nishikawa and K. Kitamura, Very simple, carbuncle-free, boundary-layer-resolving, rotated-hybrid Riemann solvers, *J. Comput. Phys.* 227, 2560–2581 (2008). [doi:10.1016/j.jcp.2007.11.003](https://doi.org/10.1016/j.jcp.2007.11.003)
  Used in: `riemann_solver = "RHLL"`, the rotated hybrid of HLL and Roe.
- <a id="quirk-1994"></a>J. J. Quirk, A contribution to the great Riemann solver debate, *Int. J. Numer. Methods Fluids* 18, 555–574 (1994). [doi:10.1002/fld.1650180603](https://doi.org/10.1002/fld.1650180603)
  Used in: the carbuncle and odd–even decoupling tests that RHLL passes and HLLC and Roe fail (`test/solver_test.cpp`; the Shu–Osher validation).

- <a id="guillard-viozat-1999"></a>H. Guillard and C. Viozat, On the behaviour of upwind schemes in the low Mach number limit, *Comput. Fluids* 28, 63–86 (1999). [doi:10.1016/S0045-7930(98)00017-6](https://doi.org/10.1016/S0045-7930%2898%2900017-6)
  Used in: the motivation of the low-Mach correction: upwind dissipation that scales with the sound speed as M → 0.
- <a id="rieper-2011"></a>F. Rieper, A low-Mach number fix for Roe's approximate Riemann solver, *J. Comput. Phys.* 230, 5263–5287 (2011). [doi:10.1016/j.jcp.2011.03.025](https://doi.org/10.1016/j.jcp.2011.03.025)
  Used in: the motivation of the low-Mach correction, and its normal-velocity-only variant that was compared against Thornber's.
- <a id="thornber-2008"></a>B. Thornber, A. Mosedale, D. Drikakis, D. Youngs and R. J. R. Williams, An improved reconstruction method for compressible flows with low Mach number features, *J. Comput. Phys.* 227, 4873–4894 (2008). [doi:10.1016/j.jcp.2008.01.036](https://doi.org/10.1016/j.jcp.2008.01.036)
  Used in: the low-Mach correction of the face velocity jump, `[numerics] low_mach_cutoff`.
- <a id="weiss-smith-1995"></a>J. M. Weiss and W. A. Smith, Preconditioning applied to variable and constant density flows, *AIAA J.* 33, 2050–2057 (1995). [doi:10.2514/3.12946](https://doi.org/10.2514/3.12946)
  Used in: the cutoff Mach number of the low-Mach correction (`low_mach_cutoff`, default 0.1).

## Viscous terms and transport

- <a id="diskin-2010"></a>B. Diskin, J. L. Thomas, E. J. Nielsen, H. Nishikawa and J. A. White, Comparison of node-centered and cell-centered unstructured finite-volume discretizations: viscous fluxes, *AIAA J.* 48, 1326–1338 (2010). [doi:10.2514/1.44940](https://doi.org/10.2514/1.44940)
  Used in: the viscous face gradients, the average of the two cell gradients corrected along the face normal so that their component along the line of centroids equals the direct difference (`src/numerics/viscous_flux.h`).
- <a id="sutherland-1893"></a>W. Sutherland, The viscosity of gases and molecular force, *Phil. Mag.* (5) 36, 507–531 (1893). [doi:10.1080/14786449308620508](https://doi.org/10.1080/14786449308620508)
  Used in: `viscosity_model = "sutherland"`.

## Time integration and time step

- <a id="shu-osher-1988"></a>C.-W. Shu and S. Osher, Efficient implementation of essentially non-oscillatory shock-capturing schemes, *J. Comput. Phys.* 77, 439–471 (1988). [doi:10.1016/0021-9991(88)90177-5](https://doi.org/10.1016/0021-9991%2888%2990177-5)
- <a id="gottlieb-shu-1998"></a>S. Gottlieb and C.-W. Shu, Total variation diminishing Runge–Kutta schemes, *Math. Comp.* 67, 73–85 (1998). [doi:10.1090/S0025-5718-98-00913-2](https://doi.org/10.1090/S0025-5718-98-00913-2)
  Used in: `time_integrator = "SSPRK3"` (the default; `src/numerics/time_integrator.cpp`).
- <a id="kutta-1901"></a>W. Kutta, Beitrag zur näherungsweisen Integration totaler Differentialgleichungen, *Z. Math. Phys.* 46, 435–453 (1901).
  Used in: `time_integrator = "RK4"`, the classical fourth-order Runge–Kutta method.
- <a id="blazek-2015"></a>J. Blazek, *Computational Fluid Dynamics: Principles and Applications*, 3rd ed., Butterworth-Heinemann (2015). [doi:10.1016/C2013-0-19038-1](https://doi.org/10.1016/C2013-0-19038-1)
  Used in: the local time step from convective and viscous spectral radii (`Solver::calc_dt_cfl1`), and the characteristic far-field condition (`type = "farfield"`), which takes the outgoing Riemann invariant from the interior and the incoming one from the free stream.

## Quadrature and geometry

- <a id="dunavant-1985"></a>D. A. Dunavant, High degree efficient symmetrical Gaussian quadrature rules for the triangle, *Int. J. Numer. Methods Eng.* 21, 1129–1148 (1985). [doi:10.1002/nme.1620210612](https://doi.org/10.1002/nme.1620210612)
  Used in: cell averages of analytical initial conditions, TENO-E integrals in 2D, and face quadrature on triangular faces in 3D (`src/numerics/quadrature.cpp`).
- <a id="duffy-1982"></a>M. G. Duffy, Quadrature over a pyramid or cube of integrands with a singularity at a vertex, *SIAM J. Numer. Anal.* 19, 1260–1262 (1982). [doi:10.1137/0719090](https://doi.org/10.1137/0719090)
- <a id="stroud-1971"></a>A. H. Stroud, *Approximate Calculation of Multiple Integrals*, Prentice-Hall (1971).
  Used in: collapsed (conical product) Gauss rules on triangles and tetrahedra, for TENO-E moments and 3D analytical initial conditions (`src/numerics/teno.cpp`, `src/solver/solver_initialize.cpp`). Gauss–Legendre rules integrate the faces.
- <a id="kuhn-1960"></a>H. W. Kuhn, Some combinatorial lemmas in topology, *IBM J. Res. Dev.* 4, 518–524 (1960). [doi:10.1147/rd.45.0518](https://doi.org/10.1147/rd.45.0518)
  Used in: `type = "cartesian_tet"`, six tetrahedra per block of a generated box.

## Parallelism, meshes and software

- <a id="skilling-2004"></a>J. Skilling, Programming the Hilbert curve, *AIP Conf. Proc.* 707, 381–387 (2004). [doi:10.1063/1.1751381](https://doi.org/10.1063/1.1751381)
  Used in: `partitioner = "hilbert"` (`src/parallel/partition.cpp`).
- <a id="sanders-seemaier-2023"></a>P. Sanders and D. Seemaier, Distributed deep multilevel graph partitioning, in *Euro-Par 2023: Parallel Processing*, Lecture Notes in Computer Science, Springer, 443–457 (2023). [doi:10.1007/978-3-031-39698-4_30](https://doi.org/10.1007/978-3-031-39698-4_30)
  Used in: `partitioner = "graph"` through dKaMinPar ([KaMinPar](https://github.com/KaHIP/KaMinPar); `Mallard_ENABLE_KAMINPAR`).
- <a id="trott-2022"></a>C. R. Trott et al., Kokkos 3: Programming model extensions for the exascale era, *IEEE Trans. Parallel Distrib. Syst.* 33, 805–817 (2022). [doi:10.1109/TPDS.2021.3097283](https://doi.org/10.1109/TPDS.2021.3097283)
  Used in: all of Mallard's kernels and data ([Kokkos](https://github.com/kokkos/kokkos)).
- <a id="geuzaine-remacle-2009"></a>C. Geuzaine and J.-F. Remacle, Gmsh: A 3-D finite element mesh generator with built-in pre- and post-processing facilities, *Int. J. Numer. Methods Eng.* 79, 1309–1331 (2009). [doi:10.1002/nme.2579](https://doi.org/10.1002/nme.2579)
  Used in: the Gmsh 2.2 and 4.1 mesh reader, and the meshes of `tools/make_cylinder_mesh.py` and `tools/make_mallard_mesh.py`.
- [toml11](https://github.com/ToruNiina/toml11) reads the input file; [exprtk](https://www.partow.net/programming/exprtk/) evaluates its expressions.

## Validation cases and reference data

- <a id="sod-1978"></a>G. A. Sod, A survey of several finite difference methods for systems of nonlinear hyperbolic conservation laws, *J. Comput. Phys.* 27, 1–31 (1978). [doi:10.1016/0021-9991(78)90023-2](https://doi.org/10.1016/0021-9991%2878%2990023-2)
  Used in: `examples/sod` and the shock-tube tests.
- <a id="shu-osher-1989"></a>C.-W. Shu and S. Osher, Efficient implementation of essentially non-oscillatory shock-capturing schemes, II, *J. Comput. Phys.* 83, 32–78 (1989). [doi:10.1016/0021-9991(89)90222-2](https://doi.org/10.1016/0021-9991%2889%2990222-2)
  Used in: `examples/shu_osher`.
- <a id="shu-1998"></a>C.-W. Shu, Essentially non-oscillatory and weighted essentially non-oscillatory schemes for hyperbolic conservation laws, in *Advanced Numerical Approximation of Nonlinear Hyperbolic Equations*, Lecture Notes in Mathematics 1697, Springer, 325–432 (1998). [doi:10.1007/BFb0096355](https://doi.org/10.1007/BFb0096355)
  Used in: the isentropic vortex of the design-order convergence study.
- <a id="woodward-colella-1984"></a>P. Woodward and P. Colella, The numerical simulation of two-dimensional fluid flow with strong shocks, *J. Comput. Phys.* 54, 115–173 (1984). [doi:10.1016/0021-9991(84)90142-6](https://doi.org/10.1016/0021-9991%2884%2990142-6)
  Used in: `examples/double_mach`.
- <a id="schulz-rinne-1993"></a>C. W. Schulz-Rinne, J. P. Collins and H. M. Glaz, Numerical solution of the Riemann problem for two-dimensional gas dynamics, *SIAM J. Sci. Comput.* 14, 1394–1414 (1993). [doi:10.1137/0914082](https://doi.org/10.1137/0914082)
- <a id="lax-liu-1998"></a>P. D. Lax and X.-D. Liu, Solution of two-dimensional Riemann problems of gas dynamics by positive schemes, *SIAM J. Sci. Comput.* 19, 319–340 (1998). [doi:10.1137/S1064827595291819](https://doi.org/10.1137/S1064827595291819)
- <a id="liska-wendroff-2003"></a>R. Liska and B. Wendroff, Comparison of several difference schemes on 1D and 2D test problems for the Euler equations, *SIAM J. Sci. Comput.* 25, 995–1017 (2003). [doi:10.1137/S1064827502402120](https://doi.org/10.1137/S1064827502402120)
  Used in: `examples/riemann_2d` and `examples/riemann_2d_quads` (configuration 3).
- <a id="naca-1953"></a>Ames Research Staff, Equations, tables, and charts for compressible flow, NACA Report 1135 (1953). [NTRS 19930091059](https://ntrs.nasa.gov/citations/19930091059)
  Used in: the oblique-shock relations against which `examples/wedge` and its test are checked.
- <a id="schlichting-gersten-2017"></a>H. Schlichting and K. Gersten, *Boundary-Layer Theory*, 9th ed., Springer (2017). [doi:10.1007/978-3-662-52919-5](https://doi.org/10.1007/978-3-662-52919-5)
  Used in: the exact viscous solutions of the tests (Couette flow, Stokes' first problem, conduction between walls).
- <a id="daru-tenaud-2009"></a>V. Daru and C. Tenaud, Numerical simulation of the viscous shock tube problem by using a high resolution monotonicity-preserving scheme, *Comput. Fluids* 38, 664–676 (2009). [doi:10.1016/j.compfluid.2008.06.008](https://doi.org/10.1016/j.compfluid.2008.06.008)
- <a id="zhou-xu-liu-2018"></a>G. Zhou, K. Xu and F. Liu, Grid-converged solution and analysis of the unsteady viscous flow in a two-dimensional shock tube, *Phys. Fluids* 30, 016102 (2018). [doi:10.1063/1.4998300](https://doi.org/10.1063/1.4998300)
  Used in: `examples/viscous_shock_tube` and the reference wall density of `tools/plot_viscous_shock_tube.py`.
- <a id="williamson-1996"></a>C. H. K. Williamson, Vortex dynamics in the cylinder wake, *Annu. Rev. Fluid Mech.* 28, 477–539 (1996). [doi:10.1146/annurev.fl.28.010196.002401](https://doi.org/10.1146/annurev.fl.28.010196.002401)
- <a id="liu-zheng-sung-1998"></a>C. Liu, X. Zheng and C. H. Sung, Preconditioned multigrid methods for unsteady incompressible flows, *J. Comput. Phys.* 139, 35–57 (1998). [doi:10.1006/jcph.1997.5859](https://doi.org/10.1006/jcph.1997.5859)
- <a id="park-kwon-choi-1998"></a>J. Park, K. Kwon and H. Choi, Numerical solutions of flow past a circular cylinder at Reynolds numbers up to 160, *KSME Int. J.* 12, 1200–1205 (1998). [doi:10.1007/BF02942594](https://doi.org/10.1007/BF02942594)
  Used in: `examples/cylinder` (Strouhal number, drag and lift at Re = 100).
- <a id="brachet-1983"></a>M. E. Brachet, D. I. Meiron, S. A. Orszag, B. G. Nickel, R. H. Morf and U. Frisch, Small-scale structure of the Taylor–Green vortex, *J. Fluid Mech.* 130, 411–452 (1983). [doi:10.1017/S0022112083001159](https://doi.org/10.1017/S0022112083001159)
- <a id="van-rees-2011"></a>W. M. van Rees, A. Leonard, D. I. Pullin and P. Koumoutsakos, A comparison of vortex and pseudo-spectral methods for the simulation of periodic vortical flows at high Reynolds numbers, *J. Comput. Phys.* 230, 2794–2805 (2011). [doi:10.1016/j.jcp.2010.11.031](https://doi.org/10.1016/j.jcp.2010.11.031)
- <a id="wang-2013"></a>Z. J. Wang et al., High-order CFD methods: current status and perspective, *Int. J. Numer. Methods Fluids* 72, 811–845 (2013). [doi:10.1002/fld.3767](https://doi.org/10.1002/fld.3767)
  Used in: `examples/taylor_green_3d`, compared with the 512³ pseudo-spectral DNS of the International Workshop on High-Order CFD Methods (case C3.5; [data](https://cfd.ku.edu/hiocfd/spectral_Re1600_512.gdiag)).
- The spherical explosion of `examples/explosion_3d` is from [Toro (2009)](#toro-2009), §17.1.3.
