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
- <a id="glaister-1988"></a>P. Glaister, An approximate linearised Riemann solver for the Euler equations for real gases, *J. Comput. Phys.* 74, 382–408 (1988). [doi:10.1016/0021-9991(88)90084-8](https://doi.org/10.1016/0021-9991%2888%2990084-8)
  Used in: the Roe and RHLL solvers for gas mixtures, a Roe linearization for a general equation of state.
- <a id="shuen-liou-van-leer-1990"></a>J.-S. Shuen, M.-S. Liou and B. van Leer, Inviscid flux-splitting algorithms for real gases with non-equilibrium chemistry, *J. Comput. Phys.* 90, 371–395 (1990). [doi:10.1016/0021-9991(90)90172-W](https://doi.org/10.1016/0021-9991%2890%2990172-W)
  Used in: the Roe and RHLL solvers for gas mixtures: the Roe average of a multicomponent gas, whose composition waves move with the contact.
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

## Large-eddy simulation

- <a id="smagorinsky-1963"></a>J. Smagorinsky, General circulation experiments with the primitive equations, *Mon. Weather Rev.* 91, 99–164 (1963). [doi:10.1175/1520-0493(1963)091<0099:GCEWTP>2.3.CO;2](https://doi.org/10.1175/1520-0493%281963%29091%3C0099%3AGCEWTP%3E2.3.CO%3B2)
  Used in: `[les] model = "smagorinsky"`.
- <a id="deardorff-1970"></a>J. W. Deardorff, A numerical study of three-dimensional turbulent channel flow at large Reynolds numbers, *J. Fluid Mech.* 41, 453–480 (1970). [doi:10.1017/S0022112070000691](https://doi.org/10.1017/S0022112070000691)
  Used in: the LES filter width `V^(1/3)`.
- <a id="nicoud-ducros-1999"></a>F. Nicoud and F. Ducros, Subgrid-scale stress modelling based on the square of the velocity gradient tensor, *Flow Turbul. Combust.* 62, 183–200 (1999). [doi:10.1023/A:1009995426001](https://doi.org/10.1023/A:1009995426001)
  Used in: `[les] model = "wale"`.
- <a id="vreman-2004"></a>A. W. Vreman, An eddy-viscosity subgrid-scale model for turbulent shear flow: algebraic theory and applications, *Phys. Fluids* 16, 3670–3681 (2004). [doi:10.1063/1.1785131](https://doi.org/10.1063/1.1785131)
  Used in: `[les] model = "vreman"`.
- <a id="nicoud-2011"></a>F. Nicoud, H. Baya Toda, O. Cabrit, S. Bose and J. Lee, Using singular values to build a subgrid-scale model for large eddy simulations, *Phys. Fluids* 23, 085106 (2011). [doi:10.1063/1.3623274](https://doi.org/10.1063/1.3623274)
  Used in: `[les] model = "sigma"` (the default in 3D), its constant and the closed-form singular values; the table of model properties checked in `test/les_test.cpp`.
- <a id="vreman-geurts-kuerten-1995"></a>B. Vreman, B. Geurts and H. Kuerten, Subgrid-modelling in LES of compressible flow, *Appl. Sci. Res.* 54, 191–203 (1995). [doi:10.1007/BF00849116](https://doi.org/10.1007/BF00849116)
  Used in: the SGS terms of the filtered energy equation that are kept and neglected.
- <a id="garnier-adams-sagaut-2009"></a>E. Garnier, N. Adams and P. Sagaut, *Large Eddy Simulation for Compressible Flows*, Springer (2009). [doi:10.1007/978-90-481-2819-8](https://doi.org/10.1007/978-90-481-2819-8)
  Used in: the Favre-filtered equations and the eddy-viscosity closure of the SGS heat flux.
- <a id="ghosal-1996"></a>S. Ghosal, An analysis of numerical errors in large-eddy simulations of turbulence, *J. Comput. Phys.* 125, 187–206 (1996). [doi:10.1006/jcph.1996.0088](https://doi.org/10.1006/jcph.1996.0088)
  Used in: the motivation of the kinetic-energy budget (`[integrals] budget`), which separates the scheme's dissipation from the model's.

## Boundary conditions

- <a id="poinsot-lele-1992"></a>T. J. Poinsot and S. K. Lele, Boundary conditions for direct simulations of compressible viscous flows, *J. Comput. Phys.* 101, 104–129 (1992). [doi:10.1016/0021-9991(92)90046-2](https://doi.org/10.1016/0021-9991%2892%2990046-2)
  Used in: the LODI relations of the characteristic boundaries (`nscbc_outlet`, `nscbc_inlet`; `BoundaryData::characteristic_W` in `src/boundary/boundary.h`) and their viscous outflow conditions.
- <a id="rudy-strikwerda-1980"></a>D. H. Rudy and J. C. Strikwerda, A nonreflecting outflow boundary condition for subsonic Navier-Stokes calculations, *J. Comput. Phys.* 36, 55–70 (1980). [doi:10.1016/0021-9991(80)90174-6](https://doi.org/10.1016/0021-9991%2880%2990174-6)
  Used in: the pressure relaxation of `nscbc_outlet` and its default `sigma`.
- <a id="selle-nicoud-poinsot-2004"></a>L. Selle, F. Nicoud and T. Poinsot, Actual impedance of nonreflecting boundary conditions: implications for computation of resonators, *AIAA J.* 42(5), 958–964 (2004). [doi:10.2514/1.1883](https://doi.org/10.2514/1.1883)
  Used in: the reflection coefficient of the relaxed outlet, against which the acoustic-pulse test checks `nscbc_outlet`.
- <a id="yoo-2005"></a>C. S. Yoo, Y. Wang, A. Trouvé and H. G. Im, Characteristic boundary conditions for direct simulations of turbulent counterflow flames, *Combust. Theory Model.* 9(4), 617–646 (2005). [doi:10.1080/13647830500307378](https://doi.org/10.1080/13647830500307378)
  Used in: the transverse terms of the characteristic boundaries.
- <a id="yoo-im-2007"></a>C. S. Yoo and H. G. Im, Characteristic boundary conditions for simulations of compressible reacting flows with multi-dimensional, viscous and reaction effects, *Combust. Theory Model.* 11(2), 259–286 (2007). [doi:10.1080/13647830600898995](https://doi.org/10.1080/13647830600898995)
  Used in: the transverse terms and the treatment of multicomponent, reacting flows at characteristic boundaries.
- <a id="lodato-2008"></a>G. Lodato, P. Domingo and L. Vervisch, Three-dimensional boundary conditions for direct and large-eddy simulation of compressible viscous flows, *J. Comput. Phys.* 227, 5105–5143 (2008). [doi:10.1016/j.jcp.2008.01.038](https://doi.org/10.1016/j.jcp.2008.01.038)
  Used in: the relaxation of the transverse terms with `beta` equal to the Mach number (`beta`).
- <a id="granet-2010"></a>V. Granet, O. Vermorel, T. Leonard, L. Gicquel and T. Poinsot, Comparison of nonreflecting outlet boundary conditions for compressible solvers on unstructured grids, *AIAA J.* 48(10), 2348–2364 (2010). [doi:10.2514/1.J050391](https://doi.org/10.2514/1.J050391)
  Used in: the default `beta` and the vortex-outflow validation of `nscbc_outlet`.
- <a id="bodony-2006"></a>D. J. Bodony, Analysis of sponge zones for computational fluid mechanics, *J. Comput. Phys.* 212, 681–702 (2006). [doi:10.1016/j.jcp.2005.07.014](https://doi.org/10.1016/j.jcp.2005.07.014)
  Used in: the sponge layers (`[[sponges]]`; `src/solver/solver_sponge.cpp`).
- <a id="mani-2012"></a>A. Mani, Analysis and optimization of numerical sponge layers as a nonreflective boundary treatment, *J. Comput. Phys.* 231, 704–716 (2012). [doi:10.1016/j.jcp.2011.10.017](https://doi.org/10.1016/j.jcp.2011.10.017)
  Used in: the sponge layers and the advice to ramp their strength.

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
  Used in: the Gmsh 2.2 and 4.1 mesh reader, and the meshes of `tools/make_cylinder_mesh.py`, `tools/make_mallard_mesh.py` and `tools/make_sphere_re300_mesh.py`.
- [toml11](https://github.com/ToruNiina/toml11) reads the input file; [exprtk](https://www.partow.net/programming/exprtk/) evaluates its expressions.

## Chemistry and multicomponent flow

- <a id="kee-coltrin-glarborg-2003"></a>R. J. Kee, M. E. Coltrin and P. Glarborg, *Chemically Reacting Flow: Theory and Practice*, Wiley (2003). [doi:10.1002/0471461296](https://doi.org/10.1002/0471461296)
  Used in: the thermally perfect mixture (NASA polynomials), rates of progress and equilibrium constants, the mixture-averaged transport model with its correction velocity (`[chemistry]`, `src/chemistry`).
- <a id="cantera"></a>D. G. Goodwin, R. L. Speth, H. K. Moffat and B. W. Weber, *Cantera: An object-oriented software toolkit for chemical kinetics, thermodynamics, and transport processes*, https://www.cantera.org. [doi:10.5281/zenodo.742000](https://doi.org/10.5281/zenodo.742000)
  Used in: the mechanism file format (YAML) Mallard reads, the transport-property fits ported from Cantera, and the reference data of the chemistry tests and validation (generated by scripts in `tools/`; Mallard does not link Cantera).
- <a id="gilbert-luther-troe-1983"></a>R. G. Gilbert, K. Luther and J. Troe, Theory of thermal unimolecular reactions in the fall-off range. II. Weak collision rate constants, *Ber. Bunsenges. Phys. Chem.* 87, 169–177 (1983). [doi:10.1002/bbpc.19830870218](https://doi.org/10.1002/bbpc.19830870218)
  Used in: Troe falloff reactions.
- <a id="hairer-wanner-1996"></a>E. Hairer and G. Wanner, *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems*, 2nd ed., Springer (1996). [doi:10.1007/978-3-642-05221-7](https://doi.org/10.1007/978-3-642-05221-7)
  Used in: the RODAS Rosenbrock integrator of the chemistry source and `MallardReactor`.
- <a id="strang-1968"></a>G. Strang, On the construction and comparison of difference schemes, *SIAM J. Numer. Anal.* 5, 506–517 (1968). [doi:10.1137/0705041](https://doi.org/10.1137/0705041)
  Used in: the Strang splitting of chemistry around the flow step.
- <a id="larrouturou-1991"></a>B. Larrouturou, How to preserve the mass fractions positivity when computing compressible multi-component flows, *J. Comput. Phys.* 95, 59–84 (1991). [doi:10.1016/0021-9991(91)90253-H](https://doi.org/10.1016/0021-9991%2891%2990253-H)
  Used in: the species fluxes (mass-flux upwinding of the reconstructed mass fractions).
- <a id="abgrall-karni-2001"></a>R. Abgrall and S. Karni, Computations of compressible multifluids, *J. Comput. Phys.* 169, 594–623 (2001). [doi:10.1006/jcph.2000.6685](https://doi.org/10.1006/jcph.2000.6685)
- <a id="billet-abgrall-2003"></a>G. Billet and R. Abgrall, An adaptive shock-capturing algorithm for solving unsteady reactive flows, *Comput. Fluids* 32, 1473–1495 (2003). [doi:10.1016/S0045-7930(03)00004-5](https://doi.org/10.1016/S0045-7930%2803%2900004-5)
- <a id="ma-lv-ihme-2017"></a>P. C. Ma, Y. Lv and M. Ihme, An entropy-stable hybrid scheme for simulations of transcritical real-fluid flows, *J. Comput. Phys.* 340, 330–357 (2017). [doi:10.1016/j.jcp.2017.03.022](https://doi.org/10.1016/j.jcp.2017.03.022)
  Used in (these three): the double-flux option for oscillation-free material and temperature interfaces.

## Validation cases and reference data

- <a id="fedkiw-merriman-osher-1997"></a>R. P. Fedkiw, B. Merriman and S. Osher, High accuracy numerical methods for thermally perfect gas flows with chemistry, *J. Comput. Phys.* 132, 175–190 (1997). [doi:10.1006/jcph.1996.5622](https://doi.org/10.1006/jcph.1996.5622)
- <a id="martinez-ferrer-2014"></a>P. J. Martínez Ferrer, R. Buttay, G. Lehnasch and A. Mura, A detailed verification procedure for compressible reactive multicomponent Navier–Stokes solvers, *Comput. Fluids* 89, 88–110 (2014). [doi:10.1016/j.compfluid.2013.10.014](https://doi.org/10.1016/j.compfluid.2013.10.014)
  Used in (these two): `examples/reactive_shock_tube` (the reflected-shock ignition of 2H2–O2–7Ar).
- <a id="sdt"></a>*Shock and Detonation Toolbox* (SDToolbox), Explosion Dynamics Laboratory, California Institute of Technology, https://shepherd.caltech.edu/EDL/PublicResources/sdt/
  Used in: the CJ speed and ZND structure of `examples/detonation_1d` (`tools/detonation_reference.py` follows its method with Cantera).
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
- <a id="johnson-patel-1999"></a>T. A. Johnson and V. C. Patel, Flow past a sphere up to a Reynolds number of 300, *J. Fluid Mech.* 378, 19–70 (1999). [doi:10.1017/S0022112098003206](https://doi.org/10.1017/S0022112098003206)
- <a id="tomboulides-1993"></a>A. G. Tomboulides, S. A. Orszag and G. E. Karniadakis, Direct and large-eddy simulation of the flow past a sphere, in *Engineering Turbulence Modelling and Experiments 2*, Elsevier, 273–282 (1993). [doi:10.1016/B978-0-444-89802-9.50030-7](https://doi.org/10.1016/B978-0-444-89802-9.50030-7)
- <a id="tomboulides-orszag-2000"></a>A. G. Tomboulides and S. A. Orszag, Numerical investigation of transitional and weak turbulent flow past a sphere, *J. Fluid Mech.* 416, 45–73 (2000). [doi:10.1017/S0022112000008880](https://doi.org/10.1017/S0022112000008880)
- <a id="kim-kim-choi-2001"></a>J. Kim, D. Kim and H. Choi, An immersed-boundary finite-volume method for simulations of flow in complex geometries, *J. Comput. Phys.* 171, 132–150 (2001). [doi:10.1006/jcph.2001.6778](https://doi.org/10.1006/jcph.2001.6778)
- <a id="constantinescu-squires-2003"></a>G. S. Constantinescu and K. D. Squires, LES and DES investigations of turbulent flow over a sphere at Re = 10,000, *Flow Turbul. Combust.* 70, 267–298 (2003). [doi:10.1023/B:APPL.0000004937.34078.71](https://doi.org/10.1023/B:APPL.0000004937.34078.71)
  Used in: `examples/sphere_re300` and `tools/plot_sphere_re300.py` (Strouhal number, mean drag and lift at Re = 300: Johnson & Patel, Tomboulides et al., Kim et al., and the laminar Re = 300 validation of Constantinescu & Squires; Tomboulides & Orszag describe the regime, a single-frequency shedding that keeps a plane of symmetry).
- <a id="brachet-1983"></a>M. E. Brachet, D. I. Meiron, S. A. Orszag, B. G. Nickel, R. H. Morf and U. Frisch, Small-scale structure of the Taylor–Green vortex, *J. Fluid Mech.* 130, 411–452 (1983). [doi:10.1017/S0022112083001159](https://doi.org/10.1017/S0022112083001159)
- <a id="van-rees-2011"></a>W. M. van Rees, A. Leonard, D. I. Pullin and P. Koumoutsakos, A comparison of vortex and pseudo-spectral methods for the simulation of periodic vortical flows at high Reynolds numbers, *J. Comput. Phys.* 230, 2794–2805 (2011). [doi:10.1016/j.jcp.2010.11.031](https://doi.org/10.1016/j.jcp.2010.11.031)
- <a id="wang-2013"></a>Z. J. Wang et al., High-order CFD methods: current status and perspective, *Int. J. Numer. Methods Fluids* 72, 811–845 (2013). [doi:10.1002/fld.3767](https://doi.org/10.1002/fld.3767)
  Used in: `examples/taylor_green_3d`, compared with the 512³ pseudo-spectral DNS of the International Workshop on High-Order CFD Methods (case C3.5; [data](https://cfd.ku.edu/hiocfd/spectral_Re1600_512.gdiag)).
- <a id="johnsen-2010"></a>E. Johnsen, J. Larsson, A. V. Bhagatwala, W. H. Cabot, P. Moin, B. J. Olson, P. S. Rawat, S. K. Shankar, B. Sjögreen, H. C. Yee, X. Zhong and S. K. Lele, Assessment of high-resolution methods for numerical simulations of compressible turbulence with shock waves, *J. Comput. Phys.* 229, 1213–1237 (2010). [doi:10.1016/j.jcp.2009.10.028](https://doi.org/10.1016/j.jcp.2009.10.028)
- <a id="lele-2012"></a>S. K. Lele, Simulations of turbulent flows with strong shocks and density variations, final report DE-FC02-06ER25787, Stanford University (2012). [doi:10.2172/1052207](https://doi.org/10.2172/1052207)
- <a id="subramaniam-2019"></a>A. Subramaniam, M. L. Wong and S. K. Lele, A high-order weighted compact high resolution scheme with boundary closures for compressible turbulent flows with shocks, *J. Comput. Phys.* 397, 108822 (2019). [doi:10.1016/j.jcp.2019.07.021](https://doi.org/10.1016/j.jcp.2019.07.021)
- <a id="samtaney-2001"></a>R. Samtaney, D. I. Pullin and B. Kosović, Direct numerical simulation of decaying compressible turbulence and shocklet statistics, *Phys. Fluids* 13, 1415–1430 (2001). [doi:10.1063/1.1355682](https://doi.org/10.1063/1.1355682)
  Used in: `examples/isotropic_turbulence` and `tools/plot_isotropic_turbulence.py`: the decaying isotropic turbulence case of Johnsen et al. and their filtered 256³ reference (read from the same figure in Lele's report), the filtered 512³ DNS and spectra of Subramaniam et al. (whose appendix F gives the post-processing of both references), and the regime of Samtaney et al.; also `viscosity_model = "power_law"`.
- <a id="ruetsch-1995"></a>G. R. Ruetsch, L. Vervisch and A. Liñán, Effects of heat release on triple flames, *Phys. Fluids* 7, 1447–1454 (1995). [doi:10.1063/1.868531](https://doi.org/10.1063/1.868531)
- <a id="veynante-1994"></a>D. Veynante, L. Vervisch, T. Poinsot, A. Liñán and G. Ruetsch, Triple flame structure and diffusion flame stabilization, *Proceedings of the Summer Program 1994*, Center for Turbulence Research, Stanford University (1994). [oa.upm.es/1537](https://oa.upm.es/1537/)
  Used in (these two): `examples/triple_flame` and `tools/triple_flame_analysis.py` (the propagation speed U<sub>F</sub> / S<sub>L</sub> against the mixing thickness at the flame, D<sub>TF</sub>, and the weak-gradient limit √(ρ<sub>u</sub> / ρ<sub>b</sub>)).
- <a id="poinsot-1990"></a>T. Poinsot, D. Veynante and S. Candel, Diagrams of premixed turbulent combustion based on direct simulation, *Proc. Combust. Inst.* 23, 613–619 (1990).
- <a id="poinsot-1991"></a>T. Poinsot, D. Veynante and S. Candel, Quenching processes and premixed turbulent combustion diagrams, *J. Fluid Mech.* 228, 561–606 (1991). [doi:10.1017/S0022112091002823](https://doi.org/10.1017/S0022112091002823)
  Used in (these two): `examples/flame_vortex` and `tools/flame_vortex_analysis.py` (the vortex-pair scales r and u′, the four outcomes and the 5% cut-off of the spectral diagram).
- The spherical explosion of `examples/explosion_3d` is from [Toro (2009)](#toro-2009), §17.1.3.
- <a id="noh-1987"></a>W. F. Noh, Errors for calculations of strong shocks using an artificial viscosity and an artificial heat flux, *J. Comput. Phys.* 72, 78–120 (1987). [doi:10.1016/0021-9991(87)90074-X](https://doi.org/10.1016/0021-9991(87)90074-X)
  Used in: `examples/noh_axisymmetric` and `tools/plot_noh.py` (the spherical Noh problem and its exact solution).
- <a id="clift-grace-weber-1978"></a>R. Clift, J. R. Grace and M. E. Weber, *Bubbles, Drops, and Particles*, Academic Press (1978).
  Used in: `examples/sphere_axisymmetric` and `tools/plot_sphere_axisymmetric.py` (correlations of the drag coefficient and the separation angle of a sphere in steady flow, 20 < Re < 260), with [Johnson & Patel (1999)](#johnson-patel-1999) for the wake length.
