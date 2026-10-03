# Design: finite-rate chemistry

Status: accepted (see [Decisions on the open questions](#decisions-on-the-open-questions)).
Implementation follows the [milestones](#10-milestones); done: 1.

Mallard today solves a single calorically perfect gas. This document adds
multicomponent, thermally perfect mixtures and finite-rate chemistry with
mechanisms read at runtime, on NVIDIA GPUs (A100 class) and CPUs, within the
existing C++20 / Kokkos / CMake / TOML stack.

## Goals

- **Arbitrary mechanisms at runtime.** A Cantera YAML file named in the input
  defines species, thermodynamics, reactions and transport data. Changing the
  mechanism never requires recompiling.
- **Mechanism sizes from 10 to about 500 species** (hydrogen to detailed
  hydrocarbon and surrogate-fuel mechanisms), with a path to 1000.
- **One code path for GPU and CPU**, through Kokkos, in 2D and 3D, in double
  and float builds.
- **Inviscid and viscous reacting flow:** reactive Euler for detonations and
  shock-induced ignition, reactive Navier-Stokes with mixture-averaged
  transport for flames.
- **No cost for the single-gas solver.** A run without a mechanism compiles to
  the code that runs today and gives bit-identical results.
- **Determinism and rank-count independence** as in [mpi.md](mpi.md) and
  #49: the result does not depend on the thread or rank count.
- **A validation suite** with reference data and pass criteria, run on every
  milestone.

## Non-goals

- Real-fluid equations of state (cubic EOS, transcritical injection),
  multiphase and spray, soot, radiation, plasma and ionized species, surface
  chemistry and catalytic walls.
- Full multicomponent (Stefan-Maxwell) transport, Soret and Dufour effects,
  bulk viscosity. The transport interface leaves room for them.
- Turbulence-chemistry interaction models (LES closures such as thickened
  flame, flamelet tables, PaSR). Mallard resolves the flame or detonation.
- Implicit time stepping of the flow. The flow stays explicit (SSPRK3); only
  the chemistry is implicit.
- A Chemkin parser. Chemkin input is converted once with Cantera's `ck2yaml`.

## Summary of decisions

| # | Decision | Main alternative | Why |
|---|---|---|---|
| 1 | Species count is a runtime value; species live in their own `View` beside the unchanged flow state | Species count as a compile-time template parameter, like `Mallard_DIM` | Arbitrary mechanisms without recompiling; register-resident species arrays do not scale past ~30 species anyway |
| 2 | The gas model is a template parameter of the RHS kernels (`PerfectGas`, `Mixture`), dispatched once per stage | Runtime branches in every kernel | Single-gas path compiles to today's code; no divergence or register cost in it |
| 3 | Riemann solvers take, per side, `[rho, u, p]` plus two thermodynamic surrogates `(gamma, e0)` with `rho E = p / (gamma - 1) + rho e0 + rho u^2 / 2` | Pass full species vectors to the Riemann solver | Fixed-width face states independent of the species count; exact for a perfect gas (`e0 = 0`); one code path |
| 4 | Species fluxes by mass-flux upwinding (Larrouturou) of linearly reconstructed mass fractions, with one shared stencil and one shared limiter per cell | Species as extra components of every Riemann solver; per-species nonlinear TENO | Positivity, sum Y = 1 at faces to round-off, and a cost that is pure bandwidth per species |
| 5 | Strang splitting of chemistry around the full SSPRK3 step | Chemistry inside the RK stages (IMEX) | Second order, keeps the explicit flow integrator untouched, lets the stiff solver adapt its own sub-steps |
| 6 | Own Kokkos-native chemistry layer: runtime mechanism tables on the device, analytical Jacobian, batched Rosenbrock integrator; BDF and codegen kernels later | Adopt TChem, Zero-RK, PelePhysics or SUNDIALS | See [Libraries](#stiff-chemistry-libraries-and-integrators); no candidate meets Kokkos 5 + runtime mechanisms + Mallard's data layout without a heavier dependency than the code it saves |
| 7 | Chemistry always in double precision, also in float builds | Chemistry in `rtype` | Stiff Newton/Rosenbrock linear algebra and equilibrium constants lose too much in float; A100 FP64 is fast |
| 8 | Mechanisms are Cantera YAML, read in C++ with yaml-cpp; transport fits computed at setup by a port of Cantera's fitting | Link Cantera's C++ library; offline Python converter | No Cantera build dependency (Cantera builds with SCons and pulls Boost, Eigen, fmt, SUNDIALS); one-step workflow |
| 9 | Load balancing of chemistry by redistributing cell states (DLBFoam-style), separate from mesh repartitioning | Only cost-weighted mesh repartitioning | Chemistry is pointwise: moving states is cheap and needs no halo rebuild |
| 10 | Fully conservative by default, primitive reconstruction; double flux as an input option | Double flux always; conservative only | Conservation where it matters (shocks, detonations) by default; oscillation-free contacts when asked for |

Each is detailed below.

## 1. Multicomponent state

### Variables

The conservative state becomes `[rho, rho u, rho E, rho Y_1 .. rho Y_Ns]`.

- `rho` is kept as its own equation and all `Ns` species are transported, so
  `sum_k rho Y_k = rho` holds to round-off by construction of the fluxes
  (section 3), not by eliminating a bath species. Eliminating the last
  species (`Y_N = 1 - sum`) breaks its positivity whenever the others
  overshoot, and the bath species is not always the most abundant one.
- `E` includes the chemical energy through the species' formation
  enthalpies (NASA polynomials are referenced to the standard state), so
  chemistry does not appear in the energy equation as a source.

### Storage

```cpp
using StateView   = Kokkos::View<rtype *[N_CONSERVATIVE]>;   // unchanged: rho, rho u, rho E
using SpeciesView = Kokkos::View<rtype **, SpeciesLayout>;    // (cell, k): rho Y_k
struct State { StateView flow; SpeciesView species; };       // species.extent(1) == 0 without a mechanism
```

- The flow block keeps its compile-time width, so every existing kernel,
  local array and Riemann solver is untouched in size.
- `SpeciesLayout` is one alias. Kokkos' default per backend
  (`LayoutLeft` on CUDA: cell index fastest) coalesces one-cell-per-thread
  kernels; `LayoutRight` (species contiguous per cell) coalesces kernels that
  put a team per cell and species across vector lanes, which is what chemistry
  and transport of larger mechanisms use (section 5). Kernels index through
  the `View`, so this is a measured choice in milestone 10, not an
  architectural one. The starting point is `LayoutRight`.
- The time integrator, `axpby`, halo exchange, restart and conservation sums
  take a `State` and loop over both parts. Without a mechanism the species
  extent is 0 and these loops vanish.

### Why runtime species count

A template parameter `NS` (like `Mallard_DIM`) would let kernels hold species
in registers and let the compiler unroll, at the price of a build per
mechanism. It buys little:

- Register arrays sized `NS` spill to local memory beyond a few tens of
  species, which is exactly where chemistry matters. Large-mechanism kernels
  must stream species from memory either way.
- The kernels that would benefit (face states, Riemann solvers, TENO
  characteristic projection) are kept free of species by decision 3.

The option stays open where measurements justify it: hot kernels can be
dispatched over a few compile-time species buckets (`NS <= 16, 32, 64`, with
runtime `Ns` below the bucket) the way TENO dispatches over its degree.

### Memory budget

Per cell and species, in `rtype` words: state 1, SSPRK3 scratch 1, RHS 1,
species face fluxes one per face side (about 3 per triangle, 6 per
hexahedron; section 3), and for viscous runs gradients `N_DIM` and the
mixture-averaged diffusivity 1. That is about 6 words in 2D Euler and 15 in
3D Navier-Stokes,
so 50 species on 10 M hexahedra in double is about 60 GB: one A100-80GB.
Species work arrays are processed in blocks of species when memory is short
(the species loop is outermost in every species kernel, so blocking is a loop
split, not a redesign).

## 2. Thermodynamics

### Thermally perfect mixture

- Ideal gas: `p = rho R_u T sum_k Y_k / W_k`.
- Species `cp_k(T)`, `h_k(T)`, `s_k(T)` from NASA-7 (two ranges) or NASA-9
  (any number of ranges) polynomials, as in Cantera's `NasaPoly2` and
  `Nasa9PolyMultiTempRegion`. Coefficients live on the device in one flat
  table: per species, its range bounds and 7 or 9 coefficients per range,
  NASA-7 stored as NASA-9 with zero extra terms so there is one evaluation
  routine.
- Outside the fitted range the outermost polynomial is extrapolated;
  `check_interval` reports cells outside the common range of
  all species' fits.

```cpp
struct Mixture {                       // POD of Views, captured by value in kernels
    int n_species;
    Kokkos::View<const double *> W;    // molar masses
    ThermoTable thermo;                // NASA ranges and coefficients
    TransportTable transport;          // fits, section 6
    KOKKOS_FUNCTION double R(const Yview &) const;
    KOKKOS_FUNCTION double e(double T, const Yview &) const;      // and cv, cp, h, g_k/RT
    KOKKOS_FUNCTION double T_from_e(double e, const Yview &, double T_guess) const;
};
```

### Temperature from energy

`e(T, Y) = sum_k Y_k e_k(T)` is monotone (`cv > 0`), so Newton on `T` with
`de/dT = cv` converges in 1 to 3 iterations from the previous value of `T`,
which is cached per cell (it is the `T` column of `primitives`). The iteration
is safeguarded by a bracket `[T_min, T_max]` (from the thermo ranges) with a
bisection fallback, and stops at `|dT| < 1e-10 T` (double) or `1e-5 T`
(float). Newton is needed once per cell per RK stage and at the start and end
of each chemistry step (which carries `T` as an unknown, section 4); face
states never need it (section 3).

The cached `T` makes the iteration count, and so the last bits of `T`,
history dependent. Restart files therefore carry `T` (section 9) so that
restarts stay bit-identical.

### Surrogate thermodynamics for the flux

From a cell's `(rho, T, Y)` the RHS computes the frozen ratio of specific heats
and an energy offset:

```text
gamma = cp(T, Y) / cv(T, Y)
e0    = e(T, Y) - R(Y) T / (gamma - 1)        so   rho e = p / (gamma - 1) + rho e0  (exact at the cell)
a^2   = gamma p / rho                          (frozen sound speed)
```

For a calorically perfect gas `e0 = 0` and `gamma` is the constant `gamma`.
These two numbers are what the Riemann solvers, the characteristic projection
and the time step need; they are reconstructed to faces like any other scalar
(section 3).

## 3. Numerics for multicomponent flow

### Riemann solvers

Today every solver takes `W = [rho, u, p]` per side and one global `gamma`.
They become functions of `W` and per-side `(gamma, e0)`:

- `rho E = p / (gamma - 1) + rho e0 + rho |u|^2 / 2`,
  `a = sqrt(gamma p / rho)`, `H = (rho E + p) / rho`.
- **Rusanov, HLL, HLLC** need only `U`, `F` and `a` per side: direct
  generalization. Einfeldt speeds use Roe-averaged `u` and
  `a_roe^2 = (sqrt(rho_L) a_L^2 + sqrt(rho_R) a_R^2) / (sqrt(rho_L) + sqrt(rho_R))
  + eta_2 (u_R - u_L)^2` (Einfeldt 1988), which needs no `gamma`.
- **Roe and RHLL** need a Roe average for a variable-`gamma` gas
  (Shuen, Liou & van Leer 1990; Glaister 1988). They come in a later
  milestone; until then a mixture run rejects them at input.
- The perfect-gas instantiation passes the constant `gamma` and `e0 = 0`
  as compile-time-known values, so it compiles to the current code.

Species are not passed to the Riemann solvers at all. The mass flux
`mdot = F_rho` at every face quadrature point is stored (one value per point)
and the species fluxes follow from it.

### Species fluxes

[Larrouturou's scheme (1991)](https://doi.org/10.1016/0021-9991(91)90253-H): the species flux is the mass flux times the
upwind mass fraction,

```text
F_k = max(mdot, 0) Y_k^L + min(mdot, 0) Y_k^R
```

- If `Y^L`, `Y^R` are in `[0, 1]` and sum to 1, then `sum_k F_k = mdot` exactly
  and `rho Y_k` stays non-negative whenever `rho` does, under the same CFL
  condition. This holds for every Riemann solver, since it only uses `mdot`.
- For HLLC it coincides with the solver's own passive-scalar flux (species
  ride the contact). Rusanov's and HLL's own species fluxes would not have
  this property, because their dissipation acts on each `rho Y_k`
  separately.

### Reconstruction

The issue: with variable `gamma`, a fully conservative scheme generates
pressure oscillations at material interfaces even with a perfect face
reconstruction ([Karni 1994](https://doi.org/10.1006/jcph.1994.1080);
[Abgrall 1996](https://doi.org/10.1006/jcph.1996.0085)), because the update
of `rho E` in a mixed cell is inconsistent with pressure equilibrium. For a
thermally perfect gas this happens at any temperature or composition jump,
not only between species. Reconstructing conservative or characteristic
variables adds oscillations of its own; primitive reconstruction with `p` is
needed for oscillation-free contacts
([Johnsen & Colonius 2006](https://doi.org/10.1016/j.jcp.2006.04.018);
[Coralic & Colonius 2014](https://doi.org/10.1016/j.jcp.2014.06.003)). The
remaining error from the conservative update is of the order of the jump in
`gamma` and `e0`: negligible across resolved premixed flames and detonation
reaction zones, but percent-level at sharp fuel/oxidizer or hot/cold contacts
(e.g. hydrogen injected into air), which is what double flux addresses
([Houim & Kuo 2011](https://doi.org/10.1016/j.jcp.2011.07.031) is the closest
analogue to Mallard: WENO with double flux for thermally perfect reacting
flow).

The design:

1. **Flow block**: reconstruct `W = [rho, u, p]` (MUSCL already does). In
   TENO, mixture runs switch the smooth-cell polynomial and the troubled-cell
   characteristic projection from conservative variables to the primitive
   system, whose eigenvectors depend only on `rho`, `u` and `a` and so are
   valid for any equation of state. The perfect-gas path keeps today's
   conservative variables.
2. **Scalars** `Y_1 .. Y_Ns`, `gamma`, `e0`: reconstructed **linearly with
   one stencil per cell and face, shared by all scalars**:
   - smooth cells: the large central stencil (as for the flow block);
   - troubled cells: the stencil TENO selected for the entropy/contact
     characteristic field at that face (species and entropy waves share the
     eigenvalue `u_n`), stored by the flow pass as one byte per face;
   - MUSCL: least-squares gradients with one limiter value per cell, the
     minimum of Barth-Jespersen (or Venkatakrishnan) over all scalars.
   Least-squares reconstruction reproduces constants, so with shared weights
   `sum_k Y_k = 1` at every face point to round-off.
3. **Bounds**: one scaling `theta` per cell
   ([Zhang & Shu 2010](https://doi.org/10.1016/j.jcp.2010.08.016)), the largest value
   in `[0, 1]` that keeps every `Y_k` at every face point in `[0, 1]` and
   `gamma > 1` in smooth cells, and within the range of the cell and its face
   neighbors in troubled cells. Smooth cells get only the physical bounds,
   because a local-range bound clips smooth extrema (e.g. radical peaks in a
   flame) and costs the design order; Zhang–Shu scaling to physical bounds does
   not. Applied to all scalars at once, it preserves `sum Y = 1`.
4. **Troubled-cell indicator**: today's density-jump variance misses species
   interfaces at constant density (common in non-premixed flames). It becomes
   the maximum of the variances of `rho` and of the mixture molar mass `W`.

This makes species reconstruction a sequence of dot products with weights
already computed for the flow block: no smoothness indicators per species and
no per-species branching. The price is that species never get the high-order
nonlinear stencil choice of their own; at a fuel/oxidizer interface they use
the contact field's choice, which is the physically relevant one.

**Option: double flux** ([Abgrall & Karni 2001](https://doi.org/10.1006/jcph.2000.6685);
[Billet & Abgrall 2003](https://doi.org/10.1016/S0045-7930(03)00004-5);
[Ma, Lv & Ihme 2017](https://doi.org/10.1016/j.jcp.2017.03.022)). Each cell's
fluxes are computed with that cell's `(gamma, e0)` frozen over the RK stage on
both sides of its faces, and `rho E` is reset from the true equation of state
after the stage. This keeps `p` and `u` exactly uniform across contacts, at the
cost of energy conservation in proportion to the jump; Houim & Kuo switch back
to the conservative flux at shocks. The surrogate interface makes it a
localized change: a face computes two fluxes, one with each side's frozen
`(gamma, e0)`, so the per-face flux array of #49's deterministic
accumulation gets a second slot for the energy flux. It is milestone 5, enabled by input,
right after the conservative scheme and the interface test that measures the
oscillations. A TENO/double-flux scheme on unstructured meshes appears not to
have been published, so this is also where Mallard would be new.

### Species fluxes without races

Following #49, species fluxes are written per face, never scattered with
atomics. Because Larrouturou's flux takes each side's contribution from one
side only, a kernel over `(cell, species block)` reconstructs the cell's
species, evaluates them at its face points, and writes
`w_q max(mdot_out, 0) Y_k(q)` into the slot `(face, side, k)` it owns. A
second kernel gathers each cell's slots in `faces_of_cell` order. This needs
`2 x n_faces x Ns` words and no face storage of reconstructed species values.

### Time step

Unchanged for inviscid flow, with the frozen sound speed. Viscous runs use
`nu_eff = max(4/3 mu / rho, lambda / (rho cv), max_k D_k)`. Chemistry does not
limit the flow step (section 4).

## 4. Coupling chemistry and flow

### Strang splitting (default)

```text
U^n --chem(dt/2)--> U* --SSPRK3 flow step(dt)--> U** --chem(dt/2)--> U^{n+1}
```

- Chemistry changes only `rho Y_k`. `rho`, `rho u` and `rho E` are constant in
  each cell during a chemistry substep, so each cell is a constant-volume,
  adiabatic 0D reactor: `dY_k/dt = W_k omega_k / rho` with `T` following from
  `e`. The integrator state is `(Y_1 .. Y_Ns, T)`; `T` is carried as an
  unknown (`dT/dt = -sum_k e_k W_k omega_k / (rho cv)`), which gives a better
  conditioned Jacobian than recomputing `T` from `e` in every RHS, and is
  checked against `e` at the end of the substep.
- The two half steps of consecutive steps are fused (one `dt` chemistry
  call between flow steps) except around output and restart times, as usual.
- Second order in time for smooth problems; the flow integrator, the CFL
  logic and the RK stages stay as they are.

### Alternatives

- **Chemistry in the RK stages, explicit.** Impossible for stiff chemistry:
  radical time scales are 1e-9 s or less, flame and detonation flow steps
  1e-8 to 1e-6 s.
- **Additive IMEX Runge-Kutta** ([Kennedy & Carpenter 2003](https://doi.org/10.1016/S0168-9274(02)00138-1)) with implicit
  chemistry per stage: higher formal order and no splitting error, but a
  Newton solve per stage at the flow step size, with no error control on the
  chemistry, and a new integrator family to maintain. Fixed steps through an
  ignition transient are inaccurate unless `dt` is small.
- **Spectral deferred corrections / chemistry with frozen advective
  forcing** (PeleLM, [Nonaka et al. 2012](https://doi.org/10.1080/13647830.2012.701019);
  PeleC, [Henry de Frahan et al. 2023](https://doi.org/10.1177/10943420221121151)):
  integrate `dU/dt = A + R(U)` per cell with the transport tendency `A` frozen
  over the step, and iterate. Removes most splitting error, notably in steady
  flames where Strang splitting is not steady-state preserving
  ([Speth et al. 2013](https://doi.org/10.1137/120878641) and
  [Lu et al. 2017](https://doi.org/10.1016/j.jcp.2017.01.044) on balanced
  splitting; [Wu, Ma & Ihme 2019](https://doi.org/10.1016/j.cpc.2019.04.016)
  compare splittings in a compressible solver). The chemistry interface takes
  an optional per-cell forcing vector, so this can be added without touching
  the integrator.

### Known splitting issues and how the design handles them

- **Spurious wave speeds for under-resolved stiff fronts**
  ([LeVeque & Yee 1990](https://doi.org/10.1016/0021-9991(90)90097-K)) come from under-resolution, not from splitting; the detonation tests
  check resolution (points per half-reaction length) and CJ speed.
- **Steady flames**: Strang splitting error appears as a `dt`-dependent
  flame speed. The flame-speed test runs at two CFL numbers and requires the
  difference to be below the pass tolerance; if it is not, the forcing option
  above is the remedy.

### Stiffness, load imbalance and skipped cells

- **Skipping inactive cells.** A cell is chemically frozen when
  `T < T_frozen` (input, off by default) or when a cheap estimate
  `dt max_k |omega_k W_k / rho| < eps_Y` holds. Active cells are compacted
  into a queue, as TENO compacts troubled cells.
- **GPU imbalance between cells.** Cells in an ignition kernel take 100x the
  sub-steps of cells in burnt or fresh gas. With one cell per thread a warp
  runs at its slowest lane. The design puts **one cell per team** (a warp, or a
  small team, with species across vector lanes) for mechanisms above about 30
  species, so the hardware block scheduler balances independent cells; for
  small mechanisms, where one cell per thread is faster, cells are **binned by
  the sub-step count of their previous step** (stored per cell) and launched
  in bins of similar cost.
- **MPI imbalance.** Flame and detonation fronts put most of the chemistry
  cost on a few ranks. Chemistry is pointwise, so it is balanced by moving
  cell states, not cells: before the chemistry step, ranks exchange cost
  estimates (`allreduce`), overloaded ranks send `(rho, e, Y, T, h_last)` of
  their most expensive active cells to underloaded ranks, which integrate them
  and send `Y, T, h_last` back ([Tekgul et al. 2021](https://doi.org/10.1016/j.cpc.2021.108073),
  [DLBFoam](https://github.com/Aalto-CFD/DLBFoam), which reports up to about
  10x speedup on reacting OpenFOAM cases). This needs no
  halo or stencil rebuild and keeps results rank-count independent (each
  cell's integration is the same wherever it runs). Long-term imbalance of the
  flow work (TENO troubled cells, species fluxes) uses the dynamic mesh
  rebalancing prepared in [mpi.md, section 10](mpi.md#10-room-for-dynamic-load-balancing),
  with the measured per-cell chemistry cost added to the partition weights.

## 5. Chemistry

### Stiff chemistry libraries and integrators

Status checked against the repositories on 2026-10-02. "Runtime" means a new
mechanism needs no recompilation.

| Library | Status | License | Fit with Mallard | Mechanisms | Jacobian / integrator | GPU strategy | Verdict |
|---|---|---|---|---|---|---|---|
| [TChem](https://github.com/sandialabs/TChem) (Sandia) | No releases; `main` last changed 2023-07; newest branch targets Kokkos 4.2 (2025-08) | BSD-2 | Kokkos-native, but **no Kokkos 5 support**; depends on [Tines](https://github.com/sandialabs/Tines) (Kokkos <= 4.5), Sacado, OpenBLAS/LAPACKE, yaml-cpp | Runtime (Chemkin, Cantera YAML) | Sacado autodiff or FD; TrBDF2 | Team per sample | Closest in spirit; unusable with Kokkos 5 without porting it and Tines. Its successor [TChem-atm](https://github.com/sandialabs/TChem-atm) (v2.0.0, 2025-09; [Kim et al., GMD 2026](https://gmd.copernicus.org/articles/19/1281/2026/)) targets atmospheric chemistry and pins a fork of Kokkos Kernels |
| [Zero-RK](https://github.com/LLNL/zero-rk) (LLNL) | Active (2026-03), no tags | BSD-3 | CPU + CUDA (no Kokkos); SUNDIALS (tested up to 5.8), SuperLU, MAGMA | Runtime (Chemkin) | Analytical sparse Jacobian; CVODE with sparse preconditioned iterative solves ([McNenly et al. 2015](https://doi.org/10.1016/j.proci.2014.05.113)) | Batched multi-reactor with MAGMA / cuSolverRf | Best algorithms for large mechanisms; the CUDA-only GPU path and old SUNDIALS rule it out as a dependency. Its ideas (static sparse pattern, preconditioned Krylov above ~300 species) carry over |
| [PelePhysics](https://github.com/Pele-Suite/PelePhysics) | v1.0.0 (2026-02), very active | BSD-3 | **Requires AMReX** and SUNDIALS | **Build-time codegen** (CEPTR, from Cantera YAML), incl. QSS reduction ([arXiv 2405.05974](https://arxiv.org/abs/2405.05974)) | Generated analytical Jacobian; CVODE (MAGMA / cuSPARSE batched), ARKODE, explicit RK64, own BDF | Lock-step batches per box | The production reference for GPU combustion, but AMReX and codegen per mechanism are both against our constraints |
| [Cantera](https://github.com/Cantera/cantera) | 3.2.0 (2025-11) | BSD-3 | CPU only; builds with SCons; Boost, Eigen, fmt, yaml-cpp, SUNDIALS | Runtime YAML: the de facto format | Analytical/approximate sparse Jacobians; CVODES | None | **The reference oracle** (Python, in tests and tools) and the input format; not linked into Mallard |
| [SUNDIALS](https://github.com/LLNL/sundials) CVODE / ARKODE | 7.9.0 (2026-09); Kokkos 5 support in the Kokkos NVector since 7.7 | BSD-3 | Kokkos NVector and `KokkosDense` batched block-diagonal solver; MAGMA, cuSolverSp batch QR, Ginkgo batched (7.5+) | n/a (user RHS) | User Jacobian; BDF, ARK, RKC/RKL (`LSRKStep`) | One big system for all cells, driven from the host | **All cells share one step size and one error norm**: the stiffest cell sets the step for the batch, and a cell's result depends on the cells it is batched with, so results change with the decomposition ([Balos et al., IJHPCA 2024](https://doi.org/10.1177/10943420241280060)). A CPU reference integrator for tests, not the device integrator |
| [Kokkos Kernels ODE](https://github.com/kokkos/kokkos-kernels/tree/5.2.2/ode) | In Kokkos Kernels 5.2.2 (2026-09) | Kokkos license | Kokkos-native, Kokkos 5 | n/a | User dense Jacobian; adaptive RK, variable-order BDF (`BDFSolve`), Newton with static-pivoting dense LU | One system per thread, no team variant | `Experimental`, Newton iteration count hard-coded, `max_step` ignored; TChem-atm had to fork it. A design reference and prototype, not a dependency. Its batched `SerialLU`/`TeamLU` are what we build on |
| [accelerInt](https://github.com/SLACKHA/accelerInt), [pyJac](https://github.com/SLACKHA/pyJac) | Unmaintained since 2018-19 | MIT | CUDA / C codegen | Codegen | Analytical (pyJac); Rosenbrock, exponential, Radau-IIA | One thread per cell | Source of the GPU integrator comparisons below |
| [KinetiX](https://github.com/bogdandanciu/KinetiX) ([Danciu et al., CPC 2025](https://doi.org/10.1016/j.cpc.2025.109504)) | 2025 | BSD-2 | OCCA codegen (nekRS) | Codegen from Cantera YAML | Rates, thermo, transport; no integrator | Thread per point | Rate evaluation up to 1.7x faster than CEPTR on GPU; a model for our optional codegen layer |
| [Pyrometheus](https://arxiv.org/abs/2503.24286), [ChemGen](https://arxiv.org/abs/2510.10005) | 2025 | various (ChemGen: not OSI) | Python/C++ codegen | Codegen | ChemGen: analytical Jacobian and implicit integrators | not stated | Not adoptable as dependencies |

What the GPU literature says about integrators and parallel layout:

- **Batched dense direct solves beat matrix-free Krylov** at moderate size:
  Balos et al. (2024) found MAGMA batched LU about 10x faster than GMRES in
  PeleLMeX for the 53- and 88-species mechanisms (GMRES won at 21 species),
  and cuSPARSE batched 1.5-10x slower than MAGMA. Ginkgo's batched solvers beat
  vendor ones on PeleLM matrices ([Aggarwal et al. 2021](https://sc21.supercomputing.org/proceedings/workshops/workshop_pages/ws_lasalss105.html); [arXiv 2308.08417](https://arxiv.org/abs/2308.08417)).
- **Implicit one-step methods map well to GPUs, but divergence hurts.**
  Radau-IIA with an analytical (pyJac) Jacobian on one GPU matched CVODE on
  12-38 CPU cores for hydrogen, but only about 3 cores for GRI-3.0 at
  `dt` = 1e-4 s because of thread divergence; exponential methods were less
  competitive, and a finite-difference Jacobian cost 7-241x
  ([Curtis, Niemeyer & Sung 2017](https://doi.org/10.1016/j.combustflame.2017.02.005)).
  Rosenbrock and RK solvers vectorize well on GPUs and CPU SIMD
  ([Stone, Alferman & Niemeyer 2018](https://arxiv.org/abs/1608.05794)).
  Explicit RKC on a GPU was up to 57x faster than CPU VODE for GRI-3.0 at
  `dt` = 1e-6 s but 2.5x slower than 6-core VODE at 1e-4 s
  ([Niemeyer & Sung 2014](https://doi.org/10.1016/j.jcp.2013.09.025)):
  explicit methods are only an option for small splitting steps.
- **Thread per cell vs team per cell**: thread per cell wins for small systems
  and many cells (up to about 2x, [arXiv 2405.17363](https://arxiv.org/abs/2405.17363)),
  but a warp runs at the pace of its stiffest cell; team per cell is needed
  once the Jacobian no longer fits per thread (roughly 50-100 species).
- **Quasi-steady-state predictor-correctors** (CHEMEQ2, Mott, Oran & van Leer
  2000; YASS) are cheap and Jacobian-free but lose accuracy and conservation on
  large stiff mechanisms. **Dynamic adaptive chemistry** (per-cell DRG) gives
  each cell its own mechanism, which is hostile to SIMT. **Offline reduction**
  ([pyMARS](https://github.com/Niemeyer-Research-Group/pyMARS)) and
  **QSS-reduced mechanisms** remain the main lever for very large mechanisms;
  both produce ordinary Cantera YAML files that need nothing from us.
- No library publishes a throughput-versus-species-count curve on an A100 for
  10-1000 species. The benchmark suite (section 11) produces one.

### Recommendation

**Write our own chemistry core in Kokkos, small and specific, and use the
libraries as oracles and design references rather than dependencies.**

1. **Mechanism data at runtime** (Cantera YAML via yaml-cpp, flattened to
   device tables). No codegen, no recompilation.
2. **Analytical Jacobian** from the same tables: dense below about 100 species,
   static-pattern sparse above.
3. **Adaptive Rosenbrock integrator per cell** (one LU per step, no Newton
   loop, fixed work per step), on Kokkos Kernels' batched `SerialLU` /
   `TeamLU`, templated on the execution pattern (thread per cell or team per
   cell, chosen from `Ns`).
4. **Later layers behind the same interface, each only if benchmarks call for
   it:** an adaptive BDF for large mechanisms (with Zero-RK-style
   preconditioned Krylov above about 300 species); a mechanism-specialized,
   code-generated kernel (KinetiX/CEPTR-style C++ compiled as an optional
   plugin) for production runs on a fixed mechanism.
5. **Cantera (Python)** produces all reference data in the tests; **CVODE**
   (SUNDIALS, CPU, one instance per cell) is a test-only reference integrator.

Tradeoffs:

- We own an integrator, Jacobians for every reaction type, and transport
  fitting: roughly 3-5k lines that TChem or PelePhysics already have. In
  exchange we get Kokkos 5 and Mallard's data layout, runtime mechanisms,
  per-cell step control (decomposition-independent results), float builds,
  and no AMReX, Tines, Sacado or SUNDIALS in the build.
- A table-driven rate kernel is slower than generated code (KinetiX reports up
  to 1.7x over CEPTR for rates alone; the LU is unaffected). The codegen layer
  recovers that when it matters.
- Rosenbrock needs a Jacobian per step; BDF reuses one over many steps and wins
  for large mechanisms at loose tolerances. The interface is built for both,
  and the benchmarks decide when BDF is worth adding.
- If Sandia ships a Kokkos 5 TChem, adopting its kinetics behind our integrator
  interface becomes a reasonable alternative to milestone 6; nothing here
  precludes it.

### Device mechanism representation

The mechanism is converted on the host into flat, read-only device arrays
(structure of arrays), grouped by reaction type so that a warp evaluates
reactions of one kind:

| Table | Content |
|---|---|
| Species | molar mass, NASA ranges and coefficients, transport fits |
| Arrhenius | `log A`, `b`, `Ea / R_u` per rate (forward, and low/high for falloff) |
| Stoichiometry | CSR per reaction: reactant and product species, coefficients, orders (non-integer orders allowed) |
| Third body | CSR of non-default efficiencies, default efficiency |
| Falloff | type (Lindemann, Troe, SRI) and parameters |
| PLOG | per reaction, list of `(log p, Arrhenius)`; Cantera's interpolation rule, including several rates per pressure |
| Chebyshev | per reaction, `T` and `p` ranges and the coefficient matrix |
| Reverse | reversible flag, explicit reverse rates where given, `sum nu` for `Kc` |

Reaction ordering within a type is by number of reactants and products, so
neighboring lanes run the same loop trip counts.

### Rates

Per cell (one thread, or one team with reactions across lanes):

1. Concentrations `C_k = rho Y_k / W_k`; `g_k / (R_u T)` from the NASA
   polynomials (one evaluation per species, shared by all reactions).
2. Forward rate constants in log form: `k_f = exp(log A + b log T - Ea / (R_u T))`.
3. Third-body concentrations; falloff blending `Pr`, `F` (Troe/SRI); PLOG
   interpolation at `p = rho R T`; Chebyshev.
4. Reverse rate constants from `Kc = exp(-sum_k nu_k g_k / (R_u T)) (p_atm / R_u T)^sum nu`,
   unless explicit.
5. Rates of progress `q_i = k_f prod C^nu' - k_r prod C^nu''`, and net
   production `omega_k = sum_i nu_ki q_i` accumulated in registers (small
   mechanisms) or team scratch memory (large ones).

### Jacobian

Analytical, from the same tables: `d omega / d C` from the rates of progress
(including third-body and falloff derivatives, and `d k / d T` for the `T`
column), then chain-ruled to `(Y, T)`.

- **Dense** for `Ns` up to a threshold (default 100): `(Ns + 1)^2` doubles in
  team scratch (level 0 shared memory where it fits, level 1 global scratch
  above that), LU with partial pivoting.
- **Sparse** above it: the sparsity pattern of `I - gamma h J` is fixed by the
  mechanism, so the host computes a fill-reducing ordering and the symbolic
  LU (pattern of L and U) once at setup; the device does a numeric
  factorization without pivoting over the fixed pattern, with a diagonal
  check that falls back to a smaller step. Zero-RK's GPU path refactorizes
  on a fixed pattern in the same way (cuSolverRf); whether static pivoting is
  robust enough across our mechanisms is checked in milestone 10, with
  threshold partial pivoting within the pattern as the fallback.
- Approximations that keep the integrator converging while cutting cost
  (dropping the third-body and falloff concentration derivatives, as some codes do)
  are a tuning option, measured against the exact Jacobian.

Unit tests compare the Jacobian against finite differences of the rates, and
the rates against Cantera, at random states.

### Integrator

The first integrator is an adaptive Rosenbrock method with an embedded error
estimate (a stiffly accurate, L-stable RODAS-type scheme, or ROS4; Hairer &
Wanner, *Solving ODEs II*, ch. IV.7), integrating the
constant-volume reactor over the splitting step:

- one Jacobian and one LU per sub-step, a fixed number of stages, no Newton
  iteration: a predictable instruction stream, which matters for SIMT
  efficiency, and simple to implement against our own Jacobian;
- error control on `Y` (absolute tolerance, default 1e-10) and `T`
  (relative, default 1e-6); first sub-step size from the previous step's last
  accepted sub-step in that cell (stored, `h_last`);
- positivity: negative `Y_k` beyond `-atol` reject the sub-step; small
  negatives are clipped and `Y` renormalized at the end of the splitting step,
  with the energy unchanged (so `T` follows).

A variable-order BDF integrator (CVODE-like, reusing the Jacobian over many
steps) is more efficient for large mechanisms with loose tolerances and is a
later milestone behind the same interface, either our own or Kokkos Kernels'
batched BDF if it proves adequate. An explicit stabilized integrator (RKC/ROCK)
is an option for mildly stiff cases (preheat zones, very small `dt`).

```cpp
struct ChemistryIntegrator {
    // Advance every active cell over dt at constant (rho, e).
    // Y, T, h_last are updated in place; forcing is optional (SDC-style coupling).
    virtual void advance(double dt, CellQueue active, SpeciesView Y, View T,
                         View h_last, ConstView rho, ConstView e,
                         std::optional<ForcingView> forcing, CostView cost) = 0;
};
```

### Precision

Chemistry runs in double in every build. Float builds convert the cell's
`(rho, e, Y)` to double on entry and back on exit. Reasons: equilibrium
constants involve exponentials of differences of large Gibbs energies;
`I - gamma h J` at large `h` is ill conditioned in float; and Newton-type
convergence tests near float round-off stall. A100 runs FP64 at half the FP32
rate, so the cost is modest. Float builds otherwise lose about seven digits of
`sum Y = rho`, which the renormalization absorbs.

### Determinism

Each cell's integration depends only on that cell's inputs, so the result is
independent of thread, team and rank assignment, and of the chemistry load
balancing, provided two rules hold: team reductions within a cell use a
fixed order (explicit loops over a fixed lane count, not `parallel_reduce`
with an implementation-defined tree), and the per-cell code path (thread or
team, team size) depends only on the mechanism and the build, never on the
cell, its bin or its rank, so binning and load balancing only reorder work.

## 6. Transport

### Models

| Model | Viscosity | Conductivity | Species diffusion | Cost per cell |
|---|---|---|---|---|
| `mixture_averaged` | Wilke | Mathur et al. (average of mole-fraction-weighted arithmetic and harmonic means) | Hirschfelder-Curtiss mixture-averaged `D_km` from binary `D_ij`, with correction velocity | `O(Ns^2)` |
| `unity_lewis` | Wilke (or Sutherland/constant from input) | Mathur (or `mu cp / Pr`) | `D_k = lambda / (rho cp)` for all `k` | `O(Ns)` |
| `constant_lewis` | as above | as above | `D_k = lambda / (rho cp Le_k)`, `Le_k` per species from input | `O(Ns)` |

These match Cantera's `mixture-averaged` and `unity-Lewis-number` models
([docs](https://cantera.org/stable/reference/transport/index.html)), which
follow Kee, Coltrin & Glarborg, *Chemically Reacting Flow*
([Wiley](https://doi.org/10.1002/0471461296)), with
[Wilke 1950](https://doi.org/10.1063/1.1747673) viscosity and
[Mathur, Tondon & Saxena 1967](https://doi.org/10.1080/00268976700100731)
conductivity, so the flame comparisons isolate Mallard's numerics.

### Species properties

Pure-species viscosity, conductivity and binary diffusion coefficients use
the polynomial fits in `log T` that Cantera computes from the Lennard-Jones
and polarity data in the YAML file (collision integrals of Monchick & Mason,
fitted over the mechanism's temperature range). The fitting runs once on the
host at setup. It is a port of Cantera's `GasTransport` fitting (BSD-3
licensed, compatible with Mallard's AGPL with attribution); the test suite
checks the fits against Cantera's to 1e-6 relative.

### Fluxes

At each face, with face-averaged `rho`, `T`, `Y` and face gradients built like
today's (averaged least-squares cell gradients with the normal correction):

```text
j_k  = -rho D_km (W_k / W) grad X_k + rho Y_k V_c,    V_c = sum_k D_km (W_k / W) grad X_k
q    = -lambda grad T + sum_k h_k j_k
```

- The correction velocity `V_c` makes `sum_k j_k = 0`, so mass is conserved
  and `rho` needs no diffusion term.
- Species gradients are of `X_k` (mole fractions), computed per cell for all
  species with the existing gradient stencils (shared weights, one pass over
  the species).
- Walls: zero normal diffusive flux (non-catalytic), the existing one-sided
  `T` and velocity treatment. Transmissive and outflow boundaries: zero normal
  gradients, as today.
- The mixture-averaged `D_km` is `O(Ns^2)` per cell, which dominates the
  transport cost for large mechanisms. An input option computes it once per
  time step instead of per RK stage (frozen transport coefficients: first
  order in `dt` for the coefficients but small, since they vary slowly; the
  two-CFL flame test measures it), and `constant_lewis` avoids it.

### Inviscid runs

`type = "euler"` with a mechanism means reactive Euler: species are advected
and react, and no transport data is needed (species in the YAML without
transport data are accepted). Numerical diffusion then sets the reaction-zone
structure, which is the standard model for detonation studies; the validation
suite measures resolution in points per half-reaction length.

## 7. Mechanism input

### Format

Cantera YAML ([format reference](https://cantera.org/stable/yaml/index.html)),
read in C++ with [yaml-cpp](https://github.com/jbeder/yaml-cpp) (MIT,
FetchContent) at startup on every rank (mechanism files are small). Supported:

| Feature | First | Later |
|---|---|---|
| `ideal-gas` phase, `elements`, `species` | yes | |
| Thermo: `NASA7`, `NASA9`, `constant-cp` (the last for verification against the perfect-gas solver) | yes | |
| Reactions: elementary (`Arrhenius`), `three-body`, explicit reverse rates, `duplicate`, non-integer `orders` | yes | |
| Falloff: Lindemann, Troe, SRI | yes | |
| `pressure-dependent-Arrhenius` (PLOG), `Chebyshev` | | milestone 8 |
| `chemically-activated`, `Blowers-Masel`, `two-temperature-plasma`, interface and electrochemical reactions | | out of scope |
| Transport: `gas` model data (LJ, polarizability, rotational relaxation, dipole) | yes | |
| Units (`units:` sections), `phase` selection by name | yes | |

Unsupported entries are errors that name the reaction, never silently dropped.
Chemkin files are converted with Cantera's `ck2yaml` once, which also checks
them.

### TOML

```toml
[physics]
type = "navier_stokes"            # or "euler"
gas = "mixture"                   # "perfect" (default) keeps the gamma/R/mu inputs
mechanism = "mechanisms/h2o2.yaml"
phase = "ohmech"                  # optional; default: the first phase
transport = "mixture_averaged"    # "unity_lewis", "constant_lewis"; ignored for euler
lewis = { H2 = 0.3, H = 0.18 }    # constant_lewis; others default to 1

[chemistry]
enabled = true                    # false: non-reacting multicomponent flow
integrator = "rosenbrock"         # later "bdf", "rkc"
coupling = "strang"               # later "sdc"
rtol = 1.0e-6
atol = 1.0e-10
T_frozen = 0.0                    # no chemistry below this T
load_balance = true               # MPI: redistribute chemistry cost (section 4)

[initialize]
type = "constant"
u = [0.0, 0.0]
p = 101325.0
T = 300.0
X = { H2 = 2.0, O2 = 1.0, AR = 7.0 }     # or Y = {...}; normalized; unlisted species are 0

# analytical: one expression per listed species, plus "balance" for the rest
# Y = { H2 = "x < 0.5 ? 0.028 : 0.0", O2 = "x < 0.5 ? 0.226 : 0.233" }
# balance = "N2"

[[boundaries]]
zone = "left"
type = "upt"
u = [10.0, 0.0]
p = 101325.0
T = 300.0
X = { CH4 = 1.0, O2 = 2.0, N2 = 7.52 }
```

`X`/`Y` are accepted wherever a boundary or initial state takes `T`.
A `[chemistry]` table without `gas = "mixture"` is an error.

## 8. Boundary conditions

Every boundary keeps its meaning; the exterior state gains `Y` and the
surrogate `(gamma, e0)`:

| Boundary | Species | Thermo |
|---|---|---|
| `extrapolation`, image faces | from the image | from the image |
| `symmetry`, walls | interior `Y` (zero normal diffusive flux) | interior; isothermal walls use `R(Y)` |
| `upt`, `farfield`, `dirichlet` | given `Y`/`X` (expressions for `dirichlet`) | from given `p`, `T`, `Y` |
| `p_out`, `p_out_average` | interior `Y` | interior `gamma` |

`farfield` uses the Riemann invariants with each side's frozen `gamma`.
Catalytic walls and species-specific wall fluxes are out of scope.

## 9. Output and restart

- **VTU variables**: `Y_<name>`, `X_<name>` (any species, or `Y_*`, `X_*`
  for all), `T`, `HRR` (heat release rate `-sum_k h_k W_k omega_k`),
  `OMEGA_<name>`, `MW`, `GAMMA`, `CP`, `CHEM_COST` (sub-steps in the last
  step), and with transport `MU`, `LAMBDA`, `D_<name>`.
- **Restart format version 2**: the header gains the list of variable names,
  so restarts map fields by name: `RHO`, `RHOU_*`, `RHOE`, then
  `RHOY_<species>`, then the auxiliary fields `T` and `CHEM_H` (last
  chemistry sub-step). The auxiliary fields make restarted runs bit-identical
  (they seed Newton and the chemistry step size). Reading checks the species
  names against the mechanism; a restart can start a reacting run from a
  non-reacting multicomponent one, and a mechanism change that keeps names
  (e.g. a reduced mechanism) maps by name, with missing species set to zero
  after an explicit `allow_missing_species = true`. Version 1 files still
  read for perfect-gas runs.
- **Diagnostics**: `check_interval` prints min/max `T`, `max |sum Y - 1|`,
  the number of active chemistry cells and the max/mean sub-steps, and the
  chemistry wall time fraction.
- **0D tool**: `MallardReactor`, a small executable built from the same
  kernels, runs constant-volume or constant-pressure reactors from TOML and
  writes CSV. It backs the 0D validation and the chemistry benchmarks without
  a mesh.

## 10. Milestones

Each is a reviewable PR with its tests; nothing reacting is user-visible
until milestone 8 (the 0D tool arrives in milestone 7).

1. **State split.** `State {flow, species}` with zero species everywhere:
   time integrators, `axpby`, halo exchange, restart v2 with names.
   Bit-identical results; restart v1 still read. (Species conservation sums
   come with the first species, in milestone 3.)
2. **Mechanism and thermo.** yaml-cpp, Cantera YAML reader (species, NASA-7/9,
   units), device thermo tables, `T` from `e`. Tests against Cantera tables
   (generated by a Python script in `tools/`, committed as small CSV files).
3. **Non-reacting mixtures, first order and MUSCL.** Gas-model template on
   the RHS kernels, Riemann solvers on `(W, gamma, e0)` (Rusanov, HLL, HLLC),
   species fluxes, shared-limiter MUSCL, `X`/`Y` in initial and boundary
   conditions, VTU species output, species conservation sums. Tests: constant-`cp` single-species mixture
   reproduces the perfect-gas solver; species interface advection;
   multicomponent shock tube.
4. **TENO for mixtures.** Primitive characteristic reconstruction of the
   flow block, shared-stencil scalars, `theta` bounds, `W`-based troubled
   indicator. Tests: design order on smooth species advection; interface and
   shock tube tests at TENO5.
5. **Double flux** (input option). Tests: interface advection keeps `p`, `u`
   uniform to round-off; shock tube energy error reported.
6. **Kinetics.** Rate tables, elementary/three-body/falloff, production
   rates and analytical Jacobian. Tests: rates vs Cantera at random states;
   Jacobian vs finite differences.
7. **0D reactor and integrator.** Rosenbrock with dense LU; `MallardReactor`.
   Tests: ignition delays and equilibrium end states vs Cantera.
8. **Coupling.** Strang splitting in the solver; `[chemistry]` input;
   active-cell queue; PLOG and Chebyshev. Tests: reactive shock tube, 1D
   detonation (CJ speed, ZND structure).
9. **Transport.** Species fits, mixture-averaged, unity and constant Lewis,
   diffusion fluxes with correction velocity, enthalpy diffusion, viscous
   time step. Tests: properties vs Cantera; premixed flame speed.
10. **GPU performance.** Team-per-cell kernels, layout study, cost binning,
    sparse LU for large mechanisms, benchmark suite with recorded baselines.
11. **MPI.** Chemistry load balancing; chemistry cost in partition weights.
    Tests: rank-count independence with chemistry; imbalance benchmark.
12. **Extensions, each optional and driven by need:** Roe/RHLL for mixtures;
    BDF integrator; SDC coupling; mechanism-specialized (code-generated)
    kernels; 2D cellular detonation, counterflow flame and mixing-layer
    cases.

## 11. Validation suite

Three tiers, all automated:

- **Unit** (GoogleTest, every PR, seconds): exact or round-off criteria.
- **Verification** (GoogleTest, every PR, under a minute on CPU): small runs
  with reference data in `test/data/`.
- **Validation and benchmarks** (`validation/`, run per milestone and before
  releases, CPU and A100): full cases with scripts that produce the
  comparison plots and a pass/fail table.

Reference data from Cantera is produced by Python scripts in `tools/` with
the mechanism file used by the run, and committed as small CSV files with the
Cantera version recorded, so the tests themselves need no Cantera. Same
mechanism, same thermo and transport, so differences are Mallard's numerics.

Mechanisms used throughout: H2/O2 of
[Burke et al. 2012](https://doi.org/10.1002/kin.20603) (13 species with Ar,
He, N2), GRI-Mech 3.0 (53 species, 325 reactions;
[source](http://combustion.berkeley.edu/gri-mech/version30/text30.html), ships
with Cantera), and for performance a ~100-species skeletal mechanism (e.g.
HyChem Jet-A, [Wang et al. 2018](https://doi.org/10.1016/j.combustflame.2018.07.012))
and a detailed ~500-1000-species mechanism (LLNL n-heptane,
[Curran et al. 1998](https://doi.org/10.1016/S0010-2180(97)00282-4), or
iso-octane, [Curran et al. 2002](https://doi.org/10.1016/S0010-2180(01)00373-X)).
Species counts are taken from the files when they are added.

### Unit and verification tests

| Test | Reference | Pass criterion |
|---|---|---|
| YAML reader | Cantera's parsed values | Every coefficient of the three mechanisms equal to Cantera's after unit conversion (relative 1e-14) |
| Thermo: `cp`, `h`, `s`, `g` per species and mixture, `T(e)` | Cantera at 200 random `(T, Y)` | Relative 1e-12 (double), 1e-5 (float); `T(e(T))` within 1e-10 K |
| Production rates and rates of progress, every reaction type | Cantera `net_production_rates` at 1000 random `(T, p, Y)` | Relative 1e-10 where `|omega| > 1e-12 max |omega|` |
| Analytical Jacobian | Centered finite differences of our rates | Relative 1e-6 per entry (scaled by row norm) |
| Transport fits and mixture properties | Cantera `mixture-averaged` | Relative 1e-6 |
| Perfect-gas regression | Current solver | All existing tests bit-identical without a mechanism |
| Single-species constant-`cp` mixture | Perfect-gas solver, same `gamma`, `R` | Sod and 2D Riemann: agreement to 1e-12 (different arithmetic, same scheme) |
| `sum Y = 1`, positivity, conservation | Exact | `max |sum_k rho Y_k - rho| / rho < 1e-13` (1e-6 float) after 1000 steps of a multispecies Riemann problem; `min Y_k >= 0`; each `sum_cells V rho Y_k` conserved to round-off without chemistry |
| Species advection, smooth | Exact (translated profile) | Design order of MUSCL and TENO3-5 on triangles, quads, tets, hexes |
| Rank and thread independence | Single rank, one thread | Reacting 2D case on 1-4 ranks and 1/4 threads: bit-identical |
| Restart | Uninterrupted run | Reacting case restarted mid-run: bit-identical |

### Validation cases

| # | Case | Reference | Pass criterion | Milestone |
|---|---|---|---|---|
| V1 | **0D constant-volume ignition delay**, H2/air (Burke) and CH4/air (GRI-3.0), `T0` 1000-1500 K, `phi` 0.5-2, 1 and 10 atm (`MallardReactor` and a single-cell solver run) | Cantera `IdealGasReactor` at tight tolerances | Ignition delay (max `dT/dt`) within 0.5%; `T(t)` and major species within 1% over the trajectory | 7 |
| V2 | **Equilibrium end states** of V1 | Cantera `equilibrate("UV")` | `T` within 0.1 K, major species within 1e-4 absolute | 7 |
| V3 | **Large-mechanism ignition** (~100 and ~500+ species) | Cantera | As V1; also the performance baseline | 7, 10 |
| V4 | **Species interface advection**, 1D and 2D (skewed triangles), H2 or He into air at different `T`, uniform `p` and `u` | Exact: uniform `p`, `u` | Conservative scheme: report max `|p - p0| / p0` and require it to converge under refinement; double flux: `p`, `u` uniform to 1e-12 after 10 flow-throughs | 3, 5 |
| V5 | **Thermally perfect multicomponent shock tube** (non-reacting, [Fedkiw, Merriman & Osher 1997](https://doi.org/10.1006/jcph.1996.5622) type, H2/O2/Ar) | Exact Riemann solution for the thermally perfect mixture (our own iterative solver in `tools/`, Cantera thermo) | L1 errors of `rho`, `u`, `p`, `T`, `Y` converge at the expected rate (about 1 for discontinuous data); no overshoot of `Y` | 3, 4 |
| V6 | **Reactive shock tube**: H2:O2:Ar = 2:1:7, reflected-shock ignition turning into a detonation (Fedkiw et al. 1997; [Martinez Ferrer et al. 2014](https://doi.org/10.1016/j.compfluid.2013.10.014) for the viscous version and a resolution study) | Converged Mallard run and published profiles at 170 and 230 us | Detonation front within 2 cells of the converged position; peak `T` and `p` converge under refinement | 8 |
| V7 | **1D CJ detonation**, 2H2-O2-7Ar at 6.67 kPa and H2/air, driven by an overdriven start | CJ speed and ZND profile from the [Shock and Detonation Toolbox](https://shepherd.caltech.edu/EDL/PublicResources/sdt/) (Cantera-based, same mechanism) | Mean front speed within 1% of `D_CJ` at 20+ cells per half-reaction length; ZND induction length within 5% and von Neumann state approached under refinement | 8 |
| V8 | **Premixed laminar flame speed**, H2/air and CH4/air at `phi` = 0.6-1.4, 1 atm, mixture-averaged and unity Lewis | Cantera `FreeFlame` (same transport model) | Flame speed within 2% at 20+ cells per thermal thickness; difference between CFL 0.4 and 0.2 below 0.5% (splitting error); temperature and HRR profiles overlaid | 9 |
| V9 | **2D cellular detonation**, 2H2-O2-7Ar at 6.67 kPa | [Oran et al. 1998](https://doi.org/10.1016/S0010-2180(97)00218-6), [Gamezo, Desbordes & Oran 1999](https://doi.org/10.1016/S0010-2180(98)00031-5), Deiterding's AMROC results; experimental cell widths | Regular cells; cell width within published numerical range at matched resolution; soot-foil (max `p`) image | 12 |
| V10 | **Counterflow diffusion flame**, H2/N2 vs air, strain-rate sweep | Cantera `CounterflowDiffusionFlame` | Peak `T` vs strain within 2% near the axis, accepting that a 2D/3D opposed-jet run only approximates the similarity solution | 12 |
| V11 | **Shock/H2-bubble interaction** with detailed transport, or a reacting mixing layer | [Billet, Giovangigli & de Gassowski 2008](https://doi.org/10.1080/13647830701545875) | Code-to-code: interface and shock positions; grid convergence | 12 |

### Performance benchmarks

- **Chemistry throughput**: cells advanced per second per A100 (and per CPU
  node) for one splitting step of `dt` = 1e-8 s (detonation regime) and 1e-6 s
  (flame regime), on states sampled from V1/V3 trajectories (mix of
  fresh, igniting and burnt), versus species count (Burke 13, GRI 53, ~100,
  ~500+). Reported with the sub-step histogram and against Cantera/CVODE on one
  CPU core, and against the published GPU numbers of
  [Niemeyer & Sung 2014](https://doi.org/10.1016/j.jcp.2013.09.025),
  [Curtis et al. 2017](https://doi.org/10.1016/j.combustflame.2017.02.005) and
  [Balos et al. 2024](https://doi.org/10.1177/10943420241280060) where the
  mechanisms match.
- **Imbalance**: the same with 1% of cells igniting, which measures the
  team-per-cell and binning strategies.
- **Full solver**: time per cell-step and the chemistry/flow split for V7 and
  V8 in 2D and 3D; multi-GPU weak scaling of V9 with and without chemistry
  load balancing.
- Baselines are recorded per milestone; a regression above 10% blocks a PR
  touching the chemistry kernels.

## Risks

| Risk | Mitigation |
|---|---|
| Chemistry dominates run time on large mechanisms and our integrator is slower than mature libraries | Benchmarks against published throughput from milestone 7; the integrator interface lets a BDF or codegen path, or a library backend, replace Rosenbrock without touching the solver |
| Conservative scheme gives unacceptable pressure oscillations for injection-type problems | Species-interface test quantifies it from milestone 3; double flux is designed in and scheduled |
| Shared-stencil species reconstruction is too dissipative for thin radical layers | TENO smooth cells already use the high-order central stencil; measured by the flame and ZND tests; per-species stencil selection can be added for selected species |
| Strang splitting error in steady flames | Two-CFL flame test; SDC-style forcing designed into the integrator interface |
| GPU register pressure in rate kernels for large mechanisms | Team-per-cell with scratch memory; reactions grouped by type; codegen path as a later option |
| Static partitions become badly imbalanced at fronts | Chemistry-state load balancing (no mesh change), then dynamic rebalancing per mpi.md |
| Float builds produce wrong chemistry | Chemistry in double always; float validated on non-reacting cases and on 0D tests only |
| Porting Cantera's transport fitting is subtle | Property tests to 1e-6 against Cantera; `unity_lewis` works without it |

## Decisions on the open questions

Accepted by the user (2026-10-02):

1. **Detonations first** (milestone 8 before transport in milestone 9).
2. **Own chemistry core** (~3-5k lines) rather than waiting for a Kokkos 5 TChem.
3. **Dependencies**: yaml-cpp via FetchContent; Cantera (Python) only for
   generating committed reference data in `tools/`.
4. **Double flux** as a non-conservative input option; conservative by default.
5. **HLLC first**; Roe/RHLL for mixtures stay in milestone 12.
6. **Chemistry always in double**, also in float builds.
7. **Scope of "done"**: as in milestone 12, the 2D cellular detonation and
   counterflow flame are extensions, not requirements for the first
   reacting release.
