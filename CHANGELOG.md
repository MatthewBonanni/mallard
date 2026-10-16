# Changelog

All notable changes to Mallard are listed here, newest first, in the format of
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Mallard uses
[semantic versioning](https://semver.org/): before 1.0, a minor release may
change results and inputs, and every such change is listed under **Changed**
with what to do about it. Numbers in parentheses are pull requests or issues.

Each pull request that changes `src/` adds a line under **Unreleased**
(see [Writing documentation](docs/design/documentation.md#changelog)). Release
notes are generated from this file: `python3 tools/changelog.py notes 0.9.0`.

## [Unreleased]

### Added

- Solution remap between meshes: `[initialize] type = "remap"` starts a run from a restart file of another mesh (any resolution, cell type, partition or rank count), conservatively by supermesh intersection (#353).
- Asynchronous output: files are written on a background thread while stepping continues (`[output] async`, default on; `io_threads`); files are byte-identical to synchronous output (#309).
- Compressed output: `[[write_data]] compression = "zlib"` for VTU and HDF5 (readable by stock ParaView and h5py), error-bounded lossy `relative_error`, and compressed restart files (format version 5) (#309).
- Numerics presets: `[numerics] preset = "robust" | "accurate" | "les" | "reacting"` fills the scheme keys an input leaves unset (#391).
- Mesh readers for CGNS (optional, `-DMallard_ENABLE_CGNS=ON`) (#354), ANSYS Fluent `.msh`/`.cas` (#357) and OpenFOAM polyMesh (#361); `[mesh.transform]` scales, rotates and translates a mesh file as it is read (#350).
- Isothermal (`wall_isothermal`) and heat-flux (`wall_heat_flux`) walls for gas mixtures (#351); `tools/flame_wall.py` for head-on flame quenching against a cold wall (#381).
- Tutorials: supersonic flow over a cone (3D) (#371) and wall-resolved channel LES at Re_τ = 180 (#375).
- User guide by topic with figures from `tools/docs` (#364), theory manual (#372), equations in LaTeX (#370).

### Changed

- **The default TFLES efficiency is `"dynamic"`** (#310): `[les.combustion] efficiency` now defaults to Charlette et al.'s saturated power law with a dynamic exponent, which changes TFLES results. `efficiency = "charlette"` restores the 0.8.0 model; `beta`, `subgrid_velocity` and `C_u` are input errors with `"dynamic"`.
- **The stable time step includes the exterior states of boundary faces** (#374): a fast inflow into slower gas now sets `dt` from the first step, so runs whose boundary states are faster than the adjacent cells take smaller steps.
- **The hybrid convective flux with SIMPLER coupling is rejected** (#300): the combination is linearly unstable and was accepted silently; use Strang coupling or `convective_flux = "riemann"`.
- Distributed runs skip the viscous gradients of halo cells beyond the first layer unless LES models are on (#365); bitwise unchanged.

### Fixed

- With `double_flux` and `navier_stokes`, the diffusive energy flux reached only one side of each face (steady conduction between walls ended 500 K off) (#351). Changes results of double-flux viscous runs only.
- Distributed runs with `double_flux` and chemistry differed from serial ones by about 1e-7: the halo is refreshed before freezing the thermodynamics (#378).
- The 3D TENO setup no longer fails with "does not fit in memory" on faceted curved boundaries that touch more than 32 mirror planes, such as the far field of `sphere_mach3` (#392).
- A fast inflow into gas at rest no longer blows up the first step (#374).

## [0.8.0] - 2026-10-15

Curved high-order boundaries, a GPU-resident 3D TENO setup, multi-jagged
partitions, 2× faster reacting mixtures, SIMPLER splitting, Soret diffusion and
optically thin radiation.

### Added

- Curved high-order boundaries (#263): Gmsh 2.2/4.1 boundary elements of order 2 and 3 (kept by `mallard-mesh-convert`) or `[[mesh.curved]]` projection onto a circle, sphere or cylinder; TENO-E orders 3–6 reach their design accuracy at curved walls (`curved_mirrors`, `curved_wall_degree`). `tools/make_annulus_mesh.py`; `--order` for the cylinder and sphere mesh tools.
- SIMPLER balanced splitting: `[chemistry] coupling = "simpler"` (#235).
- Thermal (Soret) diffusion: `physics.soret = true` (#243).
- Optically thin radiation: `[radiation]`, output `QRAD` (#226).
- CUDA graph per step: `[run] cuda_graphs`, default on in CUDA builds (#217).
- NCCL halo exchange: `Mallard_ENABLE_NCCL`, `[parallel] halo_exchange = "nccl" | "mpi"` (#217).
- Chemistry load balancing across MPI ranks: `[chemistry] load_balance = true` (#255, #262).
- Opt-in `single_precision_tables` for 3D TENO (#265) and `chemistry.C_reg` (#264).
- Examples `flame_vortex_quenching`, `premixed_flame/ch4` and `volvo_bluff_body`; performance cases `flame_3d` and `air_3d`; mechanisms `ch4_bfer.yaml` and `c3h8_2step.yaml` (#226, #241, #252).
- Builds with CUDA 13 (#259, #260).

### Changed

- **RHLL blends toward HLL across faces that carry a shock** (#248): RHLL results change wherever a face carries a pressure jump; faces with equal pressures are bitwise unchanged. Other Riemann solvers are unchanged.
- **Multi-jagged partitions are the default** (#261): results are bitwise identical on any rank count, but per-rank output pieces and TENO caches differ from 0.7.0's. `[parallel] partitioner = "hilbert"` restores the 0.7.0 partitions.
- **Gmsh high-order boundary elements curve their faces** (#263): 0.7.0 read them by their corners. `[mesh] curved_geometry = false` restores straight faces; linear meshes are bitwise unchanged.
- 3D TENO tables on macOS arm64 change at round-off (no FMA contraction in the setup) (#247).
- TENO cache format `MALLARD-TENO-10`; older caches are rebuilt (#247, #265).
- The 3D TENO setup runs on the GPU, 6–9.5× faster, bitwise identical (#247); 3D TENO device memory is about 15% lower (#265); MUSCL reconstruction is fused and overlapped with the halo exchange (#239); halo exchanges carry only the cells the stencils read (#258); reacting mixtures are 2× faster per step (#257, #264); large mechanisms solve faster on GPUs (#246).
- `-Wconversion -Wsign-conversion` are on, as errors under `Mallard_WARNINGS_AS_ERRORS` (#267, #269–#276).

### Fixed

- `[integrals] budget = true` with `double_flux = true` threw "Double flux needs the cells' thermodynamics frozen" (#252).
- The 3D TENO host retry wrote compacted cells out of bounds (#263).
- Latent truncations of `Solver::get_step()`, mesh local indices and LU substitution indices (#267).
- The radiation test failed to compile with nvcc under `-Werror` (#268).

## [0.7.0] - 2026-10-09

Large-eddy simulation with explicit subgrid-scale and combustion models, a
kinetic-energy-preserving hybrid flux, synthetic turbulent inflow, and 2.2×
faster 3D TENO. First release under Apache-2.0.

### Added

- LES eddy-viscosity models: `[les] model = "sigma" | "wale" | "vreman" | "smagorinsky"`, output `MU_T` (#203); Scotti filter width, the 3D default (#223); opt-in dynamic constant `[les] dynamic = true` (#223).
- Hybrid KEP + Ducros flux: `[numerics] convective_flux = "hybrid"` (#205).
- Kinetic-energy budget: `[integrals] budget = true` (#203).
- Thickened-flame combustion model `[les.combustion] model = "tfles"`, outputs `TF_F`, `TF_E`, `TF_OMEGA` (#215); experimental PaSR (`model = "pasr"`, output `PASR_KAPPA`) (#224); `subgrid_velocity = "eddy_viscosity"` (#222).
- Synthetic turbulent inflow for `nscbc_inlet` through `[boundaries.turbulence]`; inlet `u` accepts expressions (#221).
- 3D single-precision builds pass every test and run in CI (#237).
- Examples `autoignition_2d` (#174), `detonation_3d` and `shock_bubble_3d` (#183), `cbc_les` (#213), `channel_les` (#214, #230), `flame_turbulence` (#222), `channel_inflow_retau180` and `turbulent_inflow_duct` (#221); `mechanisms/he_air.yaml`.

### Changed

- **Licensed under Apache-2.0** (#231), with a `NOTICE` file; 0.1.0–0.6.0 remain AGPL-3.0-only.
- **Example CFLs raised toward the measured stability limits** (#210): most examples run at `cfl = 1.0`; their results are not bitwise equal to 0.6.0.
- **`channel_retau180` uses the hybrid flux** (#216).
- **3D TENO stencils are bounded by a Lebesgue constant of 4** (#232): changes 3D TENO results on tetrahedra, pyramids, mixed meshes and some prisms and hexahedra next to walls; TENO cache format `MALLARD-TENO-8`.
- **SSPRK3's last stage is `U + 2/3 (U2′ − U)`** (#237): results change at round-off; unstretched generated 3D meshes end exactly at `L`.
- **NSCBC transverse terms** use edge neighbors in 3D, skip faces next to non-characteristic boundaries and reversed outlet faces, and continue across periodic seams (#238, #221): 3D NSCBC results change.
- Restart files of runs with characteristic boundaries use format version 4, which 0.6.0 cannot read (#219).
- MallardReactor's ignition criterion is the peak of dT/dt (#220).
- `eps_numerical` is taken from the scheme's discrete pressure work, new column `pressure_work` (#222).
- 3D TENO smooth pass 2.2× faster on GPUs (#212); TENO setup host memory 6× lower on CUDA (#227).

### Fixed

- NSCBC runs restart bitwise (#219).
- `nscbc_outlet` no longer diverges when eddies reverse the flow (#238).
- A turbulent outflow could make the NSCBC outlet pressure drift (#221).
- 3D TENO grew modes next to walls and on jittered meshes (#232).

## [0.6.0] - 2026-10-05

Axisymmetric flows, non-reflecting boundaries, in-situ statistics, parallel
HDF5 output and AMD GPUs.

### Added

- Axisymmetric (r–z) flows: `[physics] axisymmetric = true` in 2D builds (#184).
- Characteristic boundaries `nscbc_outlet` and `nscbc_inlet`, and `[[sponges]]` (#176).
- Roe and RHLL for gas mixtures (#187).
- In-situ statistics `[statistics]` and probes `[[probes]]` (#169); `tools/plane_average.py`.
- Parallel HDF5 output with XDMF: `[[write_data]] format = "hdf5"` (#179).
- Output variables `Q` and `VORTICITY` (#139); `[integrals]` columns `velocity_squared`, `vorticity_squared`, `density_squared`, `temperature`, `temperature_squared` (#170).
- Inflow compositions as expressions of `x`, `y`, `z` (#167); `viscosity_model = "power_law"`, `[source] mass_flow` and `[mesh] stretching` (#170, #172).
- AMD GPUs (Kokkos HIP) (#180).
- `cache_single_precision` for TENO caches (#151).
- GPU performance suite `benchmarks/perf/` and `tools/perf_suite.py` (#168); `tools/animate_cartesian.py` (#178).
- Examples `isotropic_turbulence`, `triple_flame`, `flame_vortex`, `channel_retau180`, `counterflow_diffusion`, `poiseuille_pipe`, `noh_axisymmetric`, `sedov_axisymmetric`, `sphere_axisymmetric` and `flame_outflow`.

### Changed

- **Conventional CFL normalization** (#175): the same `cfl` gives twice the time step as in 0.5.0. **Halve `cfl` in inputs written for 0.5 or earlier to keep their results.** The `CFL` output halves accordingly.
- **3D analytical initial conditions** use a degree-5 tetrahedron rule and `n_subdivisions = 1` by default (#188): much faster, but 3D initial states differ at round-off (smooth fields) or in cells cut by a discontinuity.
- **Double-flux mixtures with HLL, Rusanov and HLLC** take wave speeds from each side's own frozen `(gamma, e0)` (#187).
- TENO-E setup 5.6× faster; results change at round-off (#181).

### Fixed

- Fractional reaction orders below 1 gave NaN Jacobians before ignition (#177).
- Monitor files of a run started from a restart file get their header (#170).
- The single-precision build compiles and passes again (#185, #190).

## Earlier releases

Versions 0.1.0–0.5.0 are described in their
[GitHub releases](https://github.com/MatthewBonanni/mallard/releases).

[Unreleased]: https://github.com/MatthewBonanni/mallard/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/MatthewBonanni/mallard/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/MatthewBonanni/mallard/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/MatthewBonanni/mallard/compare/v0.5.0...v0.6.0
