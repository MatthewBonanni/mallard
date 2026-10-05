# Input file reference

Mallard reads a single [TOML](https://toml.io) file: `Mallard -i input.toml`.
Kokkos options such as `--kokkos-num-threads=N` or `--kokkos-device-id=N` can be appended.

The spatial dimension is fixed at build time with the CMake option
`-DMallard_DIM=2` (default) or `3`. Vectors in the input (`u`, `gravity`,
`rhou`) have that many components, and expressions are in `x`, `y`, `z`
(`z` is 0 in 2D) and, where noted, `t`. Below, `[u_x, u_y]` reads
`[u_x, u_y, u_z]` in 3D.

## `[run]`

| Key | Description |
|---|---|
| `cfl` | CFL number; the time step is `cfl` times the stable step of every cell. Exactly one of `cfl` and `dt` is required. |
| `dt` | Fixed time step |
| `t_stop` | Stop at this simulation time (the last step is shortened to land on it) |
| `n_steps` | Stop after this many steps |
| `t_wall_stop` | Stop after this many seconds of wall time |

At least one stop condition is required.

## `[mesh]`

| Key | Description |
|---|---|
| `type` | `file`, `cartesian` (quads), `cartesian_tri` (each quad split into two triangles along its bottom-left to top-right diagonal), or `wedge` (quads over an 8 degree compression ramp starting at x = 0.5) |
| `filename` | (`file`) Mallard HDF5 mesh (`.h5` or `.hdf5`, see below), or ASCII Gmsh mesh, format 2.2 or 4.1, of linear triangles and/or quadrilaterals (2D), or tetrahedra, pyramids, prisms and/or hexahedra (3D) |
| `Nx`, `Ny` | Number of quads in x and y |
| `Lx`, `Ly` | Domain size; the domain is `[0, Lx] x [0, Ly]` |

Generated meshes have boundary zones named `left`, `right`, `bottom` and `top`.

`periodic = ["x", "y"]` (and `"z"` in 3D) makes a generated mesh (except
`wedge`) periodic in those directions: `left`/`right`, `bottom`/`top` and
`back`/`front` are joined into interior faces and disappear as boundary zones.
Every scheme sees the seam as interior. Each periodic direction needs at least
3 cells. Zones of mesh files are paired with `[[periodic]]` (below).

In the 3D build (`-DMallard_DIM=3`), generated meshes are boxes
`[0, Lx] x [0, Ly] x [0, Lz]` of `Nx x Ny x Nz` blocks (`Nz`, `Lz` default to
100 and 1), with the extra boundary zones `back` (z = 0) and `front` (z = Lz):

| `type` | Cells |
|---|---|
| `cartesian` | One hexahedron per block |
| `cartesian_tet` | Six tetrahedra per block (Kuhn subdivision along the block diagonal) |
| `cartesian_prism` | Two triangular prisms per block (split along the xy diagonal) |
| `cartesian_pyramid` | Six pyramids per block, with apexes at the block center |
| `cartesian_mixed` | Hexahedra, pyramids and prisms in successive thirds of x |

3D cell geometry is exact for warped (non-planar) quadrilateral faces: face
area vectors and centroids come from a triangle fan around the face's vertex
average, and cell volumes and centroids from the tetrahedra joining those
triangles to the cell's vertex average.

For Gmsh meshes, each physical curve (2D) or surface (3D) becomes a boundary
zone named after it (`physical_<tag>` if unnamed); boundary faces not in any
such group form the zone `unassigned`. Elements of other dimensions (points,
and curves in 3D) are ignored, and higher-order elements are rejected.

In a distributed run every rank reads a Gmsh file whole (keeping only its
share), so large meshes should be converted to Mallard's HDF5 mesh format
(builds with `-DMallard_ENABLE_HDF5=ON`), of which each rank reads only its
share; generated meshes are also produced per rank:

```sh
mallard-mesh-convert mesh.msh mesh.h5               # from Gmsh
mpirun -n 8 mallard-mesh-convert input.toml mesh.h5  # a generated mesh, written in parallel
```

The file holds global arrays, with global ids the row indices:
`/nodes/coordinates` (`n_nodes x dim`, float64), `/cells/offsets` and
`/cells/nodes` (CSR of node ids, uint64; the cell type follows from the node
count, Gmsh/VTK node order), `/boundary/offsets`, `/boundary/nodes` and
`/boundary/zone` (boundary faces and their zone index), the attribute
`/boundary/zone_names`, and the root attributes `format = "mallard-mesh"`,
`version = 1` and `dimension`.

## `[[periodic]]`

Pairs two boundary zones of any mesh (e.g. Gmsh physical curves or surfaces,
or zones of an HDF5 mesh) into a periodic seam, serially and in distributed
runs: zone B must be zone A translated, node by node (to 1e-6 of the zones'
shortest edge), as Gmsh's `Periodic` constraint produces. The faces of both
zones become interior faces and take no `[[boundaries]]` entry. A node with no
match is reported with its coordinates. Each zone joins at most one pair, and
as for generated meshes, each periodic direction needs at least 3 cells.

| Key | Description |
|---|---|
| `zones` | `[A, B]`: the zone names |
| `translation` | `[t_x, t_y]` (`[t_x, t_y, t_z]` in 3D) with B = A + translation |

```toml
[mesh]
type = "file"
filename = "box.msh"

[[periodic]]
zones = ["left", "right"]
translation = [1.0, 0.0]
```

## `[physics]`

| Key | Description |
|---|---|
| `type` | `euler` or `navier_stokes` |
| `gamma` | Ratio of specific heats |
| `p_ref`, `T_ref`, `rho_ref` | A reference state, which sets the gas constant `R = p_ref / (rho_ref T_ref)` |
| `mu` | (`navier_stokes`) Dynamic viscosity, or its value at `T_mu_ref` for Sutherland's law or the power law |
| `Pr` | (`navier_stokes`) Prandtl number, default 0.72 |
| `viscosity_model` | (`navier_stokes`) `constant` (default), `sutherland` or `power_law` (`mu (T / T_mu_ref)^viscosity_exponent`) |
| `T_mu_ref`, `sutherland_S` | (`sutherland`) Reference temperature (default 273.15) and Sutherland temperature (default 110.4) |
| `T_mu_ref`, `viscosity_exponent` | (`power_law`) Reference temperature and exponent, both required |
| `gas` | `perfect` (default): a calorically perfect gas set by the keys above; `mixture`: a thermally perfect mixture of the species of a mechanism (no `gamma`, `p_ref`, `T_ref`, `rho_ref`) |
| `mechanism` | (`mixture`) A [Cantera YAML](https://cantera.org/stable/yaml/index.html) file, e.g. `mechanisms/h2o2.yaml` (see `mechanisms/README.md`); Chemkin files convert with Cantera's `ck2yaml` |
| `phase` | (`mixture`) Phase of the file to use; default the first |
| `transport` | (`mixture`, `navier_stokes`) `mixture_averaged` (default: Wilke viscosity, Mathur conductivity, mixture-averaged diffusion coefficients), `unity_lewis` (diffusion coefficients `lambda / (rho cp)`) or `constant_lewis` (`lambda / (rho cp Le_k)`), as Cantera's models, from the species' transport data in the file |
| `lewis` | (`constant_lewis`) Lewis numbers by species, e.g. `lewis = { H2 = 0.3, H = 0.18 }`; others 1 |
| `axisymmetric` | (2D) `true` for flows symmetric about the x axis without swirl (default `false`); see below |

Gas mixtures (`gas = "mixture"`) react when the input has a `[chemistry]`
table; every Riemann solver and face reconstruction works (`RHLL` for strong
shocks prone to the carbuncle, e.g. blunt-body detonations). With `type = "navier_stokes"` they are viscous, heat
conducting and diffusing: species diffuse along mole-fraction gradients with a
correction velocity (so the diffusive mass fluxes sum to zero) and carry their
enthalpy, and the time step includes the largest of `4/3 mu / rho`,
`lambda / (rho cv)` and the species' diffusion coefficients. Walls are no-slip,
adiabatic and non-catalytic (`wall_isothermal` is not yet available for
mixtures). Species thermodynamics are NASA-7, NASA-9 or constant-cp
polynomials from the file, evaluated in double precision in every build. Each
species is transported (`rho Y_k`) with mass-flux upwinding, so mass fractions
stay in [0, 1] and sum to one; all mass fractions share one stencil and one
limiter per cell (MUSCL), or the stencils TENO chose for the contact field
(TENO, which reconstructs the flow in primitive variables for mixtures and
also flags cells by jumps of the molar mass). The scheme is conservative: at contacts between gases of different
`cp / cv` (e.g. cold hydrogen and hot air) the pressure is perturbed at the
percent level on coarse meshes; `double_flux` (in `[numerics]`) removes that.

### Axisymmetric flows

With `axisymmetric = true` the 2D mesh is the meridian half-plane of a body
of revolution: x is the axial coordinate and y the radius, so every node needs
`y >= 0`. Boundary faces on the axis (`y = 0`) must be `symmetry`; nothing
crosses them. The solver integrates over each cell's solid of revolution:
cell values are r-weighted averages (analytical initial conditions are
averaged that way), and fluxes, sources, forces (`[[forces]]`), `[integrals]`
and conserved totals are per radian of the revolved domain (multiply forces by
`2 pi` for the force on the 3D body; only the axial component is meaningful).
`u` is `[u_x, u_r]`. Every reconstruction, Riemann solver, viscous and
mixture model, chemistry and MPI work as in planar runs; meshes cannot be
periodic in y. Method and accuracy: `docs/design/axisymmetric.md`.

## `[initialize]`

| Key | Description |
|---|---|
| `type` | `constant`, `analytical` or `restart` |
| `u` | `constant`: `[u_x, u_y]`; `analytical`: one expression in `x`, `y`, `z` per component |
| `rho`, `p`, `T` | `constant`: `p` and `T`; `analytical`: exactly two of the three, as expressions in `x`, `y`, `z` |
| `X` or `Y` | (mixtures) Mole or mass fractions by species, e.g. `X = { H2 = 2.0, O2 = 1.0, AR = 7.0 }`; normalized, unlisted species are zero. `analytical`: expressions (or numbers) per listed species |
| `balance` | (mixtures, `analytical`) Species taking `1 - sum` of the listed fractions; without it the listed fractions are normalized |
| `n_subdivisions` | (`analytical`) Resolution of the cell averages. 2D: each cell's triangles are split into `n_subdivisions`² sub-triangles (default 4). 3D: each of the cell's tetrahedra is integrated with a 64-point rule on each of `n_subdivisions`³ pieces (default 2) |
| `file` | (`restart`) Restart file to resume from. Restart files list their variables by name (format version 2, or 3 when they also hold the weights of `[statistics]` averages) and are read by name; files of version 1 (Mallard 0.3 and earlier) are still read |

Expressions use [exprtk](https://www.partow.net/programming/exprtk/) syntax, e.g. `"x < 0.5 ? 1.0 : 0.125"`.

## `[[boundaries]]`

Every boundary face must be assigned exactly once (a fully periodic mesh has
none, and needs no `[[boundaries]]`). A zone can be split
between several entries with `where = "<expression in x, y, z>"`, which selects
the zone's faces whose centers satisfy the expression.

| `type` | Description | Keys |
|---|---|---|
| `extrapolation` | Transmissive: the exterior state is the solution translated from inside the domain | |
| `symmetry` | Slip wall / symmetry plane | |
| `wall_adiabatic` | Wall (no-slip for `navier_stokes`, slip for `euler`) with zero heat flux | `u` (wall velocity, optional) |
| `wall_isothermal` | Wall at temperature `T` | `T`, `u` (optional) |
| `wall_heat_flux` | Wall with heat flux `q` into the fluid | `q`, `u` (optional) |
| `upt` | Inflow with fixed velocity, pressure and temperature | `u`, `p`, `T` (and `X` or `Y` for mixtures: numbers, or expressions in `x`, `y`, `z` evaluated at each face center for a composition that varies along the boundary, with an optional `balance` species as in `[initialize]`) |
| `farfield` | Characteristic far field for a free stream: the outgoing Riemann invariant comes from the interior, the incoming one from the free stream, so waves leave and the boundary works for inflow, outflow and tangential flow alike | `u`, `p`, `T` (free stream) |
| `dirichlet` | Exterior state from expressions in `x`, `y`, `z`, `t`, evaluated at face centers at every stage | `rho`, `u` (one expression per component), `p` |
| `p_out` | Outlet: imposes `p` if the outflow is subsonic | `p` |
| `p_out_average` | Outlet for mixed subsonic/supersonic flow: on subsonic faces, shifts the local pressure so that its area average over the boundary equals `p`, preserving the transverse profile | `p` |

## `[numerics]`

| Key | Description |
|---|---|
| `riemann_solver` | `Rusanov`, `HLL`, `HLLC` (default), `Roe`, or `RHLL` (rotated hybrid HLL-Roe, carbuncle-free) |
| `time_integrator` | `FE`, `SSPRK3` (default) or `RK4` |
| `check_nan` | Stop if the solution becomes non-finite |
| `double_flux` | (gas mixtures) `true` for the double-flux scheme: each cell's energy is updated with its own `cp / cv` and energy offset frozen over the time step on both sides of its faces, and reset to the true equation of state after the step, so pressure and velocity stay exactly uniform across contacts between different gases. Energy is then not exactly conserved (about 0.2% over a multicomponent shock tube). Default `false` |
| `low_mach_cutoff` | Low-Mach correction of the convective flux: the velocity jump across each interior face is scaled by `z = min(1, max(M_L, M_R, low_mach_cutoff))` before the Riemann solver, so that upwind dissipation scales with the flow speed rather than the sound speed. Default 0.1; 1 disables it. See [`numerics/overview.md`](numerics/overview.md) |

### `[numerics.face_reconstruction]`

| Key | Description |
|---|---|
| `type` | `FO` (first order), `MUSCL` or `TENO` |
| `limiter` | (`MUSCL`) `venkatakrishnan` (default), `barth_jespersen` or `none` |
| `venkatakrishnan_K` | (`MUSCL`) Venkatakrishnan threshold constant, default 5 |
| `order` | (`TENO`) Order of accuracy, 3 to 6, default 5. In 3D, faces use Dunavant (triangles) or Gauss (quadrilaterals) rules exact to this order, capped at degree 5 on triangles |
| `stencil_factor` | (`TENO`) Minimum large-stencil size as a multiple of the number of polynomial coefficients, default 2; a stencil grows past it until its reconstruction's Lebesgue constant is at most 10. Smaller values (e.g. 1.5) are markedly less dissipative for fine smooth structures (Shu-Osher entropy waves: 50% more amplitude at 200 cells) but less robust at discontinuities. |
| `small_stencil_size` | (`TENO`) Cells per sector stencil, default 10 (18 in 3D) |
| `troubled_threshold` | (`TENO`) Troubled-cell threshold on the density-jump variance, default 1e-3 |
| `troubled_upper` | (`TENO`) Variance at which the adaptive cutoff reaches its largest value (most dissipative), default 1e-2 |
| `C_T` | (`TENO`) Fixed TENO cutoff; adaptive (1e-10 to 1e-6) if omitted |
| `characteristic` | (`TENO`) Select stencils on characteristic variables, default true |
| `max_condition` | (`TENO`) Stencils grow until the least-squares system's condition estimate is below this, default 1e8 |
| `cache_file` | (`TENO`) Save the precomputed stencils and matrices here, and reuse them on later runs of the same mesh, boundary assignment and TENO options; anything else is detected and recomputed. Distributed runs write one file per rank, `<cache_file>.r<rank>-of-<ranks>`, for that rank count and partition, and record the halo depth the stencils need, so a cached run sets up its halo once. Hilbert and graph partitions repeat for the same mesh and rank count. Size per cell: about 2.5 / 4 / 6 / 11 KB in 2D and 10 / 19 / 36 KB (hexahedra) or 9 / 14 / 33 / 74 KB (tetrahedra) in 3D for orders 3 / 4 / 5 / 6, e.g. 9.4 GB for 64^3 hexahedra at order 5; reading it takes seconds, against minutes of setup in 3D |
| `bound_preserving` | (`TENO`) Scale troubled-cell polynomials to keep density and pressure within the neighbors' range, default false |

## `[chemistry]`

Finite-rate chemistry of gas mixtures (`gas = "mixture"` with a mechanism
that has reactions). Each step is Strang split: every owned cell that needs
it is advanced as an adiabatic, constant-volume reactor over `dt / 2`, then
the flow takes its step, then the reactors take another `dt / 2`; the second
half step is fused with the next step's first (one chemistry call per step)
except before output, progress rows and the end of the run. The integrator
is RODAS, an adaptive Rosenbrock method with the analytical Jacobian, in
double precision in every build.

| Key | Description |
|---|---|
| `enabled` | `false` keeps the mixture non-reacting (default `true`) |
| `integrator` | `rosenbrock` (the default and only choice so far) |
| `coupling` | `strang` (the default and only choice so far) |
| `rtol` | Relative tolerance on the mass fractions and `T` (default `1e-6`) |
| `atol` | Absolute tolerance on the mass fractions (default `1e-10`) |
| `max_steps` | Sub-steps allowed per cell and half step (default 100000); more stop the run |
| `T_frozen` | No chemistry in cells below this temperature (default 0) |
| `fuse_half_steps` | `true` fuses the closing half step of a step with the next step's opening one, except where output, checks, probes, statistics or the end of the run read the state (default `false`). Faster where cells take few sub-steps, but results then depend on when output is written, and a restart reproduces an uninterrupted run only from a step at which that run also wrote output |
| `sparse` | `true` for the sparse LU (static pattern, with the Jacobian's dense rank-one part by Sherman-Morrison), `false` for the dense one; by default sparse from 30 species when its factors fill at most 60% of the dense matrix (GRI-3.0 and larger) |
| `lanes` | Vector lanes integrating one cell: 1 for one thread per cell (cells ordered by their last cost), a power of 2 up to 32 for a team per cell on GPUs; default 0, automatic: a warp per cell on GPUs from 16 species (8 warps, 16 from 512 species, for cells whose last call took 16 or more sub-steps), else one thread |

Cells whose mass fractions would change by less than `atol / 100` over the
half step at their current rates are skipped. Reaction types: elementary,
three-body, falloff (Lindemann, Troe, SRI), `pressure-dependent-Arrhenius`
(PLOG) and `Chebyshev`, with non-integer reactant `orders` (global
mechanisms such as Westbrook–Dryer's). Orders between 0 and 1 follow the
power law down to a concentration of `1e-12` kmol/m^3 and a quadratic with a
bounded slope below it (see `docs/design/chemistry.md`), so the Jacobian
stays finite where such a reactant runs out. Output variables: `HRR` (heat release rate,
W/m^3) and `CHEM_COST` (chemistry sub-steps of the cell in the last step);
restart files also hold `CHEM_H`, each cell's last sub-step, so restarted
runs repeat the uninterrupted one exactly. The progress rows add the share of
cells advanced in the last half step and the most sub-steps of a cell in the
last step, and the summary the chemistry's share of the wall time.

## `[[forces]]`

Write the force of the fluid on a boundary zone to a CSV file
(`step, t, Fx_pressure, Fy_pressure, Fx_viscous, Fy_viscous`, per unit depth, or per radian for
axisymmetric flows; in
3D `step, t, Fx_pressure, Fy_pressure, Fz_pressure, Fx_viscous, Fy_viscous, Fz_viscous`).

| Key | Description |
|---|---|
| `zone` | Boundary zone name |
| `interval` | Every this many steps, default 1 |
| `file` | Output file, default `forces_<zone>.csv` |

## `[integrals]`

Write domain integrals to a CSV file (`step, t, kinetic_energy, enstrophy,
dilatation_squared, pressure_dilatation, velocity_squared, vorticity_squared,
density_squared, temperature, temperature_squared`): the integrals of
`rho |u|^2 / 2`, `rho |omega|^2 / 2`, `(div u)^2`, `p div u`, `|u|^2`,
`|omega|^2`, `rho^2`, `T` and `T^2` (divided by the volume, the last ones give
the mean squared velocity and vorticity and the density and temperature
variances). With TENO the velocity
gradients are those of the reconstruction polynomials at the cell centroids
(order-consistent: on the Taylor-Green vortex at 64^3 per octant they match
spectral derivatives of the same field to about 1%); otherwise they are the
second-order least-squares gradients of the viscous fluxes, which
underestimate the enstrophy of under-resolved turbulence (by about 15% in
that case). For decaying
turbulence such as the Taylor-Green vortex, the kinetic energy dissipation
rate is `-dE/dt` and its viscous part `2 mu * enstrophy / rho0`.

| Key | Description |
|---|---|
| `interval` | Every this many steps, default 1 |
| `file` | Output file, default `integrals.csv` |

## `[statistics]`

Running time averages, kept per cell on the device. Every `interval` steps
once `t > t_start`, the sample at time `t` enters with weight `t - t_prev`
(the time since the previous sample, or since `t_start` for the first), so the
averages approximate time integrals over `(t_start, t]` also when the time
step varies. Each cell's averages are its own (no communication), so they are
bitwise the same on any number of ranks, and sampling reads the state without
changing it.

| Key | Description |
|---|---|
| `fields` | Variables to average: `RHO`, `U_X`, `U_Y`, (3D) `U_Z`, `P`, `T`, `H` and, for mixtures, `Y_<species>` |
| `products` | Pairs `"A*B"` of those variables (e.g. `"U_X*U_X"`, `"U_X*U_Y"`, `"T*T"`) whose covariance to keep; their means are kept too |
| `t_start` | Start time, default 0 |
| `interval` | Every this many steps, default 1 |
| `reset` | `true` starts the averages afresh on a restart instead of continuing those of the restart file (default `false`) |

The averages are output variables for `[[write_data]]`: `MEAN_<A>` and
`COV_<A>_<B>`, the covariance `<A'B'> = <AB> - <A><B>` (kept directly, by a
weighted Welford update, rather than as `<AB>`, which loses the fluctuations
to round-off when they are small), e.g. `MEAN_U` (a vector), `COV_U_X_U_Y`, or
`MEAN_*`, `COV_*`. Restart files carry them and their weights (format version
3), so a restarted run continues the averages bitwise, also on a different
number of ranks; a restart file without them starts them afresh, and one with
only some of them is an error unless `reset = true`. With
`chemistry.fuse_half_steps`, sampling steps read the state like output does.
`tools/plane_average.py` averages them further over homogeneous directions
(e.g. over x and z of a channel, as a function of y).

## `[[probes]]`

Point and line probes: the values of the cells holding a set of points, every
`interval` steps, appended by rank 0 to a CSV file with rows
`step, t, point, x, y(, z), <variables>` (one per point and step; a restarted run
appends). Each point reads the cell average of the cell containing it (on a
face between cells, the one with the lowest global id, so any number of ranks
picks the same cell), or of the nearest cell centroid if no cell contains it.
Values are written to full precision.

| Key | Description |
|---|---|
| `name` | Probe name |
| `variables` | Any of the `[statistics]` fields |
| `point` | One point `[x, y(, z)]`, or |
| `start`, `end`, `n_points` | `n_points` (at least 2) evenly spaced points from `start` to `end` |
| `interval` | Every this many steps, default 1 |
| `file` | Output file, default `probe_<name>.csv` |

## `[source]`

Optional source terms, added per unit volume.

| Key | Description |
|---|---|
| `gravity` | `[g_x, g_y]` (`[g_x, g_y, g_z]` in 3D); adds `rho g` to the momentum and `rho u . g` to the energy equation |
| `rho`, `rhou`, `rhoE` | Expressions in `x`, `y`, `z`, `t` (`rhou` has one per component) for the mass, momentum and energy sources |
| `time_dependent` | Re-evaluate the expressions at every Runge-Kutta stage (host-side, so costly on large meshes); otherwise they are evaluated once |

The scheme is not exactly well balanced: hydrostatic states carry small spurious velocities (about 1e-4 of the sound speed on a 32x32 mesh) that vanish at second order under refinement. Wall and symmetry ghost states continue the hydrostatic pressure gradient.

## `[parallel]`

Used when Mallard runs on several MPI ranks (`mpirun -n N Mallard -i input.toml`).

| Key | Description |
|---|---|
| `partitioner` | `graph` (dKaMinPar on the cell connectivity, minimizing the faces between ranks; default when built with `Mallard_ENABLE_KAMINPAR`) or `hilbert` (cells split along a Hilbert curve of their centroids; the default otherwise) |

## `[output]`

| Key | Description |
|---|---|
| `check_interval` | Print a progress row every this many steps (default 1) |

Each progress row shows the step, time `t`, time step `dt`, the fraction of the run done (by
whichever of `n_steps`, `t_stop` and `t_wall_stop` comes first), the time-stepping wall time
per step, the throughput in cell updates per second, the estimated time remaining, the minimum
density and pressure, the maximum Mach number and, with TENO, the percentage of troubled cells.
Files written appear as rows led by their step and time. The run ends with a summary of wall
time (setup, time stepping, diagnostics, output) and average throughput.

Only rank 0 prints, except for errors, which go to stderr from any rank and carry the rank in
parallel runs. Output on a terminal is colored unless `NO_COLOR` is set (`CLICOLOR_FORCE=1`
forces color, e.g. under `mpirun`); logs written to files are plain ASCII.

## `[[write_data]]`

| Key | Description |
|---|---|
| `prefix` | Output path prefix; directories are created as needed |
| `format` | `vtu` (with a `.pvd` series next to it), `hdf5` (with XDMF indexes; see below) or `restart` |
| `interval` / `time_interval` | Write every this many steps / this much simulation time (exactly one). With `time_interval` the time step is shortened to land on each output time. |
| `variables` | (`vtu`, `hdf5`) Any of `RHO`, `RHOU_X`, `RHOU_Y`, (3D) `RHOU_Z`, `RHOE`, `U_X`, `U_Y`, (3D) `U_Z`, `P`, `T`, `H`, `CFL`, the vectors `RHOU` and `U` (written with 3 components, zero z in 2D), with TENO `TENO_SIGMA` (the troubled-cell indicator; stencil selection is active where it exceeds `troubled_threshold`), and for mixtures `Y_<species>`, `X_<species>` and `RHOY_<species>` (with `navier_stokes` also `MU`, `LAMBDA` and the diffusion coefficients `D_<species>`; with `[chemistry]` also `HRR`, `CHEM_COST` and the mass production rates `OMEGA_<species>` [kg/(m^3 s)]), and `P_MAX`, the largest pressure of each cell at the end of any step so far (a numerical soot foil of detonation cells; restart files then carry it, so it continues across restarts), and the `[statistics]` averages `MEAN_<A>` and `COV_<A>_<B>`. A trailing `*` selects every variable with that prefix, e.g. `Y_*` |
| `geometry` | (`vtu`) `all` (default) for the volume, or a boundary zone name to write that zone's faces with the values of their adjacent cells (e.g. wall pressure) |

`format = "hdf5"` (builds with `-DMallard_ENABLE_HDF5=ON`; several ranks need
parallel HDF5) scales to large runs: every rank writes its part of one file per
snapshot with collective I/O instead of a VTU piece of its own. It writes

- `PREFIX_mesh.h5`, once: `nodes/coordinates` (n_nodes x dimension),
  `cells/topology`, an XDMF `Mixed` topology (each cell's XDMF type, then its
  nodes in VTK order: triangle 4, quadrilateral 5, tetrahedron 6, pyramid 7,
  wedge 8, hexahedron 9) and `cells/offsets`, where each cell's record starts
  (n_cells + 1 entries);
- `PREFIX_NNNNNN.h5` per snapshot: the attributes `step`, `time` and `mesh`
  (the mesh file's name), and `fields/<name>` for each variable (n_cells
  values, or n_cells x 3 for vectors);
- `PREFIX_NNNNNN.xmf` per snapshot and `PREFIX.xmf`, the time series, which
  ParaView opens (XDMF reader).

Row i of every cell array is global cell i, the cell order of the mesh file and
of restart files, so the files do not depend on the number of ranks.
Snapshots are about half the size of VTU pieces (the mesh is not repeated) and
two files instead of one per rank. A shared file pays off on parallel file
systems; on one node's local disk, which serializes writes to a file, VTU
pieces write faster beyond a few ranks (2.1M hexahedra, 16 ranks: 115 ms per
HDF5 snapshot against 72 ms, 1 rank: 140 ms against 1.1 s). VTU output remains
the simpler choice for small runs. `tools/mallard_h5.py` reads the files.

## `MallardReactor`

`MallardReactor -i input.toml` integrates one adiabatic, constant-volume
reactor with the solver's chemistry kernels (the RODAS Rosenbrock integrator
with the analytical Jacobian, in double precision) and writes its history as
CSV (`t`, `T`, `p`, `Y_<species>`). It reads `[physics]` (`mechanism`,
`phase`), the optional `[chemistry]` table and:

| Key | Description |
|---|---|
| `[reactor] type` | `constant_volume` (default; the only type so far) |
| `[reactor] T`, `p` | Initial temperature and pressure |
| `[reactor] X` or `Y` | Initial composition, as in `[initialize]` |
| `[reactor] end_time` | Integration time |
| `[reactor] output_interval` | Time between CSV rows (default `end_time / 100`); each interval ends like a splitting step: negative mass fractions clipped, mass fractions renormalized, `T` from the conserved energy |
| `[reactor] output` | CSV file (default `reactor.csv`) |
| `[chemistry] rtol` | Relative tolerance on `Y` and `T` (default `1e-6`) |
| `[chemistry] atol` | Absolute tolerance on `Y` (default `1e-10`) |
| `[chemistry] max_steps` | Sub-steps allowed per output interval (default 100000) |
| `[chemistry] sparse` | The linear solver, as in the solver's `[chemistry]` |
| `[benchmark]` | Instead of the run, a chemistry benchmark: states sampled along this reactor's trajectory, replicated over `cells` cells and advanced by the solver's chemistry kernels over each splitting step of `dt`; see `benchmarks/README.md` |

It prints the ignition delay (time of the maximum of `dT/dt`) when `T` rose
by more than 400 K. Example: `examples/h2_ignition`.
