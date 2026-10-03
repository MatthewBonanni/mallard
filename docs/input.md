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
| `mu` | (`navier_stokes`) Dynamic viscosity, or its value at `T_mu_ref` for Sutherland's law |
| `Pr` | (`navier_stokes`) Prandtl number, default 0.72 |
| `viscosity_model` | (`navier_stokes`) `constant` (default) or `sutherland` |
| `T_mu_ref`, `sutherland_S` | (`sutherland`) Reference temperature (default 273.15) and Sutherland temperature (default 110.4) |
| `gas` | `perfect` (default): a calorically perfect gas set by the keys above; `mixture`: a thermally perfect mixture of the species of a mechanism (no `gamma`, `p_ref`, `T_ref`, `rho_ref`) |
| `mechanism` | (`mixture`) A [Cantera YAML](https://cantera.org/stable/yaml/index.html) file, e.g. `mechanisms/h2o2.yaml` (see `mechanisms/README.md`); Chemkin files convert with Cantera's `ck2yaml` |
| `phase` | (`mixture`) Phase of the file to use; default the first |

Gas mixtures (`gas = "mixture"`) are non-reacting for now and need
`type = "euler"` and the `Rusanov`, `HLL` or `HLLC` Riemann solver; any face
reconstruction works. Species thermodynamics are NASA-7, NASA-9 or constant-cp
polynomials from the file, evaluated in double precision in every build. Each
species is transported (`rho Y_k`) with mass-flux upwinding, so mass fractions
stay in [0, 1] and sum to one; all mass fractions share one stencil and one
limiter per cell (MUSCL), or the stencils TENO chose for the contact field
(TENO, which reconstructs the flow in primitive variables for mixtures and
also flags cells by jumps of the molar mass). The scheme is conservative: at contacts between gases of different
`cp / cv` (e.g. cold hydrogen and hot air) the pressure is perturbed at the
percent level on coarse meshes; `double_flux` (in `[numerics]`) removes that.

## `[initialize]`

| Key | Description |
|---|---|
| `type` | `constant`, `analytical` or `restart` |
| `u` | `constant`: `[u_x, u_y]`; `analytical`: one expression in `x`, `y`, `z` per component |
| `rho`, `p`, `T` | `constant`: `p` and `T`; `analytical`: exactly two of the three, as expressions in `x`, `y`, `z` |
| `X` or `Y` | (mixtures) Mole or mass fractions by species, e.g. `X = { H2 = 2.0, O2 = 1.0, AR = 7.0 }`; normalized, unlisted species are zero. `analytical`: expressions (or numbers) per listed species |
| `balance` | (mixtures, `analytical`) Species taking `1 - sum` of the listed fractions; without it the listed fractions are normalized |
| `n_subdivisions` | (`analytical`) Resolution of the cell averages. 2D: each cell's triangles are split into `n_subdivisions`² sub-triangles (default 4). 3D: each of the cell's tetrahedra is integrated with a 64-point rule on each of `n_subdivisions`³ pieces (default 2) |
| `file` | (`restart`) Restart file to resume from. Restart files list their variables by name (format version 2) and are read by name; files of version 1 (Mallard 0.3 and earlier) are still read |

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
| `upt` | Inflow with fixed velocity, pressure and temperature | `u`, `p`, `T` (and `X` or `Y` for mixtures) |
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

## `[[forces]]`

Write the force of the fluid on a boundary zone to a CSV file
(`step, t, Fx_pressure, Fy_pressure, Fx_viscous, Fy_viscous`, per unit depth; in
3D `step, t, Fx_pressure, Fy_pressure, Fz_pressure, Fx_viscous, Fy_viscous, Fz_viscous`).

| Key | Description |
|---|---|
| `zone` | Boundary zone name |
| `interval` | Every this many steps, default 1 |
| `file` | Output file, default `forces_<zone>.csv` |

## `[integrals]`

Write domain integrals to a CSV file (`step, t, kinetic_energy, enstrophy,
dilatation_squared, pressure_dilatation`): the integrals of `rho |u|^2 / 2`,
`rho |omega|^2 / 2`, `(div u)^2` and `p div u`. With TENO the velocity
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
| `format` | `vtu` (with a `.pvd` series next to it) or `restart` |
| `interval` / `time_interval` | Write every this many steps / this much simulation time (exactly one). With `time_interval` the time step is shortened to land on each output time. |
| `variables` | (`vtu`) Any of `RHO`, `RHOU_X`, `RHOU_Y`, (3D) `RHOU_Z`, `RHOE`, `U_X`, `U_Y`, (3D) `U_Z`, `P`, `T`, `H`, `CFL`, the vectors `RHOU` and `U` (written with 3 components, zero z in 2D), with TENO `TENO_SIGMA` (the troubled-cell indicator; stencil selection is active where it exceeds `troubled_threshold`), and for mixtures `Y_<species>`, `X_<species>` and `RHOY_<species>`. A trailing `*` selects every variable with that prefix, e.g. `Y_*` |
| `geometry` | (`vtu`) `all` (default) for the volume, or a boundary zone name to write that zone's faces with the values of their adjacent cells (e.g. wall pressure) |
