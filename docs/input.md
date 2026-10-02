# Input file reference

Mallard reads a single [TOML](https://toml.io) file: `Mallard -i input.toml`.
Kokkos options such as `--kokkos-num-threads=N` or `--kokkos-device-id=N` can be appended.

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
| `filename` | (`file`) ASCII Gmsh mesh, format 2.2 or 4.1, of triangles and/or quadrilaterals |
| `Nx`, `Ny` | Number of quads in x and y |
| `Lx`, `Ly` | Domain size; the domain is `[0, Lx] x [0, Ly]` |

Generated meshes have boundary zones named `left`, `right`, `bottom` and `top`.
For Gmsh meshes, each named physical curve becomes a boundary zone; boundary
edges not in any physical curve form the zone `unassigned`.

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

## `[initialize]`

| Key | Description |
|---|---|
| `type` | `constant`, `analytical` or `restart` |
| `u` | `constant`: `[u_x, u_y]`; `analytical`: two expressions in `x` and `y` |
| `rho`, `p`, `T` | `constant`: `p` and `T`; `analytical`: exactly two of the three, as expressions in `x` and `y` |
| `n_subdivisions` | (`analytical`) Each cell is split into `n_subdivisions`² sub-triangles for computing cell averages, default 4 |
| `file` | (`restart`) Restart file to resume from |

Expressions use [exprtk](https://www.partow.net/programming/exprtk/) syntax, e.g. `"x < 0.5 ? 1.0 : 0.125"`.

## `[[boundaries]]`

Every boundary face must be assigned exactly once. A zone can be split
between several entries with `where = "<expression in x, y>"`, which selects
the zone's faces whose centers satisfy the expression.

| `type` | Description | Keys |
|---|---|---|
| `extrapolation` | Transmissive: the exterior state is the solution translated from inside the domain | |
| `symmetry` | Slip wall / symmetry plane | |
| `wall_adiabatic` | Wall (no-slip for `navier_stokes`, slip for `euler`) with zero heat flux | `u` (wall velocity, optional) |
| `wall_isothermal` | Wall at temperature `T` | `T`, `u` (optional) |
| `wall_heat_flux` | Wall with heat flux `q` into the fluid | `q`, `u` (optional) |
| `upt` | Inflow with fixed velocity, pressure and temperature | `u`, `p`, `T` |
| `dirichlet` | Exterior state from expressions in `x`, `y`, `t`, evaluated at face centers at every stage | `rho`, `u` (two expressions), `p` |
| `p_out` | Outlet: imposes `p` if the outflow is subsonic | `p` |
| `p_out_average` | Outlet for mixed subsonic/supersonic flow: on subsonic faces, shifts the local pressure so that its area average over the boundary equals `p`, preserving the transverse profile | `p` |

## `[numerics]`

| Key | Description |
|---|---|
| `riemann_solver` | `Rusanov`, `HLL`, `HLLC` (default), `Roe`, or `RHLL` (rotated hybrid HLL-Roe, carbuncle-free) |
| `time_integrator` | `FE`, `SSPRK3` (default) or `RK4` |
| `check_nan` | Stop if the solution becomes non-finite |

### `[numerics.face_reconstruction]`

| Key | Description |
|---|---|
| `type` | `FO` (first order), `MUSCL` or `TENO` |
| `limiter` | (`MUSCL`) `venkatakrishnan` (default), `barth_jespersen` or `none` |
| `venkatakrishnan_K` | (`MUSCL`) Venkatakrishnan threshold constant, default 5 |
| `order` | (`TENO`) Order of accuracy, 2 to 6, default 5 |
| `stencil_factor` | (`TENO`) Large-stencil size as a multiple of the number of polynomial coefficients, default 2 |
| `small_stencil_size` | (`TENO`) Cells per sector stencil, default 10 |
| `troubled_threshold` | (`TENO`) Troubled-cell threshold on the density-jump variance, default 1e-3 |
| `C_T` | (`TENO`) Fixed TENO cutoff; adaptive (1e-10 to 1e-6) if omitted |
| `characteristic` | (`TENO`) Select stencils on characteristic variables, default true |
| `max_condition` | (`TENO`) Stencils grow until the least-squares system's condition estimate is below this, default 1e8 |
| `bound_preserving` | (`TENO`) Scale troubled-cell polynomials to keep density and pressure within the neighbors' range, default false |

## `[source]`

Optional source terms, added per unit volume.

| Key | Description |
|---|---|
| `gravity` | `[g_x, g_y]`; adds `rho g` to the momentum and `rho u . g` to the energy equation |
| `rho`, `rhou`, `rhoE` | Expressions in `x`, `y`, `t` (`rhou` is a two-element array) for the mass, momentum and energy sources |
| `time_dependent` | Re-evaluate the expressions at every Runge-Kutta stage (host-side, so costly on large meshes); otherwise they are evaluated once |

The scheme is not well balanced: hydrostatic states carry small spurious velocities that vanish under refinement.

## `[output]`

| Key | Description |
|---|---|
| `check_interval` | Print solution ranges and timing every this many steps |

## `[[write_data]]`

| Key | Description |
|---|---|
| `prefix` | Output path prefix; directories are created as needed |
| `format` | `vtu` (with a `.pvd` series next to it) or `restart` |
| `interval` / `time_interval` | Write every this many steps / this much simulation time (exactly one). With `time_interval` the time step is shortened to land on each output time. |
| `variables` | (`vtu`) Any of `RHO`, `RHOU_X`, `RHOU_Y`, `RHOE`, `U_X`, `U_Y`, `P`, `T`, `H`, `CFL` |
| `geometry` | (`vtu`) `all` (default) for the volume, or a boundary zone name to write that zone's faces with the values of their adjacent cells (e.g. wall pressure) |
