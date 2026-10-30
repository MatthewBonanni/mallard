# Input file reference

Mallard reads a single [TOML](https://toml.io) file: `mallard -i input.toml`.
This page lists every key; the [user guide](guide/input_files.md) explains
by topic when to use them, with examples and figures.
Kokkos options such as `--kokkos-num-threads=N` or `--kokkos-device-id=N` can be appended.
A renamed or removed key is still read, with a warning, for at least one minor
release, and `mallard input upgrade old.toml > new.toml` rewrites old inputs
([deprecation policy and changed keys](guide/input_files.md#inputs-from-older-releases)).

The spatial dimension is fixed at build time with the CMake option
`-DMallard_DIM=2` (default) or `3`, which builds `mallard2d` or `mallard3d`;
`mallard` runs the one of the case's mesh ([Meshes](guide/meshes.md)). Vectors in the input (`u`, `gravity`,
`rhou`) have that many components, and expressions are in `x`, `y`, `z`
(`z` is 0 in 2D) and, where noted, `t`, in one language for every section:
[Expressions](guide/input_files.md#expressions). Below, `[u_x, u_y]` reads
`[u_x, u_y, u_z]` in 3D.

Every key in the file must be one the run reads: a key that is misspelled,
in the wrong table, or unused with the chosen settings (for example
`[numerics.face_reconstruction] order` with `type = "MUSCL"`) is an error,
listed with its line and the closest key Mallard looked for.

## Units

Numbers are in SI units. Every dimensional key also takes a quantity with a
unit, as a string converted to SI on read: `p = "1 atm"`, `T = "25 degC"`,
`Lx = "5 mm"`, `u = ["10 m/s", 0.0]`, `strength = "20 1/s"`. A unit of the
wrong dimension is an error naming the key and its dimension. Expressions are
in SI units (a key that takes an expression also takes a single quantity).
Syntax and the unit table: [Units](guide/input_files.md#units).

| Dimension | Keys |
|---|---|
| Time (s) | `run.dt`, `t_stop`, `t_wall_stop`; `write_data.time_interval`; `statistics.t_start`, `statistics.spectra.t_start`, `statistics.phase.t_start`, `period`, `t_reference`; `probes.time_interval`; `reactor.end_time`, `output_interval`; `benchmark.dt` |
| Length (m) | `mesh.Lx`, `Ly`, `Lz`; `mesh.transform.scale` (the length of one mesh-file unit), `translate`; `mesh.curved.center`, `radius`; `mesh.r_inner`, `r_outer` (annulus); `periodic.translation`, `center`; `boundaries.L`; `wall_model.exchange_distance`, `edge_distance`; `turbulence.length_scale`; `les.combustion.delta_L`; `probes.point`, `points`, `start`, `end`, `vertices`, `center`, `radius`, `pivot`, `translate`; `statistics.phase.probe`; region lengths (`center`, `radius`, `size`, `vertices`, `translate`, `pivot`, `edge`, ...) |
| Angle (plain numbers in degrees) | `mesh.transform.rotate` (2D), `rotate.angle` (3D); `probes.rotate` (2D), `rotate.angle` (3D), `angles`; region `rotate` (2D), `rotate.angle` (3D); `mesh.angle` (annulus); `periodic.angle` |
| Temperature (K) | `T` of `initialize`, `boundaries`, `sponges`, `reactor`; `boundaries.T0`; `physics.T_ref`, `T_mu_ref`; `les.combustion.T_unburnt`, `T_burnt`; `chemistry.T_frozen`; `radiation.T_ambient` |
| Temperature difference (K; no `degC`, `degF`) | `physics.sutherland_S` |
| Pressure (Pa) | `p` of `initialize`, `boundaries`, `sponges`, `reactor`, `probes.reference`; `boundaries.p0`; `physics.p_ref` |
| Density (kg/m^3) | `physics.rho_ref`; `rho` of `initialize` (`analytical`), `dirichlet` boundaries and `probes.reference` |
| Velocity (m/s) | `u` of `initialize`, `boundaries`, `sponges`, `probes.reference`; `turbulence.convection_velocity`; `les.combustion.s_L` |
| Velocity squared (m^2/s^2) | `turbulence.reynolds_stress` |
| Acceleration (m/s^2) | `source.gravity` |
| Rate (1/s) | `sponges.strength` |
| Dynamic viscosity (Pa s) | `physics.mu` |
| Heat flux (W/m^2) | `boundaries.q` |
| Mass flux (kg/(m^2 s)) | `source.mass_flow`; `boundaries.mass_flux` |
| Mass flow (kg/s; per metre of depth in 2D, per radian in axisymmetric runs) | `boundaries.mdot` |
| Sources (per unit volume) | `source.rho` (kg/(m^3 s)), `rhou` (N/m^3), `rhoE` (W/m^3), and the same of `[[sources]]` with `rhoY` (kg/(m^3 s)); with `total = true` integrated rates (kg/s, N, W) |
| Concentration (plain numbers in kmol/m^3) | `chemistry.C_reg` |

Every other real-valued key (`cfl`, `gamma`, `Pr`, mole fractions, model
constants, tolerances, ...) is dimensionless. `statistics.phase.level` has the dimension of its
`variable` (`P`: pressure, `T`: temperature, `U_X`: velocity, ...).

## Checking an input

`mallard check input.toml` runs the whole setup of a run (mesh, partition,
boundaries, numerics, initial solution) without time stepping and without
writing any file, and reports on the run. It exits with status 1 if there is
an error. Under `mpirun -n N` it checks the run on N ranks.

Its options: [`mallard check`](guide/cli.md#mallard-check). The report, with an example: [Checking an input](guide/input_files.md#checking-an-input).

## `[run]`

| Key | Description |
|---|---|
| `cfl` | CFL number in the usual unstructured convention ([Blazek](references.md#blazek-2015) eqs. 6.20-6.21; see [time step](numerics/overview.md)): on a uniform 2D grid of spacing h, `dt = cfl * h / (abs(u) + abs(v) + 2a)`. Measured stability limits for smooth flow are 1.26-1.6 with SSPRK3 and 1.39-1.8 with RK4, depending on reconstruction and cell type ([table](numerics/overview.md#time-step)): use at most 1.2 (SSPRK3) or 1.35 (RK4), and 1.0 with SSPRK3 as a robust default, shocks included (but 0.25 for the Noh problem's infinite-strength shock with `bound_preserving`); with `[chemistry]` the splitting error may call for less (detonations: about 0.35 with Strang splitting, 1.0 with `coupling = "simpler"`). Mallard 0.5 and earlier inputs give the same time step with half their `cfl`. Exactly one of `cfl` and `dt` is required, except that with `[numerics] preset` and neither of them the preset sets `cfl`. |
| `dt` | Fixed time step |
| `t_stop` | Stop at this simulation time (the last step is shortened to land on it) |
| `n_steps` | Stop after this many steps |
| `t_wall_stop` | Stop after this many seconds of wall time |
| `keep_last_good` | Keep a copy of the state at the start of each step (one copy of the conservatives, and partial densities, per step), so that a diverging run writes the last valid state; see [run control](guide/running.md) (default `true`) |
| `control_socket` | Serve `mallard ctl` on `mallard.sock` in the working directory (default `true`) |
| `control_file_interval` | Every this many steps, look for the files `STOP` and `CHECKPOINT` in the working directory (default 0: never) |
| `cuda_graphs` | CUDA builds: run each step (time-step reduction, halo exchanges, stages) as one CUDA graph, so the host launches one graph and waits once per step instead of launching each kernel (default `true`). Results are the same bit for bit. Steps run kernel by kernel where the host takes part in the step: gas mixtures, average-pressure outlets, characteristic boundaries, boundary expressions of `t`, sources of domain averages, and halos exchanged with MPI (`[parallel] halo_exchange`); the run log says which |

At least one stop condition is required.

## `[mesh]`

| Key | Description |
|---|---|
| `type` | `file` (the default), `cartesian` (quads), `cartesian_tri` (each quad split into two triangles along its bottom-left to top-right diagonal), `wedge` (quads over an 8 degree compression ramp starting at x = 0.5), or `annulus` (below) |
| `filename` | (`file`) Mallard HDF5 mesh (`.h5` or `.hdf5`, see below), ASCII Gmsh mesh (format 2.2 or 4.1), CGNS file (`.cgns`), ANSYS Fluent mesh or case file (`.msh`, `.cas`; ASCII or binary), or OpenFOAM polyMesh (a directory: the case, its `constant/polyMesh`, or anything between), of triangles and/or quadrilaterals (2D), or tetrahedra, pyramids, prisms and/or hexahedra (3D); see mesh files below |
| `Nx`, `Ny` | Number of quads in x and y (default 100) |
| `Lx`, `Ly` | Domain size; the domain is `[0, Lx] x [0, Ly]` (default 1) |

Generated meshes have boundary zones named `left`, `right`, `bottom` and `top`.

`stretching = [beta_x, beta_y]` (`[beta_x, beta_y, beta_z]` in 3D) clusters the
nodes of a generated mesh toward both ends of each direction with a nonzero
factor: node `j` of `N` lies at `L/2 (1 + tanh(beta (2j/N - 1)) / tanh(beta))`;
0 (the default) keeps the spacing uniform. For a channel between walls at
`y = 0` and `y = Ly`, `stretching = [0, 1.7, 0]` makes the cells next to the
walls about 8 times thinner than those at the center (with `Ny = 96`).

`periodic = ["x", "y"]` (and `"z"` in 3D) makes a generated mesh (except
`wedge`) periodic in those directions: `left`/`right`, `bottom`/`top` and
`back`/`front` are joined into interior faces and disappear as boundary zones.
Every scheme sees the seam as interior. Each periodic direction needs at least
3 cells. Zones of mesh files are paired with `[[periodic]]` (below).

`type = "annulus"` generates the annulus `r_inner <= r <= r_outer` about the
z axis, or its sector `0 <= theta <= angle`, of `Ntheta x Nr` quadrilaterals
with nodes on equispaced circles and rays (in 3D extruded over `0 <= z <= Lz`
into `Ntheta x Nr x Nz` hexahedra). Zones: `inner`, `outer`, for a sector
`start` (theta = 0) and `end` (theta = angle), and in 3D `back` and `front`.
`periodic = ["theta"]` joins `start` and `end` through the rotation by
`angle` (rotational periodicity, `[[periodic]]` below), and `"z"` (3D) `back`
and `front`.

| Key | Description |
|---|---|
| `r_inner`, `r_outer` | Radii (default 1 and 2) |
| `angle` | Sector angle in degrees, in (0, 360] (default 360, the full annulus) |
| `Ntheta`, `Nr` | Cells around and across (default 64 and 16) |
| `Nz`, `Lz` | (3D) Cells along and length of the axis (default 16 and 1) |

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
and curves in 3D) are ignored. Elements of order 2 and 3 are read by their
corners; high-order boundary elements (`line3`, `line4` in 2D; `triangle6`,
`triangle9`, `triangle10`, `quadrangle8`, `quadrangle9`, `quadrangle16` in 3D)
curve their faces (see curved boundaries below).

Every format and the transform below are illustrated in the user guide
([meshes](guide/meshes.md#other-mesh-formats)).

`[mesh.transform]` maps the nodes of a mesh file as it is read,
`x -> R (S x) + t`: scale first, then rotate about the origin, then translate.
Boundary zones, high-order nodes and cell types are unchanged; mirroring scales
(negative factors) are allowed. `[[periodic]]` translations, `[[mesh.curved]]`
shapes, probes and every other coordinate in the input refer to the
transformed mesh. `mallard mesh convert input.toml mesh.h5` writes the
transformed mesh.

| Key | Description |
|---|---|
| `scale` | A factor for every axis, the length of one unit of the mesh file (e.g. `0.001` or `"1 mm"` for a mesh in mm), or one per axis `[s_x, s_y]` (`[s_x, s_y, s_z]` in 3D); nonzero |
| `rotate` | 2D: angle in degrees, counterclockwise. 3D: `{axis = [a_x, a_y, a_z], angle = degrees}`, right-handed about the axis through the origin |
| `translate` | `[t_x, t_y]` (`[t_x, t_y, t_z]` in 3D) |

```toml
[mesh]
type = "file"
filename = "nozzle_mm.msh"

[mesh.transform]
scale = 0.001
rotate = {axis = [0, 0, 1], angle = 90}
translate = [0.0, 0.0, 0.05]
```

CGNS files (builds with `-DMallard_ENABLE_CGNS=ON`) are read from their one
base, whose cell dimension must match the build. Every zone must be
unstructured, with element sections of any of Mallard's cell types (`MIXED`
too, and higher orders by their corners); polyhedral `NGON_n`/`NFACE_n`
sections and structured zones are rejected. Several zones are joined into one
mesh where their `Abutting1to1` connectivities (vertex `PointList` and
`PointListDonor`, as Gmsh writes for partitioned meshes) identify their nodes.
Each boundary condition (`BC_t`, by element range or list, or by vertices)
becomes a boundary zone named after its family, or the family's
`FamilyName` (Gmsh writes physical names there), or else the condition
itself; boundary elements in no condition take their section's family or
name. High-order boundary elements (`BAR_3`, `BAR_4`, `TRI_6`, `TRI_9`,
`TRI_10`, `QUAD_8`, `QUAD_9`, `QUAD_16`) curve their faces like Gmsh's.

Fluent meshes (`.msh` files starting with `(`, and `.cas` files) are read in
ASCII or binary (single- or double-precision nodes). Every face zone other
than interior ones becomes a boundary zone named as its zone section (`39` or
`45`) names it (`zone_<id>` if none does); faces of such zones with cells on
both sides (internal walls, fans) are interior faces. Cells are rebuilt from
their faces and must be triangles or quadrilaterals (2D), or tetrahedra,
pyramids, prisms or hexahedra (3D); polygonal and polyhedral cells and
adapted meshes (cell and face trees) are rejected. Periodic face zones stay
boundary zones, to pair with `[[periodic]]`.

OpenFOAM meshes are read from `points`, `faces` (`faceList` or
`faceCompactList`), `owner`, `neighbour` and `boundary`, ASCII or binary
(32- or 64-bit labels and scalars); compressed (`.gz`) and decomposed
(`processor*`) meshes must be decompressed or reconstructed first. Each patch
becomes a boundary zone of its name (cyclic patches too, to pair with
`[[periodic]]`). Cells are rebuilt from their faces as for Fluent. The 2D
build reads a mesh one cell thick in z between `empty` patches (as for 2D
OpenFOAM runs): each cell is its face at the lowest z, the other patches give
the boundary edges, and the empty patches disappear.

In a distributed run every rank reads a Gmsh, CGNS, Fluent or OpenFOAM mesh whole (keeping only its
share), so large meshes should be converted to Mallard's HDF5 mesh format
(builds with `-DMallard_ENABLE_HDF5=ON`), of which each rank reads only its
share; generated meshes are also produced per rank:

```sh
mallard mesh convert mesh.msh mesh.h5               # from Gmsh (or CGNS, Fluent, OpenFOAM)
mpirun -n 8 mallard mesh convert input.toml mesh.h5  # a generated mesh, written in parallel
```

The file holds global arrays, with global ids the row indices:
`/nodes/coordinates` (`n_nodes x dim`, float64), `/cells/offsets` and
`/cells/nodes` (CSR of node ids, uint64; the cell type follows from the node
count, Gmsh/VTK node order), `/boundary/offsets`, `/boundary/nodes` and
`/boundary/zone` (boundary faces and their zone index), the attribute
`/boundary/zone_names`, and the root attributes `format = "mallard-mesh"`,
`version = 1` and `dimension`. Meshes with high-order boundary faces also have
`/boundary/high_order/offsets` (per boundary face) and
`/boundary/high_order/coordinates` (all nodes of each high-order face, corners
first in Gmsh order; none for linear faces).

### Curved boundaries

Boundary faces follow the true curved geometry where the mesh file has
high-order boundary elements, or where `[[mesh.curved]]` projects a zone onto
an analytic shape (any mesh: generated, linear Gmsh, HDF5). Curved faces get
quadrature points on the curve or surface with their own normals and
Jacobians, and the cells touching them their exact volumes, centroids and
TENO moments; in 3D the faces sharing an edge with a curved face (side faces
of prism layers, symmetry planes meeting the wall) are blended from their
edges. This brings orders 3 to 6 to their design accuracy at curved walls
(guide: [curved boundaries](guide/meshes.md#curved-boundaries)).

| Key | Description |
|---|---|
| `curved_geometry` | (`[mesh]`) `false` ignores high-order nodes and `[[mesh.curved]]`: straight-sided faces everywhere. Default `true` |
| `zone` | (`[[mesh.curved]]`) Boundary zone to curve |
| `shape` | `circle` (2D), `sphere` or `cylinder` (3D) |
| `center` | A point of the shape's center (cylinders: of their axis) |
| `radius` | Radius |
| `axis` | (`cylinder`) Direction of the axis |

```toml
[[mesh.curved]]
zone = "cylinder"
shape = "circle"
center = [0.0, 0.0]
radius = 0.5
```

Points of a straight face are projected radially onto the shape, minus the
linear interpolant of its corners' own offsets, so faces still meet at the
mesh nodes; zone nodes farther than 5% of an edge length from the shape are
an error. Curved boundaries are not yet supported with `physics.axisymmetric`.

## `[[periodic]]`

Pairs two boundary zones of any mesh (e.g. Gmsh physical curves or surfaces,
or zones of an HDF5 mesh) into a periodic seam, serially and in distributed
runs: zone B must be zone A translated, or rotated about an axis, node by node
(to 1e-6 of the zones' shortest edge), as Gmsh's `Periodic` constraint
produces. The faces of both zones become interior faces and take no
`[[boundaries]]` entry. A node with no match is reported with its coordinates.
Each zone joins at most one pair, and as for generated meshes, each periodic
direction needs at least 3 cells.

| Key | Description |
|---|---|
| `zones` | `[A, B]`: the zone names |
| `translation` | `[t_x, t_y]` (`[t_x, t_y, t_z]` in 3D) with B = A + translation |
| `angle` | Instead of `translation`: B is A rotated by `angle` degrees (right-handed about `axis`; counterclockwise in 2D) |
| `center` | With `angle`: a point of the axis, `[x, y]` (`[x, y, z]` in 3D); default the origin |
| `axis` | With `angle`, 3D only: the axis direction (default `[0, 0, 1]`) |

Across a rotational pair, velocity, momentum, gradients and the
reconstruction's geometry turn with the rotation; densities, pressure,
temperature, species and other scalars carry over unchanged. A mesh takes at
most one distinct rotation (pairs with the same angle, center and axis share
it, e.g. a sector side split into several zones) and, in 3D, translations
along its axis only (an annular sector periodic in z); 2D rotational meshes
take no translation. The sector must not reach the axis. Rotational
periodicity is not supported with `physics.axisymmetric`; transmissive
boundary images and characteristic boundaries' transverse terms do not reach
across the rotation (they treat the seam as the boundary's edge). See
`docs/design/periodic.md`.

```toml
# A 30 degree sector of an annular combustor about the z axis
[[periodic]]
zones = ["side_a", "side_b"]
angle = 30.0
center = [0.0, 0.0, 0.0]
axis = [0.0, 0.0, 1.0]
```

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
| `type` | `euler` (default) or `navier_stokes` |
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
| `soret` | (`mixture_averaged`) `true` adds thermal diffusion (the Soret effect) with Cantera's mixture-averaged thermal diffusion coefficients, as Cantera's `soret_enabled` (default `false`); output `DT_<species>` |
| `axisymmetric` | (2D) `true` for flows symmetric about the x axis without swirl (default `false`); see below |

Gas mixtures (`gas = "mixture"`) react when the input has a `[chemistry]`
table; every Riemann solver and face reconstruction works (`RHLL` for strong
shocks prone to the carbuncle, e.g. blunt-body detonations). With `type = "navier_stokes"` they are viscous, heat
conducting and diffusing: species diffuse along mole-fraction gradients with a
correction velocity (so the diffusive mass fluxes sum to zero) and carry their
enthalpy, and the time step includes the largest of `4/3 mu / rho`,
`lambda / (rho cv)` and the species' diffusion coefficients. Walls are no-slip
and non-catalytic: no species crosses them, so their species, Soret and
enthalpy diffusion fluxes vanish. `wall_adiabatic` conducts no heat,
`wall_heat_flux` the given `q`, and `wall_isothermal` `lambda dT/dn` from the
boundary cell to the wall temperature `T`, with the mean of the cell's
conductivity and that at `T` (and the cell's composition). Species thermodynamics are NASA-7, NASA-9 or constant-cp
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
periodic in y. Guide: [axisymmetric flows](guide/physics.md#axisymmetric-flows).

## `[les]`

Large-eddy simulation with an explicit subgrid-scale (SGS) model
(`navier_stokes` only, single gases and mixtures, planar runs). The model's
eddy viscosity `mu_t = rho nu_t` is added to the molecular viscosity, with
`cp mu_t / Pr_t` added to the conductivity and `mu_t / Sc_t` to every
species' diffusivity (mixtures: in the mixture-averaged form, so the
diffusive fluxes still sum to zero). Wall faces keep the molecular fluxes. The
eddy viscosity of each cell comes from its least-squares velocity gradient
`g_ij = du_i/dx_j`, with the filter width `Delta = V^(1/3)` (3D) or `A^(1/2)`
(2D) of the cell; the time step includes it. Design, choices and validation:
the [LES guide](guide/les.md) ([design notes](design/les.md)).

| Key | Description |
|---|---|
| `model` | `sigma` (default in 3D), `wale` (default in 2D), `vreman` or `smagorinsky`; see below |
| `C` | Model constant; defaults 1.35 (Sigma), 0.5 (WALE), 0.07 (Vreman's `c`), 0.17 (Smagorinsky) |
| `Pr_t` | Turbulent Prandtl number, default 0.9 |
| `Sc_t` | Turbulent Schmidt number, default 0.9 |
| `filter_width` | `scotti` (default in 3D): `V^(1/3) f(a_1, a_2)`, Scotti, Meneveau & Lilly's correction for anisotropic cells, `f = cosh(sqrt(4/27 ((ln a_1)^2 - ln a_1 ln a_2 + (ln a_2)^2)))` of the ratios `a_1 = h_1 / h_3`, `a_2 = h_2 / h_3` of the cell's extents (`V` over the eigenvalues of its projected-area tensor `1/2 sum_f A_f A_f^T / |A_f|`; 1.2 for an aspect ratio of 5, 1.4 for 10; 1 on cubes and regular cells); `volume` (the 2D default): `Delta = V^(1/d)`. For the eddy viscosity only (`[les.combustion]` keeps `V^(1/3)`). Channel at `Re_tau = 395` and 590: `Re_tau` within 0.7% and 1.6% of MKM with `scotti`, 2.6% and 4.1% with `volume` |
| `dynamic` | `true`: the constant from the global dynamic procedure (Germano et al. 1991; Lilly 1992), once per step from the state: `C^2 = <L^d : M> / <M : M>` (Vreman's `c` without the square) summed over the domain, `L = (rho u u)^ - (rho u)^ (rho u)^ / rho^`, `M = 2 ((rho Delta^2 D(g) S^d)^ - rho^ Delta_hat^2 D(g^) S^d(g^))`, with `^` the volume-weighted average over a cell and its vertex neighbors, of width `Delta_hat^2 = Delta^2 + 12 / d tr(cov)` from the covariance of the neighbors' centroids (`3 Delta` on uniform hexahedra), and filtered gradients for the gradients of the filtered field. Clipped at zero; `C` is the value until the first step. The sums are exact (fixed point), so the constant is independent of the rank count. `integrals.csv` gets the column `les_C`. Default `false` |

| `model` | `nu_t` | Zero for |
|---|---|---|
| `sigma` (Nicoud et al. 2011) | `(C Delta)^2 s3 (s1 - s2)(s2 - s3) / s1^2`, `s1 >= s2 >= s3` the singular values of `g` | pure shear, solid rotation, two-dimensional flows (so not allowed in 2D builds), isotropic and axisymmetric expansion; `~ y^3` at walls |
| `wale` (Nicoud & Ducros 1999) | `(C Delta)^2 (Sd:Sd)^(3/2) / ((S:S)^(5/2) + (Sd:Sd)^(5/4))`, `Sd` the traceless symmetric part of `g^2` | pure shear, isotropic expansion; `~ y^3` at walls |
| `vreman` (Vreman 2004) | `c Delta^2 sqrt(B / (g:g))`, `B` the second invariant of `g g^T` | pure shear |
| `smagorinsky` (Smagorinsky 1963) | `(C Delta)^2 sqrt(2 S:S)` | solid rotation (no wall damping: for comparisons only) |

`MU_T` (output variable) is the eddy viscosity. `[integrals] budget = true`
measures how much of the kinetic-energy dissipation comes from the model and
how much from the scheme.

### `[les.combustion]`

Turbulence-chemistry interaction of reacting LES (a viscous mixture with
`[chemistry]`) by the dynamically thickened flame model (TFLES; Colin et al.
2000, Légier et al. 2002): where a flame is, its species diffusivities and
conductivity are multiplied by `E F` and its reaction rates by `E / F`, which
keeps the laminar flame speed and thickens the flame by $F$ so that it spans
`n_res` cells; the efficiency $E$ restores the flame-surface wrinkling of the
subgrid scales at the filter width $F\delta_L$: by default $E = F^\beta$
(Charlette, Meneveau & Veynante 2002) with $\beta$ from the dynamic procedure
of Wang, Boileau & Veynante (2011), which compares the resolved flame surface
$\lvert\nabla c\rvert$ before and after a test filter (global, exact sums; written to the
integrals file as `tf_beta`).

$$
F = 1 + \left(\max\left(1, \frac{n_\mathrm{res}\Delta}{\delta_L}\right) - 1\right)\Omega, \qquad
\Omega = \min\left(1, \frac{c(1 - c)}{0.0475}\right), \qquad
c = \frac{T - T_\mathrm{unburnt}}{T_\mathrm{burnt} - T_\mathrm{unburnt}},
$$

with the flame sensor $\Omega$ of the progress variable $c$, so $\Omega = 1$ across
the whole flame ($0.05 \le c \le 0.95$) and 0 in fresh and burnt gas, where the
molecular and SGS transport are unchanged. In the flame the SGS heat and
species fluxes are multiplied by $1 - \Omega$. The rates are scaled by
integrating each cell's Strang reactor over $E/F$ times the half step (exact
for the autonomous constant-volume reactor). $F$, $E$ and $\Omega$ are computed once per step from the state, and written as
`TF_F`, `TF_E`, `TF_OMEGA`. Guide: [thickened flames](guide/les.md#thickened-flames-lescombustion).

| Key | Description |
|---|---|
| `model` | `tfles` |
| `delta_L` | Laminar thermal thickness `(T_b - T_u) / max dT/dx` of the flame (m), e.g. from Cantera |
| `s_L` | Laminar flame speed (m/s) |
| `T_unburnt`, `T_burnt` | Temperatures of the fresh and burnt gas (K) |
| `n_res` | Cells across the thickened flame, default 5 |
| `efficiency` | `dynamic` (default): `E = F^beta`, `beta` in [0, 1] from the dynamic procedure, no constant; turbulent flame speed within 8% of two DNS at `Delta = 0.4-1.6 delta_L`. `charlette`: Charlette's efficiency function of a subgrid velocity `u'` (the default until #294; doubles the flame speed at `Delta = 1.6 delta_L`). `none`: `E = 1`. Comparison in the [design notes](design/les.md), section 8.3 |
| `beta` | `charlette` only: exponent of the efficiency, default 0.5 |
| `subgrid_velocity` | `charlette` only: the `u'` of the efficiency: `colin` (default), `u' = 2 Delta^3 abs(lap(curl u))`; or `eddy_viscosity`, `u' = C_u nu_t / Delta` from the SGS model. Colin's operator overpredicts `u'` 5-8 times on meshes with `Delta >= 0.4 delta_L` and the flame speed by 2x at `Delta = 1.6 delta_L` |
| `C_u` | `charlette` only: constant of `subgrid_velocity = "eddy_viscosity"`, default 28 (a priori from a DNS at `Delta = 0.8 delta_L`; 22-39 over `Delta = 0.4-1.6 delta_L`) |

**Experimental: `model = "pasr"`**, the partially stirred reactor (Sabelnikov &
Fureby 2013) for non-premixed and partially premixed flames: no thickening;
each cell's rates are those of its filtered state times the reacting fraction
$\kappa = \tau_c/(\tau_c + \tau_\mathrm{mix})$, with the chemical time
$\tau_c = \rho c_p T/\lvert q\rvert$ of the cell's heat release rate $q$ and the mixing time
$\tau_\mathrm{mix} = C_\mathrm{mix}\Delta^2/(\nu + \nu_t)$ of molecular and SGS diffusion
across the cell, so $\kappa \to 1$ where the mesh resolves the mixing and the heat
release tends to $\rho c_p T/\tau_\mathrm{mix}$ where the chemistry is fast. Applied as the per-cell
chemistry time scale of TFLES (exact for the Strang reactor), once per step.
Key `C_mix` (default 1); output `PASR_KAPPA`. Against a DNS of an H2/N2-air
diffusion flame in decaying turbulence (heat release over 0.2-0.4 ms): -11%
at `Delta = 0.27 mm` (quasi-laminar +17%), but -50% and a near-extinguished
flame at 0.53 mm (quasi-laminar +41%). It corrects the right way at moderate
filter widths and overcorrects on coarse meshes; not validated beyond this case.

## `[initialize]`

| Key | Description |
|---|---|
| `type` | `constant` (default), `analytical`, `restart`, or `remap` (a restart file of another mesh). `Mallard -i input.toml --restart FILE` replaces the whole table by `type = "restart"`, `file = FILE`, so one input serves both the first run and its continuations |
| `u` | `constant`: `[u_x, u_y]`; `analytical`: one expression in `x`, `y`, `z` per component |
| `rho`, `p`, `T` | `constant`: `p` and `T`; `analytical`: exactly two of the three, as expressions in `x`, `y`, `z` |
| `X` or `Y` | (mixtures) Mole or mass fractions by species, e.g. `X = { H2 = 2.0, O2 = 1.0, AR = 7.0 }`; normalized, unlisted species are zero. `analytical`: expressions (or numbers) per listed species |
| `balance` | (mixtures, `analytical`) Species taking `1 - sum` of the listed fractions; without it the listed fractions are normalized |
| `n_subdivisions` | (`analytical`) Resolution of the cell averages. 2D: each cell's triangles are split into `n_subdivisions`² sub-triangles (default 4). 3D: each of the cell's tetrahedra is split into `n_subdivisions`³ sub-tetrahedra, each with a 14-point degree-5 rule (default 1) |
| `file` | (`restart`) Restart file to resume from. Restart files list their variables by name (format version 2, 3 when they also hold the weights of `[statistics]` averages, 4 when they also hold the state of characteristic boundary faces, 5 when compressed, or 6 when they also hold averaged `[statistics.spectra]`) and are read by name; files of version 1 (Mallard 0.3 and earlier) are still read, and files of a newer version are rejected ([compatibility](guide/output.md#file-formats-and-compatibility)) |

Expressions, e.g. `"x < 0.5 ? 1.0 : 0.125"`: [Expressions](guide/input_files.md#expressions).

### `type = "remap"`

Starts from a restart file of another mesh (resolution, cell type, partition
or rank count), remapped conservatively and to high order: see
[Initialization: remap from another mesh](guide/initialization.md#remap-from-another-mesh)
and [docs/design/remap.md](design/remap.md).

| Key | Description |
|---|---|
| `file` | Restart file of the source run |
| `[initialize.mesh]` | The source run's `[mesh]` table (a generated mesh or a mesh file) |
| `[[initialize.periodic]]` | The source mesh's periodic zone pairs, as `[[periodic]]` (default: the run's `[[periodic]]`) |
| `degree` | Degree of the source cells' least-squares polynomials, 0 to 5 (default 2) |
| `limiter` | `bounds` (default): each polynomial stays within the averages of its cell and vertex neighbors (one factor for the density and the species); `none` keeps the full order, for smooth fields |

## `[[boundaries]]`

Every boundary face must be assigned exactly once (a fully periodic mesh has
none, and needs no `[[boundaries]]`). Each entry has a `name`, the boundary
zone, and a `type`, below. A zone can be split
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
| `stagnation_inlet` | Subsonic inflow at stagnation pressure `p0` and temperature `T0`, for nozzles, ducts and intakes where the mass flow is part of the solution: the outgoing Riemann invariant of the interior sets the speed, and the state lies on the isentrope through (`T0`, `p0`) with `h(T) + |u|^2 / 2 = h(T0)`, with the mixture's own `h(T)` and `s(T, p)` for thermally perfect gases. An interior above `p0` sees the stagnation state at rest; supersonic inflow keeps the interior speed. See [the guide](guide/boundary_conditions.md#stagnation-inlet) and [docs/design/inlets.md](design/inlets.md) | `p0`, `T0`, `direction` (flow direction, into the domain at every face; default the inward normal of each face), `X` or `Y` for mixtures (as `upt`) |
| `mass_flow_inlet` | Subsonic inflow at a given mass flux and temperature: the outgoing acoustic wave of the interior sets the pressure, the density follows, and the face flux is the physical flux of that state, so the mass entering is exactly the target whatever the inlet density. See [the guide](guide/boundary_conditions.md#mass-flow-inlet) | `T`; `mass_flux` (`rho u` into the domain [kg/(m^2 s)], a number or an expression of `x`, `y`, `z`) or `mdot` (the total over the entry's faces [kg/s; per unit depth in 2D, per radian in axisymmetric runs]) with an optional `profile` (expression of `x`, `y`, `z`, default uniform) shaping it along the inlet; `direction` (default the inward normal of each face); `X` or `Y` for mixtures (as `upt`) |
| `dirichlet` | Exterior state from expressions in `x`, `y`, `z`, `t`, evaluated at face centers at every stage | `rho`, `u` (one expression per component), `p` |
| `p_out` | Outlet: imposes `p` if the outflow is subsonic | `p` |
| `p_out_average` | Outlet for mixed subsonic/supersonic flow: on subsonic faces, shifts the local pressure so that its area average over the boundary equals `p`, preserving the transverse profile | `p` |
| `nscbc_outlet` | Partially non-reflecting characteristic outlet (Poinsot & Lele): outgoing waves leave, and each face's incoming acoustic wave relaxes its pressure toward `p` at the rate `K = sigma c (1 - M^2) / L`. Waves of frequency `omega` reflect by `1 / sqrt(1 + (2 omega / K)^2)` | `p`, `L` (a length of the domain), `sigma` (default 0.25; 0 is perfectly non-reflecting but lets the mean pressure drift, large values approach `p_out`), `beta` (the weight of the transverse terms in the incoming wave, 0 to 1: default the local Mach number, as Lodato et al.; 1 is Poinsot & Lele's original condition, 0 drops them. Faces next to walls and other boundaries, and faces with reversed flow, have none, which keeps eddies that reverse the flow at the outlet from driving it unstable) |
| `nscbc_inlet` | Partially non-reflecting characteristic inlet: the incoming acoustic wave relaxes the normal velocity toward `u` as `nscbc_outlet` relaxes the pressure; temperature and tangential velocity are imposed, or relaxed with `sigma_T`, `sigma_t`; the composition is imposed. Supersonic inflow imposes `u`, `p`, `T` | `u` (numbers, or expressions in `x`, `y`, `z` evaluated at face centers), `p` (reference pressure, for supersonic inflow), `T`, `L`, `sigma` (default 0.25), `sigma_T`, `sigma_t` (optional), `beta` (default 0 with turbulence, whose own incoming wave carries the transverse terms), `X` or `Y` for mixtures (numbers or expressions, as `upt`), and an optional `[boundaries.turbulence]` table (below) |

Characteristic conditions change only the exterior state of the convective
flux; viscous terms treat them like `extrapolation` (image faces, or zero
normal derivatives: the outflow conditions of Poinsot & Lele). MUSCL boundary
cells leave out their ghosts across them, so that waves leave intact. Each face keeps its pressure and
normal velocity between steps, and restart files carry them, so a restarted
run reproduces an uninterrupted one bitwise, also on a different number of
ranks (restart files older than version 4 start them afresh from the
solution). Guide: [characteristic boundaries](guide/boundary_conditions.md#characteristic-boundaries).

### Wall models (`wall_model`)

A wall (`navier_stokes`, single gas, not axisymmetric) with a `wall_model`
table takes its shear stress and heat flux from a wall model fed by the LES
state at an exchange point a height `h` above each wall face, along its
normal (wall-modeled LES; Kawai & Larsson's exchange point, which avoids the
log-layer mismatch of the first cell). The state there is the containing
cell's value plus its gradient times the offset, on any cell type; the rest
of the LES sees a slip wall. Results do not depend on the number of ranks
(the halo deepens until every exchange point's cell is local) and restarts
are exact. `[[forces]]` of a modeled zone report the modeled stress and add
a column `Q`, the heat flux into the fluid. See
[docs/design/wall_model.md](design/wall_model.md).

| Key | Description |
|---|---|
| `model` | `gfm` (default): Griffin, Fu & Moin's (2023) inverse velocity transformation with the generalized Reynolds analogy, for isothermal and adiabatic walls; `equilibrium`: the equilibrium ODE model (thin-layer momentum and energy with a damped mixing length), for every wall type |
| `exchange_distance` or `exchange_cells` | `h`, absolute or in heights of the wall cell (twice its centroid's distance to the face); typically `h >= 3` cells and `h <= 0.1-0.3` of the boundary-layer thickness |
| `edge_distance` | (`gfm`) Height of the boundary-layer edge, where the edge velocity and temperature of the Reynolds analogy are sampled: a channel's half-height, an estimate of `delta` |
| `kappa`, `A_plus`, `Pr_t`, `s` | Model constants: 0.41, 17, 0.9 (`equilibrium`), Reynolds analogy factor 1.14 (`gfm`) |

```toml
[[boundaries]]
name = "bottom"
type = "wall_isothermal"
T = 1.0
wall_model = { model = "gfm", exchange_distance = 0.3, edge_distance = 1.0 }
```

### Synthetic turbulence (`[boundaries.turbulence]`)

An `nscbc_inlet` with a `turbulence` table adds synthetic turbulence to its
target velocity: the digital filter of Klein, Sadiki & Janicka (Gaussian
correlations of the given integral lengths across the inlet and, by Taylor's
hypothesis, in time), scaled to the given Reynolds stress by its Cholesky
factor (Lund, Wu & Squires). The tangential fluctuations are imposed, the
normal one enters through the face's incoming acoustic wave, which follows the
target's change without reflecting outgoing waves. The field depends only on
the time, the seed and the inlet, so runs are bitwise the same on any number
of ranks and across restarts. The inlet must be planar and normal to a
coordinate axis; a periodic direction across it makes the field periodic. See
the [guide](guide/boundary_conditions.md#synthetic-turbulence).

| Key | Description |
|---|---|
| `reynolds_stress` | `[R_xx, R_yy, R_zz, R_xy, R_xz, R_yz]` (2D: `[R_xx, R_yy, R_xy]`), numbers or expressions in `x`, `y`, `z`, positive semidefinite |
| `profile` | Instead of `u` and `reynolds_stress`: a CSV file of plane averages as `tools/post/plane_average.py` writes them (a first column `x`, `y` or `z`, then `MEAN_U_X`, `MEAN_U_Y`(, `MEAN_U_Z`) and `COV_U_X_U_X`, ...; missing off-diagonal covariances are zero), interpolated linearly along that coordinate: the statistics of a precursor run |
| `length_scale` | Integral length scales: a number, `[L_x, L_y, L_z]` (along each direction, for all components), or one such list per velocity component. The one along the inlet normal sets the integral time scale `L / U_c` |
| `convection_velocity` | `U_c` of Taylor's hypothesis (default the area-averaged mean inflow speed) |
| `seed` | Random seed, a non-negative integer (default 1); inlets also differ by their position in `[[boundaries]]` |
| `zero_net_flux` | `true` (default) removes the area mean of the normal fluctuation over the inlet at every instant, which would otherwise drive a plane acoustic wave; it also removes about `4 L_y L_z / A` of the normal Reynolds stress for an inlet of area `A` |
| `points_per_length` | Resolution of the generator's grid on the inlet: points per smallest integral length (default 6, at least 2) |

```toml
[[boundaries]]
name = "inlet"
type = "nscbc_inlet"
u = [1.0, 0.0, 0.0]
p = 17.857
T = 1.0
L = 6.0

[boundaries.turbulence]
reynolds_stress = [0.01, 0.01, 0.01, 0.0, 0.0, 0.0]
length_scale = 0.25
```

## `[[sponges]]`

Optional sponge layers, each adding `-strength (U - U_ref)` to the conservative variables (and species partial densities) per unit volume, with `U_ref` the conservatives of a reference state. Several sponges add up. The explicit source caps the time step at `1 / max strength`.

| Key | Description |
|---|---|
| `strength` | Damping rate (1/s): a number or an expression in `x`, `y`, `z`, zero outside the layer, e.g. `"x > 4 ? 20 * ((x - 4) / 1)^2 : 0"` (a gradual ramp reflects less than a step) |
| `u`, `p`, `T` | Reference state: numbers or expressions in `x`, `y`, `z` (`u` has one per component) |
| `X` or `Y`, `balance` | (mixtures) Reference composition, numbers or expressions as in `[initialize]` |

Strength and reference are evaluated once at cell centroids.

## `[numerics]`

| Key | Description |
|---|---|
| `preset` | `robust`, `accurate`, `les` or `reacting`: a validated combination of `face_reconstruction` (`type` and its `order` or `limiter`), `riemann_solver`, `convective_flux`, `time_integrator` and `[run] cfl`, each used only where the input does not set that key (a reconstruction option only when the reconstruction type is the preset's; `cfl` only when `[run]` sets neither `cfl` nor `dt`). What each chooses and why: [Numerics: presets](guide/numerics.md#presets). Default: none (the defaults below) |
| `riemann_solver` | `Rusanov`, `HLL`, `HLLC` (default), `Roe`, or `RHLL` (rotated hybrid HLL-Roe, carbuncle-free except at shocks at rest on cell faces; see [numerics](numerics/overview.md#stationary-shocks-on-cell-faces)) |
| `time_integrator` | `FE`, `SSPRK3` (default), `RK4`, `SSPRK54` (five-stage fourth-order SSP, low storage), `CK45` (Carpenter-Kennedy five-stage fourth-order, 2N storage) or `RDPK3Sp35` (five-stage third order with an embedded error estimate, for `[numerics.error_control]`). See [time integrators](numerics/overview.md#time-integrators) for their stability limits and cost |
| `check_nan` | Check every conservative and partial density for non-finite values right after each step, so a diverging step writes no output (one more reduction per step). Without it the run still stops at the next step, through the time-step reduction; see [run control](guide/running.md). Default `false` |
| `double_flux` | (gas mixtures) `true` for the double-flux scheme: each cell's energy is updated with its own `cp / cv` and energy offset frozen over the time step on both sides of its faces, and reset to the true equation of state after the step, so pressure and velocity stay exactly uniform across contacts between different gases. Energy is then not exactly conserved (about 0.2% over a multicomponent shock tube). Default `false` |
| `low_mach_cutoff` | Low-Mach correction of the convective flux: the velocity jump across each interior face is scaled by `z = min(1, max(M_L, M_R, low_mach_cutoff))` before the Riemann solver, so that upwind dissipation scales with the flow speed rather than the sound speed. Default 0.1; 1 disables it. See [`numerics/overview.md`](numerics/overview.md) |
| `convective_flux` | `riemann` (default): the Riemann solver at every face; `hybrid`: the kinetic-energy-preserving central flux of Kuya, Totani & Kawai (2018) on the same reconstructed states, blended into the Riemann solver where a shock sensor fires, `F = F_KEEP + phi (F_Riemann - F_KEEP)`. For large-eddy simulation: the central flux adds no dissipation, so the SGS model does the work (see `[integrals] budget`). `phi` is the larger of the face's two cells': 1 where the Ducros sensor restricted to compressions, `(div u)^2 / ((div u)^2 + |omega|^2)` with `div u < 0`, exceeds `sensor_threshold`, else `upwind_floor`. Boundary faces other than walls and symmetry planes use the Riemann solver. Species stay upwinded with the blended mass flux. Contacts and expansions are central: on Sod's problem the density error is about 1.8 times the Riemann solver's, with a 0.1% overshoot at the contact |

### `[numerics.error_control]`

Error-based time steps ([Ranocha et al. 2022](references.md#ranocha-2022)), with `time_integrator = "RDPK3Sp35"`. The first step takes `[run] cfl` or `dt`; later steps are sized by a PID controller on the embedded error estimate, and a step whose error is too large is rejected and retried with a smaller one. The CFL number is then an output (`CFL` variable, progress `dt`), not an input. Not available with `[chemistry]` or `double_flux`.

| Key | Description |
|---|---|
| `tolerance` | Absolute and relative tolerance: a step's error is the largest over cells and conservative variables (and partial densities) of `abs(e) / (abs_tolerance + rel_tolerance max(abs(U^n), abs(U^n+1)))`, which the controller keeps near 1 |
| `abs_tolerance`, `rel_tolerance` | Override `tolerance` for either part |
| `beta` | PID coefficients `[b1, b2, b3]`, default `[0.64, -0.31, 0.04]` (Ranocha et al.'s for RDPK3Sp35) |

### `[numerics.hybrid]`

| Key | Description |
|---|---|
| `sensor_threshold` | Ducros sensor above which a cell compressing the flow is a shock, default 0.65 (1 disables the Riemann solver) |
| `upwind_floor` | Upwind fraction of the other cells, default 0; a small value (e.g. 0.05) damps grid-scale noise on very irregular meshes, at the cost of numerical dissipation the budget then reports |

### `[numerics.face_reconstruction]`

| Key | Description |
|---|---|
| `type` | `FO` (first order; the default, also without this table), `MUSCL` or `TENO` |
| `limiter` | (`MUSCL`) `venkatakrishnan` (default), `barth_jespersen` or `none` |
| `venkatakrishnan_K` | (`MUSCL`) Venkatakrishnan threshold constant, default 5 |
| `order` | (`TENO`) Order of accuracy, 3 to 6, default 5. In 3D, faces use Dunavant (triangles) or Gauss (quadrilaterals) rules exact to this order, capped at degree 5 on triangles |
| `stencil_factor` | (`TENO`) Minimum large-stencil size as a multiple of the number of polynomial coefficients, default 2; a stencil grows past it until its reconstruction's Lebesgue constant is at most 4 in 3D (10 in 2D; see [TENO-E details](numerics/teno_e.md#central-stencil)). Smaller values (e.g. 1.5) are markedly less dissipative for fine smooth structures (Shu-Osher entropy waves: 50% more amplitude at 200 cells) but less robust at discontinuities. |
| `small_stencil_size` | (`TENO`) Cells per sector stencil, default 10 (18 in 3D) |
| `troubled_threshold` | (`TENO`) Troubled-cell threshold on the density-jump variance, default 1e-3 |
| `troubled_upper` | (`TENO`) Variance at which the adaptive cutoff reaches its largest value (most dissipative), default 1e-2 |
| `C_T` | (`TENO`) Fixed TENO cutoff; adaptive (1e-10 to 1e-6) if omitted |
| `characteristic` | (`TENO`) Select stencils on characteristic variables, default true |
| `max_condition` | (`TENO`) Stencils grow until the least-squares system's condition estimate is below this, default 1e8 |
| `cache_file` | (`TENO`) Save the precomputed stencils and matrices here, and reuse them on later runs of the same mesh, boundary assignment and TENO options; anything else is detected and recomputed. Distributed runs write one file per rank, `<cache_file>.r<rank>-of-<ranks>`, for that rank count and partition, and record the halo depth the stencils need, so a cached run sets up its halo once. Partitions repeat for the same mesh and rank count. Size per cell: about 2.5 / 4 / 6 / 11 KB in 2D and 10 / 18 / 32 KB (hexahedra) or 9 / 13 / 29 / 63 KB (tetrahedra) in 3D for orders 3 / 4 / 5 / 6, e.g. 8.4 GB for 64³ hexahedra at order 5; reading it takes seconds; the 3D setup takes about 5 s per million hexahedra and 20 s per million tetrahedra on an A100 (order 5), and minutes on CPUs. The cache records whether the device or the host computed it; both give the same tables |
| `cache_single_precision` | (`TENO`) Store the pseudo-inverses and smoothness-indicator matrices (3D: the cells' moments, from which the matrices are formed) of `cache_file` in single precision: about half the size (17 instead of 32 KB per cell for hexahedra at order 5). The run that writes the cache rounds them too, so it and the runs that read the cache agree with each other, but no longer with a run without it, whose results differ at single-precision round-off. Default false |
| `bound_preserving` | (`TENO`) Scale troubled-cell polynomials to keep density and pressure within the neighbors' range, default false |
| `single_precision_tables` | (`TENO`, 3D, not with gas mixtures) Keep the pseudo-inverses on the device in single precision: about 40% less device memory for 3D TENO5 (25 instead of 40 KB per hexahedron, 31 instead of 51 KB per tetrahedron on an A100, everything included), so about 1.6 times the cells per GPU, and a 3-8% faster step. Results change at single-precision round-off: below 1e-7 of each field's range for smooth flow, but up to 1e-3 (order 3: 7e-3) behind shocks, where the stencil selection of some cells goes the other way (3D blast on hexahedra, 60-80 steps; [design notes](design/performance.md)); the cache, if any, holds the tables in single precision too. Default false |
| `troubled_capacity` | (`TENO`) Fraction of the reconstructed cells that the troubled-cell passes take at a time, which sizes their scratch; more troubled cells are done in further rounds, which recompute their central coefficients (same results). Default 0.5 in 3D (1 in 2D) |
| `curved_mirrors` | (`TENO`) Also take mirror images across curved wall and symmetry faces (across each face's plane) into the stencils, default false: stencils are one-sided at curved walls, where mirroring is only second-order consistent (curved far fields and inflows keep their mirror images) |
| `curved_wall_degree` | (`TENO`) Largest degree of the central polynomials of cells whose stencils reach a curved wall or symmetry face without mirror images, default 3: one-sided fits of degree 4 and 5 grow in time there, so orders 5 and 6 are fourth order at curved walls and of their order elsewhere |

## `[chemistry]`

Finite-rate chemistry of gas mixtures (`gas = "mixture"` with a mechanism
that has reactions). By default each step is Strang split: every owned cell
that needs it is advanced as an adiabatic, constant-volume reactor over
`dt / 2`, then the flow takes its step, then the reactors take another
`dt / 2` (with `fuse_half_steps`, the second half step is fused with the
next step's first). With `coupling = "simpler"` (SIMPLER balanced splitting,
[Wu, Ma & Ihme 2019](https://doi.org/10.1016/j.cpc.2019.04.016)) each step
evaluates the flow's right-hand side `T(U^n)` once, advances every owned
cell over `dt` under its chemistry plus that constant source, then lets
the flow correct over the second half of the step with `T(U) - T(U^n)`:
steady states of chemistry and transport stay exactly steady at any `dt`
(see the [thermochemistry guide](guide/thermochemistry.md#finite-rate-chemistry)). The integrator
is RODAS, an adaptive Rosenbrock method with the analytical Jacobian, in
double precision in every build.

| Key | Description |
|---|---|
| `enabled` | `false` keeps the mixture non-reacting (default `true`) |
| `integrator` | `rosenbrock` (the default and only choice so far) |
| `coupling` | `strang` (default) or `simpler`; `simpler` does not combine with `fuse_half_steps`, `numerics.double_flux` or `numerics.convective_flux = "hybrid"` (SIMPLER's flow integrator amplifies non-dissipative operators such as the hybrid flux's central part; see the [design notes](design/chemistry.md#simpler-balanced-splitting-option)) |
| `rtol` | Relative tolerance on the mass fractions and `T` (default `1e-6`) |
| `atol` | Absolute tolerance on the mass fractions (default `1e-10`) |
| `max_steps` | Sub-steps allowed per cell and half step (default 100000); more stop the run |
| `T_frozen` | No chemistry in cells below this temperature (default 0) |
| `fuse_half_steps` | `true` fuses the closing half step of a step with the next step's opening one, except where output, checks, probes, statistics or the end of the run read the state (default `false`). Faster where cells take few sub-steps, but results then depend on when output is written, and a restart reproduces an uninterrupted run only from a step at which that run also wrote output |
| `C_reg` | concentration [kmol/m³] below which reaction orders 0 < n < 1 follow a regularized law (default `1e-12`). Larger values (e.g. `1e-10`) make trace fuel burning out in hot gas much cheaper to integrate (`flame_3d`: chemistry 20 to 11.5 ms per step) but change results: after 100 steps of `flame_3d`, T by up to 0.54 K, Y by up to 3e-5, the integrated heat release by 0.08% |
| `load_balance` | `true` moves the integration of cells from ranks with more chemistry work than the mean (sub-steps of their cells' last calls) to ranks with less, in every call where the most loaded rank exceeds the mean by 5%; only the cells' states travel, and results stay bitwise the same (default `false`). The summary reports the time ranks wait for the slowest rank's chemistry with or without it |
| `sparse` | `true` for the sparse LU (static pattern, with the Jacobian's dense rank-one part by Sherman-Morrison), `false` for the dense one; by default sparse from 30 species when its factors fill at most 60% of the dense matrix (GRI-3.0 and larger) |
| `lanes` | Vector lanes integrating one cell: 1 for one thread per cell (cells ordered by their last cost), a power of 2 up to 32 for a team per cell on GPUs; default 0, automatic: a warp per cell on GPUs from 16 species (8 warps, 16 from 512 species, for cells whose last call took 16 or more sub-steps), else one thread |

Cells whose mass fractions would change by less than `atol / 100` over the
half step at their current rates are skipped. Reaction types: elementary,
three-body, falloff (Lindemann, Troe, SRI), `pressure-dependent-Arrhenius`
(PLOG) and `Chebyshev`, with non-integer reactant `orders` (global
mechanisms such as Westbrook–Dryer's). Orders between 0 and 1 follow the
power law down to a concentration of `1e-12` kmol/m³ and a quadratic with a
bounded slope below it, so the Jacobian
stays finite where such a reactant runs out. Output variables: `HRR` (heat release rate,
W/m³) and `CHEM_COST` (chemistry sub-steps of the cell in the last step);
restart files also hold `CHEM_H`, each cell's last sub-step, so restarted
runs repeat the uninterrupted one exactly. The progress rows add the share of
cells advanced in the last half step and the most sub-steps of a cell in the
last step, and the summary the chemistry's share of the wall time.

## `[radiation]`

Radiative heat loss of gas mixtures in the optically thin limit (the TNF
workshop model): every cell loses
$q = 4\sigma\sum_i p_i a_i(T)\,(T^4 - T_\mathrm{ambient}^4)$ [W/m³], with $p_i$ the
partial pressures [atm] of the radiating species and $a_i(T)$ their Planck-mean
absorption coefficients (the TNF workshop's fits to RADCAL, 300-2500 K:
H2O and CO2 polynomials in 1000/T, CH4 and CO in T). Each cell emits but
absorbs only the background at `T_ambient`, so the model holds where the gas
is optically thin (small flames, lean or diluted ones). The loss is a source of
the energy equation in the flow step; it needs no `[chemistry]`.

| Key | Description |
|---|---|
| `model` | `optically_thin` (the default and only choice) |
| `T_ambient` | Background temperature [K] (default 300) |
| `species` | Radiating species, any of `H2O`, `CO2`, `CO`, `CH4`; default all four that the mechanism has (listing one it lacks is an error) |

Output variable: `QRAD`, the loss [W/m³]. Cantera's flames with
`radiation_enabled` use the same model with H2O and CO2 only and no
background term: `species = ["H2O", "CO2"]` and `T_ambient = 0` reproduce it.

## `[[forces]]`

Write the force of the fluid on a boundary zone to a CSV file
(`step, t, Fx_pressure, Fy_pressure, Fx_viscous, Fy_viscous`, per unit depth, or per radian for
axisymmetric flows; in
3D `step, t, Fx_pressure, Fy_pressure, Fz_pressure, Fx_viscous, Fy_viscous, Fz_viscous`).
Zones with a wall model report the modeled shear stress as the viscous force
and add a column `Q`, the modeled heat flux into the fluid over the zone.

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
(order-consistent: on the Taylor-Green vortex at $128^3$ they match
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
| `budget` | `true` adds the kinetic-energy budget (planar runs): `ke_rate_convective`, `ke_rate_viscous` and `ke_rate_sgs`, the rates of change of the resolved kinetic energy `sum V rho |u|^2 / 2` caused by the convective, molecular viscous and SGS fluxes of the current state (each `sum V (u . R_m - |u|^2 / 2 R_rho)` over that part `R` of the right-hand side), `pressure_work`, the rate a scheme without numerical dissipation would give (the two-point pressure flux `mean(p) n` of the cell values on interior faces, so `sum over faces mean(p) (u_1 - u_0) . n A`, plus the boundary faces' own fluxes), and `eps_numerical = pressure_work - ke_rate_convective`, the scheme's dissipation of kinetic energy on interior faces (the convective terms of the exact equations change the kinetic energy only by the pressure work and the boundary fluxes). `pressure_work` is the discrete counterpart of `pressure_dilatation`; unlike the latter it does not carry the gradients' discretization error, which matters where `p div u` is large (flames: thermal expansion at atmospheric pressure, several orders above the dissipation). `-ke_rate_viscous` and `-ke_rate_sgs` are the molecular and SGS dissipation. Costs one extra right-hand side per row and leaves the solution unchanged. Default `false` |

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
`tools/post/plane_average.py` averages them further over homogeneous directions
(e.g. over x and z of a channel, as a function of y). `fields` and
`products` may be left out when `[statistics.spectra]` or
`[statistics.phase]` is given; both take `interval` and `t_start` from
`[statistics]` unless they set their own, and `reset` applies to them too.

### `[statistics.spectra]`

Spectra and two-point correlations along homogeneous directions of a
generated `cartesian` mesh (quadrilaterals or hexahedra): directions that are
periodic and not stretched. At each position of the other (inhomogeneous)
directions, e.g. at each y of a channel, a sample takes the discrete Fourier
transform over the homogeneous directions H of every field,
`a_hat(k) = sum_x a(x) exp(-i k.x) / N_H`, and of every pair the cross
spectrum `a_hat conj(b_hat)` summed over the wavenumbers of all directions
of H but one, `h`: the one-dimensional spectrum along `h`. With every
direction homogeneous (a periodic box), also the shell spectrum, sums of the
real part over `|k|` in shells of width `dk = 2 pi / L_max` around `s dk`.
These and the plane means are time-averaged like the `[statistics]` means
(weight `t - t_prev`); the outputs subtract the product of the averaged plane
means, so they describe the fluctuations `a' = a - <a>`. The values are those
of the cells (cell averages, no deconvolution).

| Key | Description |
|---|---|
| `fields` | Variables whose spectra (autospectra) to keep: any of the `[statistics]` fields |
| `products` | Pairs `"A*B"` whose cross spectra (co-spectra) and cross correlations to keep |
| `directions` | Homogeneous directions, default all periodic directions of the mesh |
| `average` | `true` (default): running time averages, written to `file` with the other output (`[[write_data]]`) and at the end, and carried by restart files; `false`: each sample's spectra, written to `<file stem>_<step>.csv` (e.g. the instantaneous spectra of decaying turbulence) |
| `file` | Output file, default `spectra.csv` (`spectra/spectra.csv` with `average = false`) |
| `interval`, `t_start` | Sampling, default those of `[statistics]` |

The output CSV starts with a comment line (step, time, samples, total
weight), then the columns `quantity, a, b, direction, position, <the
inhomogeneous axes' cell-center coordinates>, index, abscissa, value`, with
`quantity`

- `mean`: the plane mean of `a`;
- `spectrum`: the one-sided spectrum `E_ab(k_n)`, `n = 0 .. N/2`, `k_n = 2 pi n / L`
  (abscissa), the real part (co-spectrum) of the two-sided one folded onto
  `n >= 0`; it sums (it is not a density) to the covariance `<a'b'>`;
- `correlation`: `R_ab(r_m) = <a'(x) b'(x + r_m e_h)>`, `r_m = m L / N`,
  `m = 0 .. N - 1` (periodic; `R(-r) = R(L - r)`), the inverse transform of
  the two-sided spectrum, so `R_ab(0) = <a'b'>`;
- `shell`: the shell spectrum (`direction` `k`), sums to `<a'b'>` too.

The results are bitwise the same on any number of ranks
([design](design/insitu_statistics.md)). Restart files (format version 6)
carry the averages as named arrays; a restart file holding spectra of other
fields or directions is an error unless `reset = true`.
`tools/post/insitu_statistics.py` recomputes the same quantities from restart
files, compares two spectra files and plots them.

### `[statistics.phase]`

Phase averages: per cell, the time-weighted means of `fields` over the
samples whose phase falls in each of `bins` equal bins, output as
`PHASE_<bin>_<field>` (bins zero-padded, `PHASE_*` selects them all). The
phase is either `(t - t_reference) / period` modulo 1, or set by a trigger:
with the upward crossings of `level` by `variable` in the cell holding
`probe` (located like a probe point), detected between consecutive samples
and interpolated linearly in time, the phase of a sample at `t` is
`(t - t_n) / (t_n - t_{n-1})` for the last two crossings `t_{n-1} < t_n`;
phases past 1 (a period longer than the last) fall in the last bin, and
samples before the second crossing are skipped. The trigger is read at every
sampling step, also before `t_start`. Restart files carry the bins, their
weights and the trigger state.

| Key | Description |
|---|---|
| `fields` | Variables to average: any of the `[statistics]` fields |
| `bins` | Number of phase bins |
| `period`, `t_reference` | A fixed period, phase 0 at `t_reference` (default 0), or |
| `probe`, `variable`, `level` | The trigger point `[x, y(, z)]`, its variable and the level of its upward crossings (default 0) |
| `interval`, `t_start` | Sampling, default those of `[statistics]` |

## `[[probes]]`

Samples of the solution on a geometry (points, curves or boundary zones;
[geometry guide](guide/geometry.md#point-sets-and-curves)), every `interval`
steps or `time_interval` of simulation time, written raw and/or reduced. Setup
locates the samples in the cells of each rank; a write evaluates them on the
device of the ranks that own them and gathers them on rank 0, which writes them
in a fixed order, so the files are the same bit for bit on any number of
ranks. A probe that is not due costs nothing.

| Key | Description |
|---|---|
| `name` | Probe name |
| `variables` | Any of the `[statistics]` fields, `U` for all velocity components, `Y_*` for every species |
| `shape` | `point`, `points`, `line`, `polyline`, `circle`, `arc` or `curve` (below); without it, `point` or `start`/`end` select a point or a line |
| `boundary` | Instead of a shape: a boundary zone name, or a list of them; each face is a sample (below) |
| `boundary_type` | Instead of, or with, `boundary`: every zone whose `[[boundaries]]` condition has this `type`, or `wall` for all walls |
| `reference` | `{rho, u = [...], p}`: free stream of `CF` and `CP` |
| `sampling` | `cell` (default): the average of the cell holding each sample. `point`: the value at the sample, from the reconstruction (see below) |
| `interval` / `time_interval` | Every this many steps (default 1) / this much simulation time; with `time_interval` the time step is shortened to land on each output time, as for `[[write_data]]` |
| `format` | `csv` (default), `hdf5` (builds with `-DMallard_ENABLE_HDF5=ON`), `vtkhdf` (likewise: one VTKHDF `PolyData` file of the samples, as vertices or polylines, with their point values at every output time; ParaView opens it as a time series) or `vtp` |
| `file` | Output file, default `probe_<name>.csv`, `.h5`, `.vtkhdf` or `.pvd` |
| `samples` | `false` writes only the reductions (default `true`) |
| `reductions` | Any of `integral`, `mean`, `favre`, `min`, `max`, `rms` (below) |
| `reductions_file` | Default `probe_<name>_reductions.csv` |
| `scale`, `rotate`, `pivot`, `translate` | Transform of the geometry, as for [regions](#regions): scaled (a factor, or one per axis), then rotated (2D: `rotate = <degrees>`; 3D: `rotate = {axis = [x, y, z], angle = <degrees>}` or `{axes = [[...], [...], [...]]}`), both about `pivot` (default the origin), then translated |

Shapes (`[x, y(, z)]` points):

| `shape` | Keys |
|---|---|
| `point` | `point` |
| `points` | `points = [[...], ...]`, or `points_file`: a text file of one point per row (numbers split by commas or blanks; `#` starts a comment, a header row is skipped) |
| `line` | `start`, `end`, `n_points` (at least 2) evenly spaced points |
| `polyline` | `vertices` (at least 2), `n_points` per segment, ends included |
| `circle` | `center`, `radius`, `n_points` (at least 3); 3D: `normal`, and optionally `reference`, the direction of the first point (default: the axis least aligned with the normal, projected) |
| `arc` | as `circle`, with `angles = [start, end]` in degrees from the x axis (3D: from `reference`) and `n_points` from start to end, ends included |
| `curve` | `x`, `y`(, `z`): expressions of the parameter `s` over `s_range = [s0, s1]` (default `[0, 1]`), `n_points` evenly spaced in `s` |

**Point values.** With TENO, `sampling = "point"` evaluates each cell's
central reconstruction polynomial at the sample (order-consistent: the error
falls at the scheme's order, e.g. fifth order for `order = 5`); cells that the
TENO indicator flags troubled, schemes without such polynomials (FO, MUSCL),
and values with a nonpositive density or pressure take the linear
least-squares reconstruction (second order), and failing that the cell
average. `RHO`, `U_*` and `P` are reconstructed; `T` and `H` follow `p / rho`
at the cell's composition (exact for a perfect gas), and mass fractions are
cell averages. A sample on a face between cells belongs to the cell with the lowest global
id, so any number of ranks pick the same cell; samples outside the domain take
the nearest cell's average in both modes.

**Boundary probes.** `boundary = "wall"` samples the zone's faces of every
rank: one sample per face, at its centroid, with the face measure as its weight
(per radian in axisymmetric runs). The values are those of the state the
boundary condition sets on the face, from the interior trace (the cell average,
or with `sampling = "point"` the reconstruction at the face centroid): walls
take their velocity (no slip; slip walls and symmetry planes drop the normal
component) and isothermal walls their temperature at the trace's pressure;
`upt`, `farfield`, `p_out` and `p_out_average` faces the exterior state of their
condition; `dirichlet` faces their prescribed state; transmissive and
characteristic faces the trace. Normals point out of the fluid, so fluxes are
positive out of the domain. In 2D the faces are chained through shared nodes
into polylines, each traversed with the fluid on its left (counterclockwise
around a domain's outer boundary, clockwise around a body), with the arc
length `s` of each face center; in 3D the faces are polygons, ordered by
their cells' global ids. Besides the `[statistics]` fields, boundary probes
have:

| Variable | Meaning |
|---|---|
| `MASS_FLUX` | `rho u . n` |
| `MOMENTUM_FLUX_<X,Y,Z>` (`MOMENTUM_FLUX`) | `rho u u . n + p n` |
| `ENTHALPY_FLUX` | `rho u . n (h + |u|^2 / 2)` |
| `SPECIES_FLUX_<species>` (`SPECIES_FLUX_*`) | `rho Y_k u . n` (convective; `upt` faces with their prescribed composition) |
| `TRACTION_PRESSURE_<X,Y,Z>` (`TRACTION_PRESSURE`) | `p n`, the pressure force of the fluid on the boundary per unit area |
| `TRACTION_VISCOUS_<X,Y,Z>` (`TRACTION_VISCOUS`) | `-tau . n`, its viscous part (molecular) |
| `TAU_W_<X,Y,Z>`, `TAU_W` | Wall shear stress: the tangential part of `-tau . n`, along the flow on a wall at rest; and its magnitude |
| `CF` | `TAU_W_<axes> . u_ref / |u_ref| / q_ref`, with `q_ref = rho_ref |u_ref|^2 / 2` |
| `CP` | `(p - p_ref) / q_ref` |
| `Q_W` | Heat flux into the fluid, `kappa dT/dn` (the convention of `wall_heat_flux`): zero on adiabatic walls, `q` on heat-flux walls |
| `U_TAU`, `Y_PLUS` | `sqrt(TAU_W / rho_w)` and `rho_w U_TAU d / mu_w`, `d` the distance from the face to its cell's centroid |

The gradients are the least-squares ones of the viscous flux, with its wall
treatment (normal derivatives from the wall velocity and temperature), and
the viscosity and conductivity those of the face (mixtures: of the cell), so
that `integral` of `TRACTION_PRESSURE` and `TRACTION_VISCOUS` with
`sampling = "cell"` gives the forces of `[[forces]]`. In axisymmetric runs a
face's `area` and the `integral` reductions are per radian of the revolved
wall (`area` is `r` times the face length, the reductions' column `area`), as
`[[forces]]` is. `TAU_W`, `Q_W` and the
wall-friction quantities need `navier_stokes`. CSV rows of boundary probes are
`step, t, face, x, y(, z), s (2D), nx, ny(, nz), area, <variables>`; HDF5
files add `normals` and `area` (and in 3D `vertices` and `polygons/vertices`,
`polygons/offsets`); VTP and VTKHDF files are polylines of the face centers (2D) or the
faces as polygons with cell data (3D).

```toml
[[probes]]
name = "wall"
boundary_type = "wall"
variables = ["P", "CP", "CF", "Q_W", "Y_PLUS"]
reference = {rho = 1.2, u = [30.0, 0.0], p = 101325.0}
format = "vtp"
interval = 100

[[probes]]
name = "outlet"
boundary = "outlet"
samples = false
variables = ["MASS_FLUX", "ENTHALPY_FLUX"]
reductions = ["integral"]
```

**Output.** `csv` appends rows `step, t, point, x, y(, z), <variables>` (one per
sample and step; a restarted run appends). `hdf5` holds `coordinates`
(samples x dimension), for curves `arc_length` and `polylines/nodes` with
`polylines/offsets`, and the extensible `step`, `time` and `fields/<variable>`
(steps x samples); `vtp` writes one VTK PolyData file per output,
`<stem>_<step>.vtp` (vertices for point sets, polylines with `arc_length`
for curves), indexed by the `.pvd` series; `vtkhdf` writes one VTKHDF PolyData
file (the same vertices or polylines, `arc_length`, and each variable's point
values at every output, with `Steps/Values` the times). A restarted run cuts the
HDF5 rows, the VTKHDF outputs and the `.pvd` entries after the restart back and
continues them.

**Reductions.** Rows `step, t, length` (curves, 2D boundaries), `area` (3D boundaries) or `count` (point sets), then
`<reduction>_<variable>` for each reduction and variable: `integral` is
`int q ds`, `mean` `int q ds / int ds`, `favre` the density-weighted mean
`int rho q ds / int rho ds`, `rms` `sqrt(int q^2 ds / int ds)`, and `min`, `max`
the extrema over the samples and the integration points. Curves integrate
with three Gauss points on each element between consecutive samples (exact for
polynomials of degree 5 along straight elements), boundaries with one point
per face (the face measure), point sets with unit weights
(`integral` is the sum, `mean` the arithmetic mean). Rank 0 sums them in a
fixed order, so they too are the same bit for bit on any number of ranks.

```toml
[[probes]]
name = "wake"
variables = ["U", "P"]
sampling = "point"
shape = "arc"
center = [0.0, 0.0]
radius = 2.0
angles = [-60.0, 60.0]
n_points = 121
time_interval = 1.0e-4
format = "vtp"
reductions = ["mean", "rms"]
```

## `[source]`

Optional source terms, added per unit volume.

| Key | Description |
|---|---|
| `gravity` | `[g_x, g_y]` (`[g_x, g_y, g_z]` in 3D); adds `rho g` to the momentum and `rho u . g` to the energy equation |
| `rho`, `rhou`, `rhoE` | Expressions (default none) for the mass, momentum and energy sources, as in `[[sources]]` without a region (they read the same variables) |
| `time_dependent` | Ignored, with a warning: expressions of `t` are re-evaluated at every stage on the device, others once (Mallard 0.8 and earlier evaluated every expression once unless this was `true`) |
| `mass_flow` | `[m_x, m_y]` (`[m_x, m_y, m_z]` in 3D): hold the volume average of `rho u` along this vector at its magnitude with a uniform body force `f` per unit volume along it (constant mass flow, e.g. a periodic channel). At every Runge-Kutta stage `f` cancels the rate of change of that average from all other terms (one global sum, exact, so the same on any number of ranks) and closes any remaining gap within the time step, so a flow started at the target stays there to rounding. The energy gains the force's work `f . u`. |

The scheme is not exactly well balanced: hydrostatic states carry small spurious velocities (about 1e-4 of the sound speed on a 32x32 mesh) that vanish at second order under refinement. Wall and symmetry ghost states continue the hydrostatic pressure gradient.

## `[[sources]]`

Source terms per unit volume (or integrated totals over a region), any number
of them, added up with `[source]`. Each is evaluated on the device: entries of
position alone once, entries of `t`, the solution or domain averages at every
Runge-Kutta stage, in one kernel without host round trips (steps stay CUDA
graphs, except with domain averages, which need a global sum per stage).

| Key | Default | Description |
|---|---|---|
| `name` | `sources[i]` | Name in the run log |
| `region` | everywhere | A [region](#regions): the values are multiplied by each cell's volume fraction of it |
| `total` | `false` | `true`: the values are integrated rates over the region (kg/s, N, W; per unit depth in 2D, per radian in axisymmetric runs), spread over it in proportion to the volume fraction, so that their integral is exact on any mesh; they may then depend on `t` only |
| `rho`, `rhou`, `rhoE` | none | Mass, momentum (one per component) and energy sources: numbers, quantities with units (`rhoE = "5 MW/m^3"`, with `total`: `"50 W"`) or expressions (in SI units) |
| `rhoY` | none | Mixtures: table of species partial-density sources, e.g. `{H2 = "1e-3"}`; the mass source `rho` is their sum (and `rho` itself is not allowed) |

Expressions ([syntax](#regions)) read the variables `x`, `y`, `z`, `t`,
`rho`, `u_x`, `u_y` (3D: `u_z`), `p`, `T`, for mixtures `Y_<species>` and
`X_<species>` (characters other than letters, digits and `_` in species names
become `_`), and the domain averages (volume-weighted over the whole mesh)
`mean_rho`, `mean_rhou_x`, `mean_rhou_y` (`mean_rhou_z`), `mean_rhoE` and
`mean_ke` (of `rho |u|^2 / 2`). Values are taken at cell centroids, from the
cell averages. Domain averages are sums over ranks and, unlike the rest of a
run, can differ in the last bits between numbers of ranks; region integrals
for totals are summed in global cell order and do not.

Sources of the solution are explicit: a source that relaxes a quantity at rate
`k` (`-k (q - q_ref)`, or linear forcing with rate `A`) needs `k dt` below
about 2.5 with SSPRK3 and 2.8 with RK4 (the real-axis stability limits), and
in practice well below 1 to stay accurate; Mallard does not limit the time
step for them.

`[[write_data]]` can write the total sources per unit volume as
`SOURCE_RHO`, `SOURCE_RHOU_X`, ..., `SOURCE_RHOE` and `SOURCE_RHOY_<species>`.

```toml
# 50 W released in a sphere of radius 2 mm during the first 0.1 ms, exactly
[[sources]]
name = "igniter"
total = true
rhoE = "t < 1e-4 ? 50 : 0"
region = {type = "sphere", center = [0, 0, 0], radius = 2e-3, edge = 5e-4}

# Linear forcing of turbulence holding k = 0.125 (see examples/forced_isotropic_turbulence)
[[sources]]
rhou = ["10 * (0.125 - mean_ke / mean_rho) / (mean_ke / mean_rho) * rho * u_x", "...", "..."]
```

## Regions

Regions select parts of space in `[[sources]]` (explained with figures in
the [geometry guide](guide/geometry.md#regions)). A region is a table: a primitive shape, or a boolean of regions, each with
optional transforms, nested to any depth. Every region has a signed distance
(negative inside) that never exceeds the distance to its surface, which gives
smooth edges their width and lets cell volume fractions skip cells away from
the surface. Volume fractions are cell averages of the region's indicator,
sampled on each cell's triangles (2D) or tetrahedra (3D), subdivided where the
surface or the smooth edge crosses them: cells entirely inside get exactly 1.
In 2D, regions are the z = 0 slices of the 3D shapes (`z` components may be
left out). Lengths take units (`radius = "5 mm"`) and angles are in degrees
unless given a unit; `axis`, `normal`, `axes` and `scale` are plain numbers.

| `type` | Keys | Shape |
|---|---|---|
| `sphere` (2D: also `disk`, `circle`) | `center`, `radius` | |
| `ellipsoid` (`ellipse`) | `center`, `semi_axes` | semi-axes along x, y, z (orient with `rotate`) |
| `cylinder` | `center`, `axis` (default z in 2D), `radius`, `inner_radius` (default 0: solid), `length` (default infinite) | annular with `inner_radius`; centered on `center` along `axis` |
| `annulus` (2D) | `center`, `radius`, `inner_radius` | |
| `cone` (`frustum`) | `base` (center of the base), `axis` (toward the top), `length`, `base_radius`, `top_radius` (default 0) | |
| `box` (`rectangle`) | `center`, `size` (edge lengths) | axis-aligned before `rotate` |
| `torus` | `center`, `axis` (default z in 2D), `major_radius`, `minor_radius` | |
| `half_space` | `point` (default origin), `normal` | the side `(x - point) . normal < 0` |
| `polygon` (2D) | `vertices` = `[[x, y], ...]` | simple polygon |
| `implicit` | `f`, an expression in `x`, `y`, `z` | `f < 0`. Smooth edges are in units of `f` (exact when `f` is a distance, e.g. `"sqrt(x^2 + y^2) - 1"`) |
| `union`, `intersection` | `regions` = `[{...}, ...]` | |
| `difference` | `a` and `b` (or `regions`, the first minus the others) | `a` without `b` |

Any region, primitive or boolean, may also have, applied in this order:

| Key | Description |
|---|---|
| `scale` | A number, or one factor per axis; about `pivot` |
| `rotate` | 3D: `{axis = [...], angle = degrees}` or `{axes = [[...], [...], [...]]}` (the region's x, y, z axes in world coordinates, orthonormal and right-handed); 2D: an angle in degrees, counterclockwise; about `pivot` |
| `pivot` | Center of `scale` and `rotate` (default the origin) |
| `translate` | A displacement |

The outermost region may also have `edge`, the width of a smooth edge: the
indicator falls from 1 to 0 as the smoothstep `s^2 (3 - 2 s)` across the band
of that width centered on the surface (default 0, sharp), which keeps the
volume of a flat surface and changes that of a curved one by the band's second
moment (e.g. `pi R edge^2 / 5` for a sphere).

```toml
# A tilted pipe wall with a hole drilled through it
region = {type = "difference",
          a = {type = "cylinder", center = [0, 0, 0], axis = [1, 1, 0], radius = 0.3, inner_radius = 0.2, length = 1.0},
          b = {type = "cylinder", center = [0, 0, 0], axis = [0, 0, 1], radius = 0.05},
          edge = 0.01}

# A box rotated 30 degrees about z around its own center
region = {type = "box", center = [1, 0, 0], size = [0.4, 0.2, 0.2], pivot = [1, 0, 0], rotate = {axis = [0, 0, 1], angle = 30}}
```

Expressions in regions are those of every section:
[Expressions](guide/input_files.md#expressions).

## `[parallel]`

Used when Mallard runs on several MPI ranks (`mpirun -n N mallard -i input.toml`).

| Key | Description |
|---|---|
| `partitioner` | `multijagged` (default: cells split into equal slabs along x, each slab along y, and in 3D along z, with the number of pieces per axis that leaves the least area between the blocks of the mesh's bounding box; a box of cuboid cells gets cuboid parts), `hilbert` (cells split along a Hilbert curve of their centroids) or `graph` (dKaMinPar on the cell connectivity, minimizing the faces between ranks; needs a build with `Mallard_ENABLE_KAMINPAR`). Results are the same bit for bit |
| `halo_exchange` | `nccl` (NCCL operations on the GPU's stream: no host synchronization per exchange; default when built with `Mallard_ENABLE_NCCL`) or `mpi` (nonblocking MPI; the default otherwise). Results are the same bit for bit |

## `[output]`

| Key | Description |
|---|---|
| `check_interval` | Print a progress row every this many steps (default 1) |
| `run_log` | Machine-readable run log, one JSON record per line: a start record with the provenance, a progress record per progress row, file writes, warnings and errors, and an end record ([format](guide/output.md#the-run-log)). Appended to when the run starts from a restart file. `""` writes none. Default `"mallard.jsonl"` |
| `hash_mesh` | Record the SHA-256 of a mesh file (or directory) in the provenance; `false` skips reading it again for very large meshes. Default `true` |
| `async` | `true` (default) writes the `[[write_data]]` files on background threads while the run continues; `false` writes them before the run goes on. When a file is due, its values are copied out (on several ranks, regrouped where the file needs it) and the run goes on; a writer's next file first waits for its previous one. The files are the same as without `async`, byte for byte. VTKHDF and HDF5 output on several ranks stays synchronous: parallel HDF5 calls MPI, which only the main thread does. The run waits for every file before it ends, also when it stops on an error |
| `io_threads` | Threads that copy out each snapshot and compress its files (default 1) |

The run's summary reports the size of the files written and the time spent writing them (in the background with `async`); `output` in its wall time is what the run waited for.

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
| `format` | `vtkhdf` (default: VTK's HDF5 format, see below), `vtu` (with a `.pvd` series next to it), `restart`, or the deprecated `hdf5` (XDMF indexes over HDF5 files; removed in 0.10.0, `mallard input upgrade` rewrites it to `vtkhdf`). Builds without HDF5, or without parallel HDF5 on several ranks, default to `vtu` with a warning |
| `series` | (`vtkhdf`) `false` (default): one file per snapshot, `PREFIX_NNNNNN.vtkhdf`, indexed by `PREFIX.vtkhdf.series`; `true`: every snapshot in one file, `PREFIX.vtkhdf` (VTKHDF temporal data), which a restarted run continues after its snapshots up to the restart time |
| `interval` / `time_interval` | Write every this many steps / this much simulation time (exactly one). With `time_interval` the time step is shortened to land on each output time. |
| `variables` | (`vtkhdf`, `vtu`, `hdf5`) Any of `RHO`, `RHOU_X`, `RHOU_Y`, (3D) `RHOU_Z`, `RHOE`, `U_X`, `U_Y`, (3D) `U_Z`, `P`, `T`, `H`, `CFL`, the vectors `RHOU` and `U` (written with 3 components, zero z in 2D), `Q` (the Q-criterion: half the squared norm of the rotation rate minus that of the strain rate, with the hoop strain in axisymmetric runs) and `VORTICITY` (3D: the vector of `VORTICITY_X`, `VORTICITY_Y`, `VORTICITY_Z`; 2D: the scalar dv/dx - du/dy), with the velocity gradients of the `[integrals]` monitor (TENO reconstruction polynomials, else least squares), with TENO `TENO_SIGMA` (the troubled-cell indicator; stencil selection is active where it exceeds `troubled_threshold`), and for mixtures `Y_<species>`, `X_<species>` and `RHOY_<species>` (with `navier_stokes` also `MU`, `LAMBDA` and the diffusion coefficients `D_<species>`, and with `soret` the thermal diffusion coefficients `DT_<species>` [kg/(m s)]; with `[chemistry]` also `HRR`, `CHEM_COST` and the mass production rates `OMEGA_<species>` [kg/(m³ s)]; with `[radiation]` also `QRAD` [W/m³]), with `[[sources]]` (or `[source]` expressions) `SOURCE_<conservative>` and `SOURCE_RHOY_<species>`, the sources per unit volume, and `P_MAX`, the largest pressure of each cell at the end of any step so far (a numerical soot foil of detonation cells; restart files then carry it, so it continues across restarts), and the `[statistics]` averages `MEAN_<A>`, `COV_<A>_<B>` and `PHASE_<bin>_<A>`. A trailing `*` selects every variable with that prefix, e.g. `Y_*` |
| `geometry` | (`vtkhdf`, `vtu`) `all` (default) for the volume, or a boundary zone name to write that zone's faces with the values of their adjacent cells (e.g. wall pressure) |
| `compression` | `none` (default) or `zlib`, lossless: VTU arrays are compressed as ParaView's `vtkZLibDataCompressor` reads them (needs `Mallard_ENABLE_ZLIB`); VTKHDF and HDF5 datasets are chunked, byte-shuffled and deflated (HDF5's standard filters, which every HDF5 reader has); restart files become version 5 (6 with averaged spectra), each variable in byte-shuffled, deflated chunks (`Mallard_ENABLE_ZLIB` to write and to read them), and restart bitwise like uncompressed ones. Not for boundary-zone VTU output (`geometry`), which is ASCII |
| `relative_error` | (`vtkhdf`, `vtu`, `hdf5`, with `compression = "zlib"`) Error bound of lossy compression: each cell value is rounded to the fewest mantissa bits that keep it within this relative error of itself (\|x′ − x\| ≤ `relative_error` · \|x\|; zeros, subnormals, infinities and NaNs are kept), and the zeroed trailing bits compress well. The mesh stays exact. Restart files are always lossless |
| `chunk_cells` | (`restart`, with `compression = "zlib"`) Cells per compressed chunk (default 65536); a rank reading some cells decompresses only the chunks that hold them |

`format = "vtkhdf"` (builds with `-DMallard_ENABLE_HDF5=ON`; several ranks need
parallel HDF5) writes [VTKHDF](https://docs.vtk.org/en/latest/vtk_file_formats/vtkhdf_file_format/index.html)
files, which ParaView 5.12 and later open directly. Every rank writes its part
of one file with collective I/O, each array in global order (cell i is global
cell i, the order of the mesh and restart files), so the files do not depend on
the number of ranks. The volume is an `UnstructuredGrid`: `Points`, `Connectivity`,
`Offsets` and `Types` (VTK cell types), then `CellData/<name>` (one value per
cell, or three for vectors) and `FieldData/TIME`. Cells touching a curved wall
(`[[mesh.curved]]` or high-order elements of the mesh file) are quadratic
Lagrange cells (VTK types 69 to 73, and 27 for pyramids), their extra nodes
on the curved geometry, after the mesh's nodes. A boundary zone (`geometry`)
is a `PolyData` of its faces (`Lines` in 2D, `Polygons` in 3D). With
`series = true` the mesh is stored once and VTKHDF's `Steps` group holds every
snapshot's time (`Steps/Values`) and where its cell values start
(`Steps/CellDataOffsets/<name>`). Nothing else is added to the VTKHDF layout:
the units and descriptions are HDF5 attributes of each dataset (`units`,
`long_name`), and the root group carries the `provenance` and the format
`version` as HDF5 attributes, which VTK ignores.
`tools/mallard/vtkhdf.py` reads the files.

`format = "hdf5"` (deprecated; builds with `-DMallard_ENABLE_HDF5=ON`; several ranks need
parallel HDF5) scales to large runs: every rank writes its part of one file per
snapshot with collective I/O instead of a VTU piece of its own. It writes

- `PREFIX_mesh.h5`, once: `nodes/coordinates` (nodes × dimension),
  `cells/topology`, an XDMF `Mixed` topology (each cell's XDMF type, then its
  nodes in VTK order: triangle 4, quadrilateral 5, tetrahedron 6, pyramid 7,
  wedge 8, hexahedron 9) and `cells/offsets`, where each cell's record starts
  (one more entry than cells);
- `PREFIX_NNNNNN.h5` per snapshot: the attributes `step`, `time` and `mesh`
  (the mesh file's name), and `fields/<name>` for each variable (one value per cell,
  or three for vectors);
- `PREFIX_NNNNNN.xmf` per snapshot and `PREFIX.xmf`, the time series, which
  ParaView opens (XDMF reader).

With `compression = "zlib"` the datasets are stored in chunks of 65536 rows.
Row i of every cell array is global cell i, the cell order of the mesh file and
of restart files, so the files do not depend on the number of ranks.
Snapshots are about half the size of VTU pieces (the mesh is not repeated) and
two files instead of one per rank. A shared file pays off on parallel file
systems; on one node's local disk, which serializes writes to a file, VTU
pieces write faster beyond a few ranks (2.1M hexahedra, 16 ranks: 115 ms per
HDF5 snapshot against 72 ms, 1 rank: 140 ms against 1.1 s). VTU output remains
the simpler choice for small runs. `tools/mallard/h5.py` reads the files.

## `mallard reactor`

`mallard reactor -i input.toml` integrates one adiabatic, constant-volume
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
| `[benchmark]` | Instead of the run, a chemistry benchmark: states sampled along this reactor's trajectory, replicated over `cells` cells and advanced by the solver's chemistry kernels over each splitting step of `dt`, with results in `output` (default `benchmark.csv`); see `benchmarks/README.md` |

It prints the ignition delay, the time of the maximum of `dT/dt`, once
`dT/dt` has fallen below half that maximum by `end_time` (the runaway is over,
however small the temperature rise of a lean mixture), and otherwise none.
Example: `examples/h2_ignition`.
