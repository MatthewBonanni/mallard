# Initialization and restarts

`[initialize]` sets the state at the start of a run: uniform, from
expressions, or from a restart file written by an earlier run or by a script.

## Uniform state

```toml
[initialize]
type = "constant"     # the default
u = [0.2, 0.0]
p = 0.7142857142857143
T = 1.0
# X = { O2 = 1.0, N2 = 3.76 }   gas mixtures: the composition
```

*Used in:* [`wedge`](https://matthewbonanni.github.io/mallard/docs/examples/#wedge), [`sphere_mach3`](https://matthewbonanni.github.io/mallard/docs/examples/#sphere-mach3), [`mallard`](https://matthewbonanni.github.io/mallard/docs/examples/#mallard), [`poiseuille_pipe`](https://matthewbonanni.github.io/mallard/docs/examples/#poiseuille-pipe), [`source_regions`](https://matthewbonanni.github.io/mallard/docs/examples/#source-regions) and 2 more.

## Expressions

```toml
# examples/riemann_2d: the four quadrants of configuration 3
[initialize]
type = "analytical"
rho = """\
    x >= 0.8 ? (y >= 0.8 ? 1.5 : 0.532258064516129) \
             : (y >= 0.8 ? 0.532258064516129 : 0.137992831541219)"""
u = ["x >= 0.8 ? 0.0 : 1.206045378311055",
     "y >= 0.8 ? 0.0 : 1.206045378311055"]
p = """\
    x >= 0.8 ? (y >= 0.8 ? 1.5 : 0.3) \
             : (y >= 0.8 ? 0.3 : 0.029032258064516)"""
```

*Used in:* [`sod`](https://matthewbonanni.github.io/mallard/docs/examples/#sod), [`shu_osher`](https://matthewbonanni.github.io/mallard/docs/examples/#shu-osher), [`riemann_2d`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d), [`riemann_2d_quads`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d-quads), [`double_mach`](https://matthewbonanni.github.io/mallard/docs/examples/#double-mach) and 15 more.

| Key | Default | Meaning |
|---|---|---|
| `type` | `"constant"` | `constant`, `analytical`, `restart`, or `remap` (a restart file of another mesh) |
| `u` | required | `constant`: numbers; `analytical`: one expression per component |
| `p`, `T` (`constant`) | required | Pressure and temperature |
| `rho`, `p`, `T` (`analytical`) | two of the three | Expressions of x, y, z |
| `X` or `Y` | required for mixtures | Composition by species: numbers, or expressions (`analytical`) |
| `balance` | none | Mixtures, `analytical`: the species that takes `1 - sum` of the others |
| `n_subdivisions` | 4 (2D), 1 (3D) | Resolution of the cell averages of `analytical` states |
| `file` | required (`restart`) | Restart file to start from |

Analytical states are integrated over each cell, not sampled at its center,
so discontinuities that cut a cell give the right average (and high-order runs
start from true cell averages). In 2D each cell's triangles are split into
`n_subdivisions`² sub-triangles; in 3D each tetrahedron of a cell into
`n_subdivisions`³ sub-tetrahedra with a degree-5 rule. Raise it for sharp
features smaller than a cell.

![A triangle with one and with sixteen sub-triangles for the cell average of an analytical state](../images/guide/subdivisions-light.svg#gh-light-mode-only)
![A triangle with one and with sixteen sub-triangles for the cell average of an analytical state](../images/guide/subdivisions-dark.svg#gh-dark-mode-only)

Expressions use [exprtk](https://www.partow.net/programming/exprtk/) syntax
(see [Input files](input_files.md#numbers-vectors-and-expressions)). Pitfall:
TOML requires the expressions to be strings, also for constants: `u =
["0.0", "0.0"]` with `analytical`.

## Restarts

```toml
[initialize]
type = "restart"
file = "solut/restart_010000.restart"

[[write_data]]                 # and keep writing them
prefix = "solut/restart"
format = "restart"
interval = 10000
```

*Used in:* [`isotropic_turbulence`](https://matthewbonanni.github.io/mallard/docs/examples/#isotropic-turbulence), [`channel_retau180`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-retau180), [`channel_inflow_retau180`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-inflow-retau180), [`cbc_les`](https://matthewbonanni.github.io/mallard/docs/examples/#cbc-les), [`channel_les`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-les) and 13 more.

`format = "restart"` in `[[write_data]]` writes the conservative variables
(and whatever else the run needs to continue: chemistry sub-steps, the soot
foil `P_MAX`, `[statistics]` averages, characteristic boundary states) in
global cell order. A restarted run continues the uninterrupted one **bit for
bit**, also on a different number of MPI ranks, or serially from a parallel
run. The step and time continue from the file; `t_stop` and `n_steps` refer to
the total.

- `[statistics] reset = true` starts the averages afresh instead of
  continuing them.
- Probes and force monitors append to their files on a restart.
- Restart files of every earlier release are read, back to 0.1, and files from a newer Mallard are rejected clearly ([compatibility](output.md#file-formats-and-compatibility)).
- `compression = "zlib"` writes compressed restart files (version 5, losslessly: they restart bit for bit), about a third smaller for smooth flows; reading them needs a build with `Mallard_ENABLE_ZLIB`, and the Python tools in `tools/` read only uncompressed ones.

### Initial states from scripts

Many examples start from a restart file that a script in `tools/` computes:
a Cantera flame, a ZND detonation, a turbulent field. These are the patterns
to copy for your own cases:

| Script | Initial state |
|---|---|
| `flame_restart.py` | Premixed flame in its own frame from a Cantera solution (1D, or a wrinkled 2D front) |
| `znd_restart.py` | ZND detonation profile, with pockets of fresh gas to trigger cells |
| `channel_init.py` | Turbulent channel: mean profile plus a divergence-free perturbation |
| `isotropic_turbulence_restart.py` | Isotropic turbulence with a given spectrum |
| `autoignition_restart.py` | Stratified temperature and velocity fields |
| `interpolate_restart.py` | A restart file interpolated onto another Gmsh mesh (nearest cells; `type = "remap"` below is conservative and high order) |
| `tile_restart.py` | A restart file of a periodic box tiled into a larger box |

A restart file is binary, little-endian: the 16 bytes `MALLARD-RESTART\0`,
then `uint32` version (2; [all versions](output.md#file-formats-and-compatibility)) and real size (8), `uint64` cell count, variable
count and step, `float64` time, each variable name as a `uint32` length and its
characters, and finally the values, variable after variable, each over all
cells in the mesh's global order. The variables are `RHO`, `RHOU_X`, `RHOU_Y`
(`RHOU_Z`), `RHOE` and, for gas mixtures, `RHOY_<species>` for every species
and `T_SEED` (a temperature guess); `channel_init.py` has a short writer.

## Remap from another mesh

```toml
[initialize]
type = "remap"
file = "coarse/run_000100.restart"   # written on another mesh

[initialize.mesh]                   # that mesh: the source run's [mesh] table
type = "cartesian"
Nx = 100
Ny = 100
```

*Used in:* [`riemann_2d_quads`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d-quads), [`taylor_green_3d`](https://matthewbonanni.github.io/mallard/docs/examples/#taylor-green-3d).

`type = "remap"` starts from a restart file written on a **different mesh**: a
finer one (grid convergence, spinning up an LES from a coarse run), another
cell type, any partition and rank count on either side. Give the source run's
mesh as `[initialize.mesh]`:

```toml
# examples/riemann_2d_quads/remap.toml: 10,000 quads at t = 0.5 onto 131,072 triangles
[initialize]
type = "remap"
file = "coarse/riemann_000218.restart"
degree = 3              # of the source polynomials, 0-5 (default 2)
# limiter = "none"      # for smooth fields; default "bounds"

[initialize.mesh]       # the source run's [mesh] table
type = "cartesian"
Nx = 100
Ny = 100
```

| Key | Default | Meaning |
|---|---|---|
| `file` | required | Restart file of the source run |
| `[initialize.mesh]` | required | The source run's `[mesh]` table |
| `[[initialize.periodic]]` | the run's `[[periodic]]` | The source mesh's periodic zone pairs |
| `degree` | 2 | Degree of each source cell's least-squares polynomial |
| `limiter` | `"bounds"` | `bounds`: no new extrema, no negative partial densities; `none`: full order |

Each source cell's polynomial is integrated **exactly** over its intersection
with every target cell, so the integrals of all conserved variables are kept
to rounding and polynomial fields up to `degree` are remapped exactly. Every
variable of the file is remapped: the flow, species, temperature seeds and
`[statistics]` averages. The run continues from the file's step and time with
fresh output series; the log reports how closely the integrals were kept.

![A target triangle cut into pieces by the source quadrilaterals it overlaps](../images/guide/remap-supermesh-light.svg#gh-light-mode-only)
![A target triangle cut into pieces by the source quadrilaterals it overlaps](../images/guide/remap-supermesh-dark.svg#gh-dark-mode-only)

![Density on quadrilaterals and remapped to triangles, and along a line](../images/guide/remap-fields-light.svg#gh-light-mode-only)
![Density on quadrilaterals and remapped to triangles, and along a line](../images/guide/remap-fields-dark.svg#gh-dark-mode-only)

Pitfalls:

- `[initialize.mesh]` must be the mesh the restart file was written on (its
  cell count is checked); a mesh file's `transform` applies to it as usual.
- The source mesh may be periodic by translation only; a source with a
  rotational pair (`[[periodic]] angle`, or an `annulus` periodic in `theta`)
  is an error.
- Where the meshes cover different domains (a curved wall discretized
  differently, a shorter box), target cells average over the covered part and
  cells outside take the nearest source cell; the log counts them, and
  conservation then holds only over the common part.
- The default `bounds` limiter flattens smooth extrema slightly; use
  `limiter = "none"` for smooth flows such as the Taylor-Green vortex.
- Restarting on the **same** mesh is `type = "restart"`, which is bitwise exact.

Method, accuracy and cost: [docs/design/remap.md](../design/remap.md).

## See also

- [Input reference: `[initialize]`](../input.md#initialize)
- [Output and diagnostics](output.md): writing restart files and snapshots
