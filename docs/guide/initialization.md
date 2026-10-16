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

| Key | Default | Meaning |
|---|---|---|
| `type` | `"constant"` | `constant`, `analytical` or `restart` |
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
- Restart files of Mallard 0.3 and earlier (version 1) are still read.

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
| `interpolate_restart.py` | A restart file interpolated onto another mesh of the same domain |
| `tile_restart.py` | A restart file of a periodic box tiled into a larger box |

A restart file is binary, little-endian: the 16 bytes `MALLARD-RESTART\0`,
then `uint32` version (2) and real size (8), `uint64` cell count, variable
count and step, `float64` time, each variable name as a `uint32` length and its
characters, and finally the values, variable after variable, each over all
cells in the mesh's global order. The variables are `RHO`, `RHOU_X`, `RHOU_Y`
(`RHOU_Z`), `RHOE` and, for gas mixtures, `RHOY_<species>` for every species
and `T_SEED` (a temperature guess); `channel_init.py` has a short writer.

## See also

- [Input reference: `[initialize]`](../input.md#initialize)
- [Output and diagnostics](output.md): writing restart files and snapshots
