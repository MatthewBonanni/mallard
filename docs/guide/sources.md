# Sources and sponges

Source terms add to the right-hand side per unit volume: gravity, body forces
that hold a mass flow, expressions of position, time and the solution
(restricted to regions of space, or given as exact totals over them), and
sponge layers that damp the solution toward a reference state.

## `[source]`

```toml
[source]
gravity = [0.0, -9.81]          # adds rho g to momentum, rho u.g to energy
```

| Key | Default | Meaning |
|---|---|---|
| `gravity` | none | Gravitational acceleration vector |
| `mass_flow` | none | Hold the volume average of `rho u` at this vector with a uniform body force (periodic channels and pipes) |
| `rho`, `rhou`, `rhoE` | none | Mass, momentum (one expression per component) and energy sources everywhere: a shorthand for one `[[sources]]` entry without a region |

### Constant mass flow

A periodic channel has no inlet to fix its flow rate. `mass_flow` adds the
uniform body force that keeps the volume-averaged momentum at the given value,
correcting every stage so that a flow started at the target stays there to
rounding; the energy gains the force's work.

```toml
# examples/channel_retau180: bulk density and velocity 1 along x
[source]
mass_flow = [1.0, 0.0, 0.0]
```

### Gravity

Pitfall: the scheme is not exactly well balanced. A hydrostatic atmosphere
carries spurious velocities of about 1e-4 of the sound speed on a 32 x 32
mesh, which vanish at second order under refinement; wall and symmetry ghost
states continue the hydrostatic pressure gradient. The `poiseuille_pipe`
example drives a pipe flow with `gravity` as a body force.

## `[[sources]]`

Any number of sources, each with its own expressions, optional region and
time dependence; they add up. Expressions are compiled once and evaluated on
the GPU: those of position alone once, those of `t` or of the solution at
every Runge–Kutta stage, at little cost (the step remains one CUDA graph).

```toml
[[sources]]
name = "heater"
rhoE = "1e6 * exp(-((t - 1e-3) / 2e-4)^2)"     # W/m^3
region = {type = "cylinder", center = [0, 0, 0], axis = [0, 0, 1], radius = 0.01, length = 0.05}
```

| Key | Default | Meaning |
|---|---|---|
| `name` | `sources[i]` | Name in the run log |
| `region` | everywhere | Where the source acts (below); values are multiplied by each cell's volume fraction |
| `total` | `false` | Values are integrated rates over the region (kg/s, N, W) instead of densities; they may depend on `t` only |
| `rho`, `rhou`, `rhoE` | none | Mass, momentum (one per component) and energy sources |
| `rhoY` | none | Mixtures: species sources by name, `{H2 = "1e-3"}`; they also add to the mass |

Expressions read `x`, `y`, `z`, `t`, the local state `rho`, `u_x`, `u_y`,
`u_z`, `p`, `T` (and `Y_<species>`, `X_<species>` in mixtures), and domain
averages `mean_rho`, `mean_rhou_x` (`_y`, `_z`), `mean_rhoE`, `mean_ke`. Write
`SOURCE_RHOE` (and `SOURCE_RHO`, `SOURCE_RHOU_X`, ...) to see where sources
act.

### Regions

A region is a primitive shape — sphere, ellipsoid, cylinder (solid or
annular), cone or frustum, box, torus, half-space, any implicit `f(x, y, z) <
0`, and in 2D disk, annulus, ellipse, rectangle and polygon — or a union,
intersection or difference of regions, nested freely, each scaled, rotated and
translated. Shapes need not follow the mesh: each cell gets its volume
fraction, computed by subdividing its simplices where the surface crosses
them, so a coarse mesh sees the right amount of the shape instead of a
staircase of whole cells.

![Cell volume fractions of each primitive over a coarse triangle mesh, with the exact outline](../images/guide/region-primitives-light.svg#gh-light-mode-only)
![Cell volume fractions of each primitive over a coarse triangle mesh, with the exact outline](../images/guide/region-primitives-dark.svg#gh-dark-mode-only)

```toml
# A disk with a rotated slot cut out of it
region = {type = "difference",
          a = {type = "disk", center = [0.5, 0.5], radius = 0.38},
          b = {type = "rectangle", center = [0.5, 0.5], size = [0.5, 0.16], rotate = 45, pivot = [0.5, 0.5]}}
```

![Volume fractions of a difference, a union, an intersection and a smooth-edged disk](../images/guide/region-booleans-light.svg#gh-light-mode-only)
![Volume fractions of a difference, a union, an intersection and a smooth-edged disk](../images/guide/region-booleans-dark.svg#gh-dark-mode-only)

In 3D the same tables describe solids; transforms take an axis and an angle,
or the region's own axes:

```toml
region = {type = "box", center = [0, 0, 0], size = [0.65, 0.35, 0.22],
          rotate = {axis = [1, 2, 3], angle = 35}, translate = [0.5, 0.5, 0.5]}
```

![Volume fractions in a tetrahedral mesh of a tilted annular cylinder, a rotated box and a drilled sphere, cut open](../images/guide/regions-3d-light.svg#gh-light-mode-only)
![Volume fractions in a tetrahedral mesh of a tilted annular cylinder, a rotated box and a drilled sphere, cut open](../images/guide/regions-3d-dark.svg#gh-dark-mode-only)

`edge` (on the outermost region) smooths the boundary over a band of that
width, so that a source switches on gradually, which avoids exciting
grid-scale waves:

![Volume fraction against radius for a sharp and a smooth-edged disk, with the smoothstep](../images/guide/region-smooth-edge-light.svg#gh-light-mode-only)
![Volume fraction against radius for a sharp and a smooth-edged disk, with the smoothstep](../images/guide/region-smooth-edge-dark.svg#gh-dark-mode-only)

### Exact totals

With `total = true` the value is the integral over the region, distributed by
the volume fractions and normalized by their integral over the mesh, so the
mesh receives exactly that rate (to rounding), however coarse or unstructured
it is — e.g. a 50 W igniter stays a 50 W igniter under refinement:

```toml
[[sources]]
name = "igniter"
total = true
rhoE = "t < 1e-4 ? 50 : 0"     # W
region = {type = "sphere", center = [0, 0, 0], radius = 2e-3, edge = 5e-4}
```

In 2D totals are per unit depth, in axisymmetric runs per radian.

### Sources of the solution

Sources may read the local state, e.g. a relaxation `-k (rho - rho_ref)`, a
heat source that switches on with temperature (`T > 1500 ? q : 0`), or linear
forcing of turbulence, `A rho u` with the work `A rho |u|^2` in the energy.
Domain averages close the loop: `examples/forced_isotropic_turbulence` holds
the kinetic energy at its target with
`A = 10 (0.125 - k) / k`, `k = mean_ke / mean_rho`.

Pitfalls:

- These sources are explicit. A relaxation at rate `k` needs `k dt` below about
  2.5 (SSPRK3; 2.8 with RK4), and well below 1 to be accurate; the time step is
  not limited for you.
- A momentum source changes the kinetic energy: add its work `f . u` to `rhoE`
  if the energy should follow.
- Mixtures take mass through `rhoY` only; give the enthalpy of injected mass in
  `rhoE` yourself.
- Domain averages cost a global sum per stage and keep steps out of CUDA graphs.

Example: `examples/source_regions` heats a ring with an exact Gaussian pulse
and pushes the gas with a tilted fan.

## Sponge layers

`[[sponges]]` add `-strength (U - U_ref)` to the conservative variables (and
species), damping the solution toward a reference state where the strength is
nonzero. They absorb eddies and waves before an outlet, where characteristic
conditions alone may reflect them or let reversed flow in.

![Sponge strength rising quadratically over the last part of the domain](../images/guide/sponge-light.svg#gh-light-mode-only)
![Sponge strength rising quadratically over the last part of the domain](../images/guide/sponge-dark.svg#gh-dark-mode-only)

```toml
[[sponges]]
strength = "x > 4 ? 20 * ((x - 4) / 1)^2 : 0"   # 1/s; a gradual ramp reflects less than a step
u = [1.0, 0.0]
p = 1.0
T = 1.0
```

| Key | Default | Meaning |
|---|---|---|
| `strength` | required | Damping rate [1/s], a number or an expression of x, y, z; zero outside the layer |
| `u`, `p`, `T` | required | Reference state, numbers or expressions of x, y, z |
| `X` or `Y`, `balance` | — | Mixtures: reference composition |

Several sponges add up. Strength and reference are evaluated once, at cell
centroids. The source is explicit and caps the time step at
`1 / max(strength)`. Example: `channel_inflow_retau180`, a sponge over the last
fifth of the channel in front of its `nscbc_outlet`.

## See also

- [Input reference: `[source]`](../input.md#source), [`[[sources]]`](../input.md#sources), [regions](../input.md#regions), [`[[sponges]]`](../input.md#sponges)
- [Boundary conditions](boundary_conditions.md#characteristic-boundaries)
- Design notes: [characteristic boundaries and sponges](../design/nscbc.md)
