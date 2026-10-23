# Sources and sponges

Source terms add to the right-hand side per unit volume: gravity, body forces
that hold a mass flow, arbitrary expressions, and sponge layers that damp the
solution toward a reference state.

## `[source]`

```toml
[source]
gravity = [0.0, -9.81]          # adds rho g to momentum, rho u.g to energy
```

*Used in:* [`poiseuille_pipe`](https://matthewbonanni.github.io/mallard/docs/examples/#poiseuille-pipe), [`channel_retau180`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-retau180), [`channel_les`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-les), [`channel_wmles`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-wmles).

| Key | Default | Meaning |
|---|---|---|
| `gravity` | none | Gravitational acceleration vector |
| `mass_flow` | none | Hold the volume average of `rho u` at this vector with a uniform body force (periodic channels and pipes) |
| `rho`, `rhou`, `rhoE` | `"0"` | Mass, momentum (one expression per component) and energy sources, expressions of x, y, z, t |
| `time_dependent` | `false` | Re-evaluate the expressions at every Runge–Kutta stage (on the host: costly on large meshes); otherwise once at the start |

### Constant mass flow

```toml
# examples/channel_retau180: bulk density and velocity 1 along x
[source]
mass_flow = [1.0, 0.0, 0.0]
```

*Used in:* [`channel_retau180`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-retau180), [`channel_les`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-les), [`channel_wmles`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-wmles).

A periodic channel has no inlet to fix its flow rate. `mass_flow` adds the
uniform body force that keeps the volume-averaged momentum at the given value,
correcting every stage so that a flow started at the target stays there to
rounding; the energy gains the force's work.

### Gravity

```toml
[source]
gravity = [0.0, -9.81]
```

*Used in:* [`poiseuille_pipe`](https://matthewbonanni.github.io/mallard/docs/examples/#poiseuille-pipe).

Pitfall: the scheme is not exactly well balanced. A hydrostatic atmosphere
carries spurious velocities of about 1e-4 of the sound speed on a 32 x 32
mesh, which vanish at second order under refinement; wall and symmetry ghost
states continue the hydrostatic pressure gradient. The `poiseuille_pipe`
example drives a pipe flow with `gravity` as a body force.

## Sponge layers

```toml
[[sponges]]
strength = "x > 4 ? 20 * ((x - 4) / 1)^2 : 0"   # 1/s; a gradual ramp reflects less than a step
u = [1.0, 0.0]
p = 1.0
T = 1.0
```

*Used in:* [`channel_inflow_retau180`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-inflow-retau180).

`[[sponges]]` add `-strength (U - U_ref)` to the conservative variables (and
species), damping the solution toward a reference state where the strength is
nonzero. They absorb eddies and waves before an outlet, where characteristic
conditions alone may reflect them or let reversed flow in.

![Sponge strength rising quadratically over the last part of the domain](../images/guide/sponge-light.svg#gh-light-mode-only)
![Sponge strength rising quadratically over the last part of the domain](../images/guide/sponge-dark.svg#gh-dark-mode-only)

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

- [Input reference: `[source]`](../input.md#source), [`[[sponges]]`](../input.md#sponges)
- [Boundary conditions](boundary_conditions.md#characteristic-boundaries)
- Design notes: [characteristic boundaries and sponges](../design/nscbc.md)
