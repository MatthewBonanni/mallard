# Physics

`[physics]` chooses the equations (Euler or Navier–Stokes), the gas (a single
calorically perfect gas, or a thermally perfect mixture of species) and the
geometry (planar or axisymmetric).

## Euler and Navier–Stokes

```toml
[physics]
type = "navier_stokes"   # or "euler" (the default)
gamma = 1.4
p_ref = 1.0              # any state of the gas: sets R = p_ref / (rho_ref T_ref)
T_ref = 1.0
rho_ref = 1.0
mu = 1.0e-3              # navier_stokes: dynamic viscosity
Pr = 0.72
```

*Used in:* [`sod`](https://matthewbonanni.github.io/mallard/docs/examples/#sod), [`shu_osher`](https://matthewbonanni.github.io/mallard/docs/examples/#shu-osher), [`riemann_2d`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d), [`riemann_2d_quads`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d-quads), [`double_mach`](https://matthewbonanni.github.io/mallard/docs/examples/#double-mach) and 35 more.

| Key | Default | Meaning |
|---|---|---|
| `type` | `"euler"` | `euler` (inviscid) or `navier_stokes` (viscous, heat-conducting) |
| `gamma` | required | Ratio of specific heats |
| `p_ref`, `T_ref`, `rho_ref` | required | A reference state; only the gas constant `R = p_ref / (rho_ref T_ref)` is taken from it |
| `mu` | required (`navier_stokes`) | Dynamic viscosity, or its value at `T_mu_ref` for the temperature-dependent laws |
| `Pr` | 0.72 | Prandtl number; the conductivity is `cp mu / Pr` |
| `viscosity_model` | `"constant"` | `constant`, `sutherland` or `power_law` |
| `T_mu_ref` | 273.15 (`sutherland`); required (`power_law`) | Temperature at which the viscosity is `mu` |
| `sutherland_S` | 110.4 | Sutherland temperature |
| `viscosity_exponent` | required (`power_law`) | `mu (T / T_mu_ref)^viscosity_exponent` |
| `gas` | `"perfect"` | `perfect` or `mixture` ([Thermochemistry](thermochemistry.md)) |
| `axisymmetric` | `false` | 2D only: flows symmetric about the x axis, [below](#axisymmetric-flows) |

When to use which:

- **Euler** for shock-dominated flows where viscosity does not matter at the
  scales of the mesh (shock tubes, blast waves, detonation fronts); the
  `wall_*` conditions are then slip walls.
- **Navier–Stokes** for boundary layers, wakes and turbulence: walls are
  no-slip, the time step includes the viscous limit, and the
  [large-eddy simulation](les.md) models add to the molecular viscosity.

Mallard is scale-free for a single gas: the reference state only fixes `R`,
so the common nondimensionalization (free-stream density and sound speed 1,
or `p_ref = T_ref = rho_ref = 1`) works as it is. The `cylinder` example, for
instance, takes the cylinder diameter and free-stream sound speed as units, so
`U = M = 0.2` and `mu = U D / Re`.

### Viscosity laws

```toml
[physics]
type = "navier_stokes"
viscosity_model = "sutherland"
mu = 1.716e-5
```

*Used in:* [`isotropic_turbulence`](https://matthewbonanni.github.io/mallard/docs/examples/#isotropic-turbulence), [`channel_wmles`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-wmles), [`volvo_bluff_body`](https://matthewbonanni.github.io/mallard/docs/examples/#volvo-bluff-body).

![Viscosity against temperature for the constant, Sutherland and power laws](../images/guide/viscosity-light.svg#gh-light-mode-only)
![Viscosity against temperature for the constant, Sutherland and power laws](../images/guide/viscosity-dark.svg#gh-dark-mode-only)

```toml
viscosity_model = "sutherland"   # air: mu(T) = mu (T / T_mu_ref)^1.5 (T_mu_ref + S) / (T + S)
mu = 1.716e-5                    # at T_mu_ref = 273.15 K
```

Use Sutherland's law (or a power law) when the temperature varies by more
than tens of percent: shock-heated boundary layers, the viscous shock tube,
hot wakes. A constant viscosity is right for nearly isothermal flows and for
comparisons with incompressible references.

## Gas mixtures

```toml
[physics]
type = "navier_stokes"
gas = "mixture"
mechanism = "../../mechanisms/h2o2.yaml"
```

*Used in:* [`volvo_bluff_body`](https://matthewbonanni.github.io/mallard/docs/examples/#volvo-bluff-body), [`flame_turbulence`](https://matthewbonanni.github.io/mallard/docs/examples/#flame-turbulence), [`h2_ignition`](https://matthewbonanni.github.io/mallard/docs/examples/#h2-ignition), [`reactive_shock_tube`](https://matthewbonanni.github.io/mallard/docs/examples/#reactive-shock-tube), [`detonation_1d`](https://matthewbonanni.github.io/mallard/docs/examples/#detonation-1d) and 11 more.

`gas = "mixture"` replaces `gamma` and the reference state with the species
of a Cantera mechanism: thermally perfect thermodynamics, mixture-averaged or
simpler transport, and finite-rate chemistry with `[chemistry]`. See
[Thermochemistry](thermochemistry.md).

## Axisymmetric flows

```toml
[physics]
type = "navier_stokes"
axisymmetric = true
# ...

[[boundaries]]
name = "axis"            # every boundary face on y = 0
type = "symmetry"
```

*Used in:* [`poiseuille_pipe`](https://matthewbonanni.github.io/mallard/docs/examples/#poiseuille-pipe), [`noh_axisymmetric`](https://matthewbonanni.github.io/mallard/docs/examples/#noh-axisymmetric), [`sedov_axisymmetric`](https://matthewbonanni.github.io/mallard/docs/examples/#sedov-axisymmetric), [`sphere_axisymmetric`](https://matthewbonanni.github.io/mallard/docs/examples/#sphere-axisymmetric).

With `axisymmetric = true` (2D builds), the mesh is the meridian half-plane of
a body of revolution: x is the axis and y the radius, so every node needs
`y >= 0`. Each cell stands for the solid it sweeps around the axis; there is
no swirl.

![The 2D mesh as the meridian half-plane of a body of revolution about the x axis](../images/guide/axisymmetric-light.svg#gh-light-mode-only)
![The 2D mesh as the meridian half-plane of a body of revolution about the x axis](../images/guide/axisymmetric-dark.svg#gh-dark-mode-only)

![The meridian half-plane of the axisymmetric sphere example with its zones](../images/guide/domain_sphere_axisymmetric-light.svg#gh-light-mode-only)
![The meridian half-plane of the axisymmetric sphere example with its zones](../images/guide/domain_sphere_axisymmetric-dark.svg#gh-dark-mode-only)

*`examples/sphere_axisymmetric`: flow past a sphere at Re = 100 and 150 on a
half O-grid (`make_cylinder_mesh.py --half`, drawn coarser than it runs).*

- Boundary faces on the axis must be `symmetry`; nothing crosses them.
- `u` is `[u_x, u_r]`. Cell values are r-weighted averages, and analytical
  initial conditions are averaged that way.
- Fluxes, sources, forces (`[[forces]]`), `[integrals]` and conserved totals
  are **per radian**: multiply a force by `2 pi` for the force on the body;
  only its axial component is meaningful.
- Every reconstruction, Riemann solver, viscous and mixture model, chemistry
  and MPI work as in planar runs, at design order up to the axis.
- Not available: periodicity in y, curved boundaries, and `[les]` (planar
  runs only).

Examples, each checked against an exact or reference solution:
[`poiseuille_pipe`](https://matthewbonanni.github.io/mallard/docs/examples/#poiseuille-pipe),
[`noh_axisymmetric`](https://matthewbonanni.github.io/mallard/docs/examples/#noh-axisymmetric),
[`sedov_axisymmetric`](https://matthewbonanni.github.io/mallard/docs/examples/#sedov-axisymmetric) and
[`sphere_axisymmetric`](https://matthewbonanni.github.io/mallard/docs/examples/#sphere-axisymmetric).

## See also

- [Input reference: `[physics]`](../input.md#physics)
- [Sources](sources.md): gravity, body forces and constant mass flow
- Design notes: [axisymmetric flows](../design/axisymmetric.md)
