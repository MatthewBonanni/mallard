# Boundary conditions

Every boundary face gets its condition from a `[[boundaries]]` entry naming its
zone (see [Meshes](meshes.md) for where zones come from). Conditions are
imposed weakly: each one supplies an exterior state, and the Riemann solver
computes the face flux from it and the reconstructed interior state.

![A boundary face between the boundary cell's reconstructed state and the exterior state of its condition](../images/guide/boundary_state-light.svg#gh-light-mode-only)
![A boundary face between the boundary cell's reconstructed state and the exterior state of its condition](../images/guide/boundary_state-dark.svg#gh-dark-mode-only)

```toml
[[boundaries]]
name = "left"           # the zone
type = "upt"            # the condition
u = [1.0, 0.0]
p = 1.0
T = 1.0

[[boundaries]]
name = "right"
type = "p_out"
p = 1.0
```

Rules:

- Each boundary face must be assigned exactly once. A zone the mesh does not
  have is an error that lists the mesh's zones, and so are a face assigned
  twice and an entry that selects no faces.
- Periodic zones are interior faces and take no entry; a fully periodic mesh
  needs no `[[boundaries]]`.
- `where = "<expression in x, y, z>"` splits a zone between entries: each takes
  the faces whose centers satisfy its expression.

![The double Mach reflection: the bottom zone split at x = 1/6 between an inflow and a symmetry condition](../images/guide/domain_double_mach-light.svg#gh-light-mode-only)
![The double Mach reflection: the bottom zone split at x = 1/6 between an inflow and a symmetry condition](../images/guide/domain_double_mach-dark.svg#gh-dark-mode-only)

*`examples/double_mach`: the `bottom` zone is post-shock inflow for x < 1/6
and a reflecting wall beyond (`where`), and `top` follows the exact moving
shock (`dirichlet` with expressions of `t`).*

```toml
[[boundaries]]
name = "bottom"
type = "dirichlet"
where = "x < 1/6"
rho = "8.0"
u = ["7.1447096811216", "-4.125"]
p = "116.5"

[[boundaries]]
name = "bottom"
type = "symmetry"
where = "x >= 1/6"
```

## Choosing a condition

```toml
[[boundaries]]
name = "wall"           # a boundary zone of the mesh
type = "wall_adiabatic"
```

*Used in:* [`sod`](https://matthewbonanni.github.io/mallard/docs/examples/#sod), [`shu_osher`](https://matthewbonanni.github.io/mallard/docs/examples/#shu-osher), [`riemann_2d`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d), [`riemann_2d_quads`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d-quads), [`double_mach`](https://matthewbonanni.github.io/mallard/docs/examples/#double-mach) and 32 more.

| `type` | Kind | Keys | Use |
|---|---|---|---|
| `wall_adiabatic` | wall | `u` (optional) | No-slip (`navier_stokes`) or slip (`euler`) wall without heat flux; `u` moves the wall |
| `wall_isothermal` | wall | `T`, `u` (optional) | Wall at temperature `T` |
| `wall_heat_flux` | wall | `q`, `u` (optional) | Wall with heat flux `q` into the fluid |
| `symmetry` | symmetry | — | Slip wall, symmetry plane, axis of axisymmetric runs |
| `upt` | inflow | `u`, `p`, `T` (+ `X`/`Y`) | Inflow at a given velocity, pressure and temperature |
| `farfield` | far field | `u`, `p`, `T` | Free stream around a body: characteristic, works for inflow, outflow and tangential flow |
| `dirichlet` | inflow | `rho`, `u`, `p` (expressions of x, y, z, t) | Prescribed exterior state, e.g. an exact solution |
| `extrapolation` | outflow | — | Transmissive: the state from inside; supersonic outflow, or waves leaving a box |
| `p_out` | outflow | `p` | Subsonic outlet at pressure `p` (reflects acoustic waves) |
| `p_out_average` | outflow | `p` | Outlet with mixed subsonic and supersonic faces: the area-averaged pressure is `p`, the profile free |
| `nscbc_outlet` | outflow | `p`, `L`, `sigma`, `beta` | Partially non-reflecting characteristic outlet ([below](#characteristic-boundaries)) |
| `nscbc_inlet` | inflow | `u`, `p`, `T`, `L`, `sigma`, ... | Partially non-reflecting characteristic inlet, optionally with [synthetic turbulence](#synthetic-turbulence) |

Gas mixtures add the composition (`X` or `Y`, and `balance`) to `upt` and
`nscbc_inlet`, as numbers or expressions of x, y, z. Their walls are
non-catalytic (no species crosses them), and all three `wall_*` types work,
e.g. a cold wall that quenches a flame:

```toml
[[boundaries]]
name = "wall"
type = "wall_isothermal"
T = 300.0
```

Pitfall: the wall heat flux is conducted across the half cell next to the
wall, so it is only as good as the resolution there; a flame quenching
against a wall needs cells of a fraction of its quenching distance.

Choosing:

- **External aerodynamics:** `farfield` on the outer boundary, a `wall_*` on
  the body (`examples/cylinder`, `sphere_re300`).
- **Ducts and combustors:** `nscbc_inlet` and `nscbc_outlet` keep acoustic waves
  from reflecting into the domain (`flame_outflow`, `volvo_bluff_body`); `upt`
  and `p_out` are simpler but reflect them (97–99% of a pulse at `p_out`,
  about 1% at `nscbc_outlet`).
- **Supersonic outflow:** `extrapolation`, or `p_out_average` if part of the
  outlet is subsonic.
- **Exact or manufactured solutions, moving shocks:** `dirichlet`.

![The wedge example: inflow, a pressure outlet and symmetry planes over the ramp](../images/guide/domain_wedge-light.svg#gh-light-mode-only)
![The wedge example: inflow, a pressure outlet and symmetry planes over the ramp](../images/guide/domain_wedge-dark.svg#gh-dark-mode-only)

![The viscous shock tube: a closed box of adiabatic walls with a symmetry plane on top](../images/guide/domain_viscous_shock_tube-light.svg#gh-light-mode-only)
![The viscous shock tube: a closed box of adiabatic walls with a symmetry plane on top](../images/guide/domain_viscous_shock_tube-dark.svg#gh-dark-mode-only)

*`examples/wedge` (Mach 1.76 over an 8° ramp) and
`examples/viscous_shock_tube` (shock / boundary-layer interaction in a closed
box), drawn on their generated meshes.*

## Characteristic boundaries

```toml
[[boundaries]]
name = "outlet"
type = "nscbc_outlet"
p = 101325.0
L = 0.1            # a length of the domain [m]
# sigma = 0.25     # relaxation: 0 is perfectly non-reflecting but drifts
```

*Used in:* [`channel_inflow_retau180`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-inflow-retau180), [`turbulent_inflow_duct`](https://matthewbonanni.github.io/mallard/docs/examples/#turbulent-inflow-duct), [`volvo_bluff_body`](https://matthewbonanni.github.io/mallard/docs/examples/#volvo-bluff-body), [`flame_outflow`](https://matthewbonanni.github.io/mallard/docs/examples/#flame-outflow).

`nscbc_outlet` and `nscbc_inlet` let outgoing waves leave and relax the
incoming ones toward a target (Poinsot & Lele 1992, with the transverse terms
of Lodato et al. 2008):

![Waves at a characteristic outlet and inlet: outgoing waves leave, incoming ones relax toward the targets](../images/guide/characteristic-light.svg#gh-light-mode-only)
![Waves at a characteristic outlet and inlet: outgoing waves leave, incoming ones relax toward the targets](../images/guide/characteristic-dark.svg#gh-dark-mode-only)

*`u` carries entropy, vorticity and species; `u ± c` are the acoustic waves.*

| Key | Default | Meaning |
|---|---|---|
| `p` | required | Target (outlet) or reference (inlet) pressure |
| `L` | required | A length of the domain, setting the relaxation rate `K = sigma c (1 - M^2) / L` |
| `sigma` | 0.25 | Relaxation coefficient; 0 lets the mean drift, large values approach a fixed pressure |
| `beta` | local Mach (outlet); 0 for a turbulent inlet | Weight of the transverse terms, 0 to 1 |
| `u`, `T` | required (inlet) | Target velocity (numbers or expressions of x, y, z) and temperature |
| `sigma_T`, `sigma_t` | imposed | Inlet: relax temperature and tangential velocity instead of imposing them |

A wave of frequency `omega` is reflected by `1 / sqrt(1 + (2 omega / K)^2)`.
Restart files carry each face's state, so restarted runs continue bitwise.

## Synthetic turbulence

```toml
[[boundaries]]
name = "left"
type = "nscbc_inlet"
u = [1.0, 0.0, 0.0]
p = 17.857
T = 1.0
L = 6.0

[boundaries.turbulence]
reynolds_stress = [0.01, 0.01, 0.01, 0.0, 0.0, 0.0]   # R_xx, R_yy, R_zz, R_xy, R_xz, R_yz
length_scale = 0.25
```

*Used in:* [`channel_inflow_retau180`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-inflow-retau180), [`turbulent_inflow_duct`](https://matthewbonanni.github.io/mallard/docs/examples/#turbulent-inflow-duct), [`volvo_bluff_body`](https://matthewbonanni.github.io/mallard/docs/examples/#volvo-bluff-body).

An `nscbc_inlet` with a `[boundaries.turbulence]` table adds turbulence to its
target velocity: the digital filter of Klein, Sadiki & Janicka (2003), with
given Reynolds stresses and integral length scales, bitwise the same on any
number of ranks and across restarts.

![A duct with a turbulent characteristic inlet, a characteristic outlet, periodic in y and z](../images/guide/domain_turbulent_inflow_duct-light.svg#gh-light-mode-only)
![A duct with a turbulent characteristic inlet, a characteristic outlet, periodic in y and z](../images/guide/domain_turbulent_inflow_duct-dark.svg#gh-dark-mode-only)

| Key | Default | Meaning |
|---|---|---|
| `reynolds_stress` | required (or `profile`) | `[R_xx, R_yy, R_zz, R_xy, R_xz, R_yz]` (2D: `[R_xx, R_yy, R_xy]`), numbers or expressions |
| `profile` | — | Instead of `u` and `reynolds_stress`: a CSV of plane averages from a precursor run (`tools/plane_average.py`) |
| `length_scale` | required | Integral length: one number, one per direction, or one list per component |
| `convection_velocity` | mean inflow speed | `U_c` of Taylor's hypothesis |
| `seed` | 1 | Random seed |
| `zero_net_flux` | `true` | Remove the area mean of the normal fluctuation (no plane acoustic wave) |
| `points_per_length` | 6 | Generator grid points per smallest integral length |

The inlet must be planar and normal to a coordinate axis. Examples:
`turbulent_inflow_duct`, `channel_inflow_retau180` (statistics from the
periodic channel via `profile`), `volvo_bluff_body`
([validation](https://matthewbonanni.github.io/mallard/validation/#synthetic-inflow)).

## Pitfalls

- **Reflections.** `p_out` and `upt` reflect acoustic waves; in long runs of
  low-Mach flows these can dominate. Use the characteristic conditions, and
  a [sponge layer](sources.md#sponge-layers) in front of the outlet where eddies
  leave.
- **Reversed flow at outlets.** Eddies leaving through an `nscbc_outlet`
  may reverse the flow in its last cells; such faces drop their transverse
  terms, and `beta = 0` is more robust (the `channel_inflow_retau180` example).
- **Viscous terms** treat characteristic and transmissive faces like
  `extrapolation` (zero normal gradients).

## See also

- [Input reference: `[[boundaries]]`](../input.md#boundaries), [`[boundaries.turbulence]`](../input.md#synthetic-turbulence-boundariesturbulence)
- [Numerical methods: boundary conditions](../numerics/overview.md#boundary-conditions)
- Design notes: [characteristic boundaries and sponges](../design/nscbc.md), [synthetic inflow](../design/synthetic_inflow.md)
