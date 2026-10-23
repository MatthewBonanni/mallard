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
| `stagnation_inlet` | inflow | `p0`, `T0`, `direction` (+ `X`/`Y`) | Subsonic inflow from a plenum: the mass flow follows from the flow ([below](#stagnation-inlet)) |
| `mass_flow_inlet` | inflow | `mass_flux` or `mdot`, `T`, `direction` (+ `X`/`Y`) | Subsonic inflow at a fixed mass flow, whatever the inlet density ([below](#mass-flow-inlet)) |
| `farfield` | far field | `u`, `p`, `T` | Free stream around a body: characteristic, works for inflow, outflow and tangential flow |
| `dirichlet` | inflow | `rho`, `u`, `p` (expressions of x, y, z, t) | Prescribed exterior state, e.g. an exact solution |
| `extrapolation` | outflow | — | Transmissive: the state from inside; supersonic outflow, or waves leaving a box |
| `p_out` | outflow | `p` | Subsonic outlet at pressure `p` (reflects acoustic waves) |
| `p_out_average` | outflow | `p` | Outlet with mixed subsonic and supersonic faces: the area-averaged pressure is `p`, the profile free |
| `nscbc_outlet` | outflow | `p`, `L`, `sigma`, `beta` | Partially non-reflecting characteristic outlet ([below](#characteristic-boundaries)) |
| `nscbc_inlet` | inflow | `u`, `p`, `T`, `L`, `sigma`, ... | Partially non-reflecting characteristic inlet, optionally with [synthetic turbulence](#synthetic-turbulence) |

Gas mixtures add the composition (`X` or `Y`, and `balance`) to `upt`,
`stagnation_inlet`, `mass_flow_inlet` and `nscbc_inlet`, as numbers or expressions of x, y, z. Their walls are
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
- **Nozzles, ducts and intakes fed from a reservoir:** `stagnation_inlet`, so
  that the mass flow (and choking) is part of the solution; `mass_flow_inlet`
  where the mass flow is given (combustors, jets).
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

## Stagnation inlet

`stagnation_inlet` feeds the domain from a reservoir at stagnation pressure
`p0` and temperature `T0`. It imposes these, the flow direction and the
composition; the acoustic wave leaving through the inlet sets the speed, and
the inflow state lies on the isentrope through (`T0`, `p0`). Mixtures use their
own `h(T)` and `s(T, p)`, not a constant `gamma`.

![The stagnation inlet: what it imposes and what it takes from the interior](../images/guide/inlet_stagnation-light.svg#gh-light-mode-only)
![The stagnation inlet: what it imposes and what it takes from the interior](../images/guide/inlet_stagnation-dark.svg#gh-dark-mode-only)

```toml
[[boundaries]]
name = "inlet"
type = "stagnation_inlet"
p0 = 2.0e5
T0 = 600.0
# direction = [1.0, 0.2]       # default: the inward normal of each face
# X = { O2 = 0.21, N2 = 0.79 } # mixtures
```

| Key | Default | Meaning |
|---|---|---|
| `p0`, `T0` | required | Stagnation pressure and temperature |
| `direction` | inward normal | Flow direction; must point into the domain at every face |
| `X` / `Y`, `balance` | required for mixtures | Composition, as for `upt` |

![Nozzle: Mach number, pressure and stagnation pressure against the quasi-one-dimensional solution](../images/guide/nozzle_validation-light.svg#gh-light-mode-only)
![Nozzle: Mach number, pressure and stagnation pressure against the quasi-one-dimensional solution](../images/guide/nozzle_validation-dark.svg#gh-dark-mode-only)

*`examples/nozzle`: a converging-diverging nozzle fed at `p0`, `T0` with a
normal shock in the diverging part, against the quasi-one-dimensional
solution: choked mass flow within 0.2%, the shock within two cells.*

Pitfalls:

- The inlet is for subsonic inflow. Supersonic inflow keeps the interior
  speed: start such runs from a field that is subsonic at the inlet.
- Where the interior pressure exceeds `p0` (a start-up transient, a back
  pressure above `p0`), the inlet sees the reservoir at rest and lets the
  flow out against `p0`; it does not hold a mass flow.
- It reflects acoustic waves like `upt` does (the mass flow follows them);
  pair it with `nscbc_outlet` or a sponge where reflections matter.

Design notes: [inlets](../design/inlets.md).

## Mass-flow inlet

`mass_flow_inlet` holds the mass flux through the inlet, whatever the density
there does: the usual inflow of combustors and jets, where heat release or
compression downstream changes the density at the inlet. It imposes the mass
flux, the temperature, the flow direction and the composition; the acoustic
wave leaving through the inlet sets the pressure, and the density follows.
The face flux is the physical flux of that boundary state, so the mass
entering is exactly the target. (`upt` fixes the velocity instead, and its
mass flow drifts with the inlet density.)

![The mass-flow inlet: what it imposes and what it takes from the interior](../images/guide/inlet_mass_flow-light.svg#gh-light-mode-only)
![The mass-flow inlet: what it imposes and what it takes from the interior](../images/guide/inlet_mass_flow-dark.svg#gh-dark-mode-only)

```toml
[[boundaries]]
name = "inlet"
type = "mass_flow_inlet"
mdot = 0.02                 # kg/s through the whole entry (2D: per unit depth)
T = 300.0
# profile = "1 - (y / 0.05)^2"  # shape of the mass flux along the inlet
# mass_flux = 20.0             # or the flux itself [kg/(m^2 s)], a number or an expression of x, y, z
# direction = [1.0, 0.0]       # default: the inward normal of each face
# X = { O2 = 0.21, N2 = 0.79 } # mixtures
```

| Key | Default | Meaning |
|---|---|---|
| `mass_flux` or `mdot` | one required | Mass flux `rho u . n_in` [kg/(m^2 s)] (number or expression of x, y, z), or the total mass flow through the entry's faces [kg/s; per unit depth in 2D, per radian in axisymmetric runs] |
| `profile` | uniform | With `mdot`: the shape of the mass flux along the inlet (an expression of x, y, z, non-negative), scaled to `mdot` |
| `T` | required | Inflow temperature |
| `direction` | inward normal | Flow direction; must point into the domain at every face |
| `X` / `Y`, `balance` | required for mixtures | Composition, as for `upt` |

![A closed duct fed through a mass-flow inlet and heated: the mass in the duct grows by exactly mdot t, while upt's drifts](../images/guide/mass_flow_held-light.svg#gh-light-mode-only)
![A closed duct fed through a mass-flow inlet and heated: the mass in the duct grows by exactly mdot t, while upt's drifts](../images/guide/mass_flow_held-dark.svg#gh-dark-mode-only)

*`examples/heated_duct` (`closed.toml`): a duct closed at the far end and
heated in its middle. The pressure rises and the inlet density with it (by
45%); the mass in the duct grows by `mdot t` to 1e-14, while `upt` at the
same initial mass flux holds its pressure and lets the gas back out.*

![Steady heated duct: pressure, velocity and temperature against Rayleigh flow](../images/guide/heated_duct-light.svg#gh-light-mode-only)
![Steady heated duct: pressure, velocity and temperature against Rayleigh flow](../images/guide/heated_duct-dark.svg#gh-dark-mode-only)

*`examples/heated_duct` (`input.toml`): the same duct open to a 1 bar outlet,
fed at 100 kg/(m^2 s) and heated between x = 0.4 and 0.6 m, against Rayleigh
flow: the inlet pressure the flow needs within 1e-5, the exit temperature
within 1e-6.*

Pitfalls:

- The inlet is for subsonic inflow: it takes the pressure from the
  interior.
- `mdot` is divided among the entry's faces by their areas (times `profile`);
  in 2D it is per unit depth.
- Like `upt`, it reflects acoustic waves (it holds the mass flux, not the
  velocity or the pressure); pair it with `nscbc_outlet` or a sponge where
  reflections matter.

## Time-dependent conditions

The velocity, pressure and temperature of `upt` and `nscbc_inlet`, `p0` and
`T0` of `stagnation_inlet`, `mass_flux` and `T` of `mass_flow_inlet`, the
wall velocity and, for mixtures, the inflow composition may be expressions of
`x`, `y`, `z` and `t`. Those of `t` are evaluated at every Runge-Kutta stage,
so a pulsed or acoustically forced inlet converges at the time integrator's
order (third with SSPRK3); evaluating them once per step would give first
order.

```toml
[[boundaries]]
name = "inlet"
type = "nscbc_inlet"
u = ["50 * (1 + 0.1 * sin(2 * pi * 1000 * t))", "0"]   # 10% at 1 kHz
T = "300 * (1 + 0.05 * sin(2 * pi * 250 * t))"
p = 1.0e5
L = 1.0
sigma = 100.0     # follow the forcing closely
```

![A forced characteristic inlet: velocity forcing leaves as acoustic waves, temperature forcing as entropy waves](../images/guide/forced_inlet-light.svg#gh-light-mode-only)
![A forced characteristic inlet: velocity forcing leaves as acoustic waves, temperature forcing as entropy waves](../images/guide/forced_inlet-dark.svg#gh-dark-mode-only)

*`examples/forced_inlet`: velocity forcing at 1 kHz leaves the inlet as
acoustic waves at `u + c`, temperature forcing at 250 Hz as entropy waves at
`u`; the inlet cell follows both targets (`tools/plot_forced_inlet.py`).*

Pitfalls:

- `nscbc_inlet` relaxes its velocity toward the target at the rate
  `K = sigma c (1 - M^2) / L`. Forcing faster than `K` reaches the domain
  attenuated (to 56% at 1 kHz with the default `sigma = 0.25` here); raise
  `sigma` to follow it, at the price of reflecting outgoing waves of lower
  frequency. `upt` imposes the forcing exactly but reflects everything.
- `nscbc_inlet` advances its boundary point once per step, so its forcing
  converges at first order in `dt`; `upt` and the other inlets at the
  integrator's order.
- Synthetic turbulence takes a mean velocity of `x`, `y`, `z` only.

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
