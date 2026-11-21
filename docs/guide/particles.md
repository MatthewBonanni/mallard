# Particles

Mallard tracks Lagrangian particles through the flow, on the device, in 2D and
3D, serial and on many ranks. Massless tracers move with the gas velocity at
their position, for pathlines, residence times and other Lagrangian analyses;
[solid particles](#solid-particles) have inertia: drag, gravity and the forces of
the gas accelerate them, convection heats them, and walls rebound or catch
them. Each particle can carry
[quantities](#per-particle-quantities) along its path (time integrals, extrema,
first times, turns about an axis), and [probes](#particle-probes) give
statistics of the particles in a region. With
[two-way coupling](#two-way-coupling) solid particles act on the gas,
[evaporating droplets](#evaporating-droplets) give it their vapor, and
[dispersed-phase fields](#dispersed-phase-fields) put them on the mesh.
Injection during the run and particle output beyond the final positions follow
in later releases ([#291](https://github.com/MatthewBonanni/mallard/issues/291)).

## Tracers: `[particles]`

```toml
[particles]
type = "tracer"
file = "tracers.csv"

[particles.lattice]
lo = [0.1, 0.1]
hi = [0.9, 0.9]
n = [9, 9]
```

*Used in:*
[`tracer_vortex`](https://matthewbonanni.github.io/mallard/docs/examples/#tracer-vortex),
[`particle_cloud`](https://matthewbonanni.github.io/mallard/docs/examples/#particle-cloud),
[`particle_vortex`](https://matthewbonanni.github.io/mallard/docs/examples/#particle-vortex).

A tracer is a zero-Stokes particle: massless, its velocity the gas velocity
where it is, with no effect on the flow. Mallard evaluates that velocity from
the reconstruction's own polynomial in the particle's cell, not from cell
averages, and advances the positions as part of the time integrator's state,
so pathlines converge at the order of the scheme in space (fourth order with
TENO of `order = 4`, second order with MUSCL) and at the integrator's order in
time (first with `FE`, third with `SSPRK3`, fourth with `RK4`). Results are
bitwise the same on any number of ranks.

| Key | Default | Meaning |
|---|---|---|
| `type` | required | `"tracer"` |
| `points` | none | Seed points, `[[x, y], ...]` (`[[x, y, z], ...]` in 3D) |
| `lattice` | none | A table `lo`, `hi`, `n`: a lattice of `n[d]` points per direction from `lo` to `hi` (one point along a direction: its middle) |
| `file` | none | CSV file of the particles at the end of the run: `id`, `x`, `y` (`z`), the velocity `v_x`, `v_y` (`v_z`) of solid particles, their temperature `T_p` with heat transfer, then the quantities, by id |
| `capacity` | the number of particles | Particles a rank can hold |

Every particle is seeded at the start of the run. Ids follow the seeds: the
`points` first, in order, then the lattice with `x` varying fastest.

**Restarts.** Restart files carry the particles, and a restarted run continues
their trajectories exactly, on any number of ranks; the seeds of `[particles]`
are then not used. A restart file written without particles (by a run without
`[particles]`) seeds them, which adds tracers to a developed flow: run the
flow to its statistically steady state, then restart from it with
`[particles]`. A `remap` start seeds them too.

At boundaries:

- walls and symmetry planes reflect a particle that crossed them within a step;
- inlets, outlets and far-field boundaries remove it;
- periodic boundaries, translational or rotational, carry it to the other side.

**Pitfalls.**

- A seed outside the mesh is an error, naming the point.
- In a troubled cell (a shock, a flame front) and with first-order or MUSCL
  reconstruction, the velocity comes from the cell's linear least-squares fit
  rather than a high-order polynomial.
- On several ranks, particles move between ranks through the host at every
  step, so such runs are not captured in CUDA graphs yet (single-rank runs
  are).
- SIMPLER chemistry coupling is refused for now.

## Solid particles

```toml
[physics]
type = "navier_stokes"
mu = 1.8e-5

[particles]
type = "solid"
diameter = 20e-6         # m
density = 2500.0         # kg/m^3
gravity = [0.0, -9.81]   # default: [source] gravity
points = [[0.1, 0.5], [0.2, 0.5]]

[[particles.walls]]
zones = ["floor"]
interaction = "stick"
```

*Used in:*
[`particle_vortex`](https://matthewbonanni.github.io/mallard/docs/examples/#particle-vortex).

A solid particle is a small rigid sphere with a velocity of its own, which
drag pulls toward the gas velocity and gravity (less the buoyancy of the gas it
displaces) accelerates:

$$\frac{d\mathbf{v}}{dt} = \frac{\mathbf{u} - \mathbf{v}}{\tau} + \mathbf{g}\left(1 - \frac{\rho}{\rho_p}\right),
\qquad \tau = \frac{\rho_p d^2}{18 \mu f}, \qquad Re_p = \frac{\rho |\mathbf{u} - \mathbf{v}| d}{\mu}.$$

The gas state at the particle (`u`, `rho`, `mu` at its temperature) comes from
the reconstruction polynomial, as for tracers, and position and velocity are
part of the time integrator's state: trajectories converge at the
integrator's order. `f = C_D Re_p / 24` is the drag correlation:

- `"stokes"`: `f = 1`, creeping flow;
- `"schiller_naumann"`: `f = 1 + 0.15 Re_p^0.687` up to `Re_p = 1000`, then
  `C_D = 0.44`;
- `"singh"`: the correlation of [Singh et al. (2022)](../references.md#singh-2022)
  for compressible and rarefied flow, of `Re_p`, the slip Mach number, the
  particle Knudsen number and the ratio of the particle's temperature to the
  gas's (the gas's without heat transfer). It reduces to Stokes drag in
  continuum creeping flow and to free-molecular drag at large Knudsen numbers,
  and adds the shock drag above Mach 1: use it for small particles in
  low-density or high-speed gas, where Schiller–Naumann overestimates the drag.

By default the particles do not act on the gas; `coupling = "two-way"` makes
them ([Two-way coupling](#two-way-coupling)).

**Forces.** `forces` adds, to drag and gravity, the forces of
[Maxey & Riley (1983)](../references.md#maxey-riley-1983) that matter when the
particle is not much denser than the gas, and lift:

| `forces` entry | Adds | Matters when |
|---|---|---|
| `"pressure_gradient"` | `-grad p / rho_p` in place of the buoyancy `-g rho / rho_p` (equal in a gas at rest) | the gas accelerates: `rho / rho_p` is not small and the flow has strong pressure gradients (shocks, nozzles, vortex cores) |
| `"added_mass"` | half the displaced gas mass, accelerated with the gas: `rho_p` becomes `rho_p + rho / 2` in `tau` and the inertia, and `(rho g - grad p) / 2` is added to the force | `rho / rho_p` is not small (bubbles, light particles in dense gas) |
| `"saffman_lift"` | the shear lift of [Saffman (1965)](../references.md#saffman-1965), `1.615 (6 / (pi d)) sqrt(rho mu / abs(w)) (u - v) x w` per volume, `w` the vorticity | particles slip through shear layers and boundary layers |

The gradients come from the cell's least-squares gradient. The Basset history
force is not included.

**Heat transfer.** With `specific_heat`, each particle carries a temperature
`T_p`, which convection relaxes toward the gas temperature at the rate of the
[Ranz & Marshall (1952)](../references.md#ranz-marshall-1952) Nusselt number:

$$\frac{dT_p}{dt} = \frac{T - T_p}{\tau_T}, \qquad \tau_T = \frac{\rho_p c_p d^2}{6 k Nu},
\qquad Nu = 2 + 0.6 Re_p^{1/2} Pr^{1/3},$$

with `k` the gas conductivity. The particle is isothermal (its Biot number is
small) and, unless coupled two-way, does not heat the gas. Particle expressions read it as `T_P`, the
`file` has it as a column `T_p`, and restart files carry it.

**Stiff particles.** The relaxation time `tau` can be far below the flow's
time step (a 1 µm droplet in air relaxes in a few microseconds). A particle
whose `tau` is below the step `dt` relaxes exponentially within the step: its
velocity is integrated exactly for a gas velocity and force that vary linearly
over the step, so tiny particles follow the gas at their equilibrium velocity
`u + tau g'` instead of making the run unstable, particles with `tau` near
`dt` keep their inertia, and the step never depends on the particles. The
same holds for `tau_T`. The trajectories still converge at the integrator's
order.

| Key | Default | Meaning |
|---|---|---|
| `type` | required | `"solid"` |
| `diameter` | required | Sphere diameter |
| `density` | required | Material density `rho_p` |
| `drag` | `"schiller_naumann"` | `"stokes"`, `"schiller_naumann"` or `"singh"` |
| `gravity` | `[source] gravity`, else none | Gravity on the particles |
| `forces` | `[]` | Further forces: `"pressure_gradient"`, `"added_mass"`, `"saffman_lift"` |
| `specific_heat` | none (no heat transfer) | Particle specific heat `c_p`: tracks the particle temperature |
| `velocity` | the gas velocity at each seed | Velocity of every seeded particle |
| `temperature` | the gas temperature at each seed | Temperature of every seeded particle (with `specific_heat`) |

The other keys of [`[particles]`](#tracers-particles) (seeds, `file`,
`capacity`) are the same. Particle expressions read the particle velocity as
`V_X`, `V_Y`, `V_Z`. Restart files carry the velocities and temperatures;
tracers restarted as solid particles start at the gas velocity, and particles
restarted with heat transfer from a file without temperatures at the gas
temperature.

**Walls.** Walls and symmetry planes rebound solid particles elastically;
`[[particles.walls]]` sets other interactions per boundary zone. A rebound
reverses the normal velocity times the restitution coefficient `e_n` and
scales the tangential velocity by `e_t`; a sticking wall removes the particle
and counts it as deposited. The bounce is located within the step: the
particle reaches the wall at the time its motion over the step puts it there,
rebounds, and moves on for the rest of the step, so the timing of bounces
does not depend on the step.

| Key | Default | Meaning |
|---|---|---|
| `zones` | required | Wall or symmetry boundary zones |
| `interaction` | `"rebound"` | `"rebound"` or `"stick"` |
| `restitution` | 1 | `e_n` in [0, 1] (rebound) |
| `tangential_restitution` | 1 | `e_t` in [0, 1] (rebound) |

**Pitfalls.**

- Solid particles need a viscous single gas (`type = "navier_stokes"` in
  `[physics]`); mixtures are refused for now.
- The bounce of a particle with `tau` below the step happens at the end of
  the step: such a particle follows the gas within the step.
- The located bounce takes the gas velocity and force at the start of the
  step, so it is exact for drag in a gas that is uniform over the step's
  travel and for constant acceleration; in a strongly varying gas it is still
  first order in the step.
- `temperature` without `specific_heat` is an error, as is an expression
  reading `T_P` in a mixture.

## Two-way coupling

```toml
[particles]
type = "solid"
coupling = "two-way"
weight = 1e4              # particles per parcel
deposition_width = 2e-3   # m
diameter = 50e-6
density = 2500.0
specific_heat = 840.0
```

*Used in:*
[`particle_cloud`](https://matthewbonanni.github.io/mallard/docs/examples/#particle-cloud).

With `coupling = "two-way"` the gas feels the particles: each stage, every
particle gives the gas the opposite of its own change of momentum and energy,
less what gravity on the particle did, times `weight`:

$$\mathbf{S}_{\rho \mathbf{u}} = -w\, m_p \left(\frac{d\mathbf{v}}{dt} - \mathbf{g}\right),
\qquad S_{\rho E} = -w\, m_p \left(\frac{d e_p}{dt} - \mathbf{g} \cdot \mathbf{v}\right),
\qquad e_p = c_p T_p + \tfrac{1}{2} |\mathbf{v}|^2,$$

summed into the cells and spread over a Gaussian of standard deviation
`deposition_width`. Drag, the pressure-gradient, added-mass and lift forces and
heat transfer all act back on the gas; gravity on the particles (`gravity` of
`[particles]`) does not, since the earth exerts it, not the gas. The sources use
the same stage rates the particles are integrated with, so the gas and the
particles together conserve momentum and energy to round-off (with heat
transfer; without it, the particles' kinetic energy is conserved to the
integrator's order), however stiff the particles are.

A parcel stands for `weight` identical particles, so that a computational
particle can represent many physical ones; in 2D a parcel stands for `weight`
particles per meter of depth. The mass loading of a cell, `w m_p / (rho V)`,
decides how strongly the particles act: near 1 and above, they drag the gas
along.

**Deposition.** With `deposition_width = 0` (the default) each particle's
source goes to the cell holding it. On a fine mesh that puts a point source in
one small cell, so the result depends on the mesh. A `deposition_width` in
meters spreads it over a Gaussian of that width whatever the cells, so coupled
results converge as the mesh is refined: second order in the cell size at a
fixed width (the deposit keeps each particle's position within its cell, not
only its cell). Choose a width of the order of the cell size of the coarsest
mesh you would trust, or of the physical size of the region a particle
disturbs. The spreading never moves sources across walls or out of the domain,
and conserves them exactly.

| Key | Default | Meaning |
|---|---|---|
| `coupling` | `"one-way"` | `"one-way"`: the particles feel the gas only; `"two-way"`: the gas feels them too |
| `weight` | 1 | Particles per parcel (per meter of depth in 2D) |
| `deposition_width` | 0 | Standard deviation of the Gaussian the sources spread over; 0: each particle's own cell |

The results are bitwise the same on any number of ranks and with or without
CUDA graphs.

**Pitfalls.**

- The point-particle model assumes particles much smaller than the cells and a
  dilute suspension. `mallard check` warns where particles are larger than a
  tenth of their cell (`W-PARTICLE-SIZE`) and where the particle volume
  fraction exceeds 1% (`W-PARTICLE-VOLUME-FRACTION`): collisions (four-way
  coupling) and the volume the particles take from the gas are left out.
- Two-way coupling needs solid particles; tracers are massless. Axisymmetric
  runs are refused.
- A wide `deposition_width` costs spreading passes per stage, about
  `4 width^2 / h^2` in 2D and `6 width^2 / h^2` in 3D for cells of size `h`,
  each with a halo exchange on several ranks.

## Evaporating droplets

```toml
[physics]
type = "navier_stokes"
gas = "mixture"
mechanism = "mechanisms/h2o2.yaml"   # holds the vapor species
transport = "mixture_averaged"

[particles]
type = "droplet"
diameter = 50e-6
temperature = 300.0
evaporation = "abramzon_sirignano"   # or "d2_law"
coupling = "two-way"
weight = 1e4

[particles.liquid]
species = "H2O"
density = 958.0
specific_heat = 4216.0
latent_heat = 2.257e6           # J/kg, at the boiling temperature
boiling_temperature = 373.15
critical_temperature = 647.1
critical_pressure = 22.064e6
```

Droplets are solid particles of a liquid that evaporates into a gas mixture:
sprays, fuel injection, humidification, quenching. Each droplet moves as a
solid particle (drag, gravity, `forces`, walls), is heated by the gas and
evaporates into the vapor species `species` of the mechanism, so its diameter
falls until, below `minimum_diameter`, it is removed. With
`coupling = "two-way"` the gas receives the vapor, with its momentum and
energy, and gives the heat; the gas and droplets together conserve mass, every
species, momentum and energy to round-off. Droplets need a viscous mixture
(`gas = "mixture"`, `type = "navier_stokes"`).

**The model.** A droplet is a sphere of uniform temperature (infinite liquid
conductivity) of one component. Its surface is at its temperature `T_p`, up to
the boiling temperature at the gas pressure, and the vapor there is at its
saturation pressure: the surface's vapor mass fraction `Y_s` follows from
Raoult's law of a pure liquid. The gas film around the droplet is evaluated by
the 1/3 rule, `T_f = T_s + (T - T_s)/3`, `Y_f = Y_s + (Y - Y_s)/3`, with the
mechanism's thermodynamics and transport (its viscosity also sets the drag).
With the [Spalding (1953)](../references.md#spalding-1953) mass transfer number `B_M = (Y_s - Y)/(1 - Y_s)`:

- `evaporation = "d2_law"`: `mdot = pi d rho_f D_f Sh0 ln(1 + B_M)` and the
  heat from the gas `Q = pi d k_f Nu0 (T - T_s)`, [Ranz & Marshall (1952)](../references.md#ranz-marshall-1952)'s
  `Sh0 = 2 + 0.6 Re^1/2 Sc^1/3`, `Nu0 = 2 + 0.6 Re^1/2 Pr^1/3`. At rest and at
  its wet-bulb temperature, d² falls linearly:
  `d² = d0² - K t`, `K = 8 rho_f D_f ln(1 + B_M) / rho_l`.
- `evaporation = "abramzon_sirignano"` (default): the film model of
  [Abramzon & Sirignano (1989)](../references.md#abramzon-sirignano-1989), which thickens the film by the Stefan flow and transfers heat by the
  number `B_T = (1 + B_M)^phi - 1`, `phi = (c_p,v / c_p,f) (Sh*/Nu*) / Le_f`:
  `mdot = pi d rho_f D_f Sh* ln(1 + B_M)`,
  `Q = pi d k_f Nu* (T - T_s) ln(1 + B_T)/B_T`. The usual choice for sprays
  in hot gas ([Miller et al. 1998](../references.md#miller-harstad-bellan-1998)).

The heat into the liquid is `Q - mdot L(T_s)`, so a droplet heats or cools to
its wet-bulb temperature, where the heat from the gas just evaporates it. The
liquid has a constant density and specific heat; its enthalpy is that of the
mechanism's vapor at the boiling point less `latent_heat`, so the latent heat
at other temperatures follows Kirchhoff's law and the liquid and the gas share
one energy reference. The saturation pressure is Clausius-Clapeyron's through
the boiling point, or Antoine's with `antoine`. A droplet at or above the
boiling temperature boils: the heat it receives evaporates it and its superheat
relaxes to the boiling temperature at its heating rate. Above the critical
pressure the boiling temperature is the critical temperature, and the
properties of a droplet driven past the critical temperature are those at it.

A droplet that would evaporate entirely within a step evaporates within that
step at its temperature, and droplets below `minimum_diameter` are removed
after the step: with two-way coupling their remains (mass, momentum, energy)
go to the gas as vapor. Restart files keep the count of evaporated droplets
with those that left the domain, were lost or stuck (`PARTICLE_REMOVED`).

| Key | Default | Meaning |
|---|---|---|
| `type` | required | `"droplet"` |
| `diameter` | required | Initial diameter of every seeded droplet |
| `temperature` | required | Initial temperature, below `critical_temperature` |
| `velocity` | gas velocity | Initial velocity |
| `evaporation` | `"abramzon_sirignano"` | `"abramzon_sirignano"` or `"d2_law"` |
| `minimum_diameter` | `diameter / 100` | Droplets below it are removed (their remains go to the gas with two-way coupling) |
| `drag`, `gravity`, `forces`, `coupling`, `weight`, `deposition_width` | | As for [solid particles](#solid-particles) and [two-way coupling](#two-way-coupling) |

| `[particles.liquid]` key | Default | Meaning |
|---|---|---|
| `species` | required | The vapor species of the mechanism |
| `density` | required | Liquid density (constant) |
| `specific_heat` | required | Liquid specific heat (constant) |
| `latent_heat` | required | Latent heat of vaporization at the boiling point |
| `boiling_temperature` | required | Boiling temperature at `boiling_pressure` |
| `boiling_pressure` | 101325 Pa | Pressure of the boiling point |
| `critical_temperature`, `critical_pressure` | required | Critical point; above the boiling point |
| `antoine` | Clausius-Clapeyron | `[A, B, C]` of the saturation pressure `log10(p / Pa) = A - B / (T / K + C)` |

Particle expressions read the droplet's temperature `T_P`, its diameter `D_P`
and the gas temperature `T`; the CSV `file` adds the diameter `d` and mass `m`
of each droplet. With two-way coupling, `PARTICLE_SOURCE_RHOY_<species>` writes
the vapor source per volume and time.

**Pitfalls.**

- The gas composition at a droplet is the average of its cell, not
  reconstructed to its position as the velocity and pressure are.
- One-component liquids only; the liquid's temperature is uniform (no internal
  conduction), its density and specific heat constant.
- Supercritical droplets are outside the model: properties stop at the
  critical temperature, and a gas above the critical pressure boils the
  droplet at the critical temperature.
- With two-way droplets, the conservation ledger of the run does not check
  the gas's totals, which the droplets change; the closed-box totals of gas and
  droplets together are conserved to round-off.
- `density` and `specific_heat` of `[particles]` are errors for droplets: they
  belong to `[particles.liquid]`. A `specific_heat` and `latent_heat` whose
  Kirchhoff latent heat vanishes below the critical temperature are refused.

## Dispersed-phase fields

```toml
[[write_data]]
prefix = "spray"
fields = ["PARTICLE_VOLUME_FRACTION", "PARTICLE_U_X", "PARTICLE_SOURCE_RHOU_X"]
```

Runs with solid particles can write the particles as mesh fields, deposited as
the coupling deposits them (over `deposition_width`):

| Field | Meaning |
|---|---|
| `PARTICLE_VOLUME_FRACTION` | Particle volume per cell volume |
| `PARTICLE_NUMBER_DENSITY` | Particles per volume (counting `weight`) |
| `PARTICLE_U_X`, `_Y`, `_Z` | Mass-weighted mean particle velocity, 0 where there are none |
| `PARTICLE_SOURCE_<A>` | What the particles give the gas per volume and time at the written state, for `<A>` each conservative variable (`RHO`, `RHOU_X`, ..., `RHOE`); with `coupling = "two-way"` |

They are written by `[[write_data]]` and listed by `mallard list fields`.

## Per-particle quantities

```toml
[[particles.quantities]]
name = "t_cavity"
integral = "in(cavity)"          # time spent in the region cavity of [regions]

[[particles.quantities]]
name = "T_max"
max = "T"

[[particles.quantities]]
name = "t_hot"
first_time = "T > 1500"

[[particles.quantities]]
name = "turns"
winding = {point = [0.052, -0.063, 0.0], axis = [0.0, 0.0, 1.0]}
```

*Used in:*
[`tracer_vortex`](https://matthewbonanni.github.io/mallard/docs/examples/#tracer-vortex).

Each `[[particles.quantities]]` follows one value along every particle's
path, for residence times, exposure (the time integral of a temperature or a
concentration), peak values and when something first happened. A quantity is
one of:

| Key | Meaning |
|---|---|
| `integral = "<expression>"` | The time integral of the expression along the path, from 0 at the seed |
| `max = "<expression>"`, `min = ...` | Its largest (smallest) value, `-inf` (`inf`) until the first sample |
| `first_time = "<condition>"` | The first time `t` at which the condition holds (is not 0), `nan` until then |
| `winding = {point = [...], axis = [...]}` | The turns of the particle about the axis through the point, unwrapped and signed (positive counterclockwise about the axis); 2D takes no `axis` (it is `z`) |

and each takes:

| Key | Default | Meaning |
|---|---|---|
| `name` | required | The quantity's name: letters, digits and `_`, not one of the variables below |
| `while` | none | A condition: the quantity integrates, or samples, only where it holds |

Expressions follow the [expression syntax](expressions.md) and read, at the
particle:

- `x`, `y` (`z`) and `t`;
- the gas: `RHO`, `U_X`, `U_Y` (`U_Z`), `P` and, for a single gas and for droplets, `T`,
  evaluated as the particle's velocity is, from the reconstruction polynomial
  of its cell;
- the particle velocity `V_X`, `V_Y` (`V_Z`), for tracers the gas velocity;
- the particle temperature `T_P`, for a single gas and for droplets, the gas
  temperature for particles without heat transfer;
- the particle diameter `D_P` (solid particles and droplets; `nan` for tracers);
- the quantities listed before this one;
- `in(name)`: 1 inside the region `name` of [regions](geometry.md#named-regions)
  and 0 outside, smooth across the region's `edge`.

Integrals and windings are part of the time integrator's state, as the
positions are: each stage evaluates them at its own position, so they converge
at the integrator's order (fourth with `RK4`). An integrand that jumps, such as
`in(name)` of a region without an `edge`, is integrated to within a step at
each crossing; give the region an `edge` of a few steps' travel to keep the
order. Extrema and first times sample the start of every step, where the
state is the accepted solution, so they are exact to the step: a peak between
two steps' starts is missed by the difference.

Quantities are written as columns of `file`, continue across restarts (a
quantity the restart file does not have starts afresh), move with their
particles between ranks, and are bitwise the same on any number of ranks.

## Particle probes

```toml
[[particles.probes]]
name = "core"
region = {type = "disk", center = [0.25, 0.25], radius = 0.12}
fields = ["turns", "hypot(U_X, U_Y)"]
interval = 100
```

*Used in:*
[`tracer_vortex`](https://matthewbonanni.github.io/mallard/docs/examples/#tracer-vortex).

A particle probe writes, every `interval` steps, the number of particles in
its region and the mean, standard deviation, minimum and maximum of each field
over them: one row per sample of its CSV file (`step`, `t`, `count`, then
`<field> mean`, `<field> std`, `<field> min`, `<field> max`).

| Key | Default | Meaning |
|---|---|---|
| `name` | required | The probe's name |
| `region` | every particle | A filled region of primitives, inline or a name of [regions](geometry.md#named-regions), that does not move |
| `fields` | none (the count only) | Expressions of the particle variables above, the quantities among them |
| `interval` | 1 | Steps between samples |
| `file` | `particle_probe_<name>.csv` | Its CSV file |

Sums are exact in fixed point, so the statistics are bitwise the same on any
number of ranks. A probe samples on the host: steps decided on the device stop
at its samples.

**Pitfalls (quantities and probes).**

- An expression reading `T` or `T_P` in a mixture is an error, except for
  droplets: particle expressions have temperatures for a single gas and for
  droplets only.
- `in(name)` and probe regions are regions of primitives that do not move
  (boolean combinations included); at most 4 regions in all particle
  expressions, 4 probes with regions, and 16 quantities.
- Probes count particles by their positions only, at the sampling step; the
  residence time of visits (per visit, total, visit counts) is a quantity:
  `integral = "in(name)"` gives the total.

*Design notes:* [particles](../design/particles.md).
