# Particles

Mallard tracks Lagrangian particles through the flow, on the device, in 2D and
3D, serial and on many ranks. Massless tracers move with the gas velocity at
their position, for pathlines, residence times and other Lagrangian analyses;
[solid particles](#solid-particles) have inertia: drag and gravity accelerate
them, and walls rebound or catch them. Each particle can carry
[quantities](#per-particle-quantities) along its path (time integrals, extrema,
first times, turns about an axis), and [probes](#particle-probes) give
statistics of the particles in a region. Particles do not act on the gas yet;
two-way coupling, sprays, injection during the run and particle output beyond
the final positions follow in later releases ([#291](https://github.com/MatthewBonanni/mallard/issues/291)).

- [Which particles?](#which-particles)
- [Tracers](#tracers-particles), [seeding](#seeding), [restarts](#restarts) and [boundaries](#boundaries)
- [Solid particles](#solid-particles): [drag](#drag-laws), [the stiff limit](#the-stiff-limit), [walls](#walls)
- [Per-particle quantities](#per-particle-quantities) and [particle probes](#particle-probes)
- [Accuracy](#accuracy), [performance](#performance) and a [checklist](#checklist)

The equations and methods are in the theory manual's
[Lagrangian particles](../theory/particles.md); the measured orders of
accuracy in the [verification manual](../verification/index.md#lagrangian-particles).

## Which particles?

The particles are *point particles*: each is much smaller than the cells and
the smallest eddies around it, sees the gas as undisturbed at its center, and
moves on its own, without collisions with other particles and without acting
on the gas (one-way coupling). That describes dilute suspensions, with a
volume fraction of particles below about 10⁻⁶ (a mass loading of 0.2% for
glass beads in air), and every tracer study.

A solid particle forgets its velocity in its **relaxation time**
$\tau = \rho_p d^2 / (18 \mu f)$, where $\rho_p$ is its material density, $d$
its diameter, $\mu$ the gas viscosity and $f \ge 1$ the drag correction (1
for creeping flow). Compared with a time scale of the flow $\tau_f$ (an eddy's
turnover time, a vortex's period, the time to cross a nozzle), it gives the
**Stokes number** $St = \tau / \tau_f$:

| $St$ | The particle | Use |
|---|---|---|
| $\ll 1$ (below 0.05) | follows the gas, plus its settling velocity $\tau g$ | `tracer` if settling does not matter; `solid` (stiff limit) for settling smoke, fog, fine dust |
| about 0.1 to 10 | lags the gas, is flung out of vortices, overshoots turns, hits walls | `solid` |
| $\gg 1$ (above 10) | moves almost ballistically; the gas changes its course slowly | `solid` |

Typical relaxation times in air at 300 K and 1 bar ($\mu$ = 1.8 × 10⁻⁵ Pa s),
with Schiller-Naumann drag at the terminal velocity:

| Particle | $d$ | $\tau$ | Terminal velocity | $Re_p$ at it |
|---|---|---|---|---|
| Water droplet | 1 µm | 3.1 µs | 0.030 mm/s | 2 × 10⁻⁶ |
| Water droplet | 10 µm | 0.31 ms | 3.0 mm/s | 0.002 |
| Water droplet | 100 µm | 26 ms | 0.25 m/s | 1.6 |
| Glass bead (2500 kg/m³) | 10 µm | 0.77 ms | 7.5 mm/s | 0.005 |
| Glass bead (2500 kg/m³) | 100 µm | 57 ms | 0.56 m/s | 3.6 |

$\tau$ grows as $d^2$: a factor 10 in diameter is a factor 100 in Stokes
number. In a jet of 10 m/s through a 1 cm nozzle ($\tau_f$ = 1 ms), the 10 µm
droplets have $St$ = 0.3 and the 1 µm ones $St$ = 0.003.

![Particles released at the gas velocity in solid-body rotation: tracers circle, inertial particles spiral out faster the larger their Stokes number](../images/guide/particle_stokes_spirals-light.svg#gh-light-mode-only)
![Particles released at the gas velocity in solid-body rotation: tracers circle, inertial particles spiral out faster the larger their Stokes number](../images/guide/particle_stokes_spirals-dark.svg#gh-dark-mode-only)

*Particles released at the gas velocity in gas rotating at $\Omega$ (Mallard
runs of the [`particle_centrifuge`](https://matthewbonanni.github.io/mallard/docs/examples/#particle-centrifuge)
flow, dots every quarter of a turn, against the exact spirals, lines):
tracers ($St = 0$) circle forever; inertial particles lag the rotation and
spiral out, fastest near $\Omega\tau = 1$.*

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
[`particle_settling`](https://matthewbonanni.github.io/mallard/docs/examples/#particle-settling),
[`particle_bounce`](https://matthewbonanni.github.io/mallard/docs/examples/#particle-bounce),
[`particle_centrifuge`](https://matthewbonanni.github.io/mallard/docs/examples/#particle-centrifuge).

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
| `file` | none | CSV file of the particles at the end of the run: `id`, `x`, `y` (`z`), the velocity `v_x`, `v_y` (`v_z`) of solid particles, then the quantities, by id |
| `capacity` | the number of particles | Particles a rank can hold |

### Seeding

Every particle is seeded at the start of the run, from `points`, a
`lattice`, or both. Ids follow the seeds: the `points` first, in order, then
the lattice with `x` varying fastest. A seed outside the mesh is an error,
naming the point; a seed on a face between two cells goes to the one with the
lowest global id, so every rank count seeds the same particles in the same
cells.

To follow particles released along a line, put `n = 1` along the other
directions (`n = [1, 50]` seeds 50 particles on the vertical line through the
middle of `lo` and `hi`); to release them in a region, cover its bounding box
with a lattice and keep the particles inside afterward (the
[quantities](#per-particle-quantities) with `in(name)` tell which particles
were where). Seeding during the run, from injectors with rates, is not
available yet ([#759](https://github.com/MatthewBonanni/mallard/issues/759)).

### Restarts

Restart files carry the particles, and a restarted run continues their
trajectories exactly, on any number of ranks; the seeds of `[particles]` are
then not used. A restart file written without particles (by a run without
`[particles]`) seeds them, which adds tracers to a developed flow: run the
flow to its statistically steady state, then restart from it with
`[particles]`. A `remap` start seeds them too.

### Boundaries

- walls and symmetry planes reflect a particle that crossed them within a step;
- inlets, outlets and far-field boundaries remove it;
- periodic boundaries, translational or rotational, carry it to the other side.

Mallard counts the particles that left the domain, that were lost (a
particle that crossed more than a halo layer of a rank in one step, or that
the location walk could not place on a badly distorted mesh) and, for solid
particles, that stuck to walls; restart files carry the three counts. The ids
missing from `file`, or the count of a [probe](#particle-probes) without a
region, give the particles removed.

**Pitfalls.**

- In a troubled cell (a shock, a flame front) and with first-order or MUSCL
  reconstruction, the velocity comes from the cell's linear least-squares fit
  rather than a high-order polynomial.
- A pathline is no more accurate than the flow it moves in: with first-order
  fluxes the pathlines converge at first order, although the particles read a
  second-order fit ([accuracy](#accuracy)).
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
[`particle_settling`](https://matthewbonanni.github.io/mallard/docs/examples/#particle-settling),
[`particle_bounce`](https://matthewbonanni.github.io/mallard/docs/examples/#particle-bounce),
[`particle_centrifuge`](https://matthewbonanni.github.io/mallard/docs/examples/#particle-centrifuge).

A solid particle is a small rigid sphere with a velocity of its own, which
drag pulls toward the gas velocity and gravity (less the buoyancy of the gas it
displaces) accelerates:

$$\frac{d\mathbf{v}}{dt} = \frac{\mathbf{u} - \mathbf{v}}{\tau} + \mathbf{g}\left(1 - \frac{\rho}{\rho_p}\right),
\qquad \tau = \frac{\rho_p d^2}{18 \mu f(Re_p)}, \qquad Re_p = \frac{\rho |\mathbf{u} - \mathbf{v}| d}{\mu}.$$

The gas state at the particle (`u`, `rho`, `mu` at its temperature) comes from
the reconstruction polynomial, as for tracers, and position and velocity are
part of the time integrator's state: trajectories converge at the
integrator's order. `f` is the drag correction of the [drag law](#drag-laws).
The particles do not act on the gas. Of the forces on a sphere, Mallard keeps
drag, gravity and buoyancy, the ones that matter for particles much denser
than the gas (droplets, dust, ash, sand, soot in air); for bubbles or
particles of a density near the gas's, the pressure-gradient, added-mass and
history forces it leaves out matter as much
([theory](../theory/particles.md#solid-particles)).

| Key | Default | Meaning |
|---|---|---|
| `type` | required | `"solid"` |
| `diameter` | required | Sphere diameter |
| `density` | required | Material density `rho_p` |
| `drag` | `"schiller_naumann"` | `"stokes"` or `"schiller_naumann"` |
| `gravity` | `[source] gravity`, else none | Gravity on the particles |
| `velocity` | the gas velocity at each seed | Velocity of every seeded particle |

The other keys of [`[particles]`](#tracers-particles) (seeds, `file`,
`capacity`) are the same. Particle expressions read the particle velocity as
`V_X`, `V_Y`, `V_Z`. Restart files carry the velocities; tracers restarted as
solid particles start at the gas velocity.

`gravity` in `[particles]` acts on the particles only. `[source] gravity`
acts on the gas too (a hydrostatic pressure gradient builds up in it) and on
the particles, unless `[particles] gravity` replaces it; for particles
settling in a small box of gas at rest, gravity on the particles alone is
simpler and the same to within the hydrostatic variation of the gas density.

### Drag laws

| `drag` | `f(Re_p)` | Use |
|---|---|---|
| `"stokes"` | 1 | Creeping flow, $Re_p \lesssim 0.1$; exact analytic solutions for verification |
| `"schiller_naumann"` (default) | `1 + 0.15 Re_p^0.687` below `Re_p = 1000`, then `0.44 Re_p / 24` (`C_D = 0.44`) | Rigid spheres up to $Re_p \approx 800$, within 5% of measured drag |

Stokes drag is 5% low at $Re_p = 0.2$ and 13% low at $Re_p = 1$: use it only
where $Re_p$ stays well below 1, or to compare with analytic solutions.
Schiller-Naumann is the safe default; its $Re_p$ uses the slip velocity of
each stage, so $\tau$ changes as the particle speeds up or slows down.

![Terminal velocity of water droplets and glass beads in air against their diameter: Stokes' law and the Schiller-Naumann correlation, with Mallard's settling velocities](../images/guide/particle_terminal_velocity-light.svg#gh-light-mode-only)
![Terminal velocity of water droplets and glass beads in air against their diameter: Stokes' law and the Schiller-Naumann correlation, with Mallard's settling velocities](../images/guide/particle_terminal_velocity-dark.svg#gh-dark-mode-only)

*Terminal velocities in air at 300 K: Stokes' law (dashed) and the
Schiller-Naumann correlation (solid), with the settling velocities of Mallard
runs of the [`particle_settling`](https://matthewbonanni.github.io/mallard/docs/examples/#particle-settling)
box (dots). Above 50 µm the drag at the terminal velocity exceeds Stokes'.*

### The stiff limit

The relaxation time `tau` can be far below the flow's time step (a 1 µm
droplet in air relaxes in a few microseconds). A particle whose `tau` is below
the step `dt` is in its stiff limit: it moves at its equilibrium velocity
`u + tau g'` and its velocity relaxes to it within the step, so tiny particles
follow the gas instead of making the run unstable, and the step never depends
on the particles.

- Settling in gas at rest or in uniform flow is exact in the stiff limit
  (the `smoke.toml` input of
  [`particle_settling`](https://matthewbonanni.github.io/mallard/docs/examples/#particle-settling)).
- Where the gas accelerates, the equilibrium velocity misses the particle's
  lag behind the gas, `tau Du/Dt`: an error first order in `tau`, small for
  the particles the switch applies to.
- Just below the switch (`dt` from `tau / 2` down) the integrator keeps its
  order; particles with `tau` within a factor of a few of `dt` are the least
  accurate. When their inertia matters, lower `dt` until `dt < tau / 2` for
  them ([verification](../verification/index.md#lagrangian-particles)).

### Walls

Walls and symmetry planes rebound solid particles elastically;
`[[particles.walls]]` sets other interactions per boundary zone. A rebound
reverses the normal velocity times the restitution coefficient `e_n` and
scales the tangential velocity by `e_t`; a sticking wall removes the particle
and counts it as deposited.

```toml
[[particles.walls]]
zones = ["bottom"]
restitution = 0.8              # e_n
tangential_restitution = 0.9   # e_t

[[particles.walls]]
zones = ["top"]
interaction = "stick"
```

| Key | Default | Meaning |
|---|---|---|
| `zones` | required | Wall or symmetry boundary zones |
| `interaction` | `"rebound"` | `"rebound"` or `"stick"` |
| `restitution` | 1 | `e_n` in [0, 1] (rebound) |
| `tangential_restitution` | 1 | `e_t` in [0, 1] (rebound) |

`e_n` is the ratio of the normal speeds after and before the impact: 1 for an
elastic bounce, 0.9 and above for hard beads on a hard wall, less for soft,
rough or small particles, 0 for a particle that slides along the wall; each
bounce keeps $e_n^2$ of the normal kinetic energy. `e_t` below 1 models friction at the impact. A
zone that is not a wall or symmetry boundary, and `[[particles.walls]]` with
tracers, are errors.

![Particles thrown sideways in a closed box: Mallard's trajectories against the exact ones, bouncing off the floor with e_n = 0.8 and off the right wall elastically](../images/guide/particle_bounce-light.svg#gh-light-mode-only)
![Particles thrown sideways in a closed box: Mallard's trajectories against the exact ones, bouncing off the floor with e_n = 0.8 and off the right wall elastically](../images/guide/particle_bounce-dark.svg#gh-dark-mode-only)

*The [`particle_bounce`](https://matthewbonanni.github.io/mallard/docs/examples/#particle-bounce)
example: Mallard's particles (dots) against their exact trajectories
(lines).*

**Pitfalls.**

- Solid particles need a viscous single gas (`type = "navier_stokes"` in
  `[physics]`); mixtures are refused for now.
- A bounce happens at the end of the step in which the particle crossed the
  wall, so its timing is first order in the step: under gravity, the particle
  leaves the wall with a velocity error of up to `(1 + e_n) g dt`.
- The stiff limit is first order in `tau / dt` for particles near it: when
  the inertia of particles with `tau` near `dt` matters, lower `dt`.
- The drag has no slip correction for particles near the mean free path of
  the gas (Cunningham's): in air at 1 bar, particles of 1 µm settle 17%
  faster than Mallard's, of 0.1 µm three times as fast.
- Droplets above about 1 mm in air are not spheres (they flatten), and
  particles of a few cells' size are not point particles: the gas at their
  center is not the flow around them.

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
- the gas: `RHO`, `U_X`, `U_Y` (`U_Z`), `P` and, for a single gas, `T`,
  evaluated as the particle's velocity is, from the reconstruction polynomial
  of its cell;
- the particle velocity `V_X`, `V_Y` (`V_Z`), for tracers the gas velocity;
- the quantities listed before this one;
- `in(name)`: 1 inside the region `name` of [regions](geometry.md#named-regions)
  and 0 outside, smooth across the region's `edge`.

Some quantities worth knowing:

| Quantity | What it measures |
|---|---|
| `integral = "1"` | The particle's age |
| `integral = "in(name)"` | Total residence time in a region |
| `integral = "hypot(V_X, V_Y)"` | Path length (2D) |
| `integral = "hypot(V_X - U_X, V_Y - U_Y)"` | Time-integrated slip of a solid particle |
| `max = "hypot(U_X - V_X, U_Y - V_Y) * RHO * 20e-6 / 1.8e-5"` | Peak particle Reynolds number (here `d` = 20 µm, `mu` = 1.8 × 10⁻⁵) |
| `integral = "T"`, `while = "T > 1200"` | Thermal exposure above a threshold |
| `first_time = "x > 0.3"` | Arrival time at a plane |

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

A probe without a region and with no fields counts the particles still in
the domain: with `[[particles.walls]]` that stick, the decrease is the
deposition history.

**Pitfalls (quantities and probes).**

- An expression reading `T` in a mixture is an error: particle expressions
  have the gas temperature for a single gas only.
- `in(name)` and probe regions are regions of primitives that do not move
  (boolean combinations included); at most 4 regions in all particle
  expressions, 4 probes with regions, and 16 quantities.
- Probes count particles by their positions only, at the sampling step; the
  residence time of visits (per visit, total, visit counts) is a quantity:
  `integral = "in(name)"` gives the total.

## Accuracy

The particles read the gas from the reconstruction's polynomial at their
positions and move with the flow's time integrator, so their error has the
flow's orders:

| | Order | Measured ([verification manual](../verification/index.md#lagrangian-particles)) |
|---|---|---|
| Gas at the particle, TENO of order `k` | `k` | 2.95 to 5.23 for TENO3 to TENO5 on quadrilaterals, triangles, hexahedra and tetrahedra |
| Gas at the particle, `FO` and `MUSCL` | 2 | 1.96 to 3.07 (the linear least-squares fit) |
| Pathlines in space | the flow's: 1 (`FO`), 2 (`MUSCL`), `k` (TENO) | 0.92 to 0.94 (`FO`), 2.28 to 3.19 (`MUSCL`), 2.94 to 5.09 (TENO3 to TENO5) |
| Trajectories in time | the integrator's: 1 (`FE`), 3 (`SSPRK3`, `RDPK3Sp35`), 4 (`RK4`, `SSPRK54`, `CK45`) | 1.02 to 1.05, 2.97 to 3.14 and 3.96 to 4.22: tracers, drag relaxation, settling and Schiller-Naumann response, also at `dt = tau / 2` |
| Stiff limit (`tau < dt`) | 1 in `tau` | 1.00 |
| A bounce under gravity | 1 in `dt` | 1.00 |

![Errors of solid particles' trajectories against the number of steps, by integrator](../images/verification/particles-time-light.svg#gh-light-mode-only)
![Errors of solid particles' trajectories against the number of steps, by integrator](../images/verification/particles-time-dark.svg#gh-dark-mode-only)

In practice:

- Choose the reconstruction for the flow; the particles inherit its order.
  In the vortex study, TENO5 on 32² cells is as accurate as MUSCL on 64²,
  and the gap grows with the accuracy asked for.
- The time step of a stable flow (CFL below 1) moves particles less than a
  cell per step and resolves their relaxation unless `tau < dt`. Check a
  trajectory statistic at two values of `cfl` when it matters.
- Particle counts at a wall (deposition) and in regions are discrete:
  converge them in the number of particles as well as the mesh.

## Performance

- **Cost.** Each particle costs an evaluation of its cell's polynomial per
  stage and a walk of a cell or two per step, independent of the other
  particles. Kernels run over every slot, so a `capacity` far above the number
  of particles costs memory and time.
- **Linear fits.** With `FO`, `MUSCL` or troubled TENO cells, the cells
  holding particles (only those) compute a least-squares gradient each stage.
- **Several ranks.** Particles that change rank go through the host at every
  step, which takes distributed runs with particles out of CUDA graphs. On one
  GPU, steps with particles are graphed.
- **Output.** The CSV file and the restart files gather every particle on
  rank 0 (restart files on every rank): fine for thousands to hundreds of
  thousands of particles, not millions, until parallel particle output
  ([#753](https://github.com/MatthewBonanni/mallard/issues/753)).
- **Probes** sample on the host: a probe every step of a GPU run stops the
  device at every step; sample every 10 to 100 steps.

## Checklist

1. Estimate $\tau$ and $St$ for your particles and flow ([table](#which-particles)):
   tracers for $St \ll 1$ without settling, solid particles otherwise.
2. Solid particles: `[physics] type = "navier_stokes"` with the right `mu`
   (its temperature dependence included), `diameter`, `density`, `gravity`.
3. Check $Re_p$ (a `max` quantity, above): above 0.1 keep
   `schiller_naumann`; above 800 the particles are outside the correlation's
   range.
4. Walls: `restitution` from experiments on your materials, or `stick` for
   deposition.
5. Converge the statistic you report in `cfl` (time), the mesh (space) and the
   number of particles.

*Design notes:* [particles](../design/particles.md).
