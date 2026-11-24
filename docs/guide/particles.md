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
[two-way coupling](#two-way-coupling) solid particles act on the gas, and
[dispersed-phase fields](#dispersed-phase-fields) put them on the mesh.
[Snapshots](#particle-snapshots) write every particle, with the mesh or
alone. Sprays and injection during the run follow in later releases
([#291](https://github.com/MatthewBonanni/mallard/issues/291)).

- [Which particles?](#which-particles)
- [Tracers](#tracers-particles), [seeding](#seeding), [restarts](#restarts) and [boundaries](#boundaries)
- [Solid particles](#solid-particles): [drag](#drag-laws), [stiff particles](#stiff-particles), [walls](#walls)
- [Per-particle quantities](#per-particle-quantities) and [particle probes](#particle-probes)
- [Accuracy](#accuracy), [performance](#performance) and a [checklist](#checklist)

The models are in the theory manual's
[Lagrangian particles](../theory/particles.md), their discretization and its
accuracy in [Numerical methods for particles](../numerics/particles.md); the
measured orders of accuracy in the
[verification manual](../verification/index.md#lagrangian-particles).

## Which particles?

The particles are *point particles*: each is much smaller than the cells and
the smallest eddies around it, sees the gas as undisturbed at its center, and
moves on its own, without collisions with other particles and, unless coupled
[two-way](#two-way-coupling), without acting on the gas (one-way coupling). That describes dilute suspensions, with a
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
| $\ll 1$ (below 0.05) | follows the gas, plus its settling velocity $\tau g$ | `tracer` if settling does not matter; `solid` for settling smoke, fog, fine dust ([stiff particles](#stiff-particles)) |
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

[particles.lattice]
lo = [0.1, 0.1]
hi = [0.9, 0.9]
n = [9, 9]

[[write_data]]
prefix = "tracers"
particle_fields = ["U"]
time_interval = 0.1
```

*Used in:*
[`tracer_vortex`](https://matthewbonanni.github.io/mallard/docs/examples/#tracer-vortex),
[`particle_settling`](https://matthewbonanni.github.io/mallard/docs/examples/#particle-settling),
[`particle_bounce`](https://matthewbonanni.github.io/mallard/docs/examples/#particle-bounce),
[`particle_centrifuge`](https://matthewbonanni.github.io/mallard/docs/examples/#particle-centrifuge),
[`particle_cloud`](https://matthewbonanni.github.io/mallard/docs/examples/#particle-cloud)
and 2 more.

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
missing from a [snapshot](#particle-snapshots), or the count of a
[probe](#particle-probes) without a region, give the particles removed.

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
[`particle_centrifuge`](https://matthewbonanni.github.io/mallard/docs/examples/#particle-centrifuge),
[`particle_cloud`](https://matthewbonanni.github.io/mallard/docs/examples/#particle-cloud),
[`particle_vortex`](https://matthewbonanni.github.io/mallard/docs/examples/#particle-vortex)
and 1 more.

A solid particle is a small rigid sphere with a velocity of its own, which
drag pulls toward the gas velocity and gravity (less the buoyancy of the gas it
displaces) accelerates:

$$\frac{d\mathbf{v}}{dt} = \frac{\mathbf{u} - \mathbf{v}}{\tau} + \mathbf{g}\left(1 - \frac{\rho}{\rho_p}\right),
\qquad \tau = \frac{\rho_p d^2}{18 \mu f}, \qquad Re_p = \frac{\rho |\mathbf{u} - \mathbf{v}| d}{\mu}.$$

The gas state at the particle (`u`, `rho`, `mu` at its temperature) comes from
the reconstruction polynomial, as for tracers, and position and velocity are
part of the time integrator's state: trajectories converge at the
integrator's order. `f` is the drag correction of the [drag law](#drag-laws)
(`drag`), `f = 1` for creeping flow. By default the particles do not act on the gas;
`coupling = "two-way"` makes them ([Two-way coupling](#two-way-coupling)).

**Forces.** Drag, gravity and buoyancy are the forces that matter for
particles much denser than the gas (droplets, dust, ash, sand, soot in air).
For bubbles, particles of a density near the gas's, or particles slipping
through shear, `forces` adds the forces of
[Maxey & Riley (1983)](../references.md#maxey-riley-1983) and lift
([theory](../theory/particles.md#solid-particles)):

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
small) and, unless coupled two-way, does not heat the gas. Particle expressions read it as `T_P`,
[snapshots](#particle-snapshots) write it as `T_P`, and restart files carry it.

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

The other keys of [`[particles]`](#tracers-particles) (seeds,
`capacity`) are the same. Particle expressions read the particle velocity as
`V_X`, `V_Y`, `V_Z`. Restart files carry the velocities and temperatures;
tracers restarted as solid particles start at the gas velocity, and particles
restarted with heat transfer from a file without temperatures at the gas
temperature.

`gravity` in `[particles]` acts on the particles only. `[source] gravity`
acts on the gas too (a hydrostatic pressure gradient builds up in it) and on
the particles, unless `[particles] gravity` replaces it; for particles
settling in a small box of gas at rest, gravity on the particles alone is
simpler and the same to within the hydrostatic variation of the gas density.

### Drag laws

`drag` in `[particles]` chooses the drag correlation, `f = C_D Re_p / 24`
(the drag over Stokes drag):

```toml
[particles]
type = "solid"
drag = "singh"   # "stokes", "schiller_naumann" (default) or "singh"
```

| `drag` | `f` | Reads |
|---|---|---|
| `"stokes"` | 1 | nothing |
| `"schiller_naumann"` (default) | `1 + 0.15 Re_p^0.687` below `Re_p = 1000`, then `0.44 Re_p / 24` (`C_D = 0.44`) | `Re_p` |
| `"singh"` | the correlation of [Singh et al. (2022)](../references.md#singh-2022), Appendix C ([theory](../theory/particles.md#drag-laws)) | `Re_p`, the slip Mach number `M = abs(u - v) / c`, the Knudsen number `Kn` and `T_p / T` |

The gas's `mu` (at its temperature), `gamma` and `R` come from `[physics]`.
For `"singh"` the Knudsen number is the gas's mean free path over the
diameter, `Kn = mu / (rho c d) sqrt(gamma pi / 2) = sqrt(gamma pi / 2) M / Re_p`,
and `T_p / T` is 1 without `specific_heat`. Each stage evaluates `f` from its
own slip velocity, so `tau` changes as the particle speeds up or slows down.

**Where each law is valid.** Outside its ranges a law still runs, but its drag
is wrong by more than the error given.

| `drag` | Particle Reynolds number $Re_p$ | Slip Mach number $M$ | Knudsen number $Kn$ | Neglects | Sources |
|---|---|---|---|---|---|
| `"stokes"` | $\lesssim 0.1$: 3% low at 0.1, 5% at 0.2, 13% at 1 | $\ll 1$ | $\lesssim 0.01$ | the inertia of the gas around the sphere (its wake), compressibility, slip at the surface | [Stokes (1851)](../references.md#stokes-1851) |
| `"schiller_naumann"` | $\lesssim 800$ within 5% of the standard drag curve; `C_D = 0.44` from 1000 to the drag crisis near $2 \times 10^5$ | $\lesssim 0.3$ | $\lesssim 0.01$ | compressibility and shocks, slip at the surface and rarefaction, the drag crisis | [Schiller & Naumann (1933)](../references.md#schiller-naumann-1933), [Clift, Grace & Weber (1978)](../references.md#clift-grace-weber-1978) |
| `"singh"` | creeping flow to $\approx 10^4$ (laminar boundary layer) | 0 to 10 (fitted from 0.1); above 5 a calorically perfect gas | 0 (continuum) to free-molecular flow | real-gas effects behind strong shocks, boundary-layer turbulence and the drag crisis; least accurate in the transition regime, $0.1 \lesssim Kn \lesssim 10$ | [Singh et al. (2022)](../references.md#singh-2022): 7.6% relative L2 error against experiments and DSMC |

All three are the drag of an isolated, smooth, rigid, non-rotating sphere in
steady, uniform flow: none accounts for neighbors, turbulence of the gas at
the particle's scale, rotation or non-spherical shapes.

**Choosing.**

- `"schiller_naumann"`, the default, for particles in a gas at ordinary
  densities whose slip is subsonic: droplets, dust and sand above a few
  microns in air at 1 bar.
- `"stokes"` where $Re_p$ stays well below 1, or to compare with analytic
  solutions; it is never more accurate than Schiller-Naumann.
- `"singh"` where the particle is not much larger than the mean free path
  ($Kn > 0.01$: below about 7 µm in air at 1 bar, larger in a low-pressure
  gas) or its slip is compressible ($M > 0.3$: shocks, nozzles, high-speed
  dusty flow). In continuum, incompressible flow it is within about 6% of
  Schiller-Naumann up to $Re_p = 1000$ and costs more.

Check $Re_p$, and for `"singh"` $M$ and $Kn$, along the paths with a `max`
quantity ([per-particle quantities](#per-particle-quantities)).

![Terminal velocity of water droplets and glass beads in air against their diameter: Stokes' law and the Schiller-Naumann and Singh et al. correlations, with Mallard's settling velocities](../images/guide/particle_terminal_velocity-light.svg#gh-light-mode-only)
![Terminal velocity of water droplets and glass beads in air against their diameter: Stokes' law and the Schiller-Naumann and Singh et al. correlations, with Mallard's settling velocities](../images/guide/particle_terminal_velocity-dark.svg#gh-dark-mode-only)

*Terminal velocities in air at 300 K and 1 bar (left) and their ratio to
the Schiller-Naumann value (right): Stokes' law (dashed), the
Schiller-Naumann correlation (solid) and Singh et al.'s (dotted), with the
settling velocities of Mallard runs of the
[`particle_settling`](https://matthewbonanni.github.io/mallard/docs/examples/#particle-settling)
box (dots: Schiller-Naumann on the left, Singh et al. on the right). Above
50 µm the drag at the terminal velocity exceeds Stokes'; below a few microns
the particles slip through the gas ($Kn$ = 0.066 at 1 µm) and settle up to
12% faster with Singh et al.'s drag; between, Singh et al.'s continuum drag
is up to 6% above Schiller-Naumann's.*

### Stiff particles

The relaxation time `tau` can be far below the flow's time step (a 1 µm
droplet in air relaxes in a few microseconds), where an explicit integrator
would be unstable. A particle whose `tau` is below the step `dt` relaxes
exponentially within the step: its velocity is integrated exactly for a gas
velocity and force that vary linearly over the step, so tiny particles follow
the gas at their equilibrium velocity `u + tau g'` instead of making the run
unstable, particles with `tau` near `dt` keep their inertia, and the step
never depends on the particles. The same holds for the temperature's `tau_T`.

- Settling in gas at rest or in uniform flow is exact (the `smoke.toml` input
  of [`particle_settling`](https://matthewbonanni.github.io/mallard/docs/examples/#particle-settling)).
- Where the gas accelerates, the trajectories still converge in the step, on
  both sides of `dt = tau`
  ([verification](../verification/index.md#lagrangian-particles)).

### Walls

Walls and symmetry planes rebound solid particles elastically;
`[[particles.walls]]` sets other interactions per boundary zone. A rebound
reverses the normal velocity times the restitution coefficient `e_n` and
scales the tangential velocity by `e_t`; a sticking wall removes the particle
and counts it as deposited. The bounce is located within the step: the
particle reaches the wall at the time its motion over the step puts it there,
rebounds, and moves on for the rest of the step, so the timing of bounces
does not depend on the step.

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
- The bounce of a particle with `tau` below the step happens at the end of
  the step: such a particle follows the gas within the step.
- The located bounce takes the gas velocity and force at the start of the
  step, so it is exact for drag in a gas that is uniform over the step's
  travel and for constant acceleration; in a strongly varying gas it is still
  first order in the step.
- `temperature` without `specific_heat` is an error, as is an expression
  reading `T_P` in a mixture.
- Stokes and Schiller-Naumann drag have no slip correction for particles near
  the mean free path of the gas: in air at 1 bar, particles of 1 µm settle
  12% faster with `"singh"` than with them, of 0.1 µm several times as fast.
  Use `"singh"` above `Kn = 0.01` ([drag laws](#drag-laws)).
- Droplets above about 1 mm in air are not spheres (they flatten), and
  particles of a few cells' size are not point particles: the gas at their
  center is not the flow around them.

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

![A cloud of heavy particles falling through gas at rest: coupled one way it falls at a single particle's terminal velocity, coupled two ways it drags the gas down and falls more than twice as fast](../images/guide/particle_cloud_coupling-light.svg#gh-light-mode-only)
![A cloud of heavy particles falling through gas at rest: coupled one way it falls at a single particle's terminal velocity, coupled two ways it drags the gas down and falls more than twice as fast](../images/guide/particle_cloud_coupling-dark.svg#gh-dark-mode-only)

*The [`particle_cloud`](https://matthewbonanni.github.io/mallard/docs/examples/#particle-cloud)
example, coupled one way and two ways (Mallard runs). Left: the cloud's mean
fall velocity from its probe. Coupled one way, each particle reaches the
terminal velocity `tau g` = 0.1 of a single particle; coupled two ways, the
cloud, as heavy as the gas within it, drags that gas along and falls at up
to 0.27. Right: the gas velocity at `t` = 2 and the parcels, whose wake
sheds a vortex pair and bends the cloud into an arc.*

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
name = "T_CAVITY"
integral = "in(cavity)"          # time spent in the region cavity of [regions]

[[particles.quantities]]
name = "T_MAX"
max = "T"

[[particles.quantities]]
name = "T_HOT"
first_time = "T > 1500"

[[particles.quantities]]
name = "TURNS"
winding = {point = [0.052, -0.063, 0.0], axis = [0.0, 0.0, 1.0]}
```

*Used in:*
[`tracer_vortex`](https://matthewbonanni.github.io/mallard/docs/examples/#tracer-vortex),
[`particle_vortex`](https://matthewbonanni.github.io/mallard/docs/examples/#particle-vortex),
[`particle_step`](https://matthewbonanni.github.io/mallard/docs/examples/#particle-step).

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
- the particle temperature `T_P`, for a single gas, the gas temperature for
  particles without heat transfer;
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

Quantities are written by [snapshots](#particle-snapshots) that list them, continue across restarts (a
quantity the restart file does not have starts afresh), move with their
particles between ranks, and are bitwise the same on any number of ranks.

## Particle probes

```toml
[[particles.probes]]
name = "core"
region = {type = "disk", center = [0.25, 0.25], radius = 0.12}
fields = ["TURNS", "hypot(U_X, U_Y)"]
interval = 100
```

*Used in:*
[`tracer_vortex`](https://matthewbonanni.github.io/mallard/docs/examples/#tracer-vortex),
[`particle_cloud`](https://matthewbonanni.github.io/mallard/docs/examples/#particle-cloud),
[`particle_vortex`](https://matthewbonanni.github.io/mallard/docs/examples/#particle-vortex),
[`particle_step`](https://matthewbonanni.github.io/mallard/docs/examples/#particle-step).

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

- An expression reading `T` or `T_P` in a mixture is an error: particle
  expressions have temperatures for a single gas only.
- `in(name)` and probe regions are regions of primitives that do not move
  (boolean combinations included); at most 4 regions in all particle
  expressions, 4 probes with regions, and 16 quantities.
- Probes count particles by their positions only, at the sampling step; the
  residence time of visits (per visit, total, visit counts) is a quantity:
  `integral = "in(name)"` gives the total.

## Particle snapshots

```toml
[[write_data]]
prefix = "flow"
time_interval = 0.5
fields = ["RHO", "U"]
particle_fields = ["V", "T_P", "TURNS"]
```

*Used in:*
[`tracer_vortex`](https://matthewbonanni.github.io/mallard/docs/examples/#tracer-vortex),
[`particle_cloud`](https://matthewbonanni.github.io/mallard/docs/examples/#particle-cloud),
[`particle_vortex`](https://matthewbonanni.github.io/mallard/docs/examples/#particle-vortex).

`particle_fields` in a [`[[write_data]]`](output.md#solution-snapshots-write_data)
entry writes the particles at each of its snapshots: their positions, `ID`,
and the variables listed (`RHO`, `U`, `P`, `T` of the gas at each particle,
`V`, `T_P`, and quantities by name). With `fields` too, the mesh and the
particles are two blocks of one series, which ParaView 6.0 and later open;
`particles = "separate"` writes them as two series instead, for ParaView 5.12
and later; `region` keeps the particles in a volume. Files, layout and
pitfalls: [Particle snapshots](output.md#particle-snapshots). Restart files
carry every particle and quantity whatever the snapshots write.

![Three particle snapshots of the particle_vortex example over the gas speed: heavy particles seeded in a vortex core are flung out of it and heat toward the gas temperature](../images/guide/particle_snapshots-light.svg#gh-light-mode-only)
![Three particle snapshots of the particle_vortex example over the gas speed: heavy particles seeded in a vortex core are flung out of it and heat toward the gas temperature](../images/guide/particle_snapshots-dark.svg#gh-dark-mode-only)

*Particle snapshots (`particle_fields = ["T_P", "SLIP_MAX"]`) of the
[`particle_vortex`](https://matthewbonanni.github.io/mallard/docs/examples/#particle-vortex)
example at `t` = 0, 3 and 6, over the gas speed of a mesh snapshot at the same
times (a Mallard run): the heavy particles seeded in one vortex are flung out
of it into the strain between the vortices and heat from 0.4 toward the gas's
1/1.4.*

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
| Stiff particles (`tau < dt`), exponential | 3, exact for a gas velocity and force linear over the step | 2.90 (position) and 2.79 (velocity), RK4 at `dt / tau` = 12.5 to 3.1 |
| A bounce | exact for drag in uniform gas and constant acceleration; else 1 in `dt` | round-off (`ABounceIsLocatedWithinTheStep`) |

![Errors of solid particles' trajectories against the number of steps, by integrator](../images/verification/particles-time-light.svg#gh-light-mode-only)
![Errors of solid particles' trajectories against the number of steps, by integrator](../images/verification/particles-time-dark.svg#gh-dark-mode-only)

In practice:

- Choose the reconstruction for the flow; the particles inherit its order.
  In the vortex study, TENO5 on 32² cells is as accurate as MUSCL on 64²,
  and the gap grows with the accuracy asked for.
- The time step of a stable flow (CFL below 1) moves particles less than a
  cell per step and resolves their relaxation, exponentially when `tau < dt`. Check a
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
- **Output.** [Snapshots](#particle-snapshots) write each rank's particles
  in parallel. Restart files carry every particle on every rank: fine for
  thousands to hundreds of thousands of particles, not millions.
- **Probes** sample on the host: a probe every step of a GPU run stops the
  device at every step; sample every 10 to 100 steps.

## Checklist

1. Estimate $\tau$ and $St$ for your particles and flow ([table](#which-particles)):
   tracers for $St \ll 1$ without settling, solid particles otherwise.
2. Solid particles: `[physics] type = "navier_stokes"` with the right `mu`
   (its temperature dependence included), `diameter`, `density`, `gravity`.
3. Choose the [drag law](#drag-laws): check $Re_p$ (a `max` quantity,
   above), and the Knudsen and slip Mach numbers: `singh` above
   $Kn = 0.01$ or $M = 0.3$; above $Re_p = 800$ Schiller-Naumann is outside
   its range.
4. Walls: `restitution` from experiments on your materials, or `stick` for
   deposition.
5. Converge the statistic you report in `cfl` (time), the mesh (space) and the
   number of particles.

*Design notes:* [particles](../design/particles.md).
