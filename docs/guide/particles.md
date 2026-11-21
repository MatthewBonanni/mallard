# Particles

Mallard tracks Lagrangian particles through the flow, on the device, in 2D and
3D, serial and on many ranks. This release has massless tracers: each
particle moves with the gas velocity at its position, for pathlines,
residence times and other Lagrangian analyses. Each particle can carry
[quantities](#per-particle-quantities) along its path (time integrals, extrema,
first times, turns about an axis), and [probes](#particle-probes) give
statistics of the particles in a region. Inertial particles, sprays, injection
during the run and particle output beyond the final positions follow in later
releases ([#291](https://github.com/MatthewBonanni/mallard/issues/291)).

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
[`tracer_vortex`](https://matthewbonanni.github.io/mallard/docs/examples/#tracer-vortex).

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
| `file` | none | CSV file of the particles at the end of the run: `id`, `x`, `y` (`z`), then the quantities, by id |
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

- An expression reading `T` in a mixture is an error: particle expressions
  have the gas temperature for a single gas only.
- `in(name)` and probe regions are regions of primitives that do not move
  (boolean combinations included); at most 4 regions in all particle
  expressions, 4 probes with regions, and 16 quantities.
- Probes count particles by their positions only, at the sampling step; the
  residence time of visits (per visit, total, visit counts) is a quantity:
  `integral = "in(name)"` gives the total.

*Design notes:* [particles](../design/particles.md).
