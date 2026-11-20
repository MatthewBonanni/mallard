# Particles

Mallard tracks Lagrangian particles through the flow, on the device, in 2D and
3D, serial and on many ranks. This first release has massless tracers: each
particle moves with the gas velocity at its position, for pathlines,
residence times and other Lagrangian analyses. Inertial particles, sprays,
injection during the run and particle output beyond the final positions follow
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
| `file` | none | CSV file of the positions at the end of the run: `id`, `x`, `y` (`z`), by id |
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

*Design notes:* [particles](../design/particles.md).
