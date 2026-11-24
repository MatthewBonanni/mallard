# Lagrangian particles

Design of the particle core (#766), the first layer of the Euler-Lagrange model
of #291. User docs: [Particles](../guide/particles.md); equations and their
accuracy: [theory](../theory/particles.md) and
[verification](../verification/index.md#lagrangian-particles). Later layers (#756-#765,
output in #753) build on the decisions here; what this layer leaves to them is
listed at the end.

## Goals and constraints

- Millions of live particles per GPU, seeded continuously, on mixed-element
  unstructured meshes, 2D and 3D.
- The gas fields at a particle come from the reconstruction's own polynomial in
  the particle's cell, so that trajectories keep the spatial order of the flow.
- Trajectories keep the flow's order in time.
- Bitwise identical results on any rank count, with NCCL or MPI, graphed or
  not, and across restarts.
- Everything on the step path runs inside the step's CUDA graph.
- A run without `[particles]` does exactly what it did before: no allocation,
  no kernel, no message.

## Storage

Particles live on the device as a structure of arrays over a fixed number of
*slots* per rank (`[particles] capacity`):

| Array | Per slot | |
|---|---|---|
| `x` | `width` reals | position, in the frame of the cell that holds it, the velocity of solid particles (#757), then the integrated quantities (#756) |
| `aux` | `n_aux` reals | extrema and first times of quantities (#756) |
| `id` | `uint64_t` | global id in injection order; `EMPTY` (all bits set) marks a free slot |
| `cell` | `uint32_t` | local cell holding the particle at the start of the step |

Fixed capacity is what makes the step graphable: no kernel allocates, every
launch has a size the graph recorded, and adding or removing a particle writes
a slot. Kernels run over every slot and skip the empty ones. Later layers add
per-particle arrays (velocity, temperature, diameter, mass, the quantities of
#756) as further arrays of the same length; positions stay the only one the
integrator sees until a layer needs more (#757 adds the velocity to the
integrated block).

Removing and inserting never moves a live particle: free slots are found by a
prefix sum over `id == EMPTY`, and the *k*-th incoming particle takes the
*k*-th free slot. Slot order never enters a result: particles do not interact
in this layer, and output sorts by id.

The default capacity is the global particle count, so that every rank can hold
every particle and migration never runs out of slots. Injection (#759) will
size it from the injectors' rates; load balancing (#763) bounds it.

## The particle block of the State

`State` (`src/common/state.h`) gains a third block, `particles`, next to the
flow block and the species: a flat view of `width * capacity` reals, `width`
per slot (the position, then what later layers integrate). Every time
integrator updates its states only through `axpby`, `lerp`, `deep_copy` and the
low-storage stage kernels, and all of them now loop over the three blocks, so
every integrator (FE, RK4, SSPRK3, SSPRK54, CK45, RDPK3Sp35) advances the
particles at its own order, with the same stage times as the flow, and the
error controller's rejected steps restore the particles with the flow.
Without particles the block is empty and its loops are not launched, exactly as
the species block of a single gas.

The right-hand side of the block is the particle velocity: after the flow's
`calc_rhs` of a stage, `Particles::velocity` evaluates the gas velocity at each
stage position from the polynomial of the cell that held the particle at the
start of the step (below), in that cell's frame. No stage walks across faces.

Alternatives considered:

- *A separate particle integrator after the flow step*, with the gas velocity
  interpolated in time between the start and the end of the step: first or
  second order in time whatever the flow's integrator, and a second copy of
  every integrator's logic.
- *Particles as extra "cells" of the flow block*: mixes per-cell and
  per-particle sizes in every flow kernel.

## Location in unstructured cells

A particle moves less than a cell per step (CFL below 1), so it is located by
walking across faces from the cell it was in: each step of the walk leaves
through the face whose plane the point lies farthest outside of, and stops when
the point is inside every face plane of the cell (to a relative tolerance of
the cell size). On convex cells this terminates; a step limit (64) catches the
rare cycle on distorted meshes, and the particle is then counted as lost.
Faces are the mesh's planar faces (centroid and normal), the same ones the flux
uses, so location matches the finite-volume cells exactly on straight-sided
meshes; on curved meshes the planar faces approximate the curved boundary.

Periodic faces carry a shift row (`Mesh::face_shift`): crossing one maps the
position into the neighbor's frame (`PeriodicMaps::forward`/`backward`), so a
particle wraps across translational and rotational periodic boundaries. The
walk runs once per step, after it (`relocate`).

Each face has a particle action, set at setup from its boundary condition:
cross (interior faces), reflect (walls and symmetry planes: the position is
mirrored in the face plane), exit (inlets, outlets, far field: the particle is
removed and counted), or lost (partition faces of the outermost halo layer,
which a particle can only reach by moving more than a halo layer in one step).

Seeds are located once at setup by `locate_points`, the point location of the
probes and statistics (a box search on the host): the owning rank of the cell
containing a seed takes it, ties go to the lowest global cell id, so every rank
count seeds the same particles in the same cells.

## Fields at the particle

The gas velocity at a point comes from the polynomial the reconstruction
already builds:

- With TENO, `FaceReconstruction::point_values` evaluates the cell's central
  polynomial (the one the probes and the transmissive boundaries use) where the
  cell is smooth by TENO's measure and the value admissible.
- Elsewhere (troubled cells, first-order and MUSCL schemes), the cell's linear
  least-squares fit (`LSQVertexGradientFunctor`, the gradient of the viscous
  fluxes and probes), with the cell average where the linear value is not
  admissible. Viscous runs and the hybrid flux compute these gradients in
  every stage, and the particles read them; elsewhere they are computed only
  in the cells that hold particles.

Both evaluate W = [rho, u, p], so the temperature, species and derived fields
of later layers take the same path (#511).

**One cell per step.** Every stage of a step evaluates the polynomial of the
cell that held the particle at the start of the step, also where the stage
position has moved into a neighbor (by less than a cell, at CFL below 1).
Three things follow:

- Within a step the velocity is one smooth function of the position, so the
  integrator keeps its order. Switching polynomials at the faces a stage
  position crosses would make the right-hand side jump by the reconstruction's
  error within a step, and cost a Runge-Kutta scheme its order there.
- A polynomial fitted on a stencil of several cells is as accurate a little
  outside its cell as inside: the pathlines keep the scheme's order in space
  (measured by the tests: fourth order with TENO of `order = 4`, second with
  MUSCL).
- The start cell is always owned (migration runs after each step), so no
  stage reads a halo cell, which has no polynomial on this rank: trajectories
  are bitwise independent of the rank count, and no stage walks, so the
  rotation of vectors across rotational periodic pairs never enters a stage.

The cell changes at the end of a step, when the particle is relocated; the
jump between neighboring polynomials is the reconstruction's error, of the
scheme's order in space.

## Time integration within the step graph

On the step path, per stage: the offsets from the start cells (one kernel),
gradients of the marked cells where the stage computes none (one kernel over
the owned cells), the linear values, `point_values`, the velocity (one kernel
each). After the step, `Particles::relocate` walks each
particle to its new cell, applies reflections and removals, and writes the
position in the new cell's frame. All are plain kernels over fixed sizes,
captured in the step graph (`record_step`) and checked by `StepGraphGuard` in
the tests: no fence, allocation or host copy.

With splitting of the chemistry (Strang), the particles move in the transport
step only, which is the integrator's step; SIMPLER splitting is refused for now.

## MPI migration

After relocation, a particle whose cell is a halo cell belongs to the halo
cell's owner. In this layer the migration is staged through the host: the
emigrants are compacted on the device (a prefix sum), copied to the host, sent
with their id, the global id of their cell and the bits of their position to
the owners (`comm::alltoallv`), and inserted in free slots there. Positions
travel bit for bit, and cells by global id, so a particle continues exactly as
on one rank.

Because it reads counts on the host, migration cannot be captured: distributed
runs with particles take their steps outside the step graph
(`device_steps_unsupported`). The graphable design, for the next layer:

- one send and one receive buffer per neighbor rank, of a fixed capacity, with
  the count in the first word, packed on the device;
- the exchange as NCCL send/receive of the full fixed-size buffers on the step
  stream (MPI as the fallback, bit for bit), like `HaloExchange`;
- insertion into free slots on the device by the same prefix sum;
- an overflow flag in the step's device reduction, which stops the run with an
  error naming `capacity`.

## Determinism

Each particle's update reads only the flow and its own state, so its result
does not depend on the slot it occupies, on the other particles or on the
order of kernels. Its stages read the polynomial of its start cell, which is
owned, and the flow is bitwise independent of the rank count, so trajectories
are too (tested on 2 and 3 ranks). Counters of removed particles are integer
atomics.

## Per-particle quantities (#756)

A quantity is a value each particle follows along its path. Two families,
stored differently:

- **Integrals and windings** are values of the particle block, after the
  position: `width = N_DIM + n_integrated`. Their right-hand side is written by
  the same kernel that writes dx/dt (`particle_rhs`), from the same stage state:
  the stage position, the gas W at it, and the stage values of the earlier
  quantities. Every integrator therefore integrates them with its own stage
  weights, at its order, and a rejected step restores them with the positions.
  A winding is the integral of d(theta)/dt = a . (r x v) / |r_perp|^2 / (2 pi)
  about the axis a through the point, so it is unwrapped by construction.
- **Extrema and first times** are outside the block (`aux`), since an
  integrator cannot combine them linearly. They are updated in the same kernel
  at the first stage of each step (c = 0 for every integrator), whose state is
  the accepted solution at the step's start. Stage states of later stages are
  not on the trajectory to the integrator's order, so sampling them would mix
  lower-order points into a maximum. A rejected step's retry samples the same
  start again, which leaves extrema and first times unchanged.

Expressions are kexpr programs over a fixed variable list: x, t, the gas
W = [rho, u, p] and T = p / (rho R) (single gas), the particle velocity, every
quantity (later ones are placeholders the compiler rejects), and `in(name)` of
the regions read, rewritten to variables before compilation. A region is a
`RegionView` of primitives (its signed distance on the device), at most 4 of
them, held by value in the functor; `in()` is the region's indicator, smooth
across its `edge`. A sharp indicator jumps within a stage, so its integral is
exact only to the crossing's step; locating the crossing within the stage
(root finding on the signed distance along the stage's path) is left for when
a case needs residence times to better than a step.

Quantities travel with their particle in migration (as raw bits, like the
position) and restart files carry each as `PARTICLE_Q_<name>`, so a restarted
run continues them bit for bit; a quantity added on restart starts afresh.

## Solid particles (#757)

Solid particles add their velocity to the block, after the position, and with
heat transfer their temperature: `width = 2 N_DIM + heated + n_integrated`.
The same `particle_rhs` kernel writes dx/dt = v and dv/dt = (u - v) / tau + a
from the stage's gas state at the particle (`particles::Solid`,
`particle_forces.h`), so drag and the forces are integrated at the
integrator's order with the flow, and each stage's drag reads that stage's gas
and velocity. tau = m d^2 / (18 mu f) with m = rho_p (+ rho / 2 with added
mass), the viscosity at the particle's temperature and the drag correlation f.
The force per volume is rho_p g, then -rho g (buoyancy) or -grad p, then
(rho g - grad p) / 2 for added mass, then Saffman lift, all divided by m.
Pressure and velocity gradients are those of the polynomial the particle reads
W from, at its position: `point_values` differentiates the central polynomial
it has just evaluated (no other pass, field or fit), and the linear fit gives
its gradient, the stage's LSQ gradient. The gas evaluation writes them per
particle only when a force reads them. The Basset
force needs the history of the slip, a convolution over the path, and is left
out.

**Drag.** Stokes and Schiller-Naumann are functions of Re_p.
Singh et al. (2022, Appendix C) also reads the slip Mach number, the Knudsen
number Kn = mu / (rho c d) sqrt(gamma pi / 2) and T_p / T. It is written in
Kn rather than M / Re so that it is finite at zero slip, and its
free-molecular term uses its Taylor series below s = 0.3, where the closed
form cancels catastrophically. Below Mach 1 there is no shock and the
post-shock state is the free stream; A_0 = 24 / delta_0^2.

**Heat transfer.** dT_p/dt = (T - T_p) / tau_T with tau_T = rho_p c_p d^2 /
(6 k Nu), Ranz-Marshall Nu and k = mu c_p / Pr of the gas. The temperature is
one more integrated component, so it shares the velocity's integration below.

**Exponential integration.** Particles of a micron in air have tau of
microseconds, far below an acoustic step. An explicit integrator is unstable
for dt / tau above about 2 to 2.8, and limiting dt by the particles would tie
the flow's step to its smallest particle. A particle with h = dt / tau > 1
(per particle and step, tau from the step's start) instead integrates its
relaxation exponentially. Its stage rates are

    dv = [(1 - e^-h)(v_eq0 - v0) + 2 (1 - phi1)(v_eq - v_eq0)] / dt
    dx = v_eq0 + phi1 (v0 - v_eq0) + (1 - 2 (1 - phi1) / h)(v_eq - v_eq0)

with phi1 = (1 - e^-h) / h, v0 and v_eq0 = u + tau a at the step's start and
v_eq = u + tau a of the stage (tau frozen at the start). For any integrator
with sum b_i = 1 and sum b_i c_i = 1/2, which every Runge-Kutta method of
order 2 and above satisfies, these integrate to the exact solution for v_eq
linear in time, so the scheme needs no stage weights from the integrators;
with the explicit rates below the threshold, the order is kept on both sides
(for FE, which has sum b_i c_i = 0, it is first order, as FE is). The
threshold 1 is where the explicit RK4 error overtakes the exponential one in
a decaying-forcing experiment (RK4 crossover near h = 0.9; SSPRK3 favors the
exponential everywhere). The old stiff limit (velocity set to v_eq) was first
order in tau / dt near the switch; at tau = 0.02, RK4 and h = 1.25 the
trajectory error drops from 9e-4 to 3e-6.

The step's start (v, u, a, tau, dt, and T_p, T, tau_T with heat transfer) is
stored per particle in `step_start` by the first stage (c = 0 for every
integrator), since later stages need it and dt itself: in a captured step the
step controller overwrites the device dt before relocation. dt is the host's
step, or inside a captured step the device scalar the step graph computes it
in (`step_dt_device`).

**Walls.** The face action table gains `STICK`, and a per-face pair of
restitution coefficients (`[[particles.walls]]` by zone). A sticking wall
removes a particle that crossed it and counts it in the third removal counter
(deposited). A rebound is located within the step: over the step the particle
is taken to move as dv/dt = (u - v) / tau + a with u, a, tau of the step's
start, for which position and velocity are closed form in phi functions of
z = b / tau. The time b before the step's end at which it reached the wall is
the root of the normal distance traveled, by a few fixed-point passes; the
particle's normal velocity is reversed times e_n and its tangential velocity
scaled by e_t at that time, and it relaxes on for b. That is exact for drag
in a uniform gas and for constant acceleration, so bounce timing is
independent of dt there. A particle with tau < dt follows the gas within the
step and is reflected at the step's end. Across rotational periodic faces u
and a rotate with the position and velocity (`Walker::cross` with three
vectors).

**Starting state.** Seeded particles start at the gas velocity (and with heat
transfer the gas temperature) at their seed unless `velocity` (`temperature`)
is given: before the first step the solver samples the gas at the particles
(`update_cell_W`, `Particles::start_from_gas`), once, on the host's step path
rather than in a captured step.

Restart files carry the velocities as `PARTICLE_V`, the temperatures as
`PARTICLE_T` and three removal counts (older files with two are read with zero
deposited).

## Particle probes (#756)

A probe samples on the host after a step it is due at: the gas at the
particles of the current state (the same path as the stage evaluation,
`Particles::sample`, after `update_cell_W`), then one kernel writes each
field's value per slot, NaN outside the probe's region. Counts, sums and sums
of squares are `ExactSums` (fixed point), so the statistics are bitwise the
same on any rank count; minima and maxima are order independent. Steps decided
on the device stop at the probes' steps (`step_horizon`), so device steps stay
on for runs with probes. Staging the reductions in the step graph, as the gas
probes do, would let the device run past them; it matters once probes sample
every few steps of a large run.

## Restart files

Restart files carry the particles as named arrays (version 6), written by rank
0 sorted by id (every rank gathers them, since each lays out the header): `PARTICLE_ID`, `PARTICLE_CELL` (the global id of the cell
holding each particle), `PARTICLE_X` (positions, exact in double precision),
`PARTICLE_V` (velocities of solid particles), `PARTICLE_T` (their temperatures, with heat transfer), `PARTICLE_Q_<name>` (quantities)
and `PARTICLE_REMOVED` (the counts of particles that left the domain, that
were lost, and that stuck to walls). The device copies them into pinned host mirrors with the other restart
data (`Particles::stage_output`, capturable), so a restart staged by the step
graph holds the particles of its own step. On a restart each rank takes the
particles of the cells it owns, so the file restarts on any rank count, and a
restarted run continues bit for bit. A restart file without particles (one
written by a run without `[particles]`) seeds them from `[particles]`, which
adds tracers to a developed flow; `type = "remap"` also seeds them (relocating
a file's particles onto another mesh is #541).

## Output

This layer writes the final positions as CSV (`[particles] file`), sorted by
id, gathered to rank 0. Trajectories, exit records, snapshots and their
parallel I/O are #753; `Particles::gather` is the interface it replaces.

## Not in this layer

- **Graphable migration** (above): distributed runs with particles are not
  graphed yet.
- **Parallel particle I/O** (#753): restart files gather every particle on
  every rank, which suits thousands of particles, not millions.
- **Remapped restarts** (#541): `type = "remap"` seeds the particles from
  `[particles]` instead of relocating the file's.
- **Seeding during the run**, regions and weights (#759), counter-based random
  numbers keyed by (seed, injector, step, index).
- **Compaction**: kernels run over every slot; dense packing of the live
  particles (or sorting them by cell, for memory locality) matters at the
  scale of #291 and comes with injection.
- **Particle fields in the registry** (#502) and Lagrangian statistics (#764).
- **Further models** (#757 follow-ups): thermophoresis, the Basset force,
  splashing, mass transfer (evaporation), drag and heat transfer in mixtures,
  and validation against DNS of particle-laden flow.
- **Two-way coupling** (#758): the particles' forces on the gas.
- **Of #756:** probes through surfaces (fluxes of particles), histograms, the
  residence time per visit and visit counts, crossings located within a stage,
  and probes staged in the step graph.
