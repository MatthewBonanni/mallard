# Lagrangian particles

Design of the particle core (#766), the first layer of the Euler-Lagrange model
of #291. User docs: [Particles](../guide/particles.md). Later layers (#756-#765,
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
| `x` | `N_DIM` reals | position, in the frame of the cell that holds it |
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
flow block and the species: a flat view of `N_DIM * capacity` reals. Every time
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
  fluxes and probes), computed only in the cells that hold particles, with the
  cell average where the linear value is not admissible.

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
gradients of the marked cells (one kernel over the owned cells), the linear
values, `point_values`, the velocity (one kernel each). After the step, `Particles::relocate` walks each
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

## Restart files

Restart files carry the particles as named arrays (version 6), written by rank
0 sorted by id (every rank gathers them, since each lays out the header): `PARTICLE_ID`, `PARTICLE_CELL` (the global id of the cell
holding each particle), `PARTICLE_X` (positions, exact in double precision) and
`PARTICLE_REMOVED` (the counts of particles that left the domain and that were
lost). The device copies them into pinned host mirrors with the other restart
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
- **Particle fields in the registry** (#502), probes and statistics of
  particles (#756, #764).
