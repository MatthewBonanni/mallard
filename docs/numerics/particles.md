# Numerical methods for particles

How Mallard discretizes the particle models of the
[theory manual](../theory/particles.md): where the gas state at a particle
comes from and how accurate it is, how particles are found in unstructured
and curved cells, how they are integrated in time with the flow (exponentially
when they relax faster than the step), how bounces are located within a step,
how particles move between ranks with bitwise identical results, and how
they are laid out on the device. The orders quoted are measured in the
[verification manual](../verification/index.md#lagrangian-particles); the
decisions behind the design are in the
[design notes](../design/particles.md). Usage:
[Particles](../guide/particles.md).

## Storage and data layout

Particles live on the device in a fixed number of *slots* per rank
(`[particles] capacity`, by default the global particle count, so that every
rank can hold every particle), as separate arrays of one entry per slot
(`src/particles/particles.h`):

| Array | Per slot | Holds |
|---|---|---|
| `x` | `width` reals, contiguous | the integrated block: position, then for solid particles the velocity and, with heat transfer, the temperature, then the integrated quantities |
| `aux` | `n_aux` reals | extrema and first times of the quantities, which an integrator cannot combine linearly |
| `id` | `uint64_t` | the global id; `EMPTY` (all bits set) marks a free slot |
| `cell` | `uint32_t` | the local cell holding the particle at the start of the step |
| `step_start` | `n_start` reals | solid particles: $\mathbf{v}$, $\mathbf{u}$, $\mathbf{a}$, $\tau$, $\Delta t$ (and $T_p$, $T$, $\tau_T$) at the step's start |

and per-stage scratch: the cell and offset from its centroid of each stage
position, and the gas $\mathbf{W}$ there. `x` is the particle block of the
solver's `State`, `width * capacity` reals: the time integrators update it
with the same element-wise kernels (`axpby`, `lerp`, the low-storage stage
updates) as the flow block, so every integrator advances the particles
without code of its own.

Kernels run over every slot and skip the empty ones. A fixed capacity is what
makes the step capturable in a CUDA graph: no kernel allocates, every launch
has the size the graph recorded, and removing or inserting a particle writes
a slot. Removing never moves a live particle; free slots are found by a
prefix sum over `id == EMPTY`, and the $k$-th incoming particle takes the
$k$-th free slot. Slot order never enters a result: particles do not
interact, and output sorts by id. A run without `[particles]` allocates
nothing and launches no particle kernel.

## The gas at a particle

At every stage, one kernel computes each particle's offset
$\mathbf{r} = \mathbf{x}_p - \mathbf{x}_i$ from the centroid of its cell $i$ and
marks the cell; a second computes the least-squares gradients of the marked
cells only; a third evaluates the linear fit at every particle; and the
reconstruction then overwrites the particles in smooth cells with its own
polynomial (`Particles::evaluate_gas`, `FaceReconstruction::point_values`).

**TENO.** In a cell that TENO finds smooth, the troubled-cell measure
$\sigma_i$ of [TENO-E](../theory/finite_volume.md#teno-e)
below `troubled_threshold`, the value at the particle is the cell's central
polynomial of degree $r = k - 1$, the $k$-exact least-squares fit over the
central stencil that the faces of smooth cells use,

$$
\begin{equation}
\mathbf{U}(\mathbf{x}_p) = \overline{\mathbf{U}}_i + \sum_{l=1}^{N_r} \mathbf{a}_l
\left(\phi_l(\boldsymbol{\xi}) - \overline{\phi_l}^{\,i}\right), \qquad
\boldsymbol{\xi} = \frac{\mathbf{x}_p - \mathbf{x}_i}{h_i}, \qquad
\mathbf{a}_l = \sum_{s\in\mathcal{S}_K} P_{ls}\left(\overline{\mathbf{U}}_s - \overline{\mathbf{U}}_i\right),
\label{eq:point-polynomial}
\end{equation}
$$

in the conservative variables of a single gas (in $\mathbf{W}$ for mixtures,
as the faces), then converted to $\mathbf{W}$. Its error at any point of the
cell is $O(h^k)$ for smooth flow, and since the fit spans a stencil of
several cells it is as accurate a short distance outside the cell as inside.
The kernel (`TENOFunctor::PointPass`, `src/numerics/teno.cpp`) recomputes,
per particle, the troubled-cell measure and the $N_r$ coefficients from the
stencil's cell averages and the pseudo-inverse $\mathbf{P}$ stored at
start-up, so it costs a pass over the stencil (about $2N_r$ cells) per
particle and stage and stores no polynomial. A value with $\rho \le 0$ or
$p \le 0$ is rejected, and the particle keeps the linear fit.

**Elsewhere** (troubled TENO cells, `FO`, `MUSCL`) the value is the cell's
linear least-squares fit,

$$
\begin{equation}
\mathbf{W}(\mathbf{x}_p) = \overline{\mathbf{W}}_i + \nabla\mathbf{W}_i\cdot\mathbf{r},
\label{eq:linear-fit}
\end{equation}
$$

with the gradient of the viscous fluxes, a weighted least-squares fit of a
quadratic over the cell's vertex neighbors
([Gradients for the viscous terms](../theory/finite_volume.md#gradients-for-the-viscous-terms)),
or the cell average where the linear value has $\rho \le 0$ or $p \le 0$.
The gradient is second-order accurate, so the error is $O(h^2)$: the
quadratic term the fit drops, and the difference between the cell average
and the point value at the centroid. That holds with first-order fluxes too,
although the flow they carry is only first-order accurate.

**Measured orders.** In a smooth periodic field (which isolates the
interpolation from the flow's own error) the gas velocity at the particles
converges at order 3, 4 and 5 for TENO of orders 3, 4 and 5 on
quadrilaterals, triangles, hexahedra and tetrahedra, and at order 2 for `FO`
and `MUSCL`. Pathlines of tracers in a steady compact vortex, which include
the flow's error, converge at the scheme's order: first order with `FO`,
whose vortex diffuses at first order however accurately the particles read
it ([verification](../verification/index.md#lagrangian-particles)).

### One cell per step

Every stage of a step evaluates the polynomial of the cell $i_n$ that held the
particle at the start of the step, $t_n$, also when the stage position has
moved into a neighbor (by less than a cell, at CFL below 1):

$$
\begin{equation}
\mathbf{u}(\mathbf{x}_p^{(s)}, t_n + c_s \Delta t) \approx \mathbf{u}_{i_n}^{(s)}(\mathbf{x}_p^{(s)}),
\label{eq:start-cell}
\end{equation}
$$

where $\mathbf{u}_{i_n}^{(s)}$ is the polynomial of cell $i_n$ reconstructed
from the stage-$s$ flow. Within the step the right-hand side of the particle
is then one smooth function of its position, and the Runge-Kutta method keeps
its order; switching to the neighbor's polynomial at the face a stage
crosses would make it jump by the reconstruction error in the middle of a
step. The error stays $O(h^k)$, since the polynomial is as accurate a little
outside its cell as inside. The particle changes cells at the end of the
step, when it is relocated, and the jump between neighboring polynomials is
then of the reconstruction's order. No stage walks across faces, and the
start cell is always owned (migration runs after every step), so no stage
reads a halo cell.

![A step of a particle crossing a face: every stage reads the polynomial of the cell it started in, and the particle is relocated at the end of the step](../images/theory/particle_start_cell-light.svg#gh-light-mode-only)
![A step of a particle crossing a face: every stage reads the polynomial of the cell it started in, and the particle is relocated at the end of the step](../images/theory/particle_start_cell-dark.svg#gh-dark-mode-only)

### Gradients for the forces

The pressure-gradient, added-mass and lift forces read $\nabla p$ and
$\nabla\mathbf{u}$, which are the cell's least-squares gradients of
eq. $\eqref{eq:linear-fit}$, copied per particle only when one of these forces
is on. The gradient is constant over the cell, so at a particle a distance
$\lvert\mathbf{r}\rvert \sim h$ from the centroid these forces are first-order
accurate in space, against the reconstruction's order for the drag. For
heavy particles they are smaller than the drag by $\rho / \rho_p$ (lift: by
up to $(\rho/\rho_p)^{1/2}$), so their error is too.

## Location in unstructured cells

A particle moves less than a cell per step, so it is located at the end of
each step by walking across faces from the cell it started in
(`Walker::walk`, `src/particles/particles.h`;
[Haselbacher et al. 2007](../references.md#haselbacher-2007)). In cell $c$,
with each face $f$'s plane through its centroid $\mathbf{x}_f$ with unit normal
$\mathbf{n}_f$ out of $c$, the walk evaluates the signed distances

$$
\begin{equation}
\delta_f = (\mathbf{x}_p - \mathbf{x}_f)\cdot\mathbf{n}_f,
\label{eq:face-distance}
\end{equation}
$$

and leaves through the face of the largest $\delta_f$, or stops in $c$ if
every $\delta_f$ is at most $\epsilon\,h_c$, with $h_c = \lvert\Omega_c\rvert^{1/d}$
and $\epsilon = 10^{-10}$ in double precision ($10^{-5}$ in single). A point
within the tolerance of a face belongs to the cell the walk reached first, so
location is unambiguous on faces. Each step of the walk costs one pass over
the cell's faces; a particle that moved less than a cell takes one or two.
On convex cells the walk terminates; a limit of 64 cells catches the rare
cycle on distorted meshes, and the particle is then counted as lost. The
face planes are the mesh's own, those of the fluxes, so on straight-sided
meshes location matches the finite-volume cells exactly.

**Curved cells.** On a curved boundary (`[[mesh.curved]]`) the centroids and
volumes of the cells come from the curved geometry, so offsets and
polynomials are those of the curved cells, but location uses face planes: a
curved face's plane passes through the centroid of its straight face with,
in 3D, the direction of the curved face's area vector, and in 2D the
chord's normal. Between a curved wall and that plane, a gap of
$O(h^2/R)$ for a radius of curvature $R$, a particle is reflected at the
plane rather than at the curve: the second-order error of a straight-sided
mesh at a curved wall.

**Periodic faces** carry a shift (`Mesh::face_shift`): crossing one maps the
position into the neighbor's frame, and for solid particles rotates the
velocity, and the gas velocity and acceleration of the step's start, with a
rotational periodic map. Positions are always in the frame of the cell that
holds them.

**Seeds** are located once at setup by the point location of the probes (a
box search on the host, `locate_points`); the rank owning the cell holding
a seed takes it, and ties go to the lowest global cell id, so every rank count
seeds the same particles in the same cells. A seed outside the mesh is an
input error.

**Boundary faces.** The walk stops at the first face whose action is not to
cross ([theory](../theory/particles.md#walls-and-other-boundaries)): it
reflects off walls and symmetry planes and walks on (up to 8 bounces per
step), removes particles at inlets, outlets and far field and at sticking
walls, and counts as lost a particle that reaches a partition face of the
outermost halo layer, which it can only do by moving farther than the halo in
one step.

## Integration in time

### The particle block

The particles' integrated values (positions; velocities and temperatures of
solid particles; integrated quantities) are the third block of the
integrator's `State`, after the cell averages and the species. After the
flow's right-hand side of each stage, `Particles::velocity` writes the
particle rates from the same stage's gas, so every Runge-Kutta method
(`FE`, `SSPRK3`, `RK4`, `SSPRK54`, `CK45`, `RDPK3Sp35`) advances the
particles at its own order with the stage times of the flow, and the error
controller's rejected steps restore them with the flow. The first stage of
every integrator is the step's start ($c = 0$): there the particles store
what later stages need (`step_start`) and sample the extrema and first times
of their quantities. With Strang splitting of the chemistry the particles
move in the transport step only.

### Explicit integration

A particle whose relaxations are slower than the step, $h = \Delta t / \tau \le 1$
(and $\Delta t / \tau_T \le 1$), integrates the equations of motion and heat transfer of the
[theory](../theory/particles.md) as written, at the integrator's order, with
$\tau$, $\mathbf{a}$ and $\tau_T$ evaluated at every stage. The relaxation
$\mathrm{d}\mathbf{v}/\mathrm{d}t = -\mathbf{v}/\tau$ is stable for $h$ up
to 2 (FE), 2.51 (SSPRK3) or 2.79 (RK4), so every integrator is stable below
the switch at $h = 1$.

### Exponential integration of stiff particles

Small particles relax in far less than a step of the flow: a 1 µm droplet in
air in 3 µs. Rather than limit the flow's step by its smallest particle,
Mallard integrates the relaxation of a particle with $h > 1$ exponentially
(`particles::Exponential`, `src/particles/particle_forces.h`). With $\tau$
frozen at the step's start, the velocity obeys
$\mathrm{d}\mathbf{v}/\mathrm{d}t = (\mathbf{v}_{eq}(t) - \mathbf{v})/\tau$
with the equilibrium velocity $\mathbf{v}_{eq} = \mathbf{u} + \tau\,\mathbf{a}$.
With $\mathbf{v}_0$ and $\mathbf{v}_{eq,0}$ the values at the step's start and
$\varphi_1(h) = (1 - e^{-h}) / h$, each stage's rates are

$$
\begin{equation}
\begin{aligned}
\frac{\mathrm{d}\mathbf{v}}{\mathrm{d}t} &= \frac{(1 - e^{-h})(\mathbf{v}_{eq,0} - \mathbf{v}_0)
+ 2 (1 - \varphi_1)(\mathbf{v}_{eq} - \mathbf{v}_{eq,0})}{\Delta t}, \\
\frac{\mathrm{d}\mathbf{x}_p}{\mathrm{d}t} &= \mathbf{v}_{eq,0} + \varphi_1 (\mathbf{v}_0 - \mathbf{v}_{eq,0})
+ \left(1 - \frac{2 (1 - \varphi_1)}{h}\right)(\mathbf{v}_{eq} - \mathbf{v}_{eq,0}),
\end{aligned}
\qquad (h > 1),
\label{eq:exponential}
\end{equation}
$$

with $\mathbf{v}_{eq}$ of the stage's gas. Any Runge-Kutta method with
$\sum b_i = 1$ and $\sum b_i c_i = 1/2$ (every method of order 2 and above)
integrates these to

$$
\begin{equation}
\mathbf{v}_1 = \mathbf{v}_{eq,1} - \varphi_1\,(\mathbf{v}_{eq,1} - \mathbf{v}_{eq,0}) + e^{-h}(\mathbf{v}_0 - \mathbf{v}_{eq,0}),
\label{eq:exponential-solution}
\end{equation}
$$

and the matching position: the exact solution wherever $\mathbf{v}_{eq}$ is
linear in time over the step. The scheme therefore needs no stage weights of
the integrators. Settling in gas at rest or in uniform flow is exact; the lag
$-\tau\,\mathrm{D}\mathbf{u}/\mathrm{D}t$ of a small particle behind an
accelerating gas (the expansion of
[the theory](../theory/particles.md#stokes-number-and-terminal-velocity)) is
kept; and a particle with $\tau$ near $\Delta t$ keeps its inertia, where
setting $\mathbf{v} = \mathbf{v}_{eq}$ would be first order in $\tau / \Delta t$.
FE, whose one stage is the step's start, is exact for constant
$\mathbf{v}_{eq}$ only, and first order, as FE is. The temperature is
integrated the same way, toward the gas temperature, when
$\Delta t / \tau_T > 1$.

**Error.** Two approximations remain. First, the curvature of
$\mathbf{v}_{eq}$ in time over the step. For a method with
$\sum b_i c_i^2 = 1/3$ (every method of order 3 and above) and
$\mathbf{v}_{eq}$ quadratic in time, the velocity's error over one step is

$$
\begin{equation}
\delta\mathbf{v} = \frac{\Delta t^2}{2}\,\ddot{\mathbf{v}}_{eq}\;\varepsilon(h), \qquad
\varepsilon(h) = 1 - \frac{2}{h} + \frac{2\varphi_1}{h} - \frac{2}{3}(1 - \varphi_1)
\;\approx\; \begin{cases} h^2 / 36, & h \to 0, \\ 1/3, & h \to \infty. \end{cases}
\label{eq:exponential-error}
\end{equation}
$$

The relaxation damps each step's error by $e^{-h}$, so the velocity error
does not accumulate: it stays $O(\Delta t^2\,\ddot{\mathbf{v}}_{eq})$, the
change of the equilibrium velocity's slope over a step. The position
integrates it, and its error grows as $O(\Delta t^2)$ over a fixed time.
Second, $\tau$ (and with it $\mathbf{v}_{eq}$) is frozen at the step's start.
Near equilibrium the slip is $\tau(\mathbf{a} - \mathrm{D}\mathbf{u}/\mathrm{D}t)$, so a change $\Delta\tau$ of $\tau$ over the step errs
by $\Delta\tau\,\lvert\mathbf{a} - \mathrm{D}\mathbf{u}/\mathrm{D}t\rvert$, a
fraction $\Delta\tau / \tau$ of the slip: first order in $\Delta t$, but
proportional to the relative change of $\tau$ over the step, small where the
gas at the particle changes little in a step, and zero for Stokes drag in gas
of uniform viscosity. At a fixed $\tau$, refining the step lowers $h$ and
$\varepsilon(h)$ with it, so convergence is faster than second order between
the limits.

**Where the switch lies.** At $h = 1$ the explicit RK4 error overtakes the
exponential one in a decaying-forcing experiment (the crossover is near
$h = 0.9$; for SSPRK3 the exponential error is smaller everywhere). The
switch is per particle and per step, from $\tau$ of the step's start, and the
trajectories converge on both sides of it.

**Measured orders.** In a uniform gas turning at a constant rate, with the
error the integrator's alone: tracer pathlines and solid particles of
$\tau = 0.2$ converge at the integrator's design order, as do particles of
$\tau = 0.02$ at $\Delta t / \tau = 1/2$ to $1/8$, just below the switch.
Above it, particles of $\tau = 0.002$ at $\Delta t / \tau = 12.5$ to 3.1,
where RK4 alone is unstable, converge to their inertial trajectories at
orders 2.90 (position) and 2.79 (velocity) on the finest pair with RK4
([verification](../verification/index.md#lagrangian-particles)).

`dt` in eq. $\eqref{eq:exponential}$ is the step of the host, or inside a
captured step the device scalar the step graph computes it in
(`step_dt_device`); the first stage stores it in `step_start`, since the step
controller overwrites the device's step size before relocation.

## Bounces within a step

A solid particle found a depth $\delta$ past a reflecting face at the end of
a step, moving into it ($v_n > 0$) with $\tau \ge \Delta t$, bounced at some
time within the step, after which it relaxed toward the gas on its way back.
`particles::rebound` (`src/particles/particle_forces.h`) locates that time.
Over the step the particle is taken to move as the relaxation form of the
[equation of motion](../theory/particles.md#solid-particles) with $\mathbf{u}$, $\mathbf{a}$ and $\tau$
of the step's start, whose velocity and position are closed form: going back
in time from the step's end by $b$, with $z = b / \tau$, $\psi(z) = (e^z - 1)/z$
and $\varphi_2(z) = (1 - \varphi_1(z))/z$, the normal distance traveled is

$$
\begin{equation}
\delta = u_n\,b + w_n\,b\,\psi(z) - a_n\,b^2\left(\psi\varphi_1 - \varphi_2\right)(z), \qquad w_n = v_n - u_n,
\label{eq:bounce-time}
\end{equation}
$$

with the normal components $u_n$, $v_n$, $a_n$. Its root $b$, the time
before the step's end at which the particle reached the wall, is found by
six fixed-point passes $b \leftarrow \delta / \bar v_n(b)$ from
$b = \delta / v_n$, $\bar v_n$ the mean normal velocity over $b$, and clipped
to $\Delta t$. At that time the [bounce](../theory/particles.md#walls-and-other-boundaries)
reverses and scales the velocity, and the particle relaxes on for $b$: with the velocity
at the wall $\mathbf{v}_c = \mathbf{u} + (\mathbf{v} - \mathbf{u})\,e^z - \mathbf{a}\,b\,\psi$
and the change $(\mathsf{R} - \mathsf{I})$ of the bounce
$\mathsf{R}\mathbf{x} = e_t\,\mathbf{x}_t - e_n\,x_n\mathbf{n}$, the
velocity gains $(\mathsf{R} - \mathsf{I})\,\mathbf{v}_c\,e^{-z}$ and the
position $(\mathsf{R} - \mathsf{I})$ applied to the displacement over $b$.
The $\varphi$ functions use their Taylor series below $z = 0.1$ and `expm1`
above.

This is exact for drag in a gas uniform over the step's travel and for
constant acceleration ($\tau \to \infty$), so there the bounce's timing does
not depend on the step (exact to round-off in the tests); in a strongly
varying gas it is first order in the step. A particle with $\tau < \Delta t$
follows the gas within the step, and is reflected at the step's end in
the face plane, with the restitution coefficients. After each bounce the walk continues from
the reflected position, so a particle bouncing in a corner reflects off each
face in turn.

## Migration across ranks

After relocation, a particle whose cell is a halo cell belongs to the halo
cell's owner (`Particles::migrate`). The emigrants are compacted on the
device in slot order (a prefix sum), copied to the host, and sent to the
owners (`comm::alltoallv`) with their id, the global id of their cell and the
bits of their integrated and auxiliary values, reinterpreted as integers
(`Kokkos::bit_cast`), so that a position, velocity or quantity arrives bit
for bit. Each rank maps the global cells to its own and inserts the arrivals
into its free slots in order. A particle whose new owner is unknown (it left
the halo) is counted as lost, and a rank with fewer free slots than arrivals
stops the run with an error naming `capacity`.

Because the migration reads counts on the host, it cannot be captured in a
step graph: distributed runs with particles take their steps outside the
step graph and without device-decided steps. Single-rank runs capture the
whole step: the stage kernels, the relocation, and the staging of restart
data into pinned host mirrors.

## Determinism

Each particle's update reads only the flow, its own state and its start
cell's polynomial: its result does not depend on the slot it occupies, on the
other particles or on the order in which kernels visit them. The start cell
is owned, the flow is bitwise independent of the rank count, and migration
moves values bit for bit with cells by global id, so trajectories are
bitwise identical on any rank count (tested on 1, 2 and 3 ranks, tracers and
solid particles, across periodic sides and partitions). Single-rank runs are
bitwise identical with and without step graphs, since a captured step runs
the same kernels on the same device step size. Restart files carry the
positions, velocities, temperatures and quantities in double precision with
the global cell of each particle, so a restarted run, on any rank count,
continues bit for bit. The only shared counters (particles removed, by
cause) are integer atomics, and the statistics of particle probes are
fixed-point sums, which are associative.

## Limitations

- Distributed runs with particles are not graphed (migration through the
  host); the graphable design (fixed-size per-neighbor buffers exchanged on
  the step stream) is in the [design notes](../design/particles.md#mpi-migration).
- Location uses planar faces: on curved boundaries, walls are located to
  $O(h^2/R)$.
- The forces that read gradients are first order in space.
- Bounces are located within the step for gas uniform over the step's
  travel; in strongly varying gas, and for $\tau < \Delta t$, at first order.
- Kernels run over every slot, live or not; the particles are not sorted by
  cell for memory locality.
