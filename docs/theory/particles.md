# Lagrangian particles

Mallard follows point particles through the flow, each with its own
position and, for solid particles, its own velocity, advanced by the time
integrator of the flow together with it. This chapter gives the equations of
motion as implemented (`src/particles/`), the drag laws, the evaluation of
the gas at a particle from the high-order reconstruction, the integration in
time (exponential for relaxations faster than the step), and the treatment of
walls and other boundaries.
Usage: [Particles](../guide/particles.md); measured orders:
[verification manual](../verification/index.md#lagrangian-particles); design
rationale: [particles](../design/particles.md).

The particles are dilute and one-way coupled: the gas moves them, they do not
act on the gas or on each other, and each is smaller than the cells and the
smallest scales of the flow around it, so that the gas state at its center
(the *undisturbed* flow) drives it
([Elghobashi 1994](../references.md#elghobashi-1994),
[Balachandar & Eaton 2010](../references.md#balachandar-eaton-2010)).

## Equations of motion

### Tracers

A tracer is a massless particle that moves with the gas velocity
$\mathbf{u}$ at its position $\mathbf{x}_p$:

$$
\begin{equation}
\frac{\mathrm{d}\mathbf{x}_p}{\mathrm{d}t} = \mathbf{u}(\mathbf{x}_p, t).
\label{eq:tracer}
\end{equation}
$$

Its trajectory is a pathline of the flow: in a steady flow, a streamline.

### Solid particles

A solid particle is a rigid sphere of diameter $d$ and material density
$\rho_p$ with a velocity $\mathbf{v}$ of its own. Of the forces of the
Maxey-Riley equation ([Maxey & Riley 1983](../references.md#maxey-riley-1983))
Mallard always keeps the drag, gravity and buoyancy, which dominate for
particles much denser than the gas, $\rho_p / \rho \gg 1$:

$$
\begin{equation}
\frac{\mathrm{d}\mathbf{x}_p}{\mathrm{d}t} = \mathbf{v}, \qquad
\frac{\mathrm{d}\mathbf{v}}{\mathrm{d}t} = \frac{\mathbf{u}(\mathbf{x}_p, t) - \mathbf{v}}{\tau}
+ \mathbf{g}' , \qquad \mathbf{g}' = \mathbf{g}\left(1 - \frac{\rho}{\rho_p}\right),
\label{eq:solid}
\end{equation}
$$

with the relaxation time

$$
\begin{equation}
\tau = \frac{\tau_{St}}{f(Re_p)}, \qquad \tau_{St} = \frac{\rho_p d^2}{18 \mu}, \qquad
Re_p = \frac{\rho \lvert\mathbf{u} - \mathbf{v}\rvert d}{\mu},
\label{eq:relaxation-time}
\end{equation}
$$

where $\rho$, $\mathbf{u}$ and $\mu$ (at the gas temperature
$T = p / (\rho R)$, from the viscosity law of `[physics]`) are the gas state at
the particle, and $f = C_D Re_p / 24$ is the drag correction of the drag
law (below): the drag over Stokes drag
([Stokes 1851](../references.md#stokes-1851)). The pressure-gradient and
added-mass forces, which scale with $\rho / \rho_p$, and the Saffman lift
([Saffman 1965](../references.md#saffman-1965)) are added by `forces`
([user guide](../guide/particles.md#solid-particles)): they add to
$\mathbf{g}'$ and, with added mass, $\rho / 2$ to $\rho_p$ in $\tau$. The
Basset history force and the Magnus lift are left out; see
[Crowe et al. 2011](../references.md#crowe-2011) for their magnitudes. With
`specific_heat` the particle's temperature relaxes to the gas's in
$\tau_T = \rho_p c_p d^2 / (6 k Nu)$, with the Nusselt number of
[Ranz & Marshall 1952](../references.md#ranz-marshall-1952), and is integrated
as the velocity is.

The ratio of $\tau$ to a time scale $\tau_f$ of the flow is the **Stokes
number** $St = \tau / \tau_f$. For $St \ll 1$ the particle follows the gas
(a tracer, up to the settling velocity); for $St \gg 1$ it moves almost
ballistically, the gas changing its velocity only over many $\tau_f$; near
$St = 1$ it lags the gas by about its own relaxation and is flung out of
vortices into the strain between them (preferential concentration,
[Squires & Eaton 1991](../references.md#squires-eaton-1991)).

In gas at rest a particle reaches the **terminal velocity**
$\mathbf{v}_t = \tau \mathbf{g}'$, where drag balances gravity; with Stokes
drag

$$
\begin{equation}
\mathbf{v}_t = \tau_{St}\,\mathbf{g}' = \frac{(\rho_p - \rho) d^2}{18 \mu}\,\mathbf{g},
\label{eq:terminal-stokes}
\end{equation}
$$

and with a drag law $f(Re_p)$ the root of
$\lvert\mathbf{v}_t\rvert f(\rho \lvert\mathbf{v}_t\rvert d / \mu) = \tau_{St} \lvert\mathbf{g}'\rvert$.
From rest, with Stokes drag, $\mathbf{v}(t) = \mathbf{v}_t (1 - e^{-t/\tau})$.

### Drag laws

`[particles] drag`:

- **`stokes`**: $f = 1$, creeping flow around the sphere: the drag is 5% low
  at $Re_p = 0.2$ and 13% at $Re_p = 1$ against Schiller-Naumann.
- **`schiller_naumann`** (default): the correlation of
    [Schiller & Naumann 1933](../references.md#schiller-naumann-1933),

    $$
    \begin{equation}
    f(Re_p) = \begin{cases} 1 + 0.15\,Re_p^{0.687}, & Re_p < 1000, \\ 0.44\,Re_p / 24, & Re_p \ge 1000, \end{cases}
    \label{eq:schiller-naumann}
    \end{equation}
    $$

    within about 5% of the standard drag curve of a sphere up to $Re_p = 800$
    ([Clift, Grace & Weber 1978](../references.md#clift-grace-weber-1978));
    above $Re_p = 1000$ the drag coefficient takes its Newton-regime
    value $C_D = 0.44$. The two branches meet within 0.4%.

- **`singh`**: the correlation of
  [Singh et al. 2022](../references.md#singh-2022) (their Appendix C), of
  $Re_p$, the slip Mach number $M = \lvert\mathbf{u} - \mathbf{v}\rvert / c$,
  the Knudsen number $Kn = \lambda / d = \sqrt{\gamma\pi/2}\,M / Re_p$ and
  $T_p / T$ (1 without heat transfer). A continuum drag, Abraham's
  $C_D = (24 / \delta_0^2)(1 + \delta_0 / Re_p^{1/2})^2$, $\delta_0 = 9.4$, at
  low Mach number, with the pressure drag of a normal shock above Mach 1,
  tending to $C_D = 0.9$ at hypersonic speed, is reduced by a slip correction
  in $Kn$ and bridged to the free-molecular drag of a sphere at large $Kn$.
  It reduces to Stokes drag in creeping continuum flow; fitted to experiments
  and DSMC from $M = 0.1$ to 10, its relative L2 error against the data is
  7.6%.

The [user guide](../guide/particles.md#drag-laws) tabulates where each law is
valid.

![Drag coefficient of a sphere against its Reynolds number: Stokes drag, the Schiller-Naumann correlation and Singh et al.'s at slip Mach numbers 0.0001, 0.5 and 2 in air](../images/theory/particle_drag-light.svg#gh-light-mode-only)
![Drag coefficient of a sphere against its Reynolds number: Stokes drag, the Schiller-Naumann correlation and Singh et al.'s at slip Mach numbers 0.0001, 0.5 and 2 in air](../images/theory/particle_drag-dark.svg#gh-dark-mode-only)

*Singh et al.'s drag at a fixed slip Mach number in air ($\gamma = 1.4$): its
Knudsen number grows as $M / Re_p$, so at low $Re_p$ rarefaction lowers the
drag below Stokes', and above Mach 1 the shock raises it.*

Every law evaluates $Re_p$ (and $M$, $Kn$), and so $\tau$, from the slip
velocity of each stage (`particles::Solid::response`,
`src/particles/particle_forces.h`).

## The gas at a particle

The gas state $\mathbf{W} = [\rho, \mathbf{u}, p]$ at a particle comes from
the polynomial the reconstruction already builds in the particle's cell $i$
(`Particles::velocity`, `src/particles/particles.cpp`), not from the cell
averages:

- with TENO of order $k$, the cell's central polynomial
  $\mathbf{W}_i(\mathbf{x})$ of degree $k - 1$
  ([Finite-volume discretization and reconstruction](finite_volume.md#reconstruction)),
  where TENO finds the cell smooth and the value at the particle is
  admissible ($\rho, p > 0$): an error $O(h^k)$;
- elsewhere (troubled cells, `FO` and `MUSCL`), the cell's linear
  least-squares fit $\overline{\mathbf{W}}_i + \nabla\mathbf{W}_i \cdot (\mathbf{x} - \mathbf{x}_i)$,
  the gradient of the viscous fluxes, computed only in the cells that hold
  particles, or the cell average where the linear value is not admissible:
  an error $O(h^2)$, also with first-order fluxes.

The derived values the particle reads (the temperature, the viscosity, the
particle expressions of the quantities) are computed from this $\mathbf{W}$.

**One cell per step.** Every stage of a step evaluates the polynomial of the
cell $i_n$ that held the particle at the start of the step, $t_n$, also when the
stage position has moved into a neighbor (by less than a cell, at CFL below 1):

$$
\begin{equation}
\mathbf{u}(\mathbf{x}_p^{(s)}, t_n + c_s \Delta t) \approx \mathbf{u}_{i_n}^{(s)}(\mathbf{x}_p^{(s)}),
\label{eq:start-cell}
\end{equation}
$$

where $\mathbf{u}_{i_n}^{(s)}$ is the polynomial of cell $i_n$ reconstructed from
the stage-$s$ flow. Within the step the right-hand side of the particle is
then one smooth function of its position, and the Runge-Kutta method keeps its
order; switching to the neighbor's polynomial at the face a stage crosses
would make it jump by the reconstruction error in the middle of a step. A
polynomial fitted on a stencil of several cells is as accurate a little
outside its cell as inside, so the error stays $O(h^k)$. The particle changes
cells at the end of the step, when it is relocated.

![A step of a particle crossing a face: every stage reads the polynomial of the cell it started in, and the particle is relocated at the end of the step](../images/theory/particle_start_cell-light.svg#gh-light-mode-only)
![A step of a particle crossing a face: every stage reads the polynomial of the cell it started in, and the particle is relocated at the end of the step](../images/theory/particle_start_cell-dark.svg#gh-dark-mode-only)

## Integration in time

The particles are a block of the integrator's state, next to the cell
averages and the species: per particle its position, its velocity (solid
particles) and its integrated quantities. Every Runge-Kutta method of
[Time integration](time_integration.md#runge-kutta-methods) advances the
block with its own stages and weights, at the stage times of the flow,
evaluating eqs. $\eqref{eq:tracer}$ or $\eqref{eq:solid}$ with each stage's
gas, position and velocity. The trajectories therefore converge at the
integrator's order in time and at the reconstruction's order in space, as the
flow does; a rejected step of the error-controlled integrator restores the
particles with the flow.

### Relaxations faster than the step

An explicit Runge-Kutta method integrates the relaxation
$\mathrm{d}\mathbf{v}/\mathrm{d}t = -\mathbf{v}/\tau$ stably only for
$h = \Delta t / \tau$ below 2 (FE), 2.51 (SSPRK3) or 2.79 (RK4), and small
particles relax in far less than a step of the flow: a 1 µm droplet in air in
3 µs. Rather than limit the flow's step by its smallest particle, Mallard
integrates the relaxation of a particle with $h > 1$ (with $\tau$ of the
step's start) exponentially. With $\mathbf{v}_{eq} = \mathbf{u} + \tau\,\mathbf{a}$
the equilibrium velocity ($\mathbf{a}$ the acceleration of gravity and the
other forces), $\mathbf{v}_0$ and $\mathbf{v}_{eq,0}$ their values at the
step's start and $\varphi_1(h) = (1 - e^{-h}) / h$, each stage's rates are

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

which every Runge-Kutta method with $\sum b_i = 1$ and $\sum b_i c_i = 1/2$
(every method of order 2 and above) integrates to the exact solution of
eq. $\eqref{eq:solid}$ for $\mathbf{v}_{eq}$ linear in time over the step.
Settling in gas at rest or in uniform flow is exact, the lag
$-\tau\,\mathrm{D}\mathbf{u}/\mathrm{D}t$ of a small particle behind an
accelerating gas ([Maxey 1987](../references.md#maxey-1987),
[Ferry & Balachandar 2001](../references.md#ferry-balachandar-2001)) is kept,
and a particle with $\tau$ near $\Delta t$ keeps its inertia. The particle's
temperature is integrated the same way when $\Delta t / \tau_T > 1$. `dt` is the
step of the host, or inside a captured step the device's own step size
(`step_dt_device`).

The switch is per particle and per step. The flow's step never depends on
the particles, and the trajectories converge in time on both sides of it: the
[verification manual](../verification/index.md#lagrangian-particles) measures
the integrator's order at $\Delta t / \tau = 1/2$ to $1/8$, just below the
switch, and the convergence of the exponential integration at
$\Delta t / \tau = 12.5$ to $3.1$.

## Location and boundaries

A particle moves less than a cell per step, so it is located at the end of
each step by walking across faces from the cell it started in
(`Walker::walk`, `src/particles/particles.h`), as in
[Haselbacher et al. 2007](../references.md#haselbacher-2007): from each cell
the walk leaves through the face whose plane the point lies farthest outside
of, and stops in the cell whose face planes the point lies inside of (to a
relative tolerance of the cell size). Each face has an action, from its
boundary condition:

- **interior and partition faces**: the particle crosses; on several ranks it
  moves to the rank owning its new cell;
- **periodic faces**: the particle crosses into the neighbor's frame, its
  position (and velocity) translated or rotated with the periodic map;
- **inlets, outlets and far field**: the particle leaves the domain and is
  removed;
- **walls and symmetry planes**: the particle is reflected, or removed as
  deposited where `[[particles.walls]]` sets `interaction = "stick"`.

A particle a depth $\delta$ past a reflecting face of unit outward normal
$\mathbf{n}$ at the end of a step is moved back in by

$$
\begin{equation}
\mathbf{x}_p' = \mathbf{x}_p - (1 + e_n)\,\delta\,\mathbf{n},
\label{eq:reflect-position}
\end{equation}
$$

and a solid particle moving into the wall, $v_n = \mathbf{v}\cdot\mathbf{n} > 0$,
leaves it with

$$
\begin{equation}
\mathbf{v}' = e_t\,(\mathbf{v} - v_n \mathbf{n}) - e_n\,v_n\,\mathbf{n},
\label{eq:reflect-velocity}
\end{equation}
$$

with the normal and tangential restitution coefficients $e_n$ and $e_t$ of
the wall (both 1 for tracers, which are only mirrored, and by default). A
particle can bounce off up to 8 faces in one step (corners); one that does
not settle in a cell within 64 cells of the walk, or reaches the outer edge of
a rank's halo, is counted as lost.

A solid particle's bounce is located within the step
(`particles::rebound`, `src/particles/particle_forces.h`): over the step the
particle is taken to move as eq. $\eqref{eq:solid}$ with $\mathbf{u}$,
$\mathbf{a}$ and $\tau$ of the step's start, whose position and velocity are
closed-form; the time $b$ before the step's end at which it reached the wall is
the root of the normal distance it then travels, eq.
$\eqref{eq:reflect-velocity}$ is applied at that time, and the particle moves
on for $b$. That is exact for drag in a gas uniform over the step's travel and
for constant acceleration, so the timing of bounces does not depend on the
step there; in a strongly varying gas it is first order in the step. A
particle with $\tau < \Delta t$ follows the gas within the step and is
reflected at the step's end by eqs. $\eqref{eq:reflect-position}$ and
$\eqref{eq:reflect-velocity}$.

## Quantities along the path

A per-particle integral $I = \int_0^t \phi\,\mathrm{d}t'$ of an expression
$\phi$ of the gas and the particle is part of the integrated block, with
$\mathrm{d}I/\mathrm{d}t = \phi$ evaluated at each stage; a winding about an
axis $\mathbf{a}$ through a point $\mathbf{x}_0$ is the integral of the
particle's angular velocity about it,

$$
\begin{equation}
\frac{\mathrm{d}N}{\mathrm{d}t} = \frac{\mathbf{a}\cdot(\mathbf{r}\times\dot{\mathbf{x}}_p)}{2\pi\,\lvert\mathbf{r}_\perp\rvert^2},
\qquad \mathbf{r} = \mathbf{x}_p - \mathbf{x}_0,
\label{eq:winding}
\end{equation}
$$

with $\mathbf{r}_\perp$ the part of $\mathbf{r}$ normal to $\mathbf{a}$, so the
turns are unwrapped by construction. Both converge at the integrator's order.
Extrema and first times sample the accepted state at the start of every step
([Per-particle quantities](../guide/particles.md#per-particle-quantities)).

## Two-way coupling

With `coupling = "two-way"` the solid particles act on the gas
(`src/particles/particle_coupling.cpp`, `Solid::exchange` in
`src/particles/particle_forces.h`). The suspension is still dilute: the
gas equations keep their form, without the volume the particles take from the
gas or collisions between particles, and gain per-volume sources
$\mathbf{S} = [0, \mathbf{S}_{\rho\mathbf{u}}, S_{\rho E}]$ in momentum and
total energy, with no mass source. This regime of two-way coupling spans
particle volume fractions from about $10^{-6}$ to $10^{-3}$
([Elghobashi 1994](../references.md#elghobashi-1994)); above it collisions
matter (four-way coupling), which Mallard does not model.

### Source terms

A computational particle, a *parcel*, stands for $w$ identical physical
particles (`weight`; in 2D, particles per unit depth). Each gives the gas the
opposite of its own change of momentum and energy, less the part gravity on
the particle accounts for:

$$
\begin{equation}
\mathbf{S}_{\rho\mathbf{u}} = -\frac{1}{V}\sum_{p} w\,m_p\left(\frac{\mathrm{d}\mathbf{v}_p}{\mathrm{d}t} - \mathbf{g}\right), \qquad
S_{\rho E} = -\frac{1}{V}\sum_{p} w\,m_p\left(\frac{\mathrm{d}e_p}{\mathrm{d}t} - \mathbf{g}\cdot\mathbf{v}_p\right), \qquad
e_p = c_p T_p + \tfrac12\lvert\mathbf{v}_p\rvert^2,
\label{eq:coupling-sources}
\end{equation}
$$

with $m_p = \rho_p \pi d^3 / 6$, $V$ the volume the source is deposited over
(below), $\mathbf{g}$ the particles' gravity and, without heat transfer,
$e_p = \tfrac12\lvert\mathbf{v}_p\rvert^2$. Gravity on the particles is the
earth's force, not the gas's, so it does not act back. Everything else does,
as the reaction to the forces of eq. $\eqref{eq:solid}$: the drag, which for
drag alone gives the familiar
$\mathbf{S}_{\rho\mathbf{u}} = \frac{1}{V}\sum_p w\,m_p(\mathbf{v}_p - \mathbf{u})/\tau$,
the buoyancy or pressure-gradient force, the added mass and the lift. The
energy source is the heat the particles take from the gas,
$w\,\pi d\,k\,Nu\,(T_p - T)$ per particle, plus the work of those forces on
the particle, $-w\,m_p(\mathrm{d}\mathbf{v}_p/\mathrm{d}t - \mathbf{g})\cdot\mathbf{v}_p$.

### Conservation

The sources are computed from the very stage rates the integrator advances
the particles with, explicit or exponential, and every Runge-Kutta method
updates the gas and the particles by the same linear combination of stage
rates. Whatever the gas loses, the particles therefore gain, to round-off, at
every stage and step (rejected steps included), however stiff the particles:
the total momentum and energy of gas and particles change only by gravity's
force and work on the particles.

That holds for the energy only if the particle's energy is itself an
integrated value: its kinetic energy $\tfrac12\lvert\mathbf{v}\rvert^2$ is
quadratic in the integrated velocity, and no linear combination of stages
reproduces it exactly. With heat transfer, two-way coupled particles
therefore integrate $e_p$ in place of $T_p$, with the rate
$c_p\,\mathrm{d}T_p/\mathrm{d}t + \mathbf{v}\cdot\mathrm{d}\mathbf{v}/\mathrm{d}t$,
and read $T_p = (e_p - \tfrac12\lvert\mathbf{v}\rvert^2) / c_p$: the
truncation error of the kinetic energy shows in $T_p$, where it is of order
$(\delta v)^2 / c_p$, rather than in the total. Without heat transfer the
kinetic energy is conserved to the integrator's order. Restart files carry
$e_p$ (`PARTICLE_E`) beside $T_p$, so restarts continue bit for bit.

### Deposition

The sources of the particles in a stage are summed into the cells holding
them at the step's start and, with `deposition_width` $\sigma > 0$, spread
over a Gaussian of standard deviation $\sigma$
(`particles::Deposition`):

1. **Exact sums per cell.** Each cell sums its particles' sources in fixed
    point, in units of its largest term over $2^{62 - \lceil\log_2(n + 1)\rceil}$
    for $n$ particles in the cell (`exact_sums.h`), with integer atomics:
    each sum is exact to $2^{-60}$ of its largest term, whatever the order of
    the slots, which differs between rank counts and migrations. The start
    cell is always owned, so no deposit lands in a halo.
2. **First moments** (with a width). Spreading from the cell centers would
    place every particle at its cell's center, an error of half a cell in the
    kernel's center. The cells also sum each source's first moment about
    their centers, $\mathbf{m}_c = \sum_p q_p(\mathbf{x}_p - \mathbf{x}_c)$, and
    move each sum by minus the divergence of the moment density
    $\boldsymbol{\mu} = \mathbf{m} / V$, face averaged, in flux form,

    $$
    \begin{equation}
    q_c \leftarrow q_c - \sum_{f} \tfrac12\left(\boldsymbol{\mu}_c + \boldsymbol{\mu}_n\right)\cdot\mathbf{n}_f A_f,
    \label{eq:coupling-moments}
    \end{equation}
    $$

    which gives the deposit the particles' first moments: the spread then
    converges at second order in the cell size rather than first.
3. **Spreading by diffusion**, following
    [Capecelatro & Desjardins 2013](../references.md#capecelatro-desjardins-2013):
    $\sigma^2 / 2$ of diffusion with unit diffusivity, whose continuum limit
    is the convolution with a Gaussian of standard deviation $\sigma$, by
    explicit steps of the face-flux Laplacian,

    $$
    \begin{equation}
    q_c \leftarrow q_c + \Delta s \sum_{f} \frac{A_f}{\delta_f}\left(\frac{q_n}{V_n} - \frac{q_c}{V_c}\right),
    \qquad \Delta s \le \frac{1}{2}\left(\max_c \frac{1}{V_c}\sum_f \frac{A_f}{\delta_f}\right)^{-1},
    \label{eq:coupling-spread}
    \end{equation}
    $$

    with $\delta_f$ the distance between the two centroids along the face
    normal, over $\lceil \sigma^2 \max_c (\sum_f A_f / \delta_f) / V_c\rceil$
    equal steps (about $2d\,\sigma^2 / h^2$ on a Cartesian mesh of spacing
    $h$). The weights stay positive, so the spread deposit keeps the sign of
    the sources. A face moves exactly the opposite amount out of one cell and
    into the other, so the sums are conserved to round-off, and walls and
    other boundaries have no flux, so no source leaves the domain or crosses
    a wall.

The gas receives $q_c / V_c$ in each owned cell. With $\sigma = 0$ (the
default) each particle's source stays in its cell: a point source of the
mesh's size, which concentrates as the mesh is refined, so coupled results do
not converge under refinement and, with cells comparable to the particles,
the gas a particle reads carries its own disturbance, which the point-particle
drag does not expect. A width in physical units makes the deposit independent
of the mesh, so it converges as the mesh is refined at fixed $\sigma$: the
volume fraction of a single particle against the Gaussian of $\sigma = 0.08$
has L1 errors $5.3 \times 10^{-2}$, $2.3 \times 10^{-2}$ and
$4.3 \times 10^{-3}$ on 16, 32 and 64 cells per side (orders 1.2 and 2.4,
the first at $\sigma / h = 1.3$), where the cell-center deposit's error did
not converge. Each cell adds its faces in the order of its neighbors' global
ids, and halo values come from their owners before each step, so the
deposit, like the particles, is bitwise independent of the rank count; on
one rank its kernels are captured in the step graph.

![Left: the volume fraction of one particle deposited over a Gaussian of width 0.08 on 16, 32 and 64 cells per side, along the particle's row of cells, against the Gaussian. Right: its L1 error against the cell size, at second order](../images/theory/particle_deposition-light.svg#gh-light-mode-only)
![Left: the volume fraction of one particle deposited over a Gaussian of width 0.08 on 16, 32 and 64 cells per side, along the particle's row of cells, against the Gaussian. Right: its L1 error against the cell size, at second order](../images/theory/particle_deposition-dark.svg#gh-dark-mode-only)

*The volume fraction of one particle off the cell centers, deposited with
$\sigma = 0.08$ on Cartesian meshes of a periodic unit box (Mallard runs):
along the particle's row of cells (left), and the $L_1$ error of the whole
deposit against the Gaussian about the particle, relative to the particle's
volume (right).*

### Dispersed-phase fields

At output the same deposition gives the fields of the dispersed phase: the
volume fraction $\alpha = \sum_p w V_p / V$, the number density
$n = \sum_p w / V$, the particles' mean velocity
$\mathbf{U}_p = \sum_p w\,m_p\mathbf{v}_p / (\rho_p \sum_p w V_p)$
(mass weighted), and with two-way coupling the sources of
eq. $\eqref{eq:coupling-sources}$ at the written state, from the explicit
rates of the particles there (`Particles::fields`). `mallard check` warns
where a particle is larger than a tenth of its cell or the volume fraction
exceeds 1%, where the point-particle and dilute assumptions fail.

## Injection

`[[particles.injectors]]` add particles during the run
(`particles::Injectors`, `src/particles/particle_injection.cpp`). Each
injector emits parcels, each of `weight` $w$ physical particles, at a rate,
from a site, with a velocity, a diameter and (with heat transfer) a
temperature of its own; injected solid particles carry their diameter and
weight per particle, so the forces, heat transfer and coupling of
eq. $\eqref{eq:solid}$ and after use each particle's own $d$ and $w$.

### Where, when and how many

- **Site.** A `point` (a nozzle smaller than the cells), the faces of
    boundary zones (`region` naming zones), or a filled region of the mesh
    or of shapes (`region` as a cell zone or a primitive). Parcels are
    distributed uniformly over the site: over a zone's faces by area, over
    a region by volume (cells by their volume times the fraction of them the
    region fills).
- **Rate.** `rate` $\dot N$ parcels per second, or `mass_flow` $\dot M$, from
    which
    $\dot N = \dot M / (w\,\rho_p\,\tfrac{\pi}{6}\langle d^3\rangle)$ with
    $\langle d^3\rangle$ the mean of $d^3$ over the injector's size
    distribution by number (in 2D, both per unit depth). An injector emits
    from `start` to `end` (for ever without it).
- **Velocity.** A vector `velocity`, or a `speed` along a `direction` (on
    boundary zones, by default each face's inward normal), spread uniformly
    over a cone of half angle `cone_angle` $\theta_c$ about it: in 3D uniform
    in solid angle, with the polar angle $\theta$ from
    $\cos\theta = 1 - \xi_1 (1 - \cos\theta_c)$ and the azimuth
    $2\pi\xi_2$; in 2D uniform in angle, $\theta = \theta_c(2\xi_1 - 1)$, with
    $\xi_1, \xi_2$ uniform in $[0, 1)$. The speed is the same for all
    particles of an injector.

### Size distributions

`diameter` is a number, every particle of that diameter, or a distribution of
the probability density $f(d)$, by number or by mass (`basis`), clipped to
`[min, max]`:

$$
\begin{equation}
\begin{aligned}
&\text{Rosin-Rammler:} && 1 - F(d) = e^{-(d/X)^q}, \quad f = \frac{q}{X}\left(\frac{d}{X}\right)^{q - 1} e^{-(d/X)^q}, \\
&\text{log-normal:} && f \propto \frac{1}{d}\exp\left[-\frac{1}{2}\left(\frac{\ln(d / d_m)}{\sigma}\right)^2\right], \\
&\text{Nukiyama-Tanasawa:} && f \propto \left(\frac{d}{X}\right)^{p} e^{-(d/X)^q}, \quad p > -1, \\
&\text{uniform:} && f \propto 1 \text{ on } [d_{min}, d_{max}], \\
&\text{table:} && f \text{ piecewise linear in } d \text{ through the given points},
\end{aligned}
\label{eq:injection-distributions}
\end{equation}
$$

with the size $X$ (`size`) and spread $q$ (`spread`), the median $d_m$
(`median`) and the standard deviation $\sigma$ of $\ln d$ (`sigma`), and the
exponent $p$ (`exponent`), the forms of sprays and powders
([Lefebvre & McDonell 2017](../references.md#lefebvre-mcdonell-2017)). A
distribution by mass is the number distribution weighted by $d^3$, so it is
sampled by number from $f(d)\,d^{-3}$. Without `min` and `max` a distribution
of unbounded support is clipped to its $10^{-9}$ and $1 - 10^{-9}$ quantiles.
`mallard check` reports each injector's mean diameter $D_{10}$ and Sauter
mean diameter $D_{32} = \langle d^3\rangle / \langle d^2\rangle$.

### Numerics

**Parcels per step.** After every accepted step, each injector emits the
parcels it owes for the part of the step within $[t_{start}, t_{end})$:
$\dot N\,\Delta t'$ plus the fraction of a parcel carried from the previous
step, whose whole part enters now and whose remainder is carried on (a
parcel due but for the rounding of the step times, within a relative
$4\epsilon$ of the time, enters now). The mean rate is therefore exact
whatever the steps, and a step the step control rejects advances no time and
injects nothing. A parcel enters at the end of the step in which it is due,
at a point drawn from the site and with its injection velocity: the entry
times are quantized to the step, so the spacing of the parcels along their
paths, not their number, carries an error of up to one step, which vanishes
as $\Delta t \to 0$.

**Uniform points.** Each injector holds the cumulative weights of its
elements (faces or cells) over every rank, ordered by global cell id and face
centroid, so it is the same on any partition. A parcel draws its element by
weight; the rank owning it places the parcel in one of the element's
simplices (a face is a fan of segments or triangles, a cell of triangles or
tetrahedra) drawn by measure, at barycentric weights from the spacings of
sorted uniform numbers, which is uniform in the simplex. In a region of
shapes, points outside the shape's signed distance are drawn again, up to
1024 times per parcel (the mean cost is the cut cells' volume over the
region's); a parcel that misses every time takes a point of the cell inside
the region found at setup, and cells the region barely touches are dropped.

**Diameters** are sampled by inverting the cumulative distribution of
$f(d)$ (by number) tabulated on 4096 intervals of $\ln d$ between the clip
bounds by the trapezoidal rule, linear in $\ln d$ within an interval. The
moments that set the rate from `mass_flow` come from the same table, so the
mean mass flow of the sampled parcels is the one given.

**Random numbers.** Draw $n$ of parcel $i$ of injector $j$ is a counter-based
hash of (`seed`, $j$, $i$, $n$), SplitMix64's finalizer chained over the key
([Steele, Lea & Flood 2014](../references.md#steele-2014)), uniform in
$[0, 1)$ with 53 random bits. A parcel's numbers depend only on its count
within its injector: not on the rank, the thread, the order of kernels or
where the steps fall. Every rank draws every parcel's element and only its
owner places it, so the same particles enter, with ids in order of
injection, on any rank count; there is no generator state to save, and
restart files carry each injector's count and carried fraction and the next
id, so a restarted run continues the injection bit for bit. The four kernels
of a step's injection (the counts, the local parcels, the free slots, the
insertion of the $k$-th local parcel into the $k$-th free slot) have fixed
sizes and are captured in the step graph. By default the capacity holds the
seeds plus $\lceil\dot N\,(t_{end} - t_{start})\rceil + 1$ parcels per
injector up to the end of the run; parcels that find no free slot are counted
and reported.
