# Lagrangian particles

Mallard follows point particles through the flow, each with its own
position and, for solid particles, its own velocity and temperature, advanced
by the time integrator of the flow together with it. This chapter gives the
models as implemented (`src/particles/`): the equations of motion, every
force with its assumptions and range of validity, the drag laws with their
Reynolds, Mach and Knudsen number dependence, the heat transfer, and the
interaction with walls and other boundaries. How the equations are
discretized (the gas at a particle, location, integration in time, bounces,
migration across ranks, data layout) and how accurate that is:
[Numerical methods for particles](../numerics/particles.md).
Usage: [Particles](../guide/particles.md); measured orders:
[verification manual](../verification/index.md#lagrangian-particles); design
rationale: [particles](../design/particles.md).

## Assumptions

The particles are dilute, one-way coupled point particles:

- **Dilute and one-way coupled.** The gas moves the particles; they do not
  act on the gas or on each other (no collisions, no wakes of one particle
  felt by another). That holds for volume fractions below about $10^{-6}$,
  and for mass loadings small enough that the particles' momentum and
  energy are negligible against the gas's
  ([Elghobashi 1994](../references.md#elghobashi-1994),
  [Balachandar & Eaton 2010](../references.md#balachandar-eaton-2010)).
- **Point particles.** Each particle is smaller than the cells and the
  smallest scales of the flow around it, so that the gas state at its center,
  the *undisturbed* flow interpolated from the resolved field, drives it,
  through correlations for an isolated sphere in a locally uniform (or
  linearly varying) flow.
- **Rigid spheres** of one diameter $d$ and one material density $\rho_p$
  for all particles of a run, which neither deform, break up nor rotate in a
  way that enters the forces (no Magnus lift).
- **Uniform particle temperature** (with heat transfer): conduction within
  the particle is fast against the transfer at its surface, a Biot number
  $Bi = h_c d / (6 k_p) \ll 1$ with $h_c$ the heat transfer coefficient and
  $k_p$ the particle's conductivity.
- **A single gas** ([Governing equations](governing_equations.md)): solid
  particles need its viscosity, so `[physics] type = "navier_stokes"`;
  tracers also move in mixtures.

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

A solid particle has a velocity $\mathbf{v}$ of its own, set by the forces
on it. Of the Maxey-Riley equation for a small sphere in a nonuniform flow
([Maxey & Riley 1983](../references.md#maxey-riley-1983)), Mallard keeps the
drag, gravity and buoyancy always, and adds the pressure-gradient,
added-mass, Saffman lift and Basset history forces on request (`forces`). Per unit particle
volume $V_p = \pi d^3 / 6$,

$$
\begin{equation}
\left(\rho_p + C_A \rho\right)\frac{\mathrm{d}\mathbf{v}}{\mathrm{d}t} =
\underbrace{\frac{18\,\mu f}{d^2}(\mathbf{u} - \mathbf{v})}_{\text{drag}}
+ \underbrace{\rho_p\,\mathbf{g}}_{\text{gravity}}
+ \underbrace{\mathbf{f}_b}_{\text{buoyancy}}
+ \underbrace{C_A\left(\rho\,\mathbf{g}_s - \nabla p\right)}_{\text{added mass}}
+ \underbrace{\mathbf{f}_L}_{\text{lift}}
+ \underbrace{\mathbf{f}_H}_{\text{history}}, \qquad
\frac{\mathrm{d}\mathbf{x}_p}{\mathrm{d}t} = \mathbf{v},
\label{eq:momentum}
\end{equation}
$$

where $\rho$, $\mathbf{u}$, $p$ and $\mu$ are the gas state at the particle
(the viscosity at the gas temperature $T = p / (\rho R)$, from the viscosity
law of `[physics]`), $f = C_D Re_p / 24$ is the drag correction of the drag
law (the drag over Stokes drag), with the particle Reynolds number

$$
\begin{equation}
Re_p = \frac{\rho \lvert\mathbf{u} - \mathbf{v}\rvert d}{\mu},
\label{eq:reynolds}
\end{equation}
$$

$\mathbf{g}$ the particles' gravity (`[particles] gravity`, by default
`[source] gravity`), $\mathbf{g}_s$ the gravity of `[source]` that
accelerates the gas, $C_A = 1/2$ with `"added_mass"` and 0 without, and the
buoyancy $\mathbf{f}_b = -\nabla p$ with `"pressure_gradient"` and
$-\rho\,\mathbf{g}$ without. Dividing by the effective density
$\rho_p + C_A\rho$ gives the relaxation form that the code integrates
(`particles::Solid::response`, `src/particles/particle_forces.h`):

$$
\begin{equation}
\frac{\mathrm{d}\mathbf{v}}{\mathrm{d}t} = \frac{\mathbf{u}(\mathbf{x}_p, t) - \mathbf{v}}{\tau} + \mathbf{a},
\qquad
\tau = \frac{(\rho_p + C_A\rho)\, d^2}{18\,\mu f(Re_p, \dots)}, \qquad
\mathbf{a} = \frac{\rho_p\,\mathbf{g} + \mathbf{f}_b + C_A(\rho\,\mathbf{g}_s - \nabla p) + \mathbf{f}_L + \mathbf{f}_H}{\rho_p + C_A\rho},
\label{eq:solid}
\end{equation}
$$

with the relaxation time $\tau$, which reduces to the Stokes relaxation time
$\tau_{St} = \rho_p d^2 / (18\mu)$ for Stokes drag without added mass, and
the acceleration $\mathbf{a}$ of the forces other than drag. Without the
optional forces, $\mathbf{a}$ is the reduced gravity
$\mathbf{g}' = \mathbf{g}(1 - \rho / \rho_p)$ (for $\mathbf{g} = \mathbf{g}_s$).
Every quantity in $\tau$ and $\mathbf{a}$ is evaluated at each stage, from
that stage's gas state at the particle and its own velocity, so $\tau$
changes as the particle speeds up or slows down and as it moves through gas
of another temperature or density.

The drag, gravity and buoyancy dominate for particles much denser than the
gas, $\rho_p / \rho \gg 1$ (solid particles and droplets in gas, where
$\rho_p / \rho \sim 10^3$): the pressure-gradient and added-mass forces
scale with $\rho / \rho_p$, and the lift with $(\rho / \rho_p)^{1/2}$ at
most. The forces are described below; the
[user guide](../guide/particles.md#solid-particles) says when to add them.

### Drag laws

The drag on a sphere moving at the slip velocity $\mathbf{u} - \mathbf{v}$
relative to the gas is

$$
\begin{equation}
\mathbf{F}_D = \tfrac12\,\rho\,C_D\,\frac{\pi d^2}{4}\,\lvert\mathbf{u} - \mathbf{v}\rvert\,(\mathbf{u} - \mathbf{v})
= 3\pi\mu d\,f\,(\mathbf{u} - \mathbf{v}), \qquad f = \frac{C_D Re_p}{24},
\label{eq:drag}
\end{equation}
$$

which is the drag term of eq. $\eqref{eq:momentum}$ times $V_p$. `[particles] drag`
chooses $C_D$:

- **`stokes`**: $f = 1$, $C_D = 24 / Re_p$, the drag of a sphere in creeping
    flow ([Stokes 1851](../references.md#stokes-1851)). It neglects the
    inertia of the gas around the sphere, compressibility and slip at the
    surface: valid for $Re_p \lesssim 0.1$, $M \ll 1$ and $Kn \lesssim 0.01$
    (definitions below). Against Schiller-Naumann its drag is 3% low at
    $Re_p = 0.1$, 5% at 0.2 and 13% at 1.
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
    value $C_D = 0.44$, up to the drag crisis near $Re_p = 2 \times 10^5$,
    which neither branch represents. The two branches meet within 0.4%. A
    function of $Re_p$ alone, it has no Mach or Knudsen number correction:
    valid for $M \lesssim 0.3$ and $Kn \lesssim 0.01$.

- **`singh`**: the correlation of
    [Singh et al. 2022](../references.md#singh-2022) (their Appendix C),
    for flow from continuum to free-molecular and from subsonic to
    hypersonic slip, in full below.

The [user guide](../guide/particles.md#drag-laws) tabulates where each law is
valid.

#### Singh et al.: Mach and Knudsen number corrections

Besides $Re_p$, the correlation reads the slip Mach number, the Knudsen
number (the gas's mean free path $\lambda$ over the diameter, from the
hard-sphere relation $\lambda = (\mu / \rho c)\sqrt{\pi\gamma / 2}$) and the
temperature ratio of the particle to the gas,

$$
\begin{equation}
M = \frac{\lvert\mathbf{u} - \mathbf{v}\rvert}{c}, \qquad
Kn = \frac{\lambda}{d} = \frac{\mu}{\rho c d}\sqrt{\frac{\pi\gamma}{2}} = \sqrt{\frac{\pi\gamma}{2}}\,\frac{M}{Re_p}, \qquad
\frac{T_p}{T},
\label{eq:singh-numbers}
\end{equation}
$$

with $c = \sqrt{\gamma p / \rho}$ the sound speed and $T_p / T = 1$ without
heat transfer. $Kn$ is computed from the gas state rather than as
$M / Re_p$, so it stays finite at zero slip. With $\omega = 0.74$ the
exponent of the viscosity's temperature dependence, it composes four
pieces:

1. **A normal shock** ahead of the sphere above Mach 1: the velocity, Mach
    number and temperature behind it,

    $$
    \begin{equation}
    \frac{U_s}{U} = \frac{2 + (\gamma - 1) M^2}{(\gamma + 1) M^2}, \quad
    M_s^2 = \frac{(\gamma - 1) M^2 + 2}{2\gamma M^2 - (\gamma - 1)}, \quad
    \frac{T_s}{T} = \frac{\left((\gamma - 1) M^2 + 2\right)\left(2\gamma M^2 - (\gamma - 1)\right)}{(\gamma + 1)^2 M^2},
    \label{eq:singh-shock}
    \end{equation}
    $$

    curved by $\alpha = 1 / (\alpha_0 M + 1 - \alpha_0)$, $\alpha_0 = 0.356$;
    below Mach 1, $U_s = U$, $M_s = M$, $T_s = T$ and $\alpha = 1$.

2. **The continuum drag**, Abraham's drag of a sphere
    ([Abraham 1970](../references.md#abraham-1970)) in the post-shock gas,
    with $\delta_0 = 9.4$ and $C_0 = 24 / \delta_0^2$, plus the pressure drag
    of the shock, which takes $C_D$ to $C_{D,\infty} = 0.9$ at hypersonic
    speed:

    $$
    \begin{equation}
    \begin{gathered}
    C_{D,c} = C_1 \left(1 - \alpha\,\frac{U_s}{U}\right) + C_0\,\Theta_s \left(1 + \frac{\delta_0}{\sqrt{\widetilde{Re}_s}}\right)^2, \qquad
    \Theta_s = \left(1 + \frac{\gamma - 1}{2} M_s^2\right)^{\gamma / (\gamma - 1)}, \\
    \frac{Re_p}{\widetilde{Re}_s} = \left(\alpha^2\,\frac{T_s}{T}\right)^{\omega}\Theta_s^{-q}, \quad
    q = \frac{\gamma + 1}{2\gamma} - \frac{(\gamma - 1)\,\omega}{\gamma}, \quad
    C_1 = \frac{C_{D,\infty} - C_0 \left(1 + \frac{(\gamma - 1)^2}{4\gamma}\right)^{\gamma / (\gamma - 1)}}{1 - \frac{\gamma - 1}{(\gamma + 1)\,\alpha_0 M}},
    \end{gathered}
    \label{eq:singh-continuum}
    \end{equation}
    $$

    with $C_1 = 0$ below Mach 1. $\Theta_s$ is the stagnation over static
    pressure of the post-shock gas, and $\widetilde{Re}_s$ the Reynolds
    number at its stagnation state. At low Mach number $C_{D,c}$ reduces to
    Abraham's $C_0(1 + \delta_0 / \sqrt{Re_p})^2$, which tends to Stokes
    drag, $24 / Re_p$, as $Re_p \to 0$.

3. **A rarefaction correction** of the continuum drag: the slip correction
    of Cunningham's form with the constants of
    [Davies 1945](../references.md#davies-1945) for $Kn = \lambda / d$,
    times a second correction in the rarefaction parameter $Wr$ (Singh et
    al.'s eqs. C3 to C5),

    $$
    \begin{equation}
    f_{Kn} = \frac{1}{\left(1 + Kn\left(A_1 + A_2\,e^{-A_3 / Kn}\right)\right)\left(1 + \alpha_h\,Wr\right)}, \qquad
    Wr = Kn\,M^{2\omega - 1}\sqrt{\frac{2}{\pi\gamma}}\left(1 + \frac{T_p}{T_s}\right)^{\omega},
    \label{eq:singh-slip}
    \end{equation}
    $$

    with $A_1 = 2.514$, $A_2 = 0.8$, $A_3 = 0.55$ and $\alpha_h = 1.27$.

4. **The free-molecular drag** of a sphere with diffuse reflection at the
    temperature $T_p$, in the speed ratio $s = M\sqrt{\gamma / 2}$,

    $$
    \begin{equation}
    C_{D,fm} = \frac{(1 + 2s^2)\,e^{-s^2}}{\sqrt{\pi}\,s^3} + \frac{(4s^4 + 4s^2 - 1)\operatorname{erf}(s)}{2 s^4}
    + \frac{2\sqrt{\pi}}{3 s}\sqrt{\frac{T_p}{T}},
    \label{eq:singh-free-molecular}
    \end{equation}
    $$

    whose first two terms cancel catastrophically at small $s$; below
    $s = 0.3$ their sum is evaluated by its Taylor series to $s^8$.

The drag bridges the continuum and free-molecular limits:

$$
\begin{equation}
C_D = \frac{C_{D,c}\,f_{Kn} + C_{D,fm}\,B}{1 + B}, \qquad
B = Br^{\eta}, \qquad
Br = Kn\sqrt{\frac{2}{\pi\gamma}}\left(1 + \frac{T_p}{T_s}\right)^{\omega}\left(M^{2\omega - 1} + 1\right), \qquad \eta = 1.8.
\label{eq:singh}
\end{equation}
$$

It reduces to Stokes drag in creeping continuum flow ($Re_p, M, Kn \to 0$)
and to the free-molecular drag as $Kn \to \infty$. Fitted to experiments and
DSMC from $M = 0.1$ to 10 and creeping flow to $Re_p \approx 10^4$, its
relative L2 error against the data is 7.6%, largest in the transition regime
$0.1 \lesssim Kn \lesssim 10$. It neglects real-gas effects behind strong
shocks (above Mach 5 the gas is taken calorically perfect), boundary-layer
turbulence and the drag crisis. The code evaluates $C_D Re_p / 24$ directly
(`particles::singh_drag_factor`), so that $f$ is finite at zero slip.

![Drag coefficient of a sphere against its Reynolds number: Stokes drag, the Schiller-Naumann correlation and Singh et al.'s at slip Mach numbers 0.0001, 0.5 and 2 in air](../images/theory/particle_drag-light.svg#gh-light-mode-only)
![Drag coefficient of a sphere against its Reynolds number: Stokes drag, the Schiller-Naumann correlation and Singh et al.'s at slip Mach numbers 0.0001, 0.5 and 2 in air](../images/theory/particle_drag-dark.svg#gh-dark-mode-only)

*Singh et al.'s drag at a fixed slip Mach number in air ($\gamma = 1.4$): its
Knudsen number grows as $M / Re_p$, so at low $Re_p$ rarefaction lowers the
drag below Stokes', and above Mach 1 the shock raises it.*

### Gravity, buoyancy and the pressure-gradient force

Gravity acts on the particle's mass, $\rho_p\,\mathbf{g}$ per volume. The gas
pushes on its surface with the pressure of the undisturbed flow, a force
$-\nabla p$ per volume ([Maxey & Riley 1983](../references.md#maxey-riley-1983)),
which in gas at rest under gravity is the buoyancy $-\rho\,\mathbf{g}$:

- by default the buoyancy is $-\rho\,\mathbf{g}$, exact in hydrostatic gas
  and wherever the gas's own acceleration is small against $\mathbf{g}$;
- with `"pressure_gradient"` it is $-\nabla p$, which also carries the
  force of the gas's acceleration on the volume the particle displaces
  (the gas around it accelerates under $-\nabla p$, and pulls the particle
  along): it matters for particles not much denser than the gas, and in
  strong pressure gradients (shocks, nozzles, vortex cores), where it
  adds $(\rho/\rho_p)\,\mathrm{D}\mathbf{u}/\mathrm{D}t$ to the
  acceleration.

$\nabla p$ is the least-squares gradient of the particle's cell
([numerics](../numerics/particles.md#gradients-for-the-forces)).

### Added mass

A particle that accelerates relative to the gas also accelerates the gas
around it. For a sphere in inviscid flow the reaction is the added mass,
half the mass of gas it displaces, times the relative acceleration
([Maxey & Riley 1983](../references.md#maxey-riley-1983); $C_A = 1/2$ holds
in unsteady nonuniform rotational flow too,
[Auton, Hunt & Prud'homme 1988](../references.md#auton-1988)):

$$
\begin{equation}
\mathbf{f}_A = C_A\,\rho\left(\frac{\mathrm{D}\mathbf{u}}{\mathrm{D}t} - \frac{\mathrm{d}\mathbf{v}}{\mathrm{d}t}\right), \qquad
\rho\,\frac{\mathrm{D}\mathbf{u}}{\mathrm{D}t} \approx \rho\,\mathbf{g}_s - \nabla p,
\label{eq:added-mass}
\end{equation}
$$

with the gas's acceleration taken from its inviscid momentum equation (the
viscous stress and the other sources of `[source]` left out). The term in
$\mathrm{d}\mathbf{v}/\mathrm{d}t$ moves to the left of
eq. $\eqref{eq:momentum}$, which gives the effective density
$\rho_p + \rho / 2$ in $\tau$ and $\mathbf{a}$. It matters for
$\rho_p / \rho$ of order 1 (bubbles, light particles), and changes the
response of heavy particles by $\rho / (2\rho_p)$.

### Saffman lift

A sphere moving relative to a gas in shear feels a lift across the slip.
`"saffman_lift"` adds the lift of
[Saffman 1965](../references.md#saffman-1965) with the coefficient of its
corrigendum ([Saffman 1968](../references.md#saffman-1968)), in the
vector form of [Crowe et al. 2011](../references.md#crowe-2011):

$$
\begin{equation}
\mathbf{F}_L = 1.615\, d^2 \sqrt{\frac{\rho\,\mu}{\lvert\boldsymbol{\omega}\rvert}}\,(\mathbf{u} - \mathbf{v})\times\boldsymbol{\omega}, \qquad
\mathbf{f}_L = \frac{\mathbf{F}_L}{V_p}, \qquad
\boldsymbol{\omega} = \nabla\times\mathbf{u},
\label{eq:saffman}
\end{equation}
$$

with the vorticity from the cell's least-squares velocity gradient (in 2D,
$\boldsymbol{\omega} = \omega_z\,\mathbf{e}_z$), and no lift where
$\boldsymbol{\omega} = 0$. Saffman's analysis assumes
$Re_p \ll Re_G^{1/2} \ll 1$, with the shear Reynolds number
$Re_G = \rho\,\lvert\boldsymbol{\omega}\rvert\,d^2 / \mu$: a small particle
in a strong shear, far from walls. Outside that range it overestimates the
lift, which decays as $Re_p$ grows past $Re_G^{1/2}$
([Crowe et al. 2011](../references.md#crowe-2011)). The lift
scales as $(\rho/\rho_p)^{1/2}$ relative to the drag at fixed $Re_G$, so it
matters for particles near walls in boundary layers and in strong shear
layers.

### Basset history force

When the slip $\mathbf{w} = \mathbf{u} - \mathbf{v}$ changes, the vorticity
it sheds diffuses away from the sphere over the viscous time $d^2 / \nu$,
and the drag lags behind its quasi-steady value. In creeping flow this
memory is the force of
[Boussinesq 1885](../references.md#boussinesq-1885) and
[Basset 1888](../references.md#basset-1888), the last term of the
Maxey-Riley equation
([Maxey & Riley 1983](../references.md#maxey-riley-1983)):

$$
\begin{equation}
\mathbf{f}_H = \frac{9}{d}\sqrt{\frac{\rho\,\mu}{\pi}}\;\mathbf{I}(t), \qquad
\mathbf{I}(t) = \int_{t_0^-}^{t} \frac{\mathrm{d}\mathbf{w}(s)}{\sqrt{t - s}},
\label{eq:basset}
\end{equation}
$$

per particle volume ($\tfrac{3}{2} d^2 \sqrt{\pi\rho\mu}$ times
$\mathbf{I}$ per particle), with $\mathrm{d}\mathbf{w}/\mathrm{d}s$ taken
along the particle's path, $\rho$ and $\mu$ of the gas at the particle, and
the history starting at the particle's first step $t_0$. A particle seeded
with a `velocity` other than the gas's starts impulsively: its slip jumps from
0 at $t_0$, and $\mathbf{I}$ holds $\mathbf{w}(t_0) / \sqrt{t - t_0}$.
`"basset_history"` adds it. Relative to the drag it scales as
$(\rho / \rho_p)^{1/2} (\tau / T)^{1/2}$ for a slip that changes over a time
$T$: it matters for $\rho_p / \rho$ of order 1 and in rapid accelerations
(shocks, impulsive starts), and changes the settling of heavy particles in
gas by a few percent at most
([Crowe et al. 2011](../references.md#crowe-2011)). The kernel
$1 / \sqrt{t - s}$ is that of creeping flow, $Re_p \ll 1$; at finite $Re_p$
the memory decays faster at long times
([Mei & Adrian 1992](../references.md#mei-adrian-1992)), so the force is an
upper bound there, as for the Stokes drag it accompanies.

The kernel is a sum of exponentials,
$1 / \sqrt{t} \approx \sum_k a_k e^{-\lambda_k t}$, the trapezoidal rule on
$1/\sqrt{t} = \pi^{-1/2} \int e^{s/2} \exp(-e^s t)\,\mathrm{d}s$
([Beylkin & Monzón 2010](../references.md#beylkin-monzon-2010)): 29 rates,
two per decade, within $7.3 \times 10^{-4}$ of $1/\sqrt{t}$ from
$10^{-6}\,t_{ref}$ to $10^{3}\,t_{ref}$, with $t_{ref}$ the particle's
Stokes relaxation time at its first step. Then
$\mathbf{I} = \sum_k a_k \mathbf{z}_k$ with
$\mathbf{z}_k = \int e^{-\lambda_k (t - s)}\,\mathrm{d}\mathbf{w}(s)$,
which each particle carries in place of its whole path: the cost and memory
per particle are bounded, and restarts and migration carry the history. The
memory older than $10^{3}\,t_{ref}$ fades, and the last $10^{-6}\,t_{ref}$
is missed, an error of about $10^{-3}\,\lvert\mathbf{I}\rvert$. How the
$\mathbf{z}_k$ advance in time:
[numerics](../numerics/particles.md#integration-in-time).

### Forces left out

- The **compressible unsteady force** of a shock passing a particle, which
  dominates its first few acoustic times $d / c$, and the history force at
  finite $Re_p$ ([#1462](https://github.com/MatthewBonanni/mallard/issues/1462)).
- The **Magnus lift** of a rotating sphere: particles have no rotation.
- **Thermophoresis**, **Brownian motion** and electrostatic forces, which
  matter for sub-micron particles.
- **Collisions** between particles and the forces of other particles' wakes
  (dense suspensions).

### Stokes number and terminal velocity

The ratio of $\tau$ to a time scale $\tau_f$ of the flow is the **Stokes
number** $St = \tau / \tau_f$. For $St \ll 1$ the particle follows the gas
(a tracer, up to the settling velocity); for $St \gg 1$ it moves almost
ballistically, the gas changing its velocity only over many $\tau_f$; near
$St = 1$ it lags the gas by about its own relaxation and is flung out of
vortices into the strain between them (preferential concentration,
[Squires & Eaton 1991](../references.md#squires-eaton-1991)). For
$\tau \ll \tau_f$ the particle's velocity is the expansion
([Maxey 1987](../references.md#maxey-1987),
[Ferry & Balachandar 2001](../references.md#ferry-balachandar-2001))

$$
\begin{equation}
\mathbf{v} = \mathbf{u} + \tau\left(\mathbf{a} - \frac{\mathrm{D}\mathbf{u}}{\mathrm{D}t}\right) + O(\tau^2),
\label{eq:equilibrium}
\end{equation}
$$

the gas velocity plus a settling drift and a lag behind the gas's
acceleration.

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

## Heat transfer

With `specific_heat` $c_p$ (the particle material's specific heat) a solid
particle carries a temperature $T_p$, uniform within it (Biot number small,
above), which exchanges heat with the gas by convection at its surface:

$$
\begin{equation}
\rho_p\,c_p\,\frac{\pi d^3}{6}\,\frac{\mathrm{d}T_p}{\mathrm{d}t} = \pi d\,k\,Nu\,(T - T_p)
\quad\Longleftrightarrow\quad
\frac{\mathrm{d}T_p}{\mathrm{d}t} = \frac{T - T_p}{\tau_T}, \qquad
\tau_T = \frac{\rho_p\,c_p\,d^2}{6\,k\,Nu},
\label{eq:heat}
\end{equation}
$$

with $T$ the gas temperature at the particle, $k = \mu c_{p,g} / Pr$ the gas
conductivity at that temperature (the gas's own $c_{p,g}$ and $Pr$ of
`[physics]`), and the Nusselt number of
[Ranz & Marshall 1952](../references.md#ranz-marshall-1952),

$$
\begin{equation}
Nu = 2 + 0.6\,Re_p^{1/2}\,Pr^{1/3},
\label{eq:ranz-marshall}
\end{equation}
$$

from the same $Re_p$ as the drag. $Nu = 2$ is conduction into gas at rest
around a sphere; the second term is the convection of the boundary layer of
a sphere in laminar flow, measured on evaporating drops for $Re_p$ up to
about 200. The ratio of the thermal to the momentum relaxation time is, for
Stokes drag without added mass, $\tau_T / \tau = 3\,(c_p / c_{p,g})\,Pr / Nu$.

The temperature is a state of the particle only: it enters the drag through
$T_p / T$ of Singh et al.'s law, and the gas temperature at the particle drives
it, but the particle's heat does not return to the gas (one-way coupling).
Left out: radiation (the particle neither emits nor absorbs, also with the
optically thin radiation of the gas,
[Chemistry and radiation](chemistry_radiation.md)), the variation of the gas
properties across the boundary layer (all at the gas temperature, not a film
temperature), Mach and Knudsen number corrections to $Nu$, and phase change.
Particles start at the gas temperature at their position unless
`temperature` gives theirs.

## The gas at a particle

The gas state $\mathbf{W} = [\rho, \mathbf{u}, p]$ at a particle comes from
the polynomial the reconstruction already builds in the particle's cell
(`Particles::evaluate_gas`, `src/particles/particles.cpp`), not from the cell
averages:

- with TENO of order $k$, the cell's central polynomial of degree $k - 1$
  ([Finite-volume discretization and reconstruction](finite_volume.md#reconstruction)),
  where TENO finds the cell smooth and the value at the particle is
  admissible ($\rho, p > 0$): an error $O(h^k)$;
- elsewhere (troubled cells, `FO` and `MUSCL`), the cell's linear
  least-squares fit $\overline{\mathbf{W}}_i + \nabla\mathbf{W}_i \cdot (\mathbf{x} - \mathbf{x}_i)$,
  with the gradient of the viscous fluxes, or the cell average where the
  linear value is not admissible: an error $O(h^2)$.

The derived values the particle reads (the temperature, the viscosity, the
sound speed, the particle expressions of the quantities) are computed from
this $\mathbf{W}$; the gradients the pressure-gradient, added-mass and lift
forces read are the cell's least-squares gradients. Every stage of a step
evaluates the polynomial of the cell that held the particle at the start of
the step, so that the right-hand side is one smooth function of the position
within the step:
[numerics](../numerics/particles.md#the-gas-at-a-particle).

## Integration in time

The particles are a block of the integrator's state, next to the cell
averages and the species: per particle its position, its velocity and
temperature (solid particles) and its integrated quantities. Every
Runge-Kutta method of
[Time integration](time_integration.md#runge-kutta-methods) advances the
block with its own stages and weights, at the stage times of the flow,
evaluating eqs. $\eqref{eq:tracer}$, $\eqref{eq:solid}$ and
$\eqref{eq:heat}$ with each stage's gas, position, velocity and
temperature. The trajectories therefore converge at the integrator's order
in time and at the reconstruction's order in space, as the flow does; a
rejected step of the error-controlled integrator restores the particles with
the flow.

A relaxation faster than the step, $\Delta t / \tau > 1$ (or
$\Delta t / \tau_T > 1$), which an explicit method would integrate unstably
above $\Delta t / \tau \approx 2$ to 2.8, is integrated exponentially
instead, exactly for a gas velocity and acceleration linear in time over the
step. The flow's step never depends on the particles. The scheme, its error
and the measured orders:
[numerics](../numerics/particles.md#integration-in-time).

## Walls and other boundaries

A particle that reaches a boundary face does what the face's boundary
condition sets:

- **interior and partition faces**: the particle crosses; on several ranks it
  moves to the rank owning its new cell;
- **periodic faces**: the particle crosses into the neighbor's frame, its
  position (and velocity) translated or rotated with the periodic map;
- **inlets, outlets and far field**: the particle leaves the domain and is
  removed;
- **walls and symmetry planes**: the particle is reflected, or removed as
  deposited where `[[particles.walls]]` sets `interaction = "stick"`.

A solid particle moving into a reflecting wall of unit outward normal
$\mathbf{n}$, with normal velocity $v_n = \mathbf{v}\cdot\mathbf{n} > 0$,
leaves it with

$$
\begin{equation}
\mathbf{v}' = e_t\,(\mathbf{v} - v_n \mathbf{n}) - e_n\,v_n\,\mathbf{n},
\label{eq:reflect-velocity}
\end{equation}
$$

with the normal and tangential restitution coefficients $e_n$
(`restitution`) and $e_t$ (`tangential_restitution`) of the wall, each in
$[0, 1]$ and 1 by default (an elastic, frictionless bounce). This is the
hard-sphere bounce without rotation: $e_n$ is the ratio of the normal speeds
after and before, which for solid particles on solid walls falls from near 1
at low impact speeds as plastic deformation sets in, and $e_t < 1$ stands for
friction without modeling the particle's spin
([Crowe et al. 2011](../references.md#crowe-2011)). Symmetry planes reflect
with the coefficients of the wall entries that name them (1 otherwise), and
tracers, which follow the gas, are only mirrored in the face plane. A
particle a depth $\delta$ past a reflecting face at the end of a step is
moved back in by

$$
\begin{equation}
\mathbf{x}_p' = \mathbf{x}_p - (1 + e_n)\,\delta\,\mathbf{n},
\label{eq:reflect-position}
\end{equation}
$$

where it would be had it bounced at the wall with constant velocity. Solid
particles whose relaxation is slower than the step instead bounce at the
time within the step at which they reached the wall
([numerics](../numerics/particles.md#bounces-within-a-step)).

A sticking wall removes a particle that reaches it, counted as deposited,
which models capture by a wet, sticky or filtering surface. No splashing,
erosion or particle-wall adhesion model is implemented. A particle can bounce
off up to 8 faces in one step (corners); one that does not settle in a cell
within 64 cells of its walk, or reaches the outer edge of a rank's halo, is
counted as lost.

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
