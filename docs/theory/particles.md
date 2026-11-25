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

The particles are dilute point particles, one-way coupled by default:

- **Dilute and, by default, one-way coupled.** The gas moves the particles;
  they do not act on each other (no collisions, no wakes of one particle felt
  by another), nor on the gas. That holds for volume fractions below about
  $10^{-6}$, and for mass loadings small enough that the particles' momentum
  and energy are negligible against the gas's
  ([Elghobashi 1994](../references.md#elghobashi-1994),
  [Balachandar & Eaton 2010](../references.md#balachandar-eaton-2010)).
  With `coupling = "two-way"` solid particles act on the gas too, which
  extends this to volume fractions of about $10^{-3}$
  ([Two-way coupling](#two-way-coupling)).
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
added-mass and Saffman lift forces on request (`forces`). Per unit particle
volume $V_p = \pi d^3 / 6$,

$$
\begin{equation}
\left(\rho_p + C_A \rho\right)\frac{\mathrm{d}\mathbf{v}}{\mathrm{d}t} =
\underbrace{\frac{18\,\mu f}{d^2}(\mathbf{u} - \mathbf{v})}_{\text{drag}}
+ \underbrace{\rho_p\,\mathbf{g}}_{\text{gravity}}
+ \underbrace{\mathbf{f}_b}_{\text{buoyancy}}
+ \underbrace{C_A\left(\rho\,\mathbf{g}_s - \nabla p\right)}_{\text{added mass}}
+ \underbrace{\mathbf{f}_L}_{\text{lift}}, \qquad
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
\mathbf{a} = \frac{\rho_p\,\mathbf{g} + \mathbf{f}_b + C_A(\rho\,\mathbf{g}_s - \nabla p) + \mathbf{f}_L}{\rho_p + C_A\rho},
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

### Forces left out

- The **Basset history force**, the viscous memory of the particle's past
  accelerations, a convolution over its path: it matters for $\rho_p / \rho$
  of order 1 and rapidly accelerating particles, and is small for heavy
  particles in gas ([Crowe et al. 2011](../references.md#crowe-2011)).
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
it, but the particle's heat returns to the gas only with two-way coupling
([Two-way coupling](#two-way-coupling)).
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

1. **Exact sums per cell.** Each cell sums its particles' sources exactly,
    in integer limbs (`exact_sums.h`), and rounds each sum once to the
    nearest double: the result does not depend on the order of the slots,
    which differs between rank counts and migrations. The start cell is
    always owned, so no deposit lands in a halo.
2. **First and second moments** (with a width). Spreading from the cell
    centers would place every particle at its cell's center, an error of half
    a cell in the kernel's center. The cells also sum each source's first and
    second moments about their centers, $\mathbf{m}_c = \sum_p q_p(\mathbf{x}_p - \mathbf{x}_c)$
    and $\mathbf{M}_c = \sum_p q_p(\mathbf{x}_p - \mathbf{x}_c)(\mathbf{x}_p - \mathbf{x}_c)^T$,
    and correct each sum by the next two terms of the Taylor series of a point
    source about the cell center,
    $q\,\delta(\mathbf{x} - \mathbf{x}_c - \mathbf{d}) \approx q\,\delta - \nabla\cdot(q\mathbf{d}\,\delta) + \tfrac12\nabla\nabla : (q\mathbf{d}\mathbf{d}^T\delta)$,
    with the moment densities $\boldsymbol{\mu} = \mathbf{m} / V$ and
    $\boldsymbol{\Pi} = \mathbf{M} / V$ and one face-averaged divergence in
    flux form, applied once to $\boldsymbol{\Pi}$ and once to the corrected
    first moments,

    $$
    \begin{equation}
    \tilde{\boldsymbol{\mu}}_c = \boldsymbol{\mu}_c - \frac{1}{2 V_c}\sum_{f} \tfrac12\left(\boldsymbol{\Pi}_c + \boldsymbol{\Pi}_n\right)\mathbf{n}_f A_f,
    \qquad
    q_c \leftarrow q_c - \sum_{f} \tfrac12\left(\tilde{\boldsymbol{\mu}}_c + \tilde{\boldsymbol{\mu}}_n\right)\cdot\mathbf{n}_f A_f,
    \label{eq:coupling-moments}
    \end{equation}
    $$

    so that $q - \nabla\cdot\boldsymbol{\mu} + \tfrac12\nabla\cdot\nabla\cdot\boldsymbol{\Pi}$
    is discretized with a Hessian built from the same divergence, on any mesh.
    The deposit then has the particles' first and second moments: the spread
    converges at second order in the cell size rather than first, and its
    error does not depend on where the particles sit in their cells.
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
the gas a particle reads carries its own disturbance, which the
point-particle drag does not expect. A width in physical units makes the
deposit independent of the mesh, so it converges as the mesh is refined at
fixed $\sigma$: the volume fraction of a single particle at $(0.4137, 0.5862)$
against the Gaussian of $\sigma = 0.08$ has L1 errors $5.2 \times 10^{-2}$,
$1.1 \times 10^{-2}$, $2.1 \times 10^{-3}$ and $5.3 \times 10^{-4}$ on 16, 32,
64 and 128 cells per side (orders 2.3, 2.4 and 2.0), where the cell-center
deposit's error did not converge. The second moments make the error
independent of where the particle sits in its cell. With the first moments
alone, a particle at offset $\mathbf{d}$ from its cell's center starts with
covariance $-\mathbf{d}\mathbf{d}^T$ rather than zero, so the spread kernel's
covariance is $\sigma^2 \mathbf{I} - \mathbf{d}\mathbf{d}^T$: still second
order, but with a constant from $8 h^2$ at a cell's center to $34 h^2$ near a
corner at $\sigma = 0.08$, so that a particle at a fixed position, which sits
at a different offset on each mesh, converged unevenly (with the same
particle, $2.1 \times 10^{-3}$ on 64 cells and $1.9 \times 10^{-3}$ on 128).
On a uniform Cartesian mesh eq. $\eqref{eq:coupling-moments}$ gives the
pre-spread deposit the particle's integral and first and second moments
exactly, and the face-flux Laplacian adds exactly $2\,\Delta s$ to the second
moment per direction and step, so the spread's covariance is $\sigma^2
\mathbf{I}$ wherever the particle sits; what remains of the offset dependence
is of higher order ($8.3 h^2$ to $8.7 h^2$ on 128 cells). On other meshes
the corrections are consistent rather than exact: the second moments still
remove the leading offset dependence, but not to round-off. Each cell adds its faces in the order
of its neighbors' global ids, and halo values come from their owners before
each step, so the deposit, like the particles, is bitwise independent of the
rank count; on one rank its kernels are captured in the step graph.

![The L1 error over h squared of one particle's deposit against its offset from the cell center along the cell's diagonal, on 32 and 64 cells per side: flat with the second moments, rising fourfold toward the corner with the first moments alone](../images/theory/particle_deposition_offset-light.svg#gh-light-mode-only)
![The L1 error over h squared of one particle's deposit against its offset from the cell center along the cell's diagonal, on 32 and 64 cells per side: flat with the second moments, rising fourfold toward the corner with the first moments alone](../images/theory/particle_deposition_offset-dark.svg#gh-dark-mode-only)

*The $L_1$ error of one particle's deposit ($\sigma = 0.08$, as below)
against where it sits in its cell, along the cell's diagonal from its
center ($d = 0$) toward a corner ($d / h = 0.5$): Mallard runs, and the
same steps with the first moments alone (dashed, NumPy).*

![Left: the volume fraction of one particle deposited over a Gaussian of width 0.08 on 16, 32, 64 and 128 cells per side, along the particle's row of cells, against the Gaussian. Right: its L1 error against the cell size, at second order](../images/theory/particle_deposition-light.svg#gh-light-mode-only)
![Left: the volume fraction of one particle deposited over a Gaussian of width 0.08 on 16, 32, 64 and 128 cells per side, along the particle's row of cells, against the Gaussian. Right: its L1 error against the cell size, at second order](../images/theory/particle_deposition-dark.svg#gh-dark-mode-only)

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

## Evaporating droplets

`type = "droplet"` makes the particles droplets of one liquid that evaporate
into a gas mixture whose mechanism holds the liquid's vapor
(`particles::Droplet`, `src/particles/particle_evaporation.h`). A droplet
moves as a solid particle, eq. $\eqref{eq:solid}$, with the liquid's density
and its own, shrinking diameter, and the drag and forces of `[particles]`;
the gas heats it, and it evaporates until, below `minimum_diameter`, it is
removed.

### Assumptions

- **One component**, of constant density $\rho_l$ and specific heat $c_l$.
- **Uniform temperature** $T_p$ within the droplet (infinite liquid
  conductivity): no internal conduction or circulation.
- **Quasi-steady gas film**: the gas around the droplet adjusts to the
  droplet's state faster than it changes, so the transfer to a droplet in a
  uniform stream holds at every instant. The film's properties are evaluated
  at reference values by the 1/3 rule (below).
- **Equilibrium at the surface**: the vapor at the surface is at the
  saturation pressure of the surface temperature (Raoult's law of a pure
  liquid), with no non-equilibrium (Langmuir-Knudsen) correction, which
  matters for droplets below about 50 µm in hot gas
  ([Miller et al. 1998](../references.md#miller-harstad-bellan-1998)).
- **Subcritical**: the saturation curve ends at the critical point, and
  real-fluid (supercritical) droplets are outside the model.
- The gas composition at a droplet is its cell's average.

### The liquid

The liquid's enthalpy is referred to the mechanism's vapor enthalpy $h_v$
(formation enthalpy included), through the latent heat $L_b$ at the boiling
temperature $T_b$ at the pressure $p_b$:

$$
\begin{equation}
h_l(T) = h_v(T_b) - L_b + c_l\,(T - T_b), \qquad L(T) = h_v(T) - h_l(T),
\label{eq:droplet-enthalpy}
\end{equation}
$$

so that the latent heat at any temperature follows Kirchhoff's law, and the
liquid and the gas, whose total energy carries formation enthalpies, share one
energy reference. The input refuses a liquid whose $L(T)$ vanishes below the
critical temperature. The saturation pressure is Clausius-Clapeyron's through
the boiling point, or Antoine's with `antoine`,

$$
\begin{equation}
p_{sat}(T) = p_b \exp\left[\frac{L_b}{R_v}\left(\frac{1}{T_b} - \frac{1}{T}\right)\right]
\quad\text{or}\quad
\log_{10}\frac{p_{sat}}{\mathrm{Pa}} = A - \frac{B}{T/\mathrm{K} + C},
\label{eq:droplet-saturation}
\end{equation}
$$

with $R_v$ the vapor's gas constant, and the boiling temperature $T_b(p)$ at
the gas pressure is its inverse, at most the critical temperature.

### The surface and the film

The surface is at $T_s = \min(T_p, T_b(p))$. The vapor's mole fraction there
is $X_s = \min(p_{sat}(T_s) / p, 1)$, and its mass fraction

$$
\begin{equation}
Y_s = \frac{X_s W_v}{X_s W_v + (1 - X_s)\,W_{dry}}, \qquad
W_{dry} = \frac{1 - Y_v}{\sum_{k \ne v} Y_k / W_k},
\label{eq:droplet-surface}
\end{equation}
$$

with $W_v$ the vapor's molar mass, $Y_v$ the gas's vapor mass fraction and
$W_{dry}$ the molar mass of the gas's other species. The film between the
surface and the gas is evaluated at the reference state of the 1/3 rule,
as in Abramzon & Sirignano's model,

$$
\begin{equation}
T_f = T_s + \tfrac13\,(T - T_s), \qquad Y_{v,f} = Y_s + \tfrac13\,(Y_v - Y_s),
\label{eq:droplet-film}
\end{equation}
$$

the other species in the gas's proportions, with the mechanism's density
$\rho_f$, specific heat $c_{p,f}$, viscosity $\mu_f$, conductivity $k_f$ and
the vapor's mixture-averaged diffusivity $D_f$, and

$$
\begin{equation}
Re_p = \frac{\rho\,\lvert\mathbf{u} - \mathbf{v}\rvert\,d}{\mu_f}, \quad
Pr_f = \frac{\mu_f c_{p,f}}{k_f}, \quad
Sc_f = \frac{\mu_f}{\rho_f D_f}, \quad
Le_f = \frac{k_f}{\rho_f c_{p,f} D_f}, \quad
Sh_0 = 2 + 0.6\,Re_p^{1/2} Sc_f^{1/3}, \quad
Nu_0 = 2 + 0.6\,Re_p^{1/2} Pr_f^{1/3},
\label{eq:droplet-numbers}
\end{equation}
$$

the Sherwood and Nusselt numbers of
[Ranz & Marshall 1952](../references.md#ranz-marshall-1952) at the film's
properties. The drag uses the film's viscosity too, with the gas's
$\gamma = c_p / (c_p - R)$ and $T_p$ for Singh et al.'s law.

### Evaporation rate and heat transfer

With Spalding's mass transfer number
([Spalding 1953](../references.md#spalding-1953))

$$
\begin{equation}
B_M = \frac{Y_s - Y_v}{1 - Y_s},
\label{eq:droplet-spalding}
\end{equation}
$$

`evaporation` chooses the rate $\dot m$ at which the droplet loses mass and
the heat $Q$ the gas gives it:

- **`d2_law`**: the classical quasi-steady droplet,

    $$
    \begin{equation}
    \dot m = \pi d\,\rho_f D_f\,Sh_0 \ln(1 + B_M), \qquad Q = \pi d\,k_f\,Nu_0\,(T - T_s).
    \label{eq:droplet-d2}
    \end{equation}
    $$

    At rest ($Sh_0 = 2$) and at a constant surface state, $d^2$ falls
    linearly in time, $d^2 = d_0^2 - K t$ with
    $K = 8 \rho_f D_f \ln(1 + B_M) / \rho_l$.

- **`abramzon_sirignano`** (default): the film model of
    [Abramzon & Sirignano 1989](../references.md#abramzon-sirignano-1989),
    in which the Stefan flow of the vapor thickens the film by
    $F(B) = (1 + B)^{0.7} \ln(1 + B) / B$, and heat crosses it with its own
    transfer number $B_T$:

    $$
    \begin{equation}
    \begin{gathered}
    Sh^* = 2 + \frac{Sh_0 - 2}{F(B_M)}, \qquad Nu^* = 2 + \frac{Nu_0 - 2}{F(B_T)}, \qquad
    B_T = (1 + B_M)^{\phi} - 1, \qquad \phi = \frac{c_{p,v}}{c_{p,f}}\,\frac{Sh^*}{Nu^*}\,\frac{1}{Le_f}, \\
    \dot m = \pi d\,\rho_f D_f\,Sh^* \ln(1 + B_M), \qquad
    Q = \pi d\,k_f\,Nu^*\,(T - T_s)\,\frac{\ln(1 + B_T)}{B_T},
    \end{gathered}
    \label{eq:droplet-abramzon-sirignano}
    \end{equation}
    $$

    with $c_{p,v}$ the vapor's specific heat at $T_f$; $B_T$ and $Nu^*$ are
    found by eight fixed-point passes from $B_T = B_M$. The factor
    $\ln(1 + B_T)/B_T$ is the blowing of the vapor, which reduces the heat
    that reaches the surface. It is the usual choice for sprays in hot gas
    ([Miller et al. 1998](../references.md#miller-harstad-bellan-1998)).

A vapor-rich gas ($Y_v > Y_s$, $B_M < 0$) condenses on the droplet. The heat
into the liquid is what the gas gives less the latent heat of what
evaporates,

$$
\begin{equation}
M c_l\,\frac{\mathrm{d}T_p}{\mathrm{d}t} = Q - \dot m\,L(T_s), \qquad
\frac{\mathrm{d}M}{\mathrm{d}t} = -\dot m, \qquad M = \rho_l \frac{\pi d^3}{6},
\label{eq:droplet-energy}
\end{equation}
$$

so a droplet heats or cools toward its wet-bulb temperature, where the heat
from the gas just evaporates it.

![Left: the vapor mass fraction at the surface of water and Spalding's transfer number against the surface temperature at 1 atm, by Clausius-Clapeyron and Antoine. Right: Abramzon and Sirignano's film thickening, blowing factor and corrected Sherwood number against the transfer number](../images/theory/droplet_transfer_numbers-light.svg#gh-light-mode-only)
![Left: the vapor mass fraction at the surface of water and Spalding's transfer number against the surface temperature at 1 atm, by Clausius-Clapeyron and Antoine. Right: Abramzon and Sirignano's film thickening, blowing factor and corrected Sherwood number against the transfer number](../images/theory/droplet_transfer_numbers-dark.svg#gh-dark-mode-only)

*Left: water's surface vapor mass fraction $Y_s$, eq. $\eqref{eq:droplet-surface}$, and
$B_M$, eq. $\eqref{eq:droplet-spalding}$, in dry air at 1 atm, from
Clausius-Clapeyron through the boiling point and from Antoine's coefficients
of water; $B_M$ diverges at the boiling point, where $Y_s \to 1$. The dotted
lines are the wet-bulb temperatures of a droplet at rest in air at 1000 K,
where eq. $\eqref{eq:droplet-energy}$'s heat into the liquid vanishes. Right:
the factors of eq. $\eqref{eq:droplet-abramzon-sirignano}$: the Stefan flow
thickens the film by $F(B)$, lowering $Sh^*$ and $Nu^*$ in a stream (here
$Re_p = 100$, $Sc_f = 0.7$), and blowing reduces the heat reaching the surface
by $\ln(1 + B_T)/B_T$, already by 30% at $B_T = 1$.*

**Boiling.** A droplet at or above the boiling temperature at the gas
pressure, or whose surface would be pure vapor ($Y_s \to 1$), boils: its
surface is at $T_b(p)$, the heat it receives evaporates it, and its superheat
relaxes to the boiling temperature on its heat-up time
$\tau_h = \rho_l c_l d^2 / (6 k_f Nu_0)$:

$$
\begin{equation}
Q = \pi d\,k_f\,Nu_0\,(T - T_s), \qquad
\dot m = \frac{\max\left(Q + M c_l (T_p - T_s)/\tau_h,\; 0\right)}{L(T_s)}.
\label{eq:droplet-boiling}
\end{equation}
$$

Above the critical pressure the boiling temperature is the critical
temperature, and a droplet driven past it is evaluated at it.

### Sources in the gas

With `coupling = "two-way"` the gas receives, besides the momentum and energy
of eq. $\eqref{eq:coupling-sources}$, the vapor: per volume, over the deposit
of each parcel,

$$
\begin{equation}
S_\rho = S_{\rho Y_v} = \frac{1}{V}\sum_p w\,\dot m_p, \qquad
\mathbf{S}_{\rho\mathbf{u}} = -\frac{1}{V}\sum_p w\left(\frac{\mathrm{d}\mathbf{P}_p}{\mathrm{d}t} - M_p\,\mathbf{g}\right), \qquad
S_{\rho E} = -\frac{1}{V}\sum_p w\left(\frac{\mathrm{d}\mathcal{E}_p}{\mathrm{d}t} - \mathbf{g}\cdot\mathbf{P}_p\right),
\label{eq:droplet-sources}
\end{equation}
$$

with the droplet's momentum $\mathbf{P} = M\mathbf{v}$ and total energy
$\mathcal{E} = M(h_l(T_p) + \tfrac12\lvert\mathbf{v}\rvert^2)$ in the gas's
reference. The vapor enters at the droplet's velocity with the enthalpy
$h_l(T_p) + L(T_s)$, which is the vapor's $h_v(T_s)$ for a droplet at its
surface temperature, while the gas loses the heat $Q$ it gives: the
energy source is $\dot m\,(h_v + \tfrac12\lvert\mathbf{v}\rvert^2) - Q$ less
the work of the forces, per droplet. The vapor source makes the gas's mass
and its vapor species grow by the same amount, so the other species' mass
fractions fall with dilution. Removed droplets give the gas what they still
carry, their mass as vapor with their momentum and energy, after the step.

### Numerics

**Conserved block values.** After its position a droplet's integrated
values are its momentum $\mathbf{P}$, its energy
$E = M(c_l(T_p - T_b) + \tfrac12\lvert\mathbf{v}\rvert^2)$ and its mass $M$,
not $\mathbf{v}$, $T_p$ and $d$; $\mathbf{v} = \mathbf{P}/M$, $T_p$ from
$E$ and $d$ from $M$ are derived. The rates are
$\mathrm{d}M/\mathrm{d}t = -\dot m$,
$\mathrm{d}\mathbf{P}/\mathrm{d}t = M\,\mathrm{d}\mathbf{v}/\mathrm{d}t - \dot m\,\mathbf{v}$
and the matching $\mathrm{d}E/\mathrm{d}t$, and the gas receives the opposite
of these stage rates (plus $\dot m\,h_l(T_b)$, the change of reference). Every
Runge-Kutta method combines both linearly in the same way, so gas and droplets
together conserve mass, every species, momentum and energy to round-off,
whatever the integrator and the stiffness; with $\mathbf{v}$ and $T_p$
integrated, the products $M\mathbf{v}$ and $ME$ would be conserved only to
the integrator's order once $M$ changes. Rebounds and periodic rotations act
on $\mathbf{P}$.

**Stiff velocity and temperature.** The velocity relaxes as a solid
particle's, explicitly for $\Delta t / \tau \le 1$ and exponentially above.
The temperature relaxes toward its wet-bulb value on the heat-up time, which
for small droplets is far below the gas step, and its rate
$f(T_p) = (Q - \dot m L) / (M c_l)$ is nonlinear in $T_p$. The first stage of
a step linearizes it, by a backward difference of $10^{-6}\,T_p$:
$J = \partial f / \partial T_p$, and where $J < 0$ the relaxation time
$\tau_T = -1/J$ and equilibrium $T_{eq} = T_p + \tau_T f$. With
$\Delta t > \tau_T$ the temperature is integrated exponentially toward
$T_{eq}$, with $\tau_T$ frozen at the step's start and $T_{eq}$ of each
stage, exactly as the velocity: exact for $T_{eq}$ linear in time over the
step, with the error of a frozen $\tau_T$ otherwise. The mass is integrated
explicitly: $d^2$ falls about linearly in time, so $\dot m$ changes slowly
until the droplet is nearly gone.

**Vanishing droplets.** The evaporation rate is at most the step's starting
mass over $\Delta t$, so no droplet's mass turns negative; a droplet capped
this way stops heating and evaporates within the step at its temperature,
which keeps the temperature of a vanishing droplet bounded. Below
`minimum_diameter` (by default 1% of `diameter`) a droplet's rates are zero
and it moves with the gas; after the step and before relocation it is
removed (`Particles::remove_evaporated`), its remains deposited into the gas
with the same deposition as the sources, and counted as evaporated.

**Gas properties.** The film properties come from the mechanism's
thermodynamic and transport tables on the device (`ThermoTable`,
`TransportTable::properties`), with $n_{species}$ values of work memory per
slot. The gas composition at the droplet is its cell's average, so it is first
order in space where the velocity and pressure the droplet reads are of the
reconstruction's order: reconstructing the species to the droplet's position
would need their gradients at every stage.
