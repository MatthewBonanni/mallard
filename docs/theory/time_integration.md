# Time integration

Mallard advances the semi-discrete equations (1) of [chapter 5](finite_volume.md#semi-discrete-equations),
$\mathrm{d}\overline{\mathbf{U}}/\mathrm{d}t = \mathbf{L}(\overline{\mathbf{U}}, t)$,
with explicit Runge-Kutta methods and a global time step from per-cell
spectral radii. Chemistry is split from this step
([chapter 3](chemistry_radiation.md#coupling-with-the-flow)). Usage:
[Numerics](../guide/numerics.md); measured stability and accuracy limits:
[Numerical methods](../numerics/overview.md#time-step).

## Right-hand side

$\mathbf{L}$ is evaluated at every stage (`Solver::calc_rhs`,
`src/solver/solver_rhs.cpp`, `solver_mixture.cpp`) in this order:
boundary states and the cell primitive variables (for mixtures, $T$ by
Newton, eq. (4) of [chapter 2](thermodynamics_transport.md#temperature-from-the-internal-energy));
the face reconstruction ([chapter 5](finite_volume.md#reconstruction)); the
shock sensor of the hybrid flux; the convective face fluxes and, for
mixtures, the species fluxes ([chapter 6](fluxes.md)); the transport
coefficients, the eddy viscosity and the viscous face fluxes; the sum over
each cell's faces; the axisymmetric source, gravity, expression sources,
mass-flow forcing and radiation, integrated over the cell; the division by
the cell volume; and the sponge layers, as rates
([chapter 8](boundaries_sources.md#sources)). Characteristic boundaries,
the dynamic SGS constant and the TFLES and PaSR fields are updated once per
step, at its first stage.

## Runge-Kutta methods

`[numerics] time_integrator` (`src/numerics/time_integrator.cpp`):

- **SSPRK3** (default), the three-stage, third-order strong-stability-preserving
    scheme of [Shu & Osher 1988](../references.md#shu-osher-1988)
    ([Gottlieb & Shu 1998](../references.md#gottlieb-shu-1998)):

    $$
    \begin{equation}
    \begin{aligned}
    \mathbf{U}^{(1)} &= \mathbf{U}^n + \Delta t\,\mathbf{L}(\mathbf{U}^n, t_n), \\
    \mathbf{U}^{(2)} &= \tfrac34\,\mathbf{U}^n + \tfrac14\left(\mathbf{U}^{(1)} + \Delta t\,\mathbf{L}(\mathbf{U}^{(1)}, t_n + \Delta t)\right), \\
    \mathbf{U}^{n+1} &= \mathbf{U}^n + \tfrac23\left(\mathbf{U}^{(2)} + \Delta t\,\mathbf{L}(\mathbf{U}^{(2)}, t_n + \tfrac12\Delta t) - \mathbf{U}^n\right);
    \end{aligned}
    \label{eq:runge-kutta-methods-1}
    \end{equation}
    $$

    the last stage is the convex combination
    $\tfrac13\mathbf{U}^n + \tfrac23(\cdots)$, written as an interpolation so
    that its weights sum to one exactly in floating point.
- **RK4**, the classical four-stage, fourth-order method
    ([Kutta 1901](../references.md#kutta-1901)):

    $$
    \begin{equation}
    \mathbf{U}^{n+1} = \mathbf{U}^n + \frac{\Delta t}{6}\left(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4\right), \quad
    \begin{aligned}
    \mathbf{k}_1 &= \mathbf{L}(\mathbf{U}^n, t_n), &
    \mathbf{k}_2 &= \mathbf{L}(\mathbf{U}^n + \tfrac12\Delta t\,\mathbf{k}_1, t_n + \tfrac12\Delta t), \\
    \mathbf{k}_3 &= \mathbf{L}(\mathbf{U}^n + \tfrac12\Delta t\,\mathbf{k}_2, t_n + \tfrac12\Delta t), &
    \mathbf{k}_4 &= \mathbf{L}(\mathbf{U}^n + \Delta t\,\mathbf{k}_3, t_n + \Delta t).
    \end{aligned}
    \label{eq:runge-kutta-methods-2}
    \end{equation}
    $$

- **FE**, forward Euler: $\mathbf{U}^{n+1} = \mathbf{U}^n + \Delta t\,\mathbf{L}(\mathbf{U}^n, t_n)$.

Positivity is not enforced between stages: the face reconstructions fall
back to first order where a face state would have a non-positive density
or pressure ([chapter 5](finite_volume.md#reconstruction)).

## Time step

With `[run] cfl`, the step is $\Delta t = \mathrm{CFL}\,\min_i\Delta t_i$
over all cells and ranks (otherwise the fixed `dt`), further limited by the
sponge layers to $\Delta t \le 1/\max_i\sigma_i$
([chapter 8](boundaries_sources.md#sponge-layers)) and shortened to land
exactly on the end time and on output times. Each cell's step for
$\mathrm{CFL} = 1$ comes from its convective and viscous spectral radii
([Blazek 2015](../references.md#blazek-2015), eqs. 6.20-6.21 with $C = 4$;
`TimeStepFunctor`, `src/solver/solver.cpp`):

$$
\begin{equation}
\Delta t_i = \frac{2\,\lvert\Omega_i\rvert}{\displaystyle\sum_f\left(\lvert u_n\rvert + a\right)_f A_f + \frac{4\,\nu_\mathrm{eff}}{\lvert\Omega_i\rvert}\sum_f A_f^2 + \frac{\nu_\mathrm{eff}\,\lvert\Omega_i\rvert}{r_i^2}},
\label{eq:time-step-1}
\end{equation}
$$

where $(\lvert u_n\rvert + a)_f$ is the larger of the face's two cells
(with the face's unit normal), the last term is present in axisymmetric
viscous runs only ($r_i$ the radius of the revolved centroid, the decay of
$u_r$ by the hoop stress), and

$$
\begin{equation}
\nu_\mathrm{eff} =
\begin{cases}
\max\left(\tfrac43,\ \dfrac{\gamma}{Pr}\right)\dfrac{\mu}{\rho} & \text{single gas}, \\[8pt]
\max\left(\tfrac43(\mu + \mu_t),\ \gamma\left(\dfrac{\mu}{Pr} + \dfrac{\mu_t}{Pr_t}\right)\right)\dfrac{1}{\rho} & \text{single gas, LES}, \\[8pt]
\max\left(\dfrac43\dfrac{\mu}{\rho},\ \dfrac{\lambda}{\rho c_v},\ \max_k D_k\right) + \max\left(\dfrac43\nu_t,\ \dfrac{\lambda_t}{\rho c_v},\ \dfrac{\nu_t}{Sc_t}\right) & \text{mixtures},
\end{cases}
\label{eq:time-step-2}
\end{equation}
$$

with the SGS term only in LES and the molecular part multiplied by
$\mathcal{E}F$ in thickened flames ([chapter 4](les.md#thickened-flame-model)).
Half the sum over faces stands for the sum over coordinate directions of the
projected areas: on a uniform grid of spacing $h$, eq. $\eqref{eq:time-step-1}$ gives
$\Delta t = \mathrm{CFL}\,h/(\lvert u\rvert + \lvert v\rvert + 2a)$ in 2D
and $\mathrm{CFL}\,h/(\lvert u\rvert + \lvert v\rvert + \lvert w\rvert + 3a)$
in 3D. Mixtures use the frozen sound speed $a = \sqrt{\gamma p/\rho}$.

## Splitting within a step

A time step of a reacting run is (`Solver::take_step`):

1. Strang splitting: chemistry over $\Delta t/2$ (fused with the previous
    step's second half with `fuse_half_steps`); SIMPLER: the transport
    tendency, then chemistry over $\Delta t$ ([chapter 3](chemistry_radiation.md#coupling-with-the-flow));
2. with double flux, freezing each cell's $(\gamma, e_0)$
    ([chapter 2](thermodynamics_transport.md#double-flux));
3. one Runge-Kutta step over $\Delta t$ (SIMPLER: over the second half step,
    with the frozen transport tendency subtracted);
4. with double flux, resetting $\rho E$ to the true equation of state;
5. Strang splitting: chemistry over $\Delta t/2$.
