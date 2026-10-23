# Governing equations

Mallard solves the compressible Euler or Navier-Stokes equations in
conservative form, for a single calorically perfect gas or for a thermally
perfect mixture of $N_s$ species, in planar 2D, 3D or axisymmetric (r-z)
geometry. This chapter states the equations and the constitutive laws of a
single gas; the thermodynamics and transport of mixtures are in
[chapter 2](thermodynamics_transport.md), the chemical source terms in
[chapter 3](chemistry_radiation.md), and the filtered equations of
large-eddy simulation in [chapter 4](les.md). Usage:
[Physics](../guide/physics.md).

## Conservation laws

With $\mathbf{x}\in\mathbb{R}^d$ ($d = 2$ or 3), Mallard advances the
conservative variables

$$
\begin{equation}
\mathbf{U} = \left[\rho,\ \rho\mathbf{u},\ \rho E,\ \rho Y_1,\ \dots,\ \rho Y_{N_s}\right]^T
\label{eq:conservation-laws-1}
\end{equation}
$$

where the partial densities $\rho Y_k$ are present for mixtures only
(`[physics] gas = "mixture"`). All $N_s$ species are transported, besides
$\rho$; the discretization keeps $\sum_k \rho Y_k = \rho$
([chapter 7](fluxes.md)). The equations are

$$
\begin{equation}
\begin{aligned}
\frac{\partial \rho}{\partial t} + \nabla\cdot(\rho\mathbf{u}) &= 0, \\
\frac{\partial (\rho\mathbf{u})}{\partial t} + \nabla\cdot(\rho\mathbf{u}\otimes\mathbf{u} + p\,\mathbf{I}) &= \nabla\cdot\boldsymbol{\tau} + \mathbf{s}_u, \\
\frac{\partial (\rho E)}{\partial t} + \nabla\cdot\big((\rho E + p)\,\mathbf{u}\big) &= \nabla\cdot(\boldsymbol{\tau}\cdot\mathbf{u} - \mathbf{q}) + s_E, \\
\frac{\partial (\rho Y_k)}{\partial t} + \nabla\cdot(\rho Y_k\mathbf{u}) &= -\nabla\cdot\mathbf{j}_k + \dot\omega_k, \qquad k = 1,\dots,N_s,
\end{aligned}
\label{eq:conservation-laws-2}
\end{equation}
$$

or, in compact form,

$$
\begin{equation}
\frac{\partial \mathbf{U}}{\partial t} + \nabla\cdot\mathbf{F}(\mathbf{U}) = \nabla\cdot\mathbf{F}^v(\mathbf{U},\nabla\mathbf{U}) + \mathbf{S}.
\label{eq:conservation-laws-3}
\end{equation}
$$

$E = e + \tfrac12\lvert\mathbf{u}\rvert^2$ is the specific total energy. For
mixtures $e$ includes the enthalpies of formation of the species, so heat
release appears as a change of composition at constant $\rho E$ and the
energy equation has no chemical source. The source $\mathbf{S}$ collects the
momentum and energy sources $\mathbf{s}_u$, $s_E$ of gravity, mass-flow
forcing, user expressions, sponge layers and radiation, and the production
rates $\dot\omega_k$ ([chapter 9](boundaries_sources.md#sources),
[chapter 3](chemistry_radiation.md)).

With `[physics] type = "euler"` (the default) the viscous flux
$\mathbf{F}^v$ is dropped; `type = "navier_stokes"` keeps it.

## Single gas

### Equation of state

A single gas is calorically perfect:

$$
\begin{equation}
p = \rho R T, \qquad e = c_v T, \qquad c_v = \frac{R}{\gamma - 1}, \qquad c_p = \gamma c_v, \qquad a = \sqrt{\frac{\gamma p}{\rho}}.
\label{eq:equation-of-state}
\end{equation}
$$

The input gives $\gamma$ and a reference state $(p_\mathrm{ref}, T_\mathrm{ref}, \rho_\mathrm{ref})$,
from which $R = p_\mathrm{ref}/(\rho_\mathrm{ref} T_\mathrm{ref})$ (`src/physics/physics.cpp`).
Nothing is nondimensionalized internally: inputs in SI units give results in
SI units, and a nondimensional reference state (for example
$p_\mathrm{ref} = 1/\gamma$, $\rho_\mathrm{ref} = T_\mathrm{ref} = 1$) gives a
nondimensional solver. Pressure follows from the conservative variables as
$p = (\gamma - 1)\left(\rho E - \tfrac12\rho\lvert\mathbf{u}\rvert^2\right)$.

### Viscous stress and heat flux

The fluid is Newtonian with Stokes' hypothesis (no bulk viscosity), and heat
conduction follows Fourier's law with a constant Prandtl number:

$$
\begin{equation}
\boldsymbol{\tau} = \mu\left(\nabla\mathbf{u} + \nabla\mathbf{u}^T - \tfrac23(\nabla\cdot\mathbf{u})\,\mathbf{I}\right),
\qquad \mathbf{q} = -\lambda\nabla T, \qquad \lambda = \frac{\mu c_p}{Pr},
\label{eq:viscous-stress-and-heat-flux-1}
\end{equation}
$$

with $Pr$ = `Pr` (default 0.72). The dynamic viscosity is one of
(`viscosity_model`, `src/physics/physics.h`)

$$
\begin{equation}
\mu(T) =
\begin{cases}
\mu_\mathrm{ref} & \text{constant (default)}, \\[2pt]
\mu_\mathrm{ref}\left(\dfrac{T}{T_{\mu}}\right)^{3/2}\dfrac{T_{\mu} + S}{T + S} & \text{Sutherland}, \\[8pt]
\mu_\mathrm{ref}\left(\dfrac{T}{T_{\mu}}\right)^{n} & \text{power law},
\end{cases}
\label{eq:viscous-stress-and-heat-flux-2}
\end{equation}
$$

with $\mu_\mathrm{ref}$ = `mu`, $T_\mu$ = `T_mu_ref` (default 273.15 K for
Sutherland's law, required for the power law), $S$ = `sutherland_S`
(default 110.4 K, air; [Sutherland 1893](../references.md#sutherland-1893))
and $n$ = `viscosity_exponent`. The viscous flux of eq. $\eqref{eq:conservation-laws-3}$ is

$$
\begin{equation}
\mathbf{F}^v = \left[0,\ \boldsymbol{\tau},\ \boldsymbol{\tau}\cdot\mathbf{u} - \mathbf{q},\ -\mathbf{j}_1,\ \dots,\ -\mathbf{j}_{N_s}\right]^T.
\label{eq:viscous-stress-and-heat-flux-3}
\end{equation}
$$

For a single gas there are no species fluxes. For mixtures, $\mu$, $\lambda$
and the diffusive mass fluxes $\mathbf{j}_k$ come from the transport model of
[chapter 2](thermodynamics_transport.md#transport), and the heat flux also
carries the enthalpy of the diffusing species,
$\mathbf{q} = -\lambda\nabla T + \sum_k h_k\mathbf{j}_k$.

## Axisymmetric flows

With `[physics] axisymmetric = true` (2D builds) the mesh is the meridian
half-plane of a flow symmetric about the $x$ axis without swirl: $x$ is the
axial coordinate and $y = r \ge 0$ the radius, and $\mathbf{u} = (u_x, u_r)$.
Mallard integrates eq. $\eqref{eq:conservation-laws-3}$ over the solid of revolution of each cell, per
radian. With $\mathrm{d}A$ and $\mathrm{d}l$ the planar area and length
elements,

$$
\begin{equation}
\frac{\mathrm{d}}{\mathrm{d}t}\int_{\Omega} \mathbf{U}\, r\,\mathrm{d}A
+ \oint_{\partial\Omega} (\mathbf{F} - \mathbf{F}^v)\cdot\mathbf{n}\; r\,\mathrm{d}l
= \int_{\Omega} \mathbf{S}\, r\,\mathrm{d}A + \int_{\Omega} \mathbf{H}\,\mathrm{d}A,
\label{eq:axisymmetric-flows-1}
\end{equation}
$$

where the only geometric source acts on the radial momentum:

$$
\begin{equation}
\mathbf{H} = \left[0,\ 0,\ p - \tau_{\theta\theta},\ 0,\ 0,\dots,0\right]^T, \qquad
\tau_{\theta\theta} = \mu\left(\frac{2u_r}{r} - \tfrac23\nabla\cdot\mathbf{u}\right), \qquad
\nabla\cdot\mathbf{u} = \frac{\partial u_x}{\partial x} + \frac{\partial u_r}{\partial r} + \frac{u_r}{r}.
\label{eq:axisymmetric-flows-2}
\end{equation}
$$

The stress tensor in the face fluxes uses the same three-dimensional
divergence, including the hoop term $u_r/r$ (on the axis, its limit
$\partial u_r/\partial r$). In this area-weighted form mass, axial momentum,
energy and species have no source and are conserved exactly in the revolved
geometry, faces on the axis have zero revolved area, and a gas at rest at
uniform pressure is an exact discrete steady state, because
$\oint p\,n_r\,r\,\mathrm{d}l = \int_\Omega p\,\mathrm{d}A$ for uniform $p$.
The cell unknowns are r-weighted averages
$\overline{\mathbf{U}} = \int_\Omega \mathbf{U}\,r\,\mathrm{d}A \big/ \int_\Omega r\,\mathrm{d}A$;
the discretization is in [chapter 6](finite_volume.md#axisymmetric-cells) and
the design in [Axisymmetric flows](../design/axisymmetric.md). All integral
outputs (forces, integrals, conserved totals) are per radian.
