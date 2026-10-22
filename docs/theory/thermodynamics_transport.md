# Thermodynamics and transport of mixtures

With `[physics] gas = "mixture"`, the gas is a thermally perfect mixture of
the $N_s$ species of a Cantera YAML mechanism, with temperature-dependent
species properties, mixture-averaged (or constant-Lewis-number) molecular
transport and optional thermal diffusion. This chapter gives the equation of
state, the transport coefficients and diffusive fluxes, and the double-flux
model. Mallard reads the mechanism itself and does not link Cantera; its
thermodynamics and transport reproduce Cantera's
([Cantera](../references.md#cantera);
[Kee, Coltrin & Glarborg 2003](../references.md#kee-coltrin-glarborg-2003)).
All of it is evaluated in double precision in every build
(`src/chemistry/thermo.h`, `transport.h`). Usage:
[Thermochemistry](../guide/thermochemistry.md).

## Species thermodynamics

Every species $k$ has piecewise polynomials over temperature ranges, stored
in the NASA-9 form (`src/chemistry/mechanism.h`):

$$
\begin{equation}
\begin{aligned}
\frac{c_{p,k}}{R_u/W_k} &= a_0 T^{-2} + a_1 T^{-1} + a_2 + a_3 T + a_4 T^2 + a_5 T^3 + a_6 T^4, \\
\frac{h_k}{R_u T/W_k} &= -a_0 T^{-2} + a_1 \frac{\ln T}{T} + a_2 + \frac{a_3}{2} T + \frac{a_4}{3} T^2 + \frac{a_5}{4} T^3 + \frac{a_6}{5} T^4 + \frac{a_7}{T}, \\
\frac{s_k^\circ}{R_u/W_k} &= -\frac{a_0}{2} T^{-2} - \frac{a_1}{T} + a_2\ln T + a_3 T + \frac{a_4}{2} T^2 + \frac{a_5}{3} T^3 + \frac{a_6}{4} T^4 + a_8,
\end{aligned}
\label{eq:species-thermodynamics}
\end{equation}
$$

with one coefficient set per range. NASA-7 data $(b_0,\dots,b_6)$ are stored
as $(0, 0, b_0, \dots, b_6)$, and a constant-$c_p$ species
$(T_0, h_0, s_0, c_{p0})$ as $a_2 = c_{p0}/R_u$,
$a_7 = (h_0 - c_{p0}T_0)/R_u$, $a_8 = (s_0 - c_{p0}\ln T_0)/R_u$ (molar
quantities). As in Cantera, a NASA-7 range boundary belongs to the range
below it and a NASA-9 boundary to the range above, and the outermost
polynomials are extrapolated beyond the data. $h_k$ includes the enthalpy of
formation. $R_u = N_A k_B$ with the exact SI values of $N_A$ and $k_B$.

## Mixture properties and equation of state

With $Y_k = \rho Y_k/\rho$ from the transported partial densities,

$$
\begin{equation}
\frac{1}{\overline{W}} = \sum_k \frac{Y_k}{W_k}, \qquad
R = \frac{R_u}{\overline{W}}, \qquad
X_k = \frac{\overline{W}}{W_k} Y_k, \qquad
c_p = \sum_k Y_k c_{p,k}, \qquad
h = \sum_k Y_k h_k,
\label{eq:mixture-properties-and-equation-of-state-1}
\end{equation}
$$

$$
\begin{equation}
p = \rho R T, \qquad
e = \sum_k Y_k\left(h_k(T) - \frac{R_u T}{W_k}\right), \qquad
c_v = c_p - R.
\label{eq:mixture-properties-and-equation-of-state-2}
\end{equation}
$$

### Temperature from the internal energy

The conservative variables give $e = \rho E/\rho - \tfrac12\lvert\mathbf{u}\rvert^2$,
and $T$ solves $e(T; Y) = e$ by Newton's method from a per-cell seed (the
previous temperature of the cell), safeguarded by bisection:

$$
\begin{equation}
T^{m+1} = T^m - \frac{e(T^m) - e}{c_v(T^m)},
\label{eq:temperature-from-the-internal-energy}
\end{equation}
$$

where a Newton iterate outside the current bracket $[T_a, T_b]$, which
starts as $[\max(T_\mathrm{min}/2, 1\,\mathrm{K}),\ 2T_\mathrm{max}]$ with
$[T_\mathrm{min}, T_\mathrm{max}]$ the range covered by every species' data,
is replaced by the bracket's midpoint. Iterations stop when
$\lvert T^{m+1} - T^m\rvert \le 10^{-10} T^m$ (at most 200). Then
$p = \rho R T$.

### Frozen surrogates

The numerical fluxes see a cell's thermodynamics through two surrogates, the
frozen ratio of specific heats and an energy offset
(`Mixture::cell_state`, `src/physics/mixture.h`):

$$
\begin{equation}
\gamma = \frac{c_p(T)}{c_v(T)}, \qquad e_0 = e - c_v(T)\,T, \qquad
\rho E = \frac{p}{\gamma - 1} + \rho e_0 + \tfrac12\rho\lvert\mathbf{u}\rvert^2,
\label{eq:frozen-surrogates}
\end{equation}
$$

so that eq. $\eqref{eq:frozen-surrogates}$ holds exactly at the cell state; $e_0 = 0$ for a
calorically perfect gas. The sound speed of the fluxes is the frozen one,
$a = \sqrt{\gamma p/\rho}$. Reconstructed face states carry their own
$(\gamma, e_0)$ ([chapter 6](finite_volume.md#gas-mixtures)), and the Riemann
solvers use each side's pair ([chapter 7](fluxes.md#gas-mixtures)).

## Transport

With `[physics] type = "navier_stokes"`, the mixture is viscous, conducts
heat and diffuses species. The transport data of the mechanism
(Lennard-Jones parameters, dipole moment, polarizability, rotational
relaxation number) give pure-species properties as Cantera's mixture-averaged
model computes them: viscosities and binary diffusion coefficients from the
collision integrals of [Monchick & Mason 1961](../references.md#monchick-mason-1961)
with Cantera's polar corrections, and conductivities with Parker's rotational
relaxation. As in Cantera, they are fitted once, at start-up, by weighted
least squares (relative error) at 50 temperatures evenly spaced over
$[T_\mathrm{min}, T_\mathrm{max}]$ (`src/chemistry/transport.cpp`), as
polynomials of degree 4 in $\ln T$:

$$
\begin{equation}
\mu_k = \sqrt{T}\left(\sum_{m=0}^{4} v_{k,m}(\ln T)^m\right)^{2}, \qquad
\lambda_k = \sqrt{T}\sum_{m=0}^{4} c_{k,m}(\ln T)^m, \qquad
\mathcal{D}_{kj} = \frac{T^{3/2}}{p}\sum_{m=0}^{4} d_{kj,m}(\ln T)^m.
\label{eq:transport}
\end{equation}
$$

In the mixing rules, negative mass fractions (round-off, or overshoots of
explicit diffusion) count as zero, and mole fractions below $10^{-20}$ as
$10^{-20}$, as in Cantera.

### Viscosity and conductivity

The viscosity follows Wilke's rule
([Wilke 1950](../references.md#wilke-1950)) and the conductivity the
combination rule of
[Mathur, Tondon & Saxena 1967](../references.md#mathur-tondon-saxena-1967):

$$
\begin{equation}
\mu = \sum_k \frac{X_k\,\mu_k}{\sum_j X_j\,\Phi_{kj}}, \qquad
\Phi_{kj} = \frac{\left(1 + \sqrt{\mu_k/\mu_j}\,(W_j/W_k)^{1/4}\right)^2}{\sqrt{8\,(1 + W_k/W_j)}},
\label{eq:viscosity-and-conductivity-1}
\end{equation}
$$

$$
\begin{equation}
\lambda = \frac12\left(\sum_k X_k\lambda_k + \frac{1}{\sum_k X_k/\lambda_k}\right).
\label{eq:viscosity-and-conductivity-2}
\end{equation}
$$

### Diffusion coefficients

The species diffusion coefficient $D_k$ depends on `[physics] transport`:

$$
\begin{equation}
D_k =
\begin{cases}
\dfrac{\sum_{j\ne k} Y_j}{\sum_{j\ne k} X_j/\mathcal{D}_{kj}} & \texttt{mixture\_averaged} \text{ (default)}, \\[10pt]
\dfrac{\lambda}{\rho c_p} & \texttt{unity\_lewis}, \\[10pt]
\dfrac{\lambda}{\rho c_p\,Le_k} & \texttt{constant\_lewis},
\end{cases}
\label{eq:diffusion-coefficients}
\end{equation}
$$

with $Le_k$ from `lewis` (default 1). The mixture-averaged numerator is
$1 - Y_k$ summed from the other species, which stays accurate where species
$k$ is nearly pure. For a single-species mechanism, $D_1 = \mathcal{D}_{11}$.

### Diffusive fluxes

Species diffuse along mole-fraction gradients with a correction velocity
([Kee, Coltrin & Glarborg 2003](../references.md#kee-coltrin-glarborg-2003)),
so that the diffusive mass fluxes sum to zero and the continuity equation
needs no diffusion term:

$$
\begin{equation}
\mathbf{j}_k = -\rho D_k\frac{W_k}{\overline{W}}\nabla X_k
+ \frac{Y_k}{\sum_j Y_j}\sum_j \rho D_j\frac{W_j}{\overline{W}}\nabla X_j
- D_k^T\,\frac{\nabla T}{T},
\label{eq:diffusive-fluxes-1}
\end{equation}
$$

where the second term is $\rho Y_k\mathbf{V}^c$ (the normalization by
$\sum_j Y_j$ removes the drift of $\sum_k Y_k$ from 1) and the last term is
present with thermal diffusion only. The energy equation carries the
enthalpy they transport (eq. (7) of [chapter 1](governing_equations.md#viscous-stress-and-heat-flux)):

$$
\begin{equation}
\mathbf{q} = -\lambda\nabla T + \sum_k h_k\,\mathbf{j}_k.
\label{eq:diffusive-fluxes-2}
\end{equation}
$$

There is no Dufour effect, as in Cantera. The face discretization of
eqs. $\eqref{eq:diffusive-fluxes-1}$-$\eqref{eq:diffusive-fluxes-2}$ is in [chapter 7](fluxes.md#mixture-diffusive-fluxes).

### Thermal diffusion (Soret effect)

With `soret = true` (mixture-averaged transport only), the thermal diffusion
coefficients are Cantera's mixture-averaged ones, the model of
[Chapman & Cowling 1970](../references.md#chapman-cowling-1970) as given by
[Zirwes & Kronenburg 2025](../references.md#zirwes-kronenburg-2025):

$$
\begin{equation}
a_k = \frac{\tfrac{15}{4}\,\mu_k/W_k}{1 + 1.065\sum_{j\ne k}\Phi_{kj}X_j/X_k}
\quad (a_k = 0 \text{ where } Y_k < 10^{-20}),
\label{eq:thermal-diffusion-soret-effect-1}
\end{equation}
$$

$$
\begin{equation}
D_k^{T\prime} = W_k\,\overline{W}\,D_k\sum_{j\ne k}\frac{1.2\,C^{\ast}_{kj} - 1}{\mathcal{D}_{kj}\,(W_k + W_j)}\left(Y_k a_j - Y_j a_k\right), \qquad
D_k^T = D_k^{T\prime} - Y_k\sum_j D_j^{T\prime},
\label{eq:thermal-diffusion-soret-effect-2}
\end{equation}
$$

with $D_k$ the mixture-averaged coefficient of eq. $\eqref{eq:diffusion-coefficients}$, $\Phi_{kj}$ Wilke's
operator of eq. $\eqref{eq:viscosity-and-conductivity-1}$, and $C^{\ast}_{kj}$ the reduced collision integral of
Monchick & Mason at $T^{\ast} = k_B T/\epsilon_{kj}$, fitted as in Cantera by a
polynomial of degree 8 in $\ln T^{\ast}$ over the table rows that bracket the
mechanism's range of $T^{\ast}$ (fitted in the reduced dipole moment for polar
pairs). The $D_k^T$ sum to zero, so the correction velocity of eq. $\eqref{eq:diffusive-fluxes-1}$
remains that of the Fickian part.

### Time-step diffusivity

Each cell's effective diffusivity for the viscous time-step limit
([chapter 8](time_integration.md#time-step)) is

$$
\begin{equation}
\nu_\mathrm{eff} = \max\left(\frac{4}{3}\frac{\mu}{\rho},\ \frac{\lambda}{\rho c_v},\ \max_k D_k\right);
\label{eq:time-step-diffusivity}
\end{equation}
$$

thermal diffusion couples the species to $T$ one way and adds no diffusive
eigenvalue.

## Double flux

A conservative scheme for a mixture whose $\gamma$ varies across a contact
(for example cold hydrogen against hot air) perturbs pressure and velocity
there: the conservative update of $\rho E$ mixes cells with different
$(\gamma, e_0)$, and eq. $\eqref{eq:frozen-surrogates}$ then returns a non-uniform pressure. The
double-flux model ([Abgrall & Karni 2001](../references.md#abgrall-karni-2001);
[Billet & Abgrall 2003](../references.md#billet-abgrall-2003);
[Ma, Lv & Ihme 2017](../references.md#ma-lv-ihme-2017)),
`[numerics] double_flux = true`, removes these oscillations at the cost of
exact energy conservation (`src/solver/solver_mixture.cpp`,
`src/numerics/mixture_flux.h`):

1. At the start of each time step, every cell freezes its
    $(\gamma_i^{\ast}, e_{0,i}^{\ast})$ from the true equation of state, eq. $\eqref{eq:frozen-surrogates}$.
2. Within all the stages of the step, cell pressures come from the frozen
    pair, $p_i = (\gamma_i^{\ast} - 1)\left(\rho E_i - \rho_i e_{0,i}^{\ast} - \tfrac12\rho_i\lvert\mathbf{u}_i\rvert^2\right)$,
    and each face computes two numerical fluxes from the same reconstructed
    $\mathbf{W}^L$, $\mathbf{W}^R$: $\hat{\mathbf{F}}^{(L)}$ converts both
    sides to conservative variables with the left cell's frozen pair, and
    $\hat{\mathbf{F}}^{(R)}$ with the right cell's. Mass and momentum use their mean, which keeps them
    conservative; the energy of each cell uses the flux computed with its own
    pair:

    $$
    \begin{equation}
    \hat F_{\rho} = \tfrac12\left(\hat F^{(L)}_{\rho} + \hat F^{(R)}_{\rho}\right), \qquad
    \hat{\mathbf{F}}_{\rho\mathbf{u}} = \tfrac12\left(\hat{\mathbf{F}}^{(L)}_{\rho\mathbf{u}} + \hat{\mathbf{F}}^{(R)}_{\rho\mathbf{u}}\right), \qquad
    \hat F_{\rho E}\big\vert_{\text{cell } L} = \hat F^{(L)}_{\rho E}, \quad
    \hat F_{\rho E}\big\vert_{\text{cell } R} = \hat F^{(R)}_{\rho E}.
    \label{eq:double-flux}
    \end{equation}
    $$

    Both fluxes estimate their wave speeds with each side's own frozen pair
    (`riemann::SideThermo`), so their mass fluxes agree at a contact for every
    Riemann solver. With the hybrid flux, each of the two is blended with the
    central flux computed with the same pair ([chapter 7](fluxes.md#hybrid-central-upwind-flux)). The
    species fluxes use the mean mass flux ([chapter 7](fluxes.md#species-fluxes)).
3. After the step, $\rho E$ of each cell is reset to the true equation of
    state at the pressure the frozen pair gives, keeping $\rho$, $\rho\mathbf{u}$
    and the composition: $T = p/(\rho R)$ and
    $\rho E = \rho\,e(T; Y) + \tfrac12\rho\lvert\mathbf{u}\rvert^2$.

A contact between gases at uniform pressure and velocity then stays exactly
uniform in $p$ and $\mathbf{u}$. Double flux cannot be combined with SIMPLER
splitting, whose reaction substep within the step would leave the frozen
thermodynamics stale ([chapter 3](chemistry_radiation.md#simpler-balanced-splitting)).
