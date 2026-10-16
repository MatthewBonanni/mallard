# Theory manual

The equations Mallard solves and the methods it solves them with, as
implemented on `main`: every model and scheme of the solver, with the
formulas, constants and defaults of the source code it cites. The
[user guide](../guide/first_simulation.md) explains how to use them;
[Numerical methods](../numerics/overview.md) collects measured stability
limits, accuracy studies and known limitations; the sources are in
[References](../references.md).

## Chapters

1. [Governing equations](governing_equations.md): the compressible
    Navier-Stokes equations, the single-gas constitutive laws and the
    axisymmetric form.
2. [Thermodynamics and transport of mixtures](thermodynamics_transport.md):
    thermally perfect mixtures, mixture-averaged transport, thermal
    diffusion and double flux.
3. [Chemistry and radiation](chemistry_radiation.md): finite-rate kinetics,
    the RODAS integrator, Strang and SIMPLER splitting, optically thin
    radiation.
4. [Large-eddy simulation](les.md): the filtered equations, SGS models,
    the dynamic procedure, thickened flames, the partially stirred reactor
    and synthetic inflow turbulence.
5. [Finite-volume discretization and reconstruction](finite_volume.md):
    face quadrature, curved boundaries, MUSCL, TENO-E and the gradients of
    the viscous terms.
6. [Numerical fluxes](fluxes.md): the Riemann solvers, the hybrid
    central-upwind flux, species fluxes and viscous fluxes.
7. [Time integration](time_integration.md): Runge-Kutta methods, the time
    step and the order of operations within a step.
8. [Boundary conditions and sources](boundaries_sources.md): exterior
    states, characteristic boundaries, body forces and sponges.

## Notation

- $\rho$, $\mathbf{u}$, $p$, $T$: density, velocity, pressure, temperature
- $e$, $E$, $h$, $H$: specific internal energy, total energy $e + \tfrac12\lvert\mathbf{u}\rvert^2$, enthalpy $e + p/\rho$, total enthalpy $E + p/\rho$
- $a$, $\gamma$, $M$: sound speed, ratio of specific heats, Mach number
- $R$, $R_u$: specific and universal gas constants
- $Y_k$, $X_k$, $C_k$, $W_k$, $\overline{W}$: mass and mole fractions, concentration $\rho Y_k/W_k$, molar mass of species $k$ and of the mixture
- $\dot\omega_k$, $\dot\omega_T$: mass production rate of species $k$, heat release rate
- $\mu$, $\lambda$, $D_k$, $D_k^T$: viscosity, thermal conductivity, mixture-averaged and thermal diffusion coefficients
- $\boldsymbol{\tau}$, $\mathbf{q}$, $\mathbf{j}_k$: viscous stress, heat flux, diffusive mass flux of species $k$
- $\mathbf{U}$, $\mathbf{W}$: conservative variables, primitive variables $[\rho, \mathbf{u}, p]$
- $\mathbf{F}$, $\mathbf{F}^v$, $\mathbf{S}$, $\hat{\mathbf{F}}$: convective and viscous fluxes, sources, numerical flux
- $\Omega_i$, $\lvert\Omega_i\rvert$, $\mathbf{x}_i$, $\overline{\mathbf{U}}_i$: cell $i$, its volume, centroid and cell average
- $f$, $A_f$, $\mathbf{n}_f$: face, its area, its unit normal from its first (left, L) cell to its second (right, R)
- $\mathbf{x}_{f,q}$, $w_{f,q}$: face quadrature points and weights
- $\Delta t$, $\mathrm{CFL}$: time step, Courant number
- $\Delta$, $\mu_t$, $\nu_t$, $Pr_t$, $Sc_t$: LES filter width, eddy viscosity (dynamic and kinematic), turbulent Prandtl and Schmidt numbers
- $F$, $\mathcal{E}$, $\Omega$: thickening factor, efficiency and flame sensor of the thickened flame model

Vectors and tensors are bold; $\delta_{ij}$ is the Kronecker delta and
$S_{ij} = \tfrac12(\partial_j u_i + \partial_i u_j)$ the strain rate.
Units are SI. Chapter 4 alone distinguishes filtered quantities
($\overline{\phi}$, and Favre-filtered $\widetilde{\phi}$); elsewhere
filter symbols are dropped. Equations are numbered within each chapter.
