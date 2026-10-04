# Design: characteristic boundary conditions and sponge layers

Status: implemented (`nscbc_outlet`, `nscbc_inlet`, `[[sponges]]`).

Flame, jet and wake simulations need open boundaries that let acoustic waves,
vortices and flames leave without reflecting. They must also hold a mean
pressure or inflow state. `p_out` and `upt` impose their state exactly and
reflect almost all of an acoustic wave. `extrapolation` reflects nothing, but
the mean state drifts. The Navier-Stokes characteristic boundary conditions
(NSCBC) of [Poinsot & Lele (1992)](../references.md#poinsot-lele-1992) sit
between the two. Outgoing waves leave unchanged, and the incoming wave is a
weak relaxation toward the target.

## From LODI to a finite-volume ghost state

In NSCBC, the normal derivatives at a boundary point are split into the
amplitudes `L_k = lambda_k dw_k/dn` of the characteristic waves. Here
`w_k` is the characteristic variable and `lambda_k` its speed along the
outward normal `n`. Outgoing amplitudes (`lambda_k > 0`) come from the
interior. Incoming ones are prescribed. Mallard has no boundary points: a
boundary face's flux comes from a Riemann solver, with the reconstructed
interior state `W_l` on one side and a ghost state `W_g` on the other.

The ghost is the face's own interior state, with the incoming characteristic
variables shifted:

```
w_k(W_g) = w_k(W_l) - (h / |lambda_k|) L_k      (incoming k)
w_k(W_g) = w_k(W_l)                             (outgoing k)
```

`h = V / A` is the boundary cell's volume over the face area. For an upwind
Riemann solver, an incoming jump `dw` on the face changes the cell's `w_k`
at the rate `|lambda_k| dw A / V`. So the boundary cell sees
`dw_k/dt = -L_k` from its boundary face. This is the LODI relation of
Poinsot & Lele, applied to the cell that touches the boundary, and the
relaxation rate does not depend on the mesh. HLL, HLLC and Roe upwind a
single-wave jump exactly. Rusanov adds dissipation, which raises the effective
relaxation rate by at most a factor of two. With `L_k = 0` the ghost equals
`W_l`, the Riemann problem has no jump, and the face is perfectly
non-reflecting for waves normal to it.

Acoustic variables are linearized about the face state:

- outgoing `w+ = p + Z u_n`, incoming `w- = p - Z u_n`, with `Z = rho c`
- incoming speed `lambda = u_n - c`
- `c^2 = gamma p / rho`, with the face's own `gamma` for mixtures (its
  frozen surrogate)

A jump `dw` in `w-` gives `p_g = p_l + dw/2` and `u_n,g = u_n,l - dw/(2Z)`.
The jump in density is isentropic, `rho_g = rho_l + (p_g - p_l)/c^2`, so the
ghost carries no entropy jump.

The ghost replaces only the exterior state seen by the convective flux.
Everything else uses a zero-gradient copy of the cell, like `extrapolation`
without image faces: the mirror ghost cells of TENO-E stencils, least-squares
gradients, scalar reconstruction, and the viscous fluxes (zero normal
derivatives, the outflow viscous conditions of Poinsot & Lele). This is
stateless. Nothing is stored between stages, it works the same at every
Runge-Kutta stage, restarts stay bit-for-bit, and it needs no communication.

## Outlet (`nscbc_outlet`)

For subsonic faces, the incoming acoustic amplitude is

```
L- = K (p_l - p_t) - (1 - beta) T-,   K = sigma c (1 - M^2) / L
```

- `p_t`: target pressure
- `L`: a length of the domain (input)
- `sigma`: relaxation coefficient, default 0.25 ([Rudy & Strikwerda
  1980](../references.md#rudy-strikwerda-1980); Poinsot & Lele)
- `M = |u| / c` at the face

The relaxation part of `h |L-| / |lambda|` is capped at 2. At the cap, the
face pressure equals `p_t` (a fully reflecting pressure outlet), which is
the limit `sigma -> infinity`.

For a plane wave at normal incidence, `dw-/dt = -K p'` gives the reflection
coefficient `R = -1 / (1 - 2 i omega / K)`
([Selle, Nicoud & Poinsot 2004](../references.md#selle-nicoud-poinsot-2004)).
Low frequencies reflect, so the mean pressure is held. High frequencies
leave.

`T-` holds the transverse terms of the incoming wave's equation. These are
the tangential convection and the divergence of the tangential velocity,
which normal derivatives do not see:

```
T- = u_t . grad p + rho c^2 div_t u_t - rho c (u_t . grad u_n)
```

They come from a least-squares gradient of the boundary cell. With
`beta = 0` the incoming wave cancels them and `w-` stays frozen against
transverse forcing. Then a vortex leaving the domain distorts nothing, but
the mean pressure drifts. [Yoo et al. (2005)](../references.md#yoo-2005),
[Yoo & Im (2007)](../references.md#yoo-im-2007) and
[Lodato, Domingo & Vervisch (2008)](../references.md#lodato-2008) relax them
with `beta` equal to the Mach number. [Granet et al.
(2010)](../references.md#granet-2010) show this matters most at low Mach
number, where the plain Poinsot-Lele outlet pushes the low pressure of a
vortex core back toward `p_t` and sends a large wave upstream. The default is
the local `beta = M`. `beta = 1` drops the terms.

Other cases:

- **Supersonic outflow faces** (`u_n >= c`): every wave is outgoing, so the
  ghost is `W_l`.
- **Backflow** (`u_n < 0`): the acoustic treatment is unchanged. Entropy,
  tangential velocity and composition still come from the interior.

## Inlet (`nscbc_inlet`)

The target is `u_t`, `T_t` and, for mixtures, the composition. At a subsonic
inflow face, four kinds of incoming waves are relaxed toward it.

**Acoustic wave**, toward the target normal velocity:

```
L- = -K_u Z (u_n,l - u_n,t) - (1 - beta) T-,   K_u = sigma c (1 - M^2) / L
```

This is Poinsot & Lele's `L_5` in Mallard's outward-normal convention, with
the same cap.

**Entropy and tangential velocity.** These waves are convective (`lambda = u_n`),
so the relaxation `dphi/dt = -K_phi (phi - phi_t)` becomes a blend:

```
phi_g = phi_l + min(1, h K_phi / |u_n|) (phi_t - phi_l),   K_phi = sigma_phi c / L
```

for `phi` = `T` or `u_t`. Without `sigma_T` or `sigma_t`, `T` and `u_t` are
imposed exactly. This does not reflect acoustic waves: in linear theory, the
reflected acoustic wave depends only on the incoming acoustic condition. The
ghost density is `p_g / (R_t T_g)`, with `R_t` the gas constant of the target
composition.

**Composition** is always imposed exactly, as for `upt`. It is a convective
wave and does not reflect acoustics. A relaxed composition would need the
ghost thermodynamics `[gamma, e0]` of a blended composition at every
evaluation. The flow block uses the target's `[gamma, e0]` on inflow faces.

**Supersonic inflow** (`u_n <= -c`) imposes the whole target state, using the
reference pressure `p`.

## Multicomponent and reacting flows

- **Species** are convective waves. They leave through outlets from the
  interior, and enter through inlets at the target composition (mass-flux
  upwinding). Acoustics uses each face's frozen `gamma`, so the
  characteristic split is the one of the double-flux and standard mixture
  fluxes.
- **Reaction terms.** Yoo & Im add the reaction source terms to the LODI
  system, so that heat release next to a boundary is not mistaken for an
  incoming wave. Mallard Strang-splits chemistry from the flow. A reaction
  half-step changes the boundary cell at constant volume, and the next flow
  step sees the resulting pressure rise as part of the state. Its acoustic
  part leaves through the outgoing characteristic, and only the slow `K`
  relaxation reacts to it. So no separate reaction term is needed, and the
  flame-exit validation below checks this.

## Sponge layers (`[[sponges]]`)

A sponge adds `S = -sigma_s(x) (U - U_ref(x))` to the conservative
variables, and the matching term to each species partial density
([Bodony 2006](../references.md#bodony-2006); [Mani 2012](../references.md#mani-2012)).

- `strength` (1/s) is an expression in `x`, `y`, `z`, zero outside the
  layer. A ramp such as `A ((x - x0)/d)^2` keeps its own reflections small.
- The reference state is `u`, `p`, `T` (plus `X` or `Y`), each a number or
  an expression.
- Strength and reference are evaluated once at cell centroids.
- Several sponges add up.
- The source is explicit, so the time step is capped at `1 / max sigma_s`
  when a sponge is present.

A sponge works with any boundary condition. The usual setup is a sponge in
front of an `nscbc_outlet` or `farfield`.

## Validation

- **Acoustic pulse at normal incidence** (1D): the reflected wave against
  the theoretical `R(omega)` response, for several `sigma`, and against
  `p_out`.
- **Vortex leaving through an outlet** (2D, M = 0.1 to 0.3): the pressure
  error against a domain three times longer, for `p_out`, the Poinsot-Lele
  outlet without transverse terms, and with them.
- **Poiseuille channel outflow** (viscous): the parabolic profile and the
  linear pressure drop hold through the outlet.
- **A composition and temperature front leaving through the outlet**
  (mixture path, cheap), and a 1D premixed H2/air flame blown out through
  the outlet (example). The pressure perturbations stay small and no wave
  comes back.
- **3D**: an acoustic pulse in a box. **MPI**: distributed runs are bitwise
  equal to serial ones.
- **Unchanged behavior**: results with the existing conditions are bitwise
  unchanged. The new code only runs on faces of the new types and in runs
  with sponges.
