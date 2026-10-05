# Design: characteristic boundary conditions and sponge layers

Status: implemented (`nscbc_outlet`, `nscbc_inlet`, `[[sponges]]`).

Flame, jet and wake simulations need open boundaries that let acoustic waves,
vortices and flames leave without reflecting. They must also hold a mean
pressure or inflow state. `p_out` and `upt` impose their state exactly and
reflect almost all of an acoustic wave. `extrapolation` holds nothing, and
(see below) still reflects. The Navier-Stokes characteristic boundary
conditions (NSCBC) of
[Poinsot & Lele (1992)](../references.md#poinsot-lele-1992) let outgoing
waves leave unchanged and make the incoming wave a weak relaxation toward
the target.

## The incoming wave of a face

In NSCBC, the normal derivatives at a boundary point are split into the
amplitudes `L_k = lambda_k dw_k/dn` of the characteristic waves. Here `w_k`
is the characteristic variable and `lambda_k` its speed along the outward
normal `n`. Outgoing amplitudes come from the interior; incoming ones are
prescribed, and the boundary point advances in time by these LODI
relations.

Mallard has no boundary points: a boundary face's flux comes from a Riemann
solver, with the reconstructed interior state `W_l` on one side and an
exterior state `W_g` on the other. The outgoing acoustic wave is that of
`W_l`:

- outgoing `w+ = p + Z u_n`, incoming `w- = p - Z u_n`, with `Z = rho c`
- `c^2 = gamma p / rho`, with the face's own `gamma` for mixtures (its
  frozen surrogate)

The incoming acoustic wave is the face's own boundary point. Each
characteristic face keeps a pressure `p_b` and normal velocity `u_b`. Once
per step, at the first Runge-Kutta stage, its `w- = p_b - Z u_b` advances by
a forward-Euler step of the LODI relation:

```
dw-/dt = -K (p_b - p_t) - beta T-            (outlets)
dw-/dt = K Z (u_b - u_n,t) - beta T-         (inlets, Poinsot & Lele's L_5)
K = sigma c (1 - M^2) / L
```

Here `p_b` and `u_b` are first refreshed from the current outgoing wave.
`K dt` is capped at 1. The face's `w-` goes into the exterior state at every
stage of the step.

The face keeps `p_b` and `u_b` rather than `w-` itself: pressure and
velocity are continuous at contacts, `w-` is not. A hot front leaving
through an outlet changes `Z` by a factor of two, and a stored `w-` would
then put an O(`Z u`) pressure jump on the face.

`p_b` and `u_b` start from the interior face state at the first step, and
again after a restart. That is the only state, and restarts are therefore
not bit-for-bit for runs with characteristic boundaries.

This is the LODI relation of Poinsot & Lele itself. For a plane wave at
normal incidence, `dw-/dt = -K p'` gives the reflection coefficient
`R = -1 / (1 - 2 i omega / K)`
([Selle, Nicoud & Poinsot 2004](../references.md#selle-nicoud-poinsot-2004)).
Low frequencies reflect, so the mean pressure is held; high frequencies
leave. The pulse test checks this.

### Stateless formulations do not work

The first implementation was stateless. It shifted the incoming wave of a
transmissive state by `-(V/A) L / |lambda|`, so that the boundary cell itself
obeyed `dw-/dt = -L`. Two problems:

- **Steady offsets.** That cell's `w-` is frozen except for `L`. Any steady
  source in the cell is balanced by `K (p - p_t)`, and the outlet settles
  away from its target by `source / K`:
  - viscous stress in a Poiseuille flow: 20% of the flow rate
  - heat release of a flame at the outlet: the flame stayed anchored there,
    or the domain kept pressurizing
- **Instability.** Taking the incoming wave from the face's own state
  instead differences the incoming wave downwind, and blows up.

With its own boundary point, the face lets the boundary cell relax toward
`w-` at the rate `|lambda| / h`. Sources in the cell are carried inward as
waves, as in the interior, and the face's pressure settles at `p_t`.

### Exterior state

Exterior state on a subsonic face:

| Wave | Source in the exterior state |
|---|---|
| Outgoing acoustic | `W_l` |
| Incoming acoustic | the face's `p_b`, `u_b`, combined with the local `Z` |
| Entropy, tangential velocity and composition at outflow | `W_l`, with an isentropic density change |
| Incoming convective waves (backflow at outlets) | the transmissive state `W_e` of `extrapolation` (the reconstructed state on the interior image face), not a copy of the boundary cell |

Special cases:

- **Supersonic outflow** (`u_n >= c`): the exterior state is `W_l`, and the
  face follows the interior.
- **Supersonic inflow at inlets** (`u_n <= -c`): the exterior state is the
  target.

**Low-Mach correction.** Characteristic faces take the low-Mach correction
(Thornber et al.) like interior faces. Without it, the boundary cell's two
sides have different dissipation. In a steady shear flow, this held the
outlet pressure away from its target.

## Reconstruction next to the boundary

A wave that leaves through an open boundary must leave the boundary cell's
reconstruction intact. Zero-gradient ghosts break this: the mirror cells of
TENO-E stencils and the ghost of the MUSCL least-squares fit force the
normal derivative toward zero, and the reconstructed states jump by O(h) at
the boundary cell's inner face. With exact upwinding, such a jump in an
outgoing wave does nothing. But the default low-Mach correction scales the
velocity jump by `z = max(M, 0.1)`, which mixes it into the incoming wave.

A normal acoustic pulse leaving through `extrapolation` (400 cells over the
domain, the pulse 20 cells wide, default low-Mach correction) reflects by:

| Reconstruction | Quadrilaterals | Triangles | Jittered mixed |
|---|---|---|---|
| MUSCL | 15% | 5% | 0.6% |
| TENO5 | 25% | 4% | 14% |

So characteristic faces take no ghosts in the boundary cell's
reconstruction:

- **MUSCL**: boundary cells use the vertex-neighbor least-squares fit
  without the ghost of the characteristic face (the fit MUSCL already uses on
  tetrahedra). On triangles the face-neighbor fit without the ghost is
  exactly determined and unstable.
- **TENO-E**: no mirror cells across characteristic faces, as across
  partition faces. Stencils grow one-sided.

Everything else treats the face like `extrapolation`:

- the image-face values and gradients of the viscous fluxes, or zero normal
  derivatives where no image exists (the outflow viscous conditions of
  Poinsot & Lele)
- the ghosts of viscous gradients (without the characteristic face's ghost,
  as for MUSCL) and of scalar reconstruction

## Transverse terms

`T-` holds the transverse terms of the incoming wave's equation. These are
the tangential convection and the divergence of the tangential velocity,
which normal derivatives do not see:

```
T- = u_t . grad p + rho c^2 div_t u_t - rho c (u_t . grad u_n)
```

With `beta = 1`, the face evolves under them in full, as in Poinsot & Lele's
original condition. At low Mach number this pushes the low pressure of a
leaving vortex core back toward `p_t`, which sends a pressure wave upstream
and sets off a transverse slosh between walls.

[Yoo et al. (2005)](../references.md#yoo-2005),
[Yoo & Im (2007)](../references.md#yoo-im-2007) and
[Lodato, Domingo & Vervisch (2008)](../references.md#lodato-2008) cancel them
in the incoming wave except for a fraction `beta`, equal to the Mach number.
[Granet et al. (2010)](../references.md#granet-2010) show this matters most
at low Mach number. The default is the local `beta = M`; `beta = 0` drops
the terms.

The tangential derivatives are a least-squares fit in the plane of the face,
as finite-difference NSCBC differentiates along the boundary:

- **Data.** The fit uses the states of the cells of the neighboring
  characteristic faces (those sharing a node with nearly the same normal).
- **Normal derivatives.** A point along the normal sets them to zero.
- **Why not the boundary cell's own gradient.** It picks up the normal
  variation of waves crossing the boundary. On triangles it also picks up
  the grid-scale odd-even velocity pattern as a transverse divergence.
- **Why not reconstructed face states.** With TENO, they made the correction
  unstable.

## Inlet (`nscbc_inlet`)

The target is `u_t`, `T_t` and, for mixtures, the composition.

- **Acoustic wave**: the normal velocity relaxes through the face's incoming
  wave, as above.
- **Entropy and tangential velocity**: these waves are convective
  (`lambda = u_n`), so the relaxation `dphi/dt = -K_phi (phi - phi_t)` is a
  blend of the transmissive state toward the target:

  ```
  phi_g = phi_e + min(1, h K_phi / |u_n|) (phi_t - phi_e),   K_phi = sigma_phi c / L
  ```

  for `phi` = `T` or `u_t`, with `h` twice the distance from the boundary
  cell's centroid to the face.

  Without `sigma_T` or `sigma_t`, `T` and `u_t` are imposed exactly. This
  does not reflect acoustic waves: in linear theory, the reflected acoustic
  wave depends only on the incoming acoustic condition.
- **Ghost density**: `p_g / (R_t T_g)`, with `R_t` the gas constant of the
  target composition.
- **Composition**: imposed exactly, as for `upt`. It is a convective wave
  and does not reflect acoustics. A relaxed composition would need the ghost
  thermodynamics `[gamma, e0]` of a blended composition at every evaluation.
  The flow block uses the target's `[gamma, e0]` on inflow faces.

## Multicomponent and reacting flows

**Species** are convective waves. They leave through outlets from the
interior, and enter through inlets at the target composition (mass-flux
upwinding). Acoustics uses each face's frozen `gamma`.

**Reaction terms.** Yoo & Im add the reaction source terms to the LODI
system, so that heat release next to a boundary is not mistaken for an
incoming wave. Mallard Strang-splits chemistry from the flow:

- A reaction half-step changes the boundary cell at constant volume.
- The cell relaxes toward the face's incoming wave and sends the pressure
  rise inward as a wave, as any interior cell does.
- The face's own state sees the heat release only through the outgoing wave.

So no separate reaction term is needed. In the flame example (an H2/air
flame blown out through the outlet), the pressure stays within 0.7% of the
target while the flame leaves, and within 0.3% across the domain. The
largest deviation is a uniform dip while the exit velocity falls from 68 to
10 m/s, relaxed on the outlet's time scale.

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

## Validation (`test/nscbc_test.cpp`, `test/boundary3d_test.cpp`, MPI tests)

- **Acoustic pulse at normal incidence**, default low-Mach correction:
  - `sigma = 0` reflects under 0.25% (MUSCL on quadrilaterals, triangles and
    jittered mixed meshes; TENO5 on quadrilaterals), against 0.6 to 25% for
    `extrapolation`.
  - `sigma = 20` reflects as `dw-/dt = -K p'` predicts, within 3%.
  - `p_out` reflects 97 to 99%.
- **Vortex** (M = 0.25, swirl 0.05) leaving through an outlet between walls,
  against a domain three times longer. RMS pressure error at t = 3:
  - default: 0.05 times the vortex's pressure deficit, on quadrilaterals,
    triangles and jittered meshes
  - `beta = 1` (Poinsot & Lele's original condition): 1.2 to 1.3 times
- **Poiseuille flow** driven by a body force through an outlet: the profile
  stays within 1.2% and the pressure within 3.5% of the force balance, on
  quadrilaterals, triangles and jittered meshes.
- **Gas mixtures**:
  - an acoustic pulse in air leaves (under 2% reflected, against over 90%
    for `p_out`)
  - a hot H2/N2 front leaves without disturbing the pressure by more than 5%
    of its dynamic pressure
  - the flame example above
- **3D**: a plane pulse leaves a box of hexahedra or tetrahedra (under 3%).
- **MPI**: runs with characteristic boundaries and sponges are bitwise equal
  to serial ones (MUSCL and TENO, 2D and 3D).
- **Unchanged behavior**: results with the existing conditions are bitwise
  unchanged; the examples were compared with the previous build.
