# Numerics

Mallard is an explicit, cell-centered finite volume method: each face flux is
an approximate Riemann solver applied to states reconstructed on both sides,
and Runge–Kutta schemes advance the cell averages. This page is about choosing
the options; [Numerical methods](../numerics/overview.md) has the formulation.

## Presets

```toml
[run]
t_stop = 0.2       # no cfl needed: the preset sets it

[numerics]
preset = "accurate"
```

*Used in:* [`cbc_les`](https://matthewbonanni.github.io/mallard/docs/examples/#cbc-les), [`detonation_1d`](https://matthewbonanni.github.io/mallard/docs/examples/#detonation-1d).

A preset picks a validated combination of every scheme choice on this page,
so most inputs need one line.

| Preset | Reconstruction | Flux | Time integrator | `cfl` | For |
|---|---|---|---|---|---|
| `robust` | MUSCL, Venkatakrishnan limiter | RHLL | SSPRK3 | 0.5 | First runs, complex geometry, strong shocks and blasts |
| `accurate` | TENO-E, order 5 | HLLC | SSPRK3 | 1.0 | Smooth flow, acoustics, vortices, moderate shocks |
| `les` | MUSCL, no limiter | hybrid (KEEP central, HLLC at shocks) | SSPRK3 | 0.8 | Turbulence: LES and DNS |
| `reacting` | MUSCL, Venkatakrishnan limiter | HLLC | SSPRK3 | 0.35 | Flames and detonations with `[chemistry]` |

Why each, and what it passes (each check runs in the test suite,
`PresetValidation.*`, with the preset alone):

- **`robust`** never needs a stencil setup, works on every cell type and
  physics, and has the fewest ways to fail. RHLL does not grow carbuncles at
  strong shocks on blunt bodies or grid-aligned faces. `cfl = 0.5` is the
  first-order positivity bound of the explicit update in Mallard's CFL
  convention: the Noh implosion (`examples/noh_axisymmetric`, an
  infinite-strength shock into gas at `p = 1e-6`) keeps a positive pressure
  at 0.5 on 40², 60² and 200² cells, but not at 0.8 on 200² (`accurate` goes to
  negative pressure on it); on 60² the plateau density is within 7% of 64.
  On Sod's problem the L1 density error is 0.0029 at 200 cells.
- **`accurate`** is fifth order where the flow is smooth (4.9 measured on a
  convected density wave on triangles, 16² to 32² cells, at its own `cfl`)
  and ENO at discontinuities: the examples `sod`, `riemann_2d`,
  `explosion_3d`, `taylor_green_3d` and `isotropic_turbulence` use exactly
  these keys ([validation](https://matthewbonanni.github.io/mallard/validation/#accuracy)).
  It costs a stencil setup (seconds per million cells on a GPU, minutes on
  CPUs) and memory, and is not bound preserving: add
  `bound_preserving = true` and a smaller `cfl` for blasts.
- **`les`** leaves the dissipation to the viscosity and the SGS model: on a
  viscous 2D Taylor–Green vortex at Mach 0.1 (16² cells) the kinetic energy
  decays at the exact viscous rate to 0.02%, against 8% too fast with
  `robust`. TENO with the central flux feeds energy into the resolved field,
  hence the unlimited MUSCL states
  ([LES design](../design/les.md#81-decaying-isotropic-turbulence-comte-bellot-corrsin-1971)).
  These are the keys of `cbc_les` and `channel_les`, within 0.7–1.6% of the
  DNS ([validation](https://matthewbonanni.github.io/mallard/validation/#les)).
- **`reacting`**: the scheme of the validated flames and detonations
  ([validation](https://matthewbonanni.github.io/mallard/validation/#reacting-flow)).
  `cfl = 0.35` keeps the Strang splitting error of a CJ detonation's
  induction length within 5% (25% at 1.0). A stoichiometric H2/air flame at
  700 K, 6 cells per thermal thickness, propagates within 0.7% of Cantera's
  speed. Steady flames alone are insensitive to `cfl` (0.05% from 0.2 to
  2.0): set `cfl = 1.0` to run them three times faster, or use
  `[chemistry] coupling = "simpler"`.

Any key the input sets wins over the preset, so a preset is also a starting
point:

```toml
[numerics]
preset = "accurate"
riemann_solver = "RHLL"     # grid-aligned strong shocks

[numerics.face_reconstruction]
order = 3                   # TENO-E 3 instead of 5
```

Reconstruction options (`order`, `limiter`) apply only when the
reconstruction type is the preset's: `type = "MUSCL"` under `accurate` gets
MUSCL's own defaults, not `order = 5`. The preset's `cfl` applies only when
`[run]` sets neither `cfl` nor `dt`. Without a preset the defaults below
apply, unchanged (first order). The log names the preset and every key the
input overrides, and the Numerics section of the log lists what is used:

```
== Numerics ===========================
  Preset             accurate (the input sets numerics.riemann_solver = RHLL)
  Reconstruction     TENO, order 5 (degree 4), 3 face quadrature points
  Riemann solver     RHLL
```

Each example's closest preset runs in the test suite
(`Examples/ExampleUnderPreset.*`).

## Time step and stopping

```toml
[run]
cfl = 1.0          # or dt = 1.0e-4 (exactly one of the two)
t_stop = 0.2       # and/or n_steps, t_wall_stop
```

*Used in:* [`sod`](https://matthewbonanni.github.io/mallard/docs/examples/#sod), [`shu_osher`](https://matthewbonanni.github.io/mallard/docs/examples/#shu-osher), [`riemann_2d`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d), [`riemann_2d_quads`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d-quads), [`double_mach`](https://matthewbonanni.github.io/mallard/docs/examples/#double-mach) and 40 more.

| Key | Default | Meaning |
|---|---|---|
| `cfl` | — | CFL number: the time step adapts to it every step |
| `dt` | — | Fixed time step |
| `t_stop` | none | Stop at this time; the last step is shortened to land on it |
| `n_steps` | none | Stop after this many steps |
| `t_wall_stop` | none | Stop after this many seconds of wall time |
| `cuda_graphs` | `true` | CUDA builds: launch each step as one CUDA graph where possible (same results) |

With a [numerics preset](#presets) `cfl` may be left out: the preset sets it.

At least one stop condition is required; the first one reached ends the run.
`cfl` is the usual unstructured CFL number: on a uniform 2D grid of spacing h,
`dt = cfl * h / (|u| + |v| + 2a)` (viscous, diffusive and source limits
included). Boundary faces count the wave speed of their exterior state too
(the inflow state of `upt`, `dirichlet` and inlets, the target of
`nscbc_inlet`), so a fast jet entering a domain at rest runs from the first
step at the requested `cfl`.

![Largest stable CFL number with SSPRK3 for each reconstruction and cell type](../images/guide/cfl_limits-light.svg#gh-light-mode-only)
![Largest stable CFL number with SSPRK3 for each reconstruction and cell type](../images/guide/cfl_limits-dark.svg#gh-dark-mode-only)

*Largest stable `cfl` for smooth flow with SSPRK3 (RK4 allows about 10%
more), measured on periodic boxes ([tables](../numerics/overview.md#time-step)).*

Rules of thumb:

- **`cfl = 1.0` with SSPRK3** is a robust default, shocks included; at most 1.2
  (SSPRK3) or 1.35 (RK4) for any cell type.
- **Extreme shocks** with bound-preserving TENO need less: 0.25 for the Noh
  implosion.
- **Chemistry** can bind first: detonations with Strang splitting need about
  0.35, or `coupling = "simpler"` at 1.0 ([Thermochemistry](thermochemistry.md#finite-rate-chemistry)).
- Inputs from Mallard 0.5 and earlier give the same time step with half their
  `cfl`.

## Reconstruction

```toml
[numerics.face_reconstruction]
type = "TENO"      # "FO", "MUSCL" or "TENO"; the default is "FO"
order = 5          # TENO: 3 to 6
```

*Used in:* [`sod`](https://matthewbonanni.github.io/mallard/docs/examples/#sod), [`shu_osher`](https://matthewbonanni.github.io/mallard/docs/examples/#shu-osher), [`riemann_2d`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d), [`riemann_2d_quads`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d-quads), [`double_mach`](https://matthewbonanni.github.io/mallard/docs/examples/#double-mach) and 40 more.

!!! warning "The default is first order"
    Without a [preset](#presets) or `[numerics.face_reconstruction]` Mallard
    runs first order. Every example sets a preset or the reconstruction.

![Density of the Sod shock tube at t = 0.2 with first order, MUSCL, TENO-E 3 and TENO-E 5 against the exact solution](../images/guide/sod_schemes-light.svg#gh-light-mode-only)
![Density of the Sod shock tube at t = 0.2 with first order, MUSCL, TENO-E 3 and TENO-E 5 against the exact solution](../images/guide/sod_schemes-dark.svg#gh-dark-mode-only)

*The Sod example (200 cells) with each reconstruction, read through a line
probe. With TENO-E 5 the contact spreads over about five cells and the shock
over two.*

| `type` | Order | Use |
|---|---|---|
| `FO` | 1 | Debugging and start-up transients; the most robust and dissipative |
| `MUSCL` | 2 | Robust and cheap; complex geometry, very large meshes, DNS of wall turbulence at fine resolution |
| `TENO` | 3–6 | TENO-E (Liang, Shyy & Fu 2025): high order in smooth regions, essentially non-oscillatory at shocks. Order 5 is the usual choice |

**MUSCL** keys: `limiter` (`venkatakrishnan`, the default; `barth_jespersen`;
`none`) and `venkatakrishnan_K` (default 5; larger is less limiting).

**TENO-E** reconstructs each cell from a large central stencil where the flow
is smooth, and switches per characteristic variable to the smoothest of its
sector stencils where a troubled-cell indicator detects a discontinuity:

![The central stencil and the three sector stencils of one triangle, as Mallard builds them for fifth order](../images/guide/teno_stencils-light.svg#gh-light-mode-only)
![The central stencil and the three sector stencils of one triangle, as Mallard builds them for fifth order](../images/guide/teno_stencils-dark.svg#gh-dark-mode-only)

*Stencils of one triangle (black) for fifth-order TENO-E, read back from
Mallard's stencil cache.*

### TENO-E options

```toml
[numerics.face_reconstruction]
type = "TENO"
order = 5
bound_preserving = true   # for very strong shocks
```

*Used in:* [`sedov_3d`](https://matthewbonanni.github.io/mallard/docs/examples/#sedov-3d), [`sphere_mach3`](https://matthewbonanni.github.io/mallard/docs/examples/#sphere-mach3), [`mallard`](https://matthewbonanni.github.io/mallard/docs/examples/#mallard), [`mallard_logo`](https://matthewbonanni.github.io/mallard/docs/examples/#mallard-logo), [`noh_axisymmetric`](https://matthewbonanni.github.io/mallard/docs/examples/#noh-axisymmetric) and 1 more.

| Key | Default | Meaning |
|---|---|---|
| `order` | 5 | 3 to 6 |
| `bound_preserving` | `false` | Keep density and pressure in the neighbors' range in troubled cells, and (planar single gas) positive in every cell at `cfl` up to 0.5: for very strong shocks, blasts and near vacuum (Noh, Sedov, Mach 8 flows) |
| `stencil_factor` | 2 | Large-stencil size over the number of coefficients; 1.5 is markedly less dissipative for fine smooth waves, less robust at shocks |
| `troubled_threshold`, `troubled_upper` | 1e-3, 1e-2 | Troubled-cell indicator: where stencil selection starts, and where its cutoff is most dissipative |
| `C_T` | adaptive | Fixed TENO cutoff instead of the adaptive one (1e-10 to 1e-6) |
| `characteristic` | `true` | Select stencils on characteristic variables |
| `small_stencil_size` | 10 (2D), 18 (3D) | Cells per sector stencil |
| `max_condition` | 1e8 | Conditioning bound of the least-squares systems |
| `cache_file` | none | Save the stencils and matrices, and reuse them on later runs of the same mesh and options |
| `cache_single_precision` | `false` | Store the cache in single precision (half the size) |
| `single_precision_tables` | `false` | 3D, single gas: keep the matrices in single precision on the device: ~40% less memory, 3–8% faster, results change at round-off |
| `troubled_capacity` | 0.5 (3D), 1 (2D) | Share of cells the troubled-cell passes handle at a time (memory) |
| `curved_mirrors`, `curved_wall_degree` | `false`, 3 | Stencils at [curved walls](meshes.md#curved-boundaries) |

Pitfalls:

- **Setup time.** TENO-E precomputes its stencils at the start: seconds per
  million cells on a GPU (3D), minutes on CPUs. Set `cache_file` for repeated
  runs of the same mesh; the cache is checked against the mesh and options and
  recomputed if they differ. Its size is several KB per cell (tens in 3D).
- **Memory.** 3D TENO5 needs about 40 KB per hexahedron on the device (an
  A100-80GB holds about 2.1M); `single_precision_tables` stretches that to
  about 3.4M.
- The **troubled-cell share** is in every progress row; output `TENO_SIGMA` to
  see where stencils are being selected.

## Riemann solvers

```toml
[numerics]
riemann_solver = "HLLC"   # default
```

*Used in:* [`sod`](https://matthewbonanni.github.io/mallard/docs/examples/#sod), [`shu_osher`](https://matthewbonanni.github.io/mallard/docs/examples/#shu-osher), [`riemann_2d`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d), [`riemann_2d_quads`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d-quads), [`double_mach`](https://matthewbonanni.github.io/mallard/docs/examples/#double-mach) and 40 more.

| `riemann_solver` | Use |
|---|---|
| `HLLC` | Default: resolves contacts and shear layers; all-round choice |
| `HLL` | More dissipative at contacts and shear layers; with bound-preserving TENO for the Mach 3 sphere on tetrahedra |
| `Roe` | Low dissipation, with Harten's entropy fix |
| `Rusanov` | The most dissipative |
| `RHLL` | Rotated hybrid HLL-Roe: no carbuncle at strong shocks on blunt bodies and grid-aligned shocks (the double Mach reflection, Mach 8 flows, blunt-body detonations) |

All five work for gas mixtures. On a 1D problem with fifth-order TENO the
choice matters little (within 20% on Sod); in 2D it decides whether strong
shocks grow carbuncles: use `RHLL` there
([stationary shocks](../numerics/overview.md#stationary-shocks-on-cell-faces)).

## Time integration and other options

```toml
[numerics]
time_integrator = "RK4"
```

*Used in:* every example keeps the default, `SSPRK3`; the others are in the time-integration tests.

| `[numerics]` key | Default | Meaning |
|---|---|---|
| `time_integrator` | `"SSPRK3"` | `FE` (forward Euler), `SSPRK3`, `RK4`, `SSPRK54`, `CK45` or `RDPK3Sp35` (below) |
| `low_mach_cutoff` | 0.1 | Low-Mach correction: the upwind dissipation scales with the flow speed down to this Mach number; 1 disables it |
| `convective_flux` | `"riemann"` | `hybrid`: central kinetic-energy-preserving flux with the Riemann solver at shocks ([LES](les.md#low-dissipation-flux-and-the-budget)) |
| `double_flux` | `false` | Gas mixtures: no pressure oscillations at contacts between different gases ([Thermochemistry](thermochemistry.md#contacts-between-different-gases)) |
| `check_nan` | `false` | Stop as soon as the solution is not finite |

The low-Mach correction matters below about Mach 0.3: without it, upwind
dissipation at M = 0.1 removes a third of the resolved enstrophy of an
under-resolved Taylor–Green vortex. Leave it on except for comparisons.

### Choosing an integrator

| Integrator | Order, stages | Registers | Safe `cfl` (smooth) | Cost per unit time vs SSPRK3 (A100, TENO5) |
|---|---|---|---|---|
| `SSPRK3` | 3, 3 | 3 | 1.2 | 1 |
| `RK4` | 4, 4 | 6 | 1.35 | 1.20 |
| `SSPRK54` | 4, 5 (SSP) | 4 | 2.2 | 0.78 |
| `CK45` | 4, 5 (2N) | 3 | 2.2 | 0.89 |
| `RDPK3Sp35` | 3, 5 (embedded 2) | 5 | 2.2 | 0.84 |

- **`SSPRK54`** is the cheapest per unit time for smooth flow and keeps SSP
  (monotone forward-Euler steps stay monotone up to `cfl` 1.5 times
  forward Euler's), with a time error about 45 times smaller than SSPRK3's at
  the same step. With shocks, `cfl = 1.5` matches SSPRK3 at 1.0 in accuracy.
- **`CK45`** needs the least memory of the fourth-order methods.
- **`RDPK3Sp35`** with `[numerics.error_control]` chooses the step itself:

```toml
[numerics]
time_integrator = "RDPK3Sp35"

[numerics.error_control]
tolerance = 1e-4   # absolute and relative; [run] cfl only sizes the first step
```

Pitfalls: error control is not available with `[chemistry]` or `double_flux`;
its steps follow the time error, not the stability limit, so a loose
tolerance can still take steps above the `cfl` you would choose by hand (the
controller rejects them once the error grows). Measured limits and costs:
[time integrators](../numerics/overview.md#time-integrators).

## See also

- [Input reference: `[numerics]`](../input.md#numerics), [`[numerics.face_reconstruction]`](../input.md#numericsface_reconstruction)
- [Numerical methods](../numerics/overview.md) and [TENO-E details](../numerics/teno_e.md)
- [Validation: order of accuracy](https://matthewbonanni.github.io/mallard/validation/#accuracy), [shocks](https://matthewbonanni.github.io/mallard/validation/#shocks)
