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
preset = "high_order"   # the default: without this line, too
```

*Used in:* [`cbc_les`](https://matthewbonanni.github.io/mallard/docs/examples/#cbc-les), [`detonation_1d`](https://matthewbonanni.github.io/mallard/docs/examples/#detonation-1d).

A preset picks a validated combination of every scheme choice on this page,
so most inputs need one line, or none: an input that sets neither `preset`
nor `[numerics.face_reconstruction]` runs `high_order`.

| Preset | Order | Reconstruction | Flux | Time integrator | `cfl` | For |
|---|---|---|---|---|---|---|
| `high_order` (default) | 5 (TENO-E) | TENO-E, order 5 | HLLC | SSPRK3 | 1.0 | Smooth flow, acoustics, vortices, moderate shocks: the least error for the cost ([why high order](#why-high-order)) |
| `second_order` | 2 (MUSCL) | MUSCL, Venkatakrishnan limiter | RHLL | SSPRK3 | 0.5 | Complex geometry, strong shocks and blasts |
| `first_order` | 1 | none (cell averages) | HLL | SSPRK3 | 0.5 | Debugging, start-up transients, the last fallback: low order and very dissipative |
| `les` | 2 (MUSCL) | MUSCL, no limiter | hybrid (KEEP central, HLLC at shocks) | SSPRK54 | 2.0 | Turbulence: LES and DNS |
| `reacting` | 2 (MUSCL) | MUSCL, Venkatakrishnan limiter | HLLC | SSPRK3 | 0.35 | Flames and detonations with `[chemistry]` |

Why each, and what it passes (each check runs in the test suite,
`PresetValidation.*`, with the preset alone):

- **`high_order`** is fifth order where the flow is smooth (4.9 measured on a
  convected density wave on triangles, 16² to 32² cells, at its own `cfl`)
  and ENO at discontinuities: the examples `sod`, `riemann_2d`,
  `explosion_3d`, `taylor_green_3d` and `isotropic_turbulence` use exactly
  these keys ([verification](https://matthewbonanni.github.io/mallard/validation/#accuracy)).
  It costs a stencil setup (seconds per million cells on a GPU, minutes on
  CPUs) and memory, and is not bound preserving: add
  `bound_preserving = true` and a smaller `cfl` for blasts, or use
  `second_order`.
- **`second_order`** never needs a stencil setup, works on every cell type and
  physics, and has the fewest ways to fail of the second-order schemes. RHLL does not grow carbuncles at
  strong shocks on blunt bodies or grid-aligned faces. `cfl = 0.5` is the
  first-order positivity bound of the explicit update in Mallard's CFL
  convention: the Noh implosion (`examples/noh_axisymmetric`, an
  infinite-strength shock into gas at `p = 1e-6`) keeps a positive pressure
  at 0.5 on 40², 60² and 200² cells, but not at 0.8 on 200² (`high_order` goes to
  negative pressure on it); on 60² the plateau density is within 7% of 64.
  On Sod's problem the L1 density error is 0.0029 at 200 cells.
- **`first_order`** is first order everywhere: it smears every wave over
  many cells, and its error falls only in proportion to the cell size. It
  is for debugging an input, for the first steps of a violent transient
  (restart from it with a higher order), and as the last fallback when
  nothing else stays positive. HLL, with Einfeldt's wave speeds, is the flux
  that keeps the first-order update positive ([Einfeldt et al. 1991](../references.md#einfeldt-1991)), here at
  `cfl = 0.5`, the bound above, and the most dissipative at contacts and shear
  layers, so it grows no carbuncles. On the Noh implosion at 60² cells it
  keeps a positive pressure, with a plateau density of 52 against 64: the
  price of its dissipation. It is not the scheme to judge a mesh or a result
  by.
- **`les`** (second order) leaves the dissipation to the viscosity and the SGS model: on a
  viscous 2D Taylor–Green vortex at Mach 0.1 (16² cells) the kinetic energy
  decays at the exact viscous rate to 0.02%, against 8% too fast with
  `second_order`. TENO with the hybrid flux gives the same result at ten times the cost,
  hence the unlimited MUSCL states
  ([LES design](../design/les.md#44-central-flux-for-teno-689)).
  These are the reconstruction and flux of `cbc_les` and `channel_les`, within 0.7–1.6% of the
  DNS ([benchmark](https://matthewbonanni.github.io/mallard/validation/#les)).
  SSPRK54 at `cfl = 2.0` takes two thirds of SSPRK3's right-hand-side
  evaluations at 0.8 for the same result: `cbc_les` gives the same spectra
  at both stations (log10 RMS deviation 0.107 / 0.104 and resolved energy
  1.250 / 1.020 on 64³ cells, 0.153 / 0.151 and 1.538 / 1.505 on 32³) in 408
  steps against 1,018 (64³), and the kinetic energy of
  the Taylor–Green vortex (2D as above, and 3D on 32³ cells to t = 10) agrees
  within 0.06%. Under-resolved double shear layers stay stable up to `cfl = 3.2`
  on quads and triangles, and a turbulent channel on stretched cells (32 x
  48 x 32) at 2.5.
- **`reacting`** (second order): the scheme of the validated flames and detonations
  ([benchmark](https://matthewbonanni.github.io/mallard/validation/#reacting-flow)).
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
preset = "high_order"
riemann_solver = "RHLL"     # grid-aligned strong shocks

[numerics.face_reconstruction]
order = 3                   # TENO-E 3 instead of 5
```

Reconstruction options (`order`, `limiter`) apply only when the
reconstruction type is the preset's: `type = "MUSCL"` under `high_order` gets
MUSCL's own defaults, not `order = 5`. The preset's `cfl` applies only when
`[run]` sets neither `cfl` nor `dt`. An input with `[numerics.face_reconstruction]`
and no preset gets no preset: the defaults below apply to every key it leaves
out (`type = "FO"`, first order, when the table leaves out `type`). The terminal output names the preset, whether it is the default, and every key the
input overrides, and its Numerics section lists what is used:

```
== Numerics ===========================
  Preset             high_order (the input sets numerics.riemann_solver = "RHLL")
  Reconstruction     TENO, order 5 (degree 4), 3 face quadrature points
  Riemann solver     RHLL
```

Each example's closest preset runs in the test suite
(`Examples/ExampleUnderPreset.*`).

### Why high order

Refining the mesh (smaller h) and raising the order (larger p) both reduce
the error, but at different rates: the error of an order-p scheme on smooth
flow falls like h<sup>p</sup>, while the cost of a 2D run grows like
h<sup>-3</sup> (cells times time steps) whatever the order. A higher order
costs more per cell, and on any but the coarsest meshes that is repaid many
times over:

![L2 density error of the isentropic vortex on irregular triangles against cell count and CPU time, for FO, MUSCL and TENO-E 3 to 6](../images/guide/high_order_cost-light.svg#gh-light-mode-only)
![L2 density error of the isentropic vortex on irregular triangles against cell count and CPU time, for FO, MUSCL and TENO-E 3 to 6](../images/guide/high_order_cost-dark.svg#gh-dark-mode-only)

*The isentropic vortex of [Shu (1998)](../references.md#shu-1998), strength
5, convected 2 units across a periodic 10 × 10 box of irregular triangles
(nodes jittered by up to a quarter cell), with HLLC, SSPRK3 and `cfl = 1.0`
for every scheme (MUSCL unlimited). Error: the L2 norm of the density cell
averages at t = 2 against the exact solution. Cost: the CPU time of the
whole run, setup and output included, on 2 threads of an Apple Silicon
laptop shared with other jobs (hence CPU rather than wall time, also
recorded). Script and data: `tools/docs/high_order_cost.py`.*

Read off the curves (interpolated between the measured meshes):

| Error | FO | MUSCL | TENO-E 3 | TENO-E 4 | TENO-E 5 | TENO-E 6 |
|---|---|---|---|---|---|---|
| 10<sup>-3</sup> | not reached (1.7 × 10<sup>-3</sup> on 524,288 cells, 395 s) | 5,800 cells, 0.57 s | 2,700 cells, 0.52 s | 960 cells, 0.25 s | 1,100 cells, 0.62 s | 750 cells, 0.76 s |
| 10<sup>-4</sup> | not reached | 64,500 cells, 20.8 s | 13,000 cells, 3.7 s | 3,300 cells, 1.1 s | 3,600 cells, 2.8 s | 1,900 cells, 2.4 s |
| 10<sup>-5</sup> | not reached | not reached | 61,000 cells, 35 s | 10,000 cells, 4.6 s | 9,700 cells, 10.4 s | 4,400 cells, 6.9 s |
| Measured order (two finest meshes) | 0.97 | 1.89 (1.23 from 131,072 to 524,288 cells) | 3.00 | 3.90 | 4.87 | 5.54 |

- **At an error of 10<sup>-4</sup>** TENO-E 5 needs 18 times fewer cells than
  MUSCL and 7.4 times less CPU time; FO would need, at its measured first
  order, about 300 times the cells of its finest run (some 150 million).
- **FO's finest run** (524,288 cells, 395 s) has an error of 1.7 × 10<sup>-3</sup>;
  TENO-E 5 has 3.8 × 10<sup>-4</sup> on 2,048 cells in 1.3 s.
- **The crossover with second order:** above an error of about 9 × 10<sup>-4</sup>
  (here, meshes under about 5,000 cells, some 7 cells per core radius of the
  vortex) MUSCL is as cheap as TENO-E 5 or cheaper: at 10<sup>-3</sup> it
  costs 0.92 times TENO-E 5's time. Below it the gap widens with every
  refinement, since each halving of h divides TENO-E 5's error by about 29
  and MUSCL's by about 4.
- **Which order is cheapest:** TENO-E 4, over the whole range from
  10<sup>-3</sup> to 10<sup>-5</sup>. It needs about as many cells as TENO-E 5
  for a given error, at a smaller stencil, and so reaches it in about 0.4
  times TENO-E 5's CPU time. TENO-E 6 is cheaper than TENO-E 5 below an error
  of about 2.5 × 10<sup>-3</sup> (0.66 times its time at 10<sup>-5</sup>) and,
  at the highest order, closes in on TENO-E 4 as the error falls. TENO-E 3 is
  cheaper than TENO-E 5 only above about 1.8 × 10<sup>-4</sup>.
- **Why `high_order` is order 5 nonetheless:** this is one smooth 2D case.
  `high_order` stays at order 5 until order 4 is shown to be as robust: the
  shock checks the presets pass, the odd-even wall-pressure mode on stretched
  wall cells, which at order 4 [only the metric and the mirror images together remove](../numerics/teno_e.md#candidates),
  and the [linear instability of periodic pyramid lattices](../numerics/teno_e.md#known-limitations)
  at order 4 in 3D ([#711](https://github.com/MatthewBonanni/mallard/issues/711)). For smooth flow, setting
  `order = 4` under `[numerics.face_reconstruction]` is the cheaper choice
  this study measured.

The advantage is for **smooth flow**. At shocks and contacts TENO-E falls
back to its ENO sector stencils and every scheme is first order there, so on
a flow dominated by discontinuities the gain shrinks to the smooth regions
between them; for the strongest shocks and blasts, `second_order` (no stencil
setup, fewer ways to fail) or, as a last resort, `first_order` is the right
tool. TENO-E also has a one-time stencil setup, included in the times above:
for TENO-E 5 it is 27% of the run on 2,048 cells and 11% on 32,768 (it grows
with the cells, the time stepping with cells times steps; seconds per
million cells on a GPU, minutes on CPUs), and it holds more memory per cell.

## Time step and stopping

```toml
[run]
cfl = 1.0          # or dt = 1.0e-4 (exactly one of the two)
t_stop = 0.2       # and/or n_steps, t_wall_stop
```

*Used in:* [`sod`](https://matthewbonanni.github.io/mallard/docs/examples/#sod), [`shu_osher`](https://matthewbonanni.github.io/mallard/docs/examples/#shu-osher), [`riemann_2d`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d), [`riemann_2d_quads`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d-quads), [`double_mach`](https://matthewbonanni.github.io/mallard/docs/examples/#double-mach) and 46 more.

| Key | Default | Meaning |
|---|---|---|
| `cfl` | — | CFL number: the time step adapts to it every step |
| `dt` | — | Fixed time step |
| `t_stop` | none | Stop at this time; the last step is shortened to land on it |
| `n_steps` | none | Stop after this many steps |
| `t_wall_stop` | none | Stop after this many seconds of wall time |
| `cuda_graphs` | `true` | CUDA builds: launch each step as one CUDA graph where possible (same results) |
| `step_queue` | 3 | Steps the host enqueues ahead of the GPU; 1 waits for each step, 0 has the host take each step (same results) |

With a [numerics preset](#presets) `cfl` may be left out: the preset sets it.

At least one stop condition is required; the first one reached ends the run.
`cfl` is the usual unstructured CFL number: on a uniform 2D grid of spacing h,
`dt = cfl * h / (|u| + |v| + 2a)` (viscous, diffusive and source limits
included). Boundary faces count the wave speed of their exterior state too
(the inflow state of `inlet_state`, `dirichlet` and the other inlets, or the
target of an `inlet_state` with `nscbc`), so a fast jet entering a domain at rest runs from the first
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
type = "TENO"      # "FO", "MUSCL" or "TENO"; without a preset, the default is "FO"
order = 5          # TENO: 3 to 6
```

*Used in:* [`sod`](https://matthewbonanni.github.io/mallard/docs/examples/#sod), [`shu_osher`](https://matthewbonanni.github.io/mallard/docs/examples/#shu-osher), [`riemann_2d`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d), [`riemann_2d_quads`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d-quads), [`double_mach`](https://matthewbonanni.github.io/mallard/docs/examples/#double-mach) and 46 more.

!!! note "Defaults"
    Without a [preset](#presets) or this table Mallard runs the `high_order`
    preset (TENO-E 5). With this table and no preset, `type` defaults to
    `"FO"`: set it. Before Mallard 0.10 the default without either was first
    order: add `type = "FO"` to reproduce older results (`mallard input upgrade`
    [notes the inputs this concerns](input_files.md#inputs-from-older-releases)).

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

*Used in:* [`sod`](https://matthewbonanni.github.io/mallard/docs/examples/#sod), [`shu_osher`](https://matthewbonanni.github.io/mallard/docs/examples/#shu-osher), [`riemann_2d`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d), [`riemann_2d_quads`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d-quads), [`double_mach`](https://matthewbonanni.github.io/mallard/docs/examples/#double-mach) and 46 more.

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

*Used in:* [`cbc_les`](https://matthewbonanni.github.io/mallard/docs/examples/#cbc-les).

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

- **`SSPRK54`**, the `les` preset's, is the cheapest per unit time for smooth flow and keeps SSP
  (monotone forward-Euler steps stay monotone up to `cfl` 1.5 times
  forward Euler's), with a time error about 45 times smaller than SSPRK3's at
  the same step. With shocks, `cfl` 1.5 to 1.8 matches SSPRK3 at 1.0 in accuracy
  (Shu–Osher, the double Mach reflection and the 2D Riemann problem with TENO5),
  for 7% fewer right-hand-side evaluations at 1.8. The `high_order` preset
  keeps SSPRK3: the saving is small, and an input that sets `cfl = 1.0`
  would take 5/3 as many evaluations with SSPRK54.
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
- [Verification: order of accuracy](https://matthewbonanni.github.io/mallard/validation/#accuracy), [shocks](https://matthewbonanni.github.io/mallard/validation/#shocks)
