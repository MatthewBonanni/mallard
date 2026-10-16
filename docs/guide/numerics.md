# Numerics

Mallard is an explicit, cell-centered finite volume method: each face flux is
an approximate Riemann solver applied to states reconstructed on both sides,
and Runge–Kutta schemes advance the cell averages. This page is about choosing
the options; [Numerical methods](../numerics/overview.md) has the formulation.

## Time step and stopping

```toml
[run]
cfl = 1.0          # or dt = 1.0e-4 (exactly one of the two)
t_stop = 0.2       # and/or n_steps, t_wall_stop
```

| Key | Default | Meaning |
|---|---|---|
| `cfl` | — | CFL number: the time step adapts to it every step |
| `dt` | — | Fixed time step |
| `t_stop` | none | Stop at this time; the last step is shortened to land on it |
| `n_steps` | none | Stop after this many steps |
| `t_wall_stop` | none | Stop after this many seconds of wall time |
| `cuda_graphs` | `true` | CUDA builds: launch each step as one CUDA graph where possible (same results) |

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

!!! warning "The default is first order"
    Without `[numerics.face_reconstruction]` Mallard runs first order. Every
    example sets the reconstruction explicitly.

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

| Key | Default | Meaning |
|---|---|---|
| `order` | 5 | 3 to 6 |
| `bound_preserving` | `false` | Keep density and pressure in the neighbors' range in troubled cells: for very strong shocks and blasts (Noh, Sedov, Mach 8 flows) |
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

| `[numerics]` key | Default | Meaning |
|---|---|---|
| `time_integrator` | `"SSPRK3"` | `FE` (forward Euler), `SSPRK3` or `RK4` |
| `low_mach_cutoff` | 0.1 | Low-Mach correction: the upwind dissipation scales with the flow speed down to this Mach number; 1 disables it |
| `convective_flux` | `"riemann"` | `hybrid`: central kinetic-energy-preserving flux with the Riemann solver at shocks ([LES](les.md#low-dissipation-flux-and-the-budget)) |
| `double_flux` | `false` | Gas mixtures: no pressure oscillations at contacts between different gases ([Thermochemistry](thermochemistry.md#contacts-between-different-gases)) |
| `check_nan` | `false` | Stop as soon as the solution is not finite |

The low-Mach correction matters below about Mach 0.3: without it, upwind
dissipation at M = 0.1 removes a third of the resolved enstrophy of an
under-resolved Taylor–Green vortex. Leave it on except for comparisons.

## See also

- [Input reference: `[numerics]`](../input.md#numerics), [`[numerics.face_reconstruction]`](../input.md#numericsface_reconstruction)
- [Numerical methods](../numerics/overview.md) and [TENO-E details](../numerics/teno_e.md)
- [Validation: order of accuracy](https://matthewbonanni.github.io/mallard/validation/#accuracy), [shocks](https://matthewbonanni.github.io/mallard/validation/#shocks)
