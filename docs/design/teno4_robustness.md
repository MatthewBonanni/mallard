# The order of the `high_order` preset

Status: `high_order` is TENO-E order 5. Order 4 reaches order 5's error on
smooth flow at 0.3 to 0.5 times the CPU time, with 0.4 to 0.6 times the
memory, but it fails three robustness checks that order 5 passes. Each failure
has its own issue, and the preset can move to order 4 once all three are
fixed ([#711](https://github.com/MatthewBonanni/mallard/issues/711)).

| Failure at order 4 | Issue |
|---|---|
| Growing modes on pyramid lattices, cubic and thin | [#236](https://github.com/MatthewBonanni/mallard/issues/236) |
| An outgoing acoustic wave amplified on a stretched O-grid (cylinder) | [#975](https://github.com/MatthewBonanni/mallard/issues/975) |
| Odd-even wall pressure at the closed trailing edge of the NACA 0012 mesh | [#976](https://github.com/MatthewBonanni/mallard/issues/976) |
| Undershoots ahead of shocks on quads, about 10 times order 5's | [#977](https://github.com/MatthewBonanni/mallard/issues/977) |

## Setup

Each case ran twice, with `preset = "high_order"` and
`[numerics.face_reconstruction] order = 4` or `5`. Every other preset choice
(HLLC, SSPRK3, `cfl = 1.0`) was the same, and so were the case's own physics,
mesh and boundary conditions. Shu–Osher and the double Mach reflection keep
their example's `riemann_solver = "RHLL"`, and both were also run with HLLC.
The machine was an Apple M4 Pro with 2 threads per run (Threads backend,
Release, double precision). Cost is the run's CPU time, setup included,
because the machine was shared. Memory is the largest resident host memory
of the progress records (`memory_host_peak`).

Health counts (`mallard.jsonl`): no run crashed or stopped on a NaN. The
`high_order` preset makes no repairs, because it is not bound preserving
(`repairs_run` stayed empty). `limited_cells` counts only MUSCL limiters, so
the troubled-cell fraction and the lowest density and pressure stand in for
it here.

## Accuracy and cost on shocks and flows

Errors are L1 norms of the density, measured against:

- **Sod:** the exact solution's cell averages.
- **Shu–Osher:** the integral over [0, 10] against TENO-E 5 on 6,400 cells.
- **2D Riemann:** the mean against TENO-E 5 on 400² triangles.
- **Double Mach reflection:** the mean for x < 2.7 against TENO-E 5 on 480 × 120 triangles.
- **3D explosion:** the mean against TENO-E 5 on 48³ hexahedra.

The CPU ratio is order 4's time over order 5's.

| Case | Cells | Error, order 4 / 5 | CPU ratio | Memory ratio |
|---|---|---|---|---|
| `sod`, 200 × 4 quads | 800 | 0.0020 / 0.0024 | 0.67 | 0.82 |
| `sod`, 400 × 4 quads | 1,600 | 0.0011 / 0.0013 | 0.71 | 0.77 |
| `shu_osher`, 400 × 4 quads, RHLL | 1,600 | 0.204 / 0.211 | 0.59 | 0.76 |
| `riemann_2d`, 100² triangles | 20,000 | 0.033 / 0.042 | 0.57 | 0.63 |
| `riemann_2d`, 200² triangles | 80,000 | 0.032 / 0.034 | 0.55 | 0.61 |
| `double_mach`, 240 × 60 triangles, RHLL | 28,800 | 0.065 / 0.076 | 0.54 | 0.63 |
| `double_mach`, the same with HLLC | 28,800 | 0.073 / 0.070 | 0.52 | 0.63 |
| `explosion_3d`, 32³ hexahedra | 32,768 | 0.00073 / 0.00085 | 0.47 | 0.59 |
| `explosion_3d`, 16³ blocks of tetrahedra | 24,576 | 0.0024 / 0.0024 | 0.34 | 0.42 |
| `explosion_3d`, 12³ blocks of pyramids | 10,368 | 0.0029 / 0.0031 | 0.34 | 0.49 |
| `explosion_3d`, 16 × 16 × 16 blocks of prisms | 8,192 | 0.0029 / 0.0029 | 0.50 | 0.62 |
| Premixed H2/air flame, 60 cells (speed error against Cantera's 9.7473 m/s) | 60 | −0.68% / −0.85% | 0.57 | 0.97 |
| `taylor_green_3d`, 32³ hexahedra, t = 3 (kinetic energy) | 32,768 | 29.70 / 29.74 | 0.38 | 0.62 |

The flame is the `PresetValidation` flame: 6 cells per thermal thickness, the
speed of the 1,000 K isotherm. On these cases order 4 is at least as accurate
as order 5 at about half the cost.

## Smooth flow in 3D

A density wave, $\rho = 1 + 0.01\sin 2\pi(x + y + z)$, was convected at
$u = 1$ once across a periodic unit cube. The error is the L2 norm of the
difference from the initial cell averages; the amplitude is small enough that
no cell is troubled.

| Mesh | Order | n = 8 | 12 | 16 | 24 |
|---|---|---|---|---|---|
| hexahedra, error | 4 | 1.1e-3 | 2.2e-4 | 6.7e-5 | 1.2e-5 |
| | 5 | 2.1e-3 | 4.2e-4 | 1.1e-4 | 1.6e-5 |
| hexahedra, CPU [s] | 4 | 0.29 | 1.1 | 2.9 | 11.6 |
| | 5 | 0.56 | 2.4 | 6.9 | 26.6 |

| Mesh | Order | n = 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|
| Kuhn tetrahedra (6 per block), error | 4 | 9.7e-4 | 3.6e-4 | 7.8e-5 | 2.5e-5 |
| | 5 | 9.2e-4 | 2.8e-4 | 4.4e-5 | 1.1e-5 |
| tetrahedra, CPU [s] | 4 | 0.76 | 2.4 | 10.2 | 30.6 |
| | 5 | 2.9 | 8.1 | 34.1 | 90.8 |
| tetrahedra, stencil setup [s] | 4 | 0.09 | 0.21 | 0.73 | 1.7 |
| | 5 | 0.61 | 1.5 | 4.9 | 11.6 |
| tetrahedra, memory [MB] | 4 | 95 | 203 | 557 | 813 |
| | 5 | 206 | 476 | 1,352 | 2,017 |

On hexahedra order 4 has the smaller error on every mesh at 0.42 to 0.52
times the CPU time. On tetrahedra order 5 converges faster: at an error of
$7.8\times 10^{-5}$, order 4 takes 10.2 s and order 5, interpolated between
its meshes, about 22 s (0.47 times). On 24,576 tetrahedra, order 4's stencil
setup takes 0.15 times order 5's and its peak memory 0.40 times. This
repeats the 2D cost study ([why high order](../guide/numerics.md#why-high-order))
in 3D.

## Linear stability

These runs perturb a gas at rest (unit density and pressure) by a velocity
of $10^{-3}$, in boxes of unit size, under the preset (HLLC, `cfl = 1.0`),
to t = 50 in 3D and t = 100 in 2D. The rate is a fit to the logarithm of the
largest Mach number over the second half of the run, in units of the sound
speed over the box size. Negative or about zero is bounded.

| Mesh | Boundaries | Order 3 | Order 4 | Order 5 |
|---|---|---|---|---|
| `cartesian_pyramid` 6³ | periodic | −0.005 | **+0.021** | −0.003 |
| `cartesian_pyramid` 6³ | symmetry walls | −0.005 | **+0.011** | +0.0004 |
| `cartesian_pyramid` 4³ | periodic | −0.0003 | **+0.009** | −0.0007 |
| `cartesian_pyramid` 6 × 6 × 4, aspect ratio 6.25 | side walls, periodic in z | −0.003 | **+0.033** | −0.005 |
| `cartesian_pyramid` 6 × 6 × 4, aspect ratio 6.25 | walls | −0.006 | −0.001 | −0.003 |
| `cartesian`, `cartesian_tet`, `cartesian_prism`, `cartesian_mixed` 6³ | periodic; walls | ≤ +0.0009 | ≤ +0.0024 | ≤ +0.0013 |
| jittered Kuhn tetrahedra 6³ (0.15 h) | periodic; walls | ≤ +0.0006 | ≤ −0.0007 | ≤ +0.0008 |
| jittered hexahedra and prisms 8³ (0.15 h) | periodic | −0.001 | +0.0007 | −0.0009 |
| `cartesian_tet`, `cartesian_prism`, aspect ratio 6.25 | walls; side walls | ≤ +0.0002 | ≤ +0.0004 | ≤ −0.0001 |
| `cartesian_tet`, `cartesian_prism` 8³, `stretching = [0, 2.5, 0]` | walls | ≤ +0.0007 | ≤ +0.0005 | ≤ +0.0002 |

In 2D the largest Mach number fell or stayed level at every order on jittered
triangles (32² squares, 0.25 h, periodic and walled), on triangles of aspect
ratio 6 (periodic), and on quads and triangles with `stretching = [0, 2]`
(walls, aspect ratio about 14 at the walls).

Only pyramids grow, and only at order 4. The thin pyramid box with side walls
grows fastest: 3.8 times its initial speed by t = 50. The cubic-lattice rates
match the linearized operator in [#236](https://github.com/MatthewBonanni/mallard/issues/236)
and [TENO-E: known limitations](../numerics/teno_e.md#known-limitations).

## Wall-bounded flows

**Cylinder** ([#975](https://github.com/MatthewBonanni/mallard/issues/975)).
The case is `examples/cylinder` at Re = 100, Mach 0.2, with a far field, on
a coarse O-grid (`make_cylinder_mesh.py --n-theta 192 --n-r 64 --r-far 25
--dr0 0.01`: 12,288 quads, radial growth ratio 1.088), run to t = 60.

| | Order 4 | Order 5 |
|---|---|---|
| largest Mach number | 1.02 | 0.33 |
| lowest density / pressure (free stream 1 / 0.714) | 0.40 / 0.22 | 0.91 / 0.63 |
| troubled cells, the most at a progress row | 59% | 0% |
| pressure drag, mean ± std over t = 30 to 60 | 0.0065 ± 0.148 | 0.0233 ± 0.023 |
| CPU time [s] | 2,576 | 781 |

At order 4 the pressure minimum travels outward at about the sound speed with
the start-up acoustic wave, and it deepens as it goes: 0.58 at r = 3.3
(t = 2.5), 0.30 at r = 7.7 (t = 9.6). The run never stops, but its flow is
wrong. Because 59% of the cells become troubled, order 4 costs 3.3 times
order 5 here.

**NACA 0012** ([#976](https://github.com/MatthewBonanni/mallard/issues/976)).
The case is the laminar tutorial mesh (28,672 quads; wall cells of aspect
ratio 10 to 30, closed trailing edge) at Mach 0.5, Re = 5,000 and 1 degree,
run to t = 2.5. The table gives the sign-alternating part of the wall
pressure's second difference, averaged over 9 faces, over the dynamic
pressure.

| | Order 4 | Order 5 |
|---|---|---|
| largest | 0.023, at x = 0.96 to 0.98 | 0.008, at the leading edge |
| largest for 0.1 < x < 0.9 | 0.0014 | 0.0002 |

The odd-even wall-pressure mode that the [mirror-image rules and the metric](../numerics/teno_e.md#candidates)
remove elsewhere remains at order 4 at the trailing edge.

**Taylor–Green vortex at Mach 0.1.** The kinetic energy agrees to 0.1%,
but order 4 leaves more compressible noise:

| Mesh | Mean squared dilatation, order 4 / 5 |
|---|---|
| 16³ hexahedra | 0.23 / 0.021 |
| 32³ hexahedra | 0.080 / 0.0040 |
| 12³ blocks of tetrahedra | 1.19 / 0.13 |

On the tetrahedra the enstrophy is 176 against 146. This is not a failure on
its own, but it fits the cylinder: order 4 damps grid-scale acoustic modes
less than order 5 does.

## Shocks: undershoots

The lowest density and pressure over every step (`check_interval = 1`):

| Case | Order 3 | Order 4 | Order 5 | Order 6 |
|---|---|---|---|---|
| Sod, quads: min ρ (exact 0.125) | 0.1246 | **0.1236** | 0.1249 | 0.1249 |
| Sod, triangles: min ρ | 0.1234 | 0.1211 | 0.1216 | 0.1206 |
| Shu–Osher, 400 × 4 quads: min p ahead of the shock (exact 1) | | **0.977** | 0.9997 | |

On quads, order 4's undershoot is about 10 times order 5's and comes and goes
as the shock crosses cells ([#977](https://github.com/MatthewBonanni/mallard/issues/977)).
On triangles every order undershoots by 2.6 to 3.5%. The 3D explosion shows
the same trend: the lowest pressure (ambient 0.1) is 0.0845 against 0.0884 on
tetrahedra and 0.0822 against 0.0857 on pyramids.

## What would change the default

When [#236](https://github.com/MatthewBonanni/mallard/issues/236),
[#975](https://github.com/MatthewBonanni/mallard/issues/975),
[#976](https://github.com/MatthewBonanni/mallard/issues/976) and
[#977](https://github.com/MatthewBonanni/mallard/issues/977) are fixed and
this page's cases pass at order 4, `high_order` can move to order 4. The
cylinder and the pyramid boxes are the deciding checks. Until then, order 4
is an explicit choice for smooth flow on meshes without pyramids:
`[numerics.face_reconstruction] order = 4` under `high_order`.
