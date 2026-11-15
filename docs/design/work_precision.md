# Work-precision study

Status: measured on v0.10.0-48 (d57440e31). The cost of reaching a given error
with MUSCL and TENO-E orders 3 to 6, on CPU threads and on one GPU, for a
smooth problem with an exact solution (the isentropic vortex) on four cell
types and for a problem with a shock (Shu-Osher). Practical guidance from it is in
[Performance: choosing a scheme for an accuracy target](../guide/performance.md#accuracy-target).

The A100 runs cover the 2D cases, and on 3D hexahedra MUSCL and TENO-E 3 only.
The rest ([#1171](https://github.com/MatthewBonanni/mallard/issues/1171))
was not run because the GPU was needed for other work.

## Results

![Error of the isentropic vortex on 2D squares against cells and wall time](../images/guide/work_precision_vortex_2d_quad-light.svg#gh-light-mode-only)
![Error of the isentropic vortex on 2D squares against cells and wall time](../images/guide/work_precision_vortex_2d_quad-dark.svg#gh-dark-mode-only)

![Error of the isentropic vortex on 2D irregular triangles against cells and wall time](../images/guide/work_precision_vortex_2d_tri-light.svg#gh-light-mode-only)
![Error of the isentropic vortex on 2D irregular triangles against cells and wall time](../images/guide/work_precision_vortex_2d_tri-dark.svg#gh-dark-mode-only)

![Error of the isentropic vortex on 3D hexahedra against cells and wall time](../images/guide/work_precision_vortex_3d_hex-light.svg#gh-light-mode-only)
![Error of the isentropic vortex on 3D hexahedra against cells and wall time](../images/guide/work_precision_vortex_3d_hex-dark.svg#gh-dark-mode-only)

![Error of the isentropic vortex on 3D tetrahedra against cells and wall time](../images/guide/work_precision_vortex_3d_tet-light.svg#gh-light-mode-only)
![Error of the isentropic vortex on 3D tetrahedra against cells and wall time](../images/guide/work_precision_vortex_3d_tet-dark.svg#gh-dark-mode-only)

![Error of the Shu-Osher problem against cells and wall time](../images/guide/work_precision_shu_osher-light.svg#gh-light-mode-only)
![Error of the Shu-Osher problem against cells and wall time](../images/guide/work_precision_shu_osher-dark.svg#gh-dark-mode-only)

Wall time to reach each error, interpolated in log-log between the meshes
around it; the cheapest scheme is in bold. "≤" means the coarsest mesh is
already below the error, and "–" that the finest mesh did not reach it.

**CPU, 2 threads:**

| Case | Error | MUSCL | TENO-E 3 | TENO-E 4 | TENO-E 5 | TENO-E 6 |
|---|---|---:|---:|---:|---:|---:|
| 2D squares | 10<sup>-3</sup> | 0.29 s | 0.65 s | **0.21 s** | 0.49 s | 0.45 s |
| | 10<sup>-4</sup> | 5.3 s | 4.0 s | **0.83 s** | 1.8 s | 1.4 s |
| | 10<sup>-5</sup> | 138 s | 34 s | **3.1 s** | 6.2 s | 4.1 s |
| | 10<sup>-6</sup> | – | – | 13.3 s | 21 s | **11.4 s** |
| 2D triangles | 10<sup>-3</sup> | 0.79 s | 0.66 s | **0.27 s** | 0.52 s | 0.53 s |
| | 10<sup>-4</sup> | 36 s | 5.6 s | **1.5 s** | 2.7 s | 1.9 s |
| | 10<sup>-5</sup> | – | 56 s | 7.1 s | 10.6 s | **5.7 s** |
| | 10<sup>-6</sup> | – | – | 38 s | 39 s | **17.5 s** |
| 3D hexahedra | 10<sup>-3</sup> | 7.0 s | 12.3 s | **5.5 s** | 10.8 s | 13.1 s |
| | 10<sup>-4</sup> | 524 s | 168 s | **38 s** | 67 s | 47 s |
| | 10<sup>-5</sup> | – | – | 248 s | 385 s | **223 s** |
| 3D tetrahedra | 10<sup>-3</sup> | 14 s | 12 s | **≤ 8.8 s** | 33 s | ≤ 93 s |
| | 10<sup>-4</sup> | – | 151 s | **28 s** | 136 s | 155 s |
| | 10<sup>-5</sup> | – | – | **162 s** | – | 405 s |
| Shu-Osher ($L_1$) | 3 × 10<sup>-2</sup> | 4.1 s | 4.7 s | **3.0 s** | 4.9 s | 7.0 s |
| | 10<sup>-2</sup> | 22 s | 19 s | **9.3 s** | 17 s | 22 s |
| | 3 × 10<sup>-3</sup> | 170 s | **142 s** | 151 s | 191 s | 309 s |

**One A100:**

| Case | Error | MUSCL | TENO-E 3 | TENO-E 4 | TENO-E 5 | TENO-E 6 |
|---|---|---:|---:|---:|---:|---:|
| 2D squares | 10<sup>-3</sup> | **0.17 s** | 0.25 s | 0.21 s | 0.26 s | 0.34 s |
| | 10<sup>-4</sup> | 0.30 s | 0.45 s | **0.27 s** | 0.36 s | 0.43 s |
| | 10<sup>-5</sup> | 1.8 s | 1.4 s | **0.42 s** | 0.54 s | 0.60 s |
| | 10<sup>-6</sup> | 29 s | 6.0 s | **0.85 s** | 0.98 s | 0.91 s |
| 2D triangles | 10<sup>-3</sup> | **0.20 s** | 0.24 s | 0.21 s | 0.25 s | 0.28 s |
| | 10<sup>-4</sup> | 0.73 s | 0.51 s | **0.32 s** | 0.40 s | 0.35 s |
| | 10<sup>-5</sup> | 42 s | 1.5 s | **0.59 s** | 0.77 s | 0.66 s |
| | 10<sup>-6</sup> | – | 7.4 s | **1.1 s** | 1.3 s | 1.2 s |
| 3D hexahedra | 10<sup>-3</sup> | 0.54 s | 2.2 s | | | |
| | 10<sup>-4</sup> | 11.8 s | – | | | |

Measured order of accuracy between the two finest meshes of each sequence (CPU;
the A100 runs give the same errors):

| Case | MUSCL | TENO-E 3 | TENO-E 4 | TENO-E 5 | TENO-E 6 |
|---|---:|---:|---:|---:|---:|
| 2D squares | 2.00 | 3.00 | 3.91 | 4.36 | 2.98 |
| 2D triangles | 1.23 | 3.00 | 3.73 | 4.70 | 1.14 |
| 3D hexahedra | 2.03 | 2.96 | 4.02 | 4.72 | 5.34 |
| 3D tetrahedra | 2.27 | 2.98 | 4.08 | 4.55 | 5.64 |
| Shu-Osher | 1.09 | 0.90 | 0.37 | 0.76 | 0.70 |

- **TENO-E 4 is the cheapest scheme** from 10<sup>-3</sup> to 10<sup>-5</sup>
  on every cell type on the CPU, and from 10<sup>-4</sup> to 10<sup>-6</sup>
  on the A100. It needs about as many cells as TENO-E 5 for a given error,
  with a smaller stencil.
- **TENO-E 6 takes over at the smallest errors** on the CPU: at 10<sup>-6</sup>
  in 2D, and at 10<sup>-5</sup> on triangles and hexahedra (on hexahedra by
  10%). On tetrahedra its setup keeps it 2.5 times dearer than TENO-E 4 at
  10<sup>-5</sup>.
- **TENO-E 5**, the order of the `high_order` preset, is never the cheapest:
  it costs 1.0 to 2.3 times TENO-E 4 on the CPU (4 to 5 times on
  tetrahedra) and 1.15 to 1.3 times on the A100. Order 5 is the preset
  for robustness, not cost ([the order of `high_order`](teno4_robustness.md)).
- **MUSCL** is the cheapest only at 10<sup>-3</sup> on the A100, where every
  2D run takes 0.15 to 0.35 s whatever the scheme. At 10<sup>-4</sup> it costs
  6 to 24 times TENO-E 4 on the CPU, and it does not reach 10<sup>-5</sup> on
  triangles or in 3D within 20 min; on the A100 it costs 34 times TENO-E 4 at
  10<sup>-6</sup> on squares.
- **The A100** reaches 10<sup>-6</sup> on 2D squares with TENO-E 4 in 0.85 s,
  16 times faster than 2 CPU threads (13.3 s). Its cost hardly grows below
  about 30,000 cells, where a run is mostly fixed cost, and its errors match
  the CPU's to the digits printed. In 2D, 45 to 73% of a TENO-E run's wall
  time on the A100 is setup, which runs on the host
  ([#1031](https://github.com/MatthewBonanni/mallard/issues/1031)); on the
  CPU it is 25 to 45% on squares and 5 to 16% on triangles.
- **In 3D on the A100** the setup sets the cost. MUSCL reaches 10<sup>-4</sup>
  on hexahedra in 11.8 s, 44 times faster than 2 CPU threads, but 72% of its
  finest run (1.77M cells) is setup. For TENO-E 3 the setup is 91% of the run
  on 221,184 cells: 14.0 s, against 0.6 s of time steps. The host-side parts
  of the TENO-E setup on lattices are
  [#1079](https://github.com/MatthewBonanni/mallard/issues/1079) and
  [#1082](https://github.com/MatthewBonanni/mallard/issues/1082).
- **With a shock**, high order gains much less. TENO-E 4 reaches an $L_1$
  error of 10<sup>-2</sup> in 0.4 times MUSCL's time, but at 3 × 10<sup>-3</sup>
  every scheme is within a factor of 2.2 and TENO-E 3 is the cheapest:
  TENO-E 4 to 6 converge below first order on the finest meshes, behind
  the shock ([#1162](https://github.com/MatthewBonanni/mallard/issues/1162)).
  The reference is itself TENO-E 5, so errors near 10<sup>-3</sup> are only
  indicative.

Two floors limit the smallest errors:

- **Time integration:** below about 10<sup>-6</sup> on squares, TENO-E 5 and 6
  fall toward third order (TENO-E 6: 3.60, then 2.98), the order of SSPRK3 at a
  fixed CFL number. At `cfl = 0.5`, TENO-E 6 on 256 × 256 squares reaches
  6.7 × 10<sup>-9</sup> instead of 3.4 × 10<sup>-8</sup>, and its order from
  128 × 128 is 5.31 instead of 3.60. Smaller errors need a smaller `cfl`.
- **Triangles:** every high order stalls at about 10<sup>-8</sup> (TENO-E 6:
  1.14 between its two finest CPU meshes, 0.76 on the A100's next one), while
  on squares the errors go on down to 6.5 × 10<sup>-10</sup>. The cause is
  not isolated ([#1172](https://github.com/MatthewBonanni/mallard/issues/1175)).

## Method

Every case is the Euler equations with SSPRK3 at `cfl = 1.0` (the `high_order`
preset's integrator and CFL number, and the default for MUSCL), run on a
sequence of meshes from coarse to fine with each scheme. Every TENO-E run uses
the defaults of `[numerics.face_reconstruction]` apart from `order`.

| Case | Mesh | Flux, MUSCL limiter | Time | Error |
|---|---|---|---|---|
| `vortex_2d_quad` | n × n squares of a periodic 10 × 10 box | HLLC, none | t = 2 | $L_2$ of $\rho$ |
| `vortex_2d_tri` | n × n squares, interior nodes moved by up to h/4, each split along its shorter diagonal ([`high_order_cost.py`](../guide/numerics.md#why-high-order)) | HLLC, none | t = 2 | $L_2$ of $\rho$ |
| `vortex_3d_hex` | n × n × max(8, n/4) cubes, 10 × 10 across, periodic | HLLC, none | t = 2 | $L_2$ of $\rho$ |
| `vortex_3d_tet` | the same cubes, six tetrahedra each (`cartesian_tet`) | HLLC, none | t = 2 | $L_2$ of $\rho$ |
| `shu_osher` | n × 4 squares of [0, 10] × [0, 40/n], symmetry walls | RHLL, Venkatakrishnan | t = 1.8 | $L_1$ of the column averages of $\rho$ |

- **The vortex** is the isentropic vortex of [Shu (1998)](../references.md#shu-1998),
  strength 5, in a free stream of $\rho = p = 1$, $u = 1$, convected 2 units.
  In 3D it is a tube along z, so the exact solution does not vary along z;
  the box keeps cubic cells and at least 8 of them along z, so that stencils
  of every order fit without wrapping onto themselves. The error is
  `mallard post convergence --field RHO --exact ...` on the run's final restart
  file: the volume-weighted $L_2$ norm against the exact cell averages
  (quadrature exact to degree 6).
- **The Shu-Osher problem** ([Shu & Osher 1989](../references.md#shu-osher-1989)),
  as the [`shu_osher`](https://matthewbonanni.github.io/mallard/docs/examples/#shu-osher)
  example: a Mach 3 shock running into an entropy wave, with RHLL because HLLC
  has a shock instability on this strip. The strip keeps square cells (its
  height shrinks with the cells), so each run's density, averaged over its 4
  cells in y, is a function of x alone. Its reference is TENO-E 5 on
  51,200 × 4 cells; the error of a run is the mean absolute difference of its
  column averages from the reference averaged over the same columns (exact,
  since every n divides 51,200).
- **Cost** is the run's wall time as its run record's `end` record gives it:
  setup (mesh, TENO-E stencils and tables), time stepping, diagnostics and the
  final snapshot. It leaves out the start of the process before the solver
  (about 0.5 s on the GPU, creating the CUDA context; milliseconds on the CPU).
  Every record of the data keeps its parts (`wall_setup`, `wall_stepping`) and
  the peak host and device memory. Progress rows every 50 steps (200 for
  Shu-Osher), the run report and the run health are on, as in a production
  run. Runs shorter than 10 s are the fastest of three.
- **CPU**: 2 threads (Kokkos `Threads`) of an AMD EPYC 7763, each run pinned
  to its own 2 cores, six runs at a time on otherwise idle cores. GCC 15,
  Release, double precision.
- **GPU**: one NVIDIA A100-SXM4-80GB, Kokkos CUDA (CUDA 13.5,
  `Kokkos_ARCH_AMPERE80`) with an OpenMP host on 16 threads; the step runs as
  a CUDA graph (the default).
- **Meshes**: each sequence refines until the next mesh would take longer than
  20 min on the CPU or 2 min on the GPU (5 min for MUSCL on 2D squares; the
  previous run's time scaled by its cells and time steps), so the finest mesh
  of a sequence is a matter of budget, not of accuracy.

Script: `tools/perf/work_precision.py` (`run`, `reference`, `collect`); data:
`tools/docs/work_precision.json`, one record per run with its run id, the
Mallard version and build it ran, its steps, wall time and parts, peak host
and device memory and its errors; figures and the tables above:
`tools/docs/work_precision.py --table`.
