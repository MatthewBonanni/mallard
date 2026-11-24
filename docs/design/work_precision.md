# Work-precision study

Status: the isentropic vortex measured on v0.10.0-217 (f685eb759), the
Shu-Osher problem on v0.10.0-48 (d57440e31). The cost of reaching a given error
with MUSCL and TENO-E orders 3 to 6, on CPU threads and on one GPU, for a
smooth problem with an exact solution (the isentropic vortex) on four cell
types and for a problem with a shock (Shu-Osher). Practical guidance from it is in
[Performance: choosing a scheme for an accuracy target](../guide/performance.md#accuracy-target).

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

**CPU, 2 threads of an AMD EPYC 7763:**

| Case | Error | MUSCL | TENO-E 3 | TENO-E 4 | TENO-E 5 | TENO-E 6 |
|---|---|---:|---:|---:|---:|---:|
| 2D squares | 10<sup>-3</sup> | 0.39 s | 0.72 s | **0.19 s** | 0.44 s | 0.33 s |
|  | 10<sup>-4</sup> | 7.1 s | 4.5 s | **0.77 s** | 1.8 s | 1.1 s |
|  | 10<sup>-5</sup> | 153 s | 38 s | **2.8 s** | 6.2 s | 3.5 s |
|  | 10<sup>-6</sup> | – | 345 s | 12.3 s | 22 s | **10.4 s** |
| 2D triangles | 10<sup>-3</sup> | 0.94 s | 0.87 s | **0.34 s** | 0.65 s | 0.58 s |
|  | 10<sup>-4</sup> | 40 s | 7.5 s | **1.8 s** | 3.2 s | 2.1 s |
|  | 10<sup>-5</sup> | – | 76 s | 8.7 s | 12.6 s | **6.5 s** |
|  | 10<sup>-6</sup> | – | 855 s | 46 s | 46 s | **19.3 s** |
| 3D hexahedra | 10<sup>-3</sup> | 9.9 s | 14.9 s | **6.0 s** | 12.5 s | 13.8 s |
|  | 10<sup>-4</sup> | 564 s | 232 s | **49 s** | 90 s | 61 s |
|  | 10<sup>-5</sup> | – | – | 362 s | 547 s | **301 s** |
| 3D tetrahedra | 10<sup>-3</sup> | 17.0 s | 15.6 s | **≤ 7.9 s** | 28 s | ≤ 33 s |
|  | 10<sup>-4</sup> | 868 s | 208 s | **32 s** | 132 s | 72 s |
|  | 10<sup>-5</sup> | – | – | **206 s** | 811 s | 215 s |
| Shu-Osher ($L_1$) | 3 × 10<sup>-2</sup> | 4.1 s | 4.6 s | **3.0 s** | 4.9 s | 7.0 s |
|  | 10<sup>-2</sup> | 22 s | 18.7 s | **9.3 s** | 17.1 s | 22 s |
|  | 3 × 10<sup>-3</sup> | 170 s | **142 s** | 151 s | 191 s | 309 s |

**One A100:**

| Case | Error | MUSCL | TENO-E 3 | TENO-E 4 | TENO-E 5 | TENO-E 6 |
|---|---|---:|---:|---:|---:|---:|
| 2D squares | 10<sup>-3</sup> | **0.29 s** | 0.33 s | 0.32 s | 0.34 s | 0.37 s |
|  | 10<sup>-4</sup> | 0.47 s | 0.50 s | **0.34 s** | 0.40 s | 0.39 s |
|  | 10<sup>-5</sup> | 2.4 s | 1.1 s | 0.45 s | 0.52 s | **0.45 s** |
|  | 10<sup>-6</sup> | 33 s | 4.3 s | 0.68 s | 0.76 s | **0.61 s** |
| 2D triangles | 10<sup>-3</sup> | **0.30 s** | 0.35 s | 0.31 s | 0.33 s | 0.33 s |
|  | 10<sup>-4</sup> | 0.84 s | 0.56 s | **0.38 s** | 0.43 s | 0.41 s |
|  | 10<sup>-5</sup> | 43 s | 1.6 s | **0.53 s** | 0.64 s | 0.55 s |
|  | 10<sup>-6</sup> | – | 8.5 s | 1.0 s | 1.2 s | **0.81 s** |
| 3D hexahedra | 10<sup>-3</sup> | 0.80 s | 0.86 s | **0.56 s** | 0.66 s | 0.73 s |
|  | 10<sup>-4</sup> | 14.8 s | 5.5 s | **1.4 s** | 1.7 s | 1.6 s |
|  | 10<sup>-5</sup> | – | – | 5.7 s | 6.0 s | **4.0 s** |
| 3D tetrahedra | 10<sup>-3</sup> | 0.99 s | 0.79 s | **≤ 0.60 s** | 0.96 s | ≤ 1.1 s |
|  | 10<sup>-4</sup> | 17.8 s | 3.9 s | **1.0 s** | 2.3 s | 1.7 s |
|  | 10<sup>-5</sup> | – | 44 s | 3.4 s | 9.6 s | **3.3 s** |

Measured order of accuracy between the two finest meshes of each sequence (CPU;
the A100 runs give the same errors to 2 × 10<sup>-7</sup> relative, and carry
most sequences one or two meshes further):

| Case | MUSCL | TENO-E 3 | TENO-E 4 | TENO-E 5 | TENO-E 6 |
|---|---:|---:|---:|---:|---:|
| 2D squares | 2.00 | 3.00 | 3.85 | 4.45 | 3.03 |
| 2D triangles | 1.42 | 2.99 | 3.84 | 4.97 | 5.77 |
| 3D hexahedra | 2.03 | 2.99 | 3.98 | 4.83 | 5.52 |
| 3D tetrahedra | 2.16 | 3.00 | 4.02 | 4.79 | 5.86 |
| Shu-Osher | 1.09 | 0.90 | 0.37 | 0.76 | 0.70 |

- **TENO-E 4 is the cheapest scheme** at 10<sup>-3</sup> and
  10<sup>-4</sup> on every cell type, on the CPU and on the A100 (apart from
  MUSCL at 10<sup>-3</sup> in 2D on the A100), and at 10<sup>-5</sup> on
  squares and tetrahedra on the CPU and on triangles on the A100. It needs
  about as many cells as TENO-E 5 for a given error, with a smaller stencil.
- **TENO-E 6 takes over at the smallest errors**: on the CPU at
  10<sup>-5</sup> on triangles and hexahedra (TENO-E 4 takes 1.3 and 1.2
  times as long) and at 10<sup>-6</sup> in 2D (1.2 times on squares, 2.4
  times on triangles); on the A100 from 10<sup>-5</sup> on squares and
  hexahedra and at 10<sup>-6</sup> on triangles. On tetrahedra TENO-E 4 and 6
  are within 5% of each other at 10<sup>-5</sup> on both.
- **TENO-E 5**, the order of the `high_order` preset, is never the cheapest:
  it costs 1.0 to 2.3 times TENO-E 4 on the CPU in 2D and on hexahedra (3.5
  to 4.1 times on tetrahedra) and 1.05 to 1.2 times on the A100 (1.6 to 2.8
  times on tetrahedra). Order 5 is the preset for robustness, not cost
  ([the order of `high_order`](teno4_robustness.md)).
- **MUSCL** is the cheapest only at 10<sup>-3</sup> in 2D on the A100, where
  every run takes 0.3 to 0.4 s whatever the scheme. At 10<sup>-4</sup> it
  costs 9 to 27 times TENO-E 4 on the CPU, and it does not reach
  10<sup>-5</sup> on triangles or in 3D within the budget; on the A100 it
  costs 48 times TENO-E 4 at 10<sup>-6</sup> on squares.
- **The A100** reaches 10<sup>-6</sup> on 2D squares with TENO-E 4 in 0.68 s,
  18 times faster than 2 CPU threads (12.3 s). Its cost hardly grows below
  about 30,000 cells, where a run is mostly fixed cost, and its errors match
  the CPU's to the digits printed. In 2D, 11 to 61% of a TENO-E run's wall
  time on the A100 is setup, which runs on the host
  ([#1031](https://github.com/MatthewBonanni/mallard/issues/1031)); on the
  CPU it is 3 to 41% on squares and 1 to 29% on triangles.
- **In 3D on the A100** the setup sets the cost: 38 to 78% of a TENO-E run
  on hexahedra and 30 to 67% on tetrahedra. MUSCL reaches 10<sup>-4</sup> on
  hexahedra in 14.8 s, 38 times faster than 2 CPU threads, but 85% of its
  finest run (1.77M cells) is setup; for TENO-E 3 the setup is 78% of the run
  on 221,184 cells and 74% on 1.77M. The host-side parts of the TENO-E setup
  on lattices are [#1079](https://github.com/MatthewBonanni/mallard/issues/1079)
  and [#1082](https://github.com/MatthewBonanni/mallard/issues/1082).
- **With a shock**, high order gains much less. TENO-E 4 reaches an $L_1$
  error of 10<sup>-2</sup> in 0.4 times MUSCL's time, but at 3 × 10<sup>-3</sup>
  every scheme is within a factor of 2.2 and TENO-E 3 is the cheapest:
  TENO-E 4 to 6 converge below first order on the finest meshes, behind
  the shock ([#1162](https://github.com/MatthewBonanni/mallard/issues/1162)).
  The reference is itself TENO-E 5, so errors near 10<sup>-3</sup> are only
  indicative.

**Time integration** limits the smallest errors: below about 10<sup>-7</sup>
on squares, TENO-E 5 and 6 fall toward third order (TENO-E 6: 3.73, then
3.03), the order of SSPRK3 at a fixed CFL number, and on triangles TENO-E 6
falls from 5.77 to 4.80 on the A100's next mesh (2.3 × 10<sup>-11</sup> on
524,288 cells). At `cfl = 0.5`, TENO-E 6 on 256 × 256 squares reaches
8.2 × 10<sup>-9</sup> instead of 4.2 × 10<sup>-8</sup>, and its order from
128 × 128 is 5.55 instead of 3.73. Smaller errors need a smaller `cfl`.

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
  strength $\beta = 5$, in a free stream of $\rho = p = 1$, $u = 1$, convected
  2 units, with a smooth cutoff at $R = 5$, half the box: $r^2$ in the exponent
  of Shu's vortex becomes $r^2 / (1 - r^2/R^2)$,

    $$f = \exp\left(\tfrac12\left(1 - \frac{r^2}{1 - r^2/R^2}\right)\right), \quad
    u_\theta = \frac{\beta}{2\pi} \frac{r f}{1 - r^2/R^2}, \quad
    T = \frac{p}{\rho} = 1 - \frac{(\gamma - 1)\beta^2}{8\gamma\pi^2} f^2,$$

    and $f = u_\theta = 0$ for $r \ge R$. It is still an exact isentropic
    vortex ($dp/dr = \rho u_\theta^2 / r$, $p = \rho^\gamma$), so convected
    it is the exact solution, and it vanishes with all its derivatives at
    $r = R$: the field is smooth across the periodic sides. Shu's vortex itself
    has a velocity of $2.4 \times 10^{-5}$ at the sides, so its periodic
    extension jumps by $4.9 \times 10^{-5}$ there, and the sound that jump sends
    out holds the error on irregular triangles at about $10^{-8}$
    ([#1175](https://github.com/MatthewBonanni/mallard/issues/1175)). The
    cutoff vortex peaks at the same radius, with a 2% higher swirl, and
    differs from Shu's by at most 10% of the peak swirl (at $r \approx 2.3$).
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
  run. Every vortex run on the CPU is the fastest of three, and the data
  keeps all three times (`tries`); other runs shorter than 10 s are the
  fastest of three.
- **CPU**: 2 threads (Kokkos `Threads`) of an AMD EPYC 7763 (2 × 64 cores),
  GCC 15, Release, double precision, in a container with dedicated CPUs on a
  node otherwise nearly idle. Each run is pinned to 2 cores of its own core
  complex (its own L3 cache), eight runs at a time for the vortex and six for
  the Shu-Osher problem. The fastest and slowest of a run's three times
  differ by a median of 0.6% and at most 5.4% for runs over 10 s.
- **GPU**: one NVIDIA A100-SXM4-80GB, Kokkos CUDA (CUDA 13.5,
  `Kokkos_ARCH_AMPERE80`) with an OpenMP host on 16 threads; the step runs as
  a CUDA graph (the default).
- **Meshes**: each sequence refines until the next mesh would take longer than
  33 min on the CPU (58 min for TENO-E 4 on triangles, TENO-E 4 and 5 on
  hexahedra and TENO-E 4 and 6 on tetrahedra) or 2 min on the GPU (5 min for
  MUSCL on 2D squares), the previous run's time scaled by its cells and time
  steps, so the finest mesh of a sequence is a matter of budget, not of
  accuracy.

Script: `tools/perf/work_precision.py` (`run`, `reference`, `collect`); data:
`tools/docs/work_precision.json`, one record per run with its run id, the
Mallard version and build it ran, its steps, wall time and parts, peak host
and device memory and its errors; figures and the tables above:
`tools/docs/work_precision.py --table`.
