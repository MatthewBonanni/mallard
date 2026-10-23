# Performance

Throughput of Mallard on the 2D Riemann problem (configuration 3, the [`riemann_2d_quads`](https://matthewbonanni.github.io/mallard/docs/examples/) example) on 1000 × 1000 quadrilaterals (the example's input with `Nx = Ny = 1000`, `n_steps = 300` in place of `t_stop`, and no `[[write_data]]`): Euler equations, HLLC flux, SSPRK3, double precision. The time per step and cell is the solver's own report over its last 100 steps (best of two runs on the CPU); it excludes setup (mesh, TENO stencil precomputation) and counts one SSPRK3 step, which evaluates the right-hand side three times.

| Reconstruction | CPU, 6 threads: ns per step and cell (0.3.0) | A100: ns per step and cell (0.4.0) | A100: million cell-steps per second | A100 / CPU |
|---|---:|---:|---:|---:|
| MUSCL (Venkatakrishnan) | 91 | 4.3 | 230 | 21× |
| TENO-E, order 3 | 125 | 7.2 | 139 | 17× |
| TENO-E, order 5 | 593 | 19.4 | 51 | 31× |

CPU: 6 of the 14 cores of an Apple M4 Pro, Kokkos `Threads` backend. GPU: one NVIDIA A100, Kokkos CUDA backend (`Kokkos_ARCH_AMPERE80`) with a serial host backend, which slows the setup but not the time steps. Both are Release builds; the GPU runs use Mallard 0.4.0, the CPU runs Mallard 0.3.0 (the CPU was too heavily shared to repeat them for 0.4.0). The machines were shared with other work, the CPU heavily (repeated CPU runs varied by up to 15%), so these numbers are indicative, not best case. On the same A100, TENO-E 5 took 21.7 ns per step and cell with Mallard 0.2.0 and 20.1 with 0.3.0 (MUSCL 4.3 and TENO-E 3 7.3 with 0.3.0); the CPU runs with 0.2.0, on a less loaded machine, 553.

At these rates the A100 advances the million-cell TENO-E 5 case by one time step in about 19 ms.

Fifth-order TENO-E costs 4.7 times as much per step as third order on the CPU and 2.7 times on the A100; its central stencil has 28 neighbors of each cell, against 10 for third order.

## 3D TENO-E: setup and memory {#teno-3d}

**Setup on the GPU** ([#247](https://github.com/MatthewBonanni/mallard/pull/247)). The 3D TENO-E setup — stencil search, moments, least-squares pseudo-inverses, Lebesgue bounds, smoothness indicators — runs one cell per team on the GPU. Its tables are bitwise identical to the host's. That makes starts without a stencil cache, restarts on another rank count and re-setup after repartitioning cheap. Fifth order on one A100 (host: 16 threads of an AMD EPYC 7763):

| Setup, TENO5 | 100³ hexahedra (1.0M) | 55³ × 6 tetrahedra (998k) |
|---|---:|---:|
| host (before 0.8.0) | 43.5 s | 123.2 s |
| GPU (0.8.0) | 4.56 s (9.5×) | 20.0 s (6.2×) |

On tetrahedra the least-squares fits are latency-bound under the bitwise constraint, hence the smaller speedup; the 2D setup still runs on the host.

**Device memory** ([#265](https://github.com/MatthewBonanni/mallard/pull/265)). Cells keep their central moments instead of their smoothness-indicator matrices, the troubled-cell passes work in bounded rounds, and the setup's peak is no higher than the steady state. Results are bitwise unchanged. Bytes per cell on an A100-80GB, everything included:

| | before 0.8.0 (setup peak / steady) | 0.8.0 | 0.8.0, `single_precision_tables` |
|---|---:|---:|---:|
| TENO5, hexahedra | 47.3 / 47.3 KB | 39.8 KB | 25.0 KB |
| TENO5, tetrahedra | 58.7 / 58.7 KB | 51.2 KB | 30.6 KB |
| TENO6, hexahedra | 129.6 / 99.6 KB | 77.9 KB | 46.5 KB |

- **Capacity:** one A100-80GB now holds about 2.1M hexahedra or 1.65M tetrahedra at TENO5 (before: about 1.8M / 1.45M), and 3.4M / 2.75M with `single_precision_tables`. A 125³ Taylor–Green vortex that ran out of memory now runs.
- **`single_precision_tables`** (opt-in, 3D, single gas) stores the pseudo-inverses in single precision and is another 3–8% faster. It changes results at single-precision round-off: below 10<sup>−7</sup> of each field's range on the Taylor–Green vortex, but up to 6.9 × 10<sup>−3</sup> in shock cells of a blast wave.

## Chemistry {#chemistry}

Mallard runs the stiff chemistry of reacting flows on GPUs with one thread per cell for small mechanisms. Larger ones use one warp per cell, or a team of up to 16 warps for the few cells that need many sub-steps, and a sparse LU factorization from 30 species up. The benchmark (`benchmarks/chemistry/` in the repository) advances states sampled along a constant-volume ignition (fresh, igniting and burnt gas, replicated over many cells) by one splitting step, as the solver does in every cell. Cells per second, the best of several calls after a warm-up, in double precision:

| Mechanism | Species | A100: dt = 10<sup>−8</sup> s | A100: dt = 10<sup>−6</sup> s | CPU, 16 cores: dt = 10<sup>−8</sup> s (0.5.0) | CPU, 16 cores: dt = 10<sup>−6</sup> s (0.5.0) |
|---|---:|---:|---:|---:|---:|
| H<sub>2</sub>/O<sub>2</sub> (`h2o2`) | 10 | 9.7M | 5.4M | 952k | 438k |
| GRI-Mech 3.0 | 53 | 535k | 541k | 79.7k | 77.5k |
| n-dodecane (Wang et al. 2014) | 100 | 307k | 60.2k | 60.7k | 24.7k |
| n-hexane (NUIG) | 1268 | 6.28k | 1.28k | 2.93k | 877 |

- **A100 columns:** the 0.8.0 development code. h2o2 is from [#264](https://github.com/MatthewBonanni/mallard/pull/264), whose thread-per-cell work memory on the stack took it from 6.5 to 9.7M cells/s at 10<sup>−8</sup> s. The others are from [#246](https://github.com/MatthewBonanni/mallard/pull/246), whose staged sparse LU took n-hexane at 10<sup>−6</sup> s from 576 to 1278 cells/s and n-dodecane from 48.0k to 60.2k.
- **CPU columns:** Mallard 0.5.0, on 16 cores of an AMD EPYC 7763 with OpenMP and one thread per cell, on a shared node with about 10% run-to-run variation.
- **Time-step sizes:** steps of 10<sup>−8</sup> s are typical of detonations, where every cell takes one sub-step. At 10<sup>−6</sup> s, typical of flames, the igniting cells of the large mechanisms take up to about 100 sub-steps and set the time.

### Reacting mixtures in the flow solver {#mixtures}

The performance case `flame_3d` is a propane–air flame with the two-step mechanism: 221,000 hexahedra, mixture-averaged transport, the Sigma model and TFLES, 72% of the cells reacting. Since [#257](https://github.com/MatthewBonanni/mallard/pull/257) and [#264](https://github.com/MatthewBonanni/mallard/pull/264), stencil kernels loop over the stencil once with the variables innermost, and the chemistry keeps its work memory on the thread's stack; results are bitwise unchanged. Time per step on one A100:

| `flame_3d`, ms per step | before | 0.8.0 | 0.8.0, `chemistry.C_reg = 1e-10` |
|---|---:|---:|---:|
| step | 64.9 | 32.7 | 24.6 |
| chemistry | 27.6 | 18.5 | 10.4 |
| mixture stencil kernels | 30.2 | 6.5 | 6.5 |
| the rest | 7.1 | 7.0 | 7.0 |
| cost per cell and step over the same flow of air | 15.5× | 7.8× | 5.9× |

The opt-in `C_reg = 1e-10` cheapens trace fuel burning out in hot gas. After 100 steps the temperature is within 0.54 K and the integrated heat release within 0.08% of the default.

### Chemistry load balancing {#chemistry-load-balancing}

`[chemistry] load_balance = true` ([#255](https://github.com/MatthewBonanni/mallard/pull/255)) moves the integration of cells from ranks with more chemistry work than the mean to ranks with less, in every call; results stay bitwise the same. Time per step on A100s, with the share of the step spent waiting for the slowest rank's chemistry:

| Case | Cells | GPUs | Without | With | Speedup |
|---|---:|---:|---:|---:|---:|
| `detonation_2d` | 1.2M | 4 | 86.3 ms (35.9%) | 69.4 ms (6.0%) | 1.24 |
| | | 8 | 56.5 ms (51.6%) | 35.1 ms (8.0%) | 1.61 |
| `flame_turbulence` DNS | 3.67M | 4 | 384 ms (17.8%) | 349 ms (3.5%) | 1.10 |
| | | 8 | 191 ms (18.7%) | 165 ms (0.3%) | 1.16 |
| `detonation_3d` | 19.2M | 4 | 1.27 s (8.2%) | 1.22 s (1.1%) | 1.04 |
| | | 8 | 631 ms (7.1%) | 596 ms (0.4%) | 1.06 |

For runs on several GPUs, see [multi-GPU scaling](parallel.md#multi-gpu-scaling). The [performance dashboard](https://matthewbonanni.github.io/mallard/docs/dashboard/) follows the GPU benchmarks from merge to merge, with their baselines.
