# Design: single-GPU performance and parallel scaling

Status: in progress. Baseline measured on `main` at v0.5.0-92 (before any change here); each change below
lists its measured effect. Results stay bitwise identical to `main` (checked on the cases under
*Verification*), and bitwise independent of the rank count, thread count and restarts.

## Hardware and method

- NVIDIA A100-SXM4-80GB: 1.94 TB/s HBM2e, 9.7 TFLOP/s FP64 (19.5 with tensor cores, unused), 108 SMs,
  64k 32-bit registers and 164 KB shared memory per SM. CUDA 12.9 (nvcc 12.9), Kokkos 5.2, OpenMP host.
- Times are per step (SSPRK3: three stages) over the time stepping of the performance set
  (`benchmarks/perf`), kernel times from Nsight Systems (`nsys profile -t cuda`). Hardware counters
  (Nsight Compute) were not available, so register and local-memory (spill) figures come from the
  compiled kernels (`cuobjdump -res-usage`), and bandwidths from the bytes a kernel must move.

## Baseline (main)

### 2D TENO5, 1024 x 1024 quadrilaterals, HLLC (`teno5_2d`)

19.5 ms per step, 35 kernel launches per step. Per stage:

| kernel | ms | share | notes |
|---|---|---|---|
| `teno_smooth` | 5.17 | 78% | 242 registers, 32 B stack; ~4 KB/cell (pseudo-inverse 29 x 14 doubles, stencil indices, face states): 0.8 TB/s, 42% of HBM peak |
| `convective_flux` (HLLC) | 0.62 | 9% | 264 B/face: 0.9 TB/s |
| `teno_troubled_select` | 0.30 | 5% | 1.2-1.4% of cells troubled |
| `teno_troubled_sectors` | 0.094 | 1.4% | |
| `face_flux_sum` | 0.096 | 1.4% | |
| `rhs_W`, `rhs_divide_volume`, `axpby` (per stage) | 0.18 | 3% | |
| `time_step` (reduction, per step) | 0.27 | 1.4% | one host synchronization per step |

### 3D TENO5, $96^3$ hexahedra, HLLC + viscous (`teno5_3d`)

335 ms per step. `teno_smooth` takes 106 ms per stage, 95% of the step: 255 registers and a 1136 B
stack. Its 34 x 5 central coefficients (`aK[NK][N_CONSERVATIVE]`) do not fit in registers, so every
multiply-add of the 69-entry stencil loop goes through local memory. It moves its 16.6 GB of
pseudo-inverses at 0.16 TB/s, 8% of HBM peak. Then come the HLLC flux (2.9 ms), the viscous vertex
gradients (1.2 ms) and fluxes (0.9 ms).

### Small meshes (strong-scaling regime)

2D TENO5 on one GPU, wall time per step against the GPU's busy time per step (sum of kernel times):

| cells | wall ms/step | kernels ms/step | launches/step |
|---|---|---|---|
| 4,096 | 1.19 | 0.97 | 33 |
| 16,384 | 1.33 | 1.16 | 33 |
| 65,536 | 2.36 | 2.19 | 34 |
| 262,144 | 5.90 | 5.73 | 34 |
| 1,048,576 | 19.8 | 19.5 | 35 |

- Host overhead (launches, the time-step reduction's synchronization) is 0.2 ms per step on one GPU.
- Below ~100k cells per GPU the kernels themselves stop shrinking: at 4,096 cells `teno_smooth` still
  takes 150 us per stage and the four troubled-cell passes 124 us. One thread reconstructs one cell
  (or one troubled sector, face, ...), so a kernel lasts at least one thread's dependent chain of loads
  and multiply-adds, however few cells there are. This latency floor, not launch overhead, is what
  limits strong scaling of TENO at small sizes.

3D TENO5 (Taylor-Green, symmetry walls) on one GPU, against cells (#304; time per step, CUDA graph, and the
smooth pass per cell and stage from Nsight Systems):

| cells | hexahedra ms/step | smooth pass ns/cell | tetrahedra ms/step | MUSCL hexahedra ms/step |
|---|---|---|---|---|
| 4,096 | 1.83 | 102 | 2.04 (4,374) | 0.367 |
| 32,768 | 6.41 | 55.5 | 7.50 (34,992) | 0.685 |
| 64,000 | 12.2 | 54.5 | 13.0 (63,888) | 1.29 |
| 110,592 | 20.6 | 53.7 | 24.9 (131,712) | 1.92 |
| 262,144 | 46.8 | 51.9 | 46.9 (257,250) | 4.12 |
| 1,000,000 | 167 | 48.7 | 173 (998,250) | 14.7 |

- The smooth pass costs within 14% of its 1M-cell rate per cell down to 32k cells (smaller boxes also have more
  cells with mirrored stencil entries); below that it hits the latency floor (about 420 us per stage).
- Timing each team with `clock64` (one A100): its stencil loop takes 270-300k cycles for 69 entries, about 4,000
  cycles per entry, and its face points 150-200k; the same in every chunk, so the loop waits on loads (stencil cell,
  then its state). Three teams per SM (128 registers), unrolling the loop by 2 or 4, and software-pipelining the
  loads by hand were all 10-50% slower, from spills.
- 4 GPUs, $64^3$ (65,536 cells per GPU, 88.6% strong efficiency against one GPU): per stage the GPU spends 3.70 ms
  reconstructing (owned and 4,160 layer-1 cells, 6.3% of them redundant) against 3.40 ms for a quarter of the
  one-GPU stage, and the rest on fluxes and gradients that also cover halo cells. The overlap already fills the
  first reconstruction's tail with the halo-dependent cells' kernel, and NCCL with graphs and MPI without are within
  2% (6.35 / 6.44 ms per step at $48^3$).
- The viscous gradients covered every halo layer (31,808 halo cells per rank at $64^3$ on 4 GPUs, 7 layers); the
  fluxes read owned and layer-1 cells only, so only those are computed now (LES models still read all). Bitwise
  identical; within noise on 4 GPUs ($48^3$: 6.43 / 6.40 ms per step), more where halos are a larger share.

## Changes

### 1. Cooperative 3D TENO smooth pass

`teno_smooth` in 3D (degree ≥ 4, i.e. TENO5 and TENO6) runs as a team kernel: 32 cells x
`N_CONSERVATIVE` threads. Over the stencil each thread accumulates a chunk of 7 of the 34 coefficients
for all variables (35 registers' worth instead of 170); the coefficients then go through shared memory,
and the cell's face points are shared out among its five threads. On CPUs and in 2D (where the
per-thread version does not spill and was measured faster at every size from 4k to 1M cells) the
per-thread kernel stays.

Bitwise identity needs care: the compiler fuses multiplies and adds into FMAs differently when a loop
index becomes a runtime value, which changes the last bits. Chunk bounds are therefore compile-time
constants (one instantiation per chunk), so every thread runs the per-thread kernel's arithmetic for
its coefficients.

| case (A100) | main | change 1 |
|---|---|---|
| `teno_smooth`, 96³ hexahedra, first stage | 111.8 ms | 44.9 ms |
| TENO5 Taylor-Green 96³ (`teno5_3d`) | 335 ms/step (2.64M cells/s) | 151 ms/step (5.85M cells/s) |
| TENO6 Taylor-Green 32³ | 47.3 ms/step | 20.7 ms/step |
| TENO5 explosion, 20³ x 6 tetrahedra | 17.5 ms/step | 11.0 ms/step |
| TENO3/TENO4 (degrees 2-3) | unchanged (per-thread kernel kept) | |

### 2. TENO setup: stencil tables streamed to the device (host memory)

The packed stencil tables (pseudo-inverses, stencil cells and faces) can only be sized on the device once every
stencil is known, so the setup kept every packed chunk on the host until the end: 27 KB per cell for 3D TENO5.
That host peak, more than device memory, set how many GPUs a large 3D run needs (#58). On CUDA each packed chunk
now goes straight to the device, into a reserved address range whose pages are mapped as it fills (CUDA virtual
memory management), so the host holds one chunk of 8192 cells at a time. The device arrays keep their exact sizes
and single base addresses; results and step times are unchanged.

| 3D TENO5, 64³ hexahedra, one A100 | main | change 2 |
|---|---|---|
| host peak during setup | 8.55 GB (32.6 KB/cell) | 1.44 GB (5.5 KB/cell) |
| device peak | 11.6 GB | 11.6 GB |

### 3. MUSCL: per-cell reconstruction, fused, overlapped with the halo exchange

MUSCL reconstructed in three kernels (least-squares gradients and limiters over cells, then face states over
faces), after the halo exchange. It now also reconstructs per cell: gradient, limiter, then the cell's own side of
each of its faces, with the same arithmetic. From $2^{17}$ cells (per rank) this runs as one fused kernel, which keeps
gradients and limiters out of memory, and in distributed runs the cells whose stencils hold no halo cell are
reconstructed while the exchange is in flight, as TENO's are. Below $2^{17}$ cells the separate kernels stay: with
fewer cells, more and shorter threads were faster.

| `muscl_2d` (Riemann problem, HLLC), A100s | main | change 3 |
|---|---|---|
| 4M cells, 1 GPU | 15.6 ms/step | 13.6 ms/step |
| 4M cells, 2 GPUs | 8.14 ms/step | 7.22 ms/step |
| 1M cells, 1 GPU (perf case) | 4.09 ms/step | 3.61 ms/step |
| 1M cells, 2 GPUs | 2.30 ms/step | 2.06 ms/step |
| 262k / 65k cells, 1 GPU | 1.21 ms / 475 us | 1.06 ms / 479 us |

### 4. TENO setup on the device

The 3D TENO setup took minutes per million cells on the host (#140): every start without a cache, every
restart on a different rank count or partition, paid it again, and it bounded the problem size a node
can set up in reasonable time. Moving it to the device makes it seconds.

**Baseline** (`main` at v0.6.0-80, `numerics` setup phase, A100 node host: EPYC 7763, 16 OpenMP threads,
symmetry walls on all sides). Per-cell work in thread-seconds (sum over threads):

| 3D TENO5 | 100³ hexahedra (1M) | 55³ x 6 tetrahedra (998k) |
|---|---|---|
| setup | **43.5 s** | **123.2 s** |
| moments (wall) | 7.5 s | 1.3 s |
| per-cell work (wall) | 26.8 s | 109 s |
| packing and upload (wall) | 9.1 s | 12.5 s |
| search: layers, mirror images, sort (thread-s) | 75 | 89 |
| means and QR (thread-s) | 178 | 1250 |
| Lebesgue constants (thread-s) | 58 | 282 |
| sector stencils (thread-s) | 70 | 40 |
| smoothness indicators, face basis, metric (thread-s) | 23 | 19 |
| least-squares fits per cell | 1.00 | 7.07 |
| host / device peak | 3.4 / 42.1 GB | 2.9 / 50.0 GB |

The time is the fits on tetrahedra (7 tried central stencils per cell, each a full QR and Lebesgue
constant), and on hexahedra the search, the fits and the packing alike.

**Change.** The per-cell setup (`teno_setup.cpp`) runs as a Kokkos team kernel with one cell per team
of 128 threads; cells whose search outgrows the device scratch are redone on the host with the same
code. Only independent sums run in parallel (columns of the QR, rows of the back substitution,
quadrature points of the Lebesgue constant, entries of the search); every value takes the host's
sequence of operations, and the file is compiled without FMA contraction, so the tables are bitwise
those of the host setup. Hot data (the visit hash set, the sort, the least-squares matrix and, for up
to 91 rows, its pseudo-inverse) is in shared memory; three teams fit an SM. Monomial means are unrolled
per degree with their operands in registers. Moments, bounding boxes and packing run on the device;
cube roots and the ranking metric (log, exp) stay on the host.

| 3D TENO5, one A100 | 100³ hexahedra | 55³ x 6 tetrahedra |
|---|---|---|
| setup | **4.56 s (9.5x)** | **20.0 s (6.2x)** |
| geometry (host metric and cube roots, device moments and boxes) | 0.50 s | 0.31 s |
| tables (device) | 3.95 s (4.0 us/cell) | 19.5 s (19.5 us/cell) |
| packing | 0.09 s | 0.12 s |
| host / device peak | 3.2 / 44.1 GB | 1.8 / 51.5 GB |

The device peak grows by the batch outputs and the per-team scratch (2 GB at 65,536 cells per batch).
Per cell on hexahedra, the device time splits into the fit (30%), the search and sort (35%), the
sector stencils (15%) and the rest (means, Lebesgue constant, smoothness indicators). On tetrahedra
the seven fits take 75%: each step of a Householder reflection is a sequential sum over the rows, which
the bitwise requirement keeps sequential, so the fits are latency-bound. More teams per SM hide that
latency best: two teams per SM with the pseudo-inverse in shared memory as well (70 KB each) took
21.7 s on tetrahedra and 5.1 s on hexahedra, and four teams (128 registers per thread) spill.

`TENO::rebuild_cells()` recomputes a list of cells and repacks the stencils on the device, keeping
every other cell's tables bitwise; `TENO::cells_within_reach()` gives the cells whose searches see a
set of changed cells. Rebuilding those after moving nodes reproduces a full setup bitwise (tests on
prisms, mixed cells and tetrahedra in 3D and triangles in 2D).

### 5. Reacting mixtures (#257)

The reacting Volvo LES cost about 24 times air per cell and step. The performance set's `flame_3d`
reproduces its configuration without the bluff body (96 x 48 x 48 hexahedra of 2 mm, propane-air with
the two-step mechanism, mixture-averaged transport, Sigma and TFLES, MUSCL without limiter, hybrid
flux, Strang-split chemistry above 500 K, 72% of the cells reacting) and `air_3d` the same flow of air.

**Baseline** (`main` at 188d169, one A100, kernel times from Nsight Systems per step of 3 stages):
air 4.20 ms per step, the mixture 64.9 ms (15.5 times).

| kernel | ms/step | why |
|---|---|---|
| chemistry (RODAS, one thread per cell) | 27.6 | work memory in rows of a global array: one sector per double per lane |
| `MixtureGradientFunctor` (u, T, X_k at cells) | 18.3 | loops over the 27-cell stencil once per variable (10 times) |
| `SpeciesSlotFunctor` (species upwind fluxes) | 6.3 | face geometry recomputed per species; 1.6 KB of stack |
| `VorticityGradientFunctor` (TFLES) | 2.1 | as the mixture gradient |
| `SpeciesSumFunctor` | 2.0 | face lists reloaded per species |
| `MixtureViscousFluxFunctor` | 1.7 | |
| `ScalarLimiterFunctor` (MUSCL scalar gradients) | 1.5 | neighbors recomputed per scalar |
| the rest (transport, MUSCL, fluxes, LES) | 5.4 | |

**Changes**, each bitwise identical (restart files after 30 steps of `flame_3d`, identical MD5):

- Stencil kernels take the variables innermost: one pass over the stencil (or the cell's faces) with
  the variables' sums in registers, the same operations in the same order. The species slots of MUSCL
  (one point per face) keep the faces' offsets and weights in registers with unrolled face loops.
- The chemistry's thread-per-cell kernel keeps the cell's work memory (Jacobian, LU, RODAS vectors) on
  the thread's stack where it fits 256 or 1024 doubles (6 to about 16 species): GPUs interleave the
  threads' local memory, so a warp's accesses at one offset share cache lines. Larger mechanisms use
  the global array as before (and lanes from 16 species). The kernel reserves local memory for its
  resident threads (up to 1.9 GB on an A100 with the 1024-double size).
- The rate derivatives reuse the reaction's mass-action factors instead of recomputing every other
  reactant's `C^order` (a `pow` for fractional orders) per reactant.
- The viscous flux evaluates each species' coefficient and normal gradient once per face.

| `flame_3d` (A100), ms per step | main | now | now, `chemistry.C_reg = 1e-10` |
|---|---|---|---|
| step | 64.9 | 32.7 | 24.6 |
| chemistry | 27.6 | 18.5 | 10.4 |
| mixture gradients, species slots and sums, vorticity gradient, scalar limiter | 30.2 | 6.5 | 6.5 |
| the rest | 7.1 | 7.0 | 7.0 |
| mixture / air (4.18 ms) per cell and step | 15.5 | 7.8 | 5.9 |

The `h2o2` reactor benchmark (`benchmarks/chemistry`) goes from 6.5 to 9.7 M cells/s at dt = 1e-8 s
and from 3.0 to 5.4 M cells/s at 1e-6 s. Bounding the chemistry kernel's registers for more resident
threads (launch bounds for 128 or 80 registers instead of 220) made it slower.

**What remains.** About 2% of the reacting cells (burnt gas into which the flow step diffused trace
fuel, $Y_\mathrm{C_3H_8}$ 1e-10 to 1e-5) take 20 to 130 RODAS sub-steps against 2 for the others: the fuel burns
out within the step along `C^0.1`, which reaches zero in finite time, through the regularization below
`C_reg` (chemistry.md). Leaving them out of the chemistry (an experiment only) took the chemistry from
19.2 to 11.3 ms per step. `chemistry.C_reg` (default 1e-12 kmol/m³) moves the regularization; larger
values cut these cells' sub-steps (a burnt cell at 1750 K with $Y_\mathrm{C_3H_8}$ = 1e-7 over 0.15 us: 32 sub-steps
at 1e-12, 13 at 1e-9, 5 at 1e-7) but change the rate law below `C_reg`. Opt-in: on `flame_3d`, 1e-10,
1e-9 and 1e-8 give 24.6, 23.4 and 22.3 ms per step; against the default after 100 steps, T differs by
at most 0.54 K, $Y_k$ by 3e-5, p by 90 Pa and u by 0.14 m/s, the integrated heat release by 0.08% (the
largest pointwise heat release differences are in cells burning out trace fuel). Thread-per-cell remains faster
than lanes for 6 species (2, 4, 8 or 32 lanes per cell: 1.8 to 5x slower).

### 6. TENO device memory (#58)

Device memory per cell (`cudaMemGetInfo`, everything included) of 3D Taylor-Green runs on one A100 at the setup
peak and in steady stepping, with a Kokkos Tools library that records every allocation by label. Results are
bitwise those of `main` (restart files after 20-96 steps of Taylor-Green and of the 3D blast, hexahedra and
tetrahedra, orders 3, 5 and 6, also with the troubled passes split into 100 rounds).

| bytes per cell, setup peak / steady | cells | main | change 6 | `single_precision_tables` |
|---|---|---|---|---|
| TENO5, 100³ hexahedra | 1.00M | 47.3 K / 47.3 K | 39.8 K / 39.8 K | 25.0 K |
| TENO5, 55³ x 6 tetrahedra | 998k | 58.7 K / 58.7 K | 51.2 K / 51.2 K | 30.6 K |
| TENO3, hexahedra | 1.00M | 18.6 K / 18.6 K | 15.9 K / 15.9 K | |
| TENO3, tetrahedra | 998k | 17.8 K / 17.8 K | 15.0 K / 15.0 K | |
| TENO6, 64³ hexahedra | 262k | 129.6 K / 99.6 K | 77.9 K / 77.9 K | 46.5 K |
| TENO6, 32³ x 6 tetrahedra | 197k | 152.0 K / 109.8 K | 85.2 K / 85.2 K | 50.5 K |

What `main` held for TENO5 on hexahedra (1M cells): the packed stencils, 31.7 KB (large-stencil pseudo-inverses
22.3 KB, sector pseudo-inverses 7.8 KB, stencil cells and mirror faces 1.5 KB), the smoothness-indicator matrices
(4.8 KB), the troubled passes' scratch for every cell (3.5 KB), the setup's team scratch, which Kokkos sizes for
every thread the device can hold and keeps (1.7 KB), basis means (0.3 KB), and the solver's fields and mesh (about
5.5 KB, of which the face states take 2.2 KB). At order 6 the setup's batches (65,536 cells of up to 135 KB of
outputs) and team scratch set a peak 30% above the steady state.

- The 3D smoothness-indicator matrices are formed in the troubled-cell selection from each cell's central moments
  (84 values at order 5, 165 at order 6) instead of stored (595 and 1540 values), with the setup's sequence of
  operations and roundings (no FMA): the selection runs one cell per team, whose threads form the matrix in team
  scratch, four entries per thread at a time.
- The troubled passes' scratch holds half the cells (`troubled_capacity`); more troubled cells are done in further
  rounds. Below a capacity of 1, every round, the first included, recomputes its cells' central coefficients with
  the smooth pass's arithmetic. Which cells fall in the first round follows the order of the queue's atomic
  increments, and two kernels need not contract the same arithmetic into the same FMAs (#612).
- The setup takes its team scratch from a pool with one slot per team that can be resident (0.3 GB instead of
  1.7-2.2 GB), freed with the setup, bounds its batches by 2 KB per cell, and hands its moments to TENO instead of
  a copy: the setup peak no longer exceeds the steady state.
- What remains is mostly the pseudo-inverses (30 KB of the 40 KB per hexahedron). Every cell's are different,
  bitwise and even at 1e-6 relative, since the equidistant entries of a lattice stencil are ranked in a different
  order from cell to cell, so tables cannot be shared without changing results. `single_precision_tables`
  (opt-in) stores them in single precision. Differences from double precision, as a fraction of each field's
  range: below 1e-7 for Taylor-Green (20 steps) and 2e-8 for the blast on tetrahedra (96 steps), but up to 2.6e-4
  (order 5), 4.9e-4 (order 6) and 6.9e-3 (order 3) for the blast on hexahedra after 60-80 steps, where the
  stencil selection near the shock goes the other way in some cells.

Largest Taylor-Green box at TENO5 that runs on one A100-80GB: main fails at $125^3$ hexahedra (1.95M cells; out of
memory allocating the face states), which change 6 runs, as does $145^3$ (3.05M) with `single_precision_tables`.
From the table: about 1.8M, 2.1M and 3.4M hexahedra; 1.45M, 1.65M and 2.75M tetrahedra; at order 6 about 0.65M,
1.1M and 1.8M hexahedra.

Step times against `main` (A100, best of 2-3): TENO5 Taylor-Green $96^3$ 151 ms (154), 3D blast $100^3$ 225 ms (235),
$64^3$ blast 67.6 ms (69.9), $32^3 \times 6$ tetrahedra blast 48.5 ms (51.5), Taylor-Green on $32^3 \times 6$ tetrahedra 37.1 ms
(37.9); TENO3 blasts 1-3% faster. At order 6 the step is 0.7-1.1% slower: 75.2 ms (74.7) for the $48^3$ blast,
48.2 ms (47.7) for the $24^3 \times 6$ tetrahedra blast, where up to 27% of the cells are troubled and forming the
1540-entry matrices costs more than reading them, and 54.7 ms (54.1) for Taylor-Green on $48^3$, where the smooth
pass, whose code is unchanged, runs 1% slower with this build's memory layout. `single_precision_tables` is
another 3-8% faster (the smooth pass reads half the bytes). Setup: 4.6 s for 1M hexahedra at order 5 (as on
main), 21 s for 1M tetrahedra (20 s); at order 6, 7.2 s (6.6 s) and 61 s (58 s), from the smaller batches.

### 7. Memory held from setup (#551)

Every Kokkos allocation alive at the first step's time-step reduction (the same Kokkos Tools library, by label),
in steady stepping (20-30 steps, output and restarts included), with host RSS and glibc's heap in use; one A100,
16 OpenMP threads. The setup's own scratch (initial-state sampling, TENO batches and search, partitioning) was
already freed with its scope. What the run held and never read again:

- on the device, the mesh's node coordinates and the cells' and faces' node lists (2D TENO keeps the faces' ones,
  and in axisymmetric runs the cells', for the points of straight faces), and the face zones' lists;
- on the host, the copies of the vertex-neighbor lists (the stencil and image searches read them); with GPUs
  these are separate from the device's;
- the regions' volume fractions (8 bytes per cell per region; `mallard check` keeps them for its region report);
- host memory the setup freed, which glibc keeps in its arenas: `malloc_trim` hands it back after setup.

| bytes per cell, first step: main / change | cells | device (Kokkos) | device (cudaMemGetInfo) | host RSS |
|---|---|---|---|---|
| 2D MUSCL, triangles | 1.84M | 975 / 926 | 1750 / 1698 | 503 / 332 |
| 2D TENO5, triangles | 461k | 2902 / 2880 | 11830 / 11807 | 896 / 682 |
| 2D MUSCL, H2/air, mixture-averaged, reacting | 500k | 5097 / 5029 | 7933 / 7866 | 1278 / 1202 |
| 3D TENO5, hexahedra | 1.00M | 7713 / 7581 | 40830 / 40694 | 2357 / 704 |
| 3D TENO5, tetrahedra | 998k | 7249 / 7185 | 52295 / 52225 | 1409 / 556 |

In steady stepping the device numbers are the same; host RSS grows again with the output's staging (3D TENO
hexahedra: 2373 / 1549, tetrahedra 1427 / 1057, 2D MUSCL 562 / 488). The largest items: the host vertex-neighbor
lists (134 B per hexahedron, 350 B per tetrahedron) and, on the device, the face and cell node lists and node
coordinates (42 B per triangle of 2D MUSCL, 120 B per hexahedron). Restart files are bitwise those of `main`.
On CPUs the device views are the host copies, so only the regions' fractions and glibc's arenas change.

### 8. Memory sizing in `mallard check` (#550)

`mallard check` records every Kokkos allocation by label (`memory::Tracker`, Kokkos Tools callbacks chained to
any tool loaded with `KOKKOS_TOOLS_LIBS`) and gives each label a category (`src/common/memory_tracker.cpp`; the
examples' test fails on a label no rule names). It keeps the peak of each setup phase and of the steady state,
each as bytes proportional to the cells plus bytes that are not (the setup's chunks and batches, the chemistry's
work memory, Kokkos' scratch), and sizes a rank by the largest of these at its cell count. TENO's packed stencils
live in CUDA virtual memory outside Kokkos; the bytes they fill are reported to the tools as allocations, and the
rest of the last 2 MB page as bytes that are not proportional to the cells (9 of the 53 MB of 2D TENO3 tables at 16k cells).

Measured: each case run on one A100-80GB (5 steps) with a Kokkos Tools library that records the peak of the
Kokkos allocations and the largest `cudaMemGetInfo` use over the run. Predicted: `mallard check --memory a100-80`
on a CPU (2 threads) of the same case with 4 to 16 times fewer cells, extrapolated, plus the typical memory outside
Kokkos; the rows checked at the measured size are the check's own prediction for its mesh.

| case | cells | checked at | Kokkos, predicted / measured | error | total, predicted / measured | error |
|---|---|---|---|---|---|---|
| MUSCL 2D | 65,536 | 65,536 | 0.08 / 0.08 GB | +1.4% | 1.68 / 1.55 GB | +8.6% |
| MUSCL 2D | 1,048,576 | 65,536 | 1.08 / 1.08 GB | -0.2% | 2.68 / 2.60 GB | +3.3% |
| TENO5 2D | 262,144 | 262,144 | 2.25 / 2.26 GB | -0.2% | 3.85 / 3.78 GB | +1.9% |
| TENO5 2D | 1,048,576 | 262,144 | 8.98 / 8.97 GB | +0.1% | 10.58 / 10.51 GB | +0.7% |
| TENO5 3D | 110,592 | 110,592 | 4.37 / 4.38 GB | -0.2% | 6.57 / 6.45 GB | +1.8% |
| TENO5 3D | 884,736 | 110,592 | 34.41 / 34.40 GB | +0.0% | 36.61 / 36.50 GB | +0.3% |
| TENO3 3D | 262,144 | 262,144 | 3.97 / 3.99 GB | -0.5% | 6.17 / 6.10 GB | +1.1% |
| TENO3 3D | 2,097,152 | 262,144 | 31.12 / 31.21 GB | -0.3% | 33.32 / 33.33 GB | -0.1% |
| TENO6 3D | 32,768 | 32,768 | 3.28 / 3.24 GB | +1.2% | 5.48 / 4.59 GB | +19.3% |
| TENO6 3D | 262,144 | 32,768 | 19.08 / 19.39 GB | -1.6% | 21.28 / 21.51 GB | -1.0% |
| propane flame 3D (`flame_3d`) | 221,184 | 221,184 | 2.22 / 1.57 GB | +41.1% | 4.42 / 3.61 GB | +22.3% |
| propane flame 3D | 1,769,472 | 221,184 | 9.65 / 9.64 GB | +0.1% | 11.85 / 11.71 GB | +1.2% |

The 2D Riemann problem (quadrilaterals) and the 3D Taylor-Green vortex (hexahedra) of the performance set, HLLC, SSPRK3. From a check of 4 to
16 times fewer cells the total is within -1.0% to +3.3% of the measured peak, inside the 5% that sizing keeps
free. Small meshes are overestimated, on the safe side:

- The memory outside Kokkos, measured after the steps: 1.48-1.54 GB in 2D and 2.05-2.13 GB in 3D, of which CUDA's
  context is 0.45-0.50 GB at the first allocation and the rest the local memory CUDA reserves for kernels with
  large stacks at their first launch. The setup's peak comes before those launches (TENO6 at 32^3), but sizing
  counts all of it at every peak. A check on a CPU counts 1.6 GB (2D) or 2.2 GB (3D); a check on the GPU with
  `--steps` measures it.
- The chemistry's work memory has a row per cell in flight, `max(4096, 4 x concurrency)` rows of 1 GiB at most: 8 MB
  on 2 CPU threads, 423 MB for the 221k cells of `flame_3d` and 1.07 GB from about 550k cells on an A100. A check
  on a CPU counts the 1 GiB bound for a GPU.
- Before the stencil pages were reported, a check missed 30 KB per cell of TENO5 in 3D (29.6 GB at 96^3).
- TENO6's tables take 2-4% more per cell on the GPU than in a check on a CPU, the largest underestimate above
  (-1.6% of the Kokkos peak).

### 9. Analytical initialization (#677)

`[initialize]` expressions and regions are cell averages: every cell is split into simplices (24 tetrahedra
for a hexahedron, from its vertex average and face averages), each carrying a 14-point degree-5 rule (7 points
in 2D), so a 1M-hexahedron start evaluates the expressions at 336M points. The Taylor-Green start of the
performance set at $100^3$, timed as the "initial solution" phase:

| A100 (16 OpenMP host threads) | main | change |
|---|---|---|
| host: simplices of each 16k-cell chunk (`h_cell_tetrahedra`) | 1.23-1.39 s | — |
| copies of the simplices to the device (2.3 GB) | 1.20-1.27 s | — |
| kernels (62 launches of 16k cells / 3 of up to 419k) | 0.63 s | about 0.3 s |
| initial solution | 3.04-3.08 s | 0.31-0.34 s |

- The kernel builds a cell's simplices itself, from the mesh's cell-node lists and node coordinates (still on
  the device during the setup), with `cell_tetrahedra`'s order and arithmetic; only curved cells' rules
  (`curved::Geometry::cell_rule`) come from the host.
- A device launch takes as many cells as 64 MB of scratch rows hold (419k here, at 160 bytes a row) instead of
  16k, which left the A100 partly idle: 16k cells per launch took 0.53 s, 64k 0.36 s, 256k 0.31 s, 1M 0.28 s.
  Host builds keep launches of 16k cells, which keep their threads busy.

On CPUs the time is the expressions and the conversions to conservatives at every point. kexpr's `evaluate`
dispatches each instruction through a switch for each point; `evaluate_lanes` runs one program over the 14
points of a simplex's rule (7 in 2D) at once, each instruction over all of them before the next, so the dispatch
is paid once per 14 points and the points' operations overlap. Every point gets `evaluate`'s operations in its
order, so the values are its bits (`KexprTest.LanesMatchPointByPointBitwise`); programs with jumps (statements,
loops) go point by point. GPU threads keep one point each. The lanes cover the base layer's programs; regions'
programs, which apply at some points only, stay per point.

| M4 Pro, 2 threads (Threads backend), fastest of 2-3 runs | 0.8 (exprtk) | main | change |
|---|---|---|---|
| Taylor-Green, $100^3$ hexahedra, machine load about 9 | 6.65 s | 18.8 s | 6.49 s |
| the same, machine load about 17 | 8.92 s | 25.4 s | 9.01 s |

Half of the change's time is `sin` and `cos`; most of the rest is the conversions and the interpreter.

Initial states are bitwise those of `main`: 2D and 3D generated meshes of every cell type, file meshes with
tetrahedra, prisms and curved (second-order) boundaries, regions with sharp and smooth edges, tabulated data,
mixtures and synthetic turbulence, on CPUs and on an A100.

### 10. Steps enqueued ahead of the GPU (#730)

With one CUDA graph per step, the host still waited for every step: it read dt, the step controller's verdict
and the divergence check, then decided on the host whether the step was due for output, statistics, probes,
monitors, a progress row or the end of the run, before launching the next graph. The GPU was idle for that
round trip at every step, the larger part of a step on small meshes.

The step's end now runs on the device, in the graph (`step_queue.h`): it advances t (the host's `t += dt`,
the same rounding), counts the step, counts rejected tries (a rejected step leaves t and the solution where
they were, so the next launch takes it again with the controller's dt), adds the step to the conservation
ledger, updates p_max, counts non-finite values (`check_nan`) and, at progress rows, reduces the row's and the
run health's local values into a pinned record. The host computes, at each step it acts on, the first step
and the first time at which it must act again (`Solver::step_horizon`: `n_steps`, `t_stop`, output and sample
intervals, time-based outputs, control files). Each such condition is `step >= s` or `t >= t_x`
(intervals: their next multiple; times: the threshold `due` compares with), so the device's comparison with
their minima is the host's decision bit for bit; a condition the device cannot evaluate exactly (spectra,
phase averages and time-averaged probes after `t_start`) stops it at every step where it may hold, and the
host decides as before.

The host keeps `step_queue` launches enqueued and reads each launch's record once its event completes.
Run-control requests (signals, `mallard ctl`, the wall time) go to a pinned array that each step's
time-step reduction reads, as before, with minus rank 0's wall time (the other ranks post the largest value),
which the reduction's minimum gives every rank for the triggers that read `wall`. The step body sits under a conditional IF node on a device flag: after
a step the host must act on (or a divergence, or the last allowed rejection), the device clears the flag and
the launches already enqueued do nothing; the host then acts on that step's state with the code of the
one-step path. Progress rows' reductions sit under a second IF node. Builds without conditional nodes (CUDA
before 12.4, or a failed capture of NCCL operations under one, tried on every rank before the first step)
launch one step at a time with the IF bodies always run.

Runs with triggers that read a probe's value stop the device at every step (probes' values are reduced on
the host).

Snapshots and restarts are staged in the graph. The writers' clocks (`OutputClock`, the arithmetic
`DataWriter` uses too) live in device memory; after the triggers, a kernel decides which writers are due at a
step the device goes on from and advances their clocks, and the step's target time (the next output time, as
`next_target_time` has it) follows them. Under an IF node on that decision the graph does `write_data`'s
device work at the device's `t` (the primitives, the transport, the heat release, the eddy viscosity, the
gradients and derived fields) and copies every array the files read, with the statistics' clocks, the step
controller's slots, the characteristic faces' state and the triggers' states, into the host's mirrors. The
mirrors are pinned (never the device's own arrays, also off GPUs), so the copies are kernels, and the host
writes the files from them as it takes the record, after the step's probes and before its progress row, as
`after_step` orders them. One set of mirrors serves every output: each rank posts the number of staged outputs
it has written in the step's time-step reduction, whose minimum the device compares with the number it
staged, so every rank stages, or stops for the host (`OUTPUT`), at the same steps. Outputs of runs with
`[sources]` (evaluated at the host's `t`), regions that move or follow the solution, fused chemistry, averaged
spectra, phase averages (whose files and restart arrays read the device) stop the
device at their steps, as do all outputs of builds without conditional nodes.

A progress row's values are those of the step it reports, at any depth. The row's local reductions (with
the mixture's and the chemistry's columns) go into a struct in the record. The run health's per-rank values
go into a pinned array per record slot: the ledger's totals, the element drift over chemistry calls and
each repair site's counts. The repair sites' cells start their next interval in the same kernels. The host
reduces them over the ranks as it takes the record.

Force and integral monitors are evaluated in the graph. After a step's end, a kernel sets one flag per
monitor (the step is accepted, the device goes on, and the step is a multiple of the monitor's interval);
each monitor's reduction sits under an IF node on its flag and is reduced over the ranks on the device
(NCCL in NCCL builds), then copied into a pinned array per record slot with a bit in the record. The host
writes the rows from the record as it takes it. At a step where the device stops, the host evaluates the
monitors itself, with the same device reduction, so rows match at any depth.

The statistics sample in the graph too (means, spectra, phase averages and conditional statistics). Each part is due by its own interval and
`t_start`, which its clock in device memory holds. The means' clock (weight, last sample time, `t_start`, sample count) lives
in device memory and every sample, the host's included, advances it in a kernel, so the device's decision
(`step % interval == 0` and `t > t_start`, after an accepted step the device goes on from) and the weights
match the host's bit for bit. The host reads the clock when it writes a snapshot or a restart.
Spectra, phases and `[[statistics.conditional]]` entries sample under flags of their own. Under a flag for any
of them, the fields they read are computed once, as the host's `sample_statistics` computes them: the
primitives, or the cell states, velocity gradients and derived fields, and the heat release rate and
production rates. Every derived field is computed on the device, at the device's `t` for the boundary states. The conditional
entries' fixed-point sums are reduced with `comm::allreduce_device`. Spectra written at each sample (`average =
false`) are copied into a pinned array per record slot on rank 0, with a bit in the record, and the host writes
the file from it as it takes the record.

Probes written every few steps (not time-based, not time-averaged, not reading `[statistics]` outputs) are
evaluated in the graph as well. A kernel sets one flag per probe and one for any; under the latter the
derived fields the due probes read are updated (the cell states, transport and gradients, as the host's
`write_probes` updates them), and under each probe's flag its samples are evaluated and copied into a pinned
array per record slot. The host gathers them over the ranks, reduces and writes them from that slot as it
takes the record; the intervals are copied to the device whenever it resumes, so control commands that change
them take effect at the host's next step.

With fused chemistry (`[chemistry] fuse_half_steps`), each step ends with a half step still pending, which the
host applies to a copy (`Solver::flush_chemistry`) for whatever reads the state. In the graph, before the first
of the monitors, statistics, probes and progress row due at a step, a kernel sets a flag if the launch has not
flushed yet; under its IF node the state, temperature seeds, last sub-steps, costs, the integrator's tallies and
last costs are copied to the kept buffers and the chemistry runs over the pending time (outside the repair
audit, as on the host). After the progress row a last IF node copies them back. The flush runs the host's code
on the same inputs, so the samples are the host-flushed ones bit for bit, and since the kept buffers include
the integrator's counts and costs, a flush leaves no trace in the steps after it: a step the device flushed
for and the host then takes (a trigger fires) is flushed again by the host from the same state. A failed
flush stops the device and the host's own flush reports it.

### 11. Read-only loads (#641, not kept)

No View in `src/` carries `Kokkos::RandomAccess` or `Kokkos::Restrict`. Those traits would not change anything
either: Mallard builds Kokkos 5.2 with its mdspan-based View (`Kokkos_ENABLE_IMPL_MDSPAN`, the default, so not
`Kokkos_ENABLE_IMPL_VIEW_LEGACY`), whose accessor depends only on the `Unmanaged` and `Atomic` traits. The
`__ldg` loads for const `RandomAccess` Views belong to the legacy View only. To measure what read-only loads
are worth, `__ldg` was tried explicitly (the read-only, non-coherent path that `const __restrict__` pointers allow
in CUDA) on the reads of data that no thread writes during the kernel. It was applied cumulatively: (a) TENO's
packed stencil tables (pseudo-inverses, stencil cells and faces), (b) the stencil states (W) that the smooth and
troubled passes gather, (c) the face states the convective flux reads and the face fluxes that `face_flux_sum`
gathers. Each change multiplied the 3D build's `LDG.E.*.CONSTANT` instructions (345 on `main`, then 2434, 3298 and
3466), and results stayed bitwise identical (`mallard diff` of the restart files after 30 steps of `teno5_3d`).

| `teno5_3d` (A100, CUDA 13.5), best of 3, interleaved | main | (a) | (a, b) | (a, b, c) |
|---|---|---|---|---|
| throughput, cells/s | 6.06M | 6.06M | 6.06M | 6.06M |
| wall time per step | 146 ms | 146 ms | 146 ms | 146 ms |
| `teno_smooth` (team pass), per launch (Nsight Systems) | 42.09 ms | | | 42.12 ms |
| `convective_flux` (HLLC), per launch | 3.432 ms | | | 3.430 ms |
| `face_flux_sum`, per launch | 179.2 us | | | 179.3 us |

None of them won, so none was kept. On the A100, ordinary global loads are already cached in L1, which is also
where the read-only path caches. The smooth pass waits on the latency of its dependent loads (section 1),
not on cache capacity, and the flux kernels stream their face data once.

### 12. Staged stencil entries in the 3D smooth pass (#641, not kept)

The team smooth pass (TENO5 on hexahedra, conservative variables) compiles to 168 registers and a 320 B stack. Its
launch bounds leave room for 10 warps per SM. Over the stencil, each entry is a chain of dependent loads (the
stencil cell, then its 5 values of W, scattered), and the 5 threads of a cell all load the same entries. Nsight
Compute's counters were not available, so these figures come from the compiled kernel (`cuobjdump`).

The tried change: each team stages its stencil entries in team scratch, in rounds of STAGE entries. Each of a cell's
threads loads every fifth entry, issuing all its loads of the round before using any, then every thread takes the
round's entries in order. This was tried in two forms. Times are for `teno5_3d` on one A100, interleaved best of 3:

| | ms/step | cells/s | bitwise |
|---|---|---|---|
| main | 8,773 / 8,961,187 | 8,585 / 9,157,200 | 8,813 / 8,920,514 | 8,625 / 9,114,733 | 11,072 / 7,005,283 | 10,065 / 7,706,501 | 17,935 / 4,324,752 |
| loader converts to conservative variables, STAGE = 10 | 136 | 6.50M | no |
| the same, STAGE = 15 | 137 | 6.46M | not checked |
| the same, STAGE = 25 | 146 | 6.08M | not checked |
| loader stages the raw loads (stencil cell, face field, W) only, STAGE = 10 | 157-158 | 5.61M | yes (`teno5_3d`, 30 steps) |

- With the conversion in the loading thread, U goes through shared memory. nvcc can then no longer fuse `W0 * W1 - U0`
  into one FMA as it does on main, and 6 of 8 cases differ from main by about 3e-16 relative.
- Order 6 does not fit two teams per SM with the staged entries (Kokkos caps the team size at 32).
- STAGE = 25 (78 KB of scratch per team) leaves the SM about 28 KB of L1, against 92 KB on main, and loses the gain.
- Staging only the raw loads keeps the arithmetic and the results, but every thread then repeats the conversion and
  reads 5 doubles of shared memory per entry, which costs more than the loads it overlaps.
- A run of main against itself on a gas-mixture case was bitwise identical, so the differences above come from
  the change.

Keeping the 7% bitwise needs the stencil loop's rounding made explicit (#931).

### 13. TENO setup on CPUs (#916)

A start without a TENO cache, and `mallard post derive` or `post extract` of a run without one, spend nearly
all their setup in the per-cell tables. Profiled on the host (`sample`, 2 Threads), the 2D setup was mostly
the monomial means over each stencil cell's quadrature points (36 per triangle at order 5) and the
smoothness-indicator integrals, the 3D setup the central least-squares fit (QR and pseudo-inverse), its
Lebesgue constant, the cells' moments and the 3D monomial means.

- **Powers by recurrence.** `teno::monomials` and the 2D smoothness integrals formed each power with
  `ipow` and each monomial's exponents by a search, at every point; they now take the powers from a table
  built by the same repeated multiplication, and the exponents and falling factorials once per cell.
- **Sector means from the central ones.** In 2D, the degree-2 means of a sector stencil's cell are the
  first of the degree-r means the central fit already has (the same sums, term by term), so they are
  copied instead of integrated again; mirror images, whose line face may change with the deeper gather,
  are still integrated.
- **Lebesgue constant along memory.** The sums over the stencil's cells read the pseudo-inverse
  transposed into the free least-squares matrix, so they run over contiguous memory and vectorize.

Every value takes the same sequence of operations, so the tables are bitwise unchanged: the TENO cache
files of the cases below and of the 2D and 3D cylinder test meshes (triangles and quadrilaterals with
curved walls, tetrahedra and pyramids, symmetry planes) are byte for byte those of the previous code.
Apple M4 Pro, 2 Threads, TENO5, `numerics` setup phase, the better of 2 alternating runs:

| case | tables before | tables after | instructions |
|---|---|---|---|
| 2D double Mach reflection, 115,200 triangles | 6.2 s | 3.7 s | -36% |
| 3D Taylor-Green vortex, 64³ hexahedra | 26.0 s | 23.4 s | -19% |

The 3D moments (5.6 s for 64³, the same before and after), the QR and the search are left: their
operations are already in order, and changing their order would change the tables.

### 14. TENO table memory (#431)

Bytes per cell of the TENO tables (the "reconstruction tables" row of `mallard check --memory a100-80`, CPU build,
periodic boxes of $16^3$ hexahedra and $10^3 \times 6$ Kuhn tetrahedra), and the cells per A100-80GB that
`mallard check` predicts:

| case | before | compact stencil entries | cells per A100-80GB before / after |
|---|---|---|---|
| TENO3, hexahedra | 10.3 kB | 9.77 kB | 5.64M / 5.85M |
| TENO4, hexahedra | 18.1 kB | 17.4 kB | 3.58M / 3.69M |
| TENO5, hexahedra | 32.0 kB | 31.3 kB | 2.08M / 2.12M |
| TENO6, hexahedra | 65.1 kB | 64.2 kB | 1.10M / 1.12M |
| TENO3, tetrahedra | 8.52 kB | 8.10 kB | 6.46M / 6.69M |
| TENO4, tetrahedra | 14.4 kB | 13.9 kB | 4.31M / 4.43M |
| TENO5, tetrahedra | 44.0 kB | 43.1 kB | 1.63M / 1.66M |
| TENO6, tetrahedra | 72.0 kB | 71.1 kB | 1.02M / 1.04M |

- **Compact stencil entries.** Each stencil slot held its cell and its mirror face (or -1), 8 bytes. Mirror
  images are rare (cells next to straight walls and symmetry planes), so a slot now holds one word: the cell, or,
  for an image, its index in a per-rank list of (cell, face) pairs. Results are bitwise unchanged.
- **Shared tables.** The pseudo-inverses are 90% of the tables, and on a lattice
  every cell's differ only in the order of their equidistant entries and at round-off (5e-14), so no two are
  bitwise equal. Ranked by their own columns on a grid of 2^-30 of their magnitude, the entries of one stencil
  shape come in one order; the cells of a shape then read the table of its cell with the lowest global id,
  agreed over the ranks, through a per-cell index. Bytes per cell of the tables and cells per A100-80GB
  (`mallard check`; the GPU's packing in blocks of 32 tables measured with the CPU build packing slices of 32):

  | case | compact entries | shared, per cell | shared, blocks of 32 | cells per A100-80GB |
  |---|---|---|---|---|
  | TENO3, hexahedra | 9.77 kB | 721 B | 774 B | 5.85M / 17.9M |
  | TENO4, hexahedra | 17.4 kB | 1.15 kB | 1.26 kB | 3.69M / 15.6M |
  | TENO5, hexahedra | 31.3 kB | 1.77 kB | 1.97 kB | 2.12M / 10.3M |
  | TENO6, hexahedra | 64.2 kB | 2.76 kB | 3.21 kB | 1.12M / 8.70M |
  | TENO3, tetrahedra | 8.10 kB | 638 B | 656 B | 6.69M / 18.3M |
  | TENO4, tetrahedra | 13.9 kB | 1.02 kB | 1.06 kB | 4.43M / 16.1M |
  | TENO5, tetrahedra | 43.1 kB | 1.90 kB | 2.06 kB | 1.66M / 12.5M |
  | TENO6, tetrahedra | 71.1 kB | 2.79 kB | 3.07 kB | 1.04M / 10.4M |

  What a TENO5 hexahedron keeps is mostly its entries' cells (0.85 kB) and its moments (0.7 kB); the face
  states and gradients (4 kB) now dominate the memory per cell. Results change once at round-off against
  per-cell tables: restart files after 5 steps of a blast on hexahedra, tetrahedra and
  mixed cells with walls differ by at most 9e-16 (absolute, fields of order 1), and do not depend on the rank
  count, slice width or thread count. Unstructured meshes have a shape per cell and keep their packing.

### 15. Working set beyond the TENO tables (#552)

`mallard check --memory a100-80` (CPU, 2 Threads, one step) of 3D TENO5 cases, HLLC, SSPRK3: the Taylor-Green
vortex of the performance set on $32^3$ hexahedra and $16^3 \times 6$ tetrahedra (perfect gas, viscous; LES adds
the Sigma model), and the propane flame of `flame_3d` with TENO5 on $48 \times 24^2$ hexahedra and
$24 \times 12^2 \times 6$ tetrahedra (two-step mechanism, 6 species), and with GRI-Mech 3.0 (53 species) on
$32 \times 16^2$ hexahedra. Bytes per cell at the steady state, and the cells that fit on an A100-80GB. The
reconstruction tables are 1.97 KB per hexahedron and 2.47 KB per tetrahedron (#431); next to them are the face states
(2.2 KB per hexahedron, 1.2 KB per tetrahedron), the troubled passes' scratch (1.8 KB) and the vertex
gradient's weights (0.6 / 1.5 KB). Results are bitwise unchanged (restart files after 4 steps of each case,
and of MUSCL, double-flux and `outlet_extrapolation` variants).

| bytes per cell, cells per A100-80GB | perfect gas, hexahedra | tetrahedra | LES, hexahedra | tetrahedra | flame, hexahedra | tetrahedra | GRI-Mech, hexahedra |
|---|---|---|---|---|---|---|---|
| main | 8,773 / 8,961,187 | 8,585 / 9,157,200 | 8,813 / 8,920,514 | 8,625 / 9,114,733 | 11,072 / 7,005,283 | 10,065 / 7,706,501 | 17,935 / 4,324,752 |
| ghost weights of boundary faces | 8,638 / 9,100,971 | 8,497 / 9,252,578 | 8,678 / 9,059,023 | 8,537 / 9,209,223 | 10,935 / 7,093,479 | 9,974 / 7,776,381 | 17,799 / 4,357,919 |

- **Ghost weights of boundary faces.** The vertex gradient (viscous fluxes, the hybrid flux's sensor, MUSCL)
  kept a weight for every face of every cell, though only boundary faces have ghosts; it keeps them for those
  faces alone, with each cell's list of them: 135 bytes less per hexahedron and 88 per tetrahedron here, about
  140 and 92 on large meshes.

## Scaling (one node)

A100-80GB GPUs of one node (NVLink), one rank per GPU, Hilbert partitions, GPU-aware MPI; steady time per step
(the last progress interval of 200-400 steps, setup excluded); main as of change 2. Efficiency is
T1 / (n Tn) for strong scaling, T1 / Tn for weak.

**Strong scaling**

| case | cells | 1 GPU | 2 GPUs | 4 GPUs | efficiency 2 / 4 |
|---|---|---|---|---|---|
| 2D TENO5, Riemann | 1M | 19.6 ms | 10.6 ms | 6.00 ms | 92% / 82% |
| 2D TENO5, Riemann | 4M | 74.0 ms | 38.3 ms | 20.4 ms | 97% / 91% |
| 2D MUSCL, Riemann | 4M | 15.6 ms | 8.13 ms | 4.50 ms | 96% / 87% |
| 2D MUSCL, Riemann | 262k | 1.22 ms | 786 us | 667 us | 78% / 46% |
| 3D TENO5, Taylor-Green (viscous) | 885k | 152 ms | 78.7 ms | 40.7 ms | 97% / 93% |

**Weak scaling**

| case | cells per GPU | 1 GPU | 2 GPUs | 4 GPUs | efficiency 4 |
|---|---|---|---|---|---|
| 2D TENO5, Riemann (1024² / GPU, square domain at 4) | 1M | 19.6 ms | 20.1 ms | 20.4 ms | 96% |
| 2D MUSCL, Riemann | 4M | 15.6 ms | 16.1 ms | 16.3 ms | 96% |

Notes:

- Weak scaling must keep the cells' shape: refining a direction alone (anisotropic cells) changes TENO's stencils
  and their cost. 3D TENO5 on $64^3$ cells per GPU refined along x and y only gave 47.2 / 52.2 / 69.1 ms on 1 / 2 / 4
  GPUs; the comparison with the domain grown instead is in progress.
- At small sizes per GPU (MUSCL, 262k cells on 4 GPUs: 65k per GPU) the step is latency-bound (about 30
  launches and four exchanges per step); the CUDA graph and NCCL exchange of #217 take it from 667 to 523 us.
- Multi-node runs (16-32 GPUs over InfiniBand) are pending free GPUs.

## Partitions

3D TENO5 spends what it loses to more GPUs mostly reconstructing halo layer 1 again (owned and layer-1
cells are reconstructed, so that one exchange per stage suffices; `mpi.md`, section 7), so the parts'
surface sets its scaling. A100-80GB, 4 GPUs per node, HDR InfiniBand with GPUDirect RDMA between nodes,
GPU-aware MPI; steady time per step. *Layer 1* is the largest number of halo layer-1 cells of a rank
(the setup summary's "Partition surface" line). Graph: dKaMinPar with its `strong` preset and 0.1%
imbalance; its default preset is worse than both others (4 GPUs, $96^3$: 44.1 ms, 3% imbalance).

**Strong scaling**, $96^3$ hexahedra (1 GPU: 154 ms)

| GPUs | Hilbert | graph | multi-jagged | efficiency (multi-jagged) | layer 1: Hilbert / graph / multi-jagged |
|---|---|---|---|---|---|
| 4 | 41.2 ms | | 41.4 ms | 93% | 9,312 / 9,776 / 9,312 |
| 8 | 23.1 ms | 23.5 ms | 23.1 ms | 83% | 7,057 / 7,958 / 7,057 |
| 12 | 17.8 ms | 17.2 ms | 16.6 ms | 77% (Hilbert 72%) | 13,038 / 8,954 / 7,906 |
| 16 | 13.4 ms | 13.3 ms | 13.1 ms | 73% | 7,130 / 9,977 / 7,130 |

**Weak scaling**, $64^3$ hexahedra per GPU (1 GPU: 48.1 ms)

| GPUs | mesh | Hilbert | graph | multi-jagged | efficiency (multi-jagged) |
|---|---|---|---|---|---|
| 4 | 128 x 128 x 64 | 49.8 ms | | 49.8 ms | 97% |
| 8 | 128³ | 50.8 ms | 51.4 ms | 50.5 ms | 95% |
| 12 | 192 x 128 x 128 | 54.7 ms | 52.7 ms | 51.9 ms | 93% (Hilbert 88%) |
| 16 | 256 x 128 x 128 | 55.8 ms | 53.9 ms | 52.7 ms | 91% (Hilbert 86%) |

2D at 16 GPUs the two built-in partitioners are within 1-4% of each other (equal parts of a square):
TENO5 4M cells 6.14 ms, 16M 20.7 ms (1M per GPU: 95% weak efficiency), MUSCL 16M 4.15 ms (Hilbert 4.12 ms).

Notes:

- Multi-jagged parts of a box are its cuboid blocks: the same as Hilbert parts at power-of-two rank counts
  on cubes, and 30-45% fewer layer-1 cells otherwise (12 and 24 ranks, the weak-scaling boxes from 12 ranks
  up). On a graded O-mesh around a cylinder (2D) and on tetrahedra they also cut fewer faces than the curve.
- At equal surface the orientation still mattered while a rank numbered its cells like the global mesh
  (x first): 4 GPUs, $96^3$ split 2 x 2 x 1 took 41.2 ms, 1 x 2 x 2 43.9 ms, 1 x 1 x 4 48.4 ms (TENO +12%,
  gradients +34% per launch). Multi-jagged breaks ties toward more pieces along x. At 16 GPUs 4 x 4 x 1 (20%
  more surface) is fastest (12.8 ms). Ranks now number their cells along a Hilbert curve ("Cell numbering").
- Multi-jagged numbers its blocks along a Hilbert curve, so that the ranks of a node hold a compact group:
  16 GPUs, $96^3$: 13.0 ms instead of 13.9 ms with the blocks numbered x-major.
- Exchanging reconstructed face states instead of reconstructing layer 1 (`mpi.md`, section 7) is not worth
  its second, dependent exchange per stage yet: at 16 GPUs ($96^3$, an nsys profile of one rank) TENO's
  reconstruction takes 11.2 of the 13.1 ms the GPU is busy per step, and layer-1 cells are 11% of the
  reconstructed cells, so it could save at most 1.3 ms (9%), less three exchanges' latency; at 12 GPUs
  weak (6% layer 1) at most 3 ms (5%). Most of the strong-scaling loss (73% at 16 GPUs, 55k cells per GPU)
  is the TENO kernel itself running less efficiently on fewer cells.

## Cell numbering

Each rank numbers its owned cells and each halo layer along the Hilbert curve of their vertex averages, and
its faces in the order those cells reach them (`mpi.md`, section 6; `[mesh] cell_order`), when the default,
`auto`, finds that this reads fewer cache lines (below). Before, a rank kept the global order, so a mesh file's
numbering set the memory locality of every kernel.

A100-80GB, ms per step (1 GPU / 4 GPUs of one node), the cases of the performance set at 1M cells and their
variants on triangles and tetrahedra. *Main* is the global (generated, x-first) order. *Random* and *z first*
number the same cells randomly and z first (y first in 2D), standing in for mesh files numbered without regard
for locality and for a numbering against the partition; Hilbert gives the same order whatever the input order.

| case | cells | main | Hilbert | random | z first |
|---|---|---|---|---|---|
| 2D TENO5, quadrilaterals | 1.05M | 19.9 / 6.39 | 19.3 / 6.19 | 52.8 / 10.7 | 20.0 / 6.36 |
| 2D TENO5, triangles | 1.05M | 19.0 / 6.05 | 19.1 / 6.10 | 49.6 / 9.60 | 19.0 / 6.05 |
| 2D MUSCL, quadrilaterals | 1.05M | 3.61 / 1.44 | 3.57 / 1.39 | 20.7 / 2.90 | 3.66 / 1.39 |
| 2D MUSCL, triangles | 2.10M | 5.87 / 2.00 | 5.89 / 1.94 | 34.9 / 6.03 | 5.94 / 1.98 |
| 3D TENO5, hexahedra | 885k | 150 / 40.9 | 143 / 39.6 | 294 / 72.2 | 151 / 40.7 |
| 3D TENO5, tetrahedra | 664k | 118 / 35.6 | 117 / 35.9 | 258 / 64.0 | 128 / 37.1 |
| 3D MUSCL, hexahedra | 885k | 13.1 / 4.27 | 13.4 / 4.36 | 75.9 / 10.7 | 13.1 / 4.20 |
| 3D MUSCL, tetrahedra | 1.05M | 33.0 / 10.4 | 36.8 / 11.3 | 198 / 32.3 | 31.2 / 10.3 |

- A numbering without locality costs 2-6x on one GPU (MUSCL on tetrahedra: 198 ms instead of 33-37 ms), less
  on four, where each rank's cells are fewer. The Hilbert numbering removes that dependence on the mesh file.
- Against the x-first numbering of a generated box, the curve is 1-5% faster for TENO on hexahedra and
  quadrilaterals and up to 11% slower for MUSCL on tetrahedra: along x, consecutive cells' neighbors are
  consecutive too, so a warp's loads coalesce, which a curve gives up at its turns.
- Tried and not kept: reverse Cuthill-McKee on the vertex-neighbor graph (2-10% slower than the curve, e.g. 155 /
  43.8 ms for 3D TENO5 on hexahedra and 39.1 / 11.9 ms for MUSCL on tetrahedra), and the curve over tiles of 512
  or 4096 cells with the input order within each (MUSCL on tetrahedra 36.3 and 34.2 ms, others as the curve).
- Restart files are bitwise identical (MD5) across the numberings and between 1 and 4 GPUs for every case.

### Which kernels gain and lose

Kernel time, x-first against Hilbert, one A100, nsys over the cases of the performance set (ms in all; the
kernels of more than 2% of the total):

| case | kernel | x-first | Hilbert | ratio |
|---|---|---|---|---|
| `air_3d` (MUSCL, LES) | all | 546 | 581 | 1.064 |
| | vertex least-squares gradient | 131 | 143 | 1.10 |
| | MUSCL cell reconstruction | 80.6 | 91.9 | 1.14 |
| | viscous flux | 82.5 | 86.9 | 1.05 |
| | convective flux | 41.1 | 45.8 | 1.11 |
| `teno5_3d` | all | 20,236 | 19,137 | 0.946 |
| | TENO reconstruction | 13,948 | 12,875 | 0.923 |
| | vertex least-squares gradient | 342 | 357 | 1.04 |
| `teno5_2d` | all | 6,770 | 6,613 | 0.977 |
| | TENO reconstruction | 5,060 | 4,887 | 0.966 |
| `muscl_2d` | all | 1,812 | 1,758 | 0.970 |
| | MUSCL cell reconstruction | 490 | 463 | 0.946 |
| | face flux sum | 89.0 | 83.5 | 0.938 |

Kernels that gather one layer of neighbors (gradients, MUSCL, fluxes) favor the x-first rows of a 3D box, whose
consecutive cells have consecutive neighbors; TENO's stencils of two or more layers favor the curve's compact
tiles, and so does every kernel in 2D, where x-first rows put a cell's y neighbors 1024 cells away. `flame_3d`
is `air_3d` plus order-independent chemistry. Neither order wins everywhere, so `auto` measures the stencil.

### The `auto` count

For blocks of 256 consecutive owned cells (every eighth block), `auto` counts the distinct 128-byte lines of a
cell field (16 cells in double precision) that the block reads when each cell reads the cells within L vertex
layers of it: L = 1 for first order and MUSCL, (p - 1) / 2 for TENO of order p. It counts the input order and
the Hilbert order and takes Hilbert when it needs fewer than 0.9 times the lines; a rank counts its own cells,
so ranks can differ, and the log says how many took the curve. The decision is the same for blocks of 64 to
1024 cells. Lines per owned cell, input / Hilbert:

| case | layers | input | Hilbert | taken |
|---|---|---|---|---|
| `sphere_3d` (Gmsh numbering) | 1 | 5.72 | 0.514 | Hilbert |
| `teno5_3d`, `probes_3d` | 2 | 0.771 | 0.525 | Hilbert |
| `teno5_2d` | 2 | 0.332 | 0.139 | Hilbert |
| `muscl_2d` | 1 | 0.199 | 0.139 | Hilbert |
| `air_3d`, `flame_3d` | 1 | 0.263 | 0.379 | input |

The count is a host loop over the node-to-cell map at setup (with the curve: +0.2 s for `air_3d`, +0.7 s for
`muscl_2d`, +2-3 s for `teno5_3d`); it reads no timings, so a mesh and its inputs always get the same
numbering.

### Against the numbering before

Performance set on one A100, cells per second, best of six runs in two interleaved rounds (before, after,
after, before), `perf_suite.py`:

| case | before | `auto` | change |
|---|---|---|---|
| `sphere_3d` | 19.1M | 24.4M | +28% |
| `teno5_3d` | 5.56M | 5.97M | +7.4% |
| `probes_3d` | 3.81M | 4.01M | +5.2% |
| `teno5_2d` | 51.7M | 53.5M | +3.5% |
| `muscl_2d` | 271M | 278M | +2.6% |
| `air_3d` | 52.3M | 52.5M | within noise (spread 4%) |
| `flame_3d` | 6.45M | 6.44M | within noise |
| chemistry (`h2o2`, `gri30`) | | | within 0.2% |

Restart files of `sphere_3d`, `teno5_3d`, `teno5_2d`, `muscl_2d`, `air_3d` and `flame_3d` are bitwise
identical before and after.

## Verification

Bitwise identity with `main` is checked on restart files (identical MD5) after N steps of: the
performance cases (2D TENO5 and MUSCL Riemann, 3D TENO5 Taylor-Green), the double Mach reflection
(triangles, Dirichlet and symmetry mirrors), axisymmetric Sedov, a viscous shock tube, the reactive
shock tube with MUSCL and TENO5 (2D, and a 3D variant: gas mixtures, primitive reconstruction), the 3D
explosion on hexahedra and tetrahedra, and the Taylor-Green vortex at orders 3, 4 and 6; for MUSCL also the
wedge, a viscous shock tube, the explosion with MUSCL on hexahedra and tetrahedra and a reacting H2/O2/Ar case
on $512^2$ cells; on 1, 2 and 3 ranks.
