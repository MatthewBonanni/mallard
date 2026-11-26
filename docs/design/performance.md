# Single-GPU performance and parallel scaling

Status: in progress. Baseline measured on `main` at v0.5.0-92 (before any change here); each change below
lists its measured effect. Results stay bitwise independent of the rank count, thread count and restarts,
and, unless a change says otherwise with a measured reason, bitwise identical to `main` (checked on the cases
under *Verification*; the contract and what it costs: section 16).

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

**2D** (#889, #999). The 2D setup runs one thread per cell (`CellSetup2D`), each with its own scratch in
global memory, so the threads in flight are the batch's cells. The kernel takes 134 registers per thread,
and Kokkos launches it in blocks of 128, three per SM: 41,472 resident threads on an A100. On CUDA a
batch is that one wave (`resident_threads()` repeats Kokkos's block-size choice from the kernel's
attributes), within a quarter of the free memory. Batches of 13,824 to 55,296 cells at orders 5 and 6
took 6.25 / 5.04 / 5.17 / **4.17** / 4.69 / 4.48 s and 19.7 / 19.3 / 15.6 / **12.2** / 14.5 / 14.7 s
(13,824 cells, the earlier byte budget's batch, then 27,648, 41,472, 51,840 and 55,296 cells): past one wave, a partial
second wave lengthens every batch.

At orders 5 and 6 a per-thread `clock64` profile of that kernel put 73-82% of the thread time in the
smoothness-indicator matrix, which added every term to the cell's interleaved scratch in global memory (a
load and a store per term, for each of up to 231 entries, quadrature point and derivative), and 16-22% in
the fits. Each thread now sums the matrix one row at a time in registers (`smoothness_rows<R>`, the terms
whose derivative vanishes left out), and on the device the Householder sums, the back-substitution and
the Lebesgue constant's coefficients likewise column by column; every entry adds the same terms in the
same order, so the tables are bitwise the same (in builds whose CUDA compiler is `nvcc_wrapper`, which
compiles the TENO sources without contracting FMAs; not yet with the host compiler as
`CMAKE_CXX_COMPILER`, #1184). The kernel then takes 255 registers and 2.3 KB of local memory per thread,
two blocks per SM; bounded to four blocks of 128 per SM (128 registers, 2.8 KB) it holds 55,296 threads
and is faster at every order. The fits are now about 70% of the thread time (#1185). The host backends
run the same row sums; their pseudo-inverse keeps updating whole rows, which vectorize.

"TENO setup" tables phase, one A100 (CUDA 13.5, 16 OpenMP threads, `n_steps = 2`); the host setup is
the same code on the 16 host threads; restart files are bitwise identical:

| tables (s) | order 3 | order 4 | order 5 | order 6 |
|---|---|---|---|---|
| double Mach, 2048 x 512 x 2 triangles (2.1M), host | 6.48 | 12.8 | 23.5 | 57.8 |
| double Mach, device, batches from a byte budget (at most 27,648 cells) | 1.03 | 1.66 | 5.12 | 19.3 |
| double Mach, device, batches of one wave (41,472) | **0.89** | **1.47** | **4.18** | **12.2** |
| box, 1024 x 1024 quadrilaterals, device, byte budget | 0.86 | 1.35 | 3.38 | 17.4 |
| box, device, one wave | **0.69** | **1.11** | **3.25** | **10.7** |
| double Mach, device, one wave, again (same session as the next row) | 0.92 | 1.45 | 4.43 | 13.5 |
| double Mach, device, sums in registers, four blocks per SM (55,296) | **0.78** | **1.07** | **2.15** | **4.83** |
| box, device, one wave, again | 0.69 | 1.11 | 3.23 | 11.4 |
| box, device, sums in registers, four blocks per SM | **0.51** | **0.75** | **1.32** | **3.42** |

On the 28 threads of the Threads backend the 2D tables take 2-43% less (box 1.63 / 3.30 / 9.34 / 25.6 s
before, 1.59 / 2.66 / 5.47 / 14.6 s after; double Mach 2.56 / 4.19 / 10.8 / 26.8 s before, 2.51 / 3.56 /
7.75 / 16.0 s after) and the 3D tables the same (Taylor-Green 96^3 at orders 3, 5 and 6: 5.74 / 11.2 /
25.0 s). The tables' SHA-256 (the cache file after its header) is the same before and after, on the A100
and on the host, for every case above.

The workspace still does not grow with the mesh (meshes of at least 16,384 cells share the batch); one
wave of 41,472 cells takes 1.5 GB at order 3 and 3.6 GB at order 6, one of 55,296 a third more. On the host, the packing of the shared tables
took 7-37 s on these meshes, more than the tables; it now runs on the device (#1031, below).

**Shared tables** (#1031). With shared tables (#994, #972), the packing copied every batch's tables to
the host, where each cell's pseudo-inverse was ranked, hashed and compared with its shape's, and the
rows of basis means and moments were then downloaded and matched the same way. Now the device ranks the
slots (an insertion sort per cell, stable like the host's), hashes each table and row, and compares the
cells with their shapes' tables; the host only looks up the hashes in cell order and receives the
tables that its shapes keep. One A100 (CUDA 13.5, 16 OpenMP threads, `n_steps = 2`), the same machine for
both; restart files at step 2 are bitwise identical:

| packing / numerics setup (s) | order 3 | order 4 | order 5 | order 6 |
|---|---|---|---|---|
| double Mach, 2048 x 512 x 2 triangles (2.1M), host | 25.7 / 29.4 | 36.7 / 49.6 | 70.2 / 87 | 108 / 168 |
| double Mach, device | **0.87 / 2.64** | **0.91 / 3.25** | **1.10 / 6.54** | **1.22 / 21.7** |
| box, 1024 x 1024 quadrilaterals (1M), host | 16.6 / 20.5 | 23.1 / 30.9 | 33.8 / 44.6 | 62.7 / 93 |
| box, device | **0.45 / 3.05** | **0.53 / 3.53** | **0.54 / 6.23** | **0.65 / 20.6** |

The numerics setup is now the tables phase plus 1-3 s; on the host it was the tables phase plus 25-150
s. Per family (central and sector stencils) and per 2.1M cells, the device ranking takes 10-70 ms, the
comparisons 25-100 ms and the host's matching 0.1-0.25 s; the rest of the packing is the cells' stencils,
as without sharing. Packing times vary by up to 3 s from run to run with the host's load (one box run at
order 5 took 3.2 s; the table gives the repeat).

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
- The troubled passes' scratch holds an eighth of the cells (`troubled_capacity`); more troubled cells are done in
  further rounds. Below a capacity of 1, every round, the first included, recomputes its cells' central coefficients with
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
before 12.4, and HIP and SYCL, whose graphs have none) take their steps kernel by kernel, one at a time, the host
reading each IF node's flag.

Where NCCL communicates over a network, it adds host nodes to the operations it captures, which a conditional
node's body cannot hold; a capture of NCCL operations under one is tried on every rank before the first step.
If it fails, the step graph gates every IF body instead. Gating only the bodies with NCCL operations and keeping
IF nodes for the others does not capture: an IF node inside a gated body ends the capture of the graph around it
and starts another one of it, and NCCL (2.32) fails the next operation captured there ("dependency created on
uncaptured work in another stream"). A gated body runs in every
launch: a kernel copies its flag, and the gate of the code around it, into a gate of its own, which every kernel
of the body tests, so the body's kernels follow the flag as it was when the body began even where the body clears
it. Every top-level kernel launches through `src/common/gated.h` (a CMake check rejects Kokkos's own launches),
which tests the innermost gate; a reduction that does nothing leaves its destination as it was, by copying it
into the result in the reduction's final step. Copies and fills there are kernels too (the tests'
`StepGraphGuard` reports others). On CUDA a thread of a parallel_for that does nothing exits rather than
returns: a return would leave the rest of the kernel in a divergent region, which costs TENO's kernels about
0.4% of a step also where nothing is gated. The NCCL operations always run, on the same buffers on every rank,
since the flags come from values reduced over the ranks. Nothing outside a gated body may therefore read a buffer
that an NCCL collective reduces in place: while the body is gated, a `SUM` buffer that its kernels no longer refill
grows by the rank count at each launch. A gated body costs its kernel launches (and its NCCL
latency) whether or not it is due: in every step for an output due only every few steps (a probe's reductions,
say), and in each launch after a stop, `step_queue` - 1 times per stop, instead of nothing. Off CUDA, the tests'
launches of the code a capture records gate every body.

Triggers that read a probe's value are evaluated on the device too. The probes they read are staged, and at
their writes, before the triggers (and after the statistics' samples, which the host's order puts first), the
graph reduces them into the trigger variables with the host's arithmetic (`probe_reduction.h`): every sum's
per-sample terms, each product rounded on its own, are summed exactly and rounded once (`ExactSums`, reduced
over the ranks with `comm::allreduce_device`, on NCCL in the graph), the extrema are reduced with `MIN` and
`MAX`, and a one-point probe's values go to every rank as bit patterns summed with zeros, so the values equal
the host's `Probe::reduce` bit for bit on any number of ranks. The host still writes the files from the staged
samples. A time-averaged probe's running means are kept on the device as well, per local sample: the clock
kernel weighs each sample as `Probes::sample_average` does (`w = t - t_last`, `a = w / W`), and every mean takes
`mean += a (x - mean)` with the product rounded on its own, as the host's means of the same staged samples do,
so the reductions of the device's means are those of rank 0's. Whenever the host samples a probe itself (a step
it stops at, a restart, averages started afresh), the device's means are copied from the host's on resuming.
Probes of `[statistics]` outputs read the device's averages, which the graph samples before the probes and
the triggers; at a step the device stops at after sampling them (a trigger that fires, an output still being
written), the record says so, and the host takes the step without sampling them again (writing the spectra's
staged sample instead). Confidence intervals `CI_<average>` are computed at each sample's cell from the
device's `BATCH_M2_<average>` and closed batches with the host's arithmetic (`batch_means::half_width`), the
Student's t quantile read from a window of 1024 the host fills on resuming once the batches closed have gone past
half of it; the horizon stops the queue before the means' samples could close more batches than the window holds.
Probes of moving and `where` regions reduce the cells covered at the write, the first `count()` of the
selection, weighed by their `f V` (`ExactSums::sum(terms, count)`, a count only the device knows). Their time
averages are kept per owned cell, as the host's per global cell (`accumulate_cells`): each cell held or covered
takes `mean += a (x - mean)` of `f V` and of `f V q`, `x` zero where the sample does not cover it, and the
reduction compacts the cells held and divides as `Probe::mean_fields` does. The sums do not depend on the order
of the terms, so the per-cell layout gives the host's values on any number of ranks. Isosurfaces (the weights of
those sampled per cell come from the host's extraction, and their averages sum each cell's points in the host's
order) and probes reading cell data are left to the host:
the device stops at their writes, the host writes them and evaluates the triggers, and on resuming copies the
values into the device's trigger variables (rank 0's, sent to every rank with one host `allgatherv`).

`[statistics.convergence]` values that a trigger reads are updated on the device too, at the rows of their signals,
between the probes and the triggers. The integrals' and forces' values are the trigger variables the monitors set;
probe signals are reduced into theirs like the probes a trigger reads. Each signal's history (times, values and the
prefix sums of its values less the first, and of their squares) lives in device memory, and one kernel, under an IF
node on whether any signal has a sample (and itself checking that flag, since graphs without conditional nodes run
every IF body), does `Convergence::add` and `update`: the MSER-5 checks, the start of the stationary window, the
integrated autocorrelation time and the batch means of the mean and the variance, in the host's order of operations
and compiled without contracted multiply-adds, so its values are the host's bit for bit. Its team sums each of
MSER's batch means and each lag's autocorrelation on one thread, as the host does, and Sokal's window closes over
chunks of 256 lags. The host updates its own estimate as it takes each record, which writes the file and the
restart histories; the device keeps the window's start, the `[statistics]` averages' start after the step's
samples, in the record, since the host's copy of the statistics' clock may be steps ahead. On resuming, the host
copies its state to the device, and its histories' samples since the last copy if it added any. The histories are
whole, with no wraparound: their device memory holds at least 4096 samples, twice the longest history when staged,
at 32 bytes per sample and signal (128 KiB per signal at least, about 1 GiB per signal at 10^7 samples, besides
MSER's batch means and the lags' chunk). A history fills it only by running that long without a stop; the device
then stops at that row (`CONVERGENCE`) and resumes with twice the room, in a new graph; while it restages, the old
and the new memory are held at once, about three times the old.

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
staged, so every rank stages, or stops for the host (`OUTPUT`), at the same steps. The staging also evaluates
the `[sources]` at the device's `t` into a device array, and copies the averaged spectra, the phase averages and
their clocks and bin weights, from which the host writes the spectra file and the restart attributes and
arrays. A region that moves or follows the solution takes its cells from its fractions, computed under an IF
node on its writer's own flag at the device's `t` (after the cell states, for a region of the solution) and
copied into pinned memory; the host computes them the same way, at its `t`, when it writes the output itself.
With fused chemistry the staging flushes the pending half step first, as for the monitors (below), and also
copies the kept, unflushed state, which the host swaps into the mirrors while it writes restart files. Outputs
of builds without conditional nodes stop the device at their steps. What the host copies from the device for an
output it writes itself also goes through pinned mirrors: the same ones for snapshots and restarts, and their
own for probe values, a moving region probe's cells and the spectra.

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

Probes written every few steps or by time, and the samples of time-averaged probes, are evaluated in the graph
as well, except those of surfaces or curves of the solution that are clipped, made of more than one
piece or cut from two conditions, or lie on a mesh with curved cells (below). A kernel sets one flag per probe and one for any; under the latter the
derived fields the due probes read are updated (the cell states, transport and gradients, as the host's
`write_probes` updates them), and under each probe's flag its samples are evaluated and copied into a pinned
array per record slot; the columns of `[statistics]` means, covariances and phase averages are copied there
from the device's averages at each sample's cell. The host gathers them over the ranks, reduces and writes them from that slot as it
takes the record. A probe whose writes need its reductions alone (no samples written, no time averages, spectra,
motion or isosurface) is reduced in the graph instead, as the probes triggers read are, and only its measure and
reduced values go to the record slot: copying a cut plane's 1.5M samples into pinned memory at each write made
the perf set's `probes_3d` two to five times slower than `teno5_3d` on an A100 (#1402). Point values come from
`FaceReconstruction::point_values` with `by_cell`: the probes' points come by cell, and TENO fits each run of
one cell once, a team of threads sharing out its coefficients, each summed in the per-point fit's order, so the
values are those of a fit per point bit for bit. The intervals are copied to the device whenever it resumes, so control commands that change
them take effect at the host's next step.

A moving region's fractions, and a `where` region's (from the levels of its conditions on the cell states, which
an IF node on a flag for any due probe of the solution updates first, after the pending chemistry flush), are
updated at the device's `t`, and a scan selects the owned cells it covers, with their count, on the device. That happens before the outputs and the statistics' samples are staged, at the steps the probe is due: if on
some rank the count is above the capacity of the probe's staging, the device stops at that step (reason
`PROBES`, reduced over the ranks), the host writes it as it writes unstaged probes, and the step graph is built
anew with each moving probe's staging at twice the cells it covers then (at least 64, at most the owned cells).
An isosurface (or another surface or curve of the solution) sampled as cell averages without normals has one
sample per cell it crosses, whose values are the cell's: the same scan selects, at the device's `t`, the owned
cells the host would read levels in (`region::CrossedLevels`), and their values are evaluated there and copied
with the cells' levels into the record slot (with the same capacity and `PROBES` stop). The host extracts the
elements from those levels as it takes the record, with the code it runs on the levels it selects itself, so the
samples, their order and weights are its own bit for bit, and each sample takes its cell's staged values.
Probes of fluxes through it, of point values or of gradients need values at each quadrature point. When the
surface is cut from one condition alone (`region::single_level`: one piece, unclipped, on straight cells), the
device cuts each crossed cell's level on its simplices as the host does (`region::cut::level_cut`, the host's
double arithmetic in its order, without contraction, and an exact angular order instead of `atan2`; the host's
`order_about` uses the same), counts the points per cell, scans their offsets and fills them into arrays of twice
the points at the stage, and at least the surface's points per crossed cell over the ranks times the cells staged
(at least 256; more stop the device with `PROBES` as above), samples them as
`Samples::in_cells` would, and stages their values with the levels. The host cuts the staged levels again at
the record and takes the staged values point by point, after checking that it found as many points. Other
surfaces of the solution stay with the host.
Otherwise the samples are evaluated over arrays the size of the owned cells, each thread past the count
returning at once, and
the covered cells, their `f V` and the count are copied into the record slot with the samples. The host orders
them by global cell, as it does for the samples it selects itself, as it takes the record. On one A100
(`air_3d.toml`, a sphere of radius 0.03 written every 10 steps) the sampling of the cells its surface crosses
is nearly all of the region's work; the selection, evaluation and copies take under 0.1 ms per write.

With fused chemistry (`[chemistry] fuse_half_steps`), each step ends with a half step still pending, which the
host applies to a copy (`Solver::flush_chemistry`) for whatever reads the state. In the graph, before the first
of the monitors, statistics, probes, outputs and progress row due at a step, a kernel sets a flag if the launch has not
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

With that rounding explicit, the smooth pass stages its entries with the conversion in the loading thread, bitwise
equal to the unstaged loop: TENO5 in rounds of 10 entries, TENO6 in rounds of 5, which keep two teams per SM
(#1004: 4.61M cells/s against 3.60M unstaged for the order-6 Taylor-Green vortex on $96^3$ hexahedra).

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

- **Shared basis means and moments (#972).** Next to its entries' cells, a shared-table cell kept its basis
  means and its central moments (2D: smoothness-indicator matrix), the same for every cell of a shape up to
  round-off. They are now one row per cell shape, found the same way as the shared tables (on a grid of 2^-30 of
  each row's magnitude, the lowest global id's row, agreed over the ranks) and reached through a per-cell index;
  the scale stays per cell, as a shape index would cost as much. Bytes per cell of the tables after the setup and
  cells per A100-80GB (`mallard check --memory a100-80` on the A100, $16^3$ hexahedra and $10^3 \times 6$
  tetrahedra, periodic):

    | case | tables, before | tables | cells per A100-80GB, before / after |
    |---|---|---|---|
    | TENO3, hexahedra | 774 B | 626 B | 18.2M / 18.8M |
    | TENO4, hexahedra | 1.26 kB | 834 B | 15.7M / 17.1M |
    | TENO5, hexahedra | 1.97 kB | 1.04 kB | 10.2M / 11.6M |
    | TENO6, hexahedra | 3.21 kB | 1.46 kB | 8.39M / 10.3M |
    | TENO3, tetrahedra | 656 B | 508 B | 18.6M / 19.2M |
    | TENO4, tetrahedra | 1.06 kB | 633 B | 16.3M / 17.8M |
    | TENO5, tetrahedra | 2.06 kB | 1.12 kB | 12.4M / 14.5M |
    | TENO6, tetrahedra | 3.07 kB | 1.32 kB | 10.1M / 13.0M |

    The smooth pass reads fewer distinct rows: the $96^3$ Taylor-Green vortex of `teno5_3d` on one A100 runs at
    8.20M cells/s against 8.08M at order 5, and 3.83M against 3.60M at order 6. Results change once at round-off
    against per-cell rows: restart files after 5 steps of a blast on $16^3$ hexahedra and tetrahedra differ by at
    most 2e-15 (absolute, fields of order 1). Unstructured meshes have a shape per cell and keep a row per cell.

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
| species slots in the face states | 8,638 / 9,100,971 | 8,497 / 9,252,578 | 8,678 / 9,059,023 | 8,537 / 9,209,223 | 10,641 / 7,289,471 | 9,778 / 7,932,252 | 15,571 / 4,981,336 |
| image tables of transmissive faces | 8,518 / 9,229,863 | 8,416 / 9,340,971 | 8,558 / 9,186,721 | 8,456 / 9,296,787 | 10,523 / 7,370,967 | 9,700 / 7,996,390 | 15,453 / 5,019,363 |

- **Ghost weights of boundary faces.** The vertex gradient (viscous fluxes, the hybrid flux's sensor, MUSCL)
  kept a weight for every face of every cell, though only boundary faces have ghosts; it keeps them for those
  faces alone, with each cell's list of them: 135 bytes less per hexahedron and 88 per tetrahedron here, about
  140 and 92 on large meshes.
- **Species slots in the face states.** In gas mixtures, each face's species fluxes out of either side (its
  slots) are written after the convective fluxes, the last readers of the face states in a stage, and read
  before the next stage's reconstruction writes them again: they share the face states' memory, which is the
  larger of the two below about 45 species at TENO5 on hexahedra (35 on tetrahedra, 5 with MUSCL). The
  saving is the smaller of the two: 48 bytes per species per hexahedron (294 for 6 species) and 32 per
  tetrahedron, at most the face states' 2.2 / 1.2 KB (2,228 bytes with GRI-Mech, 15% more cells).
- **Image tables of transmissive faces.** The image cell, image face, its side, orientation, matching
  quadrature points and offset of transmissive boundary faces were stored for every face (43 bytes in 3D);
  every face now has the row of its transmissive face in tables of those faces alone, -1 for the others (4
  bytes): 121 bytes less per hexahedron and 80 per tetrahedron, in every run.

Together: 3.0% more cells on hexahedra and 2.0% on tetrahedra for a perfect gas, 5.2% and 3.8% for the
6-species flame, 16.1% with GRI-Mech. What remains beyond the tables, and why it stays for now: #978.

#### Stage scratch of gas mixtures (#552)

A gas mixture's right-hand side runs in this order: the reconstruction (the TENO troubled passes' coefficients, or
MUSCL's gradients and limiters, and the scalar reconstruction's work memory) writes `face_thermo`; the convective
fluxes read it and write `face_mdot`; the species slots read `face_mdot`; only then does `update_transport` fill
`cell_diffusion`, `transport_values` and `transport_gradients` for the viscous fluxes. Every other reader of those
three (the time step's transport, the eddy viscosity of LES, TFLES and PaSR, output, probes, forces, derived fields)
runs after a call that fills them. Gas mixtures never start a reconstruction before the time step
(`exchanges_partition_faces`), so the two groups are never live together. They share one allocation,
`stage_scratch`, as large as the larger group: buffers placed one after another, 256-byte aligned. The memory
tracker counts it once under that label, so `mallard check --memory` stays exact; view sizes are checked as
for any allocation. The reconstructions are told not to allocate their own scratch, so the setup never holds both.

Not shared: `production` (live with the transport buffers in the TFLES reaction sensor and output), a perfect
gas's `viscous_gradients` (its reconstruction may start before the time step's gradients are read), and the
chemistry's work memory (on the thread's stack).

| bytes per cell, cells per A100-80GB | `main` | stage scratch | cells |
|---|---|---|---|
| propane flame, TENO5, hexahedra | 9,590 / 8,090,532 | 9,220 / 8,413,483 | +4.0% |
| propane flame, TENO5, tetrahedra | 8,760 / 8,854,176 | 8,390 / 9,242,432 | +4.4% |
| propane flame, MUSCL, hexahedra | 4,190 / 18,514,190 | 3,840 / 20,183,512 | +9.0% |
| GRI-Mech, TENO5, hexahedra | 14,500 / 5,342,876 | 12,300 / 6,321,797 | +18.3% |
| GRI-Mech, MUSCL, hexahedra | 12,100 / 6,435,177 | 10,600 / 7,333,000 | +14.0% |
| 2D propane flame, MUSCL | 2,100 / 37,226,797 | 1,870 / 41,879,047 | +12.5% |

With GRI-Mech at TENO5, the 2,882 bytes of `stage_scratch` replace 1,760 (TENO), 891 (`face_thermo`), 223
(`face_mdot`) and 2,248 (transport). Perfect gases and runs limited by the setup or by view sizes (2D TENO5) are
unchanged. Restart files after 4 steps are bitwise identical to `main` for 31 runs: TENO and MUSCL, 2D, 3D and
axisymmetric, GRI-Mech, Soret, PaSR, dynamic LES, double flux, MPI on 3 ranks and restarts. On one A100 (medians
of alternating runs) the time per step does not change: `flame_3d` 39.1 / 39.3 ms, with TENO5 217 / 217 ms,
with GRI-Mech at TENO5 1,567 / 1,558 ms, `teno5_3d` 105 / 104 ms; the device's peak falls by 82 MiB, 86 MiB and
482 MiB on the three flames (221,184 cells).

#### Planar face points from the face nodes (#978)

3D TENO stored every face's quadrature points (face, q, dim), read by the face evaluation of every pass. On meshes
without curved faces the kernels now place them from the faces' nodes, and the stored points are freed after the
setup. A table holds each face's four node indices (a triangle's third repeated, which marks it), next to a copy of
the node coordinates in `LayoutRight`, so a face's nodes take one 16-byte load and four coordinate rows; both replace
the mesh's node lists and coordinates, which TENO kept in 2D anyway. `planar_face_point` evaluates
$v_0 + a (v_1 - v_0) + b (v_2 - v_0)$ on triangles and $\sum_k N_k v_k$ on quadrilaterals with every product
rounded on its own (`rounded_product`), as the setup did on the host, so the points and the results are bitwise
those of the stored points; the rule's padding rows repeat the last point, as the stored padding did. A thread loads
a face's nodes once for the points it evaluates on that face: `SmoothTeamPass` gives each of a cell's threads a run
of consecutive points rather than every fifth one, and the troubled passes load them per face. The image points of
transmissive faces are placed on the host for those faces alone.

`mallard check --memory a100-80 --steps 1` (CUDA 13.5 build on an A100) of the 3D TENO5 Taylor-Green vortex on
$32^3$ hexahedra and $16^3 \times 6$ tetrahedra, `main` at a7a5432e:

| | hexahedra: `main` | change | tetrahedra: `main` | change |
|---|---|---|---|---|
| bytes per cell (setup's peak) | 6.47 kB | 5.87 kB (-600 B) | 6.51 kB | 6.20 kB (-310 B) |
| most cells per A100-80GB | 12,103,636 | 13,324,340 (+10.1%) | 12,018,308 | 12,616,797 (+5.0%) |

On one A100 (CUDA 13.5, TENO5, 300 steps, median wall time per step of the progress rows from step 100, five runs
each alternated with `main`'s on the same GPU; spread = (max - min) / median):

| case | `main` | change |
|---|---|---|
| Taylor-Green, $96^3$ hexahedra | 101.49 ms (spread 0.1%) | 100.69 ms (-0.8%, spread 0.1%) |
| explosion, $96^3$ hexahedra | 233.75 ms (1.3%) | 183.94 ms (-21.3%, 2.6%) |
| Taylor-Green, $48^3 \times 6$ tetrahedra | 87.38 ms (0.4%) | 86.81 ms (-0.7%, 0.1%) |

Restart files after 100 steps are bitwise those of `main` (`mallard diff`) in the three cases and in the hexahedral
Taylor-Green vortex at $48^3$ with `viscous_gradients = "reconstruction"`, whose viscous flux places the points as
the TENO kernels do. An earlier measurement (100 steps, three runs) gave the same picture.

Nsight Systems over the 100 steps: `SmoothTeamPass` 8.33 s / 8.23 s (Taylor-Green, hexahedra), 7.08 s / 7.00 s
(tetrahedra), and `TroubledSelectTeamPass` 6.40 s / 3.53 s in the explosion, whose threads (one per face and
variable) each read the stored points of their face again for every point. A first version that read the nodes
through the mesh's offsets per point was 4-10% slower than `main`.

### 16. Cost of bitwise reproducibility (#429)

Three places where the bitwise contract constrains the arithmetic were relaxed one at a time on a branch that is
not merged, and measured on one A100 (CUDA 13.5, best of two runs; setup times are the TENO setup's `tables`
line of the run summary):

- **TENO setup with FMA contraction and tree sums.** `teno_setup.cpp` and `teno.cpp` compiled with contraction
  (no `--fmad=false`, `-ffp-contract=off`), then also the setup's dot products (`strided_dot`, every sum of the
  Householder QR and the back substitution) with four partial sums added pairwise instead of one sequential sum.
- **Plain sums instead of `ExactSums`.** One `Kokkos::Sum<double>` per column, without the max reduction and the
  rounding to fixed point.
- **Atomic scatter of the face fluxes.** `face_flux_sum` replaced by a kernel over the faces that adds each flux
  to its two cells with `atomic_add` (the reduction order then depends on scheduling).

Unconstrained chunking in the team TENO kernel (section 1) was not measured on its own: `teno.cpp` no longer
contracts, so runtime chunk bounds would give the same bits as compile-time ones, and the compile-time bounds
stay because they keep each thread's coefficients in registers. Its cost in bits is covered by the contraction
row below.

| case | change | TENO tables | throughput (cells/s) | result vs `main` after N steps |
|---|---|---|---|---|
| `teno5_3d` ($96^3$ hexahedra) | none | 6.35 s | 8.22M | - |
| | FMA contraction | 6.10 s (-3.9%) | 8.26M (+0.4%) | all cells differ, max 4.5e-13 of each field's max (N = 100) |
| | FMA + tree sums | 6.07 s (-4.4%) | 8.24M (+0.2%) | max 4.9e-13 |
| | atomic scatter | | 8.20M (-0.3%) | max 4.7e-13 |
| 3D explosion, $55^3 \times 6$ tetrahedra | none | 24.4 s | 5.90M | - |
| | FMA contraction | 23.5 s (-3.9%) | 5.91M (+0.2%) | 52% of cells differ, max 1.8e-15 (N = 20) |
| | FMA + tree sums | 23.6 s (-3.6%) | 5.85M (-0.8%) | max 1.8e-15 |
| `sphere_3d` mesh with TENO5 (176,243 tetrahedra, 4,000 prisms) | none | 33.4 s | 3.24M | - |
| | FMA contraction | 32.0 s (-4.2%) | 3.22M (noise) | 67% of cells differ, max 6.0e-14 (N = 20) |
| | FMA + tree sums | 32.2 s (-3.5%) | 3.21M (noise) | max 5.9e-14 |
| `teno5_2d` | none | 3.35 s | 76.2M | - |
| | FMA contraction | 3.35 s | 76.4M (+0.3%) | 30% of cells differ, max 2.0e-14 (N = 300) |
| | atomic scatter | | 75.6M (-0.8%) | max 1.8e-14 |
| `air_3d` | atomic scatter | | 45.9M vs 46.0M (noise) | max 8.4e-14 (N = 100) |
| `air_3d` with `les.dynamic` | plain sums | | 35.31M vs 35.27M (+0.1%) | 71% of cells differ, max 2.5e-14 (N = 100) |
| `flame_3d` with `efficiency = "dynamic"` | plain sums | | 7.80M vs 7.79M (+0.2%) | every cell differs, momentum by up to 2%, chemistry sub-steps by 60% (N = 100) |

- **Setup.** Contraction saves 4% of the fits on every cell type; the tree sums add nothing beyond it. On these
  meshes the setup is dominated by the host's packing of shared tables (#1031), so the whole setup changes by
  less than 1%. The price would be the portable cache: with contraction the device's tables no longer equal the
  host's (cells redone on the host, CPU builds, a cache written by the other), and 13-31% of the cache's bytes
  change.
- **Per step.** `teno.cpp` already writes its FMAs explicitly where they count, so contracting the rest gains
  0.2-0.4%. The atomic scatter is slower than the gather on every case: it writes each cell's sum through
  atomics in global memory twice per face, where the gather sums in registers and writes once
  (`face_flux_sum` is 99 us per stage on `teno5_2d`, 2.1% of the stage).
- **Exact sums.** `ExactSums` takes 59 us per call for the dynamic constant (2 columns, 221k cells) and 107 us
  for the dynamic TFLES efficiency (4 columns), against 26 and 49 us for plain sums (Nsight Systems): the
  difference is 0.5% and 0.2% of a step. On CPUs (48 threads as 1, 2, 4 or 8 ranks, `air_3d` with
  `les.dynamic`) it stays inside the run-to-run spread (5%). Distributed, exact sums add one all-reduce of the
  column maxima per call to the one of the sums. Over NCCL on A100s (main 9da4ab1e7, 221k cells, median ms per
  step of two runs, exact / plain):

    | GPUs | `air_3d` with `les.dynamic` | `flame_3d` with `efficiency = "dynamic"` |
    |---|---|---|
    | 1 | 5.27 / 5.23 (+0.7%) | 27.16 / 27.12 (+0.1%) |
    | 2 | 3.25 / 3.23 (+0.8%) | 20.57 / 20.56 (+0.1%) |
    | 4 | 2.36 / 2.30 (+2.3%) | 17.12 / 17.11 (+0.1%) |
    | 8 | 1.65 / 1.60 (+3.2%) | 15.32 / 15.24 (+0.5%) |
    | 7 + 7, two nodes | 1.81 / 1.71 (+6.0%) | 15.75 / 15.61 (+0.9%) |

    On one node the difference stays within 25-50 us per step, as on one GPU; across two nodes it grows to 100-140
    us, the extra all-reduce's network latency, which shows only for the dynamic constant at 16k cells per GPU.

    Since #1097 the sums need no maxima: each term is split into 32-bit limbs indexed by its exponent, added
    exactly as integers, and the limbs go through one SUM all-reduce, after which the total is rounded once (so
    sums are now exactly rounded, and their last bits differ from those of earlier versions). A window of four
    limbs above the largest term of the rank, found without communication, stays in registers; the rest go to
    the limbs by integer atomics. Alternated against main a7a5432e4 (old exact sums and plain sums), 400 steps,
    median ms per step from step 300 with [min-max] over 5 reps on one node and 7 reps on 1 + 1 GPUs on two
    nodes, old exact / new exact / plain:

    | layout | `air_3d`, `les.dynamic`, 16k cells per GPU | `air_3d`, `les.dynamic`, 221k cells | `flame_3d`, `efficiency = "dynamic"` |
    |---|---|---|---|
    | 1 GPU | 1.022 [1.018-1.025] / 1.006 [1.003-1.016] / 1.001 | 5.265 [5.259-5.273] / 5.270 [5.266-5.276] / 5.243 | 27.65 [27.64-27.70] / 27.62 [27.57-27.64] / 27.57 |
    | 2 GPUs | 1.193 [1.187-1.202] / 1.177 [1.173-1.184] / 1.170 | 3.248 [3.243-3.254] / 3.242 [3.236-3.254] / 3.216 | 21.08 [21.06-21.14] / 21.05 [21.03-21.09] / 21.03 |
    | 1 + 1, two nodes | 1.336 [1.329-1.348] / 1.318 [1.291-1.330] / 1.296 | 3.477 [3.448-3.496] / 3.461 [3.443-3.483] / 3.452 | 21.80 [21.73-21.84] / 21.81 [21.08-21.99] / 21.75 |

    New exact sums are faster than old ones in every alternated pair of the 16k case on two nodes (median -1.0%,
    18 us per step) and by 16 us per step at 16k cells on one node; on the other cases the difference stays
    within the run-to-run spread.
    The flame shows what the plain sums would cost: last
    bits of one scalar per step grow to percent-level differences within 100 steps, so a run on another rank count,
    or a restart on another partition, could no longer be checked against the original.

**Contract.** Within one build and version, results stay bitwise independent of the rank count, partition,
thread count, step queue depth, CUDA graphs and restarts: it costs at most 0.5% per step here, and it is what
makes restarts, partitions and GPU against CPU checkable by `mallard diff` with no tolerance. Against the previous
`main`, a change may move the last bits when it brings a measured speedup, with tolerance checks and an unchanged
verification manual instead of bitwise identity ([CONTRIBUTING.md](../../CONTRIBUTING.md#reproducibility));
none of the three relaxations above qualifies, so none was kept, and the TENO setup keeps its host and device
tables equal.

### 17. Convective flux without contraction (#1253)

The Riemann solvers are inline functions, and where nvcc and Clang contract their multiplies and adds into
FMAs depends on the kernel they are inlined in: the same flux rounded differently in `convective_flux` and in a
reconstruction kernel computing it on its own (#965), by about 1e-15. The sources that call them
(`solver_convective_flux.cpp` and three tests) are compiled without contraction, as `teno.cpp` is, and the
solvers, the flux functors and the low-Mach and hybrid blends write their FMAs explicitly (`fmadd`, `fma_dot`):
every product added to something, as the compiler would have fused it. Configuring fails if another source
names `riemann::` without the same flags. Measured on one A100 (CUDA 13.5, Release, three runs each;
`convective_flux` is the mean over 60 (`teno5_3d`, 20 steps) or 300 (`sphere_3d`, 100 steps) launches, Nsight
Systems):

| case | `main` | no contraction, no explicit FMAs | this change |
|---|---|---|---|
| `teno5_3d`, `convective_flux` | 3.46 ms | 3.71 ms (+7.2%) | **3.18 ms (-8.0%)** |
| `teno5_3d`, step | 105 ms (8.41-8.44M cells/s) | 106 ms (8.36-8.37M) | **104 ms (8.48-8.51M)** |
| `sphere_3d` (MUSCL, 500 steps), `convective_flux` | 94.7 us | 98.2 us (+3.7%) | **88.0 us (-7.0%)** |
| `sphere_3d`, step | 7.33-7.36 ms | 7.30-7.40 ms | 7.32-7.35 ms |

Without contraction and without explicit FMAs the flux is 4-7% slower; with them it is 7-8% faster than `main`,
whose contraction does not fuse every product (for example `0.5 W[0] |u|^2` into the energy, or the HLLC star
state). Results change in the last bits once: after 20 steps of `teno5_3d` the cells differ from `main` by at most
3.5e-15 of each field's maximum (`mallard diff`), 1.5e-12 for the momentum along z, which starts at zero. The block-face flux of #965 (design 1, computed in the TENO smooth
pass) is now bitwise equal to `convective_flux`: on CPUs (`teno5_3d` at $32^3$, 6 steps) and on the A100
(`teno5_3d`, 20 steps).

### 18. Mesh build memory (#964)

Peak heap in use (glibc `mallinfo2`, bytes per local cell, the highest rank) while building 1M-cell meshes, and
build times of rank 0 (s, best and worst of two runs), on `main` and with the change; Linux x86_64, GCC, Release,
double, 2 threads, MPI on 1 or 3 ranks (`DistributedMesh`, as `mallard` builds a mesh for a run). Generated
meshes, and a Gmsh file of the same box; "built" is the finished mesh.

| peak / built, B per cell; time, s | serial | | distributed, 1 rank | | distributed, 3 ranks | |
|---|---|---|---|---|---|---|
| hexahedra $100^3$ | 513 / 504 → 407 / 398 | 3.74-3.86 → 3.38-3.46 | 532 → 426 | 5.45-5.60 → 4.88-4.92 | 604 → 465 | 2.21-2.38 → 1.89-2.08 |
| periodic hexahedra | 546 / 537 → 413 / 404 | 4.84-4.96 → 4.35-4.37 | 561 → 428 | 7.47-7.58 → 6.27-6.29 | 571 → 446 | 2.94-3.03 → 2.52-2.59 |
| Kuhn tetrahedra $56^3 \times 6$ | 537 / 531 → 262 / 256 | 3.10-3.13 → 1.94-2.18 | 550 → 331 | 3.92-4.19 → 2.87-2.89 | 547 → 276 | 1.62-1.86 → 1.25-1.37 |
| Gmsh hexahedra $100^3$ | 521 / 508 → 415 / 402 | 3.76-4.26 → 3.31-3.35 | 536 → 430 | 5.86-6.00 → 4.95-4.99 | 560 → 435 | 2.48-2.56 → 2.09-2.19 |
| quadrilaterals $1000^2$ | 255 / 254 → 219 / 218 | 1.33-1.35 → 1.14-1.16 | 276 → 252 | 2.08 → 1.78 | 324 → 294 | 0.86-0.94 → 0.78-0.84 |
| periodic quadrilaterals | 270 / 269 → 226 / 225 | 1.98-2.12 → 1.83-1.85 | 290 → 246 | 3.41-3.44 → 2.69-2.72 | 323 → 295 | 1.23-1.27 → 1.02-1.06 |
| triangles $700^2 \times 2$ | 222 / 222 → 170 / 170 | 0.93-0.94 → 0.70 | 239 → 223 | 1.57 → 1.28 | 253 → 228 | 0.65-0.71 → 0.58-0.64 |
| Gmsh quadrilaterals $1000^2$ | 263 / 258 → 227 / 222 | 1.59-1.64 → 1.41-1.44 | 281 → 281 | 2.38-2.57 → 2.01-2.13 | 303 → 283 | 1.14-1.15 → 0.85-0.90 |

- **Vertex neighbors on first use.** `cells_of_cell` and its offsets and shift rows (106 B per hexahedron, 275 B
  per Kuhn tetrahedron, 36 B per quadrilateral) are built by `Mesh::vertex_neighbors()` when first read, not
  with the mesh, and freed with the setup's data: only the kernels that read them in the step keep them (the
  vertex gradient of viscous fluxes, the hybrid sensor and MUSCL, the LES filters through it, moving regions).
  Building them takes 0.49 s per million hexahedra and 1.1 s per million tetrahedra, as before, at the first
  reader. Their periodic shift rows are still numbered with the mesh, so the `shifts` table and the checks of
  periodic directions are unchanged.
- **Face matching and node directory in rounds.** `DistributedMesh` sends the faces in four rounds of a hash
  bucket, and the node directory's records in four rounds of node ranges: on 3 ranks of hexahedra the
  construction peaked at 604 B per cell and now at 320, the distribution at 540 and now at 465. The records of a round are
  sorted as records, stably: `std::sort` of an index took twice as long on some meshes' rounds.

Distributed builds now peak in the distribution (465 B per hexahedron on 3 ranks against 419 built, 331 per
tetrahedron on 1 rank against 266): the cells sent to their owners next to the block. A whole run (Kokkos host
allocations, CPU, 10 steps) holds less after setup when no step kernel reads the vertex neighbors: 134 B per cell
less on the $64^3$ Taylor-Green vortex with TENO5 (6,660 → 6,526 B per cell; 5,440 → 5,309 on 3 ranks), 44 B on
$512^2$ quadrilaterals (2,476 → 2,432). With GPUs that is device memory, and the host copies of the lists go too.
The runs' peaks are set later, in the numerics setup, and do not change; nor do viscous runs, which keep the lists.
Setup and step times are within the run-to-run spread (Taylor-Green, 10 steps: 17.86 → 17.73 s). Meshes are
bitwise those of `main` on every rank (a hash of every array), and `mallard diff` finds the states of 10-step runs
(inviscid and viscous, 2D and 3D, hexahedra and tetrahedra, dynamic Smagorinsky LES) the same bit for bit on 1 and
3 ranks.

### 19. Troubled scratch for a quarter of the cells (#978)

The troubled passes' scratch (central coefficients and the sector coefficients of every face) had room for half
the cells; it now has room for a quarter (`troubled_capacity = 0.25`), and cells beyond it are done in further
rounds. Every round below a capacity of 1 recomputes its cells' central coefficients, so results do not depend
on the capacity (`TENO3DTroubled` and `RotationalPeriodicSetup` test 0.001 and 0.02 against 1 and 0.5).

The cases of section 15 (perfect gas on $32^3$ hexahedra and $16^3 \times 6$ tetrahedra, LES, the propane
flame on $48 \times 24^2$ / $24 \times 12^2 \times 6$, GRI-Mech 3.0 on $32 \times 16^2$), 3D TENO5 unless
stated, `mallard check --memory a100-80` on a CPU (2 Threads, one step): bytes per cell and the cells that fit
on an A100-80GB.

| bytes per cell, cells per A100-80GB | perfect gas, hexahedra | tetrahedra | LES, hexahedra | tetrahedra | flame, hexahedra | tetrahedra | GRI-Mech, hexahedra |
|---|---|---|---|---|---|---|---|
| capacity 0.5 | 7,558 / 10,402,669 | 7,414 / 10,603,752 | 7,598 / 10,347,899 | 7,454 / 10,546,850 | 9,212 / 8,419,977 | 8,383 / 9,252,886 | 12,046 / 6,432,167 |
| capacity 0.25 | 6,678 / 11,773,594 | 6,534 / 12,031,826 | 6,718 / 11,703,485 | 6,574 / 11,958,618 | 8,332 / 9,309,270 | 7,503 / 10,338,158 | 11,413 / 6,789,351 |

| perfect gas, bytes per cell, cells per A100-80GB | TENO3, hexahedra | tetrahedra | TENO4, hexahedra | tetrahedra | TENO6, hexahedra | tetrahedra |
|---|---|---|---|---|---|---|
| capacity 0.5 | 4,973 / 15,808,529 | 5,740 / 13,696,969 | 5,371 / 14,636,220 | 6,183 / 12,715,675 | 8,382 / 9,379,146 | 8,774 / 8,960,039 |
| capacity 0.25 | 4,343 / 18,101,651 | 5,110 / 15,385,696 | 4,641 / 16,938,167 | 5,453 / 14,418,008 | 7,292 / 10,781,086 | 7,684 / 10,230,998 |

880 bytes less per cell at TENO5 (630 at TENO3, 730 at TENO4, 1,090 at TENO6): 13% more cells per A100-80GB
for a perfect gas at TENO5, 11-12% for the 6-species flame, 6% with GRI-Mech (whose setup peak binds).
Results are bitwise unchanged (`mallard diff` of the restart files after 4 steps, on a CPU). On one A100-80GB
(CUDA 13.5), restart files after 100 steps are bitwise identical with capacities 0.25 and 0.125 on the $96^3$
Taylor-Green vortex, the $96^3$ explosion and the $48^3 \times 6$ tetrahedral Taylor-Green vortex, and on the
2D Riemann problem with 0.25 and 0.125 against 1.

The smaller scratch adds two rounds of the five troubled kernels per stage, which find no work in smooth flows:
under 0.3 ms per step on an A100 (Taylor-Green $96^3$, about 105 ms per step), and a capacity of 0.125 is within
1% of 0.25. In 2D a capacity of 0.25 costs 8% (Riemann problem, 14.2 to 15.4 ms per step), because every round
recomputes the central coefficients, so 2D keeps 1.

**An eighth of the cells.** `troubled_capacity` now defaults to 0.125 in 3D: 440 bytes less per cell at TENO5, results
again bitwise unchanged. `mallard check --memory a100-80 --steps 1`, perfect gas, TENO5, cells per A100-80GB
(the setup's peak, which holds the scratch too, binds in both):

| | $32^3$ hexahedra | $16^3 \times 6$ tetrahedra |
|---|---|---|
| CPU (2 Threads), capacity 0.25 | 12,153,598 (6.47 kB per cell) | 12,232,604 (6.43 kB) |
| CPU, capacity 0.125 | 13,040,617 (6.03 kB), +7.3% | 13,131,619 (5.99 kB), +7.4% |
| A100 (CUDA 13.5) with #1306, capacity 0.25 | 13,324,340 (5.87 kB) | 12,616,797 (6.20 kB) |
| A100 with #1306, capacity 0.125 | 14,403,365 (5.43 kB), +8.1% | 13,579,908 (5.76 kB), +7.6% |

On one A100-80GB the time per step stays within the run-to-run spread (about 2% in the explosion, under 1% in
the Taylor-Green vortex; the $96^3$ hexahedral and $48^3 \times 6$ tetrahedral Taylor-Green vortex and the $96^3$
explosion, 5 alternated runs each, numbers in #978): the two extra rounds of five kernels per stage find no work
where few cells are troubled.

### 20. TENO setup: one fit per stencil shape (#986)

With shared tables, every backend keys each cell's stencil geometry on the device before the fit and fits one
cell per key (`teno_shapes.h`); the other cells take its tables, and whole batches go through the shared-table
packing of #1031. The $64^3$ Taylor-Green vortex at TENO5 (hexahedra, periodic, HLLC, SSPRK3), Threads with 2 threads on an
Apple M-series machine shared with other jobs, base = #1031, one step and `mallard post derive` of its step-0
restart file (`--fields Q_CRITERION`):

| $64^3$ TENO5, Threads | numerics setup | of it: tables / packing / keys | run, wall / CPU | run, peak RSS | post derive, wall / CPU | post derive, peak RSS |
|---|---|---|---|---|---|---|
| base | 56.9 s | 33.5 / 12.1 / - s | 63.1 / 104.4 s | 2.43 GB | 59.6 / 95.1 s | 1.82 GB |
| keyed | 34.1 s | 0.01 / 18.5 / 4.4 s | 42.4 / 62.2 s | 2.46 GB | 47.6 / 59.8 s | 1.85 GB |

27 cells are fitted of 262,144: the interior and the 26 combinations of periodic faces. Results are bitwise
those of the base except where cells of one key used to fall on either side of the table comparison's grid:
of 3D cases on $24^3$ / $12^3$ / $8^3$ meshes (periodic and symmetry boxes, mixed, tetrahedra, prisms; orders
3-6) only the periodic order-3 box differs (restart files after 3 steps, at most 6.7e-16 in density), and of
2D double Mach ($160 \times 40 \times 2$, orders 3-6) and $96^2$ boxes only the order-3 box (4.4e-16). 2 and 3
ranks give the 1-rank result bit for bit.

On one A100 (base = #1069, device packing), restart files after 3 steps are bitwise those of the base, and a
GPU run that reads the tables cache an OpenMP run wrote gives its own result bit for bit; the two caches differ
only in the producer's name. Seconds of the numerics setup:

| A100 | base: tables / packing | keyed: tables / packing / shapes | fitted |
|---|---|---|---|
| double Mach $2048 \times 512 \times 2$, TENO3 | 0.87 / 0.96 | 0.17 / 1.17 / 0.40 | 323 of 2,097,152 |
| TENO4 | 1.43 / 0.96 | 0.20 / 1.24 / 0.40 | 323 |
| TENO5 | 4.52 / 1.16 | 0.32 / 1.47 / 0.46 | 323 |
| TENO6 | 19.35 / 1.31 | 0.66 / 1.54 / 0.56 | 323 |
| $96^3$ symmetry box, TENO5 | 5.29 / 1.14 | 0.07 / 2.03 / 1.16 | 873 of 884,736 |
| 750,000 jittered tetrahedra, TENO3 | 3.02 / 6.18 | 3.02 / 6.41 / 0.23 | 750,000 |
| TENO5 | 90.10 / 30.83 | 90.05 / 31.19 / 0.18 | 750,000 |

The peak device memory is unchanged except for double Mach at order 6 (6,995 to 7,515 MiB) and the
tetrahedra, where every cell has a shape of its own: against main's 11,279 / 40,611 MiB at orders 3 / 5, the
base peaks at 11,955 / 44,013 MiB and the keyed path at 11,279 / 42,701 MiB.

**Ranking a key's table once** (#1079). The packing ranked, hashed and compared every cell's table although the
cells of a key carry it bit for bit. Each batch now does that once per key, at its first cell there, with the
lowest global id of its cells, so the store sees the same tables in the same order and gives the same shapes.
The $64^3$ Taylor-Green vortex at TENO5 as above, under a load average of about 140, two alternating runs, base =
the keyed setup above (#986); the packing includes the decode of the records:

| $64^3$ TENO5, Threads | setup, wall | of it: geometry / tables / packing / keys | run, wall | run, peak RSS |
|---|---|---|---|---|
| base | 77.2 / 92.5 s | 21.9 / 0.07 / 28.0 / 9.2 s, 23.9 / 0.03 / 41.2 / 10.6 s | 86.1 / 100.2 s | 2.43 / 2.35 GB |
| once per key | 70.0 / 60.1 s | 25.8 / 0.07 / 11.7 / 14.6 s, 18.3 / 0.04 / 11.6 / 12.8 s | 77.1 / 68.0 s | 2.32 / 2.43 GB |

On one A100 (16 host threads, two alternating runs each) the packing, which ranks on the device, gains less:

| one A100, packing | base | once per key |
|---|---|---|
| $96^3$ symmetry box, TENO3 / TENO5 | 1.49-1.53 / 1.93-2.05 s | 1.30-1.37 / 1.63-1.70 s |
| 1,053,696 tetrahedra, TENO3 / TENO5 | 2.89-2.90 / 4.34-6.94 s | 2.61-2.66 / 3.57-3.65 s |

The $96^3$ box at TENO3 on 8 OpenMP threads packs in 4.8 s instead of 11.4 s. Restart files are bitwise those of
the base on the A100, on OpenMP (orders 3, 5 and 6; hexahedra, tetrahedra and prisms) and on 1, 2 and 3 ranks, and
the tables in the caches the A100 and OpenMP write are the same bytes. What remains: the decode's search of
every layer (#1178), the per-cell geometry (#1166) and the keys (#1167).

### 21. Step graph captures out of the throughput; lock arrays copied after a failed capture only

The capture of a step graph is a one-time cost, made in the second time step (the first runs kernel by kernel),
that time stepping used to count: over the 100 steps of the performance set it was 1 to 2 ms per step, 40% of
`air_3d`'s step at main. Time stepping, and so every throughput and wall time per step reported, now leaves it out,
and the end record gives it as `wall_capture`.

That cost had grown. Without relocatable device code every translation unit holds its own copy of the pointers
to desul's lock arrays, which Kokkos sets at the unit's first kernel launch with a copy on the legacy stream;
made inside a stream capture, that copy invalidates it (`lock_arrays.h`). Every capture used to copy them to
every unit first. Under CUDA's lazy loading (the default since CUDA 12.2) a copy to a unit's symbols loads all
of its device code, so the first capture loaded the code of every source file, also of features a run does not
use: at main, `air_3d` (MUSCL) spent 160 ms there, `cell_chemistry` 59 ms, `region_motion` 51 ms, `teno`
13 ms and 50 smaller units, as those units grew (fat binaries 45.5 MB at 0.10.0, 69.6 MB at main). Counted as
time stepping, it made `air_3d` and `sphere_3d` look 13 to 14% slower than at 0.10.0, while their steps are
faster. The capture now runs first, and only a capture that fails (on any rank) copies every unit's lock arrays
and is tried again: the units whose kernels a step graph captures have launched them in the first steps, except
kernels only in conditional bodies none of those steps took.

One A100 (CUDA 13.5), 0.10.0 and main with the capture timed alike: throughput over time stepping, best of 6
runs in 3 interleaved rounds, and the capture (`wall_capture`, median of 2 runs):

| one A100 | throughput 0.10.0 | main | this change | capture 0.10.0 | main | this change |
|---|---:|---:|---:|---:|---:|---:|
| `air_3d` | 52.8M | 56.6M | 56.5M | 60 ms | 166 ms | 0.6 ms |
| `sphere_3d` | 24.3M | 24.6M | 24.5M | 67 ms | 174 ms | 0.6 ms |
| `teno5_3d` | 6.07M | 8.64M | 8.64M | 49 ms | 170 ms | 2.4 ms |
| `flame_3d` | 6.73M | 6.84M | 6.84M | 45 ms | 111 ms | 1.4 ms |

Results are bitwise those of main (snapshots after 100 steps of `air_3d`, `sphere_3d`, `teno5_3d` and
`flame_3d`). `StepQueueTest.ACaptureThatAUnitsFirstKernelLaunchFailsIsTriedAgain` makes the first capture fail
as such a launch does; `StepQueueTest.TheStepGraphsCaptureIsLeftOutOfTimeStepping` makes it a second slower.

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
- 3D TENO5 Taylor-Green, $64^3$ hexahedra per A100 (Nx = 64 N, multi-jagged, NCCL, CUDA graphs; main
  9da4ab1e7, median of the progress intervals from step 60 to 160, two runs): 1 GPU 30.58 ms, 2 GPUs 30.77 ms
  (99.4%), 4 GPUs 31.10 ms (98.3%), 8 GPUs 31.04 ms (98.5%); over two nodes (HDR InfiniBand) 1 + 1 31.09 ms,
  4 + 4 31.05 ms and 7 + 7 31.06 ms (98.5%). With the step queue (#787) the step graph runs across nodes too,
  so crossing nodes costs nothing measurable; before it, two-node runs fell back to one step per launch and
  paid a flat 0.5 ms per step.

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
