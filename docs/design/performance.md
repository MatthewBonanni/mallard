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

### 3D TENO5, 96^3 hexahedra, HLLC + viscous (`teno5_3d`)

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

## Changes

### 1. Cooperative 3D TENO smooth pass

`teno_smooth` in 3D (degree >= 4, i.e. TENO5 and TENO6) runs as a team kernel: 32 cells x
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
| `teno_smooth`, 96^3 hexahedra, first stage | 111.8 ms | 44.9 ms |
| TENO5 Taylor-Green 96^3 (`teno5_3d`) | 335 ms/step (2.64M cells/s) | 151 ms/step (5.85M cells/s) |
| TENO6 Taylor-Green 32^3 | 47.3 ms/step | 20.7 ms/step |
| TENO5 explosion, 20^3 x 6 tetrahedra | 17.5 ms/step | 11.0 ms/step |
| TENO3/TENO4 (degrees 2-3) | unchanged (per-thread kernel kept) | |

### 2. TENO setup: stencil tables streamed to the device (host memory)

The packed stencil tables (pseudo-inverses, stencil cells and faces) can only be sized on the device once every
stencil is known, so the setup kept every packed chunk on the host until the end: 27 KB per cell for 3D TENO5.
That host peak, more than device memory, set how many GPUs a large 3D run needs (#58). On CUDA each packed chunk
now goes straight to the device, into a reserved address range whose pages are mapped as it fills (CUDA virtual
memory management), so the host holds one chunk of 8192 cells at a time. The device arrays keep their exact sizes
and single base addresses; results and step times are unchanged.

| 3D TENO5, 64^3 hexahedra, one A100 | main | change 2 |
|---|---|---|
| host peak during setup | 8.55 GB (32.6 KB/cell) | 1.44 GB (5.5 KB/cell) |
| device peak | 11.6 GB | 11.6 GB |

### 3. MUSCL: per-cell reconstruction, fused, overlapped with the halo exchange

MUSCL reconstructed in three kernels (least-squares gradients and limiters over cells, then face states over
faces), after the halo exchange. It now also reconstructs per cell: gradient, limiter, then the cell's own side of
each of its faces, with the same arithmetic. From 2^17 cells (per rank) this runs as one fused kernel, which keeps
gradients and limiters out of memory, and in distributed runs the cells whose stencils hold no halo cell are
reconstructed while the exchange is in flight, as TENO's are. Below 2^17 cells the separate kernels stay: with
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

| 3D TENO5 | 100^3 hexahedra (1M) | 55^3 x 6 tetrahedra (998k) |
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

| 3D TENO5, one A100 | 100^3 hexahedra | 55^3 x 6 tetrahedra |
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
fuel, Y_C3H8 1e-10 to 1e-5) take 20 to 130 RODAS sub-steps against 2 for the others: the fuel burns
out within the step along `C^0.1`, which reaches zero in finite time, through the regularization below
`C_reg` (chemistry.md). Leaving them out of the chemistry (an experiment only) took the chemistry from
19.2 to 11.3 ms per step. `chemistry.C_reg` (default 1e-12 kmol/m^3) moves the regularization; larger
values cut these cells' sub-steps (a burnt cell at 1750 K with Y_C3H8 = 1e-7 over 0.15 us: 32 sub-steps
at 1e-12, 13 at 1e-9, 5 at 1e-7) but change the rate law below C_reg. Opt-in: on `flame_3d`, 1e-10,
1e-9 and 1e-8 give 24.6, 23.4 and 22.3 ms per step; against the default after 100 steps, T differs by
at most 0.54 K, Y_k by 3e-5, p by 90 Pa and u by 0.14 m/s, the integrated heat release by 0.08% (the
largest pointwise heat release differences are in cells burning out trace fuel). Thread-per-cell remains faster
than lanes for 6 species (2, 4, 8 or 32 lanes per cell: 1.8 to 5x slower).

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
| 2D TENO5, Riemann (1024^2 / GPU, square domain at 4) | 1M | 19.6 ms | 20.1 ms | 20.4 ms | 96% |
| 2D MUSCL, Riemann | 4M | 15.6 ms | 16.1 ms | 16.3 ms | 96% |

Notes:

- Weak scaling must keep the cells' shape: refining a direction alone (anisotropic cells) changes TENO's stencils
  and their cost. 3D TENO5 on 64^3 cells per GPU refined along x and y only gave 47.2 / 52.2 / 69.1 ms on 1 / 2 / 4
  GPUs; the comparison with the domain grown instead is in progress.
- At small sizes per GPU (MUSCL, 262k cells on 4 GPUs: 65k per GPU) the step is latency-bound (about 30
  launches and four exchanges per step); the CUDA graph and NCCL exchange of #217 take it from 667 to 523 us.
- Multi-node runs (16-32 GPUs over InfiniBand) are pending free GPUs.

## Verification

Bitwise identity with `main` is checked on restart files (identical MD5) after N steps of: the
performance cases (2D TENO5 and MUSCL Riemann, 3D TENO5 Taylor-Green), the double Mach reflection
(triangles, Dirichlet and symmetry mirrors), axisymmetric Sedov, a viscous shock tube, the reactive
shock tube with MUSCL and TENO5 (2D, and a 3D variant: gas mixtures, primitive reconstruction), the 3D
explosion on hexahedra and tetrahedra, and the Taylor-Green vortex at orders 3, 4 and 6; for MUSCL also the
wedge, a viscous shock tube, the explosion with MUSCL on hexahedra and tetrahedra and a reacting H2/O2/Ar case
on 512^2 cells; on 1, 2 and 3 ranks.
