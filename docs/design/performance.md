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

## Verification

Bitwise identity with `main` is checked on restart files (identical MD5) after N steps of: the
performance cases (2D TENO5 and MUSCL Riemann, 3D TENO5 Taylor-Green), the double Mach reflection
(triangles, Dirichlet and symmetry mirrors), axisymmetric Sedov, a viscous shock tube, the reactive
shock tube with MUSCL and TENO5 (2D, and a 3D variant: gas mixtures, primitive reconstruction), the 3D
explosion on hexahedra and tetrahedra, and the Taylor-Green vortex at orders 3, 4 and 6.
