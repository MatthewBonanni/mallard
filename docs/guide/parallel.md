# Running in parallel

Mallard has two levels of parallelism: Kokkos inside each process (CPU threads or one GPU), and MPI between processes, which split the mesh between them.

## Threads and GPUs

The Kokkos backend is chosen when Mallard is built (see [Getting started](https://matthewbonanni.github.io/mallard/docs/installation/)). With `Threads` or `OpenMP`, set the number of threads on the command line:

```bash
Mallard -i input.toml --kokkos-num-threads=8
```

With CUDA, Mallard runs on one GPU; `--kokkos-device-id=N` picks which.

## MPI

Build with MPI enabled; this needs an MPI implementation such as Open MPI or MPICH:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_SYSTEM_KOKKOS=OFF \
      -DKokkos_ENABLE_THREADS=ON -DMallard_ENABLE_MPI=ON
cmake --build build -j
```

and launch with `mpirun`; the input file is unchanged:

```bash
mpirun -n 4 build/src/Mallard -i input.toml --kokkos-num-threads=2
```

Each rank owns a piece of the mesh, plus as many layers of halo cells as its reconstruction needs (one for first order, two for MUSCL and viscous terms, more for TENO, which Mallard determines from the stencils themselves). The log reports the split:

```text
> Distributed over 3 ranks: 160000 cells, at most 53334 per rank, 2 halo layers
> TENO stencils need 8 halo layers; rebuilding the local meshes
> Distributed over 3 ranks: 160000 cells, at most 53334 per rank, 8 halo layers
```

The pieces come from one of three partitioners, chosen with `partitioner` in a `[parallel]` table of the input. All three give the same results bit for bit:

```toml
[parallel]
partitioner = "multijagged"   # default; or "hilbert", "graph"
halo_exchange = "nccl"        # default in NCCL builds; else "mpi"
```

![Six-rank partitions of a box of triangles and of an O-grid by the Hilbert and multi-jagged partitioners](../images/guide/partitions-light.svg#gh-light-mode-only)
![Six-rank partitions of a box of triangles and of an O-grid by the Hilbert and multi-jagged partitioners](../images/guide/partitions-dark.svg#gh-dark-mode-only)

*Six ranks' pieces of a box of triangles and of an O-grid around a cylinder,
from Mallard runs with each partitioner: Hilbert pieces follow a space-filling
curve, multi-jagged ones are slabs cut along each axis in turn.*


- `multijagged` (the default since 0.8.0): equal slabs along x, each slab cut along y and, in 3D, along z, with the number of pieces per axis that leaves the least area between the blocks. A box of cuboid cells gets cuboid parts; the blocks are numbered along a Hilbert curve, so the ranks of a node hold a compact group.
- `hilbert` (the default before 0.8.0): cells split along a Hilbert curve through their centroids.
- `graph`: [dKaMinPar](https://github.com/KaHIP/KaMinPar) on the cell connectivity, which minimizes the number of faces between ranks; needs a build with `-DMallard_ENABLE_KAMINPAR=ON` (and oneTBB).

On GPUs, build with `-DMallard_GPU_AWARE_MPI=ON` if your MPI is CUDA-aware: halo data then goes between GPUs directly instead of through host memory. On NVIDIA GPUs, `-DMallard_ENABLE_NCCL=ON` exchanges halos with NCCL on the GPU's stream, with no host synchronization per exchange (`[parallel] halo_exchange = "nccl"`, the default in such builds). CUDA builds also run each step as one CUDA graph (`[run] cuda_graphs`, on by default), except where the host takes part in the step: gas mixtures, characteristic boundaries, expressions of `t`, average-pressure outlets and MPI (not NCCL) halo exchanges. Exchanges carry only the halo cells the stencils read.

Every rank builds the same stencils as a serial run and sums each cell's face fluxes in a fixed order, so the solution is bitwise identical on any number of ranks: after 100 steps of the 2D Riemann problem on 400 × 400 quadrilaterals with fifth-order TENO-E, the restart files of runs on 1 and 3 ranks are byte for byte the same. Sums over the whole domain that are written to output files, such as force monitors, are reduced across ranks and can differ in the last digits.

Each rank numbers its cells along a Hilbert curve for memory locality ([cell numbering](performance.md#cell-numbering)); restart, HDF5 and VTU files list cells in global order whatever the rank count.

With threads and ranks together, keep ranks × threads at or below the number of cores. On GPUs, use one rank per GPU; Kokkos maps each rank to its own device.

With generated meshes, and with meshes converted to Mallard's HDF5 format (`mallard-mesh-convert`, in builds with `-DMallard_ENABLE_HDF5=ON`; see the [input reference](../input.md)), each rank reads or generates only its share, and no rank ever holds the whole mesh. Gmsh files are still read whole by every rank, so convert large Gmsh meshes to HDF5 first.

### Output

Each rank writes the cells it owns as a piece, `<prefix>_<step>_p<rank>.vtu`, and rank 0 writes a `<prefix>_<step>.pvtu` that assembles them; the `.pvd` series points at the `.pvtu` files, so ParaView opens a parallel run exactly like a serial one:

```text
solut/r.pvd
solut/r_000100.pvtu
solut/r_000100_p0000.vtu
solut/r_000100_p0001.vtu
solut/r_000100_p0002.vtu
```

Boundary-zone output (`geometry = "<zone>"`) likewise holds each rank's own faces. Force monitors are summed over the ranks and written by rank 0.

### Restarts

Restart files have the same layout whatever the number of ranks, cells in global order, and each rank reads only its cells by global id. A run can therefore restart on a different number of ranks, or serially from a parallel run and vice versa: a run written at step 50 on 3 ranks and continued to step 100 on 2 ranks ends with a restart file identical to that of an uninterrupted serial run.

## Multi-GPU scaling

The [performance dashboard](https://matthewbonanni.github.io/mallard/docs/dashboard/#scaling) plots the efficiencies below.

### 3D, Mallard 0.8.0 {#scaling-3d}

The Taylor–Green vortex (viscous) on hexahedra with fifth-order TENO-E and HLLC, on NVIDIA A100-80GB GPUs: 4 per node, NVLink within a node, HDR InfiniBand with GPUDirect RDMA between nodes, GPU-aware MPI. Times are steady times per step, setup excluded; efficiency is T<sub>1</sub>/(n T<sub>n</sub>) for strong and T<sub>1</sub>/T<sub>n</sub> for weak scaling. Measured with the development code of [#261](https://github.com/MatthewBonanni/mallard/pull/261), which made multi-jagged partitions the default; `docs/design/performance.md` in the repository has the details.

**Strong scaling**, 96³ hexahedra (one GPU: 154 ms per step):

| GPUs | Hilbert | graph | multi-jagged (default) | efficiency, multi-jagged |
|---:|---:|---:|---:|---:|
| 4 | 41.2 ms | | 41.4 ms | 93% |
| 8 | 23.1 ms | 23.5 ms | 23.1 ms | 83% |
| 12 | 17.8 ms | 17.2 ms | 16.6 ms | 77% (Hilbert 72%) |
| 16 | 13.4 ms | 13.3 ms | 13.1 ms | 73% |

**Weak scaling**, 64³ hexahedra per GPU (one GPU: 48.1 ms per step):

| GPUs | mesh | Hilbert | graph | multi-jagged (default) | efficiency, multi-jagged |
|---:|---|---:|---:|---:|---:|
| 4 | 128 × 128 × 64 | 49.8 ms | | 49.8 ms | 97% |
| 8 | 128³ | 50.8 ms | 51.4 ms | 50.5 ms | 95% |
| 12 | 192 × 128 × 128 | 54.7 ms | 52.7 ms | 51.9 ms | 93% (Hilbert 88%) |
| 16 | 256 × 128 × 128 | 55.8 ms | 53.9 ms | 52.7 ms | 91% (Hilbert 86%) |

- **Multi-jagged partitions:** the parts of a box are cuboid blocks. These are the same as Hilbert parts at power-of-two rank counts on cubes, and have 30–45% fewer cells in the first halo layer otherwise (12 ranks, and the weak-scaling boxes). In 2D at 16 GPUs the partitioners are within 1–4% of each other.
- **Strong-scaling loss:** most of what is left at 16 GPUs (55k cells per GPU) is the TENO kernel itself running less efficiently on fewer cells, and the redundant reconstruction of halo layer 1, which grows with the partition surface.
- **Slimmer halo exchanges** ([#258](https://github.com/MatthewBonanni/mallard/pull/258)): 3D TENO5 at 12 GPUs went from 85% to 88% weak and from 68% to 72% strong scaling.
- **Small meshes per GPU:** there the step is latency-bound. The CUDA graph and NCCL exchange ([#217](https://github.com/MatthewBonanni/mallard/pull/217)) take 262,000 MUSCL cells on 4 GPUs from 667 to 523 µs per step, and a 4,000-cell MUSCL step from 286 to 230 µs.

### 2D, Mallard 0.3.0 {#scaling-2d}

The 2D Riemann problem (configuration 3) on Cartesian quadrilaterals, fifth-order TENO-E, HLLC, SSPRK3, double precision, with Mallard 0.3.0 on NVIDIA A100-80GB GPUs, one MPI rank per GPU, the Hilbert partition and GPU-aware MPI. Builds use `-DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON -DKokkos_ENABLE_OPENMP=ON -DMallard_ENABLE_MPI=ON -DMallard_GPU_AWARE_MPI=ON`. Up to 8 GPUs share one node over NVLink; 16 GPUs are 4 nodes of 4, connected by HDR InfiniBand with GPUDirect RDMA. Times are seconds of wall time for 50 time steps, excluding setup; efficiency is relative to one GPU (for 16M cells, to 8).

**Strong scaling** (fixed mesh):

| GPUs | 1M cells | efficiency | 4M cells | efficiency | 16M cells | efficiency |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.938 s | | 3.50 s | | | |
| 2 | 0.553 s | 85% | 1.93 s | 91% | | |
| 4 | 0.311 s | 75% | 1.00 s | 88% | | |
| 8 | 0.190 s | 62% | 0.534 s | 82% | 1.93 s | |
| 16 | 0.152 s | 39% | 0.316 s | 69% | 1.01 s | 96% |

**Weak scaling** (1M cells per GPU):

| GPUs | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| cells | 1M | 2M | 4M | 8M | 16M |
| time | 0.938 s | 0.999 s | 1.000 s | 1.005 s | 1.013 s |
| efficiency | | 94% | 94% | 93% | 93% |

Eight GPUs on 2 nodes of 4 run as fast as on one node of 8. Staging halo data through host memory, without `Mallard_GPU_AWARE_MPI`, makes multi-GPU runs 15–25% slower, and the `graph` partitioner is within a few percent of `hilbert` on this mesh. On one GPU this build takes 18.8 ms per step of 1M cells, against 20.1 ms for the plain single-GPU build of the [Performance](performance.md) page (no MPI, serial host backend), on a different A100. The halo exchange overlaps the reconstruction of interior cells, and the work of TENO's troubled cells is split across faces and characteristic variables, so that the few troubled cells of a small partition do not serialize a stage; the [MPI design notes](../design/mpi.md) have the profile behind these choices.

Design notes: [distributed memory (MPI)](../design/mpi.md), on how the partitioning, halos and exchanges work.
