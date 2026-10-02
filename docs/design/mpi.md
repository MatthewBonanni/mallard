# Design: distributed memory (MPI)

Status: accepted design (reviewed in #32). Tracks issue #2.

## Goals

- Run on one GPU per MPI rank, from one node (8 GPUs) to thousands of GPUs.
- **No rank ever holds the global mesh.** Reading, partitioning, setup and output are all distributed, so the mesh size is limited only by aggregate memory.
- Results independent of the rank count up to round-off (bitwise identical stencils and reconstructions; only the summation order of fluxes into a cell may differ).
- A single-rank build without MPI keeps working and stays the default for development and CI.
- Everything here is dimension-agnostic: it works on cells, faces as node lists and the cell graph, so 3D inherits it.

## Overview

```text
 parallel read        dual graph          partition         migrate           halo            precompute
 (HDF5, block    ->   (faces matched  ->  (dKaMinPar or ->  (cells, nodes, -> (k layers of  ->  (geometry, TENO
  of cells/rank)       by hashing)         Hilbert curve)    zone tags)        ghost cells)       stencils, BCs)
                                                                                                   |
                         time loop: exchange halo state -> reconstruct -> fluxes -> update owned cells
```

## Components

### 1. Communication layer

A thin `Comm` class wraps MPI (rank, size, `allreduce` min/sum/max, `alltoallv`, neighbor exchange). When MPI is disabled (`Mallard_ENABLE_MPI=OFF`), it is a one-rank stub, so the rest of the code has no `#ifdef`s. One rank per GPU: Kokkos maps devices by local rank (`--kokkos-map-device-id-by=mpi_rank`).

The per-stage halo exchange and the time-step `allreduce` go through a `HaloExchange` interface with two backends:

- **MPI** (default): nonblocking point-to-point; device buffers passed directly with GPU-aware MPI, or staged through host memory otherwise.
- **NCCL** (optional, `Mallard_ENABLE_NCCL`): `ncclSend`/`ncclRecv` inside a group and `ncclAllReduce`, enqueued on the Kokkos execution stream, so the exchange is stream-ordered with the kernels (no host synchronization) and can be captured in a CUDA graph. The NCCL communicator is created from the MPI communicator.

Setup-time communication (partitioning, migration, I/O) always uses MPI.

### 2. Mesh input

- **Format:** an HDF5 mesh file with global arrays, global ids being row indices: node coordinates, cell connectivity (CSR of node ids; the cell type follows from the node count), boundary faces (CSR) with zone ids, and zone names (layout in `docs/input.md`). Each rank reads a contiguous block of cells, of nodes and of boundary faces, with collective parallel HDF5 I/O (independent reads when HDF5 is not parallel). Node coordinates a rank needs but did not read are fetched later from the rank that read them (section 5).
- **Converter:** `mallard-mesh-convert` turns Gmsh (2.2/4.1) files into this format, or writes a generated mesh described by an input file in parallel. A Gmsh file is read whole, since it is converted once per mesh; later the converter can stream meshes that don't fit in memory.
- Generated meshes (`cartesian`, `wedge`, ...) are produced directly in blocks per rank, numbered as the serial generators number them.
- A Gmsh file is read whole by every rank, which keeps only its block; meshes too large for that are converted first. Every source then goes through the same distributed path (`DistributedMesh`); only single-rank runs build the mesh directly.

### 3. Distributed dual graph

Two cells are adjacent if they share a face. Each rank hashes every face of its cells (sorted global node ids) to an owner rank. One `alltoallv` brings the copies of each face together and pairs them; the boundary faces of the mesh file go to the same ranks by the same hash. A second sends each pair back as a graph edge, and each unpaired face back as a boundary face with its zone (the first boundary face in global order that matches it, else `unassigned`). Faces shared by more than two cells, and boundary faces that are interior or not cell faces, are errors on every rank. The result is a distributed CSR graph in ParMETIS layout (`vtxdist`, `xadj`, `adjncy`). Cost is O(faces / ranks) per rank and two all-to-alls.

### 4. Partitioning

A `Partitioner` interface with two backends:

| Backend | When |
|---|---|
| **dKaMinPar** ([KaHIP/KaMinPar](https://github.com/KaHIP/KaMinPar), MIT, C++20; `FetchContent`) | Default when available. Distributed, scales to trillion-edge graphs, guarantees balance. Built with 64-bit ids for hero meshes. New dependency: oneTBB. |
| **Hilbert curve** (built in) | No-dependency fallback; also used everywhere to order cells within a rank for memory locality. |

The Hilbert backend sorts cells by the curve key of their vertex average (not their centroid, which needs the geometry a rank does not have yet) with a distributed sample sort, so its partitions differ from a global-mesh Hilbert partition but are equally valid; dKaMinPar works directly on the distributed dual graph (section 3).

Vertex weights model cost: a base weight per cell plus a term for the TENO stencil size. ParMETIS or PT-Scotch can be added behind the same interface if needed.

### 5. Migration

One `alltoallv` sends each cell to its owner: its global id, owner, node ids and boundary faces (local face index and zone). A second builds a **node directory**: the rank that read each node records the cells using it and their owners. Every node shared by several owners then sends each of them the cells there it does not own, which is halo layer 1. The coordinates of the nodes a rank needs are fetched from the ranks that read them once its halo is complete. The cells' blocks stay on the ranks that read them until setup ends, and serve the halo search.

### 6. Halo

- **Local numbering:**
  - Cells: `[owned interior | owned near partition boundary | halo layer 1 | ... | halo layer k]`. Owned cells are in Hilbert order within each group. (Implemented so far: owned cells, then each halo layer, each in global id order.)
  - Faces: faces between owned cells, then faces between an owned and a halo cell, then faces between halo cells that reconstruction needs.
  - Every cell and node keeps its global id.
- **Depth k:** the number of vertex-neighbor layers any reconstruction needs:

  | Reconstruction | k |
  |---|---|
  | FO | 1 |
  | MUSCL | 2 |
  | TENO | stencil radius + 1 |

  TENO's central-stencil gather grows layer by layer until it has enough rows. So the halo is built with a default depth, stencils are computed, and if any stencil reaches the halo frontier the halo is deepened on every rank and the setup after the mesh repeats. This keeps stencils *identical* to the serial ones, which is what makes results rank-count independent.
- **Construction:** a distributed breadth-first search over vertex neighbors. Layer 1 comes from the node directory (section 5); for each further layer, a rank asks the directory for the cells around the nodes of its previous layer that it has not asked about, then fetches the new cells from the ranks whose blocks hold them. Deepening the halo continues the search from the existing layers rather than starting over.
- **Exchange plan:** for each neighbor rank, the list of local owned cells to send and of halo cells to receive, sorted by global id on both sides.

### 7. Time stepping

Per right-hand-side evaluation:

1. Post nonblocking receives and sends of the conservative state of halo cells, packed into contiguous device buffers. Sends use device pointers with GPU-aware MPI, or a host staging copy otherwise.
2. **Overlap:** reconstruct and compute fluxes for owned interior cells and their faces while messages are in flight.
3. Wait, unpack, then reconstruct the near-boundary owned cells and halo layer 1, and compute the remaining faces.

Halo layer 1 is reconstructed redundantly so that a single exchange per stage suffices. The alternative, exchanging reconstructed face states at partition faces, would save that layer's work (a surface-to-volume fraction of the TENO cost) but adds a second, dependent message round per stage; at thousands of ranks the extra latency costs more than the redundant work. The choice is contained in the exchange plan and can be revisited with measurements.

Faces between an owned and a halo cell are computed by both ranks. Each keeps only its owned side's contribution, since residuals of halo cells are never used. This costs a few redundant face fluxes but needs no reverse exchange.

Global reductions become `allreduce` calls: time step (min), NaN check, conservation sums, force monitors, average-pressure outlets.

### 8. Boundary conditions and precomputation

These are found during per-rank precomputation on owned plus halo cells, and with a sufficient halo they always resolve to local cells:
- transmissive image faces;
- TENO mirror images;
- Dirichlet face states;
- surface-zone membership.

### 9. Output and restart

- **Solution output:** per-rank VTU pieces plus a `.pvtu` index and the `.pvd` series to start. At scale, HDF5 with an XDMF index (one shared file per snapshot, collective writes).
- **Restart:** HDF5, written by global cell id, so a run can restart on a **different rank count**. This is essential for hero runs, which rarely get the same allocation twice.

### 10. Room for dynamic load balancing

Dynamic rebalancing (troubled cells, and so the TENO cost, move with shocks) is deferred, but the design keeps it cheap to add:

- **Setup is a function of a distributed cell set**, not of a file: `build_local(cells owned by this rank) -> local mesh, halo, exchange plan, stencils, boundary data`. Startup calls it after the first partition; a rebalance would call it again after migrating cells.
- **Global ids everywhere** (cells, nodes, faces of boundary zones), and **state migrates with cells** through the same `alltoallv` as setup, keyed by global id.
- **The partitioner takes per-cell weights from the caller**, so measured costs (e.g. the troubled-cell fraction from the last N steps) can replace the static model. KaMinPar can also refine an existing partition rather than start from scratch.
- **Per-rank derived data** (stencils, pseudo-inverses, halo plans) is cached by global id where it is expensive, so a rebalance recomputes only what moved.
- **No assumption that the partition is fixed** in output or restart: both are keyed by global id.

### 11. Communication patterns at scale

Setup exchanges are dense `alltoallv` calls, whose count arrays alone are O(ranks) per rank and whose latency grows with the rank count. The face matching and migration genuinely talk to many ranks, but halo growth and node-coordinate fetches have sparse patterns (a rank's partition neighbors and the few ranks whose blocks hold its cells). Beyond about 10k ranks these should move to MPI neighborhood collectives or a sparse NBX exchange (nonblocking sends, `MPI_Ibarrier` to detect completion); planned with milestone 6.

## Testing

- **Correctness:** run the solver test suite on 1, 2, 3 and 4 ranks (oversubscribed CPUs, Serial backend) in CI with OpenMPI.
  - Stencils, mirror images and halos must match the serial ones exactly by global id.
  - Solutions must match the single-rank result to round-off.
- **Unit tests:** dual-graph construction, migration and halo construction on small meshes with known answers, including cells whose stencil reaches across several ranks.
- **Restart:** write on 3 ranks, read on 2, and get bit-identical state.
- **Scaling:**
  - Strong and weak scaling on one 8-GPU node, on the 2D Riemann problem and the double Mach reflection.
  - Then across nodes once inter-pod MPI is available on the cluster.

## Milestones

1. **Comm layer and build:** `Mallard_ENABLE_MPI`, one-rank stub, CI with `mpirun`.
2. **Correct multi-rank runs at small scale:** global Gmsh read on every rank, Hilbert partition, halo, exchange, reductions. Rank-count-independence tests for FO, MUSCL, TENO and viscous fluxes.
3. **Output and restart:** `.pvtu` output; HDF5 restart independent of the partition.
4. **Scalable setup:** HDF5 mesh format and converter, distributed read, distributed dual graph, dKaMinPar, migration. Done: no rank holds the global mesh in a distributed run, and restart files are read by global id, each rank reading only its cells.
5. **Performance:** communication/computation overlap, GPU-aware MPI, the NCCL backend, single-node 8-GPU scaling study, then launch-overhead work (CUDA graphs, which the stream-ordered NCCL exchange allows) where it matters at small per-rank sizes.
6. **Multi-node:** runs across nodes; HDF5/XDMF solution output.

## Decisions from review

1. One halo exchange per stage with a redundant halo-layer-1 reconstruction (section 7).
2. Optional NCCL backend for the halo exchange and reductions (section 1).
3. Dynamic load balancing deferred, with the setup structured to allow it (section 10).
