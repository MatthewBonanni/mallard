# Design: distributed memory (MPI)

Status: accepted design (reviewed in #32), implemented through milestone 5 (see Milestones). Tracks issue #2.

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
- **NCCL** (optional, `Mallard_ENABLE_NCCL`): `ncclSend`/`ncclRecv` inside a group and `ncclAllReduce`, enqueued on the Kokkos execution stream, so the exchange is stream-ordered with the kernels (no host synchronization) and can be captured in a CUDA graph. The NCCL communicator is created from the MPI communicator. **Deferred** (milestone 5 measurements): with GPU-aware MPI over NVLink, or over InfiniBand with GPUDirect RDMA (3 us, 26 GB/s between GPUs on different nodes), an exchange costs about 0.1 ms per stage and is overlapped with the interior reconstruction, and a step launches about 30 kernels, so neither stream ordering nor CUDA graphs would recover much; the remaining strong-scaling loss is load imbalance (below).

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
2. **Overlap:** reconstruct the owned cells whose stencils and face neighbors are all owned (`FaceReconstruction::cells_independent_of_halo`, exact from the stencils rather than a distance bound) while messages are in flight, on a second execution space instance.
3. Wait, unpack, then reconstruct the near-boundary owned cells and halo layer 1 on the default instance, where they fill the device as the first instance drains. Then come the steps that need every cell: TENO's troubled-cell passes and the fluxes of the faces of owned cells (faces between two halo cells are skipped).

Implemented for TENO, the reconstruction that needs deep halos; FO and MUSCL exchange first and then reconstruct. The first stage of a step reuses the halo that the time-step computation has just filled, so an SSPRK3 step exchanges three times, not four.

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

### 10. Dynamic load balancing

Status: prototyped and measured on A100s (milestone 7); not adopted, because a rebalance does not pay for itself on the cases below. The prototype is kept in draft PRs #103 (weighted partitioners), #104 (runtime migration), #105 (cost model and trigger) and #108 (TENO data migration).

**Why it was tried.** TENO's troubled cells (near shocks) cost several times a smooth cell, cluster on the ranks a shock crosses, and move with the shocks, so a static partition can't balance them. Chemistry will add larger per-cell cost variation, which it balances by moving chemistry states between ranks without touching the mesh (`chemistry.md`, decision 9).

**Invariant.** Results must stay bitwise identical with and without rebalancing, as they are on any rank count. The prototype keeps it: stencils, pseudo-inverses and face orders are functions of global ids, reductions are exact or ordered by global keys, and state and per-cell data move as raw bits. Its tests compare rebalanced runs with serial ones bitwise (TENO with mirror faces and periodic seams, MUSCL Navier-Stokes with Dirichlet and average-pressure boundaries, 2D and 3D, 2-4 ranks, CPU and A100).

#### 10.1 The prototype

- **Cost model.** Per-cell weight `16 (1 + (kappa - 1) f_c)`, with `f_c` the fraction of steps cell c was troubled (a per-step counter kernel) and `kappa` fitted by least squares over ranks from their busy times, `B_r = a N_r + b T_r`.
- **Busy time.** Step time minus host time blocked in MPI waits, measured in the last 5 steps of every check window with the device finished before the halo wait. Without that fence, on GPUs the `Kokkos::fence` before each send also waits for communication, and the measured imbalance read 1.3-1.5 where the per-phase compute imbalance was 1.05-1.25.
- **Trigger.** Every `rebalance_interval` steps: rebalance when max/mean busy time exceeds a threshold, the saving (slowest rank brought to 1.05 x the mean) over the expected horizon beats twice the predicted cost, and total rebalancing stays within a budget of the run's wall time. Every input to the decision must be the same on all ranks: one rank-local term made ranks decide differently and mismatch collectives.
- **Repartitioning.** The `DistributedMesh` (blocks, node directory, dual graph) is kept for the run. Hilbert: the sample sort carries the weights and the curve is split at weighted prefix sums. dKaMinPar: node weights, then parts relabeled to keep the most weight in place. dKaMinPar 3.7's node balancer never terminates when single cells weigh more than a few percent of a part, so tiny parts fall back to the curve.
- **Migration.** Owned state goes to the new owners by global id; the local mesh, halo, boundaries and outputs are rebuilt as at startup. TENO data is not recomputed: each reconstructed cell's tables are exported with global keys, kept or fetched from the previous owner, and taken by `init()` instead of precomputing (recomputing would cost about 330 us per cell on one core).

#### 10.2 Measurements

2D TENO5, Hilbert partition, SSPRK3, A100-80GB with GPU-aware MPI; 8 GPUs on one node, 16 GPUs as 4 nodes x 4 over InfiniBand. Riemann problem (configuration 3, quadrilaterals, HLLC) and double Mach reflection (triangles, RHLL), each run to its final time. *Imbalance* and *excess* are means over the run's checks of the measured busy times: max/mean, and the slowest rank's time above 1.05 x the mean. The *bound* is the stepping time an instantaneous, free rebalance at every check could save.

| case | GPUs | cells/GPU | steps | ms/step | imbalance | excess (ms/step) | bound |
|---|---|---|---|---|---|---|---|
| Riemann 1M | 8 | 125k | 16,670 | 3.80 | 1.17 | 0.41 | 6.9 s of 64 s (11%) |
| Riemann 1M | 16 | 62.5k | 16,670 | 2.73 | 1.28 | 0.50 | 8.4 s of 46 s (18%) |
| Riemann 4M | 8 | 500k | 34,840 | 11.2 | 1.15 | 1.00 | 35 s of 390 s (9%) |
| Riemann 4M | 16 | 250k | 34,840 | 7.06 | 1.29 | 1.30 | 45 s of 246 s (18%) |
| DMR 1.84M | 8 | 230k | 20,274 | 5.57 | 1.12 | 0.33 | 6.7 s of 113 s (6%) |
| DMR 1.84M | 16 | 115k | 20,274 | 3.60 | 1.19 | 0.40 | 8.1 s of 73 s (11%) |

- **Where the time goes** (Riemann 1M, 8 GPUs, per stage with fences between phases): the smooth TENO pass takes 0.85 ms on every rank, the troubled passes 0.04-0.31 ms, the fluxes 0.11 ms. Splitting the troubled passes (milestone 5) already removed most of the imbalance this section was meant to fix.
- **A rebalance costs** 5.0 s at 125k cells/GPU and 21.6 s at 500k (TENO export 1.3 s, TENO repacking 1.0 s, `cells_independent_of_halo` 1.0 s, data exchange 0.7 s, local mesh 0.6 s, partition 0.2 s at 125k); recomputing the stencils instead would add the 6-7 s of the startup's TENO setup. A weighted curve split moves 9-13% of the cells (85-130k of 1M) or 6% (252k of 4M).
- **The pattern moves faster than a rebalance pays back.** Forced rebalances (Riemann 1M, 8 GPUs) took the imbalance from 1.25 to 1.05; it was back to 1.2-1.3 within 2,000-4,000 steps, and a partition weighted for earlier shock positions became worse than the uniform one. Stepping took 3.85-4.03 ms instead of 3.80, plus 5 s per rebalance. On Riemann 4M (8 GPUs) one rebalance cost 21.6 s and left 11.2 ms/step unchanged. The probe steps cost under 1%.

#### 10.3 Why not, and what would change it

If the imbalance returns over `R_d` steps after a rebalance, rebalancing every `R` steps captures a fraction `1 - R / (2 R_d)` of the excess `e` per step at cost `C` each, at best with `R = sqrt(2 R_d C / e)`. With the measured `R_d` of 3,000-6,000 steps, even a rebalance 3x cheaper than the prototype's (1.5 s per 125k cells) nets about zero on Riemann 1M at 16 GPUs and about 2% on Riemann 4M at 16 GPUs, within the optimism of assuming each rebalance resets the imbalance to 1.05. It pays clearly only below about 5 us per cell, a few tenths of a second per rank, which a full local rebuild does not reach.

It would be worth revisiting if:

- imbalance grows relative to the smooth work, e.g. at many more ranks per mesh or with costlier troubled reconstructions (3D, higher order), and lasts longer than a few thousand steps;
- an incremental scheme moves only cells near partition boundaries and patches the local mesh and TENO tables in place, instead of rebuilding them;
- a long-lived imbalance comes from a source that moves slowly, e.g. chemistry's long-term cost (which `chemistry.md` balances by redistributing chemistry states instead).

The structure the prototype relied on stays in place: setup is a function of a distributed cell set, everything is keyed by global ids, weights are a small addition to the partitioners (#103), and output and restart do not assume a fixed partition.

### 11. Communication patterns at scale

Setup exchanges are dense `alltoallv` calls, whose count arrays alone are O(ranks) per rank and whose latency grows with the rank count. The face matching and migration genuinely talk to many ranks, but halo growth and node-coordinate fetches have sparse patterns (a rank's partition neighbors and the few ranks whose blocks hold its cells). Beyond about 10k ranks these should move to MPI neighborhood collectives or a sparse NBX exchange (nonblocking sends, `MPI_Ibarrier` to detect completion); planned with milestone 6.

### 12. Periodic boundaries

See `periodic.md`. Periodic node classes are found like faces: zone nodes are
hashed by their matching-grid cell to the rank that pairs them, and the classes
resolve at the nodes' block owners by a few rounds of key propagation. No rank
gathers the periodic zones.

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
4. **Scalable setup:** HDF5 mesh format and converter, distributed read, distributed dual graph, dKaMinPar, migration. Done: with generated meshes or HDF5 mesh files no rank holds the global mesh in a distributed run (Gmsh files are still read whole by every rank), and restart files are read by global id, each rank reading only its cells. Setup only (one FO step), CPU, peak memory per rank: 4.1M hexahedra took 6.0 GiB on every rank count before (each rank built the global mesh) and now 3.1 / 1.7 / 0.9 / 0.5 GiB on 2 / 4 / 8 / 16 ranks (setup 40 s -> 19 / 11 / 6.4 / 4.5 s); 16M quadrilaterals 8.7-11 GiB per rank before, now 6.5 / 3.3 / 1.7 / 0.9 GiB; 64M quadrilaterals set up on 16 ranks in 3.3 GiB each.
5. **Performance:** communication/computation overlap, GPU-aware MPI, the NCCL backend, single-node 8-GPU scaling study, then launch-overhead work (CUDA graphs, which the stream-ordered NCCL exchange allows) where it matters at small per-rank sizes. Done except NCCL and CUDA graphs, deferred (section 1).
   - **Profile.** Profiled at 8 GPUs on 1M cells. The TENO troubled-cell pass ran one heavy thread per cell, and a rank holds fewer troubled cells than one wave of threads. The pass therefore took one thread's latency (0.6-0.9 ms per stage) on ranks crossed by shocks, and about nothing on the others. The exchanges cost about 0.1 ms per stage each.
   - **Fixes.**
     - The troubled passes are split per (cell, sector), per (cell, face, characteristic variable), per (cell, face) and per cell, with the same arithmetic.
     - The exchange is overlapped with the interior reconstruction (section 7).
     - Flux and RHS work is limited to owned cells.
     - The first-stage exchange is reused from the time-step computation.
     - Results stay bitwise identical.
   - **Scaling.** 2D Riemann problem (configuration 3), TENO5 on quadrilaterals, HLLC, SSPRK3, double precision, A100-80GB GPUs. Times are seconds per 50 steps, with GPU-aware MPI and the Hilbert partition. Up to 8 GPUs share one node over NVLink. The 16-GPU runs use 4 GPUs on each of 4 nodes, over HDR InfiniBand with GPUDirect RDMA.

     | cells | 1 GPU | 2 | 4 | 8 | 16 | efficiency at 8 / 16 |
     |---|---|---|---|---|---|---|
     | 1M | 0.938 | 0.553 | 0.311 | 0.190 | 0.152 | 62% / 38% |
     | 4M | 3.50 | 1.93 | 1.00 | 0.534 | 0.316 | 82% / 69% |
     | 16M | | | | 1.93 | 1.01 | 95% from 8 to 16 |
     | 1M per GPU (weak) | 0.938 | 0.999 | 1.000 | 1.005 | 1.013 | 93% / 93% |

     - Before this milestone, 1M cells on 8 GPUs took 0.287 s (1.038 s on 1 GPU).
     - The 8 GPUs give the same times on one node as on 2 nodes x 4.
     - Host-staged MPI is 15-25% slower; the overlap recovers about 10% of it.
     - dKaMinPar partitions give the same times within a few percent, and 0.141 s on 1M cells at 16 GPUs.
     - What remains at small per-GPU sizes is the cost of troubled cells concentrated on the ranks crossed by shocks, and of the smooth TENO pass's last partial wave of threads. Dynamic load balancing (section 10) was measured against the first and does not pay off.
6. **Multi-node:** runs across nodes; HDF5/XDMF solution output. Runs across nodes work, with GPUDirect RDMA (milestone 5 table). HDF5/XDMF output is still to do.
7. **Dynamic load balancing** (section 10): prototyped (weighted partitioners, runtime migration with TENO data carried over, measured cost model and trigger, bitwise tests) and measured on 8 and 16 A100s; not adopted, since a rebalance costs more than the imbalance it removes before the shocks move on. The prototype stays in draft PRs.

## Decisions from review

1. One halo exchange per stage with a redundant halo-layer-1 reconstruction (section 7).
2. Optional NCCL backend for the halo exchange and reductions (section 1).
3. Dynamic load balancing deferred, with the setup structured to allow it (section 10); prototyped and measured in milestone 7, and not adopted.
