# Structured blocks of hexahedra

Status: accepted (#440, Stage 1; decisions below, recorded on #440), in implementation. Stages 2
and 3 (octree and Voronoi bulk; a dimension-by-dimension scheme in blocks) are
#957.

Where the mesh is a structured block of hexahedra, the reconstruction can find
a cell's stencil entries by index arithmetic, $(i \pm 1, j, k)$, instead of
reading them, and can read one table set per stencil shape instead of one per
cell. Walls, transitions and irregular cells keep the unstructured path. The
goal is the speed and memory of a structured-grid code in the bulk of
Taylor-Green, channel, isotropic-turbulence and other box meshes, with the same
numerics and the same solver.

## Summary

- On the meshes measured below, 83-100% of the cells (88% on the 3D TENO5
  performance case, 96% on the channel, 95% on the Volvo combustor) have their whole TENO5 stencil inside
  one structured block. Counting mirror images at straight walls as ghost cells
  of the block, 99.5-100%.
- On a block of identical hexahedra every cell away from the walls has the same
  central stencil (80 entries at order 5) and the same sector stencils, up to
  translation. Their pseudo-inverses agree to $5\times 10^{-14}$ relative once
  the entries are matched by offset, but no two cells' tables are bitwise equal,
  and the entries come in a different order in nearly every cell. One shared
  table set therefore changes results in the last bits (see
  [results](#results-contract)).
- Stretched and graded blocks share tables along their uniform directions: one
  table set per $y$ plane of the channel (24,576 cells each), one per spanwise
  column of the graded Volvo mesh (62 cells each).
- A prototype CPU kernel with one shared table set reconstructs 1.43-1.49 times
  as many cells per second as Mallard's TENO5 smooth pass, and the structured
  MUSCL kernel 3.5-3.7 times as many as Mallard's fused MUSCL pass.
- A TENO5 hexahedron holds 38.8 kB, of which 32.0 kB are its reconstruction
  tables. In a block it would hold about 6.8 kB, 5.7 times less.

## Measurements

### Cells in structured blocks

A cell can take the block path if every cell its reconstruction reads lies in
its block: its face neighbors for MUSCL, and the cube of $(2R + 1)^3$ cells
around it for TENO, with $R$ the largest offset along an axis of its central and
sector stencils, measured on a periodic lattice: 1 for TENO3, 2 for TENO4 and
TENO5, 3 for TENO6. *Strict* counts only such cells; *with mirror images* also
counts cells whose cube is cut by one straight boundary plane, whose mirror
images then stand where the missing cells would. Detection as described
[below](#detection), on one rank:

| mesh | cells | MUSCL | TENO3 | TENO5 | TENO6 | stencil shapes (TENO5) |
|---|---:|---:|---:|---:|---:|---|
| `teno5_3d` ($96^3$ Taylor-Green, symmetry walls) | 885k | 93.9 / 99.9% | 93.9 / 99.9% | 88.0 / 99.5% | 82.4 / 98.9% | 1 (13 with mirror images) |
| `examples/taylor_green_3d`, isotropic turbulence (periodic boxes) | | 100% | 100% | 100% | 100% | 1 |
| `examples/channel_retau180` ($192 \times 96 \times 128$, stretched in $y$) | 2.36M | 97.9 / 100% | 97.9 / 100% | 95.8 / 100% | 93.8 / 100% | 92, one per $y$ plane |
| `examples/channel_les` ($64^3$, stretched in $y$) | 262k | 96.9 / 100% | 96.9 / 100% | 93.8 / 100% | 90.6 / 100% | 60, one per $y$ plane |
| `examples/channel_wmles` ($72 \times 24 \times 36$, uniform) | 62k | 91.7 / 100% | 91.7 / 100% | 83.3 / 100% | 75.0 / 100% | 1 (5) |
| `examples/volvo_bluff_body` (graded file mesh, extruded) | 2.71M | 97.4 / 100.0% | 97.4 / 100.0% | 94.8 / 99.9% | 92.3 / 99.9% | 41,046, one per spanwise column (62 cells each) |

Pairs are strict / with mirror images. The Volvo mesh (four planar blocks,
graded, extruded over 64 periodic cells) is one structured block, but no two of
its cells share a stencil shape except along the span: the index arithmetic
applies everywhere, the shared tables only per spanwise column. A periodic box
counts fully only if its block wraps its index across the periodic faces
(below). The other 3D examples on generated boxes
(`explosion_3d`, `sedov_3d`, `detonation_3d`, `shock_bubble_3d`) are uniform
boxes like `teno5_3d`.

Partitions cut blocks. With the layout chosen [below](#data-layout), the
cells within $R$ of a partition face take the unstructured path, as they would at
a wall. On the cuboid parts of the multi-jagged partitioner, TENO5 on `teno5_3d`
keeps 84.2, 80.5, 77.0 and 70.0% of the cells on 2, 4, 8 and 16 ranks (88.0% on
one); MUSCL keeps 91.9, 89.9, 88.0 and 84.2%. A periodic cube of $n^3$ cells per
rank keeps $((n - 4)/n)^3$ for TENO5: 82% at $64^3$, 91% at $128^3$.

### Tables of lattice cells

Every TENO5 table of a $16^3$ periodic box and of a $24^3$ box with symmetry
walls, grouped by the lattice offsets of the stencil entries (and their mirror
flags), with each cell's entries put in the order of offsets before comparing:

| | stencil shapes | entries | largest $\lvert P - P_\mathrm{ref}\rvert / \max\lvert P_\mathrm{ref}\rvert$ | cells bitwise equal to the reference | cells with the reference's entry order |
|---|---:|---:|---:|---:|---:|
| periodic, central | 1 | 80 | $4.3\times 10^{-14}$ | 0 of 4095 | 0 of 4095 |
| periodic, sectors (per face) | 6 | 18 | $1.2\times 10^{-14}$ | 0 | 0-1 |
| walls, central, interior cells | 1 | 80 | $5.0\times 10^{-14}$ | 0 | 0 |
| walls, central, all cells | 93 (125 with mirror flags) | 69-80 | $5.0\times 10^{-14}$ | 0 | 0 |
| TENO3 / TENO4 / TENO6, periodic, central | 1 | 18 / 56 / 122 | $1.6 / 1.6 / 9.4 \times 10^{-14}$ | 0 | 0 |

- One central shape covers 13,048 of the 13,824 cells of the walled box (94.4%):
  the interior cells and the cells near one wall, whose mirror images fill the
  lattice positions beyond it with the same pseudo-inverse. The other 92 shapes
  are cells near an edge or a corner, whose stencils miss the positions beyond
  two planes (one reflection per image) and have 69 entries.
- *Every cell's are different, bitwise and even at 1e-6 relative*
  ([performance.md, change 6](performance.md#6-teno-device-memory-58)) compares
  slot by slot. The difference of $10^{-6}$ is the entry order; matched by offset,
  the tables differ by round-off of the cells' coordinates.

### Prototype kernels (CPU)

A prototype linked against Mallard (the `structured-probe` program described in
the pull request, not part of the repository) runs, on the same cells, Mallard's
own kernel (`TENO::calc_cell_face_values`, `MUSCL::calc_cell_face_values`) and
a structured kernel with the same arithmetic. The structured TENO5 kernel is
Mallard's per-thread smooth pass with three changes: the stencil entries
$c + o_s$ from one list of offsets $o_s$, the pseudo-inverse $\mathbf{P}$ of one
reference cell, and the face-point basis $\psi_l(\boldsymbol{\xi}_q)$ tabulated once per face
direction instead of evaluated from each face's points. The structured MUSCL kernel
takes the six face neighbors at offsets $\pm 1, \pm N_x, \pm N_x N_y$, the
least-squares weights of the reference cell (`LSQGradientFunctor::weights`)
and its face-centroid offsets for the limiter. Both write `face_solution`
through the mesh's `faces_of_cell`.

Apple M4 Pro, 2 threads (Kokkos Threads), on a shared machine; best of 5 calls
after one warm-up. TENO5: a $64^3$ box with symmetry walls, the 216,000 cells at
least 2 cells from the walls, every cell smooth (`troubled_threshold = 1e9`),
gradients written as in viscous runs. MUSCL: a $128^3$ box, the 2.0M cells off
the walls.

| kernel | cells/s, input order | cells/s, Hilbert order | per-cell tables read |
|---|---:|---:|---:|
| TENO5, Mallard smooth pass | 3.88e5 | 4.07e5 | 22.7 kB |
| TENO5, shared table, per-cell entry lists | 5.61e5 (x1.44) | 6.09e5 (x1.49) | 0.3 kB |
| TENO5, shared table, entries at $c + o_s$ | 5.54e5 (x1.43) | | 0 |
| MUSCL, Mallard fused pass | 7.84e6 | 7.46e6 | |
| MUSCL, structured | 2.73e7 (x3.48) | | 0 |

- Face states differ from Mallard's by at most $5.3\times 10^{-14}$ (TENO5) and
  $2.1\times 10^{-14}$ (MUSCL) of each variable's range, from the tables'
  round-off; in the reference cell they are bitwise Mallard's.
- On the CPU the TENO5 pass is bound by its arithmetic (about 46,000 floating-point operations
  per cell): sharing the table removes 22.7 kB of reads per cell, and finding
  the entries by index changes nothing measurable. MUSCL gains from not reading
  the cells' and faces' coordinates and not solving the least-squares system per
  cell.
- These are kernel upper bounds on the CPU. The GPU was not measured. There the
  3D smooth pass is bound by the latency of its dependent loads, stencil cell then
  state, about 4,000 cycles per entry
  ([performance.md](performance.md#small-meshes-strong-scaling-regime)); an
  entry at $c + o_s$ removes the first load, and a shared $\mathbf{P}$ in shared memory
  replaces 34 loads per entry per cell with loads shared by the 32 cells of a team.
  As a ceiling: the arithmetic of a $96^3$ stage at the A100's FP64 peak takes
  4.2 ms, and its face states (2.2 kB per cell) 1.0 ms at full bandwidth, against
  42 ms for today's smooth pass.

## Detection

After partitioning, migration, the halo and the cell numbering, before the
reconstruction's setup, each rank finds its blocks among its owned cells:

1. **Hexahedra with three pairs of opposite faces.** A cell qualifies if it has 6
   faces that pair up (each face's centroid offset most nearly opposite one
   other face's, in the cell's frame across periodic faces).
2. **Labels.** From a seed, each face gets one of $\pm a, \pm b, \pm c$. Across a
   face, the neighbor's shared face gets the opposite label, the face opposite
   it the same label, and each of its other faces the label of the seed-side
   face it is most nearly parallel to. A neighbor reached twice with different
   labels ends the block there (an O-grid's singular line, for example).
3. **Boxes.** A block is a box of lattice indices $[0, N_a) \times [0, N_b)
   \times [0, N_c)$ of owned cells; a labeled region that is not a box splits into
   boxes. On generated meshes and multi-jagged parts, each rank has one box per
   generated block.
4. **Stencil shapes.** A cell's shape is the geometry of the cells its
   reconstruction reads: their offsets and volumes in the frame of the cell,
   rounded to $10^{-9}h$. Rounding keeps the coordinates' round-off
   ($10^{-14}h$ on generated meshes) inside one shape, while two cells of a
   stretched or graded mesh differ by the stretching ($10^{-2}$ or more). A
   block's shapes come in three kinds: one shape (uniform blocks), one per line
   or plane (stretched along one or two axes: the channel), or one per cell
   (graded along every axis). The block stores its kind and per-axis shape
   indices, so a cell's shape is a function of $(i, j, k)$.
5. **Eligible cells.** The cells whose reconstruction reads only cells of their
   own box take the block kernels; the cells within $R$ of a box face, and the
   troubled passes, take the unstructured kernels.

Detection reads only geometry and connectivity, so it gives the same shapes for
any numbering. The detection loop of the prototype, a host breadth-first search,
takes 1-3 s per million cells; it is part of the mesh setup, not of the step.

## Data layout

**Blocks as a cell numbering.** The block path adds no state arrays
and no input keys: it is automatic. With `[mesh] cell_order = "auto"` (the
default), a rank that has blocks numbers each block's owned cells
lexicographically ($i$ fastest), then its other owned cells as `auto` decides
today, then each halo layer as now. `"input"` and `"hilbert"` keep their
meaning, so they leave every cell on the unstructured path. A block is then its first cell and
its extents, and the cell at offset $(d_i, d_j, d_k)$ from cell $c$ is
$c + d_i + N_a (d_j + N_b d_k)$. `W`, `face_solution`, the fluxes, gradients and
every unstructured kernel keep their layout; owned cells stay below
`n_owned()`, which the halo exchange, the overlap and owner-computes partition
faces rely on.

- Cells within $R$ of a box face go unstructured, including at partition faces,
  since halo cells are numbered after owned cells (the partition counts
  above). A block that spans a periodic direction on one rank can wrap its
  index along that axis (one comparison per entry) and lose no cells there.
- Faces keep the numbering of the cells' order and their global orientation;
  the block kernels write face states through `faces_of_cell`, as the
  prototype does. Implicit face indices would need block faces numbered by
  direction and one orientation per direction, which holds for generated meshes
  but not for files; not in Stage 1.

**Not chosen: padded blocks with ghost layers.** Each block's states in their
own array with $R$ layers of ghost cells, filled every stage: halo cells by
the exchange's unpack (into the ghost slots), other ghost cells by a copy from
the cells they stand for. Every lattice cell, partition shells included, takes
the block path (88% instead of 70% for TENO5 on 16 ranks of `teno5_3d`), at the
cost of a second copy of the block cells' states, a copy kernel per stage, and
an exchange plan that writes into two layouts.

**Tables.** One table store holds a table set per stencil shape: the central
and sector pseudo-inverses in a canonical entry order (by lattice offset), the
offsets $(d_i, d_j, d_k)$ and mirror faces of the entries, the basis means, the
moments, the scale $h$, and the face-point basis $\psi_l(\boldsymbol{\xi}_q)$
per face direction. The unstructured kernels reach a cell's set through a
per-cell index; the block kernels through the block's shape function. This is
the store that deduplicating identical tables (#431) builds; the block path
adds the index arithmetic and the shape function, not a second store. Cells of
no block (tetrahedra, transitions, graded cells) keep their own tables in it.

**Setup.** The reconstruction's setup fits one cell per shape instead of every
cell (TENO5 setup takes 4.6 s per million hexahedra on an A100 and minutes on
CPUs, #916). The shapes are agreed over the ranks at setup: each shape's table
comes from its cell with the lowest global id, computed by the rank that owns
that cell and sent to the ranks that use the shape (MPI, at setup, like the
face matching of [mpi.md](mpi.md#3-distributed-dual-graph)). A rank's tables
then do not depend on the partition. The TENO cache stores the shapes' tables
and the cells' indices (a new format version).

## Kernels

**TENO smooth pass.** As the prototype: the smooth pass's arithmetic, with the
entries at $c + o_s$, $\mathbf{P}$ and $\psi$ from the block's table set, and
$h$ from it. On GPUs it keeps the team pass (32 cells by `N_CONSERVATIVE`
threads, compile-time chunks); 32 consecutive cells of a block row read the
same $\mathbf{P}$ and $\psi$, held in team scratch (21.8 kB and 9.2 kB for one
TENO5 shape) or read through the cache, and their states at consecutive
addresses. Sigma, the troubled queue and the gradients of smooth cells are
written as now, so the troubled passes and everything downstream are unchanged.
The kernel is a template on the entry source (per-cell lists or offsets), so the
block and unstructured passes share one body.

**Troubled passes.** Unstructured, on the queue the smooth passes fill. For a
block cell they read its shape's table set and find its entries by offset,
through the same template parameter; troubled cells are a few percent of the
cells, so a branch per cell costs little.

**MUSCL.** Per block cell: the six face neighbors by offset; the gradient from
the shape's least-squares weights; the limiter from the shape's face-centroid
offsets and $h$; face states through `faces_of_cell`. Cells on characteristic
boundaries (vertex gradients) are not eligible.

**Mirror images.** A cell near one straight boundary plane has the stencil
shape of the interior with mirror images in place of the cells beyond the plane
(measured above). The block kernel can serve it with the interior shape's table
if it evaluates each image's ghost state as `entry_W` does (the boundary
condition applied to the reflected cell's state at the nearest face of the
plane). This lifts `teno5_3d` from 88% to 99.5% of the cells. Part of Stage 1,
after the interior.

## MPI, NCCL and the step graph

- **Exchange.** Unchanged: halo states travel in `W`'s layout. Block cells are
  halo-independent by construction (their reconstruction reads only owned cells
  of their box), so the block kernels run among the cells that
  `cells_independent_of_halo` returns, on the execution-space instance that
  overlaps the exchange; partition shells run after it with the unstructured
  kernels. Owner-computes partition faces (#305) is unchanged.
- **NCCL.** No new communication in the step. The only new cross-rank exchange,
  the shapes' tables, is at setup, over MPI, as all setup communication is.
- **Step graph.** The block kernels are launches of fixed extents, set at setup,
  inside the existing CUDA graph; nothing is decided on the host per step. The
  split between block and unstructured cells is fixed device data.

## Results contract

Following #429's split, as decided for this change:

- **Independent of rank count, thread count and restarts (kept).** A cell's
  result must not depend on which path reconstructed it, since that depends on
  the partition. Two rules give this: every cell of a shape, on either path,
  reads the same table set (the store's, agreed over the ranks); and the two
  paths share one kernel body with its rounding made explicit (`fma()` where the
  compiler contracts today, as #931 proposes for the staged smooth pass), so
  that two kernels compiled from it round alike. A test compares the block and
  unstructured paths on the same cells bitwise, in CPU and GPU CI.
- **Bitwise identical to the previous `main` (not kept).** The shared tables
  differ from the per-cell tables by $5\times 10^{-14}$ relative and sum their
  entries in another order, so results change once in the last bits: face states
  by $5\times 10^{-14}$ of each variable's range per pass in the prototype. This
  is #429's "relax when justified": a measured speedup, restart files within
  tolerance of `main`, and an unchanged verification manual (#342).

## Memory

Bytes per cell after setup, `mallard check` on $64^3$ boxes (Kokkos
allocations; with CUDA the pseudo-inverses live outside Kokkos, change 2 in
[performance.md](performance.md#2-teno-setup-stencil-tables-streamed-to-the-device-host-memory)):

| | TENO5, today | TENO5, block cell | MUSCL, today and block cell |
|---|---:|---:|---:|
| reconstruction tables | 32.0 kB | 0 (one table set per shape: 36.8 kB per block) | 0 |
| faces and gradients (face states 2.2 kB, troubled-pass scratch) | 4.20 kB | 4.20 kB | 0.65 kB |
| mesh and geometry | 2.29 kB | 2.29 kB | 2.26 kB |
| solution and stages, boundaries | 0.28 kB | 0.28 kB | 0.28 kB |
| total | 38.8 kB | 6.8 kB | 3.2 kB |

A 3D TENO5 hexahedron measured 40.7 kB on an A100 (`cudaMemGetInfo`,
[performance.md](performance.md#7-memory-held-from-setup-551)), which caps a
TENO5 box at about 1.8M cells on 80 GB; at 6.8 kB per block cell, about
10M. What remains is the face states and the mesh: fusing reconstruction and
fluxes on block faces, and dropping per-face geometry inside blocks, is
filed as a follow-up (#965). MUSCL gains throughput, not memory.

## Documentation

The block path is automatic: no input keys, nothing to turn on. Once it is
implemented, its documentation describes what it does in the current state,
with Mallard's own names (structured blocks, stencil shapes, the unstructured
path):

- **User guide.** [Performance](../guide/performance.md) gets a section on
  structured blocks: which cells take the block path (hexahedra of a structured
  block whose stencils lie in the block), the speed and memory per cell against
  the unstructured path, and the numbers of the A100 measurements.
  [Meshes](../guide/meshes.md) says how to get more of a mesh onto it: generated
  `cartesian` meshes are blocks throughout; stretching keeps them blocks, with a
  table set per plane; file meshes gain from uniform spacing in the bulk and
  hexahedral regions away from walls. [Parallel](../guide/parallel.md) says that
  the cells within a stencil of a partition face take the unstructured path, so
  the share falls as the cells per rank fall. [Numerics](../guide/numerics.md)
  states that the numerics are the same on both paths.
- **`mallard check`.** The Mesh section reports, per reconstruction, the share
  of cells on the block path, the number of blocks and stencil shapes, and the
  share lost to partition faces with `--ranks`; the Memory section's
  reconstruction tables category shrinks accordingly (one table set per shape),
  and the run log's Mesh section and the run record's setup provenance carry the
  same counts, as they carry the cell order today.
- **Performance page numbers.** `docs/guide/performance.md` and
  [performance.md](performance.md) get the cells/s and device bytes per cell of
  `teno5_3d`, `air_3d` and a channel case, block path against unstructured, from
  `perf_suite.py` on an A100.
- **README and website.** One feature bullet: structured blocks of hexahedra
  reconstructed with implicit neighbor indexing and shared tables, at the
  measured speedup and memory per cell; the website's features page in sync.
- **Release notes.** The 0.11.0 headline: hexahedral meshes run faster and in
  about a sixth of the memory per cell with TENO, automatically, with
  the measured numbers; the `changelog/` fragment of the implementation PR
  (category `added`, and `changed` for the last-bits change of results) carries it.

## Validation (Phase 2)

- Detection: unit tests on generated boxes (uniform, stretched, periodic,
  partitioned on 1-3 ranks), a file mesh in Gmsh's numbering, and meshes with
  no blocks (tetrahedra, O-grids).
- Paths: the block and unstructured kernels bitwise equal on the same cells
  (both orders 3-6 and MUSCL; CPU and GPU CI); results independent of rank count
  bitwise, as today.
- Against `main`: restart files of the verification cases within tolerance;
  verification manual unchanged.
- Performance: `perf_suite.py` on an A100 (`teno5_3d`, `air_3d`, a channel
  case), interleaved best of 3, and device memory per cell from `mallard check`.

## Decisions

Decided by the maintainer on this note (#967), recorded on #440:

1. **Layout.** Blocks are a cell numbering. Cells near partition faces stay on
   the unstructured path; there are no new state arrays and owned cells stay
   first.
2. **Results.** Every cell of a stencil shape reads one shared table set on
   either path, through one kernel body with explicit `fma()`, with the tables
   agreed over the ranks at setup. Results stay bitwise independent of the rank
   count; the one-time change of the last bits against `main` is accepted.
3. **Scope.** Stretched and graded structured blocks are in Stage 1, with
   tables per plane, column or cell as the mesh requires.
4. **Mirror images** at straight walls are in Stage 1.
5. **Order.** The shared table store (canonical entry order, a table set per
   stencil shape, the cells' index) is #431's and lands first. #440 adds the
   block numbering, the $(i, j, k)$ to shape map and the kernels.

## Implementation

One pull request per step, as a stack on #431:

1. Detection and the block numbering.
2. The $(i, j, k)$ to stencil shape map, on #431's table store.
3. The TENO and MUSCL block kernels.
4. Mirror images at straight walls.
5. `mallard check` reporting.
6. The documentation of [Documentation](#documentation).
