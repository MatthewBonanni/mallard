# Design: adaptive mesh refinement

Status: proposed, for review (#161). Nothing here is implemented. Decisions
are marked **Decision**; the open questions for review are at the
[end](#open-questions).

Thin fronts (detonations, shocks, flames) need cells of a few micrometers
over a band a few dozen cells thick. A uniform mesh at that spacing wastes
almost all of its cells. This note designs dynamic h-refinement for
Mallard's unstructured mixed meshes. It must stay compatible with TENO-E,
MPI with bitwise rank-count independence, exact restarts, chemistry, LES
and NSCBC. It ends with a staged plan, the alternatives, and a
recommendation.

## Summary

**The cost that decides AMR in Mallard is TENO-E's setup, not the tree.**
A TENO-E stencil search and pseudo-inverse costs about 330 us per cell on
one core in 2D and several ms in 3D (TENO5; [mpi.md §10](mpi.md#10-dynamic-load-balancing),
#140). One time step costs 0.17 us per cell (3D TENO5, A100;
[performance.md](performance.md)). A cell whose tables are rebuilt
therefore costs as much as roughly a thousand of its steps. Along a front
that moves a third of a cell per step, the rebuilds make AMR about as
expensive as the uniform fine mesh it replaces (section 9). Below about
5 us per cell, AMR wins by an order of magnitude. That is also the
threshold at which mpi.md §10.3 found dynamic load balancing pays.

**Recommendation** (section 12):

1. Make the TENO setup fast, incremental and device-resident first
   (stage 0, #140). It pays off on its own: startup time, restarts, and
   dynamic load balancing.
2. Then build a forest of refinement trees over the input mesh, with
   hanging faces, 2:1 balance and isotropic refinement of all six cell
   types. Ship it first for static and restart-time adaptation, then for
   slow fronts.
3. Commit to frequent adaptation for shocks and detonations only after
   stage 0 measures under about 5 us per cell.
4. Until then, run moving fronts in a frame moving with the front, on
   statically refined meshes.

| # | Decision | Main alternative | Why |
|---|---|---|---|
| 1 | h-refinement by element subdivision with **hanging faces**, 2:1 balanced across faces and edges | Conforming red-green closure | A cell-centered FV scheme handles non-conforming faces exactly and conservatively. Green closures of mixed elements need dozens of templates, create the thin cells TENO-E is fragile on, and must be undone before refining again |
| 2 | A **forest of trees**: every input cell is a root, leaves form the flat mesh the solver sees | Flat remeshing; block-structured AMR | Keeps the solver's data structures. Gives stable hierarchical ids, coarsening by families and a space-filling-curve order |
| 3 | **Isotropic** refinement only at first (tri/quad 1:4, tet/prism/hex 1:8, pyramid 1:10); anisotropic later, and only for quads, hexes and prisms | Anisotropic from the start | Anisotropic children create the high-aspect-ratio and mixed configurations that TENO-E stencils are unstable on today ([teno_e.md](../numerics/teno_e.md)) |
| 4 | **Level-view stencils**: each cell builds its stencil from the mesh as seen at its own level, with finer regions replaced by virtual parent cells (restricted averages) | Stencils from raw leaves | Coarse cells keep their tables bitwise when neighbors refine, so the rebuild band halves. Coarse stencils do not become one-sided bundles of fine cells |
| 5 | **Conservative high-order prolongation** from the parent's TENO polynomial, scaled toward the mean for positivity (Zhang-Shu). Restriction by volume-weighted averaging | Injection | Design order through refinement, conservation to round-off, positivity |
| 6 | **Global time step**, adaptation only between full steps | Subcycling (Berger-Colella) | Local time stepping was rejected as a general feature. It would save at most about a quarter of the work in front-dominated cases (section 6) |
| 7 | **Space-filling-curve partition of leaves**, rebalanced at every adaptation, families kept on one rank | Graph repartitioning | Incremental (only chunk ends move), deterministic, coarsening stays local; refinement makes imbalance large and long-lived, so balancing pays |
| 8 | **Ids from the tree** (root gid, child path); adaptation flags and closures are functions of the solution only | Renumbering after each adaptation | Bitwise rank-count independence, and restarts on any rank count |
| 9 | **In-house forest**, algorithms after p4est/t8code | p4est or t8code as dependencies | GPL-2.0 against Mallard's Apache-2.0. Neither is GPU-resident. Neither knows TENO stencils, which are most of the work |
| 10 | Stage 0 (fast incremental TENO setup) is a **go/no-go gate** for fast-front AMR | Build AMR first and optimize later | The economics of AMR depend on the setup cost (section 9) |

## Goals

- Dynamic refinement and coarsening on 2D and 3D mixed meshes (triangles,
  quadrilaterals, tetrahedra, prisms, pyramids, hexahedra), driven by
  solution sensors and by user-defined regions.
- Every reconstruction (FO, MUSCL, TENO-E orders 3 to 6) and every physics
  module keeps working on adapted meshes.
- Conservation to round-off through adaptation.
- Design order of accuracy through static refinement interfaces and
  through prolongation.
- Results bitwise independent of the rank count, the partition and
  rebalancing, as today ([mpi.md](mpi.md)).
- Exact restarts on any rank count, including mid-adaptation history.
- No cost and no change without `[adaptation]`: unadapted runs stay
  byte-identical and keep their speed.

## Non-goals

- Local time stepping or subcycling (section 6).
- Order adaptation (p-refinement): TENO-E's order is a global choice.
- Metric-based anisotropic remeshing (moving nodes, swapping edges). The
  forest only subdivides cells.
- Refining toward curved geometry. New boundary nodes lie on the input's
  straight faces until #162 (curved boundaries) provides a surface to
  project them onto.
- Adjoint-based (output-targeted) error estimation. Mallard has no adjoint.

## 1. Refinement strategy

### 1.1 Hanging faces or conforming refinement

| | Hanging faces (non-conforming) | Conforming red-green | Conforming bisection |
|---|---|---|---|
| Element types | all, one template each (section 2) | all, plus transition templates for every pattern of refined edges per type (dozens for hexes and pyramids) | simplices only ([Rivara 1984](#ref-rivara-1984); [Stevenson 2008](#ref-stevenson-2008)) |
| FV fluxes | exact: the coarse cell's face splits into subfaces, and each subface flux is computed once and summed by both sides | conforming | conforming |
| Cell quality | children similar to the parent (with Bey's rule for tetrahedra) | green cells are thin, and green closures must be removed before the next refinement | bounded classes for simplices |
| Locality of a change | the refined family and 2:1 ripple | the closure spreads, and coarsening undoes it | bisection closure spreads |
| TENO-E | cells keep their shapes. Stencils near level jumps mix cell sizes (section 4) | green cells: aspect ratios 2-4 and mixed tilings, the configurations with growing modes in [teno_e.md](../numerics/teno_e.md) | as red-green for simplices; hex and prism meshes are excluded |

**Decision 1: hanging faces with 2:1 balance.** For a cell-centered FV
scheme, a hanging node costs nothing in the flux: the face list of the
coarse cell grows by subfaces. The FE complications (constraining hanging
DOFs) do not arise. Balance is enforced across faces and, in 3D, across
edges:

- Face balance bounds a coarse face at 4 subfaces.
- Edge balance keeps vertex neighborhoods (stencil gathering, halo layers,
  viscous vertex fits) from mixing levels three apart.
- Vertex balance is not required.

The balance closure only ever refines, so its result does not depend on the
order in which it is applied (section 7).

### 1.2 Tree or flat

**Decision 2: a forest of trees** ([Burstedde, Wilcox & Ghattas 2011](#ref-burstedde-2011);
[Holke, Knapp & Burstedde 2021](#ref-holke-2021) for hybrid elements;
[Khokhlov 1998](#ref-khokhlov-1998) for the cell-based tree of compressible
reacting codes).

- Each input cell is the root of a tree. Its children follow from its type
  and refinement template, and the leaves are the cells of the mesh.
- The solver keeps working on a flat local `Mesh` of leaves. The forest is
  host-side bookkeeping used only to adapt, to name cells, to order them
  and to coarsen them.
- Internal nodes keep their geometry (volume, centroid, moments). They
  serve as virtual cells for stencils (section 4) and as the targets of
  coarsening.

A flat approach regenerates an unrelated mesh at each adaptation (MMG /
ParMmg remeshing, [Dapogny, Dobrzynski & Frey 2014](#ref-dapogny-2014)).
It needs conservative remapping between unrelated meshes, offers no
coarsening by restriction, gives no stable ids, and is limited to
tetrahedra. Block-structured AMR (AMReX, [Zhang et al. 2019](#ref-zhang-2019);
PeleC, [Henry de Frahan et al. 2023](#ref-henry-de-frahan-2023)) works on
Cartesian patches and represents geometry by embedded boundaries. It would
be a different code.

### 1.3 Anisotropic refinement

Boundary layers in Mallard are already resolved by prism and hex layers in
the input mesh (LES design: walls resolved, y+ about 1). Flames and
detonations wrinkle, so a normal-only refinement must follow a changing
direction.

The cost is TENO-E. Anisotropic children of quads, hexes and prisms raise
the aspect ratio by 2 per level. Spacing ratios above 2.5 switch stencil
ranking to the metric. Prisms of aspect ratio 2 next to walls needed the 3D
Lebesgue bound of 4, and order 6 on jittered prisms is still open (#229,
#126).

**Decision 3: isotropic only** until stage 4. The one benign anisotropic
case is a later candidate: in-plane-only refinement of thin boundary-layer
prisms and hexes, which lowers their aspect ratio.

## 2. Refinement templates by element type

Every face subdivides in the same way, whatever cell it belongs to:

- a triangle into 4 by its edge midpoints;
- a quadrilateral into 4 by its edge midpoints and its vertex average.

So any two refined cells match on the shared face, across types.

| Type | Children | Template | Notes |
|---|---|---|---|
| Triangle | 4 | red (midpoints) | all children similar to the parent |
| Quadrilateral | 4 | midpoints + vertex average | bilinear parent: children lie on the parent's map |
| Tetrahedron | 8 | 4 corner tets + octahedron split along one diagonal | diagonal chosen by Bey's rule ([Bey 1995](#ref-bey-1995)), from the local vertex order with ties broken by node key: at most 3 similarity classes under repeated refinement. with a consistent vertex order, Kuhn tetrahedra refine into Kuhn tetrahedra, the tiling already validated in teno_e.md |
| Prism | 8 | triangle 1:4 x 2 layers | anisotropic later: 4 (in-plane) or 2 (axial) |
| Pyramid | 10 | 6 pyramids + 4 tetrahedra | quad base 1:4, triangle faces 1:4. Pyramid lattices are already TENO-fragile (#236), so prefer orders 3 or 5 |
| Hexahedron | 8 | trilinear midpoints | anisotropic later: 4 or 2 |

New node coordinates:

- An edge midpoint is `0.5 (x_a + x_b)` with the operands ordered by node
  key.
- A face or cell center is the vertex average in the order of its key.

So every rank computes the same bits. Node keys (section 7) are functions of
the parents' keys. Midpoints on periodic zones therefore get matching
periodic keys without a new matching pass.

**Warped faces.** Bilinear quadrilateral faces are not planar. Mallard
fans them around their vertex average. After refinement, the four subfaces
fan around their own averages, so the neighbor's volume changes slightly
once its face is split. **Decision:** a cell whose geometry changes keeps
its integral `V U`: `U_new = (V_old / V_new) U_old`. This is exactly
conservative. The change is O(warp), and it is zero for triangles, planar
quads and straight-sided tetrahedra.

## 3. Refinement criteria

Each criterion is evaluated per leaf on the device from the state and the
cell's stencil. Their outputs combine into a flag: refine, keep or coarsen.

| Criterion | Detects | Cost | Notes |
|---|---|---|---|
| TENO troubled-cell indicator `sigma` (density variance over the large stencil) | shocks, contacts | free: computed every stage | already the scheme's own discontinuity sensor (`TENO_SIGMA`) |
| Löhner's normalized second difference ([Löhner 1987](#ref-lohner-1987)) of a chosen variable (`rho`, `p`, `T`, `Y_k`) | shocks, flames, shear layers | one pass over vertex neighbors | scale-free, threshold O(0.1-0.3) |
| Heat release `HRR / max HRR`, or a progress-variable gradient | flames, reaction zones of detonations | `HRR` exists | a global max is exact and rank-independent |
| Top-degree coefficients of the TENO polynomial, scaled by `h^r` | smooth under-resolution (truncation error estimate) | one adaptation-time reconstruction pass | an analogue of the modal-decay sensor of [Persson & Peraire 2006](#ref-persson-2006) |
| Multiresolution details: children's averages against their prediction from the parent ([Harten 1995](#ref-harten-1995); [Cohen et al. 2003](#ref-cohen-2003)) | coarsening (where the fine data adds nothing) | uses the prolongation operator (section 5) | the natural coarsening test, with a threshold tied to the scheme's error |
| Regions: boxes, spheres or expressions in `x`, `t`, with min/max level | static refinement, wakes, inflow | none | also caps levels near NSCBC zones and inlets (section 8) |

**Decision:** stage 1 ships regions and `sigma`. Stage 2 adds Löhner, `HRR`
and multiresolution coarsening. Every flag is a function of the state and
of rank-independent global reductions, never of the partition. A target
cell count, if wanted, is reached by bisecting a threshold on exact
integer counts.

Each criterion has a refine threshold above its coarsen threshold
(hysteresis). Flags are then dilated by `n_buffer` cells. The buffer must
cover the distance a front travels between adaptations, `b >= v n_a`, with
`v` the front's speed in fine cells per step and `n_a` the steps between
adaptations. Section 9 picks `n_a`.

## 4. TENO-E stencils and caches on a changing mesh

### 4.1 What a stencil depends on

A cell's TENO-E tables are its central and sector stencils, pseudo-inverses
and smoothness-indicator matrices. They depend on:

- the geometry of every candidate cell gathered by the vertex-neighbor
  layer search, not only the selected ones;
- the metric from the vertex neighbors' offsets;
- the mirror images;
- the Lebesgue constant at the cell's face quadrature points.

Any change inside a cell's search reach invalidates its tables.

### 4.2 Level-view stencils (decision 4)

Built naively, a coarse cell next to a refined region would see eight
small cells per neighbor on that side. Ranked by distance, they make up
most of its 2 x DOFs entries. The resulting stencil is one-sided, with large
Lebesgue constants, and every coarse cell within reach of a change must be
rebuilt.

Instead, each cell of level `L` gathers its stencil in the **level-`L`
view** of the forest:

- a region refined beyond `L` appears as its level-`L` ancestors;
- coarser cells appear as they are.

A level-`L` ancestor is a virtual cell: its geometry is the tree node's
own, and its average is the volume-weighted restriction of its leaves.

- A coarse cell's tables are then **identical** before and after a neighbor
  refines, since its level view did not change. They are kept bitwise and
  need no recomputation.
- A fine cell next to a coarse region sees larger cells in that direction,
  as in any graded mesh. Section 10's validation decides whether the
  Lebesgue bound and the metric's 2.5 cut-off hold there, or whether
  sizes need normalizing in the ranking.
- Virtual cells are extra rows of the state array, filled by a restriction
  kernel after each halo exchange and never advanced in time. TENO kernels
  index them like leaves, so the kernels do not change. The halo must hold
  all leaves of every virtual cell it references.
- With warped faces, a virtual cell's own volume differs from the sum of
  its children by O(warp). That costs accuracy near such interfaces
  (reconstruction data only, not conservation).
- Fluxes are unaffected: a coarse cell evaluates its polynomial at the
  quadrature points of its subfaces. The face-point basis values are
  recomputed for the split face, which is cheap and involves no least
  squares. The Lebesgue acceptance test uses the cell's unsplit faces, so
  the stencil does not depend on its neighbors' levels.
- Sector stencils stay one per logical face of the cell type, not one per
  subface.

### 4.3 Invalidation, reuse and cost

Each cell stores the radius, in level-view layers, that its search
reached. After an adaptation, a cell is rebuilt if:

- it is new; or
- a cell in its level view within that radius changed.

That set is computed by a breadth-first search from the changed families.
Reused tables equal recomputed ones bitwise, because they are a function
of the level view alone. So adaptation history cannot leak into the
results, and a restart that recomputes everything reproduces a run that
reused tables.

Cost, per rebuilt cell (`c_r`), against a step (`c_s` = 0.17 us per cell for
3D TENO5 on an A100, 0.019 us for 2D TENO5):

| Path | `c_r` | Source |
|---|---|---|
| Host setup today, 2D TENO5 | 330 us per core | mpi.md §10.1 |
| Host setup today, 3D TENO5 hexes | about 3-5 ms per core, about 200 us per GPU with 16 cores | #140 (over 10 min for 1M cells on 8 threads) |
| Moving existing tables between ranks | about 1.5 us (36 KB per cell over 25 GB/s) | mpi.md §10 |
| GPU batched setup (stage 0 target) | 1-5 us | estimate: about 1 MFLOP per 3D TENO5 cell, at 3-30% of A100 FP64 peak |

The host setup is not flop-bound: 1 MFLOP in 3 ms is 0.3 GFLOP/s. The time
goes into neighbor search and host data structures. **Stage 0** therefore
moves the per-cell setup to the device:

- layer search over the level-view CSR graph;
- batched QR least squares;
- the Lebesgue loop with a per-cell fixed code path, so results do not
  depend on batch composition.

Its other parts:

- an incremental API: rebuild a list of cells, keep the rest;
- a device-side repack of the slice-interleaved `PackedStencils` by
  permutation, chunked to bound peak memory;
- one rank-count-independent cache keyed by leaf id, written with restarts
  (HDF5, collective), replacing the per-rank `cache_file`s.

The viscous quadratic least-squares fits over vertex neighbors are
invalidated on the same rule. They are much cheaper.

## 5. Prolongation and restriction

**Restriction** (coarsening a family): `U_p = sum_c V_c U_c / V_p`, summed
in child order. It is exact and conservative. With warped faces, `V_p`
is the sum of the children's volumes, and section 2's rescaling follows.

**Prolongation** (refining a parent):

1. **Smooth parents** (not troubled): child averages come from the
   parent's large-stencil polynomial,
   `U_c = U_p + sum_l a_l mean_c(psi_l)`.
   - `mean_c(psi_l)` are child moments in the parent's reference frame. They
     come from the same tetrahedral-decomposition moment integration as the
     cell geometry.
   - The basis has zero mean on the parent, so `sum_c V_c U_c = V_p U_p` up
     to round-off.
   - The order is `r + 1`, the scheme's own. This is the high-order FV
     prolongation of [Ivan & Groth 2014](#ref-ivan-groth-2014) (k-exact
     CENO on AMR) and [Dumbser et al. 2013](#ref-dumbser-2013).
2. **Troubled parents**: limited linear prolongation (the MUSCL gradient
   with Barth-Jespersen limiting against the parent's neighbors).
3. **Positivity**: if any child has `rho`, `p` or `rho Y_k` out of bounds,
   the deviations of the whole family are scaled by `theta` in [0, 1]
   toward `U_p`, as in `bound_preserving`
   ([Zhang & Shu 2010](../references.md#zhang-shu-2010)). Scaling a
   mean-zero deviation keeps conservation. `theta = 0` is injection.
4. **Species**: `rho Y_k` prolong linearly with limiting, matching the
   linear species reconstruction of chemistry.md decision 4. Then
   `rho_child = sum_k rho Y_k` keeps `sum Y = 1` to round-off.
   - **Open question:** this replaces the high-order `rho` of step 1 in
     mixtures. The alternative is high-order `rho` with `Y` prolonged
     separately, which conserves `rho` but `rho Y_k` only to the limiter's
     accuracy.

The prolongation runs on the device before the parent's tables are
discarded. It needs the parent's stencil, which exists because the parent
was a leaf.

**Auxiliary per-cell fields:**

| Field | Prolongation (copied to children) | Restriction |
|---|---|---|
| `T_SEED`, `CHEM_H` | parent's value | child 0's value (deterministic) |
| `P_MAX` | parent's value | max |
| `[statistics]` means, covariances and weights | parent's value | volume-weighted average |

Synthetic-inflow and NSCBC face states are split to child faces and
averaged back.

**Axisymmetric runs** use `cell_measure` (the r-weighted volume) in place
of `V` everywhere above, and r-weighted child moments.

## 6. Time stepping

**Decision 6: one global time step.** The CFL reduction already takes the
minimum over all cells, and the fastest waves usually sit in the finest
cells at the front. Adaptation happens only between full steps:

- after any closing Strang chemistry half step;
- not at fused half steps (`fuse_half_steps`), the rule output already
  follows;
- RK stages never see a changing mesh.

The dt reduction after an adaptation uses the new mesh.

Local time stepping is out of scope. For the record, what AMR would gain
from subcycling ([Berger & Oliger 1984](#ref-berger-oliger-1984);
[Berger & Colella 1989](#ref-berger-colella-1989)):

- In section 9's example, levels coarser than the finest hold about 40% of
  the cells. Subcycling would step them a quarter and half as often,
  saving at most about 27% of the step work.
- Mallard's stencils span several coarse cells across a level jump. Their
  time interpolation of ghost data would be far wider than the 2-4 ghost
  cells of structured AMR.
- Subcycling needs a refluxing correction for conservation, and breaks the
  one-exchange-per-stage overlap of mpi.md §7.

None of this is needed for AMR to pay off.

## 7. Determinism, ids and rank-count independence

The invariant of mpi.md §10 holds unchanged: results are bitwise
identical:

- on any rank count;
- with or without rebalancing;
- with tables reused or recomputed;
- across restarts.

The ingredients:

- **Ids.**
  - A cell key is `(root gid, path)`, with 4 bits per level (pyramids
    have 10 children).
  - The key fits 64 bits for 2^40 roots and 6 levels. Deeper forests use
    128-bit keys.
  - Node keys are `(min, max)` parent node keys plus a tag (edge midpoint),
    or the owning cell or face key (centers). They are hashed to 64 bits
    for the directory, and a collision is checked and fatal.
  - Global order: roots in Hilbert order of their vertex averages, then
    the depth-first child order (a Morton-like order within each root).
    For tetrahedra a TM-type order ([Burstedde & Holke 2016](#ref-burstedde-holke-2016))
    keeps children local.
  - Every tie-break that uses global ids today (stencil candidates,
    face order, ordered reductions) uses this key.
- **Flags.** Per leaf, from the state and the stencil, which are identical
  on all ranks. Global thresholds come from exact reductions (min, max,
  integer counts) or the existing key-ordered sums.
- **Balance closure.** It only refines, so its least fixed point is unique
  and independent of processing order. It is computed in rounds of local
  passes plus halo exchanges of levels, until a global "changed" flag is
  false.
- **Coarsening.** A family coarsens (at most one level per adaptation) iff:
  - all its children are leaves flagged for coarsening;
  - every face and edge neighbor's level *after refinement* is at most the
    children's level.

  The decision depends only on post-refinement levels, so it does not
  depend on order.
- **Numerics.** Prolongation and restriction sum in child order. New
  geometry follows section 2's rules. The tables follow section 4.3.
- **Schedule.** Adaptation is triggered by step count or simulated time,
  never wall time. Rebalancing may use measured times (as in mpi.md
  §10.1), since the partition does not affect results.

Tests (section 10) compare adapted runs on 1-4 ranks, with forced
rebalancing, against serial runs bitwise, as the DLB prototype did.

## 8. Physics and boundary interactions

- **Chemistry.**
  - Pointwise, so it only needs section 5's auxiliary fields.
  - Chemistry load balancing (#141) moves states and is unaffected.
  - Refinement concentrates chemistry cost, which the partition weights
    must include (mpi.md §10.1's cost model plus `CHEM_COST`).
- **LES.**
  - The filter width `V^(1/d)` changes across level jumps. This causes
    commutation errors and spurious energy transfer at grid discontinuities
    ([Ghosal & Moin 1995](#ref-ghosal-moin-1995);
    [Vanella, Piomelli & Balaras 2008](#ref-vanella-2008)).
  - Recommendation: in LES, keep the turbulent region at a fixed level
    (regions), and adapt only around flames and shocks.
  - TFLES's dynamic thickening already follows `Delta`, so refined flames
    thicken less.
  - The dissipation budget works per cell unchanged.
- **NSCBC, inlets, sponges.**
  - Characteristic zones and synthetic inlets carry per-face state and
    transverse terms. Stage 1 caps levels within `n` cells of them, which
    keeps their faces unsplit at a fixed level.
  - Sponge strengths and reference states are expressions, re-evaluated
    at new cells.
- **Mirror images and periodic seams.**
  - Mirrors are recomputed for rebuilt cells, as at setup.
  - Periodic keys propagate to new nodes (section 2).
  - 2:1 balance crosses seams through the forest's face connectivity,
    which includes periodic faces.
- **Curved boundaries.** New boundary nodes stay on the input's straight
  faces. With #162 they would be projected onto the surface, a geometry
  change handled by section 2's rescaling.
- **Viscous fluxes.**
  - The vertex-neighbor quadratic fits are rebuilt like TENO tables.
  - A hanging node joins the node-to-cell incidence of the coarse cell
    whose face or edge it lies on. Vertex adjacency, halos and gradients
    therefore see the coarse cell.
- **Probes and integrals.** Probes are relocated after adaptation. Surface
  integrals iterate over subfaces.

## 9. Load balancing and the cost of adaptation

### 9.1 Partitioning

**Decision 7.** The leaves in global key order form a space-filling
curve:

- A partition is a split of that order into contiguous chunks of equal
  weight.
- Cut points are snapped to family boundaries, so coarsening is local
  ("partition for coarsening", as in p4est).
- Weights follow mpi.md §10.1's model (base, troubled fraction, stencil
  size), plus chemistry cost.
- Repartitioning moves only the cells near chunk ends, the incremental
  scheme mpi.md §10.3 named as what would make rebalancing pay.
- The Hilbert backend generalizes directly. dKaMinPar stays available
  for unadapted runs.

Refinement changes a rank's cell count by factors of 8 per level, not the
15-30% imbalance §10 measured. So AMR runs rebalance at every adaptation
whose imbalance exceeds a threshold. Moved cells carry their tables
(about 1.5 us per cell) instead of rebuilding them.

### 9.2 When does adaptation pay?

Take a front of area `A` (in fine cells) and thickness `w` fine cells,
moving `v` fine cells per step, with stencil reach `R`, adapting every
`n_a` steps with a buffer `v n_a`. Per step:

- stepping the fine band costs `c_s A (w + 2 v n_a)`;
- rebuilding costs `c_r A (2 v + 2 R / n_a)`. New cells and fine cells
  within `R` of changes are rebuilt; level-view stencils spare the coarse
  side.

The optimum is `n_a = sqrt(R c_r / (v c_s))`.

With 3D TENO5, `c_s` = 0.17 us, `w` = 20 and `R` = 3, the cost of the fine
band relative to an ideal static band of width `w`:

| `c_r` per cell | shock / detonation (`v` = 0.3): `n_a`, cost | flame (`v` = 0.005): `n_a`, cost |
|---|---|---|
| 190 us (host today) | 106 steps, **41x** | 820 steps, 2.4x |
| 20 us | 34, 6.6x | 270, 1.3x |
| 5 us | 17, 2.9x | 130, 1.15x |
| 1 us | 8, 1.6x | 60, 1.06x |

Whole problem: a 3D detonation in a cube resolved at 1000^3
fine-equivalent cells (planar-ish front, area 10^6 fine faces, base level 4x
coarser, two levels):

| `c_r` | cell-steps per step | gain over uniform 10^9 |
|---|---|---|
| 190 us | 8.5e8 | 1.2x |
| 5 us | 8.1e7 | 12x |
| 1 us | 5.5e7 | 18x |
| flame (`v` = 0.005) at 190 us | 7.0e7 | 14x |

Memory follows the cell count: TENO5 needs about 80 KB per 3D cell on the
device (#58). The 10^9-cell mesh needs about 1,400 A100-80GB. The adapted
one fits on about 70-100.

The model leaves out the mesh rebuild (`c_m`, per cell of the rank, per
adaptation). The DLB prototype's rebuild is about 20 us per cell on the
host without TENO data (local mesh, halo independence, exchange,
partition). At `n_a` = 17 that alone adds 7 times the step cost. Fast-front AMR
therefore needs **both** an O(changed) host side and an O(N) side made of
device passes (sort, scan, gather), not host loops. Slow fronts tolerate
today's full rebuild (about 15% at `n_a` = 800).

**Conclusion:**

- Slow fronts (flames, quasi-steady features) are viable with today's
  setup and a full rebuild per adaptation.
- Shocks and detonations need `c_r` under about 5 us per cell and a
  device-side incremental rebuild.

## 10. Data structures, GPU and I/O

- **Host.**
  - The forest holds per leaf its key, level, type and flags, and per
    internal node its geometry moments and children range. It is O(cells)
    with small constants, and updated only for changed families.
  - Per root it holds the type and template.
  - The `DistributedMesh` directory (nodes to cells) is kept for the run,
    as in the DLB prototype.
- **Device.**
  - The leaf `Mesh` (CSR arrays as today), plus virtual-cell rows and their
    children CSR.
  - An explicit cell type, since node count no longer identifies the type
    once a cell has subfaces.
  - A logical-face-to-subface map for sector stencils and the Lebesgue
    test.
  - Kernels work on leaves as today.
- **Adaptation pipeline.**
  1. Flags kernel, then copy 1 byte per cell to the host.
  2. Host: closure, coarsening rule, new families, new keys. Each rank
     handles its families, with halo-level exchanges.
  3. Device: prolongation and restriction into new state arrays.
  4. Local mesh patch: the full rebuild in stage 2, device and
     incremental in stage 3.
  5. Halo patch, exchange plan, `cells_independent_of_halo`.
  6. Repack kept tables by permutation, and set up the invalidated
     cells (stage 0 code).
  7. Repartition if imbalanced, moving states and tables by key (the
     DLB prototype's #104 and #108 mechanisms).
- **Restart (format 5).**
  - Contents:
    - a reference to the root mesh (path and content hash);
    - the leaf keys in global order;
    - the per-leaf state and auxiliary fields of format 4;
    - NSCBC face states keyed by (leaf key, face, subface);
    - the adaptation scheduler (step of the last adaptation).
  - A reader on any rank count takes a contiguous range of leaves, which is
    already a valid SFC partition, and rebuilds the forest from the root
    mesh.
  - The optional global TENO cache (section 4.3) makes it start in
    seconds.
- **Output.**
  - Leaves are ordinary cells with their own nodes, and hanging nodes are
    ordinary nodes, so VTU and HDF5/XDMF need no new cell types. On planar
    faces there are no visible cracks.
  - HDF5/XDMF writes the mesh only when it changed since the last
    snapshot, and later snapshots reference it.

## 11. Staged plan

Each stage is a PR series with its own validation, and stage 0 is the
gate. Effort is a rough estimate in developer-weeks.

| Stage | Content | Validation (each must fail under a plausible bug) | Effort | Runtime cost |
|---|---|---|---|---|
| 0 | Device TENO setup, incremental API, device repack, global leaf-keyed cache (#140) | Stencils and tables bitwise equal between full and incremental setup, and across rank counts and batch sizes. Measured `c_r` (3D TENO5 hex/tet, 2D tri/quad). **Gate:** `c_r <= 5 us` on an A100 | 4-6 | setup: minutes to seconds; DLB (mpi.md §10.3) becomes viable |
| 1 | Forest, keys, templates for 6 types, hanging subfaces in `Mesh`, 2:1 closure, level-view stencils with virtual cells, prolongation and restriction, SFC partition of leaves, restart 5, output. Adaptation at startup (regions, or criteria on the initial condition) and at restart | Free-stream preservation on refined meshes of each type and order. Conservation to round-off over 100 random refine/coarsen cycles. Isentropic vortex through a static 2:1 patch, convergence order of TENO3-6 within 0.2 of uniform. **Linear stability** (acoustics at rest, max Re lambda as in teno_e.md) on 2:1 patches, per type and order. Sod and shock-vortex through level jumps (spurious reflection < 1%). 1-4 ranks bitwise; restart on 3 ranks read on 2 bitwise | 10-14 | adaptation = a setup (seconds to minutes) |
| 2 | In-run adaptation at low frequency with a full rebuild: criteria (sigma, Löhner, HRR, multiresolution), buffers, hysteresis, rebalancing at every adaptation | 2D H2/air premixed flame: speed within 1% of uniform-fine. DMR and Riemann problem 3: error against uniform fine at equal finest h, and cost. Adapted runs bitwise on 1-4 ranks with forced rebalancing. Restart mid-run bitwise | 6-8 | per adaptation: full rebuild (5-20 s per 125-500k cells per GPU) + `c_r` x changed. Pays for `n_a` of hundreds or more |
| 3 | Fast fronts: device-side incremental mesh, halo and table patching, O(changed) host work | Cellular H2-O2-Ar detonation, cell size against [Deiterding 2011](#ref-deiterding-2011). 3D DMR-type shock reflection. Cost against uniform fine at equal accuracy, against section 9's model | 8-12 | per adaptation: target ≲ 10 steps |
| 4 | Optional: anisotropic quads, hexes and prisms (in-plane first); projection onto curved boundaries with #162; output-targeted criteria | eigenvalue stability on anisotropic children; BL profiles | 4-8 | — |

Risks:

- **TENO-E at level jumps.**
  - Fine cells next to coarse ones may break the Lebesgue bound or the
    metric cut-off.
  - Stage 1's eigenvalue tests find this before anything dynamic is built.
  - Mitigations: size-normalized candidate ranking; order reduction in the
    one fine layer at a jump.
- **Pyramids** (#236) and **jittered prisms** (#229) are already fragile.
  Refinement adds more of both.
- **Stage 0 may not reach 5 us per cell.** Then stage 3 is not built, and
  stages 1-2 plus the moving frame cover the use cases.

## 12. Alternatives considered and recommendation

| Alternative | For | Against |
|---|---|---|
| **A. No AMR: static refinement (Gmsh size fields) + scaling** | No solver complexity. Mallard weak-scales at 93% (16 GPUs). Covers stationary flames, jets, wakes and boundary layers | Moving fronts in large domains need uniform fine spacing: 10-20x more cells (section 9), so about 1,400 GPUs instead of 100 for the 1000^3 example |
| **B. Frame moving with the front + static refinement** | Steady propagating detonations and flames (the cellular-detonation demos, #147) stay in a short refined domain with existing BCs | Only steady propagation: not DDT, reflections, blasts or ignition |
| **C. Restart-time adaptation only** (stage 1) | Most of the machinery at low risk; solution-adaptive meshes for quasi-steady problems | Fronts must be slow compared with a restart cycle |
| **D. Remeshing (MMG/ParMmg) + conservative remap** | Anisotropic metric meshes ([Alauzet & Loseille 2016](#ref-alauzet-2016)) | Tetrahedra only; a full setup per remesh; supermesh remap of high-order data; no coarsening by restriction |
| **E. Conforming red-green or bisection** | No hanging faces | See section 1.1 |
| **F. p4est / t8code** | Mature balance, ghost and SFC algorithms; t8code has hybrid elements | GPL-2.0 against Apache-2.0; host only; the stencil, determinism and GPU work stays ours. Their algorithms are reused, not their code |
| **G. Block-structured AMR (AMReX/PeleC)** | Mature and GPU-ready, with subcycling | Cartesian with embedded boundaries: a different code |
| **H. Subcycling** | Saves coarse-level steps | Rejected (section 6) |

**Recommendation.**

1. **Do stage 0 now.** It is worth doing without AMR: 3D TENO setup drops
   from minutes to seconds, and fast incremental tables are exactly what
   mpi.md §10.3 said dynamic load balancing lacks.
2. **Then stage 1.** It delivers static and restart-time adaptation on all
   cell types, and settles the one real numerical risk: TENO-E stability
   at 2:1 interfaces.
3. **Stage 2** when a flame or LES case needs a mesh that follows its
   flame.
4. **Stage 3 only if** stage 0 meets the 5 us per cell gate and a
   moving-front science case (DDT, detonation reflection, blasts) needs
   it. Until then, use alternatives A and B for moving fronts.

## Open questions

1. Is the gate right? 5 us per cell for fast fronts, judged on section 9's
   model, or require a measured stage-2 prototype first.
2. Species prolongation (section 5): `rho = sum rho Y_k` with linear
   species, or high-order `rho` with separately limited `Y`.
3. Key width: 64-bit (6 levels over 2^40 roots) or 128-bit from the start.
4. Whether stage 1 should allow different levels in LES regions at all, or
   only static regions (section 8).
5. Edge balance in 3D. It is required here for stencil locality. Is
   face-only balance worth measuring?

## References

DOIs checked against Crossref. Entries move to [references.md](../references.md)
with the stage that implements them.

- <a id="ref-alauzet-2016"></a>Alauzet & Loseille 2016, *Comput.-Aided Des.* 72, 13. [doi:10.1016/j.cad.2015.09.005](https://doi.org/10.1016/j.cad.2015.09.005)
- <a id="ref-berger-colella-1989"></a>Berger & Colella 1989, *J. Comput. Phys.* 82, 64. [doi:10.1016/0021-9991(89)90035-1](https://doi.org/10.1016/0021-9991%2889%2990035-1)
- <a id="ref-berger-oliger-1984"></a>Berger & Oliger 1984, *J. Comput. Phys.* 53, 484. [doi:10.1016/0021-9991(84)90073-1](https://doi.org/10.1016/0021-9991%2884%2990073-1)
- <a id="ref-bey-1995"></a>Bey 1995, *Computing* 55, 355. [doi:10.1007/BF02238487](https://doi.org/10.1007/BF02238487)
- <a id="ref-burstedde-2011"></a>Burstedde, Wilcox & Ghattas 2011, *SIAM J. Sci. Comput.* 33, 1103 (p4est). [doi:10.1137/100791634](https://doi.org/10.1137/100791634)
- <a id="ref-burstedde-holke-2016"></a>Burstedde & Holke 2016, *SIAM J. Sci. Comput.* 38, C471. [doi:10.1137/15M1040049](https://doi.org/10.1137/15M1040049)
- <a id="ref-cohen-2003"></a>Cohen, Kaber, Müller & Postel 2003, *Math. Comp.* 72, 183. [doi:10.1090/S0025-5718-01-01391-6](https://doi.org/10.1090/S0025-5718-01-01391-6)
- <a id="ref-dapogny-2014"></a>Dapogny, Dobrzynski & Frey 2014, *J. Comput. Phys.* 262, 358. [doi:10.1016/j.jcp.2014.01.005](https://doi.org/10.1016/j.jcp.2014.01.005)
- <a id="ref-deiterding-2011"></a>Deiterding 2011, *J. Combust.* 2011, 738969. [doi:10.1155/2011/738969](https://doi.org/10.1155/2011/738969)
- <a id="ref-dumbser-2013"></a>Dumbser, Zanotti, Hidalgo & Balsara 2013, *J. Comput. Phys.* 248, 257. [doi:10.1016/j.jcp.2013.04.017](https://doi.org/10.1016/j.jcp.2013.04.017)
- <a id="ref-gamezo-2008"></a>Gamezo, Ogawa & Oran 2008, *Combust. Flame* 155, 302 (DDT with cell-based AMR). [doi:10.1016/j.combustflame.2008.06.004](https://doi.org/10.1016/j.combustflame.2008.06.004)
- <a id="ref-ghosal-moin-1995"></a>Ghosal & Moin 1995, *J. Comput. Phys.* 118, 24. [doi:10.1006/jcph.1995.1077](https://doi.org/10.1006/jcph.1995.1077)
- <a id="ref-harten-1995"></a>Harten 1995, *Comm. Pure Appl. Math.* 48, 1305. [doi:10.1002/cpa.3160481201](https://doi.org/10.1002/cpa.3160481201)
- <a id="ref-henry-de-frahan-2023"></a>Henry de Frahan et al. 2023, *Int. J. High Perform. Comput. Appl.* 37, 115 (PeleC). [doi:10.1177/10943420221121151](https://doi.org/10.1177/10943420221121151)
- <a id="ref-holke-2021"></a>Holke, Knapp & Burstedde 2021, *SIAM J. Sci. Comput.* 43, C359 (t8code hybrid forests). [doi:10.1137/20M1383033](https://doi.org/10.1137/20M1383033)
- <a id="ref-ivan-groth-2014"></a>Ivan & Groth 2014, *J. Comput. Phys.* 257, 830. [doi:10.1016/j.jcp.2013.09.045](https://doi.org/10.1016/j.jcp.2013.09.045)
- <a id="ref-khokhlov-1998"></a>Khokhlov 1998, *J. Comput. Phys.* 143, 519. [doi:10.1006/jcph.1998.9998](https://doi.org/10.1006/jcph.1998.9998)
- <a id="ref-lohner-1987"></a>Löhner 1987, *Comput. Methods Appl. Mech. Eng.* 61, 323. [doi:10.1016/0045-7825(87)90098-3](https://doi.org/10.1016/0045-7825%2887%2990098-3)
- <a id="ref-persson-2006"></a>Persson & Peraire 2006, AIAA Paper 2006-112. [doi:10.2514/6.2006-112](https://doi.org/10.2514/6.2006-112)
- <a id="ref-rivara-1984"></a>Rivara 1984, *Int. J. Numer. Methods Eng.* 20, 745. [doi:10.1002/nme.1620200412](https://doi.org/10.1002/nme.1620200412)
- <a id="ref-shen-2011"></a>Shen, Qiu & Christlieb 2011, *J. Comput. Phys.* 230, 3780 (WENO with AMR). [doi:10.1016/j.jcp.2011.02.008](https://doi.org/10.1016/j.jcp.2011.02.008)
- <a id="ref-stevenson-2008"></a>Stevenson 2008, *Math. Comp.* 77, 227. [doi:10.1090/S0025-5718-07-01959-X](https://doi.org/10.1090/S0025-5718-07-01959-X)
- <a id="ref-vanella-2008"></a>Vanella, Piomelli & Balaras 2008, *J. Turbul.* 9, N32. [doi:10.1080/14685240802446737](https://doi.org/10.1080/14685240802446737)
- <a id="ref-venditti-2002"></a>Venditti & Darmofal 2002, *J. Comput. Phys.* 176, 40 (adjoint adaptation, not adopted). [doi:10.1006/jcph.2001.6967](https://doi.org/10.1006/jcph.2001.6967)
- <a id="ref-zhang-2019"></a>Zhang et al. 2019, *J. Open Source Softw.* 4, 1370 (AMReX). [doi:10.21105/joss.01370](https://doi.org/10.21105/joss.01370)
