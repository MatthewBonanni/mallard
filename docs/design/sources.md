# Source terms on the device, and regions

Design notes for `[[sources]]` and regions (#317): why expressions are
compiled by Mallard instead of evaluated by exprtk, how stage times reach
captured steps, and how regions give exact totals on any mesh.

## Compiled expressions

exprtk evaluates a tree of virtual nodes in host memory, so device kernels
cannot call it. Until 0.8 a time-dependent `[source]` was evaluated on the
host at every Runge–Kutta stage, cell by cell, and copied to the device: the
step left the CUDA graph and spent most of its time on the host.

`src/common/kexpr.{h,cpp}` parses the same syntax once on the
host into a postfix program of 16-byte instructions (`kexpr::Program`), which
`kexpr::evaluate` runs with a small stack. It is one `KOKKOS_INLINE_FUNCTION`,
so the host and the kernels execute the same code; variables come from a
callable index → value, so a cell, a face or a point binds its own values
without copying them into an array. `kexpr::ProgramTable` packs many programs
into one device array.

The values are exprtk's to the last bit on the host (tested against exprtk on
45 expressions × 203 points until exprtk was removed), which needs exprtk's own
semantics, not just its grammar:

- precedence climbing with exprtk's levels: `^` right-associative, unary
  minus binding below `^` (`-x^2 = -(x^2)`), comparisons left-associative,
  implicit multiplication (`2x`, `2(x + 1)`);
- exprtk's conversion of literals (its own, not `strtod`), constants folded;
- integer powers as exprtk's `fast_exp` multiplications (hand-unrolled up to
  10), `x^-n = 1 / x^n`;
- its formulas for `round`, `roundn`, `hypot`, `log2`, `root`, `sinc`, ... with
  products in those helpers never contracted into FMAs (`__dmul_rn` on the
  device).

The reference is exprtk with strength reduction off, compiled without FMA
contraction: with reassociation (`(x + c0) + c1 → x + (c0 + c1)`) or
contracted fused nodes, exprtk itself changes the last bits. On the device,
library functions (`sin`, `exp`, ...) come from CUDA's libm and agree with the
host to a few ulp; arithmetic is bitwise the same.

Unlike exprtk, variables are case sensitive: sources need both `T` and `t`.

### Statements and control flow (#450)

The engine also compiles exprtk's statements: `;` sequences, `var` locals and
assignments (`:=`, `+=`, ...), `if`/`else` and `switch`, `while`, `for`,
`repeat ... until`, `break[v]` and `continue`. The parser builds a tree and
emits it with jumps (`JUMP`, `JUMP_IF_NOT`, and `ITERATE`/`UNTIL` for loop
conditions), reproducing exprtk's semantics node by node: a loop's value is
its body's last complete value (0 if it never ran; `continue` keeps the
previous one), `break` without a value gives NaN, an `if` without `else` gives
NaN, a sequence drops statements without effects and everything after a
`break` or `continue`. Conditionals whose branches have no effects stay a
`SELECT` of both, as before, so expressions without statements compile to the
same instructions as in 0.8 and run the same code.

Locals live in the bottom slots of the evaluation stack (`LOCALS n` moves the
stack above them; block scopes reuse slots), so a thread's storage is still the
one `double[32]` array, and a loop's value sits in a stack slot that `break`
and `continue` cut the stack back to (`KEEP`, `DROP`). The only new
per-thread state is an iteration counter: after `kexpr::MAX_ITERATIONS`
(100000) loop iterations in all, an evaluation returns NaN and records which
program stopped in a device flag of its `ProgramTable`. The flag is read only
where the host already waits (setup, output, a diverged step), and only for
tables that have loops, so a runaway loop becomes an input error instead of a
hung kernel without a synchronization per stage.

### One evaluation path, without exprtk (#449)

Since 0.10 exprtk is no longer part of Mallard. The remaining host sites
(`Expression`: sponges, boundary `where`, mass-flow profiles, compositions,
synthetic-inflow profiles, parametric probes) evaluate the same `kexpr::Program`;
those that evaluate at many mesh points (sponges and their compositions,
`where`, mass-flow profiles, boundary compositions) do so in one kernel
(`evaluate_at`), and parametric probes and synthetic-inflow profiles, which
evaluate at a few points on the host, call the same `kexpr::evaluate`. The
literal conversion is a port of exprtk's `string_to_real` and `derivative`
(probe curves) of exprtk's fourth-order difference, so no value changed
where exprtk ran with strength reduction off. exprtk ran these sites with it
on, which rewrites a few patterns of division (`(a / b) / c → a / (b * c)`,
`a / (b / c) → (a * c) / b`, ...); expressions of those forms now round as
written. Of the 77 expressions in the examples and tutorials, only one would
round differently (by up to 34 ulp, through `cos` of a rewritten argument),
and it is a boundary value, compiled since 0.9. Where the host compiler
contracts multiply-adds (Apple silicon by default), exprtk's fused nodes such
as `a * b + c` also rounded once; the engine never contracts. In a 2D test
case with a sponge, `where` filters and a mass-flow profile free of such
divisions, 20 steps of the
solution are bitwise the same as before without contraction, and differ in
the last bits (relative 1e-15) with it. The tests check the engine
against hand-written references of each expression and the device against
the host.

Measured on the host (Apple M4 Pro, 2 threads, 2M points, best of 7), the
expressions of the tests and the source below compile to the same
instructions as before and evaluate 10–25% faster (the source: 40.5 → 36.4 ns
per evaluation), from the interpreter's reordered dispatch.

### No expression on the host during a step (#449)

Every expression read during a step runs in a kernel at the stage time held
in a device scalar: sources, boundary values of `t` (`TimedBoundaryFields`,
`DirichletStates`, `TimedInletTargets`) and the target of an average-pressure
outlet. The outlet's target was the last one evaluated on the host: each stage
copied its condition to the host, evaluated `p(t)` there and copied it back.
It is now one entry of `TimedBoundaryFields` without a face, and the outlet's
offset is written by a one-thread kernel; the outlet still copies its faces'
`p A` and `A` to the host once per stage, to sum them in an order every rank
count agrees on (a 3D outlet case with `p(t)`, 40 steps, is bitwise the same as
before). The setup sites evaluate once: on the device at mesh points, and on
the host (with the same `kexpr::evaluate`) only for parametric probe curves
and synthetic-inflow profiles. `ForcedInletTest.ConditionsOfTKeepTheStepGraph`
shows the step path needs the host for none of them: on CUDA it replays the
whole step, conditions of `t` included, as one captured graph, with the same
solution as stepping kernel by kernel.

### Geometry, expressions and inputs apart (#449)

Three layers, each unaware of the one above:

- `geometry.h`: primitives, transforms and booleans (`region::Node`,
  `distance`, `indicator`), for kernels and the host. An implicit shape is
  whatever callable the caller passes, `implicit(i, q)` at the local point
  `q`, so the geometry knows neither kexpr nor inputs.
- `kexpr.{h,cpp}`: the expression engine. Where an input-facing caller wants
  more in the error for an unknown symbol (the hint that expressions take no
  unit symbols, from `units.h`), it passes a `SymbolHint`.
- `region.{h,cpp}` joins them: `ImplicitPrograms` evaluates implicit shape
  `i` as program `i` of a `kexpr::ProgramTable`, and a `Region` is built
  from a `region::Definition`, plain data. `region_input.{h,cpp}` parses
  input tables into definitions (`region::parse`, `region::read`) and holds
  what refers to the input: named regions, regions of the mesh and
  `region::Regions`.

`src/CMakeLists.txt` checks that `geometry.h` and `kexpr.{h,cpp}` include no
other Mallard header. The code that evaluates is unchanged, and so is every
result: a 2D case (48² triangles, 4 steps) and a 3D one (20³, 3 steps) whose
initial regions, sources, sponges and region probes use every kind of node
(each primitive and boolean), smooth edges and boundaries of implicit shapes
write the same solution and probe files, byte for byte, before and after.

## Stage times in captured steps

A graph-captured step replays kernels whose time and time step live on the
device (`step_scalars`). The right-hand side therefore gets the stage's
fraction `c` of the step besides its host time (`RHSFunction(U, R, t, c)`):
inside a capture a one-thread kernel sets the device stage time to
`t_n + c dt` from the step's device scalars; otherwise it copies the host
stage time. For FE, SSPRK3 and RK4 `c dt` is exact, so both give the same
bits as the host's `t + c dt`. A step with time-dependent sources is a CUDA
graph like any other, also with domain averages (`mean_*`): their global sum
per stage is an `ExactSums` on the device, reduced over the ranks with NCCL on
the step's stream.

## Evaluation

Terms of position alone are summed once into a field; terms of `t`, the local
state or domain averages are evaluated per stage in one kernel over the owned
cells, which also adds the field: per cell it binds the variables lazily (`T`
from `p / (rho R)`, mixture `X_k` from the mass fractions) and skips cells of
zero weight. `W_cells` (and the mixture scalars) already hold the stage's
primitive state when sources are added.

## Regions

`src/common/region.{h,cpp}` builds constructive solid geometry from input
tables. Each primitive has a signed distance that is 1-Lipschitz (so
`|phi(x)|` never exceeds the distance to the surface): exact for spheres,
boxes, cylinders (annular included: the exact distance to a rectangle in the
`(rho, z)` half plane), capped cones, tori, half-spaces and polygons; for
ellipsoids and non-uniform scales, the distance in the scaled frame times the
smallest scale, a lower bound. Unions (min), intersections (max) and
differences (max(a, -b)) keep the bound. Transforms compose onto the
primitives when the table is parsed, so evaluation is a flat postfix program,
the same on host and device (`RegionView`).

Volume fractions are cell averages of the indicator, integrated over the
cell's triangles (2D fan) or tetrahedra (`Mesh::h_cell_tetrahedra`), each
subdivided by midpoints only where the surface (or the smooth band) can cross
it: a simplex whose centroid is farther from the surface than the simplex's
radius is uniform, by the Lipschitz bound. Up to 8 levels in 2D and 5 in 3D,
the volumes of the tests' regions come out within 2e-5 (2D) and 2e-4 (3D) on
meshes of 8 and 5 cells across. A cell entirely inside gets exactly 1. Implicit
regions have no bound; they subdivide where `f` may change sign over a simplex
by twice its slope there.

Smooth edges use the smoothstep of the signed distance across a band of width
`edge`. It is odd about the surface, so a flat surface keeps its volume; a
curved one gains the band's second moment (`pi edge^2 / 20` for a disk,
`pi R edge^2 / 5` for a sphere), which the tests check.

### Exact totals

With `total = true` a value `Q(t)` becomes the density `Q(t) f_c / W`, with `f_c`
the volume fraction and `W = sum_c f_c V_c` over the owned cells of all ranks
(computed once). Its integral `sum_c Q f_c / W V_c` is `Q` to rounding on any
mesh; the volume fractions only decide how it is distributed. A total may
depend on `t` alone, so `W` never changes.

## Cost

Measured on one A100 (2D Euler, MUSCL, 1M triangles, SSPRK3, 300 steps,
best of 3), the source `rhoE = 1e-3 sin(50 t) exp(-((x - 1)^2 + (y - 0.5)^2) /
0.01)`:

| Case | ms per step | CUDA graph |
|---|---|---|
| No source (0.8 / now) | 3.20 / 3.14 | yes |
| 0.8: `[source]` with `time_dependent = true` (host evaluation, copy per stage) | 173 | no |
| Now: `[[sources]]`, everywhere | 3.80 (+21%) | yes |
| Now: `[[sources]]` in a disk of radius 0.3 with a smooth edge | 3.32 (+6%) | yes |

The interpreter runs about 20 instructions, `sin` and `exp` among them, in every cell at
every stage; cells outside a region skip it.

## Reuse

Boundary expressions (#316), sponges and `[initialize]` (#324) evaluate kexpr
on the device like sources (#449). Sponges and probes take regions too (#499).

`region::Regions` holds the regions of one setup: inline ones, and each name of
`[regions]` once, with its volume fractions computed the first time a key asks
and shared by every key that names it (a source, the sponge and the probe of
the same region read one array). It reads `[regions]` lazily, entry by entry,
so a name no key uses stays an unused key, reported like any other. Each
primitive type has one fixed dimension (`types()` in `region_input.cpp`; all of
them fill space so far), a boolean requires its parts to share it, and
consumers check it (`require_full_dimension` for sources, sponges and region
probes), so that
points, curves and surfaces can join the same table without any consumer
inferring a dimension from where it is used.

Region probes reduce cell averages weighted by `f V` over the cells with
`f > 0`. Each rank evaluates its cells on the device; rank 0 gathers the values
and the weights in global cell order and sums them in that order, as the
probes of points and curves do, so the reductions are the same bit for bit on
any number of ranks.
