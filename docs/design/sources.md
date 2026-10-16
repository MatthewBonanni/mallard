# Source terms on the device, and regions

Design notes for `[[sources]]` and regions (#317): why expressions are
compiled by Mallard instead of evaluated by exprtk, how stage times reach
captured steps, and how regions give exact totals on any mesh.

## Compiled expressions

exprtk evaluates a tree of virtual nodes in host memory, so device kernels
cannot call it. Until 0.8 a time-dependent `[source]` was evaluated on the
host at every Runge–Kutta stage, cell by cell, and copied to the device: the
step left the CUDA graph and spent most of its time on the host.

`src/common/compiled_expression.{h,cpp}` parses the same syntax once on the
host into a postfix program of 16-byte instructions (`expr::Program`), which
`expr::evaluate` runs with a small stack. It is one `KOKKOS_INLINE_FUNCTION`,
so the host and the kernels execute the same code; variables come from a
callable index → value, so a cell, a face or a point binds its own values
without copying them into an array. `expr::ProgramTable` packs many programs
into one device array.

The values are exprtk's to the last bit on the host (tested on 22 expressions ×
203 points), which needs exprtk's own semantics, not just its grammar:

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

## Stage times in captured steps

A graph-captured step replays kernels whose time and time step live on the
device (`step_scalars`). The right-hand side therefore gets the stage's
fraction `c` of the step besides its host time (`RHSFunction(U, R, t, c)`):
inside a capture a one-thread kernel sets the device stage time to
`t_n + c dt` from the step's device scalars; otherwise it copies the host
stage time. For FE, SSPRK3 and RK4 `c dt` is exact, so both give the same
bits as the host's `t + c dt`. A step with time-dependent sources is a CUDA
graph like any other; only domain averages (`mean_*`), which need a global sum
on the host per stage, keep it out.

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
the volumes of the tests' shapes come out within 2e-5 (2D) and 2e-4 (3D) on
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

Measured on one A100 (2D Euler, MUSCL, 1M triangles, SSPRK3), a Gaussian
`rhoE` source of `t` and `x, y`:

PERF_TABLE

## Reuse

Boundary expressions (#316), sponges and `[initialize]` still evaluate exprtk
on the host where they are evaluated once; adopting `expr::Program` is a
matter of binding their variables. Regions are meant for sponge and
initialization regions and for probes (#318).
