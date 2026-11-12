# Grid convergence

How large is the discretization error of a result? Run the case at three or more
resolutions, then let `mallard post convergence` compare them. For each quantity
it reports the order of accuracy the runs show, the value extrapolated to zero
cell size, and the grid convergence index (GCI): an error band on the finest
run's value. A verdict in words says whether the quantity converges and how far
to trust the band. Use it to show that a result is mesh independent, to put an
uncertainty on a force coefficient, or to verify that a scheme reaches its design
order on a case with an exact solution.

```bash
mallard post convergence coarse medium fine
```

Each argument is a directory a run ran in. The order you give them in doesn't
matter.

## Preparing the runs

Make the runs identical except for the mesh. Refine it by the same factor in every
direction, keep the domain, and give each run its own directory:

```text
coarse/input.toml   Nx = 32,  Ny = 16
medium/input.toml   Nx = 64,  Ny = 32
fine/input.toml     Nx = 128, Ny = 64
```

Then run each one in its directory (`cd coarse && mallard -i input.toml`). Write
the quantities you want to study as tables: `[[forces]]` on walls, `[integrals]`
over the domain, `[[probes]]` reductions or points ([Probes and monitors](probes.md)).
Use a time step that follows the mesh (`cfl`, not `dt`), so that the time error
shrinks with the cells. A refinement ratio of at least 1.3 between runs keeps the
differences above the noise; 2 is common.

## What it reads

Runs are identified by what they record, never by their names. Each directory's
run record (`mallard.jsonl`, [Output](output.md#run-record-mallardjsonl)) gives:

- the runs' ids, so a table counts only rows written by this run or by the runs
  it was restarted from, and rows `mallard post extract` wrote from their
  snapshots;
- its cell count `N`, from the progress records;
- its dimension `d` and resolved input, from its [run block](output.md#run-block).

The runs are sorted by cell count, and the representative cell size of each is
$h = (1/N)^{1/d}$ ([Celik et al. 2008](../references.md#celik-2008), eq. 1, for a
fixed domain). Only ratios of $h$ matter. If the runs' inputs differ in anything
other than `[mesh]`, `[run]`, `[output]`, `[[write_data]]` and `[parallel]`, the
report gives a warning: a convergence study should change only the resolution.

The quantities are the columns of every CSV table in the run directories with one
row per time: forces, integrals, probe reductions and single-point probes, each
named `TABLE:COLUMN` (for example `integrals:kinetic_energy` or
`forces_cylinder:F_x`). By default each run's last value is taken. That's right for
a steady case even when the runs stopped at different times; the report notes when
they did. For an unsteady case, pick one time with `--time T` or average over a
window with `--average T0:T1`. A row is at `T` when its time is within $10^{-9}$ of
it, relative, and in the window when it is within $10^{-12}$ of its ends; for runs
in single precision, both are $1.2 \times 10^{-5}$ (see below).

A probe added after the runs (a separation point, a recirculation length, a flux
through a contour) enters the study once [`mallard post extract`](probes.md#probes-of-existing-output)
has evaluated it on each run's snapshots into the run's directory:

```bash
for run in coarse medium fine; do
    (cd $run && mallard post extract restart_*.restart --probe wall)
done
mallard post convergence coarse medium fine --quantity extract/probe_wall_reductions
```

The table `extract/probe_wall_reductions.csv` was written by the extraction, not by
the run, but its run block lists the snapshots it read with the run id, step and
time each records ([run block `files`](output.md#run-block)). A row counts when
it is of a snapshot of one of the run's parts, so a table extracted from another
run's snapshots is not the run's. The report names the tables that came from an
extraction, and `--format json` marks their quantities `"extracted": true`.

## The method

The procedure follows [Celik et al. (2008)](../references.md#celik-2008).
Number the grids from fine to coarse, $h_1 < h_2 < h_3$, with values
$\phi_1, \phi_2, \phi_3$, ratios $r_{21} = h_2/h_1$ and $r_{32} = h_3/h_2$, and
changes $\varepsilon_{21} = \phi_2 - \phi_1$ and $\varepsilon_{32} = \phi_3 - \phi_2$.
The observed order $p$ solves

$$
\begin{equation}
p = \frac{1}{\ln r_{21}} \left| \ln \left| \frac{\varepsilon_{32}}{\varepsilon_{21}} \right| + q(p) \right|,
\qquad
q(p) = \ln \frac{r_{21}^p - s}{r_{32}^p - s},
\qquad
s = \operatorname{sign} \frac{\varepsilon_{32}}{\varepsilon_{21}}
\label{eq:observed-order}
\end{equation}
$$

by fixed-point iteration from $q = 0$. For a constant ratio, $q = 0$ and the first
iterate is exact. The extrapolated value, the relative errors and the fine-grid
GCI are then

$$
\begin{equation}
\phi_\text{ext} = \frac{r_{21}^p \phi_1 - \phi_2}{r_{21}^p - 1},
\qquad
e_a = \left| \frac{\phi_1 - \phi_2}{\phi_1} \right|,
\qquad
e_\text{ext} = \left| \frac{\phi_\text{ext} - \phi_1}{\phi_\text{ext}} \right|,
\qquad
\text{GCI} = \frac{F_s\, e_a}{r_{21}^p - 1}
\label{eq:gci}
\end{equation}
$$

The GCI is a relative band on $\phi_1$, reported as a percentage. Its factor of
safety $F_s$ is Roache's: 1.25 when the order is observed on three or more grids,
and 3 when it is assumed with two
([Roache 1994](../references.md#roache-1994), [1998](../references.md#roache-1998)).

The ratio $R = \varepsilon_{21}/\varepsilon_{32}$ classifies the quantity:

| Behavior | When | Status |
|---|---|---|
| `monotonic` | $0 < R < 1$: the changes shrink, with one sign | ok |
| `oscillatory` | $R < 0$: the changes alternate in sign; the order and GCI are rough | WARN |
| `divergent` | $R \ge 1$: the changes grow with refinement; no order or GCI | ERROR |
| `unchanged` | every change is round-off: grid independent | ok |
| `zero` | every value is below $10^{-12}$ of the largest column of its table: round-off around zero | ok |
| `indeterminate` | the finest change is round-off, the coarser one is not; add a grid | WARN |
| `assumed` | two runs: the order is assumed, not observed, and $F_s = 3$ | WARN |

Changes smaller than $10^{-10}$ times the largest value count as round-off.
When any of the runs was made in single precision (`build.precision` in its run
record), this threshold, the $10^{-12}$ of `zero` and the tolerances of `--time` and
`--average` are all 100 machine epsilons of single precision instead,
$1.2 \times 10^{-5}$.
A converging quantity also gets a warning when its order is more than 1 away from
the design order, or when more than three runs give orders that are not yet
steady. The design order is that of the runs' reconstruction (`FO` 1, `MUSCL` 2,
`TENO` its `order`), or `--order P`.

**More than three runs.** Each three successive runs give an estimate. The finest
three are the result, and the others are listed so you can see whether the order
settles. When the order on the finest three differs from the next by more than 20%,
the coarser runs are outside the asymptotic range.

**Two runs.** With `--order P`, or the design order, the order is assumed rather
than observed, so $F_s = 3$. The verdict always starts
"2 grids: order assumed, not observed; Fs = 3" and the status is WARN.

## Errors against an exact solution

For a case with an exact solution (a manufactured solution, an advected wave,
an exact Riemann solution before the waves interact), `--field` and `--exact` give
the L1, L2 and Linf errors of a field on each run, and their observed orders:

```bash
mallard post convergence n16 n32 n64 --field RHO \
    --exact '1 + 0.2 * sin(2 * pi * (x - t)) * sin(2 * pi * (y - 0.5 * t))'
```

The exact solution is an [expression](expressions.md) of `x`, `y`, `z` and `t`.
It is averaged over each cell, with quadrature exact to degree 6 (weighted by
`y` in axisymmetric runs), at the time of the run's last snapshot, or of `--time T`.
That makes it comparable with the cell averages the run stores. The norms are
volume-weighted:

$$
L_1 = \frac{\sum_i \lvert e_i \rvert V_i}{\sum_i V_i}, \qquad
L_2 = \left( \frac{\sum_i e_i^2 V_i}{\sum_i V_i} \right)^{1/2}, \qquad
L_\infty = \max_i \lvert e_i \rvert
$$

The snapshot is the latest restart file, VTKHDF or XDMF snapshot of the whole mesh
that holds the field. The mesh is the one the run block records, so run the
command where it can find the mesh file. Give a component of a vector as `U[0]`.
Each argument can also be a snapshot or restart file. The order of each pair of
successive runs is $p = \ln(e_\text{coarse}/e_\text{fine}) / \ln(h_\text{coarse}/h_\text{fine})$;
with three or more runs, the least-squares slope over all of them is also reported.
An error that grows from the next-finest run to the finest is an error status.

Norms against the finest run, for cases without an exact solution, are planned;
`--field` without `--exact` is a usage error for now.

## Reading the report

The density wave above, on a periodic unit square with `u = [1.0, 0.5]` and
`p = 1`, advected to `t_stop = 0.05` with unlimited MUSCL (design order 2) on
16², 32² and 64² cells, writing `[integrals]` and a restart file at the end:

```toml
[initialize]
type = "analytical"
rho = "1.0 + 0.2 * sin(2 * pi * x) * sin(2 * pi * y)"
u = ["1.0", "0.5"]
p = "1.0"

[numerics.face_reconstruction]
type = "MUSCL"
limiter = "none"

[integrals]
interval = 1

[[write_data]]
prefix = "restart"
format = "restart"
time_interval = 0.05
```

```text
$ mallard post convergence n16 n32 n64 --quantity density_squared,kinetic_energy,enstrophy
== Runs ===================================================================================
  run  cells  h / h_fine  run id
  n16  256    4           9bc1ba6b-7635-4738-8b28-adabd49875cb
  n32  1,024  2           a5f2cab1-a77c-435e-ab3a-6f05102b55d0
  n64  4,096  1           494054ce-9a8a-4f5d-a668-e1aaafb12721
  design order 2: the runs' [numerics.face_reconstruction]

== Grid convergence ==================================================================== ok
  status  quantity                   fine           extrapolated  units       order  GCI       behavior
  ok      integrals:density_squared  1.0099828      1.0099989     kg2 m-4     2.16   0.00199%  monotonic
  ok      integrals:enstrophy        4.6460009e-29  -             kg m-1 s-2  -      -         zero
  ok      integrals:kinetic_energy   0.625          0.625         kg m s-2    -      0%        unchanged

  integrals:density_squared: converging monotonically, order 2.16 (design 2), extrapolated
  1.0099989; fine-grid value 1.0099828 +/- 0.00199% (GCI, Fs = 1.25)
  integrals:enstrophy: zero to round-off: below 1e-12 of the largest column of its table on
  every run
  integrals:kinetic_energy: the same on every grid, to round-off: grid independent
```

The kinetic energy is conserved exactly, so it's `unchanged`. The enstrophy of
this vorticity-free flow is round-off, `zero`: it's below $10^{-12}$ of the largest
column of its table on every run, so it isn't mistaken for a diverging quantity.

```text
$ mallard post convergence n16 n32 n64 --field RHO \
      --exact '1 + 0.2 * sin(2 * pi * (x - t)) * sin(2 * pi * (y - 0.5 * t))'
== Runs ===================================================================================
  run  cells  h / h_fine  run id
  n16  256    4           9bc1ba6b-7635-4738-8b28-adabd49875cb
  n32  1,024  2           a5f2cab1-a77c-435e-ab3a-6f05102b55d0
  n64  4,096  1           494054ce-9a8a-4f5d-a668-e1aaafb12721
  design order 2: the runs' [numerics.face_reconstruction]

== Error of RHO against the exact solution ============================================= ok
  RHO = 1 + 0.2 * sin(2 * pi * (x - t)) * sin(2 * pi * (y - 0.5 * t)), cell averages
  status  norm  256 cells  1,024 cells  4,096 cells  orders
  ok      L1    0.0004448  0.0001024    2.498e-05    2.12, 2.03
  ok      L2    0.0005277  0.0001182    2.856e-05    2.16, 2.05
  ok      Linf  0.001003   0.0002193    5.163e-05    2.19, 2.09

  L1: order 2.03 on the finest pair (design 2), 2.08 fitted over every grid
  L2: order 2.05 on the finest pair (design 2), 2.10 fitted over every grid
  Linf: order 2.09 on the finest pair (design 2), 2.14 fitted over every grid

== Notes ==================================================================================
  n16: RHO of n16/restart_000007.restart at t = 0.05 s
  n32: RHO of n32/restart_000014.restart at t = 0.05 s
  n64: RHO of n64/restart_000027.restart at t = 0.05 s
```

The runs table lists each run's cells and its size relative to the finest. The
`fine` column is the finest run's value; `extrapolated` is $\phi_\text{ext}$;
`GCI` is the band on `fine`. `--format json` prints everything as one object
(`schema` `"mallard-convergence"`): the runs, and for each quantity its values per
run, the estimate of the finest three runs (`order`, `r21`, `r32`, `R`,
`extrapolated`, `e_a`, `e_ext`, `gci`, `safety_factor`), the estimate of every
three successive runs, its status, its verdict, and whether it is `extracted`
from the runs' snapshots.

The exit status is 0 without errors (warnings included), 1 if a quantity or error
diverges or the runs cannot be read or compared, and 2 for a usage error.

## Pitfalls

- **Not in the asymptotic range.** An order far from the design order, or orders
  that change from one set of three runs to the next, mean the coarse runs are too
  coarse. Add a finer run rather than trusting the band.
- **Shocks and limiters.** Discontinuities and active limiters reduce the
  observed order to about 1 or less, whatever the design order. That's a property
  of the solution, not a fault of the study.
- **Time error.** With a fixed `dt` the time error doesn't shrink with the mesh,
  and the order levels off. Use `cfl`, or a `dt` that scales with the cells.
- **Unsteady flows.** The last values of runs that stopped at different times, or
  of a single cycle, aren't comparable. Use `--time`, or `--average` over the same
  window of several periods.
- **Non-uniform refinement.** $h = (1/N)^{1/d}$ assumes the domain is the same and
  every region is refined alike. For local refinement or a time-step study, give the
  sizes with `--sizes`, one per run in the order given.
- **Values near zero.** $e_a$ and the GCI are relative to $\phi_1$, so a quantity
  that is nearly zero (the lift of a symmetric body) has a huge relative band. Judge
  it by the absolute change instead.

## See also

- [Command line: `mallard post convergence`](cli.md#mallard-post-convergence), every option
- [Probes and monitors](probes.md): the forces, integrals and probes it studies
- [Verification manual](../verification/index.md): the observed orders of Mallard's discretizations
- [Celik et al. (2008)](../references.md#celik-2008), [Roache (1994)](../references.md#roache-1994)
