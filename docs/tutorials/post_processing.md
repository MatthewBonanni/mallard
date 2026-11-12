# From run to figure

A run ends with a directory of files. This tutorial takes such directories, three runs of one case on
three meshes, and turns them into numbers and figures: it finds out what each run wrote and how, measures
the flow after the fact with probes the input never had, computes a field the run never wrote, estimates
the discretization error from the three meshes, and draws the figures in Python. The case is the
steady wake of a cylinder at Re = 40, a classic with reference values for every quantity measured here.
The three runs take about 40 minutes on two CPU threads; everything after them takes seconds.

You will use: `mallard info` on a run directory and on a snapshot, the resolved input a file records,
`mallard post extract` with probes added after the run and with a snapshot of a derived field,
`mallard post convergence` with its observed order, extrapolated value and grid convergence index, and the
[Python package](../guide/python.md) that reads every file into numpy and runs the commands. Files:
[`tutorials/post_processing/`](../../tutorials/post_processing). Setup: [Tutorials](index.md), with the
Python package installed.

## 1. The case

The [cylinder tutorial](cylinder.md)'s flow, slowed to Re = U D / ν = 40 (μ = 0.005 in its units: D = 1,
density and sound speed 1, U = 0.2). Below Re ≈ 47 the wake does not shed: two vortices stay attached
behind the cylinder, and the flow becomes steady. Three quantities describe it, all known from computations
and experiment:

| Quantity | References (incompressible) |
|---|---|
| Drag coefficient C_D | 1.50-1.54 |
| Length of the recirculation behind the cylinder, L / D | 2.13-2.35 |
| Separation angle, from the rear | 53.7°-55.6° |

The ranges span [Dennis & Chang (1970)](../references.md#dennis-chang-1970),
[Fornberg (1980)](../references.md#fornberg-1980),
[Coutanceau & Bouard (1977)](../references.md#coutanceau-bouard-1977) (experiment) and
[Taira & Colonius (2007)](../references.md#taira-colonius-2007).

[`input.toml`](../../tutorials/post_processing/input.toml) is the cylinder tutorial's input with the
viscosity of Re = 40, no transverse kick in the initial state and a run to t = 200 (40 D / U). It writes
what a production run would: the forces on the wall (`[[forces]]`, every 20 steps) and a snapshot of `RHO`,
`U` and `P` every 50 time units. Nothing else: everything below is measured afterwards.

## 2. Three meshes, three runs

[`cylinder.geo`](../../tutorials/cylinder/cylinder.geo) takes a `refine` factor that divides every cell
size, so one script gives a family of meshes refined uniformly. The base here is coarser at the wall than
the cylinder tutorial's (a first cell of 0.04 D: the boundary layer at Re = 40 is thick), and the three
meshes are refined by √2 in each direction from one to the next. With the build's `src` directory on
`PATH`, `mallard` is the command below, and the Python package finds it and its library there:

```bash
export PATH=$MALLARD/build/src:$PATH
mkdir -p run/post_processing && cd run/post_processing
for mesh in coarse medium fine; do
    case $mesh in coarse) refine=0.7071;; medium) refine=1;; fine) refine=1.4142;; esac
    mkdir -p $mesh && cp $MALLARD/tutorials/post_processing/input.toml $mesh/
    gmsh $MALLARD/tutorials/cylinder/cylinder.geo -2 -setnumber n_theta 64 -setnumber n_ring 10 \
        -setnumber dn0 0.04 -setnumber h_wake 0.2 -setnumber refine $refine -o $mesh/cylinder.msh
    (cd $mesh && mallard -i input.toml --kokkos-num-threads=2)
done
```

| Run | Cells | Steps | Wall time (2 threads of a busy machine) |
|---|---|---|---|
| `coarse` | 3,656 | 10,357 | 4.5 minutes |
| `medium` | 7,266 | 20,609 | 6.4 minutes |
| `fine` | 14,444 | 41,181 | 27 minutes |

The time step halves from one mesh to the next (viscosity in the cells at the wall limits it), so each
step costs twice as much on half the time step: each run about four times the one before.

## 3. What a run wrote: `mallard info`

`mallard info` on a run directory reads its run record (`mallard.jsonl`) and lists every file the run
wrote, checking each against the size it recorded (`--verify` checks their SHA-256 too):

```bash
mallard info medium
```

```text
== Run medium =============================================================================
  Run record         mallard.jsonl, the last run in it
  Run id             602d29dc-0767-4de4-92ce-03ddd353c83b
  Restarted from     no: a fresh start
  Started            2026-11-11T16:05:33Z
  Ranks              1 rank
  Mallard            v0.9.1-63-gc51fd4e2
  Ended              finished: t_stop = 200 reached
  Last step          20609
  Last snapshot      ./solut/wake_000004.vtkhdf, step 20609, t = 200 s
  Simulated time     t = 200 s at the end
  Wall time          6m 26s, the whole run

== Output files =================================================================== 2 files
  file                      kind      times  first t [s]  last t [s]  last step  fields  format
  forces_cylinder.csv       table     1,030  0.190063414  199.92089   20600      6       CSV
  solut/wake.vtkhdf.series  snapshot  5      0            200         20609      3       VTKHDF series index (VTKHDF snapshots)

== Files against the run record ======================= 5 recorded as written, size checked
  Recorded files     all present, as written
```

On a file, it reports what the file holds: its mesh, times and steps, fields with units and ranges, and the
[run block](../guide/output.md#run-block) of the run that wrote it, the same in every file of the run:

```bash
mallard info medium/solut/wake.vtkhdf.series
```

```text
== Fields =================================================== ranges at t = 200 s, the last
  field  units       components  min          max       stored   description
  P      kg m-1 s-2  1           0.695269     0.737551  float64  pressure
  RHO    kg m-3      1           0.968085     1.02349   float64  density
  U      m s-1       3           0.000556964  0.238571  float64  velocity
  min and max of fields with several components are of their magnitude.

== Run ===================================================== same for every file of the run
  Mallard            v0.9.1-63-gc51fd4e2 (commit c51fd4e20d45)
  Run block          version 3
  Build              2D, double precision, Release, Threads backend, MPI no, HDF5 yes
  Run id             602d29dc-0767-4de4-92ce-03ddd353c83b
  Restarted from     no: a fresh start
  Run started        2026-11-11T16:05:33Z
  Ranks              1 rank on 1 host
  Directory          .../run/post_processing/medium
  Input file         input.toml
  Input sha256       08d40f8d28212543686f394791824a02a93499fd3c90d876390b4ddf19a3b775
  Input recorded     resolved; mallard info --input prints it
  Mesh               file
  Mesh file          cylinder.msh
  Mesh sha256        cdb4a86d12c3dbc69b608b57157097bfa14b0de5501689bdc3a9c4847f84498b
```

Every file records the input that produced it, resolved: every key the run read, defaults filled in.
`mallard info FILE --input > rerun.toml` gives an input that reruns the case, however long ago the run was
and whatever happened to its input file since. The run is not identified by the file's name or directory:
step, time and run come from what the file records.

## 4. Measuring after the run: `mallard post extract`

The run recorded the forces, but not the distribution of pressure and friction on the wall, nor the
velocity along the wake axis. [`probes.toml`](../../tutorials/post_processing/probes.toml) adds them as
`[[probes]]`, exactly as an input would:

| Probe | Geometry | Variables, reductions | File |
|---|---|---|---|
| `wall` | `boundary = "cylinder"` | `CP`, `CF` of the free stream, `TRACTION_PRESSURE`, `TRACTION_VISCOUS`: `integral` | `probe_wall.csv`, `probe_wall_reductions.csv` |
| `axis` | a line from (0.5, 0) to (5.5, 0), 201 points | `U_X`, point values | `probe_axis.csv` |
| `contour` | a circle of radius 5 around the cylinder | `MASS_FLUX`, `MOMENTUM_FLUX`: `integral` | `probe_contour_reductions.csv` |

`mallard post extract` evaluates them on the snapshots, with the run's own code, so each probe gives what it
would have given in the run (the same values for snapshots of the conservative variables, the same to
rounding for snapshots of `RHO`, `U` and `P`, as here):

```bash
for mesh in coarse medium fine; do
    (cd $mesh && cat input.toml $MALLARD/tutorials/post_processing/probes.toml > measure.toml &&
     mallard post extract solut/wake.vtkhdf.series -i measure.toml)
done
```

The input given with `-i` must be the run's case: its mesh, physics and numerics; only its `[[probes]]`
and `[regions]` are new. Run it in the run's directory, so the mesh file resolves. The files go to
`extract/` under the names the probes give them, never over the run's own. The report ends with the
reductions of the last snapshot, in the fine run:

```text
== probe_wall_reductions.csv ========================================== t = 200 s, the last
  column                        value            units
  length                        3.14098214       m
  integral_CP                   -1.02092069      m
  integral_CF                   0.530380813      m
  integral_TRACTION_PRESSURE_X  0.0207651252     kg s-2
  integral_TRACTION_PRESSURE_Y  -4.20926906e-07  kg s-2
  integral_TRACTION_VISCOUS_X   0.011134492      kg s-2
  integral_TRACTION_VISCOUS_Y   -1.00614686e-07  kg s-2

== probe_contour_reductions.csv ======================================= t = 200 s, the last
  column                    value           units
  length                    31.4159265      m
  integral_MASS_FLUX        5.34867893e-05  kg m-1 s-1
  integral_MOMENTUM_FLUX_X  -0.0311596759   kg s-2
  integral_MOMENTUM_FLUX_Y  1.9290865e-06   kg s-2
```

`integral_TRACTION_PRESSURE_X` plus `integral_TRACTION_VISCOUS_X` is the drag of `[[forces]]`. The circle's
`MOMENTUM_FLUX` is the momentum, pressure included, flowing out through it. In a steady flow nothing
accumulates inside, so it is minus the drag, apart from the viscous stress on the circle, which the flux
variables leave out. The circle gives C_D = 1.558 against 1.595 on the wall in the fine run, a gap that
shrinks with the mesh: 5.1%, 3.3% and 2.3% on the coarse, medium and fine runs. The net `MASS_FLUX`,
5.3e-5 against the 2.0 entering through the circle's upstream half, shows how steady the flow has become.

Without an input, `--region` extracts one region or probe given on the command line, such as
`--region '{type = "line", start = [0.5, 0], end = [5.5, 0], n_points = 201}' --fields U_X`; `--steps`,
`--times` and `--latest` choose the snapshots.

**A field the run did not write.** `--snapshot` writes the extracted fields as a snapshot, so any field of
[`[[write_data]]`](../guide/output.md#derived-fields) can be computed after the run. The Q-criterion,
positive where rotation dominates strain:

```bash
cd fine && mallard post extract solut/wake.vtkhdf.series --latest --snapshot vtkhdf \
    --fields Q_CRITERION --name wake_q && cd ..
```

writes `fine/extract/wake_q.vtkhdf.series`, to open in ParaView or Python like the run's own snapshots.

## 5. How large is the discretization error? `mallard post convergence`

`mallard post convergence` compares the runs' forces, integrals and probe tables. It identifies each run by
its run record, not its name, orders them by cell count, and follows the procedure of
[Celik et al. (2008)](../references.md#celik-2008): from three meshes, the order of accuracy they show, the
value extrapolated to zero cell size, and the grid convergence index (GCI), an error band on the fine run's
value. For the two parts of the drag:

```bash
mallard post convergence coarse medium fine --quantity Fx_pressure,Fx_viscous
```

```text
== Runs ===================================================================================
  run     cells   h / h_fine  run id
  coarse  3,656   1.988       10e8f857-e923-4431-8fff-2f9181635b39
  medium  7,266   1.41        602d29dc-0767-4de4-92ce-03ddd353c83b
  fine    14,444  1           56b67a3e-bd79-49d1-9942-3e953b5cf843
  design order 3: the runs' [numerics.face_reconstruction]

== Grid convergence ================================================================== WARN
  status  quantity                     fine         extrapolated  units   order  GCI    behavior
  WARN    forces_cylinder:Fx_pressure  0.020765125  0.020581499   kg s-2  1.72   1.11%  monotonic
  WARN    forces_cylinder:Fx_viscous   0.011134492  0.0099813681  kg s-2  0.38   12.9%  monotonic

  forces_cylinder:Fx_pressure: converging monotonically, order 1.72 (design 3), extrapolated
  0.020581499; fine-grid value 0.020765125 +/- 1.11% (GCI, Fs = 1.25); order 1.72 is more
  than 1 from the design order 3
  forces_cylinder:Fx_viscous: converging monotonically, order 0.38 (design 3), extrapolated
  0.0099813681; fine-grid value 0.011134492 +/- 12.9% (GCI, Fs = 1.25); order 0.38 is more
  than 1 from the design order 3

== Notes ==================================================================================
  the runs' last values are at different times (t = 199.68934 to 199.99679 s): right for a
  steady case; for an unsteady one, compare at one time with --time or over a window with
  --average
```

The pressure drag converges monotonically at order 1.7, with an error band (GCI) of 1.1% on the fine
run's value. The viscous drag converges too, but at order 0.4, far from the scheme's third order: these
meshes are not yet in its asymptotic range, so its extrapolated value is not to be trusted, and its band,
13%, says as much. `WARN` marks both orders as far from the design order. Only more meshes tell whether the
viscous order rises toward it ([Things to try](#7-things-to-try)).

The quantities are the columns of the tables the runs wrote themselves (here `forces_cylinder.csv`); the
`extract/` tables of section 4 are not among them. For an unsteady flow, `--average T0:T1` compares time
averages, and `--time T` the values at one time.

## 6. Figures: the Python package

The [Python package](../guide/python.md) reads every file above into numpy arrays through Mallard's own
reader, and runs the commands as functions that return their JSON reports:

```python
import mallard

forces = mallard.open("fine/forces_cylinder.csv")
cd = (forces.read_all("Fx_pressure") + forces.read_all("Fx_viscous")) / 0.02   # every row

wall = mallard.open("fine/extract/probe_wall.csv")
x, y, cf = wall.read("x"), wall.read("y"), wall.read("CF")                   # the last snapshot's samples

q = mallard.open("fine/extract/wake_q.vtkhdf.series")
cells = q.geometry()                                                          # points, connectivity, VTK types

study = mallard.post.convergence(["coarse", "medium", "fine"], quantity=["Fx_pressure", "Fx_viscous"])
[(s["name"], s["estimate"]["extrapolated"], s["estimate"]["gci"]) for s in study["quantities"]]
```

[`plot_post_processing.py`](../../tutorials/post_processing/plot_post_processing.py) does this for every
figure of this page, and prints the measurements of each run:

```bash
python3 $MALLARD/tutorials/post_processing/plot_post_processing.py coarse medium fine --out .
```

```text
run        cells     C_D C_D circle  L / D separation
coarse     3,656  1.6330     1.5502  2.198     52.75°
medium     7,266  1.6104     1.5570  2.249     53.54°
fine      14,444  1.5950     1.5580  2.270     53.52°
reference C_D: 1.5 to 1.54
reference L / D: 2.13 to 2.35
reference separation angle [deg]: 53.7 to 55.6
forces_cylinder:Fx_pressure: converging monotonically, order 1.72 (design 3), extrapolated 0.020581499; fine-grid value 0.020765125 +/- 1.11% (GCI, Fs = 1.25); order 1.72 is more than 1 from the design order 3
forces_cylinder:Fx_viscous: converging monotonically, order 0.38 (design 3), extrapolated 0.0099813681; fine-grid value 0.011134492 +/- 12.9% (GCI, Fs = 1.25); order 0.38 is more than 1 from the design order 3
```

**The wake.** The fine run's U_X with the recirculation outlined, and the Q-criterion from section 4.

![Streamwise velocity around the cylinder with the recirculation outlined, and the Q-criterion computed after the run](../images/tutorials/post_processing_field-light.svg#gh-light-mode-only)
![Streamwise velocity around the cylinder with the recirculation outlined, and the Q-criterion computed after the run](../images/tutorials/post_processing_field-dark.svg#gh-dark-mode-only)

**The wall and the axis.** The pressure coefficient around the upper half of the cylinder, from the front
stagnation point (C_p = 1.17: at low Reynolds numbers viscous stresses raise it above the inviscid 1) to
the rear; the skin friction, which changes sign where the boundary layer separates; and U_X on the wake axis, negative in the recirculation.
The three meshes agree closely. The separation angle, 53.5° from the rear on the fine run, is within 0.2°
of the references' 53.7°-55.6°, and the recirculation length, L / D = 2.27, within their 2.13-2.35.

![Wall pressure and skin-friction coefficients against the angle from the front, and the streamwise velocity on the wake axis, on three meshes](../images/tutorials/post_processing_wall-light.svg#gh-light-mode-only)
![Wall pressure and skin-friction coefficients against the angle from the front, and the streamwise velocity on the wake axis, on three meshes](../images/tutorials/post_processing_wall-dark.svg#gh-dark-mode-only)

**The drag.** The drag coefficient while the wake settles, and its convergence with the mesh: the three
runs, the extrapolated value and the fine run's GCI band. C_D settles by t ≈ 150 (30 D / U) on every mesh. The fine run's 1.595 is 4% above the references'
1.50-1.54; the extrapolated value, 1.528, is within them, though its viscous part rests on the low observed
order above. Refinement removes only the discretization error, not the far field at 20 D or the Mach
number of 0.2, which the incompressible, unbounded references do not have: with the far field at 60 D,
the medium run's C_D falls from 1.610 to 1.573.

![The drag coefficient against time on three meshes, and against the cell size with its extrapolated value and error band](../images/tutorials/post_processing_convergence-light.svg#gh-light-mode-only)
![The drag coefficient against time on three meshes, and against the cell size with its extrapolated value and error band](../images/tutorials/post_processing_convergence-dark.svg#gh-dark-mode-only)

## 7. Things to try

- **A larger domain.** `-setnumber R_far 60 -setnumber h_far 6` moves the far field from 20 to 60
  diameters (8,628 cells for the medium mesh, 11 minutes): the drag of the paragraph above.
- **Other probes, the same runs.** Add a `line` probe across the wake at x = 3 to `probes.toml` and
  extract again: the runs need not be repeated.
- **A fourth mesh.** `refine = 2` (about 29,000 cells, four times the fine run's cost): with four runs,
  `mallard post convergence` reports an order for each three successive meshes, which agree once the
  meshes are in the asymptotic range.
- **An unsteady wake.** Run the [cylinder tutorial](cylinder.md)'s shedding wake at Re = 100 on two or
  three of these meshes: `mallard post convergence --average 100:160` compares their mean forces over the
  same window.

Next: [Supersonic flow over a cone](cone.md), in 3D.
