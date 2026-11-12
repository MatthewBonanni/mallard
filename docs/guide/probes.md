# Probes and monitors

Probes sample the solution where you choose: at points, along curves, on cut
planes and surfaces, over boundary zones and over regions. Monitors reduce the solution to a
few numbers per output: the force on a wall and integrals over the domain. Each
writes its own files ([What a run writes](output.md#what-a-run-writes)), with
[units](output.md#units), [run block](output.md#run-block) and [snapshot metadata](output.md#snapshot-metadata).

## Probes: `[[probes]]`

A probe samples the solution on a geometry every `interval` steps (or
`time_interval` of simulation time) and writes the samples, reductions over
them, or both; a probe of a [region](#region-probes) writes integrals, means and
extrema over it. The files are the same bit for bit on any number of ranks.

On a GPU the files have the CPU's rows, in the same order, with the same steps,
times, coordinates, normals, areas and region volumes: the geometry and the
region fractions are set up on the host, and rank 0 reduces the gathered samples
(cells, for a region) on the host in sample order. The values agree
with a CPU run to rounding, not bit for bit, since the device rounds the solution
and its point values differently (fused multiply-adds, its own math functions).
The test suite checks every backend against a CPU reference (`test/data/probes`):
indices, steps and times exactly, geometry to 1e-14 and values to 1e-10 of each
column's largest magnitude. The geometry tolerance covers other CPUs and
compilers only; on one host the geometry of every backend is the same bit for bit.


```toml
[[probes]]
name = "centerline"
fields = ["RHO", "P"]
start = [0.0, 0.01]
end = [1.0, 0.01]
n_points = 200
interval = 10
```

*Used in:* [`pulsed_jet`](https://matthewbonanni.github.io/mallard/docs/examples/#pulsed-jet), [`source_regions`](https://matthewbonanni.github.io/mallard/docs/examples/#source-regions), [`moving_heaters`](https://matthewbonanni.github.io/mallard/docs/examples/#moving-heaters), [`hot_gas_regions`](https://matthewbonanni.github.io/mallard/docs/examples/#hot-gas-regions), [`cylinder`](https://matthewbonanni.github.io/mallard/docs/examples/#cylinder) and 4 more.

```toml
# Wake survey: high-order point values on an arc, a VTK series for ParaView,
# and the arc's mean and RMS in probe_wake_reductions.csv
[[probes]]
name = "wake"
fields = ["U", "P"]
sampling = "point"
type = "arc"
center = [0.0, 0.0]
radius = 5.0
angles = [-45.0, 45.0]
n_points = 91
time_interval = 1.0
format = "vtp"
reductions = ["mean", "rms"]
```

![Probes of every type over a triangle mesh, colored by their point values of the pressure](../images/guide/probe_shapes-light.svg#gh-light-mode-only)
![Probes of every type over a triangle mesh, colored by their point values of the pressure](../images/guide/probe_shapes-dark.svg#gh-dark-mode-only)

| Key | Default | Meaning |
|---|---|---|
| `name` | required | Names the files |
| `fields` | required | `[statistics]` fields; `U`, `RHOU`, `Y_*` and `RHOY_*` expand to all components and species |
| `type` | from the keys | `point`, `points` (`points` or `points_file`), `line`, `polyline`, `circle`, `arc`, `curve` (`x`, `y`, `z` of `s`), `cut_plane` (`point`, `normal`, optional `bounds`); 3D `surface` (`x`, `y`, `z` of `u`, `v`) |
| `region` | none | Instead of `type`: a [region](geometry.md#regions) or its name, reduced over its cells ([below](#region-probes)) |
| `n_points` | per type | Samples on a curve; `"cells"`: one per cell it crosses ([below](#curves-on-their-cells)); `[n_u, n_v]` on a surface; on a region, a grid ([below](#grids-in-regions)) |
| `sampling` | `cell` | `cell`: the average of the cell holding each sample; `point`: the reconstruction's value at the sample (a filled region: its integral over the region, [below](#region-probes)) |
| `interval` / `time_interval` | 1 step | Output times; `time_interval` shortens the step to land on them |
| `format` | `csv` | `csv`, `hdf5` (one extensible file), `vtkhdf` (one VTKHDF file of every output, which ParaView opens as a time series) or `vtp` (one file per output and a `.pvd` series) |
| `reductions` | none (`region`: `integral`, `mean`, `min`, `max`) | `integral`, `mean`, `favre`, `min`, `max`, `rms` over the geometry; boundaries also `moment` (about `moment_center`, default the origin) |
| `samples` | `true` | `false` writes only the reductions |
| `spectra` | none | Time spectra of the samples or reductions ([below](#time-spectra)) |
| `scale`, `rotate`, `pivot`, `translate` | none | Move the geometry (scale, then rotate, about `pivot`, then translate) |

### Curves on their cells

With `n_points = "cells"`, a `line`, `polyline`, `circle`, `arc` or `curve` is
cut exactly by the cells it crosses: one sample per cell and crossing, at the
middle of its length in the cell, weighing that length. The samples are
written as segments with their values as cell data, and each crossing
integrates with three Gauss points, so the `integral` of a line is exact for
the reconstruction polynomials up to degree 5, on any mesh and resolution. A
curve along a face between two cells is sampled once, in one of them; one along
a boundary face, in its cell. Curved cells cut it by their curved shapes.

```toml
[[probes]]
name = "cut"
type = "line"
start = [0.0, -1.0]
end = [0.0, 1.0]
n_points = "cells"
fields = ["U", "MASS_FLUX"]
sampling = "point"
reductions = ["integral"]
```

### Cut planes and surfaces

`type = "cut_plane"` intersects the mesh exactly with a plane: one sample per
cut cell, and integrals exact for the reconstruction polynomials. Surfaces
are regions, resampled on a grid with `n_points`:
`region = {type = "sphere", center = [0.5, 0.5, 0.5], radius = 0.3, filled = false}`
with `n_points = [24, 48]`. Both have normals, so fluxes through them are
variables: a closed sphere's `integral` of `MASS_FLUX` is the net outflow.

![An exact cut plane through a tetrahedral mesh, one polygon per cut cell, and a resampled sphere, both colored by point values of the pressure](../images/guide/probe_cut_3d-light.svg#gh-light-mode-only)
![An exact cut plane through a tetrahedral mesh, one polygon per cut cell, and a resampled sphere, both colored by point values of the pressure](../images/guide/probe_cut_3d-dark.svg#gh-dark-mode-only)

```toml
[[probes]]
name = "midplane"
type = "cut_plane"
point = [0.0, 0.0, 0.5]
normal = [0.0, 0.0, 1.0]
fields = ["U", "P", "MASS_FLUX"]
sampling = "point"
format = "vtp"
time_interval = 0.01
reductions = ["integral"]
```

### Boundary data

`boundary = "<zone>"` (a name or a list) or `boundary_type = "wall"` (every
wall) turns each face of the zones into a sample: the state the boundary
condition sets on the face, wall quantities and fluxes. In 2D the faces form
ordered polylines with their arc length `s`; in 3D, polygons for ParaView.

```toml
[[probes]]
name = "cylinder"
boundary = "cylinder"
fields = ["P", "CP", "CF", "Q_W", "Y_PLUS"]
reference = {rho = 1.0, u = [0.2, 0.0], p = 0.7142857142857143}
time_interval = 1.0

[[probes]]
name = "inflow"
boundary = ["inlet", "outlet"]
samples = false
fields = ["MASS_FLUX", "MOMENTUM_FLUX", "ENTHALPY_FLUX"]
reductions = ["integral"]
```

![Boundary probes on the curved walls of an annulus: each wall a closed polyline traversed with the fluid on its left, and the inner wall's pressure against arc length, point values against cell averages and the exact value](../images/guide/probe_walls-light.svg#gh-light-mode-only)
![Boundary probes on the curved walls of an annulus: each wall a closed polyline traversed with the fluid on its left, and the inner wall's pressure against arc length, point values against cell averages and the exact value](../images/guide/probe_walls-dark.svg#gh-dark-mode-only)

In 3D the faces are polygons; ParaView colors the `.vtp` series by any variable:

![Heat flux into an isothermal wall under a sheared warm blob, from a 3D boundary probe](../images/guide/probe_wall_3d-light.svg#gh-light-mode-only)
![Heat flux into an isothermal wall under a sheared warm blob, from a 3D boundary probe](../images/guide/probe_wall_3d-dark.svg#gh-dark-mode-only)

| Variable | Meaning |
|---|---|
| `MASS_FLUX`, `MOMENTUM_FLUX`, `ENTHALPY_FLUX`, `SPECIES_FLUX_*` | Convective fluxes through the face, positive out of the domain |
| `TRACTION_PRESSURE`, `TRACTION_VISCOUS` | Force per unit area on the boundary; their `integral` is the `[[forces]]` force (per radian in axisymmetric runs, as `area`) |
| `TAU_W`, `TAU_W_X`, ... | Wall shear stress (magnitude, components) |
| `CF`, `CP` | Skin friction and pressure coefficients of `reference` (the probe's, else `[output] reference`) |
| `Q_W` | Heat flux into the fluid |
| `U_TAU`, `Y_PLUS` | Friction velocity and first-cell `y+` |
| `ENERGY_DIFFUSIVE_FLUX`, `SPECIES_DIFFUSIVE_FLUX_*` | The viscous flux's diffusive energy flux (conduction, viscous work, species enthalpy) and species mass fluxes, positive out of the domain |
| `ENERGY_TOTAL_FLUX`, `SPECIES_TOTAL_FLUX_*` | Convective plus diffusive |
| `WALL_MODEL_U`, `WALL_MODEL_T`, `WALL_MODEL_H` | On [modeled walls](les.md#wall-models-wall_model): the exchange point's tangential speed, temperature and height |

On walls with a [wall model](les.md#wall-models-wall_model), the shear stress,
the tractions and `Q_W` are the model's, so `integral` of
`TRACTION_VISCOUS` and `Q_W` are the `[[forces]]` force and `Q`. The
`moment` reduction gives the moment of each vector variable about
`moment_center`, from the same faces as the forces:

```toml
[[probes]]
name = "wing"
boundary = "wing"
samples = false
fields = ["TRACTION_PRESSURE", "TRACTION_VISCOUS"]
reductions = ["integral", "moment"]   # forces and moments
moment_center = [0.25, 0.0, 0.0]      # quarter chord
```

Around a cylinder at Re = 40: the wall probe's `CP`, `CF` and `Y_PLUS`, the
`mean` and `max` of `Y_PLUS`, the `integral` of the tractions (the drag) and
the `MASS_FLUX` through a closed circle and a cut, whose net outflow settles
to zero:

![Measurements around a cylinder at Re = 40: mass flux samples on a closed circle and a cut line, the wall pressure and skin-friction coefficients against angle, first-cell y+ with its mean and maximum, and the drag and the circle's net outflow over time](../images/guide/probe_measurements-light.svg#gh-light-mode-only)
![Measurements around a cylinder at Re = 40: mass flux samples on a closed circle and a cut line, the wall pressure and skin-friction coefficients against angle, first-cell y+ with its mean and maximum, and the drag and the circle's net outflow over time](../images/guide/probe_measurements-dark.svg#gh-dark-mode-only)

### In the run record and report

Each progress row of the [run record](output.md#run-record-mallardjsonl)
holds each probe's latest reductions, and a one-point probe's latest values,
with the step and time of its last write; the start record gives their units.
These are the values [triggers](running.md#triggers) read, and
[`report.html`](output.md#run-report-reporthtml) plots them, a group per
probe. The record keeps the first 64 over every probe, in the order of
`[[probes]]`; the probe's own files hold all of them at every write.

### Accuracy and pitfalls

With TENO, `sampling = "point"` reads each cell's reconstruction polynomial at
the sample, so a probe converges at the scheme's order (fifth order for
`order = 5`) instead of the first order of cell averages at points off the
centroids. The [Sod comparison](numerics.md#reconstruction) on the Numerics
page was read from a line probe. The full key list is in the [input reference](../input.md#probes);
each type, where its samples go and how to read the files, with figures:
[Geometry](geometry.md#points-and-curves). Each probe file records where every sample is read (owning
cell, centroid, distance) and how: [Probe sampling](output.md#probe-sampling).

Pitfalls:

- Point values fall back to the linear least-squares reconstruction in cells
  the TENO indicator flags troubled (shocks, and coarse meshes of smooth
  fields), and to the cell average where neither gives a positive density
  and energy. VTP and VTKHDF
  probe files record the method and each variable's kind of value
  ([Cell averages and point values](output.md#cell-averages-and-point-values)).
- Mass fractions are cell averages in both modes; `T` and `H` follow the
  reconstructed `p / rho` at the cell's composition.
- CSV is for small probes: it repeats the coordinates every output. Use `vtkhdf`, `hdf5`
  or `vtp` for thousands of samples.
- Wall shear and heat flux use the viscous flux's one-sided wall gradients:
  second-order shear on smooth meshes, first-order heat flux at isothermal
  walls. `CF` is signed along `reference.u`.
- A cut plane is exact for the mesh's straight-sided cells (curved cells are
  cut as their straight versions); `bounds` clips it to a box.
- Convective fluxes of boundary probes are those of the face states, not the
  scheme's numerical fluxes: they agree as the mesh is refined. The diffusive
  ones are the scheme's own (molecular: without the SGS part of LES); walls
  carry no species flux, outflows without an image face none at all.
- Reductions of points weigh each point 1 (`integral` is a sum); curves
  integrate with three Gauss points per element between samples, or per
  crossing of a cell with `n_points = "cells"`.

### Samples outside the domain

Every sample a probe writes lies in the mesh. A point on a boundary face, to
round-off relative to the cell size, counts as inside, so a line from wall to
wall keeps its ends.

- `point` and `points` probes: a point outside the domain is an input error
  at setup (in a run, `mallard check` and `mallard post extract` alike) that
  names the probe, how many points are outside, and for the first few the
  point's index, its coordinates and its distance to the nearest boundary
  zone.
- Curves (`line`, `polyline`, `arc`, `circle`, `curve`) and resampled
  surfaces: the samples outside the domain are dropped, and a warning names
  the probe, how many samples and what share of the length or area were
  dropped, and where (the runs of dropped samples). The warning is printed and
  written to `mallard.jsonl`, and the file records it as `clipped` in its
  [probe sampling block](output.md#probe-sampling). A curve or surface
  entirely outside the domain is an error. A line drawn past a curved wall
  ([`[[mesh.curved]]`](meshes.md#curved-boundaries)) is clipped at the curved
  wall, and at its faceted surface without one.
- Cut planes sample the cells they cut, so they are always clipped to the
  mesh; one that misses the mesh is an error.
- Region probes cover the region's part in the domain. Where part of a filled
  region lies outside the domain (more than 0.1% of its volume, or area in
  planar 2D), a warning gives that share and the volumes inside and in all,
  and the file records it as `clipped` (`volume_fraction`, or
  `area_fraction` in planar 2D). One that misses the mesh is an error.

![Volume fractions of a disk partly outside a square triangle mesh, its part outside the domain hatched, under the warning the run prints with the share outside](../images/guide/region-outside-light.svg#gh-light-mode-only)
![Volume fractions of a disk partly outside a square triangle mesh, its part outside the domain hatched, under the warning the run prints with the share outside](../images/guide/region-outside-dark.svg#gh-dark-mode-only)

### Time averages

`[statistics]` keeps time averages per cell. A probe reads them: `MEAN_P`,
`COV_U_X_U_Y` and the other averages are probe variables, each sample the
value of its cell, so a line probe of `MEAN_P` writes the `[statistics]`
field along the line. For what has no per-cell average, such as wall
quantities, fluxes and point values, `average = true` averages the probe's own
samples in time instead:

```toml
[statistics]
t_start = 0.2            # also the start of the probe averages
interval = 5             # and their sampling

[[probes]]
name = "wall_mean"
boundary_type = "wall"
fields = ["CF", "Q_W", "Y_PLUS"]
reference = {rho = 1.0, u = [1.0, 0.0], p = 1.0}
average = true
interval = 1000          # writes the means so far every 1000 steps
reductions = ["integral", "mean"]
```

The probe samples on the `[statistics]` schedule, each sample weighted by the
time since the last, and writes `MEAN_CF`, `MEAN_Q_W`, ... every `interval`,
with reductions of the averaged samples (`integral_MEAN_CF`). Rank 0 keeps the
means in sample order, so they are the same on any number of ranks, and
restart files carry them: a restarted run continues them bit for bit.
`statistics.reset = true` starts them afresh. An averaging probe evaluates
its samples at every sampling step, not only at its outputs.

### Time spectra

A probe with `spectra` computes, during the run, the power spectral density
of its samples in time: the shedding frequency of a wake, the tones of a
cavity, the band of a combustion instability, without post-processing the
probe file.

```toml
[[probes]]
name = "wake"
type = "point"
point = [3.0, 0.5]
fields = ["U_Y", "P"]
sampling = "point"
time_interval = 0.5      # the sampling interval of the spectra

[probes.spectra]
fields = ["U_Y"]
t_start = 60.0           # once the flow is periodic
length = 140.0           # the last 140 time units: bin width 1 / 140
strouhal = {length = 1.0, velocity = 0.2}   # St = f D / U
```

Each output of the probe from `t_start` on adds a sample to a history of the
last `length / time_interval` samples. Once the history is full, the probe
writes the one-sided PSD of every listed variable at every sample point to
`probe_<name>_spectra.csv`, with columns `step, t, point, f, St, PSD_U_Y`
(units in the second line: `PSD_U_Y` in `m2 s-1`, velocity squared per
frequency), at the end of the run and, with `time_interval`, at its
multiples. Reductions are signals too: `fields = ["integral_TRACTION_PRESSURE_Y"]`
on a wall probe gives the spectrum of the lift, written to
`probe_<name>_reductions_spectra.csv` (`step, t, f, St, PSD_...`). At each
write the run's log reports the peak of each spectrum, and the run record
(`mallard.jsonl`) carries it as the monitor `probe_<name>_spectra`
(`f_peak_<variable>`, `St_peak_<variable>`):

```text
   85208 2.0000e+02  spectra  probe wake: U_Y peaks at f = 0.032942 (St = 0.1647) (bin width 0.00714)
```

The spectrum is Welch's: the history is cut into `segments` (default 1)
segments overlapping by half, the last ending at the latest sample; each has
its mean removed and is windowed (`window = "hann"`, the default, or
`"rectangular"`), and their periodograms are averaged, normalized as a
density (`scipy.signal.welch` with `scaling = "density"` gives the same
values): the PSD summed over the bins times the bin width is the variance of
the windowed signal. A sine of amplitude A at a bin center puts `A^2 / 2` of
power in its bin without a window, and over three bins (`A^2 / (3 df)` at its
own, `A^2 / (12 df)` at each neighbor) with the Hann window. The peak frequency
is refined by a parabola through the logarithms of the largest bin and its
neighbors, so it is found to a fraction of the bin width.

Restart files carry the history, so a restarted run continues it bit for bit
(rank 0 keeps it in sample order, the same on any number of ranks);
`statistics.reset = true` starts it afresh.

| Key | Default | Meaning |
|---|---|---|
| `fields` | required | Fields of the probe (`U` for its components), and reduction columns (`mean_P`, `integral_MASS_FLUX`) |
| `length` | required | Time spanned by the history; the bin width is `(segments + 1) / (2 length)` |
| `t_start` | `0` | First sample |
| `segments` | `1` | Welch segments, overlapping by half: more average out noise, at a coarser bin width |
| `window` | `hann` | `hann` or `rectangular` |
| `time_interval` | none | Also write at its multiples, once the history is full |
| `strouhal` | none | `{length, velocity}`: adds `St = f length / velocity` |
| `file`, `reductions_file` | `probe_<name>_spectra.csv`, `probe_<name>_reductions_spectra.csv` | Output files |

Pitfalls:

- The probe needs a `time_interval` (equally spaced samples), and its
  Nyquist frequency is `1 / (2 time_interval)`: sample at least a few times
  per period of the fastest motion of interest, or it aliases.
- The bin width is the resolution: four periods in the history put the peak
  4 bins from zero. The parabola locates a clean peak well within a bin, but
  close tones need a longer `length`.
- A spectrum is not written before the history is full: a run shorter than
  `t_start + length` writes none.

### Region probes

```toml
# Mass, mean temperature and pressure range in a box, every 10 steps
[[probes]]
name = "box"
fields = ["RHO", "T", "P"]
region = {type = "rectangle", center = [0.5, 0.5], size = [0.4, 0.2]}
reductions = ["integral", "mean", "min", "max"]   # the default
interval = 10
```

*Used in:* [`source_regions`](https://matthewbonanni.github.io/mallard/docs/examples/#source-regions), [`moving_heaters`](https://matthewbonanni.github.io/mallard/docs/examples/#moving-heaters), [`hot_gas_regions`](https://matthewbonanni.github.io/mallard/docs/examples/#hot-gas-regions).

A probe with `region` reduces the cell averages of the cells the region covers,
each weighted by its volume times its volume fraction `f`, so a region need not
follow the mesh: `integral` is `sum f V q` (the mass in the region for `RHO`, its momentum,
total energy and species masses for `RHOU_*`, `RHOE` and `RHOY_<species>`),
`mean` that over `sum f V`, `favre` the density-weighted mean, `rms` the root
mean square, and `min`, `max` the extremes over the cells with `f > 0`. Rank 0
sums them in global cell order, so they are the same bit for bit on any number
of ranks. The region may be one of the mesh's own: `region = "domain"` gives
the totals of the whole domain (the mass, for `RHO`), a cell zone those of its
cells ([regions of the mesh](geometry.md#regions-of-the-mesh)). A surface,
curve or set of points weighs each cell by its area, length or count in the
cell instead of `f V` ([on a mesh](geometry.md#surfaces-and-curves-on-a-mesh)):
`integral` is then the surface integral, `mean` the area-weighted mean,
evaluated at the elements' quadrature points (cell averages, or point values
with `sampling = "point"`), and the third column is `area`, `length` or
`count`. Surfaces (2D: curves) have normals, so their fluxes (`MASS_FLUX`, ...)
are variables too: a closed surface's `integral` of `MASS_FLUX` is the net
outflow. Each output
appends a row to `reductions_file` (default `probe_<name>_reductions.csv`):

```
step,t,area,integral_RHO,integral_T,integral_P,mean_RHO,...,max_P
```

The third column is `sum f V`: `volume` in 3D and axisymmetric runs (per
radian), `area` in planar 2D runs (per unit depth).

**Measures.** A filled region probe records the region's own volume and
surface area (planar 2D: area and perimeter; axisymmetric: per radian, of the
solid its part at r >= 0 sweeps), independent of the mesh, and those of its
part in the domain, in the `measures` of its
[probe sampling block](output.md#probe-sampling), which heads its reductions
file and is written to `mallard.jsonl` as a `probe` record:

- Its own: in closed form for one primitive, moved, rotated and scaled
  (`tolerance` 0); otherwise, as for booleans, polygons that cross themselves,
  ellipses and ellipsoids of unequal axes, smooth edges and implicit regions,
  by subdividing a box around it, extrapolated in the subdivision's size,
  with `tolerance` the estimated relative error (larger for the surface area
  of sharp edges and corners, which converges at first order). `null` for a
  region of the mesh or one without a bounded box.
- In the domain: the volume `sum f V`, the reductions' third column, and the
  area of its surface: the region's own surface inside the domain and the
  domain's boundary inside the region, the closed surface
  `{type = "boundary", of = {type = "intersection", regions = ["domain", ...]}}`
  that a control-volume balance integrates over.

On a mesh with [curved boundaries](meshes.md#curved-boundaries), both follow
the curved walls: the fractions `f` of the cells at a curved wall are of their
curved shapes, and the surface's part on the wall is the curved faces'.

![Volume fractions of a disk across the curved outer wall of a coarse annulus mesh, the cells drawn with their curved faces, the disk's part in the domain outlined, and its recorded area and perimeter against the closed form](../images/guide/region-curved-wall-light.svg#gh-light-mode-only)
![Volume fractions of a disk across the curved outer wall of a coarse annulus mesh, the cells drawn with their curved faces, the disk's part in the domain outlined, and its recorded area and perimeter against the closed form](../images/guide/region-curved-wall-dark.svg#gh-dark-mode-only)

![A sphere across the curved wall of a coarse tetrahedral ball: the cells' volume fractions on the plane through its center, the closed surface of its part in the domain with outward normals, and its recorded volume and surface area against the closed form](../images/guide/region-curved-wall-3d-light.svg#gh-light-mode-only)
![A sphere across the curved wall of a coarse tetrahedral ball: the cells' volume fractions on the plane through its center, the closed surface of its part in the domain with outward normals, and its recorded volume and surface area against the closed form](../images/guide/region-curved-wall-3d-dark.svg#gh-dark-mode-only)

**Moving regions and copies.** A filled region that
[moves](geometry.md#moving-regions) is placed where it is at each write: its
fractions, and so the third column, are those of that time, the region's part
in the domain as it enters or leaves it. Its own measures are recorded while it
only translates and rotates (`null` once it changes shape), and its part in
the domain is the third column of each row (`in_domain` is `null`). With
`average = true` its reductions are those of the time averages over the cells
it sweeps, each cell weighed by its mean `f V`: the integral is the mean of
the integrals, the mean that integral over the mean measure in the domain
(the third column), and the extremes those of the cells it has covered. With
`copies = "each"`, a region with [`repeat`](geometry.md#arrays-of-copies) gives
one probe per copy, `<name>_copy<k>`, each with its own reductions file.

**Regions defined by the solution.** A `where` region and an `isosurface`
([Regions defined by the solution](geometry.md#regions-defined-by-the-solution))
are placed on the solution at each write: a `where` region's fractions, and an
isosurface's elements, extracted anew from the cells' levels. The reductions
file's measure column is the region's at that time, and its `measures` are
`null`. Isosurfaces take no `n_points` or `average`.

**Point sampling.** With `sampling = "point"`, a static filled region
integrates the reconstruction over its own shape instead of `f V` times the
cell averages: a cell entirely inside is one sample of its cell value weighing
its volume (the integral of its reconstruction polynomial), and a cell the
region's surface cuts has quadrature points over its part inside, high-order
for curved surfaces and fitted to a few points of the reconstruction's
degree. On a mesh with [curved boundaries](meshes.md#curved-boundaries), the
cells at curved walls are integrated over their curved shapes. `integral` is
then exact for the reconstruction polynomials, and the third column is the
volume of that quadrature: the region's part in the domain to round-off for a
smooth surface.

### Grids in regions

A filled region with `n_points = [n_x, n_y]` (3D: `[n_x, n_y, n_z]`, each at
least 2) also writes samples at the nodes of a grid over the box around the
region and the mesh, those inside both, in grid order: the same values as
`points` probes at those nodes, in `format` (`csv`, `hdf5`, `vtkhdf` or `vtp`,
as vertices) with `sampling` as for points. Its reductions stay those of the
region's cells.

```toml
[[probes]]
name = "plume"
region = {type = "sphere", center = [0.5, 0.5, 0.5], radius = 0.2}
n_points = [32, 32, 32]
fields = ["T"]
sampling = "point"
format = "vtkhdf"
time_interval = 0.01
```

Give the region by name to measure exactly where a source or sponge acts:

**Control-volume balances.** The conserved variables and the flux variables
of the region's boundary give a region's instantaneous balances: with a
region probe of `RHO`, `RHOU` and `RHOE` (`integral`) and a probe of its
closed boundary with `MASS_FLUX`, `MOMENTUM_FLUX` and `ENTHALPY_FLUX`
(`integral`, `sampling = "point"`), d/dt of each integral plus the net
outflow is the source inside: zero for mass, minus the force on any body
inside for momentum, its heat and work for energy. The fluxes are
convective (pressure included), so viscous stresses and heat conduction
through the boundary are left out, and the time derivative comes from
successive outputs. [Tutorial 8](../tutorials/measuring.md) closes the
balances of a box around a cylinder this way.

```toml
[regions]
heater = {type = "sphere", center = [0, 0, 0], radius = 0.01, edge = 0.002}

[[sources]]
rhoE = "1 MW/m^3"
region = "heater"

[[probes]]
name = "heater"
fields = ["T"]
region = "heater"
```

Pitfalls:

- Values are cell averages (with `sampling = "point"`, point values at the
  quadrature points and cell values inside), so `min` and `max` are those of
  the samples, not of the field between them; with cell averages they
  include cells the region only touches.
- Only a surface or a filled region with `n_points` has samples: otherwise
  `type`, `format`, `file` and `samples` do not apply to a region probe.
- A [moving](geometry.md#moving-regions) region and one
  [defined by the solution](geometry.md#regions-defined-by-the-solution) take
  neither `sampling = "point"` nor `n_points`.

### Probes of existing output

A probe added after the run, or a run that wrote too few, does not need a
rerun: `mallard post extract` sets the case up again from a restart file or a
snapshot and evaluates probes on its stored state, with the run's own probe
code. The case comes from the file's [run block](output.md#run-block) (every output of Mallard 0.10
and later records its resolved input), or from `-i input.toml`, which may add
probes the run did not have:

```sh
mallard post extract solut/restart_*.restart                  # every [[probes]] of the input, at each restart
mallard post extract solut/flow.vtkhdf.series --probe wake -o wake
mallard post extract solut/flow.vtkhdf.series --latest --region domain --fields RHO,RHOE
mallard post extract solut/flow.vtkhdf.series --times 0.1:0.5 --fields MASS_FLUX --reductions integral \
    --region '{type = "cut_plane", point = [0.5, 0, 0], normal = [1, 0, 0]}'
```

`--region` takes what a probe's `region` takes: a name of `[regions]` or of
the mesh (`domain`, a boundary or cell zone), or an inline table, a region or
a probe type; its files are `probe_extract.csv` and
`probe_extract_reductions.csv` (`--name` renames them). `--steps`, `--times`,
`--every` and `--latest` choose the snapshots, and `--snapshot vtkhdf`
also writes the chosen fields (`--fields`, of `--region` if given) as
`[[write_data]]` would have. Files go to `extract/` (`-o DIR`) under the names
the input gives them, every probe at every snapshot chosen whatever its
`interval`. [Regions defined by the solution](geometry.md#regions-defined-by-the-solution)
are placed on each file's state, as the run places them on its own: `--region
'{type = "isosurface", field = "T", value = 1500}'` gives the flame's area and
the flux through it at every snapshot. The report lists what it wrote and the
reductions of the last snapshot, each with its units:

```
$ mallard post extract solut/restart_000002.restart solut/restart_000004.restart
== Extract ================================================================================
  Files              solut/restart_000002.restart (restart), solut/restart_000004.restart (restart)
  Case               the resolved input in the run block of solut/restart_000002.restart
  Snapshots          2 of 2, t = 0.002 to 0.004 s
  Output             extract

== Snapshots ================================================================== 2 extracted
  #  t [s]  step  file
  0  0.002  2     restart_000002.restart
  1  0.004  4     restart_000004.restart

== Written ======================================================================== 2 files
  file                              kind   times  first t [s]  last t [s]  fields  format
  extract/probe_line.csv            table  2      0.002        0.004       9       CSV
  extract/probe_box_reductions.csv  table  2      0.002        0.004       15      CSV

== probe_box_reductions.csv ========================================= t = 0.004 s, the last
  column        value         units
  area          0.0799609423  m2
  integral_RHO  0.079960745   kg m-1
  mean_RHO      0.999997533   kg m-3
  ...
```

`--format json` prints the same as one object. Under `mpirun -n N` the probes
are evaluated on N ranks, with the same values as on one.

The tables' run block lists each snapshot read with the run id, step and time
it records, so tables extracted into a run's directory are that run's to
[`mallard post convergence`](convergence.md#what-it-reads).

Pitfalls:

- The values are those the run's probe gave at that step: bit for bit from
  restart files and snapshots of the conservative variables (`RHO`, `RHOU`,
  `RHOE`, `RHOY_*`), to rounding from snapshots of `RHO`, `U` and `P`. Mixture
  snapshots need `P`, `T`, `U` and every `Y_*`, or the conservative variables.
- The file must be of the whole mesh (restart files, VTKHDF or XDMF
  snapshots), and the case's mesh must be the file's: run the command where
  the run ran, so that a mesh file named by a relative path is found.
- VTKHDF files record no step: their snapshots are numbered by their
  position in the files given, and `--steps` does not apply to them.
- Statistics (`MEAN_*`) and `[[forces]]`, `[integrals]` are not extracted,
  nor are [time-averaged probes](#time-averages) (`average = true`, or
  `MEAN_`/`COV_` variables): their averages are over the run's steps, which
  output does not hold. Without `--probe` they are skipped with a warning.

## Wall forces: `[[forces]]`

```toml
[[forces]]
zone = "cylinder"
interval = 10        # default 1
# file = "forces_cylinder.csv"
```

*Used in:* [`sphere_mach3`](https://matthewbonanni.github.io/mallard/docs/examples/#sphere-mach3), [`ellipsoid`](https://matthewbonanni.github.io/mallard/docs/examples/#ellipsoid), [`sphere_axisymmetric`](https://matthewbonanni.github.io/mallard/docs/examples/#sphere-axisymmetric), [`cylinder`](https://matthewbonanni.github.io/mallard/docs/examples/#cylinder), [`sphere_re300`](https://matthewbonanni.github.io/mallard/docs/examples/#sphere-re300) and 3 more.

Writes `step, t` and the pressure and viscous force components of the fluid
on the zone, per unit depth in 2D (per radian in axisymmetric runs). Divide by
the dynamic pressure and a reference length (or area) for drag and lift
coefficients. On walls with a [wall model](les.md#wall-models-wall_model)
the viscous force is the modeled stress, and a last column `Q` gives the
modeled heat flux into the fluid.

## Domain integrals: `[integrals]`

```toml
[integrals]
interval = 1
budget = true        # kinetic-energy budget (LES), default false
```

*Used in:* [`turbulent_hot_spot_2d`](https://matthewbonanni.github.io/mallard/docs/examples/#turbulent-hot-spot-2d), [`taylor_green_3d`](https://matthewbonanni.github.io/mallard/docs/examples/#taylor-green-3d), [`isotropic_turbulence`](https://matthewbonanni.github.io/mallard/docs/examples/#isotropic-turbulence), [`channel_retau180`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-retau180), [`forced_isotropic_turbulence`](https://matthewbonanni.github.io/mallard/docs/examples/#forced-isotropic-turbulence) and 5 more.

Writes `integrals.csv` with the volume integrals of the kinetic energy,
enstrophy, squared dilatation and pressure dilatation, and of `|u|^2`,
`|omega|^2`, `rho^2`, `T` and `T^2` (from which mean squares and variances
follow). For decaying turbulence (the Taylor–Green
vortex) the dissipation rate is `-dE/dt`. With TENO, velocity gradients come
from the reconstruction polynomials (order-consistent: within 1% of spectral
derivatives of the same field). `budget = true` adds the
[kinetic-energy budget](les.md#low-dissipation-flux-and-the-budget):
molecular, SGS and numerical dissipation.

## See also

- [Input reference: `[[probes]]`](../input.md#probes), [`[[forces]]`](../input.md#forces), [`[integrals]`](../input.md#integrals)
- [Geometry](geometry.md): probe types, sample placement, reading probe files
- [Statistics](statistics.md): time averages, spectra and phase averages
- [Grid convergence](convergence.md): the order, extrapolated value and GCI of forces, integrals and probes over runs at several resolutions
- [Output](output.md): snapshots, the run record (which repeats the latest forces and integrals at each progress row), units and file formats
