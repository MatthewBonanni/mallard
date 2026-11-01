# Measuring a flow

A simulation is only as useful as what you measure in it. This tutorial takes the cylinder wake of the
[previous tutorial](cylinder.md) and measures it the ways an experiment would: the force on the wall,
the mass, momentum and energy that cross a control volume around the cylinder, the drag again from that
control volume's momentum balance, the shedding frequency from a probe's spectrum, and the flow at each
phase of the shedding cycle. Nothing here needs post-processing of full snapshots: Mallard reduces
everything while it runs, into small CSV files. It takes 85,000 steps on 13,000 cells, about 8 minutes on two CPU threads.

You will use: named regions shared by several probes, a region probe (integrals and density-weighted
means over a box), cut planes with flux variables, boundary probes of the wall's tractions and pressure
coefficient, a point probe as a time series, time and phase averages (`[statistics]`,
`[statistics.phase]`), and a snapshot of just the box's cells. Files:
[`tutorials/measuring/`](../../tutorials/measuring). Setup: [Tutorials](index.md).

## 1. The case

The flow is the cylinder tutorial's: a cylinder of diameter D = 1 at Re = 100 and Mach 0.2, on the same
Gmsh mesh, in the same units (D = 1, density and sound speed 1, U = 0.2), run to t = 200, about
four and a half shedding periods after the shedding saturates at t = 60.

```bash
mkdir -p run/measuring && cd run/measuring
cp $MALLARD/tutorials/measuring/input.toml .
gmsh $MALLARD/tutorials/cylinder/cylinder.geo -2 -o cylinder.msh     # 13,000 cells
```

[`input.toml`](../../tutorials/measuring/input.toml) is the cylinder's input with measurements added.
They do not change the solution, except that the wake probe's `time_interval` shortens a step now and
then to land on its output times.

## 2. A control volume, by name

```toml
[regions]
box = {type = "rectangle", center = [1.0, 0.0], size = [6.0, 4.0]}
west = {type = "cut_plane", point = [-2.0, 0.0], normal = [-1.0, 0.0], bounds = [[-3.0, -2.0], [5.0, 2.0]]}
east = {type = "cut_plane", point = [4.0, 0.0], normal = [1.0, 0.0], bounds = [[-3.0, -2.0], [5.0, 2.0]]}
south = {type = "cut_plane", point = [0.0, -2.0], normal = [0.0, -1.0], bounds = [[-2.0, -3.0], [4.0, 3.0]]}
north = {type = "cut_plane", point = [0.0, 2.0], normal = [0.0, 1.0], bounds = [[-2.0, -3.0], [4.0, 3.0]]}
```

`[regions]` names geometry once, for every feature that takes a `region`
([named regions](../guide/geometry.md#named-regions)). `box` is the rectangle -2 < x < 4, -2 < y < 2
around the cylinder; it need not follow the mesh, since each cell counts with the fraction of it inside.
The four sides are [probe types as regions](../guide/geometry.md#probe-types-as-regions): cut planes
(lines, in 2D) that intersect the mesh exactly, each clipped by `bounds` to its side of the box and with
its `normal` pointing out of it.

## 3. The probes

Each `[[probes]]` entry samples or reduces the solution every `interval` steps (here 20, as
`[[forces]]`) and appends a row to its own CSV file
([probes](../guide/probes.md#probes-probes)):

| Probe | Geometry | Variables, reductions | File |
|---|---|---|---|
| `wall` | `boundary = "cylinder"` | `TRACTION_PRESSURE`, `TRACTION_VISCOUS`: `integral` | `probe_wall_reductions.csv` |
| `box` | `region = "box"` | `RHO`, `RHOU`, `RHOE`: `integral` | `probe_box_reductions.csv` |
| `west`, `east`, `south`, `north` | `region = "<side>"` | `MASS_FLUX`, `MOMENTUM_FLUX`, `ENTHALPY_FLUX`: `integral` | `probe_<side>_reductions.csv` |
| `wake` | a point at (3, 0.5) | `U_Y`, `P` every 0.5 time units | `probe_wake.csv` |

The `integral`s of `RHO`, `RHOU` and `RHOE` over the box are its mass, momentum and total energy. The flux variables of a cut plane are along its normal: `MASS_FLUX` is ρu·n,
`MOMENTUM_FLUX` ρu(u·n) + pn (pressure included), `ENTHALPY_FLUX` ρu·n(h + |u|²/2). The wall's traction
integrals are the force of `[[forces]]`.

The startup lists each one with the cells or samples it found (the other sides are like `west`):

```text
  phase averages     8 bins every 10 steps from t = 60, upward crossings of U_Y = 0 at (3, 0): U_X, U_Y, P
  probe wall         probe_wall_reductions.csv, boundary cylinder of 96 points every 20 steps: TRACTION_PRESSURE_X, TRACTION_PRESSURE_Y, TRACTION_VISCOUS_X, TRACTION_VISCOUS_Y; integral
  probe box          probe_box_reductions.csv, region box = rectangle over 5,036 cells every 20 steps: RHO, RHOU_X, RHOU_Y, RHOE; integral
  probe west         probe_west_reductions.csv, cut_plane of 50 points every 20 steps (point values): MASS_FLUX, MOMENTUM_FLUX_X, MOMENTUM_FLUX_Y, ENTHALPY_FLUX; integral
  probe wake         probe_wake.csv, point of 1 point every t = 0.5 (point values): U_Y, P
```

## 4. Time and phase averages

```toml
[statistics]
fields = ["U_X", "U_Y", "P"]
products = ["U_X*U_X", "U_Y*U_Y", "U_X*U_Y"]
t_start = 60.0
interval = 10

[statistics.phase]
fields = ["U_X", "U_Y", "P"]
bins = 8
probe = [3.0, 0.0]
variable = "U_Y"
level = 0.0

[[write_data]]
prefix = "./solut/box"
region = "box"
time_interval = 200.0
variables = ["U_X", "U_Y", "P", "MEAN_*", "COV_*", "PHASE_*"]
```

From t = 60, `[statistics]` keeps each cell's running means and covariances, and
`[statistics.phase]` its means in 8 bins of the shedding cycle. The cycle is measured, not given: each
upward crossing of U_Y = 0 on the wake axis at (3, 0) starts a new one
([phase averages](../guide/statistics.md#phase-averages-statisticsphase)). The snapshot writes only the box's
cells ([snapshots of regions](../guide/output.md#snapshots-of-regions)), at the start and the end.

## 5. Run

```bash
$MALLARD/build/src/mallard -i input.toml --kokkos-num-threads=4
```

This is 85,208 steps: 8 minutes on two CPU threads of an Apple M4 Pro.

## 6. What the measurements say

```bash
python3 $MALLARD/tutorials/measuring/plot_measurements.py . --t-start 60 --out .
```

```text
t >= 60, 4 shedding periods
wall:     mean Cd 1.3691 ([[forces]]), 1.3691 (wall probe); largest difference 3.92e-07
mass:     box 23.0665, inflow 0.73687 per unit time, residual d/dt + net outflow: RMS 6.17e-05 (8.4e-05 of the inflow)
energy:   residual d/dt + net enthalpy outflow: RMS 1.71e-04 (9.2e-05 of the inflow), net outflow alone 8.5e-04
momentum: mean Cd 1.3658 from the box against 1.3691 on the wall; Cl amplitude 0.3715 against 0.3336
Strouhal: 0.1654 from the lift's crossings, 0.1651 from the wake spectrum
```

**The wall.** The integral of the wall probe's tractions agrees with the force of `[[forces]]` to 4e-7 (in C_D). The wall probe's samples also give the force's distribution around the
cylinder, which `[[forces]]` does not.

**Mass.** Over the box, dM/dt plus the net mass flux out through the four sides should vanish. The
residual is 8e-5 of the mass flow entering through `west`: the sides are cut exactly and integrated
exactly for the reconstruction polynomials, and what remains is the difference between those point values
and the cell averages the scheme conserves, and the time derivative taken from samples 20 steps apart.

**Energy.** The box holds no heat source and its wall is adiabatic, so d/dt of its total energy ∫ρE plus
the total enthalpy flowing out vanishes at every instant: the residual is 9e-5 of the enthalpy flowing in,
nine times smaller than the net outflow alone, which the unsteady term balances. What remains is the same
as for the mass, plus the heat conduction and viscous work through the sides, which the flux variables
leave out.

**Momentum and drag.** The force of the fluid on the cylinder is minus the momentum flowing out of the
box, pressure included, minus the rate of change of the momentum inside. Measured this way, far from the
wall, the mean drag coefficient is 1.366 against 1.369 on the wall itself. The flux variables leave out the
viscous stress on the box's sides, small this far from the wall: the two agree within 0.3%. The fluctuating lift is 11% larger
from the box than on the wall: the rate of change of the box's momentum, a time derivative of samples 20
steps apart, is the noisy term.

![The mass residual of the box over the mass flow into it, and the drag coefficient from the wall and from the box's momentum balance](../images/tutorials/measuring_balance-light.svg#gh-light-mode-only)
![The mass residual of the box over the mass flow into it, and the drag coefficient from the wall and from the box's momentum balance](../images/tutorials/measuring_balance-dark.svg#gh-dark-mode-only)

**The shedding frequency.** The upward zero crossings of the lift after t = 60 give St = 0.1654, and the
peak of the spectrum of U_Y at the wake point gives 0.1651. The spectrum is of a probe's time series
(Hann window, peak located by a parabola through the three highest values): with four and a half periods
its resolution is 0.036 in St, so the parabola does the locating. Its harmonics, at 2, 3, 4 and 5 times
the shedding frequency, are the wake's departure from a pure sine. `[statistics.spectra]` computes
spectra in space, along the periodic directions of a generated mesh, which this wake has none of.

![Power spectrum of the cross-stream velocity at a point of the wake, peaking at the Strouhal number](../images/tutorials/measuring_spectrum-light.svg#gh-light-mode-only)
![Power spectrum of the cross-stream velocity at a point of the wake, peaking at the Strouhal number](../images/tutorials/measuring_spectrum-dark.svg#gh-dark-mode-only)

**The cycle.** The box snapshot at t = 200 holds the 8 phase averages; four of them, a quarter period
apart, show the alternating vortices of U_Y forming behind the cylinder and moving downstream. Each bin
averages its part of each of the four cycles; a longer run smooths them. In ParaView, open
`solut/box.vtkhdf.series` and color by `PHASE_<bin>_U_Y`, or by `MEAN_U_X` for the mean recirculation.

![Phase-averaged cross-stream velocity in the box at four phases of the shedding cycle](../images/tutorials/measuring_phase-light.svg#gh-light-mode-only)
![Phase-averaged cross-stream velocity in the box at four phases of the shedding cycle](../images/tutorials/measuring_phase-dark.svg#gh-dark-mode-only)

## 7. Things to try

- **Move the box.** Make it larger (`size = [12.0, 8.0]`, and the sides with it): the mass residual stays
  small, while the drag from the momentum balance changes as the neglected viscous stresses on the sides
  and the wake's pressure recovery change.
- **More bins.** `bins = 16` resolves the cycle more finely, at the price of fewer samples per bin.
- **A rake.** A `line` probe across the wake at x = 3 (`reductions = ["mean", "rms"]`) gives the wake's
  mean velocity deficit and its fluctuations, the way a hot-wire rake would.

Next: [Supersonic flow over a cone](cone.md), in 3D.
