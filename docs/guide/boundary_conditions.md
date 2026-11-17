# Boundary conditions

Every boundary face gets its condition from a `[[boundaries]]` entry naming its
zone (see [Meshes](meshes.md) for where zones come from). Conditions are
imposed weakly: each one supplies an exterior state, and the Riemann solver
computes the face flux from it and the reconstructed interior state.

![A boundary face between the boundary cell's reconstructed state and the exterior state of its condition](../images/guide/boundary_state-light.svg#gh-light-mode-only)
![A boundary face between the boundary cell's reconstructed state and the exterior state of its condition](../images/guide/boundary_state-dark.svg#gh-dark-mode-only)

```toml
[[boundaries]]
name = "left"           # the zone
type = "inlet_state"    # the condition: its kind, then what it prescribes
u = [1.0, 0.0]
p = 1.0
T = 1.0

[[boundaries]]
name = "right"
type = "outlet_pressure"
p = 1.0
nscbc = { L = 1.0 }     # an option: non-reflecting
```

*Used in:*
[`sod`](https://matthewbonanni.github.io/mallard/docs/examples/#sod),
[`shu_osher`](https://matthewbonanni.github.io/mallard/docs/examples/#shu-osher),
[`riemann_2d`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d),
[`riemann_2d_quads`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d-quads),
[`double_mach`](https://matthewbonanni.github.io/mallard/docs/examples/#double-mach)
and 47 more.

The `type` names the condition: its kind first (`wall_`, `inlet_`,
`outlet_`), then what it prescribes. It is always explicit, never inferred
from the keys present. Each type requires its keys and rejects keys that
don't apply to it, with an error naming the type that takes them:

```text
line 14: boundaries[name = "wall"].q: `q` doesn't apply to `wall_isothermal`. Use `wall_heat_flux`, or remove q.
See docs/guide/boundary_conditions.md#wall_isothermal.
```

*Options* are keys that add a treatment to a type (`nscbc`, `average`,
`turbulence`, `wall_model`); each type takes the options of its row in the
[support matrix](#support-matrix).

Rules:

- Each boundary face must be assigned exactly once. A zone the mesh does not
  have is an error that lists the mesh's zones, and so are a face assigned
  twice and an entry that selects no faces.
- Periodic zones are interior faces and take no entry; a fully periodic mesh
  needs no `[[boundaries]]` ([Periodic](#periodic)).
- `where = "<expression in x, y, z>"` splits a zone between entries: each takes
  the faces whose centers satisfy its expression.

![The double Mach reflection: the bottom zone split at x = 1/6 between an inflow and a symmetry condition](../images/guide/domain_double_mach-light.svg#gh-light-mode-only)
![The double Mach reflection: the bottom zone split at x = 1/6 between an inflow and a symmetry condition](../images/guide/domain_double_mach-dark.svg#gh-dark-mode-only)

*`examples/double_mach`: the `bottom` zone is post-shock inflow for x < 1/6
and a reflecting wall beyond (`where`), and `top` follows the exact moving
shock (`dirichlet` with expressions of `t`).*

```toml
[[boundaries]]
name = "bottom"
type = "dirichlet"
where = "x < 1/6"
rho = "8.0"
u = ["7.1447096811216", "-4.125"]
p = "116.5"

[[boundaries]]
name = "bottom"
type = "symmetry"
where = "x >= 1/6"
```

## Available boundary conditions

| `type` | Kind | Requires | Options | Use |
|---|---|---|---|---|
| [`wall_adiabatic`](#wall_adiabatic) | wall | — | `wall_model` | No-slip (`navier_stokes`) or slip (`euler`) wall without heat flux; `u` moves the wall |
| [`wall_isothermal`](#wall_isothermal) | wall | `T` | `wall_model` | Wall at temperature `T` |
| [`wall_heat_flux`](#wall_heat_flux) | wall | `q` | `wall_model` | Wall with heat flux `q` into the fluid |
| [`inlet_state`](#inlet_state) | inlet | `u`, `p`, `T` | `nscbc`, `turbulence` (with `nscbc`), composition | The full inflow state: exact for supersonic inflow, over-specified for subsonic inflow unless `nscbc` |
| [`inlet_stagnation`](#inlet_stagnation) | inlet | `p0`, `T0` | composition | Subsonic inflow from a plenum: the mass flow follows from the flow |
| [`inlet_mass_flow`](#inlet_mass_flow) | inlet | `mass_flux` or `mdot`, `T` | composition | Subsonic inflow at a fixed mass flow, whatever the inlet density |
| [`outlet_pressure`](#outlet_pressure) | outlet | `p` | `average` or `nscbc` | Subsonic outlet at pressure `p` |
| [`outlet_extrapolation`](#outlet_extrapolation) | outlet | — | — | Transmissive: supersonic outflow, or waves leaving a box |
| [`farfield`](#farfield) | far field | `u`, `p`, `T` | — | Free stream around a body: inflow, outflow and tangential flow |
| [`symmetry`](#symmetry) | symmetry | — | — | Slip wall, symmetry plane, axis of axisymmetric runs |
| [`dirichlet`](#dirichlet) | prescribed | `rho`, `u`, `p` | — | Prescribed exterior state, e.g. an exact solution |
| [Periodic](#periodic) | periodic | set with `[[periodic]]`, not `[[boundaries]]` | — | Two zones joined into interior faces |

*Composition* is `X` or `Y` (and `balance`) for gas mixtures. Mixtures run
every type except `farfield`, `dirichlet` and `outlet_pressure` with
`average`; their walls are non-catalytic (no species crosses them). Further
keys of a type (`u` and `direction` of walls, `direction` of inlets,
`profile`) are in its section.

### Support matrix

Which options each type takes, as the parser checks them (a test compares
this table with the parser's, and the notes match its errors). *u field*: `u`
may be an expression of x, y, z, t, as a vector or a speed along `direction`
([values](#values-numbers-expressions-and-vectors)).

| `type` | `nscbc` | `average` | `turbulence` | `wall_model` | composition | u field | Not supported (yet) |
|---|---|---|---|---|---|---|---|
| `wall_adiabatic` | — | — | — | ✓ | — | ✓ | `nscbc` and transpiration (blowing, suction) on walls: [#565](https://github.com/MatthewBonanni/mallard/issues/565) |
| `wall_isothermal` | — | — | — | ✓ | — | ✓ | as `wall_adiabatic` |
| `wall_heat_flux` | — | — | — | ✓ | — | ✓ | as `wall_adiabatic` |
| `inlet_state` | ✓ | — | ✓ | — | ✓ | ✓ | `turbulence` without `nscbc`: [#565](https://github.com/MatthewBonanni/mallard/issues/565) |
| `inlet_stagnation` | — | — | — | — | ✓ | — | `nscbc`, `turbulence`: [#565](https://github.com/MatthewBonanni/mallard/issues/565) |
| `inlet_mass_flow` | — | — | — | — | ✓ | — | `nscbc`, `turbulence`: [#565](https://github.com/MatthewBonanni/mallard/issues/565) |
| `outlet_pressure` | ✓ | ✓ | — | — | — | — | `average` with `nscbc`, backflow, convective outflow: [#565](https://github.com/MatthewBonanni/mallard/issues/565) |
| `outlet_extrapolation` | — | — | — | — | — | — | |
| `farfield` | — | — | — | — | — | ✓ | gas mixtures (composition): [#565](https://github.com/MatthewBonanni/mallard/issues/565) |
| `symmetry` | — | — | — | — | — | — | |
| `dirichlet` | — | — | — | — | — | ✓ | |

A mass-flow outlet (`outlet_mass_flow`, for bleeds and duct splits) is
planned as a type of its own in [#565](https://github.com/MatthewBonanni/mallard/issues/565).

Choosing:

- **External aerodynamics:** `farfield` on the outer boundary, a `wall_*` on
  the body (`examples/cylinder`, `sphere_re300`).
- **Ducts and combustors:** `inlet_state` and `outlet_pressure` with `nscbc`
  keep acoustic waves from reflecting into the domain (`flame_outflow`,
  `volvo_bluff_body`); without `nscbc` they are simpler but reflect them (97–99%
  of a pulse at `outlet_pressure`, about 1% with `nscbc`).
- **Nozzles, ducts and intakes fed from a reservoir:** `inlet_stagnation`, so
  that the mass flow (and choking) is part of the solution; `inlet_mass_flow`
  where the mass flow is given (combustors, jets).
- **Supersonic inflow:** `inlet_state`. **Supersonic outflow:**
  `outlet_extrapolation`, or `outlet_pressure` with `average = true` if part
  of the outlet is subsonic.
- **Exact or manufactured solutions, moving shocks:** `dirichlet`.

![The wedge example: inflow, a pressure outlet and symmetry planes over the ramp](../images/guide/domain_wedge-light.svg#gh-light-mode-only)
![The wedge example: inflow, a pressure outlet and symmetry planes over the ramp](../images/guide/domain_wedge-dark.svg#gh-dark-mode-only)

![The viscous shock tube: a closed box of adiabatic walls with a symmetry plane on top](../images/guide/domain_viscous_shock_tube-light.svg#gh-light-mode-only)
![The viscous shock tube: a closed box of adiabatic walls with a symmetry plane on top](../images/guide/domain_viscous_shock_tube-dark.svg#gh-dark-mode-only)

*`examples/wedge` (Mach 1.76 over an 8° ramp) and
`examples/viscous_shock_tube` (shock / boundary-layer interaction in a closed
box), drawn on their generated meshes.*

## Values: numbers, expressions and vectors

Every prescribed number may be a number (SI units), a quantity with a unit
(`"300 K"`, `"1 bar"`) or an expression of `x`, `y`, `z` and `t` (`"300 + 50 *
y"`, in SI units): wall temperatures and heat fluxes, outlet pressures, inflow
and stagnation states, mass fluxes, wall velocities, far-field states and
compositions. Expressions also read the boundary's
[local coordinates](#local-coordinates): `u` and `v` from 0 to 1 across the
boundary whatever its shape, and the wall distance `n`; and the columns of
[profiles from data](#profiles-from-data).

Vectors (`u` of `inlet_state`, `farfield` and walls; `mass_flux` of
`inlet_mass_flow`) take either form:

```toml
u = [10.0, "2 * sin(2 * pi * 50 * t)"]   # a vector: each component a number or an expression
```

```toml
u = 10.0                 # a magnitude (a number or an expression)...
direction = [1.0, 0.2]   # ...along a direction (normalized); default: each face's inward normal
```

A vector together with `direction` is an error. A wall speed needs its
`direction` (the inward-normal default would be blowing, planned in
[#565](https://github.com/MatthewBonanni/mallard/issues/565)), and `farfield`
takes a vector only. A `mass_flux` vector `m` flows along `m / |m|` with the
flux `m . n_in` at each face; `mdot` stays a scalar total.

`direction` (of a speed `u`, and the flow direction of `inlet_stagnation`
and `inlet_mass_flow`) may have expressions of `x`, `y`, `z`, `t` among its
components. It is normalized at each face, and at inlets it must point into
the domain at every face and time: at setup the run stops with the first
face where it does not, and a direction of `t` that turns outward or to zero
later stops the run at the next progress check with the face, time and
value. A swirling inflow about the axis of a circular inlet in the x = 0
plane, with a swirl angle of 30°:

```toml
[[boundaries]]
name = "inlet"
type = "inlet_mass_flow"
mass_flux = 50.0
T = 300.0
direction = [1.0, "-0.577 * z / sqrt(y * y + z * z + 1e-12)", "0.577 * y / sqrt(y * y + z * z + 1e-12)"]
```

### Time-dependent conditions

Expressions of `x`, `y`, `z` are evaluated once at each face center, so each
face gets a condition of its own (`dirichlet` states too); those of `t` on
the device at every Runge-Kutta stage, inside the step's CUDA graph, so a pulsed or acoustically forced inlet converges at the
time integrator's order (third with SSPRK3; evaluating them once per step
would give first order). Inputs that do not depend on `t` cost nothing per
stage. A number of `t` that leaves its range (`p`, `T`, `p0`, `T0` or a wall
`T` not positive, `mass_flux` negative, a `mass_flux` vector out of the
domain) stops the run at the next progress check with the face, time and
value.

```toml
[[boundaries]]
name = "inlet"
type = "inlet_state"
u = ["50 * (1 + 0.1 * sin(2 * pi * 1000 * t))", "0"]   # 10% at 1 kHz
T = "300 * (1 + 0.05 * sin(2 * pi * 250 * t))"
p = 1.0e5
nscbc = { L = 1.0, sigma = 100.0 }   # follow the forcing closely
```

![A forced characteristic inlet: velocity forcing leaves as acoustic waves, temperature forcing as entropy waves](../images/guide/forced_inlet-light.svg#gh-light-mode-only)
![A forced characteristic inlet: velocity forcing leaves as acoustic waves, temperature forcing as entropy waves](../images/guide/forced_inlet-dark.svg#gh-dark-mode-only)

*`examples/forced_inlet`: velocity forcing at 1 kHz leaves the inlet as
acoustic waves at `u + c`, temperature forcing at 250 Hz as entropy waves at
`u`; the inlet cell follows both targets (`examples/forced_inlet/plot_forced_inlet.py`).*

An expression of `t` pulses a jet: `where` splits the wall into a nozzle,
an `inlet_mass_flow` whose mass flux is a smoothed square wave in time, and an
adiabatic wall around it. The inlet holds the mass flux, so each pulse ejects
exactly the stroke the expression integrates to.

```toml
[[boundaries]]
name = "left"                 # the wall, with a nozzle of diameter 1 in it
type = "inlet_mass_flow"
where = "y < 0.5"
# rho0 U(t): U0 = 0.2 from rest, period 30 (stroke L / D = 3)
mass_flux = "0.2 * (1 - tanh(4 * cos(2 * pi * t / 30))) / 2"
T = 0.7142857142857143

[[boundaries]]
name = "left"
type = "wall_adiabatic"
where = "y >= 0.5"
```

![A pulsed jet launching vortex rings: vorticity, ring core positions and the inlet velocity against its target](../images/guide/pulsed_jet-light.svg#gh-light-mode-only)
![A pulsed jet launching vortex rings: vorticity, ring core positions and the inlet velocity against its target](../images/guide/pulsed_jet-dark.svg#gh-dark-mode-only)

*`examples/pulsed_jet` (axisymmetric, D / 96, Re_D = 2000, six pulses to
t U0 / D = 36): each pulse rolls up into a vortex ring. The inlet cell delivers
strokes of 2.995 to 3.000 D against 3. The first ring travels at 0.40 U0 with
1.06 times the circulation of the slug model; the rings behind it, in its wake,
at 0.454 to 0.463 U0 with 1.02 to 1.12 times (`examples/pulsed_jet/plot_pulsed_jet.py`).*

Pitfalls:

- With `nscbc`, an inlet relaxes its velocity toward the target at the rate
  `K = sigma c (1 - M^2) / L`. Forcing faster than `K` reaches the domain
  attenuated (to 56% at 1 kHz with the default `sigma = 0.25` here); raise
  `sigma` to follow it, at the price of reflecting outgoing waves of lower
  frequency. `inlet_state` without `nscbc` imposes the forcing exactly but
  reflects everything.
- With `nscbc`, the boundary point advances once per step, so its forcing
  converges at first order in `dt`; the other inlets at the integrator's
  order.
- With `average = true`, `p` is one target for the whole outlet: it may
  depend on `t`, not on `x`, `y`, `z`.

## Local coordinates

Expressions of a `[[boundaries]]` entry also read coordinates that belong to
the boundary itself: `u` and `v` run from 0 to 1 across the entry's faces,
whatever their shape, size and place, and `n` from 0 on the walls to 1 at the
point farthest from them. A profile written in them needs no dimensions and
no axes, and fits the same inlet on any mesh:

| For | Write |
|---|---|
| A rectangular duct of any size, anywhere: zero on its four sides, `U` at the middle | `u = "U * 16 * u * (1 - u) * v * (1 - v)"` |
| A pipe of any radius: Poiseuille flow of bulk velocity 10 m/s (`r` is 0 at the center, 1 at the wall) | `u = "20 * (1 - r^2)"` |
| A turbulent profile on any duct or pipe: a 1/7 power law in the wall distance | `u = "30 * n^(1/7)"` |
| The same with a boundary layer 0.01 m thick, in meters from the wall | `u = "30 * min(n_m / 0.01, 1)^(1/7)"` |
| Eight hot streaks around an annular inlet | `T = "300 + 50 * sin(8 * theta)"` |
| 2D ([`channel_inlet`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-inlet)): plane Poiseuille flow along an inlet of any height | `profile = "u * (1 - u)"` |

(`u = "..."` on the left is the entry's velocity key, a speed along the
inward normal; the `u` inside the expression is the coordinate. Replace `U`
by a number.)

Each coordinate comes as a fraction and as a length. The fraction has the
bare name; its twin in meters, measured along the boundary, ends in `_m`:

| Fraction | In meters | Runs | Meaning |
|---|---|---|---|
| `u` | `u_m` | 2D, 3D | Along the boundary: from one end of a 2D boundary to the other, or from one side of a four-sided patch to the opposite one; around a ring or a disk |
| `v` | `v_m` | 3D | Across the patch, from the side `v = 0` to the opposite one; from one rim of a ring to the other; from the center of a disk to its edge |
| `n` | `n_m` | 2D, 3D | Distance to the nearest wall face; `n` is `n_m` over its largest value on the boundary |
| `r` | `r_m` | 3D, rings and disks | Another name for `v`, `v_m`: the radius of a disk, 0 at its center and 1 at its edge |
| `theta` | | 3D, rings and disks | The angle around: `2 * pi * u`, in [0, 2 pi) |

![The inlet of a rectangular duct in its 3D mesh, colored by u with lines of constant u and v, and one face's u, v and wall distance n](../images/guide/boundary_patch-light.svg#gh-light-mode-only)
![The inlet of a rectangular duct in its 3D mesh, colored by u with lines of constant u and v, and one face's u, v and wall distance n](../images/guide/boundary_patch-dark.svg#gh-dark-mode-only)

*The inlet of a duct in its mesh: `u` and `v` across the inlet's faces, and
the wall distance `n` of one face (`tools/docs/boundary_coordinates.py`).*

**The patch.** The faces of an entry (after `where`) are its patch, and its
outline decides what `u` and `v` are. The run log names the patch and shows
the range of every coordinate the entry reads, as `local coordinates:
four-sided patch, (u, v) = (0, 0) at (0, 0, 0), (1, 0) at (0, 0.2, 0), ...; u
0.03125 to 0.96875`, so a coordinate that runs the wrong way is visible
before the run starts.

| Patch | Outline | `u` | `v` |
|---|---|---|---|
| Four-sided | One loop with four corners (where it turns by more than 30 degrees): a rectangle, a parallelogram, a sector, a curved or warped panel | 0 on one side, 1 on the opposite one | 0 on a third side, 1 on the fourth |
| Ring | Two loops: an annulus, the side of a pipe, of a cone or of a nozzle | Around the ring, in [0, 1) | 0 on one rim, 1 on the other |
| Disk | One loop without corners: a circle, an ellipse, a rounded slot | Around the center, in [0, 1) | 0 at the center, 1 on the edge |
| Curve (2D) | A 2D boundary: one open curve | Arc length over the boundary's length | |

![u and v over a rectangle, a parallelogram, a warped four-sided patch, a circle, an annulus, the side of a cone and the side of a pipe, as face colors and contour lines](../images/guide/boundary_coordinates-light.svg#gh-light-mode-only)
![u and v over a rectangle, a parallelogram, a warped four-sided patch, a circle, an annulus, the side of a cone and the side of a pipe, as face colors and contour lines](../images/guide/boundary_coordinates-dark.svg#gh-dark-mode-only)

*`u` (top) and `v` (bottom) over seven patches: each face in the color of
its value, the lines those of the coordinate between the faces
(`tools/docs/boundary_coordinates.py`, which checks the ranges against
Mallard's own on the same meshes).*

`u` and `v` are smooth everywhere on the patch and follow its sides: on a
rectangle or a parallelogram they are the fractions of the way across, on a
circle the radius over the circle's and the angle, and on any other shape the
coordinates that Laplace's equation on the patch's own surface gives
([design note](../design/patch_coordinates.md)). They are computed once at
setup from all of the entry's faces, the same on any number of ranks.

**Where `u` starts and which way it grows.** Two optional points say it,
the same way on every patch: `origin`, where `u` (and `v`) is 0, and
`toward`, where `u` grows to. Neither has to lie exactly on the patch.

```toml
# Four-sided: (u, v) = (0, 0) at the corner nearest to origin; u runs along the
# side to the next corner nearest to toward, v along the other side
[[boundaries]]
name = "panel"
type = "wall_isothermal"
T = "300 + 200 * u"
origin = [0.0, 0.0, 0.0]
toward = [0.0, 0.3, 0.0]

# Ring or disk: u (and theta) is 0 at the face nearest to origin, and grows
# around the way that reaches toward sooner
[[boundaries]]
name = "annular_inlet"
type = "inlet_state"
u = 30.0
p = 1.0e5
T = "300 + 50 * cos(8 * theta)"   # eight streaks, the first at the origin
origin = [0.0, 0.0, 0.12]
toward = [0.0, 0.12, 0.0]
```

In 2D `origin = [x, y]` picks the end of the boundary `u` starts from.
`toward` is optional: without it a four-sided patch keeps the side `u` runs
along by default, starting at the origin's corner, and a ring or a disk its
default direction.

The points only have to say which: the nearest corner, face or end is taken
however far the point is, so `origin = [0, -1, 0]` is a fine way to say "the
corner on the -y side". The run log shows what was picked, as `(u, v) = (0,
0) at (0, 0, 0), (1, 0) at (0, 0.3, 0)` or `u = 0 at (0.07, 0, 0.11), growing
toward (0, 0.12, 0.05)`. A point that does not decide is an error listing
the candidates: one as near to two corners (or ends) as each other, on the
axis of a ring or at the center of a disk, or a `toward` at the origin or
half way around from it. "As near" is within 1% of the patch's size (the
square root of its area; the length of a 2D boundary).

Without either:

- **Four-sided.** `u` grows along the coordinate axis after the one the patch
  faces, and `v` along the next (x to y to z to x): on an inlet across x, `u`
  runs along +y and `v` along +z; across y, `u` along +z and `v` along +x;
  across z, `u` along +x and `v` along +y. A patch across no coordinate axis
  counts as across the one its mean normal is nearest to (of two as near,
  the first), and `u` runs between the pair of opposite sides most nearly
  along the next axis as seen across the normal.
- **Ring.** `v` is 0 on the shorter rim (the inner circle of an annulus, the
  narrow end of a cone) and, between rims as long as each other (a pipe), on
  the lower one along the ring's axis. `u` is 0 on the side of the coordinate
  axis after the one the ring is around, and grows toward the next (x to y
  to z to x): around z, `theta` is the angle from +x toward +y; around x,
  from +y toward +z. A ring around no coordinate axis counts as around the
  one its own axis is nearest to (of two as near, the first), and `u` is 0
  toward the next axis as seen across the ring's own.
- **Disk.** `v` is 0 at the center of the faces' area; `u` as on a ring.
- **Curve (2D).** `u` is 0 at the end with the lower coordinate along the axis
  the boundary mostly runs along (the lower y on an inlet across x).

`1 - u` in an expression also turns a coordinate around.

**Corners.** A four-sided patch is found by the four corners of its outline.
For an outline with no clear corners or more than four, `corners` lists the
four to use, in the order `(u, v)` = (0, 0), (1, 0), (1, 1), (0, 1) (each is
moved to the nearest node of the outline; `origin` and `toward` are then not
needed). `corners = []` treats a patch with corners as a disk.

```toml
[[boundaries]]
name = "curved_panel"
type = "wall_isothermal"
T = "300 + 200 * u"   # from the side through the first and last corner to the opposite one
corners = [[0, 0, 0], [0.4, 0, 0.1], [0.4, 0.3, 0.1], [0, 0.3, 0]]
```

**Lengths.** `v_m` is the distance along the patch's surface from where `v`
is 0 (a side, a rim or the center), in meters, and `u_m` that from the side
`u = 0` of a four-sided patch; around a ring or a disk `u_m` is the arc
length from where `u` is 0, on the circle through the face. On a rectangle
they are the distances from two of its sides, on a pipe's wall the height and
the arc length, on a disk `r_m` is the radius. In 2D `u_m` is the arc length
along the boundary.

**The wall distance** `n_m` is the exact distance from the face center to the
nearest wall face: to segments in 2D and to the faces' triangles in 3D, so
it holds on curved and slanted walls, in corners (a rectangular duct's
inlet gets the distance to its nearest side) and on inlets that do not touch
a wall. By default the walls are the faces of every `wall_adiabatic`,
`wall_isothermal` and `wall_heat_flux` entry; `walls = ["bottom"]` names the
zones instead, of any type (a symmetry plane too). A curved wall counts by
its straight faces. `n` divides it by its largest value over the boundary's
faces and nodes (the half height of a channel, the radius of a pipe).

![A 2D inlet on the outer arc of half a ring: u along the arc from 0 to 1, and n to the nearer straight wall](../images/guide/boundary_arc_length-light.svg#gh-light-mode-only)
![A 2D inlet on the outer arc of half a ring: u along the arc from 0 to 1, and n to the nearer straight wall](../images/guide/boundary_arc_length-dark.svg#gh-dark-mode-only)

*A curved 2D inlet: `u` follows the arc from one end to the other, `n` is 1
where the walls are farthest (`tools/docs/boundary_coordinates.py`).*

The coordinates are available in every value of the entry (`u`,
`direction`, `p`, `T`, `p0`, `T0`, `mass_flux`, `profile`, `q`, `X` and `Y`,
`dirichlet` states and the `turbulence` table's `reynolds_stress`), alone or
with `x`, `y`, `z` and `t`: `"30 * n^(1/7) * (1 + 0.05 * sin(2 * pi * 100 *
t))"` pulses the profile. `where` selects faces by `x`, `y`, `z` only (the
patch is made of the faces it selects), and `mdot` and an `average` outlet
pressure, totals over the boundary, read neither.

A boundary layer of thickness 0.01 m on every wall an inlet touches, a fully
developed pipe flow of bulk velocity 10 m/s on a circular inlet (3D),
wherever the pipe is and however it is turned, and a mass-flow inlet shaped
like a boundary layer:

```toml
[[boundaries]]
name = "inlet"
type = "inlet_state"
u = ["30 * min(n_m / 0.01, 1)^(1/7)", "0"]
p = 1.0e5
T = 300.0

[[boundaries]]
name = "pipe_inlet"
type = "inlet_state"
u = "20 * (1 - r^2)"  # a speed along the inward normal
p = 1.0e5
T = 300.0

[[boundaries]]
name = "slot"
type = "inlet_mass_flow"
mdot = 0.05
profile = "min(n_m / 0.002, 1)^(1/7)"  # rescaled so that the faces carry mdot
T = 300.0
walls = ["slot_wall"]
```

| Key | Default | Meaning |
|---|---|---|
| `corners` | found from the patch's outline | 3D: the four corners of the patch [m], where `(u, v)` is (0, 0), (1, 0), (1, 1), (0, 1); `[]` for a disk |
| `origin` | from the axes, as above | A point [m]: `u` (and `v`) is 0 at the corner (four-sided), the face (ring, disk) or the end (2D) nearest to it |
| `toward` | from the axes, as above | 3D, with `origin`: a point [m] that `u` grows toward: the next corner nearest to it, or the way around that reaches it sooner |
| `walls` | every wall entry's faces | Zones `n` and `n_m` measure the distance from |

Pitfalls:

- The coordinates are computed once at setup, at the face centers, from the
  mesh the run starts with; a profile is sampled at the face centers, so a
  boundary layer needs faces finer than its thickness to be resolved.
- `u` and `v` belong to one patch: an entry whose faces form separate pieces,
  a closed surface, or a patch with more than one hole is an error naming
  the entry. Split it with `where`, one entry per patch.
- An outline with corners other than four (an L, a triangle, a half disk, a
  staircase cut by `where`) is an error listing the corners found: give the
  four to use in `corners`, or `corners = []`.
- A circle of fewer than about 12 faces around turns by more than 30 degrees
  at every node, so its nodes count as corners: give `corners = []`.
- `r`, `r_m` and `theta` exist on rings and disks; on a four-sided patch they
  are an error (use `u`, `v` or `n`). On a ring `r` is 0 on the rim `v = 0`,
  not on the axis, and `v` varies as the logarithm of the radius across a
  flat annulus (`r_m` is the distance from the inner rim in meters).
- The center of a disk, where `v` is 0, is the center of the faces' area to
  within a face, and the lengths `u_m`, `v_m` are accurate to first order in
  the face size (`u`, `v` to second).
- `n` reaches 1 only if a face center or node lies at the point farthest
  from the walls; otherwise it is `n_m` over the largest distance the mesh
  holds.
- A local variable named like a coordinate (`var r := sqrt(x^2 + y^2)`)
  hides it inside that expression.
- On a wall entry that reads `n`, the default walls include the entry
  itself, so `n_m` is 0 on every face, which is an error: name the other walls
  in `walls`. `walls`, `corners`, `origin` and `toward` are errors on an entry
  whose expressions don't read the coordinates they set.
- Around a ring or a disk `u` jumps from 1 back to 0 where it starts: write
  profiles around it in `sin` and `cos` of `theta`, which do not.
- 2D runs have `u`, `n` and their lengths only: the error says so.

## Profiles from data

A `[[boundaries.data]]` table after a `[[boundaries]]` entry reads a profile
from a CSV file, in the format and with the keys of
[`[[initialize.data]]`](initialization.md#tabulated-data-and-point-clouds),
and gives every column other than the coordinates to the entry's expressions
as a variable `<name>_<column>`. Its `at` expressions read `x`, `y`, `z` and
the [local coordinates](#local-coordinates), so one measured or precomputed
boundary-layer profile fits every wall-bounded inlet, wherever it is and
however it is turned:

```toml
[[boundaries]]
name = "inlet"
type = "inlet_state"
u = "bl_u * (1 + 0.05 * sin(2 * pi * 100 * t))"  # a speed along the inward normal
p = 1.0e5
T = "bl_T"

[[boundaries.data]]
name = "bl"
file = "boundary_layer.csv"  # y,u,T: wall distance in mm
at = ["n_m"]                 # the distance to the nearest wall, in meters
units = { y = "mm" }
interpolation = "cubic"
```

A coordinate in meters (`n_m`, `u_m`, `v_m`) keeps the table's own lengths,
whatever the inlet's size; a normalized one (`n`, `u`, `v`, from 0 to 1)
stretches the table over the boundary, so `at = ["u"]` with a table from 0 to
1 fits a profile across an inlet of any width.

The data are read once at setup, at each face center, and stored with the
face like its coordinates: an expression of the data and `t` is evaluated on
the device at the stage times like any other, with no work on the host and
the same bits on every rank count. A table holds its end values beyond its
range (a boundary-layer table up to its edge gives the free stream above
it). The run log shows, per entry, each coordinate's range at the faces, how
many faces fall outside the table, and each variable's range, e.g. `data bl
(boundary_layer.csv: table 41, cubic; y = n_m 0.000625 to 0.05, 24 faces
outside the table's 0 to 0.02 take its end values): bl_u 0 to 30, bl_T 300 to
340`, and the run block records the file's hash.

| Key | Default | Meaning |
|---|---|---|
| `name` | required | Prefix of the variables: `<name>_<column>` |
| `file` | required | CSV file, relative to where Mallard runs |
| `type` | `"table"` | `table` (on a grid of 1-3 coordinates, interpolated) or `points` (scattered, by nearest points) |
| `coordinates` | first column (`table`); `x`, `y` (`z`) (`points`) | Names of the coordinate columns |
| `at` | the coordinates, if named `x`, `y`, `z` or like a local coordinate | One expression of `x`, `y`, `z` and the local coordinates per coordinate (not of `t`) |
| `interpolation` | `"linear"` | Tables: `linear` or `cubic` |
| `neighbors` | 1 | Points: how many nearest points to weight (1-16) |
| `units` | SI | Units of columns, e.g. `{y = "mm", T = "degC"}` |

Pitfalls:

- An entry's data have at most 16 value columns in all; data that none of
  the entry's expressions read are an error.
- `at` may not read `t`: read the data at a fixed place and combine them with
  `t` in the entry's expressions.
- `where`, `mdot` and an `average` outlet pressure do not read the data.

## Walls

Walls are no-slip under `navier_stokes` and slip under `euler`. Every wall
may move, may take its own wall model, and is non-catalytic for gas mixtures;
the three types differ in what they impose on the energy equation.

| Key | Default | Meaning |
|---|---|---|
| `u` | at rest | Wall velocity [m/s]: a vector, or a speed along `direction`; numbers or expressions of x, y, z, t (a rotating wall: `u = ["-omega * y", "omega * x"]`) |
| `direction` | — | With a speed `u`: the direction the wall moves in (numbers or expressions of x, y, z, t) |
| `wall_model` | none | A [wall model](#wall-models-wall_model) for this wall, with its own model and parameters |

A wall moving along its own normal (blowing, suction) is not supported yet:
planned as `transpiration` in [#565](https://github.com/MatthewBonanni/mallard/issues/565).

### `wall_adiabatic`

```toml
[[boundaries]]
name = "wall"           # a boundary zone of the mesh
type = "wall_adiabatic"
```

*Used in:*
[`sphere_mach3`](https://matthewbonanni.github.io/mallard/docs/examples/#sphere-mach3),
[`ellipsoid`](https://matthewbonanni.github.io/mallard/docs/examples/#ellipsoid),
[`mallard`](https://matthewbonanni.github.io/mallard/docs/examples/#mallard),
[`heated_duct`](https://matthewbonanni.github.io/mallard/docs/examples/#heated-duct),
[`heated_wall`](https://matthewbonanni.github.io/mallard/docs/examples/#heated-wall)
and 7 more.

**Imposes** zero heat flux and, in viscous runs, the wall velocity (through a
mirrored ghost velocity, `2 u_w - u`); density and pressure come from the
interior. Inviscid runs make it a slip wall, as `symmetry`.

**Use** for walls whose heat loss does not matter or is unknown. A moving
lid:

```toml
[[boundaries]]
name = "top"
type = "wall_adiabatic"
u = [1.0, 0.0]
```

**Pitfall:** under `euler` a wall's `u` has no effect (slip).

### `wall_isothermal`

```toml
[[boundaries]]
name = "wall"
type = "wall_isothermal"
T = 300.0                   # [K]; or an expression: "300 + 200 * x"
```

*Used in:*
[`channel_inlet`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-inlet),
[`channel_inlet_data`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-inlet-data),
[`poiseuille_pipe`](https://matthewbonanni.github.io/mallard/docs/examples/#poiseuille-pipe),
[`channel_retau180`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-retau180),
[`channel_inflow_retau180`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-inflow-retau180)
and 2 more.

**Imposes** the wall temperature `T` (required) and the wall velocity as
`wall_adiabatic`: a single gas through the ghost density and the viscous
flux, a gas mixture through the diffusive fluxes alone. The pressure comes
from the interior.

**Use** for cooled or heated walls, and a cold wall that quenches a flame
(mixtures too).

**Pitfall:** the wall heat flux is conducted across the half cell next to the
wall, so it is only as good as the resolution there; a flame quenching
against a wall needs cells of a fraction of its quenching distance.

### `wall_heat_flux`

```toml
[[boundaries]]
name = "wall"
type = "wall_heat_flux"
q = 2.0e4                   # [W/m^2] into the fluid; negative cools it
```

*Used in:*
[`heated_wall`](https://matthewbonanni.github.io/mallard/docs/examples/#heated-wall).

**Imposes** the heat flux `q` into the fluid (required) in the energy
equation and the temperature gradient, and the wall velocity as
`wall_adiabatic`.

**Use** where the heat input is known (an electrically heated wall, a
radiative load).

*`examples/heated_wall`: a closed box heated through its floor with
`q = 2e4 (1 + 0.5 cos(2 pi x / 0.1))` W/m^2; the energy in the box grows by
the heat that entered, `q A t`, within 2e-14, and after 1 ms each stretch of
the floor has taken up its own share, the integral of `q` over it, within
0.3% (`check_heated_wall.py`).*

**Pitfall:** a large `q` on a coarse wall cell overshoots the wall
temperature; check the near-wall temperature against `q dy / lambda`.

### Wall models (`wall_model`)

A `wall_model` table on a wall gives that wall its shear stress and heat flux
from a model fed by the LES state at an exchange point a height `h` above each
wall face (wall-modeled LES, [LES guide](les.md#wall-models-wall_model)). Each
wall has its own: different walls of one run may use different models and
parameters.

```toml
[[boundaries]]
name = "bottom"
type = "wall_isothermal"
T = 1.0
wall_model = { model = "gfm", exchange_distance = 0.3, edge_distance = 1.0 }

[[boundaries]]
name = "top"
type = "wall_adiabatic"
wall_model = { model = "equilibrium", exchange_cells = 3 }
```

*Used in:*
[`channel_wmles`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-wmles).

| Key | Default | Meaning |
|---|---|---|
| `model` | `gfm` | `gfm`: Griffin, Fu & Moin (2023), for isothermal and adiabatic walls; `equilibrium`: the equilibrium ODE model, for every wall type |
| `exchange_distance` / `exchange_cells` | one required | `h`, absolute or in wall-cell heights |
| `edge_distance` | required by `gfm` | Where the edge velocity and temperature are sampled |
| `kappa`, `A_plus`, `Pr_t`, `s` | 0.41, 17, 0.9, 1.14 | Model constants |

**Not supported:** gas mixtures, axisymmetric runs. Pitfalls:
[LES guide](les.md#wall-models-wall_model); design:
[wall models](../design/wall_model.md).

## Inlets

Every inlet takes the composition of the gas it admits for mixtures: `X` or
`Y` by species (numbers or expressions), with an optional `balance` species,
as in [`[initialize]`](initialization.md).

### `inlet_state`

```toml
[[boundaries]]
name = "inlet"
type = "inlet_state"
u = [500.0, 0.0]            # or a speed along direction (default: the inward normal)
p = 1.0e5
T = 300.0
# X = { O2 = 0.21, N2 = 0.79 } # mixtures
```

*Used in:*
[`wedge`](https://matthewbonanni.github.io/mallard/docs/examples/#wedge),
[`sphere_mach3`](https://matthewbonanni.github.io/mallard/docs/examples/#sphere-mach3),
[`mallard`](https://matthewbonanni.github.io/mallard/docs/examples/#mallard),
[`heated_duct`](https://matthewbonanni.github.io/mallard/docs/examples/#heated-duct),
[`average_outlet`](https://matthewbonanni.github.io/mallard/docs/examples/#average-outlet)
and 14 more.

**Imposes** the full exterior state: velocity, pressure, temperature and
composition, nothing from the interior. For **supersonic inflow** this is
exactly right: all waves enter, and the upwind flux is that of the prescribed
state. For **subsonic inflow** it is over-specified: it also fixes the
pressure, which an outgoing acoustic wave should set, so it reflects acoustic
waves back into the domain. **Reversed flow** (flow leaving through it) still
sees the Riemann flux against the prescribed state; it is not an outlet.

| Key | Default | Meaning |
|---|---|---|
| `u` | required | Velocity [m/s]: a vector, or a speed along `direction` |
| `direction` | inward normal | With a speed `u`: numbers or expressions of x, y, z, t, into the domain at every face and time |
| `p`, `T` | required | Pressure and temperature |
| `X` / `Y`, `balance` | required for mixtures | Composition |
| `nscbc` | none | [Non-reflecting treatment](#nscbc-characteristic-boundaries) |
| `turbulence` | none | [Synthetic turbulence](#synthetic-turbulence-turbulence), with `nscbc` |

**Use** for supersonic inflow, and subsonic inflow where reflections do not
matter (short runs, inflow far from the region of interest). Otherwise add
`nscbc` (a partially non-reflecting inlet that relaxes toward this state), or
use `inlet_stagnation` (reservoir-fed) or `inlet_mass_flow` (given mass flow).

**Pitfall:** the mass flow follows the velocity, not the density: when heat
release raises the pressure, the inlet keeps its velocity and lets mass
through at the prescribed density. `inlet_mass_flow` holds the mass flow
([figure below](#inlet_mass_flow)).

**Not supported (yet):** `turbulence` without `nscbc`, planned in
[#565](https://github.com/MatthewBonanni/mallard/issues/565).

Design notes: [inlets](../design/inlets.md); theory:
[weak imposition](../theory/boundaries_sources.md#weak-imposition).

### `inlet_stagnation`

```toml
[[boundaries]]
name = "inlet"
type = "inlet_stagnation"
p0 = 2.0e5
T0 = 600.0
# direction = [1.0, 0.2]       # default: the inward normal of each face
# X = { O2 = 0.21, N2 = 0.79 } # mixtures
```

*Used in:*
[`nozzle`](https://matthewbonanni.github.io/mallard/docs/examples/#nozzle).

**Imposes** the stagnation pressure `p0` and temperature `T0`, the flow
direction and the composition: the domain is fed from a reservoir. The one
outgoing acoustic wave of subsonic inflow comes from the interior and sets
the speed, and the inflow state lies on the isentrope through (`T0`, `p0`).
Mixtures use their own `h(T)` and `s(T, p)`, not a constant `gamma`.

![The stagnation inlet: what it imposes and what it takes from the interior](../images/guide/inlet_stagnation-light.svg#gh-light-mode-only)
![The stagnation inlet: what it imposes and what it takes from the interior](../images/guide/inlet_stagnation-dark.svg#gh-dark-mode-only)

| Key | Default | Meaning |
|---|---|---|
| `p0`, `T0` | required | Stagnation pressure and temperature |
| `direction` | inward normal | Flow direction (numbers or expressions of x, y, z, t); must point into the domain at every face and time |
| `X` / `Y`, `balance` | required for mixtures | Composition |

![Nozzle: Mach number, pressure and stagnation pressure against the quasi-one-dimensional solution](../images/guide/nozzle_validation-light.svg#gh-light-mode-only)
![Nozzle: Mach number, pressure and stagnation pressure against the quasi-one-dimensional solution](../images/guide/nozzle_validation-dark.svg#gh-dark-mode-only)

*`examples/nozzle`: a converging-diverging nozzle fed at `p0`, `T0` with a
normal shock in the diverging part, against the quasi-one-dimensional
solution: choked mass flow within 0.2%, the shock within two cells.*

**Use** for nozzles, ducts and intakes fed from a plenum, where the mass flow
(and choking) is part of the solution.

Pitfalls:

- The inlet is for subsonic inflow. Supersonic inflow keeps the interior
  speed: start such runs from a field that is subsonic at the inlet, or use
  `inlet_state`.
- **Reversed flow:** where the interior pressure exceeds `p0` (a start-up
  transient, a back pressure above `p0`), the inlet sees the reservoir at rest
  and lets the flow out against `p0`; it does not hold a mass flow.
- It reflects acoustic waves like `inlet_state` (the mass flow follows them);
  pair it with an `nscbc` outlet or a sponge where reflections matter.

**Not supported (yet):** `nscbc` and `turbulence`, planned in
[#565](https://github.com/MatthewBonanni/mallard/issues/565).

Design notes: [inlets](../design/inlets.md).

### `inlet_mass_flow`

```toml
[[boundaries]]
name = "inlet"
type = "inlet_mass_flow"
mdot = 0.02                 # kg/s through the whole entry (2D: per unit depth)
T = 300.0
# profile = "1 - (y / 0.05)^2"  # shape of the mass flux along the inlet
# mass_flux = 20.0             # or the flux itself [kg/(m^2 s)]: a number, an expression or a vector
# direction = [1.0, 0.0]       # default: the inward normal of each face
# X = { O2 = 0.21, N2 = 0.79 } # mixtures
```

*Used in:*
[`heated_duct`](https://matthewbonanni.github.io/mallard/docs/examples/#heated-duct),
[`channel_inlet`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-inlet),
[`channel_inlet_data`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-inlet-data),
[`pulsed_jet`](https://matthewbonanni.github.io/mallard/docs/examples/#pulsed-jet).

**Imposes** the mass flux, the temperature, the flow direction and the
composition, whatever the density at the inlet does: the usual inflow of
combustors and jets, where heat release or compression downstream changes the
density there. The acoustic wave leaving through the inlet sets the
pressure, and the density follows. The face flux is the physical flux of that
boundary state, so the mass entering is exactly the target (`inlet_state`
fixes the velocity instead, and its mass flow drifts with the inlet density).

![The mass-flow inlet: what it imposes and what it takes from the interior](../images/guide/inlet_mass_flow-light.svg#gh-light-mode-only)
![The mass-flow inlet: what it imposes and what it takes from the interior](../images/guide/inlet_mass_flow-dark.svg#gh-dark-mode-only)

| Key | Default | Meaning |
|---|---|---|
| `mass_flux` or `mdot` | one required | Mass flux `rho u . n_in` [kg/(m^2 s)] (a number or expression of x, y, z, t, or a vector, which gives the direction too), or the total mass flow through the entry's faces [kg/s; per unit depth in 2D, per radian in axisymmetric runs] (a number or expression of t, non-negative) |
| `profile` | uniform | With `mdot`: the shape of the mass flux along the inlet (an expression of x, y, z, t, non-negative), scaled to `mdot` at every stage |
| `T` | required | Inflow temperature |
| `direction` | inward normal | Flow direction (numbers or expressions of x, y, z, t); must point into the domain at every face and time; not with a `mass_flux` vector |
| `X` / `Y`, `balance` | required for mixtures | Composition |

![A closed duct fed through a mass-flow inlet and heated: the mass in the duct grows by exactly mdot t, while inlet_state's drifts](../images/guide/mass_flow_held-light.svg#gh-light-mode-only)
![A closed duct fed through a mass-flow inlet and heated: the mass in the duct grows by exactly mdot t, while inlet_state's drifts](../images/guide/mass_flow_held-dark.svg#gh-dark-mode-only)

*`examples/heated_duct` (`closed.toml`): a duct closed at the far end and
heated in its middle. The pressure rises and the inlet density with it (by
45%); the mass in the duct grows by `mdot t` to 1e-14, while `inlet_state` at
the same initial mass flux (`closed_upt.toml`) holds its pressure and lets the
gas back out.*

![Steady heated duct: pressure, velocity and temperature against Rayleigh flow](../images/guide/heated_duct-light.svg#gh-light-mode-only)
![Steady heated duct: pressure, velocity and temperature against Rayleigh flow](../images/guide/heated_duct-dark.svg#gh-dark-mode-only)

*`examples/heated_duct` (`input.toml`): the same duct open to a 1 bar outlet,
fed at 100 kg/(m^2 s) and heated between x = 0.4 and 0.6 m, against Rayleigh
flow: the inlet pressure the flow needs within 1e-5, the exit temperature
within 1e-6.*

**Use** for combustors, jets and any inflow whose mass flow is given.

**A mass flow or profile that depends on `t`.** `mdot` may be an expression
of `t` (not of `x`, `y`, `z`: it is a total; its shape across the inlet is the
`profile`), and so may the `profile`. Either is evaluated on the device at
every Runge-Kutta stage, and the faces carry exactly `mdot(t)`, the same on
any number of ranks. A jet pulsating at 200 Hz whose peak also sweeps across
the inlet:

```toml
[[boundaries]]
name = "inlet"
type = "inlet_mass_flow"
mdot = "0.02 * (1 + 0.3 * sin(2 * pi * 200 * t))"       # kg/s: 30% pulsation
profile = "1 + 0.8 * sin(2 * pi * 100 * t) * y / 0.05"  # tilts back and forth at 100 Hz
T = 300.0
```

`mdot` and the profile must stay non-negative at every face and time: a run
where either turns negative stops at the next progress check, with the time
and value (and the face, for the profile).

Pitfalls:

- The inlet is for subsonic inflow: it takes the pressure from the
  interior. For supersonic inflow use `inlet_state`.
- `mdot` is divided among the entry's faces by their areas (times `profile`);
  in 2D it is per unit depth.
- A `profile` of `t` moves the flow across the inlet, not its total: the
  total is `mdot`, renormalized at every stage. For a total that changes in
  time, make `mdot` an expression of `t`.
- Like `inlet_state`, it reflects acoustic waves (it holds the mass flux, not
  the velocity or the pressure); pair it with an `nscbc` outlet or a sponge
  where reflections matter.

**Not supported (yet):** `nscbc` and `turbulence`, planned in
[#565](https://github.com/MatthewBonanni/mallard/issues/565).

## Outlets

### `outlet_pressure`

```toml
[[boundaries]]
name = "outlet"
type = "outlet_pressure"
p = 101325.0
```

*Used in:*
[`wedge`](https://matthewbonanni.github.io/mallard/docs/examples/#wedge),
[`nozzle`](https://matthewbonanni.github.io/mallard/docs/examples/#nozzle),
[`heated_duct`](https://matthewbonanni.github.io/mallard/docs/examples/#heated-duct),
[`average_outlet`](https://matthewbonanni.github.io/mallard/docs/examples/#average-outlet),
[`forced_inlet`](https://matthewbonanni.github.io/mallard/docs/examples/#forced-inlet)
and 15 more.

**Imposes** the pressure `p` on subsonic faces (the incoming acoustic wave),
with the velocity and temperature of the interior (the other waves leave);
supersonic faces take the interior state (all waves leave). **Reversed flow**
(subsonic backflow) re-enters at the interior's temperature and velocity at
the pressure `p` (a backflow option is planned in
[#565](https://github.com/MatthewBonanni/mallard/issues/565)).

| Key | Default | Meaning |
|---|---|---|
| `p` | required | Back pressure (a number or an expression; with `average`, of `t` only) |
| `average` | `false` | [Area-averaged pressure](#average-area-averaged-pressure) instead of a local one |
| `nscbc` | none | [Non-reflecting treatment](#nscbc-characteristic-boundaries) |

**Use** for subsonic outflow; add `nscbc` where acoustic reflections matter
(long runs of low-Mach flows, combustors), `average` where part of the outlet
is supersonic.

Pitfalls:

- Without `nscbc` it reflects acoustic waves (97–99% of a pulse).
- A uniform `p` across a profile that should carry a pressure gradient (a
  swirling or curved outflow) distorts the flow upstream: use `average`.

**Not supported (yet):** `average` with `nscbc`, backflow and convective
outflow, planned in [#565](https://github.com/MatthewBonanni/mallard/issues/565).

#### `average`: area-averaged pressure

```toml
[[boundaries]]
name = "outlet"
type = "outlet_pressure"
p = 1.0e5
average = true
```

*Used in:*
[`average_outlet`](https://matthewbonanni.github.io/mallard/docs/examples/#average-outlet).

On subsonic faces, the local pressure is shifted by one number, updated every
stage, so that its area average over the outlet's faces is `p`; the transverse
profile is kept (with `p >= 1e-3` of the local pressure). Supersonic faces
take the interior state. Use it where an outlet mixes subsonic and supersonic
faces or carries a pressure profile.

*`examples/average_outlet`: a Mach 1.7 core between Mach 0.4 streams; the
outlet's area-averaged pressure stays within 0.05% of `p` while the pressure
varies by 6% across it, where a local outlet (`local.toml`) ends 2% above `p`
(`check_average_outlet.py`).*

The average is an exact sum over all
ranks (in fixed point), so results do not depend on the rank count. Not with gas
mixtures, and not with `nscbc` (both planned in
[#565](https://github.com/MatthewBonanni/mallard/issues/565)).

### `outlet_extrapolation`

```toml
[[boundaries]]
name = "outlet"
type = "outlet_extrapolation"
```

*Used in:*
[`shu_osher`](https://matthewbonanni.github.io/mallard/docs/examples/#shu-osher),
[`riemann_2d`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d),
[`riemann_2d_quads`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d-quads),
[`double_mach`](https://matthewbonanni.github.io/mallard/docs/examples/#double-mach),
[`explosion_3d`](https://matthewbonanni.github.io/mallard/docs/examples/#explosion-3d)
and 8 more.

**Imposes** nothing: the exterior state is the solution translated from
inside the domain (the reconstructed state on the image face one cell
inward), so every wave leaves without a reference state. Exact for supersonic
outflow.

**Use** for supersonic outflow, and to let waves leave a box in short runs
(shock tubes, blasts).

Pitfalls:

- Subsonic outflow has nothing to fix its pressure, which drifts in long runs:
  use `outlet_pressure` (with `nscbc`).
- **Reversed flow** takes the interior state in as well, unconstrained.

Theory: [transmissive boundaries](../theory/boundaries_sources.md#transmissive-boundaries).

## Other

### `farfield`

```toml
[[boundaries]]
name = "farfield"
type = "farfield"
u = [0.5, 0.0]               # free stream, a vector
p = 1.0
T = 1.0
```

*Used in:*
[`sphere_mach3`](https://matthewbonanni.github.io/mallard/docs/examples/#sphere-mach3),
[`ellipsoid`](https://matthewbonanni.github.io/mallard/docs/examples/#ellipsoid),
[`sphere_axisymmetric`](https://matthewbonanni.github.io/mallard/docs/examples/#sphere-axisymmetric),
[`pulsed_jet`](https://matthewbonanni.github.io/mallard/docs/examples/#pulsed-jet),
[`cylinder`](https://matthewbonanni.github.io/mallard/docs/examples/#cylinder)
and 1 more.

**Imposes** a characteristic free stream: the outgoing Riemann invariant
comes from the interior and the incoming one from the free stream, entropy and
tangential velocity from the upwind side; supersonic faces take the free
stream (inflow) or the interior (outflow). It works for inflow, outflow and
tangential flow alike and lets waves leave.

**Use** on the outer boundary of external flows (airfoils, cylinders,
spheres), well away from the body.

**Pitfall:** it assumes a uniform free stream; a wake or a lifting body's
circulation reaching the boundary reflects. Place it far enough away.

**Not supported (yet):** gas mixtures (a free-stream composition), planned in
[#565](https://github.com/MatthewBonanni/mallard/issues/565).

Theory: [far field](../theory/boundaries_sources.md#far-field).

### `symmetry`

```toml
[[boundaries]]
name = "bottom"
type = "symmetry"
```

*Used in:*
[`sod`](https://matthewbonanni.github.io/mallard/docs/examples/#sod),
[`shu_osher`](https://matthewbonanni.github.io/mallard/docs/examples/#shu-osher),
[`double_mach`](https://matthewbonanni.github.io/mallard/docs/examples/#double-mach),
[`wedge`](https://matthewbonanni.github.io/mallard/docs/examples/#wedge),
[`explosion_3d`](https://matthewbonanni.github.io/mallard/docs/examples/#explosion-3d)
and 31 more.

**Imposes** zero normal velocity and zero normal gradients (the ghost mirrors
the interior); under gravity it continues the hydrostatic pressure.

**Use** for symmetry planes, slip walls, and the axis (y = 0) of axisymmetric
runs, where it is required.

### `dirichlet`

```toml
[[boundaries]]
name = "top"
type = "dirichlet"
rho = "8.0"                       # expressions of x, y, z, t
u = ["7.1447096811216", "-4.125"]
p = "116.5"
```

*Used in:*
[`shu_osher`](https://matthewbonanni.github.io/mallard/docs/examples/#shu-osher),
[`double_mach`](https://matthewbonanni.github.io/mallard/docs/examples/#double-mach),
[`noh_axisymmetric`](https://matthewbonanni.github.io/mallard/docs/examples/#noh-axisymmetric).

**Imposes** the exterior state, evaluated at each face center at every stage;
nothing comes from the interior.

**Use** for exact and manufactured solutions, and moving shocks along a
boundary.

**Pitfall:** like `inlet_state`, it is over-specified wherever the flow is
subsonic.

### Periodic

Periodic zones are not a `[[boundaries]]` type: a `[[periodic]]` entry joins
two zones, and their faces become interior faces. Every scheme sees the seam
as the inside of the domain; the zones take no `[[boundaries]]` entry, and a
fully periodic mesh needs no `[[boundaries]]` at all.

```toml
# Translational: zone B is zone A moved by translation
[[periodic]]
zones = ["left", "right"]
translation = [1.0, 0.0]
```

```toml
# Rotational: a 30 degree sector about the z axis
[[periodic]]
zones = ["side_a", "side_b"]
angle = 30.0
axis = [0.0, 0.0, 1.0]
```

*Used in:*
[`rotational_sector`](https://matthewbonanni.github.io/mallard/docs/examples/#rotational-sector),
[`turbulent_hot_spot_2d`](https://matthewbonanni.github.io/mallard/docs/examples/#turbulent-hot-spot-2d),
[`poiseuille_pipe`](https://matthewbonanni.github.io/mallard/docs/examples/#poiseuille-pipe),
[`moving_heaters`](https://matthewbonanni.github.io/mallard/docs/examples/#moving-heaters),
[`vortex_ring_3d`](https://matthewbonanni.github.io/mallard/docs/examples/#vortex-ring-3d)
and 13 more.

Rules: zone B must match zone A node for node (to 1e-6 of the zones' shortest
edge) after the translation or rotation, and a node without a partner is
reported with its coordinates; each periodic direction needs at least 3
cells; each zone joins one pair at most; a mesh takes at most one distinct
rotation. Details and rotational periodicity:
[Meshes](meshes.md#periodic-boundaries); keys:
[`[[periodic]]`](../input.md#periodic).

## Options

### `nscbc`: characteristic boundaries

```toml
[[boundaries]]
name = "outlet"
type = "outlet_pressure"
p = 101325.0
nscbc = { L = 0.1 }        # a length of the domain [m]; sigma = 0.25 by default
```

*Used in:*
[`forced_inlet`](https://matthewbonanni.github.io/mallard/docs/examples/#forced-inlet),
[`pulsed_jet`](https://matthewbonanni.github.io/mallard/docs/examples/#pulsed-jet),
[`channel_inflow_retau180`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-inflow-retau180),
[`turbulent_inflow_duct`](https://matthewbonanni.github.io/mallard/docs/examples/#turbulent-inflow-duct),
[`volvo_bluff_body`](https://matthewbonanni.github.io/mallard/docs/examples/#volvo-bluff-body)
and 2 more.

`nscbc` makes `outlet_pressure` and `inlet_state` Navier–Stokes
characteristic boundary conditions, partially non-reflecting (Poinsot & Lele
1992, with the transverse terms of Lodato et al. 2008): outgoing waves leave,
and the incoming acoustic wave relaxes toward the type's target, the pressure
`p` at an outlet and the normal velocity `u` at an inlet. Inlets impose
temperature and tangential velocity (or relax them with `sigma_T`,
`sigma_t`). Supersonic inflow imposes the target state; supersonic outflow
takes the interior's.

![Waves at a characteristic outlet and inlet: outgoing waves leave, incoming ones relax toward the targets](../images/guide/characteristic-light.svg#gh-light-mode-only)
![Waves at a characteristic outlet and inlet: outgoing waves leave, incoming ones relax toward the targets](../images/guide/characteristic-dark.svg#gh-dark-mode-only)

*`u` carries entropy, vorticity and species; `u ± c` are the acoustic waves.*

| Key | Default | Meaning |
|---|---|---|
| `L` | required | A length of the domain, setting the relaxation rate `K = sigma c (1 - M^2) / L` |
| `sigma` | 0.25 | Relaxation coefficient; 0 lets the mean drift, large values approach a fixed pressure (velocity) |
| `beta` | local Mach (outlet); 0 for a turbulent inlet | Weight of the transverse terms, 0 to 1 |
| `sigma_T`, `sigma_t` | imposed | Inlets only: relax temperature and tangential velocity instead of imposing them |

The table may also be a sub-table:

```toml
[[boundaries]]
name = "inlet"
type = "inlet_state"
u = [50.0, 0.0]
p = 101325.0
T = 300.0

[boundaries.nscbc]
L = 2.4e-3
sigma = 0.25
```

A wave of frequency `omega` is reflected by `1 / sqrt(1 + (2 omega / K)^2)`.
Restart files carry each face's state, so restarted runs continue bitwise.

Pitfalls:

- **Reversed flow at outlets.** Eddies leaving through the outlet may reverse
  the flow in its last cells; such faces drop their transverse terms and take
  the entropy of the exterior state, and `beta = 0` is more robust (the
  `channel_inflow_retau180` example).
- **Viscous terms** treat characteristic faces like `outlet_extrapolation`
  (zero normal gradients).

**Not supported (yet):** `nscbc` on `inlet_stagnation`, `inlet_mass_flow` and
walls, and together with `average`, planned in
[#565](https://github.com/MatthewBonanni/mallard/issues/565).

Design notes: [characteristic boundaries and sponges](../design/nscbc.md);
theory: [characteristic boundaries](../theory/boundaries_sources.md#characteristic-boundaries).

### Synthetic turbulence (`turbulence`)

```toml
[[boundaries]]
name = "left"
type = "inlet_state"
u = [1.0, 0.0, 0.0]
p = 17.857
T = 1.0
nscbc = { L = 6.0 }

[boundaries.turbulence]
reynolds_stress = [0.01, 0.01, 0.01, 0.0, 0.0, 0.0]   # R_xx, R_yy, R_zz, R_xy, R_xz, R_yz
length_scale = 0.25
```

*Used in:*
[`channel_inflow_retau180`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-inflow-retau180),
[`turbulent_inflow_duct`](https://matthewbonanni.github.io/mallard/docs/examples/#turbulent-inflow-duct),
[`volvo_bluff_body`](https://matthewbonanni.github.io/mallard/docs/examples/#volvo-bluff-body).

An `inlet_state` with `nscbc` and a `[boundaries.turbulence]` table adds
turbulence to its target velocity: the digital filter of Klein, Sadiki &
Janicka (2003), with given Reynolds stresses and integral length scales,
bitwise the same on any number of ranks and across restarts.

![A duct with a turbulent characteristic inlet, a characteristic outlet, periodic in y and z](../images/guide/domain_turbulent_inflow_duct-light.svg#gh-light-mode-only)
![A duct with a turbulent characteristic inlet, a characteristic outlet, periodic in y and z](../images/guide/domain_turbulent_inflow_duct-dark.svg#gh-dark-mode-only)

| Key | Default | Meaning |
|---|---|---|
| `reynolds_stress` | required (or `profile`) | `[R_xx, R_yy, R_zz, R_xy, R_xz, R_yz]` (2D: `[R_xx, R_yy, R_xy]`), numbers or expressions |
| `profile` | — | Instead of `u` and `reynolds_stress`: a CSV of plane averages from a precursor run (`mallard post average --homogeneous`) |
| `length_scale` | required | Integral length: one number, one per direction, or one list per component |
| `convection_velocity` | mean inflow speed (at `t = 0`) | `U_c` of Taylor's hypothesis |
| `seed` | 1 | Random seed |
| `zero_net_flux` | `true` | Remove the area mean of the normal fluctuation (no plane acoustic wave) |
| `points_per_length` | 6 | Generator grid points per smallest integral length |

**A mean that depends on `t`.** `u` may be an expression of `t`: the
fluctuations are added on the device, at every stage, to the mean evaluated
there, like any other inlet of `t`. The Reynolds stresses and length scales
are those of the fluctuations alone, set at setup in `x`, `y`, `z`; they do
not follow the mean. The convection velocity of Taylor's hypothesis sets the
time between the generator's planes, so it stays fixed: by default the mean
inflow speed at `t = 0`. Set `convection_velocity` when the mean starts at
rest or changes widely. A ramped inlet with turbulence:

```toml
[[boundaries]]
name = "left"
type = "inlet_state"
u = ["10 * min(t / 0.01, 1)", 0.0, 0.0]   # from rest to 10 m/s over 10 ms
p = 1.0e5
T = 300.0
nscbc = { L = 1.0 }

[boundaries.turbulence]
reynolds_stress = [1.0, 0.5, 0.5, 0.0, 0.0, 0.0]   # (1 m/s)^2 streamwise from the start
length_scale = 0.01
convection_velocity = 10.0   # the mean at t = 0 is 0, so give it
```

The inlet must be planar and normal to a coordinate axis. Examples:
`turbulent_inflow_duct`, `channel_inflow_retau180` (statistics from the
periodic channel via `profile`), `volvo_bluff_body`
([benchmark](https://matthewbonanni.github.io/mallard/validation/#synthetic-inflow)).

**Not supported (yet):** turbulence on inlets without `nscbc`, and precursor
or recycling inflow, planned in
[#565](https://github.com/MatthewBonanni/mallard/issues/565).

## Inputs from older releases

Types renamed in 0.9.0 were removed in 0.10.0: a run with one stops, naming
the replacement. `mallard input upgrade old.toml > new.toml` rewrites them, keeping comments and layout
([inputs from older releases](input_files.md#inputs-from-older-releases)).

| Old | New |
|---|---|
| `upt` | `inlet_state` |
| `nscbc_inlet` | `inlet_state` with `nscbc` (its `L`, `sigma`, `beta`, `sigma_T`, `sigma_t` move into the table) |
| `p_out` | `outlet_pressure` |
| `p_out_average` | `outlet_pressure` with `average = true` |
| `nscbc_outlet` | `outlet_pressure` with `nscbc` (its `L`, `sigma`, `beta` move into the table) |
| `extrapolation` | `outlet_extrapolation` |

## Pitfalls

- **Reflections.** `outlet_pressure` and `inlet_state` without `nscbc`
  reflect acoustic waves; in long runs of low-Mach flows these can dominate.
  Use `nscbc`, and a [sponge layer](sources.md#sponge-layers) in front of the
  outlet where eddies leave.
- **Over-specified inflow.** A subsonic inlet that fixes the pressure as well
  as the velocity (`inlet_state`, `dirichlet`) cannot let an acoustic wave
  out.
- **Viscous terms** treat characteristic and transmissive faces like
  `outlet_extrapolation` (zero normal gradients).

## Related

- [Periodic and rotational-periodic boundaries](meshes.md#periodic-boundaries) (meshes)
- [Wall models](les.md#wall-models-wall_model) (LES)
- [Sponge layers](sources.md#sponge-layers) (sources)
- [Turbulent inflow](#synthetic-turbulence-turbulence) and its [design](../design/synthetic_inflow.md)

## See also

- [Input reference: `[[boundaries]]`](../input.md#boundaries), [`[boundaries.turbulence]`](../input.md#synthetic-turbulence-boundariesturbulence)
- [Numerical methods: boundary conditions](../numerics/overview.md#boundary-conditions)
- Design notes: [characteristic boundaries and sponges](../design/nscbc.md), [inlets](../design/inlets.md), [synthetic inflow](../design/synthetic_inflow.md)
