# Tabulated chemistry (flamelets)

`gas = "flamelet"` replaces the species of a mechanism with two transported
scalars, the mixture fraction `Z` and the progress variable `C`, and looks the
gas's thermodynamics, transport and the source of `C` up in a flamelet table
computed beforehand (flamelet/progress variable, FPV). A run then costs little
more than a non-reacting one: six or seven values per cell instead of
`5 + N_species`, and no stiff chemistry. Reading tables needs a build with
HDF5. Design and rationale: [design notes](../design/flamelet.md); the table
format: [table_format.md](../design/table_format.md).

Mallard does not generate tables: they come from a separate generator library
that writes [Mallard's format](../design/table_format.md). Laminar runs look
the table up at zero subgrid variance of `Z`; with `[les]` the variance comes
from the resolved gradient of `Z` ([LES](#les)), or is zero with
`variance = "none"` and with thickened flames ([thickened flames](#thickened-flames)).

```toml
[physics]
type = "navier_stokes"      # or "euler": no transport
gas = "flamelet"
table = "h2_fpv.h5"         # relative to the directory the run starts in

[initialize]
type = "analytical"
p = "101325"
T = "300"
u = ["0", "0"]
Z = "x < 0 ? 1 : 0"         # 1: fuel stream, 0: oxidizer
C = "0"                     # fresh; "burnt" is the table's burnt value at Z

[[boundaries]]
name = "fuel"
type = "inlet_state"
u = 1.0
p = 101325.0
T = 300.0
Z = 1.0
C = 0.0
```

## The model

The transported scalars are `rho Z` and `rho C`, in the species block
(output `Y_Z`, `Y_C`, restart and probes as any species). The table's axes are
`(Z, Sz, Lambda)`, named as in FPVgen: `Sz` the normalized variance of `Z`
and `Lambda = (C - C_min) / (C_max - C_min)` the normalized progress variable
at `(Z, Sz)`. Mallard finds `Lambda` from the transported `C`, exactly,
because the format requires `C` to be linear in `Lambda`.

FPV tables hold the whole S-curve of steady counterflow flamelets, not only
its burning flames. The S-curve has three branches: the stable (burning)
branch, from low strain to extinction; the unstable branch, from extinction
back toward low strain at lower temperatures; and the extinguished branch, pure
mixing at `C = 0`. The unstable flamelets fill the range between burning and
extinguished, so that ignition, extinction and lifted flames move through
states the table holds. Each flamelet is a profile in `Z`. At each `Z` the
flamelets, ordered by their progress variable, become the `Lambda` axis.

The figures on this page come from a sample table built that way
(`tools/docs/flamelet.py`). It holds Cantera flamelets of H2/N2 (1:1 by mole)
against air, both at 300 K and 1 atm, with `mechanisms/h2o2.yaml`, unity Lewis
number and `C = Y_H2O`. The stable branch runs from `chi_st` = 0.32 1/s
(`T_max` = 1978 K) to extinction at 2777 1/s and 1246 K. The unstable branch
runs from there back to 0.52 1/s, where it levels off at `T_max` = 986 K.

![S-curve of the sample table's counterflow flamelets: peak temperature and source of C against the stoichiometric scalar dissipation rate, with the stable, unstable and extinguished branches, and the temperature of flamelets of each branch against the mixture fraction](../images/guide/flamelet_s_curve-light.svg#gh-light-mode-only)
![S-curve of the sample table's counterflow flamelets: peak temperature and source of C against the stoichiometric scalar dissipation rate, with the stable, unstable and extinguished branches, and the temperature of flamelets of each branch against the mixture fraction](../images/guide/flamelet_s_curve-dark.svg#gh-dark-mode-only)

At the stoichiometric `Z`, the stable branch fills `Lambda` from 0.63 to 1 and
the unstable branch from 0.34 to 0.63. Below 0.34 no steady flamelet of the
sample exists, and the table interpolates linearly to the mixing solution at
`Lambda = 0`. The source of `C` peaks at 707 kg/m³/s at `Lambda` = 0.67, near
extinction; the most burning flamelet, at `Lambda = 1`, has the highest
`T_TABLE` and a small source.

![Temperature, source of C and OH mass fraction of the sample table at the stoichiometric mixture fraction against Lambda, with the nodes from the stable and the unstable branch and the interpolated range below them](../images/guide/flamelet_progress-light.svg#gh-light-mode-only)
![Temperature, source of C and OH mass fraction of the sample table at the stoichiometric mixture fraction against Lambda, with the nodes from the stable and the unstable branch and the interpolated range below them](../images/guide/flamelet_progress-dark.svg#gh-dark-mode-only)

Over `(Z, Lambda)` at zero variance, the dashed line is the extinction
flamelet and the dotted line the last unstable one:

![Temperature, density, source of C, heat release rate, C and OH mass fraction of the sample table at zero variance, over the mixture fraction Z and the normalized progress variable Lambda, with the extinction flamelet and the last unstable flamelet marked](../images/guide/flamelet_table-light.svg#gh-light-mode-only)
![Temperature, density, source of C, heat release rate, C and OH mass fraction of the sample table at zero variance, over the mixture fraction Z and the normalized progress variable Lambda, with the extinction flamelet and the last unstable flamelet marked](../images/guide/flamelet_table-dark.svg#gh-dark-mode-only)

A lookup maps the transported `(Z, C)` to `Lambda` with the table's bounds of
`C` at that `(Z, Sz)`, then interpolates multilinearly between the `2^3`
nodes of the table cell around `(Z, Sz, Lambda)`
([design](../design/flamelet.md#4-lookup-on-the-device)).

![Bounds of C against Z at 3 normalized variances, with a transported state mapped to Lambda = 0.6, and the 8 corners of a table cell around a query point](../images/guide/flamelet_lookup-light.svg#gh-light-mode-only)
![Bounds of C against Z at 3 normalized variances, with a transported state mapped to Lambda = 0.6, and the 8 corners of a table cell around a query point](../images/guide/flamelet_lookup-dark.svg#gh-dark-mode-only)

- **Thermodynamics** (Saghafian et al. 2015): the table holds, per node, `T0`,
  `e0 = e(T0)`, `R`, `gamma0` and `a_gamma`, with `gamma(T) = gamma0 +
  a_gamma (T - T0)`. The temperature follows from the transported energy in
  closed form, `p = rho R T`, so heat release, compression and acoustics are
  consistent with the energy equation; every Riemann solver and face
  reconstruction works.
- **Transport** (`navier_stokes`): `mu`, `kappa` and `cp` from the table;
  unity Lewis number, `rho D = kappa / cp` for both scalars, and the heat
  flux `-(kappa / cp) grad h` (so that enthalpy diffuses with `Z`, as in the
  flamelets the table came from). The time step includes the largest of
  `4/3 mu / rho`, `kappa / (rho cv)` and `D`.
- **Source**: `omega_C` from the table, an explicit source of `rho C` in the
  RHS, scaled from the table's density `rho0` at that node to the local density
  by `(rho / rho0)^2` (a second-order rate). It is a pointwise kernel on the
  device, with no chemistry step.

Each scalar is limited by its own slope cut in reconstruction (they are not
mass fractions that sum to one).

## Keys

| Key | Where | Meaning |
|---|---|---|
| `gas = "flamelet"`, `table` | `[physics]` | The table file, an FPV table with `saghafian2015` thermodynamics |
| `Z`, `C` | `[initialize]`, `[[initialize.regions]]`, `inlet_state` | Mixture fraction in [0, 1] (default 0) and progress variable in the table's range at `Z` (default 0); numbers, expressions, or `C = "burnt"` |
| `variance` | `[physics]`, with `[les]` | Closure of the subgrid variance of `Z`: `algebraic` (default), `transported`, or `none` (zero variance; the default, and the only choice, with `[les.combustion]`) |
| `C_Z` | `[physics]`, `variance = "algebraic"` | Constant of the algebraic variance, default 0.1, non-negative |
| `C_chi` | `[physics]`, `variance = "transported"` | Constant of the subgrid dissipation rate, default 2, positive |

`C` must lie between the table's `C_min` and `C_max` at its `Z`; values
outside are clamped in lookups.

## Output

Besides `Y_Z` and `Y_C` (the transported scalars), a flamelet run writes the
table's fields at each cell's `Z` and `C` ([[write_data]] only):

| Field | Meaning |
|---|---|
| `Y_<species>` | Mass fraction of each species of the table's species block (`Y_*` selects them with `Y_Z` and `Y_C`); a species named `Z` or `C` is not written |
| `T_TABLE` | Temperature of the tabulated flamelet state, `T0`. `T` is the temperature of the energy equation, which compression and heat loss move away from it |
| `HRR` | Heat release rate, if the table has the optional field `HRR` (W m-3), scaled from the table's density by `(rho / rho0)^2` as the source of `C` (and by `E / F` in a thickened flame) |
| `MIXTURE_FRACTION` | `Z` |
| `C_NORM` | The table's normalized progress variable `Lambda = (C - C_min) / (C_max - C_min)` at `(Z, Sz)`, in [0, 1] (FPVgen's Λ). `PROGRESS_VARIABLE` is the mixtures' own, defined by `[output.mixture]`, and not a field of flamelet runs |
| `MIXTURE_FRACTION_VARIANCE` | (`[les]`, `variance` other than `none`) The subgrid variance of `Z` of each cell's lookups |

With `[les]` the table fields are those at the cell's variance.

```toml
[[write_data]]
prefix = "flame"
format = "vtkhdf"
interval = 100
fields = ["RHO", "U", "P", "T", "T_TABLE", "HRR", "Y_*", "C_NORM"]
```

## Pressure

A table holds flamelets computed at one pressure. A run whose initial pressure,
or a boundary's prescribed pressure (`inlet_state`, `inlet_stagnation`,
`outlet_pressure`), differs from it by more than 10% warns `W-TABLE-PRESSURE`:
the tabulated source, transport and composition are those of the table's
pressure. Generate the table at the run's pressure.

## Boundaries

`Z` and `C` take the place of `X` or `Y` at `inlet_state`, `inlet_stagnation`
and `inlet_mass_flow`: numbers, expressions of `x`, `y`, `z`, `t` (as other
boundary values), or `C = "burnt"`, the table's burnt value at `Z`. A `"burnt"`
`C` may follow a `Z` that varies in space, but not one that varies in time:
give `C` as an expression of `t` too. The inlets' gas constant, energy and, for
`inlet_stagnation`, isentrope are the table's at their `Z` and `C`.

```toml
[[boundaries]]
name = "pilot"
type = "inlet_stagnation"
p0 = 1.1e5
T0 = 1800.0
Z = "0.27 + 0.02 * sin(2000 * t)"
C = "0.12"
```

Walls pass no flux of `Z` or `C`. `wall_isothermal` conducts heat with
`(kappa / cp) dh/dn`, `h` the enthalpy at the wall temperature and the
cell's `Z` and `C`; `wall_heat_flux` passes its `q`; `wall_adiabatic`
nothing.

## LES

With `[les]` every lookup (thermodynamics, transport, the source of `C` and the
output fields) is at the cell's subgrid variance of `Z`, the table's second
axis. The algebraic closure of Pierce & Moin (1998) sets it from the resolved
gradient:

$$
\widetilde{Z''^2} = \min\left(C_Z \Delta^2 |\nabla \tilde Z|^2,\ \tilde Z (1 - \tilde Z)\right)
$$

with `Delta` the LES filter width (`filter_width`) and the least-squares
gradient of the viscous fluxes. The table maps it to its normalized axis
`Sz = Zvar / (Z (1 - Z))`, so the clip is the table's edge.

The variance smooths the flamelets over a beta distribution of `Z`: the peak
temperature and the source of `C` fall and spread with `Sz`, and `Sz = 1`
leaves only the two pure streams.

![Temperature at Lambda = 1 and source of C at Lambda = 0.7 of the sample table against the mean mixture fraction, at 5 normalized variances Sz](../images/guide/flamelet_variance-light.svg#gh-light-mode-only)
![Temperature at Lambda = 1 and source of C at Lambda = 0.7 of the sample table against the mean mixture fraction, at 5 normalized variances Sz](../images/guide/flamelet_variance-dark.svg#gh-dark-mode-only)

On the rich side, `Lambda = 1` at `Sz = 0` is not one flamelet: at each `Z` it
is the flamelet with the most `C` there, hence the kinks of the laminar curve.

The eddy viscosity adds `mu_t / Sc_t` to the diffusivity of `Z` and `C`, and
`mu_t / Pr_t` to the conductivity of the enthalpy (`h` is conducted with
`kappa / cp`, so its eddy part is `mu_t / Pr_t`, not `cp mu_t / Pr_t`). With
`Sc_t = Pr_t` (both default 0.9) enthalpy mixes with `Z`, as the unity Lewis
number of the table assumes; other values let the energy drift from the
flamelets'. Wall faces keep the molecular fluxes.

```toml
[physics]
type = "navier_stokes"
gas = "flamelet"
table = "h2_fpv.h5"
C_Z = 0.1                   # variance = "algebraic" is the default

[les]
model = "sigma"
```

With `variance = "transported"` the second moment `Z2`, the filtered `Z^2`, is
a third transported scalar (written as `Y_Z2`), and the variance is
`Zvar = Z2 - Z^2`, within `[0, Z (1 - Z)]`. `rho Z2` is conserved and diffuses
with `rho (D + D_t)`, as `Z`; its sink is the filtered dissipation rate

$$
\bar\rho \tilde\chi = 2 \bar\rho D |\nabla \tilde Z|^2 + \bar\rho \chi_{sgs}, \qquad
\chi_{sgs} = C_\chi (D + D_t) \frac{\widetilde{Z''^2}}{\Delta^2}
$$

The eddy diffusion of `Z2 = Z^2 + Zvar` produces the variance at
`2 rho D_t |grad Z|^2`, so it relaxes in `Delta^2 / (C_chi (D + D_t))` toward
`2 D_t / (C_chi (D + D_t)) Delta^2 |grad Z|^2`: with `D_t >> D` the algebraic
model with `C_Z = 2 / C_chi`. The default `C_chi = 2` relaxes within a few
explicit time steps; much larger values make the sink stiff at the viscous time
step, as `C_chi (D + D_t) dt / Delta^2` reaches `C_chi cfl / 8` on regular cells.
Transporting `Z2` rather than `Zvar` keeps the production out of the scheme's
discretization (Kaul et al. 2013).

```toml
[physics]
type = "navier_stokes"
gas = "flamelet"
table = "h2_fpv.h5"
variance = "transported"
C_chi = 2.0
```

With `variance = "none"` every lookup is at zero variance, the table's
`Sz = 0` slice (the laminar flamelets), as in laminar runs; `Z` and `C` still
take the eddy diffusivity.

Inlets and the boundary states are laminar (zero variance; with the transported
variance, inlets carry `Z2 = Z^2`). The initial state
is converted at its variance: `p`, `T`, `u`, `Z` and `C` are those given, the
density and energy those of the table at the cell's variance. A transported
`Z2` starts at zero variance.

## Thickened flames

`[les.combustion]` with `model = "tfles"` thickens the flame of the table so
that the mesh resolves it, as for finite-rate chemistry
([thickened flames](les.md#thickened-flames-lescombustion)): where the flame
sensor `Omega` is 1, the diffusivity of `Z` and `C` and the conductivity of the
enthalpy are multiplied by `E F`, the source of `C` (and the `HRR` output) by
`E / F`, and the eddy diffusivities by `1 - Omega`. The thickened flame keeps
the laminar flame speed times the efficiency `E` and spans `n_res` cells.

- **Sensor**: the table's normalized progress variable `Lambda` (`C_NORM`)
  in place of the temperature, `Omega = min(1, Lambda (1 - Lambda) / 0.0475)`:
  1 across the flame (`0.05 <= Lambda <= 0.95`), 0 in fresh and burnt gas
  and where the table has no flame at the cell's `Z`. The dynamic efficiency
  filters `Lambda` too. `sensor = "progress"` is the default and the only
  sensor with `gas = "flamelet"`; `T_unburnt` and `T_burnt` are not used.
- **Variance**: none. The thickened flame is resolved on the mesh, so the
  table is looked up at zero subgrid variance of `Z` (`Sz = 0`, the laminar
  flamelets): `variance = "none"` is the default, and `algebraic` or
  `transported` is an input error with `[les.combustion]`, as are `C_Z` and
  `C_chi`. The table keeps its `Sz` axis; only its first slice is read.
- `delta_L` and `s_L`, the thickness and speed of the laminar flame that `F`
  and the efficiency scale with, are required, as with `sensor =
  "temperature"`.

The sensor is 1 wherever the table's source of `C` is (the sample table at
3 mixture fractions):

![The thickened flame's sensor against Lambda, with the source of C of the sample table at 3 mixture fractions, each normalized by its peak](../images/guide/flamelet_tfles_sensor-light.svg#gh-light-mode-only)
![The thickened flame's sensor against Lambda, with the source of C of the sample table at 3 mixture fractions, each normalized by its peak](../images/guide/flamelet_tfles_sensor-dark.svg#gh-dark-mode-only)

```toml
[physics]
type = "navier_stokes"
gas = "flamelet"
table = "ch4_fpv.h5"        # variance = "none" is implied

[les]
model = "sigma"

[les.combustion]
model = "tfles"
delta_L = 4.5e-4            # m, of the laminar flame
s_L = 0.38                  # m/s
n_res = 5
```

The partially stirred reactor (`model = "pasr"`) scales finite-rate rates and
is an input error with the flamelet gas.

## Not yet supported

These are input errors with `gas = "flamelet"`, each naming the key:
`[les.combustion]` `model = "pasr"`, `[chemistry]`, `[radiation]`, `[output.mixture]`, the fields `ENTROPY` and the
mixture's other derived fields, and nondimensional units. Probes and statistics
do not sample the table fields (`T_TABLE`, `HRR`, the table's `Y_<species>`).
