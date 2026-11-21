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
from the resolved gradient of `Z` ([LES](#les)).

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
| `variance` | `[physics]`, with `[les]` | Closure of the subgrid variance of `Z`: `algebraic` (default) or `transported` |
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
| `HRR` | Heat release rate, if the table has the optional field `HRR` (W m-3), scaled from the table's density by `(rho / rho0)^2` as the source of `C` |
| `MIXTURE_FRACTION` | `Z` |
| `FPV_LAMBDA` | The table's normalized progress variable `Lambda = (C - C_min) / (C_max - C_min)` at `(Z, Sz)`, in [0, 1] (FPVgen's Λ). `PROGRESS_VARIABLE` is the mixtures' own, defined by `[output.mixture]`, and not a field of flamelet runs |
| `MIXTURE_FRACTION_VARIANCE` | (`[les]`) The subgrid variance of `Z` of each cell's lookups |

With `[les]` the table fields are those at the cell's variance.

```toml
[[write_data]]
prefix = "flame"
format = "vtkhdf"
interval = 100
fields = ["RHO", "U", "P", "T", "T_TABLE", "HRR", "Y_*", "FPV_LAMBDA"]
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

Inlets and the boundary states are laminar (zero variance; with the transported
variance, inlets carry `Z2 = Z^2`). The initial state
is converted at its variance: `p`, `T`, `u`, `Z` and `C` are those given, the
density and energy those of the table at the cell's variance. A transported
`Z2` starts at zero variance. `[les.combustion]`
(thickened flames) is an input error with the flamelet gas: the table already
represents the subgrid flame.

## Not yet supported

These are input errors with `gas = "flamelet"`, each naming the key:
`[les.combustion]`, `[chemistry]`, `[radiation]`, `[output.mixture]`, the fields `ENTROPY` and the
mixture's other derived fields, and nondimensional units. Probes and statistics
do not sample the table fields (`T_TABLE`, `HRR`, the table's `Y_<species>`).
