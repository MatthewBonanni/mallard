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
that writes [Mallard's format](../design/table_format.md). This first stage
runs laminar flames (no `[les]`); the subgrid variance of `Z` is zero.

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
`(Z, S, c)` with `c = (C - C_min) / (C_max - C_min)` the normalized progress
variable at `Z`; Mallard finds `c` from the transported `C`, exactly, because
the format requires `C` to be linear in `c`.

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

`C` must lie between the table's `C_min` and `C_max` at its `Z`; values
outside are clamped in lookups.

## Not yet supported

These are input errors with `gas = "flamelet"`, each naming the key:
`[les]` (LES closures and the subgrid variance are the next stage),
`[chemistry]`, `[radiation]`, `[output.mixture]`, the fields `ENTROPY` and the
mixture's derived fields, nondimensional units, `wall_isothermal`,
`wall_heat_flux`, `inlet_stagnation`, `inlet_mass_flow`, expressions of `T`,
`Z` or `C` at `inlet_state`, and boundary values that vary in time. Walls are
adiabatic and pass no scalar flux. Output of the table's species, `T` from the
table and `HRR` as derived fields comes later.
