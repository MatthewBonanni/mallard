# Tabulated chemistry: flamelets, FPV and FGM

Status: proposed (#631). [Stage](#11-stages) 1 is implemented: the table
generator, the table format and the device lookup, verified against Cantera.
Recommendations are marked **Decision**; the alternatives considered are listed
with each, and the [open questions](#12-open-questions) need an answer before
stage 2 changes the solver.

Mallard's reacting flows today carry every species of a mechanism and integrate
finite-rate chemistry in each cell ([chemistry.md](chemistry.md)); in LES the
flame is thickened to be resolved (TFLES, [les.md](les.md#6-turbulence-chemistry-interaction)).
That is the most general closure and the most expensive one: memory and
bandwidth scale with the species count, the chemistry with the mechanism, and
TFLES needs the flame on the mesh. Tabulated chemistry replaces the species by a
few transported control variables (mixture fraction, its variance, a progress
variable, an enthalpy defect) and looks everything else up in a table computed
beforehand from laminar flames, so that large reacting LES costs little more
than non-reacting LES.

## Goals

- **One framework, four models.** Flamelet/progress variable (FPV) for
  non-premixed and lifted flames (#287), flamelet-generated manifolds (FGM,
  FPI) for premixed and partially premixed flames, steady laminar flamelets
  (SLFM), and heat loss and slow chemistry (NOx, #280) through further table
  axes and unsteady flamelets. They differ in how the table is generated and in
  which control variables are transported, not in the solver.
- **Our own table generator on Cantera's 1D solvers.** Mallard does not write a
  flamelet solver: `tools/setup/flamelet_table.py` drives Cantera's counterflow
  and free flames, integrates the presumed PDFs and writes the table.
- **A versioned table format with provenance**, read by every build (no
  optional dependency) and by numpy.
- **Device lookup** in the RHS and the output kernels: multilinear, a few cache
  lines per lookup, bitwise independent of the backend, thread and rank count.
- **Ignorable when off.** A run without `gas = "flamelet"` compiles to and runs
  the code it runs today, byte-identical; the model's code sits in its own files
  and is reached from a few well-defined points.
- **Validation** against the table's own flamelets, independent Cantera
  solutions, FPVgen's tables, Mallard's finite-rate solutions of the same
  laminar flames, and the standard turbulent flames (section 10).

## Non-goals

- A flamelet solver of our own, interactive flamelet models (RIF), CMC,
  transported PDFs, linear-eddy models.
- Tables with more than four axes, adaptive or sparse tables, neural-network
  regression of tables. The format leaves room for them.
- Multiphase and spray flamelets.
- Replacing finite-rate chemistry: both stay, and the [shared
  case](#10-validation) compares them.

## Summary of decisions

| # | Decision | Main alternatives | Why |
|---|---|---|---|
| 1 | Tables generated offline by a Python tool on Cantera's 1D flames (counterflow S-curves by two-point flame control; free flames for FGM) | FPVgen; FlameMaster; a flamelet solver in Mallard | Cantera is already Mallard's chemistry reference; FPVgen's continuation is reproduced, its gaps (section 2.5) are not |
| 2 | Presumed PDFs integrated at generation: beta in Z, exact for piecewise-linear profiles through the incomplete beta function; delta in the progress parameter | Quadrature of the PDF (singular at 0 and 1); FPVgen's copies of the laminar flamelet along the variance axis | Exact, no quadrature parameter, no singularity; the variance axis then means what it says |
| 3 | FPV tables indexed by the normalized mean progress variable `c` in place of `Lambda` | `Lambda` axis with a search for `Lambda(C)` per lookup; `C` axis on a different grid per `(Z, S)` | One lookup path, no iteration on the device; `C` is reproduced exactly (section 4.3) |
| 4 | Mallard's table format: one magic line, one line of JSON (axes, fields, units, provenance), little-endian float64 values, node-major with a node's fields contiguous | HDF5; netCDF; CSV | Readable in every build and with numpy alone; self-describing; one lookup reads 2^d short runs |
| 5 | Own lookup (`src/chemistry/flamelet_table.h`), not `tabulated::Data` | Extend `tabulated` | Up to 4 axes, 64-bit offsets, weights computed once for many fields; `tabulated` serves single-field boundary and source tables |
| 6 | (proposed) A third gas model, `gas = "flamelet"`, whose control variables live in the species block | Control variables beside a mechanism's species; one-way passive scalars | Reuses the species transport, halo exchange, restart and time integration unchanged; the thermodynamics comes from the table (section 5) |
| 7 | (proposed) Compressible FPV thermodynamics after Saghafian et al. (2015): tabulated `T0`, `e0`, `R`, `gamma0`, `a_gamma`, closed form `T(e)` | Tabulated `T(Z, C)` with `p = rho R T` (low-Mach FPV); a Newton solve on tabulated `cp(T)` | Consistent energy with heat release and acoustics; closed form, no iteration; fits the Riemann solvers' `(gamma, e0)` surrogate (chemistry.md, decision 3) |
| 8 | (proposed) Unity Lewis number for all control variables, with the table generated with unity-Lewis transport by default | Differential diffusion in the table | Consistent transport and table; differential diffusion is an option of the generator for later studies |

## 1. Models and control variables

| Model | Flamelets | Control variables (transported) | Table axes | Source |
|---|---|---|---|---|
| SLFM | steady counterflow, stable branch | `Z`, `Zvar` (+ `chi` from a closure) | `Z`, `S`, `chi_st` | none |
| FPV | steady counterflow S-curve (stable, unstable) | `Z`, `Zvar`, `C` | `Z`, `S`, `c` | `omega_C` |
| FGM | free premixed flames over equivalence ratio | `Z`, `Zvar`, `C` (or `c`) | `Z`, `S`, `c` | `omega_C` |
| + heat loss | burner-stabilized or radiating flamelets | + enthalpy `h` | + `h_defect` | |
| + slow species | unsteady flamelets, or a transported `Y_NO` with tabulated production and consumption | + `Y_NO` | as the base model | split source (Ihme & Pitsch 2008c) |

`Z` is Bilger's mixture fraction (Bilger, Stårner & Kee 1990), `Zvar` its
subgrid variance, `S = Zvar / (Z (1 - Z))` the normalized variance in [0, 1],
`C = sum_k w_k Y_k` the progress variable (weights in the table input; `Y_H2O`
for hydrogen, `Y_CO2 + Y_CO + Y_H2O + Y_H2` for hydrocarbons), `Lambda` the
flamelet parameter (`C` at the stoichiometric `Z`, Ihme & Pitsch 2008a), and
`c` the normalized mean progress variable of decision 3.

## 2. Table generation (`tools/setup/flamelet_table.py`)

### 2.1 Input

TOML, with the solver's conventions: paths relative to the input file, unknown
keys are errors, every key has a default except the mechanism, the streams and
the progress variable.

```toml
output = "h2_fpv.mtab"

[mechanism]
file = "../../mechanisms/h2o2.yaml"
phase = "ohmech"
transport = "unity-Lewis-number"   # or "mixture-averaged"

[streams]
pressure = 101325.0
fuel = { T = 300.0, X = { H2 = 0.25, N2 = 0.75 } }
oxidizer = { T = 300.0, X = { O2 = 0.21, N2 = 0.79 } }

[flamelets]
width = 0.01            # nozzle separation (m)
U_ox = 0.1              # oxidizer velocity of the first, most burning flamelet (m/s)
temperature_step = 10.0 # two-point control decrement (K)
T_min = 500.0           # stop below this peak temperature
max_flamelets = 1000

[progress_variable]
H2O = 1.0

[table]
Z_points = 101          # sinh-clustered about Z_st by Z_cluster (4)
S_points = 9            # quadratic in [0, 1]
c_points = 41           # uniform in [0, 1]
lambda_points = 101     # internal Lambda grid
integration_points = 2001
species = ["H2", "O2", "H2O", "OH"]
```

### 2.2 The S-curve

One flamelet is solved at `U_ox` (momentum-balanced jets), then Cantera's
two-point flame control (`two_point_control_enabled`; the continuation of
Nishioka, Law & Takeno 1996, as in Cantera's `diffusion_flame_continuation`
sample and FPVgen) lowers the temperature at two control points on either side
of the peak by `temperature_step` per flamelet. This walks the stable branch
to the turning point (extinction) and continues on the unstable branch, with
the strain rate as an unknown. When a step fails the tool restarts from the
last flamelet with half the step, so the turning point is passed with small
steps and the step grows back afterwards.

The unstable branch of a flame whose streams cannot autoignite ends at
vanishing strain: the mass flux goes to zero and the solver stops. For the H2/N2
test table, the stable branch runs from 1404 K (`U_ox` = 0.1 m/s) to the turning
point at 1161 K (maximum axial velocity gradient 1718 1/s) and the unstable
branch to 1033 K, 28 flamelets in 35 s. Between the last flamelet and the
mixing solution (`Lambda` = 0) the table interpolates linearly in `Lambda`; for
LES of lifted flames the lower part of the unstable branch matters
(Ihme & See 2010), and stage 4 extends it (smaller domain at low strain, or
the ignition branch of preheated streams, which is how the Cabra flame's
vitiated coflow ends it).

### 2.3 The laminar table `phi(Z, Lambda)`

Each flamelet is mapped to `Z` (Bilger, per grid point) and interpolated to a
fine `Z` grid (`integration_points`, clustered about `Z_st`); where a flamelet's
nozzles do not reach `Z = 0` or `1` (diffusion through the inlet at low strain)
it is closed by the pure streams. The flamelets are sorted by `Lambda`, the
mixing solution (frozen adiabatic mixture of the streams) is added at
`Lambda = 0`, and each field is interpolated to a uniform `Lambda` grid at every
`Z`. Fields: `C`, `omega_C` (kg/m^3/s), `T`, `rho`, the mean molar mass `W`,
`cp`, the internal energy `e` (with formation energies), `mu`, `kappa`, the heat
release rate, and the mass fractions of the listed species.

### 2.4 Presumed PDFs (decision 2)

`P(Z, Lambda) = beta(Z; Z_mean, S) delta(Lambda - Lambda_mean)` (Ihme & Pitsch
2008a). For a field piecewise linear in `Z` on the fine grid, the beta mean is a
fixed linear combination of the nodal values:

$$
\int_0^1 \phi(Z) P(Z) \mathrm{d}Z = \sum_i \frac{z_{i+1} \Delta F_i - \Delta G_i}{h_i} \phi_i + \frac{\Delta G_i - z_i \Delta F_i}{h_i} \phi_{i+1},
$$

with `F = I(Z; a, b)` the regularized incomplete beta function, `G = Z_mean
I(Z; a + 1, b)` the partial first moment, `a = Z_mean (1/S - 1)`,
`b = (1 - Z_mean)(1/S - 1)` and `Delta` the change over `[z_i, z_i+1]`. It is
exact, whatever the singularities of the PDF at 0 and 1 where `a` or `b` is
below one, where adaptive quadrature of the PDF reports round-off and agrees
only to about 1e-8 (`S` = 0.8 in the generator's tests). `S = 0` is the laminar value, `S = 1` the two pure
streams `(1 - Z_mean) phi(0) + Z_mean phi(1)`. The weights are computed once per
`(Z_mean, S)` and applied to every field and `Lambda` at once.

Alternatives: beta in `C` too (Ihme & Pitsch 2008a, part 1, found the delta
adequate for Sandia D/E away from extinction); statistically most likely
distributions (Ihme & Pitsch 2008a). Both change only this step.

### 2.5 What is not taken from FPVgen

FPVgen (IhmeGroup/FPVgen) is the reference for the continuation logic. Not
taken: it fills the variance axis with copies of the laminar flamelet instead of
integrating a PDF, has no premixed or heat-loss flamelets, writes only
CharLES^X's layout, and has no tests. An importer for CharLES^X and FlameMaster
tables (writing this format) is optional, later.

## 3. Table format (decision 4)

```
mallard-table 1\n
{"format":"mallard-table","version":1,"model":"fpv","axes":[{"name":"Z","values":[...]},...],
 "fields":[{"name":"C","units":"1"},...],"Z_st":...,"progress_variable":{...},"streams":{...},
 "flamelets":{...},"provenance":{"tool":...,"cantera":"3.2.0","created":...,
 "mechanism":{"file":...,"sha256":...},"input":"<the table input, verbatim>"}}\n
<little-endian float64 values>
```

- The first line names the format and its version; a reader refuses any other.
- The header is one line of JSON: the axes (name, strictly increasing nodes),
  the fields (name, UDUNITS units), the model, and what produced the table: the
  tool, Cantera's version, the mechanism's SHA-256, the time, and the table
  input verbatim, so a table regenerates from its own header.
- The values follow: node-major, the last axis fastest, the fields of a node
  contiguous. A lookup of `n` fields in `d` dimensions reads `2^d` runs of `n`
  values.
- Readers check the magic, the version, the axes, the value count and that
  every value is finite, and name the file and the problem in their errors.
- Why not HDF5: Mallard builds without it (`Mallard_ENABLE_HDF5` is optional),
  and a table must be readable by every build and by numpy alone. Tables are at
  most a few hundred MB (200 x 30 x 100 nodes x 40 fields in float64 is 190 MB),
  read once at setup.
- Values are float64. A float32 variant (half the bandwidth and memory) is a
  version-2 question for when lookups show up in profiles.

## 4. Lookup on the device

### 4.1 Multilinear lookup (decision 5)

`flamelet::TableView` is a plain aggregate of two rank-1 Views (axis nodes,
values, the latter indexed in 64 bits) and the axis sizes, captured by value in
kernels. `locate(q)` finds each coordinate's cell by bisection on its axis
(non-uniform axes; clamped to the ends, NaN to the first node) and the fraction
across it; `evaluate(point, fields, n, out)` sums the `2^d` corners with their
weights once for all requested fields. The order of the sum is fixed, so the
result depends only on the query: bitwise identical across backends, thread
counts and ranks, and in captured steps.

`tabulated::Data` (boundary and source tables) is not reused: it holds at most
three coordinates with 32-bit offsets and recomputes the weights for each field,
and a flamelet lookup evaluates ten or more fields per cell in every stage.

### 4.2 Memory and layout

The table is replicated on every rank and GPU (190 MB for the large example
above; an A100 has 80 GB). Node-major with fields contiguous makes one lookup
read `2^d n_fields` values in `2^d` contiguous runs; field-major would make it
`2^d n_fields` scattered reads. Neighboring threads look up different points in
either layout. To be measured in stage 2 against a field-major layout and
float32 values.

### 4.3 The FPV map (decision 3)

The transported `(Z, Zvar, C)` is mapped to the table's `(Z, S, c)`:

1. `S = Zvar / (Z (1 - Z))`, clamped to [0, 1] (0 where `Z (1 - Z)` < 1e-12).
2. `C_min` and `C_max`: the table's `C` at `(Z, S, c = 0)` and `(Z, S, c = 1)`.
3. `c = (C - C_min) / (C_max - C_min)`, clamped to [0, 1] (0 where the span
   vanishes: at `Z = 0`, `Z = 1` and `S = 1` every `Lambda` gives the same
   state).

The generator tabulates `C` exactly linear in `c` at each `(Z, S)` node, so the
multilinear lookup of `C` returns the transported `C` itself (to round-off)
wherever it lies within the bounds: the table cannot drift from the transported
variable it is indexed by. Three lookups of one field and one of all the fields
needed; no iteration. The `Lambda` alternative needs a search (bisection or
Newton) for `Lambda(Z, S, C)` per lookup, with a data-dependent iteration count
on the device; tabulating `C` on a different grid per `(Z, S)` needs the same
search in disguise.

`C` must increase with `Lambda` at each `(Z_mean, S)` for `c` to be a coordinate.
`Lambda` is chosen because it is monotonic along the S-curve at `Z_st`; away
from it the generator checks monotonicity after integration and follows the
running maximum where it fails, reporting how many nodes did. In the test table
none do (the only non-monotonic nodes are those where `C` does not vary at all).

## 5. Transported scalars and coupling with the flow (stage 2, proposed)

### 5.1 Equations

Favre-filtered, with the LES closures of [les.md](les.md):

$$
\partial_t (\bar\rho \tilde Z) + \nabla \cdot (\bar\rho \tilde u \tilde Z) = \nabla \cdot ((\bar\rho D + \bar\rho D_t) \nabla \tilde Z)
$$

$$
\partial_t (\bar\rho \tilde C) + \nabla \cdot (\bar\rho \tilde u \tilde C) = \nabla \cdot ((\bar\rho D + \bar\rho D_t) \nabla \tilde C) + \tilde{\dot\omega}_C
$$

with `rho D = kappa / cp` from the table (unity Lewis number, decision 8),
`rho D_t = mu_t / Sc_t` from the eddy viscosity, and `omega_C` from the table.
The variance has two closures, selected in the input:

- **algebraic** (default): `Zvar = C_Z Delta^2 |grad Z|^2` (Pierce & Moin
  1998), `C_Z = 0.1` or dynamic; no extra equation;
- **transported second moment** `rho Z2` (stage 3): conserved, so it is
  transported like `Z` with the sink `-rho chi_sgs`, `chi_sgs = C_chi (D + D_t)
  Zvar / Delta^2`; `Zvar = Z2 - Z^2`.
  Transporting `Z2` rather than `Zvar` avoids the production term's
  dependence on the scheme (Kaul et al. 2013).

Without `[les]` (laminar flamelet runs, the verification cases) `Zvar = 0`.

### 5.2 Thermodynamics (decision 7)

The energy equation is unchanged (`rho E`, formation energies included). The
table holds, per node, `T0` (the flamelet temperature), `e0 = e(T0)`, `R`,
`gamma0` and `a_gamma = d gamma / d T` at `T0` (Saghafian, Terrapon & Pitsch
2015). With `gamma(T) = gamma0 + a_gamma (T - T0)`, `cv = R / (gamma - 1)` and
`e(T) = e0 + int_T0^T cv dT`:

$$
T = T_0 + \frac{\gamma_0 - 1}{a_\gamma} \left( \exp\left( \frac{a_\gamma (e - e_0)}{R} \right) - 1 \right), \qquad p = \rho R T,
$$

in closed form, valid for compressible heating and cooling around the
flamelet state (shocks, acoustics) where tabulated `T(Z, C)` alone would
ignore the energy equation. The Riemann solvers' surrogate `(gamma, e0')`
(chemistry.md, decision 3) is `gamma = gamma(T)`, `e0' = e - p / (rho (gamma -
1))`, so the flux kernels are unchanged.

### 5.3 Where the control variables live (decision 6, open question 1)

**Proposed:** `gas = "flamelet"`, a third gas model beside the perfect gas and
`Mixture`, whose state is `[rho, rho u, rho E]` plus the species block holding
`[rho Z, rho C]` (and `rho Z2`, `rho h` when transported). The species block
already has everything a transported scalar needs: storage beside the flow
block, the time integrator and `axpby`, halo exchange, restart, conservation
sums, linear reconstruction with one shared limiter, Larrouturou mass-flux
upwinding (which keeps a scalar between its neighbors' values when `rho` does
not vary), boundary values at inlets, viscous fluxes with per-scalar
diffusivities and the SGS diffusivity. What changes is confined to the gas
model:

- `cell_state` (W, `gamma`, `e0`, T from the conservatives): the FPV map and the
  closed-form `T(e)` above, in place of the NASA polynomials and the Newton
  iteration on `T`;
- transport: `mu`, `kappa` and `rho D = kappa / cp` from the same lookup, in
  place of the mixture-averaged model; one diffusivity for all scalars;
- the chemistry step: `omega_C` from the table as an explicit source in the RHS
  in place of the Strang-split reactors (its time scale is that of the resolved
  flame, not of the fastest radicals);
- output: `T`, `Y_<species>`, `HRR` and any table field as derived fields.

**Alternatives.** (a) Control variables beside a mechanism's species
(`gas = "mixture"` with extra columns, species set from the table): exact NASA
thermodynamics, but it keeps the species memory and bandwidth that tabulation is
meant to save, and the transported species and the table disagree. (b) One-way
passive scalars on any gas model: no feedback on density and temperature, so not
a combustion model; useful only as a verification step. (c) A new scalar block
in `State` beside the species: cleaner naming, but duplicates every loop the
species block already has. The open question is how the mixture path's
functors become generic over the gas model: templated on it (as the RHS already
is over `PerfectGas` / `Mixture`) or dispatched once per stage.

### 5.4 Boundary and initial values

Inlets give `Z` (1 for fuel, 0 for oxidizer, or any mixture) and `C` (0 for
fresh streams, or the burnt value for a pilot: `C_max` at that `Z`), instead of
species; walls are zero-flux for all scalars. Initial states use the same keys,
through the existing expression machinery (`Z = "x < 0 ? 1 : 0"`). A pilot or an
equilibrium region is `C = "burnt"`, resolved through the table.

## 6. Interaction with the existing models

- **Finite-rate chemistry:** exclusive. `gas = "flamelet"` loads no mechanism
  into the solver (the table carries the species it outputs); `[chemistry]` is
  an input error with it.
- **LES:** the eddy viscosity enters `rho D_t = mu_t / Sc_t` of every scalar
  and the variance closure through the filter width `Delta` (les.md, section 3).
  The kinetic-energy budget and the low-dissipation flux are independent of the
  gas model.
- **TFLES:** not combined with FPV (the table already represents the subgrid
  flame). For FGM in premixed LES on meshes that do not resolve the tabulated
  flame, thickening the progress-variable equation (`D -> E F D`,
  `omega_C -> E omega_C / F`) is the standard combination (e.g. Kuenne et al.
  2011); it reuses `ThickenedFlame` and its dynamic efficiency unchanged, with
  the sensor on `c` instead of `T`. Stage 5.
- **PaSR:** not applicable (it scales finite-rate rates).
- **Radiation:** the optically thin model needs `T` and `X_H2O`, `X_CO2` from the
  table; it then removes enthalpy, which needs the enthalpy-defect axis (stage
  6). Until then radiation is an input error with `gas = "flamelet"`.
- **Double flux:** applies unchanged to the surrogate `(gamma, e0)`.
- **Output, probes, statistics:** table fields are derived fields
  (`src/solver/derived_catalog.h`): `T`, `Y_<species>` of the table's species,
  `HRR`, `omega_C`, `Z`, `Zvar`, `C`, `c`; probes and statistics read them
  as they read the mixture's.
- **Determinism and step graphs:** lookups are pointwise kernels without host
  work; no new reduction or exchange.

## 7. Input keys (stage 2, proposed)

```toml
[physics]
gas = "flamelet"
table = "h2_fpv.mtab"        # relative to the input file
variance = "algebraic"       # or "transported" (stage 3); ignored without [les]
C_Z = 0.1                    # algebraic variance constant

[[boundary]]
name = "fuel"
type = "inflow"
Z = 1.0
C = 0.0
```

`model` comes from the table's header (`fpv`, `fgm`, `slfm`); the solver checks
that the table's pressure matches the run's reference pressure within 10% and
warns otherwise (flamelet tables are generated at one pressure; the
pressure-dependent table axis is a stage-6 option).

## 8. Performance

Per cell and RHS: one FPV map (three single-field lookups) and one lookup of
about eight fields (`T0`, `e0`, `R`, `gamma0`, `a_gamma`, `mu`, `kappa`, `cp`,
`omega_C`), each `2^3` corners: about 100 values read through the cache. The
state per cell is 6 values in 3D instead of `5 + N_s` (58 for GRI-3.0), and
there is no stiff chemistry step. The expected cost is close to that of a
non-reacting run; stage 2 measures it against a finite-rate run of the same
case.

## 9. Tests

Each must fail under a plausible bug (CONTRIBUTING.md).

- **Generator (Python, `test/python/test_flamelet_table.py`)**: beta weights
  against an independent integration by parts with the beta CDF for profiles
  with a sharp peak at `Z_st`, at singular and regular PDFs; the `S = 0`,
  `S = 1` and `Z = 0, 1` limits and continuity towards them; the `Lambda -> c`
  remap reproduces `C` linearly and fields that are functions of `C`; the file
  round trip; the repository table regenerates from its header's input and is
  monotonic in `c`. Needs scipy (skipped without it).
- **Lookup (C++, `test/flamelet_table_test.cpp`)**: multilinear functions of
  four uneven axes reproduced to round-off in a device kernel, and clamped
  outside; the FPV map returns the transported `C` within its bounds and
  clamps outside; malformed files (truncated, wrong version, NaN, wrong axes)
  refused with the file and reason.
- **Verification against Cantera (C++)**: a counterflow flamelet at
  `U_ox` = 1.5 m/s, which is not one of the table's, looked up from its own `Z`
  and `C` through the device FPV map on the 41 x 5 x 21 test table:
  `T` within 4.6 K (peak 1256 K), `Y_OH` within 7% and `omega_C` within 3% of
  their peaks (tolerances twice these).
- **Stage 2**: the gas model's `cell_state` against the table's `T0`, `p` at
  the flamelet states and the closed-form `T(e)` against a quadrature of
  `cv(T)`; transport of `Z` with constant diffusivity against the exact error
  function solution; a 2D counterflow (below) against Cantera.

## 10. Validation

| Case | Model | Reference | Target |
|---|---|---|---|
| Table flamelets and off-table flamelets (stage 1) | FPV | Cantera counterflow, unity Lewis | `T` within 0.5% of the peak on the test table |
| PDF integration (stage 1) | all | direct integration of the beta PDF | 1e-9 relative |
| FPVgen tables on the same flamelets | FPV | FPVgen (laminar part) | node values within interpolation error |
| 2D laminar counterflow, `examples/counterflow_diffusion` | FPV | Cantera (as `counterflow_reference.py --match`), Mallard finite-rate on the same mesh | `T_max`, flame position and `Y_OH` peak within 2% of Cantera; same extinction strain within 5% |
| Sandia flame D (Barlow & Frank 1998) | FPV, SLFM | TNF data: mean and rms `Z`, `T`, `Y_k` profiles at x/D = 7.5 to 45 | within the scatter of the TNF FPV results (Ihme & Pitsch 2008b) |
| Cabra lifted H2/N2 flame (Cabra et al. 2002) | FPV | lift-off height, centerline and radial `Z`, `T` | lift-off within 20% of the measured height; the methane variant against Ihme & See (2010) |
| Premixed jet or Volvo rig (#197, #278) | FGM | Volvo (Sjunnesson et al.), as les.md section 8.3 | mean velocity and `T` profiles; comparison with TFLES |
| Shared case: flame in decaying turbulence (`examples/flame_turbulence`) | FGM vs finite-rate + TFLES | each other, DNS | turbulent consumption speed within 10% |

## 11. Stages

1. **Table generator, format and device lookup** (this change): the FPV
   generator, the format, `flamelet::Table` / `TableView` / `FPV`, and their
   tests and verification against Cantera. No solver change; no input key.
2. **Laminar FPV in the solver**: `gas = "flamelet"` with transported `Z` and
   `C`, compressible thermodynamics (table fields `T0`, `e0`, `R`, `gamma0`,
   `a_gamma` added by the generator), table transport, `omega_C`, inlet values,
   derived fields; the 2D counterflow against Cantera and Mallard's finite-rate
   run. Device code: GPU CI.
3. **LES closures**: algebraic and transported variance, SGS scalar fluxes;
   Sandia flame D.
4. **Unstable branch and lifted flames**: the low-strain end of the unstable
   branch, preheated streams; the Cabra flame.
5. **FGM**: free premixed flamelets over `Z`, the FGM table model, optional
   thickening with TFLES; Volvo or the premixed jet; the shared case against
   finite-rate + TFLES.
6. **Heat loss and slow chemistry**: enthalpy-defect axis from burner-stabilized
   or radiating flamelets, radiation; `Y_NO` with tabulated production and
   consumption, unsteady flamelets (#280); pressure axis.
7. **Importers** (optional): CharLES^X and FlameMaster tables.

## 12. Open questions

1. **Gas-model structure (decision 6).** A third gas model whose control
   variables use the species block, with the mixture path's functors templated
   on the gas model, is proposed. Is reusing the species block (and its naming:
   `RHOY_Z`, `RHOY_C` in restarts) acceptable, or should control variables get
   their own block in `State` (alternative c)?
2. **Thermodynamics.** Saghafian's linearized `gamma(T)` is proposed for the
   compressible coupling. Low-Mach FPV (tabulated `rho` and `T`, pressure from
   the EOS only) is simpler but inconsistent with Mallard's compressible
   energy equation; is there a case that needs the pressure axis earlier?
3. **Progress-variable source.** Explicit in the RHS is proposed. FPV sources
   near extinction are stiff on fine meshes; a Strang-split implicit update of
   `C` alone (one scalar ODE per cell, through the existing chemistry step) is
   the fallback.
4. **Table precision.** float64 now; float32 values as a format option when
   lookups are measured to matter.

## References

- R. W. Bilger, S. H. Stårner and R. J. Kee, On reduced mechanisms for methane-air combustion in nonpremixed flames, *Combust. Flame* 80, 135–149 (1990).
- M. Nishioka, C. K. Law and T. Takeno, A flame-controlling continuation method for generating S-curve responses with detailed chemistry, *Combust. Flame* 104, 328–342 (1996).
- N. Peters, Laminar diffusion flamelet models in non-premixed turbulent combustion, *Prog. Energy Combust. Sci.* 10, 319–339 (1984).
- C. D. Pierce and P. Moin, A dynamic model for subgrid-scale variance and dissipation rate of a conserved scalar, *Phys. Fluids* 10, 3041–3044 (1998).
- C. D. Pierce and P. Moin, Progress-variable approach for large-eddy simulation of non-premixed turbulent combustion, *J. Fluid Mech.* 504, 73–97 (2004).
- M. Ihme and H. Pitsch, Prediction of extinction and reignition in nonpremixed turbulent flames using a flamelet/progress variable model. 1. A priori study and presumed PDF closure, *Combust. Flame* 155, 70–89 (2008a).
- M. Ihme and H. Pitsch, Prediction of extinction and reignition in nonpremixed turbulent flames using a flamelet/progress variable model. 2. Application in LES of Sandia flames D and E, *Combust. Flame* 155, 90–107 (2008b).
- M. Ihme and H. Pitsch, Modeling of radiation and nitric oxide formation in turbulent nonpremixed flames using a flamelet/progress variable formulation, *Phys. Fluids* 20, 055110 (2008c).
- M. Ihme and Y. C. See, Prediction of autoignition in a lifted methane/air flame using an unsteady flamelet/progress variable model, *Combust. Flame* 157, 1850–1862 (2010).
- A. Saghafian, V. E. Terrapon and H. Pitsch, An efficient flamelet-based combustion model for compressible flows, *Combust. Flame* 162, 652–667 (2015).
- J. A. van Oijen and L. P. H. de Goey, *Combust. Sci. Technol.* 161, 113–137 (2000): flamelet-generated manifolds (FGM) for premixed flames.
- O. Gicquel, N. Darabiha and D. Thévenin, Laminar premixed hydrogen/air counterflow flame simulations using flame prolongation of ILDM with differential diffusion, *Proc. Combust. Inst.* 28, 1901–1908 (2000).
- C. M. Kaul, V. Raman, E. Knudsen, E. S. Richardson and J. H. Chen, Large eddy simulation of a lifted ethylene flame using a dynamic nonequilibrium model for subfilter scalar variance and dissipation rate, *Proc. Combust. Inst.* 34, 1289–1297 (2013).
- G. Kuenne, A. Ketelheun and J. Janicka, LES modeling of premixed combustion using a thickened flame approach coupled with FGM tabulated chemistry, *Combust. Flame* 158, 1750–1767 (2011).
- R. S. Barlow and J. H. Frank, Effects of turbulence on species mass fractions in methane/air jet flames, *Proc. Combust. Inst.* 27, 1087–1095 (1998).
- R. Cabra, T. Myhrvold, J. Y. Chen, R. W. Dibble, A. N. Karpetis and R. S. Barlow, Simultaneous laser Raman-Rayleigh-LIF measurements and numerical modeling results of a lifted turbulent H2/N2 jet flame in a vitiated coflow, *Proc. Combust. Inst.* 29, 1881–1888 (2002).
