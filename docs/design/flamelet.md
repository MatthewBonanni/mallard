# Tabulated chemistry: flamelets, FPV and FGM

Status: proposed (#631). [Stage](#11-stages) 1 is this design, the table
format ([table_format.md](table_format.md), the contract with table
generators) and Mallard's reader and device lookup. Mallard does not generate
tables. Tables come from an external generator library, **`<generator, name
TBD>`**: a separate project that starts from FPVgen (IhmeGroup/FPVgen), credited
in its README. The [open questions](#12-open-questions) of the first draft are
resolved. Stage 2 couples the tables to the solver; its laminar part is built
([section 5.5](#55-stage-2-as-built)); the LES closures and the counterflow
validation are next.

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
- **Table generation outside Mallard.** The generator library
  (`<generator, name TBD>`, starting from FPVgen) drives Cantera's counterflow
  and free flames, integrates the presumed PDFs and writes the table. It will
  grow to FGM/FPI, heat loss, several streams and other thermodynamics.
  Mallard's side is the format and the reader.
- **A generic, versioned table format with provenance**
  ([table_format.md](table_format.md)): N axes of named control variables,
  named fields, a species block, model and thermodynamics tags, in HDF5:
  read by every build with HDF5 and by h5py.
- **Device lookup** in the RHS and the output kernels: multilinear, a few cache
  lines per lookup, bitwise independent of the backend, thread and rank count.
- **Ignorable when off.** A run without `gas = "flamelet"` compiles to and runs
  the code it runs today, byte-identical; the model's code sits in its own files
  and is reached from a few well-defined points.
- **Validation** against the table's own flamelets and independent Cantera
  solutions (in the generator), Mallard's finite-rate solutions of the same
  laminar flames, and the standard turbulent flames (section 10).

## Non-goals

- Table generation in Mallard: no generator in `tools/`. Generation, its
  tests and its validation against Cantera live in the generator library.
- A flamelet solver of our own, interactive flamelet models (RIF), CMC,
  transported PDFs, linear-eddy models.
- A Strang-split or implicit update of `C`: `omega_C` is an explicit source
  (decision 9).
- Tables with more than four axes, adaptive or sparse tables, neural-network
  regression of tables. The format leaves room for them.
- Multiphase and spray flamelets.
- Replacing finite-rate chemistry: both stay, and the [shared
  case](#10-validation) compares them.

## Summary of decisions

| # | Decision | Main alternatives | Why |
|---|---|---|---|
| 1 | Tables generated offline by an external library, `<generator, name TBD>`, that starts from FPVgen and runs Cantera's 1D flames (counterflow S-curves by two-point flame control; free flames for FGM) | A generator in Mallard's `tools/`; FlameMaster; a flamelet solver in Mallard | Generation is a separate product with its own users, tests and release cycle, and it will cover more models than Mallard reads. Mallard keeps only the contract |
| 2 | Presumed PDFs integrated by the generator: beta in Z, exact for piecewise-linear profiles through the incomplete beta function; delta in the progress parameter | Quadrature of the PDF (singular at 0 and 1); copies of the laminar flamelet along the variance axis, as FPVgen does today | Exact, no quadrature parameter, no singularity; the variance axis then means what it says |
| 3 | FPV tables indexed by the normalized mean progress variable `Lambda` (FPVgen's), in place of the flamelet label `C_st = C(Z_st)`, with `C` linear in `Lambda` (a checked part of the contract) | `C_st` axis with a search for `C_st(C)` per lookup; `C` axis on a different grid per `(Z, Sz)` | One lookup path, no iteration on the device; `C` is reproduced exactly (section 4.3) |
| 4 | Mallard's table format in HDF5 ([table_format.md](table_format.md)): tags, species and pressure as root attributes, named axes and fields with units, model data and provenance as groups, and the values in one contiguous, unfiltered dataset, node-major with a node's fields contiguous, read into memory on CPU backends and mapped from the file to stage the copy on GPU backends | A custom binary format (magic line, JSON header, raw values); netCDF; CSV; CharLES^X's HDF5 layout (FPVgen's output today) | Standard, self-describing, readable with h5py alone; generic over models; one lookup reads 2^d short runs. HDF5 is optional in Mallard, but most CI jobs, the GPU suites, the containers and Spack have it, and it builds from its submodule |
| 5 | Own lookup (`src/chemistry/flamelet_table.h`), not `tabulated::Data` | Extend `tabulated` | Any number of axes, fixed at compile time per model so the corner loop unrolls; 64-bit offsets; weights computed once for many fields; `tabulated` serves single-field boundary and source tables |
| 6 | A third gas model, `gas = "flamelet"`, whose `Z` and `C` live in the existing species block, with the mixture path's functors templated on the gas model | A separate scalar block in `State`; control variables beside a mechanism's species; one-way passive scalars | Reuses the species transport, halo exchange, restart and time integration unchanged; the thermodynamics comes from the table (section 5) |
| 7 | Compressible FPV thermodynamics after Saghafian et al. (2015): tabulated `T0`, `e0`, `R`, `gamma0`, `a_gamma`, closed form `T(e)` (thermodynamics tag `saghafian2015`) | Low-Mach FPV (tabulated `T`, `rho`); a Newton solve on tabulated `cp(T)` | Consistent energy with heat release and acoustics; closed form, no iteration; fits the Riemann solvers' `(gamma, e0)` surrogate (chemistry.md, decision 3) |
| 8 | Unity Lewis number for all control variables, `rho D = kappa / cp`, with tables generated with unity-Lewis transport by default | Differential diffusion in the table | Consistent transport and table; differential diffusion is a generator option for later studies |
| 9 | `omega_C` an explicit source in the RHS | A Strang-split implicit update of `C` | The FPV progress source is not stiff: its time scale is the resolved flame's, not the fastest radicals' |
| 10 | Table values in float64 or float32, lookups summed in double | float64 only | float32 halves a table's memory and lookup bandwidth at a relative error of 6e-8, for when lookups show up in profiles; float64 stays the default and is exact |

## 1. Models and control variables

| Model | Flamelets | Control variables (transported) | Table axes | Source |
|---|---|---|---|---|
| SLFM | steady counterflow, stable branch | `Z`, `Zvar` (+ `chi` from a closure) | `Z`, `Sz`, `chi_st` | none |
| FPV | steady counterflow S-curve (stable, unstable) | `Z`, `Zvar`, `C` | `Z`, `Sz`, `Lambda` | `omega_C` |
| FGM | free premixed flames over equivalence ratio | `Z`, `Zvar`, `C` (or `Lambda`) | `Z`, `Sz`, `Lambda` | `omega_C` |
| + heat loss | burner-stabilized or radiating flamelets | + enthalpy `h` | + `h_defect` | |
| + slow species | unsteady flamelets, or a transported `Y_NO` with tabulated production and consumption | + `Y_NO` | as the base model | split source (Ihme & Pitsch 2008c) |

`Z` is Bilger's mixture fraction (Bilger, Stårner & Kee 1990), `Zvar` its
subgrid variance, `Sz = Zvar / (Z (1 - Z))` the normalized variance in [0, 1],
`C = sum_k w_k Y_k` the progress variable (weights in the table input; `Y_H2O`
for hydrogen, `Y_CO2 + Y_CO + Y_H2O + Y_H2` for hydrocarbons), `C_st` the
flamelet parameter (`C` at the stoichiometric `Z`), and
`Lambda = (C - C_min) / (C_max - C_min)` the normalized mean progress variable
of decision 3, between the least and largest `C` of the table at that
`(Z, Sz)`.

The names are FPVgen's: `Sz` is its normalized variance axis and `Lambda` its
normalized progress variable (its field `PROG_NORM`, plotted as Λ; FPVgen's
CharLES^X tables name that axis `C`). FPVgen normalizes each laminar
flamelet's `C` by the least and largest `C` of all flamelets at the same `Z`;
Mallard's `Lambda` is the same normalization at each `(Z, Sz)` after PDF
integration (at `Sz = 0`, of the laminar flamelets). Ihme & Pitsch (2008a) write Λ for
the flamelet parameter `C(Z_st)`, which is `C_st` here: it labels flamelets
in generation and is not a table axis.

## 2. Table generation (external)

Mallard does not generate tables. The generator library, **`<generator, name
TBD>`**, is a separate project that starts from FPVgen (IhmeGroup/FPVgen, MIT
License), not a GitHub fork of it, and credits FPVgen in its README. It writes
[Mallard's table format](table_format.md). Mallard's tests read a small
synthetic fixture in that format and do not need the generator. Generation is
verified in the generator's own tests, against Cantera.

What Mallard needs from the generator, for `fpv` tables:

- **The S-curve.** One counterflow flamelet is solved, then Cantera's two-point
  flame control (the continuation of Nishioka, Law & Takeno 1996, as in
  Cantera's `diffusion_flame_continuation` sample and FPVgen) walks the stable
  branch to extinction and continues on the unstable branch. The mixing
  solution closes the table at `C_st = 0`. For LES of lifted flames the lower
  part of the unstable branch matters (Ihme & See 2010).
- **The laminar table `phi(Z, C_st)`.** Each flamelet is mapped to Bilger's
  `Z` and labeled by `C_st = C(Z_st)` (Ihme & Pitsch 2008a).
- **Presumed PDFs (decision 2).** `P(Z, C_st) = beta(Z; Z_mean, Sz)
  delta(C_st - C_st_mean)`. For a field piecewise linear in `Z` on a fine
  grid, the beta mean is a fixed linear combination of the nodal values:

    $$
    \int_0^1 \phi(Z) P(Z) \mathrm{d}Z = \sum_i \frac{z_{i+1} \Delta F_i - \Delta G_i}{h_i} \phi_i + \frac{\Delta G_i - z_i \Delta F_i}{h_i} \phi_{i+1},
    $$

    with `F = I(Z; a, b)` the regularized incomplete beta function, `G = Z_mean
    I(Z; a + 1, b)` the partial first moment, `a = Z_mean (1/Sz - 1)`,
    `b = (1 - Z_mean)(1/Sz - 1)` and `Delta` the change over `[z_i, z_i+1]`. It is
    exact for any singularity of the PDF at 0 and 1. `Sz = 0` is the laminar
    value, and `Sz = 1` gives the two pure streams. FPVgen today fills the
    variance axis with copies of the laminar flamelet; the generator replaces
    that with this integration. A first implementation, verified against an
    independent integration by parts, is in the history of #1222 and can seed
    the generator.

- **The `C_st -> Lambda` remap (decision 3).** After integration, each `(Z, Sz)`
  column is retabulated against `Lambda`, with `C` exactly linear in `Lambda` (a checked
  part of the format). `C` must increase with `C_st` at each `(Z, Sz)`. Where
  integration breaks monotonicity, the generator follows the running maximum
  and reports how many nodes did.
- **The fields of the contract**: `C`, `omega_C`, `rho0`, the Saghafian
  coefficients `T0`, `e0`, `R`, `gamma0`, `a_gamma`, transport `mu`, `kappa`,
  `cp`, and the species block. FPVgen already computes the Saghafian
  coefficients and power-law transport for CharLES^X. The generator writes
  them in this format, together with the provenance.

`tools/docs/flamelet.py` follows these steps to draw the documentation's
figures, and writes no table. Its sample is Cantera flamelets of H2/N2 (1:1 by
mole) against air at 300 K and 1 atm, `mechanisms/h2o2.yaml`, unity Lewis
number, `C = Y_H2O`. The control temperatures march down. Where that stalls
(near `T_max` = 1025 K), they march up, and every solution whose `C_st` falls
is kept. The stable branch runs from `chi_st` = 0.32 1/s to extinction at
2777 1/s and `T_max` = 1246 K. The unstable branch runs back to 0.52 1/s and
levels off at `T_max` = 986 K and `C_st` = 0.059 (`Lambda` = 0.34 at `Z_st`).
Below that, the table interpolates linearly to the mixing solution.

![S-curve of the sample table's counterflow flamelets: peak temperature and source of C against the stoichiometric scalar dissipation rate, with the stable, unstable and extinguished branches, and the temperature of flamelets of each branch against the mixture fraction](../images/guide/flamelet_s_curve-light.svg#gh-light-mode-only)
![S-curve of the sample table's counterflow flamelets: peak temperature and source of C against the stoichiometric scalar dissipation rate, with the stable, unstable and extinguished branches, and the temperature of flamelets of each branch against the mixture fraction](../images/guide/flamelet_s_curve-dark.svg#gh-dark-mode-only)

![Temperature, source of C and OH mass fraction of the sample table at the stoichiometric mixture fraction against Lambda, with the nodes from the stable and the unstable branch and the interpolated range below them](../images/guide/flamelet_progress-light.svg#gh-light-mode-only)
![Temperature, source of C and OH mass fraction of the sample table at the stoichiometric mixture fraction against Lambda, with the nodes from the stable and the unstable branch and the interpolated range below them](../images/guide/flamelet_progress-dark.svg#gh-dark-mode-only)

The beta integration of the same flamelets, at the most burning `Lambda` and
at the `Lambda` of the largest source:

![Temperature at Lambda = 1 and source of C at Lambda = 0.7 of the sample table against the mean mixture fraction, at 5 normalized variances](../images/guide/flamelet_variance-light.svg#gh-light-mode-only)
![Temperature at Lambda = 1 and source of C at Lambda = 0.7 of the sample table against the mean mixture fraction, at 5 normalized variances](../images/guide/flamelet_variance-dark.svg#gh-dark-mode-only)

Later models (FGM/FPI from free premixed flames, heat-loss axes, several
streams, other thermodynamics) are added to the generator first, and to the
format as new tags ([table_format.md](table_format.md#8-versioning-and-new-models)).

## 3. Table format (decision 4)

The specification is [table_format.md](table_format.md). In short:

- An HDF5 file: the tags, species and pressure as attributes of the root,
  each axis as a float64 dataset `/axes/<name>` with its units, and the values
  as one dataset `/values` of shape `(n_0, ..., n_(d-1), n_fields)`, float64
  or float32, contiguous and unfiltered, whose attributes name its axes,
  fields and units. The data is node-major with the last axis fastest, and the
  fields of a node are contiguous: a lookup of `n` fields in `d` dimensions
  reads `2^d` runs of `n` values.
- The format is generic. It has N axes of named control variables, named
  fields with units, and a species block (`Y_k` fields). A **model tag**
  (`fpv`) sets the axes and the model's fields. A **thermodynamics tag**
  (`saghafian2015`) sets the fields the equation of state reads. FGM/FPI, heat
  loss, several streams and other thermodynamics are new tags in the same
  format. Mallard refuses tags it does not support, by name.
- Provenance (generator and version, creation time, mechanism SHA-256, the
  generator's input verbatim) is required, so that a table can be regenerated
  from its own file.
- Mallard checks the format and the contract of the table's tags at setup:
  the format and version, the required attributes, groups and datasets, the
  axes, the units of every field it uses, physical bounds, `C` linear in `Lambda`,
  the shape and finiteness. Its errors name the file and the problem.
- Loading reads the values into memory (`H5Dread`) on CPU backends. On GPU
  backends, where the host copy only stages the copy to the device, it maps
  them from the file (`H5Dget_offset`, then `mmap`) when the dataset is
  contiguous, unfiltered, in native byte order and aligned. Both give bitwise
  identical values. Mapping on CPU backends would keep one copy per node in
  the page cache instead of one per rank, but lookups on page-cache pages ran
  about 20% slower than on the read copy, which gets transparent huge pages;
  page-cache and shared-memory pages get none on typical kernels
  ([table_format.md](table_format.md#2-file-layout)).

**Why HDF5 and not a custom format.** The first version of this design used
a custom format (a magic line, one line of JSON, raw little-endian float64),
so that every build could read tables, HDF5 or not. HDF5 was chosen instead:

- *Builds without HDF5.* Tabulated chemistry then needs HDF5, and a build
  without it refuses a table with an input error that says how to rebuild.
  That costs little: HDF5 is in 10 of CI's 18 backend jobs (`openmp`, `mpi`,
  `mpi-3d`, `float-3d` of the gate; `serial-3d`, `clang-22` and four arm64
  jobs), in the GPU suites (CUDA with a parallel HDF5), in the containers and
  in Spack, and `-DUSE_SYSTEM_HDF5=OFF` builds it from its submodule wherever
  a system HDF5 is missing. The jobs without it (`serial`, `float`, `gcc-16`,
  `clang-23`, `clang-22-arm64`, macOS) test the refusal.
- *Device builds.* None: tables are read on the host at setup and copied to
  the device; HDF5 never appears in device code. The CUDA, HIP and SYCL
  compile jobs build without HDF5, and the GPU suites with it.
- *Load speed.* Equal or better: the values are one contiguous dataset, so
  reading them is one `H5Dread` at disk speed (measurements in the PR). The metadata are a few
  dozen attributes.
- *Gains.* A standard container that h5py, HDF5's tools (`h5dump`, `h5ls`)
  and other codes read; attributes and groups instead of a JSON parser;
  float32 in the same version; room for more datasets (for example
  precomputed bounds) without a new layout.

## 4. Lookup on the device

### 4.1 Multilinear lookup (decision 5)

`flamelet::TableView<D>` is a plain aggregate of two rank-1 Views (axis nodes,
values, the latter indexed in 64 bits), the axis sizes and strides, captured by
value in kernels. The reader takes any number of axes; the lookup's dimension
`D` is a template parameter, set by the model (3 for `fpv`), so the `2^D`
corner loop unrolls and a point's `D` cells and fractions stay in registers on
GPUs. `locate(q)` finds each coordinate's cell by bisection on its axis
(non-uniform axes; clamped to the ends, NaN to the first node) and the fraction
across it. The bisection's trip count depends on the axis alone and its steps
are selects, not branches on the query, so neighboring threads do not diverge.
`evaluate(point, fields, n, out)` sums the `2^D` corners with their weights once
for all requested fields, without branches on the weights. The order of the sum is fixed, so the
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
either layout. To be measured in stage 2 against a field-major layout, and
float32 values against float64. The FPV map reads the bounds of `C` at the ends of the `Lambda` axis
from the `2^(D-1)` nodes of the `(Z, Sz)` cell alone.

### 4.3 The FPV map (decision 3)

The transported `(Z, Zvar, C)` is mapped to the table's `(Z, Sz, Lambda)`:

1. `Sz = Zvar / (Z (1 - Z))`, clamped to [0, 1] (0 where `Z (1 - Z)` < 1e-12).
2. `C_min` and `C_max`: the table's `C` at `(Z, Sz, Lambda = 0)` and `(Z, Sz, Lambda = 1)`.
3. `Lambda = (C - C_min) / (C_max - C_min)`, clamped to [0, 1] (0 where the span
   vanishes: at `Z = 0`, `Z = 1` and `Sz = 1` every `C_st` gives the same
   state).

The format requires `C` exactly linear in `Lambda` at each `(Z, Sz)` node, so the
multilinear lookup of `C` returns the transported `C` itself (to round-off)
wherever it lies within the bounds: the table cannot drift from the transported
variable it is indexed by. Three lookups of one field and one of all the fields
needed; no iteration. The `C_st` alternative needs a search (bisection or
Newton) for `C_st(Z, Sz, C)` per lookup, with a data-dependent iteration count
on the device; tabulating `C` on a different grid per `(Z, Sz)` needs the same
search in disguise.

`C` must increase with `C_st` at each `(Z_mean, Sz)` for `Lambda` to be a coordinate.
`C_st` is chosen because it is monotonic along the S-curve at `Z_st`; away
from it the generator checks monotonicity after integration and follows the
running maximum where it fails, reporting how many nodes did. In the first
draft's H2/N2 table none did (the only non-monotonic nodes were those where `C`
does not vary at all).

The bounds of `C` of the sample table, and a lookup's cell:

![Bounds of C against Z at 3 normalized variances, with a transported state mapped to Lambda = 0.6, and the 8 corners of a table cell around a query point](../images/guide/flamelet_lookup-light.svg#gh-light-mode-only)
![Bounds of C against Z at 3 normalized variances, with a transported state mapped to Lambda = 0.6, and the 8 corners of a table cell around a query point](../images/guide/flamelet_lookup-dark.svg#gh-dark-mode-only)

## 5. Transported scalars and coupling with the flow (stage 2)

### 5.1 Equations

Favre-filtered, with the LES closures of [les.md](les.md):

$$
\partial_t (\bar\rho \tilde Z) + \nabla \cdot (\bar\rho \tilde u \tilde Z) = \nabla \cdot ((\bar\rho D + \bar\rho D_t) \nabla \tilde Z)
$$

$$
\partial_t (\bar\rho \tilde C) + \nabla \cdot (\bar\rho \tilde u \tilde C) = \nabla \cdot ((\bar\rho D + \bar\rho D_t) \nabla \tilde C) + \tilde{\dot\omega}_C
$$

with `rho D = kappa / cp` from the table (unity Lewis number, decision 8),
`rho D_t = mu_t / Sc_t` from the eddy viscosity, and `omega_C` from the table
as an explicit source (decision 9), scaled from the flamelet density `rho0` to
the local `rho`.
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

### 5.3 Where the control variables live (decision 6)

**Decision:** `gas = "flamelet"`, a third gas model beside the perfect gas and
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
  in place of the Strang-split reactors (decision 9);
- output: `T`, `Y_<species>`, `HRR` and any table field as derived fields.

**Alternatives.** (a) Control variables beside a mechanism's species
(`gas = "mixture"` with extra columns, species set from the table): exact NASA
thermodynamics, but it keeps the species memory and bandwidth that tabulation is
meant to save, and the transported species and the table disagree. (b) One-way
passive scalars on any gas model: no feedback on density and temperature, so not
a combustion model; useful only as a verification step. (c) A new scalar block
in `State` beside the species: cleaner naming, but duplicates every loop the
species block already has. The mixture path's functors become generic over the
gas model by templating on it, as the RHS already is over `PerfectGas` /
`Mixture`.

### 5.4 Boundary and initial values

Inlets give `Z` (1 for fuel, 0 for oxidizer, or any mixture) and `C` (0 for
fresh streams, or the burnt value for a pilot: `C_max` at that `Z`), instead of
species; walls are zero-flux for all scalars. Initial states use the same keys,
through the existing expression machinery (`Z = "x < 0 ? 1 : 0"`). A pilot or an
equilibrium region is `C = "burnt"`, resolved through the table.

### 5.5 Stage 2 as built

The laminar coupling (`src/physics/flamelet.{h,cpp}`,
`src/solver/solver_flamelet.cpp`, user guide [flamelet.md](../guide/flamelet.md))
settles four points the sections above left open:

- **Dispatch.** `Mixture` holds a `flamelet::Gas` and branches at run time on
  `tabulated()` in `cell_state`, the gas constant, `cp`, `e(T)` and `T(e)`,
  instead of templating the mixture path's functors on the gas model
  (decision 6). The branch is uniform across a launch (one gas per run), so it
  costs no divergence, and it avoids instantiating every mixture kernel twice.
  Runs without `gas = "flamelet"` take the untaken branch only.
- **Source scaling.** `omega_C` is tabulated with the flamelet density `rho0`
  per node and scaled by `(rho / rho0)^2`, the scaling of a second-order rate
  (Saghafian et al. 2015), so compression and expansion change the rate
  consistently.
- **Heat flux.** With unity Lewis number the conduction and the enthalpy
  carried by diffusion combine to `-(kappa / cp) grad h`, `h = e + p / rho`;
  the solver conducts `h` with `kappa / cp` and drops the species enthalpy flux
  and the correction velocity (`Z` and `C` are not mass fractions). Conducting
  along `grad T` instead would heat a mixing layer of streams at one
  temperature by hundreds of kelvin.
- **float32 tables** are widened to float64 when copied to the solver's
  device view: stage 2 reads few fields per lookup and keeps one lookup path.

Each scalar is limited on its own in reconstruction, without the mass-fraction
coupling (major species, shared bound) of mixtures. The tests
(`test/flamelet_gas_test.cpp`, `test/flamelet_solver_test.cpp`) use synthetic
tables with exact answers: the closed-form `T(e)` against a quadrature of
`cv`, device lookups against the multilinear source, the error-function
mixing of `Z` at uniform `T`, a closed reactor following the tabulated source,
and the exact traveling front of `C_t = D C_xx + k C^2 (1 - C)`, which checks
diffusivity, source and the map `C -> Lambda` together.

- **Output.** The table's species, `T0` (as `T_TABLE`) and the optional `HRR`
  field are computed on the host when the output is staged, from the host copy
  of the table: output is infrequent, so they add no device kernel and nothing
  to the step. `HRR` is scaled by `(rho / rho0)^2` like `omega_C`, since both
  are rates of the same reactions. `MIXTURE_FRACTION` and `C_NORM`
  are the derived fields of the catalog, `Z` and the table's `Lambda`.
  Probes and statistics of the table fields are left for later.
- **Table pressure.** The setup compares the initial pressure field (its
  extremes over all ranks) and the prescribed pressures of `inlet_state`,
  `inlet_stagnation` and `outlet_pressure` with the table's, and warns
  (`W-TABLE-PRESSURE`) beyond 10%. It is a warning, not an error: a run may
  pass through other pressures on purpose, and acoustics or a shock move the
  pressure without invalidating the table.

- **Boundaries.** The boundary data carry the `Mixture` (species tables or
  the flamelet gas) instead of the species thermodynamics alone, so inlet ghost
  states take `R`, `e(T)`, `h(T)` and the isentrope from the same dispatch as
  the cells. For `saghafian2015`, `int cp / T dT = R ln(T / T0) + R / b
  (ln(T / T0) - ln((gamma(T) - 1) / (gamma0 - 1)))` with `b = gamma0 - 1 -
  a_gamma T0`, in closed form. Expressions of `Z` and `C` are compiled as the
  species' are and are not normalized. Walls convert their temperature to the
  conducted enthalpy at the cell's `Z` and `C`; the table's `kappa / cp` does
  not depend on `T`, so the wall's conductivity is the cell's.

Still to come in stage 2: the 2D counterflow against Cantera.

**The counterflow validation needs a generator.** It compares Mallard's 2D
counterflow (`examples/counterflow_diffusion`) with Cantera's
`CounterflowDiffusionFlame` and with Mallard's finite-rate run on the same
mesh, which needs a table made from Cantera flamelets of the same mechanism
and transport. Mallard has no generator (section 2), and FPVgen is not used
from Mallard. The validation waits for the generator library to provide:

1. an `fpv` / `saghafian2015` table in this format of the S-curve of the
   example's counterflow (H2/N2 against air, `mechanisms/h2o2.yaml`, 300 K,
   1 atm), with unity Lewis number, `HRR` and at least `Y_OH` and `Y_H2O`;
2. its own check that table flamelets match Cantera (section 10, first row);
3. the table, small enough to commit (a few MB), and the generator input in its
   `/provenance/input`.

The case then runs `gas = "flamelet"` beside the finite-rate example, with the
targets of section 10: `T_max`, flame position and the `Y_OH` peak within 2% of
Cantera.

### 5.6 Stage 3 as built: LES with the algebraic variance

`[les]` with `gas = "flamelet"` (`src/solver/solver_flamelet.cpp`,
`src/solver/solver_les.cpp`, guide [flamelet.md](../guide/flamelet.md#les)):

- **Variance.** One device view of `Zvar` per cell, held by
  `flamelet::Gas` so that every lookup (`cell_state`, transport, the source of
  `C`, the enthalpy conducted at walls) reads it through `cell_variance(c)`;
  laminar runs leave it empty and read zero. The algebraic closure uses the
  viscous least-squares stencil of `Z = rho Z / rho` and the LES filter width,
  on owned cells only, and the halo cells receive their owners' values by an
  exchange: a halo cell's stencil is truncated, so computing it locally would
  make the result depend on the partition. It is recomputed wherever the
  state's temperature is (`update_cell_states`, the stage updates, freezing
  for step graphs), before the conversion that uses it.
- **Initial state.** The initial fields convert at zero variance (the gradient
  needs the conserved `Z`); the cells then reconvert at their variance to the
  same `p`, `T`, `u`, `Z` and `C`. Restarts read the conserved state and only
  recompute the variance.
- **Boundaries** are laminar: their states have no gradient of their own, and
  an inlet's `Zvar` would need a modeled inflow variance. Inlet faces of
  `inlet_state` enter the cell gradients as the viscous fluxes' do.
- **Eddy fluxes.** The SGS coefficients of the flamelet gas are `[mu_t, mu_t /
  Pr_t, mu_t / Sc_t]`: enthalpy is conducted with `kappa / cp`, so its eddy
  conductivity is `mu_t / Pr_t` (the mixtures' `cp mu_t / Pr_t` multiplies
  `grad T`), and `Z` and `C` take `mu_t / Sc_t` without the molar mass the
  mixtures' mixture-averaged form carries. The time step bounds use `R` and
  `cp` of the table at the cell's variance.
- **Variance in the table.** The FPV map's `Sz = Zvar / (Z (1 - Z))` axis
  already ran from 0 to 1 in the stage-1 reader; the clip `Zvar <= Z (1 - Z)`
  keeps lookups at the edge instead of relying on the map's clamp.

The tests (`test/flamelet_les_test.cpp`) use a table linear in `Sz` in every
field the solver reads, so lookups at the variance are exact: the variance of a
linear `Z` against `C_Z h^2 |grad Z|^2`, and `p`, `T`, `rho`, `mu`, `Y_H2O`,
the normalized `Lambda` and one step of the source at it; the decay of a cosine of
`Z` in uniform shear at `(D + nu_t / Sc_t) k^2` with the Smagorinsky `nu_t`,
with enthalpy mixing with `Z` only when `Pr_t = Sc_t`; and bit-for-bit
agreement on any rank count and with step graphs.

### 5.7 Stage 3 as built: the transported second moment

`variance = "transported"` adds `Z2` to the species block after `Z` and `C`
(`flamelet::Gas::second_moment`), so storage, transport, reconstruction, halo
exchange, restart and output come with it, and the variance is the pointwise
`Z2 - Z^2` of every cell, with no exchange. Its sink, `2 rho D |grad Z|^2 +
C_chi rho (D + D_t) Zvar / Delta^2`, is an explicit source beside that of `C`,
from the transport gradient of `Z`, the conducted `kappa / cp` and the SGS
coefficient `mu_t / Sc_t`. Inlets carry `Z2 = Z^2` (laminar), completed from
their `Z` wherever compositions are made (`flamelet::Gas::complete`), and fresh
initial states start at `Z2 = Z^2` of the cell averages: the averages of a
pointwise `Z^2` would hold the variance of the profile within each cell.

`C_chi` defaults to 2, not to the
`2 / C_Z = 20` that would match the algebraic model's equilibrium: the sink
relaxes in `Delta^2 / (C_chi (D + D_t))`, which the viscous time step resolves
only for `C_chi` of a few. The test relaxes a linear `Z` in uniform shear from
zero variance to `2 D_t Delta^2 |grad Z|^2 / (C_chi (D + D_t))`, exact on a
uniform mesh because the discrete diffusion of `Z^2` is `2 |grad Z|^2` there.

### 5.8 Thickened flames (TFLES) with the table

`[les.combustion]` `model = "tfles"` with the flamelet gas: the flame of the
table is thickened until the mesh resolves it, the use FPV/FGM tables get on
meshes that do not resolve the laminar flame (Kuenne et al. 2011). Decisions:

- **Thickening as for finite-rate chemistry**, with no flamelet-specific
  path: `ThickenFunctor` already multiplies every scalar's `rho D` and the
  conducted `kappa / cp` by `E F` (for the flamelet gas these are `Z`, `C`,
  and enthalpy, with unity Lewis number) and the eddy coefficients by
  `1 - Omega`, and the source of `C` takes the chemistry time scale `E / F`
  of the step, as the Strang reactors do (`ProgressSourceFunctor`). `Z` is
  thickened with `C`, as every species is in TFLES: unity Lewis number keeps
  `Z`, `C` and `h` diffusing alike, which the table assumes. The efficiency
  multiplies the source (Colin et al. 2000; Charlette et al. 2002): the
  thickened flame then propagates at `E s_L`, its subgrid wrinkling. `HRR`
  output is scaled as the source.
- **Sensor on `Lambda`**, not `T`: the transport values of the flamelet gas
  hold `h`, not `T`, and the table's normalized progress variable is the
  progress variable the table is built on, 0 and 1 at the fresh and burnt
  states of every `Z`, so a non-premixed or stratified flame needs no
  `T_unburnt`, `T_burnt`. Where the table has no flame (`C_max = C_min`),
  `Lambda = 0` and nothing is thickened. The dynamic efficiency filters the
  same `Lambda`. Mixture sensors (`temperature`, `reaction`) are input errors.
- **Zero subgrid variance**: the thickened flame is resolved, and a presumed
  PDF over a subgrid variance would represent the same unresolved structure a
  second time. With `[les.combustion]` the variance closure is `none` (the
  default; `algebraic` and `transported` are input errors), which allocates no
  variance and leaves every lookup on the table's `Sz = 0` slice, exactly the
  laminar flamelets (the axis starts at 0 by the format). `variance = "none"`
  is also accepted in LES without thickening. Tables keep their `Sz` axis
  (the format needs every axis to have two nodes); only its first slice is read.
- **PaSR** stays an input error: it scales finite-rate rates.

With the sample table, the sensor covers the source of `C` at every `Z`
(`C_SENSOR = 0.05` in `src/physics/tfles.h`):

![The thickened flame's sensor against Lambda, with the source of C of the sample table at 3 mixture fractions, each normalized by its peak](../images/guide/flamelet_tfles_sensor-light.svg#gh-light-mode-only)
![The thickened flame's sensor against Lambda, with the source of C of the sample table at 3 mixture fractions, each normalized by its peak](../images/guide/flamelet_tfles_sensor-dark.svg#gh-dark-mode-only)

Tests (`test/flamelet_les_test.cpp`): lookups with `variance = "none"` and with
a thickened flame are those of the `Sz = 0` slice of the variance table; in
uniform shear with `Omega = 1`, a cosine of `Z` decays at `E F D k^2` (no eddy
diffusivity) and `C` follows `dC/dt = (E / F) k C^2 (1 - C)` with Charlette's
saturated `E = F^0.5`; a reaction-diffusion front thickened by `F = 3` keeps its
laminar speed within 1% and is `F` times thicker.

## 6. Interaction with the existing models

- **Finite-rate chemistry:** exclusive. `gas = "flamelet"` loads no mechanism
  into the solver (the table carries the species it outputs); `[chemistry]` is
  an input error with it.
- **LES:** the eddy viscosity enters `rho D_t = mu_t / Sc_t` of every scalar
  and the variance closure through the filter width `Delta` (les.md, section 3).
  The kinetic-energy budget and the low-dissipation flux are independent of the
  gas model.
- **TFLES:** thickens `Z`, `C` and enthalpy (`D -> E F D`,
  `omega_C -> E omega_C / F`), the standard combination (e.g. Kuenne et al.
  2011), with `ThickenedFlame` and its dynamic efficiency unchanged, the sensor
  on `Lambda` instead of `T`, and the table at zero subgrid variance (5.8).
- **PaSR:** not applicable (it scales finite-rate rates).
- **Radiation:** the optically thin model needs `T` and `X_H2O`, `X_CO2` from the
  table; it then removes enthalpy, which needs the enthalpy-defect axis (stage
  6). Until then radiation is an input error with `gas = "flamelet"`.
- **Double flux:** applies unchanged to the surrogate `(gamma, e0)`.
- **Output, probes, statistics:** table fields are derived fields
  (`src/solver/derived_catalog.h`): `T`, `Y_<species>` of the table's species,
  `HRR`, `omega_C`, `Z`, `Zvar`, `C`, `Lambda`; probes and statistics read them
  as they read the mixture's.
- **Determinism and step graphs:** lookups are pointwise kernels without host
  work; no new reduction or exchange.

## 7. Input keys

```toml
[physics]
gas = "flamelet"
table = "h2_fpv.h5"          # relative to the directory the run starts in
variance = "algebraic"       # with [les] only; or "transported" (C_chi = 2)
C_Z = 0.1                    # algebraic variance constant

[[boundaries]]
name = "fuel"
type = "inlet_state"
Z = 1.0
C = 0.0
```

`model` and `thermodynamics` come from the table's header; the solver checks
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

- **Reader and lookup (C++, `test/flamelet_table_test.cpp`)**: the fixture
  ([table_format.md](table_format.md#10-reference-fixtures)), written by an
  independent Python writer, reads with its tags, species, pressure and
  provenance. Every field looks up in a device kernel as the writer computed
  it, which catches node order, field order, stride and byte order.
  Multilinear functions of four uneven axes are reproduced to round-off and
  clamped outside the axes. The FPV map returns the transported `C` within its
  bounds and clamps outside. Files that break the format or the contract are
  refused with the file and the reason: truncated data, a wrong version, NaN,
  an unknown model or thermodynamics tag, wrong units, a missing field, a field
  name outside the allowed form, a `Y_` field outside the species block, wrong
  axes or too few of them, bad provenance, `Z_st` out of range, a weight of `C`
  that is not a number, a stream mass fraction outside [0, 1], `gamma0 <= 1`, a
  mass fraction outside [0, 1], and `C` not linear in `Lambda`.
- **Generation** (the generator library): beta weights against an independent
  integration, the PDF limits, the `C_st -> Lambda` remap, and off-table flamelets
  against direct Cantera solutions. The first draft of #1222 measured, on a
  41 x 5 x 21 H2/N2 table, `T` within 4.6 K of an off-table Cantera flamelet.
- **Stage 2**: the gas model's `cell_state` against the table's `T0`, `p` at
  the flamelet states, and the closed-form `T(e)` against a quadrature of
  `cv(T)`; transport of `Z` with constant diffusivity against the exact error
  function solution; a 2D counterflow (below) against Cantera, with a
  generator-made table committed for it.

## 10. Validation

| Case | Model | Reference | Target |
|---|---|---|---|
| Table flamelets and off-table flamelets (generator) | FPV | Cantera counterflow, unity Lewis | `T` within 0.5% of the peak |
| PDF integration (generator) | all | direct integration of the beta PDF | 1e-9 relative |
| 2D laminar counterflow, `examples/counterflow_diffusion` | FPV | Cantera (as `counterflow_reference.py --match`), Mallard finite-rate on the same mesh | `T_max`, flame position and `Y_OH` peak within 2% of Cantera; same extinction strain within 5% |
| Sandia flame D (Barlow & Frank 1998) | FPV, SLFM | TNF data: mean and rms `Z`, `T`, `Y_k` profiles at x/D = 7.5 to 45 | within the scatter of the TNF FPV results (Ihme & Pitsch 2008b) |
| Cabra lifted H2/N2 flame (Cabra et al. 2002) | FPV | lift-off height, centerline and radial `Z`, `T` | lift-off within 20% of the measured height; the methane variant against Ihme & See (2010) |
| Premixed jet or Volvo rig (#197, #278) | FGM | Volvo (Sjunnesson et al.), as les.md section 8.3 | mean velocity and `T` profiles; comparison with TFLES |
| Shared case: flame in decaying turbulence (`examples/flame_turbulence`) | FGM vs finite-rate + TFLES | each other, DNS | turbulent consumption speed within 10% |

## 11. Stages

1. **Design, table format and reader** (this change): this design,
   [table_format.md](table_format.md), `flamelet::Table` (reading and checking
   the contract), `TableView` and `FPV` (device lookup), the fixture and their
   tests. No solver change, no input key. Table generation moves to the
   generator library.
2. **Laminar FPV in the solver**: `gas = "flamelet"` with transported `Z` and
   `C`, `saghafian2015` thermodynamics, table transport, explicit `omega_C`,
   inlet values, derived fields; the 2D counterflow against Cantera and
   Mallard's finite-rate run, with a table from the generator. Device code: GPU
   CI.
3. **LES closures**: algebraic and transported variance, SGS scalar fluxes;
   Sandia flame D.
4. **Unstable branch and lifted flames**: the low-strain end of the unstable
   branch, preheated streams; the Cabra flame.
5. **FGM**: the `fgm` tag (free premixed flamelets, from the generator), optional
   thickening with TFLES; Volvo or the premixed jet; the shared case against
   finite-rate + TFLES.
6. **Heat loss and slow chemistry**: an enthalpy-defect axis, radiation; `Y_NO`
   with tabulated production and consumption, unsteady flamelets (#280);
   pressure axis.

Importers of CharLES^X and FlameMaster tables, if wanted, belong in the
generator library, which writes this format.

## 12. Open questions

Resolved with the maintainer before stage 2:

1. **Gas-model structure (decision 6). Resolved:** a third gas model whose `Z`
   and `C` live in the existing species block, with the mixture path's functors
   templated on the gas model; no separate scalar block in `State`. The species
   block already has storage, transport, halo exchange, restart and time
   integration, and a second block would duplicate all of it.
2. **Thermodynamics (decision 7). Resolved:** compressible FPV with linearized
   `gamma(T)` after Saghafian et al. (2015), not low-Mach FPV. Mallard solves
   the compressible energy equation, and low-Mach FPV would ignore it at
   shocks and acoustics.
3. **Progress-variable source (decision 9). Resolved:** `omega_C` is an explicit
   RHS source, because the FPV progress source is not stiff. The Strang-split
   fallback is dropped and is listed as a non-goal.
4. **Table precision (decision 10). Resolved:** float64 by default, float32
   allowed in the same format version. float64 is exact; float32 halves the
   memory and bandwidth of lookups if they show up in profiles.
5. **Table generation. Resolved:** outside Mallard, in a separate library
   (`<generator, name TBD>`) that starts from FPVgen and credits it. Mallard
   specifies the format and reads it. Generation has its own users and tests
   and will cover more models than Mallard reads.

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
