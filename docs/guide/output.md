# Output

A run writes snapshots of the solution, restart files, and the diagnostics it
is asked for: probes, wall forces, domain integrals and statistics. It reports
its progress on the terminal and in a machine-readable run record.

## What a run writes

| Output | File(s) | Turned on by | Details |
|---|---|---|---|
| Solution snapshots | `PREFIX_<n>.vtkhdf` (`<n>`: the step, or with `time_interval` the snapshot's index) and `PREFIX.vtkhdf.series`; VTU: `.vtu`, `.pvtu`, `.pvd` | `[[write_data]]` | [Solution snapshots](#solution-snapshots-write_data) |
| Restart files | `PREFIX_<step>.restart` | `[[write_data]]` with `format = "restart"` | [Initialization and restarts](initialization.md#restarts) |
| Terminal output | standard output (errors: standard error) | always; `[output] check_interval` sets the progress rows | [Terminal output](#terminal-output) |
| Run record | `mallard.jsonl` | on by default; `[output] run_log` | [Run record](#run-record-mallardjsonl) |
| Divergence report | `divergence_report.txt`, `.json`, `divergence_report_neighborhood_<step>.vtkhdf` | a run that diverges; `[output] divergence_report` | [The divergence report](running.md#the-divergence-report) |
| Run health | the run record's `health` and the terminal output's **Health** section | on by default; `[output] health` | [Run health](#run-health) |
| Probes | `probe_<name>.csv` (or `.h5`, `.vtkhdf`, `.pvd`), `probe_<name>_reductions.csv` | `[[probes]]` | [Probes and monitors](probes.md#probes-probes) |
| Wall forces | `forces_<zone>.csv` | `[[forces]]` | [Wall forces](probes.md#wall-forces-forces) |
| Domain integrals | `integrals.csv` | `[integrals]` | [Domain integrals](probes.md#domain-integrals-integrals) |
| Time and phase averages | `MEAN_*`, `COV_*` and `PHASE_*` fields of the snapshots and restart files; probe columns `MEAN_<variable>` with `average = true` | `[statistics]`, `[statistics.phase]`, `[[probes]]` | [Statistics](statistics.md), [Probes](probes.md#time-averages) |
| Spectra and two-point correlations | `spectra.csv` | `[statistics.spectra]` | [Spectra](statistics.md#spectra-and-two-point-correlations-statisticsspectra) |
| Time spectra of probes | `probe_<name>_spectra.csv`, `probe_<name>_reductions_spectra.csv` | `[[probes]]` with `spectra` | [Time spectra](probes.md#time-spectra) |

Every file records its [units](#units), the [run block](#run-block) of the run
that wrote it, each snapshot's [snapshot metadata](#snapshot-metadata) (step,
time, why it was written, by which run) and a
[format version](#formats-and-compatibility). File names are only names:
Mallard's readers take step, time and run from what the files record.

## Solution snapshots: `[[write_data]]`

```toml
[[write_data]]
prefix = "solut/flow"         # directories are created
time_interval = 0.01          # or interval = 100 (steps)
fields = ["RHO", "U", "P", "T", "VORTICITY"]

[[write_data]]                # several entries write independently
prefix = "solut/wall"
interval = 50
region = "cylinder"           # a boundary zone, a slice, a volume: any region (below)
fields = ["P", "T"]
```

*Used in:*
[`sod`](https://matthewbonanni.github.io/mallard/docs/examples/#sod),
[`shu_osher`](https://matthewbonanni.github.io/mallard/docs/examples/#shu-osher),
[`riemann_2d`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d),
[`riemann_2d_quads`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d-quads),
[`double_mach`](https://matthewbonanni.github.io/mallard/docs/examples/#double-mach)
and 50 more.

| Key | Default | Meaning |
|---|---|---|
| `prefix` | required | Output path prefix |
| `format` | `"vtkhdf"` | `vtkhdf` (ParaView 5.12+, with a `.vtkhdf.series` index), `vtu` (ParaView, with a `.pvd` time series), `xdmf` (XDMF indexes over HDF5 files) or `restart`; see [which format](#which-format). Builds without HDF5 write `vtu`, with a warning |
| `series` | `false` | `vtkhdf`: `true` keeps every snapshot in one file, `PREFIX.vtkhdf` |
| `interval` / `time_interval` | one required | Every this many steps, or this much simulated time (the step is shortened to land on each output time) |
| `fields` | required (all but `restart`) | Fields to write, below (`mallard list fields input.toml` lists those of a case); `Y_*` selects every field with that prefix |
| `region` | none (the whole mesh) | `vtkhdf`, `vtu`: write only this [region](geometry.md#regions), inline or by name: a filled region's cells, or a boundary zone, surface, curve, points or cut plane ([below](#snapshots-of-regions)). Replaces `geometry` (removed in 0.10.0; `mallard input upgrade` rewrites it) |
| `regions` | none | Instead of `region`: a list of regions, each written as its own series, `PREFIX_<name>` (inline ones: `PREFIX_<place in the list>`) |
| `compression` | `"none"` | `"zlib"`: lossless compression of `vtkhdf`, `vtu`, `xdmf` and `restart` files (below) |
| `relative_error` | none (lossless) | `vtkhdf`, `vtu`, `xdmf` with `compression = "zlib"`: lossy, each cell value within this relative error |
| `chunk_cells` | 65536 | Compressed `restart` files: cells per chunk |

**Fields:** `mallard list fields input.toml` lists exactly those of a case, and
`mallard list fields` all of them with their units and what each needs
([Command line](cli.md#mallard-list-fields)); the table below is the latter.
Its last column says where each field can be had: written by
`[[write_data]]`, sampled by [`[[probes]]`](probes.md), averaged by
[`[statistics]`](statistics.md), or derived from restart files and snapshots
after the run by [`mallard post derive`](#derived-fields-after-the-run).
Vectors (`U`, `RHOU`, `VORTICITY` in 3D) also go by their components
`<name>_X`, `_Y`, `_Z`; `<s>` is a species of the mechanism, `<A>` and `<B>`
other fields.

<!-- BEGIN GENERATED fields -->
| Group | Field | Units | Description | Needs | Available in |
|---|---|---|---|---|---|
| conservative | `RHO` | `kg m-3` | density | every case | write_data, probes, statistics |
|  | `RHOE` | `kg m-1 s-2` | total energy density | every case | write_data, probes, statistics |
|  | `RHOU` | `kg m-2 s-1` | momentum density (vector) | every case | write_data, probes, statistics |
|  | `RHOY_<s>` | `kg m-3` | partial density of species s | a mixture | write_data, probes, statistics |
| primitive | `P` | `kg m-1 s-2` | pressure | every case | write_data, probes, statistics |
|  | `T` | `K` | temperature | every case | write_data, probes, statistics |
|  | `H` | `m2 s-2` | specific enthalpy | every case | write_data, probes, statistics |
|  | `U` | `m s-1` | velocity (vector) | every case | write_data, probes, statistics |
| derived | `MACH` | `1` | Mach number | every case | write_data, post derive |
|  | `Q_CRITERION` | `s-2` | Q-criterion | every case | write_data, post derive |
|  | `LAMBDA2` | `s-2` | lambda2 vortex criterion | every case | write_data, post derive |
|  | `SCHLIEREN` | `kg m-4` | density gradient magnitude | every case | write_data, post derive |
|  | `P0` | `kg m-1 s-2` | stagnation pressure | every case | write_data, post derive |
|  | `T0` | `K` | stagnation temperature | every case | write_data, post derive |
|  | `H0` | `m2 s-2` | stagnation (total) specific enthalpy | every case | write_data, post derive |
|  | `SOUND_SPEED` | `m s-1` | speed of sound | every case | write_data, post derive |
|  | `ENTROPY` | `m2 s-2 K-1` | specific entropy | every case | write_data, post derive |
|  | `DILATATION` | `s-1` | dilatation (velocity divergence) | every case | write_data, post derive |
|  | `HELICITY` | `m s-2` | helicity density | a 3D build (in 2D the vorticity is normal to the velocity) | write_data, post derive |
|  | `ENSTROPHY` | `s-2` | enstrophy | every case | write_data, post derive |
|  | `STRAIN_RATE` | `s-1` | strain-rate magnitude | every case | write_data, post derive |
|  | `CP` | `1` | pressure coefficient | [output] reference = {rho, u, p} | write_data, probes, post derive |
|  | `CFL` | `1` | local CFL number | every case | write_data |
|  | `P_MAX` | `kg m-1 s-2` | largest pressure seen | every case | write_data |
|  | `TENO_SIGMA` | `1` | TENO troubled-cell indicator | face_reconstruction type = "TENO" | write_data |
|  | `REGION_FRACTION` | `1` | volume fraction of the region in the cell | a [[write_data]] region, which writes it | write_data |
|  | `DIVERGED` | `1` | 1 in the cells that became invalid | a run that diverges, whose divergence report snapshot writes it | - |
|  | `VORTICITY` | `s-1` | vorticity (vector) | every case | write_data |
| transport | `MU` | `kg m-1 s-1` | dynamic viscosity | a mixture with navier_stokes | write_data |
|  | `LAMBDA` | `kg m s-3 K-1` | thermal conductivity | a mixture with navier_stokes | write_data |
|  | `D_<s>` | `m2 s-1` | diffusion coefficient of species s | a mixture with navier_stokes | write_data |
|  | `DT_<s>` | `kg m-1 s-1` | thermal diffusion coefficient of species s | a mixture with navier_stokes and soret = true | write_data |
| mixture | `MIXTURE_FRACTION` | `1` | Bilger mixture fraction | a mixture with [output.mixture] fuel and oxidizer | write_data, post derive |
|  | `PROGRESS_VARIABLE` | `1` | progress variable | a mixture with [output.mixture] progress | write_data, post derive |
|  | `FLAME_INDEX` | `1` | Takeno flame index | a mixture with [output.mixture] fuel and oxidizer, or fuel_species and oxidizer_species | write_data, post derive |
|  | `SCALAR_DISSIPATION` | `s-1` | scalar dissipation rate of the mixture fraction | a mixture with navier_stokes and [output.mixture] fuel and oxidizer | write_data, post derive |
|  | `T_SEED` | `K` | temperature guess of the thermodynamic state | a mixture | write_data |
|  | `Y_<s>` | `1` | mass fraction of species s | a mixture | write_data, probes, statistics |
|  | `X_<s>` | `1` | mole fraction of species s | a mixture | write_data |
| chemistry | `HRR` | `kg m-1 s-3` | heat release rate | a mixture with [chemistry] | write_data |
|  | `CHEM_H` | `s` | last chemistry sub-step | a mixture with [chemistry] | write_data |
|  | `CHEM_COST` | `1` | chemistry sub-steps in the last step | a mixture with [chemistry] | write_data |
|  | `QRAD` | `kg m-1 s-3` | radiative heat source | [radiation] | write_data |
|  | `OMEGA_<s>` | `kg m-3 s-1` | net production rate of species s | a mixture with [chemistry] | write_data |
| LES | `MU_T` | `kg m-1 s-1` | eddy viscosity | [les] | write_data |
|  | `TF_F` | `1` | flame thickening factor | [les.combustion] (TFLES) | write_data |
|  | `TF_E` | `1` | flame wrinkling efficiency | [les.combustion] (TFLES) | write_data |
|  | `TF_OMEGA` | `1` | flame sensor | [les.combustion] (TFLES) | write_data |
|  | `PASR_KAPPA` | `1` | PaSR reacting fraction | [les.combustion] model = "pasr" | write_data |
| sources | `SOURCE_<A>` | `A's s-1` | source of A, per second | [[sources]] or [source] | write_data |
| statistics | `MEAN_<A>` | `A's` | time average of A | [statistics] fields with A, or [[probes]] averaging A | write_data, probes |
|  | `COV_<A>_<B>` | `A's times B's` | covariance of A and B | [statistics] products with A*B | write_data, probes |
|  | `PHASE_<bin>_<A>` | `A's` | phase average of A in a bin | [statistics.phase] | write_data, probes |
|  | `RMS_<A>` | `A's` | root mean square fluctuation of A | mallard post average, of A with its variance | write_data, probes |
|  | `FAVRE_MEAN_<A>` | `A's` | Favre (density-weighted) time average of A | mallard post average --favre | write_data, probes |
|  | `FAVRE_COV_<A>_<B>` | `A's times B's` | Favre covariance of A and B | mallard post average --favre, of A*B | write_data, probes |
|  | `CI_<average>` | `the average's` | confidence interval half width of an average | [statistics] batch, of a MEAN_ or COV_ average | write_data, probes |
| surface probes | `MASS_FLUX` | `kg m-2 s-1` | normal mass flux | [[probes]] on a surface or boundary zone | probes |
|  | `ENTHALPY_FLUX` | `kg s-3` | normal total enthalpy flux | [[probes]] on a surface or boundary zone | probes |
|  | `TAU_W` | `kg m-1 s-2` | wall shear stress magnitude | [[probes]] on a wall zone | probes |
|  | `CF` | `1` | skin friction coefficient | [[probes]] on a wall zone | probes |
|  | `Q_W` | `kg s-3` | wall heat flux into the fluid | [[probes]] on a wall zone | probes |
|  | `U_TAU` | `m s-1` | friction velocity | [[probes]] on a wall zone | probes |
|  | `Y_PLUS` | `1` | first-cell y+ | [[probes]] on a wall zone | probes |
|  | `ENERGY_DIFFUSIVE_FLUX` | `kg s-3` | normal diffusive energy flux (conduction, viscous work, species enthalpy) | [[probes]] on a surface or boundary zone | probes |
|  | `ENERGY_TOTAL_FLUX` | `kg s-3` | normal total energy flux (convective and diffusive) | [[probes]] on a surface or boundary zone | probes |
|  | `WALL_MODEL_U` | `m s-1` | wall model exchange-point tangential speed | [[probes]] on a wall zone | probes |
|  | `WALL_MODEL_T` | `K` | wall model exchange-point temperature | [[probes]] on a wall zone | probes |
|  | `WALL_MODEL_H` | `m` | wall model exchange-point height | [[probes]] on a wall zone | probes |
|  | `MOMENTUM_FLUX` | `kg m-1 s-2` | normal momentum flux (vector) | [[probes]] on a surface or boundary zone | probes |
|  | `TRACTION_PRESSURE` | `kg m-1 s-2` | pressure traction on the boundary (vector) | [[probes]] on a wall zone | probes |
|  | `TRACTION_VISCOUS` | `kg m-1 s-2` | viscous traction on the boundary (vector) | [[probes]] on a wall zone | probes |
|  | `SPECIES_FLUX_<s>` | `kg m-2 s-1` | normal mass flux of species s | a mixture; [[probes]] on a surface or boundary zone | probes |
|  | `SPECIES_DIFFUSIVE_FLUX_<s>` | `kg m-2 s-1` | normal diffusive mass flux of species s | a mixture; [[probes]] on a surface or boundary zone | probes |
|  | `SPECIES_TOTAL_FLUX_<s>` | `kg m-2 s-1` | normal total mass flux of species s | a mixture; [[probes]] on a surface or boundary zone | probes |
| mesh quality | `NON_ORTHOGONALITY` | `degree` | largest face non-orthogonality | every case | write_data |
|  | `SKEWNESS` | `1` | largest face skewness | every case | write_data |
|  | `ASPECT_RATIO` | `1` | cell aspect ratio | every case | write_data |
|  | `VOLUME_RATIO` | `1` | largest volume ratio to a neighbor | every case | write_data |
|  | `DELTA_OVER_ETA` | `1` | cell size over the estimated Kolmogorov length | mallard check --mesh-fields (resolution estimates) | - |
|  | `CELLS_PER_DELTA_L` | `1` | cells across the laminar flame thickness | mallard check --mesh-fields (resolution estimates) with [les.combustion] (TFLES) | - |
| stencils | `STENCIL_SIZE` | `1` | cells in the central stencil | every case | write_data |
|  | `STENCIL_SECTORS` | `1` | sector stencils kept (TENO) | every case | write_data |
|  | `STENCIL_DEGREE` | `1` | degree of the central reconstruction polynomial | every case | write_data |
|  | `STENCIL_CONDITION` | `1` | condition number of the central least-squares fit | every case | write_data |
|  | `STENCIL_WALL` | `1` | 1 where a stencil reaches a wall | every case | write_data |
|  | `STENCIL_SEAM` | `1` | 1 where a stencil reaches across a periodic seam | every case | write_data |
|  | `STENCIL_PARTITION` | `1` | 1 where a stencil reaches cells of another rank | every case | write_data |
|  | `STENCIL_<id>` | `1` | membership in the stencils of cell id | mallard check --mesh-fields with --cells or --region | - |
<!-- END GENERATED fields -->

### Derived fields

Fields that show the flow's structure and state, computed from the state
being written (with the same cell gradients as `VORTICITY` and the
`[integrals]` monitor: the TENO reconstruction polynomials, else least
squares), so they cost nothing in steps that write none. Choosing them
before the run is not required: [`mallard post derive`](#derived-fields-after-the-run)
computes the same values from restart files and snapshots afterwards.

```toml
[output]
reference = {rho = 1.2, u = [30.0, 0.0], p = 101325.0}   # for CP only

[[write_data]]
prefix = "solut/jet"
interval = 500
fields = ["RHO", "U", "P", "MACH", "P0", "Q_CRITERION", "LAMBDA2", "SCHLIEREN", "CP"]
```

| Name | Units | Definition |
|---|---|---|
| `MACH` | `1` | Mach number \|u\| / c, with c = √(γ p / ρ) and, for mixtures, each cell's frozen γ |
| `P0` | `Pa` | Stagnation pressure p (1 + (γ − 1) M² / 2)^(γ / (γ − 1)), the isentropic relation with the cell's γ (frozen, for mixtures) |
| `T0` | `K` | Stagnation temperature T (1 + (γ − 1) M² / 2), likewise |
| `H0` | `m2 s-2` | Stagnation (total) specific enthalpy h + \|u\|² / 2, with the exact h of `H` |
| `SOUND_SPEED` | `m s-1` | Speed of sound c = √(γ p / ρ) (frozen, for mixtures) |
| `ENTROPY` | `m2 s-2 K-1` | Specific entropy. **Perfect gas:** s = c_p ln(T / T_ref) − R ln(p / p_ref), zero at the `[physics]` reference state `p_ref`, `T_ref`. **Mixtures:** the absolute (third-law) entropy Σ Y_k (s°_k(T) − R_k ln(X_k p / p°)) of the mechanism's NASA polynomials, with the entropy of mixing, p° = 1 atm. The two references differ: never compare a perfect-gas value with a mixture value |
| `Q_CRITERION` | `s-2` | Q = (‖Ω‖² − ‖S‖²) / 2 ([Hunt, Wray & Moin 1988](../references.md#hunt-wray-moin-1988)): positive where rotation dominates strain, i.e. in vortices. Before 0.10.0 it was called `Q`, which is now an error that names `Q_CRITERION` |
| `LAMBDA2` | `s-2` | λ₂, the middle eigenvalue of S² + Ω² ([Jeong & Hussain 1995](../references.md#jeong-hussain-1995)): negative in vortex cores |
| `SCHLIEREN` | `kg m-4` | \|∇ρ\|, a numerical schlieren image of shocks, contact surfaces and mixing layers |
| `DILATATION` | `s-1` | Dilatation ∇·u, the trace of the velocity gradient (with u_r / r in axisymmetric runs): negative in compressions and shocks |
| `HELICITY` | `m s-2` | Helicity density u·ω, ω = ∇×u. 3D only: in 2D ω is normal to u, and `HELICITY` is an error |
| `ENSTROPHY` | `s-2` | Enstrophy \|ω\|² / 2 |
| `STRAIN_RATE` | `s-1` | Strain-rate magnitude \|S\| = √(2 S_ij S_ij), as LES models use it |
| `CP` | `1` | Pressure coefficient (p − p_ref) / q_ref, q_ref = ρ_ref \|u_ref\|² / 2 of `[output] reference = {rho, u, p}`; the same definition and reference as the wall probes' `CP`. Without a reference it is an error that asks for one |

S and Ω are the symmetric and antisymmetric parts of the velocity
gradient ∂u_i/∂x_j. A 2D run treats its flow as a 3D one
with no third velocity component and no variation along z, so that Q and
λ₂ mean what they do in 3D; axisymmetric runs add the hoop strain
u_r / r to the gradient. In 2D, λ₂ = −Q for an incompressible flow,
so `LAMBDA2` adds little to `Q_CRITERION` there.

Pitfalls: both vortex criteria are squares of gradients, so their magnitude
scales with the flow's rate squared; pick isosurface levels relative to a
reference rate (e.g. Q = 0.1 (U/D)²) rather than absolute values. On coarse
meshes the gradients, and hence the criteria, are smoothed. `SCHLIEREN` spans
orders of magnitude between shocks and the rest of the flow; a logarithmic
color map or exp(−k \|∇ρ\| / max \|∇ρ\|) in
ParaView's calculator shows both. `P0`, `T0` and `SOUND_SPEED` of mixtures use
the frozen γ, consistent with `MACH`; they are not the stagnation state of a
reacting or equilibrium expansion. The wall quantities `Y_PLUS`, `CF`,
`TAU_W` and `Q_W` are [boundary probe](probes.md#boundary-data) variables,
which `[output] reference` also serves.

### Mixture fields

Fields of gas mixtures, relative to the streams, species and definitions of
`[output.mixture]` ([keys](../input.md#outputmixture)):

```toml
[output.mixture]
fuel = "fuel_inlet"                      # a [[boundaries]] name, or {X = {CH4 = 1.0}}
oxidizer = {X = {O2 = 1.0, N2 = 3.76}}
progress = {species = ["CO2", "CO"], burnt = 0.15}

[[write_data]]
prefix = "solut/flame"
interval = 200
fields = ["T", "HRR", "MIXTURE_FRACTION", "PROGRESS_VARIABLE", "FLAME_INDEX", "SCALAR_DISSIPATION"]
```

| Name | Units | Definition |
|---|---|---|
| `MIXTURE_FRACTION` | `1` | Bilger's mixture fraction Z = (β − β_O) / (β_F − β_O), β = 2 Z_C / W_C + Z_H / (2 W_H) − Z_O / W_O of the element mass fractions ([Bilger, Stårner & Kee 1990](../references.md#bilger-starner-kee-1990)), β_F and β_O those of the `fuel` and `oxidizer` streams: 1 in the fuel, 0 in the oxidizer, conserved by reactions |
| `PROGRESS_VARIABLE` | `1` | c of `progress`: Σ Y_k of its `species` divided by `burnt` (without `burnt`, the raw sum); or (T − T_unburnt) / (T_burnt − T_unburnt) |
| `FLAME_INDEX` | `1` | Takeno's flame index ∇Y_F·∇Y_O / (\|∇Y_F\| \|∇Y_O\|) of `fuel_species` and `oxidizer_species` ([Yamashita, Shimada & Takeno 1996](../references.md#yamashita-shimada-takeno-1996)): +1 where the gradients align (premixed), −1 where they oppose (non-premixed), 0 where either vanishes. Euler runs too |
| `SCALAR_DISSIPATION` | `s-1` | χ = 2 D \|∇Z\|² ([Peters 2000](../references.md#peters-2000)), with D = λ / (ρ c_p) of each cell (a unity-Lewis diffusivity). Needs `navier_stokes`, which has λ |

The mass-fraction gradients of `FLAME_INDEX` and `SCALAR_DISSIPATION` are
least squares over the cell's neighbors, the stencil of the velocity
gradient's fallback: the species have no TENO polynomials. Boundary faces
take the prescribed composition of `inlet_state` conditions and the cell's
own elsewhere.

Pitfalls: Z needs streams that differ in β (a fuel with C or H, an
oxidizer with O); species in neither stream (products, diluents) count by
their elements, so Z stays conserved through the flame. Mass fractions of
roughly constant value have gradients of round-off size whose direction is
noise: `FLAME_INDEX` is ±1 there, so read it where the flame is (mask it with
`HRR` or c between 0.05 and 0.95). A progress variable of species needs
`burnt` to reach 1 in the burnt gas; give the burnt gas's sum, e.g. from
`mallard reactor` or Cantera's equilibrium.

### Derived fields after the run

`mallard post derive` sets the case of restart files or snapshots up again
and writes snapshots of any of the fields above, from the run's own code
(reconstruction, gradients and kernels), so they equal what `[[write_data]]`
would have written at those steps, bit for bit from restart files
([Command line](cli.md#mallard-post-derive)):

```
mallard post derive solut/flow.vtkhdf.series --fields MACH,P0,Q_CRITERION
mallard post derive solut/restart_*.restart --latest --fields ENTROPY,DILATATION --with-state
mallard post derive solut/flow.vtkhdf.series --fields CP --reference '{rho = 1.2, u = [30, 0], p = 101325}'
mallard post derive solut/flame.vtkhdf.series --fields MIXTURE_FRACTION,FLAME_INDEX \
    --mixture '{fuel = "fuel_inlet", oxidizer = "air_inlet"}'
```

The snapshots go to `derived/derived_<step>.vtkhdf` (`-o`, `--name`; `.vtu`
without HDF5), with only the fields asked for unless `--with-state` adds
`RHO`, `U`, `P` and `T`. The case's `[output] reference` and
`[output.mixture]` come from the run's input; `--reference` and `--mixture`
give them for runs that set none, or replace them.

### Which format

| `format` | Opens in | Files |
|---|---|---|
| `vtkhdf` (default) | ParaView 5.12+, PyVista, the [Python package](python.md), h5py | One file per snapshot, or one for the series (`series = true`) |
| `xdmf` | VisIt, meshio, ParaView of any version ([open by a path with a directory](#xdmf-in-paraview)); the [Python package](python.md), h5py | The mesh once, one HDF5 file of fields per snapshot, and XDMF indexes |
| `vtu` | Anything with a VTK XML reader (e.g. Tecplot), the [Python package](python.md); builds without HDF5 | One file per rank per snapshot, with a `.pvtu` and a `.pvd` series |
| `restart` | Mallard, to continue or remap a run | One file per snapshot |

`vtkhdf` and `xdmf` need a build with HDF5 (parallel HDF5 on several ranks).
Before 0.10.0, `xdmf` was called `format = "hdf5"`, which is now an error since
VTKHDF is HDF5 too; `mallard input upgrade` rewrites it to `"vtkhdf"`.

**VTKHDF, the default.** `format = "vtkhdf"` writes VTK's HDF5 format, which
ParaView 5.12 and later open directly: open `PREFIX.vtkhdf.series` for the time
series. Every rank writes its part of one file per snapshot with collective I/O
(parallel HDF5 on several ranks), each array in global cell order, so the files
are the same on any number of ranks. Cells touching a curved wall are written as
quadratic Lagrange cells, so ParaView shows the curved geometry. With `series = true` the mesh is stored
once and every snapshot goes into `PREFIX.vtkhdf`; a restarted run drops the
snapshots after its restart time and continues the file. Read the files in
Python with the [Python package](python.md) (`mallard.open("PREFIX.vtkhdf.series")`),
or with h5py alone.

**Restarted runs continue the series.** A run resumed from a restart file
numbers its next snapshot after the snapshots up to the restart time, which the
restart file counts, and keeps those entries of the `.vtkhdf.series`, `.pvd` or
`.xmf` index; this holds for every `time_interval` writer, regions included
(`interval` writers are numbered by step). Resuming from an earlier restart file
than the last one rewinds: the entries after the restart time are dropped and
their snapshots are overwritten as the run reaches them again. If the index is
missing, the run says so, still never overwrites an earlier snapshot, and lists
only the snapshots it writes. Restart files older than this count number from
the index, as before.

Builds without HDF5 write VTU instead (and say so): one file per snapshot, and
per rank in parallel, assembled by a `.pvtu`, with a `.pvd` series. Set
`format = "vtu"` to choose it explicitly. `format = "xdmf"` writes the mesh
once (`PREFIX_mesh.h5`), one HDF5 file of fields per snapshot and XDMF
indexes (`PREFIX.xmf` for the series), all in global cell order; read it in
Python with the [Python package](python.md) (`mallard.open("PREFIX.xmf")`).

<a id="xdmf-in-paraview"></a>**XDMF in ParaView.** The `.xmf` files name their
HDF5 files relative to themselves, as XDMF allows. ParaView's XDMF reader
(ParaView 6.0.1, Xdmf3) gets this wrong when it is given a bare file name, as in
`pvpython script.py flow.xmf` or `OpenDataFile("flow.xmf")`: it fails to open
`flow.xmfflow_mesh.h5` and aborts with a misleading error about the number of
dimensions. Give it a path with a directory in it: `./flow.xmf`,
`solut/flow.xmf` or an absolute path. The ParaView GUI passes absolute paths, so
files opened from it are not affected. VTKHDF does not have this problem.
Mallard does not write absolute paths into the `.xmf` files, because then a run
directory could no longer be moved or copied.

**Asynchronous and compressed output.** Writing large snapshots can take longer than many steps. By default (`[output] async = true`) each snapshot is copied out and written on a background thread while the run goes on, with the same files byte for byte as `async = false`; `io_threads` copies and compresses on several threads (GPU runs leave the host cores idle):

```toml
[output]
io_threads = 8              # async = true is the default

[[write_data]]
prefix = "solut/flow"
interval = 500
fields = ["RHO", "U", "P", "T"]
compression = "zlib"       # lossless; readable by any ParaView
relative_error = 1e-4       # optional: lossy, |x' - x| <= 1e-4 |x| per value
```

Compressed VTU uses VTK's own zlib compressor and compressed VTKHDF and HDF5 the standard shuffle and deflate filters, so ParaView and h5py read them unchanged. VTU and restart compression needs a build with `-DMallard_ENABLE_ZLIB=ON`. `relative_error` rounds each value to the fewest mantissa bits within the bound (the mesh stays exact), which compresses smooth fields several times further; restart files are always lossless. On 4.2M quads on an A100, a VTU snapshot of `RHO`, `U`, `P`, `T` is 474 MB, 164 MB with `zlib` and 59 MB with `relative_error = 1e-4`.

Pitfalls: the run's summary reports the time spent writing; if each file takes longer to write than the steps between files, the run waits for the previous file anyway (write less often, or compress with more `io_threads`). Errors writing a file surface at the next file or at the end of the run, which waits for every file; a run killed outright (e.g. `kill -9`) can leave the file being written incomplete, while the files before it are complete. VTKHDF and HDF5 output on several ranks stays synchronous (parallel HDF5 calls MPI only from the main thread).

**Restart files** (`format = "restart"`) are described in
[Initialization and restarts](initialization.md#restarts).

### Snapshots of regions

Large runs need the whole field rarely and a few planes, a near field or the
walls often. `region` writes just that, at its own interval, with the same keys
as a whole-mesh snapshot:

```toml
[regions]
near_field = {type = "sphere", center = [0, 0, 0], radius = 0.05}
x1 = {type = "cut_plane", point = [0.01, 0, 0], normal = [1, 0, 0]}
x2 = {type = "cut_plane", point = [0.02, 0, 0], normal = [1, 0, 0]}
x3 = {type = "cut_plane", point = [0.04, 0, 0], normal = [1, 0, 0]}

[[write_data]]                # the cells of a volume
prefix = "solut/near"
time_interval = 1e-4
region = "near_field"
fields = ["RHO", "U", "P"]

[[write_data]]                # three slices: solut/x_x1.vtkhdf, solut/x_x2.vtkhdf, solut/x_x3.vtkhdf
prefix = "solut/x"
time_interval = 2e-5
regions = ["x1", "x2", "x3"]
fields = ["U", "P", "MASS_FLUX"]

[[write_data]]                # a boundary zone, by its name
prefix = "solut/wall"
interval = 50
region = "cylinder"
fields = ["P", "T"]
```

![A disk's cells with their REGION_FRACTION, and their pressure with three slices across the field, on a triangle mesh](../images/guide/region_snapshots-light.svg#gh-light-mode-only)
![A disk's cells with their REGION_FRACTION, and their pressure with three slices across the field, on a triangle mesh](../images/guide/region_snapshots-dark.svg#gh-dark-mode-only)

What a snapshot holds depends on the region's dimension:

| Region | Written | Values |
|---|---|---|
| A volume (an area in 2D): any region of [`[regions]`](geometry.md#regions), e.g. `sphere`, `box`, booleans | The cells it covers (cell fraction above 0), as a mesh of its own: only their nodes, curved cells as quadratic ones | The cells' fields, and `REGION_FRACTION`, each cell's [volume fraction](geometry.md#regions-on-a-mesh) of the region |
| `"domain"` | The whole mesh, as without `region` | The cells' fields |
| A boundary zone, by its name | Its faces: polygons (3D), a polyline (2D) | Point values at the face centroids, as a [boundary probe](probes.md#boundary-data) with `sampling = "point"` writes them, wall quantities included; other cell fields (`P_MAX`, `HRR`) from each face's cell |
| A surface, curve or points of [`[regions]`](geometry.md#surfaces-and-curves-on-a-mesh) (`filled = false`, `boundary`, flat disks, circles, ...) | Its elements on the mesh: polygons (3D), segments (2D), points | Point values at each element's center, with its normal and measure; other cell fields from the element's cell |
| A cut plane or other [probe type as a region](geometry.md#probe-types-as-regions) | The exact cut of the mesh, or the resampled surface, curve or points | Point values of the reconstruction, as a [probe](probes.md) of it with `sampling = "point"`; fluxes (`MASS_FLUX`, ...) through surfaces and 2D curves; other cell fields from the cell of each sample |

A volume is written by every rank in global order, like the whole mesh;
`compression`, `relative_error` and `series` apply as they do there. A boundary
zone, cut plane, surface, curve or points is written as a probe writes it, by
the same code: `vtkhdf` writes one PolyData file `PREFIX.vtkhdf` of every
snapshot (`series` is always on), `vtu` one VTK PolyData file per snapshot,
`PREFIX_<step>.vtp`, and the `PREFIX.pvd` series. Its `fields` are sampled
as [`[[probes]]`](probes.md) sample them (`U` writes `U_X`, `U_Y`, `U_Z`);
any other variable of a snapshot (`HRR`, `P_MAX`, `MEAN_P`, ...) takes the value
of the cell holding each sample. Either kind is
the same on any number of ranks, takes [units](#units) and descriptions, and
continues its series on restart. Every region is also written at the end of the
run, as the whole mesh is.

Pitfalls:

- A slice is sampled where the plane cuts each cell, from the reconstruction:
  the values are those of a [probe](probes.md#cut-planes-and-surfaces) of the plane, not
  a cell average. Across a shock, use a volume region around it instead.
- On a surface, curve or points, `RHO`, the primitives and `Y_<s>` are point
  values, while cell fields such as `HRR`, `VORTICITY` or `MEAN_*` are the
  averages of the cells the samples lie in: first order across the cell.
- A volume region much smaller than a cell can cover no cell, which is an error.
- A boundary zone's samples lie at the face centroids, on the wall: `P` is the
  reconstruction's wall pressure, not the first cell's average. Cell fields
  such as `P_MAX` are still the cell's.
- Regions are written as `vtkhdf` or `vtu` only, not as `xdmf`.
- `compression` on a region of lower dimension, alone in its entry, is an
  unused key: those files are small and written uncompressed.

## Run progress

### Terminal output

```toml
[output]
check_interval = 100   # a progress row every 100 steps
```

*Used in:*
[`sod`](https://matthewbonanni.github.io/mallard/docs/examples/#sod),
[`shu_osher`](https://matthewbonanni.github.io/mallard/docs/examples/#shu-osher),
[`riemann_2d`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d),
[`riemann_2d_quads`](https://matthewbonanni.github.io/mallard/docs/examples/#riemann-2d-quads),
[`double_mach`](https://matthewbonanni.github.io/mallard/docs/examples/#double-mach)
and 50 more.

Rank 0 prints a header (version, build, device), the setup time of each
stage, then the Mesh, Physics (with the case's units, `physics.units`),
Numerics, Boundaries and Output sections. Each section gives its settings in
words, then its quantities in a table with a units column: the cells and faces,
the extent, cell sizes and owned cells per rank as min and max, the partition's
ranks, halo layers and exchange sizes, the time step or CFL number and the
stops, and the constants of the models: the gas's (`gamma`, `R`, `cp`, `cv`),
the viscosity law's, the source terms', TENO's stencil sizes and switch, the
step controller's tolerances, LES's, TFLES's and PaSR's; the boundaries as
zone, condition, faces and details; the outputs as file, interval (in steps,
or in simulated time) and contents:

```text
== Mesh ===================================================================================
  setting     value
  Type        cartesian_tri
  Cell order  Hilbert curve (auto: 0.195 cache lines per cell, 0.348 in the input order)

  quantity        value    units  basis
  cells           460,800  cells  tri
  faces           692,400  faces
  boundary faces  2,400    faces

  quantity   min        max        units  basis
  x extent   0          4          m      nodes
  y extent   0          1          m      nodes
  cell area  8.681e-06  8.681e-06  m^2    ratio 1
...
== Output =================================================================================
  Intervals in steps, or in simulated time:
  output    file                 every  units  contents
  progress  -                    200    steps  a row of the progress table
  vtkhdf    ./solut/dmr.vtkhdf   0.002  s      RHO, U_X, U_Y, P, MACH, SCHLIEREN
```

Then comes one progress row every `[output] check_interval` steps (default 1).
A note above the table says what the columns are, and under the column names a
row gives their units:

```text
       step           t        dt   done  wall/step  throughput        ETA     min rho       min p   max Ma  troubled  written
          -           s         s      %         ms     cells/s    h:mm:ss      kg/m^3          Pa        -         %  files
          0  0.0000e+00  8.78e-04    0.0          -           -          -   1.400e+00   1.000e+00    1.827      0.00  1 vtu, 1 restart
         10  8.6159e-03  8.53e-04   16.7       37.9    1.90e+05    0:00:02   1.399e+00   9.991e-01    1.881      5.72  2 vtu
         20  1.7029e-02  8.28e-04   33.3       34.2    2.10e+05    0:00:02   1.399e+00   9.990e-01    1.882      5.89  3 vtu
         30  2.4894e-02  8.08e-04   50.0       40.5    1.78e+05    0:00:02   1.399e+00   9.989e-01    1.878      6.19  2 vtu, 1 restart
```

| Column | Units | Meaning |
|---|---|---|
| `step`, `t`, `dt` | -, s, s | Step, simulated time and time step |
| `done` | % | Of the run, by whichever stop condition comes first |
| `wall/step`, `throughput`, `ETA` | ms, cells/s, h:mm:ss | Time-stepping wall time per step and cell updates per second since the previous row, wall time to the stop |
| `min rho`, `min p`, `max Ma` | kg/m^3, Pa, - | Positivity and Mach number at a glance, over the cells |
| `troubled` | % | TENO: share of cells selecting stencils (shocks) |
| `min T`, `max T`, `sum Y-1` | K, K, - | Mixtures: temperature range, and the largest \|sum of the mass fractions - 1\| |
| `reacting`, `chem/cell` | %, steps | Chemistry: share of cells advanced, most integrator steps of a cell |
| `written` | files | Files written since the previous row, by kind (`2 vtu, 1 restart`) |

Every column keeps one unit, so a column reads straight down. In a
nondimensional case (`physics.units = "nondimensional"`, [Units](input_files.md#units))
the units row shows `-` for the simulated quantities. The files themselves
are named in the [run record](#run-record-mallardjsonl) and the summary;
`mallard -i input.toml --verbose` adds a line for each between the rows.
Requests acted on during the run (a checkpoint, a stop, a setting changed,
[Run control](running.md)) and a divergence print a remark between the rows:

```text
  >> step 339, t = 0.0610224: stop: stop requested by mallard ctl
```

The run ends with a **Summary**: why and where it stopped, a table of wall
time by phase (setup, time stepping and its chemistry, diagnostics, output),
the throughput, the bytes written and every file kind written, then the
[run health](#run-health). The same information, in machine-readable form,
goes to the run record. Errors go to stderr from any rank. Colors are off when
the output is not a terminal or `NO_COLOR` is set (`CLICOLOR_FORCE=1` forces
them, e.g. under `mpirun`).

For live monitoring and plotting, `mallard -i input.toml --log-format jsonl`
prints on stdout only the run record's `progress` records, one JSON object per
line at each progress row, exactly as `mallard.jsonl` has them, even when
`[output] run_log` is off; the text above goes to stderr:

```sh
mallard -i input.toml --log-format jsonl | jq --unbuffered '[.step, .t, .monitors.min_rho]'
```

### Run record: `mallard.jsonl`

Next to the terminal output, rank 0 writes `mallard.jsonl` (`[output] run_log`):
one JSON object per line ([JSON Lines](https://jsonlines.org)), flushed as
it is written, so scripts and dashboards can follow a running case:

```python
import json
records = [json.loads(line) for line in open("mallard.jsonl")]
progress = [r for r in records if r["type"] == "progress"]
```

A run starting from a restart file appends to the record, so a case continued
several times keeps one history, each part led by its own `start` record, which
names the restart file. Resuming from an earlier restart file than the last one
(a rewind) keeps the records after the restart step, and the `start` record says
which step the record had reached.
Each record has a `type`:

| `type` | When | Fields |
|---|---|---|
| `start` | After setup | `schema` (`"mallard-run-log"`), `version` (2), `run_id` (the run's, as in its [run block](#run-block)), `restart_of` (`run_id`, `file` and `step` of the restart file the run started from, `run_id` `null` for a file of a run before 0.10; `null` for a fresh start), `provenance` (the [run block](#run-block); its key keeps its name), `placement` (one object per rank, as the [rank report](parallel.md#the-rank-report) finds it: `rank`, `host`, `threads`, `cpus` (the CPU set before Kokkos starts, e.g. `"0-15"`; `null` where unknown), `cpu_numa`, `omp_bind`, and on GPUs `gpu` (PCI address), `gpu_uuid`, `gpu_index`, `gpu_numa`; `null` where unknown or on CPUs), `setup_phases` (seconds of each setup phase: `max` and `mean` over ranks); runs started from a restart file: `restart` (`file`, `step`, `t`, and on a rewind `rewound_from`, the last step recorded before it) |
| `progress` | Every progress row | as below |
| `output` | A snapshot completely written (also in the background): the run's output index | `step`, `t`, `kind` (`vtkhdf`, `vtu`, `xdmf`, `restart`; probes: `vtp`, `vtkhdf`; `spectra`), `path`, `reason` (as the [snapshot metadata](#snapshot-metadata)'s), `writer` (the `[[write_data]]` prefix, probe or output that wrote it), `run_id`, `complete` (`true`), `bytes` (of its files), `files` (every file of it, each rank's piece too: `path`, `bytes`, `sha256`; no `sha256` for a [file appended to](#complete-files)) |
| `event` | A request acted on (the rows between progress rows) | `step`, `t`, `kind` (`checkpoint`, `stop`, `set` and `diverged` from [run control](running.md); `trigger` from [triggers](running.md#triggers), with also `name`, `firing` (1 for the first) and `values`, the values its condition read), `text` |
| `divergence` | A run that diverged, before its `error` record | the [divergence report](running.md#the-divergence-report): `step`, `t`, `invalid_cells`, `failed` (cells by the quantity that failed first), `box` (their extent), `last_valid` (`step`, `t`, `file`, or `null`), `causes` (most likely first: `kind` (`time_step`, `mesh`, `boundary`, `initial_state`, `chemistry`, `numerics`), `title`, `score`, `evidence`, `fix`), `cell`, `neighbors`, `others` (`id`, `rank`, `x`, `failed`, `rho`, `speed`, `p`, `T`, `cfl`, `quality`, `repaired`, `troubled`), `time_step` (`integrator`, `limit`, `dt`, `dt_cfl1`, `cfl`, `cells_over_limit`), `zones` (`name`, `type`, `near`, and where compared `inadmissible`, `ratio`, `jump`), `history` (the last progress rows), `snapshot`, `units` (of `t`, `dt`, `x`, `rho`, `speed`, `p` and `T`, as the CSV files' units lines write them; `"1"` for each in a nondimensional case) |
| `warning`, `error` | A warning (those of the setup right after the `start` record), or the error that ended the run | `code` (stable: [Warnings and errors](running.md#warnings-and-errors)), `message`, `docs` (the page and anchor that explain the code) |
| `probe` | After the `start` record, for each filled region probe | `probe` (its name), `sampling` (its [probe sampling block](#probe-sampling), with the region's `measures`) |
| `end` | The end of the run, also on a stop signal, divergence or an error (on rank 0): a run without one was killed (`SIGKILL`, a lost node) or is still running | `outcome` (`finished`, `stopped`, `signal`, `wall_time_limit`, `diverged`, `error`), `cause` (`finished`: `n_steps` or `t_stop`; `stopped`: `stop_file`, `mallard_ctl` or `trigger`; `signal`: `SIGTERM`, `SIGINT` or `SIGUSR2`), `trigger` (the [trigger](running.md#triggers) that stopped the run, by name), `reason` (in words; the error's message), `step`, `t`, `last_snapshot` and `last_restart` (the last of each recorded complete: `step`, `t`, `path`; none if the run wrote none), `steps` (this run's), `wall` and its parts `wall_setup`, `wall_stepping`, `wall_diagnostics`, `wall_output` (and `wall_chemistry`), `rejected_steps` (with error control), `cells_per_second` |

Progress records:

| Field | Meaning |
|---|---|
| `step`, `t`, `dt` | Step, time, time step |
| `cfl` | `dt` over the stable time step at CFL 1 |
| `progress` | Fraction of the run, by the first stop condition to be reached |
| `wall` | Seconds since the solver started |
| `wall_per_step`, `cells_per_second` | Time stepping since the previous progress record; `null` before any step |
| `eta` | Seconds remaining, `null` until known |
| `cells`, `ranks` | Global cell count and rank count |
| `memory_host`, `memory_host_peak` | Resident memory of the largest rank, and its peak, in bytes |
| `memory_device`, `memory_device_total` | GPU memory in use (by all processes) on the fullest device, and its size; `null` without a GPU |
| `monitors` | The progress row's diagnostics: `min_rho`, `min_p`, `max_mach`; TENO: `troubled_cells` (fraction); mixtures: `min_T`, `max_T`, `max_sum_Y_error`; chemistry: `reacting_cells` (fraction), `max_chemistry_cost` |
| `integrals` | The latest `[integrals]` (`step`, `t` and the columns of `integrals.csv`) |
| `forces_<zone>` | The latest `[[forces]]` of each zone (`step`, `t` and the columns of its file) |
| `health` | The [run health](#run-health) at this row (`[output] health`) |

Numbers are in SI units (seconds, bytes) and read back exactly; unknown
values are `null`. Within a `version`, records only gain fields; renaming,
removing or changing the meaning of one increments it. Version 2 (0.10) added
`run_id`, `restart_of` and the `output` records, which replace version 1's
`event` records of written files; `--restart latest` and `mallard info DIR`
find a run's files through them.

#### Complete files

A file is written under the name `NAME.partial` and gets its name `NAME`
once complete, as restart files always were: snapshots (each rank's piece
too), their indexes (`.pvd`, `.vtkhdf.series`, `.xmf`), XDMF data and mesh
files, probe snapshots and spectra. A file under its own name is therefore
never cut short by a job killed while writing it, and a `NAME.partial` left
behind is one whose write did not finish: `mallard info`, `mallard diff` and
`mallard post extract` refuse it and say so, and `--restart latest` never
picks it.

The `output` record of a snapshot is written once all its files are
complete, with each file's size and SHA-256, so a snapshot without a record
was not completely written. One thread hashes the files after their write, on
the CPU's SHA instructions where it has them (over 1 GB/s); a run that writes
faster than that waits for it, and the run summary says how long.
`mallard info DIR` checks every recorded file against its record (its size;
`--verify` also its SHA-256) and reports one that is missing or differs.

Files appended to in place cannot be renamed: a series in one file
(`series = true`), a probe's HDF5 or VTKHDF series and the CSV files of
probes, forces and integrals. A job killed while appending can leave such a
file damaged (HDF5 files are not safe against it) or ending in a partial
row; the step's missing `output` record says that step was not completely
written. Their records give the size after each step and no SHA-256, which
would rehash the whole file at every step.

## Run health

At every progress row Mallard also checks that the run is healthy, and
writes what it finds into the progress record's `health` (and a short
**Health** section at the end of the terminal output). Everything is computed at
progress rows only, from the state of that step. Between rows the ledger
adds a reduction over the boundary faces (and one over the cells where
there are volume sources) per stage, chemistry two reductions of the
element totals per call and a sub-step count per integrated cell; none of
these waits for the device. `[output] health = false` turns
it all off.

| Field | Contents |
|---|---|
| `margins` | The smallest `rho`, `p`, `T` and, for mixtures, mass fraction (`Y_<species>`) over the domain: `value`, cell centroid `x` and `step` |
| `margins_run` | The smallest of each at any progress row so far |
| `activity` | Fractions of cells, over the domain (`all`) and the least and most active rank (`rank_min`, `rank_max`): `troubled_cells` (TENO's troubled-cell indicator), `limited_cells` (MUSCL cells whose limiter is below one in some variable), `upwind_fraction` (the hybrid flux's mean upwind fraction) |
| `chemistry` | Since the previous row: `substeps` and `rejected_substeps` of the Rosenbrock integration, `failures`, `reacting_cells` (fraction of the last call), `wall` (seconds of the slowest rank) and `imbalance` (the slowest rank's chemistry time over the ranks' mean) |
| `timing` | Per step since the previous row, `all` (mean), `rank_min`, `rank_max` over ranks: `output_sync` (the run's own output time), `output_async` (time of the background writers) and, with probes of [regions defined by the solution](geometry.md#regions-defined-by-the-solution), `regions` (the part of `output_sync` spent placing them on the solution). The step's own wall time and throughput are the row's `wall_per_step` and `cells_per_second` |
| `repairs`, `repairs_run` | Each [numerical repair](#numerical-repairs), since the previous row and since the start: `activations`, `cells` (distinct owned cells) and `change`, the integrated change of each quantity it changes |
| `balance`, `balance_run` | The [conservation ledger](#conservation-ledger), since the previous row and since the first row |
| `element_drift` | Mixtures with chemistry: the largest change of a rank's element mass over one chemistry call, beyond the recorded repairs, relative to the rank's mass (round-off) |

A breakdown of the step by phase is a profiling question, not a health check:
timing phases needs a fence after each one, which a CUDA-graph step does not
have; use a profiler (Nsight Systems, Kokkos Tools) for it.

Mallard warns once per kind when the density or pressure falls below
10<sup>-3</sup> of its domain mean, when a mass fraction is below
-10<sup>-6</sup>, and when a checked balance has an unexplained relative
change above 10<sup>5</sup> machine epsilons.

### Conservation ledger

The ledger is an independent account of the domain totals of mass,
momentum, total energy and, for mixtures, each element's mass. It does not
read the solver's face fluxes or residual: at each progress row it
integrates the totals from the conservative variables, and compares their
change with

- `boundary`: the transfers through the boundary faces, which it evaluates
  itself at every stage from the faces' states (the interior reconstruction
  and the boundary condition's exterior state), normals and areas, with the
  convective and viscous point fluxes of the scheme (wall models included),
- `sources`: gravity, `[source]` fields, the mass flow forcing and the
  sponges, re-evaluated from each stage's state,
- `repairs`: the recorded changes of the [numerical repairs](#numerical-repairs),

each stage weighted by the time integrator's weights b<sub>i</sub> and
committed with its time step once the step is accepted. The fields of each
quantity are `total`, `change`, `boundary`, `sources`, `repairs`,
`residual` (the change none of these explain), `relative` (residual over
the larger of the totals and their absolute integral; for momentum,
sum |&rho;u| + (2 &rho; &rho;E)<sup>1/2</sup>), `checked` and, where the
ledger cannot close a quantity, a `note` saying why. A conservative setup
closes at round-off, in a periodic box as with inflow, outflow and walls;
a missing factor or a wrong sign in the update shows as a residual many
orders larger.

Not closed (`checked = false`): boundary transfers of gas mixtures (their
mass and element balances close where the boundaries are walls, symmetry
planes or periodic), the energy with double flux (not conservative) or
radiation, the radial momentum of axisymmetric runs, element masses with
sponges, and every quantity with `coupling = "simpler"`.

### Numerical repairs

Every place that changes the state outside the discretized equations is
audited:

| Site | What it does | Recorded change |
|---|---|---|
| `composition_clip` | Chemistry: negative mass fractions are clipped to zero and the composition renormalized, before and after a cell's integration (the energy stays exact) | `species_<name>` and `element_<symbol>` masses |
| `energy_reset` | Double flux: after each step, &rho;E is reset to the true equation of state at the pressure of the frozen one | `energy` |

The temperature inversion (Newton from the cell's previous temperature,
safeguarded by bisection within the thermodynamic fits' range) never
changes the state, and a failed chemistry integration ends the run with an
error, so neither is a repair.

## In every output file

### Units

Every value Mallard writes is in SI units (the mesh-quality angle of
`mallard check` excepted, in degrees), and every output file says which,
as metadata in the way its format's tools expect; field names never change.
Restart files are the exception: they hold the conservative variables, in SI
units by definition, for Mallard to read back.

| File | Where the units are |
|---|---|
| VTKHDF, HDF5 (snapshots, their mesh file, probes) | CF-convention string attributes of each dataset: `units` ([UDUNITS](https://docs.unidata.ucar.edu/udunits/current/) syntax, `1` for dimensionless) and `long_name`; `standard_name` only where a CF standard name fits, which no field does yet (`air_pressure` and the like are for atmospheres); `cell_methods` for [cell averages and point values](#cell-averages-and-point-values) |
| XDMF (`.xmf`) | `<Information Name="units" Value="..."/>` and `<Information Name="long_name" .../>` in each `<Attribute>` |
| VTU, VTP (each piece of a PVTU) | Field data: the string arrays `MALLARD_FIELD_NAMES`, `MALLARD_FIELD_UNITS` and `MALLARD_FIELD_LONG_NAMES`, entry *i* of each for the same array (VTK has no units attribute) |
| CSV (forces, integrals, probes) | A line `# units: ...` after the header, one entry per column |
| Spectra CSV (`[statistics.spectra]`) | A last column `units`: the units of each row's `value` |

```python
import mallard                                     # the Python package: any file Mallard writes
f = mallard.open("solut/flow_000100.vtkhdf")
f.fields["U"].units, f.fields["U"].long_name       # ('m s-1', 'velocity')
mallard.open("forces_cylinder.csv").fields["Fx_pressure"].units   # 'kg s-2' (2D: per unit depth)

import h5py                                        # or the attributes themselves
f = h5py.File("solut/flow_000100.h5")
f["fields/P"].attrs["units"], f["fields/P"].attrs["long_name"]   # ('kg m-1 s-2', 'pressure')
```

These are the attributes xarray and netCDF tools look for, readable as
they are. Units are written in SI base units (`kg m-1 s-2`, not `Pa`), so that
derived fields compose: `MEAN_<A>` and `PHASE_<bin>_<A>` have the units of
`A`, `COV_<A>_<B>` their product (`COV_U_X_U_Y`: `m2 s-2`), and `SOURCE_<A>`
those of `A` per second.

| Field | Units | Field | Units |
|---|---|---|---|
| `RHO`, `RHOY_<s>` | `kg m-3` | `MU`, `MU_T` | `kg m-1 s-1` |
| `RHOU` | `kg m-2 s-1` | `LAMBDA` | `kg m s-3 K-1` |
| `RHOE`, `P`, `P_MAX` | `kg m-1 s-2` | `D_<s>` | `m2 s-1` |
| `U` | `m s-1` | `DT_<s>` | `kg m-1 s-1` |
| `T`, `T_SEED` | `K` | `HRR`, `QRAD` | `kg m-1 s-3` (W/m³) |
| `H` | `m2 s-2` (J/kg) | `OMEGA_<s>` | `kg m-3 s-1` |
| `VORTICITY` | `s-1` | `CHEM_H` | `s` |
| `Q_CRITERION`, `LAMBDA2` | `s-2` | `CFL`, `MACH`, `TENO_SIGMA`, `Y_<s>`, `X_<s>`, `CHEM_COST`, `TF_F`, `TF_E`, `TF_OMEGA`, `PASR_KAPPA` | `1` |
| `NON_ORTHOGONALITY` (`mallard check`) | `degree` | `SKEWNESS`, `ASPECT_RATIO`, `VOLUME_RATIO`, `DELTA_OVER_ETA`, `CELLS_PER_DELTA_L` (`mallard check`) | `1` |
| `MASS_FLUX`, `SPECIES_FLUX_<s>`, `SPECIES_DIFFUSIVE_FLUX_<s>`, `SPECIES_TOTAL_FLUX_<s>` (probes) | `kg m-2 s-1` | `ENTHALPY_FLUX`, `ENERGY_DIFFUSIVE_FLUX`, `ENERGY_TOTAL_FLUX`, `Q_W` | `kg s-3` (W/m²) |
| `MOMENTUM_FLUX`, `TRACTION_PRESSURE`, `TRACTION_VISCOUS`, `TAU_W` | `kg m-1 s-2` | `U_TAU` | `m s-1` |
| `CF`, `CP`, `Y_PLUS` | `1` | `REGION_FRACTION` (regions) | `1` |
| `SCHLIEREN` | `kg m-4` | | |
| `WALL_MODEL_U` | `m s-1` | `WALL_MODEL_T`, `WALL_MODEL_H` | `K`, `m` |

CSV columns: `t` in `s`, coordinates, `s` and probe `length` in `m`, `step`,
`point`, `face`, `cell`, `count` and the normal `nx`, `ny`, `nz` `1`; the
`area` of a sample of a boundary, cut plane or surface, and its reductions'
`area`, in `m2` (in 2D, where these are curves, `m`, the reductions' column
`length`); forces in `N` (`kg m s-2`), per unit depth in 2D
(`kg s-2`), and the wall model's `Q` in `W`; domain integrals carry the
volume's `m3` (per unit depth in 2D: `m2`); probe reductions have their
variable's units, `integral` times those of the measure (`m` along a curve,
`m2` over a 3D boundary, cut plane or surface: `integral_MASS_FLUX` in `kg s-1`; `moment` those of
`integral` times `m`; over a region its
`volume`, `m3`, or `area`, `m2`, per unit depth in 2D). In axisymmetric runs,
forces, integrals and region probes are per radian of the revolved domain, and their units
say so: `kg m s-2 rad-1`, `kg m2 s-2 rad-1`; so are boundary probes' `area`
(`m2 rad-1`), region probes' `volume` (`m3 rad-1`) and `integral` reductions. Spectra rows: a `mean` has its
field's units, `spectrum`, `correlation` and `shell` (sums, not densities)
the product of their pair's (`U_X*U_Y`: `m2 s-2`); the `abscissa` is a
wavenumber in `m-1` (spectra, shells) or a separation in `m` (correlations).
Files written before Mallard 0.9 have no units; the readers return them empty.

### Run block

Every output file records the run that wrote it: the run block, one JSON
object, the same in every file of that run (where [snapshot
metadata](#snapshot-metadata) is each snapshot's own). It is written into
restart files (a trailer after the data, also of compressed files), VTU,
`.pvtu`, `.pvd`, XDMF and probe `.vtp` files (an XML comment `<!-- mallard-run
{...} -->` after the XML declaration), VTKHDF and HDF5 files, probe files
included (the string attribute `mallard_run` of the root group; a continued
run's file holds the latest run's), mesh quality files and the case view, CSV
files of forces, integrals, probes, region reductions, spectra and the
reactor (a line `# mallard-run {...}` after the header and its [units
line](#units), and another where a continued run starts appending: the rows
after it are that run's), and the run record's `start` record.

| Field | Contents |
|---|---|
| `schema`, `version` | `"mallard-run"`, 3 |
| `mallard` | `version` (`git describe`), `git_sha`, `git_dirty` (uncommitted changes in the build); `null` outside a git checkout. Taken at build time |
| `dependencies` | Versions of `compiler`, `kokkos`, `mpi` (the library's own string and the standard), `hdf5`, `cgns`, `zlib`, `nccl`, `gpu_runtime` and `gpu_driver` (CUDA or HIP); `null` for those not built in |
| `build` | `dim`, `precision`, `build_type`, `compiler`, `cxx_flags`, `kokkos` (version), `backend` and `host_backend` (Kokkos execution spaces), `mpi`, `gpu_aware_mpi`, `nccl`, `hdf5`, `hdf5_parallel` (HDF5 built with MPI), `kaminpar`, `species_layout` |
| `run` | `run_id` (a random UUID, new for every run, restarts included), `start_time` (UTC, ISO 8601), `ranks`, `host`, `device` (rank 0's), `directory`, `hosts` (the number of hosts; where each rank ran is the run record's `placement`), `restart_of` (as the run record's: `run_id`, `file`, `step`; `null` for a fresh start) |
| `input` | `file`, `sha256` of its bytes, its `text`, and `resolved`: the input with every key explicit, as the run used it ([the resolved input](input_files.md#the-resolved-input), what `mallard input resolve` writes) |
| `mesh` | `type`; for mesh files `file` and the `sha256` of its bytes (of a directory: of each file's relative path and bytes, by path; skipped with `[output] hash_mesh = false`) |
| `files` | Each other file the run read, once (absent if none): `kind` (`mechanism`, `table`, `inflow_profile`, `probe_points`, `restart`, `remap_source`, `remap_mesh`), `path` as given, `resolved`, `bytes`, and `sha256` of its bytes. A restart file or remap source is not hashed again: it records the `run_id`, `step` and `t` of the run that wrote it, and the `sha256` from that run's [output record](#run-record-mallardjsonl) of the file (`sha256_from`). Where there is none (a copied or moved file, a run before 0.10, a changed file) and for a remap source mesh, `sha256` is `null`, with the file's `modified` time and a `note` saying why |
| `preset` | The `[numerics] preset` (`null` without one): `name`, `default` (`true` when the input names no preset and gets the default, `high_order`), and per key it covers its `key`, its `value` in the resolved input (as TOML: `"\"HLLC\""`, `0.5`) and `from_preset` (`false` where the input overrides it) |

Before 0.10 the block was named the provenance block: schema
`"mallard-provenance"`, the comment `<!-- provenance {...} -->`, the attribute
`provenance` and the CSV line `# provenance {...}`, without `run_id` and
`restart_of`. Every reader (`mallard info`, the post-processing commands,
the [Python package](python.md)) reads those names too, and says so. Version 2 (development builds after 0.9.1) added
`input.resolved`; version 1 (0.9) had none, its preset values were plain text
(`"HLLC"`), and a `setup` object held the terminal output's setup summary as
`[key, value]` pairs per section.

Check that a file belongs to a given input with `sha256sum input.toml`; read
the block in Python as `mallard.open(path).run`, a dict (any file; each
snapshot's `run_id` says which run of a continued file wrote it).
`mallard info FILE` summarizes the block, and `mallard info --input FILE`
prints the input that produced the file, ready to run again
([Command line](cli.md#mallard-info)).

### Snapshot metadata

Every snapshot records which snapshot it is, in its format's own slot and in
a snapshot block, one JSON object per snapshot:

| Format | Step and time | Snapshot block |
|---|---|---|
| VTU, probe VTP | field data `TIME` (s) and `STEP` | XML comment `<!-- mallard-snapshot {...} -->` |
| PVTU | its pieces' | XML comment, as VTU |
| PVD index | `timestep` and `step` of each `DataSet` | its members' |
| VTKHDF, one snapshot per file | field data `TIME` and `STEP` | dataset `mallard_snapshot` of the root group |
| VTKHDF, `series = true`, and probe VTKHDF | `Steps/Values`, and field data `TIME` and `STEP` per step | dataset `mallard_snapshot`, one row per step |
| `.vtkhdf.series` index | `time` and `step` of each entry | its members' |
| XDMF | `<Time>` and `<Information Name="step">` of each grid; root attributes `time` and `step` of each `.h5` | dataset `mallard_snapshot` of each `.h5` |
| Restart | header `step` and `t` | trailer before the run block's |
| Probe HDF5 | datasets `time` and `step` | root attribute `mallard_runs`: each run's `run_id`, `restart_of`, `first_row`, `first_step` |
| CSV: forces, integrals, probes, reductions | columns `step` and `t` | each run's `# mallard-run` line precedes its rows |
| Spectra CSV | in its block | line `# mallard-snapshot {...}` |
| Reactor CSV | columns `step` and `t` | `# mallard-run` line |

The block's fields:

| Field | Contents |
|---|---|
| `schema`, `version` | `"mallard-snapshot"`, 1 |
| `run_id`, `restart_of` | The run that wrote it, as its [run block](#run-block)'s: a continued series holds the snapshots of each run that wrote to it |
| `step`, `time` | Step; simulated time in s |
| `written_at` | UTC, ISO 8601 |
| `reason` | `interval` (its `interval` or `time_interval`), `start` (the initial state), `end` (the run's end), `stop` (a stop request or signal), `checkpoint` (a [run control](running.md) request), `divergence` (the last valid state, written as the run diverged), `trigger` (a [trigger](running.md#triggers)'s `write`) |
| `format`, `format_version`, `solver_precision` | The file's format and its [version](#formats-and-compatibility); the precision the solver ran in (`double`, `single`) |
| `statistics` | Snapshots of time-averaged fields (`MEAN_`, `COV_`, `PHASE_`) and spectra: `average_start`, `average_end` (s), `samples`; absent otherwise |
| `reconstruction` | Snapshots with point values: how they were evaluated, `method` (`TENO`, `linear_least_squares`), its design `order`, and `fallback`, what replaces it in order where it cannot be used ([below](#cell-averages-and-point-values)); absent otherwise |
| `fields` | Per field: `name`, `units`, `long_name`, `stored` (`float64`, `float32`: as written, which a single-precision build or output makes `float32`), `value` (`cell_average`, `cell_state`, `point_value`: [below](#cell-averages-and-point-values)), and for time-averaged fields `average` ([below](#time-averaged-fields)) |
| `units` | The units of the block's own numbers (`time`: `s`) |

Files written before 0.10 have no snapshot block. Readers then take step and
time from the format's own slots, and say so: VTU, VTKHDF and the indexes had
no step, so their snapshots are numbered by their position in the file or
index, from 0; their run is that of the file's run block, and their reason
and stored precision are unknown. In Python, `mallard.open(path).snapshots[k].block`
is a snapshot's block as a dict (`None` for older files), and
`mallard info FILE` prints the last one under **This snapshot**.

### Cell averages and point values

A finite-volume solution is a set of cell averages: a cell's `RHO` is the
mean density over the cell, not the density at its centroid, and above
second order the two differ by the scheme's truncation error. Every snapshot
says which of three kinds of value each field holds, as `value` in its
[snapshot block](#snapshot-metadata):

| `value` | What it is | Fields |
|---|---|---|
| `cell_average` | The average of the quantity over the cell | The solver's unknowns `RHO`, `RHOU`, `RHOE`, `RHOY_<s>` (restart files hold only these), their `MEAN_` and `PHASE_` averages, `REGION_FRACTION` |
| `cell_state` | Computed pointwise from the cell's averaged state: at high order neither the cell average nor the centroid value of that quantity (the pressure of the mean state is not the mean pressure) | Every other cell field: `P`, `T`, `U`, `MACH`, gradient fields such as `VORTICITY`, `COV_`; probes and boundary samples with `sampling = "cell"` other than the unknowns; mass fractions in every mode |
| `point_value` | The reconstruction evaluated at the sample | [Probes](probes.md) with `sampling = "point"` and [snapshots of regions](#snapshots-of-regions) of lower dimension, except mass fractions and the cell fields they copy |

Point values are a snapshot's `reconstruction`: with TENO its polynomials
at the scheme's `order`, falling back to linear least squares in troubled
cells and to the cell average where neither gives a positive density and
energy
(`{"method": "TENO", "order": 5, "fallback": ["linear_least_squares", "cell_average"]}`);
with MUSCL or first order, linear least squares (order 2). The same
distinction is in the format's own metadata where it has a slot: the
[CF](https://cfconventions.org/) attribute `cell_methods` of each VTKHDF and
HDF5 dataset and an `<Information Name="cell_methods">` of each XDMF
attribute, `area: mean` (3D: `volume: mean`) for a cell average and
`area: point` (`volume: point`) for a point value, and none for a
`cell_state`, which no CF method describes.

Compare like with like. An error norm against an exact solution, or a
convergence study, of cell averages takes the exact solution's cell averages
(or `RHO` against `RHO`); a point value is compared with the exact value at
the point. Mixing the two hides an error of the scheme's order at high order.
`mallard diff` warns when A and B hold different kinds of values of a field
(a **Value kinds** table, `value_a` and `value_b` per field in its JSON),
`mallard info FILE` lists each field's `value` and the reconstruction under
**This snapshot**, and `value_kinds(path)` of `tools/mallard/vtu.py` returns
them. Files written before 0.10 record none.

### Time-averaged fields

A `MEAN_`, `COV_` or `PHASE_` field records what it averages, as `average`
in its field's entry of the [snapshot block](#snapshot-metadata), so a
script never infers it from the name:

| Key | Contents |
|---|---|
| `kind` | `reynolds` (a time mean, not weighted by density), `covariance` (`<a'b'>`) or `phase` (a [phase mean](statistics.md#phase-averages-statisticsphase)) |
| `of` | The variables averaged: `["U_X"]`, two for a covariance (`["P", "U_X"]`) |
| `weighting` | `time`: each sample weighs the time since the previous one (the first, since the start) |
| `start`, `end` | The window, s: where the first sample's weight starts, and the last sample |
| `samples`, `interval` | The samples taken, every `interval` steps |
| `bin`, `bins`, `bin_weight`, `period` or `trigger` | Phase means: the field's bin, of how many, the time its samples weigh (s), and the fixed period (s) or the variable whose crossings set it |

```json
{"name": "COV_P_U_X", "units": "kg s-3", "value": "cell_state",
 "average": {"kind": "covariance", "of": ["P", "U_X"], "weighting": "time",
             "start": 0.25, "end": 1.25, "samples": 64, "interval": 2}}
```

Mallard keeps Reynolds statistics only; [Favre means follow from
them](statistics.md#time-averages-statistics) with `RHO` among the fields.
The averages of a probe with `average = true` record the probe's own window,
and VTKHDF, HDF5 and XDMF datasets of time and phase means add `time: mean`
to their `cell_methods` (`area: mean time: mean` for `MEAN_RHO`). `mallard
info FILE` tabulates them under **This snapshot**, and `field_averages(path)`
of `tools/mallard/vtu.py` returns them.

### Probe sampling

A probe file says where each of its samples is read, so it stands alone
without the input or the mesh. Its probe sampling block (`schema`
`mallard-probe-sampling`, version 1) is the line `# mallard-probe-sampling
{...}` after each run's `# mallard-run` line of a CSV file (of the
reductions file, for a probe without a samples file, such as a region probe), the root
attribute `mallard_sampling` of an HDF5 or VTKHDF file, and an XML comment
of each VTP file:

| Key | Contents |
|---|---|
| `probe`, `sampling` | The probe's name and `cell` or `point` |
| `reconstruction` | For `sampling = "point"`, the method, order and fallbacks the values are read with (as in [Cell averages and point values](#cell-averages-and-point-values)); `null` for `cell` |
| `variables` | Each column's `name` and `value` kind, and with `average = true` its `average` (`kind`, `of`, `weighting`, window `start`) |
| `points` | For `point` and `points` probes, curves and resampled surfaces: `n` samples, `max_distance` and `mean_distance`, and per sample, in sample order, the `requested` point, the owning `cell` (global id), its `centroid` and the `distance` from the requested point to where the value is read; `null` for probes of a region's cells, a cut plane or a boundary's faces |
| `clipped` | For a curve or resampled surface partly outside the mesh: the samples `dropped`, `of` how many, and the dropped share of its length (`length_fraction`) or area (`area_fraction`); for a filled region partly outside the domain, the share of its volume outside (`volume_fraction`; planar 2D: `area_fraction`); `null` when nothing was dropped |
| `measures` | For a filled region probe ([Measures](probes.md#region-probes)): the `region`'s own `volume` and `surface_area` (planar 2D: `area` and `perimeter`; axisymmetric: per radian) with their relative `tolerance` (0: closed form), `null` for a region of the mesh or an unbounded one, and the same of its part in the domain (`in_domain`); `null` for other probes |
| `units` | Units of the dimensional keys (m, s) |

The distance is zero for a point value and the distance to the centroid of
the cell for a cell sample. Every sample lies in the mesh: a point on a
boundary face counts as inside, a `point` or `points` probe with a point
outside is an input error, and the samples of a curve or resampled surface
outside the mesh are dropped with a warning
([Samples outside the domain](probes.md#samples-outside-the-domain)), absent
from `points` and counted in `clipped`, so a clipped file never looks like
the whole curve.

```json
{"schema": "mallard-probe-sampling", "version": 1, "probe": "wake",
 "sampling": "point",
 "reconstruction": {"method": "TENO", "order": 3, "fallback": ["linear_least_squares", "cell_average"]},
 "variables": [{"name": "RHO", "value": "point_value"}, {"name": "P", "value": "point_value"}],
 "points": {"n": 6, "max_distance": 0.0, "mean_distance": 0.0,
            "requested": [[0.3, 0.5], [0.3, 0.6], ...], "cell": [5, 9, ...],
            "centroid": [[0.375, 0.375], [0.375, 0.625], ...], "distance": [0.0, 0.0, ...]},
 "clipped": {"dropped": 5, "of": 11, "length_fraction": 0.5},
 "units": {"points.requested": "m", "points.centroid": "m", "points.distance": "m",
           "points.max_distance": "m", "points.mean_distance": "m"}}
```

`mallard info FILE` prints it under **Probe sampling** (with `--json`, as
`sampling`), and `probe_sampling(path)` of `tools/mallard/vtu.py` returns it
for any probe file. Files written before 0.10 record none.

### Formats and compatibility

Every file Mallard writes carries a format version. **The promise:** a release
reads the restart and mesh files of every earlier release of the same major
version. In 0.x that means every restart file since 0.1 and every HDF5 mesh since
0.3. A file from a *newer*
Mallard is rejected with a message that gives its version and the versions this
build reads. It is never misread.

| File | Version, where | Versions (written from) | Read by |
|---|---|---|---|
| Restart (`format = "restart"`) | `uint32` after `MALLARD-RESTART` | 1 (0.1), 2: named variables (0.4), 3: `[statistics]` weights (0.6), 4: characteristic boundary states (0.7), 5: `compression = "zlib"` (0.9), 6: `[statistics.spectra]` and `[statistics.phase]` state (0.9), 7: the [snapshot block](#snapshot-metadata) trailer (0.10) | Mallard: all |
| HDF5 mesh (`mallard mesh convert`) | root attribute `version` | 1 (0.3) | Mallard: all |
| VTKHDF output (`format = "vtkhdf"`) | root attribute `version` (VTK's own in `/VTKHDF`'s `Version`: 2.0) | 1 (0.9) | ParaView 5.12+, the [Python package](python.md) |
| XDMF output (`format = "xdmf"`; `"hdf5"` before 0.10) | root attribute `version` of both files | 1 (0.6) | ParaView, the [Python package](python.md) |
| VTU output | field data `MALLARD_FORMAT_VERSION` | none before 0.9, 1 (0.9) | ParaView, the [Python package](python.md) |
| Run record (`mallard.jsonl`, `[output] run_log`) | `version` of each `start` record | 1 (0.9), 2: `run_id`, `restart_of`, `output` records (0.10) | Your scripts (records only gain fields within a version) |
| [Run block](#run-block) (every output file) | its `version` | 3: `mallard-run`, `run_id`, `restart_of` (0.10); as the provenance block: 2 (development builds after 0.9.1), 1 (0.9) | `mallard info`, the [Python package](python.md) (`File.run`, every version) |
| [Snapshot block](#snapshot-metadata) (every snapshot) | its `version` | 1 (0.10) | `mallard info`, the [Python package](python.md) (`Snapshot.block`) |
| Probe output (`[[probes]]` `format = "hdf5"`, `"vtkhdf"`, `"vtp"`) | root attribute `version` (HDF5, VTKHDF), field data `MALLARD_FORMAT_VERSION` (VTP) | 1 (0.9) | Mallard (restarts append to HDF5), ParaView |
| TENO cache (`cache_file`) | magic `MALLARD-TENO-<n>` | - | Rebuilt automatically when it does not match |

- Restart files are written with version 7 since 0.10, whose snapshot block
  trailer earlier releases would take for a truncated file: they reject it
  with their newer-version message. (Until 0.9 they were written with the
  lowest version that held their contents.)
- A restart file must also match the build's dimension and precision
  (`Mallard_DIM`, `Mallard_USE_DOUBLE`); the message says which differs.
- Output files gain arrays and attributes within a version. A version changes
  only when existing content changes meaning or layout. The Python tools refuse
  versions newer than they know; update `tools/` with Mallard.
- A change to a format adds a version and keeps the reader of every older one,
  with a fixture in `test/data/formats/` that a test reads (see its
  [README](../../test/data/formats/README.md)).

## Post-processing

`mallard info` tells what any of these files holds, without an input file:
its kind, format version, mesh, times, fields with their units, ranges and
stored precision, its run block (**Run**, the same for every file of the run)
and the last snapshot's block (**This snapshot**); `mallard info solut` does so for each file a run wrote, one table
row each ([Command line](cli.md#mallard-info)). Every number comes in a table
column or a labeled line that names its units (`t [s]`, `min [m]`, a field's
units, `dimensionless`); a `-` is something the file does not record:

```
$ mallard info solut/flow.vtkhdf.series
== solut/flow.vtkhdf.series ===============================================================
  Kind               snapshot
  Format             VTKHDF series index (VTKHDF snapshots), version 1
  Values             2D, double precision
  Simulated time     3 snapshots, t = 0 to 0.002 s

== Mesh ===================================================================================
  cells  points
  12     20
...
== Snapshots ================================================================== 3 snapshots
  #  t [s]  file
  0  0      flow_000000.vtkhdf
  1  0.001  flow_000001.vtkhdf
  2  0.002  flow_000002.vtkhdf

== Fields ================================================= ranges at t = 0.002 s, the last
  field  units       components  min      max      description
  P      kg m-1 s-2  1           1        1        pressure
  RHO    kg m-3      1           1.03804  1.26232  density
  U      m s-1       3           0.3      0.3      velocity
  min and max of fields with several components are of their magnitude.
```

`--format json` prints the same as one JSON object, for scripts; in Python,
`mallard.open(path)` holds it as attributes and reads the values
([Python](python.md)).

`mallard post extract` evaluates probes, region integrals and snapshots of
chosen fields on the state of restart files and snapshots after the run, as
the run would have written them ([Probes of existing output](probes.md#probes-of-existing-output)),
and `mallard post derive` their [derived fields](#derived-fields-after-the-run).

`mallard post convergence` compares runs of a case at several resolutions:
the observed order, the extrapolated value and the grid convergence index of
their forces, integrals and probes ([Grid convergence](convergence.md)).

### Comparing two solutions: `mallard diff`

`mallard diff A B` tells whether two files hold the same solution and, if not,
how far apart and where: for a regression check in CI, two settings of a case,
a restart that should reproduce a run, or a serial run against an MPI one
([Command line](cli.md#mallard-diff)). A and B are snapshot, restart, probe or
CSV files, of any format each; it compares every field both hold (`--fields`
picks some), at the last snapshot of each, or at the one of a step (`--step N`) or
nearest a time (`--time T`) as the files record them:

```
$ mallard diff serial/solut/restart_000020.restart mpi/solut/restart_000020.restart
== Diff ================================================================================= ok
  Result: ok, identical, the same state bit for bit (exit status 0)
  A: serial/solut/restart_000020.restart (restart)
  B: mpi/solut/restart_000020.restart (restart)
  Compared: cell by cell, in the mesh's global cell order (768 cells)
  Tolerance: 0 (values must be equal); on max |B - A| / max |A| per field
...
== Fields =============================================================================== ok
  |B - A| per cell, in the field's units: L1, L2, Linf are its mean, root mean square and
  max over the cells, not weighted by volume; differing: values not bitwise equal.
  status  field   units       differing cells  L1  L2  Linf
  ok      RHO     kg m-3      0 of 768         0   0   0
  ok      RHOU_X  kg m-2 s-1  0 of 768         0   0   0
...
```

- **Per field**, the `Fields` table gives the cells whose values are not
  bitwise equal and the L1, L2 and Linf norms of B - A (mean, root mean square
  and max over the cells, not weighted by volume) in the field's units;
  `Relative differences` divides each by the same norm of A; `Largest
  differences` names the cell of the max, its coordinates and both values.
  Vectors are compared by the magnitude of the difference; a NaN in one file
  only is an infinite difference.
- **Restart files** are also compared in the rest of their state, bit for bit:
  scalar attributes, named arrays (statistics weights) and the face records of
  characteristic boundaries. The [run block](#run-block) and snapshot blocks
  are not compared: they record when and where each run ran.
- **Tolerance and exit status.** A field passes if its max |B - A| / max |A|
  is at most `--tol` (default 0: any difference in value fails), and so does t,
  relative to the larger |t|; `--tol 1e-12,P=1e-6` sets one field apart. Steps
  that differ, or fields only one file holds, are warnings. As in
  [`mallard check`](cli.md#mallard-check), the status column, each section's
  heading and the first and last lines read `ok` (identical), `WARN` (within
  tolerance) or `ERROR` (beyond it); the exit status is 0, 0 and 1, and 3 when the files cannot be
  compared, so `mallard diff a b --tol 1e-10 || exit 1` gates a CI job.
  `--format json` prints the same as one object, its statuses `ok`, `warning`
  and `error`.
- **Cells are matched by their global index**, which restart files and VTKHDF
  or XDMF snapshots of the whole mesh keep whatever the ranks that wrote them;
  a restart file against a snapshot compares the fields both hold (RHO). VTU
  pieces and region snapshots are in their own order: `mallard diff` refuses
  to match them against those files, and compares two of the same layout value
  by value.
- **Different meshes** are refused, unless `--remap`: B's cell values are
  remapped onto A's mesh by the conservative remap of
  [`[initialize] type = "remap"`](initialization.md) (degree 2, limited) and
  compared there. The meshes, and the cell coordinates of the largest
  differences, come from the input each file's run block records (a mesh file
  relative to the run's directory, checked against its SHA-256), or from
  `--mesh-a` and `--mesh-b`.

```
$ mallard diff coarse/solut/restart_000020.restart fine/solut/restart_000020.restart --remap --tol 1e-2
...
== Remap ============================================================================== WARN
  Conservative, degree 2, limited. Remapped results differ from A by the remap's own error
  too, so a remap is never identical. A cells outside B's mesh take the nearest B cell's
  value.
  B cells  A cells  integrals kept to  A cells partly covered  A cells outside B
  3,072    768      1.2e-15            0                       0
...
== Relative differences =============================================================== WARN
  A norm of B - A over the same norm of A; a field passes if its Linf is at most its
  tolerance.
  status  field   units          L1           L2           Linf         tolerance on Linf
  WARN    RHO     dimensionless  0.000213803  0.000262748  0.000440052  0.01
...
```

For your own scripts and notebooks, the [Python package](python.md) reads every
file into numpy arrays through Mallard's own reader and runs these commands as
functions (`mallard.open`, `mallard.post.extract`, `mallard.diff`). The scripts
in `tools/` (numpy, matplotlib; see
[Tools](https://matthewbonanni.github.io/mallard/docs/tools/)) plot and animate output:

| Script | Use |
|---|---|
| `mallard/` | The [Python package](python.md); its modules `vtu`, `vtkhdf`, `h5` and `tables` are the h5py and numpy readers these scripts use |
| `post/plot_vtu.py`, `post/compare_png.py` | A field of one or several snapshots |
| `post/animate.py` | MP4/GIF animation of a 2D series with numerical schlieren |
| `post/plane_average.py` | Averages over homogeneous directions |
| `post/insitu_statistics.py` | Plots `[statistics.spectra]` output; recomputes it, or phase-bins a probe series, to check the in-situ values |
| `plot_<case>.py` | Each validation case against its reference |

ParaView opens `.vtkhdf.series` and `.vtkhdf` (VTKHDF), `.pvd` (VTU) and `.xmf` (HDF5) series directly, and `mallard.open` any of them.

## See also

- [Input reference: `[[write_data]]`](../input.md#write_data), [`[output]`](../input.md#output)
- [Probes and monitors](probes.md): probes, wall forces, domain integrals
- [Grid convergence](convergence.md): discretization error from runs at several resolutions
- [Python](python.md): every output file as numpy arrays, and the commands as functions
- [Statistics](statistics.md): time averages, spectra and correlations, phase averages
- [Initialization and restarts](initialization.md): restart files and their format
- [Running](running.md): run control of a running case
