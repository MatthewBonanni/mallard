# Thermochemistry

Gas mixtures carry one mass fraction per species of a mechanism, with
thermally perfect thermodynamics, mixture-averaged (or simpler) transport,
finite-rate chemistry and optically thin radiation. All of it is in double
precision in every build, on CPUs and GPUs.

## Gas mixtures

```toml
[physics]
type = "navier_stokes"          # or "euler": no transport
gas = "mixture"
mechanism = "../../mechanisms/h2o2.yaml"
# phase = "ohmech"              # default: the file's first phase
transport = "mixture_averaged"  # navier_stokes only

[initialize]
type = "constant"
u = [0.0, 0.0]
p = 101325.0
T = 300.0
X = { H2 = 2.0, O2 = 1.0, N2 = 3.76 }   # mole fractions, normalized
```

*Used in:*
[`volvo_bluff_body`](https://matthewbonanni.github.io/mallard/docs/examples/#volvo-bluff-body),
[`flame_turbulence`](https://matthewbonanni.github.io/mallard/docs/examples/#flame-turbulence),
[`h2_ignition`](https://matthewbonanni.github.io/mallard/docs/examples/#h2-ignition),
[`reactive_shock_tube`](https://matthewbonanni.github.io/mallard/docs/examples/#reactive-shock-tube),
[`detonation_1d`](https://matthewbonanni.github.io/mallard/docs/examples/#detonation-1d)
and 12 more.

| Key | Default | Meaning |
|---|---|---|
| `gas` | `"perfect"` | `mixture` for a thermally perfect mixture; then no `gamma`, `p_ref`, `T_ref`, `rho_ref` |
| `mechanism` | required | A [Cantera YAML](https://cantera.org/stable/yaml/index.html) file |
| `phase` | first phase | Phase of the file to use |
| `transport` | `"mixture_averaged"` | `mixture_averaged`, `unity_lewis` or `constant_lewis` (`navier_stokes` only) |
| `lewis` | 1 for all | `constant_lewis`: Lewis numbers by species, e.g. `{ H2 = 0.3, H = 0.18 }` |
| `soret` | `false` | `mixture_averaged`: thermal diffusion (the Soret effect) |

**Mechanisms.** `mechanisms/` holds H2/O2 (`h2o2.yaml`, 10 species), GRI-Mech
3.0 (`gri30.yaml`, 53), two-step propane and methane mechanisms for LES,
helium-air for shock-bubble runs, air to 20,000 K (NASA-9, thermo only) and a
100-species n-dodecane mechanism; its README lists them. Chemkin files convert
with Cantera's `ck2yaml`. Species thermodynamics may be NASA-7, NASA-9 or
constant-cp; reactions may be elementary, three-body, falloff (Lindemann,
Troe, SRI), pressure-dependent Arrhenius (PLOG) and Chebyshev, with
non-integer reactant orders.

**Compositions.** Everywhere a state is given (`[initialize]`, `inlet_state` (also with `nscbc`), `inlet_stagnation` and
`inlet_mass_flow` boundaries, sponges, `mallard reactor`), give exactly one of `X`
(mole fractions) or `Y` (mass fractions), as a table by species; listed values
are normalized and unlisted species are zero. Analytical initial conditions
and inflows take expressions per species, and `balance = "N2"` gives one
species the rest (`1 - sum`) instead of normalizing:

```toml
[initialize]
type = "analytical"
Y = { H2 = "x < 0.5 ? 0.028 : 0.0", O2 = "x < 0.5 ? 0.226 : 0.233" }
balance = "N2"
```

**Transport.** With `navier_stokes`, species diffuse along mole-fraction
gradients with a correction velocity (the diffusive mass fluxes sum to zero)
and carry their enthalpy; viscosity and conductivity follow the mixture rules
(Wilke, Mathur), as Cantera's mixture-averaged model.

| `transport` | Diffusion coefficients | When |
|---|---|---|
| `mixture_averaged` | Mixture-averaged, from the species' transport data | Default; differential diffusion (lean hydrogen flames) |
| `unity_lewis` | `lambda / (rho cp)` for every species | Comparisons with unity-Lewis theory or DNS |
| `constant_lewis` | `lambda / (rho cp Le_k)` | Simplified chemistry with fitted Lewis numbers |

Walls are no-slip and non-catalytic: no species crosses them, so their
species, Soret and enthalpy diffusion fluxes vanish. `wall_adiabatic`,
`wall_isothermal` and `wall_heat_flux` set the heat flux (see
[boundary conditions](boundary_conditions.md)).

### Contacts between different gases

```toml
[numerics]
double_flux = true
```

*Used in:* the multicomponent shock-tube tests.

The conservative scheme perturbs pressure at the percent level at contacts
between gases of different `cp / cv` (cold hydrogen in hot air) on coarse
meshes. `[numerics] double_flux = true` removes those oscillations: pressure
and velocity stay exactly uniform across such contacts, at the price of exact
energy conservation (about 0.2% over a multicomponent shock tube). Use it for
interfaces of very different gases (the `shock_bubble_3d` example), not for
flames.

## Finite-rate chemistry

```toml
[chemistry]
rtol = 1.0e-6
atol = 1.0e-10
# coupling = "simpler"   # balanced splitting, for large steps
# T_frozen = 400.0       # no chemistry below 400 K
```

*Used in:*
[`volvo_bluff_body`](https://matthewbonanni.github.io/mallard/docs/examples/#volvo-bluff-body),
[`flame_turbulence`](https://matthewbonanni.github.io/mallard/docs/examples/#flame-turbulence),
[`h2_ignition`](https://matthewbonanni.github.io/mallard/docs/examples/#h2-ignition),
[`reactive_shock_tube`](https://matthewbonanni.github.io/mallard/docs/examples/#reactive-shock-tube),
[`detonation_1d`](https://matthewbonanni.github.io/mallard/docs/examples/#detonation-1d)
and 11 more.

A mixture whose mechanism has reactions reacts when the input has a
`[chemistry]` table.

Each owned cell is advanced as an adiabatic, constant-volume reactor by RODAS,
an adaptive Rosenbrock method with the analytical Jacobian; cells whose
composition would change by less than `atol / 100` over the step are skipped.
Chemistry and flow are split:

![Time line of one step with Strang and with SIMPLER splitting](../images/guide/splitting-light.svg#gh-light-mode-only)
![Time line of one step with Strang and with SIMPLER splitting](../images/guide/splitting-dark.svg#gh-dark-mode-only)

- **Strang** (default): half a step of chemistry, a step of flow, another half
  step of chemistry. Its splitting error binds before the CFL condition in
  detonations: the 1D CJ detonation keeps its induction length within 5% up to
  `cfl = 0.35`.
- **SIMPLER** (`coupling = "simpler"`, Wu, Ma & Ihme 2019): the chemistry
  integrates over the whole step with the flow's right-hand side as a constant
  source, so steady flames stay exactly steady at any step; the detonation's
  induction length stays within 5% at `cfl = 1.0`
  ([verification](https://matthewbonanni.github.io/mallard/validation/#simpler)).
  Not with `fuse_half_steps`, `double_flux` or the `hybrid` flux.

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | `false` keeps the mixture frozen |
| `coupling` | `"strang"` | `strang` or `simpler` |
| `rtol`, `atol` | 1e-6, 1e-10 | Integrator tolerances (relative on Y and T, absolute on Y) |
| `max_steps` | 100000 | Sub-steps allowed per cell and call; more stop the run |
| `T_frozen` | 0 | No chemistry in cells below this temperature |
| `fuse_half_steps` | `true` | Strang: one chemistry call per step, the closing half step fused with the next opening one; output still shows the state at the step's end, and results do not depend on output or restarts. `false` takes the two half steps separately |
| `C_reg` | 1e-12 | Concentration [kmol/m³] below which orders between 0 and 1 are regularized; 1e-10 speeds up trace-fuel burnout at a small cost in accuracy |
| `load_balance` | `false` | MPI: move chemistry work from loaded to idle ranks, bitwise the same results ([performance](performance.md#chemistry-load-balancing)) |
| `sparse` | automatic | Sparse LU (`true`) or dense (`false`); automatic picks sparse from 30 species when it pays (GRI-3.0 and larger) |
| `lanes` | 0 (automatic) | GPU lanes per cell; automatic: a warp per cell from 16 species |
| `integrator` | `"rosenbrock"` | The only choice so far |

Output variables: `HRR` (heat release rate, W/m³), `OMEGA_<species>`
(production rates), `CHEM_COST` (sub-steps of each cell in the last step).
Restart files carry each cell's last sub-step, so restarts repeat an
uninterrupted run exactly. The progress rows show the share of cells advanced
and the most sub-steps of any cell.

Pitfalls:

- **Time step.** With Strang splitting, detonations need `cfl` of about 0.35;
  flames are insensitive (speed unchanged to 0.001% between 0.2 and 1.0).
- **Ignition from a cold start** needs no special treatment, but `T_frozen`
  saves the work of cold fresh gas in large runs.
- **Large mechanisms on GPUs** are memory-bound: see
  [Performance](performance.md#chemistry) for cells per second up to 1268
  species.

## Radiation

```toml
[radiation]
model = "optically_thin"
T_ambient = 300.0
# species = ["H2O", "CO2"]
```

*Used in:*
[`flame_vortex_quenching`](https://matthewbonanni.github.io/mallard/docs/examples/#flame-vortex-quenching).

Optically thin radiative heat loss (the TNF workshop model):
`q = 4 sigma sum_i p_i a_i(T) (T^4 - T_ambient^4)`, with Planck-mean
absorption coefficients of H2O, CO2, CO and CH4. It holds for small, lean or
diluted flames, which emit but barely absorb. It needs no `[chemistry]` and
outputs `QRAD` [W/m³].

| Key | Default | Meaning |
|---|---|---|
| `model` | `"optically_thin"` | The only model |
| `T_ambient` | 300 | Background temperature [K] |
| `species` | those of H2O, CO2, CO, CH4 in the mechanism | Radiating species |

Cantera's `radiation_enabled` flames are `species = ["H2O", "CO2"]` with
`T_ambient = 0` ([benchmark](https://matthewbonanni.github.io/mallard/validation/#radiation)).

## 0D reactors: `mallard reactor`

```toml
# examples/h2_ignition/input.toml
[physics]
gas = "mixture"
mechanism = "../../mechanisms/h2o2.yaml"

[reactor]
T = 1200.0
p = 101325.0
X = { H2 = 2.0, O2 = 1.0, N2 = 3.76 }
end_time = 2.0e-4
output_interval = 1.0e-6

[chemistry]
rtol = 1.0e-6
atol = 1.0e-10
```

*Used in:*
[`h2_ignition`](https://matthewbonanni.github.io/mallard/docs/examples/#h2-ignition),
[`autoignition_2d`](https://matthewbonanni.github.io/mallard/docs/examples/#autoignition-2d).

`mallard reactor` integrates one adiabatic, constant-volume reactor with the
solver's own chemistry, and writes `t`, `T`, `p` and the mass fractions as
CSV. Use it to check a mechanism and tolerances before a run, or to get an
ignition delay.

```sh
cd examples/h2_ignition
../../build/src/mallard reactor -i input.toml    # writes reactor.csv, prints the ignition delay
```

![Temperature and mass fractions of the constant-volume ignition of stoichiometric H2/air at 1200 K](../images/guide/ignition-light.svg#gh-light-mode-only)
![Temperature and mass fractions of the constant-volume ignition of stoichiometric H2/air at 1200 K](../images/guide/ignition-dark.svg#gh-dark-mode-only)

*`examples/h2_ignition` with `mallard reactor`: the ignition delay (maximum of
dT/dt) is 44.2 µs, as Cantera's
([benchmark](https://matthewbonanni.github.io/mallard/validation/#ignition)).*

| `[reactor]` key | Default | Meaning |
|---|---|---|
| `type` | `"constant_volume"` | The only type |
| `T`, `p` | required | Initial temperature [K] and pressure [Pa] |
| `X` or `Y` | required | Initial composition |
| `end_time` | required | Integration time [s] |
| `output_interval` | `end_time / 100` | Time between CSV rows |
| `output` | `"reactor.csv"` | CSV file |

A `[benchmark]` table turns the run into a chemistry benchmark: states sampled
along the reactor's trajectory, replicated over many cells and advanced by the
solver's chemistry kernels, which prints cells per second (see
`benchmarks/README.md`; keys `cells`, `samples`, `dt`, `repeats`,
`igniting_fraction`, `lanes`, `threads`, `shared`, `bin_by_cost`, `output`).

## Flame speed and thickness

`mallard post flame` measures a premixed flame, planar on average, in a run's
snapshots or restart files:

```sh
mallard post flame solut/flame.vtkhdf --reference-speed 2.33237
```

For each snapshot it averages the state over the directions normal to
`--direction` (default `x`; cells weighted by volume) into a profile, with the
fresh gas at its colder end, and reports:

| Column | Meaning |
|---|---|
| `x_f` | Flame position: where T is midway between the fresh temperature T_u (at the colder end) and the burnt temperature T_b: the peak temperature, or `--burnt-temperature` |
| `delta_T` | Thermal thickness (T_b − T_u) / max \|dT/dx\| |
| `dTdx_max` | max \|dT/dx\|, between neighbouring rows |
| `S_c` | Consumption speed −∫ ω_k dx / (ρ_u (Y_k,u − Y_k,b)) of the deficient reactant k: in a steady flame the flame speed, in any frame |
| `S_d` | Displacement speed: the speed of `x_f` relative to the fresh gas |

The production rates ω_k are the run's own: the `OMEGA_*` fields of snapshots
that hold them with `T`, `RHO`, the velocity and `Y_*` (read on the mesh
alone), else computed from each stored state by the case's `[chemistry]`
(restart files, snapshots of the state). Unless `--species` names it, k
follows from the equivalence ratio φ = (2 C + H/2) / O of the fresh gas's
atoms: the fuel when φ ≤ 1.01 (H2 of a lean or stoichiometric H2/air flame),
else the oxidizer (O2 of a rich one); of several, the one the flame depletes
most relative to its fresh mass fraction. The summary averages `S_c`,
`delta_T` and `dTdx_max` over the last third of the series and fits `x_f` over
its last half for `S_d`; choose the
snapshots of a settled flame with `--times`. The table, with a row per
snapshot, goes to `flame/flame.csv` (`-o`, `--name`); `--format json` gives the
report to scripts.

`delta_T` depends on T_b. The burnt gas of a short domain is often still below
its adiabatic equilibrium temperature, so with the peak as T_b the thickness
falls short of a reference flame's while the slope agrees: in the premixed
flame tutorial the burnt gas reaches 2231 K against Cantera's 2379 K, and
`delta_T` is 6.8% below Cantera's thickness while max |dT/dx| is within 0.4%:
the gap is the burnt gas's, not the flame's. Compare `dTdx_max`, or give the
reference's T_b with `--burnt-temperature` (`2379.43` there brings `delta_T`
within 0.4% of Cantera's).

On a mesh whose cells do not line up in rows normal to `--direction`, each
cell is a row of its own and the profile is not a flame's: the command is for
strips and channels (the
[premixed flame tutorial](../tutorials/premixed_flame.md), `flame_2d`). Every
option: [`mallard post flame --help`](cli.md#mallard-post-flame).

## Examples

| Example | Shows |
|---|---|
| `h2_ignition` | `mallard reactor` |
| `reactive_shock_tube` | Ignition behind a reflected shock, Strang splitting |
| `detonation_1d`, `detonation_2d`, `detonation_3d` | CJ and cellular detonations, numerical soot foils (`P_MAX`) |
| `premixed_flame` | Freely propagating H2/air and CH4/air (GRI-3.0) flames against Cantera |
| `flame_2d`, `triple_flame`, `counterflow_diffusion`, `reacting_mixing_layer` | Multidimensional laminar flames, differential diffusion, autoignition in a mixing layer |
| `flame_vortex`, `flame_vortex_quenching` | Flame-vortex interaction, with radiation |
| `autoignition_2d` | Stratified autoignition DNS |
| `shock_bubble_3d` | Non-reacting mixture, `double_flux` |

![The flame_2d example: a periodic channel with an inflow on the left and a pressure outlet on the right](../images/guide/domain_flame_2d-light.svg#gh-light-mode-only)
![The flame_2d example: a periodic channel with an inflow on the left and a pressure outlet on the right](../images/guide/domain_flame_2d-dark.svg#gh-dark-mode-only)

*`examples/flame_2d`: a lean H2/air flame enters through `inlet_state` (with its
composition) and leaves through `outlet_pressure`, periodic across the channel.*

## See also

- [Input reference: `[chemistry]`](../input.md#chemistry), [`[radiation]`](../input.md#radiation), [`mallard reactor`](../input.md#mallard-reactor)
- [Benchmarks: reacting flow](https://matthewbonanni.github.io/mallard/validation/#reacting-flow)
- [Large-eddy simulation](les.md): thickened flames and PaSR
- Design notes: [finite-rate chemistry](../design/chemistry.md)
