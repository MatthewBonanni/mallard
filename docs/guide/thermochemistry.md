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

*Used in:* [`volvo_bluff_body`](https://matthewbonanni.github.io/mallard/docs/examples/#volvo-bluff-body), [`flame_turbulence`](https://matthewbonanni.github.io/mallard/docs/examples/#flame-turbulence), [`h2_ignition`](https://matthewbonanni.github.io/mallard/docs/examples/#h2-ignition), [`reactive_shock_tube`](https://matthewbonanni.github.io/mallard/docs/examples/#reactive-shock-tube), [`detonation_1d`](https://matthewbonanni.github.io/mallard/docs/examples/#detonation-1d) and 11 more.

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

**Compositions.** Everywhere a state is given (`[initialize]`, `upt`, `stagnation_inlet` and
`nscbc_inlet` boundaries, sponges, `MallardReactor`), give exactly one of `X`
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

*Used in:* [`volvo_bluff_body`](https://matthewbonanni.github.io/mallard/docs/examples/#volvo-bluff-body), [`flame_turbulence`](https://matthewbonanni.github.io/mallard/docs/examples/#flame-turbulence), [`h2_ignition`](https://matthewbonanni.github.io/mallard/docs/examples/#h2-ignition), [`reactive_shock_tube`](https://matthewbonanni.github.io/mallard/docs/examples/#reactive-shock-tube), [`detonation_1d`](https://matthewbonanni.github.io/mallard/docs/examples/#detonation-1d) and 10 more.

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
  ([validation](https://matthewbonanni.github.io/mallard/validation/#simpler)).
  Not with `fuse_half_steps`, `double_flux` or the `hybrid` flux.

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | `false` keeps the mixture frozen |
| `coupling` | `"strang"` | `strang` or `simpler` |
| `rtol`, `atol` | 1e-6, 1e-10 | Integrator tolerances (relative on Y and T, absolute on Y) |
| `max_steps` | 100000 | Sub-steps allowed per cell and call; more stop the run |
| `T_frozen` | 0 | No chemistry in cells below this temperature |
| `fuse_half_steps` | `false` | Strang: fuse the closing half step with the next opening one, except where output reads the state (faster; results then depend on output times) |
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

*Used in:* [`flame_vortex_quenching`](https://matthewbonanni.github.io/mallard/docs/examples/#flame-vortex-quenching).

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
`T_ambient = 0` ([validation](https://matthewbonanni.github.io/mallard/validation/#radiation)).

## 0D reactors: `MallardReactor`

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

*Used in:* [`h2_ignition`](https://matthewbonanni.github.io/mallard/docs/examples/#h2-ignition), [`autoignition_2d`](https://matthewbonanni.github.io/mallard/docs/examples/#autoignition-2d).

`MallardReactor` integrates one adiabatic, constant-volume reactor with the
solver's own chemistry, and writes `t`, `T`, `p` and the mass fractions as
CSV. Use it to check a mechanism and tolerances before a run, or to get an
ignition delay.

```sh
cd examples/h2_ignition
../../build/src/MallardReactor -i input.toml    # writes reactor.csv, prints the ignition delay
```

![Temperature and mass fractions of the constant-volume ignition of stoichiometric H2/air at 1200 K](../images/guide/ignition-light.svg#gh-light-mode-only)
![Temperature and mass fractions of the constant-volume ignition of stoichiometric H2/air at 1200 K](../images/guide/ignition-dark.svg#gh-dark-mode-only)

*`examples/h2_ignition` with `MallardReactor`: the ignition delay (maximum of
dT/dt) is 44.2 µs, as Cantera's
([validation](https://matthewbonanni.github.io/mallard/validation/#ignition)).*

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

## Examples

| Example | Shows |
|---|---|
| `h2_ignition` | `MallardReactor` |
| `reactive_shock_tube` | Ignition behind a reflected shock, Strang splitting |
| `detonation_1d`, `detonation_2d`, `detonation_3d` | CJ and cellular detonations, numerical soot foils (`P_MAX`) |
| `premixed_flame` | Freely propagating H2/air and CH4/air (GRI-3.0) flames against Cantera |
| `flame_2d`, `triple_flame`, `counterflow_diffusion` | Multidimensional laminar flames, differential diffusion |
| `flame_vortex`, `flame_vortex_quenching` | Flame-vortex interaction, with radiation |
| `autoignition_2d` | Stratified autoignition DNS |
| `shock_bubble_3d` | Non-reacting mixture, `double_flux` |

![The flame_2d example: a periodic channel with an inflow on the left and a pressure outlet on the right](../images/guide/domain_flame_2d-light.svg#gh-light-mode-only)
![The flame_2d example: a periodic channel with an inflow on the left and a pressure outlet on the right](../images/guide/domain_flame_2d-dark.svg#gh-dark-mode-only)

*`examples/flame_2d`: a lean H2/air flame enters through `upt` (with its
composition) and leaves through `p_out`, periodic across the channel.*

## See also

- [Input reference: `[chemistry]`](../input.md#chemistry), [`[radiation]`](../input.md#radiation), [`MallardReactor`](../input.md#mallardreactor)
- [Validation: reacting flow](https://matthewbonanni.github.io/mallard/validation/#reacting-flow)
- [Large-eddy simulation](les.md): thickened flames and PaSR
- Design notes: [finite-rate chemistry](../design/chemistry.md)
