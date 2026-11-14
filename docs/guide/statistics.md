# Statistics

Mallard accumulates statistics during the run, on the device, from every
`interval`-th step: per-cell time averages and covariances, spectra and
two-point correlations along homogeneous directions, phase averages over a
periodic cycle, and PDFs and conditional statistics (for flames: conditional
means on the mixture fraction or the progress variable). Per-cell averages are
fields of the [snapshots](output.md#solution-snapshots-write_data), spectra and
conditional statistics CSV files; restart files carry the running averages, so a
continued run keeps averaging.

## Time averages: `[statistics]`

```toml
[statistics]
fields = ["RHO", "U_X", "U_Y", "U_Z", "P", "T"]
products = ["U_X*U_X", "U_Y*U_Y", "U_Z*U_Z", "U_X*U_Y"]
t_start = 40.0
interval = 10
```

*Used in:*
[`cylinder`](https://matthewbonanni.github.io/mallard/docs/examples/#cylinder),
[`channel_retau180`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-retau180),
[`channel_inflow_retau180`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-inflow-retau180),
[`turbulent_inflow_duct`](https://matthewbonanni.github.io/mallard/docs/examples/#turbulent-inflow-duct),
[`forced_isotropic_turbulence`](https://matthewbonanni.github.io/mallard/docs/examples/#forced-isotropic-turbulence)
and 4 more.

`[statistics]` keeps running time averages and covariances per cell, on the
device, without communication: bitwise the same on any number of ranks, and
carried across restarts.

| Key | Default | Meaning |
|---|---|---|
| `fields` | none | `RHO`, `U_X`, `U_Y`, `U_Z`, `P`, `T`, `H`, `RHOU_X`, `RHOU_Y`, `RHOU_Z`, `RHOE`, `Y_<s>`, `RHOY_<s>`, with `[chemistry]` the heat release rate `HRR` and the production rates `OMEGA_<s>`, and the [derived fields](output.md#derived-fields) (`MACH`, `P0`, `ENSTROPHY`, `MIXTURE_FRACTION`, ...: the derived and mixture groups of `mallard list fields`) |
| `products` | none | Pairs `"A*B"` whose covariance to keep |
| `t_start` | 0 | Start averaging after this time; or let a [trigger](running.md#triggers) start them (`do = "start statistics"`) once a condition holds |
| `interval` | 1 | Sample every this many steps (weighted by the time since the last sample) |
| `reset` | `false` | Start afresh on a restart |

Write them with `fields = ["MEAN_*", "COV_*"]`, sample them with
[probes](probes.md#time-averages) (`fields = ["MEAN_P"]`), and average further over
homogeneous directions with [`mallard post average --statistics`](#averages-of-saved-snapshots-mallard-post-average)
(e.g. over x and z of a channel, giving profiles in y; its CSV also feeds
[synthetic turbulence](boundary_conditions.md#synthetic-turbulence-turbulence)).
Each snapshot records what each of these fields averages, over which window
([Time-averaged fields](output.md#time-averaged-fields)).

A derived field such as `MACH` is computed at each sample from the cell's
state and its gradients, exactly as `[[write_data]]` writes it, so its mean is
the mean of the field, not the field of the mean state; a field that needs
input (the `reference` of `CP`, the streams of `MIXTURE_FRACTION`) takes it
from `[output]` as when written. Sampling one costs a gradient evaluation.

### Confidence intervals

```toml
[statistics]
fields = ["U_X", "P"]
products = ["U_X*U_X"]
batch = 2.0          # >= 10 integrated autocorrelation times
```

With `batch`, every cell also gets the half width of a confidence interval
on each of its averages, `CI_MEAN_<A>` and `CI_COV_<A>_<B>` (write them with
`variables = ["CI_*"]`): the samples fall in consecutive batches of that
duration, each batch is averaged on its own, and the spread of the batch
averages gives the interval (batch means, Student's t over the closed
batches; NaN before two have closed). This is what
[`[statistics.convergence]`](#convergence-statisticsconvergence) does for
monitored values, cell by cell.

- The batches must be long compared with the flow's correlation time, or
  the intervals come out too narrow: at least 10 times the integrated
  autocorrelation time `tau` that `[statistics.convergence]` reports for a
  representative signal (in samples of that signal: multiply by its
  sampling period). Fewer, longer batches are safe; 10 to 30 closed batches
  give useful intervals.
- Costs four times the device memory of the averages (three more arrays
  per average) and a little more work per sample; no communication.
- `confidence` (default 0.95) sets the confidence. Restart files carry the
  batches, so a continued run continues them bitwise.

**Favre averages** are not computed directly. Favre means follow exactly from
Reynolds statistics that include `RHO`: with `fields = ["RHO", "U_X", "U_Y", ...]`
and `products = ["RHO*U_X", "RHO*U_Y", "U_X*U_Y", ...]`,

```text
Favre mean        ũ = <ρu> / <ρ> = MEAN_U_X + COV_RHO_U_X / MEAN_RHO
Favre covariance  ũ''v'' = COV_U_X_U_Y - COV_RHO_U_X * COV_RHO_U_Y / MEAN_RHO^2 + <ρ'u'v'> / MEAN_RHO
```

The triple correlation `<ρ'u'v'>` is not kept, so a Favre covariance computed
from these fields leaves it out. That is a good approximation where density
fluctuations are small (low Mach number, no heat release), but not in flames
or across strong shocks. `mallard post average --favre` computes Favre means
and covariances exactly, the triple correlation included, from saved snapshots.

## Averages of saved snapshots: `mallard post average`

```sh
mallard post average solut/restart_*.restart --t-start 40          # MEAN_, COV_, RMS_ per cell
mallard post average solut/flow.vtkhdf --favre              # and FAVRE_MEAN_, FAVRE_COV_
mallard post average solut/flow.vtkhdf --homogeneous x,z    # profiles in y: average/average.csv
mallard post average stats/channel.vtkhdf --latest --statistics --homogeneous x,z
```

For a run without `[statistics]`, or with other fields than it averaged,
`mallard post average` averages the states of saved restart files or
snapshots in time, per cell, with `[statistics]`'s own code: the same
`MEAN_<a>` and `COV_<a>_<b>`, `RMS_<a>` (the square root of a variance) and,
with `--favre`, the Favre means `FAVRE_MEAN_<a>` and covariances
`FAVRE_COV_<a>_<b>`. It reads one state at a time, so its memory does not grow
with the series. The result is one snapshot at the last step,
`average/average.vtkhdf`, with the averaging window in its
[snapshot block](output.md#snapshot-metadata).

| Option | Default | Meaning |
|---|---|---|
| `--fields` | `RHO`, `U_X`, `U_Y`, `U_Z` (3D), `P`, `T` | What `[statistics]` `fields` takes, derived fields included |
| `--products` | each field's variance, and the velocity covariances | Pairs `A*B`, or `none` |
| `--favre` | off | Favre means and covariances of the fields and products other than `RHO` |
| `--t-start` | as long before the first snapshot as the second is after it | Where the first snapshot's weight starts |
| `--homogeneous` | none | Also average over these directions (`x,z`): a CSV profile instead of a snapshot |
| `--statistics` | off | Average over `--homogeneous` the `[statistics]` fields one snapshot holds |
| `--region` | the whole mesh | Average over a region only: as `[[write_data]]` `region`; with `--homogeneous`, a filled region, cells weighted by their fraction in it; a region defined by the solution: its integrals, means, covariances and profiles (below) |
| `--reference`, `--mixture` | the input's | `[output]` `reference` and `mixture` of derived fields, as for [`post derive`](output.md#derived-fields-after-the-run) |

The full list is under `mallard post average --help` and in the
[CLI reference](cli.md). Each snapshot weighs the time since the previous
one, as a run's samples do: restart files of the steps a run sampled, with
`--t-start` at its `t_start`, give its `[statistics]` bit for bit. Snapshots
are identified by the steps and times they record, never by their names, and
must come in increasing time. Evenly spaced snapshots weigh the same by
default.

**Favre averages** weigh each sample by the time and the cell's density:
`ã = Σ w ρ a / Σ w ρ` and `<ρ a'' b''>/<ρ> = Σ w ρ (a - ã)(b - b̃) / Σ w ρ`,
exactly, with no triple correlation left out. `RHO` is added to the fields.

**Homogeneous directions.** `--homogeneous x,z` averages the per-cell
results over x and z as functions of y, weighted by cell volume (per radian
in an axisymmetric run, centers those of that volume), on any mesh: cells
whose centers agree in the kept directions (to `--rtol` of the mesh's
extent, 1e-6) form a row. A covariance becomes that over time and the
averaged directions together,

```text
<a'b'> = avg(COV_a_b + MEAN_a MEAN_b) - avg(MEAN_a) avg(MEAN_b)
```

adding the dispersive part (the spatial variation of the time means); Favre
columns are weighted by volume times `MEAN_RHO`. The CSV, `average/average.csv`,
holds the kept coordinates, `n_cells`, then each column, and is the `profile`
[synthetic turbulence](boundary_conditions.md#synthetic-turbulence-turbulence)
reads. `--statistics` does the same to the `MEAN_`, `COV_` (and
`FAVRE_`) fields a snapshot of a run's `[statistics]` holds, pairing
`COV_<a>_<b>` with `MEAN_<a>` and `MEAN_<b>` where both are there.

**Regions defined by the solution.** With a `--region` that follows the
solution ([Regions defined by the solution](geometry.md#regions-defined-by-the-solution)),
a `where` region, an isosurface, or a boolean of one with other regions
(`domain`, a cell zone, a primitive), the region is placed on each snapshot's
own state, as `mallard post extract` places it, and the averages are over the
region in space and time together: `average/average_reductions.csv` holds

- the time-weighted mean measure of the region;
- `integral_MEAN_<a>`, the time-weighted mean of the integral of each field
  over it, and `integral_MEAN_<a>_<b>` of each product of `--products`;
- `mean_MEAN_<a>`, the mean integral over the mean measure, and the
  covariances of these means, `COV_<a>_<b>` (mean integral of `a b` over the
  mean measure, less the product of the means), with `RMS_<a>`;
- with `--favre`, `FAVRE_MEAN_<a>`, the mean integral of `RHO a` over the mean
  integral of `RHO`, and `FAVRE_COV_<a>_<b>` likewise.

A snapshot where the region is empty counts with zero measure.

```sh
mallard post average solut/restart_*.restart --fields RHO,T --favre \
    --region '{type = "where", condition = "T > 1500"}'
```

With `--homogeneous`, a filled region gives a profile of the same averages per
row: each cell weighs the time average of its measure in the region, so a
row's `MEAN_<a>` is the time average of the integral of `a` over the region's
cells in that row over the time average of their measure (a conditional
average), and its `n_cells` counts the cells the region reached in some
snapshot. `--snapshot` and `--conditional` are errors with such a region: its
averages are integrals, not fields on cells.

- Snapshots of `RHO`, `U` and `P` give the run's averages to their stored
  precision; restart files and snapshots of the conservative variables, bit for bit.
- It reads restart files and VTKHDF or XDMF snapshots of the whole mesh,
  as `mallard post extract` does; VTU files hold cells in an order of their own.
- A single snapshot has no interval to weigh it by: give `--t-start`.
- Under `mpirun -n N` the cells are split over N ranks; the profile's rows
  are summed over them.

## Spectra and two-point correlations: `[statistics.spectra]`

```toml
[statistics]
t_start = 120.0
interval = 20

[statistics.spectra]
fields = ["U_X", "U_Y", "U_Z"]   # autospectra and autocorrelations
products = ["U_X*U_Y"]           # cross spectra and correlations
# directions = ["x", "z"]        # default: the mesh's periodic directions
# average = false                # per-sample files instead of time averages
```

*Used in:* the in-situ statistics tests.

On a generated `cartesian` mesh, spectra and two-point correlations along the
homogeneous (periodic, unstretched) directions, at each position of the
others: per y of a channel, or shell spectra of a periodic box. Computed by
FFTs on the device, bitwise the same on any number of ranks. Spectra in time
of a probe's signal, on any mesh, are [time spectra](probes.md#time-spectra) of probes.

| Key | Default | Meaning |
|---|---|---|
| `fields`, `products` | none | Variables (as `[statistics]`) and pairs `"A*B"` |
| `directions` | periodic directions | Homogeneous directions |
| `average` | `true` | Time averages, written to `file` with every snapshot and carried by restart files; `false`: one `<file stem>_<step>.csv` per sample |
| `file` | `spectra.csv` | Output CSV (`quantity, a, b, direction, position, <coordinates>, index, abscissa, value, units`) |
| `interval`, `t_start` | `[statistics]`'s | Sampling |

The file holds plane means, one-sided spectra (sums, not densities: they
add up to the covariance `<a'b'>`), correlations `<a'(x) b'(x + r)>` and, in
a periodic box, shell spectra. `tools/post/insitu_statistics.py` plots them
(`plot`) and recomputes them from restart snapshots (`spectra`, `compare`).

- Values are cell averages: at the grid cutoff a spectrum is lower than one
  of point values by the cell-average factor (sinc^2(k h / 2) per direction,
  0.41 at k = N/2), as `examples/isotropic_turbulence/isotropic_turbulence_stats.py --no-deconvolve`.
- A sample costs about a third of a step of the Re_tau 180 channel (11 ms
  on an A100): sample every 10-20 steps, not every step.
- Correlations are periodic: `R(L - r) = R(-r)`; use `r <= L/2`.

## Phase averages: `[statistics.phase]`

```toml
[statistics.phase]
fields = ["U_X", "U_Y", "P"]
bins = 12
probe = [3.0, 0.0]               # or: period = 30.3 (and t_reference)
variable = "U_Y"
level = 0.0
```

*Used in:*
[`cylinder`](https://matthewbonanni.github.io/mallard/docs/examples/#cylinder).

Per-cell means over the samples whose phase in a periodic cycle falls in each
of `bins` bins, output as `PHASE_<bin>_<field>` (`fields = ["PHASE_*"]`).
The phase comes from a fixed `period`, or from the upward crossings of a
level by a variable at a probe point.

| Key | Default | Meaning |
|---|---|---|
| `fields` | none | Variables to average |
| `bins` | none | Phase bins |
| `period`, `t_reference` | none, 0 | Fixed period, phase 0 at `t_reference` |
| `probe`, `variable`, `level` | none, none, 0 | Trigger point, variable and level |
| `interval`, `t_start` | `[statistics]`'s | Sampling |

- The trigger needs two crossings before it bins anything, and is sampled
  only every `interval` steps: keep `interval` well below a period / `bins`.
- A cycle longer than the previous one puts its tail in the last bin.
- `examples/cylinder` phase-averages
  its vortex street this way.

## Conditional statistics and PDFs: `[[statistics.conditional]]`

```toml
[[statistics.conditional]]
on = "MIXTURE_FRACTION"          # two variables, ["MIXTURE_FRACTION", "T"], for a joint PDF
bins = 50
fields = ["T", "HRR", "Y_OH"]    # their conditional means and variances
weight = "mass"                  # density-weighted (Favre); default "volume"
```

*Used in:*
[`flame_turbulence`](https://matthewbonanni.github.io/mallard/docs/examples/#flame-turbulence).

Use these for flame DNS and LES: the PDF of the mixture fraction (Bilger) or
of the progress variable, joint PDFs of two variables, and the mean and
variance of any field conditioned on them, such as the temperature, the heat
release rate or a radical against the mixture fraction, the quantities flamelet
and conditional moment closure models are built from and compared with. The
conditioning variables and the fields are any of the `[statistics]` fields:
`MIXTURE_FRACTION` and `PROGRESS_VARIABLE` take their streams from
[`[output.mixture]`](output.md#derived-fields) (`fuel` and `oxidizer`, or `progress`),
and `HRR` and `OMEGA_<s>` need `[chemistry]`. Each entry writes one CSV
file with the other output.

| Key | Default | Meaning |
|---|---|---|
| `on` | none | The variable of a PDF, or two variables of a joint PDF |
| `bins` | 50 | Equal bins of each range: one count, or one per variable |
| `range` | `[0.0, 1.0]` for `MIXTURE_FRACTION` and `PROGRESS_VARIABLE`, else none | `[low, high]` in the variable's units, `[[low, high], [low, high]]` for two |
| `fields` | none | Variables whose conditional mean and variance to keep |
| `weight` | `"volume"` | `"volume"`, or `"mass"` for density-weighted (Favre) statistics |
| `region` | the whole mesh | A filled [region](geometry.md#regions) (or its name): cells weighted by their fraction in it |
| `file` | `conditional_<on>.csv` | The output file (`conditional_<on>_<on>.csv` for two variables) |
| `interval`, `t_start` | `[statistics]`'s | Sampling |

Each sample adds every cell, weighted by its volume (or mass) times the time
since the previous sample, to the bin of its value of `on`. The file has a row
per bin (for two variables, the second fastest): the bin centers, `PDF` (per
unit of each variable, integrating to one over the whole sample), and for each
field `MEAN_<f>`, `COV_<f>_<f>` (the conditional variance) and `RMS_<f>`; a
comment line records the bins, the weighting and the fraction of the weight
outside the ranges.

- Samples outside the ranges count in the total only: the PDF then
  integrates to one minus that fraction, which the file records. Widen the
  range of a variable that leaves it (a temperature, a scalar dissipation rate).
- A bin no cell reached holds `nan` means; fewer bins give smoother means in
  short averages.
- Bins are of the cell values (cell averages): resolve the flame, or the PDF
  of an under-resolved front is that of its smeared profile.
- The sums are exact (fixed point), so the results are the same bit for bit
  on any number of ranks, and restart files continue them bitwise. They hold
  each entry by its position in the list: a restart file whose entry there has
  other variables, bins or fields is an error unless `statistics.reset = true`.
- On GPUs, entries are sampled without stopping the steps queued ahead of the
  host, including those reading derived fields, `HRR` or `OMEGA_<species>`.
- `mallard post average --conditional` computes the same files from saved
  snapshots of a run without them; from restart files of the run's sampling
  steps, they are the run's to the digit (below).

`[[statistics.conditional]]` may stand alone in `[statistics]`, without
`fields` and `products`.

```bash
mallard post average solut/flame.vtkhdf --conditional \
    '{on = "MIXTURE_FRACTION", fields = ["T", "HRR"], weight = "mass"}'
```

![Conditional mean temperature and heat release rate against the mixture fraction, with the PDF of the mixture fraction, in an igniting hydrogen mixing layer](../images/guide/conditional-light.svg#gh-light-mode-only)
![Conditional mean temperature and heat release rate against the mixture fraction, with the PDF of the mixture fraction, in an igniting hydrogen mixing layer](../images/guide/conditional-dark.svg#gh-dark-mode-only)

*Density-weighted conditional means of the temperature (with one RMS either
side) and of the heat release rate against the mixture fraction, and the PDF of
the mixture fraction, in a layer of hydrogen in nitrogen igniting in hot air,
averaged over 0.1 to 0.3 ms: heat is released lean of the stoichiometric
mixture fraction (0.51), where the mixture is hottest
(`tools/docs/runs.py conditional`).*

## Convergence: `[statistics.convergence]`

```toml
[statistics]
fields = ["U_X", "U_Y", "U_Z", "P"]
products = ["U_X*U_X", "U_X*U_Y"]
interval = 10

[statistics.convergence]
signals = ["wall_integral_TAU_WALL_X", "integrals_kinetic_energy"]
tolerance = 0.01                    # converged at +-1% of the means

[[triggers]]
name = "developed"
when = "convergence_stationary"     # the initial transient has ended
do = "start statistics"

[[triggers]]
name = "converged"
when = "convergence_converged"      # every interval within the tolerance
do = "stop"
```

*Used in:* [`forced_isotropic_turbulence`](https://matthewbonanni.github.io/mallard/docs/examples/#forced-isotropic-turbulence).

Answers two questions every averaging run asks: when has the initial
transient ended, and how close are the averages to their long-time values?
It follows monitored values, those [triggers](running.md#triggers) read
(`[integrals]`, `[[forces]]`, reductions of `[[probes]]` and probes of one
point, with the same names), one sample each time the monitor writes one.

- **End of the transient**: MSER-5 (White 1997) on each signal: the start
  that minimizes the standard error of the mean of what remains, over the
  first half of the series; a minimum at the end of that half means the
  signal still drifts. MSER-5 runs again each time a signal's series has
  grown by 5%. Once it has accepted a start for every signal, the run is
  *stationary* (and stays so), from the latest start, `t_stationary`. A
  longer series can show that a slow transient lasted longer: each accepted
  start replaces the previous one, and `t_stationary` moves with it.
- **Confidence intervals** on the mean and on the variance of each signal, of
  the samples after `t_stationary` (and, once the `[statistics]` averages
  run, after their start, so the intervals are those of the samples the
  averages cover): non-overlapping batch means (Schmeiser 1982) with
  Student's t, which account for the samples' correlation in time; the
  variance is the mean of `(x - mean)^2`.
- **Integrated autocorrelation time** `tau` (in samples, by Sokal's automatic
  window) and the effective sample size `n_eff = samples / tau`: how many
  independent samples the run holds.
- **Converged** (with `tolerance`): stationary, every interval within
  `tolerance` times the mean (and `variance_tolerance` times the variance),
  and every batch at least 10 `tau` long, so the batch means are nearly
  independent and the intervals trustworthy.

| Key | Default | Meaning |
|---|---|---|
| `signals` | none | Monitored values, named as in [triggers](running.md#triggers) |
| `confidence` | 0.95 | Confidence of the intervals |
| `batches` | 20 | Batches of the batch means (10 to 30 is usual) |
| `tolerance` | none | Converged when every mean's half width is within this fraction of the mean |
| `variance_tolerance` | none | And every variance's within this fraction of the variance |
| `min_samples` | 100 | Samples of a signal before MSER-5 is tried |
| `file` | `convergence.csv` | Output CSV, a row each time a signal has a new sample |

The values, a row of `file` and the monitor `convergence` of the run record,
are `stationary` (0 or 1), `t_stationary`, `converged` (with `tolerance`) and,
per signal, `<signal>_mean`, `_ci` (the half width), `_variance`,
`_variance_ci`, `_tau`, `_n_eff` and `_samples`; they are NaN before the run
is stationary. Triggers read them as `convergence_<value>`, e.g.
`convergence_stationary`, `convergence_integrals_kinetic_energy_ci`: start
the statistics or stop the run with them, as above, or with a tolerance of
your own (`convergence_forces_cylinder_Fy_pressure_ci < 1e-3` for a mean near
zero, where a relative tolerance means nothing). Restart files carry the
histories, so a continued run keeps them.

- Pick signals that settle with the statistics you want: a wall shear stress
  integral for a channel, a force for a bluff body. Integrals over the whole
  domain converge faster than any point; a point probe is the strict test.
- Samples equally spaced in time (a probe's `time_interval`) are best: a
  sample is a sample, whatever time separates it from the last.
- The statistics start when the trigger fires, after the transient has been
  detected, not at `t_stationary`: what came before was not kept.
- Costs nothing on the device: the analysis runs on the host each time a
  signal has a new sample, on every rank alike.

## See also

- [Input reference: `[statistics]`](../input.md#statistics), [`[statistics.spectra]`](../input.md#statisticsspectra), [`[statistics.phase]`](../input.md#statisticsphase), [`[[statistics.conditional]]`](../input.md#statisticsconditional)
- [Large-eddy simulation](les.md): turbulent inflow and the dissipation budget
- [Probes and monitors](probes.md): samples and reductions of the instantaneous solution
- [Units](output.md#units) of `MEAN_`, `COV_`, `RMS_`, `FAVRE_` and `PHASE_` fields and of spectra
- [`mallard post average`](cli.md): its options in full
