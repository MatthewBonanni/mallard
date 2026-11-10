# Statistics

Mallard accumulates statistics during the run, on the device, from every
`interval`-th step: per-cell time averages and covariances, spectra and
two-point correlations along homogeneous directions, and phase averages over a
periodic cycle. Per-cell averages are fields of the
[snapshots](output.md#solution-snapshots-write_data), spectra a CSV file; restart
files carry the running averages, so a continued run keeps averaging.

## Time averages: `[statistics]`

```toml
[statistics]
fields = ["RHO", "U_X", "U_Y", "U_Z", "P", "T"]
products = ["U_X*U_X", "U_Y*U_Y", "U_Z*U_Z", "U_X*U_Y"]
t_start = 40.0
interval = 10
```

*Used in:* [`cylinder`](https://matthewbonanni.github.io/mallard/docs/examples/#cylinder), [`channel_retau180`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-retau180), [`channel_inflow_retau180`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-inflow-retau180), [`turbulent_inflow_duct`](https://matthewbonanni.github.io/mallard/docs/examples/#turbulent-inflow-duct), [`forced_isotropic_turbulence`](https://matthewbonanni.github.io/mallard/docs/examples/#forced-isotropic-turbulence) and 3 more.

`[statistics]` keeps running time averages and covariances per cell, on the
device, without communication: bitwise the same on any number of ranks, and
carried across restarts.

| Key | Default | Meaning |
|---|---|---|
| `fields` | none | `RHO`, `U_X`, `U_Y`, `U_Z`, `P`, `T`, `H`, `RHOU_X`, `RHOU_Y`, `RHOU_Z`, `RHOE`, `Y_<s>`, `RHOY_<s>` |
| `products` | none | Pairs `"A*B"` whose covariance to keep |
| `t_start` | 0 | Start averaging after this time; or let a [trigger](running.md#triggers) start them (`do = "start statistics"`) once a condition holds |
| `interval` | 1 | Sample every this many steps (weighted by the time since the last sample) |
| `reset` | `false` | Start afresh on a restart |

Write them with `variables = ["MEAN_*", "COV_*"]`, sample them with
[probes](probes.md#time-averages) (`variables = ["MEAN_P"]`), and average further over
homogeneous directions with `tools/post/plane_average.py` (e.g. over x and z of a
channel, giving profiles in y; its CSV also feeds
[synthetic inflow](boundary_conditions.md#synthetic-turbulence-turbulence)).

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
or across strong shocks.

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

*Used in:* [`cylinder`](https://matthewbonanni.github.io/mallard/docs/examples/#cylinder).

Per-cell means over the samples whose phase in a periodic cycle falls in each
of `bins` bins, output as `PHASE_<bin>_<field>` (`variables = ["PHASE_*"]`).
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

- [Input reference: `[statistics]`](../input.md#statistics), [`[statistics.spectra]`](../input.md#statisticsspectra), [`[statistics.phase]`](../input.md#statisticsphase)
- [Large-eddy simulation](les.md): turbulent inflow and the dissipation budget
- [Probes and monitors](probes.md): samples and reductions of the instantaneous solution
- [Units](output.md#units) of `MEAN_`, `COV_` and `PHASE_` fields and of spectra
