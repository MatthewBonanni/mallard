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

*Used in:* [`cylinder`](https://matthewbonanni.github.io/mallard/docs/examples/#cylinder), [`channel_retau180`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-retau180), [`channel_inflow_retau180`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-inflow-retau180), [`turbulent_inflow_duct`](https://matthewbonanni.github.io/mallard/docs/examples/#turbulent-inflow-duct), [`channel_les`](https://matthewbonanni.github.io/mallard/docs/examples/#channel-les) and 2 more.

`[statistics]` keeps running time averages and covariances per cell, on the
device, without communication: bitwise the same on any number of ranks, and
carried across restarts.

| Key | Default | Meaning |
|---|---|---|
| `fields` | none | `RHO`, `U_X`, `U_Y`, `U_Z`, `P`, `T`, `H`, `Y_<s>` |
| `products` | none | Pairs `"A*B"` whose covariance to keep |
| `t_start` | 0 | Start averaging after this time |
| `interval` | 1 | Sample every this many steps (weighted by the time since the last sample) |
| `reset` | `false` | Start afresh on a restart |

Write them with `variables = ["MEAN_*", "COV_*"]`, and average further over
homogeneous directions with `tools/post/plane_average.py` (e.g. over x and z of a
channel, giving profiles in y; its CSV also feeds
[synthetic inflow](boundary_conditions.md#synthetic-turbulence)).

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
FFTs on the device, bitwise the same on any number of ranks.

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

## See also

- [Input reference: `[statistics]`](../input.md#statistics), [`[statistics.spectra]`](../input.md#statisticsspectra), [`[statistics.phase]`](../input.md#statisticsphase)
- [Large-eddy simulation](les.md): turbulent inflow and the dissipation budget
- [Probes and monitors](probes.md): samples and reductions of the instantaneous solution
- [Units](output.md#units) of `MEAN_`, `COV_` and `PHASE_` fields and of spectra
