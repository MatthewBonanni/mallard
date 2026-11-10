# Statistical convergence

`[statistics.convergence]` (#337): the end of the initial transient of
monitored values and confidence intervals on their means and variances. User
docs: [Statistics](../guide/statistics.md#convergence-statisticsconvergence).

## Decisions

1. **Signals are monitored values already reduced.** The same values
   `[[triggers]]` read (integrals, forces, probe reductions and one-point
   probes), under the same names, sampled each time their monitor writes
   them. Nothing new runs on the device or on the step path: the analysis is
   output-interval work. Region reductions of probes give what per-cell
   fields cannot without device work, e.g. a plane mean.
2. **Every rank computes it.** Integrals and forces are summed over the ranks
   already; probe values live on rank 0, so at the steps where a probe signal
   has a new sample rank 0 sends those values to the others (one
   `allgatherv`, at the probe's interval). Every rank then runs the same
   arithmetic on the same doubles, so the state agrees bitwise without a
   per-step broadcast, and the values reach triggers through their existing
   variable list (`convergence_<column>`).
3. **Starting and stopping are trigger actions.** `stationary` and
   `converged` are values; `do = "start statistics"` and `do = "stop"` act on
   them as on any condition. No second start or stop path.
4. **MSER-5 for the transient** (White 1997; Hoad, Robinson and Davies 2010):
   simple, without parameters to tune, and among the best performers in Hoad
   et al.'s comparison of 40-odd methods. The minimum is searched over the
   first half of the batches, and one at the end of the search is rejected:
   a series still drifting puts its minimum there. The check runs each time a
   signal's history grows by 5% (at least one batch), so the cost over a run
   stays linear in its samples. MSER-5 trades the bias of what it keeps
   against the variance of a shorter series, so on a short series it
   accepts an early start: on a decay over 400 samples from 9 standard
   deviations (`tau` = 9), the first 100 samples. `t_stationary` therefore
   follows every accepted check rather than the first: as the series grows,
   the start moves to where the decay no longer biases the mean, and settles.
   `stationary` stays 1 once set, so a trigger on it fires once.
5. **Batch means for the intervals** (Schmeiser 1982), 20 batches by
   default: Student's t with b - 1 degrees of freedom accounts for the
   uncertainty of the batch variance, and nothing is assumed about the shape
   of the autocorrelation. The variance's interval is that of the mean of
   `(x - mean)^2`. The integrated autocorrelation time by Sokal's window is
   reported independently and guards `converged`: each batch must span at
   least 10 `tau` (for AR(1), the lag-one correlation of batch means of m
   samples is about `tau / 2m`, 5% at `m = 10 tau`), so a short run with few
   effective samples is not declared converged on intervals that are too
   narrow. The means, variances and batch sums come from prefix sums of each
   history (less its first value, against cancellation), so a new sample
   costs O(batches); `tau` is recomputed when the window has grown by 5%.
6. **Window.** Once `[statistics]` averages run, the intervals are those of
   the samples after their start: the uncertainty of what the averages hold.
   The averages start when the trigger fires, after the detection; the
   samples between `t_stationary` and then were not kept per cell.
7. **Restarts** carry each history (time and value), the truncation state and
   the check schedule (the prefix sums are rebuilt from the history in the
   same order), so a resumed run writes the same rows bitwise
   (tested); `statistics.reset` starts it afresh.

## Validation

- `test/convergence_test.cpp`, against synthetic AR(1) signals with known
  answers: Student's t quantiles against tables to 1e-8; `tau` of AR(1) with
  phi = 0, 0.5, 0.8 within 8% of `(1 + phi) / (1 - phi)`; over 1000
  replicates (phi = 0.8, `tau` = 9), 95% intervals hold the exact mean and
  variance 95% of the time within three binomial standard deviations, and
  their mean width is that `tau` sets; MSER-5 finds a 1000-sample offset of 5
  standard deviations at 1000-1050 samples, rejects a drift and deletes
  little from a stationary series; a run is converged at the sample count
  the tolerance implies.
- `test/triggers_test.cpp`: a trigger on `convergence_stationary` starts
  the statistics at the step the file first says stationary; a resumed run
  writes the same rows bitwise.
