# In-situ spectra, correlations and phase averages

`[statistics.spectra]` and `[statistics.phase]` (issue #284) extend the
per-cell running averages of `[statistics]` (#169) with quantities that
otherwise need raw snapshots: spectra and two-point correlations along
homogeneous directions, and phase averages of a periodic flow. Like the means,
they are deterministic, independent of the rank count and carried by restart
files.

## Spectra and correlations

Homogeneous directions H are periodic, unstretched directions of a generated
`cartesian` mesh, whose cell (i, j, k) has the global id (i Ny + j) Nz + k
(checked against the centroids at setup). For a sample:

1. The owned cells' values of every field move (one all-to-all) to slabs of
   whole planes normal to the first inhomogeneous axis; with every direction
   homogeneous, to x slabs.
2. Each line along a homogeneous direction in the slab is transformed by one
   device thread: a mixed-radix Stockham FFT over the prime factors of N
   (radix p costs O(N p), so any N works; twiddles from one table of
   exp(-2 pi i m / N), m < N, computed on the host). In a periodic box the x
   slabs, transformed along y (and z), are transposed to y slabs (a second
   all-to-all) for the x transforms.
3. Per plane, sums of a_hat conj(b_hat) over the wavenumbers of all
   homogeneous directions but one give the two-sided 1D spectra; in a
   periodic box, also the shell sums (one thread per plane and shell, looping
   only over the wavenumbers whose |k| may fall in the shell). Sums are in
   double precision.
4. The per-plane sums are gathered in plane order; in a periodic box they are
   added in plane order on every rank.

Every line and sum sees the same values in the same order on any rank count,
and the combination order is fixed by plane indices, so the results are
bitwise identical on 1-4 ranks (tested) and after restarts on another rank
count. The time averages (weight t - t_prev, like the means) are kept on the
host and stored in restart files as named arrays (format version 6). Outputs
subtract the product of the averaged plane means from the k = 0 terms, so they
describe fluctuations about the time-and-plane mean; correlations are the
inverse transforms of the averaged two-sided spectra (linear in the samples,
so equal to averaged correlations).

Cost on one A100 per sample: channel 192 x 96 x 128 (3 fields, 4 pairs)
11.4 ms (a step takes 32.3 ms); isotropic box 64^3 (3 fields) 2 ms (a TENO
step takes 155 ms). Sampling every 20 steps adds under 2% to the channel.

## Phase averages

Per cell, one time-weighted running mean per bin and field. The phase is
(t - t_reference) / period modulo 1, or (t - t_n) / (t_n - t_{n-1}) from the
last two upward crossings of a level by a variable in the cell holding a
probe point (located like `[[probes]]`; its value reaches every rank by an
all-reduce in which the other ranks add exact zeros), interpolated linearly
between samples. The trigger state (last sample, last two crossings) and the
bin weights are restart attributes, the bins restart fields.

## Validation

- Isotropic turbulence, 64^3, per-sample spectra every 20 steps to t / tau =
  4: shell spectra of velocity, density and pressure equal
  `tools/isotropic_turbulence_stats.py --no-deconvolve` on restart files of
  the same steps within 1e-14 (relative, every shell, 16 snapshots). Against
  the tool's default (deconvolved point values) they are lower by the cell
  average factor: 0.95 at k = 16, 0.41 at k = 32.
- Channel at Re_tau 180 (`examples/channel_retau180`, from a developed state,
  t = 320): averaged spectra and correlations of 5 samples equal
  `tools/insitu_statistics.py spectra` on restart files of the same steps
  within 3e-13 (relative to each quantity's largest value). Over 12,000 steps
  (Re_tau 182), the near-wall (y+ = 4.5) streamwise correlation in z has its
  minimum, -0.19, at dz+ = 54, the streak half-spacing of Kim, Moin & Moser
  (1987).
- Cylinder at Re 100, t = 100-400, 12 bins triggered by U_Y on the wake axis
  at x = 3: crossings give St = 0.1651; the bins at the probe cell equal the
  probe series binned by `tools/insitu_statistics.py phase` bitwise, and the
  phase-averaged U_Y keeps 0.991 of the signal's amplitude (a sinusoid binned
  in 12 keeps 0.989), i.e. no phase jitter.
- Unit tests: correlations against direct shifted products, shell spectra
  against a direct DFT, on box sizes with factors 2, 3 and 5, in a periodic
  box and a stretched channel; phase bins against the analytical means of an
  oscillating flow, trigger against period bitwise; bitwise restarts; rank
  independence on 1-4 ranks (`MPITest`).
