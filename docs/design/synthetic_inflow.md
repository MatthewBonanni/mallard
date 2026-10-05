# Design: synthetic turbulent inflow

Status: implemented (`[boundaries.turbulence]` of `nscbc_inlet`).

Spatially developing DNS (jets, flames, channels, boundary layers) need
turbulence entering through the inlet: a prescribed mean profile, Reynolds
stress `R_ij` and integral length and time scales, with the right spatial and
temporal correlations, and without spurious acoustics. Mallard's
requirements on top of that:

- **Unstructured inlets.** Faces have no (i, j) indexing.
- **GPUs.** The generator runs on the device every stage.
- **MPI.** The field must be bitwise the same on any number of ranks.
- **Restarts.** A restarted run must reproduce an uninterrupted one bitwise.

## Choice of method

| | Digital filter (Klein et al. 2003) | Synthetic eddies (Jarrin et al. 2006; Poletto et al. 2013) | Recycling-rescaling (Lund et al. 1998) |
|---|---|---|---|
| `R_ij` | exact, any realizable tensor (Cholesky factor) | exact for SEM; DFSEM only within its anisotropy limits | follows from the simulation |
| Length scales | per component and direction, by the kernel widths | one eddy size (per direction in variants) | follow from the simulation |
| Divergence | not solenoidal | DFSEM solenoidal for a uniform convection | solenoidal |
| Unstructured inlet | auxiliary grid on the plane, interpolated to faces | direct, at any point | needs point location and interpolation on a recycle plane, and communication between the ranks owning the two planes |
| Cost | independent of the scales (separable filters) | grows with the number of eddies overlapping a point | one extra plane of the solution |
| State | none (see below) | eddy positions, or a counter-based variant | none, but needs the domain to hold an equilibrium flow downstream |
| Applies to | any flow with known statistics | any flow with known statistics | equilibrium wall-bounded flows only (not jets, flames) |

Recycling-rescaling is ruled out as the primary method: it does not apply to
free shear flows or flames, and it needs an equilibrium flow at the recycle
plane. Between the synthetic methods, the digital filter reproduces any
realizable `R_ij` exactly and controls the length scales of every component
in every direction independently, and its cost does not grow with the
ratio of the inlet to the scales. Its fluctuations are not solenoidal, which
the synthetic-eddy variant of Poletto et al. avoids; the injection below
removes what of that radiates (the plane acoustic mode), and the validation
measures the rest. Mallard therefore implements the digital filter, with the
temporal correlation of Klein's third filter direction by Taylor's
hypothesis rather than the exponential forward-stepwise correlation of
[Xie & Castro (2008)](../references.md#xie-castro-2008), because the latter
carries state from step to step.

## The generator

On an auxiliary uniform grid on the inlet plane, one plane of unit-variance
random numbers per generator time step `dt_g` is filtered, separately for
each velocity component, with Gaussian kernels
([Klein et al. 2003](../references.md#klein-2003))

```
b_k ~ exp(-pi k^2 / (2 n^2)),  |k| <= N = ceil(2 n),  sum b_k^2 = 1
```

along the two transverse directions and in time, with `n = L / spacing`
(`n = L_normal / (U_c dt_g)` in time). The filtered field has unit variance
and the correlation `exp(-pi r^2 / (4 L^2))`, whose integral is `L`. Each
face then:

1. interpolates the filtered planes bilinearly in space and linearly in time
   to its centroid and the current time;
2. divides by the exact standard deviation of that interpolation. For
   unit-variance values with neighbor correlation `rho1 = sum_k b_k b_(k+1)`,
   it is `sqrt(prod_d ((1 - f_d)^2 + f_d^2 + 2 f_d (1 - f_d) rho1_d))` over
   the fractions `f_d` in each direction and in time. Without this, faces
   between grid points would carry less energy (5% at `L / 4` spacing);
3. multiplies by the lower Cholesky factor `a` of the local `R_ij`
   ([Lund et al. 1998](../references.md#lund-1998)): `u' = a v`, so that
   `<u'u'^T> = a a^T = R`;
4. adds the mean velocity.

The grid spacing is the smallest transverse length over 6
(`points_per_length`), and `dt_g` is the smallest streamwise length over
`6 U_c`. The inlet must be planar and normal to a coordinate axis. A
periodic direction across the inlet (from the mesh's periodic translations)
makes the grid periodic with the period, so that the field is too.

### Determinism without state

The random number of component `j` at grid point `(i_1, i_2)` of plane `m`
is a hash of `(seed, inlet, j, i_1, i_2, m)` (SplitMix64 rounds), mapped to a
uniform number of unit variance. Every plane is therefore a function of its
index alone:

- **Ranks.** Every rank computes the planes it needs itself, for its own
  faces; nothing is communicated, and the result does not depend on the
  partition.
- **Restarts.** The field at time `t` uses planes `floor(t / dt_g)` and the
  next one, and the time filter the `2 N_t` around them; a restarted run
  recomputes them. The generator has no state.
- **Cost.** Each new plane costs one random plane plus three filter passes on
  the auxiliary grid, about once per time step; the face kernel
  interpolates four points (two in 2D) per component.

### Zero net flux

The area mean of the normal fluctuation over the inlet is a plane wave
source: `p' = rho c <u_n'>`, much larger than the hydrodynamic pressure at
low Mach number. With `zero_net_flux` (the default) it is subtracted from
every face at every instant. The mean is a sum over all the inlet's faces,
which a rank does not all have, so it is computed on the auxiliary grid:
at setup, every rank gathers all faces (in the order of their global keys)
and accumulates their weights `A_f a_nj s_fj w_p` onto the grid points; at
run time the mean is a dot product of these weights with each new plane, in
fixed chunks summed in order. The result is the same on every rank and
partition. The removed mode carries about `4 L_y L_z / A` of the normal
Reynolds stress (`2 L_y / W` in 2D).

## Injection through `nscbc_inlet`

The characteristic inlet ([nscbc.md](nscbc.md)) now takes its target
velocity per face (`BoundaryData::char_target`), set from expressions or a
profile once, and from the generator at every stage:

- **Tangential velocity** is imposed exactly by default. In linear theory
  this does not reflect acoustic waves.
- **Normal velocity** relaxes toward the target through the face's incoming
  acoustic wave. A relaxation alone lags the fluctuations by `1/K`;
  imposing them exactly reflects outgoing waves. As
  [Guézennec & Poinsot (2009)](../references.md#guezennec-poinsot-2009)
  propose, the target's own incoming wave enters the face's:

  ```
  dw-/dt = K Z (u_n - u_n,t) - Z du_n,t/dt - beta T-,   w- = p - Z u_n
  ```

  If the interior carries the target, the face's pressure stays at the
  target and its normal velocity follows `u_n,t`. Each step adds
  `-Z (u_n,t(t + dt) - u_n,t(t))`, with the generator evaluated at both
  times.
- **Temperature** stays imposed (no temperature fluctuations).

Restart files carry the faces' `p_b` and `u_b` ([nscbc.md](nscbc.md)), and
the generator has no state, so turbulent inlets restart bitwise.

## Mean profile and statistics from a precursor

`u` of `nscbc_inlet` takes expressions in `x`, `y`, `z`. For turbulence,
`reynolds_stress` takes numbers or expressions, or `profile` reads a CSV of
plane averages as `tools/plane_average.py` writes it from a precursor's
`[statistics]` (`MEAN_U_*`, `COV_U_*_U_*`), interpolated along its
coordinate. Pivots of the Cholesky factor that round-off makes slightly
negative (next to walls) become zero.

## Validation

### Unit tests (`test/synthetic_inflow_test.cpp`, MPI tests)

- **Inlet statistics.** Anisotropic `R_ij` with a shear stress, and length
  scales that differ by component and direction (3D, 40 x 40 faces, 3000
  samples, `zero_net_flux = false`):
  - `R_xx`, `R_yy`, `R_zz`, `R_xy` within 1% of the targets
  - `L_y` of `u` 0.193 for 0.2; `U_c T` of `u` 0.507 for 0.5
  - 2D: within 1% (stresses), 5% (scales)
- **Zero net flux** to round-off (1e-14) with it; 0.003 or more without.
- **Determinism.** The field at a time is the same whatever the generator
  evaluated before, and differs for another inlet.
- **Restarts.** A turbulent inlet run of 40 steps equals 20 + 20 from a
  restart, bitwise; without restoring the faces' states it does not.
- **MPI.** A turbulent inlet on tetrahedra matches the serial run bitwise on
  2, 3 and 4 ranks.

VALIDATION_RESULTS
