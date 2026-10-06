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
- **Transverse terms.** The target's own incoming wave already holds the
  target's transverse terms; adding the face's (`beta T-`) counts them
  twice. Turbulent inlets therefore default to `beta = 0`.
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

### Decaying turbulence in a duct (DNS, `examples/turbulent_inflow_duct`)

**Setup.**
- Box `6 x 2 x 2`, periodic across, 384 x 128 x 128 hexahedra.
- `U = 1`, `M = 0.2`, isotropic `u' = 0.1`, `L = 0.25`, `Re_L = u' L / nu = 100`.
- MUSCL (no limiter), HLLC, SSPRK3; `nscbc_inlet` and `nscbc_outlet`, no sponge.
- Statistics over `t = 8-30` (3.7 flow-throughs), averaged over y and z.

**Inlet.**
- **Reynolds stresses.** The cells next to the inlet carry `R_uu, R_vv, R_ww = 0.0097, 0.0106, 0.0094` for 0.01 (`k` within 1%); `R_uv` is `2e-5`.
- **Scales.** Probes in those cells give `L_y = 0.23` for 0.25. The integral time `U T = 0.32` is longer than at the faces: the first cell averages over a cell and smooths.

**Adjustment and decay.** The digital-filter field is not solenoidal. Of an isotropic one, a third of the energy is dilatational, which a low-Mach flow cannot carry as vortices. Within `0.4 L` the energy drops to `0.85 k_in`, well short of the `2/3` bound. After that:

| `x / L` | 1 | 4 | 8 | 12 | 16 | 20 |
|---|---|---|---|---|---|---|
| `k / k_in` | 0.89 | 0.85 | 0.80 | 0.73 | 0.65 | 0.58 |
| `R_uu / R_vv` | 1.34 | 1.29 | 1.26 | 1.24 | 1.18 | 1.16 |

- **Dissipation lags.** The dissipation `eps = -U dk/dx` is low at first: `eps L / u'^3 = 0.32` up to `x / L = 8`, rising to 0.44 at 12 and 0.60 at 16, as the cascade the synthetic field lacks builds up over 1-2 eddy turnovers (`t u' / L = x u' / (U L)`).
- **No power law yet.** The power-law decay of grid turbulence
  ([Comte-Bellot & Corrsin 1966](../references.md#comte-bellot-corrsin-1966),
  `k ~ t^-1.25`) is not yet established within the domain.
- **Transient anisotropy.** `R_uu` exceeds the transverse stresses by 30% after the adjustment. The normal component's dilatational part leaves differently from the tangential ones, and the anisotropy then relaxes toward isotropy.

**Spurious pressure.**

| `x / L` | 0.4 | 1 | 4 | 8 | 20 |
|---|---|---|---|---|---|
| `p'_rms / (rho U u')`, `zero_net_flux = true` | 0.90 | 0.55 | 0.15 | 0.13 | 0.11 |
| `p'_rms / (rho U u')`, `zero_net_flux = false` | 1.19 | 0.95 | 0.80 | 0.79 | 0.78 |

- **With `zero_net_flux`** (the default), the pressure fluctuations decay away from the inlet to `1.1-1.3 rho u'^2`, the level of the turbulence's own (hydrodynamic) pressure.
- **Inlet near field.** The near field, `0.9 rho U u'` in the first cells, is the evanescent response to the non-solenoidal part of the injected field. It decays over a few `L` and does not radiate.
- **Without `zero_net_flux`**, a plane acoustic wave of `p' = 0.078 = 7.8 rho u'^2` fills the duct: its average along a transverse line equals the local value. This is `rho c` times the inlet-mean normal fluctuation.
- **Ratio.** The default removes 6-8 times the hydrodynamic level of spurious pressure downstream.

### Spatially developing channel at `Re_tau = 180` (DNS)

`examples/channel_inflow_retau180`.

**Setup.**
- Box `8 pi h x 2h x 4 pi / 3 h`, periodic in z, isothermal walls, 384 x 96 x 128 hexahedra (the mesh of the periodic channel example, repeated twice along x).
- `Re_b = 5600`, `M_b = 0.2`. MUSCL (no limiter), HLLC, SSPRK3.
- The inlet's `profile` holds the statistics of Mallard's periodic channel (`Re_tau = 172`). The length scales are `0.5h, 0.15h, 0.15h` for `u` and `0.1-0.2h` for `v`, `w`.
- The flow starts from the periodic channel repeated twice along x (`tools/tile_restart.py`). Statistics run over `t = 40-110 h / U_b` (2.8 flow-throughs), averaged over z.
- **Outlet.** A sponge covers `x > 20h` (not analyzed), and the `nscbc_outlet` runs with `beta = 0`. With the default `beta` (the local Mach number), a near-wall eddy reversed the flow in the outlet's last cells at `t = 18-19 h / U_b` and the run diverged, also with the sponge. That is a property of the outlet, not of the inflow.

**Development.** Quantities against the precursor, smoothed over `+-0.5h`:

| `x / h` | 0.5 | 2 | 4 | 6 | 8 | 12 | 16 | 19.5 |
|---|---|---|---|---|---|---|---|---|
| `C_f` | 1.05 | 1.03 | 1.01 | 1.00 | 1.00 | 0.99 | 0.98 | 1.01 |
| peak `u_rms+` | 0.91 | 0.86 | 0.85 | 0.89 | 0.92 | 0.95 | 0.96 | 0.96 |
| peak `v_rms+` | 0.94 | 0.86 | 0.85 | 0.89 | 0.90 | 0.92 | 0.94 | 0.97 |
| peak `w_rms+` | 1.07 | 1.07 | 1.02 | 0.98 | 0.96 | 0.97 | 0.99 | 0.99 |
| peak `-u'v'+` | 0.57 | 0.75 | 0.92 | 0.98 | 0.98 | 0.97 | 1.01 | 1.00 |

- **Wall shear.** It is within 5% of the precursor everywhere and within 2% from `x = 2h`.
- **Shear stress.** `-u'v'` recovers by `4.6h`. The synthetic field has the right `R_xy` at the inlet, but not the eddies that carry it to the wall.
- **Normal stresses.** These take longest. `u_rms` and `v_rms` first dip by 15% (the dilatational part shed, as in the duct). They are back within 10% by about `8-10h`, and within 5% from `13.7h` (`u_rms`) and `16.6h` (`v_rms`).
- **Overall.** All the quantities stay within 5% of the fully developed channel from about `17h`. [Keating et al. (2004)](../references.md#keating-2004) report that synthetic inflow needs long development lengths, about `20h` for a channel. Wall shear and shear stress recover faster here because the Reynolds stress is imposed in full, the mean profile is the precursor's, and the length scales are those of the outer layer.
- **Inlet pressure.** Next to the inlet (`x = 0.02h`, channel center), `p'_rms = 10 tau_w`, of which the transverse line average is `3.7 tau_w`. This is the near field of the non-solenoidal injection, as in the duct, against about `1 tau_w` in the developed flow.
