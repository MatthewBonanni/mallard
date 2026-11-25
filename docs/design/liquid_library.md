# Library of named liquids

Status: implemented (issue #1571), on the evaporating droplets of #1390.
User docs: [named liquids](../guide/particles.md#named-liquids); keys:
[`[particles.liquid]`](../input.md#particlesliquid).

`[particles.liquid] fluid = "water"` makes every property of a droplet's
liquid a function of its temperature, from CoolProp's reference equations,
instead of the constants given property by property.

## Decision

- CoolProp ([Bell et al. 2014](../references.md#bell-2014)) is an optional
  dependency, `-DMallard_ENABLE_COOLPROP=ON`, off by default as HDF5 and CGNS
  are. Without it a named fluid is an input error that says how to enable
  it, and liquids given property by property behave exactly as before.
- CoolProp runs on the host at setup only. It fills tables, which the
  droplets' kernels interpolate on the device: no CoolProp call, branch on
  the fluid or virtual function on the step path.
- Any key of `[particles.liquid]` overrides the table of its property with
  the constant, so a named liquid is a starting point, not a closed box.

Alternatives rejected: porting the equations of state to device code (a
Helmholtz energy per fluid, saturation by iteration, and the transport
correlations: thousands of lines to keep in step with CoolProp, slower per
evaluation than a table); polynomial fits from the literature (fixed
accuracy, ranges that stop short of the critical point, a different form per
property and source); calling CoolProp from kernels (impossible on GPUs).

## Host-only build

CMake builds CoolProp (a pinned tag, `v8.0.0`) as an `ExternalProject` with
the host C++ compiler, the one `nvcc_wrapper` or `hipcc` would call, into a
static library. Mallard includes only its C interface, `CoolPropLib.h`, in one
file, `src/particles/liquid_coolprop.cpp`, so nvcc, hipcc and SYCL compilers
never see CoolProp's headers, and the option needs nothing of the device
toolchain. `-DMallard_COOLPROP_ROOT` takes an installed static CoolProp
instead (offline builds, packages). Its first build takes about 5 minutes on 3
cores; the library is 24 MB. CoolProp is MIT licensed
(`licenses/CoolProp-MIT.txt`). A Spack variant `+coolprop` waits for the
Spack package itself (#1027).

## Tables

Each property is a cubic Hermite interpolant in T (`src/particles/liquid_curve.h`)
on knots crowded toward the critical point,
$T_k = T_{max} - (T_{max} - T_{min})\, u_k (u_k + a)/(1 + a)$, $u_k = 1 - k/(n-1)$,
with CoolProp's values and slopes (central differences of width $2\cdot10^{-6} T$)
at the knots. The map from T to the knot index is closed form, so an
evaluation is a square root, one table row and a cubic: no search. The
tables run from the triple point to $T_{max} = 0.999\,T_c$, or 0.999 of the
temperature where a correlation ends below $T_c$ (CoolProp's surface tension of
ethanol stops 0.8 K short).

| Table | Of | Error measured on |
|---|---|---|
| density | $\rho_l$ | value |
| enthalpy | $h_s(T) = \int_{T_b}^T c_p\,dT$ (Gauss-Legendre, 5 points per interval) | value and slope ($c_p$) |
| temperature | $T(h_s)$ | value |
| latent heat | $L = h_v - h_l$ | value |
| saturation pressure | $\ln p_{sat}$ | value (the relative error of $p_{sat}$) |
| boiling temperature | $T(\ln p)$ | value |
| viscosity | $\ln \mu$ | value |
| conductivity | $k$ | value |
| surface tension | $\sigma$ | value |

Each table doubles its intervals from 8 until its largest relative error at
the quarters of its intervals (and, for the enthalpy, its slope's at
$t = (3 \pm \sqrt 3)/6$, where a Hermite slope's error peaks) is within
`tolerance`, 1e-6 by default, or it reaches 8193 knots. The setup log reports
each table's knots and error. The inverse tables, $T(h_s)$ and $T(\ln p)$, use
$a = 1$ (their curves are steep, not singular, at the end) and are followed
by two Newton steps on the forward table, so that `temperature_of(sensible(T))`
and the boiling temperature at the table's own saturation pressure are exact to
round-off. Past its ends the enthalpy continues linearly at its end slopes, so
the inversion stays defined for any energy; the other tables clamp.

The stretched knots matter: on uniform knots the error near the critical
point, where $\rho_l$, $L$ and $\sigma$ go to their critical values as powers of
$1 - T/T_c$ below one, falls only after thousands of knots (dashed below). The
quarters matter too: CoolProp's conductivity has a kink where its critical
enhancement starts (n-hexane at 278 K, n-octane at 299 K), which midpoints
alone missed. That kink, and the enhancement's growth near $T_c$, make the
conductivity the only table that reaches 8193 knots before 1e-6 for most
fluids, at 1e-6 to 7e-6 there, far below the correlations' own uncertainty
(a few percent).

![Largest relative error of the tables of water and n-dodecane against their knots, on stretched and uniform knots](../images/guide/liquid_library_resolution-light.svg#gh-light-mode-only)
![Largest relative error of the tables of water and n-dodecane against their knots, on stretched and uniform knots](../images/guide/liquid_library_resolution-dark.svg#gh-dark-mode-only)

The tables are one `Kokkos::View` per liquid on the device and its host copy
(seeding, gathering and the setup log evaluate on the host); a `Curve` is
five numbers in the `Liquid` struct the kernels already capture.

## Energy reference

The droplet model of #1390 keeps one energy reference between gas and
liquid: the liquid's enthalpy is the vapor's mechanism enthalpy at the boiling
point less the latent heat there, plus the sensible enthalpy above it. With a
named liquid the sensible enthalpy is $h_s(T)$, so $c_p(T)$ is its exact slope
and the droplet temperature is $T(h_s)$, and the latent heat at the surface is
the table's $L(T_s)$, not Kirchhoff's continuation. The droplet's state is
its sensible enthalpy and mass, and the gas still receives exactly
$-(\Delta E + \Delta M\, h_{ref})$ of them, so gas and droplets conserve energy
to round-off whatever the tables are. The diameter follows
$\rho_l(T)$: a droplet's mass is its state, its diameter is
$(6 M / \pi \rho_l(T))^{1/3}$.

## Pressure

The tables are of the saturated liquid: the liquid has no pressure of its own
in the model, and the droplets' heat and mass transfer read the gas pressure
only at the surface (saturation, film properties). CoolProp gives the
compressed liquid's deviation from the saturated one:

| Liquid | T, p | $\rho$ | $c_p$ |
|---|---|---|---|
| water | 350 K, 5 MPa | +0.23% | -0.26% |
| water | 450 K, 5 MPa | +0.30% | -0.39% |
| n-heptane | 400 K, 5 MPa | +1.8% | -1.3% |
| n-dodecane | 450 K, 6 MPa | +1.8% | -0.8% |
| n-dodecane | 600 K, 6 MPa | +9.8% | -6.0% |
| oxygen | 120 K, 10 MPa | +4.2% | -9.0% |

At engine pressures and near the critical temperature that is more than the
tables' error by orders of magnitude, so tables in $(T, p)$ for density and
$c_p$ are a follow-up, with the energy reference extended to a
pressure-dependent liquid enthalpy.

## Verification

`test/liquid_library_test.cpp` checks:

- against an analytic liquid with the critical-point singularities of a real
  one ($c_p \sim (1 - T/T_c)^{-1/2}$, $\rho$, $L$ and $\sigma$ as powers below
  one): every table, evaluated on the device at 4001 temperatures, within 3
  times the tolerance (1e-4 and 1e-7), and the inverses exact to 1e-13;
- with CoolProp, every library fluid's tables against CoolProp at 4001
  temperatures from the triple point to the end of the tables, on the device:
  the largest relative errors at the default tolerance are 1e-6 or less for
  density (9.5e-7, methanol), $c_p$ (8.6e-7, n-heptane), latent heat,
  saturation pressure, viscosity and surface tension, 2.2e-8 for the boiling
  temperature, and 6.4e-6 for the conductivity (nitrogen);
- an isolated water droplet in air at 1000 K reaching the wet-bulb
  temperature of Abramzon and Sirignano's model with CoolProp's water to 1e-6.

Liquids given property by property are bitwise unchanged without CoolProp
and with it.

## Cost

Setup: about 2 s for water on one core (CoolProp evaluations; the
conductivity's 8193 knots are most of it), once per run. Memory: about 11,000
knots, 180 kB, per liquid. Per evaluation in a kernel: one square root, a
table row of four values and a cubic, plus two Newton steps for the
temperature of an enthalpy.

## Not in this layer

- Tables in $(T, p)$ of the compressed liquid ([pressure](#pressure)).
- Multicomponent liquids and mixtures (CoolProp's mixture models are slower
  and narrower; tables in composition would be another dimension).
- Using the surface tension in breakup or Weber-number models; for now it is a
  particle expression variable (`SIGMA_P`).
- Spack `+coolprop` (#1027).
