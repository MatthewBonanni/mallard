# Laminar reacting mixing layer (V11)

A spatially developing planar mixing layer: H2/N2 (1:1 by volume) at 300 K
and 50 m/s below air at 1150 K and 100 m/s, both at 1 atm, meeting at the
trailing edge of a splitter plate of zero thickness, with detailed chemistry
(`mechanisms/h2o2.yaml`) and mixture-averaged transport. The hot air
autoignites the mixture that forms in the layer: OH and heat release grow
along the layer until it ignites some 15 mm downstream, and a diffusion flame
trails behind. The flame is lifted and held by autoignition, not by
propagation: the flow is much faster than the premixed flames of the layer's
mixtures.

The reference solves the same flow without streamwise diffusion: the steady
planar boundary-layer equations with the same mechanism and Cantera's
mixture-averaged transport (diffusion velocities with a correction velocity,
enthalpy transport by them, viscous heating), marched in x from the plate
(`mixing_layer_reference.py`). In von Mises form with the similarity variable
`eta = psi / sqrt(x)` and `tau = ln x` the equations are parabolic in `tau`
on a fixed `eta` grid; without chemistry the solution is Lock's self-similar
mixing layer, so the march starts 1 nm from the plate and the layer is
resolved by the same points at every x.

The Mallard run (2D build) covers 30 mm downstream of the plate and
y = -8 to 12 mm, with a buffer of 10 mm beyond it: fuel enters through the
left boundary below y = 0 and air above it (two `inlet_state` conditions),
and the right, top and bottom boundaries are characteristic `outlet_pressure`
conditions at 1 atm (`nscbc`), through which acoustic waves leave and the
layer's displacement flows out (outward on both sides). The mesh
(`mixing_layer_setup.py`) is a tensor product of quadrilaterals, uniform
along x up to 30 mm and growing by 10% per cell in the buffer, uniform across
the layer and growing by 8% per cell outside it; the run starts from the
reference solution and lasts two flow-throughs of the slower stream
(1.2 ms). MUSCL, HLLC, SSPRK3, CFL 0.8.

## Running

From this directory (2D build):

```sh
python mixing_layer_reference.py reference --field
python mixing_layer_setup.py reference/field.npz .
../../build/src/mallard -i input.toml
python plot_mixing_layer.py --run base . --average 8 --history
```

The first command writes the reference (`ignition.csv`, `profiles.csv`, and
the initial state `field.npz`, 20 MB, which is not kept here; 20-40 minutes
on one CPU core). The second writes the mesh, the initial state and
`input.toml` (replacing this one, without its header) with cells of 100 um
along x and 30 um across the layer; `--dx` and `--dy` set others. The last
reduces the run to `results/base.csv` (peaks along x, the mean over the last
8 snapshots, 0.4 ms) and `results/base_profiles.csv` (profiles across the
layer) and compares every run in `results/` with the reference;
`--figure` draws the figures below from them.

## Validation

The lift-off (ignition) length is measured as where the peak `Y_OH` across
the layer first reaches 2e-3 (`x_OH`) and where the peak temperature first
exceeds the air's by 250 K (`x_T`), both from the per-snapshot peaks averaged
over the last 0.4 ms of each run. The position of the largest heat release
rate is also printed, but its maximum is too flat to locate within 1 mm.

| Run | Cells (dx, dy) | Cost on one A100 | `x_OH` [mm] | error | `x_T` [mm] | error | peak T at 20 / 30 mm [K] |
|---|---|---|---|---|---|---|---|
| reference | 401 points across | | 15.26 | | 16.40 | | 1782 / 2072 |
| coarse | 23,352 (200, 60 um) | 37,654 steps, 2.3 min | 14.33 | -6.1% | 15.44 | -5.8% | 1841 / 2070 |
| base (this input) | 75,168 (100, 30 um) | 87,898 steps, 12 min | 14.93 | -2.2% | 16.05 | -2.1% | 1805 / 2062 |
| fine | 260,603 (50, 15 um) | 226,466 steps, 89 min | 15.36 | +0.6% | 16.47 | +0.5% | 1770 / 2058 |

![Peak temperature, Y_OH and heat release rate along the layer](../../docs/images/guide/mixing_layer_peaks-light.svg#gh-light-mode-only)
![Peak temperature, Y_OH and heat release rate along the layer](../../docs/images/guide/mixing_layer_peaks-dark.svg#gh-dark-mode-only)

The lift-off length converges to the reference's: from 200 to 50 um cells
the error of `x_OH` goes from -0.93 to -0.33 to +0.10 mm, and on the fine mesh
`x_OH` and `x_T` are within 0.6% and 0.5% of the reference's. The peak temperature at 25 and 30 mm
is 13 and 14 K below the reference on the fine mesh.

Across the layer (aligned at the stoichiometric mixture fraction `Z_st` =
0.304), temperature and species on the fine mesh overlay the reference at
every station.

![Temperature and species across the layer](../../docs/images/guide/mixing_layer_profiles-light.svg#gh-light-mode-only)
![Temperature and species across the layer](../../docs/images/guide/mixing_layer_profiles-dark.svg#gh-dark-mode-only)

On the fine mesh the largest temperature difference across the layer is
124 K, at 25 mm, on the steep fuel-side flank 0.5 mm below `Z_st`: a shift of the profile by about
0.06 mm. It is 42 K at 30 mm. 1.5 mm into the fuel side the fine mesh has
300 and 319 K at 25 and 30 mm, against the reference's 301 and 318 K.

The layer is steady because no disturbance returns to the plate. The mixing
layer is convectively unstable (Kelvin-Helmholtz). Linear stability theory
for the reference's own profiles (variable density, viscous) gives spatial
growth rates of about 500, 300 and 230 1/m at 5, 15 and 20 mm, at 25-45 kHz.
Disturbances grow by a factor of e^5 or more between 5 and 25 mm, and by more
from closer to the plate. A disturbance that reaches the plate grows into
waves that roll the layer up downstream of the flame base, carrying products
into the fuel side. Sound returns to the plate in two ways:
- **Reflecting boundaries.** With fixed-pressure `outlet_pressure` sides and
  outlet, the start-up transient keeps returning to the plate as sound, and
  the waves persist: several m/s of transverse velocity at 15-25 mm on
  50 x 15 um cells after two flow-throughs. Their time average then puts
  1000 K and `Y_H2O` = 0.1 1.5 mm into the fuel side at 30 mm.
- **Waves reaching the outlet.** With characteristic boundaries the start-up
  wave packet convects out by about 0.5 ms. Waves that reach the outlet still
  make sound there, and on 50 x 15 um cells they return as intermittent
  waves of 1-4 m/s at 20-30 mm. The buffer, where the growing cells damp the
  waves before the outlet, removes them: on the fine mesh `x_OH` stays
  between 15.35 and 15.37 mm over the last 0.6 ms.

The buffer leaves the flow upstream unchanged: it changes the base mesh's
profiles up to 15 mm by less than 1 K, and its `x_OH` by 0.005 mm.

A smooth inflow profile in place of the step at the plate makes no
difference while the boundaries reflect. Characteristic conditions on the
inlet across the step drive spurious velocities at the step
([#1179](https://github.com/MatthewBonanni/mallard/issues/1179)), so the
inlet keeps its fixed states.

The reference is converged: with 201, 401 and 801 points across the layer
`x_OH` is 15.235, 15.261 and 15.266 mm (the last on a similarity grid half
again as wide), and a relative tolerance of 1e-7 instead of 1e-6 moves it by
0.002 mm.
