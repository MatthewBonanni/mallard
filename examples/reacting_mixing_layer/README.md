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
y = -8 to 12 mm: fuel enters through the left boundary below y = 0 and air
above it (two `inlet_state` conditions), and the right, top and bottom
boundaries are characteristic `outlet_pressure` conditions at 1 atm
(`nscbc`), through which the layer's displacement leaves (outward on both
sides) and acoustic waves leave too. The mesh (`mixing_layer_setup.py`) is a tensor product of
quadrilaterals, uniform along x, uniform across the layer and growing by 8%
per cell outside it; the run starts from the reference solution and lasts
two flow-throughs of the slower stream (1.2 ms). MUSCL, HLLC, SSPRK3, CFL 0.8.

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
| base (this input) | 69,600 (100, 30 um) | 88,000 steps | 14.93 | -2.2% | 16.04 | -2.2% | 1806 / 2056 |

![Peak temperature, Y_OH and heat release rate along the layer](../../docs/images/guide/mixing_layer_peaks-light.svg#gh-light-mode-only)
![Peak temperature, Y_OH and heat release rate along the layer](../../docs/images/guide/mixing_layer_peaks-dark.svg#gh-dark-mode-only)

![Temperature and species across the layer](../../docs/images/guide/mixing_layer_profiles-light.svg#gh-light-mode-only)
![Temperature and species across the layer](../../docs/images/guide/mixing_layer_profiles-dark.svg#gh-dark-mode-only)

Across the layer (aligned at the stoichiometric mixture fraction `Z_st` =
0.304), temperature and species follow the reference at every station. At 30
mm, 1.5 mm into the fuel side of `Z_st`, the base mesh has 360 K against the
reference's 318 K, and the peak temperature at 25 and 30 mm is within 5 and
15 K of it.

The layer is steady because no acoustic wave returns to the plate. The
mixing layer is convectively unstable (Kelvin-Helmholtz). In linear
stability theory for the reference's own profiles (inviscid and with
viscosity, variable density), the spatial growth rate peaks at about 500,
300 and 230 1/m at 5, 15 and 20 mm, at 25-45 kHz, and disturbances grow by a
factor of e^5 or more between 5 and 25 mm, more from closer to the plate.
Disturbances that reach the plate are amplified into waves that roll the
layer up downstream of the flame base and carry products into the fuel side.
With reflecting `outlet_pressure` sides and outlet, the start-up transient
keeps returning to the plate as sound. The waves then persist (several m/s
of transverse velocity at 15-25 mm on 50 x 15 um cells after two
flow-throughs), and their time average puts 1000 K and `Y_H2O` = 0.1 1.5 mm
into the fuel side at 30 mm. With characteristic boundaries the start-up
wave packet convects out by about 0.5 ms and the transverse velocity along
the layer falls below 0.1 m/s at 10-15 mm, with intermittent waves of
under 1 m/s at 20-30 mm on the base mesh. A smooth inflow profile in place
of the step at the plate makes no difference while the boundaries reflect.
Characteristic conditions on the inlet across the step drive spurious
velocities at the step
([#1179](https://github.com/MatthewBonanni/mallard/issues/1179)), so the
inlet keeps its fixed states.

The reference is converged: with 201, 401 and 801 points across the layer
`x_OH` is 15.235, 15.261 and 15.266 mm (the last on a similarity grid half
again as wide), and a relative tolerance of 1e-7 instead of 1e-6 moves it by
0.002 mm.
