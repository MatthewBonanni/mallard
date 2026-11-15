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
boundaries are `outlet_pressure` at 1 atm, through which the free streams
also entrain. The mesh (`mixing_layer_setup.py`) is a tensor product of
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
| coarse | 20,700 (200, 60 um) | 37,865 steps, 2.3 min | 14.20 | -6.9% | 15.31 | -6.6% | 1848 / 2071 |
| base (this input) | 69,300 (100, 30 um) | 88,759 steps, 10 min | 14.86 | -2.6% | 15.92 | -2.9% | 1822 / 2042 |
| base, dx / 2 | 138,600 (50, 30 um) | 110,584 steps, 24 min | 14.94 | -2.1% | 15.99 | -2.5% | 1817 / 2046 |
| fine | 246,600 (50, 15 um) | 228,802 steps, 87 min | 15.18 | -0.5% | 16.22 | -1.0% | 1809 / 2022 |

![Peak temperature, Y_OH and heat release rate along the layer](../../docs/images/guide/mixing_layer_peaks-light.svg#gh-light-mode-only)
![Peak temperature, Y_OH and heat release rate along the layer](../../docs/images/guide/mixing_layer_peaks-dark.svg#gh-dark-mode-only)

The lift-off length converges to the reference's: from 200 to 50 um cells the
error of `x_OH` falls from 1.06 to 0.40 to 0.08 mm, close to second order,
and the refinement across the layer accounts for most of it (halving dx alone
moves `x_OH` by 0.08 mm). The fine mesh puts the flame base within 0.5%
(`x_OH`) and 1.0% (`x_T`) of the reference. Before ignition the peak `Y_OH`
and heat release rate grow exponentially along x, as in the reference, and
the fine mesh follows their growth within 30%, a shift of about 0.3 mm.

![Temperature and species across the layer](../../docs/images/guide/mixing_layer_profiles-light.svg#gh-light-mode-only)
![Temperature and species across the layer](../../docs/images/guide/mixing_layer_profiles-dark.svg#gh-dark-mode-only)

Across the layer (fine mesh, aligned at the stoichiometric mixture fraction
`Z_st` = 0.304), temperature and species overlay the reference at 15 mm and
remain close downstream. The layer is not steady there: the mixing layer is
unstable (Kelvin-Helmholtz), and disturbances from the step at the plate and
from the reflecting boundaries grow along it into waves of transverse
velocity of a few m/s that wrinkle the flame beyond about 18 mm. They grow
under refinement, as numerical dissipation falls, and thicken the burning
layer: the time-averaged peak temperature at 25-30 mm is 46-50 K (2.4%)
below the steady reference on the fine mesh, against 0-30 K on the coarser
ones. Characteristic boundaries (`nscbc` on every boundary) made the waves
stronger, not weaker. A quieter setup, which would make the downstream flame
comparable at the finest resolutions, is tracked in
[#1051](https://github.com/MatthewBonanni/mallard/issues/1051).

The reference is converged: with 201, 401 and 801 points across the layer
`x_OH` is 15.235, 15.261 and 15.266 mm (the last on a similarity grid half
again as wide), and a relative tolerance of 1e-7 instead of 1e-6 moves it by
0.002 mm. The runs in the table started from the 201-point reference, whose
slightly thinner layer gives the base mesh one row fewer than this input's
(300 x 232 = 69,600 cells).
