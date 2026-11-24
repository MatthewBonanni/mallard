# Shock wave hitting a curtain of glass beads

A Mach 1.66 shock in air hits a curtain of glass beads 2 mm thick, after the
multiphase shock tube of Sandia National Laboratories (Wagner et al., 2012).
Part of the shock reflects off the curtain, a weaker one passes through it,
and the post-shock flow drags the beads downstream. The flow behind the
reflected shock is slower than the flow behind the transmitted one, so the
upstream beads feel less drag than the downstream ones and the curtain
spreads as it moves.

The beads are point particles coupled two ways (`coupling = "two-way"`):
their drag, pressure-gradient and added-mass forces and their heating act
back on the gas, so the curtain reflects a shock and attenuates the one it
lets through. The run compares the curtain's edges with the experiment's
schlieren edges.

## The experiment and its parameters

| Quantity | Value | Source |
|---|---|---|
| Incident shock | Mach 1.66 ± 0.02, 568 ± 15 m/s | Wagner et al. (2011), Table 1 |
| Gas ahead of it | air at 82.7 kPa, 296.4 K | Wagner et al. (2011), Table 1 |
| Test section | 76.2 mm square | Wagner et al. (2012) |
| Beads | soda-lime glass, sieved to 106 to 125 µm (115 µm here), 2520 kg/m³ | Wagner et al. (2011); Beresh et al. (2011) |
| Curtain | 2 mm thick over the bottom 75% of the height, spanning the test section | Wagner et al. (2011) |
| Volume fraction | 19 ± 2% at mid-height (24% at the ceiling, 13% at the floor) | Wagner et al. (2011) |

The run takes the mid-height values everywhere: a uniform curtain of
volume fraction 0.19, 2 mm thick, across the whole height, in a planar
tube. Behind the incident shock the gas is at 252.1 kPa, 423.8 K and
304.2 m/s (Rankine-Hugoniot for Mach 1.66; the inflow keeps this state).
The beads start at rest and at the gas temperature, with a specific heat
of 840 J/(kg K). Drag is `"singh"`, which corrects the drag of a sphere for
the slip Mach number: the beads peak at slip Mach numbers of 0.54 to 0.70
and particle Reynolds numbers of 2,200 to 3,000 (the quantities
`M_slip_max` and `Re_p_max`), well inside its range.

| Mesh and numerics | |
|---|---|
| Domain | 0.65 m x 76.2 mm, the curtain's upstream edge at x = 0.2 m |
| Cells | 6,500 x 16 quadrilaterals: 0.1 mm along the tube; the flow is planar |
| Scheme | TENO5, HLLC, SSPRK3, CFL 0.8; Navier-Stokes with Sutherland's viscosity |
| Beads | 304,000 parcels of 119.6 beads per meter of depth (200 x 1,520 lattice) |
| Deposition | `deposition_width` 0.3 mm, about two bead spacings |
| Boundaries | `inlet_state` (the post-shock gas) upstream, `outlet_extrapolation` downstream, symmetry planes as the walls |

## Results

The edges are the positions that 1% and 99% of the beads have not passed,
as displacements from their initial positions, against the time since the
shock reached the curtain. Wagner et al. tracked them in schlieren movies at
mid-height in four runs at Mach 1.66; `wagner_edges.csv` holds the median
and range of the four, digitized from Beresh et al. (2011), Fig. 20.

![Edges of the curtain against the experiment, coupled two ways and one way](../../docs/images/guide/shock_curtain_edges-light.svg#gh-light-mode-only)
![Edges of the curtain against the experiment, coupled two ways and one way](../../docs/images/guide/shock_curtain_edges-dark.svg#gh-dark-mode-only)

| t after impact (µs) | Upstream edge, experiment (mm) | Upstream edge, Mallard (mm) | Downstream edge, experiment (mm) | Downstream edge, Mallard (mm) |
|---|---|---|---|---|
| 100 | 0.5 | 0.6 | 2.6 | 0.8 |
| 200 | 1.8 | 2.2 | 8.6 | 3.3 |
| 300 | 4.1 | 4.7 | 16.5 | 7.1 |
| 400 | 7.7 | 8.2 | 26.1 | 12.3 |
| 500 | 12.3 | 12.5 | 36.5 | 18.7 |
| 600 | 17.1 | 17.8 | 48.0 | 26.2 |
| 650 | 21.1 | 20.7 | 53.4 | 30.3 |

- **The upstream edge follows the experiment** within 0.9 mm over 650 µs:
  it is set by the drag of the gas behind the reflected shock on the
  upstream beads, which the model reproduces.
- **The downstream edge moves about half as fast as in the experiment**
  (30.3 mm against 53.4 mm at 650 µs), so the curtain spreads to 11.5 mm
  instead of 34 mm.
  The downstream beads sit in gas that the beads upstream of them have
  slowed: in the model the gas loses momentum to the beads but keeps all
  the volume, and the beads feel an isolated sphere's drag. In the curtain
  the gas squeezes between the beads and speeds up, and neighbors raise
  each bead's drag several times; DeMauro et al. (2017) also saw the gas jet
  through the curtain once it deformed.
- **The beads pass through one another.** Beads from the middle of the
  curtain overtake those ahead of them (73 of the 200 columns of the
  lattice by 650 µs) and collect at the downstream edge in a sheet denser
  than the initial curtain (up to 0.3 to 0.5 in 0.1 mm bins). The
  steeper gradient on the downstream side is what the X-ray measurements
  show (Wagner et al., 2011), but there the peak volume fraction falls to
  about 5% by 280 µs: real beads collide (#1463).
- **Pressures.** The reflected shock passes the wall sensors at
  x = -68.6 and -99.1 mm 158 µs apart, as in the experiment (Beresh et al.,
  2011), but it is stronger: 4.68 p₁ behind it at -68.6 mm against 4.3 p₁,
  and it does not weaken upstream (4.60 p₁ at -99.1 mm against about
  3.9 p₁). The transmitted shock has a pressure ratio of 2.60, 85% of the
  incident shock's 3.05; far downstream the experiment's transmitted shock
  is 84% of the shock without a curtain (2.45 against 2.92 at 425 mm).
- **Numerics.** The edges move by less than 0.5% between cells of 0.2 and
  0.1 mm. The flow around the curtain stays planar to 2 × 10⁻⁴; on the
  0.1 mm cells, whose aspect ratio is 48, the foot of the transmitted shock
  shows transverse oscillations of up to 2.5% in density, which the
  quantities above do not depend on.

Edges at 650 µs, against the model choices. Coupled one way, every bead
sees the undisturbed post-shock flow: the curtain moves as a block, its
upstream edge almost twice as far as in the experiment, and does not
spread. The two-way coupling is what spreads it. The pressure-gradient and
added-mass forces add 12% to the downstream edge's travel; the drag law,
the deposition width and the mesh change the edges by a few percent.

| Run | Upstream edge (mm) | Downstream edge (mm) |
|---|---|---|
| Experiment | 21.1 | 53.4 |
| This input (0.1 mm cells) | 20.7 | 30.3 |
| 0.2 mm cells | 20.8 | 30.1 |
| 0.2 mm, `deposition_width` 0.6 mm | 21.3 | 28.6 |
| 0.2 mm, `drag = "schiller_naumann"` | 21.0 | 29.4 |
| 0.2 mm, without `forces` (drag only) | 20.4 | 26.9 |
| 0.2 mm, `coupling = "one-way"` | 38.5 | 38.1 |

## Limits

This is the dilute point-particle model applied well outside its range,
which `mallard check` says (`W-PARTICLE-VOLUME-FRACTION` at 0.19,
`W-PARTICLE-SIZE`: the beads are larger than a tenth of the cells). What it
leaves out, and the issues that would add it:

- the volume the beads take from the gas: no blockage, no acceleration of
  the gas through the curtain (#1460);
- the effect of neighbors on drag and heat transfer (#1461);
- the compressible unsteady force during the shock's passage, which
  dominates the first microseconds, and the history force (#1462);
- collisions and the granular stress (#1463);
- the curtain's variation with height, the slits in the floor and ceiling,
  the walls' boundary layers and three-dimensional motion.

Quantitative agreement with the downstream edge and the pressure histories
needs the first two at least.

## Running

From this directory (2D build):

```sh
../../build/src/mallard -i input.toml --kokkos-num-threads=2
python animate_shock_curtain.py extract .
python animate_shock_curtain.py edges .
python animate_shock_curtain.py render shock_curtain.mp4 . --poster shock_curtain.jpg --poster-t 300
```

The figure of the edges is redrawn from the edges of this run and of a run
with `coupling = "one-way"` (each with its `frames.npz`) by
`python tools/docs/shock_curtain.py --two-way RUN --one-way RUN`, from the
repository's root.

6,440 steps, 73 minutes on 2 threads of an Apple M4 Pro. The restart frames of the animation (every 2.5 µs, 281 frames,
which carry the beads) take 7.8 GB; `extract` reduces them to `frames.npz`.
`render` draws the test section at mid-height (numerical schlieren of the
gas, the beads colored by their velocity), the pressure along the tube, the
edges against the experiment, and the beads' volume fraction, at 3840 x 2160
and at half that size.

## References

- J. L. Wagner, S. J. Beresh, S. P. Kearney, W. M. Trott, J. N. Castaneda,
  B. O. Pruett and M. R. Baer, "A multiphase shock tube for shock wave
  interactions with dense particle fields", Exp. Fluids 52, 1507-1517
  (2012).
- J. L. Wagner, S. J. Beresh, S. P. Kearney, B. O. Pruett, E. K. Wright and
  G. C. Stoker, "Flash X-ray measurements of shock wave interactions with
  dense particle fields", Sandia report SAND2011-9369C (2011).
- S. J. Beresh, J. L. Wagner, S. P. Kearney, E. K. Wright, M. R. Baer and
  B. O. Pruett, "Experimental characterization of energetic material
  dynamics for multiphase blast simulation", Sandia report SAND2011-8421
  (2011).
- E. P. DeMauro, J. L. Wagner, S. J. Beresh and P. A. Farias, "Unsteady drag
  following shock wave impingement on a dense particle curtain measured
  using pulse-burst PIV", Phys. Rev. Fluids 2, 064301 (2017).
- Y. Ling, J. L. Wagner, S. J. Beresh, S. P. Kearney and S. Balachandar,
  "Interaction of a planar shock wave with a dense particle curtain:
  modeling and experiments", Phys. Fluids 24, 113301 (2012).
