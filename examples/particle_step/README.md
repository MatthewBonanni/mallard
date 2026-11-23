# Solid particles over a backward-facing step

Particles of four Stokes numbers carried by channel flow over a planar
backward-facing step, after the particle-laden step flows of Fessler & Eaton
(1997, 1999). The flow separates at the edge of the step into a shear layer
that rolls up into vortices, and a recirculation zone forms under it. How the
particles respond depends on their inertia: tracers roll up with the vortices
and fill the recirculation zone, particles of Stokes number near 1 are flung
out of the vortex cores and enter the recirculation zone only in tongues
between the vortices, and heavy particles cross the shear layer almost
undisturbed and never enter it.

The particles are point particles coupled one way: in a dilute suspension
they do not change the flow, so every Stokes number runs in the same flow.
Fessler & Eaton (1999) measured how far the particles modify the gas at mass
loadings of a few percent to 40%; this example leaves that out.

## The experiment and this model

| | Fessler & Eaton | This example |
|---|---|---|
| Geometry | step height H = 26.7 mm, inlet channel 40 mm, expansion ratio 5:3 | the same ratios, planar (2D), H = 1 |
| Inflow | fully developed turbulent channel flow, Re_H = 18,400 on the centerline velocity U0 | laminar channel flow (Poiseuille), Re_H = 3000, Mach 0.3 |
| Unsteadiness | the turbulence of the inflow and of the shear layer | the shear layer's own roll-up, locked by a weak forcing at the step's edge |
| Particles | glass and copper beads of 70 to 150 µm, large-eddy Stokes numbers 0.5 to 7.4; the flow is vertical, downward | St = 0 (tracers), 0.25, 1 and 4; no gravity |
| Wall impacts | | elastic rebound |

The Stokes number is the particle's relaxation time over the large-eddy time,
$St = \tau / (5 H / U_0)$ with $\tau = \rho_p d^2 / (18 \mu)$: $5 H / U_0$ is the
period of the shear layer's vortices here ($f H / U_0 = 0.16$ at their
roll-up). The particles have a density of 2000 and diameters of 1.9 × 10⁻³ H to
7.7 × 10⁻³ H (`run_particle_step.py` computes them); their drag is
Schiller-Naumann's, so their actual relaxation time is a little shorter than
$\tau$ when they slip (peak particle Reynolds numbers in the quantity
`RE_P_MAX`).

| Mesh and numerics | |
|---|---|
| Domain | inlet channel 20 H long, 30 H after the step |
| Cells | 29,280 quadrilaterals from `make_step_mesh.py`: squares of H / 24 from the step to x = 14 H, stretched upstream (where the channel flow does not change along x) and toward the outlet |
| Scheme | TENO5, HLLC, SSPRK3, CFL 1; Navier-Stokes, μ = U0 H / 3000 |
| Boundaries | `inlet_state` with the parabolic profile and `outlet_pressure`, both with `nscbc`; a sponge from x = 22 H; adiabatic walls |
| Forcing | a transverse body force of peak 0.033 U0² / H, over a Gaussian of radius 0.1 H at the step's edge, at $f H / U_0 = 0.16$ (`[[sources]] lip_forcing`) |
| Particles | 45,600 per Stokes number, a 760 x 60 lattice filling the inlet channel from x = -19.5 H to -0.5 H, seeded at the gas velocity in the developed flow |

Without the forcing the laminar shear layer is only convectively unstable at
this Reynolds number: its first vortices, from the start of the run, leave
the domain and the layer near the step settles into a steady state while the
flow downstream stays unsteady. The forcing stands in for the turbulence of
the experiment's inflow.

## Results

RESULTS

## Limits

- The flow is planar and laminar at a sixth of the experiment's Reynolds
  number: its vortices are coherent two-dimensional rollers that pair, not
  turbulent eddies, and its recirculation zone is longer than the
  experiment's (about 7 H). The trends with the Stokes number are what
  compares, not the profiles.
- The particles are seeded once, in the inlet channel, rather than fed
  continuously (injectors are #759): the statistics come from the window in
  which they pass the step, and the particles near the inlet's walls, which
  move slowly, arrive last.
- No gravity: the experiment's flow was vertical, along gravity, which
  speeds the heavy particles up without moving them across the shear layer.
- No collisions between particles, and one-way coupling, which is right for
  the experiment's dilute loadings.

## Running

From this directory (2D build):

```sh
python make_step_mesh.py step.msh
python run_particle_step.py --args --kokkos-num-threads=2
python animate_particle_step.py profiles profiles.png st_0 st_0.25 st_1 st_4
python animate_particle_step.py render particle_step.mp4 st_0 st_0.25 st_1 st_4 --poster particle_step.jpg --poster-t 30
```

RUNTIME

## References

- J. R. Fessler and J. K. Eaton, "Particle response in a planar sudden
  expansion flow", Exp. Therm. Fluid Sci. 15, 413-423 (1997).
- J. R. Fessler and J. K. Eaton, "Turbulence modification by particles in a
  backward-facing step flow", J. Fluid Mech. 394, 97-117 (1999).
