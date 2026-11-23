# Solid particles over a backward-facing step

Particles of three Stokes numbers carried by channel flow over a planar
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
| Geometry | planar sudden expansion of ratio 5:3 (inlet channel 1.5 H high) | the same, in 2D, H = 1 |
| Inflow | fully developed turbulent channel flow, Re_H = 18,400 on the centerline velocity U0 | laminar channel flow (Poiseuille), Re_H = 3000, Mach 0.3 |
| Unsteadiness | the turbulence of the inflow and of the shear layer | the shear layer's own roll-up, locked by a weak forcing at the step's edge |
| Particles | 90 and 150 µm glass and 70 µm copper (1999); large-eddy Stokes numbers 0.5 to 7.4 (1997); the flow is vertical, downward | St = 0 (tracers), 1 and 4; no gravity |

The Stokes number is the particle's relaxation time over the large-eddy time,
$St = \tau / (5 H / U_0)$ with $\tau = \rho_p d^2 / (18 \mu)$: $5 H / U_0$ is the
period of the shear layer's vortices here ($f H / U_0 = 0.16$ at their
roll-up). The particles have a density of 2000 and diameters of 3.9 × 10⁻³ H
(St = 1) and 7.7 × 10⁻³ H (St = 4), which `run_particle_step.py` computes, and
rebound elastically off the walls. Their drag is Schiller-Naumann's, so their
actual relaxation time is shorter than $\tau$ when they slip: the particle
Reynolds number (the quantity `RE_P_MAX`) peaks at 9.1 for St = 1 and 25 for
St = 4, where the drag is 1.7 and 2.4 times Stokes'.

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

The statistics are over U0 t / H = 15 to 45 after the seeding, while the
cloud passes the step (`animate_particle_step.py profiles`). The mean flow
reattaches at x = 12.1 H (where the mean velocity at the floor turns
positive), with a weak secondary eddy along the floor from x = 2.3 to 7.7 H.

| | Tracers | St = 1 | St = 4 |
|---|---|---|---|
| Particles under the shear layer (0 < x < 12.1 H, y < H), on average | 4,096 | 1,131 (0.28 of the tracers) | 577 (0.14) |
| Near the floor (y < H / 2) at x = 4 H: concentration / inlet | 0 | 0.026 | 0 |
| at x = 6 H: concentration, mean streamwise velocity | 0.053, -0.22 U0 | 0.067, -0.03 U0 | 0 |
| at x = 8 H | 0.17, -0.35 U0 | 0.027, -0.10 U0 | 0.018, +0.05 U0 |
| at x = 10 H | 0.24, -0.20 U0 | 0.009, +0.10 U0 | 0.020, +0.14 U0 |

Against the trends Fessler & Eaton (1997) report for large-eddy Stokes
numbers of 0.5, about 1.4, and above 3:

- **Tracers fill the recirculation zone and roll up with the vortices**, as
  their smallest particles did, but in this laminar flow they fill it slowly:
  they enter at its downstream end, where the vortices hit the floor, and
  travel upstream with the reversed flow, which reaches back only to
  x = 5 H within the window. The corner of the zone exchanges its gas far
  more slowly than in the turbulent experiment.
- **St = 1 particles are flung out of the vortex cores** (the cores are empty
  in the video, where the tracers spiral in) **and enter the recirculation
  zone only in tongues** thrown down between the vortices, as Fessler &
  Eaton saw at St ≈ 1.4. A quarter as many are under the shear layer as
  tracers. The ones there lag the reversed flow (-0.03 to -0.10 U0 against
  -0.22 to -0.35 U0 for the gas), and near the reattachment they still move
  downstream.
- **St = 4 particles do not enter the recirculation zone**, as Fessler &
  Eaton found above St = 3: none reach the floor upstream of x = 8 H. The
  shear layer only bends their paths; those under it are crossing its lower
  edge on their way to the floor near the reattachment, where they bounce
  and keep moving downstream (+0.05 to +0.14 U0 where the gas moves
  upstream).

The comparison is qualitative: the flow is not the experiment's (below).

## Limits

- The flow is planar and laminar at a sixth of the experiment's Reynolds
  number: its vortices are coherent two-dimensional rollers, not turbulent
  eddies, and its recirculation zone is longer than the turbulent
  experiment's. The trends with the Stokes number are what compares, not the
  profiles.
- The mesh resolves the flow as a coarse direct simulation: `mallard check`
  reports cells of 4.9 Kolmogorov lengths on average and wall cells up to
  y+ = 6, and the mesh was not refined to converge the statistics.
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
python animate_particle_step.py profiles profiles.png st_0 st_1 st_4
python animate_particle_step.py render particle_step.mp4 st_0 st_1 st_4 --poster particle_step.jpg --poster-t 28
```

The flow takes 14,499 steps and each particle run 8,982 (U0 t / H = 45):
41,445 steps in all, 109 minutes on 2 threads of an Apple M4 Pro shared with
other jobs (68 ms per flow step when the machine was quieter, against 93 ms
under this load). The restart frames of each run (every 0.5, 300 frames, which
carry the particles) take about 1 GB; `extract` reduces them to
`frames.npz`. `render` draws the three runs over the tracer run's vorticity
at 3840 x 2160 and at half that size.

## References

- J. R. Fessler and J. K. Eaton, "Particle response in a planar sudden
  expansion flow", Exp. Therm. Fluid Sci. 15, 413-423 (1997).
- J. R. Fessler and J. K. Eaton, "Turbulence modification by particles in a
  backward-facing step flow", J. Fluid Mech. 394, 97-117 (1999).
