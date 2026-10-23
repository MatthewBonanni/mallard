# Vortex shedding behind a cylinder

At Reynolds number 100, the wake of a circular cylinder becomes unstable and sheds vortices alternately
from its two sides: the Kármán vortex street. The shedding frequency, as a Strouhal number, and the
drag and lift on the cylinder are known accurately from many computations. This tutorial builds the mesh
from a Gmsh script of your own, solves the Navier-Stokes equations, records the forces on the wall, and
measures all three. It runs in about 5 minutes on a laptop CPU.

You will use: a Gmsh geometry with physical groups, a structured ring of quadrilaterals around the wall
and triangles elsewhere, the Navier-Stokes equations with a viscosity set by the Reynolds number, a
no-slip wall and a characteristic far field, initial conditions as expressions, and wall forces. Files:
[`tutorials/cylinder/`](../../tutorials/cylinder). Setup: [Tutorials](index.md).

## 1. The problem

A cylinder of diameter D = 1 in a uniform stream U at Re = U D / ν = 100. Mallard solves the compressible
equations, so the stream also has a Mach number. At Mach 0.2 compressibility changes the forces by about
2% (the Prandtl-Glauert factor), small enough to compare with incompressible references:

| Quantity | References (incompressible) |
|---|---|
| Strouhal number St = f D / U | 0.164-0.166 |
| Mean drag coefficient C_D | 1.33-1.35 |
| Lift amplitude C_L' | 0.33-0.34 |

The [validation](https://matthewbonanni.github.io/mallard/validation/#cylinder-at-re-100) page lists the
computations behind these ranges, and runs the same case on a much finer mesh.

## 2. Mesh

[`cylinder.geo`](../../tutorials/cylinder/cylinder.geo) is a Gmsh script with four parts:

- **Points and curves.** The cylinder (radius 0.5), a ring around it (radius 1) and the far field
  (radius 20) are each four arcs. Four radial lines join the cylinder to the ring.
- **A structured ring.** The four blocks between cylinder and ring are *transfinite* surfaces with
  *recombined* cells. That makes them quadrilaterals: 96 around the cylinder and 16 across, growing
  geometrically from a first cell of 0.008 D, which resolves the thin boundary layer.
- **Triangles elsewhere.** The ring's outer cell size grows to 2 at the far field, with a box of size 0.12
  through the first 12 diameters of the wake, where the vortices travel. Gmsh *fields* set these sizes.
- **Physical groups.** The curves `cylinder` and `farfield` become Mallard's boundary zones of the same
  names, and the surface `fluid` holds the cells.

```bash
mkdir -p run/cylinder && cd run/cylinder
cp $MALLARD/tutorials/cylinder/input.toml .
gmsh $MALLARD/tutorials/cylinder/cylinder.geo -2 -o cylinder.msh     # 13,000 cells
python3 $MALLARD/tutorials/cylinder/plot_mesh.py cylinder.msh --out .
```

![The whole mesh with the far field, and the quadrilateral ring and the triangles near the cylinder](../images/tutorials/cylinder_mesh-light.svg#gh-light-mode-only)
![The whole mesh with the far field, and the quadrilateral ring and the triangles near the cylinder](../images/tutorials/cylinder_mesh-dark.svg#gh-dark-mode-only)

Every parameter can be changed from the command line, e.g. `-setnumber n_theta 192 -setnumber dn0 0.004`
for a finer ring, or interactively in the Gmsh GUI (`gmsh cylinder.geo`, then *Mesh → 2D*). Mallard reads
the mixed mesh as it is: the reconstruction and the fluxes work on any mix of triangles and
quadrilaterals. [Meshes](../guide/meshes.md#mesh-files-gmsh) has the details.

## 3. Input file

[`input.toml`](../../tutorials/cylinder/input.toml) uses units of the cylinder and of the free stream:
D = 1, density and sound speed 1, so U = M = 0.2, p = T = 1/γ (gas constant 1), and μ = ρ U D / Re = 0.002.
One time unit is D / a. The flow passes the cylinder in 5, and a shedding period lasts about 30.

| Table | In this case |
|---|---|
| `[run]` | `t_stop = 160`, about five shedding periods after the shedding has saturated, at CFL 1 |
| `[mesh]` | `type = "file"`, `cylinder.msh` |
| `[initialize]` | Expressions in x and y: the free stream, slowed to rest at the wall, plus a small upward kick behind the cylinder |
| `[[boundaries]]` | `cylinder`: `wall_adiabatic` (no slip, no heat flux); `farfield`: `farfield`, the free stream outside, entering or leaving by characteristics |
| `[numerics]` | HLLC, SSPRK3, third-order TENO-E |
| `[physics]` | `navier_stokes`, γ = 1.4, μ = 0.002, Pr = 0.72 |
| `[[forces]]` | The force on `cylinder`, pressure and viscous parts, every 20 steps: `forces_cylinder.csv` |
| `[[write_data]]` | Every 2 time units: density, velocity, pressure and vorticity, `solut/cylinder_*.vtu` |

The kick matters. A perfectly symmetric start keeps the wake symmetric until round-off breaks the
symmetry, and the instability only grows by a factor e every ~50 time units. With the kick, shedding
saturates by t = 60. Expressions accept the usual functions and operators; see
[Initialization](../guide/initialization.md#expressions).

## 4. Run

```bash
$MALLARD/build/src/mallard -i input.toml --kokkos-num-threads=4
```

This is about 70,000 steps: viscous flow on a fine wall mesh makes the time step small. The run takes
about 5 minutes on 3 laptop CPU threads. The progress rows show the maximum Mach
number rising from 0.2 to about 0.3 as the flow accelerates around the cylinder.

## 5. Forces

`forces_cylinder.csv` holds, every 20 steps, `step, t` and the force on the wall, split into pressure
and viscous parts: `Fx_pressure, Fy_pressure, Fx_viscous, Fy_viscous`, per unit depth. Divided by the
dynamic pressure times the diameter, ½ ρ U² D = 0.02, they give the drag and lift coefficients:

```bash
python3 $MALLARD/tutorials/cylinder/plot_results.py . --t-start 60 --out .
```

```text
t >= 60: St 0.1660, mean Cd 1.3446, Cl amplitude 0.3157 (references: St 0.164-0.166, Cd 1.33-1.35, Cl 0.33-0.34)
```

![Drag and lift coefficients against time, with the reference ranges shaded](../images/tutorials/cylinder_history-light.svg#gh-light-mode-only)
![Drag and lift coefficients against time, with the reference ranges shaded](../images/tutorials/cylinder_history-dark.svg#gh-dark-mode-only)

The kick starts the shedding at once, and by t = 60 the lift is a steady sine. The drag oscillates at
twice that frequency, once per vortex shed from either side. The Strouhal number comes from the period
of the lift (its upward zero crossings) after t = 60.

| | This mesh (13,000 cells) | References |
|---|---|---|
| St | 0.166 | 0.164-0.166 |
| Mean C_D | 1.345 | 1.33-1.35 |
| C_L' | 0.316 | 0.33-0.34 |

The frequency and the drag fall within the references. The lift amplitude is 5% low: this mesh is coarse
around the cylinder, and the 384 × 128 O-grid of the
[validation](https://matthewbonanni.github.io/mallard/validation/#cylinder-at-re-100) page gets 0.331.

## 6. The vortex street

The last snapshot's vorticity shows the street: vortices of alternating sign shed from the two sides of
the cylinder and carried downstream a little slower than the free stream. Past x = 12 the triangles
coarsen and the vortices fade, as they should do only far downstream: extend the wake box (`L_wake`) to
follow them further.

![Vorticity in the wake: alternating vortices shed from the cylinder](../images/tutorials/cylinder_field-light.svg#gh-light-mode-only)
![Vorticity in the wake: alternating vortices shed from the cylinder](../images/tutorials/cylinder_field-dark.svg#gh-dark-mode-only)

In ParaView, open `solut/cylinder.pvd` and color by `VORTICITY` to watch the street form.
`python3 $MALLARD/tools/post/animate.py solut cylinder --var VORTICITY --xlim -2 16 --ylim -3 3` makes a movie
of it.

## 7. Things to try

- **Refine.** `-setnumber n_theta 192 -setnumber n_ring 32 -setnumber dn0 0.004 -setnumber h_wake 0.06`
  roughly quadruples the cells. Compare St, C_D and C_L' with the table above: which one moves most?
- **Another Reynolds number.** Below Re ≈ 47 the wake stays steady: set `mu = 0.005` (Re = 40) and the
  lift decays to zero, leaving two attached eddies behind the cylinder. At Re = 200, `mu = 0.001`, St rises
  to about 0.19.
- **Curved walls.** The straight faces between the ring's nodes cut slightly into the circle. Writing the
  mesh with second-order elements, `gmsh ... -order 2`, lets Mallard curve the wall
  ([curved boundaries](../guide/meshes.md#curved-boundaries)).

Next: [Supersonic flow over a cone](cone.md), in 3D.
