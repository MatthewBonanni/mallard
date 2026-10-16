# Laminar flow over an airfoil

A NACA 0012 airfoil at Mach 0.5, 1° angle of attack and a chord Reynolds
number of 5000: subsonic, viscous and steady, with a thin laminar boundary
layer that separates near the trailing edge. This is case C1.3 of the 1st
International Workshop on High-Order CFD Methods (2012), whose reference drag
and lift come from adjoint-adapted high-order solutions. This tutorial builds
a structured C-grid in Gmsh, marches Mallard's Navier-Stokes solver to the
steady state, and compares the drag, lift and surface distributions on two
meshes with the reference.

You will use: a Gmsh script with transfinite (structured) blocks and
clustering, wall and far-field boundaries, a steady state by time marching,
wall forces, and post-processing of wall quantities from the volume output.
Files: [`tutorials/airfoil/`](../../tutorials/airfoil). Setup:
[Tutorials](index.md).

## 1. Geometry

The airfoil has unit chord, its leading edge at the origin and a closed
trailing edge: the workshop's NACA 0012, whose last coefficient is −0.1036
instead of −0.1015, so the thickness vanishes at x = 1. The domain is a "C"
around it: a half circle of radius R = 100 chords about the trailing edge,
closed downstream by the outlet at x = 1 + R. A far field that far away
changes drag and lift by less than the discretization does.

![The C-shaped domain with its boundary zones, and the airfoil in the free stream](../images/tutorials/airfoil_geometry-light.svg#gh-light-mode-only)
![The C-shaped domain with its boundary zones, and the airfoil in the free stream](../images/tutorials/airfoil_geometry-dark.svg#gh-dark-mode-only)

## 2. Mesh

[`naca0012.geo`](../../tutorials/airfoil/naca0012.geo) splits the domain
into six four-sided blocks, above and below the airfoil, the near wake (to
four chords behind the trailing edge) and the far wake, and meshes each as a
transfinite surface of quadrilaterals:

- along each surface, 96 cells clustered at both edges (`Using Bump`);
- away from the wall, 64 cells growing geometrically from a first cell of
  0.001 chords (`Using Progression`, with the growth ratio solved in the
  script), about 20 cells across the boundary layer at mid-chord;
- along the wake, 96 cells over the near wake, starting at the trailing-edge
  spacing, and 32 more to the outlet.

The lines between blocks (the wake cut and the line ahead of the leading
edge) are interior faces; only the airfoil (`"airfoil"`) and the outer
boundary (`"farfield"`) are physical groups, so they are Mallard's boundary
zones. `refine` multiplies every cell count and divides the first cell
height:

```bash
mkdir -p run/airfoil && cd run/airfoil
cp $MALLARD/tutorials/airfoil/input.toml .
gmsh $MALLARD/tutorials/airfoil/naca0012.geo -2 -o naca0012.msh                      # 28,672 cells
gmsh $MALLARD/tutorials/airfoil/naca0012.geo -2 -setnumber refine 2 -o naca0012.msh  # 114,688 cells
python3 $MALLARD/tutorials/airfoil/plot_mesh.py naca0012.msh --out .
```

![The whole mesh, the airfoil, and the leading and trailing edges, with the boundary zones](../images/tutorials/airfoil_mesh-light.svg#gh-light-mode-only)
![The whole mesh, the airfoil, and the leading and trailing edges, with the boundary zones](../images/tutorials/airfoil_mesh-dark.svg#gh-dark-mode-only)

## 3. Input file

[`input.toml`](../../tutorials/airfoil/input.toml) works in units of the
chord and of the free stream's density and sound speed: the free stream is
U = 0.5 at 1° (`u = [0.49992, 0.00873]`), p = T = 1/γ with a gas constant of
1, and μ = ρ U c / Re = 1e-4.

| Table | In this case |
|---|---|
| `[run]` | `t_stop = 120`: 60 chord lengths of flow, at CFL 1 |
| `[mesh]` | `naca0012.msh` |
| `[initialize]` | The free stream everywhere: an impulsive start |
| `[[boundaries]]` | `airfoil`: `wall_adiabatic` (no slip); `farfield`: `farfield` with the free stream, a characteristic condition through which the start-up waves leave |
| `[numerics]` | HLLC, SSPRK3, MUSCL with the Venkatakrishnan limiter |
| `[physics]` | `navier_stokes`, γ = 1.4, constant μ, Pr = 0.72 |
| `[[forces]]` | The pressure and viscous forces on the airfoil every 50 steps, `forces_airfoil.csv` |
| `[[write_data]]` | Density, velocity, pressure, temperature and vorticity every 20 time units |

## 4. Run

Explicit time steps are set by the thinnest cells at the wall (about
1.5e-4 here, 4.6e-5 on the refined mesh, where the viscous limit takes
over), so reaching t = 120 takes 800,000 steps on the first mesh and 2.6
million on the second:

```bash
$MALLARD/build/src/Mallard -i input.toml --kokkos-num-threads=8    # CPU threads
mpirun -n 4 $MALLARD/build/src/Mallard -i input.toml               # MPI ranks
```

This is a case for a GPU: on one A100 the first mesh takes about 25 minutes
and the second 1.5 hours (estimated), with the CUDA build of the
[first simulation](../guide/first_simulation.md) and the same input.

## 5. Monitor

The progress rows show the minimum density and pressure and the maximum
Mach number: after the start-up waves have left, the minimum pressure
settles near 0.65 and the maximum Mach number near 0.61, at the suction peak
of the leading edge. The forces file grows as the run goes; `plot_results.py`
(below) plots the drag and lift coefficients over time, or a quick look from
the shell:

```bash
tail -n 3 forces_airfoil.csv     # step, t, Fx and Fy of pressure, then of viscous stress
```

![Drag and lift coefficients over time on both meshes, against the workshop reference](../images/tutorials/airfoil_history-light.svg#gh-light-mode-only)
![Drag and lift coefficients over time on both meshes, against the workshop reference](../images/tutorials/airfoil_history-dark.svg#gh-dark-mode-only)

After the start-up transient the coefficients settle to steady values; on the first mesh drag and lift change by less than 1e-6 over the last 10% of the run.

## 6. Post-process and compare

```bash
python3 $MALLARD/tutorials/airfoil/plot_results.py coarse fine --labels coarse fine --out .
```

takes one directory per mesh, coarse first. It averages drag and lift over
the last 10% of each run, rotating the forces into the free-stream frame,
c_d = (F_x cos α + F_y sin α) / q∞ and c_l = (−F_x sin α + F_y cos α) / q∞
with q∞ = ½ ρ U² c, and computes the wall quantities from the cells next to
the airfoil in the last snapshot: Cp from their pressure, and the skin
friction Cf = μ u_t / (d q∞) from their tangential velocity and the
distance d of their centroid from the wall.

![Mach number and streamlines around the airfoil, and the separated flow at the trailing edge](../images/tutorials/airfoil_field-light.svg#gh-light-mode-only)
![Mach number and streamlines around the airfoil, and the separated flow at the trailing edge](../images/tutorials/airfoil_field-dark.svg#gh-dark-mode-only)

![Pressure and skin-friction coefficients along the upper and lower surfaces on both meshes](../images/tutorials/airfoil_surface-light.svg#gh-light-mode-only)
![Pressure and skin-friction coefficients along the upper and lower surfaces on both meshes](../images/tutorials/airfoil_surface-dark.svg#gh-dark-mode-only)

![Drag and lift coefficients against the mesh size, approaching the workshop reference](../images/tutorials/airfoil_convergence-light.svg#gh-light-mode-only)
![Drag and lift coefficients against the mesh size, approaching the workshop reference](../images/tutorials/airfoil_convergence-dark.svg#gh-dark-mode-only)

| Mesh | c_d | c_l |
|---|---|---|
| 28,672 cells (t = 60 c/U) | 0.05612 (+1.5%) | 0.01717 (−6.1%) |
| 114,688 cells (t = 19 c/U, unfinished) | 0.05566 (+0.6%) | 0.01802 (−1.4%) |
| Workshop reference | 0.05532 | 0.01827 |

## 7. Things to try

- **Transonic.** The workshop's inviscid case: Mach 0.8 at 1.25°,
  `type = "euler"`, `wall_adiabatic` becomes a slip wall; reference
  c_l = 0.3520, c_d = 0.0226 (shock drag). Use `riemann_solver = "RHLL"` for
  the shocks.
- **Curved faces.** `Mesh.ElementOrder = 2;` in the `.geo` file writes
  quadratic elements, whose curved airfoil faces Mallard follows
  ([meshes](../guide/meshes.md)).
- **The far field.** Halve `R`: the drag changes by about the far field's
  influence, which a study of R at fixed resolution measures.
