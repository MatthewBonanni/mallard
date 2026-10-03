# Examples

Run each case from its own directory, e.g.

```sh
cd examples/riemann_2d
../../build/src/Mallard -i input.toml --kokkos-num-threads=8
python ../../tools/animate.py solut riemann
```

| Case | What it shows | Size, time on 12 CPU threads |
|---|---|---|
| `sod` | Sod shock tube with TENO5 on a 200x4 strip | 800 cells, seconds |
| `shu_osher` | Mach 3 shock running into an entropy wave, the standard test for high-order schemes | 800 cells, seconds |
| `riemann_2d` | 2D Riemann problem, configuration 3, on 320,000 triangles with TENO5 | ~20 minutes |
| `riemann_2d_quads` | The same problem on 160,000 quadrilaterals | ~9 minutes on 6 threads |
| `double_mach` | Double Mach reflection of a Mach 10 shock: a split bottom boundary (inflow, then wall) and an exact moving shock on the top boundary | 460,000 triangles, ~1 hour |
| `explosion_3d` | Spherical explosion (3D build): one octant with symmetry planes, TENO5 on 64^3 hexahedra; the solution stays spherically symmetric | 262,144 cells, ~5 minutes on 8 threads |
| `taylor_green_3d` | Taylor-Green vortex at Re = 1600 (3D build): the high-order workshop DNS case on the octant [0, pi]^3 with symmetry planes. At 64^3 per octant (128^3 full box) the dissipation peak is 0.01302 at t = 8.1, against 0.01286 at t = 9.0 for the 512^3 spectral DNS | 262,144 cells, 32,000 steps: ~1 hour on one A100 |
| `taylor_green_3d` (`input_periodic.toml`) | The same case on the full periodic box [0, 2 pi]^3 (no symmetry planes), for runs on several GPUs | 128^3 = 2.1M cells |
| `sphere_mach3` | Mach 3 flow over a sphere (3D build) on an unstructured Gmsh tetrahedral mesh of the quarter domain (generate it with `tools/make_sphere_mesh.py`): HLL with bound-preserving TENO5. The bow-shock standoff settles at Delta / R = 0.226, 10% above Billig's correlation (0.205): 0.01 D, under half the tetrahedron edge length (0.025 D) in the shock layer | 800,000 tetrahedra, ~1.5 hours on four A100s |
| `sedov_3d` | Sedov-Taylor point blast (3D build): one octant, bound-preserving TENO5 on 128^3 hexahedra, against the exact similarity solution (`tools/sedov.py`). The shock radius stays within 1-3% of R = 1.0328 (E t^2 / rho0)^(1/5) | 2.1M cells, ~1.5 hours on four A100s |
| `wedge` | Mach 1.76 flow over an 8 degree ramp; the oblique shock matches theory (p2/p1 = 1.498) | 19,200 quads, minutes |
| `cylinder` | Viscous flow past a cylinder at Re = 100 (vortex shedding) on a Gmsh O-grid; generate the mesh with `tools/make_cylinder_mesh.py` first. Reproduces St, Cd and lift amplitude to within 2% | 49,152 quads, ~50 min on a laptop CPU |
| `sphere_re300` | Viscous flow past a sphere at Re = 300 (3D build): periodic shedding of hairpin vortices, on a Gmsh mesh of prisms (boundary layer) and tetrahedra from `tools/make_sphere_re300_mesh.py`; MUSCL. St = 0.133, Cd = 0.669, Cl = 0.073 against 0.137, 0.656, 0.069 (Johnson & Patel 1999), within 0.2%, 0.4% and 5% of a mesh with 2.4 times the cells | 850,000 cells, 1.04M steps: ~4.7 hours on four A100s |
| `viscous_shock_tube` | Daru & Tenaud viscous shock tube at Re = 200: shock / boundary-layer interaction in a closed box, giving a lambda shock and a primary vortex by t = 1. Matches the grid-converged wall density of Zhou et al. to 0.5 RMS; check with `tools/plot_viscous_shock_tube.py` | 500,000 quads, 58,500 viscous-limited steps: use a GPU build (or Nx = 500, Ny = 250: ~20 min on 6 threads) |
| `mallard` | Mach 8 flow over a flying mallard from an impulsive start: RHLL with bound-preserving TENO5 on a Gmsh triangle mesh; generate the mesh with `tools/make_mallard_mesh.py` first | 1.07M triangles, ~1.5 hours on a GPU |
| `premixed_flame` | Freely propagating stoichiometric H2/air flame at 1 atm with mixture-averaged transport (V8), started from Cantera's flame in its frame (generate the restart file first, see the input's header); the consumption speed settles within 0.1% of Cantera's 2.332 m/s | 300 cells, 20 per thermal thickness, ~10 min on one thread |
| `h2_ignition` | Constant-volume ignition of stoichiometric H2/air at 1200 K, 1 atm with `MallardReactor` (`../../build/src/MallardReactor -i input.toml`, writes `reactor.csv`); ignition delay 44.2 us as Cantera | 0D, milliseconds |
