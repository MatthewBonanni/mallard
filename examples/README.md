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
| `wedge` | Mach 1.76 flow over an 8 degree ramp; the oblique shock matches theory (p2/p1 = 1.498) | 19,200 quads, minutes |
| `cylinder` | Viscous flow past a cylinder at Re = 100 (vortex shedding) on a Gmsh O-grid; generate the mesh with `tools/make_cylinder_mesh.py` first. Reproduces St, Cd and lift amplitude to within 2% | 49,152 quads, ~50 min on a laptop CPU |
| `viscous_shock_tube` | Daru & Tenaud viscous shock tube at Re = 200: shock / boundary-layer interaction in a closed box, giving a lambda shock and a primary vortex by t = 1. Matches the grid-converged wall density of Zhou et al. to 0.5 RMS; check with `tools/plot_viscous_shock_tube.py` | 500,000 quads, 58,500 viscous-limited steps: use a GPU build (or Nx = 500, Ny = 250: ~20 min on 6 threads) |
