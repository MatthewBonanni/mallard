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
| `double_mach` | Double Mach reflection of a Mach 10 shock: a split bottom boundary (inflow, then wall) and an exact moving shock on the top boundary | 460,000 triangles, ~1 hour |
| `wedge` | Mach 1.76 flow over an 8 degree ramp; the oblique shock matches theory (p2/p1 = 1.498) | 19,200 quads, minutes |
| `cylinder` | Viscous flow past a cylinder at Re = 100 (vortex shedding) on a Gmsh O-grid; generate the mesh with `tools/make_cylinder_mesh.py` first | 24,576 quads, ~1 hour |
