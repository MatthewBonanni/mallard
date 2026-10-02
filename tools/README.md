# Tools

Python scripts need numpy, scipy, matplotlib, imageio and imageio-ffmpeg; the 3D animations also need pyvista.

| Script | Purpose |
|---|---|
| `mallard_vtu.py` | Minimal reader for Mallard's VTU files (cells, cell data, time) |
| `plot_vtu.py` | Plot one cell field of a VTU file |
| `compare_png.py` | Side-by-side plots of a field from several VTU files |
| `plot_sod.py` | Density profiles against the exact Sod solution |
| `plot_viscous_shock_tube.py` | Viscous shock tube at t = 1 against the grid-converged reference of Zhou et al.: wall density, lambda-shock triple point |
| `animate.py` | MP4/GIF animation of a VTU series: density (or `--var`, including derived `VORTICITY` and `MACH`) with contours, plus numerical schlieren |
| `animate_grid.py` | Several cases side by side in one animation, synchronized in normalized time: field panels, wall profiles against reference data, pre-rendered frame sequences (e.g. 3D views) and time series traced over a reference curve. `hero_grid.json` makes the README animation (run from the repository root after running the double Mach, 2D Riemann (quads), viscous shock tube and Taylor-Green examples at the resolutions given in their headers; the Taylor-Green panels read 3D frames rendered from its snapshots, one PNG per snapshot with a `times.txt`, and the dissipation rate and the spectral DNS reference as CSV files with columns `t,y`) |
| `make_cylinder_mesh.py` | Gmsh O-grid around a cylinder |
| `make_mallard_mesh.py` | Triangle mesh (via the gmsh Python module) around a flying mallard silhouette |
| `plot_taylor_green.py` | Taylor-Green vortex: kinetic energy and dissipation rate (`-dE/dt` and enstrophy-based) from `[integrals]` output against the spectral DNS reference |
| `animate_taylor_green.py` | Taylor-Green vortex animation: Q-criterion isosurfaces on the box mirrored from the computed octant, orbiting camera, and the dissipation rate tracing the reference |
