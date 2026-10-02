# Tools

Python scripts need numpy, scipy, matplotlib, imageio and imageio-ffmpeg.

| Script | Purpose |
|---|---|
| `mallard_vtu.py` | Minimal reader for Mallard's VTU files (cells, cell data, time) |
| `plot_vtu.py` | Plot one cell field of a VTU file |
| `compare_png.py` | Side-by-side plots of a field from several VTU files |
| `plot_sod.py` | Density profiles against the exact Sod solution |
| `plot_viscous_shock_tube.py` | Viscous shock tube at t = 1 against the grid-converged reference of Zhou et al.: wall density, lambda-shock triple point |
| `animate.py` | MP4/GIF animation of a VTU series: density (or `--var`, including derived `VORTICITY` and `MACH`) with contours, plus numerical schlieren |
| `animate_grid.py` | Several cases side by side in one animation, synchronized in normalized time, with optional wall-profile panels against reference data. `hero_grid.json` makes the README animation (run from the repository root after running the three examples at the resolutions given in their headers) |
| `make_cylinder_mesh.py` | Gmsh O-grid around a cylinder |
| `make_mallard_mesh.py` | Triangle mesh (via the gmsh Python module) around a flying mallard silhouette |
