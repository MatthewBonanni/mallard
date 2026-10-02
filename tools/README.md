# Tools

Python scripts need numpy, scipy, matplotlib, imageio and imageio-ffmpeg.

| Script | Purpose |
|---|---|
| `mallard_vtu.py` | Minimal reader for Mallard's VTU files (cells, cell data, time) |
| `plot_vtu.py` | Plot one cell field of a VTU file |
| `compare_png.py` | Side-by-side plots of a field from several VTU files |
| `plot_sod.py` | Density profiles against the exact Sod solution |
| `animate.py` | MP4/GIF animation of a VTU series: density (or `--var`, including `VORTICITY`) with contours, plus numerical schlieren |
| `make_cylinder_mesh.py` | Gmsh O-grid around a cylinder |
