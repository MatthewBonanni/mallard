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
| `chemistry_reference.py` | Reference data for the chemistry tests from Cantera (`pip install cantera`): writes the CSV files in `test/data/chemistry/`, which are committed so the tests need no Cantera |
| `make_sphere_mesh.py` | Gmsh tetrahedral mesh of the quarter domain around a sphere (symmetry planes y = 0 and z = 0), refined on the sphere and through the bow-shock layer |
| `plot_sphere.py` | Bow-shock standoff of supersonic flow over a sphere, from the meridian-plane output, against Billig's correlation |
| `animate_sphere.py` | Sphere animation: Mach number on the horizontal meridian plane, schlieren on the vertical one, the bow shock revolved into a 3D surface, and the standoff history against Billig |
| `sedov.py` | Exact 3D Sedov-Taylor blast (similarity ODEs): the constant xi0 of R = xi0 (E t^2 / rho0)^(1/5), and a run's shock radius and density profile against it |
| `animate_sedov.py` | Sedov-Taylor animation: density on the three symmetry planes (mirrored to full disks) with the exact shock sphere, orbiting camera, shock radius and density profile against the exact solution |
