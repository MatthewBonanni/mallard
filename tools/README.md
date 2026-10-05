# Tools

Python scripts need numpy, scipy, matplotlib, imageio and imageio-ffmpeg; the 3D animations also need pyvista.

| Script | Purpose |
|---|---|
| `mallard_vtu.py` | Minimal reader for Mallard's VTU files (cells, cell data, time); also reads HDF5 snapshots (`.h5`), so the scripts below accept them too |
| `mallard_h5.py` | Reader for `format = "hdf5"` output (needs h5py): mesh, cell fields of a snapshot, and the snapshots of a series `PREFIX.xmf` (`animate.py` takes `--glob "PREFIX_[0-9]*.h5"`) |
| `plot_vtu.py` | Plot one cell field of a VTU file |
| `compare_png.py` | Side-by-side plots of a field from several VTU files |
| `plot_sod.py` | Density profiles against the exact Sod solution |
| `plot_viscous_shock_tube.py` | Viscous shock tube at t = 1 against the grid-converged reference of Zhou et al.: wall density, lambda-shock triple point |
| `animate.py` | MP4/GIF animation of a VTU series: density (or `--var`, including derived `VORTICITY` and `MACH`) with contours, plus numerical schlieren |
| `animate_grid.py` | Several cases side by side in one animation, synchronized in normalized time: field panels, wall profiles against reference data, 2D detonations with their soot foil, pre-rendered frame sequences (e.g. 3D views) and time series traced over a reference curve. `hero_grid.json` makes the README animation from the double Mach, 2D detonation, Taylor-Green (full box) and Re = 300 sphere examples; `detonation` panels show a detonation run's pressure and numerical soot foil, and the 3D panels are frames cropped from the case animations (the config's `_inputs` lists where each input comes from) |
| `make_cylinder_mesh.py` | Gmsh O-grid around a cylinder; `--half` gives the upper half with the axis cuts, the meridian plane of a sphere for axisymmetric runs |
| `plot_poiseuille.py` | Axisymmetric Hagen-Poiseuille flow: axial velocity against the exact r-weighted cell averages |
| `plot_noh.py` | Spherical Noh problem from an axisymmetric run: density against the exact solution, shock radius by direction |
| `plot_sphere_axisymmetric.py` | Steady axisymmetric flow past a sphere: drag, separation angle and wake length against the literature |
| `make_mallard_mesh.py` | Triangle mesh (via the gmsh Python module) around a flying mallard silhouette |
| `plot_taylor_green.py` | Taylor-Green vortex: kinetic energy and dissipation rate (`-dE/dt` and enstrophy-based) from `[integrals]` output against the spectral DNS reference |
| `animate_taylor_green.py` | Taylor-Green vortex animation: Q-criterion isosurfaces on the box mirrored from the computed octant, orbiting camera, and the dissipation rate tracing the reference |
| `chemistry_reference.py` | Reference data for the chemistry tests from Cantera (`pip install cantera`): writes the CSV files in `test/data/chemistry/`, which are committed so the tests need no Cantera |
| `transport_reference.py` | Transport reference data from Cantera: species viscosities, conductivities and binary diffusion coefficients, and mixture-averaged and unity-Lewis mixture properties, as CSV files in `test/data/chemistry/` |
| `flame_reference.py` | Freely propagating premixed flames (Cantera `FreeFlame`) of H2/air and CH4/air at 300 K and 1 atm, phi 0.6-1.4, mixture-averaged and unity Lewis: flame speeds, thermal thicknesses and profiles in `examples/premixed_flame/reference/`, and optionally each flame's full solution; `--single FUEL PHI T_U DIR` computes only one mixture's flames (both transport models, any fresh-gas temperature) |
| `flame_restart.py` | A premixed-flame run (input and restart file) in the flame's frame from a Cantera flame's full solution, at a given number of cells per thermal thickness: a 1D strip, or with `--ny` a periodic 2D channel whose front is wrinkled by random modes (`--perturb`); `--ref-delta` gives flames of different transport the same mesh |
| `flame_speed.py` | Consumption and displacement speeds of a premixed-flame run over time, against a reference flame speed |
| `plot_flame.py` | Temperature and heat release profiles of a premixed-flame run against Cantera's, aligned at the maximum of dT/dx |
| `isotropic_turbulence_restart.py` | Initial state of `examples/isotropic_turbulence` as a restart file: random solenoidal velocity with E(k) ~ k^4 exp(-2 (k / k0)^2) at a given Mt, uniform density and pressure, as exact cell averages of the same band-limited field on any N^3 mesh; prints the derived viscosity and tau |
| `isotropic_turbulence_stats.py` | Statistics of isotropic turbulence snapshots (restart files of a uniform hexahedral box): kinetic energy, enstrophy, dilatation, temperature, density and pressure variances on the full mesh and filtered to 64^3 (sixth-order or spectral derivatives, the references' protocols), Mt, Re_lambda, k_max eta; shell spectra and fields for the animation |
| `plot_isotropic_turbulence.py` | Isotropic turbulence runs against the filtered DNS of Johnsen et al. (2010) and Subramaniam et al. (2019): filtered and unfiltered histories (grid convergence) and spectra |
| `animate_isotropic_turbulence.py` | Isotropic turbulence animation: Q-criterion isosurfaces colored by dilatation, orbiting camera, the dilatation on a plane (shocklets) and the enstrophy and dilatation variance tracing the reference |
| `make_sphere_mesh.py` | Gmsh tetrahedral mesh of the quarter domain around a sphere (symmetry planes y = 0 and z = 0), refined on the sphere and through the bow-shock layer |
| `plot_sphere.py` | Bow-shock standoff of supersonic flow over a sphere, from the meridian-plane output, against Billig's correlation |
| `animate_sphere.py` | Sphere animation: Mach number on the horizontal meridian plane, schlieren on the vertical one, the bow shock revolved into a 3D surface, and the standoff history against Billig |
| `sedov.py` | Exact 3D Sedov-Taylor blast (similarity ODEs): the constant xi0 of R = xi0 (E t^2 / rho0)^(1/5), and a run's shock radius and density profile against it |
| `animate_sedov.py` | Sedov-Taylor animation: density on the three symmetry planes (mirrored to full disks) with the exact shock sphere, orbiting camera, shock radius and density profile against the exact solution |
| `make_sphere_re300_mesh.py` | Gmsh mesh of the full domain around a sphere for viscous flow: prism boundary layer, tetrahedra refined through the near wake; `--scale` refines every size for mesh studies |
| `plot_sphere_re300.py` | Sphere at Re = 300: Strouhal number from the lift, mean drag and lift over whole shedding periods, against Johnson & Patel and other simulations |
| `animate_sphere_re300.py` | Sphere wake animation: Q-criterion isosurfaces colored by streamwise velocity, orbiting camera, and the drag and lift histories |
| `plane_average.py` | Average `[statistics]` output (`MEAN_`, `COV_`) of a quad or hex mesh over homogeneous directions as a function of the others (e.g. a channel's profiles in y), grouping equal cell-center coordinates (stretched meshes included) and adding the dispersive part to the covariances; reads VTU or PVTU |
| `interpolate_restart.py` | Interpolate a restart file onto another Gmsh mesh of the same domain (inverse-distance weighting of the nearest cells), to start a refined run from a developed flow |
| `znd_restart.py` | Initial state of a detonation run from a ZND profile (`detonation_reference.py`): 1D, or 2D (`--ny`, `--ly`) with disks of unreacted gas behind the front (`--pocket`, repeatable; `--fresh-pocket` for fresh gas at rest) to trigger cells |
| `soot_foil.py` | 2D cellular detonation: mean front speed against D_CJ, triple points on the front and the cell width they imply, and the numerical soot foil (`P_MAX`) as an image |
| `animate_detonation_2d.py` | Cellular detonation animation: pressure and the soot foil building up in a window that follows the front, and the whole foil so far |
| `animate_flame_2d.py` | Side-by-side animation of 2D flame runs (`flame_restart.py --ny`): temperature over the adiabatic flame temperature with heat-release contours, and the consumption speeds against time |
| `perf_suite.py` | The performance set (`benchmarks/perf/`): runs it on 2D and 3D builds (best of several runs after a warm-up, with the spread), writes JSON/CSV, compares with the baselines for the hardware (fails when slower by more than the threshold and the spread) and updates them; standard library only |
