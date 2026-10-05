![C++](https://img.shields.io/badge/C%2B%2B-20-blue)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23112953.svg)](https://doi.org/10.5281/zenodo.23112953)

![logo_dark](./docs/images/mallard_dark.png#gh-dark-mode-only)
![logo_light](./docs/images/mallard_light.png#gh-light-mode-only)

Mallard is a high-order unstructured finite volume solver for the compressible Euler and Navier-Stokes equations, written in C++ with [Kokkos](https://github.com/kokkos/kokkos) for performance portability.

![Mallard simulations](./docs/images/hero.gif)

*A cellular detonation in 2H2-O2-7Ar with finite-rate chemistry, its numerical soot foil recording the triple-point tracks (front speed within 0.01% of the Chapman-Jouguet speed); the Taylor-Green vortex at Re = 1600 on the full periodic box, whose kinetic-energy dissipation rate follows the 512³ spectral DNS of the [High-Order CFD Workshop](https://cfd.ku.edu/hiocfd/); the Mach 10 double Mach reflection; and the hairpin vortices shed by a sphere at Re = 300, with Strouhal number, drag and lift within 3%, 2% and 6% of Johnson & Patel ([`examples/`](examples)).*

> **NOTE:** Mallard is under active development; finite-rate chemistry is new, and its GPU performance is still being tuned.

## Features

- Compressible Euler and Navier-Stokes equations: a calorically perfect gas (constant or Sutherland viscosity), or thermally perfect gas mixtures
- Finite-rate chemistry with arbitrary Cantera (YAML) mechanisms read at run time: elementary, three-body, falloff, PLOG and Chebyshev reactions; an adaptive Rosenbrock integrator per cell with analytical Jacobians, Strang-split from the flow; mixture-averaged, unity-Lewis or constant-Lewis transport; an optional double-flux scheme for interfaces; `MallardReactor`, a 0D reactor tool
- 2D or 3D (a build option): unstructured meshes of triangles and quadrilaterals, or of tetrahedra, hexahedra, prisms and pyramids, read from Gmsh or HDF5 files or generated
- Face reconstruction:
  - First order
  - Second-order MUSCL with least-squares gradients and Barth-Jespersen or Venkatakrishnan limiting
  - TENO-E of orders 3 to 6 ([Liang, Shyy & Fu, J. Sci. Comput. 2025](https://doi.org/10.1007/s10915-025-02918-w)): k-exact least squares on a large central stencil and one sector stencil per face, a density-based troubled-cell indicator, characteristic-wise stencil selection with an adaptive cutoff, and mirror ghost cells at boundaries
- Riemann solvers: Rusanov, HLL, HLLC, Roe, and the carbuncle-free rotated-hybrid HLL-Roe
- Source terms: gravity and arbitrary expressions
- Time integration: forward Euler, SSPRK3, RK4, with the time step set by a CFL number
- Boundary conditions: transmissive, symmetry, adiabatic, isothermal and heat-flux walls (optionally moving), inflow with fixed velocity, pressure and temperature, pressure outlet, partially non-reflecting characteristic (NSCBC) inlets and outlets with transverse terms, and time-dependent states given as expressions; zones can be split between conditions; periodic boundaries (generated meshes, or paired zones of mesh files); sponge layers
- Initial conditions given as analytical expressions, integrated over each cell
- Restart files
- Output to VTU (ParaView), with `.pvd` time series
- Simple TOML input files

The sources of every method and of the validation data are listed in [docs/references.md](docs/references.md).

## Building

Mallard depends on [Kokkos](https://github.com/kokkos/kokkos) (5.x), [toml11](https://github.com/ToruNiina/toml11) and [exprtk](https://github.com/ArashPartow/exprtk), all included in this repository; Kokkos and toml11 are submodules.

```sh
git clone --recursive https://github.com/MatthewBonanni/mallard.git
cd mallard
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_SYSTEM_KOKKOS=OFF -DKokkos_ENABLE_THREADS=ON
cmake --build build -j
```

Add `-DMallard_DIM=3` (in a separate build directory) for the 3D solver.

Pick the Kokkos backend at configure time, for example `-DKokkos_ENABLE_OPENMP=ON`, or `-DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON -DCMAKE_CXX_COMPILER=$PWD/src/external/kokkos/bin/nvcc_wrapper` for NVIDIA A100 GPUs (add `-DKokkos_ENABLE_OPENMP=ON` too, so host-side setup such as TENO's precomputation runs in parallel). To use an installed Kokkos instead, pass `-DUSE_SYSTEM_KOKKOS=ON -DKokkos_DIR=/path/to/kokkos`.

For AMD GPUs, build with ROCm's Clang (ROCm 6.2 or newer, with the HIP development headers):

```sh
cmake -S . -B build-hip -DCMAKE_BUILD_TYPE=Release -DUSE_SYSTEM_KOKKOS=OFF \
    -DKokkos_ENABLE_HIP=ON -DKokkos_ARCH_AMD_GFX950=ON -DKokkos_ENABLE_OPENMP=ON \
    -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++
```

`AMD_GFX950` is the MI350X/MI355X; use `AMD_GFX942` for the MI300X/MI300A or `AMD_GFX90A` for the MI250X. Instinct GPUs run 64-wide wavefronts where NVIDIA GPUs run 32-wide warps; the chemistry gives each cell a wavefront, and a team's results do not depend on its width. With MPI and dKaMinPar, also pass `-DCMAKE_POSITION_INDEPENDENT_CODE=ON`. On a node with automatic NUMA balancing on (`/proc/sys/kernel/numa_balancing` is 1), the ROCm driver stalls a process's GPU queues for up to seconds at a time: turn it off, as AMD recommends for Instinct GPUs, or bind each rank to its GPU's NUMA node (`numactl --cpunodebind=N --membind=N`).

| CMake option | Default | Description |
|---|---|---|
| `USE_SYSTEM_KOKKOS` | `ON` | Use an installed Kokkos instead of the submodule |
| `Mallard_DIM` | `2` | Spatial dimension, `2` or `3` (one binary per dimension) |
| `Mallard_USE_DOUBLE` | `ON` | Double precision (single precision otherwise) |
| `Mallard_ENABLE_MPI` | `OFF` | Distributed memory with MPI: `mpirun -n N Mallard -i input.toml` splits the mesh between ranks; with generated meshes or HDF5 mesh files (`mallard-mesh-convert`) no rank ever holds the whole mesh, while Gmsh files are read whole by every rank |
| `Mallard_GPU_AWARE_MPI` | `OFF` | With MPI on GPUs: hand device buffers to a CUDA- or ROCm-aware MPI (e.g. Open MPI over UCX built with CUDA or ROCm) instead of staging halos through host memory |
| `Mallard_ENABLE_KAMINPAR` | `OFF` | With MPI: partition the mesh with the [dKaMinPar](https://github.com/KaHIP/KaMinPar) graph partitioner (fetched at configure time; needs oneTBB) instead of a Hilbert curve. With CUDA, configure with the host compiler (`-DCMAKE_CXX_COMPILER=g++`) instead of `nvcc_wrapper`: Kokkos then compiles the code that uses it through `nvcc_wrapper` itself, and dKaMinPar does not compile with nvcc |
| `Mallard_ENABLE_HDF5` | `OFF` | HDF5 mesh files and solution output (`format = "hdf5"`, with XDMF for ParaView; parallel HDF5 with MPI, when available) and the `mallard-mesh-convert` tool |
| `Mallard_WARNINGS_AS_ERRORS` | `OFF` | Treat compiler warnings in Mallard's own code as errors (on in CI) |
| `BUILD_DOCS` | `OFF` | Doxygen documentation target |

## Running

```sh
cd examples/riemann_2d
../../build/src/Mallard -i input.toml --kokkos-num-threads=8
```

See [`examples/`](examples) for complete input files and [`docs/input.md`](docs/input.md) for every input option.

## Testing

```sh
./build/test/MallardTest
```

The test suite checks mesh geometry, the Riemann solvers against an exact Riemann solver, time integrator convergence orders, gradient and limiter properties, TENO design order on triangles and quadrilaterals and on 3D tetrahedra and hexahedra (with polynomial exactness on prisms, pyramids and mixed meshes), free-stream preservation, discrete conservation, symmetry preservation, shock tubes against exact solutions, viscous flows against exact solutions (Couette, Stokes' first problem, conduction), and bit-for-bit restarts.

## Postprocessing

`tools/` contains Python scripts (numpy, matplotlib, scipy, imageio) for reading Mallard's VTU files, plotting profiles against exact solutions, and rendering animations:

```sh
python tools/animate.py examples/riemann_2d/solut riemann
```

## Contributing

Mallard uses the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).

## Citing

If you use Mallard, please cite it ([doi:10.5281/zenodo.23112953](https://doi.org/10.5281/zenodo.23112953), or the DOI of the version you used on Zenodo) with the metadata in [CITATION.cff](CITATION.cff) (GitHub's "Cite this repository" button), and the papers behind the methods you use ([docs/references.md](docs/references.md)).

## License

Mallard is licensed under the AGPL v3.0 License. See the [LICENSE](LICENSE) file for more details.

---

<p align="center">
  <img src="docs/images/mallard.gif" alt="Mach 8 flow over a mallard" width="100%">
  <br>
  <em>Mach 8 flow over a mallard: 1.07M triangles, TENO5 + RHLL (<a href="examples/mallard">examples/mallard</a>).</em>
</p>
