![C++](https://img.shields.io/badge/C%2B%2B-20-blue)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

![logo_dark](./docs/images/mallard_dark.png#gh-dark-mode-only)
![logo_light](./docs/images/mallard_light.png#gh-light-mode-only)

Mallard is a high-order unstructured finite volume solver for the compressible Euler and Navier-Stokes equations, written in C++ with [Kokkos](https://github.com/kokkos/kokkos) for performance portability.

![2D Riemann problem](./docs/images/riemann_2d.gif)

*2D Riemann problem (configuration 3) on 320,000 triangles with fifth-order TENO-E reconstruction.*

> **NOTE:** Mallard is a **work in progress**. Its performance **has not been extensively characterized**, and it has not yet been run on GPUs.

## Features

- Compressible Euler and Navier-Stokes equations (calorically perfect gas, constant or Sutherland viscosity)
- Unstructured meshes of triangles and quadrilaterals
- Face reconstruction:
  - First order
  - Second-order MUSCL with least-squares gradients and Barth-Jespersen or Venkatakrishnan limiting
  - TENO-E of orders 2 to 6 ([Liang, Shyy & Fu, J. Sci. Comput. 2025](https://doi.org/10.1007/s10915-025-02918-w)): k-exact least squares on a large central stencil and three or four sector stencils, a density-based troubled-cell indicator, characteristic-wise stencil selection with an adaptive cutoff, and mirror ghost cells at boundaries
- Riemann solvers: Rusanov, HLL and HLLC (Einfeldt wave speeds)
- Time integration: forward Euler, SSPRK3, RK4, with the time step set by a CFL number
- Boundary conditions: transmissive, symmetry, adiabatic, isothermal and heat-flux walls (optionally moving), inflow with fixed velocity, pressure and temperature, and pressure outlet
- Initial conditions given as analytical expressions, integrated over each cell
- Restart files
- Output to VTU (ParaView), with `.pvd` time series
- Simple TOML input files

## Building

Mallard depends on [Kokkos](https://github.com/kokkos/kokkos) (5.x), [toml11](https://github.com/ToruNiina/toml11) and [exprtk](https://github.com/ArashPartow/exprtk), all included in this repository; Kokkos and toml11 are submodules.

```sh
git clone --recursive https://github.com/MatthewBonanni/Mallard.git
cd Mallard
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_SYSTEM_KOKKOS=OFF -DKokkos_ENABLE_THREADS=ON
cmake --build build -j
```

Pick the Kokkos backend at configure time, for example `-DKokkos_ENABLE_OPENMP=ON`, or `-DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON` for NVIDIA A100 GPUs. To use an installed Kokkos instead, pass `-DUSE_SYSTEM_KOKKOS=ON -DKokkos_DIR=/path/to/kokkos`.

| CMake option | Default | Description |
|---|---|---|
| `USE_SYSTEM_KOKKOS` | `ON` | Use an installed Kokkos instead of the submodule |
| `Mallard_USE_DOUBLE` | `ON` | Double precision (single precision otherwise) |
| `Mallard_ENABLE_HDF5` | `OFF` | Find or build HDF5 (not used by the solver yet) |
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

The test suite checks mesh geometry, the Riemann solvers against an exact Riemann solver, time integrator convergence orders, gradient and limiter properties, TENO design order on triangles and quadrilaterals, free-stream preservation, discrete conservation, symmetry preservation, shock tubes against exact solutions, viscous flows against exact solutions (Couette, Stokes' first problem, conduction), and bit-for-bit restarts.

## Postprocessing

`tools/` contains Python scripts (numpy, matplotlib, scipy, imageio) for reading Mallard's VTU files, plotting profiles against exact solutions, and rendering animations:

```sh
python tools/animate.py examples/riemann_2d/solut riemann
```

## Contributing

Mallard uses the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).

## License

Mallard is licensed under the AGPL v3.0 License. See the [LICENSE](LICENSE) file for more details.
