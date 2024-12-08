![C++](https://img.shields.io/badge/C%2B%2B-17-blue)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

![logo_dark](./docs/images/mallard_dark.png#gh-dark-mode-only)
![logo_light](./docs/images/mallard_light.png#gh-light-mode-only)

Mallard is a high-order unstructured finite volume solver for the compressible Navier-Stokes equations, written in C++.

> **NOTE:** Mallard is a **work in progress** and many features are incomplete or **not validated**. Its performance **has not been extensively characterized**.

## Features (IN PROGRESS)

- Solves the fully compressible Navier-Stokes equations
- Supports unstructured meshes
- Performance portability and GPU support via Kokkos and MPI
- High-order TENO reconstruction
- Simple TOML-based input file format

## Installation

### Kokkos

Mallard depends on the Kokkos Ecosystem, specifically:
- [kokkos](https://github.com/kokkos/kokkos)
- [kokkos-kernels](https://github.com/kokkos/kokkos-kernels)

Both of these are included in this repository as submodules.
You may choose to use these in-tree submodules, or use an existing Kokkos and/or KokkosKernels installation.
All other prerequisites will be installed automatically.

### Building

Mallard uses CMake for building. Follow the steps below to install Mallard:

1. Clone the repository and set up its submodules:
    ```sh
    git clone https://github.com/MatthewBonanni/Mallard.git
    cd Mallard
    git submodule init
    ```

2. Create a build directory and navigate to it:
    ```sh
    mkdir build
    cd build
    ```

3. Configure, build, and install Mallard:
   1. If using in-tree Kokkos and KokkosKernels (be sure to enable the desired backends):
      ```sh
      ccmake .. -DCMAKE_INSTALL_PREFIX=/path/to/Mallard/build -DUSE_SYSTEM_KOKKOS=OFF
      make -j install
      ```
   2. If using existing Kokkos and KokkosKernels installations:
      ```sh
      ccmake .. -DCMAKE_INSTALL_PREFIX=/path/to/Mallard/build -DUSE_SYSTEM_KOKKOS=ON -DKokkos_DIR=/path/to/kokkos
      make -j install
      ```

## Contributing

Mallard uses the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)

## License

Mallard is licensed under the AGPL v3.0 License. See the [LICENSE](LICENSE) file for more details.
