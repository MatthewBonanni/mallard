![C++](https://img.shields.io/badge/C%2B%2B-17-blue)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

<p align="center">
    <picture>
      <source media="(prefers-color-scheme: dark)" width="800" srcset="docs/images/mallard.png">
      <source media="(prefers-color-scheme: light)" width="800" srcset="docs/images/mallard.png">
      <img alt="mallard logotype" width="800" src="docs/images/mallard.png">
    </picture>
</p>

Mallard is a high-order unstructured finite volume solver for the compressible Navier-Stokes equations.

## Features

- Solves the fully compressible Navier-Stokes equations
- Supports unstructured meshes
- Performance portability on heterogeneous computing architectures, including GPUs, via Kokkos and MPI
- High-order TENO reconstruction
- Simple TOML-based input file format

## Installation

### Prerequisites

Mallard depends on the Kokkos Ecosystem, specifically:
- [kokkos](https://github.com/kokkos/kokkos)
- [kokkos-kernels](https://github.com/kokkos/kokkos-kernels)
Follow the instructions for each of these, compiling with the desired backends enabled.
All other prerequisites will be installed automatically.

### Building

Mallard uses CMake for building. Follow the steps below to install Mallard:

1. Clone the repository:
    ```sh
    git clone https://github.com/MatthewBonanni/Mallard.git
    cd Mallard
    ```

2. Create a build directory and navigate to it:
    ```sh
    mkdir build
    cd build
    ```

3. Configure the project using CMake:
    ```sh
    cmake ..
    ```

4. Build the project:
    ```sh
    make
    ```

5. (Optional) Install the project:
    ```sh
    make install
    ```

## Contributing

Mallard uses the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)

## License

Mallard is licensed under the GPL v3.0 License. See the [LICENSE](LICENSE) file for more details.
