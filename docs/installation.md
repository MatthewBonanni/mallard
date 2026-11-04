# Installation

Three ways to install Mallard, quickest first:

| | Best for | Compiling |
|---|---|---|
| [Container](#container) (Docker, Apptainer, Kubernetes) | Trying Mallard; workstations; GPU nodes; Kubernetes clusters | None |
| [Spack](#spack) | Clusters: your compilers, MPI and GPUs | Automatic |
| [From source](#from-source) | Development; options the others lack | CMake |

Each installs the solvers `mallard2d` and `mallard3d`, the `mallard` command that runs whichever a case needs (by its mesh: a generated mesh is 3D when `[mesh]` sets `Nz` or `Lz`; `mallard --dim 2|3` states it), and, under `share/mallard`, the `examples`, `mechanisms`, `tools` and `tutorials`. One command does everything: `mallard -i input.toml` runs a case, `mallard check`, `new`, `ctl` and `input upgrade` help set it up and run it, `mallard reactor` integrates 0D reactors and `mallard mesh convert` writes meshes to HDF5; `mallard --help` lists them by purpose and `mallard <command> --help` explains each ([command line](guide/cli.md)).

## Container

Each release, from 0.9.1 on, publishes images to the GitHub Container Registry:

| Image | For |
|---|---|
| `ghcr.io/matthewbonanni/mallard` (`:latest`, or a version: `:0.9.1`) | CPUs: OpenMP, MPI with dKaMinPar; x86-64 and ARM (also Apple Silicon) |
| `ghcr.io/matthewbonanni/mallard:cuda` (or `:0.9.1-cuda`) | NVIDIA GPUs: A100, A40 and RTX 30xx, L40S and RTX 40xx, H100 (driver R525 or newer); MPI with NCCL |

Both have HDF5, CGNS, zlib and the Python tools, and set `MALLARD_SHARE` to the examples' directory.

With Docker, run the Sod shock tube in the current directory:

```sh
docker run --rm -u "$(id -u):$(id -g)" -v "$PWD:/work" ghcr.io/matthewbonanni/mallard \
    bash -c 'cp -r $MALLARD_SHARE/examples $MALLARD_SHARE/mechanisms . && cd examples/sod && mallard -i input.toml'
```

and then any case, e.g. on 4 MPI ranks (`-it ... bash` for a shell in the container):

```sh
docker run --rm -u "$(id -u):$(id -g)" -v "$PWD:/work" -w /work/examples/riemann_2d ghcr.io/matthewbonanni/mallard \
    mpirun -n 4 mallard -i input.toml
```

On an NVIDIA GPU (with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)), add `--gpus all` and use the `cuda` image: `docker run --rm --gpus all ... ghcr.io/matthewbonanni/mallard:cuda ...`.

With Apptainer (or Singularity), which runs as you and mounts the current directory, on clusters too:

```sh
apptainer pull mallard.sif docker://ghcr.io/matthewbonanni/mallard
apptainer exec mallard.sif bash -c 'cp -r $MALLARD_SHARE/examples $MALLARD_SHARE/mechanisms .'
cd examples/sod && apptainer exec ../../mallard.sif mallard -i input.toml
```

For a GPU, pull `docker://ghcr.io/matthewbonanni/mallard:cuda` and add `--nv`: `apptainer exec --nv mallard-cuda.sif mallard -i input.toml`.

On Kubernetes, a `Job` runs a case on one GPU node with its files on a volume, and resumes from a restart file after preemption; an `MPIJob` ([Kubeflow MPI Operator](https://github.com/kubeflow/mpi-operator)) runs it across nodes, with the images' Open MPI in the launcher and the workers ([manifests and guide](guide/running.md#kubernetes)). On clusters with their own MPI and no Kubernetes, run across nodes with Spack or a source build.

## Spack

[Spack](https://spack.io) (1.0 or newer) builds Mallard and its dependencies for your machine, with the package in this repository:

```sh
git clone --depth 1 https://github.com/spack/spack.git ~/spack && . ~/spack/share/spack/setup-env.sh
spack repo add --path spack/spack_repo/mallard https://github.com/MatthewBonanni/mallard.git
spack install mallard
spack load mallard
```

`spack install mallard` builds the latest release, `mallard@0.9.0` an earlier one (0.9.0 or newer), and `mallard@main` the development version. `spack repo add` also takes a local checkout's package repository, `spack repo add mallard/spack/spack_repo/mallard`; `spack repo update mallard` pulls a newer package, with newer releases. The default is OpenMP (C++ threads on macOS), MPI, HDF5, CGNS and zlib, in 2D and 3D, double precision. Other builds name their variants:

```sh
spack install mallard +cuda cuda_arch=80                 # NVIDIA A100 (90: H100)
spack install mallard +cuda cuda_arch=90 +nccl           # H100, halo exchanges with NCCL
spack install mallard +rocm amdgpu_target=gfx90a         # AMD MI250X (gfx942: MI300X)
spack install mallard ~mpi dims=3 precision=single       # one workstation, 3D only, single precision
spack install mallard ^openmpi@5                         # a particular MPI
```

| Variant | Default | Description |
|---|---|---|
| `dims` | `2,3` | Dimensions to build (`dims=3`: the 3D solver only) |
| `precision` | `double` | `double` or `single` |
| `openmp`, `threads`, `serial` | `+openmp` (macOS: `+threads`) | Kokkos host backend |
| `cuda` + `cuda_arch`, `rocm` + `amdgpu_target`, `sycl` | off | Kokkos GPU backend |
| `mpi`, `gpu_aware_mpi`, `nccl` | `+mpi` | MPI; device buffers to a GPU-aware MPI; halo exchanges with NCCL |
| `hdf5`, `cgns`, `zlib` | on | HDF5 meshes and output; CGNS meshes; zlib compression |

Spack builds its own Kokkos to match. A cluster's MPI, CUDA or compilers can be registered with `spack external find` (e.g. `spack external find openmpi cuda`). dKaMinPar is not in the package: build from source for `partitioner = "graph"`.

## From source

Mallard needs a C++20 compiler (GCC 10.4, Clang 14 or newer), CMake 3.22 or newer, and git. It depends on [Kokkos](https://github.com/kokkos/kokkos) (5.x) and [toml11](https://github.com/ToruNiina/toml11), both included in this repository as submodules; CMake fetches yaml-cpp.

```sh
git clone --recursive https://github.com/MatthewBonanni/mallard.git
cd mallard
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_SYSTEM_KOKKOS=OFF -DKokkos_ENABLE_THREADS=ON
cmake --build build -j
```

This builds the 2D solver, `build/src/mallard2d`, and `build/src/mallard`, which runs it. Add `-DMallard_DIM=3` (in a separate build directory) for the 3D solver, `mallard3d`; installed to the same prefix, one `mallard` runs both.

With everything (MPI with dKaMinPar, HDF5, CGNS and zlib), on Ubuntu 24.04, in 2D and 3D, installed into `~/mallard` (the CPU container is built this way):

```sh
sudo apt-get install -y build-essential cmake git libopenmpi-dev openmpi-bin libhdf5-openmpi-dev zlib1g-dev libtbb-dev
for dim in 2 3; do
    cmake -S . -B build-$dim -DCMAKE_BUILD_TYPE=Release -DUSE_SYSTEM_KOKKOS=OFF -DKokkos_ENABLE_OPENMP=ON \
        -DMallard_ENABLE_MPI=ON -DMallard_ENABLE_KAMINPAR=ON -DMallard_ENABLE_HDF5=ON \
        -DMallard_ENABLE_CGNS=ON -DUSE_SYSTEM_CGNS=OFF -DMallard_ENABLE_ZLIB=ON \
        -DMallard_DIM=$dim -DCMAKE_INSTALL_PREFIX=$HOME/mallard
    cmake --build build-$dim -j && cmake --install build-$dim
done
export PATH=$HOME/mallard/bin:$PATH MALLARD_SHARE=$HOME/mallard/share/mallard
```

On macOS with Homebrew, `brew install cmake open-mpi hdf5 cgns` and the same loop with `-DKokkos_ENABLE_THREADS=ON` in place of OpenMP, without dKaMinPar, and `-DUSE_SYSTEM_CGNS=ON`. `mallard test` runs the test suites of a build or an install (`mallard-test2d` and `mallard-test3d`, `--dim 2|3` for one), e.g. `mallard test --gtest_filter='Sod*'`.

Pick the Kokkos backend at configure time, for example `-DKokkos_ENABLE_OPENMP=ON`, or `-DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON -DCMAKE_CXX_COMPILER=$PWD/src/external/kokkos/bin/nvcc_wrapper` for NVIDIA A100 GPUs (add `-DKokkos_ENABLE_OPENMP=ON` too, so host-side setup such as TENO's precomputation runs in parallel; with MPI, `-DMallard_ENABLE_NCCL=ON` for halo exchanges with NCCL). The CUDA container is built this way, for several GPUs at once (`docker/Dockerfile.cuda`). To use an installed Kokkos instead, pass `-DUSE_SYSTEM_KOKKOS=ON -DKokkos_DIR=/path/to/kokkos`.

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
| `Mallard_DIM` | `2` | Spatial dimension, `2` or `3`: the solver `mallard2d` or `mallard3d` (one per build; `mallard` runs either) |
| `Mallard_USE_DOUBLE` | `ON` | Double precision (single precision otherwise) |
| `Mallard_ENABLE_MPI` | `OFF` | Distributed memory with MPI: `mpirun -n N mallard -i input.toml` splits the mesh between ranks; with generated meshes or HDF5 mesh files (`mallard mesh convert`) no rank ever holds the whole mesh, while other mesh files are read whole by every rank |
| `Mallard_GPU_AWARE_MPI` | `OFF` | With MPI on GPUs: hand device buffers to a CUDA- or ROCm-aware MPI (e.g. Open MPI over UCX built with CUDA or ROCm) instead of staging halos through host memory |
| `Mallard_ENABLE_NCCL` | `OFF` | With MPI on NVIDIA GPUs: exchange halos with NCCL, stream-ordered with the kernels, so the host never waits for a halo (finds NCCL through `NCCL_HOME` or `NCCL_ROOT`); `[parallel] halo_exchange` chooses at run time |
| `Mallard_ENABLE_KAMINPAR` | `OFF` | With MPI: the [dKaMinPar](https://github.com/KaHIP/KaMinPar) graph partitioner for `[parallel] partitioner = "graph"` (fetched at configure time; needs [oneTBB](https://github.com/uxlfoundation/oneTBB), found through `CMAKE_PREFIX_PATH`, e.g. built with `cmake -S oneTBB -B build -DTBB_TEST=OFF -DCMAKE_INSTALL_PREFIX=$HOME/tbb && cmake --build build -j && cmake --install build`). The default multi-jagged partitions need no library and cut box meshes at least as well. With CUDA, configure with the host compiler (`-DCMAKE_CXX_COMPILER=g++`) instead of `nvcc_wrapper`: Kokkos then compiles the code that uses it through `nvcc_wrapper` itself, and dKaMinPar does not compile with nvcc |
| `Mallard_ENABLE_HDF5` | `OFF` | HDF5 mesh files, `mallard mesh convert`, and the HDF5 outputs: `[[write_data]]` `format = "vtkhdf"` (the default) and `"xdmf"`, and `[[probes]]` `"hdf5"` and `"vtkhdf"` (parallel HDF5 with MPI, when available) |
| `Mallard_ENABLE_ZLIB` | `OFF` | zlib compression of VTU and restart files (`[[write_data]] compression = "zlib"`; HDF5 output compresses with HDF5's own deflate filter and needs no option) |
| `Mallard_ENABLE_CGNS` | `OFF` | CGNS mesh files, through the [CGNS library](https://cgns.github.io/) (an installed one, found through `CMAKE_PREFIX_PATH`; with `-DUSE_SYSTEM_CGNS=OFF`, v4.5.0 is fetched and built with the HDF5 of `Mallard_ENABLE_HDF5`) |
| `Mallard_ENABLE_TESTS` | `ON` | Build the test suite, `mallard-test2d` or `mallard-test3d` (fetches GoogleTest) |
| `Mallard_INSTALL_TESTS` | `ON` | Install the test suite and its data, so that `mallard test` checks an installed build |
| `USE_SYSTEM_YAML_CPP` | `OFF` | Use an installed yaml-cpp (0.8 or newer) instead of fetching it |
| `Mallard_WARNINGS_AS_ERRORS` | `OFF` | Treat compiler warnings in Mallard's own code as errors (on in CI) |
| `BUILD_DOCS` | `OFF` | Doxygen documentation target |
