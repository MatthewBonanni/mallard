# Supported platforms

What Mallard's CI builds and runs, and how often. **Run** means the test suite (or
the named checks) runs on that build; **compile** means every source is built with
warnings as errors, but nothing runs; anything not listed is untested. Every
GitHub-hosted job runs on x86-64 Ubuntu 24.04 unless the table says otherwise.

When a job runs:

- **every PR**: the CI run a pull request needs before it merges (`ci.yml`), and every push to `main`;
- **main**: every push to `main` only;
- **device PRs**: pull requests whose changes touch device code ([`device_changes.sh`](../../tools/dev/device_changes.sh)), before they merge;
- **nightly**, **release** (every version tag, and by hand on a branch) and **by hand** (before a release).

## Builds

| Build | Hardware | Tested | What runs | When | Job |
|---|---|---|---|---|---|
| OpenMP, 2D, HDF5, CGNS, zlib | x86-64 CPU | Run | Full suite (2 threads); upgrading every release's inputs; output units through h5py | every PR | `ci.yml/openmp` |
| OpenMP, single precision, 2D, zlib | x86-64 CPU | Run | Full suite, two copies at once | every PR | `ci.yml/float` |
| OpenMP, single precision, 3D, HDF5, CGNS | x86-64 CPU | Run | Full suite | every PR | `ci.yml/float-3d` |
| Serial, MPI, 2D, dKaMinPar, HDF5, zlib | x86-64 CPU | Run | Full suite on 3 ranks; MPI tests on 1, 2 and 4 ranks; rank report on 2 and 3 | every PR | `ci.yml/mpi` |
| Serial, MPI, 3D, HDF5 | x86-64 CPU | Run | Full suite on 1 and 3 ranks; MPI tests on 2 and 4 ranks | every PR | `ci.yml/mpi-3d` |
| Serial, MPI, HDF5, 2D and 3D, installed | x86-64 CPU | Run | Every [tutorial](../tutorials/index.md) at reduced size on 2 ranks, with its Gmsh mesh and scripts; every case template; the installed bash, zsh and fish completion | every PR | `ci.yml/tutorials` |
| Serial, 2D | x86-64 CPU | Run | Full suite | main | `ci.yml/serial` |
| Serial, 3D, HDF5 | x86-64 CPU | Run | Full suite | main | `ci.yml/serial-3d` |
| Serial, single precision, 2D, HDF5, Clang | | Compile | | every PR | `ci.yml/clang` |
| CUDA 12.9 (sm_80), OpenMP host, MPI, GPU-aware MPI, 2D and 3D | NVIDIA A100 | Run | Full suite on one GPU (with CUDA graphs, the default); MPI tests on 2 ranks, one GPU each, GPU-aware; [performance suite](../../benchmarks/README.md) against its baselines | device PRs (GPU CI) | |
| CUDA 12.9 (sm_80), OpenMP host, MPI, GPU-aware MPI, 2D and 3D | | Compile | | main, device PRs | `ci.yml/cuda` |
| CUDA 13.2 (sm_80), same options | | Compile | | device PRs (GPU CI) | |
| HIP (ROCm 7.2, gfx90a), MPI, GPU-aware MPI, 2D and 3D | | Compile | Never run on an AMD GPU | device PRs | `ci.yml/hip` |
| SYCL (oneAPI 2025.3, Intel GEN), MPI, GPU-aware MPI, 2D and 3D | | Compile | Never run on an Intel GPU | device PRs | `ci.yml/sycl` |
| Serial, MPI, HDF5, 2D, AddressSanitizer and UBSan | x86-64 CPU | Run | Full suite on 1 rank; MPI tests on 2 ranks | nightly | `sanitizers.yml/asan-ubsan` |
| Serial, HDF5, CGNS, Clang, libFuzzer, ASan and UBSan | x86-64 CPU | Run | Input and mesh reader harnesses: a minute each in 2D (every PR); ten minutes each in 2D and 3D (nightly) | every PR, nightly | `ci.yml/fuzz`, `fuzz.yml/fuzz` |
| Serial, 2D and 3D | x86-64 CPU | Run | [Convergence studies](../verification/index.md) | by hand | `verification.yml/verification` |

## Containers and Spack

| Package | Hardware | Tested | What runs | When | Job |
|---|---|---|---|---|---|
| CPU image: OpenMP, MPI, dKaMinPar, HDF5, CGNS, zlib, 2D and 3D | x86-64 and arm64 CPU | Run | Examples compared with reference solutions (`docker/smoke`); also through Apptainer on x86-64 | release | `release.yml/cpu` |
| CUDA image: sm_80, 86, 89 and 90, OpenMP host, MPI, NCCL, HDF5, CGNS, zlib, 2D and 3D | | Compile | Libraries resolve and the executables start (no GPU) | release | `release.yml/cuda` |
| Spack `mallard@main`: OpenMP, MPI, HDF5, CGNS, zlib, 2D and 3D | x86-64 CPU | Run | `spack test run` and the container smoke test | release | `release.yml/spack` |
| Spack `+cuda`, `+rocm`, `+cuda +nccl`, `~mpi precision=single` | | Concretized only | | release | `release.yml/spack` |

## Toolchains

| | Tested with |
|---|---|
| Kokkos | 5.2.2, the submodule; Spack builds use the Kokkos 5 of Spack's package repository |
| C++ compilers | GCC 13 (every CPU job, the containers, Spack, and the host compiler of nvcc); Clang 18 (`clang`, fuzzing) |
| Device compilers | nvcc 12.9 and 13.2 through Kokkos' `nvcc_wrapper`; ROCm 7.2 `hipcc`; oneAPI 2025.3 `icpx` |
| MPI | Open MPI 4.1 (Ubuntu 24.04's): CPU jobs, containers, Spack; HPC-X's Open MPI 4.1 (NVIDIA HPC SDK 25.7), CUDA-aware: GPU CI |
| HDF5, CGNS | Ubuntu 24.04's HDF5 1.10 (serial and parallel) and CGNS 4.4; CGNS 4.5.0 built by Mallard (`USE_SYSTEM_CGNS=OFF`) in the containers |
| Precision | Double everywhere; single in `float`, `float-3d` (run) and `clang` (compile) |

## Not tested

- **macOS** and the **Kokkos Threads backend**, which macOS builds use: no CI job builds either.
- **Linux on arm64**, beyond the CPU container's smoke test.
- **NCCL halo exchanges** (`Mallard_ENABLE_NCCL`): compiled in the CUDA image only, never run.
- **GPUs other than the A100**, single precision or HDF5 and CGNS on GPUs, and AMD or Intel GPUs at all.
- **More than 4 MPI ranks**, or more than one node.
- **Other MPI implementations** (MPICH, Intel MPI, Cray MPICH), compilers other than those above (including the oldest supported, GCC 10.4 and Clang 14, and `nvc++`), an installed Kokkos outside Spack, and Debug builds.

Some MPI distributions start the real launcher from a wrapper that passes no signals on, so
signals sent to `mpirun` may not reach the run: see [MPI launchers and Slurm](running.md#mpi-launchers-and-slurm).

`python3 tools/dev/check_support_matrix.py` (run by CI) fails when a workflow gains or loses a
job without this page naming it in its *Job* column.
