# Supported platforms

What Mallard's CI builds and runs, and how often. **Run** means the test suite (or
the named checks) runs on that build; **compile** means every source is built with
warnings as errors, but nothing runs; anything not listed is untested. Which
features are tested together is in the [feature matrix](features.md). Every
GitHub-hosted job runs on x86-64 Ubuntu 26.04 with its GCC 15 and CMake 4.4 unless the
table says otherwise. [Installation](../installation.md#supported-platforms) summarizes
this page by platform, with the support policy.

When a job runs:

- **every PR**: the gate, the CI run a pull request needs before it merges (`ci.yml`), and every push to `main`;
- **main**: on `main` every 3 hours when it has changed since its last full run, and on a branch with `-f full=true`; a failure opens an issue labeled `ci-main`;
- **device PRs**: pull requests whose changes touch device code ([`device_changes.sh`](../../tools/dev/device_changes.sh)), before they merge;
- **nightly**, **release** (every version tag, and by hand on a branch) and **by hand** (before a release).

A job *split across* N jobs builds once, in `build-<job>`, and runs as `<job>-1of<N>` to
`<job>-<N>of<N>`, each running a share of the job's tests of about equal duration on that build.
A suite without MPI runs as several processes at once, one per core (Serial) or per two
cores (OpenMP, Threads), each running a share of the job's tests. The shares follow the test
durations of `main`'s newest passed run, or [`test_times.json`](../../tools/dev/test_times.json)
([`test_split.py`](../../tools/dev/test_split.py)).

## Builds

| Build | Hardware | Tested | What runs | When | Job |
|---|---|---|---|---|---|
| OpenMP, 2D, HDF5, CGNS, zlib | x86-64 CPU | Run | Full suite (2 threads); upgrading every release's inputs; output units through h5py; the Python package's tests | every PR | `ci.yml/openmp` |
| OpenMP, single precision, 2D, zlib | x86-64 CPU | Run | Full suite, two copies at once | every PR | `ci.yml/float` |
| OpenMP, single precision, 3D, HDF5, CGNS | x86-64 CPU | Run | Full suite | every PR | `ci.yml/float-3d` |
| Serial, MPI, 2D, dKaMinPar, HDF5, zlib | x86-64 CPU | Run | Full suite on 3 ranks; MPI tests on 1, 2 and 4 ranks; rank report on 2 and 3; split across 2 jobs | every PR | `ci.yml/mpi` |
| Serial, MPI, 3D, HDF5 | x86-64 CPU | Run | Full suite on 3 ranks, and on 1 rank on main (PRs: the MPI tests on 1 rank); MPI tests on 2 and 4 ranks; split across 4 jobs | every PR | `ci.yml/mpi-3d` |
| Serial, MPI, HDF5, 2D and 3D, installed | x86-64 CPU | Run | Every [tutorial](../tutorials/index.md) at reduced size on 2 ranks, with its Gmsh mesh and scripts; every case template; the installed bash, zsh and fish completion | main | `ci.yml/tutorials` |
| Serial, 2D | x86-64 CPU | Run | Full suite | main | `ci.yml/serial` |
| Serial, 3D, HDF5 | x86-64 CPU | Run | Full suite | main | `ci.yml/serial-3d` |
| OpenMP, 2D, GCC 16 (Docker's `gcc:16.2.0`, Debian 13), CMake 4.2.3 | x86-64 CPU | Run | Full suite (2 threads) | every PR | `ci.yml/gcc-16` |
| Serial, single precision, 2D, HDF5, Clang 22 (Ubuntu's) | x86-64 CPU | Run | Full suite | main | `ci.yml/clang-22` |
| Serial, 3D, Clang 23 (apt.llvm.org) | x86-64 CPU | Run | Full suite | every PR | `ci.yml/clang-23` |
| OpenMP, 2D, HDF5, CGNS, zlib, GCC 15 | arm64 CPU (Ubuntu 26.04) | Run | Full suite (2 threads) | main | `ci.yml/gcc-15-arm64` |
| Serial, MPI, 2D, dKaMinPar, HDF5, zlib, GCC 15 | arm64 CPU (Ubuntu 26.04) | Run | Full suite on 3 ranks; MPI tests on 1, 2 and 4 ranks | main | `ci.yml/mpi-arm64` |
| OpenMP, single precision, 3D, HDF5, CGNS, GCC 16 (Docker's `gcc:16.2.0`), CMake 4.2.3 | arm64 CPU | Run | Full suite (2 threads) | main | `ci.yml/gcc-16-arm64` |
| OpenMP, single precision, 2D, zlib, Clang 22 (Ubuntu's) | arm64 CPU (Ubuntu 26.04) | Run | Full suite, two copies at once | main | `ci.yml/clang-22-arm64` |
| Serial, 3D, HDF5, Clang 23 (apt.llvm.org) | arm64 CPU (Ubuntu 26.04) | Run | Full suite | main | `ci.yml/clang-23-arm64` |
| Threads, 2D, Apple Clang 21 (Xcode 26) | arm64 CPU (macOS 26) | Run | Full suite (2 threads) | main | `ci.yml/macos` |
| Threads, 3D, Apple Clang 21 (Xcode 26) | arm64 CPU (macOS 26) | Run | Full suite (2 threads) | main | `ci.yml/macos-3d` |
| Threads, 2D, GCC 16 (Homebrew's) | arm64 CPU (macOS 26) | Run | Full suite (2 threads) | main | `ci.yml/macos-gcc` |
| Spack `mallard@main %gcc@15`: OpenMP, MPI, HDF5, CGNS, zlib, 2D and 3D, with Spack's GCC 15, Open MPI and HDF5 | x86-64 CPU (Red Hat UBI 8) | Run | Full 2D and 3D suites; `spack test run` | main | `ci.yml/ubi8-spack` |
| Same, on RHEL 9 | x86-64 CPU (Red Hat UBI 9) | Run | Full 2D and 3D suites; `spack test run` | main | `ci.yml/ubi9-spack` |
| CUDA 12.9 (sm_80), OpenMP host, MPI, GPU-aware MPI, NCCL, parallel HDF5, CGNS, 2D and 3D | NVIDIA A100 | Run | Full suite on one GPU (with CUDA graphs, the default); MPI tests on 2 ranks, one GPU each, GPU-aware, with NCCL's and MPI's halo exchanges, each compared bit for bit with a serial run; [performance suite](../../benchmarks/README.md) against its baselines | device PRs (GPU CI) | |
| CUDA 12.9 (sm_80), single precision, same options, 2D | NVIDIA A100 | Run | Full suite on one GPU | device PRs (GPU CI) | |
| CUDA 13.2 (sm_80), same options, 2D and 3D | NVIDIA A100 | Run | Full suites on one GPU, except the tests that signal runs through `mpiexec` (the image's MPI is Open MPI 5) | device PRs (GPU CI) | |
| CUDA 13.5 (sm_80), OpenMP host, MPI, GPU-aware MPI, 2D and 3D | | Compile | | main, device PRs | `ci.yml/cuda-13.5` |
| CUDA 13.4 (sm_80), same options | | Compile | | main | `ci.yml/cuda-13.4` |
| CUDA 13.5 (sm_90), same options | arm64 (Grace) | Compile | | main | `ci.yml/cuda-13.5-arm64` |
| HIP (ROCm 10.1, gfx90a), MPI, GPU-aware MPI, 2D and 3D | | Compile | Never run on an AMD GPU | device PRs | `ci.yml/hip` |
| SYCL (oneAPI 2026.2, Intel GEN), MPI, GPU-aware MPI, 2D and 3D | | Compile | Never run on an Intel GPU | device PRs | `ci.yml/sycl` |
| Serial, MPI, HDF5, 2D, AddressSanitizer and UBSan | x86-64 CPU | Run | Full suite on 1 rank; MPI tests on 2 ranks | nightly | `sanitizers.yml/asan-ubsan` |
| Serial, HDF5, CGNS, Clang, libFuzzer, ASan and UBSan | x86-64 CPU | Run | Input and mesh reader harnesses: a minute each in 2D (every PR); ten minutes each in 2D and 3D (nightly) | main, nightly | `ci.yml/fuzz`, `fuzz.yml/fuzz` |
| Serial, 2D and 3D | x86-64 CPU | Run | [Convergence studies](../verification/index.md) | by hand | `verification.yml/verification` |

## Containers and Spack

| Package | Hardware | Tested | What runs | When | Job |
|---|---|---|---|---|---|
| CPU image: OpenMP, MPI, dKaMinPar, HDF5, CGNS, zlib, 2D and 3D | x86-64 and arm64 CPU | Run | Examples compared with reference solutions (`docker/smoke`); also through Apptainer on x86-64 | release | `release.yml/cpu` |
| CUDA image: sm_80, 86, 89 and 90, OpenMP host, MPI, NCCL, HDF5, CGNS, zlib, 2D and 3D | | Compile | Libraries resolve and the executables start (no GPU) | release | `release.yml/cuda` |
| Spack `mallard@main`: OpenMP, MPI, HDF5, CGNS, zlib, 2D and 3D, with Ubuntu's GCC, CMake and Open MPI | x86-64 CPU | Run | `spack test run` and the container smoke test | release | `release.yml/spack` |
| Spack `+cuda`, `+rocm`, `+cuda +nccl`, `~mpi precision=single` | | Concretized only | | release | `release.yml/spack` |

## Toolchains

| | Tested with |
|---|---|
| Kokkos | 5.2.2, the submodule; Spack builds use the Kokkos 5 of Spack's package repository |
| C++ compilers | GCC 15.2 (Ubuntu 26.04's: most Linux jobs, the containers, the host compiler of nvcc); GCC 15 built by Spack (`ubi8-spack`, `ubi9-spack`); GCC 16.2 (`gcc-16`, `gcc-16-arm64`; Homebrew's on macOS, `macos-gcc`); Clang 22 (Ubuntu 26.04's: `clang-22`, `clang-22-arm64`, fuzzing); Clang 23 (apt.llvm.org's: `clang-23`, `clang-23-arm64`); Apple Clang 21 (Xcode 26: `macos`, `macos-3d`) |
| CMake | 4.2.3, the oldest supported (`gcc-16`, `gcc-16-arm64`; Ubuntu 26.04's, in the CUDA, HIP and SYCL builds and the containers); 4.4 (the runners'); Spack's in the UBI jobs |
| Device compilers | nvcc 13.4 and 13.5 through Kokkos' `nvcc_wrapper` (12.9 and 13.2 in GPU CI); ROCm 10.1 `hipcc`; oneAPI 2026.2 `icpx` |
| MPI | Open MPI 5.0 (Ubuntu 26.04's): CPU jobs, containers, release Spack job; Open MPI built by Spack: `ubi8-spack`, `ubi9-spack`; HPC-X's Open MPI (NVIDIA HPC SDK), CUDA-aware: GPU CI |
| HDF5, CGNS | Ubuntu 26.04's HDF5 1.14 (serial and parallel) and CGNS; Debian 13's in `gcc-16-arm64`; Spack's in the UBI jobs; CGNS 4.5.0 built by Mallard (`USE_SYSTEM_CGNS=OFF`) in the containers and GPU CI, there with a parallel HDF5 1.14.6 built with HPC-X's Open MPI |
| Precision | Double everywhere; single in `float`, `float-3d`, `clang-22`, `gcc-16-arm64`, `clang-22-arm64` and GPU CI |

## Not tested

- **The Kokkos Threads backend on Linux**, and MPI, HDF5 or CGNS on macOS.
- **GPUs other than the A100**, and AMD or Intel GPUs at all.
- **More than 4 MPI ranks**, or more than one node.
- **Other MPI implementations** (MPICH, Intel MPI, Cray MPICH, MVAPICH, HPE MPI, IBM Spectrum MPI), compilers other than those above (including `nvc++`), an installed Kokkos outside Spack, and Debug builds.
- **macOS on x86-64**, Linux distributions other than Ubuntu 26.04, Debian 13 and RHEL 8 and 9 (as UBI containers), and real TOSS or Cray systems.

Some MPI distributions start the real launcher from a wrapper that passes no signals on, so
signals sent to `mpirun` may not reach the run: see [MPI launchers and Slurm](running.md#mpi-launchers-and-slurm).

`python3 tools/dev/check_support_matrix.py` (run by CI) fails when a workflow gains or loses a
job without this page naming it in its *Job* column, or the
[supported-platforms table](../installation.md#supported-platforms) naming it in its row.
