# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import os
import shutil
import sys

from spack_repo.builtin.build_systems import cmake
from spack_repo.builtin.build_systems.cmake import CMakePackage
from spack_repo.builtin.build_systems.cuda import CudaPackage
from spack_repo.builtin.build_systems.rocm import ROCmPackage

from spack.package import *


IS_DARWIN = sys.platform == "darwin"


def _submodules(pkg):
    # Kokkos and HDF5 come from Spack; only the header-only toml11 is vendored
    return ["src/external/toml11"]


class Mallard(CMakePackage, CudaPackage, ROCmPackage):
    """Mallard is a high-order finite volume solver for compressible, reacting
    flows on unstructured 2D and 3D meshes, running on CPUs and GPUs through
    Kokkos."""

    homepage = "https://matthewbonanni.github.io/mallard/"
    git = "https://github.com/MatthewBonanni/mallard.git"

    maintainers("MatthewBonanni")

    license("Apache-2.0", checked_by="MatthewBonanni")

    version("main", branch="main", submodules=_submodules)

    variant(
        "dims",
        default="2,3",
        values=("2", "3"),
        multi=True,
        description="Spatial dimensions to build: one solver per dimension "
        "(mallard2d, mallard3d), both run as mallard",
    )
    variant(
        "precision",
        default="double",
        values=("double", "single"),
        multi=False,
        description="Floating-point precision",
    )
    # Host backend: OpenMP, except on macOS, where Apple Clang has no OpenMP
    variant("openmp", default=not IS_DARWIN, description="Kokkos OpenMP host backend")
    variant("threads", default=IS_DARWIN, description="Kokkos C++ threads host backend")
    variant("serial", default=False, description="Kokkos serial backend on the host")
    variant("sycl", default=False, description="Kokkos SYCL backend (Intel GPUs)")
    variant("mpi", default=True, description="Distributed memory parallelism with MPI")
    variant(
        "gpu_aware_mpi",
        default=False,
        when="+mpi",
        description="Pass device buffers to MPI directly (CUDA- or ROCm-aware MPI)",
    )
    variant("nccl", default=False, when="+mpi+cuda", description="Halo exchanges with NCCL")
    variant("hdf5", default=True, description="HDF5 meshes and output, mallard mesh convert")
    variant("cgns", default=True, description="Read CGNS mesh files")
    variant("zlib", default=True, description="zlib compression of VTU and restart files")

    depends_on("c", type="build")
    depends_on("cxx", type="build")
    depends_on("cmake@3.22:", type="build")

    depends_on("kokkos@5: cxxstd=20")
    depends_on("kokkos+openmp", when="+openmp")
    depends_on("kokkos+threads", when="+threads")
    depends_on("kokkos+serial", when="+serial")
    depends_on("kokkos+sycl", when="+sycl")
    depends_on("kokkos+cuda+wrapper", when="+cuda")
    depends_on("kokkos+rocm", when="+rocm")
    for _arch in CudaPackage.cuda_arch_values:
        depends_on(f"kokkos cuda_arch={_arch}", when=f"+cuda cuda_arch={_arch}")
    for _target in ROCmPackage.amdgpu_targets:
        depends_on(f"kokkos amdgpu_target={_target}", when=f"+rocm amdgpu_target={_target}")

    depends_on("yaml-cpp@0.8:")
    depends_on("mpi", when="+mpi")
    depends_on("nccl", when="+nccl")
    for _arch in CudaPackage.cuda_arch_values:
        depends_on(f"nccl cuda_arch={_arch}", when=f"+nccl cuda_arch={_arch}")
    depends_on("hdf5~mpi", when="+hdf5~mpi")
    depends_on("hdf5+mpi", when="+hdf5+mpi")
    depends_on("cgns+hdf5", when="+cgns")
    depends_on("cgns~mpi", when="+cgns~mpi")
    depends_on("zlib-api", when="+zlib")

    conflicts("+openmp", when="+threads", msg="Kokkos builds one parallel host backend")
    conflicts("+cuda", when="+rocm", msg="Kokkos builds one device backend")
    conflicts("+cuda", when="+sycl", msg="Kokkos builds one device backend")
    conflicts("+rocm", when="+sycl", msg="Kokkos builds one device backend")
    conflicts("+cuda", when="cuda_arch=none", msg="Set cuda_arch, e.g. cuda_arch=80 for A100")
    conflicts("+rocm", when="amdgpu_target=none", msg="Set amdgpu_target, e.g. gfx90a")
    conflicts("+cgns", when="~hdf5", msg="CGNS meshes need +hdf5")

    def setup_run_environment(self, env):
        env.set("MALLARD_SHARE", join_path(self.prefix.share, "mallard"))

    @run_after("install")
    def check_install(self):
        """mallard and the solvers start and report their version."""
        for name in ["mallard"] + [f"mallard{dim}d" for dim in self.spec.variants["dims"].value]:
            Executable(join_path(self.prefix.bin, name))("--version", output=str, error=str)

    def _run_example(self, example, *command):
        share = join_path(self.prefix.share, "mallard")
        work = join_path(self.test_suite.stage, example)
        if os.path.exists(work):
            shutil.rmtree(work)
        shutil.copytree(join_path(share, "mechanisms"), join_path(work, "mechanisms"))
        case = join_path(work, "examples", example)
        shutil.copytree(join_path(share, "examples", example), case)
        with working_dir(case):
            Executable(join_path(self.prefix.bin, "mallard"))(*command, "-i", "input.toml")

    def test_sod(self):
        """Run the Sod shock tube example in 2D"""
        if not self.spec.satisfies("dims=2"):
            raise SkipTest("Mallard was built without dims=2")
        self._run_example("sod")

    def test_reactor(self):
        """Run the hydrogen ignition example with mallard reactor"""
        self._run_example("h2_ignition", "reactor")


class CMakeBuilder(cmake.CMakeBuilder):
    """One build per spatial dimension, installed side by side."""

    def _build_dir(self, dim):
        return join_path(self.pkg.stage.path, f"spack-build-{dim}d")

    def _dims(self):
        return sorted(self.spec.variants["dims"].value)

    def cmake_args(self):
        spec = self.spec
        args = [
            self.define("USE_SYSTEM_KOKKOS", True),
            self.define("Kokkos_ROOT", spec["kokkos"].prefix),
            self.define("USE_SYSTEM_YAML_CPP", True),
            self.define("Mallard_USE_DOUBLE", spec.satisfies("precision=double")),
            self.define_from_variant("Mallard_ENABLE_MPI", "mpi"),
            self.define_from_variant("Mallard_ENABLE_HDF5", "hdf5"),
            self.define("USE_SYSTEM_HDF5", True),
            self.define_from_variant("Mallard_ENABLE_CGNS", "cgns"),
            self.define("USE_SYSTEM_CGNS", True),
            self.define_from_variant("Mallard_ENABLE_ZLIB", "zlib"),
            self.define("Mallard_ENABLE_TESTS", self.pkg.run_tests),
        ]
        if spec.satisfies("+mpi"):
            args.append(self.define_from_variant("Mallard_GPU_AWARE_MPI", "gpu_aware_mpi"))
            if spec.satisfies("+cuda"):
                args.append(self.define_from_variant("Mallard_ENABLE_NCCL", "nccl"))
        if spec.satisfies("+cuda"):
            args.append(self.define("CMAKE_CXX_COMPILER", self.pkg["kokkos"].kokkos_cxx))
        elif spec.satisfies("+rocm") and not (
            spec.satisfies("%cxx=clang") or spec.satisfies("%cxx=rocmcc")
        ):
            args.append(self.define("CMAKE_CXX_COMPILER", spec["hip"].hipcc))
        return args

    def cmake(self, pkg, spec, prefix):
        for dim in self._dims():
            args = self.std_cmake_args + self.cmake_args()
            args += [
                self.define("Mallard_DIM", dim),
                os.path.abspath(self.root_cmakelists_dir),
            ]
            with working_dir(self._build_dir(dim), create=True):
                pkg.module.cmake(*args)

    def build(self, pkg, spec, prefix):
        for dim in self._dims():
            pkg.module.cmake("--build", self._build_dir(dim), "--parallel", str(make_jobs))

    def install(self, pkg, spec, prefix):
        for dim in self._dims():
            pkg.module.cmake("--install", self._build_dir(dim))

    def check(self):
        for dim in self._dims():
            with working_dir(self._build_dir(dim)):
                Executable(join_path(self._build_dir(dim), "test", "mallard-test" + dim + "d"))()
