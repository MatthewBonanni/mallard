#!/usr/bin/env bash
# Builds the portable binaries of Mallard, 2D and 3D, for one architecture, backend and MPI, in a container of
# an old glibc (manylinux_2_28: glibc 2.28, RHEL 8 and newer), for the release:
#
#   archive.sh SOURCE OUTDIR VARIANT MPI [CMAKE_ARGS...]
#
# VARIANT is cpu (OpenMP), cuda12, cuda13 (CUDA with an OpenMP host backend) or rocm (HIP); the container
# brings the compilers and toolkits (release.yml names them). The archive unpacks anywhere:
#
#   OUTDIR/mallard-<version>-linux-<arch>-<variant>-<mpi>.tar.gz   bin/, lib/, share/ under one directory
#   OUTDIR/<archive>.sha256                                        for sha256sum -c
#   OUTDIR/<archive>.json                                          its entry of release-manifest.json
#
# MPI is the ABI of one MPI family, its C interface only, and the archive carries a library of it, so it runs
# on one node as it is:
#   mpich    libmpi.so.12: MPICH, Cray MPICH, Intel MPI, MVAPICH. Built and carried: the container's MPICH
#            (mpich-devel).
#   openmpi  libmpi.so.40: Open MPI 4.x and 5.x. Built against the container's Open MPI 4.1 (openmpi-devel),
#            whose binaries Open MPI 5 runs too; carries Open MPI 5 built here with its components linked in
#            (--disable-dlopen), which runs a single process without a launcher or its install prefix.
# Multi-node runs load the site's MPI of that family with `module load`, whose library comes first on
# LD_LIBRARY_PATH: every executable and library finds the rest of the archive through its RUNPATH
# ($ORIGIN/../lib), which LD_LIBRARY_PATH overrides. CGNS and Kokkos are linked in; HDF5 (parallel, from the
# submodule) and every other library but the C and C++ runtimes of the system (and the GPU driver) are copied
# into lib/.
set -euo pipefail

src=$(cd "$1" && pwd)
out=$(mkdir -p "$2" && cd "$2" && pwd)
variant=$3
mpi_family=$4
shift 4
arch=$(uname -m)
case "$variant" in
    cpu) backend=(-DKokkos_ENABLE_OPENMP=ON) ;;
    cuda12 | cuda13)
        # One fat binary: machine code for A100, A40/RTX 30xx, L40S/RTX 40xx and H100, PTX of the lowest
        backend=(-DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON)
        export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-} -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_90,code=sm_90"
        ;;
    rocm) backend=(-DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_HIP=ON -DKokkos_ARCH_AMD_GFX90A=ON -DCMAKE_CXX_COMPILER=hipcc) ;;
    *) echo "archive.sh: variant $variant: cpu, cuda12, cuda13 or rocm" >&2; exit 2 ;;
esac
version=$(git -C "$src" describe --tags --match 'v[0-9]*' 2> /dev/null | sed 's/^v//') ||
    version=$(sed -n 's/^project(Mallard VERSION \([0-9.]*\) .*/\1/p' "$src/CMakeLists.txt")
name="mallard-$version-linux-$arch-$variant-$mpi_family"
stage="$out/stage/$name"
rm -rf "$stage" && mkdir -p "$stage"

case "$mpi_family" in
    mpich) mpicc=/usr/lib64/mpich/bin/mpicc ;;
    openmpi) mpicc=/usr/lib64/openmpi/bin/mpicc ;;
    *) echo "archive.sh: MPI $mpi_family: mpich or openmpi" >&2; exit 2 ;;
esac
[ -x "$mpicc" ] || { echo "archive.sh: no $mpicc; the container must provide $mpi_family-devel" >&2; exit 1; }
export PATH="$(dirname "$mpicc"):$PATH"
mpicc -show | grep -q "$mpi_family" || { echo "archive.sh: $mpicc is not $mpi_family's" >&2; exit 1; }
# MPI's C library and headers alone: no libmpicxx, and none of the distribution's flags that its compiler
# wrappers add (RHEL's add _GLIBCXX_ASSERTIONS and hardening options)
mpi_lib=$(dirname "$(dirname "$mpicc")")/lib
mpi_inc=$(mpicc -show | tr ' ' '\n' | sed -n 's/^-I//p' | head -n 1)
carried_lib=$mpi_lib
if [ "$mpi_family" = openmpi ]; then
    ompi=openmpi-5.0.10
    ompi_sha256=0acecc4fc218e5debdbcb8a41d182c6b0f1d29393015ed763b2a91d5d7374cc6
    curl -fsSL -o "$out/$ompi.tar.bz2" "https://download.open-mpi.org/release/open-mpi/v5.0/$ompi.tar.bz2"
    echo "$ompi_sha256  $out/$ompi.tar.bz2" | sha256sum -c -
    tar -C "$out" -xjf "$out/$ompi.tar.bz2"
    # One node's transports (shared memory, TCP); a site's Open MPI brings its network's
    (cd "$out/$ompi" && ./configure --prefix="$out/openmpi" --disable-dlopen --disable-mpi-fortran --disable-oshmem \
        --without-ofi --without-ucx --without-psm2 --without-cuda --with-hwloc=internal \
        --with-libevent=internal --with-pmix=internal --with-prrte=internal > "$out/$ompi.configure.log" &&
        make -j "${JOBS:-$(nproc)}" > "$out/$ompi.make.log" && make install > /dev/null) ||
        { tail -n 50 "$out/$ompi".*.log >&2; exit 1; }
    carried_lib=$out/openmpi/lib
fi
mpi=(-DMPI_SKIP_COMPILER_WRAPPER=ON -DMPI_C_HEADER_DIR="$mpi_inc" -DMPI_CXX_HEADER_DIR="$mpi_inc"
    -DMPI_C_LIB_NAMES=mpi -DMPI_CXX_LIB_NAMES=mpi -DMPI_mpi_LIBRARY="$mpi_lib/libmpi.so" -DMPI_CXX_SKIP_MPICXX=ON)

# Parallel HDF5 from the submodule, installed on its own first: CGNS (built from source) finds it there
hdf5="$out/hdf5"
cmake -S "$src/src/external/hdf5" -B "$out/build-hdf5" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$hdf5" \
    "${mpi[@]}" -DHDF5_ENABLE_PARALLEL=ON -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF \
    -DHDF5_BUILD_TOOLS=OFF -DHDF5_BUILD_EXAMPLES=OFF -DHDF5_BUILD_CPP_LIB=OFF -DHDF5_BUILD_FORTRAN=OFF \
    -DHDF5_BUILD_JAVA=OFF -DHDF5_BUILD_UTILS=OFF
cmake --build "$out/build-hdf5" --parallel "${JOBS:-$(nproc)}"
cmake --install "$out/build-hdf5"
# What ldd resolves while collecting lib/: the libraries the build linked from outside system paths, and
# the MPI library the archive carries
export LD_LIBRARY_PATH="$hdf5/lib:$hdf5/lib64:$carried_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

for dim in 2 3; do
    cmake -S "$src" -B "$out/build-$dim" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$stage" \
        -DUSE_SYSTEM_KOKKOS=OFF -DMallard_ENABLE_TESTS=OFF -DMallard_ENABLE_PYTHON=OFF -DMallard_ENABLE_LIBRARY=OFF \
        -DMallard_ENABLE_MPI=ON -DMallard_ENABLE_HDF5=ON -DHDF5_ROOT="$hdf5" \
        -DMallard_ENABLE_CGNS=ON -DUSE_SYSTEM_CGNS=OFF -DMallard_ENABLE_ZLIB=ON \
        -DCMAKE_INSTALL_RPATH='$ORIGIN/../lib' -DCMAKE_BUILD_WITH_INSTALL_RPATH=OFF \
        "${mpi[@]}" -DMallard_DIM="$dim" "${backend[@]}" "$@"
    cmake --build "$out/build-$dim" --parallel "${JOBS:-$(nproc)}"
    cmake --install "$out/build-$dim"
done
# Mallard's executables only, not the bundled Kokkos's tools
find "$stage/bin" -type f ! -name 'mallard*' -delete

# The libraries the system provides everywhere (manylinux's policy), and the GPU driver's
system_lib() {
    case "$1" in
        linux-vdso.so* | ld-linux*.so* | libc.so* | libm.so* | libdl.so* | libpthread.so* | librt.so* | libutil.so* | \
            libstdc++.so* | libgcc_s.so* | libcuda.so* | libnvidia-*.so* | libdrm*.so*) return 0 ;;
    esac
    return 1
}
mkdir -p "$stage/lib"
elves() { find "$stage/bin" "$stage/lib" -type f -exec sh -c 'head -c 4 "$1" | grep -q ELF' sh {} \; -print; }
changed=1
while [ "$changed" = 1 ]; do
    changed=0
    for f in $(elves); do
        while read -r lib path; do
            system_lib "$lib" && continue
            [ -n "$path" ] && [ -f "$path" ] || { echo "archive.sh: $f needs $lib, not found" >&2; exit 1; }
            [ -e "$stage/lib/$lib" ] && continue
            cp -L "$path" "$stage/lib/$lib"
            chmod u+w "$stage/lib/$lib"
            changed=1
        done < <(ldd "$f" | awk '/=>/ { print $1, ($3 ~ /^\//) ? $3 : "" }')
    done
done
for f in $(elves); do
    case "$f" in "$stage/bin/"*) rpath='$ORIGIN/../lib' ;; *) rpath='$ORIGIN' ;; esac
    # RUNPATH (patchelf's default), not RPATH: LD_LIBRARY_PATH (a site's MPI module) comes first
    patchelf --remove-rpath "$f"
    patchelf --set-rpath "$rpath" "$f"
    strip --strip-unneeded "$f" 2> /dev/null || true
done
# The oldest glibc the archive runs on: the newest symbol version any of its files needs
glibc=$(for f in $(elves); do objdump -T "$f" 2> /dev/null; done | grep -o 'GLIBC_[0-9.]*' | sed 's/GLIBC_//' |
    sort -t. -k1,1n -k2,2n -k3,3n -u | tail -n 1)
cat > "$stage/share/mallard/archive.toml" << EOF
# Portable binaries of Mallard (docker/archive.sh): unpack them anywhere and run bin/mallard.
version = "$version"
variant = "$variant"
platform = "linux-$arch"
glibc = "$glibc"
mpi = "$mpi_family"
EOF

# Every executable runs from the moved tree, with nothing on LD_LIBRARY_PATH
moved="$out/moved/$name"
rm -rf "$out/moved" && mkdir -p "$out/moved" && cp -a "$stage" "$moved"
env -u LD_LIBRARY_PATH "$moved/bin/mallard" --version
for f in $(find "$moved/bin" -type f -perm -u+x); do
    ! env -u LD_LIBRARY_PATH ldd "$f" 2> /dev/null | grep 'not found' | grep -v libcuda || { echo "archive.sh: $f" >&2; exit 1; }
done
rm -rf "$out/moved"

tar -C "$out/stage" -czf "$out/$name.tar.gz" "$name"
(cd "$out" && sha256sum "$name.tar.gz" > "$name.tar.gz.sha256")
sha=$(cut -d' ' -f1 "$out/$name.tar.gz.sha256")
size=$(stat -c %s "$out/$name.tar.gz")
cat > "$out/$name.tar.gz.json" << EOF
{"kind": "archive", "name": "$name.tar.gz", "variant": "$variant", "platform": "linux-$arch", "sha256": "$sha", "size": $size, "glibc": "$glibc", "mpi": "$mpi_family", "dims": [2, 3]}
EOF
echo "archive.sh: $out/$name.tar.gz ($size bytes, glibc $glibc, sha256 $sha)"
