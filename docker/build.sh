#!/usr/bin/env bash
# Builds Mallard in 2D and 3D and installs both into PREFIX, for the container images:
#
#   build.sh SOURCE PREFIX [CMAKE_ARGS...]
#
# The 3D executables get the suffix 3D (Mallard3D, mallard-mesh-convert3D).
set -euo pipefail

src="$1"
prefix="$2"
shift 2
for dim in 2 3; do
    suffix=""
    [ "$dim" = 3 ] && suffix=3D
    cmake -S "$src" -B "/tmp/build-$dim" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$prefix" \
        -DUSE_SYSTEM_KOKKOS=OFF -DMallard_ENABLE_TESTS=OFF \
        -DMallard_ENABLE_MPI=ON -DMallard_ENABLE_HDF5=ON -DMallard_ENABLE_CGNS=ON -DUSE_SYSTEM_CGNS=OFF \
        -DMallard_DIM="$dim" -DMallard_EXECUTABLE_SUFFIX="$suffix" "$@"
    cmake --build "/tmp/build-$dim" --parallel "${JOBS:-$(nproc)}"
    cmake --install "/tmp/build-$dim"
done
