#!/usr/bin/env bash
# Builds Mallard in 2D and 3D and installs both into PREFIX, for the container images:
#
#   build.sh SOURCE PREFIX [CMAKE_ARGS...]
#
# The solvers are mallard2d and mallard3d; mallard runs the one of a case's dimension.
set -euo pipefail

src="$1"
prefix="$2"
shift 2
# .dockerignore leaves tracked files (docs, CI) out of the context; without this the
# version would read as modified
if git -C "$src" rev-parse --git-dir >/dev/null 2>&1; then
    git -C "$src" ls-files -z --deleted | xargs -0r git -C "$src" update-index --skip-worktree
fi
for dim in 2 3; do
    cmake -S "$src" -B "/tmp/build-$dim" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$prefix" \
        -DUSE_SYSTEM_KOKKOS=OFF -DMallard_ENABLE_TESTS=OFF \
        -DMallard_ENABLE_MPI=ON -DMallard_ENABLE_HDF5=ON -DMallard_ENABLE_CGNS=ON -DUSE_SYSTEM_CGNS=OFF -DMallard_ENABLE_ZLIB=ON \
        -DMallard_DIM="$dim" "$@"
    cmake --build "/tmp/build-$dim" --parallel "${JOBS:-$(nproc)}"
    cmake --install "/tmp/build-$dim"
done
