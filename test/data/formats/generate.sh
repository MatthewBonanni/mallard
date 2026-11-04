#!/bin/bash
# Writes the format fixtures of a release with that release's 2D, double-precision
# build (with HDF5): test/data/formats/generate.sh BUILD_DIR vX.Y.Z
# then add the release to RESTART_FIXTURES and the mesh releases in
# test/format_compat_test.cpp.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
build=$(cd "$1" && pwd)
out=$here/$2
mkdir -p "$out"
# The inputs refer to ../../../mechanisms, so they run from this directory
cd "$here"
tmp=$(mktemp -d "$here/.generate.XXXXXX")
trap 'rm -rf "$tmp"' EXIT
for case in gas statistics mixture nscbc; do
  sed "s|^prefix = \"$case\"|prefix = \"$tmp/$case\"|" $case.toml > "$tmp/$case.toml"
  "$build/src/mallard" -i "$tmp/$case.toml" --kokkos-num-threads=1 > "$tmp/$case.log"
  cp "$tmp/${case}_000000.restart" "$tmp/${case}_000002.restart" "$out/"
done
"$build/src/mallard" mesh convert gas.toml "$out/mesh.h5"
"$build/src/mallard" mesh convert annulus.msh "$out/curved_mesh.h5"
ls -l "$out"
