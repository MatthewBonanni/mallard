#!/bin/bash
# Writes the format fixtures of a release with that release's 2D, double-precision
# build (with HDF5): test/data/formats/generate.sh BUILD_DIR vX.Y.Z
# then add the release to RESTART_FIXTURES and the mesh releases in
# test/format_compat_test.cpp, and its snapshots to the output reader's test.
# The snapshot files' run blocks name the host and directory that wrote them:
# they are replaced (python3 with h5py rewrites the HDF5 attributes).
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

# Snapshots in every format, run in their own directory (format = "hdf5" before 0.10)
mkdir "$tmp/snapshots"
case "$2" in
  v0.[0-9].*) sed 's/format = "xdmf"/format = "hdf5"/' snapshots.toml > "$tmp/snapshots/snapshots.toml" ;;
  *) cp snapshots.toml "$tmp/snapshots/" ;;
esac
(cd "$tmp/snapshots" && "$build/src/mallard" -i snapshots.toml --kokkos-num-threads=1 > ../snapshots.log)
dir=$(cd "$tmp/snapshots" && pwd -P)
host=$(hostname)
for f in "$tmp"/snapshots/snapshots*; do
  case "$f" in *.toml) continue ;; esac
  name=$(basename "$f")
  case "$f" in
    *.h5|*.vtkhdf)
      python3 - "$f" "$dir" "$host" <<'EOF'
import sys
import h5py
import numpy as np
path, directory, host = sys.argv[1:]
with h5py.File(path, "r+") as f:
    for name, value in list(f["/"].attrs.items()):
        if value.dtype.kind != "S":
            continue
        text = [v.decode().replace(directory, "/fixtures/run").replace(host, "fixture-host") for v in value]
        del f["/"].attrs[name]
        f["/"].attrs[name] = np.array([t.encode() for t in text])
EOF
      h5repack "$f" "$out/$name"
      ;;
    *)
      # Same length, so the restart files' trailer lengths hold
      DIR="$dir" HOST="$host" perl -pe 'BEGIN { $d = substr("/fixtures/" . "x" x 4096, 0, length $ENV{DIR});
        $h = substr("fixture-host" . "x" x 256, 0, length $ENV{HOST}) } s/\Q$ENV{DIR}\E/$d/g; s/\Q$ENV{HOST}\E/$h/g' \
        "$f" > "$out/$name"
      ;;
  esac
done
ls -l "$out"
