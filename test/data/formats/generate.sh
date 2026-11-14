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
# The cases' run log and report land here
trap 'rm -rf "$tmp" mallard.jsonl report.html' EXIT
host=$(hostname)
# Replaces the directory and the host in the run blocks of text and restart files:
# same length, so the restart files' trailer lengths hold
scrub() {
  DIR="$1" HOST="$host" perl -pe 'BEGIN { $d = substr("/fixtures/" . "x" x 4096, 0, length $ENV{DIR});
    $h = substr("fixture-host" . "x" x 256, 0, length $ENV{HOST}) } s/\Q$ENV{DIR}\E/$d/g; s/\Q$ENV{HOST}\E/$h/g' \
    "$2" > "$3"
}
root=$(cd "$here/../../.." && pwd -P)
for case in gas statistics mixture nscbc; do
  sed "s|^prefix = \"$case\"|prefix = \"$tmp/$case\"|" $case.toml > "$tmp/$case.toml"
  "$build/src/mallard" -i "$tmp/$case.toml" --kokkos-num-threads=1 > "$tmp/$case.log"
  for step in 000000 000002; do scrub "$root" "$tmp/${case}_$step.restart" "$out/${case}_$step.restart"; done
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
for f in "$tmp"/snapshots/snapshots*; do
  case "$f" in *.toml) continue ;; esac
  name=$(basename "$f")
  case "$f" in
    *.h5|*.vtkhdf)
      # Prints h5repack options that shrink the extendible datasets' chunks to their extent
      layout=$(python3 - "$f" "$dir" "$host" <<'EOF'
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
    def chunk(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.chunks and obj.chunks != obj.shape:
            print(f"-l/{name}:CHUNK={'x'.join(str(max(n, 1)) for n in obj.shape)}")
    f.visititems(chunk)
EOF
      )
      # shellcheck disable=SC2086
      h5repack $layout "$f" "$out/$name"
      ;;
    *)
      scrub "$dir" "$f" "$out/$name"
      ;;
  esac
done
ls -l "$out"
