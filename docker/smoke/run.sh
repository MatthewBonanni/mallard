#!/usr/bin/env bash
# Smoke test of an installed Mallard (a container image, a Spack install or `cmake --install`):
# runs examples from share/mallard and compares them with reference solutions.
#
#   run.sh [PREFIX]
#
# PREFIX defaults to the installation of the mallard on PATH. Runs the Sod shock tube
# through mallard in 2D (also on two MPI ranks, when built with MPI) and in 3D, each
# dimension that is installed (mallard2d, mallard3d), and hydrogen ignition with
# mallard reactor. Needs Python 3 with NumPy and SciPy, and h5py for builds with HDF5.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
prefix="${1:-$(dirname "$(dirname "$(command -v mallard)")")}"
bin="$prefix/bin"
share="$prefix/share/mallard"
tools="$share/tools"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
mkdir -p "$work/examples"
cp -R "$share/mechanisms" "$work/"

run() {
    local name="$1"
    shift
    echo "== $name: $*"
    "$@" > "$name.log" 2>&1 || { cat "$name.log"; return 1; }
    tail -n 3 "$name.log"
}

"$bin/mallard" --version
for dim in 2 3; do
    [ -x "$bin/mallard${dim}d" ] || continue
    case="$work/examples/sod_${dim}d"
    cp -R "$share/examples/sod" "$case"
    [ "$dim" = 3 ] && cp "$here/sod_3d.toml" "$case/input.toml"
    (cd "$case" && run "sod_${dim}d" "$bin/mallard" -i input.toml)
    grep -q "Build.* ${dim}D," "$case/sod_${dim}d.log" || { echo "mallard ran sod_${dim}d in another dimension"; exit 1; }
    python3 "$here/check.py" "$tools" sod "$case/solut/sod_000010"
done

links_mpi() { { ldd "$1" || otool -L "$1"; } 2> /dev/null | grep -q libmpi; }
if [ -x "$bin/mallard2d" ] && command -v mpirun > /dev/null && links_mpi "$bin/mallard2d"; then
    case="$work/examples/sod_mpi"
    cp -R "$share/examples/sod" "$case"
    (cd "$case" && OMP_NUM_THREADS=1 run mpi mpirun -n 2 --oversubscribe "$bin/mallard" -i input.toml)
    python3 "$here/check.py" "$tools" sod "$case/solut/sod_000010"
fi

out="$("$bin/mallard" mesh convert --help 2>&1 || true)"
grep -q Usage <<< "$out" || { echo "mallard mesh convert: $out"; exit 1; }
echo "== mallard mesh convert starts"

case="$work/examples/h2_ignition"
cp -R "$share/examples/h2_ignition" "$case"
(cd "$case" && run reactor "$bin/mallard" reactor -i input.toml)
python3 "$here/check.py" "$tools" reactor "$case/reactor.log"
echo "Smoke test passed"
