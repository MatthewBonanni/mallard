#!/usr/bin/env bash
# Smoke test of an installed Mallard (a container image, a Spack install or `cmake --install`):
# runs examples from share/mallard and compares them with reference solutions.
#
#   run.sh [PREFIX]
#
# PREFIX defaults to the installation of the Mallard on PATH. Runs the Sod shock tube
# with mallard (and on two MPI ranks, when built with MPI) and with mallard3d, and
# hydrogen ignition with mallard reactor, each where installed. Needs Python 3 with
# NumPy and SciPy.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
prefix="${1:-$(dirname "$(dirname "$(command -v mallard || command -v mallard3d)")")}"
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

for exe in mallard mallard3d; do
    [ -x "$bin/$exe" ] || continue
    "$bin/$exe" --version
    case="$work/examples/sod_$exe"
    cp -R "$share/examples/sod" "$case"
    [ "$exe" = mallard3d ] && cp "$here/sod_3d.toml" "$case/input.toml"
    (cd "$case" && run "$exe" "$bin/$exe" -i input.toml)
    python3 "$here/check.py" "$tools" sod "$case"/solut/sod_000010.*vtu
done

links_mpi() { { ldd "$1" || otool -L "$1"; } 2> /dev/null | grep -q libmpi; }
if [ -x "$bin/mallard" ] && command -v mpirun > /dev/null && links_mpi "$bin/mallard"; then
    case="$work/examples/sod_mpi"
    cp -R "$share/examples/sod" "$case"
    (cd "$case" && OMP_NUM_THREADS=1 run mpi mpirun -n 2 --oversubscribe "$bin/mallard" -i input.toml)
    python3 "$here/check.py" "$tools" sod "$case/solut/sod_000010.pvtu"
fi

for exe in mallard mallard3d; do
    [ -x "$bin/$exe" ] || continue
    out="$("$bin/$exe" mesh convert 2>&1 || true)"
    grep -q Usage <<< "$out" || { echo "$exe mesh convert: $out"; exit 1; }
    echo "== $exe mesh convert starts"
done

if [ -x "$bin/mallard" ]; then
    case="$work/examples/h2_ignition"
    cp -R "$share/examples/h2_ignition" "$case"
    (cd "$case" && run reactor "$bin/mallard" reactor -i input.toml)
    python3 "$here/check.py" "$tools" reactor "$case/reactor.log"
fi
echo "Smoke test passed"
