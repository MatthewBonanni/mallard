#!/usr/bin/env bash
# Smoke test of an installed Mallard (a container image, a Spack install or `cmake --install`):
# runs examples from share/mallard and compares them with reference solutions.
#
#   run.sh [PREFIX]
#
# PREFIX defaults to the installation of the Mallard on PATH. Runs the Sod shock tube
# with Mallard (and on two MPI ranks, when built with MPI) and with Mallard3D, and
# hydrogen ignition with MallardReactor, each where installed. Needs Python 3 with
# NumPy and SciPy.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
prefix="${1:-$(dirname "$(dirname "$(command -v Mallard || command -v Mallard3D)")")}"
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

for exe in Mallard Mallard3D; do
    [ -x "$bin/$exe" ] || continue
    "$bin/$exe" --version
    case="$work/examples/sod_$exe"
    cp -R "$share/examples/sod" "$case"
    [ "$exe" = Mallard3D ] && cp "$here/sod_3d.toml" "$case/input.toml"
    (cd "$case" && run "$exe" "$bin/$exe" -i input.toml)
    python3 "$here/check.py" "$tools" sod "$case"/solut/sod_000010.*vtu
done

links_mpi() { { ldd "$1" || otool -L "$1"; } 2> /dev/null | grep -q libmpi; }
if [ -x "$bin/Mallard" ] && command -v mpirun > /dev/null && links_mpi "$bin/Mallard"; then
    case="$work/examples/sod_mpi"
    cp -R "$share/examples/sod" "$case"
    (cd "$case" && OMP_NUM_THREADS=1 run mpi mpirun -n 2 --oversubscribe "$bin/Mallard" -i input.toml)
    python3 "$here/check.py" "$tools" sod "$case/solut/sod_000010.pvtu"
fi

if [ -x "$bin/MallardReactor" ]; then
    case="$work/examples/h2_ignition"
    cp -R "$share/examples/h2_ignition" "$case"
    (cd "$case" && run reactor "$bin/MallardReactor" -i input.toml)
    python3 "$here/check.py" "$tools" reactor "$case/reactor.log"
fi

for exe in mallard-mesh-convert mallard-mesh-convert3D; do
    [ -x "$bin/$exe" ] || continue
    out="$("$bin/$exe" 2>&1 || true)"
    grep -q Usage <<< "$out" || { echo "$exe: $out"; exit 1; }
    echo "== $exe starts"
done
echo "Smoke test passed"
