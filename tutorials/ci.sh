#!/usr/bin/env bash
# Run the tutorials at reduced size, as CI does: mesh, run on two MPI ranks,
# post-process. Checks that every step still works, not the accuracy.
#
#   tutorials/ci.sh MALLARD_2D MALLARD_3D [WORK_DIR] [TUTORIAL ...]
#
# MALLARD_2D and MALLARD_3D are the solvers of 2D and 3D builds with MPI;
# gmsh and python3 (with tutorials/requirements.txt) must be on the PATH.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
m2d="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
m3d="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
work="${3:-$(mktemp -d)}"
shift $(( $# < 3 ? $# : 3 ))
mkdir -p "$work"
work="$(cd "$work" && pwd)"
export OMP_NUM_THREADS=1
mpi() { mpirun -n 2 --oversubscribe "$@"; }

# set_key FILE KEY VALUE: replace the value of every line "KEY = ..." of FILE
set_key() { sed -i.bak "s|^$2 = .*|$2 = $3|" "$1" && rm -f "$1.bak"; }

cone() {
    cp "$here/cone/input.toml" .
    gmsh "$here/cone/cone.geo" -3 -setnumber h 0.06 -setnumber h_out 0.12 -o cone.msh > gmsh.log
    mpi "$m3d" -i input.toml > run.log
    python3 "$here/cone/taylor_maccoll.py" --csv taylor_maccoll.csv
    python3 "$here/cone/plot_mesh.py" cone.msh --out figures
    python3 "$here/cone/plot_results.py" . --out figures
}

tutorials=("$@")
[ ${#tutorials[@]} -gt 0 ] || tutorials=(cone)
for t in "${tutorials[@]}"; do
    echo "== $t"
    mkdir -p "$work/$t"
    start=$SECONDS
    (cd "$work/$t" && "$t") || { echo "$t FAILED (logs in $work/$t)"; tail -n 30 "$work/$t"/run.log 2>/dev/null; exit 1; }
    echo "$t passed in $(( SECONDS - start )) s"
done
