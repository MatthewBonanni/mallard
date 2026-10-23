#!/usr/bin/env bash
# Create every case template with `Mallard new --quick` and run it as a user
# would: mesh.sh, init.sh, run.sh on two MPI ranks, post.sh. Checks that each
# step works, not the accuracy.
#
#   templates/ci.sh MALLARD_2D MALLARD_3D [WORK_DIR] [TEMPLATE ...]
#
# MALLARD_2D and MALLARD_3D are 2D and 3D builds with MPI (from a build or an
# install tree); python3 must have the packages the scripts list.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
m2d="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
m3d="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
work="${3:-$(mktemp -d)}"
shift $(( $# < 3 ? $# : 3 ))
mkdir -p "$work"
work="$(cd "$work" && pwd)"
export OMP_NUM_THREADS=1 MPLBACKEND=Agg

field() {
    python3 - "$here/manifest.toml" "$1" "${2:-}" <<'EOF'
import sys, tomllib
templates = tomllib.load(open(sys.argv[1], "rb"))["templates"]
if sys.argv[2] == "names":
    print(" ".join(t["name"] for t in templates))
else:
    print(next(t for t in templates if t["name"] == sys.argv[2]).get(sys.argv[3], 2))
EOF
}

templates=("$@")
[ ${#templates[@]} -gt 0 ] || read -r -a templates <<< "$(field names)"
failed=()
for t in "${templates[@]}"; do
    echo "== $t"
    start=$SECONDS
    mallard="$m2d"
    [ "$(field "$t" dim)" = 3 ] && mallard="$m3d"
    dir="$work/$t"
    rm -rf "$dir"
    if (
        set -e
        "$mallard" new "$t" "$dir" --quick > "$work/$t.new.log"
        cd "$dir"
        if [ -x mesh.sh ]; then ./mesh.sh > mesh.log 2>&1; fi
        if [ -x init.sh ]; then ./init.sh > init.log 2>&1; fi
        ./run.sh -n 2 > run.log 2>&1
        if [ -x post.sh ]; then ./post.sh > post.log 2>&1; fi
    ); then
        echo "$t passed in $(( SECONDS - start )) s"
    else
        echo "$t FAILED (files in $dir)"
        for log in "$work/$t.new.log" "$dir"/*.log; do
            [ -f "$log" ] && { echo "--- $log"; tail -n 25 "$log"; }
        done
        failed+=("$t")
    fi
done

# run.sh on threads instead of MPI ranks
(cd "$work/${templates[0]}" && ./run.sh -t 2 > run_threads.log 2>&1) || failed+=("${templates[0]} (threads)")

if [ ${#failed[@]} -gt 0 ]; then
    echo "FAILED: ${failed[*]}"
    exit 1
fi
echo "All ${#templates[@]} templates passed"
