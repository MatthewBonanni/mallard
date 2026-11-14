#!/usr/bin/env bash
# Create every case template with `mallard new --quick` and run it as a user
# would: mesh.sh, init.sh, run.sh on two MPI ranks, post.sh. Checks that each
# step works, not the accuracy.
#
#   templates/ci.sh MALLARD [WORK_DIR] [TEMPLATE ...]
#
# MALLARD is the mallard of an install with both dimensions (mallard2d and
# mallard3d), built with MPI; python3 must have the packages the scripts list.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
mallard="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
work="${2:-$(mktemp -d)}"
shift $(( $# < 2 ? $# : 2 ))
mkdir -p "$work"
work="$(cd "$work" && pwd)"
export OMP_NUM_THREADS=1 MPLBACKEND=Agg

names() {
    python3 -c 'import sys, tomllib; print(" ".join(t["name"] for t in tomllib.load(open(sys.argv[1], "rb"))["templates"]))' \
        "$here/manifest.toml"
}

templates=("$@")
[ ${#templates[@]} -gt 0 ] || read -r -a templates <<< "$(names)"
failed=()
for t in "${templates[@]}"; do
    echo "== $t"
    start=$SECONDS
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
