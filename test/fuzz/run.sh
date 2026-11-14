#!/usr/bin/env bash
# Run the libFuzzer harnesses of a Mallard_ENABLE_FUZZING build, in parallel:
#   test/fuzz/run.sh BUILD_DIR SECONDS WORK_DIR [HARNESS...]
# Seeds come from test/data, examples, tutorials and test/fuzz/corpus.
# WORK_DIR/corpus/<harness> keeps what the run found (reuse it to continue),
# WORK_DIR/crashes the inputs that crashed. Fails if any harness crashed.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)

if [ "${1:-}" = --one ]; then
    harness=$2 build=$3 seconds=$4 work=$5
    seeds=$work/seeds/$harness
    mkdir -p "$seeds" "$work/corpus/$harness" "$work/crashes"
    case $harness in
        input) find "$root/examples" "$root/tutorials" -name '*.toml' -exec cp {} "$seeds" \;
               cp "$root"/test/fuzz/corpus/input/* "$seeds" ;;
        expression) cp "$root"/test/fuzz/corpus/expression/* "$seeds" ;;
        gmsh) for f in "$root"/test/data/meshes/*.msh; do
                  case $f in *fluent*) ;; *) cp "$f" "$seeds" ;; esac
              done ;;
        fluent) cp "$root"/test/data/meshes/*fluent*.msh "$root"/test/fuzz/corpus/fluent/* "$seeds" ;;
        openfoam) for d in "$root"/test/data/meshes/*_foam*; do
                      for f in points faces owner neighbour boundary; do
                          cat "$d/constant/polyMesh/$f"
                          [ $f = boundary ] || printf '\n//@@\n'
                      done > "$seeds/$(basename "$d")"
                  done ;;
    esac
    max_len=()
    case $harness in
        expression) max_len=(-max_len=512) ;;
        input) max_len=(-max_len=16384) ;;
    esac
    log=$work/$harness.log
    if "$build/test/fuzz/fuzz_$harness" -max_total_time="$seconds" -timeout=10 -rss_limit_mb=2048 \
        ${max_len[@]+"${max_len[@]}"} -print_final_stats=1 -artifact_prefix="$work/crashes/$harness-" \
        "$work/corpus/$harness" "$seeds" > "$log" 2>&1; then
        echo "$harness: $(grep -E '^stat::number_of_executed_units' "$log" | awk '{print $2}') runs, no crash"
    else
        echo "$harness: CRASH"
        grep -m1 -A30 -E 'ERROR|runtime error' "$log"
        exit 1
    fi
    exit 0
fi

build=$(cd "$1" && pwd) seconds=$2 work=$3
shift 3
mkdir -p "$work"
work=$(cd "$work" && pwd)
harnesses=("$@")
if [ ${#harnesses[@]} -eq 0 ]; then
    for h in input expression gmsh fluent openfoam mesh_h5 cgns; do
        [ -x "$build/test/fuzz/fuzz_$h" ] && harnesses+=("$h")
    done
fi
jobs=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)
printf '%s\n' "${harnesses[@]}" | xargs -P "$jobs" -I{} "$0" --one {} "$build" "$seconds" "$work"
