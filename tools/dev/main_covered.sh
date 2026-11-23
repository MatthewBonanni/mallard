#!/usr/bin/env bash
# Whether a commit pushed to main lands a tree that already passed the gate in a batch, so
# ci.yml's push run need not run the gate again (the device job). Covered when the pull request
# the commit merged carries batch.yml's `CI` status ("Passed in batch <sha> (run <id>)"), that
# run is a passed, dispatched ci.yml run whose commit has `CI` success, and main reaches the run's
# tree at this commit or after it (a batch of several pull requests lands as several commits;
# the ones before its last carry trees no run tested, as every batch's do). Waits up to WAIT
# seconds (default 900) for main to reach that tree. Prints the covering run and exits 0, or
# says why not and exits 1. Needs gh (GH_TOKEN), and the commit and origin/main in the clone.
#
#   tools/dev/main_covered.sh <commit> [owner/repo]
set -euo pipefail

sha=$(git rev-parse "${1:?usage: main_covered.sh <commit> [owner/repo]}^{commit}")
repo=${2:-${GITHUB_REPOSITORY:?}}
wait=${WAIT:-900}
api() { gh api "repos/$repo/$1" "${@:2}"; }
no() { echo "not covered: $*"; exit 1; }

head=$(api "commits/$sha/pulls" --jq '[.[] | select(.merged_at)][0].head.sha // ""')
[ -n "$head" ] || no "${sha:0:8} merged no pull request"
status=$(api "commits/$head/statuses?per_page=100" \
    --jq '[.[] | select(.context == "CI")][0] | "\(.state)\t\(.creator.login)\t\(.description)"')
stamp='^Passed in batch [0-9a-f]{8} \(run ([0-9]+)\)'
IFS=$'\t' read -r state creator desc <<< "$status"
[ "$state" = success ] && [ "$creator" = "github-actions[bot]" ] \
    && [[ $desc =~ $stamp ]] \
    || no "the head ${head:0:8} of its pull request has no batch.yml CI status ($state, $creator: $desc)"
run=${BASH_REMATCH[1]}
info=$(api "actions/runs/$run" --jq '"\(.path)\t\(.event)\t\(.conclusion)\t\(.head_sha)"')
IFS=$'\t' read -r path event conclusion tested <<< "$info"
[ "$path" = .github/workflows/ci.yml ] && [ "$event" = workflow_dispatch ] && [ "$conclusion" = success ] \
    || no "run $run is not a passed, dispatched ci.yml run ($path $event $conclusion)"
ci=$(api "commits/$tested/status?per_page=100" --jq '[.statuses[] | select(.context == "CI")][0].state // "none"')
[ "$ci" = success ] || no "CI on run $run's commit ${tested:0:8} is $ci"
tree=$(api "git/commits/$tested" --jq .tree.sha)

deadline=$(( $(date +%s) + wait ))
git fetch -q origin main
while :; do
    for c in "$sha" $(git rev-list --reverse --first-parent --ancestry-path "$sha..origin/main"); do
        if [ "$(git rev-parse "$c^{tree}")" = "$tree" ]; then
            echo "covered: main reaches the tree ${tree:0:8} of run $run at ${c:0:8}"
            echo "run=$run"
            exit 0
        fi
    done
    [ "$(date +%s)" -lt "$deadline" ] || no "main has not reached the tree ${tree:0:8} of run $run in $wait s"
    sleep 20
    git fetch -q origin main
done
