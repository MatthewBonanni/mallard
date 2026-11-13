#!/bin/sh
# The checks of CI's first job, in seconds and without a build, on what the branch would
# land: its merge into origin/main, which also has to be free of conflicts. Run it before
# asking for a CI run. ci.yml's changelog job runs it too, so the two stay the same.
#
#   tools/dev/precheck.sh [BASE]     (BASE: default origin/main)
#
# PRECHECK_SKIP_CHANGELOG=1 leaves out the changelog fragment check (CI does it itself,
# where a no-changelog label can waive it). PYTHON is the python to use (default python3);
# the two workflow checks need PyYAML.
set -u
repo=$(git rev-parse --show-toplevel) || exit 1
cd "$repo" || exit 1
base=${1:-origin/main}
python=${PYTHON:-python3}
failed=""

run() {
    name=$1
    shift
    if [ -n "${GITHUB_ACTIONS:-}" ]; then echo "::group::$name"; else echo "== $name"; fi
    "$@" || failed="$failed
  $name"
    if [ -n "${GITHUB_ACTIONS:-}" ]; then echo "::endgroup::"; fi
}

# The tree of the merge, checked out where the checks run
dir=$repo
if [ "$(git rev-parse HEAD)" != "$(git rev-parse "$base")" ]; then
    echo "== merges into $base"
    if merge=$(git merge-tree --write-tree --name-only --no-messages "$base" HEAD); then
        # A throwaway commit of the merge (CI runners have no git identity)
        commit=$(GIT_AUTHOR_NAME=precheck GIT_AUTHOR_EMAIL=precheck@localhost GIT_COMMITTER_NAME=precheck \
            GIT_COMMITTER_EMAIL=precheck@localhost git commit-tree -p "$base" -p HEAD -m precheck \
            "$(echo "$merge" | head -n 1)") || exit 1
        dir=$(mktemp -d)
        trap 'git -C "$repo" worktree remove --force "$dir"' EXIT
        git worktree add -q --detach "$dir" "$commit" || exit 1
    else
        echo "conflicts in:"
        echo "$merge" | sed -n '2,$s/^/  /p'
        failed="$failed
  merges into $base"
    fi
fi

checks() {
    cd "$dir" || return 1
    if "$python" -c 'import yaml' 2> /dev/null; then
        run "workflow triggers" "$python" tools/dev/check_workflow_triggers.py
        run "support matrix names every CI job" "$python" tools/dev/check_support_matrix.py
    else
        echo "== workflow triggers, support matrix: skipped (no PyYAML; PYTHON= a python with it)"
    fi
    run "spelling" "$python" tools/dev/check_spelling.py
    run "CI's test splitting" "$python" tools/dev/test_split_test.py
    run "documented input keys and defaults" "$python" tools/dev/input_catalog.py docs
    run "TOML snippets of the docs" "$python" tools/dev/check_doc_toml.py
    run "\"Used in\" lines of the guide" "$python" tools/docs/used_in.py --check
    run "feature matrix" "$python" tools/docs/feature_matrix.py --check
    cd "$repo" || return 1
}
if [ -z "$failed" ]; then checks; fi
if [ "${PRECHECK_SKIP_CHANGELOG:-}" != 1 ]; then
    run "changelog fragment" "$python" tools/dev/changelog.py check "$base"
fi

if [ -n "$failed" ]; then
    echo "precheck failed:$failed"
    exit 1
fi
echo "precheck passed"
