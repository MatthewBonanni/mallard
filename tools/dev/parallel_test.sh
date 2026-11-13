#!/bin/sh
# Runs a GoogleTest suite as several processes at once, for CI jobs whose suite would use
# only some of the runner's cores: the tests the job selects (GTEST_FILTER, if set) are split
# into one piece per worker of about equal duration by TIMES (tools/dev/test_split.py), and
# the pieces run at once. Every selected test runs once. Each piece's output follows when all
# are done, a failed piece's outside a collapsed group; the exit status is 1 if any piece
# failed.
#
#   tools/dev/parallel_test.sh WORKERS TIMES COMMAND [ARG...]
workers=$1
times=$2
shift 2
[ "$workers" -ge 1 ] || workers=1
if [ -n "${GTEST_TOTAL_SHARDS:-}" ]; then
    echo "parallel_test.sh: googletest's sharding is not supported; select a shard with GTEST_FILTER" >&2
    exit 1
fi
here=$(dirname "$0")
logs=$(mktemp -d)
export logs
# The listing writes no JUnit report
env -u MALLARD_TEST_XML "$@" --gtest_list_tests > "$logs/tests.txt" || { cat "$logs/tests.txt"; exit 1; }
python3 "$here/test_split.py" split "$times" "$workers" < "$logs/tests.txt" > "$logs/pieces.txt" || exit 1
pieces=$(wc -l < "$logs/pieces.txt" | tr -d ' ')
export pieces
p=0
while IFS= read -r filter; do
    printf '%s\n' "$filter" > "$logs/$p.filter"
    p=$((p + 1))
done < "$logs/pieces.txt"
echo "$pieces pieces at once"

seq 0 $((pieces - 1)) | xargs -P "$workers" -I{} sh -c '
    p=$1
    shift
    start=$(date +%s)
    if [ "$pieces" = 1 ]; then
        "$@" > "$logs/$p.log" 2>&1
    else
        "$@" --gtest_filter="$(cat "$logs/$p.filter")" > "$logs/$p.log" 2>&1
    fi
    status=$?
    echo "$status" > "$logs/$p.status"
    echo "piece $((p + 1)) of $pieces: exit status $status after $(($(date +%s) - start)) s"
' sh {} "$@"

failed=""
p=0
while [ "$p" -lt "$pieces" ]; do
    if [ "$(cat "$logs/$p.status" 2> /dev/null)" = 0 ]; then
        echo "::group::Piece $((p + 1)) of $pieces"
        cat "$logs/$p.log"
        echo "::endgroup::"
    else
        echo "== Piece $((p + 1)) of $pieces (failed)"
        cat "$logs/$p.log"
        failed="$failed $((p + 1))"
    fi
    p=$((p + 1))
done
rm -rf "$logs"
if [ -n "$failed" ]; then
    echo "Failed pieces:$failed"
    exit 1
fi
