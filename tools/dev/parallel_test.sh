#!/bin/sh
# Runs a GoogleTest suite as several processes at once, for CI jobs whose suite would use
# only some of the runner's cores: the job's share of the suite (GTEST_TOTAL_SHARDS and
# GTEST_SHARD_INDEX, if set) is cut into 4 pieces per worker, and each worker runs the next
# piece left, so a few long tests do not hold up the rest. Every test of the job's share
# runs once. Each piece's output follows when all are done, a failed piece's outside a
# collapsed group; the exit status is 1 if any piece failed.
#
#   tools/dev/parallel_test.sh WORKERS COMMAND [ARG...]
workers=$1
shift
[ "$workers" -ge 1 ] || workers=1
shards=${GTEST_TOTAL_SHARDS:-1}
shard=${GTEST_SHARD_INDEX:-0}
pieces=$((workers * 4))
logs=$(mktemp -d)
export logs shards shard pieces
echo "$pieces pieces, $workers at a time"

# Piece p of job shard s of S is googletest's shard s + S*p of S*pieces: together the
# pieces run exactly the tests of shard s of S
seq 0 $((pieces - 1)) | xargs -P "$workers" -I{} sh -c '
    p=$1
    shift
    start=$(date +%s)
    GTEST_TOTAL_SHARDS=$((shards * pieces)) GTEST_SHARD_INDEX=$((shard + shards * p)) "$@" > "$logs/$p.log" 2>&1
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
