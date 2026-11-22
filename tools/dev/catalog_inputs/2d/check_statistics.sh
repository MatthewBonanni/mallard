#!/bin/sh
# Writes check_statistics.restart, the [statistics] averages check.toml's
# check.statistics reads: one step of check.toml's case (without its [regions] and [check]) averaging ENSTROPHY.
#   tools/dev/catalog_inputs/2d/check_statistics.sh BUILD_DIR   (a 2D build)
# The run block's host and directory are replaced as in test/data/formats/generate.sh.
set -eu
here=$(cd "$(dirname "$0")" && pwd -P)
build=$(cd "$1" && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
sed '/^\[regions\]/,$d' "$here/check.toml" > "$tmp/case.toml"
cat >> "$tmp/case.toml" <<TOML
[output]
check_interval = 1000000

[statistics]
fields = ["ENSTROPHY"]

[[write_data]]
prefix = "stats"
format = "restart"
interval = 1
TOML
(cd "$tmp" && "$build/src/mallard" -i case.toml --kokkos-num-threads=1 > run.log)
dir=$(cd "$tmp" && pwd -P)
DIR="$dir" HOST="$(hostname)" perl -pe 'BEGIN { $d = substr("/fixtures/" . "x" x 4096, 0, length $ENV{DIR});
  $h = substr("fixture-host" . "x" x 256, 0, length $ENV{HOST}) } s/\Q$ENV{DIR}\E/$d/g; s/\Q$ENV{HOST}\E/$h/g' \
  "$tmp/stats_000001.restart" > "$here/check_statistics.restart"
