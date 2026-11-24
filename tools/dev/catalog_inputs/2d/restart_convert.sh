#!/bin/sh
# Writes restart_convert.restart, the restart file restart_convert.toml starts from:
# the initial state of its case, a constant H2/air state.
#   tools/dev/catalog_inputs/2d/restart_convert.sh BUILD_DIR   (a 2D build)
# The run block's host and directories are replaced as in test/data/formats/generate.sh.
set -eu
here=$(cd "$(dirname "$0")" && pwd -P)
build=$(cd "$1" && pwd)
root=$(cd "$here/../../../.." && pwd -P)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
sed -n '/^\[mesh\]/,/^mechanism/p' "$here/restart_convert.toml" | sed "s|\.\./\.\./\.\./\.\./|$root/|" > "$tmp/case.toml"
sed -n '/^\[\[boundaries\]\]/,$p' "$here/restart_convert.toml" >> "$tmp/case.toml"
cat >> "$tmp/case.toml" <<TOML

[run]
n_steps = 1
dt = 1.0e-9

[initialize]
p = 101325.0
T = 300.0
u = [10.0, 0.0]
X = {H2 = 2.0, O2 = 1.0, N2 = 3.76}

[output]
check_interval = 1000000

[[write_data]]
prefix = "start"
format = "restart"
interval = 1
TOML
(cd "$tmp" && "$build/src/mallard" -i case.toml --kokkos-num-threads=1 > run.log)
dir=$(cd "$tmp" && pwd -P)
DIR="$dir" ROOT="$root" HOST="$(hostname)" perl -pe 'BEGIN { $d = substr("/fixtures/" . "x" x 4096, 0, length $ENV{DIR});
  $r = substr("/repository/" . "x" x 4096, 0, length $ENV{ROOT});
  $h = substr("fixture-host" . "x" x 256, 0, length $ENV{HOST}) } s/\Q$ENV{DIR}\E/$d/g; s/\Q$ENV{ROOT}\E/$r/g; s/\Q$ENV{HOST}\E/$h/g' \
  "$tmp/start_000001.restart" > "$here/restart_convert.restart"
