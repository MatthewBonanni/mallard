#!/bin/sh
# Builds run_case against an installed Mallard (find_package(Mallard)), runs examples/sod with it and
# with the installed mallard, and checks that both write the same files bit for bit.
#
#   examples/cpp_api/ci.sh PREFIX WORKDIR
set -eu
prefix=$(cd "$1" && pwd)
mkdir -p "$2"
work=$(cd "$2" && pwd)
here=$(cd "$(dirname "$0")" && pwd)
cmake -S "$here" -B "$work/build" -DCMAKE_PREFIX_PATH="$prefix" -DCMAKE_BUILD_TYPE=Release
cmake --build "$work/build"
for run in api executable; do
    rm -rf "${work:?}/$run"
    mkdir -p "$work/$run"
    cp "$here/../sod/input.toml" "$work/$run/input.toml"
done
(cd "$work/api" && "$work/build/run_case" input.toml --kokkos-num-threads=2)
(cd "$work/executable" && "$prefix/bin/mallard" -i input.toml --kokkos-num-threads=2)
n=0
for file in "$work/executable/solut/"*; do
    name=$(basename "$file")
    "$prefix/bin/mallard" diff "$file" "$work/api/solut/$name" > "$work/diff.txt" || {
        cat "$work/diff.txt"
        echo "run_case and mallard -i differ in solut/$name"
        exit 1
    }
    n=$((n + 1))
done
test "$n" -gt 0 || { echo "mallard -i wrote no files in solut/"; exit 1; }
echo "run_case and mallard -i wrote the same $n files bit for bit"
