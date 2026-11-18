#!/bin/bash
# run.sh MALLARD PVBATCH INPUT [REPEATS]: in the directory where INPUT was run,
# the wall time and peak memory of mallard post derive and of pv_derive.py on
# the last snapshot and on the whole series, without and with the run's TENO
# cache_file (benchmarks/README.md). macOS /usr/bin/time -l.
set -eu
MALLARD=$1; PVBATCH=$2; INPUT=$3; N=${4:-3}
HERE=$(cd "$(dirname "$0")" && pwd)
F=MACH,Q_CRITERION,LAMBDA2,SCHLIEREN,VORTICITY
SERIES=$(ls solut/*.vtkhdf.series)
LAST=$(ls solut/*_[0-9]*.vtkhdf | tail -1)
sed '/^cache_file = /d' "$INPUT" > uncached.toml
measure() {
  local label=$1; shift
  for _ in $(seq "$N"); do
    rm -rf bench_out
    /usr/bin/time -l "$@" > bench.log 2>&1
    awk -v l="$label" '/ real /{t=$1} /peak memory footprint/{m=$1}
      END{printf "%-24s %8.2f s %8.0f MB\n", l, t, m/1048576}' bench.log
  done
}
DERIVE=("$MALLARD" post derive --fields $F -o bench_out --kokkos-num-threads=2)
measure "mallard, one file" "${DERIVE[@]}" "$LAST" -i uncached.toml
measure "mallard, series" "${DERIVE[@]}" "$SERIES" -i uncached.toml
measure "mallard cached, one file" "${DERIVE[@]}" "$LAST" -i "$INPUT"
measure "mallard cached, series" "${DERIVE[@]}" "$SERIES" -i "$INPUT"
export VTK_SMP_MAX_THREADS=2
measure "pvbatch, one file" "$PVBATCH" "$HERE/pv_derive.py" 1.4 bench_out "$LAST"
measure "pvbatch, series" "$PVBATCH" "$HERE/pv_derive.py" 1.4 bench_out solut/*_[0-9]*.vtkhdf
