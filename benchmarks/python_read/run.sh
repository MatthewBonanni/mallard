#!/bin/bash
# run.sh PVBATCH CASE_DIR... : for each case run by make_cases.py, each file (VTKHDF series, XDMF and PVTU
# series indexes) and task (open, one, series), REPEATS (default 3) fresh processes of read_mallard.py (the
# mallard package and library of the environment: PYTHONPATH, MALLARD_LIBRARY) and of read_pvbatch.py, one
# thread each. CSV on stdout: case,file,task,tool,repeat,seconds in the script,checksum,wall s,peak RSS kB
# (GNU time). Values saved to npy/ for comparison (benchmarks/README.md).
set -u
PVBATCH=$1; shift
HERE=$(cd "$(dirname "$0")" && pwd)
N=${REPEATS:-3}
export OMP_NUM_THREADS=1 VTK_SMP_MAX_THREADS=1
mkdir -p npy
for dir in "$@"; do
  case=$(basename "$dir")
  for file in vtkhdf.vtkhdf xdmf.xmf pvtu.pvd; do
    for task in open one series; do
      for tool in mallard pvbatch; do
        if [ $tool = mallard ]; then cmd=(python3 "$HERE/read_mallard.py"); else
          cmd=("$PVBATCH" --force-offscreen-rendering "$HERE/read_pvbatch.py"); fi
        for i in $(seq "$N"); do
          out=$( { /usr/bin/time -f "TIME %e %M" "${cmd[@]}" "$(realpath "$dir/$file")" $task \
                   "npy/${case}_${file%%.*}_${task}_${tool}.npy"; } 2>&1 )
          read -r _ t check <<< "$(grep '^READ' <<< "$out" | tail -1)"
          read -r _ wall rss <<< "$(grep '^TIME' <<< "$out" | tail -1)"
          echo "$case,$file,$task,$tool,$i,${t:-failed},${check:-},$wall,$rss"
        done
      done
    done
  done
done
