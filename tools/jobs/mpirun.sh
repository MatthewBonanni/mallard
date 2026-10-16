#!/bin/bash
# Mallard on one machine (a workstation, or a node you are logged in to) with mpirun.
#
#   cp mpirun.sh <case directory> && cd <case directory>
#   ./mpirun.sh                    # one rank per GPU, or per 8 cores without GPUs
#   NRANKS=2 THREADS=4 ./mpirun.sh # or choose ranks and threads per rank
#
# Ctrl-C (or kill -TERM) writes a restart and stops after the current step; run
# the script again to resume from the newest restart. A second Ctrl-C stops at once.
# Each rank gets the GPU nearest its cores automatically; the "Ranks" table at
# the top of the log shows where each rank runs.

set -u

MALLARD=${MALLARD:-Mallard}        # <Mallard, Mallard3D, or the path to the executable>
INPUT=${INPUT:-input.toml}
RESTARTS='restart_*.restart'       # <the restart files the run writes; the newest is resumed from>

# Physical cores (not hardware threads), which mpirun binds ranks to
cores=$(lscpu -p=core,socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)
[ "$cores" -gt 0 ] || cores=$(sysctl -n hw.physicalcpu 2>/dev/null || getconf _NPROCESSORS_ONLN)
gpus=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)
[ "$gpus" -eq 0 ] && gpus=$(rocm-smi --showid 2>/dev/null | grep -c '^GPU\[' || true)
if [ "$gpus" -gt 0 ]; then
    NRANKS=${NRANKS:-$gpus}
else
    NRANKS=${NRANKS:-$(( cores >= 8 ? cores / 8 : 1 ))}
fi
threads=${THREADS:-$(( cores / NRANKS > 0 ? cores / NRANKS : 1 ))}
export OMP_NUM_THREADS=$threads OMP_PROC_BIND=spread OMP_PLACES=threads

args=(-i "$INPUT")
latest=$(ls -1t $RESTARTS 2>/dev/null | head -n 1)
[ -n "$latest" ] && args+=(--restart "$latest")

# Open MPI: each rank on $threads consecutive cores. MPICH: replace the
# options by "-bind-to core:$threads". macOS cannot bind processes to cores.
bind=(--map-by "slot:PE=$threads" --bind-to core)
[ "$(uname)" = Darwin ] && bind=(--bind-to none)
set -m  # mpirun in a process group of its own, so that Ctrl-C reaches only this script
mpirun -n "$NRANKS" "${bind[@]}" "$MALLARD" "${args[@]}" --kokkos-num-threads="$threads" &
pid=$!

# mpirun ends the ranks at once on INT and TERM, but forwards USR2, which
# makes Mallard write a restart and stop after the current step
stops=0
stop() {
    stops=$((stops + 1))
    if [ "$stops" -eq 1 ]; then
        echo "Stopping after the current step (Ctrl-C again to stop now)" >&2
        kill -USR2 $pid 2>/dev/null
    else
        kill -TERM $pid 2>/dev/null
    fi
}
trap stop INT TERM
wait $pid
status=$?
while kill -0 $pid 2>/dev/null; do
    wait $pid
    status=$?
done
# Exit status 0: finished; 1: error or divergence; 3: stopped on request
exit "$status"
