#!/bin/bash
# Mallard under PBS Pro or Torque.
#
#   cp pbs.sh <case directory> && cd <case directory>
#   (edit the lines marked <...>)
#   qsub pbs.sh
#
# One MPI rank per GPU. Each rank gets the GPU nearest its cores automatically;
# the "Ranks" table at the top of the log shows where each rank runs.
# Five minutes before the walltime, Mallard writes a restart and stops, and the
# job submits its continuation, which resumes from that restart.
# CPU-only nodes: drop ngpus and set mpiprocs x ompthreads to the node's cores.

#PBS -N mallard
#PBS -l select=2:ncpus=64:ngpus=4:mpiprocs=4:ompthreads=16
#     <nodes>:ncpus=<cores per node>:ngpus=<GPUs per node>:mpiprocs=<GPUs per node>:ompthreads=<cores / GPUs>
#     Torque instead: #PBS -l nodes=2:ppn=64:gpus=4
#PBS -l walltime=24:00:00
#PBS -j oe
##PBS -q <queue>
##PBS -A <account>

set -u
cd "$PBS_O_WORKDIR" || exit 1

MALLARD=mallard                    # <mallard, mallard3d, or the path to the executable>
INPUT=input.toml
SCRIPT=pbs.sh                      # <this file's name, to submit the continuation>
RESTARTS='restart_*.restart'       # <the restart files the run writes; the newest is resumed from>
STOP_BEFORE=300                    # seconds before the walltime to stop (longer than a step plus a restart write)
MAX_JOBS=20                        # resubmit at most this many times

# module load <the compiler, MPI and CUDA that Mallard was built with>

nodes=$(sort -u "$PBS_NODEFILE" | wc -l)
ranks_per_node=4                   # <GPUs per node>
threads=16                         # <cores per node / ranks per node>
export OMP_NUM_THREADS=$threads OMP_PROC_BIND=spread OMP_PLACES=threads

rm -f .at_limit
args=(-i "$INPUT")
latest=$(ls -1t $RESTARTS 2>/dev/null | head -n 1)
[ -n "$latest" ] && args+=(--restart "$latest")

# Open MPI. Cray PALS: mpiexec -n N --ppn $ranks_per_node --depth $threads --cpu-bind depth.
# MPICH (Hydra): mpiexec -n N -ppn $ranks_per_node -bind-to core:$threads
mpirun -n $((nodes * ranks_per_node)) --map-by "ppr:$ranks_per_node:node:PE=$threads" --bind-to core \
    "$MALLARD" "${args[@]}" &
pid=$!

# Ask Mallard to write a restart and stop after the current step: through its
# control socket (rank 0 runs on this node), as some mpirun wrappers (the NVIDIA
# HPC SDK's) pass no signal on; else (before the time loop opens the socket)
# with USR2, which mpirun forwards
stop() { "$MALLARD" ctl . stop >/dev/null 2>&1 || kill -USR2 $pid 2>/dev/null; }

# PBS sends TERM only seconds before it kills the job, too late for a restart:
# stop ahead of the walltime
walltime=$(qstat -f "$PBS_JOBID" 2>/dev/null | sed -n 's/^ *Resource_List.walltime = //p')
if [ -n "$walltime" ]; then
    IFS=: read -r h m s <<< "$walltime"
    ( sleep $(( 10#$h * 3600 + 10#$m * 60 + 10#$s - STOP_BEFORE )) && touch .at_limit && stop ) \
        2>/dev/null &
    timer=$!
fi
trap 'stop &' TERM
wait $pid
status=$?
while kill -0 $pid 2>/dev/null; do
    wait $pid
    status=$?
done
[ -n "${timer:-}" ] && kill "$timer" 2>/dev/null

# Exit status 0: finished; 1: error or divergence; 3: stopped on request
job=${MALLARD_JOB:-1}
if [ "$status" -eq 3 ] && [ -e .at_limit ] && [ "$job" -lt "$MAX_JOBS" ]; then
    qsub -v MALLARD_JOB=$((job + 1)) "$SCRIPT"
fi
rm -f .at_limit
exit "$status"
