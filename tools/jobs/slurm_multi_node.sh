#!/bin/bash
# Mallard on several GPU nodes under Slurm.
#
#   cp slurm_multi_node.sh <case directory> && cd <case directory>
#   (edit the lines marked <...>)
#   sbatch slurm_multi_node.sh
#
# One MPI rank per GPU. Each rank gets the GPU nearest its cores automatically;
# the "Ranks" table at the top of the log shows where each rank runs.
# Five minutes before the time limit, Mallard writes a restart and stops, and
# the job submits its continuation, which resumes from that restart.
# Run it from a directory every node sees (a shared file system).
# CPU-only nodes: drop --gpus-per-node and set --ntasks-per-node x
# --cpus-per-task to the node's cores.

#SBATCH --job-name=mallard
#SBATCH --nodes=4                  # <nodes>
#SBATCH --ntasks-per-node=4        # <GPUs per node>: one rank per GPU
#SBATCH --gpus-per-node=4          # <GPUs per node>
#SBATCH --cpus-per-task=16         # <cores per node / ranks per node>
#SBATCH --time=24:00:00            # <time limit>
#SBATCH --signal=B:USR2@300        # stop 300 s before the limit (longer than a step plus a restart write)
#SBATCH --output=mallard-%j.out
##SBATCH --partition=<partition>
##SBATCH --account=<account>

set -u

MALLARD=mallard                    # <mallard, mallard3d, or the path to the executable>
INPUT=input.toml
RESTARTS='restart_*.restart'       # <the restart files the run writes; the newest is resumed from>
MAX_JOBS=20                        # resubmit at most this many times

# module load <the compiler, MPI and CUDA that Mallard was built with>

# Halo exchanges go between GPUs directly with a GPU-aware MPI (Mallard built
# with -DMallard_GPU_AWARE_MPI=ON) or NCCL (-DMallard_ENABLE_NCCL=ON). Some
# MPIs need to be told: <uncomment the line for yours>
# export MPICH_GPU_SUPPORT_ENABLED=1           # Cray MPICH
# export OMPI_MCA_pml=ucx                      # Open MPI over InfiniBand

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=spread OMP_PLACES=threads

args=(-i "$INPUT")
latest=$(ls -1t $RESTARTS 2>/dev/null | head -n 1)
[ -n "$latest" ] && args+=(--restart "$latest")

# Every rank sees every GPU of the node (no --gpus-per-task): Mallard picks
# one per rank, and GPU-aware MPI can copy between them directly
srun --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=cores "$MALLARD" "${args[@]}" &
pid=$!

# Time limit (USR2 from --signal) or scancel (TERM): ask Mallard to write a
# restart and stop after the current step
at_limit=0
trap 'at_limit=1; kill -USR2 $pid 2>/dev/null' USR2
trap 'kill -USR2 $pid 2>/dev/null' TERM
wait $pid
status=$?
while kill -0 $pid 2>/dev/null; do
    wait $pid
    status=$?
done

# Exit status 0: finished; 1: error or divergence; 3: stopped on request
job=${MALLARD_JOB:-1}
if [ "$status" -eq 3 ] && [ "$at_limit" -eq 1 ] && [ "$job" -lt "$MAX_JOBS" ]; then
    script=$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^ *Command=\([^ ]*\).*/\1/p')
    sbatch --export=ALL,MALLARD_JOB=$((job + 1)) "$script"
fi
exit "$status"
