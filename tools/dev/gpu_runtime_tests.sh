#!/usr/bin/env bash
# GPU runtime checks of a CUDA build: the step graph and stream ordering tests under NVIDIA's
# compute-sanitizer (memcheck, racecheck and synccheck), on one GPU and on 2 ranks. --full also runs
# initcheck, more tests under each tool, and every MPI test on the ranks MPIEXEC starts, so that
# with ranks on several nodes NCCL's halo exchanges and collectives cross the network. Exits 1 if any
# check failed.
#
#   tools/dev/gpu_runtime_tests.sh [--full] BUILD_DIR
#
# MPIEXEC: the launcher of the MPI checks (default "mpiexec -n 2"); each rank takes the GPU of its
# rank on its node.
set -u

full=0
if [ "${1:-}" = --full ]; then
    full=1
    shift
fi
[ $# -eq 1 ] || { echo "usage: $0 [--full] BUILD_DIR" >&2; exit 2; }
build=$1
bin=
for name in mallard-test2d mallard-test3d; do
    [ -x "$build/test/$name" ] && bin=$build/test/$name
done
[ -n "$bin" ] || { echo "no test binary in $build/test" >&2; exit 2; }
command -v compute-sanitizer > /dev/null || { echo "compute-sanitizer is not on PATH" >&2; exit 2; }
read -r -a mpiexec <<< "${MPIEXEC:-mpiexec -n 2}"

serial='StepQueueTest.OutputsByStep:StepQueueTest.SnapshotsAndRestartsAreStagedWithoutStoppingTheDevice'
serial+=':StepQueueTest.FusedChemistryIsFlushedOnTheDeviceForWhatReadsTheState'
serial+=':StepQueueTest.RestartedRunsCaptureTheStepGraphAgainAndContinueBitForBit'
serial+=':StepQueueTest.AFailedStepGraphCaptureFallsBackToStepsKernelByKernel'
distributed='MPITest.StreamsWaitForEachOtherWhenOneRunsLate:StepQueueTest.DistributedRunsAsTheHostTakesThem'
tools=(memcheck racecheck synccheck)
if [ "$full" = 1 ]; then
    serial='StepQueueTest.*:StepQueueExample.*:RestartTest.*:IOTest.Asynchronous*'
    distributed+=':MPITest.*WithAndWithoutStepGraphs:MPI3DTest.*WithAndWithoutStepGraphs'
    tools+=(initcheck)
fi
mpi_tests='CommTest.*:MeshBlockTest.*:DistributedMeshTest.*:MPITest.*:MPI3DTest.*:StepQueueTest.Distributed*'

failed=()
check() {
    local what=$1 start=$SECONDS
    shift
    echo "== $what"
    if "$@"; then
        echo "== $what: passed ($((SECONDS - start)) s)"
    else
        echo "== $what: FAILED ($((SECONDS - start)) s)"
        failed+=("$what")
    fi
}

for tool in "${tools[@]}"; do
    sanitizer=(compute-sanitizer --tool "$tool" --error-exitcode 99 --print-limit 20)
    check "$tool, 1 GPU" "${sanitizer[@]}" "$bin" --gtest_color=no --gtest_filter="$serial"
    check "$tool, ${mpiexec[*]}" "${mpiexec[@]}" "${sanitizer[@]}" "$bin" --gtest_color=no \
        --gtest_filter="$distributed" --kokkos-map-device-id-by=mpi_rank
done
if [ "$full" = 1 ]; then
    check "MPI tests, ${mpiexec[*]}" "${mpiexec[@]}" "$bin" --gtest_color=no --gtest_filter="$mpi_tests" \
        --kokkos-map-device-id-by=mpi_rank
fi

if [ ${#failed[@]} -gt 0 ]; then
    printf 'failed: %s\n' "${failed[@]}"
    exit 1
fi
echo "all passed"
