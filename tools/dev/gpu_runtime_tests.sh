#!/usr/bin/env bash
# GPU runtime checks of a CUDA build, on one GPU and on 2 ranks:
#  - the step graph and stream ordering tests (graph capture and replay, restarts, asynchronous output,
#    streams held back after each wait), with step graphs on;
#  - NVIDIA's compute-sanitizer (memcheck, racecheck, synccheck) on a representative set of tests, with
#    MALLARD_TEST_CUDA_GRAPHS=off: compute-sanitizer misreports kernels in the step graph (#1076), so the
#    steps run kernel by kernel, with the same results.
# --full also runs initcheck, memcheck on the whole suite but the chemistry reactor tests, and every MPI
# test on the ranks MPIEXEC starts, so that with ranks on several nodes NCCL's halo exchanges and collectives
# cross the network. Exits 1 if any check failed.
#
#   tools/dev/gpu_runtime_tests.sh [--full] BUILD_DIR
#
# MPIEXEC: the launcher of the MPI checks (default "mpiexec -n 2"; "none" skips them, e.g. on one
# GPU, which NCCL cannot share between ranks); each rank takes the GPU of its rank on its node.
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

graphs='StepQueueTest.*:*StepQueueExample.*:IOTest.Asynchronous*:ForcedInletTest.ConditionsOfTKeepTheStepGraph'
graphs+=':InsituStatisticsTest.SpectraAndPhasesAreSampledInTheStepGraph'
graphs_distributed='MPITest.StreamsWaitForEachOtherWhenOneRunsLate:StepQueueTest.Distributed*'
graphs_distributed+=':MPITest.*WithAndWithoutStepGraphs:MPI3DTest.*WithAndWithoutStepGraphs'
# A few tests per kernel family (schemes, boundaries, chemistry, LES, statistics, output): minutes under racecheck
sanitized='StepQueueTest.OutputsByStep:StepQueueTest.FusedChemistryIsFlushedOnTheDeviceForWhatReadsTheState'
sanitized+=':StepQueueTest.SnapshotsAndRestartsAreStagedWithoutStoppingTheDevice'
sanitized+=':StepQueueTest.RestartedRunsCaptureTheStepGraphAgainAndContinueBitForBit'
sanitized+=':StepQueueTest.DerivedFieldsAndRatesAreSampledWithoutStoppingTheDevice'
sanitized+=':InsituStatisticsTest.SpectraAndPhasesAreSampledInTheStepGraph'
sanitized+=':IOTest.AsynchronousOutputWritesTheSameFiles'
sanitized+=':MPITest.TENOOnQuadsMatchesSerial:MPITest.LargeEddySimulationMatchesSerialWithAndWithoutStepGraphs'
sanitized+=':MPITest.CharacteristicBoundariesAndSpongesMatchSerialWithAndWithoutStepGraphs'
sanitized_distributed='MPITest.StreamsWaitForEachOtherWhenOneRunsLate:StepQueueTest.DistributedRunsAsTheHostTakesThem'
tools=(memcheck racecheck synccheck)
if [ "$full" = 1 ]; then
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
mpi() { [ "${mpiexec[*]}" != none ]; }
# The test binary only: the runs it starts (the mallard executable) keep their step graphs. racecheck's warnings
# (hazards it does not rate as errors) include Kokkos' CUDA reductions of values over 16 bytes: every lane of the
# first warp stores the result, and thread 0 reads it for final() with no __syncwarp between. memcheck and
# initcheck skip the CUDA API errors NCCL and Open MPI make and handle (the suppressions file); memcheck
# reports explicit API errors only, as the driver's log lines ("extended" errors) cannot be suppressed
suppressions=$(cd "$(dirname "$0")" && pwd)/compute_sanitizer_suppressions.xml
sanitizer() {
    sanitize=(env MALLARD_TEST_CUDA_GRAPHS=off compute-sanitizer --tool "$1" --target-processes application-only
              --error-exitcode 99 --print-limit 20)
    case $1 in
        memcheck) sanitize+=(--report-api-errors explicit --suppressions "$suppressions") ;;
        initcheck) sanitize+=(--suppressions "$suppressions") ;;
        racecheck) sanitize+=(--racecheck-detect-level error) ;;
    esac
}

check "step graphs, 1 GPU" "$bin" --gtest_color=no --gtest_filter="$graphs"
mpi && check "step graphs, ${mpiexec[*]}" "${mpiexec[@]}" "$bin" --gtest_color=no \
    --gtest_filter="$graphs_distributed" --kokkos-map-device-id-by=mpi_rank
for tool in "${tools[@]}"; do
    sanitizer "$tool"
    check "$tool, 1 GPU" "${sanitize[@]}" "$bin" --gtest_color=no --gtest_filter="$sanitized"
    mpi && check "$tool, ${mpiexec[*]}" "${mpiexec[@]}" "${sanitize[@]}" "$bin" --gtest_color=no \
        --gtest_filter="$sanitized_distributed" --kokkos-map-device-id-by=mpi_rank
done
if [ "$full" = 1 ]; then
    sanitizer memcheck
    # The chemistry reactor tests integrate thousands of reactors: hours under memcheck
    check "memcheck, whole suite, 1 GPU" "${sanitize[@]}" "$bin" --gtest_color=no \
        --gtest_filter='-ChemistryReactorTest.*:ChemistryBenchmarkTest.*'
    mpi && check "MPI tests, ${mpiexec[*]}" "${mpiexec[@]}" "$bin" --gtest_color=no --gtest_filter="$mpi_tests" \
        --kokkos-map-device-id-by=mpi_rank
fi

if [ ${#failed[@]} -gt 0 ]; then
    printf 'failed: %s\n' "${failed[@]}"
    exit 1
fi
echo "all passed"
