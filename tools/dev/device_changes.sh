#!/usr/bin/env bash
# Whether the changes since BASE touch code that the CUDA build runs on the device:
# a changed or deleted source file under src/ with Kokkos device constructs, or a new
# version of the bundled Kokkos. Prints "true" or "false" and, on stderr,
# the files that decided it. Used by CI (the CUDA build jobs) and before GPU CI.
#
#   tools/dev/device_changes.sh [BASE]    (default: the merge base with origin/main)
set -euo pipefail

base=${1:-$(git merge-base origin/main HEAD)}
pattern='KOKKOS_|Kokkos::(parallel_|View|TeamPolicy|RangePolicy|MDRangePolicy|deep_copy|atomic_|Experimental::)'

found=()
while IFS=$'\t' read -r status path; do
    case "$path" in
        src/external/kokkos) found+=("$path") ;;
        src/*.cpp | src/*.h | src/*.hpp | src/*.inc)
            if [ "$status" = D ] || grep -qE "$pattern" "$path"; then found+=("$path"); fi ;;
    esac
done < <(git diff --name-status --no-renames "$base" HEAD -- src)

if [ ${#found[@]} -gt 0 ]; then
    printf '%s\n' "${found[@]}" >&2
    echo true
else
    echo false
fi
