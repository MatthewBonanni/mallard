#!/usr/bin/env python3
"""Fails if the `only` input of a ci.yml dispatch names a job it doesn't know, so a
typo or a job's display name (cuda-2d) fails the run instead of silently skipping
jobs. The valid names are the backends of the matrix job, tutorials, device and fuzz.
Used by the first job of ci.yml.

    python3 tools/dev/check_ci_only.py ONLY [WORKFLOW]
"""
import pathlib
import sys

import yaml

from check_support_matrix import from_json_names

OTHERS = ["tutorials", "device", "fuzz"]
HINTS = {"cuda": "device", "hip": "device", "sycl": "device", "fuzz": "fuzz"}


def valid_names(workflow):
    jobs = yaml.safe_load(workflow.read_text())["jobs"]
    return from_json_names(jobs, "needs.matrix.outputs.backends") + OTHERS


def main():
    only = sys.argv[1]
    workflow = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else ".github/workflows/ci.yml")
    valid = valid_names(workflow)
    unknown = [name for name in only.split(",") if name and name not in valid]
    for name in unknown:
        hint = HINTS.get(name.split("-")[0])
        print(f"only: unknown job '{name}'" + (f" (did you mean '{hint}'?)" if hint else ""))
    if unknown:
        print(f"Valid names, comma-separated without spaces: {', '.join(valid)}")
    return 1 if unknown else 0


if __name__ == "__main__":
    sys.exit(main())
