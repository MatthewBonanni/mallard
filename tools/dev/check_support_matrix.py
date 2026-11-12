#!/usr/bin/env python3
"""Fails if the support matrix (docs/guide/support.md) or the supported-platforms table
of docs/installation.md disagrees with the GitHub workflows: every test or build job
must appear on each page as `<workflow>/<job>`, and every such reference must name a
job that exists. A job with a matrix of named entries (ci.yml's backends, CUDA, HIP and
SYCL, the UBI jobs) counts once per entry. Used by the first job of ci.yml.

    python3 tools/dev/check_support_matrix.py [WORKFLOW_DIR] [PAGE...]
"""
import json
import pathlib
import re
import sys

import yaml

# Jobs that only orchestrate, report or publish, and workflows that build no Mallard
NOT_TESTS = {
    "ci.yml": {"changelog", "device", "matrix", "result"},
    "release.yml": {"meta", "publish"},
    "batch.yml": {"stamp"},
}
SKIPPED_WORKFLOWS = {"pages.yml"}
REFERENCE = re.compile(r"`([\w-]+\.ya?ml/[\w.-]+)`")
JSON_LIST = re.compile(r"'(\[\s*\{.*?\}\s*\])'", re.DOTALL)


def from_json_names(jobs, expression):
    """The `name`s of the JSON list that `fromJSON(needs.JOB.outputs.X)` reads, as
    written literally in a step of JOB."""
    match = re.search(r"needs\.([\w-]+)\.outputs", expression)
    if not match:
        return None
    for step in jobs.get(match.group(1), {}).get("steps", []):
        for literal in JSON_LIST.findall(step.get("run", "")):
            return [entry["name"] for entry in json.loads(literal)]
    return None


def job_names(jobs, job_id):
    matrix = (jobs[job_id].get("strategy") or {}).get("matrix") or {}
    for value in matrix.values():
        if isinstance(value, list) and value and all(isinstance(v, dict) and "name" in v for v in value):
            return [v["name"] for v in value]
        if isinstance(value, str) and "fromJSON" in value:
            names = from_json_names(jobs, value)
            if names:
                return names
    return [job_id]


def workflow_jobs(directory):
    found = set()
    for path in sorted(directory.glob("*.y*ml")):
        if path.name in SKIPPED_WORKFLOWS:
            continue
        jobs = yaml.safe_load(path.read_text()).get("jobs", {})
        for job_id in jobs:
            if job_id not in NOT_TESTS.get(path.name, set()):
                found.update(f"{path.name}/{name}" for name in job_names(jobs, job_id))
    return found


def main():
    directory = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".github/workflows")
    pages = [pathlib.Path(p) for p in sys.argv[2:]] or [pathlib.Path("docs/guide/support.md"),
                                                       pathlib.Path("docs/installation.md")]
    jobs = workflow_jobs(directory)
    problems = []
    for page in pages:
        documented = set(REFERENCE.findall(page.read_text()))
        problems += [f"{job}: a CI job missing from {page}" for job in sorted(jobs - documented)]
        problems += [f"{job}: in {page}, but no workflow has this job" for job in sorted(documented - jobs)]
    for problem in problems:
        print(problem)
    if problems:
        print("Update the support matrix (docs/guide/support.md) with what the job builds and runs, and when,")
        print("and the supported-platforms table of docs/installation.md with its platform.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
