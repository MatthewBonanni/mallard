#!/usr/bin/env python3
"""Fails if a GitHub workflow runs on pull requests or on pushes to branches other
than main: CI on branches runs only by hand (gh workflow run --ref <branch>), and
release workflows run on tags. Used by the first job of ci.yml.

    python3 tools/check_workflow_triggers.py [WORKFLOW_DIR]
"""
import pathlib
import sys

import yaml

ALLOWED_BRANCHES = {"main"}
FORBIDDEN_EVENTS = {"pull_request", "pull_request_target"}


def problems(path):
    doc = yaml.safe_load(path.read_text())
    on = doc.get("on", doc.get(True))  # YAML 1.1 reads a bare `on` as true
    if isinstance(on, str):
        on = {on: None}
    elif isinstance(on, list):
        on = {event: None for event in on}
    found = [f"{path}: runs on {event}" for event in FORBIDDEN_EVENTS & set(on or {})]
    push = (on or {}).get("push", "absent")
    if push is None:
        found.append(f"{path}: runs on every push (give push: branches: [main] or tags)")
    elif isinstance(push, dict):
        branches = push.get("branches")
        if branches is None and "tags" not in push:
            found.append(f"{path}: runs on pushes to any branch")
        for branch in branches or []:
            if branch not in ALLOWED_BRANCHES:
                found.append(f"{path}: runs on pushes to branch '{branch}'")
    return found


def main():
    directory = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".github/workflows")
    found = [p for path in sorted(directory.glob("*.y*ml")) for p in problems(path)]
    for problem in found:
        print(problem)
    if found:
        print("CI runs on branches only by hand (gh workflow run --ref <branch>); releases run on tags.")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
