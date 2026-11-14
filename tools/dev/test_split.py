#!/usr/bin/env python3
"""Splits the tests a CI job selects into shards of about equal duration: longest first,
each test to the shard with the least estimated time so far (ties to the lower shard).
The split depends only on the test list and the durations, so every shard of a backend,
given the same build and the same durations, computes the same split on its own. Every
selected test lands in exactly one shard (checked on every split).

    test_split.py times DIR BACKEND [FALLBACK]
        Prints the backend's durations, {test: seconds}: the longest time of each test in
        the JUnit reports of DIR/junit-BACKEND and DIR/junit-BACKEND-<i>of<n>/ (artifacts of a
        ci.yml run), or else the backend's entry of FALLBACK (tools/dev/test_times.json).
    test_split.py split TIMES N [I] < LIST
        LIST is the output of `--gtest_list_tests`. Prints shard I's googletest filter (1 to N,
        -* for none), or with no I every non-empty shard's (-* if none), one per line, longest
        first. A test without a duration counts as TIMES["*"], or else the median of TIMES.
        A summary goes to stderr.
    test_split.py refresh [RUN...]
        Rewrites tools/dev/test_times.json from the JUnit reports of the newest passed ci.yml
        runs on main (or the runs given), each backend from the newest run that has it (gh).
        It keeps the tests of 0.1 s or more; "*" is the mean of the others.
"""
import hashlib
import json
import pathlib
import re
import statistics
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

TIMES = pathlib.Path(__file__).resolve().parent / "test_times.json"
SHARD = re.compile(r"-\d+of\d+$")


def parse_list(text):
    """Full test names of `--gtest_list_tests` output, in its order."""
    tests, suite = [], None
    for line in text.splitlines():
        if not line.strip():
            continue
        first = line.split()[0]
        if not line[0].isspace():
            suite = first if first.endswith(".") else None
        elif suite:
            tests.append(suite + first)
    return tests


def report_times(paths):
    """{test: seconds}, the longest time of each test that ran in these JUnit reports."""
    times = {}
    for path in paths:
        for case in ET.parse(path).getroot().iter("testcase"):
            if case.get("status") == "notrun":
                continue
            name = f"{case.get('classname')}.{case.get('name')}"
            times[name] = max(times.get(name, 0.0), float(case.get("time") or 0))
    return times


def backend_reports(root, backend):
    """The JUnit reports of the backend's jobs among the artifacts under root."""
    root = pathlib.Path(root)
    dirs = [d for d in sorted(root.iterdir()) if d.is_dir() and SHARD.sub("", d.name) == f"junit-{backend}"] \
        if root.is_dir() else []
    return [p for d in dirs for p in sorted(d.glob("*.xml"))]


def split(tests, times, n):
    """The tests in n shards of about equal duration: (shards, estimated ms of each)."""
    if n < 1:
        raise ValueError(f"{n} shards")
    if len(set(tests)) != len(tests):
        raise ValueError("a test is listed twice")
    known = [round(1000 * s) for t, s in times.items() if t != "*"]
    default = round(1000 * times["*"]) if "*" in times else round(statistics.median(known)) if known else 1000
    cost = {t: round(1000 * times[t]) if t in times else default for t in tests}
    shards, load = [[] for _ in range(n)], [0] * n
    for t in sorted(tests, key=lambda t: (-cost[t], t)):
        i = min(range(n), key=lambda i: (load[i], i))
        shards[i].append(t)
        load[i] += cost[t]
    placed = [t for s in shards for t in s]
    if len(placed) != len(tests) or set(placed) != set(tests):
        raise AssertionError("the shards do not hold every test exactly once")
    return [sorted(s) for s in shards], load


def gtest_filter(tests):
    return ":".join(tests) if tests else "-*"


def digest(shards):
    return hashlib.sha256("\n".join(gtest_filter(s) for s in shards).encode()).hexdigest()[:12]


def cmd_times(root, backend, fallback=None):
    reports = backend_reports(root, backend)
    if reports:
        times, source = report_times(reports), f"{len(reports)} JUnit reports"
    else:
        table = json.loads(pathlib.Path(fallback).read_text()) if fallback else {}
        times, source = table.get(backend, {}), f"{fallback}" if backend in table else "none"
    print(f"{backend}: durations of {len(times)} tests from {source}", file=sys.stderr)
    json.dump({t: round(s, 3) for t, s in sorted(times.items())}, sys.stdout, indent=0)
    print()


def cmd_split(times_path, n, index=None):
    times = json.loads(pathlib.Path(times_path).read_text())
    tests = parse_list(sys.stdin.read())
    shards, load = split(tests, times, n)
    unknown = sum(t not in times for t in tests)
    print(f"split {digest(shards)}: {len(tests)} tests ({unknown} not in {times_path}) in {n} shards, "
          f"estimated {', '.join(f'{ms / 1000:.0f}' for ms in load)} s", file=sys.stderr)
    if index is not None:
        if not 1 <= index <= n:
            raise ValueError(f"shard {index} of {n}")
        print(gtest_filter(shards[index - 1]))
        return
    for i in sorted(range(n), key=lambda i: (-load[i], i)):
        if shards[i] or i == 0 and not tests:
            print(gtest_filter(shards[i]))


def gh(*args):
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout


def cmd_refresh(runs):
    if not runs:
        runs = gh("run", "list", "--workflow", "ci.yml", "--branch", "main", "--status", "success",
                  "--limit", "30", "--json", "databaseId", "-q", ".[].databaseId").split()
    table = json.loads(TIMES.read_text()) if TIMES.exists() else {}
    found = set()
    for run in runs:
        names = gh("api", f"repos/{{owner}}/{{repo}}/actions/runs/{run}/artifacts?per_page=100",
                   "-q", ".artifacts[].name").split()
        backends = {SHARD.sub("", n[len("junit-"):]) for n in names if n.startswith("junit-")} - found
        if not backends:
            continue
        with tempfile.TemporaryDirectory() as d:
            gh("run", "download", run, "-p", "junit-*", "-D", d)
            for b in sorted(backends):
                times = report_times(backend_reports(d, b))
                short = [s for s in times.values() if s < 0.1]
                table[b] = {t: round(s, 1) for t, s in sorted(times.items()) if s >= 0.1}
                table[b]["*"] = round(statistics.mean(short), 3) if short else 0.0
                print(f"{b}: {len(table[b])} tests from run {run}", file=sys.stderr)
        found |= backends
    TIMES.write_text(json.dumps(dict(sorted(table.items())), indent=1, sort_keys=True) + "\n")


def main(argv):
    if len(argv) >= 3 and argv[0] == "times" and len(argv) <= 4:
        cmd_times(*argv[1:])
    elif len(argv) in (3, 4) and argv[0] == "split":
        cmd_split(argv[1], int(argv[2]), int(argv[3]) if len(argv) == 4 else None)
    elif argv and argv[0] == "refresh":
        cmd_refresh(argv[1:])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
