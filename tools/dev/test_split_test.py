#!/usr/bin/env python3
"""Tests of tools/dev/test_split.py (run by tools/dev/precheck.sh)."""
import contextlib
import io
import pathlib
import random
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_split  # noqa: E402


def run_split(times, listing, *args):
    with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
        f.write(__import__("json").dumps(times))
        f.flush()
        out = io.StringIO()
        with mock.patch("sys.stdin", io.StringIO(listing)), contextlib.redirect_stdout(out), \
                contextlib.redirect_stderr(io.StringIO()):
            test_split.cmd_split(f.name, *args)
    return out.getvalue().splitlines()


class Split(unittest.TestCase):
    def test_long_tests_go_to_different_shards(self):
        # clang-23's two round-robin shards took 15.0 and 5.9 min: googletest's sharding put the
        # tests of 204 s and 94 s, two apart in its order, in the same shard
        tests = [f"S.T{i:02}" for i in range(40)]
        times = {t: 3.0 for t in tests} | {"S.T00": 204.0, "S.T02": 94.0}
        shards, load = test_split.split(tests, times, 2)
        self.assertNotEqual("S.T00" in shards[0], "S.T02" in shards[0])
        self.assertEqual(sorted(load), [205000, 207000])

    def test_every_test_once_and_the_same_split_for_any_order(self):
        rng = random.Random(1)
        for n in (1, 2, 3, 4, 7):
            tests = [f"Suite{i % 5}/P.Test/{i}" for i in range(200)]
            times = {t: rng.choice([0.0, 0.01, 0.5, 2.0, 2.0, 30.0]) for t in tests[:150]}
            shards, load = test_split.split(tests, times, n)
            self.assertEqual(sorted(t for s in shards for t in s), sorted(tests))
            shuffled = tests[:]
            rng.shuffle(shuffled)
            self.assertEqual(test_split.split(shuffled, dict(reversed(list(times.items()))), n), (shards, load))

    def test_a_test_without_a_duration_counts_as_the_median(self):
        times = {"A.a": 9.0, "A.b": 1.0, "A.c": 4.0}
        _, load = test_split.split(["A.a", "A.new1", "A.new2"], times, 2)
        self.assertEqual(load, [9000, 8000])
        _, load = test_split.split(["A.a", "A.new1", "A.new2"], times | {"*": 0.5}, 2)
        self.assertEqual(load, [9000, 1000])

    def test_no_test_selected(self):
        shards, load = test_split.split([], {"A.a": 1.0}, 3)
        self.assertEqual((shards, load), ([[], [], []], [0, 0, 0]))
        self.assertEqual(run_split({}, "", 3, 2), ["-*"])
        self.assertEqual(run_split({}, "", 3), ["-*"])

    def test_shards_of_a_listing(self):
        listing = ("A.\n  x\n  y\nP/B.  # TypeParam = float\n  z/0  # GetParam() = 1\n"
                   "  z/1  # GetParam() = 2\n")
        times = {"A.x": 5.0, "A.y": 2.0, "P/B.z/0": 2.0, "P/B.z/1": 2.0}
        self.assertEqual(run_split(times, listing, 2, 1), ["A.x"])
        self.assertEqual(run_split(times, listing, 2, 2), ["A.y:P/B.z/0:P/B.z/1"])
        self.assertEqual(run_split(times, listing, 2), ["A.y:P/B.z/0:P/B.z/1", "A.x"])
        with self.assertRaises(ValueError):
            run_split(times, listing, 2, 3)


class Times(unittest.TestCase):
    def test_reports_of_the_backend_only(self):
        report = ('<testsuites><testsuite name="A"><testcase name="x" classname="A" status="run" time="{}"/>'
                  '<testcase name="y" classname="A" status="notrun" time="0"/></testsuite></testsuites>')
        with tempfile.TemporaryDirectory() as d:
            for job, t in (("mpi-1of2", 7), ("mpi-2of2", 3), ("mpi-3d-1of4", 99), ("mpi", 1)):
                (pathlib.Path(d) / f"junit-{job}").mkdir()
                (pathlib.Path(d) / f"junit-{job}" / "1.xml").write_text(report.format(t))
            self.assertEqual(test_split.report_times(test_split.backend_reports(d, "mpi")), {"A.x": 7.0})
            self.assertEqual(test_split.report_times(test_split.backend_reports(d, "mpi-3d")), {"A.x": 99.0})


if __name__ == "__main__":
    unittest.main()
