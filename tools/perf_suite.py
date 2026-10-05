#!/usr/bin/env python3
"""Run the performance set (benchmarks/perf/suite.json) and compare it with baselines.

    perf_suite.py run --build-2d build --build-3d build3d [--launcher "mpirun -n 1"]
                      [--repeats 3] [--cases teno5_2d,h2o2] [--out perf.json] [--csv perf.csv]
    perf_suite.py compare perf.json [--hardware NAME] [--threshold 0.05]
    perf_suite.py update-baselines perf.json [--hardware NAME]

`run` runs each case once to warm up, then --repeats times, and records every
run's throughput (cells/s: the solver's time stepping, or the chemistry
benchmark's cells per second at each dt), the best and the spread
((best - worst) / best). `compare` reads benchmarks/perf/baselines.csv and
exits 1 if a metric is slower than its baseline by more than both the
threshold and the run's spread; metrics without a baseline are listed, not
failed. The hardware defaults to the one recorded in the results (the GPU
name from nvidia-smi or amd-smi, else the CPU model).
"""

import argparse
import csv
import json
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PERF_DIR = ROOT / "benchmarks" / "perf"
SUITE = PERF_DIR / "suite.json"
BASELINES = PERF_DIR / "baselines.csv"
SI = {"": 1.0, "k": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}


def detect_hardware():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0].strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        out = subprocess.run(["amd-smi", "static", "--asic", "--json"], capture_output=True, text=True, timeout=30)
        if out.returncode == 0:
            data = json.loads(out.stdout)
            gpus = data.get("gpu_data", []) if isinstance(data, dict) else data
            if gpus:
                return gpus[0]["asic"]["market_name"]
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, IndexError):
        pass
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def stage_input(case_input, work):
    """Copy the input into `work` with its mechanism path made absolute."""
    text = case_input.read_text()

    def absolute(match):
        path = Path(match.group(2))
        if not path.is_absolute():
            path = (case_input.parent / path).resolve()
        return f'{match.group(1)}"{path}"'

    text = re.sub(r'^(\s*mechanism\s*=\s*)"([^"]+)"', absolute, text, flags=re.M)
    staged = work / case_input.name
    staged.write_text(text)
    return staged


def run_once(case, binary, launcher, work, log=None):
    staged = stage_input((PERF_DIR / case["input"]).resolve(), work)
    cmd = launcher + [str(binary), "-i", staged.name]
    out = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
    if log:
        log.write_text(out.stdout + out.stderr)
    if out.returncode != 0:
        raise RuntimeError(f"{case['name']}: {shlex.join(cmd)} exited with {out.returncode}\n"
                           f"{out.stdout[-3000:]}\n{out.stderr[-3000:]}")
    if case["program"] == "Mallard":
        m = re.search(r"Throughput\s+([0-9.]+)([kMGT]?) cells/s", out.stdout)
        if not m:
            raise RuntimeError(f"{case['name']}: no throughput in the output\n{out.stdout[-3000:]}")
        return {case["name"]: float(m.group(1)) * SI[m.group(2)]}
    result = {}
    csv_files = sorted(work.glob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"{case['name']}: the chemistry benchmark wrote no CSV")
    with open(csv_files[0]) as f:
        for row in csv.DictReader(f):
            result[f"{case['name']}@dt={float(row['dt']):g}"] = float(row["cells_per_second"])
    for f in csv_files:
        f.unlink()
    return result


def cmd_run(args):
    suite = json.loads(SUITE.read_text())["cases"]
    if args.cases:
        wanted = set(args.cases.split(","))
        unknown = wanted - {c["name"] for c in suite}
        if unknown:
            sys.exit(f"unknown cases: {', '.join(sorted(unknown))}")
        suite = [c for c in suite if c["name"] in wanted]
    builds = {"2d": args.build_2d, "3d": args.build_3d}
    launcher = shlex.split(args.launcher) if args.launcher else []
    logs = Path(args.logs) if args.logs else None
    if logs:
        logs.mkdir(parents=True, exist_ok=True)
    runs = {}
    for case in suite:
        build = builds[case["build"]]
        if build is None:
            print(f"skip {case['name']}: no --build-{case['build']}", flush=True)
            continue
        binary = Path(build).resolve() / "src" / case["program"]
        with tempfile.TemporaryDirectory(prefix=f"perf-{case['name']}-") as tmp:
            for i in range(args.repeats + 1):
                log = logs / f"{case['name']}-{i}.log" if logs else None
                result = run_once(case, binary, launcher, Path(tmp), log)
                if i == 0:
                    continue  # warm-up
                for name, value in result.items():
                    runs.setdefault(name, []).append(value)
        for name in runs:
            if name == case["name"] or name.startswith(case["name"] + "@"):
                print(f"{name:24s} best {max(runs[name]):.4g} cells/s over {len(runs[name])} runs", flush=True)
    metrics = []
    for name, values in runs.items():
        best = max(values)
        metrics.append({"name": name, "unit": "cells/s", "best": best, "runs": values,
                        "spread": (best - min(values)) / best})
    results = {"hardware": args.hardware or detect_hardware(), "repeats": args.repeats, "metrics": metrics}
    Path(args.out).write_text(json.dumps(results, indent=2) + "\n")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["hardware", "benchmark", "cells_per_s", "spread", "runs"])
            for m in metrics:
                w.writerow([results["hardware"], m["name"], f"{m['best']:.4g}", f"{m['spread']:.4f}",
                            " ".join(f"{v:.4g}" for v in m["runs"])])
    print(f"wrote {args.out}")


def read_baselines(hardware):
    if not BASELINES.exists():
        return {}
    with open(BASELINES) as f:
        return {r["benchmark"]: float(r["cells_per_s"]) for r in csv.DictReader(f) if r["hardware"] == hardware}


def compare(results, hardware, threshold):
    """Rows (name, best, baseline, change, spread, status); status is ok, faster, SLOWER or new."""
    baselines = read_baselines(hardware)
    rows = []
    for m in results["metrics"]:
        base = baselines.get(m["name"])
        if base is None:
            rows.append((m["name"], m["best"], None, None, m["spread"], "new"))
            continue
        change = m["best"] / base - 1.0
        if -change > max(threshold, m["spread"]):
            status = "SLOWER"
        elif change > max(threshold, m["spread"]):
            status = "faster"
        else:
            status = "ok"
        rows.append((m["name"], m["best"], base, change, m["spread"], status))
    return rows


def cmd_compare(args):
    results = json.loads(Path(args.results).read_text())
    hardware = args.hardware or results["hardware"]
    rows = compare(results, hardware, args.threshold)
    print(f"{hardware}, threshold {args.threshold:.0%}")
    print(f"{'benchmark':24s} {'cells/s':>10s} {'baseline':>10s} {'change':>8s} {'spread':>7s}  status")
    for name, best, base, change, spread, status in rows:
        print(f"{name:24s} {best:10.4g} {base if base is not None else float('nan'):10.4g} "
              f"{change if change is not None else float('nan'):+8.1%} {spread:7.1%}  {status}")
    if args.json:
        Path(args.json).write_text(json.dumps(
            [dict(zip(("name", "best", "baseline", "change", "spread", "status"), r)) for r in rows], indent=2) + "\n")
    return 1 if any(r[5] == "SLOWER" for r in rows) else 0


def cmd_update(args):
    results = json.loads(Path(args.results).read_text())
    hardware = args.hardware or results["hardware"]
    kept = []
    if BASELINES.exists():
        with open(BASELINES) as f:
            kept = [r for r in csv.DictReader(f) if r["hardware"] != hardware]
    new = [{"hardware": hardware, "benchmark": m["name"], "cells_per_s": f"{m['best']:.4g}"}
           for m in results["metrics"]]
    with open(BASELINES, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["hardware", "benchmark", "cells_per_s"], lineterminator="\n")
        w.writeheader()
        w.writerows(kept + new)
    print(f"{BASELINES}: {len(new)} baselines for {hardware}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--build-2d")
    r.add_argument("--build-3d")
    r.add_argument("--launcher", help='command prefix, e.g. "mpirun -n 1" for MPI builds')
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--cases", help="comma-separated subset of the suite")
    r.add_argument("--hardware", help="override the detected hardware name")
    r.add_argument("--out", default="perf.json")
    r.add_argument("--csv")
    r.add_argument("--logs", help="directory for each run's output (run 0 is the warm-up)")
    c = sub.add_parser("compare")
    c.add_argument("results")
    c.add_argument("--hardware")
    c.add_argument("--threshold", type=float, default=0.05)
    c.add_argument("--json", help="write the comparison as JSON")
    u = sub.add_parser("update-baselines")
    u.add_argument("results")
    u.add_argument("--hardware")
    args = p.parse_args()
    if args.command == "run":
        if args.repeats < 1:
            p.error("--repeats must be at least 1")
        cmd_run(args)
        return 0
    if args.command == "compare":
        return cmd_compare(args)
    cmd_update(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
