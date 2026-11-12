#!/usr/bin/env python3
"""The coverage map of the test suite: for each googletest test, the repository files it
depends on, so CI can rerun only the tests a change can affect.

    python3 tools/dev/test_coverage.py run --build BUILD --dim 2 --out map2d.json
        [--unit test|suite] [--jobs N] [--launcher "mpirun -n 1"] [--ranks N --rank-pattern PAT]
        [--llvm-cov llvm-cov] [--llvm-profdata llvm-profdata]
    python3 tools/dev/test_coverage.py merge --out test-coverage.json map2d.json map3d.json

`run` needs a build with Clang's source-based coverage (-fprofile-instr-generate
-fcoverage-mapping). It runs each test (or each suite) in its own process, with every
process it starts writing its own profile, and records the source files with an executed
line. A test then depends on:
- those files, and every header the translation units among them include (from the
  build's dependency files), so a change to a constant or a type in a header counts too;
- the data files that any of them names (`"test/data/meshes"`, `"examples" / file`):
  a path named in a string, or joined from strings with `/`, as a file or a directory;
- the files of RULES["global"] (the build system, the CI workflow, the submodules).
With --ranks, the tests matching --rank-pattern also run on that many ranks, and the
files of both runs count.

The gate (outside this repository) reads the merged map: a changed file decides the tests
that rerun when the map can answer for it, and the whole suite otherwise: a file no test
depends on that is neither in RULES["none"] nor built only in the other dimension, a new
file, or a test source that differs from the map's (its tests may have changed).
"""
import argparse
import concurrent.futures
import fnmatch
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

# Paths that every test depends on, and paths that no test reads
RULES = {
    "global": ["CMakeLists.txt", "*/CMakeLists.txt", "cmake/*", ".gitmodules", ".github/workflows/ci.yml",
               "src/external/*", "test/test_main.cpp", "test/data_paths.h"],
    "none": [".github/*", ".gitignore", ".gitattributes", ".dockerignore", "*.md", "CITATION.cff", "LICENSE",
             "NOTICE", "licenses/*", "changelog/*", "docker/*", "spack/*", "benchmarks/*", "tutorials/*",
             "tools/*", "test/fuzz/*", "test/features.toml", "docs/*"],
}
# Directories whose files tests read as data when the sources name them
DATA_ROOTS = ("test/data", "examples", "templates", "tutorials", "docs", "mechanisms", "tools", "benchmarks")
# Files whose data names are not reads (data_paths.h only checks that test/data exists)
REFS_IGNORED = {"test/data_paths.h"}
TEST_MACRO = re.compile(r"^\s*(?:TEST|TEST_F|TEST_P|TYPED_TEST|TYPED_TEST_P)\(\s*(\w+)\s*,", re.M)
STRING = re.compile(r'"((?:[^"\\\n]|\\.)*)"')
JOINED = re.compile(r'"((?:[^"\\\n]|\\.)*)"\s*/\s*"((?:[^"\\\n]|\\.)*)"')


def git_files(src):
    out = subprocess.run(["git", "-C", src, "ls-files"], capture_output=True, text=True, check=True).stdout
    return set(out.split())


def data_refs(text, tracked):
    """The data paths a source names: files, or directories (ending in /)."""
    while True:
        joined = JOINED.sub(lambda m: '"' + m.group(1).rstrip("/") + "/" + m.group(2).lstrip("/") + '"', text)
        if joined == text:
            break
        text = joined
    refs = set()
    for literal in STRING.findall(text):
        for word in re.split(r"[\s\"'`(),;]+", literal.replace("\\n", " ").replace('\\"', " ")):
            word = word.split("#")[0].strip("/.:")
            for root in DATA_ROOTS:
                i = word.find(root)
                if i < 0 or (i > 0 and word[i - 1] != "/") or word[i + len(root):i + len(root) + 1] not in ("", "/"):
                    continue
                path = word[i:].rstrip("/")
                if path in tracked:
                    refs.add(path)
                    continue
                # A directory, or a prefix completed at run time: its longest existing directory
                while path and not any(f.startswith(path + "/") for f in tracked):
                    path = path.rsplit("/", 1)[0] if "/" in path else ""
                if path:
                    refs.add(path + "/")
    return sorted(refs)


def parse_depfiles(build, src, tracked):
    """{translation unit: [repository headers it includes]} from the build's .d files."""
    graph = {}
    src_real = os.path.realpath(src)
    for root, _, files in os.walk(build):
        for name in files:
            if not name.endswith(".o.d"):
                continue
            with open(os.path.join(root, name), errors="replace") as f:
                text = f.read().replace("\\\n", " ")
            _, _, deps = text.partition(": ")
            paths = []
            for dep in deps.split():
                full = os.path.realpath(dep if os.path.isabs(dep) else os.path.join(build, dep))
                if full.startswith(src_real + os.sep):
                    rel = os.path.relpath(full, src_real)
                    if rel in tracked and not rel.startswith("src/external/"):
                        paths.append(rel)
            if paths and paths[0].endswith((".cpp", ".cc", ".c", ".cu")):
                graph.setdefault(paths[0], set()).update(p for p in paths[1:])
    return {tu: sorted(h) for tu, h in graph.items()}


def list_tests(launcher, binary):
    out = subprocess.run(launcher + [binary, "--gtest_list_tests"], capture_output=True, text=True, check=True).stdout
    tests, suite = [], None
    for line in out.splitlines():
        if not line.strip() or line.startswith(("Running", "[")):
            continue
        if not line.startswith(" "):
            suite = line.split("#")[0].strip()
        else:
            tests.append(suite + line.split("#")[0].strip())
    return tests


def executed(args, unit, launchers, tmp, src_real, tracked):
    """The repository files the unit's processes executed, and whether all its runs passed."""
    d = tempfile.mkdtemp(dir=tmp)
    env = dict(os.environ, LLVM_PROFILE_FILE=os.path.join(d, "%p.profraw"))
    ok = True
    for launcher in launchers:
        r = subprocess.run(launcher + [args.binary, f"--gtest_filter={unit}", "--gtest_brief=1"], cwd=d, env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ok = ok and r.returncode == 0
    raws = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".profraw")]
    files = set()
    if raws:
        prof = os.path.join(d, "merged.profdata")
        subprocess.run(args.llvm_profdata + ["merge", "-sparse", "-o", prof] + raws, check=True,
                       stdout=subprocess.DEVNULL)
        out = subprocess.run(args.llvm_cov + ["export", "-summary-only", f"-instr-profile={prof}", args.binary]
                             + [x for o in args.objects for x in ("-object", o)],
                             capture_output=True, text=True, check=True).stdout
        for entry in json.loads(out)["data"][0]["files"]:
            if entry["summary"]["lines"]["covered"] > 0:
                full = os.path.realpath(entry["filename"])
                rel = os.path.relpath(full, src_real) if full.startswith(src_real + os.sep) else None
                if rel in tracked and not rel.startswith("src/external/"):
                    files.add(rel)
    shutil.rmtree(d, ignore_errors=True)
    return files, ok


def cmd_run(args):
    src = os.path.realpath(args.source)
    tracked = git_files(src)
    args.binary = os.path.join(args.build, "test", f"mallard-test{args.dim}d")
    args.objects = [os.path.join(args.build, "src", "mallard")]
    args.llvm_cov, args.llvm_profdata = shlex.split(args.llvm_cov), shlex.split(args.llvm_profdata)
    launcher = shlex.split(args.launcher)
    tests = list_tests(launcher, args.binary)
    if args.unit == "suite":
        units = sorted({t.split(".")[0] + ".*" for t in tests})
    else:
        units = tests
    many = shlex.split(args.launcher.replace("-n 1", f"-n {args.ranks}") + " --oversubscribe") if args.ranks else None
    graph = parse_depfiles(args.build, src, tracked)
    tmp = tempfile.mkdtemp(prefix="test-coverage-")
    start = time.time()
    results = {}

    def one(unit):
        launchers = [launcher]
        if many and any(fnmatch.fnmatchcase(unit.split("/")[-1], p) or fnmatch.fnmatchcase(unit, p)
                        for p in args.rank_pattern.split(":")):
            launchers.append(many)
        return unit, executed(args, unit, launchers, tmp, src, tracked)

    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        for i, (unit, res) in enumerate(pool.map(one, units), 1):
            results[unit] = res
            if i % 50 == 0 or i == len(units):
                print(f"{i}/{len(units)} units, {time.time() - start:.0f} s", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)

    def expand(files):
        return files | {h for f in files for h in graph.get(f, ())}

    unit_files = {u: expand(f) for u, (f, _) in results.items()}
    per_test = {t: unit_files[t.split(".")[0] + ".*" if args.unit == "suite" else t] for t in tests}
    build_files = set(graph) | {h for hs in graph.values() for h in hs}
    names = sorted(set().union(*per_test.values(), build_files)) if per_test else sorted(build_files)
    index = {f: i for i, f in enumerate(names)}
    refs = {}
    for f in sorted(tracked):
        if f.startswith(("src/", "test/")) and f.endswith((".cpp", ".h", ".hpp")) and f not in REFS_IGNORED \
                and not f.startswith(("src/external/", "test/fuzz/")):
            with open(os.path.join(src, f), errors="replace") as fh:
                r = data_refs(fh.read(), tracked)
            if r:
                refs[f] = r
    suites = {}
    for f in sorted(tracked):
        if f.startswith("test/") and f.endswith((".cpp", ".h")) and not f.startswith("test/fuzz/"):
            with open(os.path.join(src, f), errors="replace") as fh:
                for s in TEST_MACRO.findall(fh.read()):
                    suites.setdefault(s, set()).add(f)
    out = {"dim": args.dim, "unit": args.unit, "seconds": round(time.time() - start),
           "files": names, "build_files": sorted(index[f] for f in build_files),
           "tests": {t: sorted(index[f] for f in fs) for t, fs in sorted(per_test.items())},
           "failed": sorted(u for u, (_, ok) in results.items() if not ok),
           "refs": refs, "suites": {s: sorted(fs) for s, fs in sorted(suites.items())}}
    with open(args.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"{len(tests)} tests in {len(units)} units, {out['seconds']} s; {len(out['failed'])} failed; {args.out}")
    return 0


def cmd_merge(args):
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    tree = subprocess.run(["git", "rev-parse", "HEAD^{tree}"], capture_output=True, text=True, check=True).stdout.strip()
    dims = {}
    for path in args.maps:
        with open(path) as f:
            m = json.load(f)
        dims[str(m["dim"])] = m
    with open(args.out, "w") as f:
        json.dump({"version": 1, "commit": commit, "tree": tree, "rules": RULES, "data_roots": DATA_ROOTS,
                   "dims": dims}, f, separators=(",", ":"))
    print(f"{args.out}: {', '.join(f'{d}D {len(m['tests'])} tests' for d, m in sorted(dims.items()))} at {commit}")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--build", required=True)
    r.add_argument("--source", default=".")
    r.add_argument("--dim", type=int, choices=(2, 3), required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--unit", choices=("test", "suite"), default="test")
    r.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    r.add_argument("--launcher", default="", help='e.g. "mpirun -n 1" for an MPI build')
    r.add_argument("--ranks", type=int, help="also run the tests matching --rank-pattern on this many ranks")
    r.add_argument("--rank-pattern", default="*")
    r.add_argument("--llvm-cov", default="llvm-cov")
    r.add_argument("--llvm-profdata", default="llvm-profdata")
    m = sub.add_parser("merge")
    m.add_argument("--out", required=True)
    m.add_argument("maps", nargs="+")
    args = p.parse_args()
    return cmd_run(args) if args.cmd == "run" else cmd_merge(args)


if __name__ == "__main__":
    sys.exit(main())
