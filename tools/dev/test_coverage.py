#!/usr/bin/env python3
"""The coverage map of the test suite: for each googletest test, the repository files it
depends on, so CI can rerun only the tests a change can affect.

    python3 tools/dev/test_coverage.py run --build BUILD --dim 2 --out map2d.json
        [--unit test|suite] [--jobs N] [--launcher "mpirun -n 1"] [--ranks N --rank-pattern PAT]
        [--llvm-cov llvm-cov] [--llvm-profdata llvm-profdata]
    python3 tools/dev/test_coverage.py merge --out test-coverage.json.gz map2d.json map3d.json

`run` needs a build with Clang's source-based coverage (-fprofile-instr-generate
-fcoverage-mapping). It runs each test (or each suite) in its own process, with every
process it starts writing its own profile, and records the source files with an executed
line. A test then depends on:
- those files, and the headers they include directly, so a change to a constant or a type
  in a header counts too (a header included only through another header counts for the
  tests that execute a line of it or of a file that includes it directly);
- the data files that a file it executes names (`"test/data/meshes"`, `"examples" / file`):
  a path named in a string, or joined from strings with `/`, as a file or a directory;
- the files of RULES["global"] (the build system, the CI workflow, the submodules).
It also records the functions each test executes, as (file, first line, last line) spans of
the instrumented functions (the function records of `llvm-cov export`), so a change inside
a function body reruns only the tests that executed that function.
With --ranks, the tests matching --rank-pattern also run on that many ranks, and the
files of both runs count.

The gate (outside this repository) reads the merged map (gzip JSON). A change confined to
function bodies reruns the tests that executed those functions; any other change to a file
(declarations, types, constants, macros, a function's signature or first line) reruns the
tests that depend on the file. A file no test depends on that is neither in RULES["none"]
nor built only in the other dimension, a new file, or a test source that differs from the
map's (its tests may have changed) needs the whole suite.
"""
import argparse
import concurrent.futures
import fnmatch
import gzip
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


INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.M)


def direct_includes(src, tracked):
    """{source: [repository headers its #include "..." lines name]}."""
    by_name = {}
    for f in tracked:
        by_name.setdefault(os.path.basename(f), []).append(f)
    out = {}
    for f in tracked:
        if not f.startswith(("src/", "test/")) or not f.endswith((".cpp", ".h", ".hpp")) \
                or f.startswith("src/external/"):
            continue
        with open(os.path.join(src, f), errors="replace") as fh:
            names = INCLUDE.findall(fh.read())
        found = set()
        for inc in names:
            local = os.path.normpath(os.path.join(os.path.dirname(f), inc))
            if local in tracked:
                found.add(local)
                continue
            found.update(c for c in by_name.get(os.path.basename(inc), [])
                         if c.endswith("/" + inc) and not c.startswith("src/external/"))
        if found:
            out[f] = sorted(found)
    return out


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


def repo_path(path, src_real, tracked):
    full = os.path.realpath(path)
    rel = os.path.relpath(full, src_real) if full.startswith(src_real + os.sep) else None
    return rel if rel in tracked and not rel.startswith("src/external/") else None


def coverage_args(args, prof):
    return [f"-instr-profile={prof}", args.binary] + [x for o in args.objects for x in ("-object", o)]


def executed(args, unit, launchers, tmp, src_real, tracked):
    """The repository files and functions ((file, name)) the unit's processes executed, and whether
    all its runs passed."""
    d = tempfile.mkdtemp(dir=tmp)
    env = dict(os.environ, LLVM_PROFILE_FILE=os.path.join(d, "%p.profraw"))
    ok = True
    for launcher in launchers:
        r = subprocess.run(launcher + [args.binary, f"--gtest_filter={unit}", "--gtest_brief=1"], cwd=d, env=env,
                           capture_output=True, text=True, errors="replace")
        if r.returncode != 0:
            print(f"{unit} failed ({r.returncode}):\n{(r.stdout + r.stderr)[-600:]}", flush=True)
            ok = False
    raws = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".profraw")]
    files, functions = set(), set()
    if raws:
        prof = os.path.join(d, "merged.profdata")
        subprocess.run(args.llvm_profdata + ["merge", "-sparse", "-o", prof] + raws, check=True,
                       stdout=subprocess.DEVNULL)
        spans = os.path.join(tmp, "spans.profdata")
        if not os.path.exists(spans):
            shutil.copy(prof, spans + f".{os.getpid()}.{unit.__hash__()}")
            os.replace(spans + f".{os.getpid()}.{unit.__hash__()}", spans)
        out = subprocess.run(args.llvm_cov + ["export", "-format=lcov", "-skip-expansions"] + coverage_args(args, prof),
                             capture_output=True, text=True, check=True, errors="replace").stdout
        rel = None
        for line in out.splitlines():
            if line.startswith("SF:"):
                rel = repo_path(line[3:], src_real, tracked)
            elif rel is None:
                continue
            elif line.startswith("FNDA:"):
                count, name = line[5:].split(",", 1)
                if int(count) > 0:
                    functions.add((rel, name))
            elif line.startswith("LH:") and int(line[3:]) > 0:
                files.add(rel)
    shutil.rmtree(d, ignore_errors=True)
    return files, functions, ok


def function_spans(args, prof, src_real, tracked):
    """{(file, name): (first line, last line)} of every instrumented function of the build."""
    out = subprocess.run(args.llvm_cov + ["export", "-skip-expansions"] + coverage_args(args, prof),
                         capture_output=True, text=True, check=True, errors="replace").stdout
    spans = {}
    for fn in json.loads(out)["data"][0]["functions"]:
        rel = repo_path(fn["filenames"][0], src_real, tracked)
        lines = [(r[0], r[2]) for r in fn["regions"] if r[5] == 0]
        if rel and lines:
            span = (min(a for a, _ in lines), max(b for _, b in lines))
            old = spans.get((rel, fn["name"]))
            spans[(rel, fn["name"])] = span if old is None else (min(old[0], span[0]), max(old[1], span[1]))
    return spans


def cmd_run(args):
    src = os.path.realpath(args.source)
    tracked = git_files(src)
    args.build = os.path.realpath(args.build)
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
    includes = direct_includes(src, tracked)
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
    spans = function_spans(args, os.path.join(tmp, "spans.profdata"), src, tracked)
    shutil.rmtree(tmp, ignore_errors=True)

    def expand(files):
        return files | {h for f in files for h in includes.get(f, ())}

    unit_files = {u: expand(f) for u, (f, _, _) in results.items()}
    unit_of = {t: t.split(".")[0] + ".*" if args.unit == "suite" else t for t in tests}
    per_test = {t: unit_files[unit_of[t]] for t in tests}
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
    # Data a test reads: named in a file it executes (a header's name counts only where its code runs)
    order = sorted(per_test)
    by_ref = {}
    for i, t in enumerate(order):
        for r in {r for f in results[unit_of[t]][0] for r in refs.get(f, ())}:
            by_ref.setdefault(r, []).append(i)
    suites = {}
    for f in sorted(tracked):
        if f.startswith("test/") and f.endswith((".cpp", ".h")) and not f.startswith("test/fuzz/"):
            with open(os.path.join(src, f), errors="replace") as fh:
                for s in TEST_MACRO.findall(fh.read()):
                    suites.setdefault(s, set()).add(f)
    # Functions as unique (file, first line, last line) spans; per test the spans it executed
    span_ids, functions = {}, []
    for (rel, _), (a, b) in sorted(spans.items()):
        if rel in index and (rel, a, b) not in span_ids:
            span_ids[(rel, a, b)] = len(functions)
            functions.append([index[rel], a, b])
    test_functions = {}
    for t in order:
        ids = sorted({span_ids[(f, *spans[(f, n)])] for f, n in results[unit_of[t]][1]
                      if (f, *spans.get((f, n), (0, 0))) in span_ids})
        test_functions[t] = encode(ids)
    out = {"dim": args.dim, "unit": args.unit, "seconds": round(time.time() - start),
           "functions": functions, "test_functions": test_functions,
           "files": names, "build_files": sorted(index[f] for f in build_files),
           "tests": {t: sorted(index[f] for f in fs) for t, fs in sorted(per_test.items())},
           "failed": sorted(u for u, (_, _, ok) in results.items() if not ok),
           "data": by_ref, "suites": {s: sorted(fs) for s, fs in sorted(suites.items())}}
    with open(args.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"{len(tests)} tests in {len(units)} units, {out['seconds']} s; {len(out['failed'])} failed; {args.out}")
    return 0


def encode(ids):
    """Sorted ids as the base-36 gaps between them, space-separated: a quarter of the JSON list."""
    out, prev = [], -1
    for i in ids:
        out.append(base36(i - prev))
        prev = i
    return " ".join(out)


def base36(n):
    digits = ""
    while True:
        n, r = divmod(n, 36)
        digits = "0123456789abcdefghijklmnopqrstuvwxyz"[r] + digits
        if not n:
            return digits


def cmd_merge(args):
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    tree = subprocess.run(["git", "rev-parse", "HEAD^{tree}"], capture_output=True, text=True, check=True).stdout.strip()
    dims = {}
    for path in args.maps:
        with open(path) as f:
            m = json.load(f)
        dims[str(m["dim"])] = m
    with gzip.open(args.out, "wt") as f:
        json.dump({"version": 2, "commit": commit, "tree": tree, "rules": RULES, "data_roots": DATA_ROOTS,
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
