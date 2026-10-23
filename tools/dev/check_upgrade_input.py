#!/usr/bin/env python3
"""Run `mallard upgrade-input` on every input file of every release.

    check_upgrade_input.py build/src/mallard [--tags v0.1.0,v0.8.0]

For each tag v* (git show TAG:path) and each .toml under examples/,
tutorials/ and benchmarks/, checks that upgrade-input succeeds, that its
output is valid TOML with nothing left to upgrade (`--check` passes on it),
and that a file with nothing to upgrade comes back byte for byte. Needs the
tags (git fetch --tags). Standard library only (Python 3.11+).
"""

import argparse
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIRS = ("examples/", "tutorials/", "benchmarks/")


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True, text=True).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mallard")
    parser.add_argument("--tags", help="comma-separated (default: every v* tag)")
    args = parser.parse_args()
    tags = args.tags.split(",") if args.tags else git("tag", "--list", "v*").split()
    if not tags:
        sys.exit("check_upgrade_input.py: no tags (git fetch --tags)")
    failures, files, upgraded = [], 0, 0
    with tempfile.TemporaryDirectory() as tmp:
        for tag in tags:
            for path in git("ls-tree", "-r", "--name-only", tag).split():
                if not (path.endswith(".toml") and path.startswith(DIRS)):
                    continue
                text = git("show", f"{tag}:{path}")
                try:
                    tomllib.loads(text)
                except tomllib.TOMLDecodeError:
                    continue  # templates and fragments that are not TOML on their own
                files += 1
                old, new = Path(tmp) / "old.toml", Path(tmp) / "new.toml"
                old.write_text(text)
                run = subprocess.run([args.mallard, "upgrade-input", str(old)], capture_output=True, text=True)
                name = f"{tag}:{path}"
                if run.returncode != 0:
                    failures.append(f"{name}: exit {run.returncode}: {run.stderr.strip()}")
                    continue
                new.write_text(run.stdout)
                try:
                    tomllib.loads(run.stdout)
                except tomllib.TOMLDecodeError as e:
                    failures.append(f"{name}: the output is not TOML: {e}")
                    continue
                again = subprocess.run([args.mallard, "upgrade-input", "--check", str(new)],
                                       capture_output=True, text=True)
                if again.returncode != 0:
                    failures.append(f"{name}: the output still has something to upgrade: {again.stderr.strip()}")
                if run.stderr.strip():
                    upgraded += 1
                elif run.stdout != text:
                    failures.append(f"{name}: nothing to upgrade, but the output differs from the input")
    print(f"{files} input files of {len(tags)} releases, {upgraded} upgraded, {len(failures)} failures")
    for f in failures:
        print("  " + f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
