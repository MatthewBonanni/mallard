#!/usr/bin/env python3
"""Check CHANGELOG.md and generate release notes from it.

    changelog.py notes 0.9.0       print a release's section (or "Unreleased")
    changelog.py check [BASE]      exit 1 if the branch changes src/ without
                                   adding a bullet under "## [Unreleased]"

`check` compares HEAD with its merge base with BASE (default origin/main) and
also checks the file's structure: an Unreleased section, release headings
`## [X.Y.Z] - YYYY-MM-DD`, and only Keep a Changelog subsections. Standard
library only.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHANGELOG = ROOT / "CHANGELOG.md"
CATEGORIES = {"Added", "Changed", "Deprecated", "Removed", "Fixed", "Security"}
RELEASE = re.compile(r"^## \[(\d+\.\d+\.\d+)\] - \d{4}-\d{2}-\d{2}$")


def sections(lines):
    """(name, first line index, end index) of each `## ` section."""
    starts = [i for i, line in enumerate(lines) if line.startswith("## ")]
    out = []
    for k, i in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(lines)
        m = RELEASE.match(lines[i])
        if m:
            name = m.group(1)
        elif lines[i] == "## [Unreleased]":
            name = "Unreleased"
        else:
            name = lines[i][3:]
        out.append((name, i, end))
    return out


def structure_errors(lines):
    errors = []
    secs = sections(lines)
    if not secs or secs[0][0] != "Unreleased":
        errors.append('the first section must be "## [Unreleased]"')
    for name, start, end in secs:
        if name == "Earlier releases":
            continue
        if name != "Unreleased" and not RELEASE.match(lines[start]):
            errors.append(f'line {start + 1}: expected "## [X.Y.Z] - YYYY-MM-DD", got "{lines[start]}"')
        for i in range(start + 1, end):
            if lines[i].startswith("### ") and lines[i][4:] not in CATEGORIES:
                errors.append(f"line {i + 1}: unknown subsection \"{lines[i][4:]}\" "
                              f"(use {', '.join(sorted(CATEGORIES))})")
    return errors


def notes(version):
    lines = CHANGELOG.read_text().splitlines()
    for name, start, end in sections(lines):
        if name == version.removeprefix("v"):
            body = lines[start + 1:end]
            while body and not body[-1].strip():
                body.pop()
            while body and not body[0].strip():
                body.pop(0)
            body = [line for line in body if not re.match(r"^\[[^\]]+\]: ", line)]
            print("\n".join(body))
            return 0
    print(f"changelog.py: no section for {version} in CHANGELOG.md", file=sys.stderr)
    return 1


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True,
                          text=True).stdout


def check(base):
    lines = CHANGELOG.read_text().splitlines()
    errors = structure_errors(lines)
    for e in errors:
        print(f"CHANGELOG.md: {e}", file=sys.stderr)

    merge_base = git("merge-base", base, "HEAD").strip()
    changed = git("diff", "--name-only", f"{merge_base}...HEAD").split()
    if not any(f.startswith("src/") for f in changed):
        print("No changes under src/: no changelog entry needed.")
        return 1 if errors else 0

    unreleased = next(((s, e) for n, s, e in sections(lines) if n == "Unreleased"), None)
    added = []
    line_no = 0
    for line in git("diff", "-U0", f"{merge_base}...HEAD", "--", "CHANGELOG.md").splitlines():
        hunk = re.match(r"^@@ -\S+ \+(\d+)", line)
        if hunk:
            line_no = int(hunk.group(1))
        elif line.startswith("+") and not line.startswith("+++"):
            added.append((line_no - 1, line[1:]))
            line_no += 1
    entries = [text for i, text in added
               if unreleased and unreleased[0] < i < unreleased[1] and text.startswith("- ")]
    if entries:
        print(f"Changelog entries under Unreleased: {len(entries)}")
        return 1 if errors else 0
    print("This branch changes src/ but adds no bullet under \"## [Unreleased]\" in "
          "CHANGELOG.md.\nAdd one (Added, Changed, Deprecated, Removed or Fixed; say what a "
          "user notices and what to do about it), or label the pull request "
          "\"no-changelog\" if users notice nothing (refactors, tests, comments).",
          file=sys.stderr)
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("notes")
    p.add_argument("version")
    p = sub.add_parser("check")
    p.add_argument("base", nargs="?", default="origin/main")
    args = parser.parse_args()
    return notes(args.version) if args.cmd == "notes" else check(args.base)


if __name__ == "__main__":
    sys.exit(main())
