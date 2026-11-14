#!/usr/bin/env python3
"""Changelog fragments: check them, preview them, fold them into CHANGELOG.md.

Each pull request that changes src/ adds a fragment
changelog/<name>.<category>.md (category: added, changed, deprecated,
removed, fixed or security) holding one or more "- " bullets, instead of
editing CHANGELOG.md, so that parallel pull requests do not conflict.

    changelog.py check [BASE]          exit 1 if the branch changes src/ without
                                       adding a fragment (or a fragment is malformed)
    changelog.py notes VERSION         print a release's section; "Unreleased"
                                       previews CHANGELOG.md's Unreleased plus the fragments
    changelog.py release X.Y.Z [DATE]  fold Unreleased and the fragments into
                                       "## [X.Y.Z] - DATE" (default today) and delete the fragments

`check` compares HEAD with its merge base with BASE (default origin/main) and
also checks CHANGELOG.md's structure. Standard library only.
"""

import argparse
import datetime
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHANGELOG = ROOT / "CHANGELOG.md"
FRAGMENTS = ROOT / "changelog"
CATEGORIES = ["Added", "Changed", "Deprecated", "Removed", "Fixed", "Security"]
RELEASE = re.compile(r"^## \[(\d+\.\d+\.\d+)\] - \d{4}-\d{2}-\d{2}$")
FRAGMENT = re.compile(r"^[A-Za-z0-9_.-]+\.(" + "|".join(c.lower() for c in CATEGORIES) + r")\.md$")
REPO = "https://github.com/MatthewBonanni/mallard"


def sections(lines):
    """(name, first line index, end index) of each `## ` section."""
    starts = [i for i, line in enumerate(lines) if line.startswith("## ")]
    out = []
    for k, i in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(lines)
        m = RELEASE.match(lines[i])
        name = m.group(1) if m else "Unreleased" if lines[i] == "## [Unreleased]" else lines[i][3:]
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
                errors.append(f'line {i + 1}: unknown subsection "{lines[i][4:]}" (use {", ".join(CATEGORIES)})')
    return errors


def fragment_errors(path, text):
    if not FRAGMENT.match(path.name):
        return [f"{path.name}: name it <name>.<category>.md, category one of "
                f"{', '.join(c.lower() for c in CATEGORIES)}"]
    if not text.strip().startswith("- "):
        return [f"{path.name}: write one or more Markdown bullets starting with \"- \""]
    return []


def fragments():
    """{category: [bullet lines]} of changelog/, in file-name order."""
    out = {c: [] for c in CATEGORIES}
    for path in sorted(FRAGMENTS.glob("*.md")):
        if path.name == "README.md":
            continue
        m = FRAGMENT.match(path.name)
        if m:
            out[m.group(1).capitalize()] += path.read_text().strip().splitlines()
    return out


def merged_unreleased(lines):
    """{category: [lines]} of CHANGELOG.md's Unreleased section plus the fragments."""
    out = {c: [] for c in CATEGORIES}
    start, end = next((s, e) for n, s, e in sections(lines) if n == "Unreleased")
    category = None
    for line in lines[start + 1:end]:
        if line.startswith("### "):
            category = line[4:]
        elif category and line.strip():
            out[category].append(line)
    for c, bullets in fragments().items():
        out[c] += bullets
    return out


def render(categories):
    body = []
    for c in CATEGORIES:
        if categories[c]:
            body += [f"### {c}", ""] + categories[c] + [""]
    return body


def notes(version):
    lines = CHANGELOG.read_text().splitlines()
    if version == "Unreleased":
        print("\n".join(render(merged_unreleased(lines))).rstrip())
        return 0
    for name, start, end in sections(lines):
        if name == version.removeprefix("v"):
            body = [line for line in lines[start + 1:end] if not re.match(r"^\[[^\]]+\]: ", line)]
            print("\n".join(body).strip())
            return 0
    print(f"changelog.py: no section for {version} in CHANGELOG.md", file=sys.stderr)
    return 1


def release(version, date):
    lines = CHANGELOG.read_text().splitlines()
    if any(name == version for name, _, _ in sections(lines)):
        print(f"changelog.py: CHANGELOG.md already has {version}", file=sys.stderr)
        return 1
    body = render(merged_unreleased(lines))
    if not body:
        print("changelog.py: nothing to release", file=sys.stderr)
        return 1
    start, end = next((s, e) for n, s, e in sections(lines) if n == "Unreleased")
    new = lines[:start] + ["## [Unreleased]", "", f"## [{version}] - {date}", ""] + body + lines[end:]
    link = next((i for i, line in enumerate(new) if line.startswith("[Unreleased]: ")), None)
    if link is not None:
        previous = re.search(r"compare/(v[\d.]+)\.\.\.HEAD", new[link])
        new[link] = f"[Unreleased]: {REPO}/compare/v{version}...HEAD"
        if previous:
            new.insert(link + 1, f"[{version}]: {REPO}/compare/{previous.group(1)}...v{version}")
    CHANGELOG.write_text("\n".join(new) + "\n")
    for path in FRAGMENTS.glob("*.md"):
        if path.name != "README.md":
            path.unlink()
    print(f"CHANGELOG.md: released {version}; fragments removed")
    return 0


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def check(base):
    errors = [f"CHANGELOG.md: {e}" for e in structure_errors(CHANGELOG.read_text().splitlines())]
    merge_base = git("merge-base", base, "HEAD").strip()
    status = [line.split("\t") for line in git("diff", "--name-status", f"{merge_base}...HEAD").splitlines()]
    changed = [entry[-1] for entry in status]
    added = [ROOT / entry[-1] for entry in status
             if entry[0][0] in "AR" and entry[-1].startswith("changelog/") and entry[-1] != "changelog/README.md"]
    for path in added:
        errors += fragment_errors(path, git("show", f"HEAD:{path.relative_to(ROOT).as_posix()}"))
    for e in errors:
        print(e, file=sys.stderr)
    if not any(f.startswith("src/") for f in changed):
        print("No changes under src/: no changelog fragment needed.")
    elif added:
        print(f"Changelog fragments: {', '.join(p.name for p in added)}")
    else:
        print("This branch changes src/ but adds no changelog fragment.\n"
              "Add changelog/<branch>.<added|changed|deprecated|removed|fixed|security>.md with one "
              "\"- \" bullet per change: what a user notices and what to do about it, with the PR or "
              "issue number. If users notice nothing (refactors, tests, comments), label the pull "
              "request \"no-changelog\". See docs/design/documentation.md#changelog.", file=sys.stderr)
        return 1
    return 1 if errors else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check")
    p.add_argument("base", nargs="?", default="origin/main")
    p = sub.add_parser("notes")
    p.add_argument("version")
    p = sub.add_parser("release")
    p.add_argument("version")
    p.add_argument("date", nargs="?", default=datetime.date.today().isoformat())
    args = parser.parse_args()
    if args.cmd == "check":
        return check(args.base)
    if args.cmd == "notes":
        return notes(args.version)
    return release(args.version, args.date)


if __name__ == "__main__":
    sys.exit(main())
