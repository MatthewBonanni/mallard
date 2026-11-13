#!/usr/bin/env python3
"""Fails if a page of docs/ that the tests read is not a test input to ci.yml: its push
trigger and the docs-only rule of its device job, which skip documentation alone. The
pages the tests read: those test sources open under source_dir() or data_dirs(), and
those of diagnostic_code::ALL (check_test). Used by the first job of ci.yml.

    python3 tools/dev/check_tested_docs.py
"""
import pathlib
import re
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
JOINED = re.compile(r'"docs"((?:\s*/\s*"[^"]+")+)')
SLASHED = re.compile(r'"/(docs/[^"#]+)"')
PAGE = re.compile(r'_PAGE = "(docs/[^"#]+)"')
RULE = re.compile(r"grep -qE '(\^docs/[^']*)'")


def tested():
    found = set()
    for path in sorted((ROOT / "test").glob("*.cpp")):
        text = path.read_text()
        found |= {"docs/" + "/".join(re.findall(r'"([^"]+)"', m)) for m in JOINED.findall(text)}
        found |= set(SLASHED.findall(text))
    found |= set(PAGE.findall((ROOT / "src/common/diagnostic_code.h").read_text()))
    return sorted(p + "/" if (ROOT / p).is_dir() else p for p in found)


def problems(workflow):
    text = workflow.read_text()
    rule = RULE.search(text)
    if not rule:
        return [f"{workflow}: no docs-only rule for the pages the tests read (grep -qE '^docs/...')"]
    doc = yaml.safe_load(text)
    paths = doc.get("on", doc.get(True))["push"]["paths"]
    found = []
    for page in tested():
        if not re.search(rule.group(1), page + ("x" if page.endswith("/") else "")):
            found.append(f"{workflow}: the device job's docs-only rule skips {page}, which the tests read")
        if (page + "**" if page.endswith("/") else page) not in paths:
            found.append(f"{workflow}: the push trigger skips {page}, which the tests read")
    return found


def main():
    found = problems(ROOT / ".github/workflows/ci.yml")
    for problem in found:
        print(problem)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
