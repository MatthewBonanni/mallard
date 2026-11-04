#!/usr/bin/env python3
"""Parse every ```toml snippet of the Markdown docs (docs/, README.md, examples/, tutorials/).

    python tools/dev/check_doc_toml.py

Exits 1 and lists the file and line of each snippet that is not valid TOML.
Standard library only (Python 3.11+).
"""
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SNIPPET = re.compile(r"^```toml\n(.*?)^```", re.S | re.M)


def main():
    files = [ROOT / "README.md"] + sorted(p for d in ("docs", "examples", "tutorials") for p in (ROOT / d).rglob("*.md"))
    snippets, failures = 0, []
    for path in files:
        text = path.read_text()
        for m in SNIPPET.finditer(text):
            snippets += 1
            try:
                tomllib.loads(m.group(1))
            except tomllib.TOMLDecodeError as e:
                failures.append(f"{path.relative_to(ROOT)}:{text.count(chr(10), 0, m.start()) + 1}: {e}")
    print(f"{snippets} TOML snippets, {len(failures)} invalid")
    for f in failures:
        print("  " + f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
