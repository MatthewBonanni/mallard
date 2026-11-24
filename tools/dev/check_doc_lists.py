#!/usr/bin/env python3
"""Check the lists of the Markdown pages the website builds for what its Markdown breaks.

    python tools/dev/check_doc_lists.py

GitHub follows CommonMark; the website's Python-Markdown is stricter, and renders these
differently, as one paragraph with literal "- " where GitHub shows a list:

- a code block, paragraph, table or list nested in a list item and indented by less than 4
  spaces past the item's marker (GitHub needs only the marker's width): it ends the list,
  or flattens a nested list into its parent;
- a list item right after a nested block that follows a blank line: it runs on into
  that block;
- a list right after a paragraph, with no blank line between: it runs on into the
  paragraph.

Exits 1 and lists the file and line of each. Standard library only.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ITEM = re.compile(r"^( *)(?:[-*+]|\d+\.) ")
FENCE = re.compile(r"^ *(```|~~~)")
INTERRUPTS = re.compile(r"^ *(?:[-*+]|1\.) ")  # on GitHub, a list that can follow a paragraph line
NOT_PARAGRAPH = re.compile(r"^ *(#|\||<|\$\$|!\[|```|~~~)")


def indent(line):
    return len(line) - len(line.lstrip(" "))


def check(text):
    """Yield (line number, message) for each list block of `text` that Python-Markdown breaks."""
    items = []  # marker indents of the open list items, outermost first
    fence = None
    blank = True
    block = None  # indent of the nested block begun after a blank line, until the next blank
    paragraph = False  # the previous line is text of a paragraph (not of a list item's first one)
    for n, line in enumerate(text.splitlines(), 1):
        if fence:
            if line.strip().startswith(fence):
                fence = None
            continue
        if not line.strip():
            blank = True
            continue
        ind = indent(line)
        item = ITEM.match(line)
        if item and paragraph and not blank and not INTERRUPTS.match(line):
            item = None  # a number ending a sentence at a line break, text on both sites
        opens_fence = FENCE.match(line)
        if item and not blank:
            if block is not None and ind < block:
                yield n, "list item right after a nested block: add a blank line before it"
            elif paragraph:
                yield n, "list right after a paragraph: add a blank line before it"
        if item and block is not None and ind < block:
            block = None
        while items and (ind < items[-1] or (ind == items[-1] and (item or blank))):
            items.pop()
        if blank:
            block = ind if items and ind > items[-1] else None
        nested = bool(items) and items[-1] < ind < items[-1] + 4 and bool(blank or opens_fence or item)
        if nested:
            what = "nested list" if item else "code block" if opens_fence else "table" if line.lstrip().startswith("|") else "paragraph"
            yield n, f"{what} in a list item indented {ind - items[-1]} spaces past the item's marker, not 4"
        if item and not nested:
            items.append(ind)
        if opens_fence:
            fence = opens_fence.group(1)
        paragraph = not item and not NOT_PARAGRAPH.match(line) and (paragraph or blank)
        blank = False


def main():
    files = [ROOT / f for f in ("README.md", "CONTRIBUTING.md", "examples/README.md", "tools/README.md")]
    files += sorted((ROOT / "docs").rglob("*.md"))
    failures = [f"{p.relative_to(ROOT)}:{n}: {msg}" for p in files for n, msg in check(p.read_text())]
    print(f"{len(files)} pages, {len(failures)} list blocks that the website breaks")
    for f in failures:
        print("  " + f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
