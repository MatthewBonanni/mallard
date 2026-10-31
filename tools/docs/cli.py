"""The help pages of docs/guide/cli.md, from the mallard executable itself.

    python tools/docs/cli.py --mallard build/src/mallard           rewrite them
    python tools/docs/cli.py --mallard build/src/mallard --check   exit 1 if out of date

Runs `mallard --help`, takes the commands and families it lists, and writes each
one's `--help` page between the GENERATED markers of docs/guide/cli.md, with the
build's version shown as <version>. The test suite (CliTest.UserDocsShowEveryHelpPage)
fails when a page there differs from the executable's.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "guide" / "cli.md"
BEGIN, END = "<!-- BEGIN GENERATED pages -->", "<!-- END GENERATED pages -->"


def run(mallard, *args):
    return subprocess.run([mallard, *args], capture_output=True, text=True, check=True).stdout


def commands(top):
    """Command names of the top-level page's "Commands:" section, in order."""
    names, inside = [], False
    for line in top.splitlines():
        if line == "Commands:":
            inside = True
        elif inside and not line.strip():
            break
        elif inside and line.startswith("    mallard "):
            name = re.split(r"\s{2,}", line.strip())[0].removeprefix("mallard ")
            if not name.startswith("-i "):
                names.append(name)
    return names


def section(title, page, version):
    page = page.replace(version, "<version>")
    return f"### {title}\n\n```text\n{page}```\n"


def generate(mallard):
    version = run(mallard, "--version").strip().removeprefix("Mallard ")
    top = run(mallard, "--help")
    out = [section("mallard", top, version)]
    families = []
    for name in commands(top):
        family = name.split()[0] if " " in name else None
        if family and family not in families:
            families.append(family)
            out.append(section(f"mallard {family}", run(mallard, family, "--help"), version))
        out.append(section(f"mallard {name}", run(mallard, *name.split(), "--help"), version))
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mallard", required=True, help="the mallard executable (build/src/mallard)")
    parser.add_argument("--check", action="store_true", help="exit 1 if the page is out of date")
    args = parser.parse_args()
    text = PAGE.read_text()
    start, end = text.index(BEGIN) + len(BEGIN), text.index(END)
    new = text[:start] + "\n\n" + generate(args.mallard) + "\n" + text[end:]
    if args.check:
        if new != text:
            print(f"{PAGE.relative_to(ROOT)} is out of date: python tools/docs/cli.py --mallard {args.mallard}")
            sys.exit(1)
    elif new != text:
        PAGE.write_text(new)
        print(f"wrote {PAGE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
