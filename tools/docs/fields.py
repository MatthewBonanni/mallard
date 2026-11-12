"""The field table of docs/guide/output.md, from the mallard executable itself.

    python tools/docs/fields.py --mallard build/src/mallard           rewrite it
    python tools/docs/fields.py --mallard build/src/mallard --check   exit 1 if out of date

Runs `mallard list fields --format json` and writes its catalog between the
GENERATED markers of docs/guide/output.md, one row per field or pattern. The test
suite (CliTest.OutputGuideTableIsTheFieldCatalog) fails when a row there differs
from the executable's catalog.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "guide" / "output.md"
BEGIN, END = "<!-- BEGIN GENERATED fields -->", "<!-- END GENERATED fields -->"


def cell(text):
    return text.replace("|", "\\|")


def generate(mallard):
    out = subprocess.run([mallard, "list", "fields", "--format", "json"], capture_output=True, text=True, check=True)
    rows = ["| Group | Field | Units | Description | Needs | Available in |", "|---|---|---|---|---|---|"]
    group = None
    for f in json.loads(out.stdout)["fields"]:
        shown = f["group"] if f["group"] != group else ""
        group = f["group"]
        rows.append(
            f"| {cell(shown)} | `{f['name']}` | `{f['units']}` | {cell(f['long_name'])} | "
            f"{cell(f.get('needs', '')) or 'every case'} | {', '.join(f['available']) or '-'} |"
        )
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mallard", required=True, help="the mallard executable (build/src/mallard)")
    parser.add_argument("--check", action="store_true", help="exit 1 if the table is out of date")
    args = parser.parse_args()
    text = PAGE.read_text()
    start, end = text.index(BEGIN) + len(BEGIN), text.index(END)
    new = text[:start] + "\n" + generate(args.mallard) + "\n" + text[end:]
    if args.check:
        if new != text:
            print(f"{PAGE.relative_to(ROOT)} is out of date: python tools/docs/fields.py --mallard {args.mallard}")
            sys.exit(1)
    elif new != text:
        PAGE.write_text(new)
        print(f"wrote {PAGE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
