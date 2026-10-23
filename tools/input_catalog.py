#!/usr/bin/env python3
"""The key catalog of Mallard's input (docs/input-keys.json), made by the setup itself.

    python tools/input_catalog.py build --mallard-2d build/src/mallard --mallard-3d build-3d/src/mallard [--check]
    python tools/input_catalog.py docs [--catalog docs/input-keys.json]

build: runs `mallard check INPUT --catalog KEYS.json` on every input of examples/
and tutorials/, in its directory on a temporary copy whose generated mesh is cut to
a few cells per direction (each check takes seconds; inputs whose mesh file is not
in the repository are skipped) and
merges what setup recorded into docs/input-keys.json: for each key path, its type,
physical dimension (and the unit of a plain number, when not SI), default (as TOML
text) or that it is required, the accepted values, the builds (2D, 3D) whose setup
read it, and the inputs that set it. With --check it compares with the
committed file instead of writing it, and exits 1 if they differ.

docs: compares the catalog's defaults with the defaults docs/input.md states
("default X", "Default: X", "X (default)" in a key's row) and exits 1 on any
difference. CI runs both (the tutorials job, which has both builds).
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "docs" / "input-keys.json"
SIZE_KEYS = ("Nx", "Ny", "Nz")
TUTORIAL_DIMS = {"wedge": 2, "cylinder": 2, "cone": 3, "channel_les": 3}


def inputs():
    """(path, dimension) of every input of examples/ and tutorials/."""
    out = []
    for f in sorted((ROOT / "examples").rglob("*.toml")):
        if f.name == "card.toml" or "reference" in f.parts:
            continue
        case = f.relative_to(ROOT / "examples").parts[0]
        card = ROOT / "examples" / case / "card.toml"
        build = tomllib.loads(card.read_text()).get("build", "2d") if card.exists() else "2d"
        out.append((f, 3 if build == "3d" else 2))
    for name, dim in TUTORIAL_DIMS.items():
        f = ROOT / "tutorials" / name / "input.toml"
        if f.exists():
            out.append((f, dim))
    return out


def shrink(text):
    """The input with the cell counts of a generated mesh cut to at most 8."""
    return re.sub(rf"^(\s*(?:{'|'.join(SIZE_KEYS)})\s*=\s*)(\d+)", lambda m: m.group(1) + str(min(int(m.group(2)), 8)),
                  text, flags=re.M)


def check(mallard, path, work):
    """The catalog of one input, from a check run in the input's own directory (so its
    relative paths hold) on a copy with a small mesh; None if its mesh file is missing."""
    inp = tomllib.loads(path.read_text())
    mesh = inp.get("mesh", {})
    if mesh.get("type", "file") == "file" and not (path.parent / mesh.get("filename", "")).is_file():
        return None
    copy = path.parent / f".catalog_{path.name}"
    keys = work / (path.relative_to(ROOT).as_posix().replace("/", "_") + ".json")
    try:
        copy.write_text(shrink(path.read_text()))
        subprocess.run([str(mallard), "check", copy.name, "--catalog", str(keys), "--kokkos-num-threads=1"],
                       cwd=path.parent, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
    finally:
        copy.unlink(missing_ok=True)
    return json.loads(keys.read_text()) if keys.exists() else None


def build(args):
    merged = {}
    with tempfile.TemporaryDirectory() as tmp:
        for path, dim in inputs():
            mallard = Path(args.mallard_2d if dim == 2 else args.mallard_3d).resolve()
            keys = check(mallard, path, Path(tmp))
            if keys is None:
                continue
            source = path.relative_to(ROOT).as_posix()
            for key, entry in keys.items():
                m = merged.setdefault(key, {"dims": [], "set_in": []})
                if dim not in m["dims"]:
                    m["dims"].append(dim)
                if entry.get("set") and source not in m["set_in"]:
                    m["set_in"].append(source)
                if "type" in entry:
                    types = set(filter(None, m.get("type", "").split("|"))) | {entry["type"]}
                    m["type"] = "|".join(sorted(types))
                if entry.get("required"):
                    m["required"] = True
                if "default" in entry:
                    if m.get("default") in (None, entry["default"]):
                        m["default"] = entry["default"]
                    else:
                        m.setdefault("defaults_by_dim", {})[str(dim)] = entry["default"]
                for field in ("dimension", "unit"):
                    if field in entry:
                        m[field] = entry[field]
                if "values" in entry:
                    m["values"] = sorted(set(m.get("values", [])) | set(entry["values"]))
    for m in merged.values():
        m["dims"].sort()
        m["set_in"].sort()
    text = json.dumps(dict(sorted(merged.items())), indent=1, sort_keys=True) + "\n"
    if args.check:
        if not CATALOG.exists() or CATALOG.read_text() != text:
            sys.exit(f"{CATALOG.relative_to(ROOT)} is out of date: run python tools/input_catalog.py build ...")
        print(f"{CATALOG.relative_to(ROOT)}: up to date ({len(merged)} keys)")
    else:
        CATALOG.write_text(text)
        print(f"wrote {CATALOG.relative_to(ROOT)}: {len(merged)} keys")


# A default stated as a literal: a number, true/false, or a quoted or backticked value
# without spaces or placeholders (prose like "default the first" is not compared)
LITERAL = r"(?>`[^`<\s]+`|-?\d[\d.]*(?:[eE][-+]?\d+)?|true|false)"
DEFAULT = re.compile(rf"[Dd]efault:?\s+({LITERAL})(?![\w<])(?!\s+in\s+[23]D)|({LITERAL})\s+\(default\)")


def documented_defaults():
    """{path: default text} of the keys whose row in docs/input.md states one."""
    out, section = {}, ""
    for line in (ROOT / "docs" / "input.md").read_text().splitlines():
        h = re.match(r"^#{2,4} `\[*([\w.]+)\]*`", line)
        if h:
            section = h.group(1)
            continue
        row = re.match(r"^\| `([\w]+)`(?: / `[\w]+`)* \| (.*)$", line)
        if not row or not section:
            continue
        m = DEFAULT.search(row.group(2))
        if m:
            out[f"{section}.{row.group(1)}"] = (m.group(1) or m.group(2)).strip("`\"'")
    return out


def same(catalog_text, doc_text):
    a, b = catalog_text.strip('"'), doc_text
    try:
        return float(a) == float(b)
    except ValueError:
        return a.lower() == b.lower()


def docs(args):
    catalog = json.loads(Path(args.catalog).read_text())
    documented = documented_defaults()
    wrong = []
    for path, text in documented.items():
        entry = catalog.get(path)
        if not entry or "default" not in entry or entry.get("type") not in ("bool", "integer", "number", "string"):
            continue
        if not same(entry["default"], text) and not any(same(d, text) for d in entry.get("defaults_by_dim", {}).values()):
            wrong.append(f"{path}: setup uses {entry['default']}, docs/input.md says {text}")
    checked = sum(1 for p in documented if p in catalog and "default" in catalog[p])
    if wrong:
        sys.exit("defaults in docs/input.md that setup does not use:\n  " + "\n  ".join(wrong))
    print(f"{checked} documented defaults match the catalog")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--mallard-2d", required=True)
    b.add_argument("--mallard-3d", required=True)
    b.add_argument("--check", action="store_true")
    d = sub.add_parser("docs")
    d.add_argument("--catalog", default=str(CATALOG))
    args = ap.parse_args()
    {"build": build, "docs": docs}[args.cmd](args)


if __name__ == "__main__":
    main()
