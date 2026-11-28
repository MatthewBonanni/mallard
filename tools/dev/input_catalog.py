#!/usr/bin/env python3
"""The key catalog of Mallard's input (docs/input-keys/), made by the setup itself.

    python tools/dev/input_catalog.py build --mallard-2d build/src/mallard --mallard-3d build-3d/src/mallard [--check]
    python tools/dev/input_catalog.py docs [--catalog docs/input-keys]

build: runs `mallard check INPUT --catalog KEYS.json` on every input of examples/,
tutorials/ and tools/dev/catalog_inputs/ (2d/, 3d/: inputs of the keys no example or
tutorial sets up), in its directory on a temporary copy whose generated mesh is cut to
a few cells per direction (each check takes seconds; inputs whose mesh file is not
in the repository are skipped) and
merges what setup recorded into docs/input-keys/, one file per key path: its type,
physical dimension (and the unit of a plain number, when not SI), default (as TOML
text) or that it is required, the accepted values, the builds (2D, 3D) whose setup
read it, the inputs that set it, and what setup records besides (path, references,
for_types, required_for, alternatives: docs/guide/input_files.md#the-key-catalog;
for_types and required_for only if every input read the key under a type). A key's file is its path, percent-encoded
outside [a-z0-9_.-] (file systems that ignore case keep sigma_t and sigma_T apart),
plus .json. With --check it compares with the committed files instead of writing
them, and exits 1 if they differ. After a build, python tools/dev/input_config.py
regenerates the typed configs (include/mallard/config.h, tools/mallard/config.py)
from the catalog.

docs: checks docs/input.md against the catalog and exits 1 on any difference:
every key of its key tables (those headed "| Key |") must be in the catalog, and
the defaults it states ("default X", "Default: X", "X (default)" in a key's row)
must be the catalog's. CI runs
build --check in the tutorials job, which has both builds, and docs in
tools/dev/precheck.sh.
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
import tomllib
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "docs" / "input-keys"
PLAIN = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_.-")
SIZE_KEYS = ("Nx", "Ny", "Nz")
TUTORIAL_DIMS = {"wedge": 2, "cylinder": 2, "measuring": 2, "cone": 3, "channel_les": 3, "cluster": 2, "post_processing": 2}


def inputs():
    """(path, dimension) of every input of examples/, tutorials/ and tools/dev/catalog_inputs/."""
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
    for dim in (2, 3):
        out += [(f, dim) for f in sorted((ROOT / "tools" / "dev" / "catalog_inputs" / f"{dim}d").glob("*.toml"))]
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


def merge(merged, keys, dim, source, untyped):
    """Merges one input's catalog (keys, of the build of dimension dim) into merged. untyped
    collects the keys some input read in a table without a type (their for_types and
    required_for are dropped at the end: they apply whatever the type) or read required
    without a type (required_for alone)."""
    for key, entry in keys.items():
        m = merged.setdefault(key, {"dims": [], "set_in": []})
        if dim not in m["dims"]:
            m["dims"].append(dim)
        if entry.get("set") and source not in m["set_in"]:
            m["set_in"].append(source)
        if "type" in entry:
            types = set(filter(None, m.get("type", "").split("|"))) | {entry["type"]}
            m["type"] = "|".join(sorted(types))
        if ("type" in entry or entry.get("set")) and "for_types" not in entry:
            untyped.add((key, "for_types"))
        if entry.get("required") and "required_for" not in entry:
            untyped.add((key, "required_for"))
        if entry.get("required"):
            m["required"] = True
        if "default" in entry:
            if m.get("default") in (None, entry["default"]):
                m["default"] = entry["default"]
            else:
                m.setdefault("defaults_by_dim", {})[str(dim)] = entry["default"]
        for field in ("dimension", "unit", "references", "path"):
            if field in entry:
                m[field] = entry[field]
        for field in ("values", "for_types", "required_for"):
            if field in entry:
                m[field] = sorted(set(m.get(field, [])) | set(entry[field]))
        if "alternatives" in entry:
            required = {}
            for a in m.get("alternatives", []) + entry["alternatives"]:
                groups = json.dumps(a["keys"])
                required[groups] = required.get(groups, True) and a["required"]
            m["alternatives"] = [{"keys": json.loads(g), "required": r} for g, r in sorted(required.items())]


def finish(merged, untyped):
    for key, field in untyped:
        merged[key].pop(field, None)
        if field == "for_types":
            merged[key].pop("required_for", None)
    for m in merged.values():
        m["dims"].sort()
        m["set_in"].sort()


def require_hdf5(mallard):
    """Defaults that depend on the build (write_data's format) are the HDF5 builds' ones."""
    out = subprocess.run([str(Path(mallard).resolve()), "version", "--format", "json"],
                         capture_output=True, text=True, check=True).stdout
    if not json.loads(out)["build"]["hdf5"]:
        sys.exit(f"{mallard} is built without HDF5, whose defaults differ: build with -DMallard_ENABLE_HDF5=ON")


def build(args):
    require_hdf5(args.mallard_2d)
    require_hdf5(args.mallard_3d)
    merged, untyped = {}, set()
    with tempfile.TemporaryDirectory() as tmp:
        for path, dim in inputs():
            mallard = Path(args.mallard_2d if dim == 2 else args.mallard_3d).resolve()
            keys = check(mallard, path, Path(tmp))
            if keys is not None:
                merge(merged, keys, dim, path.relative_to(ROOT).as_posix(), untyped)
    finish(merged, untyped)
    files = {file_name(key): json.dumps(entry, indent=1, sort_keys=True) + "\n" for key, entry in merged.items()}
    present = {f.name: f.read_text() for f in CATALOG.glob("*.json")}
    where = CATALOG.relative_to(ROOT)
    if args.check:
        if present != files:
            stale = sorted(set(present) ^ set(files) | {n for n in files if present.get(n, files[n]) != files[n]})
            sys.exit(f"{where}/ is out of date ({', '.join(stale[:5])}{', ...' if len(stale) > 5 else ''}): "
                     "run python tools/dev/input_catalog.py build ...")
        print(f"{where}/: up to date ({len(files)} keys)")
    else:
        CATALOG.mkdir(exist_ok=True)
        for name in set(present) - set(files):
            (CATALOG / name).unlink()
        for name, text in files.items():
            if present.get(name) != text:
                (CATALOG / name).write_text(text)
        print(f"wrote {where}/: {len(files)} keys")


def file_name(key):
    return "".join(c if c in PLAIN else "".join(f"%{b:02X}" for b in c.encode()) for c in key) + ".json"


def key_of(name):
    return urllib.parse.unquote(name.removesuffix(".json"))


def load(directory):
    """{key path: entry} of a catalog directory."""
    return {key_of(f.name): json.loads(f.read_text()) for f in sorted(Path(directory).glob("*.json"))}


# A default stated as a literal: a number, true/false, or a quoted or backticked value
# without spaces or placeholders (prose like "default the first" is not compared)
LITERAL = r"(?>`[^`<\s]+`|-?\d[\d.]*(?:[eE][-+]?\d+)?|true|false)"
DEFAULT = re.compile(rf"[Dd]efault:?\s+({LITERAL})(?![\w<])(?!\s+in\s+[23]D)|({LITERAL})\s+\(default\)")


def documented_rows():
    """(paths, whether they are alternatives, row text) of every row of the key tables of
    docs/input.md; a row may name several keys (`Nx`, `Ny`) or alternatives (`p` / `p0`,
    `X` or `Y`).
    A heading names its table as `[table]`, a subtable of its section as (`name`), or a
    variant of its section as `type = "..."`; the tables under other headings are skipped."""
    out, top, section, key_table, previous = [], "", "", False, ""
    for line in (ROOT / "docs" / "input.md").read_text().splitlines():
        h = re.match(r"^(#{2,4}) (.*)$", line)
        if h:
            table = re.search(r"`\[\[?([\w.]+)\]\]?`", h.group(2))
            subtable = re.search(r"\(`(\w+)`\)$", h.group(2))
            if table:
                section = table.group(1)
            elif subtable and top:
                section = f"{top}.{subtable.group(1)}"
            elif re.match(r'`type = "\w+"`$', h.group(2)):
                section = top
            else:
                section = ""
            if len(h.group(1)) == 2:
                top = section
        if line.startswith("|") and not previous.startswith("|"):
            key_table = line.startswith("| Key |")
        previous = line
        row = re.match(r"^\| (`\w+`(?:(?:, | / | or )`\w+`)*) \| (.*)$", line)
        if row and section and key_table:
            paths = [f"{section}.{key}" for key in re.findall(r"`(\w+)`", row.group(1))]
            out.append((paths, ", " not in row.group(1), row.group(2)))
    return out


def documented_defaults():
    """{path: default text} of the keys whose row in docs/input.md states one (the first
    key of a row of alternatives, `p` / `p0`; rows of several keys, `Nx`, `Ny`, are not
    compared)."""
    out = {}
    for paths, alternatives, text in documented_rows():
        m = DEFAULT.search(text)
        if m and alternatives:
            out[paths[0]] = (m.group(1) or m.group(2)).strip("`\"'")
    return out


def same(catalog_text, doc_text):
    a, b = catalog_text.strip('"'), doc_text
    try:
        return float(a) == float(b)
    except ValueError:
        return a.lower() == b.lower()


def docs(args):
    catalog = load(args.catalog)
    documented = documented_defaults()
    wrong = []
    for path, text in documented.items():
        entry = catalog.get(path)
        if not entry or "default" not in entry or entry.get("type") not in ("bool", "integer", "number", "string"):
            continue
        if not same(entry["default"], text) and not any(same(d, text) for d in entry.get("defaults_by_dim", {}).values()):
            wrong.append(f"{path}: setup uses {entry['default']}, docs/input.md says {text}")
    keys = {path for paths, _, _ in documented_rows() for path in paths}
    unread = sorted(keys - catalog.keys())
    checked = sum(1 for p in documented if p in catalog and "default" in catalog[p])
    errors = []
    if unread:
        errors.append("keys in docs/input.md that setup never reads (or that no input of the catalog sets up):\n  "
                      + "\n  ".join(unread))
    if wrong:
        errors.append("defaults in docs/input.md that setup does not use:\n  " + "\n  ".join(wrong))
    if errors:
        sys.exit("\n".join(errors))
    print(f"{len(keys)} documented keys are in the catalog, {checked} documented defaults match it")


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
