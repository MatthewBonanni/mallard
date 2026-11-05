#!/usr/bin/env python3
"""Fails on British spellings in tracked text files: Mallard uses American spelling
everywhere (code, comments, docs, messages). Used by the first job of ci.yml.

    python3 tools/dev/check_spelling.py

Names that other formats or published titles fix are allowed where they appear:
OpenFOAM's `neighbour` file, GitHub's `cancelled` conclusion, the units parser's
`metre` and `litre` aliases, reference titles in docs/references.md, and the input-error test
that suggests "center" for "centre".
"""
import re
import subprocess
import sys

BRITISH = {
    "behaviour": "behavior", "centre": "center", "centred": "centered", "colour": "color",
    "favour": "favor", "flavour": "flavor", "grey": "gray", "harbour": "harbor", "honour": "honor",
    "labour": "labor", "neighbour": "neighbor", "neighbourhood": "neighborhood", "vapour": "vapor",
    "metre": "meter", "litre": "liter", "fibre": "fiber", "calibre": "caliber", "licence": "license",
    "defence": "defense", "catalogue": "catalog", "analogue": "analog", "programme": "program",
    "modelling": "modeling", "modelled": "modeled", "labelled": "labeled", "labelling": "labeling",
    "cancelled": "canceled", "cancelling": "canceling", "travelled": "traveled", "travelling": "traveling",
    "signalling": "signaling", "fuelled": "fueled", "levelled": "leveled", "tunnelling": "tunneling",
    "whilst": "while", "amongst": "among", "artefact": "artifact", "aluminium": "aluminum",
    "sulphur": "sulfur", "aeroplane": "airplane", "acknowledgement": "acknowledgment",
}
IZE_STEMS = ["initial", "normal", "optim", "parallel", "discret", "minim", "maxim", "organ", "recogn",
             "util", "visual", "real", "summar", "synchron", "token", "serial", "general", "special",
             "regular", "vector", "local", "linear", "symmetr", "stabil", "equal", "emphas", "character",
             "categor", "custom", "standard", "prior", "author", "central", "random", "material", "finali"]
IZE = re.compile(r"\b(" + "|".join(IZE_STEMS) + r")is(e|es|ed|ing|ation|ations|er|ers)\b", re.IGNORECASE)
YSE = re.compile(r"\b(analy|paraly|cataly)s(e|ed|ing)\b", re.IGNORECASE)
OUR_ETC = re.compile(r"\b(" + "|".join(BRITISH) + r")s?\b", re.IGNORECASE)

EXCLUDED_PREFIXES = ("src/external/", "licenses/", "mechanisms/", "test/data/")
EXCLUDED_SUFFIXES = (".svg", ".png", ".jpg", ".pdf", ".h5", ".cgns", ".msh", ".vtu", ".vtkhdf")
ALLOWED = [
    (re.compile(r"^src/mesh/read_openfoam\.cpp$|^test/(fuzz/|mesh_import_test)|^docs/input\.md$"), "neighbour"),
    (re.compile(r"^\.github/workflows/|^tools/dev/ci_status\.sh$"), "cancelled"),
    (re.compile(r"^src/common/units\.cpp$"), "metre"),
    (re.compile(r"^src/common/units\.cpp$"), "litre"),
    (re.compile(r"^docs/references\.md$"), None),
    (re.compile(r"^test/error_messages_test\.cpp$"), "centre"),
    (re.compile(r"^tools/dev/check_spelling\.py$"), None),
]


def allowed(path, word):
    stem = word.lower().rstrip("s")
    return any(p.search(path) and (w is None or stem.startswith(w)) for p, w in ALLOWED)


def main():
    files = subprocess.run(["git", "ls-files"], check=True, capture_output=True, text=True).stdout.split()
    errors = []
    for path in files:
        if path.startswith(EXCLUDED_PREFIXES) or path.lower().endswith(EXCLUDED_SUFFIXES):
            continue
        try:
            text = open(path, encoding="utf-8").read()
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            for pattern in (OUR_ETC, IZE, YSE):
                for m in pattern.finditer(line):
                    if not allowed(path, m.group(0)):
                        errors.append(f"{path}:{n}: \"{m.group(0)}\" is British spelling; use American")
    for e in errors:
        print(e, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
