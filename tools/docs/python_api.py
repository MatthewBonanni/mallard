"""The reference of the Python package (docs/guide/python_api.md), from its docstrings.

    python tools/docs/python_api.py           rewrite it
    python tools/docs/python_api.py --check   exit 1 if out of date

Reads tools/mallard/*.py with ast (no import, so no numpy or library is needed) and writes each public
function and class, with its signature and docstring, between the GENERATED markers of the page.
"""
import argparse
import ast
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "tools" / "mallard"
PAGE = ROOT / "docs" / "guide" / "python_api.md"
BEGIN, END = "<!-- BEGIN GENERATED reference -->", "<!-- END GENERATED reference -->"

# (section, module, prefix, names), in the order of the page
SECTIONS = [
    ("Reading files", "reader", "mallard.", ["open", "File", "Field", "Snapshot", "Geometry", "run_files",
                                              "run_record"]),
    ("Commands", "commands", "mallard.", ["info", "diff", "command", "CommandError"]),
    ("`mallard.post`", "post", "mallard.post.", ["extract", "convergence"]),
]


def signature(node, skip_self=False):
    args = node.args
    positional = args.posonlyargs + args.args
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    parts = [a.arg + ("" if d is None else "=" + ast.unparse(d)) for a, d in zip(positional, defaults)]
    if skip_self:
        parts = parts[1:]
    if args.vararg:
        parts.append("*" + args.vararg.arg)
    elif args.kwonlyargs:
        parts.append("*")
    for a, d in zip(args.kwonlyargs, args.kw_defaults):
        parts.append(a.arg + ("" if d is None else "=" + ast.unparse(d)))
    if args.kwarg:
        parts.append("**" + args.kwarg.arg)
    return "(" + ", ".join(parts) + ")"


def doc(node):
    text = ast.get_docstring(node)
    if not text:
        sys.exit(f"python_api.py: {node.name} has no docstring")
    return inspect.cleandoc(text)


def member(node, owner):
    """A method's or property's entry; None for private members."""
    if node.name == "__getitem__":
        return f"#### `{owner.lower()}[field]`\n\n{doc(node)}\n"
    if node.name.startswith("_"):
        return None
    is_property = any(isinstance(d, ast.Name) and d.id == "property" for d in node.decorator_list)
    head = node.name if is_property else node.name + signature(node, skip_self=True)
    return f"#### `{owner.lower()}.{head}`\n\n{doc(node)}\n"


def reference():
    out = []
    for section, module, prefix, names in SECTIONS:
        tree = ast.parse((PACKAGE / f"{module}.py").read_text())
        defs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        out.append(f"## {section}\n")
        for name in names:
            node = defs[name]
            if isinstance(node, ast.FunctionDef):
                out.append(f"### `{prefix}{name}{signature(node)}`\n\n{doc(node)}\n")
                continue
            out.append(f"### `{prefix}{name}`\n\n{doc(node)}\n")
            for child in node.body:
                if isinstance(child, ast.FunctionDef):
                    entry = member(child, name)
                    if entry:
                        out.append(entry)
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if the page is out of date")
    args = parser.parse_args()
    text = PAGE.read_text()
    start, end = text.index(BEGIN) + len(BEGIN), text.index(END)
    new = text[:start] + "\n\n" + reference() + "\n" + text[end:]
    if args.check:
        if new != text:
            sys.exit("docs/guide/python_api.md is out of date: run python3 tools/docs/python_api.py")
        return
    PAGE.write_text(new)


if __name__ == "__main__":
    main()
