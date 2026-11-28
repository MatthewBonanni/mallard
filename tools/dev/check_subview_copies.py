#!/usr/bin/env python3
"""Fails on copies between memory spaces of a subview that may not be contiguous.

Kokkos copies a subview across memory spaces (device to host or back) only if it
is contiguous. A row range of a multidimensional View is contiguous in LayoutRight
but not in LayoutLeft, the default layout of CUDA and HIP, so such a copy works on
host backends and throws on GPUs ("Kokkos::deep_copy with no available copy
mechanism"). This flags `deep_copy`, `gated::copy` and `create_mirror_view_and_copy`
of a `subview` of a View of rank 2 or more, with a host View on the other side,
unless the subview is contiguous in the View's explicit layout (LayoutRight:
leading scalar indices, then one range, then ALL; LayoutLeft: the reverse) or in
any layout (a single element). Mark a copy proven contiguous otherwise with a
comment `subview-copy: contiguous` on its lines or the line above.

    python3 tools/dev/check_subview_copies.py [FILE...]   (default: src/ and test/)
"""
import os
import re
import subprocess
import sys

OPT_OUT = "subview-copy: contiguous"
CALL = re.compile(r"\b(Kokkos::deep_copy|deep_copy|gated::copy|Kokkos::create_mirror_view_and_copy|"
                  r"create_mirror_view_and_copy)\s*\(")
SUBVIEW = re.compile(r"\b(?:Kokkos::)?subview\s*\(")
INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.M)
HOST = re.compile(r"Host|host_mirror|\bh_\w|std::vector")
DEVICE = re.compile(r"memory_space|execution_space|DefaultExecutionSpace|DeviceUnmanaged|Kokkos::CudaSpace|"
                    r"Kokkos::HIPSpace|SharedSpace")
RANGE = re.compile(r"^(Kokkos::ALL(\(\))?|ALL|(std::|Kokkos::)?make_pair\s*\(.*|(std::|Kokkos::)?pair\s*<.*|\{.*\})$",
                   re.S)


def strip(text):
    """Comments and string literals blanked, keeping offsets."""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(re.sub(r"[^\n]", " ", text[i:j]))
            i = j
        elif c == '"' or (c == "'" and not (i > 0 and text[i - 1].isalnum())):
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == "\\" else 1
            out.append(c + " " * (min(j, n) - i - 1) + (c if j < n else ""))
            i = j + 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def arguments(text, start):
    """The top-level arguments of the call whose '(' is at start, and the offset after its ')'."""
    depth, angle, args, begin, i = 0, 0, [], start + 1, start
    while i < len(text):
        c = text[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                args.append(text[begin:i].strip())
                return args, i + 1
        elif c == "<" and text[i - 1:i + 1] != "<<" and text[i:i + 2] not in ("<<", "<=") and \
                re.match(r"[\w:>]", text[i - 1]):
            angle += 1
        elif c == ">" and angle > 0 and text[i - 1] != "-" and text[i:i + 2] != ">=":
            angle -= 1
        elif c == "," and depth == 1 and angle == 0:
            args.append(text[begin:i].strip())
            begin = i + 1
        i += 1
    return args, len(text)


def declaration(text, m):
    """The type before the name matched by m and, if assigned, the right-hand side."""
    depth, i = 0, m.start() - 1
    while i >= 0:
        c = text[i]
        if c == ">":
            depth += 1
        elif c == "<":
            depth -= 1
        elif depth == 0 and c in ",(;{}":
            break
        i -= 1
    kind = text[i + 1:m.start()].strip()
    if not re.search(r"[\w>]\s*[&*]?$", kind) or re.search(r"\b(return|if|while|for)\b|[=+\-/]", kind):
        kind = ""
    rhs = ""
    if m.group(1).startswith("="):
        j = text.find(";", m.end())
        rhs = text[m.end():j if j >= 0 else len(text)]
    if not kind and not rhs:
        return None
    return kind + " = " + rhs if rhs else kind


def last_name(expr):
    names = re.findall(r"[A-Za-z_]\w*", re.sub(r"\[[^\]]*\]|\([^()]*\)", "", expr))
    return names[-1] if names else ""


class Source:
    def __init__(self, path, root):
        self.path = path
        with open(path, encoding="utf-8", errors="replace") as f:
            self.raw = f.read()
        self.text = strip(self.raw)
        self.context = [self.text]
        seen = {os.path.abspath(path)}
        stem = os.path.splitext(path)[0]
        candidates = [stem + ".h"] + self.includes(path, root)
        for c in candidates:
            if c and os.path.isfile(c) and os.path.abspath(c) not in seen:
                seen.add(os.path.abspath(c))
                with open(c, encoding="utf-8", errors="replace") as f:
                    self.context.append(strip(f.read()))

    def includes(self, path, root):
        bases = [os.path.dirname(path), os.path.join(root, "src"), os.path.join(root, "test")]
        bases += sorted(d.path for d in os.scandir(os.path.join(root, "src")) if d.is_dir()) \
            if os.path.isdir(os.path.join(root, "src")) else []
        out = []
        for inc in INCLUDE.findall(self.raw):
            out += [p for p in (os.path.join(b, inc) for b in bases) if os.path.isfile(p)][:1]
        return out

    def declarations(self, name, before=None):
        """Declarations and assignments of name, as their type and right-hand side: the file's
        own before offset first, nearest first, then its header's and includes'."""
        pattern = re.compile(r"(?<![\w.>])" + re.escape(name) + r"\s*(=(?!=)|[;{,)\[]|\((?=[^)]))")
        texts = [self.text[:before] if before is not None else self.text] + self.context[1:]
        found = []
        for t in texts:
            here = [declaration(t, m) for m in pattern.finditer(t)]
            here = [d for d in here if d]
            found.extend(reversed(here) if t is texts[0] else here)
        return found

    def layout(self, name):
        for d in self.declarations(name):
            if "<" not in d:
                continue
            if "LayoutRight" in d:
                return "right"
            if "LayoutLeft" in d:
                return "left"
            alias = re.search(r"(?:\b|::)(\w+)\s*<", d)
            if alias and alias.group(1) not in ("View", "pair"):
                for t in self.context:
                    m = re.search(r"using\s+" + re.escape(alias.group(1)) + r"\s*=([^;]*);", t)
                    if m and "LayoutRight" in m.group(1):
                        return "right"
                    if m and "LayoutLeft" in m.group(1):
                        return "left"
            if re.search(r"View\s*<", d):
                return "default"
        return "default"

    def is_range(self, arg, before):
        if RANGE.match(arg):
            return True
        if re.fullmatch(r"[A-Za-z_]\w*", arg):
            for d in self.declarations(arg, before)[:3]:
                if re.search(r"\bpair\b|make_pair|Kokkos::ALL", d):
                    return True
        return False

    def is_host(self, expr, before):
        if HOST.search(expr):
            return True
        if DEVICE.search(expr) or SUBVIEW.search(expr):
            return False
        name = expr if re.fullmatch(r"[A-Za-z_]\w*", expr) else None
        if name is None:
            return False
        for d in self.declarations(name, before)[:1]:
            if DEVICE.search(d):
                return False
            if HOST.search(d) or re.search(r"create_mirror(_view)?\s*\(\s*[A-Za-z_]\w*\s*\)", d):
                return True
        return False

    def line(self, offset):
        return self.raw.count("\n", 0, offset) + 1


def contiguous(kinds, layout):
    """kinds: 's' scalar, 'r' range, 'a' ALL, one per index of the View."""
    if all(k == "s" for k in kinds) or all(k == "a" for k in kinds):
        return True
    right = re.fullmatch(r"s*[ra]?a*", "".join(kinds)) is not None
    left = re.fullmatch(r"a*[ra]?s*", "".join(kinds)) is not None
    return {"right": right, "left": left, "default": right and left}[layout]


def subview_problem(src, expr, offset):
    """A description if expr is a subview of a rank >= 2 View that may not be contiguous."""
    m = SUBVIEW.match(expr) if expr.startswith(("subview", "Kokkos::subview")) else None
    if not m:
        return None
    args, _ = arguments(expr, m.end() - 1)
    if len(args) < 3:
        return None
    kinds = ["a" if re.match(r"^(Kokkos::)?ALL(\(\))?$", a) else "r" if src.is_range(a, offset) else "s"
             for a in args[1:]]
    layout = src.layout(last_name(args[0]))
    if contiguous(kinds, layout):
        return None
    shown = "LayoutRight" if layout == "right" else "LayoutLeft" if layout == "left" else "the default layout"
    return f"subview of {args[0]} (rank {len(args) - 1}, {shown})"


def check(path, root):
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    if "subview" not in raw or not CALL.search(raw):
        return []
    src = Source(path, root)
    problems = []
    for m in CALL.finditer(src.text):
        args, end = arguments(src.text, m.end() - 1)
        func = m.group(1)
        if "gated::copy" in func:
            args = args[1:]
        if len(args) != 2:
            continue
        if "create_mirror_view_and_copy" in func:
            pairs = [(args[1], args[0])] if HOST.search(args[0]) else []
        else:
            pairs = [(args[1], args[0]), (args[0], args[1])]
        for view, other in pairs:
            problem = subview_problem(src, view, m.start())
            if not problem or ("create_mirror" not in func and not src.is_host(other, m.start())):
                continue
            first, last = src.line(m.start()), src.line(end)
            lines = src.raw.split("\n")[max(0, first - 2):last]
            if any(OPT_OUT in l for l in lines):
                continue
            problems.append(f"{os.path.relpath(path, root)}:{first}: {func} between host and device of a "
                            f"{problem}, which may not be contiguous (CUDA and HIP default to LayoutLeft). "
                            f"Copy the whole View, pack the rows on the device first, or mark the copy "
                            f"`// {OPT_OUT}` with the reason.")
    return problems


def main():
    root = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                          check=True).stdout.strip()
    files = sys.argv[1:]
    if not files:
        listed = subprocess.run(["git", "ls-files", "src", "test"], cwd=root, capture_output=True, text=True,
                                check=True).stdout.split()
        files = [os.path.join(root, f) for f in listed
                 if f.endswith((".cpp", ".h", ".hpp")) and not f.startswith("src/external/")]
    problems = [p for f in files for p in check(f, root)]
    for p in problems:
        print(p)
    if problems:
        print(f"{len(problems)} copies of subviews that may not be contiguous")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
