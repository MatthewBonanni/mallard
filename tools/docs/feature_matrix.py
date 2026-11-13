"""The feature matrix (docs/guide/features.md): each pair of features marked
validated, verified, implemented, rejected or untested, with a link to the
evidence, from the repository's own records.

    python tools/docs/feature_matrix.py           rewrite the page's generated blocks
    python tools/docs/feature_matrix.py --check   exit 1 if they are out of date or a record is inconsistent

Each pair takes the strongest of:

- rejected: an input guard of src/ throws for the pair. GUARDS names each guard by
  its file and a part of its message; every throw in src/ whose message says
  "not supported" or "not available" must be in GUARDS or OTHER_GUARDS.
- validated: an entry of `validation` in an example's card.toml (a section of the
  website's validation page and the regime it covers), with the features read
  from the example's input file.
- verified: a convergence study of the verification manual
  (docs/verification/results.json), with the features its test declares in
  test/features.toml (those of all its runs) and those its method names.
- implemented: a run of a test declared in test/features.toml uses both; for MPI
  and the GPU backends, a build of the support matrix (docs/guide/support.md)
  that runs the full test suite, with every feature a declared test runs the flow
  solver (Euler or Navier-Stokes) with.
- untested: none of these; the page lists these pairs as the test gaps.

Standard library only.
"""
import json
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "guide" / "features.md"
REGISTRY = ROOT / "test" / "features.toml"
RESULTS = ROOT / "docs" / "verification" / "results.json"
SUPPORT = ROOT / "docs" / "guide" / "support.md"
PRESETS = ROOT / "src" / "solver" / "numerics_preset.cpp"
PRESETS_HEADER = ROOT / "src" / "solver" / "numerics_preset.h"
VALIDATION = "https://matthewbonanni.github.io/mallard/validation/"


@dataclass
class Feature:
    id: str
    label: str
    group: str
    has: object = None  # test of a parsed input (validation entries); None: not read from inputs


def physics(t):
    return t.get("physics", {})


def numerics(t):
    return t.get("numerics", {})


def recon(t):
    return numerics(t).get("face_reconstruction", {})


def teno(order):
    return lambda t: recon(t).get("type", "FO") == "TENO" and recon(t).get("order", 5) == order


def periodic_pairs(t):
    return t.get("periodic", [])


FEATURES = [
    Feature("euler", "Euler", "Physics", lambda t: physics(t).get("type", "euler") == "euler"),
    Feature("navier_stokes", "Navier-Stokes", "Physics", lambda t: physics(t).get("type") == "navier_stokes"),
    Feature("mixture", "Gas mixture", "Physics", lambda t: physics(t).get("gas") == "mixture"),
    Feature("double_flux", "Double flux", "Physics", lambda t: numerics(t).get("double_flux", False)),
    Feature("chemistry", "Chemistry", "Physics",
            lambda t: "chemistry" in t and t["chemistry"].get("enabled", True)),
    Feature("simpler", "SIMPLER coupling", "Physics",
            lambda t: "chemistry" in t and t["chemistry"].get("coupling") == "simpler"),
    Feature("les", "LES", "Physics", lambda t: "les" in t),
    Feature("wall_model", "Wall model", "Physics", lambda t: any("wall_model" in b for b in t.get("boundaries", []))),
    Feature("axisymmetric", "Axisymmetric", "Physics", lambda t: physics(t).get("axisymmetric", False)),
    Feature("fo", "FO", "Reconstruction", lambda t: recon(t).get("type", "FO") == "FO"),
    Feature("muscl", "MUSCL", "Reconstruction", lambda t: recon(t).get("type") == "MUSCL"),
    Feature("teno3", "TENO3", "Reconstruction", teno(3)),
    Feature("teno4", "TENO4", "Reconstruction", teno(4)),
    Feature("teno5", "TENO5", "Reconstruction", teno(5)),
    Feature("teno6", "TENO6", "Reconstruction", teno(6)),
    Feature("hllc", "HLLC", "Flux", lambda t: numerics(t).get("riemann_solver", "HLLC") == "HLLC"),
    Feature("hll", "HLL", "Flux", lambda t: numerics(t).get("riemann_solver") == "HLL"),
    Feature("rhll", "RHLL", "Flux", lambda t: numerics(t).get("riemann_solver") == "RHLL"),
    Feature("roe", "Roe", "Flux", lambda t: numerics(t).get("riemann_solver") == "Roe"),
    Feature("rusanov", "Rusanov", "Flux", lambda t: numerics(t).get("riemann_solver") == "Rusanov"),
    Feature("hybrid", "Hybrid flux", "Flux", lambda t: numerics(t).get("convective_flux") == "hybrid"),
    Feature("fe", "FE", "Time integration", lambda t: numerics(t).get("time_integrator") == "FE"),
    Feature("ssprk3", "SSPRK3", "Time integration",
            lambda t: numerics(t).get("time_integrator", "SSPRK3") == "SSPRK3"),
    Feature("rk4", "RK4", "Time integration", lambda t: numerics(t).get("time_integrator") == "RK4"),
    Feature("ssprk54", "SSPRK54", "Time integration", lambda t: numerics(t).get("time_integrator") == "SSPRK54"),
    Feature("ck45", "CK45", "Time integration", lambda t: numerics(t).get("time_integrator") == "CK45"),
    Feature("rdpk3sp35", "RDPK3Sp35", "Time integration",
            lambda t: numerics(t).get("time_integrator") == "RDPK3Sp35"),
    Feature("error_control", "Error control", "Time integration", lambda t: "error_control" in numerics(t)),
    Feature("curved", "Curved walls", "Geometry", lambda t: "curved" in t.get("mesh", {})),
    Feature("periodic", "Periodic", "Geometry", lambda t: bool(periodic_pairs(t)) or "periodic" in t.get("mesh", {})),
    Feature("rotational", "Rotational periodicity", "Geometry",
            lambda t: any("angle" in p for p in periodic_pairs(t))),
    Feature("mpi", "MPI", "Parallel"),
    Feature("cuda", "CUDA", "Parallel"),
    Feature("hip", "HIP", "Parallel"),
    Feature("sycl", "SYCL", "Parallel"),
    Feature("remap", "Remap", "Output"),
    Feature("async", "Asynchronous output", "Output"),
    Feature("compression", "Compression", "Output"),
    Feature("restart", "Restart", "Output"),
]
BY_ID = {f.id: f for f in FEATURES}
ORDER = {f.id: i for i, f in enumerate(FEATURES)}

# Features of which a run uses exactly one: their pairs are not combinations
CHOICES = [
    {"euler", "navier_stokes"},
    {"fo", "muscl", "teno3", "teno4", "teno5", "teno6"},
    {"hllc", "hll", "rhll", "roe", "rusanov"},
    {"fe", "ssprk3", "rk4", "ssprk54", "ck45", "rdpk3sp35"},
    {"cuda", "hip", "sycl"},
]

# The page's tables: (heading, generated block, row groups, column groups)
NUMERICS = ["Reconstruction", "Flux", "Time integration"]
OTHER = ["Geometry", "Parallel", "Output"]
TABLES = [
    ("Physics", "physics", ["Physics"], ["Physics"]),
    ("Physics and numerics", "physics-numerics", ["Physics"], NUMERICS),
    ("Physics, geometry, parallel runs and output", "physics-other", ["Physics"], OTHER),
    ("Numerics", "numerics", NUMERICS, NUMERICS),
    ("Numerics, geometry, parallel runs and output", "numerics-other", NUMERICS, OTHER),
    ("Geometry, parallel runs and output", "other", OTHER, OTHER),
]

# Input guards that reject a pair of features: (file, part of the message, feature, feature)
GUARDS = [
    ("src/solver/solver.cpp", "curved boundaries are not supported in axisymmetric runs", "axisymmetric", "curved"),
    ("src/solver/solver.cpp", "rotational periodicity is not supported in axisymmetric runs",
     "axisymmetric", "rotational"),
    ("src/solver/solver.cpp", "its thermodynamics are frozen over a step", "double_flux", "simpler"),
    ("src/solver/solver.cpp", '"hybrid" is not available with chemistry.coupling', "hybrid", "simpler"),
    ("src/solver/solver_les.cpp", "[les] is not available for axisymmetric runs", "les", "axisymmetric"),
    ("src/solver/solver_wall_model.cpp", "gas mixtures are not supported", "wall_model", "mixture"),
    ("src/solver/solver_wall_model.cpp", "axisymmetric runs are not supported", "wall_model", "axisymmetric"),
    ("src/solver/solver_step_control.cpp", "is not available with [chemistry]", "error_control", "chemistry"),
    ("src/solver/solver_step_control.cpp", "is not available with numerics.double_flux", "error_control", "double_flux"),
]
# Input guards that make a feature need another: (file, part of the message, feature, needed feature).
# The feature is rejected with the other choices of the needed one, and with whatever the needed one is.
REQUIRES = [
    ("src/physics/les.cpp", '[les] needs physics.type = "navier_stokes"', "les", "navier_stokes"),
    ("src/solver/solver_wall_model.cpp", 'needs [physics] type = "navier_stokes"', "wall_model", "navier_stokes"),
    ("src/solver/solver_step_control.cpp", "needs a time integrator with an embedded error estimate",
     "error_control", "rdpk3sp35"),
    ("src/solver/solver.cpp", "[chemistry] needs a mixture of species", "chemistry", "mixture"),
    ("src/solver/solver.cpp", 'is for mixtures, but physics.gas is "perfect"', "double_flux", "mixture"),
]
# Features that need another by definition: a coupling of [chemistry], a kind of periodic pair
DEFINITIONS = [("simpler", "chemistry"), ("rotational", "periodic")]
# Guards of combinations with something that is not a feature of the matrix
OTHER_GUARDS = [
    ("src/solver/probes.cpp", "through a cut plane or curve is not supported with physics.axisymmetric"),
    ("src/solver/probes.cpp", "moment is not supported with physics.axisymmetric"),
    ("src/solver/synthetic_turbulence.cpp", "is not available on axisymmetric meshes"),
    ("src/numerics/teno.cpp", "single_precision_tables is not available with gas mixtures"),
    ("src/chemistry/mechanism.cpp", "reaction type"),
    ("src/chemistry/mechanism.cpp", "(only ideal-gas)"),
    ("src/chemistry/mechanism.cpp", "transport model"),
    ("src/chemistry/mechanism.cpp", "(NASA7, NASA9, constant-cp)"),
    ("src/mesh/mesh_io.cpp", "parametric nodes are not supported"),
    ("src/common/region_input.cpp", "only filled regions move"),
]
GUARD_WORDS = re.compile(r"not (supported|available)")

# Section of the verification manual of each study section of results.json
VERIFICATION_ANCHORS = {
    "time": "time-integrators",
    "reconstruction": "reconstruction-on-every-cell-type",
    "advection": "euler-equations-advection-of-a-smooth-wave",
    "viscous": "navier-stokes-equations-viscous-operators",
    "boundaries": "boundary-conditions",
    "axisymmetric": "axisymmetric-flows",
    "curved": "curved-boundaries",
    "remap": "remapping",
    "mixtures": "gas-mixtures",
}
METHOD_TOKENS = {"FO": "fo", "MUSCL": "muscl", "TENO3": "teno3", "TENO4": "teno4", "TENO5": "teno5",
                 "TENO6": "teno6", "FE": "fe", "SSPRK3": "ssprk3", "RK4": "rk4", "SSPRK54": "ssprk54",
                 "CK45": "ck45", "RDPK3Sp35": "rdpk3sp35"}
# Builds of the support matrix that stand for each parallel feature
BACKENDS = {"cuda": "CUDA", "hip": "HIP", "sycl": "SYCL"}

RANK = {"rejected": 4, "validated": 3, "verified": 2, "implemented": 1}
SYMBOL = {"rejected": "rej", "validated": "val", "verified": "ver", "implemented": "imp"}


@dataclass
class Evidence:
    status: str
    link: str
    title: str


def problem(problems, text):
    problems.append(text)


def unescape(literal):
    return literal.replace('\\"', '"').replace("\\n", " ").replace("\\\\", "\\")


def throws(path):
    """(message, line) of each throw statement of a source file: its string literals
    joined, those of `.message` alone where it has one."""
    text = path.read_text()
    out = []
    for m in re.finditer(r"\bthrow\s+[\w:]+\s*\(", text):
        end = m.end()
        for token in re.finditer(r'"(?:[^"\\]|\\.)*"|;', text[m.end():]):
            if token.group(0) == ";":
                end = m.end() + token.start()
                break
        statement = text[m.end():end]
        field = re.search(r"\.message\s*=(.*?)(,\s*\.\w+\s*=|\}\s*\)\s*$)", statement, re.S)
        literals = re.findall(r'"((?:[^"\\]|\\.)*)"', field.group(1) if field else statement)
        out.append((unescape("".join(literals)), text.count("\n", 0, m.start()) + 1))
    return out


def guard_evidence(problems):
    """Rejected pairs from the guards, and problems with GUARDS, REQUIRES and OTHER_GUARDS."""
    entries = [(f, part, (a, b, False)) for f, part, a, b in GUARDS] + \
              [(f, part, (a, b, True)) for f, part, a, b in REQUIRES] + [(f, part, None) for f, part in OTHER_GUARDS]
    used = [""] * len(entries)
    pairs = {}
    needs = [(a, b, None) for a, b in DEFINITIONS]
    for path in sorted((ROOT / "src").rglob("*")):
        if "external" in path.relative_to(ROOT / "src").parts or path.suffix not in (".cpp", ".h"):
            continue
        rel = path.relative_to(ROOT).as_posix()
        for message, line in throws(path):
            hits = [i for i, (f, part, _) in enumerate(entries) if f == rel and part in message]
            for i in hits:
                used[i] = used[i] or message
            if not hits and GUARD_WORDS.search(message):
                problem(problems, f"{rel}:{line}: a guard missing from GUARDS or OTHER_GUARDS: {message[:120]}")
    for (f, part, guarded), message in zip(entries, used):
        if not message:
            problem(problems, f"{f}: no throw says {part!r} (GUARDS, REQUIRES or OTHER_GUARDS)")
        if guarded is None:
            continue
        a, b, required = guarded
        if a not in BY_ID or b not in BY_ID:
            problem(problems, f"GUARDS or REQUIRES: unknown feature in {(a, b)}")
            continue
        ev = Evidence("rejected", f"../../{f}", message or part)
        if not required:
            pairs.setdefault(key(a, b), ev)
            continue
        needs.append((a, b, ev))
        for choice in CHOICES:
            for other in choice - {b} if b in choice else ():
                pairs.setdefault(key(a, other), ev)
    # A feature that needs another is rejected with whatever that one is rejected with
    changed = True
    while changed:
        changed = False
        for a, b, _ in needs:
            for (x, y), ev in list(pairs.items()):
                other = y if x == b else x if y == b else None
                if other and not exclusive(a, other) and key(a, other) not in pairs:
                    pairs[key(a, other)] = ev
                    changed = True
    return pairs


def key(a, b):
    return tuple(sorted((a, b), key=lambda x: ORDER.get(x, -1)))


def exclusive(a, b):
    return a == b or any(a in c and b in c for c in CHOICES)


def test_names(path):
    text = path.read_text()
    return set(f"{s}.{n}" for s, n in re.findall(r"^\s*TEST(?:_F|_P)?\(\s*(\w+)\s*,\s*(\w+)\s*\)", text, re.M))


def registry(problems):
    """{test name: (file, [runs])}, each run a list of feature ids."""
    data = tomllib.loads(REGISTRY.read_text())
    out = {}
    for stem, tests in data.items():
        path = ROOT / "test" / f"{stem}.cpp"
        if not path.exists():
            problem(problems, f"{REGISTRY.name}: [{stem}]: no test/{stem}.cpp")
            continue
        names = test_names(path)
        for name, value in tests.items():
            runs = value if value and isinstance(value[0], list) else [value]
            if name not in names:
                problem(problems, f"{REGISTRY.name}: [{stem}] {name}: no such test in test/{stem}.cpp")
            for run in runs:
                for f in run:
                    if f not in BY_ID:
                        problem(problems, f"{REGISTRY.name}: [{stem}] {name}: unknown feature {f!r}")
            out[name] = (stem, [[f for f in run if f in BY_ID] for run in runs])
    return out


def test_evidence(tests):
    """Implemented pairs, the features of any run, and those of runs of the flow solver."""
    found, tested, flow = {}, {}, {}
    for name in sorted(tests):
        stem, runs = tests[name]
        ev = Evidence("implemented", f"../../test/{stem}.cpp", name)
        for run in runs:
            for a in run:
                tested.setdefault(a, ev)
                if "euler" in run or "navier_stokes" in run:
                    flow.setdefault(a, ev)
                for b in run:
                    if not exclusive(a, b):
                        found.setdefault(key(a, b), ev)
    return found, tested, flow


def study_test(name):
    """The test of a study: GoogleTest's full name without the instantiation and parameter."""
    suite, test = name.split(".", 1)
    return suite.split("/")[-1] + "." + test.split("/")[0]


def verification_evidence(tests, problems):
    found, alone = {}, {}
    studies = json.loads(RESULTS.read_text())["studies"]
    index = (ROOT / "docs" / "verification" / "index.md").read_text()
    slugs = {slug(h) for h in re.findall(r"^#+ (.*)$", index, re.M)}
    for s in studies:
        name = study_test(s["test"])
        if name not in tests:
            problem(problems, f"{RESULTS.relative_to(ROOT)}: {name} is not in {REGISTRY.relative_to(ROOT)}")
            continue
        runs = tests[name][1]
        common = set(runs[0]).intersection(*runs[1:])
        method = {METHOD_TOKENS[t] for t in re.findall(r"[A-Za-z0-9]+", s["method"]) if t in METHOD_TOKENS}
        features = sorted(common | method, key=ORDER.get)
        anchor = VERIFICATION_ANCHORS.get(s["section"])
        if anchor not in slugs:
            problem(problems, f"no section of the verification manual for {s['section']!r} (VERIFICATION_ANCHORS)")
        title = f"{s['method']}, {s['cells']}" if s["cells"] else s["method"]
        ev = Evidence("verified", f"../verification/index.md#{anchor}", f"{title}: {s['error']}")
        for a in features:
            alone.setdefault(a, ev)
            for b in features:
                if not exclusive(a, b):
                    found.setdefault(key(a, b), ev)
    return found, alone


def slug(heading):
    return re.sub(r"[\s]+", "-", re.sub(r"[^\w\s-]", "", heading.strip().lower()))


def presets(problems):
    """{preset: {(table path...): value}} from the preset table of the source."""
    text = PRESETS.read_text()
    out = {}
    for name, body in re.findall(r'\{"(\w+)", \{(.*?)\n        \}\}', text, re.S):
        entries = {}
        for path, value in re.findall(r"\{(reconstruction\(\"\w+\"\)|\{[^}]*\}), Input\(([^)]*)\)\}", body):
            parts = re.findall(r'"(\w+)"', path)
            if path.startswith("reconstruction"):
                parts = ["numerics", "face_reconstruction"] + parts
            entries[tuple(parts)] = json.loads(value) if value.startswith('"') else float(value)
        out[name] = entries
    if not out:
        problem(problems, f"{PRESETS.relative_to(ROOT)}: no presets found")
    default = re.search(r'DEFAULT = "(\w+)"', PRESETS_HEADER.read_text())
    if not default or default[1] not in out:
        problem(problems, f"{PRESETS_HEADER.relative_to(ROOT)}: no DEFAULT preset found")
    else:
        out[None] = out[default[1]]
    return out


def with_preset(t, table):
    name = numerics(t).get("preset")
    if not name and "face_reconstruction" in numerics(t):
        return t
    entries = table.get(name, {})
    out = json.loads(json.dumps(t))
    rec_type = recon(t).get("type")
    for path, value in entries.items():
        if path[:2] == ("numerics", "face_reconstruction") and path[2] != "type" and rec_type not in (
                None, entries.get(("numerics", "face_reconstruction", "type"))):
            continue
        node = out
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node.setdefault(path[-1], int(value) if isinstance(value, float) and value.is_integer() else value)
    return out


def validation_evidence(problems):
    found, alone = {}, {}
    table = presets(problems)
    for card in sorted((ROOT / "examples").glob("*/card.toml")):
        entries = tomllib.loads(card.read_text()).get("validation", [])
        for entry in entries:
            path = card.parent / entry.get("input", "input.toml")
            if not path.exists() or "anchor" not in entry or "regime" not in entry:
                problem(problems, f"{card.relative_to(ROOT)}: a validation entry needs anchor, regime and an "
                                  f"existing input")
                continue
            t = with_preset(tomllib.loads(path.read_text()), table)
            features = [f.id for f in FEATURES if f.has and f.has(t)]
            ev = Evidence("validated", VALIDATION + "#" + entry["anchor"],
                          f"{card.parent.name}: {entry['regime']}")
            for a in features:
                alone.setdefault(a, ev)
                for b in features:
                    if not exclusive(a, b):
                        found.setdefault(key(a, b), ev)
    return found, alone


def support_builds():
    """(build, tested, what runs) of each row of the support matrix's table of builds."""
    text = SUPPORT.read_text()
    section = text.split("## Builds", 1)[1].split("\n## ", 1)[0]
    rows = []
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 4 and cells[0] not in ("Build", "") and not cells[0].startswith("---"):
            rows.append((cells[0], cells[2], cells[3]))
    return rows


def backend_evidence(flow, problems):
    """Pairs of MPI and the GPU backends with every feature a test runs the flow solver with, from the
    builds that run the full suite."""
    found = {}
    rows = support_builds()
    runs_suite = {}
    for b, name in BACKENDS.items():
        for build, status, what in rows:
            if build.startswith(name) and status == "Run" and what.startswith("Full suite"):
                runs_suite.setdefault(b, build)
            if build.startswith(name) and status == "Run" and "MPI tests on" in what:
                found.setdefault(key(b, "mpi"), Evidence("implemented", "support.md#builds", f"{build}: {what}"))
    for build, status, what in rows:
        if re.search(r"\bMPI\b", build) and status == "Run" and re.match(r"Full suite on \d", what):
            runs_suite.setdefault("mpi", build)
    if "mpi" not in runs_suite:
        problem(problems, f"{SUPPORT.relative_to(ROOT)}: no MPI build runs the full suite")
    for b, build in runs_suite.items():
        ev = Evidence("implemented", "support.md#builds", f"full test suite: {build}")
        for a in flow:
            if not exclusive(a, b):
                found.setdefault(key(a, b), ev)
    return found, runs_suite


def collect(problems):
    tests = registry(problems)
    rejected = guard_evidence(problems)
    validated, validated_alone = validation_evidence(problems)
    verified, verified_alone = verification_evidence(tests, problems)
    implemented, tested, flow = test_evidence(tests)
    backends, runs_suite = backend_evidence(flow, problems)
    for source in (validated, verified, implemented):
        for k, ev in source.items():
            if k in rejected:
                problem(problems, f"{ev.title} ({ev.link}): runs {' and '.join(k)}, which a guard rejects")
    cells = {}
    for source in (rejected, validated, verified, implemented, backends):
        for k, ev in source.items():
            if k not in cells or RANK[ev.status] > RANK[cells[k].status]:
                cells[k] = ev
    alone = {}
    for source in (validated_alone, verified_alone, tested):
        for a, ev in source.items():
            if a not in alone or RANK[ev.status] > RANK[alone[a].status]:
                alone[a] = ev
    for b in runs_suite:
        alone.setdefault(b, Evidence("implemented", "support.md#builds", f"full test suite: {runs_suite[b]}"))
    return cells, alone


def cell(ev):
    if ev is None:
        return "·"
    title = re.sub(r"[()]", "", ev.title.replace('"', "'").replace("|", "/"))
    return f'[{SYMBOL[ev.status]}]({ev.link} "{title}")'


def features_of(groups):
    return [f for f in FEATURES if f.group in groups]


def table(cells, alone, rows, cols):
    head = "| | " + " | ".join(f.label for f in features_of(cols)) + " |"
    out = [head, "|---|" + "---|" * len(features_of(cols))]
    for r in features_of(rows):
        line = []
        for c in features_of(cols):
            if r.id == c.id:
                line.append(f"**{cell(alone.get(r.id))}**")
            elif exclusive(r.id, c.id):
                line.append("")
            else:
                line.append(cell(cells.get(key(r.id, c.id))))
        out.append(f"| **{r.label}** | " + " | ".join(line) + " |")
    return "\n".join(out)


def pairs():
    out = []
    for i, a in enumerate(FEATURES):
        for b in FEATURES[i + 1:]:
            if not exclusive(a.id, b.id):
                out.append((a, b))
    return out


def blocks(cells, alone):
    out = {}
    counts = {s: 0 for s in RANK}
    untested = 0
    for a, b in pairs():
        ev = cells.get(key(a.id, b.id))
        if ev:
            counts[ev.status] += 1
        else:
            untested += 1
    total = len(pairs())
    out["summary"] = (f"{total} pairs of {len(FEATURES)} features: " + ", ".join(
        f"{counts[s]} {s}" for s in ("validated", "verified", "implemented", "rejected")) + f", {untested} untested.")
    for _, name, rows, cols in TABLES:
        out[name] = table(cells, alone, rows, cols)
    gaps = []
    for f in FEATURES:
        missing = [b.label for a, b in pairs() if a.id == f.id and key(a.id, b.id) not in cells]
        if missing:
            gaps.append(f"- **{f.label}** with " + ", ".join(missing))
    alone_gaps = [f.label for f in FEATURES if f.id not in alone]
    if alone_gaps:
        gaps.append("- **Alone, in no test:** " + ", ".join(alone_gaps))
    out["gaps"] = "\n".join(gaps)
    return out


def render(text, generated):
    for name, body in generated.items():
        pattern = re.compile(rf"(<!-- BEGIN GENERATED {name} -->\n).*?(<!-- END GENERATED {name} -->)", re.S)
        if not pattern.search(text):
            sys.exit(f"{PAGE.relative_to(ROOT)}: no generated block {name}")
        text = pattern.sub(lambda m: m.group(1) + body + "\n" + m.group(2), text)
    return text


def main():
    check = "--check" in sys.argv
    problems = []
    cells, alone = collect(problems)
    text = PAGE.read_text()
    new = render(text, blocks(cells, alone))
    if new != text:
        if check:
            problems.append(f"{PAGE.relative_to(ROOT)} is out of date: run python3 tools/docs/feature_matrix.py")
        else:
            PAGE.write_text(new)
    for p in dict.fromkeys(problems):
        print(p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
