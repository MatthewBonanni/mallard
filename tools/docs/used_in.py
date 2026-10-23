"""The "Used in" line under each feature's opening snippet in the user guide: the
examples whose input files use the feature, found by reading them.

    python tools/docs/used_in.py           rewrite the lines in docs/guide/
    python tools/docs/used_in.py --check   exit 1 if a line is out of date

Each section of RULES names a guide page and heading, and a test of a parsed input
file. The line is the first one starting with "*Used in:*" in that section; it
lists up to five examples in the order of examples/README.md, linked to their cards
on the website's Examples page, or the rule's fallback text when no example uses the
feature.
"""
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "docs" / "guide"
CARDS = "https://matthewbonanni.github.io/mallard/docs/examples/"
MAX = 5


def mesh(t):
    return t.get("mesh", {})


def num(t):
    return t.get("numerics", {})


def boundaries(t):
    return t.get("boundaries", [])


def recon(t):
    return num(t).get("face_reconstruction", {})


RULES = {
    ("boundary_conditions.md", "## Choosing a condition"): (lambda t: bool(boundaries(t)), None),
    ("boundary_conditions.md", "## Characteristic boundaries"):
        (lambda t: any(b.get("type", "").startswith("nscbc") for b in boundaries(t)), None),
    ("boundary_conditions.md", "## Synthetic turbulence"):
        (lambda t: any("turbulence" in b for b in boundaries(t)), None),
    ("initialization.md", "## Uniform state"): (lambda t: t.get("initialize", {}).get("type") == "constant", None),
    ("initialization.md", "## Expressions"): (lambda t: t.get("initialize", {}).get("type") == "analytical", None),
    ("initialization.md", "## Restarts"): (lambda t: t.get("initialize", {}).get("type") == "restart", None),
    ("initialization.md", "## Remap from another mesh"):
        (lambda t: t.get("initialize", {}).get("type") == "remap", None),
    ("les.md", "## A minimal LES"): (lambda t: "les" in t, None),
    ("les.md", "## Subgrid-scale models: `[les]`"): (lambda t: "model" in t.get("les", {}), None),
    ("les.md", "## Low-dissipation flux and the budget"):
        (lambda t: num(t).get("convective_flux") == "hybrid", None),
    ("les.md", "## Thickened flames: `[les.combustion]`"):
        (lambda t: t.get("les", {}).get("combustion", {}).get("model") == "tfles", None),
    ("les.md", "### PaSR (experimental)"):
        (lambda t: t.get("les", {}).get("combustion", {}).get("model") == "pasr", "no example yet"),
    ("les.md", "## Wall models: `wall_model`"): (lambda t: any("wall_model" in b for b in boundaries(t)), None),
    ("meshes.md", "## Generated meshes"): (lambda t: mesh(t).get("type", "file") != "file", None),
    ("meshes.md", "### Stretching"): (lambda t: "stretching" in mesh(t), None),
    ("meshes.md", "## Periodic boundaries"): (lambda t: "periodic" in mesh(t) or "periodic" in t, None),
    ("meshes.md", "## Mesh files (Gmsh)"): (lambda t: str(mesh(t).get("filename", "")).endswith(".msh"), None),
    ("meshes.md", "## Other mesh formats"):
        (lambda t: re.search(r"\.(cgns|cas)$|polyMesh", str(mesh(t).get("filename", ""))) is not None,
         "the mesh-import tests (`test/data/meshes/`)"),
    ("meshes.md", "## Transform on read"): (lambda t: "transform" in mesh(t), "the mesh-import tests"),
    ("meshes.md", "## HDF5 meshes for large runs"):
        (lambda t: str(mesh(t).get("filename", "")).endswith(".h5"),
         "any example, after `mallard-mesh-convert input.toml mesh.h5`"),
    ("meshes.md", "## Curved boundaries"):
        (lambda t: "curved" in mesh(t), "the curved-boundary validation (`tools/curved_convergence.py`)"),
    ("numerics.md", "## Presets"): (lambda t: "preset" in num(t), None),
    ("numerics.md", "## Time step and stopping"): (lambda t: "cfl" in t.get("run", {}), None),
    ("numerics.md", "## Reconstruction"): (lambda t: "type" in recon(t), None),
    ("numerics.md", "### TENO-E options"): (lambda t: recon(t).get("bound_preserving", False), None),
    ("numerics.md", "## Riemann solvers"): (lambda t: "riemann_solver" in num(t), None),
    ("numerics.md", "## Time integration and other options"):
        (lambda t: num(t).get("time_integrator", "SSPRK3") != "SSPRK3",
         "every example keeps the default, `SSPRK3`; the others are in the time-integration tests"),
    ("output.md", "## Snapshots: `[[write_data]]`"): (lambda t: "write_data" in t, None),
    ("output.md", "## The log"): (lambda t: "check_interval" in t.get("output", {}), None),
    ("output.md", "## Wall forces: `[[forces]]`"): (lambda t: "forces" in t, None),
    ("output.md", "## Domain integrals: `[integrals]`"): (lambda t: "integrals" in t, None),
    ("output.md", "## Statistics"): (lambda t: "statistics" in t, None),
    ("output.md", "## Probes: `[[probes]]`"): (lambda t: "probes" in t, None),
    ("physics.md", "## Euler and Navier–Stokes"): (lambda t: "type" in t.get("physics", {}), None),
    ("physics.md", "### Viscosity laws"): (lambda t: "viscosity_model" in t.get("physics", {}), None),
    ("physics.md", "## Gas mixtures"): (lambda t: t.get("physics", {}).get("gas") == "mixture", None),
    ("physics.md", "## Axisymmetric flows"): (lambda t: t.get("physics", {}).get("axisymmetric", False), None),
    ("sources.md", "## `[source]`"): (lambda t: "source" in t, None),
    ("sources.md", "### Constant mass flow"): (lambda t: "mass_flow" in t.get("source", {}), None),
    ("sources.md", "### Gravity"): (lambda t: "gravity" in t.get("source", {}), None),
    ("sources.md", "## Sponge layers"): (lambda t: "sponges" in t, None),
    ("thermochemistry.md", "## Gas mixtures"): (lambda t: t.get("physics", {}).get("gas") == "mixture", None),
    ("thermochemistry.md", "### Contacts between different gases"):
        (lambda t: num(t).get("double_flux", False), "the multicomponent shock-tube tests"),
    ("thermochemistry.md", "## Finite-rate chemistry"): (lambda t: "chemistry" in t, None),
    ("thermochemistry.md", "## Radiation"): (lambda t: "radiation" in t, None),
    ("thermochemistry.md", "## 0D reactors: `MallardReactor`"): (lambda t: "reactor" in t, None),
}


def examples():
    """(name, parsed input files) of each example, in the order of examples/README.md."""
    readme = (ROOT / "examples" / "README.md").read_text()
    order = list(dict.fromkeys(re.findall(r"^\| `([a-z0-9_]+)", readme, re.M)))
    out = []
    for name in order:
        inputs = []
        for f in sorted((ROOT / "examples" / name).rglob("*.toml")):
            if f.name == "card.toml" or "reference" in f.parts:
                continue
            try:
                inputs.append(tomllib.loads(f.read_text()))
            except tomllib.TOMLDecodeError:
                pass
        out.append((name, inputs))
    return out


def line(test, fallback, cases):
    names = [n for n, inputs in cases if any(test(t) for t in inputs)]
    if not names:
        return f"*Used in:* {fallback or 'no example yet'}."
    links = [f"[`{n}`]({CARDS}#{n.replace('_', '-')})" for n in names[:MAX]]
    more = f" and {len(names) - MAX} more" if len(names) > MAX else ""
    return "*Used in:* " + ", ".join(links) + more + "."


def section_end(text, start, level):
    """End of the section starting at START: the next heading of LEVEL or higher, outside code."""
    pos, fence = start, False
    for ln in text[start:].splitlines(keepends=True)[1:]:
        pos += len(text[pos:].split("\n", 1)[0]) + 1
        if ln.startswith("```"):
            fence = not fence
        elif not fence and re.match(rf"#{{1,{level}}} ", ln):
            return pos
    return len(text)


def update(check):
    cases = examples()
    stale = []
    for (page, heading), (test, fallback) in RULES.items():
        path = GUIDE / page
        text = path.read_text()
        start = text.index(heading + "\n")
        end = section_end(text, start, len(heading.split(" ")[0]))
        m = re.compile(r"^\*Used in:\*.*$", re.M).search(text, start, end)
        if not m:
            sys.exit(f"{page}: no *Used in:* line under {heading!r}")
        new = line(test, fallback, cases)
        if m.group(0) != new:
            stale.append(f"{page}: {heading}")
            if not check:
                path.write_text(text[:m.start()] + new + text[m.end():])
    if check and stale:
        sys.exit("out of date (python tools/docs/used_in.py):\n  " + "\n  ".join(stale))


if __name__ == "__main__":
    update("--check" in sys.argv)
