"""The coefficients of mallard check's cost estimate (src/solver/cost_model_data.h), from measured runs only.

Three steps, each reading what the previous one wrote:

    python tools/perf/cost_model.py measure --mallard2d B2 --mallard3d B3 --out DIR   # local CPU runs (optional)
    python tools/perf/cost_model.py collect --runs DIR ... --gui-table FILE --mallard2d B2 --mallard3d B3
    python tools/perf/cost_model.py fit

`measure` runs the CPU cases of CPU_CASES on this machine (at most 3 ranks, MPI builds), each in DIR/<name>/:
strong scaling over 1 to 3 ranks of one thread, one rank on two threads, reacting boxes of mechanisms of 6 to 53
species, and tracers. Runs already finished are skipped.

`collect` writes the measurements, with where each comes from, to benchmarks/cost_model/:

- throughput.csv: the throughput of one rank alone (cells/s; per thread on CPUs) by device class, dimension,
  reconstruction, physics and species: every entry of the GUI's measured table (`--gui-table`, mallard-gui's
  throughput_table.py, which gathers the performance set's baselines, docs/guide/performance.md and campaign
  runs), and every single-rank run of `--runs` not in it.
- scaling.csv: the time per step of every run of `--runs` (median over repetitions), with the largest part of its
  partition as `mallard check --ranks N --ranks-per-node P --format json` gives it (owned and halo cells,
  neighbors, halo cells and neighbors on other nodes). A periodic box too large for a serial check here
  (`--max-check-cells`) gets its partition from checks of the same box with its cells coarsened f and 2f times:
  the factorization of the ranks into equal blocks and the halo depth that reproduce both, scaled to the box
  (exact for equal cuboid blocks; anything else is an error).

Run directories hold `input.toml` and `mallard.jsonl`, and from the campaign driver `result.json` (its median wall
time per step and repetitions); the run's start record gives the build, the device, the placement and the resolved
input. Runs named by SKIP are left out, and reacting runs are kept out of the step-time fit (their chemistry, where
the flame is, unbalances the ranks; the fit reports their error separately).

`fit` writes src/solver/cost_model_data.h from the two files:

- Throughput: the measured keys as they are (check takes the geometric mean of a key's measurements), and for other
  keys a least-squares fit of log(throughput) = base(device, dim) + reconstruction(kind) + physics(kind) +
  slope(kind) log((species + 1) / 7) over every measurement, kind CPU or GPU. Its error is the root mean square of
  the log error of each measurement predicted by the fit to the others (leave one out); the error of a measured key
  is the spread between the cases of one key.
- Step time of the largest part on N ranks, per device class: s (o + beta h) + t0 + gamma k + delta h_remote +
  particle P (+ particle_mpi with particles on several ranks), s the seconds per cell of the case on one rank,
  fitted with the coefficients (least squares on the relative error, coefficients >= 0). Its held-out error: each
  case predicted by the coefficients fitted to the other cases, from its own single-rank step.

Requires numpy and scipy (tools/README.md).
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
import tomllib
from collections import defaultdict
from dataclasses import dataclass, field, fields
from datetime import date
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "benchmarks" / "cost_model"
THROUGHPUT_CSV = DATA / "throughput.csv"
SCALING_CSV = DATA / "scaling.csv"
HEADER = ROOT / "src" / "solver" / "cost_model_data.h"

SKIP = ("nsys", "rt2d", "rt3d", "pre1042", "-1042-", "-exact", "smoke", "pwake", "records", "c1077")
RECONSTRUCTIONS = ("fo", "muscl", "teno3", "teno4", "teno5", "teno6")
PHYSICS = ("euler", "ns", "les", "reacting")
REFERENCE_SPECIES = 6


# ------------------------------------------------------------------ measurements


@dataclass
class ThroughputRow:
    device: str
    hardware: str
    dim: int
    reconstruction: str
    physics: str
    species: int
    cells_per_second: float  # per thread on CPUs
    threads: int
    cells: int
    case: str
    date: str
    source: str


@dataclass
class ScalingRow:
    device: str
    hardware: str
    case: str  # runs of one case share their cost per cell (same build, physics, numerics and cell type)
    dim: int
    reconstruction: str
    physics: str
    species: int
    ranks: int
    ranks_per_node: int  # 0: one node
    cells: int
    owned: int
    halo: int
    neighbors: int
    remote_neighbors: int
    halo_remote: int
    particles: int  # slots per rank
    seconds_per_step: float
    repetitions: int
    use: str  # "fit", or why not
    date: str
    source: str
    partition: str = ""  # how the partition was found


def write_csv(path: Path, rows: list, kind: type) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [f.name for f in fields(kind)]
    with path.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(names)
        for r in rows:
            w.writerow([getattr(r, n) for n in names])


def read_csv(path: Path, kind: type) -> list:
    types = {f.name: f.type for f in fields(kind)}
    out = []
    with path.open() as f:
        for row in csv.DictReader(f):
            out.append(kind(**{k: (int(v) if types[k] == "int" else float(v) if types[k] == "float" else v)
                               for k, v in row.items()}))
    return out


# ------------------------------------------------------------------ inputs and runs


def device_class(name: str) -> str:
    if "A100" in name:
        return "a100"
    if name.startswith("Apple M"):
        return "cpu-apple-m"
    return "gpu" if any(k in name for k in ("NVIDIA", "AMD Instinct", "Intel(R) Data Center GPU")) else "cpu"


def species_count(mechanism: str) -> int:
    path = Path(mechanism)
    if not path.exists():
        path = ROOT / "mechanisms" / path.name
    text = path.read_text()
    phases = text.split("\nspecies:")[0]
    m = re.search(r"species:\s*\[([^\]]*)\]", phases, re.S)
    return len([s for s in m.group(1).split(",") if s.strip()]) if m else 0


def classify(inp: dict) -> tuple[str, str, int]:
    """(reconstruction, physics, species) of a resolved input."""
    rec = inp.get("numerics", {}).get("face_reconstruction", {})
    kind = rec.get("type", "FO")
    reconstruction = {"FO": "fo", "MUSCL": "muscl"}.get(kind, f"teno{rec.get('order', 5)}")
    phys = inp.get("physics", {})
    mixture = phys.get("gas") == "mixture"
    species = species_count(phys["mechanism"]) if mixture and "mechanism" in phys else 0
    if mixture and "chemistry" in inp:
        return reconstruction, "reacting", species
    if "les" in inp:
        return reconstruction, "les", species
    return reconstruction, "ns" if phys.get("type") == "navier_stokes" else "euler", species


@dataclass
class Run:
    path: Path
    name: str  # without the repetition
    case: str
    device: str
    hardware: str
    dim: int
    ranks: int
    ranks_per_node: int
    threads: int
    cells: int
    particles: int
    reconstruction: str
    physics: str
    species: int
    seconds_per_step: float
    date: str
    sha: str
    input_text: str
    resolved: dict = field(default_factory=dict)


def particle_slots(inp: dict) -> int:
    p = inp.get("particles")
    if not p:
        return 0
    if p.get("capacity"):
        return int(p["capacity"])
    n = len(p.get("points", []))
    if "lattice" in p:
        n += math.prod(p["lattice"]["n"])
    return n


def read_run(d: Path) -> Run | None:
    lines = (d / "mallard.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    start = next((r for r in records if r.get("type") == "start"), None)
    end = next((r for r in records if r.get("type") == "end"), None)
    if start is None or end is None or end.get("outcome") not in ("finished", "wall_time_limit"):
        return None
    prov = start["provenance"]
    resolved = tomllib.loads(prov["input"]["resolved"])
    hosts = [p["host"] for p in start.get("placement") or []] or [prov["run"].get("host", "")]
    ranks = int(prov["run"]["ranks"])
    per_node = max(hosts.count(h) for h in set(hosts))
    progress = [r for r in records if r.get("type") == "progress" and r.get("wall_per_step")]
    result = json.loads((d / "result.json").read_text()) if (d / "result.json").exists() else {}
    cells = int(progress[-1]["cells"]) if progress else 0
    if not cells:
        m = resolved["mesh"]
        cells = int(m.get("Nx", 0)) * int(m.get("Ny", 1)) * int(m.get("Nz", 1))
    if result.get("median_wall_per_step"):
        step = float(result["median_wall_per_step"]) / 1e3
    elif len(progress) >= 3:
        step = statistics.median(r["wall_per_step"] for r in progress[len(progress) // 3:])
    else:
        step = cells / float(end["cells_per_second"])
    name = result.get("name") or re.sub(r"-r\d+-[0-9a-f]{8}$", "", d.name)
    device = prov["run"].get("device", "")
    reconstruction, physics, species = classify(resolved)
    family = re.sub(r"-n\d+(x\d+)?(?=-|$)", "", name)
    family = re.sub(r"^p\d+-", "", family)  # tracers: the same case with particles
    sha = prov["mallard"]["git_sha"][:10]
    return Run(path=d, name=name, case=f"{d.parent.parent.name}/{family}@{sha}", device=device_class(device),
               hardware=re.sub(r" \(\d+ GB\)$", "", device), dim=int(prov["build"]["dim"]), ranks=ranks,
               ranks_per_node=0 if per_node >= ranks else per_node,
               threads=int((start.get("placement") or [{}])[0].get("threads", 1)), cells=cells,
               particles=particle_slots(resolved), reconstruction=reconstruction, physics=physics,
               species=species, seconds_per_step=step, date=prov["run"]["start_time"][:10], sha=sha,
               input_text=(d / "input.toml").read_text(), resolved=resolved)


def find_runs(dirs: list[Path]) -> list[Run]:
    runs = []
    for top in dirs:
        for log in sorted(list(top.glob("runs/*/mallard.jsonl")) + list(top.glob("*/mallard.jsonl"))):
            if any(s in log.parent.name for s in SKIP):
                continue
            run = read_run(log.parent)
            if run is not None:
                runs.append(run)
    return runs


# ------------------------------------------------------------------ partitions


def check_partition(binary: str, input_text: str, ranks: int, per_node: int, work: Path) -> dict:
    """The largest part as `mallard check --ranks N --ranks-per-node P` gives it (cached by content)."""
    key = hashlib.sha1(f"{binary}\0{input_text}\0{ranks}\0{per_node}".encode()).hexdigest()[:16]
    cache = work / f"{key}.json"
    if not cache.exists():
        d = work / key
        d.mkdir(parents=True, exist_ok=True)
        # Mechanisms from where the run read them to this checkout's
        local = re.sub(r'(?m)^(mechanism\s*=\s*)"([^"]*)"',
                       lambda m: f'{m.group(1)}"{ROOT / "mechanisms" / Path(m.group(2)).name}"', input_text)
        (d / "input.toml").write_text(local)
        out = subprocess.run([binary, "check", "input.toml", "--ranks", str(ranks), "--ranks-per-node",
                              str(per_node), "--format", "json", "--kokkos-num-threads=2"],
                             cwd=d, capture_output=True, text=True)
        memory = json.loads(out.stdout).get("memory")
        if not memory:
            raise RuntimeError(f"no partition from {binary} check of {d}/input.toml: {out.stderr[-2000:]}")
        cache.write_text(json.dumps(memory))
    m = json.loads(cache.read_text())
    return {"owned": m["target_owned"], "halo": m["target_halo"], "neighbors": m["target_neighbors"],
            "remote_neighbors": m["target_remote_neighbors"], "halo_remote": m["target_halo_remote"]}


def coarsened(text: str, inp: dict, f: int) -> str:
    """The input text with its Cartesian box's cells coarsened f times along each side."""
    m = inp["mesh"]
    for k in ("Nx", "Ny", "Nz"):
        if k in m:
            if int(m[k]) % f:
                raise RuntimeError(f"{k} = {m[k]} is not a multiple of {f}")
            text, n = re.subn(rf"(?m)^{k}\s*=\s*\d+\s*$", f"{k} = {int(m[k]) // f}", text, count=1)
            if n != 1:
                raise RuntimeError(f"no line {k} = ... in the input")
    return text


def box_partition(binary: str, run: Run, work: Path, max_cells: int) -> dict:
    """The partition of a periodic Cartesian box too large to check here, from checks of it coarsened f and 2f times
    (f the smallest power of 2 that fits)."""
    m = run.resolved["mesh"]
    sides = [int(m[k]) for k in ("Nx", "Ny", "Nz")[: run.dim]]
    f = 2
    while run.cells / f ** run.dim > max_cells:
        f *= 2
    coarse = {k: check_partition(binary, coarsened(run.input_text, run.resolved, k), run.ranks,
                                 run.ranks_per_node, work)
              for k in (f, 2 * f)}
    periodic = {z for p in run.resolved.get("periodic", []) for z in p["zones"]}
    if len(periodic) != 2 * run.dim:
        raise RuntimeError(f"{run.path}: not a periodic box, and too large to check here")
    found = []
    for counts in product(*[[n for n in range(1, run.ranks + 1) if run.ranks % n == 0]] * run.dim):
        if math.prod(counts) != run.ranks:
            continue
        for layers in range(1, 8):
            ok = True
            for k, part in coarse.items():
                block = [s // k // n for s, n in zip(sides, counts)]
                if any((s // k) % n for s, n in zip(sides, counts)) or min(block) < layers:
                    ok = False
                    break
                halo = math.prod(b + 2 * layers * (n > 1) for b, n in zip(block, counts)) - math.prod(block)
                if part["owned"] != math.prod(block) or part["halo"] != halo:
                    ok = False
                    break
            if ok:
                found.append((counts, layers))

    def full(counts, layers):
        block = [s // n for s, n in zip(sides, counts)]
        return math.prod(block), math.prod(b + 2 * layers * (n > 1) for b, n in zip(block, counts)) - math.prod(block)

    if not found or len({full(*x) for x in found}) != 1:
        raise RuntimeError(f"{run.path}: {len(found)} block factorizations reproduce the coarsened partitions")
    counts, layers = found[0]
    owned, halo = full(counts, layers)
    cut = [n > 1 for n in counts]
    remote = coarse[f]["halo_remote"]
    if remote and sum(cut) > 1:
        raise RuntimeError(f"{run.path}: halo on other nodes of blocks cut along several sides")
    # Blocks cut along one side: every halo cell lies across a face, so halo on other nodes scales as the face
    return {"owned": owned, "halo": halo, "neighbors": coarse[f]["neighbors"],
            "remote_neighbors": coarse[f]["remote_neighbors"],
            "halo_remote": remote * halo // max(coarse[f]["halo"], 1),
            "how": f"blocks {'x'.join(map(str, counts))}, {layers} halo layers, from checks coarsened {f} and "
                   f"{2 * f} times"}


def partition(run: Run, binaries: dict, max_cells: int, work: Path) -> dict:
    if run.ranks == 1:
        return {"owned": run.cells, "halo": 0, "neighbors": 0, "remote_neighbors": 0, "halo_remote": 0, "how": "one rank"}
    if run.cells <= max_cells:
        p = check_partition(binaries[run.dim], run.input_text, run.ranks, run.ranks_per_node, work)
        return p | {"how": "mallard check"}
    return box_partition(binaries[run.dim], run, work, max_cells)


# ------------------------------------------------------------------ collect


def gui_table(path: Path) -> list[ThroughputRow]:
    """The entries of mallard-gui's throughput_table.py (TABLE), without importing it."""
    tree = ast.parse(path.read_text())
    rows = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "Entry":
            kw = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
            rows.append(ThroughputRow(device=kw["device_class"], hardware=kw["hardware"], dim=kw["dim"],
                                      reconstruction=kw["reconstruction"], physics=kw["physics"],
                                      species=kw["species"], cells_per_second=kw["cells_per_second_per_rank"],
                                      threads=kw["threads"], cells=kw["cells"], case=kw["case"], date=kw["date"],
                                      source=f"mallard-gui throughput table: {kw['source']}"))
    return rows


def collect(args) -> None:
    binaries = {2: str(Path(args.mallard2d).resolve()), 3: str(Path(args.mallard3d).resolve())}
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    runs = find_runs([Path(d) for d in args.runs])
    throughput = gui_table(Path(args.gui_table)) if args.gui_table else []
    known = " ".join(r.source for r in throughput)

    groups = defaultdict(list)
    for r in runs:
        groups[(r.case, r.ranks, r.ranks_per_node, r.cells, r.particles, r.threads)].append(r)
    scaling = []
    for (case, ranks, per_node, cells, particles, threads), reps in sorted(groups.items()):
        r = reps[0]
        # CPUs: the fastest repetition, since other jobs on a shared machine only slow a run down
        steps = [x.seconds_per_step for x in reps]
        step = min(steps) if r.device.startswith("cpu") else statistics.median(steps)
        names = ", ".join(sorted({x.path.name for x in reps}))
        source = f"{r.path.parent.parent.name if r.path.parent.name != r.name else r.path.parent.name}: {names}"
        if ranks == 1 and particles == 0 and r.name not in known:
            per = threads if r.device.startswith("cpu") else 1
            throughput.append(ThroughputRow(device=r.device, hardware=r.hardware, dim=r.dim,
                                            reconstruction=r.reconstruction, physics=r.physics,
                                            species=r.species, cells_per_second=cells / step / per,
                                            threads=threads if per > 1 else 0, cells=cells, case=r.name,
                                            date=r.date, source=f"{source} (Mallard {r.sha})"))
        if r.device.startswith("cpu") and threads != 1:
            use = "threads: the step-time fit is per rank of one thread"
        elif r.physics == "reacting":
            use = "reacting: the flame's chemistry unbalances the ranks"
        else:
            use = "fit"
        p = partition(r, binaries, args.max_check_cells, work)
        scaling.append(ScalingRow(device=r.device, hardware=r.hardware, case=case, dim=r.dim,
                                  reconstruction=r.reconstruction, physics=r.physics, species=r.species,
                                  ranks=ranks, ranks_per_node=per_node, cells=cells, owned=p["owned"],
                                  halo=p["halo"], neighbors=p["neighbors"], remote_neighbors=p["remote_neighbors"],
                                  halo_remote=p["halo_remote"], particles=particles, seconds_per_step=step,
                                  repetitions=len(reps), use=use, date=r.date, source=f"{source} (Mallard {r.sha})",
                                  partition=p["how"]))
        print(f"{case} n{ranks} p{per_node}: {step * 1e3:.3f} ms/step, owned {p['owned']}, halo {p['halo']}, "
              f"neighbors {p['neighbors']} ({p['how']})", file=sys.stderr)
    throughput.sort(key=lambda t: (t.device, t.dim, t.reconstruction, t.physics, t.species, t.date, t.case))
    write_csv(THROUGHPUT_CSV, throughput, ThroughputRow)
    write_csv(SCALING_CSV, scaling, ScalingRow)
    print(f"wrote {THROUGHPUT_CSV.relative_to(ROOT)} ({len(throughput)} rows), "
          f"{SCALING_CSV.relative_to(ROOT)} ({len(scaling)} rows)")


# ------------------------------------------------------------------ measure (local CPU runs)

TWO_PI = 6.283185307179586
P0 = 71.42857142857143


def tgv(dim: int, n: int, rec: str, wall: float, tracers: int = 0) -> str:
    mesh = f'type = "cartesian"\nNx = {n}\nNy = {n}\nLx = {TWO_PI}\nLy = {TWO_PI}\n'
    zones = [("left", "right"), ("bottom", "top")]
    if dim == 3:
        mesh += f"Nz = {n}\nLz = {TWO_PI}\n"
        zones.append(("back", "front"))
        u = '["sin(x) * cos(y) * cos(z)", "-cos(x) * sin(y) * cos(z)", "0.0"]'
        p = f'"{P0} + (cos(2 * x) + cos(2 * y)) * (cos(2 * z) + 2) / 16"'
    else:
        u = '["sin(x) * cos(y)", "-cos(x) * sin(y)"]'
        p = f'"{P0} + (cos(2 * x) + cos(2 * y)) / 4"'
    face = '{ type = "MUSCL" }' if rec == "muscl" else f'{{ type = "TENO", order = {rec[4:]} }}'
    periodic = "".join(f'\n[[periodic]]\nzones = ["{a}", "{b}"]\n' for a, b in zones)
    text = f"""[run]
n_steps = 1000000
cfl = 0.4
t_wall_stop = {wall}

[mesh]
{mesh}{periodic}
[initialize]
type = "analytical"
u = {u}
p = {p}
T = "1.0"

[numerics]
riemann_solver = "HLLC"
time_integrator = "SSPRK3"
face_reconstruction = {face}

[physics]
type = "navier_stokes"
units = "nondimensional"
gamma = 1.4
p_ref = {P0}
T_ref = 1.0
rho_ref = 1.0
mu = 0.000625
Pr = 0.71

[output]
check_interval = 20
"""
    if tracers:
        h = TWO_PI / n
        lo = ", ".join([repr(0.25 * h)] * dim)
        hi = ", ".join([repr(TWO_PI - 0.25 * h)] * dim)
        counts = ", ".join(str(n * (tracers if i == 0 else 1)) for i in range(dim))
        text += f'\n[particles]\ntype = "tracer"\n\n[particles.lattice]\nlo = [{lo}]\nhi = [{hi}]\nn = [{counts}]\n'
    return text


# Homogeneous igniting mixtures in a periodic box, with the box's vortex: the chemistry of every cell is active
MIXTURES = {
    "c3h8_2step": ("c3h8_2step", "{ C3H8 = 1.0, O2 = 5.0, N2 = 18.8 }", 1600.0),
    "h2o2": ("ohmech", "{ H2 = 2.0, O2 = 1.0, N2 = 3.76 }", 1100.0),
    "gri30": ("gri30", "{ CH4 = 1.0, O2 = 2.0, N2 = 7.52 }", 1600.0),
}


def reacting_box(mechanism: str, n: int, wall: float) -> str:
    phase, x, temperature = MIXTURES[mechanism]
    return f"""[run]
n_steps = 1000000
cfl = 0.8
t_wall_stop = {wall}

[mesh]
type = "cartesian"
Nx = {n}
Ny = {n}
Lx = 0.01
Ly = 0.01

[[periodic]]
zones = ["left", "right"]

[[periodic]]
zones = ["bottom", "top"]

[initialize]
type = "analytical"
u = ["10.0 * sin(628.3185307179586 * x) * cos(628.3185307179586 * y)",
     "-10.0 * cos(628.3185307179586 * x) * sin(628.3185307179586 * y)"]
p = "101325.0"
T = "{temperature}"
X = {x}

[numerics]
preset = "reacting"

[physics]
type = "navier_stokes"
gas = "mixture"
mechanism = "{ROOT / 'mechanisms' / (mechanism + '.yaml')}"
phase = "{phase}"
transport = "mixture_averaged"

[chemistry]

[output]
check_interval = 20
"""


def cpu_cases(wall: float) -> list[tuple[str, int, int, str]]:
    """(name, ranks, threads, input) of the local CPU runs."""
    cases = []
    for dim, n, rec in ((2, 384, "muscl"), (2, 256, "teno5"), (3, 48, "muscl"), (3, 32, "teno5")):
        for ranks in (1, 2, 3):
            cases.append((f"cpu-ns{dim}d-{rec}-{n}-n{ranks}", ranks, 1, tgv(dim, n, rec, wall)))
    cases.append(("cpu-ns2d-muscl-384-t2-n1", 1, 2, tgv(2, 384, "muscl", wall)))
    for tracers in (1, 4):
        cases.append((f"p{tracers}-cpu-ns2d-muscl-384-n1", 1, 1, tgv(2, 384, "muscl", wall, tracers)))
    cases.append(("p1-cpu-ns2d-muscl-384-n2", 2, 1, tgv(2, 384, "muscl", wall, 1)))
    for mechanism in MIXTURES:
        cases.append((f"cpu-reacting2d-{mechanism}-n1", 1, 2, reacting_box(mechanism, 64, wall)))
    return cases


def measure(args) -> None:
    out = Path(args.out)
    binaries = {2: str(Path(args.mallard2d).resolve()), 3: str(Path(args.mallard3d).resolve())}
    jobs = [(name, k, ranks, threads, text) for k in range(1, args.repeat + 1)
            for name, ranks, threads, text in cpu_cases(args.wall)]
    for name, k, ranks, threads, text in jobs:
        if args.only and not re.search(args.only, name):
            continue
        d = out / (name if k == 1 else f"{name}-r{k}-{hashlib.sha1(name.encode()).hexdigest()[:8]}")
        log = d / "mallard.jsonl"
        if log.exists() and '"type":"end"' in log.read_text():
            continue
        d.mkdir(parents=True, exist_ok=True)
        (d / "input.toml").write_text(text)
        dim = 3 if "Nz =" in text else 2
        cmd = [binaries[dim], "-i", "input.toml", f"--kokkos-num-threads={threads}"]
        if ranks > 1:
            cmd = ["mpirun", "-n", str(ranks), "--bind-to", "none"] + cmd
        print(f"{name}: {' '.join(cmd)}", file=sys.stderr)
        with (d / "out.log").open("w") as f:
            rc = subprocess.run(cmd, cwd=d, stdout=f, stderr=subprocess.STDOUT).returncode
        if rc not in (0, 3):
            print(f"  failed (exit status {rc}): {d / 'out.log'}", file=sys.stderr)


# ------------------------------------------------------------------ fit


def fit_throughput(rows: list[ThroughputRow]):
    """Least squares in log space; the leave-one-out error over measurements whose key is fitted from others."""
    import numpy as np

    def kind(r):
        return "cpu" if r.device.startswith("cpu") else "gpu"

    def columns(rs):
        cols = sorted({("base", r.device, r.dim) for r in rs})
        for r in rs:
            for var, level in (("reconstruction", r.reconstruction), ("physics", r.physics)):
                cols.append((var, kind(r), level))
        for k in sorted({kind(r) for r in rs}):
            if len({r.species for r in rs if kind(r) == k and r.physics == "reacting"}) > 1:
                cols.append(("species", k))
        cols = list(dict.fromkeys(cols))
        # references (0): MUSCL and NS of each kind
        return [c for c in cols if not (c[0] in ("reconstruction", "physics") and c[2] in ("muscl", "ns"))]

    def design(rs, cols):
        a = np.zeros((len(rs), len(cols)))
        index = {c: i for i, c in enumerate(cols)}
        for i, r in enumerate(rs):
            a[i, index[("base", r.device, r.dim)]] = 1
            for var, level in (("reconstruction", r.reconstruction), ("physics", r.physics)):
                c = (var, kind(r), level)
                if c in index:
                    a[i, index[c]] = 1
            if r.physics == "reacting" and ("species", kind(r)) in index:
                a[i, index[("species", kind(r))]] = math.log((r.species + 1) / (REFERENCE_SPECIES + 1))
        return a

    def solve(rs):
        cols = columns(rs)
        a = design(rs, cols)
        y = np.log([r.cells_per_second for r in rs])
        x, *_ = np.linalg.lstsq(a, y, rcond=None)
        return cols, x

    cols, x = solve(rows)
    # Leave one out: a measurement whose every column stays determined without it
    errors = []
    for i, r in enumerate(rows):
        rest = rows[:i] + rows[i + 1:]
        rc = columns(rest)
        if set(columns([r])) - set(rc):
            continue
        a = design(rest, rc)
        if np.linalg.matrix_rank(a) < len(rc):
            continue
        xr, *_ = np.linalg.lstsq(a, np.log([q.cells_per_second for q in rest]), rcond=None)
        errors.append(float(design([r], rc)[0] @ xr) - math.log(r.cells_per_second))
    model_error = math.sqrt(statistics.fmean(e * e for e in errors)) if errors else 1.0
    # Spread between the measurements of one key
    keys = defaultdict(list)
    for r in rows:
        keys[(r.device, r.dim, r.reconstruction, r.physics, r.species)].append(math.log(r.cells_per_second))
    spread = [v - statistics.fmean(vs) for vs in keys.values() if len(vs) > 1 for v in vs]
    dof = sum(len(vs) - 1 for vs in keys.values() if len(vs) > 1)
    same_key = math.sqrt(sum(s * s for s in spread) / dof) if dof else model_error
    return dict(zip(cols, x)), model_error, same_key, len(errors)


STEP_PARAMS = ("t0", "beta", "gamma", "delta", "particle", "particle_mpi")


def step_time(p: dict, s: float, r) -> float:
    t = s * (r.owned + p["beta"] * r.halo) + p["t0"] + p["gamma"] * r.neighbors + p["delta"] * r.halo_remote + \
        p["particle"] * r.particles
    if r.particles and r.ranks > 1:
        t += p["particle_mpi"]
    return t


def fit_scaling(rows: list[ScalingRow], free: tuple[str, ...]):
    """Coefficients (and each case's seconds per cell) minimizing the relative step-time error."""
    import numpy as np
    from scipy.optimize import least_squares

    cases = sorted({r.case for r in rows})
    scale = {c: min(r.seconds_per_step / r.owned for r in rows if r.case == c) for c in cases}
    units = {"t0": 1e-3, "beta": 1.0, "gamma": 1e-5, "delta": 1e-8, "particle": 1e-8, "particle_mpi": 1e-3}

    def unpack(v):
        p = {k: 0.0 for k in STEP_PARAMS}
        for i, k in enumerate(free):
            p[k] = v[i] * units[k]
        s = {c: v[len(free) + j] * scale[c] for j, c in enumerate(cases)}
        return p, s

    def residuals(v):
        p, s = unpack(v)
        return [step_time(p, s[r.case], r) / r.seconds_per_step - 1.0 for r in rows]

    v0 = np.array([0.1] * len(free) + [0.9] * len(cases))
    sol = least_squares(residuals, v0, bounds=(0, np.inf), x_scale="jac")
    p, s = unpack(sol.x)
    return p, s, float(np.sqrt(np.mean(np.square(residuals(sol.x)))))


def free_parameters(rows: list[ScalingRow]) -> tuple[str, ...]:
    free = ["t0", "beta", "gamma"]
    if any(r.halo_remote for r in rows):
        free.append("delta")
    if any(r.particles for r in rows):
        free.append("particle")
    if any(r.particles and r.ranks > 1 for r in rows):
        free.append("particle_mpi")
    return tuple(free)


def held_out(rows: list[ScalingRow], free: tuple[str, ...]) -> tuple[float, list[tuple[str, float]]]:
    """Each case's multi-rank runs predicted by the coefficients fitted to the others, from its own single-rank run
    (its seconds per cell, as check takes it from --steps on one rank)."""
    errors, by_case = [], []
    for case in sorted({r.case for r in rows}):
        mine = [r for r in rows if r.case == case]
        one = [r for r in mine if r.ranks == 1 and not r.particles]
        rest = [r for r in rows if r.case != case]
        if not one or len(mine) < 2:
            continue
        p, _, _ = fit_scaling(rest, tuple(k for k in free if any(_needs(k, r) for r in rest)))
        s = max(one[0].seconds_per_step - step_time(p, 0.0, one[0]), 0.0) / one[0].owned
        e = [step_time(p, s, r) / r.seconds_per_step - 1.0 for r in mine if r.ranks > 1 or r.particles]
        if e:
            errors += e
            by_case.append((case, math.sqrt(statistics.fmean(x * x for x in e))))
    return (math.sqrt(statistics.fmean(x * x for x in errors)) if errors else float("nan")), by_case


def _needs(k: str, r: ScalingRow) -> bool:
    return {"delta": r.halo_remote > 0, "particle": r.particles > 0,
            "particle_mpi": r.particles > 0 and r.ranks > 1}.get(k, True)


def c_string(s: str) -> str:
    return json.dumps(s)


def fit(args) -> None:
    measurements = read_csv(THROUGHPUT_CSV, ThroughputRow)
    scaling = read_csv(SCALING_CSV, ScalingRow)
    coef, model_error, same_key, n_held = fit_throughput(measurements)
    print(f"throughput: {len(measurements)} measurements; leave-one-out error {100 * model_error:.0f}% "
          f"({n_held} predicted), between cases of one key {100 * same_key:.0f}%")

    devices = sorted({r.device for r in scaling if r.use == "fit"})
    scalings, checks = [], []
    for device in devices:
        rows = [r for r in scaling if r.device == device and r.use == "fit"]
        free = free_parameters(rows)
        p, s, rms = fit_scaling(rows, free)
        error, by_case = held_out(rows, free)
        hardware = sorted({r.hardware for r in rows})
        dates = sorted({r.date for r in rows})
        names = ", ".join(f"{c.split('/', 1)[1][:-3]} {100 * e:.1f}%" for c, e in by_case)
        print(f"{device}: {len(rows)} runs, {len({r.case for r in rows})} cases, fit {100 * rms:.1f}%, held out "
              f"{100 * error:.1f}% ({names})")
        print("  " + ", ".join(f"{k} {p[k]:.4g}" for k in STEP_PARAMS))
        for r in scaling:
            if r.device == device and r.use != "fit":
                one = [q for q in scaling if q.case == r.case and q.ranks == 1]
                if one and r.ranks > 1:
                    sc = max(one[0].seconds_per_step - step_time(p, 0.0, one[0]), 0.0) / one[0].owned
                    print(f"  not fitted ({r.use}): {r.case} on {r.ranks}: predicted "
                          f"{step_time(p, sc, r) * 1e3:.3f} ms, measured {r.seconds_per_step * 1e3:.3f} ms")
        basis = f"{' and '.join(hardware)}, {dates[0]}" + (f" to {dates[-1]}" if dates[-1] != dates[0] else "")
        scalings.append((device, p, len(rows), len({r.case for r in rows}), rms, error, names, basis))
        for r in rows[:: max(1, len(rows) // 4)]:
            checks.append((device, s[r.case], r, step_time(p, s[r.case], r)))

    bases = sorted((k[1], k[2], v) for k, v in coef.items() if k[0] == "base")
    factors = sorted((k[1], k[0], k[2], v) for k, v in coef.items() if k[0] in ("reconstruction", "physics"))
    for kind in ("cpu", "gpu"):
        for var, ref in (("reconstruction", "muscl"), ("physics", "ns")):
            factors.append((kind, var, ref, 0.0))
    factors.sort()
    slopes = sorted((k[1], v) for k, v in coef.items() if k[0] == "species")

    lines = [
        "/**",
        " * @file cost_model_data.h",
        " * @brief The coefficients of mallard check's cost model (cost_model.h), fitted to the measurements of",
        " *        benchmarks/cost_model/ by tools/perf/cost_model.py fit: do not edit, regenerate.",
        " *",
        " * @copyright Copyright (c) 2026 Matthew Bonanni",
        " */",
        "",
        "#ifndef COST_MODEL_DATA_H",
        "#define COST_MODEL_DATA_H",
        "",
        "#include <cstdint>",
        "",
        "#include \"cost_model.h\"",
        "",
        "namespace cost_model {",
        "",
        "/** @brief A throughput measured on one rank alone (per thread on CPUs). */",
        "struct Measurement {",
        "    const char * device;",
        "    const char * hardware;",
        "    int dim;",
        "    const char * reconstruction;",
        "    const char * physics;",
        "    int species;",
        "    double cells_per_second;",
        "    int threads;",
        "    const char * case_name;",
        "    const char * date;",
        "    const char * source;",
        "};",
        "",
        "/** @brief log(cells/s) of MUSCL NS on a device class in a dimension. */",
        "struct Base {",
        "    const char * device;",
        "    int dim;",
        "    double log_value;",
        "};",
        "",
        "/** @brief log of the throughput of a reconstruction or physics over MUSCL or NS, on CPUs or GPUs. */",
        "struct Factor {",
        "    const char * kind;",
        "    const char * variable;",
        "    const char * level;",
        "    double log_value;",
        "};",
        "",
        "/** @brief d log(cells/s) / d log(species + 1) of reacting cases, from reference species. */",
        "struct SpeciesSlope {",
        "    const char * kind;",
        "    double slope;",
        "    int reference;",
        "};",
        "",
        "/** @brief A step time the fit gives, for the test that check's evaluation matches the fit's. */",
        "struct StepCheck {",
        "    const char * device;",
        "    double seconds_per_cell;",
        "    double owned, halo, neighbors, halo_remote, particles;",
        "    int ranks;",
        "    double step;",
        "};",
        "",
        f"/** @brief Root mean square of the log error of measurements predicted from the others (leave one out). */",
        f"inline constexpr double MODEL_ERROR = {model_error:.4f};",
        "/** @brief Root mean square spread of log(cells/s) between the measurements of one key. */",
        f"inline constexpr double SAME_KEY_ERROR = {same_key:.4f};",
        "",
        "inline constexpr Measurement MEASUREMENTS[] = {",
    ]
    for m in measurements:
        lines.append(f"    {{{c_string(m.device)}, {c_string(m.hardware)}, {m.dim}, {c_string(m.reconstruction)}, "
                     f"{c_string(m.physics)}, {m.species}, {m.cells_per_second:.6g}, {m.threads}, {c_string(m.case)}, "
                     f"{c_string(m.date)},")
        lines.append(f"     {c_string(m.source)}}},")
    lines += ["};", "", "inline constexpr Base BASES[] = {"]
    lines += [f"    {{{c_string(d)}, {dim}, {v:.6f}}}," for d, dim, v in bases]
    lines += ["};", "", "inline constexpr Factor FACTORS[] = {"]
    lines += [f"    {{{c_string(k)}, {c_string(var)}, {c_string(level)}, {v:.6f}}}," for k, var, level, v in factors]
    lines += ["};", "", "inline constexpr SpeciesSlope SPECIES_SLOPES[] = {"]
    lines += [f"    {{{c_string(k)}, {v:.6f}, {REFERENCE_SPECIES}}}," for k, v in slopes] or ["    {\"none\", 0.0, 6},"]
    lines += ["};", "", "inline constexpr Scaling SCALINGS[] = {"]
    for device, p, runs, cases, rms, error, names, basis in scalings:
        lines.append(f"    {{{c_string(device)}, {p['t0']:.6g}, {p['beta']:.6g}, {p['gamma']:.6g}, {p['delta']:.6g}, "
                     f"{p['particle']:.6g}, {p['particle_mpi']:.6g}, {runs}, {cases}, {rms:.4f}, {error:.4f},")
        lines.append(f"     {c_string(names)},")
        lines.append(f"     {c_string(basis)}}},")
    lines += ["};", "", "inline constexpr StepCheck STEP_CHECKS[] = {"]
    for device, s, r, t in checks:
        lines.append(f"    {{{c_string(device)}, {s:.10g}, {r.owned}, {r.halo}, {r.neighbors}, {r.halo_remote}, "
                     f"{r.particles}, {r.ranks}, {t:.10g}}},")
    lines += ["};", "", "} // namespace cost_model", "", "#endif // COST_MODEL_DATA_H", ""]
    HEADER.write_text("\n".join(lines))
    print(f"wrote {HEADER.relative_to(ROOT)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    m = sub.add_parser("measure", help="run the local CPU cases")
    m.add_argument("--mallard2d", required=True)
    m.add_argument("--mallard3d", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--wall", type=float, default=25.0, help="seconds of stepping per run")
    m.add_argument("--repeat", type=int, default=1, help="runs of each case (CPUs: the fastest is kept)")
    m.add_argument("--only", default="", help="regular expression on the case names")
    c = sub.add_parser("collect", help="write benchmarks/cost_model/*.csv from run directories")
    c.add_argument("--runs", nargs="+", default=[], help="directories of runs (<dir>/runs/*/ or <dir>/*/)")
    c.add_argument("--gui-table", default="", help="mallard-gui's src/mallard_gui/throughput_table.py")
    c.add_argument("--mallard2d", required=True)
    c.add_argument("--mallard3d", required=True)
    c.add_argument("--max-check-cells", type=int, default=1_100_000,
                   help="larger periodic boxes get their partition from coarsened checks")
    c.add_argument("--work", default=str(Path.home() / ".cache" / "mallard-cost-model"),
                   help="directory of the partition checks (cached by content)")
    sub.add_parser("fit", help="write src/solver/cost_model_data.h from benchmarks/cost_model/*.csv")
    args = ap.parse_args(argv)
    {"measure": measure, "collect": collect, "fit": fit}[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
