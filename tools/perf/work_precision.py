#!/usr/bin/env python3
"""Work-precision study: the error each reconstruction reaches against the cost of reaching it.

    work_precision.py run --mallard-2d B2 --mallard-3d B3 --out DIR [--threads 2] [--jobs 4 --cpus 0-15]
                          [--cases vortex_2d_quad,...] [--schemes MUSCL,TENO5] [--max-wall 900] [--max-cells N]
    work_precision.py reference --mallard-2d B2 --out DIR [--threads 24]
    work_precision.py collect DIR... [--data tools/docs/work_precision.json]

Cases (all Euler, SSPRK3 at cfl = 1.0, each scheme on a sequence of meshes):

- vortex_2d_quad, vortex_2d_tri: the isentropic vortex (Shu 1998, strength 5, with a smooth cutoff at radius 5:
  vortex_fields) convected to t = 2 across a periodic 10 x 10 box of squares, or of irregular triangles
  (high_order_cost.triangles);
- vortex_3d_hex, vortex_3d_tet: the same vortex as a tube along z in a periodic box of cubes, 10 x 10 across and max(8, n / 4) cubes along z, or of
  six tetrahedra per cube (cartesian_tet);
- shu_osher: the Shu-Osher problem (a Mach 3 shock into an entropy wave) to t = 1.8 on Nx x 4 squares.

Smooth cases use HLLC and unlimited MUSCL; their error is the L2 norm of the density against the exact cell
averages, from `mallard post convergence --exact`. The Shu-Osher problem uses RHLL and the default MUSCL limiter;
its error is the L1 norm of the density column averages against those of a TENO-E 5 run on 51,200 x 4 cells
(`reference`), averaged onto each run's columns.

`run` runs every case and scheme on its meshes from coarse to fine, and stops refining a sequence before a run whose
wall time, extrapolated from the previous run by its cells and time steps, would exceed --max-wall. Runs shorter than --repeat-below seconds are repeated (three in all) and the fastest
kept. Each run is its own directory; what the study records about it (results.jsonl in DIR, one record per run)
comes from the run's own record (mallard.jsonl) and the error computed from the snapshot that record names. With
--jobs, sequences run concurrently, each run pinned (taskset) to its own --threads CPUs of --cpus.
`collect` merges the records of one or more DIRs into the data file that tools/docs/work_precision.py draws.
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tools" / "docs")]
from mallard.vtu import read_vtu_cells  # noqa: E402

DATA = ROOT / "tools" / "docs" / "work_precision.json"
GAMMA, BETA, LENGTH = 1.4, 5.0, 10.0
VORTEX_T, SHU_OSHER_T = 2.0, 1.8
SHU_OSHER_NY, REFERENCE_NX = 4, 51200

CASES = {
    "vortex_2d_quad": dict(dim=2, sizes=(16, 32, 64, 128, 256, 512, 1024, 2048)),
    "vortex_2d_tri": dict(dim=2, sizes=(16, 32, 64, 128, 256, 512, 1024)),
    "vortex_3d_hex": dict(dim=3, sizes=(16, 24, 32, 48, 64, 96, 128, 192)),
    "vortex_3d_tet": dict(dim=3, sizes=(16, 24, 32, 48, 64, 96, 128)),
    "shu_osher": dict(dim=2, sizes=(100, 200, 400, 800, 1600, 3200, 6400)),
}
SCHEMES = ("MUSCL", "TENO3", "TENO4", "TENO5", "TENO6")


def layers(n):
    """Cells along z of the 3D vortex: a quarter of those across, at least 8 (the flow does not vary along z)."""
    return max(8, n // 4)


def cells(case, n):
    return {"vortex_2d_quad": n * n, "vortex_2d_tri": 2 * n * n, "vortex_3d_hex": n * n * layers(n),
            "vortex_3d_tet": 6 * n * n * layers(n), "shu_osher": SHU_OSHER_NY * n}[case]


def growth(case, n0, n1):
    """Expected ratio of the costs of two meshes: cells times time steps (which follow the cells across)."""
    return cells(case, n1) / cells(case, n0) * n1 / n0


def reconstruction(scheme, shock):
    if scheme == "MUSCL":
        return 'type = "MUSCL"' + ("" if shock else '\nlimiter = "none"')
    return f'type = "TENO"\norder = {int(scheme[4:])}'


def vortex_fields(t=None):
    """Density, velocity and pressure of the vortex as expressions; at time t (convected at u = 1, wrapped into the
    periodic box) when given, else at t = 0. The vortex is Shu's with r^2 in its exponent replaced by
    r^2 / (1 - r^2 / R^2), R = LENGTH / 2: still an exact steady isentropic vortex, and zero with all its
    derivatives from r = R on, so the field is smooth across the periodic box."""
    c = LENGTH / 2
    dx = f"x - {c}" if t is None else f"x - {c} - {t} - {LENGTH} * round((x - {c} - {t}) / {LENGTH})"
    k = (GAMMA - 1.0) * BETA**2 / (8.0 * GAMMA * np.pi**2)
    a = BETA / (2 * np.pi)
    v = (f"var dx := {dx}; var dy := y - {c}; var r2 := dx * dx + dy * dy; var s := 1.0 - r2 / {c * c}; "
         f"var f := s > 0.0 ? exp(0.5 * (1.0 - r2 / s)) : 0.0; var g := s > 0.0 ? f / s : 0.0; ")
    T = f"(1.0 - {k:.17g} * f * f)"
    return (v + f"pow({T}, {1.0 / (GAMMA - 1.0):.17g})", v + f"1.0 - {a:.17g} * g * dy", v + f"{a:.17g} * g * dx",
            v + f"pow({T}, {GAMMA / (GAMMA - 1.0):.17g})")


PHYSICS = f"""[physics]
type = "euler"
units = "nondimensional"
gamma = {GAMMA}
p_ref = 1.0
T_ref = 1.0
rho_ref = 1.0
"""


def vortex_input(case, n, scheme):
    dim = CASES[case]["dim"]
    if case == "vortex_2d_tri":
        mesh = 'type = "file"\nfilename = "mesh.msh"'
        periodic = [("left", "right", f"[{LENGTH}, 0.0]"), ("bottom", "top", f"[0.0, {LENGTH}]")]
    else:
        kind = "cartesian_tet" if case == "vortex_3d_tet" else "cartesian"
        mesh = f'type = "{kind}"\nNx = {n}\nNy = {n}\nLx = {LENGTH}\nLy = {LENGTH}'
        if dim == 3:
            mesh += f"\nNz = {layers(n)}\nLz = {LENGTH * layers(n) / n:.17g}"
        periodic = [("left", "right", None), ("bottom", "top", None)] + ([("back", "front", None)] if dim == 3 else [])
    blocks = "".join(f'[[periodic]]\nzones = ["{a}", "{b}"]\n' + (f"translation = {tr}\n" if tr else "")
                     for a, b, tr in periodic)
    rho, u, v, p = vortex_fields()
    vel = f'["{u}", "{v}"' + (', "0.0"]' if dim == 3 else "]")
    return f"""[run]
t_stop = {VORTEX_T}
cfl = 1.0
[mesh]
{mesh}
{blocks}[initialize]
type = "analytical"
rho = "{rho}"
u = {vel}
p = "{p}"
[numerics]
riemann_solver = "HLLC"
time_integrator = "SSPRK3"
[numerics.face_reconstruction]
{reconstruction(scheme, False)}
{PHYSICS}[output]
check_interval = 50
[[write_data]]
prefix = "final"
format = "restart"
time_interval = {VORTEX_T}
"""


def shu_osher_input(nx, scheme):
    ly = SHU_OSHER_NY * LENGTH / nx
    return f"""[run]
t_stop = {SHU_OSHER_T}
cfl = 1.0
[mesh]
type = "cartesian"
Nx = {nx}
Ny = {SHU_OSHER_NY}
Lx = {LENGTH}
Ly = {ly:.17g}
[initialize]
type = "analytical"
rho = "x < 1 ? 3.857143 : 1 + 0.2 * sin(5 * (x - 5))"
u = ["x < 1 ? 2.629369 : 0", "0"]
p = "x < 1 ? 10.33333 : 1"
[[boundaries]]
name = "left"
type = "dirichlet"
rho = "3.857143"
u = ["2.629369", "0"]
p = "10.33333"
[[boundaries]]
name = "right"
type = "outlet_extrapolation"
[[boundaries]]
name = "top"
type = "symmetry"
[[boundaries]]
name = "bottom"
type = "symmetry"
[numerics]
riemann_solver = "RHLL"
time_integrator = "SSPRK3"
[numerics.face_reconstruction]
{reconstruction(scheme, True)}
{PHYSICS}[output]
check_interval = 200
[[write_data]]
prefix = "final"
format = "vtu"
time_interval = {SHU_OSHER_T}
fields = ["RHO"]
"""


def hardware():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=30)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0].strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    try:
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=30)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return platform.processor() or platform.machine()


def run_record(directory):
    records = [json.loads(line) for line in (Path(directory) / "mallard.jsonl").read_text().splitlines() if line]
    start = next(r for r in records if r["type"] == "start")
    end = next(r for r in reversed(records) if r["type"] == "end")
    progress = [r for r in records if r["type"] == "progress"]
    outputs = [r for r in records if r["type"] == "output" and r.get("run_id") == start["run_id"]]
    return start, end, progress, outputs


def final_snapshot(directory, outputs, kind, t):
    """The path of the run's snapshot of this kind at time t, from its output records."""
    match = [r for r in outputs if r["kind"] == kind and abs(r["t"] - t) <= 1e-9 * max(1.0, t)]
    if not match:
        raise RuntimeError(f"{directory}: no {kind} snapshot at t = {t} in its run record")
    return Path(directory) / match[-1]["path"]


def vortex_errors(binary, directories):
    """L1, L2 and Linf errors of the density of each run against the exact cell averages, by run id (`mallard post
    convergence` compares two or more runs)."""
    out = subprocess.run([str(binary), "post", "convergence", *map(str, directories), "--field", "RHO", "--exact",
                          vortex_fields(t="t")[0], "--format", "json"], capture_output=True, text=True)
    if not out.stdout.strip():
        raise RuntimeError(f"mallard post convergence failed:\n{out.stderr}")
    study = json.loads(out.stdout)
    norms = {n["norm"]: n["errors"] for n in study["field"]["norms"]}
    return {r["run_id"]: {k: v[i] for k, v in norms.items()} for i, r in enumerate(study["runs"])}


def column_means(directory, outputs, nx):
    """Density averaged over each column of cells of the Shu-Osher strip, by cell centroid."""
    pts, conn, offs, _, data = read_vtu_cells(str(final_snapshot(directory, outputs, "vtu", SHU_OSHER_T)))
    if np.any(np.diff(np.concatenate([[0], offs])) != 4):
        raise RuntimeError("expected quadrilaterals only")
    x = pts[conn.reshape(-1, 4), 0].mean(axis=1)
    column = np.clip(np.floor(x / (LENGTH / nx)).astype(int), 0, nx - 1)
    counts = np.bincount(column, minlength=nx)
    if np.any(counts != SHU_OSHER_NY):
        raise RuntimeError("columns do not have the expected cell count")
    return np.bincount(column, weights=data["RHO"], minlength=nx) / counts


def shu_osher_error(means, reference):
    ratio = len(reference) // len(means)
    if ratio * len(means) != len(reference):
        raise RuntimeError(f"reference columns {len(reference)} not a multiple of {len(means)}")
    e = np.asarray(means) - reference.reshape(len(means), ratio).mean(axis=1)
    return dict(L1=float(np.mean(np.abs(e))), L2=float(np.sqrt(np.mean(e * e))), Linf=float(np.max(np.abs(e))))


def launch(binary, directory, threads, cpus):
    cmd = [str(Path(binary).resolve()), "-i", "input.toml"] + ([f"--kokkos-num-threads={threads}"] if threads else [])
    if cpus:
        cmd = ["taskset", "-c", ",".join(map(str, cpus))] + cmd
    start = time.perf_counter()
    out = subprocess.run(cmd, cwd=directory, capture_output=True, text=True)
    wall = time.perf_counter() - start
    (Path(directory) / "stdout.txt").write_text(out.stdout + out.stderr)
    return out.returncode, wall


def one_run(args, case, scheme, n, directory, cpus):
    """Runs one case, repeated when short; returns the study's record of it. A try whose directory already holds a
    finished run of the same input (an interrupted study) is read back instead of run again."""
    binary = args.mallard_2d if CASES[case]["dim"] == 2 else args.mallard_3d
    text = shu_osher_input(n, scheme) if case == "shu_osher" else vortex_input(case, n, scheme)
    tries = []
    while not tries or (tries[-1]["wall"] < args.repeat_below and len(tries) < 3):
        d = Path(directory) / f"try{len(tries)}"
        wall = None
        try:
            start, end, progress, outputs = run_record(d)
            reuse = (d / "input.toml").read_text() == text and end["outcome"] == "finished" and \
                (case != "shu_osher" or final_snapshot(d, outputs, "vtu", SHU_OSHER_T).exists())
        except (OSError, StopIteration, RuntimeError, ValueError):
            reuse = False
        if not reuse:
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True)
            if case == "vortex_2d_tri":
                import high_order_cost
                high_order_cost.triangles(d / "mesh.msh", n)
            (d / "input.toml").write_text(text)
            code, wall = launch(binary, d, args.threads, cpus)
            start, end, progress, outputs = run_record(d)
            if code != 0 or end["outcome"] != "finished":
                return dict(case=case, scheme=scheme, n=n, failed=end.get("reason") or f"exit {code}", directory=str(d))
        prov = start["provenance"]
        rec = dict(case=case, scheme=scheme, n=n, cells=progress[-1]["cells"], directory=str(d),
                   run_id=start["run_id"], steps=end["steps"], wall_process=wall,
                   **{k: end[k] for k in ("wall", "wall_setup", "wall_stepping", "wall_output", "wall_diagnostics")},
                   memory_host_peak=max((p["memory_host_peak"] or 0) for p in progress),
                   memory_device=max((p["memory_device"] or 0) for p in progress) or None,
                   mallard=prov["mallard"], backend=prov["build"]["backend"], device=prov["run"]["device"],
                   threads=start["placement"][0]["threads"], cpus=cpus)
        if case == "shu_osher":
            rec["column_means"] = column_means(d, outputs, n).tolist()
        tries.append(rec)
    best = dict(min(tries, key=lambda r: r["wall"]), snapshot_directory=tries[0]["directory"],
                snapshot_run_id=tries[0]["run_id"])
    best["tries"] = [r["wall"] for r in tries]
    return best


def cpu_list(spec):
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.jsonl"
    done = {}
    if results.exists():
        for line in results.read_text().splitlines():
            r = json.loads(line)
            if "case" in r:
                done[(r["case"], r["scheme"], r["n"])] = r
    meta = dict(type="meta", hardware=hardware(), host=platform.node(), threads=args.threads, time=time.time())
    lock = threading.Lock()
    with results.open("a") as f:
        f.write(json.dumps(meta) + "\n")
    slots = None
    if args.cpus:
        cpus = cpu_list(args.cpus)
        slots = [cpus[i * args.threads:(i + 1) * args.threads] for i in range(len(cpus) // args.threads)][:args.jobs]
    pool_slots = list(range(args.jobs))

    def sequence(case, scheme):
        with lock:
            slot = pool_slots.pop()
        pending = []
        binary = args.mallard_2d if CASES[case]["dim"] == 2 else args.mallard_3d

        def flush():
            if args.defer_errors:
                return
            ok = [r for r in pending if "failed" not in r]
            if case != "shu_osher" and len(ok) >= 2:
                try:
                    errors = vortex_errors(binary, [r["snapshot_directory"] for r in ok])
                    for r in ok:
                        r["error"] = errors[r["snapshot_run_id"]]
                except RuntimeError as e:
                    print(f"{case} {scheme}: {e}", flush=True)
            elif case != "shu_osher" and ok:
                print(f"{case} {scheme}: one run only, no error", flush=True)
            with lock, results.open("a") as f:
                for r in pending:
                    for x in Path(r["directory"]).parent.glob("try*/*"):
                        if x.suffix in (".restart", ".msh", ".vtu"):
                            x.unlink()
                    f.write(json.dumps(r) + "\n")

        try:
            last = None
            for n in CASES[case]["sizes"]:
                if cells(case, n) > args.max_cells:
                    break
                if last and "failed" not in last and last["wall"] * growth(case, last["n"], n) > args.max_wall:
                    break
                if (case, scheme, n) in done:
                    last = done[(case, scheme, n)]
                    continue
                rec = one_run(args, case, scheme, n, out / case / scheme / f"n{n}", slots[slot] if slots else None)
                pending.append(rec)
                last = rec
                print(f"{case:15s} {scheme:6s} {n:5d} {rec.get('cells', 0):9d} cells  "
                      + (f"FAILED: {rec['failed']}" if "failed" in rec else
                         f"wall {rec['wall']:8.2f} s  setup {rec['wall_setup']:7.2f} s  steps {rec['steps']}"),
                      flush=True)
                if "failed" in rec:
                    break
        finally:
            flush()
            with lock:
                pool_slots.append(slot)

    work = [(c, s) for c in args.cases.split(",") for s in args.schemes.split(",")]
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for fut in [ex.submit(sequence, c, s) for c, s in work]:
            fut.result()


def reference(args):
    d = Path(args.out) / "shu_osher_reference"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    (d / "input.toml").write_text(shu_osher_input(REFERENCE_NX, "TENO5"))
    code, wall = launch(args.mallard_2d, d, args.threads, None)
    start, end, _, outputs = run_record(d)
    if code != 0 or end["outcome"] != "finished":
        raise RuntimeError(f"reference run failed: {end.get('reason')}")
    means = column_means(d, outputs, REFERENCE_NX)
    rec = dict(run_id=start["run_id"], scheme="TENO5", nx=REFERENCE_NX, ny=SHU_OSHER_NY, wall=wall,
               mallard=start["provenance"]["mallard"], device=start["provenance"]["run"]["device"],
               column_means=means.tolist())
    (Path(args.out) / "shu_osher_reference.json").write_text(json.dumps(rec) + "\n")
    print(f"reference: {REFERENCE_NX} columns in {wall:.1f} s")


def collect(args):
    """Merges the records of the study directories into the data file, one platform per directory (the hardware of its first run)."""
    platforms = []
    reference = None
    for directory in args.dirs:
        ref = Path(directory) / "shu_osher_reference.json"
        if ref.exists():
            reference = json.loads(ref.read_text())
    for directory in args.dirs:
        lines = [json.loads(line) for line in (Path(directory) / "results.jsonl").read_text().splitlines() if line]
        metas = [r for r in lines if r.get("type") == "meta"]
        runs = {}
        for r in lines:
            if r.get("type") == "meta":
                continue
            r = dict(r)
            for k in ("directory", "snapshot_directory", "snapshot_run_id"):
                r.pop(k, None)
            if "column_means" in r:
                means = r.pop("column_means")
                if reference is None:
                    continue
                r["error"] = shu_osher_error(means, np.asarray(reference["column_means"]))
            runs[(r["case"], r["scheme"], r["n"])] = r
        backends = {r.get("backend") for r in runs.values() if "backend" in r}
        platforms.append(dict(hardware=metas[0]["hardware"], threads=metas[0]["threads"],
                              backend=sorted(b for b in backends if b),
                              runs=sorted(runs.values(), key=lambda r: (r["case"], r["scheme"], r["n"]))))
    data = dict(reference=None if reference is None else {k: v for k, v in reference.items() if k != "column_means"},
                platforms=platforms)
    Path(args.data).write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {args.data}: " + ", ".join(f"{p['hardware']} ({len(p['runs'])} runs)" for p in platforms))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--mallard-2d", required=False)
    r.add_argument("--mallard-3d", required=False)
    r.add_argument("--out", required=True)
    r.add_argument("--threads", type=int, default=0, help="--kokkos-num-threads (0: the backend's default)")
    r.add_argument("--jobs", type=int, default=1)
    r.add_argument("--cpus", help="CPUs to pin the runs to, e.g. 0-15 (each run gets --threads of them)")
    r.add_argument("--cases", default=",".join(CASES))
    r.add_argument("--schemes", default=",".join(SCHEMES))
    r.add_argument("--max-wall", type=float, default=900.0)
    r.add_argument("--max-cells", type=int, default=10**8)
    r.add_argument("--repeat-below", type=float, default=10.0)
    r.add_argument("--defer-errors", action="store_true",
                   help="keep the snapshots and write no records, so that the same command run again elsewhere (on a "
                        "copy of --out, with any build) reads the runs back and computes their errors there")
    f = sub.add_parser("reference")
    f.add_argument("--mallard-2d", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--threads", type=int, default=0)
    c = sub.add_parser("collect")
    c.add_argument("dirs", nargs="+")
    c.add_argument("--data", default=str(DATA))
    args = ap.parse_args()
    {"run": run, "reference": reference, "collect": collect}[args.command](args)


if __name__ == "__main__":
    main()
