"""Build the sample run report of the user guide, docs/images/guide/run_report_sample.html.

    python tools/docs/run_report.py --mallard build/src/mallard

Runs a short 2D Euler case (a density and pressure jump over a wall, with
[integrals] and [[forces]], so every section of the report has something to
show), then rebuilds its report.html with `mallard info DIR --html` from its
run record, after replacing the host and directory the record names with
placeholders. Everything else on the page is what the run recorded.
"""
import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "images" / "guide" / "run_report_sample.html"

INPUT = """\
# The sample run report of the user guide (tools/docs/run_report.py)
[run]
n_steps = 120
cfl = 0.5

[mesh]
type = "cartesian_tri"
Nx = 32
Ny = 24
Lx = 1.0
Ly = 1.0

[initialize]
type = "analytical"
rho = "1.0 + (x < 0.4 ? 1.0 : 0.0)"
u = ["0.2", "0.1"]
p = "x < 0.4 ? 2.0 : 1.0"

[[boundaries]]
name = "left"
type = "outlet_extrapolation"

[[boundaries]]
name = "right"
type = "outlet_extrapolation"

[[boundaries]]
name = "top"
type = "symmetry"

[[boundaries]]
name = "bottom"
type = "wall_adiabatic"

[numerics]
riemann_solver = "HLLC"
time_integrator = "SSPRK3"

[numerics.face_reconstruction]
type = "MUSCL"

[physics]
type = "euler"
gamma = 1.4
p_ref = 1.0
T_ref = 1.0
rho_ref = 1.0

[output]
check_interval = 5

[integrals]
interval = 5

[[forces]]
zone = "bottom"
interval = 5

[[write_data]]
prefix = "solut/restart"
format = "restart"
interval = 60

[[write_data]]
prefix = "solut/flow"
format = "vtu"
interval = 60
variables = ["RHO", "P"]
"""


def anonymized(record):
    """The record with the machine it ran on replaced by placeholders."""
    run = record.get("provenance", {}).get("run")
    if run:
        run["host"] = "workstation"
        run["directory"] = "/home/user/run_report_sample"
    for rank in record.get("placement") or []:
        rank["host"] = "workstation"
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mallard", required=True, help="2D Mallard executable")
    args = ap.parse_args()
    mallard = str(Path(args.mallard).resolve())
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "input.toml").write_text(INPUT)
        subprocess.run([mallard, "-i", "input.toml", "--kokkos-num-threads=2"], cwd=work, check=True,
                       stdout=subprocess.DEVNULL)
        log = work / "mallard.jsonl"
        records = [anonymized(json.loads(line)) for line in log.read_text().splitlines()]
        log.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in records))
        subprocess.run([mallard, "info", str(work), "--html"], check=True, stdout=subprocess.DEVNULL)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(work / "report.html", OUT)
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1000:.0f} kB)")


if __name__ == "__main__":
    main()
