"""The solver bindings (mallard.Case, mallard.Solver: tools/mallard/_solver.py over src/api/python_module.cpp).

Needs a build with -DMallard_ENABLE_PYTHON_BINDINGS=ON: the mallard executable on PATH (or MALLARD_EXECUTABLE)
and the extension module next to its library; skipped without them. One session for the whole module: Kokkos
starts once per process.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import mallard
from mallard import _solver

DIMS = [d for d in (2, 3) if _solver._has_extension(d)]
pytestmark = pytest.mark.skipif(not DIMS, reason="no solver bindings: a build with -DMallard_ENABLE_PYTHON_BINDINGS=ON")
DIM = DIMS[0] if DIMS else 2


def case_text(steps, extra=""):
    """A small shock tube writing integrals, a VTU snapshot at the end and a restart file, in the working directory."""
    if DIM == 2:
        mesh = '[mesh]\ntype = "cartesian_tri"\nNx = 12\nNy = 10\nLx = 1.0\nLy = 1.0\n'
        zones, u = ["left", "right", "top", "bottom"], '["0.2", "0.1"]'
    else:
        mesh = '[mesh]\ntype = "cartesian"\nNx = 6\nNy = 5\nNz = 4\nLx = 1.0\nLy = 1.0\nLz = 1.0\n'
        zones, u = ["left", "right", "bottom", "top", "back", "front"], '["0.2", "0.1", "0.0"]'
    text = f"[run]\nn_steps = {steps}\ncfl = 0.4\n" + mesh
    text += ('[initialize]\ntype = "analytical"\nrho = "1.0 + (x < 0.4 ? 1.0 : 0.0)"\n'
             f'u = {u}\np = "x < 0.4 ? 2.0 : 1.0"\n')
    text += "".join(f'[[boundaries]]\nname = "{z}"\ntype = "outlet_extrapolation"\n' for z in zones)
    text += ('[numerics]\nriemann_solver = "HLLC"\ntime_integrator = "SSPRK3"\n'
             '[numerics.face_reconstruction]\ntype = "MUSCL"\n'
             '[physics]\ntype = "euler"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n'
             '[output]\ncheck_interval = 1000000\n[integrals]\ninterval = 5\n'
             f'[[write_data]]\nprefix = "solut/flow"\nformat = "vtu"\nfields = ["RHO", "U", "P"]\ninterval = {steps}\n'
             f'[[write_data]]\nprefix = "solut/restart"\nformat = "restart"\ninterval = {steps}\n')
    return text + extra


@pytest.fixture(scope="module", autouse=True)
def session():
    if not DIMS:
        yield None
        return
    s = mallard.Session(["--kokkos-num-threads=2"], dim=DIM)
    yield s


def run_executable(d):
    subprocess.run([mallard._lib.executable(), "-i", "input.toml", "--kokkos-num-threads=2"], cwd=d, check=True,
                   capture_output=True)


def events(d):
    with open(d / "mallard.jsonl") as f:
        records = [json.loads(line) for line in f]
    return [r for r in records if r["type"] == "event"], records


def test_a_run_driven_from_python_matches_the_executable_bit_for_bit(tmp_path, monkeypatch):
    exe, py = tmp_path / "executable", tmp_path / "python"
    for d in (exe, py):
        d.mkdir()
        (d / "input.toml").write_text(case_text(12))
    run_executable(exe)
    monkeypatch.chdir(py)
    solver = mallard.Solver(mallard.Case.from_toml_file("input.toml"))
    seen = []
    # Callbacks that only read leave the run as it is
    solver.on_step(lambda s: seen.append((s.step_count(), float(s.field("P").sum()))))
    while solver.step():
        pass
    assert [s for s, _ in seen] == list(range(1, 13))
    assert solver.finish() == 0
    files = sorted(os.listdir(exe / "solut"))
    assert files and files == sorted(os.listdir(py / "solut"))
    for name in files:
        report = mallard.diff(exe / "solut" / name, py / "solut" / name)
        # Tolerance 0: every value bitwise the same
        assert report["result"] == "ok" and report["time"]["difference"] == 0, (name, report)
        assert report["fields"] and all(f["differing"] == 0 for f in report["fields"]), (name, report)


def test_fields_are_the_solvers_memory_without_copies(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    solver = mallard.Solver(mallard.Case.from_toml(case_text(4)))
    solver.setup()
    solver.step()
    rho, block = solver.field("RHO"), solver.field("conservatives")
    if isinstance(rho, mallard.DeviceArray):
        cupy = pytest.importorskip("cupy")
        a, b = cupy.asarray(rho), cupy.asarray(solver.field("RHO"))
        assert a.data.ptr == b.data.ptr == rho.__cuda_array_interface__["data"][0]
        assert cupy.from_dlpack(rho).data.ptr == a.data.ptr
        w = cupy.asarray(solver.writable_field("RHO"))
        assert w.data.ptr == a.data.ptr
        return
    assert block.shape[1] == DIM + 2 and rho.shape == (block.shape[0],)
    assert np.shares_memory(rho, block) and not rho.flags.writeable
    assert rho.__array_interface__["data"][0] == solver.field("RHO").__array_interface__["data"][0]
    # DLPack hands the same memory on
    assert np.shares_memory(np.from_dlpack(rho), rho)
    # A writable view is the state itself: writes are seen by every view at once
    w = solver.writable_field("RHO")
    assert w.flags.writeable and np.shares_memory(w, rho)
    before = rho.copy()
    w *= 1.5
    np.testing.assert_array_equal(solver.field("RHO"), 1.5 * before)
    solver.commit_state()
    solver.step()
    assert solver.field("GLOBAL_ID").dtype == np.uint64
    assert solver.field("CENTROID").shape == (block.shape[0], DIM)
    assert "MEAN_RHO" not in solver.field_names()
    solver.finish()
    kinds = [e["kind"] for e in events(tmp_path)[0]]
    assert kinds.count("state_set") == 1


def test_stage_callbacks_see_each_stage_and_errors_end_the_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    solver = mallard.Solver(mallard.Case.from_toml(case_text(10)))
    stages = []
    callback = solver.on_stage(lambda s, stage: stages.append((stage.step, stage.index, stage.time)))
    solver.step()
    solver.remove_callback(callback)
    solver.step()
    # SSPRK3: at t_n, t_n + dt and t_n + dt / 2
    assert [(s, i) for s, i, _ in stages] == [(0, 0), (0, 1), (0, 2)]
    assert stages[2][2] == 0.5 * stages[1][2]

    def fails(s):
        raise ValueError("from the callback")

    solver.on_step(fails)
    with pytest.raises(ValueError, match="from the callback"):
        solver.step()
    end = [r for r in events(tmp_path)[1] if r["type"] == "end"][-1]
    assert end["outcome"] == "error"


def same_files(a, b):
    files = sorted(os.listdir(a))
    assert files and files == sorted(os.listdir(b))
    for name in files:
        report = mallard.diff(a / name, b / name)
        assert report["result"] == "ok" and report["time"]["difference"] == 0, (name, report)
        assert report["fields"] and all(f["differing"] == 0 for f in report["fields"]), (name, report)


def test_a_case_set_from_python_emits_the_toml_mallard_runs_bit_for_bit(tmp_path, monkeypatch):
    base = mallard.Case.from_toml(case_text(4))
    case = (base.set("run.n_steps", 8).set("run.cfl", 0.3).set("numerics.riemann_solver", "HLL")
            .set("statistics", {"fields": ["RHO", "P"], "interval": 1}))
    text = case.toml()
    # Only the lines of the keys set change; the new table goes at the end
    changed = base.toml().replace("n_steps = 4", "n_steps = 8").replace("cfl = 0.4", "cfl = 0.3")
    assert text.startswith(changed.replace('"HLLC"', '"HLL"') + "\n[statistics]\n"), text
    with pytest.raises(mallard.Error) as e:
        base.set("probes[1].interval", 1)
    assert e.value.code == "E-INPUT"

    exe, py = tmp_path / "executable", tmp_path / "python"
    exe.mkdir()
    py.mkdir()
    (exe / "input.toml").write_text(text)
    run_executable(exe)
    monkeypatch.chdir(py)
    saved = case.save("input.toml")
    assert saved.path() == "input.toml" and saved.toml() == text
    solver = mallard.Solver(saved)
    assert solver.step(3) and solver.step_count() == 3
    assert not solver.step(100) and solver.step_count() == 8
    assert solver.finish() == 0
    same_files(exe / "solut", py / "solut")
    start = [r for r in events(py)[1] if r["type"] == "start"][-1]
    assert start["provenance"]["input"]["text"] == text


def test_a_typed_config_emits_the_toml_mallard_runs_bit_for_bit(tmp_path, monkeypatch):
    from mallard import config
    base = mallard.Case.from_toml(case_text(4))
    case = (base.set(config.Run(n_steps=8, cfl=0.3, t_stop=(60, "ms")))
            .set(config.Numerics(riemann_solver="HLL", face_reconstruction={"type": "MUSCL", "limiter": "barth_jespersen"}))
            .set(config.Probes(0, name="left", fields=["RHO"], region=config.Region(type="box" if DIM == 3 else "rectangle",
                                                                                     center=[0.3, 0.5, 0.5][:DIM],
                                                                                     size=[0.2, 0.4, 0.6][:DIM]),
                               reductions=["max"], interval=2)))
    # The same keys set one by one: a typed config is only a typed way to write them
    by_key = (base.set("run.n_steps", 8).set("run.cfl", 0.3).set("run.t_stop", "60 ms")
              .set("numerics.riemann_solver", "HLL").set("numerics.face_reconstruction.type", "MUSCL")
              .set("numerics.face_reconstruction.limiter", "barth_jespersen"))
    text = case.toml()
    # In the order of the class's keywords
    assert text.startswith(by_key.toml() + '\n[[probes]]\nfields = ["RHO"]\ninterval = 2\nname = "left"\n'), text
    with pytest.raises(ValueError):
        base.set(config.Region(type="circle"))
    with pytest.raises(TypeError):
        config.Run(n_step=8)

    exe, py = tmp_path / "executable", tmp_path / "python"
    exe.mkdir()
    py.mkdir()
    (exe / "input.toml").write_text(text)
    run_executable(exe)
    monkeypatch.chdir(py)
    solver = mallard.Solver(case.save("input.toml"))
    assert solver.run() == 0
    assert solver.monitor("left_max_RHO") is not None and solver.time() <= 0.06
    same_files(exe / "solut", py / "solut")


def test_check_reports_mallard_checks_results_as_structs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report = mallard.Case.from_toml(case_text(4)).check()
    assert report.ok and report.setup_done and report.status != "error", report.text
    assert report.time_step.set_by == "CFL" and report.time_step.dt > 0
    assert report.cost.n_steps == 4
    assert json.loads(report.json)["status"] == report.status
    assert not os.path.exists("solut")
    bad = mallard.Case.from_toml(case_text(4)).set("run.t_stpo", 0.5).check(log=lambda m: None)
    assert not bad.ok and bad.status == "error"
    assert [(k.key, k.suggestion) for k in bad.unused_keys] == [("run.t_stpo", "run.t_stop")]
    assert any(d.code == "E-KEY-UNUSED" and d.level == "error" for d in bad.diagnostics)


def test_a_region_anywhere_is_one_typed_class_names_and_regions_mixed(tmp_path, monkeypatch):
    from mallard import config
    monkeypatch.chdir(tmp_path)
    hole = config.Region(type="box" if DIM == 3 else "rectangle", center=[0.4, 0.5, 0.5][:DIM], size=[0.1, 0.2, 0.2][:DIM],
                         repeat=config.RegionRepeat(count=2, translate=[0.2, 0.0, 0.0][:DIM]))
    case = (mallard.Case.from_toml(case_text(1))
            .set(config.Regions("ball", type="sphere", center=[0.5] * DIM, radius=0.3))
            .set(config.Probes(0, name="parts", fields=["P"], reductions=["mean"],
                               region=config.Region(type="union", regions=["ball", hole]))))
    report = case.check()
    assert report.ok, report.text
    keys = {k.path for k in report.keys}
    assert {"regions.ball.radius", "probes[0].region.regions[1].size", "probes[0].region.regions[1].repeat.count"} <= keys


def test_check_gives_the_resolved_input_with_each_keys_catalog_record(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report = mallard.Case.from_toml(case_text(4)).set("physics.p_ref", "1 bar").check()
    keys = {k.path: k for k in report.keys}
    p_ref = keys["physics.p_ref"]
    assert (p_ref.set, p_ref.type, p_ref.unit, float(p_ref.value)) == (True, "number", "Pa", 1e5)
    assert "HLLC" in keys["numerics.riemann_solver"].values
    assert keys["boundaries[1].type"].catalog_path == "boundaries.type"
    defaults = [k for k in report.keys if not k.set and k.default]
    assert defaults and all(k.value == k.default for k in defaults)


def test_progress_output_and_record_callbacks_and_the_log_sink(tmp_path, monkeypatch, capfd):
    monkeypatch.chdir(tmp_path)
    solver = mallard.Solver(mallard.Case.from_toml(case_text(8)).set("output.check_interval", 4))
    rows, outputs, records, messages = [], [], [], []
    solver.on_progress(rows.append)
    solver.on_output(outputs.append)
    solver.on_record(records.append)
    solver.set_log(messages.append)
    capfd.readouterr()
    assert solver.run() == 0
    assert capfd.readouterr().out == ""
    assert [r.step for r in rows] == [0, 4, 8] and rows[-1].fraction == 1.0
    assert rows[0].diagnostics[0][0] == "min_rho"
    with open(tmp_path / "mallard.jsonl") as f:
        lines = f.read().splitlines()
    assert [r.json for r in records] == lines
    assert [r.json for r in rows] == [line for line in lines if json.loads(line)["type"] == "progress"]
    assert {(o.kind, o.step) for o in outputs} >= {("vtu", 8), ("restart", 8)}
    assert all(o.files and o.files[0].bytes > 0 for o in outputs)
    assert messages and all(m.level == "info" and "\033" not in m.text for m in messages)
    assert any(m.text.startswith("== Run ") for m in messages)


def test_inputs_are_checked_as_the_executable_checks_them(tmp_path):
    with pytest.raises(mallard.Error) as e:
        mallard.Case.from_toml("[run\n")
    assert e.value.code == "E-INPUT"
    case = mallard.Case.from_toml(case_text(2).replace("cfl =", "cfll = 0.5\ncfl ="))
    assert case.dimension() == DIM
    solver = mallard.Solver(case)
    with pytest.raises(mallard.Error) as e:
        solver.setup()
    assert e.value.code == "E-KEY-UNUSED" and "cfll" in str(e.value)


def test_the_package_reads_files_without_loading_the_bindings():
    code = "import sys, mallard; mallard.Field; assert not any(m.startswith('mallard._core') for m in sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True)
