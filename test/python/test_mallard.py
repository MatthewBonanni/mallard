"""The Python package (tools/mallard) on the output of a run of test/data/formats/snapshots.toml.

Needs a build with -DMallard_ENABLE_PYTHON=ON and HDF5: MALLARD_EXECUTABLE (or mallard on PATH) and its library.
"""
import os
import shutil
import subprocess
import tarfile

import numpy as np
import pytest

import mallard

HERE = os.path.dirname(os.path.abspath(__file__))
FORMATS = os.path.join(HERE, "..", "data", "formats")


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    d = tmp_path_factory.mktemp("run")
    shutil.copy(os.path.join(FORMATS, "snapshots.toml"), d)
    subprocess.run([mallard._lib.executable(), "-i", "snapshots.toml", "--kokkos-num-threads=2"], cwd=d,
                   check=True, capture_output=True)
    return d


SERIES = ["snapshots_vtkhdf.vtkhdf.series", "snapshots_series.vtkhdf", "snapshots_vtu.pvd", "snapshots_xdmf.xmf"]


def cell_corners(g):
    """Each cell's corner coordinates, sorted, so that cells compare whatever their point numbering."""
    return [np.array(sorted(map(tuple, g.points[g.connectivity[a:b]]))) for a, b in zip(g.offsets[:-1], g.offsets[1:])]


def test_every_format_holds_the_same_snapshots(run):
    files = [mallard.open(run / name) for name in SERIES]
    for f in files:
        assert [(s.t, s.step) for s in f.snapshots] == [(s.t, s.step) for s in files[0].snapshots]
        assert f.fields["U"].components == 3 and f.fields["RHO"].units == "kg m-3"
        for k in range(len(f.snapshots)):
            np.testing.assert_array_equal(f.read("RHO", k), files[0].read("RHO", k))
            np.testing.assert_array_equal(f.read("U", k), files[0].read("U", k))
        ref = cell_corners(files[0].geometry())
        assert all(np.array_equal(a, b) for a, b in zip(cell_corners(f.geometry()), ref))


def test_values_line_up_with_their_cells(run):
    # The initial density is 1 + 0.3 x + 0.2 y^2, whose average over a cell of height h is 1 + 0.3 xc + 0.2 (yc^2 + h^2 / 12)
    f = mallard.open(run / "snapshots_vtkhdf.vtkhdf.series")
    g = f.geometry(0)
    assert g.n_cells == f.mesh["cells"] == 30 and set(g.types) == {9}
    corners = [g.points[g.connectivity[a:b]] for a, b in zip(g.offsets[:-1], g.offsets[1:])]
    xc = np.array([c[:, 0].mean() for c in corners])
    yc = np.array([c[:, 1].mean() for c in corners])
    h = np.array([np.ptp(c[:, 1]) for c in corners])
    np.testing.assert_allclose(f.read("RHO", 0), 1 + 0.3 * xc + 0.2 * (yc**2 + h**2 / 12), rtol=1e-12)


def test_read_all_is_every_snapshot(run):
    f = mallard.open(run / "snapshots_series.vtkhdf")
    u = f.read_all("U")
    assert u.shape == (2, 30, 3)
    for k in range(2):
        np.testing.assert_array_equal(u[k], f.read("U", k))
    np.testing.assert_array_equal(f["RHO"], f.read("RHO", -1))
    with pytest.raises(IndexError):
        f.read("RHO", 2)


def test_restart_files(run):
    f = mallard.open(run / "snapshots_000001.restart")
    assert f.kind == "restart" and f.snapshots[0].step == 1
    snapshot = mallard.open(run / "snapshots_vtkhdf.vtkhdf.series")
    np.testing.assert_array_equal(f.read("RHO"), snapshot.read("RHO", 1))
    with pytest.raises(RuntimeError, match="holds no mesh"):
        f.geometry()


def test_probes_and_tables(run):
    csv = mallard.open(run / "snapshots_probe.csv")
    h5 = mallard.open(run / "snapshots_probe.h5")
    assert h5.kind == "probe" and csv.kind == "table"
    np.testing.assert_array_equal(h5.read_all("RHO").ravel(), csv.read_all("RHO"))
    np.testing.assert_allclose(h5.geometry().points[:, 0], [0.1, 0.5, 0.9])
    integrals = mallard.open(run / "snapshots_integrals.csv")
    np.testing.assert_array_equal(integrals.read_all("t"), integrals.times)


def test_mesh_files():
    f = mallard.open(os.path.join(FORMATS, "v0.8.0", "mesh.h5"), dim=2)
    g = f.geometry()
    assert f.kind == "mesh" and g.n_cells == f.mesh["cells"]
    assert np.all(g.points[:, 2] == 0) and g.offsets[-1] == g.connectivity.size


def test_run_directory(run):
    files = [os.path.basename(p) for p in mallard.run_files(run)]
    assert "snapshots_vtkhdf.vtkhdf.series" in files and "snapshots_vtkhdf_000000.vtkhdf" not in files
    records = mallard.run_record(run / "snapshots.jsonl")
    assert records[0]["type"] == "start" and records[-1]["type"] == "end"


def test_post_extract_gives_the_run_probe(run, tmp_path):
    restarts = [run / "snapshots_000000.restart", run / "snapshots_000001.restart"]
    report = mallard.post.extract(restarts, probe="line", output=tmp_path, threads=2, cwd=run)
    assert report is not None
    extracted = mallard.open(tmp_path / "snapshots_probe.csv")
    np.testing.assert_array_equal(extracted.read_all("RHO"), mallard.open(run / "snapshots_probe.csv").read_all("RHO"))


def test_command_errors(run):
    with pytest.raises(mallard.CommandError) as e:
        mallard.post.extract(run / "snapshots_000001.restart", no_such_option=True, cwd=run)
    assert e.value.returncode == 2


def test_diff_reports_differences_as_a_result(run):
    same = mallard.diff(run / "snapshots_vtkhdf_000001.vtkhdf", run / "snapshots_xdmf_000001.h5")
    assert same["exit_status"] == 0 and all(f["absolute"]["linf"] == 0 for f in same["fields"])
    changed = mallard.diff(run / "snapshots_vtkhdf_000000.vtkhdf", run / "snapshots_vtkhdf_000001.vtkhdf")
    assert changed["exit_status"] == 1 and changed["fields"][0]["absolute"]["linf"] > 0


@pytest.fixture(scope="module")
def logged_run(tmp_path_factory):
    """The run of snapshots.toml with its run record in mallard.jsonl, as mallard pack and --html need."""
    d = tmp_path_factory.mktemp("logged")
    text = open(os.path.join(FORMATS, "snapshots.toml")).read()
    (d / "input.toml").write_text(text.replace('run_log = "snapshots.jsonl"\n', ""))
    subprocess.run([mallard._lib.executable(), "-i", "input.toml", "--kokkos-num-threads=2"], cwd=d, check=True,
                   capture_output=True)
    return d


RESTARTS = ["snapshots_000000.restart", "snapshots_000001.restart"]


def mach(restart, gamma=1.4):
    r = mallard.open(restart)
    rho = r.read("RHO")
    u2 = (r.read("RHOU_X") ** 2 + r.read("RHOU_Y") ** 2) / rho**2
    p = (gamma - 1) * (r.read("RHOE") - 0.5 * rho * u2)
    return np.sqrt(u2 / (gamma * p / rho))


def test_post_derive(run, tmp_path):
    report = mallard.post.derive([run / f for f in RESTARTS], ["MACH"], latest=True, output=tmp_path, threads=2,
                                 cwd=run)
    assert len(report["snapshots"]) == 1
    derived = mallard.open(report["written"][0]["path"])
    assert len(derived.snapshots) == 1 and "MACH" in derived.fields and "RHO" not in derived.fields
    np.testing.assert_allclose(derived.read("MACH"), mach(run / RESTARTS[1]), rtol=1e-12)
    with_state = mallard.post.derive(run / RESTARTS[1], ["MACH"], with_state=True, output=tmp_path / "state",
                                     threads=2, cwd=run)
    assert "RHO" in mallard.open(with_state["written"][0]["path"]).fields


def test_post_average(run, tmp_path):
    report = mallard.post.average([run / f for f in RESTARTS], fields=["RHO"], products="none", output=tmp_path,
                                  threads=2, cwd=run)
    assert report["fields"] == ["RHO"] and report["products"] == []
    averages = mallard.open(report["written"][0])
    assert "MEAN_RHO" in averages.fields and "MEAN_P" not in averages.fields
    assert not any(name.startswith("COV_") for name in averages.fields)
    w = np.array([s["weight"] for s in report["snapshots"]])
    rho = np.array([mallard.open(run / f).read("RHO") for f in RESTARTS])
    np.testing.assert_allclose(averages.read("MEAN_RHO"), w @ rho / w.sum(), rtol=1e-12)
    one = mallard.post.average([run / f for f in RESTARTS], fields=["RHO"], steps=[1], output=tmp_path / "one",
                               threads=2, cwd=run)
    assert [s["snapshot"]["step"] for s in one["snapshots"]] == [1]


def test_report(logged_run, tmp_path):
    (logged_run / "report.html").unlink(missing_ok=True)
    report = mallard.report(logged_run)
    assert report["report"] == os.path.join(str(logged_run), "report.html")
    assert "<html" in open(report["report"]).read().lower()
    with pytest.raises(mallard.CommandError, match="has no mallard.jsonl") as e:
        mallard.report(tmp_path)
    assert e.value.returncode == 1


def test_pack_and_verify(logged_run, tmp_path):
    small = mallard.pack(logged_run, output=tmp_path / "small.tar")
    assert os.path.exists(small["archive"]) and small["with_outputs"] == "small"
    outputs = {f["path"] for f in small["files"] if f["role"] == "output"}
    assert "output/snapshots_probe.csv" in outputs and not any(p.endswith(".restart") for p in outputs)
    assert small["rerun"]["command"] == "mallard -i input.toml"
    none = mallard.pack(logged_run, output=tmp_path / "none.tar", with_outputs="none")
    assert not any(f["role"] == "output" for f in none["files"])

    ok = mallard.pack(small["archive"], verify=True)
    assert ok["result"] == "ok" and ok["exit_status"] == 0 and ok["problems"] == []
    with tarfile.open(small["archive"]) as t:
        t.extractall(tmp_path / "x", filter="data")
    top = small["rerun"]["directory"].split("/")[0]
    with open(tmp_path / "x" / top / "resolved.toml", "a") as f:
        f.write("\n")
    with tarfile.open(tmp_path / "tampered.tar", "w") as t:
        t.add(tmp_path / "x" / top, arcname=top)
    tampered = mallard.pack(tmp_path / "tampered.tar", verify=True)
    assert tampered["result"] == "differs" and tampered["exit_status"] == 1
    assert [p["path"] for p in tampered["problems"]] == ["resolved.toml"]

    with pytest.raises(mallard.CommandError, match="not a run directory"):
        mallard.pack(tmp_path / "tampered.tar")
