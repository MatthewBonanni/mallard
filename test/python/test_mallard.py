"""The Python package (tools/mallard) on the output of a run of test/data/formats/snapshots.toml.

Needs a build with -DMallard_ENABLE_PYTHON=ON and HDF5: MALLARD_EXECUTABLE (or mallard on PATH) and its library.
"""
import os
import shutil
import subprocess

import numpy as np
import pytest

import mallard

HERE = os.path.dirname(os.path.abspath(__file__))
FORMATS = os.path.join(HERE, "..", "data", "formats")


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    d = tmp_path_factory.mktemp("run")
    shutil.copy(os.path.join(FORMATS, "snapshots.toml"), d)
    subprocess.run([mallard.commands._lib.executable(), "-i", "snapshots.toml", "--kokkos-num-threads=2"], cwd=d,
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
