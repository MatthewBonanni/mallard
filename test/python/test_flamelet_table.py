"""The FPV table generator (tools/setup/flamelet_table.py): the beta-PDF integration against direct quadrature,
the progress-variable remap, and the table file. Needs scipy; Cantera is needed only to regenerate the tables."""
import os
import sys

import numpy as np
import pytest

scipy_stats = pytest.importorskip("scipy.stats")
from scipy import integrate  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools", "setup"))
import flamelet_table as ft  # noqa: E402


def quadrature(z, phi, Zm, S):
    """The mean of phi (piecewise linear on z) over the beta PDF of mean Zm and normalized variance S, by parts:
    phi(1) - integral of phi' F, with F the beta CDF, which is bounded where the PDF is singular."""
    g = 1.0 / S - 1.0
    F = scipy_stats.beta(Zm * g, (1.0 - Zm) * g).cdf
    slope = np.diff(phi) / np.diff(z)
    parts = [s * integrate.quad(F, z[i], z[i + 1], epsabs=1e-14, epsrel=1e-12)[0] for i, s in enumerate(slope)]
    return phi[-1] - np.sum(parts)


@pytest.mark.parametrize("Zm,S", [(0.3, 0.05), (0.1, 0.4), (0.55, 0.8), (0.9, 0.2), (0.02, 0.01)])
def test_beta_weights_match_quadrature(Zm, S):
    # Exact for phi piecewise linear on the nodes (a flamelet profile with a sharp peak at Z_st), whatever the PDF's
    # singularities at 0 and 1; a wrong incomplete-beta argument or a missing first moment fails it.
    z = ft.z_axis(101, 0.3, 4.0)
    phi = np.interp(z, [0.0, 0.3, 1.0], [300.0, 2000.0, 300.0]) + 50.0 * z ** 2
    exact = quadrature(z, phi, Zm, S)
    assert np.dot(ft.beta_weights(z, Zm, S), phi) == pytest.approx(exact, rel=1e-9)


def test_beta_weights_limits():
    z = ft.z_axis(101, 0.2, 3.0)
    phi = np.sin(3.0 * z) + z
    # S = 0: the laminar value; S = 1: the two pure streams; Z = 0 or 1: that stream
    assert np.dot(ft.beta_weights(z, 0.37, 0.0), phi) == pytest.approx(np.interp(0.37, z, phi), abs=1e-14)
    assert np.dot(ft.beta_weights(z, 0.37, 1.0), phi) == pytest.approx(0.63 * phi[0] + 0.37 * phi[-1], abs=1e-14)
    assert np.dot(ft.beta_weights(z, 1.0, 0.5), phi) == pytest.approx(phi[-1], abs=1e-14)
    # Continuous towards both ends of the variance axis
    for S in (1e-7, 1.0 - 1e-7):
        lim = np.dot(ft.beta_weights(z, 0.37, 0.0 if S < 0.5 else 1.0), phi)
        assert np.dot(ft.beta_weights(z, 0.37, S), phi) == pytest.approx(lim, abs=1e-4)


def test_progress_remap_reproduces_C_and_its_fields():
    # A table phi(Z, S, Lambda) whose C rises with Lambda and a field that is a known function of C: after the
    # remap to c, C is linear in c between its ends and the field follows C.
    lam = np.linspace(0.0, 1.0, 30)
    C = 0.1 * np.sqrt(lam) + 0.02 * lam
    table = np.zeros((2, 3, lam.size, 2))
    for i in range(2):
        for j in range(3):
            table[i, j, :, 0] = (1.0 + i + j) * C
            table[i, j, :, 1] = np.exp(table[i, j, :, 0])
    c = np.linspace(0.0, 1.0, 11)
    out, nonmonotonic = ft.to_progress(table, 0, c)
    assert nonmonotonic == 0
    for i in range(2):
        for j in range(3):
            Cmax = (1.0 + i + j) * C[-1]
            np.testing.assert_allclose(out[i, j, :, 0], c * Cmax, rtol=1e-12, atol=1e-15)
            np.testing.assert_allclose(out[i, j, :, 1], np.exp(c * Cmax), rtol=2e-3)


def test_table_file_round_trip(tmp_path):
    header = {"format": ft.FORMAT, "version": ft.VERSION, "model": "fpv",
              "axes": [{"name": "Z", "values": [0.0, 0.5, 1.0]}, {"name": "S", "values": [0.0, 1.0]},
                       {"name": "c", "values": [0.0, 0.25, 1.0]}],
              "fields": [{"name": "C", "units": "1"}, {"name": "T", "units": "K"}]}
    data = np.random.default_rng(631).random((3, 2, 3, 2))
    path = tmp_path / "t.mtab"
    ft.write_table(path, header, data)
    h, d = ft.read_table(path)
    assert h == header
    np.testing.assert_array_equal(d, data)
    with pytest.raises(ValueError):
        ft.write_table(path, header, data[:, :, :2])


def test_repository_table_is_consistent():
    # test/data/flamelet/h2_fpv.mtab, which the C++ tests read: written from its input by this generator
    h, d = ft.read_table(os.path.join(HERE, "..", "data", "flamelet", "h2_fpv.mtab"))
    names = [f["name"] for f in h["fields"]]
    iC, iT = names.index("C"), names.index("T")
    assert [a["name"] for a in h["axes"]] == ["Z", "S", "c"]
    assert h["provenance"]["input"] == open(os.path.join(HERE, "..", "data", "flamelet", "h2_fpv.toml")).read()
    # C rises along c, and the burning end is hotter than mixing near stoichiometry
    assert np.all(np.diff(d[..., iC], axis=2) >= -1e-15)
    i_st = int(np.argmin(np.abs(np.array(h["axes"][0]["values"]) - h["Z_st"])))
    assert d[i_st, 0, -1, iT] > d[i_st, 0, 0, iT] + 700.0
