"""tools/setup/profiles.py: analytic profiles, rescaling as the solver's scale, checks."""
import importlib.util
import math
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = importlib.util.spec_from_file_location("profiles", os.path.join(HERE, "..", "..", "tools", "setup", "profiles.py"))
profiles = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profiles)


def run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["profiles.py", *map(str, argv)])
    return profiles.main()


def test_spalding_profile_satisfies_the_law(tmp_path, monkeypatch):
    out = tmp_path / "bl.csv"
    run(monkeypatch, "law", "spalding", out, "--delta", 0.02, "--u-inf", 30, "--u-tau", 1.2, "--nu", 1.5e-5)
    metadata, header, rows = profiles.read(out)
    assert header == ["n_m", "u"] and metadata["at"] == ["n_m"] and metadata["reference"]["length"] == 0.02
    kappa, b = 0.41, 5.0
    for n, u in rows:
        up, yp = u / 1.2, n * 1.2 / 1.5e-5
        k = kappa * up
        assert up + math.exp(-kappa * b) * (math.exp(k) - 1 - k - k * k / 2 - k ** 3 / 6) == pytest.approx(yp, rel=1e-10, abs=1e-12)


def test_transform_rescales_each_column_by_its_dimension(tmp_path, monkeypatch):
    # As the solver's ScaleRescalesEachColumnByItsDimension: lengths x3, velocities x1.25, density x2
    src = tmp_path / "in.csv"
    profiles.write(src, {"units": {"d": "mm", "U": "m/s", "R": "m^2/s^2", "T": "K", "p": "bar"},
                         "reference": {"length": 0.01, "velocity": 4.0, "density": 1.0}},
                   ["d", "U", "R", "T", "p"], [[4, 3, 0.25, 300, 2]])
    out = tmp_path / "out.csv"
    run(monkeypatch, "transform", src, out, "--scale", "length=0.03", "--scale", "velocity=5", "--scale", "density=2")
    metadata, _, rows = profiles.read(out)
    assert rows[0] == pytest.approx([12, 3.75, 0.25 * 1.5625, 300, 2 * 2 * 1.5625])
    assert metadata["reference"] == {"length": 0.03, "velocity": 5.0, "density": 2.0}
    assert metadata["transforms"] == ["scale length x3, velocity x1.25, density x2"]


def test_show_reports_a_coordinate_that_does_not_increase(tmp_path, monkeypatch, capsys):
    src = tmp_path / "bad.csv"
    profiles.write(src, {"units": {"y": "furlong"}}, ["y", "u"], [[0, 1], [0.2, 2], [0.1, 3]])
    assert run(monkeypatch, "show", src) == 1
    text = capsys.readouterr().out
    assert "the coordinate y does not increase" in text and "unknown unit furlong" in text
