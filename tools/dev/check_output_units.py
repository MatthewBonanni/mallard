"""Runs a small case with every snapshot and monitor format and checks, with
h5py and the tools/mallard readers, that each file carries the same units for
the same field (docs/guide/output.md#units).

    python3 tools/dev/check_output_units.py build/src/mallard
"""
import os
import subprocess
import sys
import tempfile

import h5py

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mallard.tables import read_csv  # noqa: E402
from mallard.vtu import long_names, units  # noqa: E402

INPUT = """
[run]
n_steps = 2
cfl = 0.5
[mesh]
type = "cartesian"
Nx = 8
Ny = 4
Lx = 2.0
Ly = 1.0
[[periodic]]
zones = ["left", "right"]
[[boundaries]]
name = "bottom"
type = "wall_adiabatic"
[[boundaries]]
name = "top"
type = "wall_adiabatic"
[initialize]
type = "analytical"
rho = "1.176"
p = "1 atm"
u = ["10.0", "0.0"]
[physics]
type = "euler"
gamma = 1.4
p_ref = 101325.0
T_ref = 300.0
rho_ref = 1.176
[statistics]
fields = ["U_X", "P"]
products = ["U_X*P"]
[integrals]
[[forces]]
zone = "bottom"
[[probes]]
name = "line"
start = [0.1, 0.5]
end = [1.9, 0.5]
n_points = 5
variables = ["RHO", "U", "P"]
reductions = ["integral", "mean"]
[[probes]]
name = "h5"
format = "hdf5"
point = [1.0, 0.5]
variables = ["RHO", "T"]
[[write_data]]
prefix = "flow"
format = "vtu"
interval = 2
variables = ["RHO", "U", "P", "T", "H", "MEAN_*", "COV_*"]
[[write_data]]
prefix = "flow"
format = "hdf5"
interval = 2
variables = ["RHO", "U", "P", "T", "H", "MEAN_*", "COV_*"][[write_data]]
prefix = "flow"
interval = 2
variables = ["RHO", "U", "P", "T", "H", "MEAN_*", "COV_*"]
[[probes]]
name = "vtkhdf"
format = "vtkhdf"
point = [1.0, 0.5]
variables = ["RHO", "T"]
"""


def main():
    mallard = os.path.abspath(sys.argv[1])
    with tempfile.TemporaryDirectory() as work:
        with open(os.path.join(work, "input.toml"), "w") as f:
            f.write(INPUT)
        subprocess.run([mallard, "--dim", "2", "-i", "input.toml", "--kokkos-num-threads=2"], cwd=work, check=True,
                       stdout=subprocess.DEVNULL)

        def path(name):
            return os.path.join(work, name)

        vtu = units(path("flow_000002.vtu"))
        h5 = units(path("flow_000002.h5"))
        xmf = units(path("flow_000002.xmf"))
        vtkhdf = units(path("flow_000002.vtkhdf"))
        assert vtu == h5 == xmf == vtkhdf, (vtu, h5, vtkhdf)
        assert long_names(path("flow_000002.vtkhdf")) == long_names(path("flow_000002.vtu"))
        assert vtu["RHO"] == "kg m-3" and vtu["U"] == "m s-1" and vtu["COV_U_X_P"] == "kg s-3", vtu
        assert long_names(path("flow_000002.vtu")) == long_names(path("flow_000002.h5"))
        # h5py alone: the CF attributes are plain strings
        with h5py.File(path("flow_000002.h5")) as f:
            for name, dataset in f["fields"].items():
                assert dataset.attrs["units"] == vtu[name], name
                assert isinstance(dataset.attrs["long_name"], str) and dataset.attrs["long_name"], name
        with h5py.File(path("flow_mesh.h5")) as f:
            assert f["nodes/coordinates"].attrs["units"] == "m"
        assert units(path("probe_vtkhdf.vtkhdf"))["T"] == vtu["T"]
        probe = units(path("probe_h5.h5"))
        assert probe["time"] == "s" and probe["RHO"] == vtu["RHO"], probe

        columns, csv_units = read_csv(path("probe_line.csv"))
        assert list(csv_units) == list(columns), (csv_units, columns)
        assert (csv_units["RHO"], csv_units["P"], csv_units["U_X"]) == (vtu["RHO"], vtu["P"], vtu["U"]), csv_units
        assert csv_units["t"] == "s" and csv_units["x"] == "m"
        assert read_csv(path("probe_line_reductions.csv"))[1]["integral_RHO"] == "kg m-2"
        assert read_csv(path("forces_bottom.csv"))[1]["Fx_pressure"] == "kg s-2"
        assert read_csv(path("integrals.csv"))[1]["kinetic_energy"] == "kg m s-2"
        assert len(read_csv(path("integrals.csv"))[0]["t"]) == 3
    print("output units: consistent across VTKHDF, VTU, HDF5, XDMF and CSV")


if __name__ == "__main__":
    main()
