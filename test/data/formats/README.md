# Format fixtures

Files written by past releases, which `test/format_compat_test.cpp` reads to
keep the [compatibility promise](../../../docs/guide/output.md#formats-and-compatibility):
each release reads the restart and mesh files of the earlier releases.

| Input | Restart files it writes (version) |
|---|---|
| `gas.toml` | perfect gas, analytical initial state (2; 7 from 0.10.0) |
| `statistics.toml` | with `[statistics]` averages (3; 7 from 0.10.0) |
| `mixture.toml` | H2/air mixture with chemistry: species and `CHEM_H` (2; 7 from 0.10.0) |
| `nscbc.toml` | characteristic inlet and outlet: face states (4, 0.7.0 on; 7 from 0.10.0) |

Each `vX.Y.Z/` but `v0.9.1/` holds `<case>_000000.restart` (the initial state) and
`<case>_000002.restart` (two steps later), and `mesh.h5`, the mesh of
`gas.toml` from `mallard mesh convert`; `curved_mesh.h5` (from 0.8.0) is
`annulus.msh` (second-order walls, from `tools/mesh/make_annulus_mesh.py
--n-r 2 --n-theta 8 --order 2`).

From 0.9.1, `vX.Y.Z/snapshots*` are the output of `snapshots.toml` (one
step): snapshots in every format (VTKHDF files and their `.vtkhdf.series`,
a `series = true` VTKHDF file, VTU and their `.pvd`, XDMF output), restart
files, probe CSV and HDF5 files, forces and integrals, and the run record,
which `test/output_reader_test.cpp` reads: 0.9.1's for the readers' fallback
for files without snapshot blocks, later ones from their blocks. The run
blocks' host and directory are replaced (`fixture-host`, `/fixtures/...`), and
the chunks of extendible HDF5 datasets shrunk to their extent.

Each set was written by that tag's own 2D double-precision build
(`git archive vX.Y.Z`, built with `-DMallard_ENABLE_HDF5=ON`). At a release,
add its set with `./generate.sh <build> vX.Y.Z` and list it in the test.
