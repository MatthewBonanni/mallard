# Format fixtures

Files written by past releases, which `test/format_compat_test.cpp` reads to
keep the [compatibility promise](../../../docs/guide/output.md#file-formats-and-compatibility):
each release reads the restart and mesh files of the earlier releases.

| Input | Restart files it writes (version) |
|---|---|
| `gas.toml` | perfect gas, analytical initial state (2) |
| `statistics.toml` | with `[statistics]` averages (3) |
| `mixture.toml` | H2/air mixture with chemistry: species and `CHEM_H` (2) |
| `nscbc.toml` | characteristic inlet and outlet: face states (4; 0.7.0 on) |

Each `vX.Y.Z/` holds `<case>_000000.restart` (the initial state) and
`<case>_000002.restart` (two steps later), and `mesh.h5`, the mesh of
`gas.toml` from `mallard-mesh-convert`; `v0.8.0/curved_mesh.h5` is
`annulus.msh` (second-order walls, from `tools/make_annulus_mesh.py
--n-r 2 --n-theta 8 --order 2`).

Each set was written by that tag's own 2D double-precision build
(`git archive vX.Y.Z`, built with `-DMallard_ENABLE_HDF5=ON`). At a release,
add its set with `./generate.sh <build> vX.Y.Z` and list it in the test.
