# The Python package

How the `mallard` Python package is laid out, and how the programmatic
interface ([#345](https://github.com/MatthewBonanni/mallard/issues/345)) fits
into it. The user-facing pages are [Python](../guide/python.md) and the
[Python API](../guide/python_api.md); the reader under it is
[Reading output](output_reader.md).

## One package

There is one Python distribution, `mallard-python`, imported as `mallard`, for both
post-processing and driving the solver. Its source is `tools/mallard`, and its
version is Mallard's (`tools/mallard/_version.py`, checked against
`project(VERSION)` by every build with `Mallard_ENABLE_PYTHON`, on by default).
The containers, Spack and `cmake --install` install it with Mallard
([Installation](../installation.md)).

## Post-processing layer

`mallard.open`, `File`, `run_record`, `run_files`, `mallard.command`, `info`,
`diff`, `report`, `pack` and `post.*` read what Mallard writes and run its commands. They depend
on numpy and on the reader library (`libmallard2d`/`libmallard3d`, the C
interface of `src/io/reader_c_api.h`, loaded with ctypes) and on nothing else:
no solver bindings, no compiled extension module, no MPI. Commands run the
`mallard` executable as a process. This layer works with any Mallard build that
has the reader library, and with any Python 3.10+.

## Solver bindings

The bindings of #345 (`mallard.Case`, `mallard.Solver`: a case built in code,
`setup`, `step`, `advance_to`, fields as arrays without copies) are a compiled
extension module of the same package, built by the same CMake build
and imported only by the code that uses it. The post-processing layer does not
import it, so a build or environment without it still reads files and runs
commands.

The extension is `mallard._core2d` or `mallard._core3d` (nanobind, built with
`-DMallard_ENABLE_PYTHON_BINDINGS=ON` from `src/api/python_module.cpp`),
linked against `libmallard<dim>d` through the C++ API only, and installed into
the package's copies (`share/mallard/tools/mallard`, `Mallard_PYTHON_INSTALL_DIR`),
or found next to the library in a build tree. `tools/mallard/_solver.py` is the
Python side: `Case` (the TOML text, checked by the extension), `Session` (one
per process; the dimension of its extension), `Solver` (one per case, closed
before Kokkos is finalized at exit), callbacks held in Python (so that a
callback holding its solver is a cycle the garbage collector sees, not one
through C++) and called with the GIL from the C++ hooks, and `DeviceArray`,
which adds `__cuda_array_interface__` to nanobind's DLPack tensor. Host fields
are NumPy arrays of the solver's memory whose base keeps the solver alive.
`mallard/__init__.py` exports these names lazily (PEP 562), so `import mallard`
never loads the extension.

Once the extension exists, the reader moves into it: `mallard.open` and `File`
call the extension instead of ctypes, and the C interface goes away. Their
Python API (functions, arguments, returned types) stays the same, which the
tests in `test/python` pin.

## Dependencies

numpy is the only required dependency. Heavier ones are extras of the same
distribution, installed only by those who need them: `mallard-python[mpi]` (mpi4py,
for the bindings under MPI), and others as the bindings need them. A
post-processing wheel of `mallard-python` without the extension comes first
([#792](https://github.com/MatthewBonanni/mallard/issues/792)); wheels with the
compiled extension follow with #345.

## Performance

Reading is measured against ParaView's `pvbatch` on the same files
([#794](https://github.com/MatthewBonanni/mallard/issues/794);
`benchmarks/python_read`). Each case is one run writing 5 snapshots of `RHO`,
`U` and `P` as a VTKHDF series file and an XDMF series, and, on 4 ranks, as
a PVTU series of 4 pieces. A script reads the last snapshot (geometry, `RHO`
and `U`) or `RHO` at every snapshot, one at a time; the `pvbatch` script
enables only those arrays and takes them from the reader's output without
`Fetch`'s copy. Each figure is the median of 3 fresh processes on one thread
of an AMD EPYC 7763, warm page cache, ParaView 6.2.0 (its Linux binary) and
a Release build with OpenMP, MPI and HDF5 1.14.6. "In the script" is from
`mallard.open` or `OpenDataFile` to the last array; the whole script includes
starting Python or `pvbatch` (GNU `time`); peak memory is the process's
maximum resident set, for the package, the package before #794 (which copied
each array) and `pvbatch`:

| Mesh | Format | Read | In the script (s) | Whole script (s) | Peak memory (MB) |
|---|---|---|---|---|---|
| 2D, 115k triangles | VTKHDF | last snapshot | 0.18 / 0.13 | 0.29 / 3.25 | 55 / 58 / 419 |
| 2D, 115k triangles | VTKHDF | series | 0.17 / 0.15 | 0.29 / 3.26 | 48 / 48 / 421 |
| 2D, 115k triangles | XDMF | last snapshot | 0.18 / 0.21 | 0.30 / 3.38 | 55 / 60 / 423 |
| 2D, 115k triangles | XDMF | series | 0.18 / 0.51 | 0.30 / 3.67 | 50 / 50 / 446 |
| 2D, 115k triangles | PVTU | last snapshot | 0.19 / 0.14 | 0.31 / 3.21 | 52 / 52 / 427 |
| 2D, 115k triangles | PVTU | series | 0.18 / 0.18 | 0.30 / 3.42 | 42 / 42 / 431 |
| 2D, 1M quadrilaterals | VTKHDF | last snapshot | 0.24 / 0.17 | 0.37 / 3.53 | 144 / 188 / 502 |
| 2D, 1M quadrilaterals | VTKHDF | series | 0.21 / 0.26 | 0.33 / 3.57 | 68 / 71 / 548 |
| 2D, 1M quadrilaterals | XDMF | last snapshot | 0.29 / 0.84 | 0.42 / 4.15 | 168 / 177 / 560 |
| 2D, 1M quadrilaterals | XDMF | series | 0.22 / 3.48 | 0.34 / 6.72 | 85 / 91 / 778 |
| 2D, 1M quadrilaterals | PVTU | last snapshot | 0.31 / 0.22 | 0.44 / 3.49 | 166 / 170 / 591 |
| 2D, 1M quadrilaterals | PVTU | series | 0.24 / 0.44 | 0.38 / 3.76 | 65 / 66 / 656 |
| 3D, 1M hexahedra | VTKHDF | last snapshot | 0.25 / 0.17 | 0.38 / 3.38 | 175 / 249 / 534 |
| 3D, 1M hexahedra | VTKHDF | series | 0.21 / 0.30 | 0.33 / 3.64 | 69 / 71 / 611 |
| 3D, 1M hexahedra | XDMF | last snapshot | 0.33 / 1.37 | 0.48 / 4.58 | 236 / 265 / 604 |
| 3D, 1M hexahedra | XDMF | series | 0.23 / 6.36 | 0.35 / 9.63 | 85 / 122 / 1015 |
| 3D, 1M hexahedra | PVTU | last snapshot | 0.35 / 0.23 | 0.46 / 3.50 | 204 / 245 / 660 |
| 3D, 1M hexahedra | PVTU | series | 0.24 / 0.48 | 0.36 / 3.76 | 65 / 67 / 774 |
| 3D, 8M hexahedra | VTKHDF | last snapshot | 0.64 / 0.36 | 0.81 / 3.66 | 1033 / 1532 / 1398 |
| 3D, 8M hexahedra | VTKHDF | series | 0.45 / 1.03 | 0.62 / 4.25 | 231 / 231 / 2020 |
| 3D, 8M hexahedra | XDMF | last snapshot | 1.31 / 9.97 | 1.49 / 13.35 | 1590 / 1612 / 1949 |
| 3D, 8M hexahedra | XDMF | series | 0.59 / 48.22 | 0.77 / 51.43 | 231 / 655 / 4797 |
| 3D, 8M hexahedra | PVTU | last snapshot | 1.54 / 0.71 | 1.75 / 3.91 | 1205 / 1614 / 2405 |
| 3D, 8M hexahedra | PVTU | series | 0.74 / 2.64 | 0.91 / 5.74 | 225 / 255 / 2858 |

- **Same values.** `RHO` agrees bit for bit with `pvbatch`'s, in the same
  order, in all 24 cases, and the sums of the points and fields read agree.
- **Whole scripts.** The package's take 0.3-1.8 s, `pvbatch`'s 3.2-51 s:
  starting `pvbatch` alone takes 3.2-3.4 s and 420 MB.
- **Series.** The package reads them faster in every case but the smallest
  VTKHDF one (0.17 s against 0.15 s): 2.3x on the 8M-cell VTKHDF series, 3.6x
  on its PVTU series and 80x on its XDMF series, where ParaView's XDMF reader
  takes as long at each time as for the first. It holds one snapshot at a
  time; `pvbatch` peaks at 8-21x more memory, 420 MB of it its own.
- **One snapshot.** ParaView's VTKHDF and PVTU readers are faster in the
  script (0.36 s against 0.64 s for 8M hexahedra in VTKHDF). Two costs of the
  package's: `mallard.open` reads every point to give the mesh's bounds, and
  most of a small file's 0.17 s is loading the library, whose MPI
  dependencies take 0.15 s to load although the reader never calls MPI.
  ParaView's XDMF reader is 1.2-8x slower than the package.
- **Memory.** Arrays view the library's memory, so a read peaks at what it
  returns: the 8M-cell snapshot returns 996 MB and peaks at 1,033 MB (1,532 MB
  before #794, 1,398 MB in `pvbatch`). Opening an XDMF series reads its mesh
  topology in parts (231 MB instead of 655 MB for 8M hexahedra).
