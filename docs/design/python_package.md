# The Python package

How the `mallard` Python package is laid out, and how the programmatic
interface ([#345](https://github.com/MatthewBonanni/mallard/issues/345)) fits
into it. The user-facing pages are [Python](../guide/python.md) and the
[Python API](../guide/python_api.md); the reader under it is
[Reading output](output_reader.md).

## One package

There is one Python distribution, `mallard`, imported as `mallard`, for both
post-processing and driving the solver. Its source is `tools/mallard`, and its
version is Mallard's (`tools/mallard/_version.py`, checked against
`project(VERSION)` by every build with `Mallard_ENABLE_PYTHON`, on by default).
The containers, Spack and `cmake --install` install it with Mallard
([Installation](../installation.md)).

## Post-processing layer

`mallard.open`, `File`, `run_record`, `run_files`, `mallard.command`, `info`,
`diff` and `post.*` read what Mallard writes and run its commands. They depend
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

Once the extension exists, the reader moves into it: `mallard.open` and `File`
call the extension instead of ctypes, and the C interface goes away. Their
Python API (functions, arguments, returned types) stays the same, which the
tests in `test/python` pin.

## Dependencies

numpy is the only required dependency. Heavier ones are extras of the same
distribution, installed only by those who need them: `mallard[mpi]` (mpi4py,
for the bindings under MPI), and others as the bindings need them. A
post-processing wheel without the extension comes first
([#792](https://github.com/MatthewBonanni/mallard/issues/792)); wheels with the
compiled extension follow with #345.
