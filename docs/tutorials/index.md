# Tutorials

Each tutorial takes one case from start to finish: the geometry, a mesh from
a Gmsh script, the input file, running on a CPU, a GPU or several MPI ranks,
monitoring the run, post-processing, and the final comparison with reference
data. Everything they use is in the repository under
[`tutorials/`](../../tutorials): the `.geo` scripts, the input files and the
Python scripts that draw every figure on these pages. CI runs each tutorial
at reduced size on every change, so the commands stay working.

| Tutorial | What it shows | Reference |
|---|---|---|
| [Supersonic cone](cone.md) | 3D Euler flow at Mach 2 over a 15° cone: a tetrahedral Gmsh mesh, symmetry planes, probes and surface output | Taylor-Maccoll exact solution |

The tutorials assume a build of Mallard (see the
[first simulation](../guide/first_simulation.md)), here with MPI for the
multi-rank runs, and Python 3 with the packages of
`tutorials/requirements.txt` (numpy, scipy, matplotlib, gmsh, cantera), which
also provide the `gmsh` command:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_SYSTEM_KOKKOS=OFF \
      -DKokkos_ENABLE_OPENMP=ON -DMallard_ENABLE_MPI=ON
cmake --build build -j
cmake -S . -B build-3d -DCMAKE_BUILD_TYPE=Release -DUSE_SYSTEM_KOKKOS=OFF \
      -DKokkos_ENABLE_OPENMP=ON -DMallard_ENABLE_MPI=ON -DMallard_DIM=3
cmake --build build-3d -j
python3 -m venv .venv
.venv/bin/pip install -r tutorials/requirements.txt
export PATH=$PWD/.venv/bin:$PATH
```

On macOS use `-DKokkos_ENABLE_THREADS=ON` instead of OpenMP. Each tutorial
runs in a directory of your choice; the commands below use `run/<tutorial>`
inside the repository and refer to the repository as `$MALLARD`:

```bash
export MALLARD=$PWD
```

`tutorials/ci.sh` runs the same steps at reduced size, as CI does:

```bash
tutorials/ci.sh build/src/Mallard build-3d/src/Mallard /tmp/tutorials
```
