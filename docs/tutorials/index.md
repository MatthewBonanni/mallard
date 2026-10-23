# Tutorials

The tutorials are a learning path. Each one takes a case from start to finish and adds a few
concepts to the ones before it, from a first run on a laptop to large-eddy simulation on several GPUs.
Each covers the problem and its reference solution, the mesh, the input file, running on a CPU, a GPU or
several MPI ranks, monitoring, post-processing, and the comparison. Everything they use is in the
repository under [`tutorials/`](../../tutorials): the `.geo` scripts, the input files and the Python
scripts that draw every figure on these pages. CI runs each tutorial at reduced size on every change, so
the commands stay working.

| | Tutorial | You learn | Run time | Reference |
|---|---|---|---|---|
| 1 | [First simulation](first_simulation.md) | Building Mallard, running a case, the input file's tables, presets, schemes and Riemann solvers | seconds, laptop | Exact Riemann solution |
| 2 | [Oblique shock on a ramp](wedge.md) | Generated meshes and boundary zones, inflow, outflow and slip walls, units and the gas constant, line probes, ParaView, mesh refinement | seconds, laptop | θ-β-M relation |
| 3 | [Vortex shedding behind a cylinder](cylinder.md) | A Gmsh geometry with a structured ring and triangles, Navier-Stokes, initial conditions as expressions, wall forces, Strouhal number | 5 minutes, laptop | Reference computations |
| 4 | [Supersonic flow over a cone](cone.md) | 3D builds, tetrahedra, symmetry planes, a characteristic far field, surface output, MPI ranks and GPUs | minutes, laptop | Taylor-Maccoll solution |
| 5 | Premixed flame against Cantera (in preparation) | Gas mixtures, mechanisms, finite-rate chemistry, flame speed | | Cantera |
| 6 | [Channel LES](channel_les.md) | Periodic Gmsh hexahedra, the Sigma model, constant mass flow, running statistics, the dissipation budget | an hour, one GPU | DNS of Moser, Kim & Mansour |
| 7 | [Laminar airfoil](airfoil.md) | A structured Gmsh C-grid, steady state by time marching, wall forces, Cp and Cf | | High-Order CFD Workshop case C1.3 |

Tutorials 1 to 4 run on a laptop CPU. For more cases, the
[examples](https://matthewbonanni.github.io/mallard/docs/examples/) are a catalog of ready-to-run inputs,
each with what it demonstrates and how long it takes, and the
[validation](https://matthewbonanni.github.io/mallard/validation/) page shows the accuracy to expect.

The tutorials assume a build of Mallard (see the
[first simulation](first_simulation.md)), here with MPI for the
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
tutorials/ci.sh build/src/mallard build-3d/src/mallard /tmp/tutorials
```
