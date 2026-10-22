# Quick start

Build Mallard on a CPU and run your first case in three commands (a C++20 compiler, CMake 3.16 or
newer and git are all it needs):

```bash
git clone --recursive https://github.com/MatthewBonanni/mallard.git && cd mallard
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_SYSTEM_KOKKOS=OFF -DKokkos_ENABLE_THREADS=ON && cmake --build build -j
cd examples/sod && ../../build/src/Mallard -i input.toml
```

The last line runs the Sod shock tube in about a second and writes `solut/sod.pvd`, which ParaView
opens. GPU builds (CUDA, HIP), MPI and the other options are in [Installation](https://matthewbonanni.github.io/mallard/docs/installation/).

## Where next

The [tutorials](../tutorials/index.md) are an ordered path, each building on the last:

1. [First simulation](../tutorials/first_simulation.md): the Sod shock tube against the exact solution, then changing the case
2. [Oblique shock on a ramp](../tutorials/wedge.md): a 2D generated mesh, boundary conditions, probes, ParaView
3. [Vortex shedding behind a cylinder](../tutorials/cylinder.md): a Gmsh mesh, viscous flow, wall forces
4. [Supersonic flow over a cone](../tutorials/cone.md): 3D, tetrahedra, symmetry planes, MPI and GPUs
5. Premixed flame against Cantera: mixtures and chemistry (in preparation)
6. [Channel LES](../tutorials/channel_les.md): large-eddy simulation, statistics, multi-GPU runs

The [examples](https://matthewbonanni.github.io/mallard/docs/examples/) are 40 ready-to-run cases to
browse, and the user guide covers each topic: [input files](input_files.md), [meshes](meshes.md),
[boundary conditions](boundary_conditions.md), [numerics](numerics.md), [output](output.md).
