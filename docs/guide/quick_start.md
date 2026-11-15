# Quick start

Build Mallard on a CPU and run your first case in three commands (GCC 15, Clang 22 or Apple Clang 21
or newer, CMake 4.2 or newer and git are all it needs; see [supported platforms](../installation.md#supported-platforms)):

```bash
git clone --recursive https://github.com/MatthewBonanni/mallard.git && cd mallard
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_SYSTEM_KOKKOS=OFF -DKokkos_ENABLE_THREADS=ON && cmake --build build -j
cd examples/sod && ../../build/src/mallard -i input.toml
```

The last line runs the Sod shock tube in about a second and writes `solut/sod.pvd` (VTU: this build has no
HDF5; with `-DMallard_ENABLE_HDF5=ON` it writes VTKHDF, `solut/sod.vtkhdf`), which ParaView
opens. `build/src/mallard list templates` lists ready-to-run cases to copy (`mallard new cylinder` creates
one with its mesh, run and post-processing scripts). GPU builds (CUDA, HIP), MPI and the other options are in [Installation](https://matthewbonanni.github.io/mallard/docs/installation/).

## Where next

The [tutorials](../tutorials/index.md) are an ordered path from this first run to
multi-GPU large-eddy simulation, each building on the last. The
[examples](https://matthewbonanni.github.io/mallard/docs/examples/) are ready-to-run cases to
browse, and the user guide covers each topic: [input files](input_files.md), [expressions](expressions.md), [meshes](meshes.md),
[boundary conditions](boundary_conditions.md), [numerics](numerics.md), [output](output.md).
