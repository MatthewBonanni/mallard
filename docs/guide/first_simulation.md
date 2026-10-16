# First simulation

This walk-through takes about ten minutes: build Mallard on a CPU, run the Sod shock tube, check the result against the exact solution, and then change the case. It assumes a C++20 compiler, CMake 3.16 or newer, git, and Python 3.

## 1. Build

```bash
git clone --recursive https://github.com/MatthewBonanni/mallard.git
cd mallard
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_SYSTEM_KOKKOS=OFF -DKokkos_ENABLE_THREADS=ON
cmake --build build -j
```

This builds Kokkos from the bundled submodule with its `Threads` backend, which works on Linux and macOS. The solver is `build/src/Mallard`; `build/test/MallardTest` runs the test suite. For OpenMP, use `-DKokkos_ENABLE_OPENMP=ON` instead (see [Installation](https://matthewbonanni.github.io/mallard/docs/installation/)). For an NVIDIA GPU, enable CUDA with the architecture of your GPU (here an A100), compile through Kokkos' `nvcc_wrapper`, and enable OpenMP as the host backend so that the setup work done on the CPU (TENO's stencil precomputation, which takes minutes for a million cells on one thread) runs in parallel:

```bash
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DUSE_SYSTEM_KOKKOS=OFF \
      -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON -DKokkos_ENABLE_OPENMP=ON \
      -DCMAKE_CXX_COMPILER=$PWD/src/external/kokkos/bin/nvcc_wrapper
cmake --build build-cuda -j
```

The commands below use the CPU build; the CUDA build runs the same inputs with `build-cuda/src/Mallard -i input.toml`.

The post-processing scripts in `tools/` need a few Python packages:

```bash
python3 -m venv .venv
.venv/bin/pip install numpy scipy matplotlib imageio imageio-ffmpeg
```

## 2. Run the Sod shock tube

```bash
cd examples/sod
../../build/src/Mallard -i input.toml --kokkos-num-threads=4
```

The run takes about a second. Mallard prints a summary of the setup, then a progress row every 100 steps (step, time, time step, throughput, minimum density and pressure, maximum Mach number, share of troubled cells; see [the log](output.md#the-log)), and writes eleven snapshots, `solut/sod_000000.vtu` to `solut/sod_000010.vtu` (t = 0 to 0.2), plus `solut/sod.pvd`, which ParaView opens as a time series.

The input file, `input.toml`, has one table per part of the problem; every key is documented in the [input reference](../input.md):

| Table | In this case |
|---|---|
| `[run]` | Stop at `t_stop = 0.2` with time steps at CFL 1 |
| `[mesh]` | A generated 200 × 4 quadrilateral strip on [0, 1] × [0, 0.02] |
| `[initialize]` | Density and pressure as expressions in `x`: (1, 1) left of x = 0.5, (0.125, 0.1) right of it, at rest |
| `[[boundaries]]` | Symmetry (slip walls) on all four sides, which makes the strip one-dimensional |
| `[numerics]` | HLLC flux, SSPRK3 time integration, fifth-order TENO-E reconstruction |
| `[physics]` | Euler equations, γ = 1.4 |
| `[[write_data]]` | VTU output of density, velocity, pressure and temperature every 0.02 |

## 3. Compare with the exact solution

```bash
../../.venv/bin/python ../../tools/plot_sod.py sod.png solut/sod_000010.vtu
```

`sod.png` shows the computed density along the strip against the exact Riemann solution at t = 0.2: the rarefaction, the contact discontinuity (spread over about five cells) and the shock (two cells). The [validation page](https://matthewbonanni.github.io/mallard/validation/#sod-shock-tube) has the same comparison for velocity and pressure, and the error under mesh refinement.

## 4. Change the case

Edit `input.toml` and run again:

- **Resolution.** Set `Nx = 400` (and `Ly = 0.01` to keep square cells). The discontinuities become half as wide.
- **Scheme.** Replace the reconstruction with second-order MUSCL,

    ```toml
    [numerics.face_reconstruction]
    type = "MUSCL"
    limiter = "venkatakrishnan"
    ```

    or try `order = 3` with TENO. The contact discontinuity is where the schemes differ most. The [Numerics](numerics.md#reconstruction) page compares them on this case.
- **Riemann solver.** `Rusanov`, `HLL`, `HLLC`, `Roe` or `RHLL`. With fifth-order reconstruction the choice changes the error on this case by less than 20%, Rusanov being the most dissipative. It matters more in 2D: for shocks aligned with a quadrilateral grid, use `RHLL` (see [Shu–Osher](https://matthewbonanni.github.io/mallard/validation/#shu-osher-problem)).

## 5. A 2D case

The 2D Riemann problem (configuration 3 of Lax & Liu) is the case in the [gallery](https://matthewbonanni.github.io/mallard/gallery/), on a coarser mesh: 320,000 triangles, about 20 minutes on 12 CPU threads.

```bash
cd ../riemann_2d
../../build/src/Mallard -i input.toml --kokkos-num-threads=8
../../.venv/bin/python ../../tools/animate.py solut riemann
```

`animate.py` writes `riemann.mp4`, `riemann.gif` and the last frame as `riemann_final.png`: density with contours beside a numerical schlieren image, one frame per snapshot. Open `solut/riemann.pvd` in ParaView for anything else.

## 6. A 3D case {#6-a-3d-case}

The spatial dimension is fixed when Mallard is built. `-DMallard_DIM=3` gives a 3D solver for meshes of tetrahedra, hexahedra, prisms and pyramids, in any mix; build it in its own directory, next to the 2D one:

```bash
cmake -S . -B build3d -DCMAKE_BUILD_TYPE=Release -DUSE_SYSTEM_KOKKOS=OFF \
      -DKokkos_ENABLE_THREADS=ON -DMallard_DIM=3
cmake --build build3d -j
```

3D inputs use the same keys as 2D, with three differences: vectors have three components (`u = [u_x, u_y, u_z]`, `gravity`, `rhou`); expressions are in `x`, `y` and `z`; and generated meshes are `Nx × Ny × Nz` boxes with two more boundary zones, `back` (z = 0) and `front` (z = Lz). Each block of the box holds:

| `[mesh] type` | Cells per block |
|---|---|
| `cartesian` | one hexahedron |
| `cartesian_tet` | six tetrahedra |
| `cartesian_prism` | two triangular prisms |
| `cartesian_pyramid` | six pyramids |
| `cartesian_mixed` | hexahedra, pyramids and prisms in successive thirds of x |

Gmsh meshes (format 2.2 or 4.1) work as in 2D, each physical surface becoming a boundary zone. The output is VTU with 3D cells and the velocity as a 3-vector `U`.

The `explosion_3d` example, a spherical explosion on 64³ hexahedra, takes about 11 minutes on 6 threads:

```bash
cd examples/explosion_3d
../../build3d/src/Mallard -i input.toml --kokkos-num-threads=6
```

The [validation page](https://matthewbonanni.github.io/mallard/validation/#spherical-explosion) compares it with a one-dimensional radial solution, and also shows the Sod shock tube in 3D and the Taylor–Green vortex.

## Next steps

- [Input files](input_files.md): how an input is organized, expressions, units.
- The user guide by topic: [meshes](meshes.md), [physics](physics.md),
  [boundary conditions](boundary_conditions.md), [numerics](numerics.md),
  [output](output.md), [running in parallel](parallel.md).
- [Examples](https://matthewbonanni.github.io/mallard/docs/examples/): 40 cases, from shock tubes to turbulent flames, and how long they take.
- [Validation](https://matthewbonanni.github.io/mallard/validation/): what accuracy to expect, case by case.
