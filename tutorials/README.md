# Tutorials

The files of the step-by-step tutorials in [`docs/tutorials/`](../docs/tutorials/index.md)
(on the website: *Tutorials*): one directory per tutorial with its Gmsh script (if any),
input file and the Python scripts that set it up and draw its figures, and
`common/` with what they share (figure style, mesh and snapshot readers).

| Directory | Tutorial |
|---|---|
| `wedge/` | [Oblique shock on a ramp](../docs/tutorials/wedge.md) (2D, generated mesh) against the θ-β-M relation |
| `cylinder/` | [Vortex shedding behind a cylinder](../docs/tutorials/cylinder.md) (2D, Gmsh) at Re = 100 against reference computations |
| `cone/` | [Supersonic flow over a cone](../docs/tutorials/cone.md) (3D) against the Taylor-Maccoll solution |
| `channel_les/` | [Channel LES](../docs/tutorials/channel_les.md): turbulent channel at Re_τ = 180 against DNS |

`ci.sh` runs every tutorial at reduced size on two MPI ranks, as CI does, with the steps in each
tutorial's own `ci.sh`, through the `mallard` of an install with both dimensions:

```sh
cmake --install build --prefix install && cmake --install build-3d --prefix install
tutorials/ci.sh install/bin/mallard /tmp/tutorials
```

Python packages: `pip install -r tutorials/requirements.txt` (which also
installs the `gmsh` command). The figure scripts write light and dark SVGs to
`docs/images/tutorials/`, or to the directory given with `--out`.
