"""Runs a small periodic box with each given mallard executable and checks that tools/mallard/restart.py and the
tools built on it read the restart files it writes (uncompressed and compressed), so that a new restart format
version fails here until the reader learns it:

- read: compressed and uncompressed files hold the same values; written back by restart.write, a file is bitwise
  the original for mallard diff;
- tools/setup/tile_restart.py, and (3D) tools/setup/interpolate_restart.py onto the same mesh, whose output
  mallard reads;
- tools/post/insitu_statistics.py spectra, which matches the solver's [statistics.spectra] of the same sample.

    python3 tools/dev/check_restart_tools.py build/src/mallard [build-3d/src/mallard]
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "tools"))
from mallard import restart  # noqa: E402

N = (8, 6, 4)
L = (2 * np.pi, 3.0, 2.0)


def case(dim):
    z = " + 0.4 * z" if dim == 3 else ""
    mesh = "".join(f"N{a} = {n}\nL{a} = {length!r}\n" for a, n, length in zip("xyz", N[:dim], L[:dim]))
    periodic = "".join(f'[[periodic]]\nzones = ["{a}", "{b}"]\n'
                       for a, b in (("left", "right"), ("bottom", "top"), ("back", "front"))[:dim])
    u = ['"0.3 * sin(x + 0.5) * cos(1.3 * y' + z + ')"', '"0.2 * cos(x - y)"', '"0.1 * sin(z + x)"'][:dim]
    return f"""[run]
n_steps = 4
cfl = 0.4
[mesh]
type = "cartesian"
{mesh}{periodic}[initialize]
type = "analytical"
rho = "1.0 + 0.1 * sin(x + 2 * y{z})"
u = [{", ".join(u)}]
p = "1.0 + 0.05 * cos(x - y{z})"
[physics]
type = "navier_stokes"
gamma = 1.4
p_ref = 1.0
T_ref = 1.0
rho_ref = 1.0
mu = 0.01
Pr = 0.72
[statistics]
fields = ["U_X", "P"]
interval = 4
[statistics.spectra]
fields = ["U_X", "P"]
products = ["U_X*P"]
[[write_data]]
prefix = "plain"
format = "restart"
interval = 4
[[write_data]]
prefix = "zlib"
format = "restart"
compression = "zlib"
chunk_cells = 20
interval = 4
"""


def gmsh_box(path, dim):
    """The generated cartesian box of case(3) as an ASCII Gmsh 2.2 mesh of hexahedra."""
    n = N[:dim]
    axes = [np.linspace(0.0, length, m + 1) for length, m in zip(L, n)]
    node = np.arange(np.prod([m + 1 for m in n])).reshape([m + 1 for m in n]) + 1
    x, y, z = np.meshgrid(*axes, indexing="ij")
    with open(path, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n")
        f.write(f"{node.size}\n")
        for i, p in zip(node.ravel(), zip(x.ravel(), y.ravel(), z.ravel())):
            f.write(f"{i} {p[0]!r} {p[1]!r} {p[2]!r}\n")
        f.write(f"$EndNodes\n$Elements\n{np.prod(n)}\n")
        e = 0
        for i in range(n[0]):
            for j in range(n[1]):
                for k in range(n[2]):
                    e += 1
                    v = [node[i, j, k], node[i + 1, j, k], node[i + 1, j + 1, k], node[i, j + 1, k],
                         node[i, j, k + 1], node[i + 1, j, k + 1], node[i + 1, j + 1, k + 1], node[i, j + 1, k + 1]]
                    f.write(f"{e} 5 2 1 1 {' '.join(map(str, v))}\n")
        f.write("$EndElements\n")


def check(mallard):
    dim = int(os.path.basename(os.path.realpath(mallard))[len("mallard")])
    with tempfile.TemporaryDirectory() as work:
        def path(name):
            return os.path.join(work, name)

        def run(*args, **kwargs):
            return subprocess.run([*map(str, args)], cwd=work, check=True, capture_output=True, text=True, **kwargs)

        def tool(name, *args):
            return run(sys.executable, os.path.join(ROOT, "tools", name), *args)

        with open(path("input.toml"), "w") as f:
            f.write(case(dim))
        run(mallard, "-i", "input.toml", "--kokkos-num-threads=2")
        plain = restart.read(path("plain_000004.restart"))
        zlib = restart.read(path("zlib_000004.restart"))
        assert zlib.chunk_cells == 20 and plain.chunk_cells == 0, (zlib.chunk_cells, plain.chunk_cells)
        assert plain.n_cells == np.prod(N[:dim]) and plain.step == 4 and plain.arrays and plain.attributes
        assert (plain.names, plain.t, plain.attributes) == (zlib.names, zlib.t, zlib.attributes)
        assert np.array_equal(plain.values, zlib.values)
        assert all(np.array_equal(plain.arrays[k], zlib.arrays[k]) for k in plain.arrays)

        restart.write(path("rewritten.restart"), plain.fields, plain.step, plain.t, plain.attributes, plain.arrays,
                      plain.real_size)
        run(mallard, "diff", "plain_000004.restart", "rewritten.restart")

        tool("setup/tile_restart.py", "zlib_000004.restart", N[0], 2, "tiled.restart")
        tiled = restart.read(path("tiled.restart"))
        assert tiled.names == [n for n in plain.names if n.startswith("RHO")]
        assert all(np.array_equal(tiled[n], np.tile(plain[n], 2)) for n in tiled.names)
        run(mallard, "info", "tiled.restart")

        if dim == 3:
            gmsh_box(path("box.msh"), dim)
            tool("setup/interpolate_restart.py", "box.msh", "zlib_000004.restart", "box.msh", "moved.restart")
            moved = restart.read(path("moved.restart"))
            assert (moved.names, moved.step, moved.t, moved.attributes) == \
                (plain.names, plain.step, plain.t, plain.attributes)
            assert np.allclose(moved.values, plain.values, rtol=1e-12, atol=0.0)
            run(mallard, "info", "moved.restart")

        tool("post/insitu_statistics.py", "spectra", "input.toml", "zlib_000004.restart", "-o", "from_restart.csv")
        tool("post/insitu_statistics.py", "compare", "spectra.csv", "from_restart.csv", "--tol", "1e-10")
    print(f"restart files of {mallard} ({dim}D, version {plain.version}): read by tools/mallard/restart.py, "
          f"tile_restart.py{', interpolate_restart.py' if dim == 3 else ''} and insitu_statistics.py")


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for mallard in sys.argv[1:]:
        check(os.path.abspath(mallard))


if __name__ == "__main__":
    main()
