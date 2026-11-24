#!/usr/bin/env python3
"""Inputs of a strong and weak scaling study of the Taylor-Green vortex, from scaling.toml.

    scaling.py --gpus 1 2 4 8 [--strong-n 128] [--weak-n 64] [--out .] [--mallard BUILD/src/mallard]

Writes strong/nG/input.toml (scaling.toml's box of strong-n^3 hexahedra) and
weak/nG/input.toml (weak-n^3 hexahedra per GPU: Nx = weak-n G, Lx = 2 pi G)
for each GPU count G, and prints the command that runs each one. Run them one
after the other, so that no two share a GPU; then

    python ../../tools/docs/scaling.py --strong strong/n* --weak weak/n* --out scaling.json

reduces their run logs (mallard.jsonl) to times per step and efficiencies.
"""
import argparse
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def box(template, nx, ny, nz):
    text = template
    for key, n in (("Nx", nx), ("Ny", ny), ("Nz", nz)):
        text, count = re.subn(rf"^{key} = \d+$", f"{key} = {n}", text, count=1, flags=re.M)
        assert count == 1, key
    for key, n in (("Lx", nx), ("Ly", ny), ("Lz", nz)):
        text, count = re.subn(rf"^{key} = [\d.]+$", f"{key} = {2 * math.pi * n / ny!r}", text, count=1, flags=re.M)
        assert count == 1, key
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gpus", type=int, nargs="+", required=True)
    ap.add_argument("--strong-n", type=int, default=128, help="cells per side of the strong-scaling box")
    ap.add_argument("--weak-n", type=int, default=64, help="cells per side per GPU of the weak-scaling boxes")
    ap.add_argument("--out", type=Path, default=Path("."))
    ap.add_argument("--mallard", type=Path, default=HERE.parents[1] / "build3d" / "src" / "mallard")
    args = ap.parse_args()
    template = (HERE / "scaling.toml").read_text()
    template = template[template.index("[run]"):]
    mallard = args.mallard.resolve()
    for series, n_of in (("strong", lambda g: (args.strong_n,) * 3),
                         ("weak", lambda g: (args.weak_n * g, args.weak_n, args.weak_n))):
        for g in args.gpus:
            d = args.out / series / f"n{g}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "input.toml").write_text(box(template, *n_of(g)))
            print(f"(cd {d} && mpirun -n {g} --bind-to none {mallard} -i input.toml "
                  f"--kokkos-map-device-id-by=mpi_rank)")


if __name__ == "__main__":
    main()
