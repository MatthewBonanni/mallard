"""Reader of the stencil file (PREFIX_stencils.h5 of mallard check --mesh-fields with --stencils all, --cells or
--region, or [output] write_stencils; needs h5py): every target cell's stencils, member by member.

    from mallard.stencils import read_stencils
    s = read_stencils("mesh_stencils.h5")
    st = s.stencil(120)        # the stencils of cell 120 (global id)
    st["cell"][st["stencil"] == 0]    # its central stencil's cells
"""
import h5py
import numpy as np

# Newest format version (the root attribute "version") this reader knows
STENCIL_FORMAT_VERSION = 1

# Member roles (bits of "role")
MIRROR = 1    # taken across the boundary face face_cell/face: a mirror image, or a boundary's ghost or image
PERIODIC = 2  # a periodic image, across a periodic seam

# What a cell's stencils reach (bits of "reach")
WALL = 1
SEAM = 2


class Stencils:
    """The arrays of a stencil file, by target and by member, and each target's stencils."""

    def __init__(self, path):
        with h5py.File(path, "r") as f:
            if f.attrs["format"][0] not in (b"mallard-stencils", "mallard-stencils"):
                raise ValueError(f"{path} is not a Mallard stencil file")
            version = int(f.attrs["version"])
            if version > STENCIL_FORMAT_VERSION:
                raise ValueError(f"{path} is a stencil file of format version {version}, written by a newer "
                                 f"Mallard; this script reads versions up to {STENCIL_FORMAT_VERSION}: update tools/")
            scheme = f.attrs["scheme"][0]
            self.scheme = scheme.decode() if isinstance(scheme, bytes) else str(scheme)
            self.dimension = int(f.attrs["dimension"])
            self.n_cells = int(f.attrs["n_cells"])
            g = f["stencils"]
            self.cells = g["cells"][...]
            self.offsets = g["offsets"][...]
            self.degree = g["degree"][...]
            self.sizes = g["sizes"][...].reshape(len(self.cells), -1)
            self.condition = g["condition"][...]
            self.reach = g["reach"][...]
            m = g["members"]
            self.members = {name: m[name][...] for name in m}
            self.rank = f["partition/rank"][...]
            self.member_rank = f["partition/member_rank"][...]
        self._row = {int(c): t for t, c in enumerate(self.cells)}

    def __contains__(self, cell):
        return int(cell) in self._row

    def stencil(self, cell):
        """The stencils of a target cell (global id): its members' cells, stencil (0 central, 1 + k the sector
        stencil of its face k), role, face_cell and face, weights if the file has them, and its degree,
        sizes, condition and reach."""
        t = self._row[int(cell)]
        a, b = self.offsets[t], self.offsets[t + 1]
        out = {name: values[a:b] for name, values in self.members.items()}
        out.update(degree=int(self.degree[t]), sizes=self.sizes[t], condition=float(self.condition[t]),
                   reach=int(self.reach[t]), rank=int(self.rank[t]), member_rank=self.member_rank[a:b])
        return out

    def sectors(self, cell):
        """The sector stencils of a target cell, by face: {k: member cells}."""
        st = self.stencil(cell)
        return {int(k) - 1: st["cell"][st["stencil"] == k] for k in np.unique(st["stencil"]) if k > 0}


def read_stencils(path):
    """The stencil file at path."""
    return Stencils(path)
