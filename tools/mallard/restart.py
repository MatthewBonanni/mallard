"""Reader and writer of Mallard restart files (docs/guide/output.md#formats-and-compatibility).

read(path) returns a Restart: the header (version, step, t, names, attributes, arrays) and the values of every
variable over all cells in the mesh's global order. Compressed files (compression = "zlib") are read too; the
faces of characteristic boundaries and the trailers are not. write(path, fields, ...) writes the lowest version
that holds what it is given, which every Mallard since 0.6 reads.
"""
import struct
import zlib
from dataclasses import dataclass, field

import numpy as np

MAGIC = b"MALLARD-RESTART\0"
VERSIONS = range(2, 8)
FACE_WIDTH = 2  # values per characteristic boundary face (RestartFaces::WIDTH)


@dataclass
class Restart:
    path: str
    version: int
    real_size: int
    n_cells: int
    step: int
    t: float
    names: list
    attributes: dict = field(default_factory=dict)
    arrays: dict = field(default_factory=dict)
    chunk_cells: int = 0
    data_offset: int = 0
    values: np.ndarray = None

    @property
    def dtype(self):
        return np.dtype("<f8" if self.real_size == 8 else "<f4")

    @property
    def fields(self):
        """{name: values over the cells} (views of values)."""
        return dict(zip(self.names, self.values))

    def __getitem__(self, name):
        return self.values[self.names.index(name)]


def read_header(path):
    """The Restart of path without its values; data_offset is where they start (uncompressed files)."""
    with open(path, "rb") as f:
        if f.read(16) != MAGIC:
            raise SystemExit(f"{path}: not a Mallard restart file")
        version, real_size, n_cells, n_vars, step, t = struct.unpack("<IIQQQd", f.read(40))
        if version not in VERSIONS:
            raise SystemExit(f"{path}: restart file of version {version}; tools/mallard/restart.py reads versions "
                             f"{VERSIONS.start}-{VERSIONS.stop - 1}")
        if real_size not in (4, 8):
            raise SystemExit(f"{path}: real size {real_size}")

        def name():
            (length,) = struct.unpack("<I", f.read(4))
            return f.read(length).decode()

        names = [name() for _ in range(n_vars)]
        attributes, arrays, chunk_cells = {}, {}, 0
        if version >= 3:
            (n,) = struct.unpack("<Q", f.read(8))
            for _ in range(n):
                key = name()
                (attributes[key],) = struct.unpack("<d", f.read(8))
        if version >= 4:
            f.read(12)  # face count and values per face
        if version >= 5:
            _, chunk_cells = struct.unpack("<IQ", f.read(12))
        if version >= 6:
            (n,) = struct.unpack("<Q", f.read(8))
            for _ in range(n):
                key = name()
                (length,) = struct.unpack("<Q", f.read(8))
                arrays[key] = np.frombuffer(f.read(8 * length), dtype="<f8")
        return Restart(str(path), version, real_size, n_cells, step, t, names, attributes, arrays, chunk_cells,
                       f.tell())


def read(path):
    """The Restart of path with its values, an array (variables, cells) of the file's precision."""
    r = read_header(path)
    n_vars, dtype = len(r.names), r.dtype
    with open(path, "rb") as f:
        f.seek(r.data_offset)
        if r.chunk_cells == 0:
            r.values = np.fromfile(f, dtype=dtype, count=n_vars * r.n_cells).reshape(n_vars, r.n_cells)
            return r
        n_chunks = -(-r.n_cells // r.chunk_cells)
        offsets = np.fromfile(f, dtype="<u8", count=n_vars * n_chunks + 1)
        r.values = np.empty((n_vars, r.n_cells), dtype=dtype)
        for k in range(n_vars * n_chunks):
            f.seek(int(offsets[k]))
            raw = zlib.decompress(f.read(int(offsets[k + 1] - offsets[k])))
            v, c = divmod(k, n_chunks)
            first = c * r.chunk_cells
            n = min(r.chunk_cells, r.n_cells - first)
            # Chunks are byte-shuffled: byte b of every value, then byte b + 1
            shuffled = np.frombuffer(raw, dtype=np.uint8).reshape(r.real_size, n)
            r.values[v, first:first + n] = np.ascontiguousarray(shuffled.T).view(dtype).ravel()
    return r


def memmap(path, mode="r"):
    """(Restart without values, its values as a writable or read-only memmap (variables, cells)) of an
    uncompressed file, to change a large file in place."""
    r = read_header(path)
    if r.chunk_cells:
        raise SystemExit(f"{path}: compressed restart file; write it with compression = \"none\"")
    return r, np.memmap(path, dtype=r.dtype, mode=mode, offset=r.data_offset, shape=(len(r.names), r.n_cells))


def write(path, fields, step=0, t=0.0, attributes=None, arrays=None, real_size=8):
    """Write {name: values} (each the same number of cells, in the mesh's global order) as a restart file of
    version 2, or 3 with attributes, or 6 with arrays ({name: float64 values})."""
    names = list(fields)
    n_cells = np.asarray(fields[names[0]]).size
    attributes, arrays = attributes or {}, arrays or {}
    version = 6 if arrays else 3 if attributes else 2
    dtype = "<f8" if real_size == 8 else "<f4"

    def name(s):
        return struct.pack("<I", len(s)) + s.encode()

    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<IIQQQd", version, real_size, n_cells, len(names), step, t))
        f.write(b"".join(name(n) for n in names))
        if version >= 3:
            f.write(struct.pack("<Q", len(attributes)))
            f.write(b"".join(name(k) + struct.pack("<d", v) for k, v in attributes.items()))
        if version >= 6:
            f.write(struct.pack("<QIIQQ", 0, FACE_WIDTH, 0, 0, len(arrays)))
            for k, v in arrays.items():
                v = np.ascontiguousarray(v, dtype="<f8").ravel()
                f.write(name(k) + struct.pack("<Q", v.size) + v.tobytes())
        for n in names:
            values = np.ascontiguousarray(fields[n], dtype=dtype).ravel()
            if values.size != n_cells:
                raise ValueError(f"{n}: {values.size} values, {n_cells} cells")
            f.write(values.tobytes())
