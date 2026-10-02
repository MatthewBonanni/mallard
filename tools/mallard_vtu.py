"""Minimal reader for Mallard's raw-appended VTU files."""
import re

import numpy as np

_DTYPES = {"Float32": "<f4", "Float64": "<f8", "UInt32": "<u4", "Int32": "<i4",
           "Int64": "<i8", "UInt64": "<u8", "UInt8": "u1"}


def read_vtu(path):
    raw = open(path, "rb").read()
    start = raw.index(b"<AppendedData")
    start = raw.index(b"_", start) + 1
    header = raw[:start].decode("latin-1")
    hdr_type = re.search(r'header_type="(\w+)"', header)
    hdr_dt = np.dtype(_DTYPES[hdr_type.group(1)] if hdr_type else "<u4")
    arrays = {}
    for m in re.finditer(r"<DataArray([^>]*)>", header):
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
        if "offset" not in attrs:
            continue
        off = start + int(attrs["offset"])
        nbytes = int(np.frombuffer(raw, hdr_dt, 1, off)[0])
        dt = np.dtype(_DTYPES[attrs["type"]])
        data = np.frombuffer(raw, dt, nbytes // dt.itemsize, off + hdr_dt.itemsize)
        ncomp = int(attrs.get("NumberOfComponents", 1))
        if ncomp > 1:
            data = data.reshape(-1, ncomp)
        arrays[attrs.get("Name", "Points")] = data
    conn, offs = arrays.pop("connectivity"), arrays.pop("offsets")
    arrays.pop("types", None)
    pts = arrays.pop("Points")[:, :2].astype(float)
    starts = np.concatenate([[0], offs[:-1]])
    sizes = offs - starts
    tri_parts, idx_parts = [], []
    for n in np.unique(sizes):
        sel = np.nonzero(sizes == n)[0]
        cells = conn[starts[sel, None] + np.arange(n)]
        for k in range(1, n - 1):
            tri_parts.append(cells[:, [0, k, k + 1]])
            idx_parts.append(sel)
    tris = np.vstack(tri_parts).astype(np.int64)
    tri_cell = np.concatenate(idx_parts)
    tm = re.search(r'Name="TIME"[^>]*>([^<]*)<', header)
    if tm:
        arrays["TIME"] = float(tm.group(1))
    return pts, tris, tri_cell, {k: (v if k == 'TIME' else np.asarray(v, dtype=float)) for k, v in arrays.items()}
