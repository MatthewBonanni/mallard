#!/usr/bin/env python3
"""Moving window for detonation runs on a generated cartesian box, between restarts.

    detonation_window.py run INPUT --keep BEHIND --segment DT --t-end T [--frames DIR] -- MALLARD COMMAND...
    detonation_window.py shift INPUT RESTART --keep BEHIND
    detonation_window.py frame INPUT RESTART OUT.npz [--depth D]
    detonation_window.py extrude INPUT_2D RESTART_2D NZ OUT.restart [--pocket BEHIND Y Z R]...

A detonation runs to +x through fresh gas at rest, and only the front, its
reaction zone and a stretch of the flow behind it matter. `shift` moves the
box along with the front: it finds the front (the last cross-section in x
whose largest density exceeds 1.5 times that of the fresh gas in the last
one), drops the columns more than BEHIND meters behind it and appends as
many copies of the last (fresh) column, in place, so the front sits BEHIND
from the left end again. The transmissive left boundary then cuts the burnt
gas BEHIND behind the front. Before the columns go, the P_MAX of their cells
next to the side walls (y = 0, y = LY and, in 3D, z = 0, z = LZ) is saved as
the next numbered chunk of the soot foils in foil/, and window.json (both
next to INPUT) records the total shift, i.e. where the box's left end is in
the duct, and each chunk. The step and time of the restart are kept.

`run` alternates Mallard runs and shifts until T: INPUT's `[run] t_stop` is
set to the end of each segment of DT, the command runs (it must read INPUT,
start from `[initialize] file` and write restart files at least at the
segment ends, e.g. `time_interval` = DT or a divisor), every new restart file
in the restart directory becomes a frame (`frame`, written to DIR) and is
deleted, except the last, which is shifted and becomes the next segment's
initial state (`[initialize] file`, overwritten). Rerunning the same command
resumes after the last completed segment.

`frame` writes an animation frame: the pressure (kPa, float16) in the D
meters (default 0.03) behind the front at full resolution, and P_MAX on the
side walls over the whole box, with the shift, step and time.

`extrude` turns a 2D restart of an NX x NY box into a 3D one of NX x NY x
NZ, uniform in z (zero z momentum, LZ = LY / NY * NZ), e.g. to start a 3D
run from a developed 2D cellular detonation; INPUT_2D is the 2D run's input.
Each --pocket fills a sphere of radius R centered BEHIND meters behind the
front at (Y, Z) with the fresh gas of the last column, at rest, to start
transverse waves in z.

Cells are numbered as Mallard's generated boxes number them, z fastest, then
y: c = (i NY + j) NZ + k. Double-precision restart files of version 2 or 3.
`foils(INPUT)` (import this module) assembles the soot foils of the whole
duct so far. Needs only numpy (Python 3.11 or later).
"""
import argparse
import glob
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tomllib

import numpy as np

ATOMIC_WEIGHTS = {"H": 1.008, "O": 15.999, "N": 14.007, "C": 12.011, "Ar": 39.95, "He": 4.002602}
GAS_CONSTANT = 8314.462618  # J / (kmol K)


class Case:
    """The box of a Mallard input file with a generated cartesian mesh."""

    def __init__(self, input_path):
        self.input = os.path.abspath(input_path)
        self.dir = os.path.dirname(self.input)
        with open(self.input, "rb") as f:
            self.toml = tomllib.load(f)
        m = self.toml["mesh"]
        self.shape = (int(m["Nx"]), int(m["Ny"]), int(m.get("Nz", 1)))
        self.size = (float(m["Lx"]), float(m["Ly"]), float(m.get("Lz", 1.0)))
        self.dx = self.size[0] / self.shape[0]
        self.mechanism = os.path.join(self.dir, self.toml["physics"]["mechanism"])
        self.state_file = os.path.join(self.dir, "window.json")
        self.foil_dir = os.path.join(self.dir, "foil")

    def path(self, p):
        return os.path.join(self.dir, p)

    def load_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file) as f:
                return json.load(f)
        return {"shift_cells": 0, "chunks": [], "segments": [], "frames": 0}

    def save_state(self, state):
        state["dx"] = self.dx
        tmp = self.state_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=1)
        os.replace(tmp, self.state_file)


def read_header(f):
    magic = f.read(16)
    version, real_size = struct.unpack("<II", f.read(8))
    n_cells, n_vars, step = struct.unpack("<QQQ", f.read(24))
    (t,) = struct.unpack("<d", f.read(8))
    if not magic.startswith(b"MALLARD-RESTART") or version not in (2, 3) or real_size != 8:
        raise SystemExit("expected a double-precision Mallard restart file of version 2 or 3")

    def name():
        (length,) = struct.unpack("<I", f.read(4))
        return f.read(length).decode()

    names = [name() for _ in range(n_vars)]
    if version == 3:
        (n_attributes,) = struct.unpack("<Q", f.read(8))
        for _ in range(n_attributes):
            name()
            f.read(8)
    return {"version": version, "n_cells": n_cells, "names": names, "step": step, "t": t, "offset": f.tell()}


def open_restart(path, shape, mode="r"):
    with open(path, "rb") as f:
        h = read_header(f)
    if h["n_cells"] != np.prod(shape):
        raise SystemExit(f"{path}: {h['n_cells']} cells, not {' x '.join(map(str, shape))}")
    data = np.memmap(path, dtype="<f8", mode=mode, offset=h["offset"], shape=(len(h["names"]),) + tuple(shape))
    return h, data


def front_index(rho):
    """Last x index whose largest density exceeds 1.5 times the fresh gas's (the last column's)."""
    rho0 = rho[-1].mean()
    shocked = np.nonzero(rho.reshape(rho.shape[0], -1).max(axis=1) > 1.5 * rho0)[0]
    if shocked.size == 0:
        raise SystemExit("no detonation front in the box")
    return int(shocked[-1])


def wall_strips(p_max):
    """P_MAX next to the side walls, (x cells, transverse cells) each: y walls, then z walls in 3D."""
    strips = {"bottom": p_max[:, 0, :], "top": p_max[:, -1, :]}
    if p_max.shape[2] > 1:
        strips.update({"back": p_max[:, :, 0], "front": p_max[:, :, -1]})
    return strips


def shift(case, restart, keep, state):
    """Shift RESTART in place to keep KEEP meters behind the front; returns the shift in cells."""
    h, data = open_restart(restart, case.shape, mode="r+")
    i_front = front_index(data[h["names"].index("RHO")])
    s = i_front - int(round(keep / case.dx))
    if s < 1:
        print(f"front at x = {(state['shift_cells'] + i_front + 0.5) * case.dx:.4f} m in the duct: no shift")
        return 0
    if "P_MAX" in h["names"]:
        os.makedirs(case.foil_dir, exist_ok=True)
        chunk = f"chunk_{len(state['chunks']):04d}.npz"
        np.savez(os.path.join(case.foil_dir, chunk), **wall_strips(np.asarray(data[h["names"].index("P_MAX"), :s])))
        state["chunks"].append({"file": chunk, "first_cell": state["shift_cells"], "cells": s,
                                "step": h["step"], "t": h["t"]})
    for v in range(len(h["names"])):
        field = data[v]
        field[:-s] = field[s:].copy()
        field[-s:] = field[-1 - s]
    data.flush()
    del data
    state["shift_cells"] += s
    print(f"step {h['step']}, t = {h['t']:.6e}: front at cell {i_front}, shifted by {s} cells; "
          f"box starts at x = {state['shift_cells'] * case.dx:.4f} m in the duct", flush=True)
    return s


def molecular_weights(mechanism):
    """Species molecular weights [kg/kmol] from the composition maps of a Cantera YAML file (no Cantera needed)."""
    weights, name = {}, None
    with open(mechanism) as f:
        for line in f:
            if line.strip().startswith("- name:"):
                name = line.split(":", 1)[1].strip()
            elif name and line.strip().startswith("composition:"):
                comp = line.split("{", 1)[1].split("}", 1)[0]
                weights[name] = sum(ATOMIC_WEIGHTS[e.split(":")[0].strip()] * float(e.split(":")[1])
                                    for e in comp.split(","))
                name = None
    return weights


def pressure(h, data, i0, i1, mechanism):
    """Pressure of columns i0:i1 from the partial densities and T_SEED (the temperature of the last state)."""
    names = h["names"]
    weights = molecular_weights(mechanism)
    rho_over_w = np.zeros((i1 - i0,) + data.shape[2:])
    for name in names:
        if name.startswith("RHOY_"):
            rho_over_w += np.asarray(data[names.index(name), i0:i1]) / weights[name[5:]]
    return rho_over_w * GAS_CONSTANT * np.asarray(data[names.index("T_SEED"), i0:i1])


def frame(case, restart, out, depth, state):
    h, data = open_restart(restart, case.shape)
    i_front = front_index(data[h["names"].index("RHO")])
    i0 = max(0, i_front + 3 - int(round(depth / case.dx)))
    i1 = min(case.shape[0], i_front + 4)
    p = pressure(h, data, i0, i1, case.mechanism)
    result = {"p_kpa": (p / 1000).astype(np.float16), "i0": i0, "shift_cells": state["shift_cells"],
              "dx": case.dx, "t": h["t"], "step": h["step"], "i_front": i_front, "size": case.size}
    if "P_MAX" in h["names"]:
        for wall, strip in wall_strips(np.asarray(data[h["names"].index("P_MAX")])).items():
            result["pmax_" + wall] = strip.astype(np.float32)
    np.savez(out, **result)
    print(f"{out}: t = {h['t']:.6e}, front at x = {(state['shift_cells'] + i_front + 0.5) * case.dx:.4f} m",
          flush=True)


def foils(input_path, restart=None):
    """The wall soot foils of the whole duct so far, {wall: (x cells, transverse cells)}, and dx."""
    case = Case(input_path)
    state = case.load_state()
    parts = {}
    for c in state["chunks"]:
        with np.load(os.path.join(case.foil_dir, c["file"])) as z:
            for w in z.files:
                parts.setdefault(w, []).append(z[w])
    restart = restart or case.path(case.toml["initialize"]["file"])
    h, data = open_restart(restart, case.shape)
    for w, strip in wall_strips(np.asarray(data[h["names"].index("P_MAX")])).items():
        parts.setdefault(w, []).append(strip)
    return {w: np.concatenate(p) for w, p in parts.items()}, case.dx


def extrude(input_2d, restart_2d, nz, out, pockets):
    case = Case(input_2d)
    with open(restart_2d, "rb") as f:
        h = read_header(f)
    nx, ny = case.shape[:2]
    _, data = open_restart(restart_2d, (nx, ny, 1))
    names = list(h["names"])
    names.insert(names.index("RHOU_Y") + 1, "RHOU_Z")
    i_front = front_index(data[h["names"].index("RHO")])
    h_cell = case.size[1] / ny
    x = (np.arange(nx) + 0.5) * case.dx
    y = (np.arange(ny) + 0.5) * h_cell
    z = (np.arange(nz) + 0.5) * h_cell
    inside = np.zeros((nx, ny, nz), dtype=bool)
    for behind, yc, zc, r in pockets:
        xc = (i_front + 0.5) * case.dx - behind
        inside |= ((x[:, None, None] - xc) ** 2 + (y[None, :, None] - yc) ** 2 + (z[None, None, :] - zc) ** 2
                   < r ** 2)
    with open(out, "wb") as f:
        f.write(b"MALLARD-RESTART\0")
        f.write(struct.pack("<IIQQQd", 2, 8, nx * ny * nz, len(names), h["step"], h["t"]))
        for name in names:
            f.write(struct.pack("<I", len(name)) + name.encode())
        for name in names:
            if name == "RHOU_Z":
                f.write(np.zeros(nx * ny * nz, dtype="<f8").tobytes())
                continue
            column = np.asarray(data[h["names"].index(name), :, :, 0])
            field = np.repeat(column[:, :, None], nz, axis=2)
            if name not in ("CHEM_H", "P_MAX"):
                fresh = 0.0 if name.startswith("RHOU_") else column[-1].mean()
                field[inside] = fresh
            field.astype("<f8").tofile(f)
    print(f"wrote {out}: {nx} x {ny} x {nz} cells at t = {h['t']:.6e}, LZ = {h_cell * nz} m; "
          f"{inside.sum()} cells of fresh gas in {len(pockets)} pockets")


def set_toml_value(text, table, key, value):
    """Replace KEY's value in [TABLE] of a TOML text, keeping everything else."""
    pattern = re.compile(rf"(^\[{re.escape(table)}\][^\[]*?^{key}\s*=\s*)([^\n]*)", re.M | re.S)
    if not pattern.search(text):
        raise SystemExit(f"no {key} in [{table}]")
    return pattern.sub(lambda m: m.group(1) + value, text, count=1)


def restart_dir(case):
    for w in case.toml.get("write_data", []):
        if w.get("format") == "restart":
            return case.path(os.path.dirname(w["prefix"]) or ".")
    raise SystemExit("the input writes no restart files")


def run(case, keep, segment, t_end, frames_dir, depth, command):
    state = case.load_state()
    init = case.path(case.toml["initialize"]["file"])
    rdir = restart_dir(case)
    if frames_dir:
        os.makedirs(frames_dir, exist_ok=True)
    with open(init, "rb") as f:
        t = read_header(f)["t"]
    while t < t_end * (1 - 1e-9):
        t_next = min(t_end, segment * (round(t / segment) + 1))
        with open(case.input) as f:
            text = f.read()
        with open(case.input, "w") as f:
            f.write(set_toml_value(text, "run", "t_stop", f"{t_next:.10e}"))
        for old in glob.glob(os.path.join(rdir, "*.restart")):
            os.remove(old)
        print(f"segment t = {t:.6e} -> {t_next:.6e}: {' '.join(command)}", flush=True)
        rc = subprocess.call(command, cwd=case.dir)
        restarts = sorted(glob.glob(os.path.join(rdir, "*.restart")))
        if rc != 0 or not restarts:
            raise SystemExit(f"Mallard exited with {rc}; resume by rerunning this command")
        with open(restarts[-1], "rb") as f:
            t_last = read_header(f)["t"]
        if abs(t_last - t_next) > 1e-9 * t_next:
            raise SystemExit(f"the last restart is at t = {t_last}, not the segment end {t_next}")
        for r in restarts:
            if frames_dir:
                frame(case, r, os.path.join(frames_dir, f"frame_{state['frames']:05d}.npz"), depth, state)
                state["frames"] += 1
            if r != restarts[-1]:
                os.remove(r)
        shift(case, restarts[-1], keep, state)
        tmp = init + ".tmp"
        shutil.move(restarts[-1], tmp)
        os.replace(tmp, init)
        state["segments"].append({"t": t_next, "shift_cells": state["shift_cells"]})
        case.save_state(state)
        t = t_next


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("input")
    p.add_argument("--keep", type=float, required=True)
    p.add_argument("--segment", type=float, required=True)
    p.add_argument("--t-end", type=float, required=True)
    p.add_argument("--frames")
    p.add_argument("--depth", type=float, default=0.03)
    p = sub.add_parser("shift")
    p.add_argument("input")
    p.add_argument("restart")
    p.add_argument("--keep", type=float, required=True)
    p = sub.add_parser("frame")
    p.add_argument("input")
    p.add_argument("restart")
    p.add_argument("out")
    p.add_argument("--depth", type=float, default=0.03)
    p = sub.add_parser("extrude")
    p.add_argument("input")
    p.add_argument("restart")
    p.add_argument("nz", type=int)
    p.add_argument("out")
    p.add_argument("--pocket", type=float, nargs=4, action="append", default=[])
    argv = sys.argv[1:]
    command = argv[argv.index("--") + 1:] if "--" in argv else []
    args = ap.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
    if args.cmd == "extrude":
        extrude(args.input, args.restart, args.nz, args.out, args.pocket)
        return
    case = Case(args.input)
    state = case.load_state()
    if args.cmd == "run":
        if not command:
            sys.exit("run needs the Mallard command after --")
        run(case, args.keep, args.segment, args.t_end, args.frames, args.depth, command)
    elif args.cmd == "shift":
        shift(case, args.restart, args.keep, state)
        case.save_state(state)
    else:
        frame(case, args.restart, args.out, args.depth, state)


if __name__ == "__main__":
    main()
