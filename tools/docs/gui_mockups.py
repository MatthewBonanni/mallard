"""Mockups of three screens of the Mallard GUI (docs/design/gui.md): case setup, run
monitor and data manager, each in the GUI's light and dark theme, as SVG with real text
in docs/images/design/gui-<screen>-<theme>.svg.

    python tools/docs/gui_mockups.py

Needs only the standard library. The tokens below are the GUI's design language
(docs/design/gui.md#design-language); the shared token file replaces them once it exists.
"""
import math
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "docs" / "images" / "design"
W, H = 1280, 800
SANS = "Inter, 'Helvetica Neue', Arial, sans-serif"
MONO = "'JetBrains Mono', 'SF Mono', Menlo, monospace"

LIGHT = dict(name="light", bg="#f7f7f8", surface="#ffffff", raised="#fbfbfc", border="#e4e4e7", fg="#18181b",
             muted="#71717a", faint="#a1a1aa", accent="#0f766e", accent_soft="#ccfbf1", error="#b42318",
             error_soft="#fee4e2", warn="#b54708", warn_soft="#fef0c7", ok="#3f6212", mesh="#d4d4d8",
             zones=["#0072b2", "#e69f00", "#009e73", "#cc79a7", "#56b4e9"])
DARK = dict(name="dark", bg="#0e0f11", surface="#16171a", raised="#1c1d21", border="#2a2b30", fg="#ececee",
            muted="#9b9ba3", faint="#62626a", accent="#2dd4bf", accent_soft="#123a36", error="#f97066",
            error_soft="#3b1514", warn="#fdb022", warn_soft="#3a2a0c", ok="#a3e635", mesh="#3a3b41",
            zones=["#56b4e9", "#e69f00", "#009e73", "#cc79a7", "#f0e442"])


class Svg:
    def __init__(self, P):
        self.P = P
        self.items = []

    def add(self, s):
        self.items.append(s)

    def rect(self, x, y, w, h, fill, stroke=None, r=0, sw=1):
        s = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}"{s}/>')

    def text(self, x, y, t, size=13, fill=None, weight=400, mono=False, anchor="start"):
        fam = MONO if mono else SANS
        feat = ' font-feature-settings="\'tnum\'"' if mono else ""
        t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.add(f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" font-weight="{weight}" '
                 f'fill="{fill or self.P["fg"]}" text-anchor="{anchor}"{feat}>{t}</text>')

    def line(self, x1, y1, x2, y2, stroke, sw=1, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def path(self, d, stroke, sw=1.5, fill="none"):
        self.add(f'<path d="{d}" stroke="{stroke}" stroke-width="{sw}" fill="{fill}" stroke-linejoin="round"/>')

    def circle(self, x, y, r, fill="none", stroke=None, sw=1):
        s = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}"{s}/>')

    def str(self):
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n'
                + "\n".join(self.items) + "\n</svg>\n")


def badge(s, x, y, label, fill, fg):
    w = 8 * len(label) * 0.78 + 16
    s.rect(x, y, w, 20, fill, r=10)
    s.text(x + w / 2, y + 14, label, 11, fg, 500, anchor="middle")
    return w


def chrome(s, title, location, tab):
    """The window: top bar with project, location, actions; left rail of the app's sections."""
    P = s.P
    s.rect(0, 0, W, H, P["bg"])
    s.rect(0, 0, W, 48, P["surface"])
    s.line(0, 48, W, 48, P["border"])
    s.circle(24, 24, 9, P["accent"])
    s.text(42, 29, "cylinder-re100", 14, weight=600)
    s.text(160, 29, "/", 14, P["faint"])
    s.text(174, 29, title, 14, P["muted"])
    badge(s, 300, 14, location, P["accent_soft"], P["accent"])
    s.rect(W - 380, 12, 220, 24, P["raised"], P["border"], r=6)
    s.text(W - 368, 28, "Search keys, runs, actions", 12, P["faint"])
    s.text(W - 172, 28, "Ctrl K", 11, P["faint"], mono=True, anchor="end")
    s.rect(0, 48, 56, H - 48, P["surface"])
    s.line(56, 48, 56, H, P["border"])
    for i, name in enumerate(["Case", "Runs", "Data", "View", "Remotes"]):
        y = 76 + 56 * i
        active = name == tab
        if active:
            s.rect(8, y - 18, 40, 44, P["accent_soft"], r=8)
        c = P["accent"] if active else P["muted"]
        s.rect(20, y - 8, 16, 16, "none", c, r=4, sw=1.5)
        s.text(28, y + 22, name, 9, c, 500, anchor="middle")


def cylinder_mesh(s, cx, cy, R, r0):
    P = s.P
    for k in range(9):
        r = r0 + (R - r0) * (1.32 ** k - 1) / (1.32 ** 8 - 1)
        s.circle(cx, cy, r, stroke=P["mesh"], sw=0.8)
    for k in range(48):
        a = 2 * math.pi * k / 48
        s.line(cx + r0 * math.cos(a), cy + r0 * math.sin(a), cx + R * math.cos(a), cy + R * math.sin(a), P["mesh"], 0.8)
    s.circle(cx, cy, R, stroke=P["zones"][1], sw=3)
    s.circle(cx, cy, r0, stroke=P["zones"][0], sw=3.5)


def field(s, x, y, w, label, value, unit="", error=None, select=False, focus=False):
    P = s.P
    s.text(x, y, label, 12, P["muted"], 500)
    stroke = P["error"] if error else (P["accent"] if focus else P["border"])
    s.rect(x, y + 6, w, 30, P["raised"], stroke, r=6, sw=1.5 if (error or focus) else 1)
    s.text(x + 10, y + 26, value, 13, mono=not select)
    if unit:
        s.text(x + w - 10, y + 26, unit, 12, P["muted"], mono=True, anchor="end")
    if select:
        s.path(f"M{x + w - 20} {y + 18} l5 5 l5 -5", P["muted"])
    if error:
        s.text(x, y + 52, error, 11.5, P["error"])
        return 74
    return 60


def case_setup(P):
    s = Svg(P)
    chrome(s, "Case setup", "Local", "Case")
    s.rect(56, 48, 220, H - 48, P["surface"])
    s.line(276, 48, 276, H, P["border"])
    s.text(76, 80, "SETUP", 11, P["faint"], 600)
    steps = [("Mesh", "ok"), ("Physics", "ok"), ("Boundaries", "error"), ("Initial state", "ok"),
             ("Numerics", "ok"), ("Output", "warn"), ("Probes & regions", "ok"), ("Check & run", "")]
    for i, (name, st) in enumerate(steps):
        y = 108 + 36 * i
        if name == "Boundaries":
            s.rect(64, y - 18, 204, 30, P["accent_soft"], r=6)
        col = {"ok": P["ok"], "error": P["error"], "warn": P["warn"], "": P["faint"]}[st]
        s.circle(84, y - 3, 4, col)
        s.text(98, y + 1, name, 13, P["accent"] if name == "Boundaries" else P["fg"], 500 if name == "Boundaries" else 400)
        if st == "error":
            s.text(256, y + 1, "1", 11, P["error"], 600, mono=True, anchor="end")
        if st == "warn":
            s.text(256, y + 1, "1", 11, P["warn"], 600, mono=True, anchor="end")
    s.text(76, 420, "BOUNDARY ZONES", 11, P["faint"], 600)
    for i, (z, bc, c) in enumerate([("cylinder", "wall_adiabatic", 0), ("farfield", "farfield", 1)]):
        y = 448 + 30 * i
        s.rect(78, y - 10, 10, 10, P["zones"][c], r=2)
        s.text(96, y, z, 13, mono=True)
        s.text(256, y, bc, 11.5, P["muted"], anchor="end")
    s.text(76, 530, "cylinder.msh", 12, P["muted"], mono=True)
    s.text(76, 548, "1 536 quads · 2 zones", 11.5, P["faint"], mono=True)

    vx, vw = 276, 664
    s.rect(vx, 48, vw, H - 48 - 120, P["bg"])
    cylinder_mesh(s, vx + vw / 2, 330, 230, 26)
    s.rect(vx + 16, 64, 210, 28, P["surface"], P["border"], r=14)
    s.text(vx + 30, 82, "Zones  ·  Regions  ·  Probes  ·  Quality", 11.5, P["muted"])
    s.rect(vx + vw / 2 + 170, 160, 140, 26, P["surface"], P["zones"][1], r=6)
    s.text(vx + vw / 2 + 180, 177, "farfield · 64 faces", 11, mono=True)
    s.line(vx + 30, 610, vx + 130, 610, P["muted"], 1.5)
    s.text(vx + 30, 626, "10 D", 11, P["muted"], mono=True)

    py = H - 120
    s.rect(vx, py, vw, 120, P["surface"])
    s.line(vx, py, vx + vw, py, P["border"])
    s.text(vx + 16, py + 24, "PROBLEMS", 11, P["faint"], 600)
    s.text(vx + 100, py + 24, "mallard check · 0.14 s · on input change", 11, P["faint"])
    rows = [(P["error"], "E-INPUT", "boundaries.farfield.T: required for farfield", "line 34"),
            (P["warn"], "W-FORMAT-VTU", "write_data: VTU files, build has no HDF5", "line 72"),
            (P["muted"], "W-DNS-ETA", "cells of 6.4 eta on average: refine or use [les]", "")]
    for i, (c, code, msg, ln) in enumerate(rows):
        y = py + 52 + 22 * i
        s.circle(vx + 22, y - 4, 4, c)
        s.text(vx + 34, y, code, 11.5, c, 600, mono=True)
        s.text(vx + 150, y, msg, 12.5)
        s.text(vx + vw - 16, y, ln, 11.5, P["muted"], mono=True, anchor="end")

    ix = 940
    s.rect(ix, 48, W - ix, H - 48, P["surface"])
    s.line(ix, 48, ix, H, P["border"])
    s.rect(ix + 20, 68, 10, 10, P["zones"][1], r=2)
    s.text(ix + 38, 78, "farfield", 15, weight=600, mono=True)
    s.text(ix + 20, 100, "Boundary condition of zone farfield", 12, P["muted"])
    y = 128
    y += field(s, ix + 20, y, 300, "type", "farfield", select=True)
    y += field(s, ix + 20, y, 300, "u  velocity", "[69.4, 0.0]", "m/s")
    y += field(s, ix + 20, y, 300, "p  pressure", "101 325", "Pa")
    y += field(s, ix + 20, y, 300, "T  temperature", "", "K", error="Required for farfield. E-INPUT  ·  docs", focus=True)
    s.text(ix + 20, y + 4, "Expert", 12, P["muted"], 500)
    s.path(f"M{ix + 70} {y - 1} l4 4 l4 -4", P["muted"])
    s.line(ix + 20, y + 22, W - 20, y + 22, P["border"])
    s.text(ix + 20, y + 46, "boundaries[1]  ·  input.toml line 30", 11.5, P["faint"], mono=True)
    s.rect(ix + 20, H - 60, 140, 36, P["surface"], P["border"], r=8)
    s.text(ix + 90, H - 37, "Check", 13, weight=500, anchor="middle")
    s.rect(ix + 172, H - 60, 148, 36, P["accent"], r=8)
    s.text(ix + 246, H - 37, "Run locally", 13, P["surface"], 600, anchor="middle")
    return s.str()


def series(s, x, y, w, h, f, color, n=240, sw=1.5):
    pts = [(x + w * i / (n - 1), y + h * (0.5 - 0.5 * f(i / (n - 1)))) for i in range(n)]
    s.path("M" + " L".join(f"{a:.1f} {b:.1f}" for a, b in pts), color, sw)


def chart(s, x, y, w, h, title, unit, curves):
    P = s.P
    s.rect(x, y, w, h, P["surface"], P["border"], r=10)
    s.text(x + 16, y + 26, title, 13, weight=600)
    s.text(x + w - 16, y + 26, unit, 11.5, P["muted"], mono=True, anchor="end")
    for k in range(4):
        gy = y + 48 + (h - 72) * k / 3
        s.line(x + 16, gy, x + w - 16, gy, P["border"], 1, "2 4")
    for f, c in curves:
        series(s, x + 16, y + 48, w - 32, h - 72, f, c)


def run_monitor(P):
    s = Svg(P)
    chrome(s, "Runs / re100-fine", "hpc · Slurm", "Runs")
    x0 = 76
    s.text(x0, 86, "re100-fine", 20, weight=600)
    w = badge(s, x0 + 128, 70, "running", P["accent_soft"], P["accent"])
    s.text(x0 + 140 + w, 85, "job 48213 · 4 nodes · 16 GPUs · rank 0 on its first node", 12, P["muted"], mono=True)
    s.rect(W - 316, 66, 112, 32, P["surface"], P["border"], r=8)
    s.text(W - 260, 87, "Checkpoint", 13, weight=500, anchor="middle")
    s.rect(W - 196, 66, 72, 32, P["surface"], P["border"], r=8)
    s.text(W - 160, 87, "Set…", 13, weight=500, anchor="middle")
    s.rect(W - 116, 66, 96, 32, P["error_soft"], P["error"], r=8)
    s.text(W - 68, 87, "Stop", 13, P["error"], 600, anchor="middle")

    s.rect(x0, 116, W - x0 - 20, 6, P["border"], r=3)
    s.rect(x0, 116, (W - x0 - 20) * 0.62, 6, P["accent"], r=3)
    s.text(x0, 142, "t = 99.4 of 160  ·  62 %  ·  ETA 3 h 12 min  ·  stops at t_stop = 160", 12, P["muted"], mono=True)
    tiles = [("step", "1 284 516", ""), ("dt", "7.74e-5", ""), ("CFL", "1.00", ""), ("throughput", "2.31e9", "cells/s"),
             ("memory", "61.2", "GB/GPU"), ("min p", "0.6931", "Pa")]
    tw = (W - x0 - 20 - 5 * 12) / 6
    for i, (k, v, u) in enumerate(tiles):
        tx = x0 + i * (tw + 12)
        s.rect(tx, 160, tw, 72, P["surface"], P["border"], r=10)
        s.text(tx + 14, 182, k, 11.5, P["muted"], 500)
        s.text(tx + 14, 214, v, 20, weight=500, mono=True)
        if u:
            s.text(tx + tw - 14, 214, u, 11, P["muted"], mono=True, anchor="end")
    cw = (W - x0 - 20 - 12) / 2
    z = P["zones"]
    chart(s, x0, 248, cw, 230, "Probe wake_point · U_Y", "m/s",
          [(lambda t: 0.85 * math.sin(2 * math.pi * 9.5 * t) * (1 - math.exp(-6 * t)), z[0])])
    chart(s, x0 + cw + 12, 248, cw, 230, "Forces cylinder · C_D, C_L", "1",
          [(lambda t: 0.55 + 0.05 * math.sin(2 * math.pi * 19 * t) * (1 - math.exp(-6 * t)), z[1]),
           (lambda t: 0.6 * math.sin(2 * math.pi * 9.5 * t) * (1 - math.exp(-6 * t)), z[2])])
    s.rect(x0, 494, cw, H - 494 - 20, P["surface"], P["border"], r=10)
    s.text(x0 + 16, 520, "Events", 13, weight=600)
    s.text(x0 + cw - 16, 520, "mallard.jsonl · followed over SSH", 11.5, P["muted"], anchor="end")
    ev = [("14:02", "output", "solut/cylinder_000099.vtkhdf  ·  412 MB"), ("13:58", "set", "t_stop = 160 (was 120)"),
          ("13:31", "checkpoint", "solut/restart_1210000.restart  ·  1.9 GB"),
          ("12:10", "warning", "W-CFL-LIMIT cleared"), ("11:47", "start", "16 ranks · v0.11.0 · cuda")]
    for i, (t, k, m) in enumerate(ev):
        y = 552 + 34 * i
        s.text(x0 + 16, y, t, 12, P["muted"], mono=True)
        s.text(x0 + 70, y, k, 12, P["accent"] if k in ("output", "checkpoint") else P["fg"], 500)
        s.text(x0 + 160, y, m, 12, mono=True)
    vx = x0 + cw + 12
    s.rect(vx, 494, cw, H - 494 - 20, P["surface"], P["border"], r=10)
    s.text(vx + 16, 520, "3D view", 13, weight=600)
    s.text(vx + cw - 16, 520, "viewer job on cpu-small · idle stop 15 min", 11.5, P["muted"], anchor="end")
    s.rect(vx + 16, 536, cw - 32, H - 536 - 36, P["bg"], r=8)
    cx, cy = vx + cw / 2 - 160, 634
    for k in range(6):
        s.path(f"M{cx + 24 * k} {cy} q 36 {-40 + 10 * (k % 2)} 72 0 t 72 0 t 72 0", P["zones"][0] if k % 2 else P["zones"][3], 2.5 - 0.2 * k)
    s.circle(cx - 10, cy, 14, P["mesh"])
    s.text(vx + 30, H - 52, "VORTICITY_Z  ·  t = 99.0  ·  streamed, nothing copied", 11.5, P["muted"], mono=True)
    return s.str()


def data_manager(P):
    s = Svg(P)
    chrome(s, "Data / re100-fine", "hpc · Slurm", "Data")
    x0 = 76
    s.text(x0, 86, "Files of re100-fine", 20, weight=600)
    s.text(x0, 108, "run 3a256c99  ·  home: hpc:/scratch/project/runs/re100-fine  ·  objects by SHA-256", 12, P["muted"], mono=True)
    for i, f in enumerate(["All", "Snapshots", "Restarts", "Probes & forces", "Inputs"]):
        fx = x0 + [0, 48, 148, 236, 366][i]
        active = i == 0
        s.rect(fx, 124, [40, 92, 80, 122, 64][i], 26, P["accent_soft"] if active else P["surface"], None if active else P["border"], r=13)
        s.text(fx + [20, 46, 40, 61, 32][i], 141, f, 12, P["accent"] if active else P["fg"], 500, anchor="middle")
    s.text(x0 + 450, 141, "t from 60.0 to 99.0", 12, P["muted"], mono=True)

    tx, tw = x0, 820
    s.rect(tx, 164, tw, H - 184, P["surface"], P["border"], r=10)
    cols = [("", 16), ("file", 44), ("kind", 330), ("t", 420), ("size", 510), ("hpc", 610), ("local", 680), ("k8s-lab", 750)]
    for name, cx in cols:
        s.text(tx + cx, 192, name, 11.5, P["muted"], 600, mono=name in ("hpc", "local", "k8s-lab"))
    s.line(tx, 204, tx + tw, 204, P["border"])
    rows = [("cylinder_000060.vtkhdf", "snapshot", "60.0", "412 MB", "s", "s", ""),
            ("cylinder_000070.vtkhdf", "snapshot", "70.0", "412 MB", "s", "s", ""),
            ("cylinder_000080.vtkhdf", "snapshot", "80.0", "412 MB", "s", "", ""),
            ("cylinder_000090.vtkhdf", "snapshot", "90.0", "412 MB", "s", "part", ""),
            ("cylinder_000099.vtkhdf", "snapshot", "99.0", "412 MB", "s", "", ""),
            ("restart_1210000.restart", "restart", "93.6", "1.9 GB", "s", "", "s"),
            ("probe_wake_point.csv", "probe", "0–99.4", "2.1 MB", "s", "s", ""),
            ("forces_cylinder.csv", "forces", "0–99.4", "0.8 MB", "s", "s", ""),
            ("input.toml", "input", "", "4 kB", "s", "s", "s"),
            ("cylinder_fine.msh", "mesh", "", "1.4 GB", "s", "s", "s")]
    sel = {2, 3, 4}
    for i, (f, k, t, size, a, b, c) in enumerate(rows):
        y = 230 + 34 * i
        if i in sel:
            s.rect(tx + 1, y - 21, tw - 2, 34, P["accent_soft"])
        s.rect(tx + 16, y - 12, 14, 14, P["accent"] if i in sel else "none", P["accent"] if i in sel else P["faint"], r=3)
        if i in sel:
            s.path(f"M{tx + 19} {y - 5} l3 3 l6 -6", P["surface"], 1.8)
        s.text(tx + 44, y, f, 12.5, mono=True)
        s.text(tx + 330, y, k, 12, P["muted"])
        s.text(tx + 420, y, t, 12, mono=True)
        s.text(tx + 510, y, size, 12, mono=True)
        for cx, st in ((610, a), (680, b), (750, c)):
            if st == "s":
                s.circle(tx + cx + 12, y - 4, 5, P["ok"])
            elif st == "part":
                s.add(f'<path d="M{tx + cx + 12} {y - 9} a5 5 0 0 1 0 10 z" fill="{P["accent"]}"/>')
                s.circle(tx + cx + 12, y - 4, 5, "none", P["accent"], 1.5)
                s.text(tx + cx + 22, y, "41 %", 10.5, P["accent"], mono=True)
            else:
                s.circle(tx + cx + 12, y - 4, 5, "none", P["faint"], 1)
    s.text(tx + 16, H - 34, "● present (SHA-256 matches the run log)   ○ absent   ◐ partial (resumes from any copy with the same hash)",
           11.5, P["muted"])

    px = tx + tw + 16
    pw = W - px - 20
    s.rect(px, 164, pw, H - 184, P["surface"], P["border"], r=10)
    s.text(px + 18, 192, "Copy 3 files", 15, weight=600)
    s.text(px + 18, 214, "1.24 GB  ·  1 partial, resumes at 41 %", 12, P["muted"], mono=True)
    s.text(px + 18, 248, "from", 12, P["muted"], 500)
    s.rect(px + 18, 256, pw - 36, 32, P["raised"], P["border"], r=6)
    s.text(px + 30, 277, "hpc  ·  /scratch/project/runs/…", 12.5, mono=True)
    s.text(px + 18, 310, "to", 12, P["muted"], 500)
    s.rect(px + 18, 318, pw - 36, 32, P["raised"], P["accent"], r=6, sw=1.5)
    s.text(px + 30, 339, "local  ·  ~/cases/cylinder/runs/…", 12.5, mono=True)
    s.text(px + 18, 378, "ROUTE", 11, P["faint"], 600)
    for i, (r, est, chosen) in enumerate([("direct  rsync over SSH", "≈ 58 s at 22 MB/s", True),
                                          ("view remotely instead", "no copy · viewer job", False)]):
        y = 396 + 52 * i
        s.rect(px + 18, y, pw - 36, 44, P["accent_soft"] if chosen else P["raised"], P["accent"] if chosen else P["border"], r=8)
        s.circle(px + 36, y + 22, 6, P["accent"] if chosen else "none", P["accent"] if chosen else P["faint"], 1.5)
        s.text(px + 52, y + 19, r, 12.5, weight=500)
        s.text(px + 52, y + 35, est, 11.5, P["muted"], mono=True)
    s.text(px + 18, 528, "Checked by SHA-256 against the run log's records.", 11.5, P["muted"])
    s.text(px + 18, 546, "Indexed on this machine; nothing committed changes.", 11.5, P["muted"])
    s.rect(px + 18, 568, pw - 36, 38, P["accent"], r=8)
    s.text(px + pw / 2, 592, "Copy 1.24 GB", 13, P["surface"], 600, anchor="middle")
    s.text(px + 18, 640, "HISTORY", 11, P["faint"], 600)
    for i, (h, m) in enumerate([("Nov 20 13:40", "2 files hpc → local · 4.2 MB"),
                                ("Nov 19 09:12", "inputs local → hpc · 1.4 GB")]):
        s.text(px + 18, 664 + 24 * i, h, 11.5, P["muted"], mono=True)
        s.text(px + 124, 664 + 24 * i, m, 11.5, mono=True)
    return s.str()


def figures():
    OUT.mkdir(parents=True, exist_ok=True)
    for P in (LIGHT, DARK):
        for name, draw in (("case-setup", case_setup), ("run-monitor", run_monitor), ("data-manager", data_manager)):
            (OUT / f"gui-{name}-{P['name']}.svg").write_text(draw(P))


if __name__ == "__main__":
    figures()
