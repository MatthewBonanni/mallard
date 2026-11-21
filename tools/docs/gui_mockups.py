"""High-fidelity mockups of three screens of the Mallard GUI (docs/design/gui.md): case
setup, run monitor and data manager, in the compact density, each in the GUI's light and
dark theme, as SVG with real text in docs/images/design/gui-<screen>-<theme>.svg.

    python tools/docs/gui_mockups.py

Needs only the standard library, and is deterministic. The case is examples/cylinder
(Re = 100, Mach 0.2, nondimensional) with its real keys, zones, writers and probes; the
plotted histories are synthetic, built from the example's reference results (St = 0.164,
mean C_D = 1.36, C_L amplitude 0.33). The icons are this script's own outline set. The
tokens below are the GUI's design language (docs/design/gui.md#design-language); the
shared token file replaces them once it exists.
"""
import math
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "docs" / "images" / "design"
W, H = 1440, 900
SANS = "Inter, 'Helvetica Neue', Arial, sans-serif"
MONO = "'JetBrains Mono', 'SF Mono', Menlo, monospace"

# Compact density: 4 px base, 24 px rows and inputs, 12 px text, 11 px secondary text
ROW, TEXT, SMALL, TOP, TABS, STATUS, HEAD = 24, 12, 11, 36, 28, 24, 24

LIGHT = dict(name="light", bg="#ffffff", panel="#f7f7f8", hover="#f0f0f2", sel="#e3f2ef", border="#e2e2e6",
             fg="#18181b", muted="#5f5f67", icon="#71717a", faint="#c4c4cc", accent="#0f766e", on_accent="#ffffff",
             error="#b42318", error_bg="#fdecea", warn="#9a5a06", warn_bg="#fdf3e1", ok="#3f7a12", mesh="#cfcfd6",
             view="#f3f3f5", tip="#18181b", on_tip="#ffffff",
             zones=["#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9"])
DARK = dict(name="dark", bg="#131417", panel="#191a1e", hover="#202126", sel="#123430", border="#2b2c31",
            fg="#ececee", muted="#a3a3ab", icon="#8e8e96", faint="#45464d", accent="#2dd4bf", on_accent="#0b1f1c",
            error="#f97066", error_bg="#3a1715", warn="#f5b544", warn_bg="#33270f", ok="#9ad56a", mesh="#3a3b42",
            view="#0e0f12", tip="#ececee", on_tip="#131417",
            zones=["#56b4e9", "#e69f00", "#009e73", "#cc79a7", "#f0e442", "#d55e00"])

# vik (Crameri), sampled: perceptually uniform, diverging, colorblind-safe
VIK = ["#001261", "#033e7d", "#1e6f9d", "#71a8c4", "#c9dde7", "#eacebd", "#d29773", "#aa5b2e", "#71200b", "#590007"]

# The icon set: outline glyphs on a 24-unit grid, drawn at 16 px with 1.6 px strokes
ICONS = {
    "case": "M6 3h8l4 4v14H6z M14 3v4h4 M9 12h6 M9 16h6",
    "runs": "M12 3a9 9 0 1 0 0.01 0z M10 8.5v7l6-3.5z",
    "data": "M5 6c0-1.7 3.1-3 7-3s7 1.3 7 3-3.1 3-7 3-7-1.3-7-3z M5 6v12c0 1.7 3.1 3 7 3s7-1.3 7-3V6 M5 12c0 1.7 3.1 3 7 3s7-1.3 7-3",
    "view": "M12 3l8 4.5v9L12 21l-8-4.5v-9z M12 12l8-4.5 M12 12v9 M12 12L4 7.5",
    "remotes": "M4 4h16v6H4z M4 14h16v6H4z M8 7h.01 M8 17h.01",
    "search": "M11 4a7 7 0 1 0 0.01 0z M16 16l4 4",
    "mesh": "M4 4h16v16H4z M4 12h16 M12 4v16 M4 4l16 16",
    "physics": "M3 8c3-3 6 3 9 0s6 3 9 0 M3 16c3-3 6 3 9 0s6 3 9 0",
    "boundary": "M4 4h16v16H4z M4 20V4",
    "initial": "M12 3v3 M12 18v3 M3 12h3 M18 12h3 M12 8a4 4 0 1 0 0.01 0z",
    "numerics": "M17 5H7l6 7-6 7h10",
    "output": "M12 4v11 M7 10l5 5 5-5 M4 19h16",
    "regions": "M9 4a5 5 0 1 0 0.01 0z M11 11h9v9h-9z",
    "probes": "M12 3v5 M12 16v5 M3 12h5 M16 12h5 M12 9a3 3 0 1 0 0.01 0z",
    "statistics": "M5 20V11 M10 20V5 M15 20v-7 M20 20V9",
    "run": "M12 4a8 8 0 1 0 0.01 0z M12 8v4l3 2",
    "play": "M7 4l12 8-12 8z",
    "stop": "M6 6h12v12H6z",
    "bookmark": "M6 3h12v18l-6-4-6 4z",
    "sliders": "M4 7h10 M18 7h2 M4 17h4 M12 17h8 M14 4v6 M8 14v6",
    "laptop": "M5 5h14v10H5z M2 19h20",
    "server": "M4 4h16v6H4z M4 14h16v6H4z M8 7h.01 M8 17h.01",
    "k8s": "M12 2l8.7 5v10L12 22l-8.7-5V7z M12 8v8 M8.5 10l7 4 M15.5 10l-7 4",
    "check": "M5 12l5 5 9-10",
    "alert": "M12 3l9.5 17h-19z M12 10v4 M12 17h.01",
    "error": "M12 3a9 9 0 1 0 0.01 0z M9 9l6 6 M15 9l-6 6",
    "chev_r": "M9 6l6 6-6 6",
    "chev_d": "M6 9l6 6 6-6",
    "arrow": "M4 12h16 M14 6l6 6-6 6",
    "pause": "M8 5v14 M16 5v14",
    "refresh": "M20 11a8 8 0 1 0-2.3 5.7 M20 4v7h-7",
    "eye": "M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z M12 9a3 3 0 1 0 0.01 0z",
    "camera": "M4 7h4l2-3h4l2 3h4v13H4z M12 10a3.5 3.5 0 1 0 0.01 0z",
    "layers": "M12 3l9 5-9 5-9-5z M3 13l9 5 9-5",
    "more": "M5 12h.01 M12 12h.01 M19 12h.01",
    "filter": "M4 5h16l-6 8v6l-4-2v-4z",
    "pick": "M5 3l14 7-6 2-2 6z",
    "link": "M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1 M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1",
    "fit": "M4 9V4h5 M15 4h5v5 M20 15v5h-5 M9 20H4v-5",
    "plus": "M12 5v14 M5 12h14",
}


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def tw(t, size=TEXT, mono=False):
    """Approximate text width."""
    return len(t) * size * (0.6 if mono else 0.54)


class Svg:
    def __init__(self, P):
        self.P = P
        self.items = []

    def add(self, s):
        self.items.append(s)

    def rect(self, x, y, w, h, fill, stroke=None, r=0, sw=1):
        s = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" fill="{fill}"{s}/>')

    def text(self, x, y, t, size=TEXT, fill=None, weight=400, mono=False, anchor="start", rotate=None):
        fam = MONO if mono else SANS
        feat = ' font-feature-settings="\'tnum\'"' if mono else ""
        rot = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate else ""
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{fam}" font-size="{size}" font-weight="{weight}" '
                 f'fill="{fill or self.P["fg"]}" text-anchor="{anchor}"{feat}{rot}>{esc(t)}</text>')

    def line(self, x1, y1, x2, y2, stroke=None, sw=1, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                 f'stroke="{stroke or self.P["border"]}" stroke-width="{sw}"{d}/>')

    def path(self, d, stroke, sw=1.5, fill="none"):
        self.add(f'<path d="{d}" stroke="{stroke}" stroke-width="{sw}" fill="{fill}" stroke-linejoin="round" '
                 f'stroke-linecap="round"/>')

    def circle(self, x, y, r, fill="none", stroke=None, sw=1):
        s = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}"{s}/>')

    def icon(self, name, x, y, color=None, size=16):
        k = size / 24
        self.add(f'<path transform="translate({x:.1f} {y:.1f}) scale({k:.4f})" d="{ICONS[name]}" fill="none" '
                 f'stroke="{color or self.P["icon"]}" stroke-width="{1.6 / k:.2f}" stroke-linecap="round" '
                 f'stroke-linejoin="round"/>')

    def str(self):
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n'
                + "\n".join(self.items) + "\n</svg>\n")


def badge(s, x, y, text, kind):
    P = s.P
    fg, bg = {"error": (P["error"], P["error_bg"]), "warn": (P["warn"], P["warn_bg"]),
              "accent": (P["accent"], P["sel"]), "muted": (P["muted"], P["hover"])}[kind]
    w = tw(text, SMALL - 1, True) + 10
    s.rect(x - w, y - 7, w, 14, bg, r=7)
    s.text(x - w / 2, y + 3.5, text, SMALL - 1, fg, 600, mono=True, anchor="middle")
    return w


def head(s, x, y, w, label, right="", icons=()):
    """A panel header: 24 px, uppercase label, optional actions, divider below."""
    P = s.P
    s.rect(x, y, w, HEAD, P["panel"])
    s.text(x + 8, y + 16, label.upper(), SMALL, P["muted"], 600)
    rx = x + w - 6
    for name in icons:
        rx -= 20
        s.icon(name, rx + 2, y + 4, size=16)
    if right:
        s.text(rx - 4, y + 16, right, SMALL, P["muted"], mono=True, anchor="end")
    s.line(x, y + HEAD, x + w, y + HEAD)


def button(s, x, y, label, kind="plain", icon=None, w=None):
    P = s.P
    w = w or 16 + tw(label) + (18 if icon else 0)
    fill, stroke, fg = {"plain": (P["bg"], P["border"], P["fg"]), "accent": (P["accent"], None, P["on_accent"]),
                        "danger": (P["bg"], P["error"], P["error"])}[kind]
    s.rect(x, y, w, 24, fill, stroke, r=4)
    tx = x + 8
    if icon:
        s.icon(icon, tx, y + 4, fg if kind != "plain" else P["icon"], 16)
        tx += 20
    s.text(tx, y + 16, label, TEXT, fg, 500)
    return w


def tabs(s, x, y, w, names, active):
    P = s.P
    s.rect(x, y, w, TABS, P["bg"])
    tx = x
    for n in names:
        tw_ = 20 + tw(n)
        if n == active:
            s.rect(tx, y + TABS - 2, tw_, 2, P["accent"])
        s.text(tx + 10, y + 18, n, TEXT, P["fg"] if n == active else P["muted"], 600 if n == active else 400)
        tx += tw_
    s.line(x, y + TABS, x + w, y + TABS)
    return tx


def chrome(s, crumbs, tab, status, project="cylinder-re100"):
    """Top bar with project, breadcrumbs and search; left rail; status bar."""
    P = s.P
    s.rect(0, 0, W, H, P["bg"])
    s.rect(0, 0, W, TOP, P["panel"])
    s.line(0, TOP, W, TOP)
    s.circle(20, 18, 7, P["accent"])
    s.path("M16.5 18.5c1.5-3 5.5-3 7 0", P["on_accent"], 1.4)
    s.text(34, 22, project, 13, weight=600)
    x = 34 + tw(project, 13) + 8
    for i, (c, loc) in enumerate(crumbs):
        s.icon("chev_r", x, 11, P["faint"], 14)
        x += 16
        s.text(x, 22, c, TEXT, P["fg"] if i == len(crumbs) - 1 else P["muted"], 500 if i == len(crumbs) - 1 else 400)
        x += tw(c) + 6
        if loc:
            icon, label = loc
            w = 28 + tw(label, SMALL)
            s.rect(x, 8, w, 20, P["sel"], r=10)
            s.icon(icon, x + 6, 12, P["accent"], 12)
            s.text(x + 22, 22, label, SMALL, P["accent"], 600)
            x += w + 6
    s.rect(W - 420, 6, 260, 24, P["bg"], P["border"], r=4)
    s.icon("search", W - 412, 10, size=16)
    s.text(W - 392, 22, "Keys, runs, files, actions", TEXT, P["muted"])
    s.rect(W - 206, 10, 40, 16, P["hover"], r=3)
    s.text(W - 186, 22, "⌘ K", SMALL, P["muted"], mono=True, anchor="middle")
    rx = W - 150
    for icon, label in (("server", "hpc"), ("k8s", "k8s-lab")):
        s.icon(icon, rx, 10, size=16)
        s.circle(rx + 15, 25, 3, P["ok"])
        s.text(rx + 22, 22, label, SMALL, P["muted"])
        rx += 26 + tw(label, SMALL) + 10
    s.rect(0, TOP, 44, H - TOP - STATUS, P["panel"])
    s.line(44, TOP, 44, H - STATUS)
    for i, name in enumerate(["case", "runs", "data", "view", "remotes"]):
        y = TOP + 8 + 36 * i
        if name == tab:
            s.rect(8, y, 28, 28, P["sel"], r=4)
            s.rect(0, y + 4, 2, 20, P["accent"])
        s.icon(name, 14, y + 6, P["accent"] if name == tab else P["icon"], 16)
    s.icon("sliders", 14, H - STATUS - 34, size=16)
    s.rect(0, H - STATUS, W, STATUS, P["panel"])
    s.line(0, H - STATUS, W, H - STATUS)
    x = 10
    for t, c, icon in status:
        if icon:
            s.icon(icon, x, H - STATUS + 5, c or P["icon"], 14)
            x += 18
        s.text(x, H - 8, t, SMALL, c or P["muted"], mono=True)
        x += tw(t, SMALL, True) + 10
        s.line(x, H - STATUS + 6, x, H - 6)
        x += 10


def tree(s, x, y, w, items, selected, hovered=None):
    """Rows of (depth, icon, label, open, status, right); open: True, False or None (leaf)."""
    P = s.P
    for i, (depth, icon, label, is_open, st, right) in enumerate(items):
        ry = y + ROW * i
        if label == selected:
            s.rect(x, ry, w, ROW, P["sel"])
            s.rect(x, ry, 2, ROW, P["accent"])
        elif label == hovered:
            s.rect(x, ry, w, ROW, P["hover"])
        tx = x + 6 + 14 * depth
        if is_open is not None:
            s.icon("chev_d" if is_open else "chev_r", tx, ry + 6, P["icon"], 12)
        tx += 14
        if icon:
            s.icon(icon, tx, ry + 4, P["accent"] if label == selected else P["icon"], 16)
            tx += 20
        mono = depth > 0
        s.text(tx, ry + 16, label, TEXT, P["accent"] if label == selected else P["fg"],
               600 if label == selected else (500 if depth == 0 else 400), mono=mono)
        rx = x + w - 8
        if st:
            kind, n = st
            rx -= badge(s, rx, ry + 12, n, kind) + 4
        if right:
            s.text(rx, ry + 16, right, SMALL, P["muted"], mono=True, anchor="end")


def grid(s, x, y, w, rows, label_w=112):
    """A property grid: label | value | unit, 24 px rows, thin dividers."""
    P = s.P
    yy = y
    for r in rows:
        if r[0] == "group":
            s.rect(x, yy, w, ROW, P["panel"])
            s.icon("chev_d", x + 6, yy + 6, P["icon"], 12)
            s.text(x + 22, yy + 16, r[1], SMALL, P["muted"], 600)
            if len(r) > 2:
                s.text(x + w - 8, yy + 16, r[2], SMALL, P["muted"], mono=True, anchor="end")
            s.line(x, yy + ROW, x + w, yy + ROW)
            yy += ROW
            continue
        label, value, unit, kind = (list(r) + ["", ""])[:4]
        if kind == "focus":
            s.rect(x, yy, w, ROW, P["sel"])
        s.text(x + 8, yy + 16, label, TEXT, P["muted"], mono=True)
        s.line(x + label_w, yy, x + label_w, yy + ROW)
        vx = x + label_w + 8
        if kind == "select":
            s.text(vx, yy + 16, value, TEXT)
            s.icon("chev_d", x + w - 20, yy + 6, P["icon"], 12)
        elif kind == "pick":
            s.icon("pick", vx, yy + 5, P["accent"], 14)
            s.text(vx + 18, yy + 16, value, TEXT, P["accent"])
        elif kind == "link":
            s.icon("link", vx, yy + 5, P["icon"], 14)
            s.text(vx + 18, yy + 16, value, TEXT, P["accent"], mono=True)
        elif kind == "focus":
            s.rect(x + label_w + 3, yy + 3, w - label_w - 6, ROW - 6, P["bg"], P["accent"], r=3, sw=1.5)
            s.text(vx, yy + 16, value, TEXT, mono=True)
            s.line(vx + tw(value, TEXT, True) + 1, yy + 6, vx + tw(value, TEXT, True) + 1, yy + 18, P["fg"])
        else:
            s.text(vx, yy + 16, value, TEXT, mono=True)
        if unit:
            s.text(x + w - (26 if kind == "select" else 8), yy + 16, unit, SMALL, P["muted"], mono=True, anchor="end")
        yy += ROW
        s.line(x, yy, x + w, yy)
    return yy


def nice_ticks(lo, hi, n=5):
    span = hi - lo
    step = 10 ** math.floor(math.log10(span / n))
    for m in (1, 2, 2.5, 5, 10):
        if span / (step * m) <= n:
            step *= m
            break
    t = math.ceil(lo / step) * step
    out = []
    while t <= hi + 1e-12:
        out.append(round(t, 10))
        t += step
    return out


def fmt(v):
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1000 or a < 0.01:
        return f"{v:.3g}" if a >= 1000 else f"{v:.3e}".rstrip("0").replace("0e", "e").replace("e-0", "e-")
    return f"{v:g}"


def plot(s, x, y, w, h, title, xlab, ylab, series, ylim, xlim, unit="", note=""):
    """A chart: title and legend, gridlines, ticks with labels, axis labels with units."""
    P = s.P
    s.text(x + 10, y + 16, title, TEXT, weight=600)
    lx = x + 10 + tw(title) + 14
    for name, c, _ in series:
        s.rect(lx, y + 11, 12, 2, c)
        s.text(lx + 16, y + 16, name, SMALL, P["muted"], mono=True)
        lx += 16 + tw(name, SMALL, True) + 12
    if note:
        s.text(x + w - 10, y + 16, note, SMALL, P["muted"], mono=True, anchor="end")
    px, py, pw, ph = x + 52, y + 28, w - 64, h - 62
    s.rect(px, py, pw, ph, "none", P["border"])
    for t in nice_ticks(*ylim):
        gy = py + ph * (1 - (t - ylim[0]) / (ylim[1] - ylim[0]))
        s.line(px, gy, px + pw, gy, P["border"], 1, "2 3")
        s.line(px - 3, gy, px, gy, P["muted"])
        s.text(px - 6, gy + 4, fmt(t), SMALL - 1, P["muted"], mono=True, anchor="end")
    for t in nice_ticks(*xlim, 6):
        gx = px + pw * (t - xlim[0]) / (xlim[1] - xlim[0])
        s.line(gx, py, gx, py + ph, P["border"], 1, "2 3")
        s.line(gx, py + ph, gx, py + ph + 3, P["muted"])
        s.text(gx, py + ph + 14, fmt(t), SMALL - 1, P["muted"], mono=True, anchor="middle")
    s.text(px + pw / 2, y + h - 6, xlab, SMALL, P["muted"], anchor="middle")
    s.text(x + 12, py + ph / 2, ylab, SMALL, P["muted"], anchor="middle", rotate=-90)
    for _, c, pts in series:
        d = []
        for tx, ty in pts:
            gx = px + pw * (tx - xlim[0]) / (xlim[1] - xlim[0])
            gy = py + ph * (1 - (ty - ylim[0]) / (ylim[1] - ylim[0]))
            d.append(f"{gx:.1f} {min(max(gy, py), py + ph):.1f}")
        s.path("M" + " L".join(d), c, 1.3)


def cmap(v):
    """vik at v in [0, 1]."""
    v = min(max(v, 0.0), 1.0) * (len(VIK) - 1)
    i = min(int(v), len(VIK) - 2)
    f = v - i
    a = [int(VIK[i][k:k + 2], 16) for k in (1, 3, 5)]
    b = [int(VIK[i + 1][k:k + 2], 16) for k in (1, 3, 5)]
    return "#" + "".join(f"{round(p + (q - p) * f):02x}" for p, q in zip(a, b))


def vorticity(x, y, t=0.0):
    """A Karman street behind a unit cylinder: alternating Gaussian vortices plus the shear layers."""
    w = 0.0
    a, b, ux = 5.2, 1.1, 0.17
    phase = (t * ux) % a
    for k in range(-1, 6):
        xc = 1.4 + k * a / 2 + phase
        sgn = 1 if k % 2 == 0 else -1
        yc = sgn * b / 2
        core = 0.45 + 0.08 * (xc - 1.4)
        strength = 2.2 * math.exp(-0.03 * (xc - 1.4))
        w += sgn * strength * math.exp(-((x - xc) ** 2 + (y - yc) ** 2) / core ** 2)
    if x > 0:
        for sgn in (1, -1):
            w += -sgn * 2.6 * math.exp(-((y - sgn * 0.5 * (1 + 0.12 * x)) ** 2) / 0.02) * math.exp(-x / 1.2)
    return w


def field_view(s, x, y, w, h, x0, x1, y0, y1, nx, ny, clim):
    """VORTICITY_Z on a raster, merged into runs of equal color, with the cylinder."""
    P = s.P
    s.rect(x, y, w, h, P["view"])
    dx, dy = w / nx, h / ny
    for j in range(ny):
        yy = y1 - (j + 0.5) * (y1 - y0) / ny
        run_c, run_i = None, 0
        for i in range(nx + 1):
            c = None
            if i < nx:
                xx = x0 + (i + 0.5) * (x1 - x0) / nx
                if xx * xx + yy * yy < 0.25:
                    c = P["mesh"]
                else:
                    v = vorticity(xx, yy)
                    c = cmap(round(((v - clim[0]) / (clim[1] - clim[0])) * 24) / 24)
            if c != run_c:
                if run_c is not None:
                    s.rect(x + run_i * dx, y + j * dy, (i - run_i) * dx + 0.4, dy + 0.4, run_c)
                run_c, run_i = c, i
    cx = x + w * (0 - x0) / (x1 - x0)
    cy = y + h * (y1 - 0) / (y1 - y0)
    s.circle(cx, cy, 0.5 * w / (x1 - x0), P["mesh"], P["fg"], 1)


def colorbar(s, x, y, w, label, clim, ticks):
    P = s.P
    n = 48
    for k in range(n):
        s.rect(x + w * k / n, y, w / n + 0.4, 10, cmap(k / (n - 1)))
    s.rect(x, y, w, 10, "none", P["border"])
    for t in ticks:
        tx = x + w * (t - clim[0]) / (clim[1] - clim[0])
        s.line(tx, y + 10, tx, y + 13, P["muted"])
        s.text(tx, y + 24, fmt(t), SMALL - 1, P["muted"], mono=True, anchor="middle")
    s.text(x, y - 5, label, SMALL, P["fg"], 600, mono=True)


def cylinder_case_view(s, x, y, w, h):
    """The case view of examples/cylinder near the body: mesh, zones, the near_wake region, cuts, probes."""
    P = s.P
    z = P["zones"]
    s.add(f'<clipPath id="cv"><rect x="{x}" y="{y}" width="{w}" height="{h}"/></clipPath><g clip-path="url(#cv)">')
    s.rect(x, y, w, h, P["view"])
    sc = 46.0
    cx, cy = x + w * 0.30, y + h * 0.5

    def X(px):
        return cx + px * sc

    def Y(py):
        return cy - py * sc

    nr, nt = 30, 96
    for k in range(nr):
        r = 0.5 * math.exp(k * 0.12)
        if r * sc > w:
            break
        s.circle(cx, cy, r * sc, stroke=P["mesh"], sw=0.6)
    for k in range(nt):
        a = 2 * math.pi * k / nt
        s.line(X(0.5 * math.cos(a)), Y(0.5 * math.sin(a)), X(20 * math.cos(a)), Y(20 * math.sin(a)), P["mesh"], 0.6)
    s.rect(x, y, w, h, "none", P["border"])
    s.add(f'<rect x="{X(0.0):.1f}" y="{Y(1.5):.1f}" width="{6 * sc:.1f}" height="{3 * sc:.1f}" '
          f'fill="{z[3]}" fill-opacity="0.16" stroke="{z[3]}" stroke-width="2"/>')
    s.text(X(0.08), Y(1.5) - 6, "near_wake", SMALL, z[3], 600, mono=True)
    for xc in (1.0, 1.4, 1.8):
        s.line(X(xc), Y(2.0), X(xc), Y(-2.0), z[2], 2)
    s.text(X(1.8) + 4, Y(-2.0) + 12, "wake cuts x = 1.0, 1.4, 1.8", SMALL, z[2], 600, mono=True)
    s.circle(cx, cy, 0.5 * sc, P["view"], z[0], 3)
    s.text(cx - 0.5 * sc - 4, cy - 0.5 * sc - 6, "cylinder", SMALL, z[0], 600, mono=True, anchor="end")
    for k in range(0, 91, 3):
        a = math.radians(-45 + k)
        s.circle(X(5 * math.cos(a)), Y(5 * math.sin(a)), 1.4, z[4])
    s.text(X(5 * math.cos(math.radians(40))) + 6, Y(5 * math.sin(math.radians(40))) - 4, "wake_arc", SMALL, z[4], 600,
           mono=True)
    for px, py, lab in ((2.5, 0.0, "wake_point"), (3.0, 0.0, "phase probe")):
        s.path(f"M{X(px) - 5} {Y(py)} h10 M{X(px)} {Y(py) - 5} v10", P["fg"], 1.5)
    s.text(X(2.5) + 6, Y(0.0) + 14, "wake_point", SMALL, P["fg"], 600, mono=True)
    s.add(f'<rect x="{X(0.0) - 1:.1f}" y="{Y(1.5) - 1:.1f}" width="{6 * sc + 2:.1f}" height="{3 * sc + 2:.1f}" '
          f'fill="none" stroke="{P["accent"]}" stroke-width="1" stroke-dasharray="4 3"/>')
    s.add("</g>")
    return X, Y


def case_setup(P):
    s = Svg(P)
    chrome(s, [("Case", None), ("regions", None), ("near_wake", None)], "case",
           [("Local", P["ok"], "laptop"), ("mallard 0.11.0 · 2D · double · Threads", None, None),
            ("49 152 cells · nondimensional (D = 1, a∞ = 1, ρ∞ = 1)", None, None),
            ("check 0.14 s ago", None, "check"), ("1 error", P["error"], "error"), ("2 warnings", P["warn"], "alert"),
            ("input.toml · modified", None, None)])
    x0, tw0 = 44, 248
    head(s, x0, TOP, tw0, "Case", "input.toml", ("plus", "more"))
    items = [(0, "mesh", "Mesh", False, None, "file"),
             (0, "physics", "Physics", False, None, "navier_stokes"),
             (0, "boundary", "Boundaries", True, ("error", "1"), ""),
             (1, None, "cylinder", None, None, "wall_adiabatic"),
             (1, None, "farfield", None, ("error", "T"), "farfield"),
             (0, "initial", "Initial state", False, None, "analytical"),
             (0, "numerics", "Numerics", False, None, "TENO 3 · HLLC"),
             (0, "regions", "Regions", True, None, "8"),
             (1, None, "near_wake", None, None, "rectangle"),
             (1, None, "wake_cut_1", None, None, "cut_plane"),
             (1, None, "wake_cut_2", None, None, "cut_plane"),
             (1, None, "wake_cut_3", None, None, "cut_plane"),
             (1, None, "cylinder", None, None, "zone"),
             (1, None, "farfield", None, None, "zone"),
             (1, None, "domain", None, None, "mesh"),
             (0, "probes", "Probes", True, None, "3"),
             (1, None, "wake_point", None, None, "point"),
             (1, None, "wake_arc", None, None, "arc · 91"),
             (1, None, "wall", None, None, "zone"),
             (0, "output", "Output", False, ("warn", "1"), "5 writers"),
             (0, "statistics", "Statistics", False, None, "phase"),
             (0, "run", "Run", False, None, "t_stop 160")]
    tree(s, x0, TOP + HEAD, tw0, items, "near_wake", hovered="wake_cut_2")
    s.line(x0 + tw0, TOP, x0 + tw0, H - STATUS)

    gx, gw = W - 352, 352
    vx, vw = x0 + tw0, gx - x0 - tw0
    tabs(s, vx, TOP, vw, ["Case view", "input.toml", "Mesh quality", "Check report"], "Case view")
    ty = TOP + TABS
    s.rect(vx, ty, vw, 30, P["bg"])
    bx = vx + 6
    for lab, on, icon in (("Zones", True, "boundary"), ("Regions", True, "regions"), ("Probes", True, "probes"),
                          ("Mesh", True, "mesh"), ("Quality", False, None), ("Partition", False, None),
                          ("2D lock", True, None), ("Clip", False, "layers")):
        w = 12 + tw(lab, SMALL) + (18 if icon else 0)
        s.rect(bx, ty + 4, w, 22, P["sel"] if on else P["bg"], None if on else P["border"], r=4)
        if icon:
            s.icon(icon, bx + 5, ty + 7, P["accent"] if on else P["icon"], 14)
        s.text(bx + (23 if icon else 6), ty + 19, lab, SMALL, P["accent"] if on else P["muted"], 500)
        bx += w + 4
    rx = vx + vw - 6
    for icon in ("camera", "fit", "eye"):
        rx -= 26
        s.rect(rx, ty + 3, 24, 24, P["bg"], P["border"], r=4)
        s.icon(icon, rx + 4, ty + 7, size=16)
    s.line(vx, ty + 30, vx + vw, ty + 30)
    pb = 4 * ROW + HEAD
    vy, vh = ty + 30, H - STATUS - ty - 30 - pb
    X, Y = cylinder_case_view(s, vx, vy, vw, vh)
    s.rect(vx + 8, vy + 8, 168, 82, P["bg"], P["border"], r=4)
    for i, (c, lab) in enumerate(((P["zones"][0], "cylinder  wall_adiabatic"), (P["zones"][3], "near_wake  region"),
                                  (P["zones"][2], "wake cuts  regions"), (P["zones"][4], "wake_arc  probe"))):
        s.rect(vx + 16, vy + 18 + 18 * i, 10, 10, c, r=2)
        s.text(vx + 32, vy + 27 + 18 * i, lab, SMALL, mono=True)
    s.line(vx + 16, vy + vh - 18, vx + 16 + 46, vy + vh - 18, P["fg"], 1.5)
    s.text(vx + 68, vy + vh - 14, "1 D", SMALL, P["muted"], mono=True)
    s.text(vx + vw - 8, vy + vh - 14, "x 3.12  y -0.84  ·  cell 20 417", SMALL, P["muted"], mono=True, anchor="end")
    s.path(f"M{vx + vw - 40} {vy + vh - 64} l0 -26 M{vx + vw - 40} {vy + vh - 64} l26 0", P["muted"], 1.2)
    s.text(vx + vw - 12, vy + vh - 60, "x", SMALL, P["muted"], mono=True)
    s.text(vx + vw - 44, vy + vh - 94, "y", SMALL, P["muted"], mono=True)

    py = vy + vh
    s.line(vx, py, gx, py)
    head(s, vx, py, vw, "Problems", "mallard check on change", ("filter",))
    rows = [("error", "E-INPUT", "boundaries.farfield.T", "required for type = \"farfield\": the free-stream temperature", "36"),
            ("alert", "W-FORMAT-VTU", "write_data[cylinder].format", "VTU files: this build has no HDF5 (vtkhdf is the default)", "66"),
            ("alert", "W-DNS-ETA", "mesh", "cells of 6.38 eta on average: refine to 0.71 or model with [les]", "—"),
            ("check", "regions", "near_wake", "398 cells, area 18.0, fully inside the domain", "84")]
    for i, (icon, code, key, msg, ln) in enumerate(rows):
        y = py + HEAD + ROW * i
        c = {"error": P["error"], "alert": P["warn"], "check": P["ok"]}[icon]
        if i == 0:
            s.rect(vx, y, vw, ROW, P["error_bg"])
        s.icon(icon, vx + 8, y + 5, c, 14)
        s.text(vx + 28, y + 16, code, SMALL, c, 600, mono=True)
        s.text(vx + 128, y + 16, key, SMALL, mono=True)
        s.text(vx + 318, y + 16, msg, TEXT)
        s.text(vx + vw - 8, y + 16, "line " + ln, SMALL, P["muted"], mono=True, anchor="end")
        s.line(vx, y + ROW, gx, y + ROW)

    s.line(gx, TOP, gx, H - STATUS)
    head(s, gx, TOP, gw, "Region", "write_data[2].region", ("more",))
    s.icon("regions", gx + 8, TOP + HEAD + 4, P["zones"][3], 16)
    s.text(gx + 30, TOP + HEAD + 17, "near_wake", 13, weight=600, mono=True)
    badge(s, gx + gw - 8, TOP + HEAD + 12, "inline", "muted")
    s.line(gx, TOP + HEAD + 24, gx + gw, TOP + HEAD + 24)
    y = grid(s, gx, TOP + HEAD + 24, gw, [
        ("group", "SHAPE"),
        ("type", "rectangle", "", "select"),
        ("center", "[3.0, 0.0]", "nd"),
        ("size", "[6.0, 3.0", "nd", "focus"),
        ("edge", "0.0", "nd"),
        ("group", "ON THE MESH", "from check"),
        ("dimension", "2 (filled)", ""),
        ("cells", "398", ""),
        ("area", "18.00", "nd²"),
        ("cut cells", "36 · fractions", ""),
        ("group", "USED BY", "1"),
        ("write_data", "./solut/near_wake", "", "link"),
        ("use in…", "probes · sources · sponges", "", "pick"),
        ("", "initialize · statistics", "", "pick"),
        ("group", "EXPERT"),
        ("transform", "none", "", "select"),
        ("copies", "none", "", "select"),
        ("motion", "static", "", "select"),
    ], 96)
    s.text(gx + 8, y + 18, "input.toml line 84 · docs: geometry#regions", SMALL, P["muted"], mono=True)
    button(s, gx + 8, y + 28, "Promote to [regions]", icon="regions")
    by = H - STATUS - 32
    s.line(gx, by - 4, gx + gw, by - 4)
    button(s, gx + 8, by, "TOML", icon="case")
    button(s, gx + gw - 8 - 108 - 6 - 72, by, "Check", icon="check", w=72)
    button(s, gx + gw - 8 - 108, by, "Run…", "accent", icon="play", w=108)
    hy = TOP + HEAD + ROW * 10
    s.rect(x0 + tw0 + 4, hy + 1, 196, 22, P["tip"], r=3)
    s.text(x0 + tw0 + 12, hy + 16, "cut_plane x = 1.4, |y| < 2", SMALL, P["on_tip"], mono=True)
    return s.str()


def histories():
    """Synthetic histories of examples/cylinder from its reference results."""
    f = 0.164 * 0.2
    ts = [150 + 0.25 * k for k in range(0, 251)]
    uy = [(t, 0.118 * math.sin(2 * math.pi * f * t) + 0.004 * math.sin(2 * math.pi * 3 * f * t + 0.4)) for t in ts]
    cl = [(t, 0.33 * math.sin(2 * math.pi * f * t - 0.9)) for t in ts]
    cd = [(t, 1.36 + 0.032 * math.sin(4 * math.pi * f * t - 0.5)) for t in ts]
    dt = [(t, 1.0e-3 * (1.880 + 0.012 * math.sin(4 * math.pi * f * t))) for t in ts]
    mach = [(t, 0.300 + 0.006 * math.sin(4 * math.pi * f * t + 1.2)) for t in ts]
    minp = [(t, 0.6512 - 0.004 * math.sin(4 * math.pi * f * t + 0.3)) for t in ts]
    return dict(uy=uy, cl=cl, cd=cd, dt=dt, mach=mach, minp=minp)


def run_monitor(P):
    s = Svg(P)
    z = P["zones"]
    hist = histories()
    chrome(s, [("Runs", None), ("re100-fine", ("server", "hpc · Slurm"))], "runs",
           [("hpc", P["ok"], "server"), ("job 48213 · gpu-a100 · 1 node · 1 GPU", None, None),
            ("step 108 523", None, None), ("t 212.60 / 400", None, None), ("wall 4 min 31 s", None, None),
            ("ETA 3 min 58 s", None, None), ("mallard.jsonl +1.2 kB/s", None, "refresh"),
            ("viewer cpu-small · idle stop 15 min", None, "eye")])
    x0, tw0 = 44, 248
    head(s, x0, TOP, tw0, "Runs", "", ("plus", "filter"))
    tree(s, x0, TOP + HEAD, tw0, [(0, "server", "hpc", True, None, ""),
                                  (1, "play", "re100-fine", None, ("accent", "53%"), ""),
                                  (1, "check", "re100-coarse", None, None, "done"),
                                  (0, "k8s", "k8s-lab", True, None, ""),
                                  (1, "pause", "re200-les", None, ("muted", "queued"), ""),
                                  (0, "laptop", "Local", True, None, ""),
                                  (1, "check", "re100-quick", None, None, "done")], "re100-fine", hovered="re100-coarse")
    s.line(x0 + tw0, TOP, x0 + tw0, H - STATUS)
    gy0 = TOP + HEAD + ROW * 7 + 8
    s.line(x0, gy0, x0 + tw0, gy0)
    head(s, x0, gy0, tw0, "Run", "re100-fine")
    grid(s, x0, gy0 + HEAD, tw0, [
        ("run_id", "3a256c99", ""), ("mallard", "0.11.0-cuda", ""), ("input", "9b1f3c02", ""), ("mesh", "384 × 128 quads", ""),
        ("cells", "49 152", ""), ("ranks", "1 · A100 40GB", ""), ("started", "11-20 14:01", ""), ("restart of", "—", ""),
        ("directory", "…/runs/re100-fine", ""), ("job script", "run.sbatch", "", "link")], 84)

    cx0 = x0 + tw0
    vw = 440
    cw = W - cx0 - vw
    y = TOP
    s.rect(cx0, y, cw, 40, P["bg"])
    s.text(cx0 + 10, y + 25, "re100-fine", 14, weight=600, mono=True)
    w = 66
    s.rect(cx0 + 100, y + 12, w, 18, P["sel"], r=9)
    s.circle(cx0 + 110, y + 21, 3, P["accent"])
    s.text(cx0 + 118, y + 25, "running", SMALL, P["accent"], 600)
    bx = cx0 + cw - 10
    for lab, kind, icon in (("Stop", "danger", "stop"), ("Set…", "plain", "sliders"), ("Checkpoint", "plain", "bookmark")):
        w = 16 + tw(lab) + 18
        bx -= w
        button(s, bx, y + 8, lab, kind, icon, w)
        bx -= 6
    s.line(cx0, y + 40, cx0 + cw, y + 40)
    y += 40
    s.rect(cx0 + 10, y + 9, cw - 20, 4, P["border"], r=2)
    s.rect(cx0 + 10, y + 9, (cw - 20) * 0.5315, 4, P["accent"], r=2)
    s.text(cx0 + 10, y + 29, "t 212.60 of 400 (t_stop)  ·  53.2 %  ·  2.41 ms/step  ·  ETA 3 min 58 s", SMALL, P["muted"],
           mono=True)
    y += 36
    s.line(cx0, y, cx0 + cw, y)
    kv = [("step", "108 523", ""), ("dt", "1.889e-3", "nd"), ("cfl", "1.000", ""), ("cells_per_second", "2.04e7", "1/s"),
          ("memory_device", "1.84", "GB"), ("troubled_cells", "0", ""),
          ("min_rho", "0.9617", "nd"), ("min_p", "0.6508", "nd"), ("max_mach", "0.304", ""),
          ("C_D (mean)", "1.361", ""), ("C_L (rms)", "0.233", ""), ("St (probe)", "0.1647", "")]
    cols = 6
    kw = cw / cols
    for i, (k, v, u) in enumerate(kv):
        kx, ky = cx0 + kw * (i % cols), y + 42 * (i // cols)
        s.text(kx + 10, ky + 15, k, SMALL, P["muted"], mono=True)
        s.text(kx + 10, ky + 34, v, 15, weight=500, mono=True)
        if u:
            s.text(kx + kw - 10, ky + 34, u, SMALL, P["muted"], mono=True, anchor="end")
        if i % cols:
            s.line(kx, ky + 4, kx, ky + 38)
    y += 84
    s.line(cx0, y, cx0 + cw, y)
    ch = 196
    hw = cw / 2
    xl = (150, 212.6)
    plot(s, cx0, y, hw, ch, "Probe wake_point", "t (nd)", "U_Y (nd)", [("U_Y", z[0], hist["uy"])], (-0.15, 0.15), xl,
         note="x = 2.5, y = 0")
    s.line(cx0 + hw, y, cx0 + hw, y + 2 * ch)
    plot(s, cx0 + hw, y, hw, ch, "Forces cylinder", "t (nd)", "coefficient", [("C_D", z[1], hist["cd"]),
                                                                             ("C_L", z[2], hist["cl"])], (-0.5, 1.5), xl)
    s.line(cx0, y + ch, cx0 + cw, y + ch)
    plot(s, cx0, y + ch, hw, ch, "Time step", "t (nd)", "dt (nd)", [("dt", z[3], hist["dt"])], (1.86e-3, 1.90e-3), xl,
         note="cfl 1.000, SSPRK3")
    plot(s, cx0 + hw, y + ch, hw, ch, "Health", "t (nd)", "value", [("max_mach", z[4], hist["mach"]),
                                                                   ("min_p", z[5], hist["minp"])], (0.25, 0.7), xl)
    y += 2 * ch
    s.line(cx0, y, cx0 + cw, y)
    head(s, cx0, y, cw, "Events", "mallard.jsonl", ("filter",))
    ev = [("14:05:58", "output", "bookmark", "solut/cylinder.vtkhdf  ·  snapshot 212", "4.4 MB"),
          ("14:04:40", "set", "sliders", "t_stop = 400 (was 160)", ""),
          ("14:03:12", "checkpoint", "bookmark", "solut/restart_000101000.restart", "2.1 MB"),
          ("14:01:08", "start", "play", "1 rank · gpu-a100 · v0.11.0 · cuda", "")]
    for i, (t, k, icon, m, sz) in enumerate(ev):
        ey = y + HEAD + ROW * i
        if ey + ROW > H - STATUS:
            break
        s.text(cx0 + 10, ey + 16, t, SMALL, P["muted"], mono=True)
        s.icon(icon, cx0 + 72, ey + 5, P["accent"] if k in ("output", "checkpoint") else P["icon"], 14)
        s.text(cx0 + 92, ey + 16, k, TEXT, weight=500)
        s.text(cx0 + 172, ey + 16, m, SMALL, mono=True)
        s.text(cx0 + cw - 10, ey + 16, sz, SMALL, P["muted"], mono=True, anchor="end")
        s.line(cx0, ey + ROW, cx0 + cw, ey + ROW)

    vx = W - vw
    s.line(vx, TOP, vx, H - STATUS)
    tabs(s, vx, TOP, vw, ["3D view", "Snapshots", "Log"], "3D view")
    gy = grid(s, vx, TOP + TABS, vw, [("field", "VORTICITY_Z", "1/t (nd)", "select"),
                                      ("snapshot", "t 212.0 · step 108 200", "", "select"),
                                      ("colormap", "vik · symmetric ±2", "", "select")], 84)
    iy, ih = gy, 300
    field_view(s, vx, iy, vw, ih, -1.2, 9.8, -2.6, 2.6, 132, 72, (-2.0, 2.0))
    s.rect(vx + 8, iy + 8, 150, 20, P["bg"], P["border"], r=3)
    s.icon("server", vx + 13, iy + 11, P["icon"], 14)
    s.text(vx + 32, iy + 22, "rendered on hpc", SMALL, P["muted"])
    s.line(vx, iy + ih, W, iy + ih)
    colorbar(s, vx + 16, iy + ih + 26, vw - 32, "VORTICITY_Z  (1/t, nd)", (-2.0, 2.0), [-2, -1, 0, 1, 2])
    gy2 = iy + ih + 66
    s.line(vx, gy2, W, gy2)
    head(s, vx, gy2, vw, "Viewer", "job 48219 · cpu-small")
    grid(s, vx, gy2 + HEAD, vw, [("mode", "image · 1280×720 · JPEG 85", ""), ("round trip", "28", "ms"),
                                 ("frame rate", "24", "fps"), ("data", "nothing copied", ""),
                                 ("idle stop", "in 15 min", "")], 84)
    return s.str()


def data_manager(P):
    s = Svg(P)
    chrome(s, [("Data", None), ("re100-fine", ("server", "hpc · Slurm"))], "data",
           [("hpc", P["ok"], "server"), ("direct route hpc → k8s-lab: yes (agent forwarding)", None, "link"),
            ("cache index: 3 locations · rebuilt 2 min ago", None, None), ("1 transfer running", P["accent"], "refresh"),
            ("1 paused", P["warn"], "pause")])
    x0, tw0 = 44, 248
    head(s, x0, TOP, tw0, "Locations", "", ("refresh",))
    tree(s, x0, TOP + HEAD, tw0, [(0, "server", "hpc", True, None, "home"),
                                  (1, "data", "re100-fine", None, None, "1.13 GB"),
                                  (1, "data", "re100-coarse", None, None, "212 MB"),
                                  (0, "laptop", "Local", True, None, ""),
                                  (1, "data", "re100-fine ", None, ("accent", "41%"), "318 MB"),
                                  (1, "data", "re100-quick", None, None, "9 MB"),
                                  (0, "k8s", "k8s-lab", True, None, "pvc mallard-data"),
                                  (1, "data", "re100-fine  ", None, None, "96 MB"),
                                  (1, "data", "re200-les", None, None, "home")], "re100-fine", hovered="re100-coarse")
    s.line(x0 + tw0, TOP, x0 + tw0, H - STATUS)
    gy0 = TOP + HEAD + ROW * 9 + 8
    s.line(x0, gy0, x0 + tw0, gy0)
    head(s, x0, gy0, tw0, "Run", "re100-fine")
    grid(s, x0, gy0 + HEAD, tw0, [("run_id", "3a256c99", ""), ("home", "hpc", ""), ("objects", "226", ""),
                                  ("streams", "7", ""), ("total", "1.13", "GB"), ("snapshot", "4.4", "MB"),
                                  ("restart", "2.1", "MB")], 84)

    px = W - 360
    tx, tw2 = x0 + tw0, px - x0 - tw0
    tabs(s, tx, TOP, tw2, ["Files", "Transfers", "History"], "Files")
    fy = TOP + TABS
    bx = tx + 6
    for i, lab in enumerate(["All 233", "Snapshots 212", "Restarts 9", "Streams 7", "Inputs 5"]):
        w = 12 + tw(lab, SMALL)
        on = i == 0
        s.rect(bx, fy + 4, w, 22, P["sel"] if on else P["bg"], None if on else P["border"], r=4)
        s.text(bx + 6, fy + 19, lab, SMALL, P["accent"] if on else P["muted"], 500)
        bx += w + 4
    s.rect(tx + tw2 - 208, fy + 4, 200, 22, P["bg"], P["border"], r=4)
    s.icon("filter", tx + tw2 - 202, fy + 8, size=14)
    s.text(tx + tw2 - 184, fy + 19, "t ≥ 200 and t ≤ 212", SMALL, P["fg"], mono=True)
    fy += 30
    s.line(tx, fy, tx + tw2, fy)
    cols = [("", 8), ("path", 32), ("kind", 300), ("t (nd)", 372), ("size", 450), ("sha256", 512), ("hpc", 612),
            ("local", 664), ("k8s-lab", 718)]
    s.rect(tx, fy, tw2, ROW, P["panel"])
    for name, cx in cols:
        s.text(tx + cx, fy + 16, name, SMALL, P["muted"], 600, mono=name in ("sha256", "t (nd)"))
    fy += ROW
    s.line(tx, fy, tx + tw2, fy)
    rows = []
    for k, tt in enumerate(range(200, 213)):
        h = f"{(tt * 2654435761) % 16 ** 8:08x}"
        loc = "p" if tt <= 203 else ("part" if tt == 204 else "")
        k8 = "p" if tt in (200, 205, 210) else ""
        rows.append((f"solut/cylinder_{tt:06d}.vtkhdf", "snapshot", f"{tt:.1f}", "4.4 MB", h, "p", loc, k8))
    rows += [("solut/restart_000101000.restart", "restart", "197.3", "2.1 MB", "a19e77f0", "p", "", "p"),
             ("solut/restart_000108000.restart", "restart", "211.6", "2.1 MB", "4f6d2a90", "p", "", ""),
             ("mallard.jsonl", "stream", "0 … 212.6", "1.9 MB", "prefix", "p", "short", "short")]
    sel = {5, 6, 7, 8, 9, 10, 11, 12}
    for i, (f, k, t, size, h, a, b, c) in enumerate(rows):
        y = fy + ROW * i
        if y + ROW > H - STATUS - 26:
            break
        if i in sel:
            s.rect(tx, y, tw2, ROW, P["sel"])
        elif i == 2:
            s.rect(tx, y, tw2, ROW, P["hover"])
        on = i in sel
        s.rect(tx + 8, y + 6, 12, 12, P["accent"] if on else P["bg"], P["accent"] if on else P["icon"], r=2)
        if on:
            s.path(f"M{tx + 10.5} {y + 12} l2.5 2.5 l5 -5", P["on_accent"], 1.6)
        s.text(tx + 32, y + 16, f, TEXT, mono=True)
        s.text(tx + 300, y + 16, k, TEXT, P["muted"])
        s.text(tx + 372, y + 16, t, SMALL, mono=True)
        s.text(tx + 500, y + 16, size, SMALL, mono=True, anchor="end")
        s.text(tx + 512, y + 16, h, SMALL, P["muted"], mono=True)
        for cx, st in ((612, a), (664, b), (718, c)):
            dx, dy = tx + cx + 6, y + 12
            if st == "p":
                s.circle(dx, dy, 4, P["ok"])
            elif st == "part":
                s.add(f'<path d="M{dx} {dy - 4} a4 4 0 0 1 0 8 z" fill="{P["accent"]}"/>')
                s.circle(dx, dy, 4, "none", P["accent"], 1.2)
                s.text(dx + 8, dy + 4, "41%", SMALL - 1, P["accent"], mono=True)
            elif st == "short":
                s.add(f'<path d="M{dx} {dy - 4} a4 4 0 0 1 0 8 z" fill="{P["ok"]}"/>')
                s.circle(dx, dy, 4, "none", P["ok"], 1.2)
            else:
                s.circle(dx, dy, 4, "none", P["faint"], 1)
        s.line(tx, y + ROW, tx + tw2, y + ROW)
    y = fy + ROW * 2
    h2 = rows[2][4]
    s.rect(tx + tw2 - 296, y + ROW + 2, 288, 40, P["tip"], r=4)
    s.text(tx + tw2 - 288, y + ROW + 17, f"sha256 {h2}3e91c0…  4 417 280 B", SMALL, P["on_tip"], mono=True)
    s.text(tx + tw2 - 288, y + ROW + 33, "on hpc and local · from the run log", SMALL, P["on_tip"], mono=True)
    ly = H - STATUS - 26
    s.rect(tx, ly, tw2, 26, P["panel"])
    s.line(tx, ly, tx + tw2, ly)
    s.circle(tx + 14, ly + 13, 4, P["ok"])
    s.text(tx + 22, ly + 17, "object (SHA-256 matches)", SMALL, P["muted"])
    s.add(f'<path d="M{tx + 184} {ly + 9} a4 4 0 0 1 0 8 z" fill="{P["ok"]}"/>')
    s.circle(tx + 184, ly + 13, 4, "none", P["ok"], 1.2)
    s.text(tx + 192, ly + 17, "stream prefix (shorter)", SMALL, P["muted"])
    s.add(f'<path d="M{tx + 344} {ly + 9} a4 4 0 0 1 0 8 z" fill="{P["accent"]}"/>')
    s.circle(tx + 344, ly + 13, 4, "none", P["accent"], 1.2)
    s.text(tx + 352, ly + 17, "partial (resumes from any holder)", SMALL, P["muted"])
    s.circle(tx + 552, ly + 13, 4, "none", P["faint"], 1)
    s.text(tx + 560, ly + 17, "absent", SMALL, P["muted"])
    s.text(tx + tw2 - 8, ly + 17, "8 selected · 35.3 MB", SMALL, P["fg"], 600, mono=True, anchor="end")

    pw = W - px
    s.line(px, TOP, px, H - STATUS)
    head(s, px, TOP, pw, "Transfer", "8 objects", ("more",))
    y = grid(s, px, TOP + HEAD, pw, [
        ("group", "SELECTION"),
        ("objects", "8 snapshots", ""), ("size", "35.3", "MB"), ("at destination", "0", ""),
        ("group", "ROUTE"),
        ("from", "hpc", "", "select"), ("to", "local", "", "select"),
        ("method", "rsync over SSH", "", "select"), ("estimate", "1.6 s at 22.4 MB/s", ""),
        ("group", "OTHER ROUTES", "estimates"),
        ("k8s-lab → local", "4.1 s at 8.6 MB/s", ""), ("view remotely", "no copy · viewer", ""),
    ], 116)
    button(s, px + 8, y + 8, "Estimate routes", icon="refresh")
    button(s, px + pw - 8 - 132, y + 8, "Copy 35.3 MB", "accent", "output", 132)
    y += 40
    head(s, px, y, pw, "Running", "this machine")
    trs = [("hpc → local", "snapshot 204", 0.41, "1.8 / 4.4 MB", "21.9 MB/s", "accent"),
           ("hpc → k8s-lab", "restart_000108000", 0.78, "1.6 / 2.1 MB", "direct · 9.2 MB/s", "accent"),
           ("k8s-lab → local", "mallard.jsonl tail", 0.0, "0 / 0.6 MB", "paused", "warn")]
    yy = y + HEAD
    for route, what, frac, sz, rate, kind in trs:
        s.text(px + 8, yy + 16, route, SMALL, weight=600, mono=True)
        s.text(px + pw - 8, yy + 16, rate, SMALL, P["warn"] if kind == "warn" else P["muted"], mono=True, anchor="end")
        s.text(px + 8, yy + 34, what, SMALL, P["muted"], mono=True)
        s.text(px + pw - 8, yy + 34, sz, SMALL, P["muted"], mono=True, anchor="end")
        s.rect(px + 8, yy + 42, pw - 16, 4, P["border"], r=2)
        s.rect(px + 8, yy + 42, (pw - 16) * frac, 4, P["accent"] if kind == "accent" else P["warn"], r=2)
        yy += 54
        s.line(px, yy, W, yy)
    head(s, px, yy, pw, "History", "verified")
    for i, (when, m) in enumerate([("11-20 14:06", "4 objects hpc → local  17.6 MB"),
                                   ("11-20 13:48", "3 objects hpc → k8s-lab  12.9 MB"),
                                   ("11-20 13:40", "inputs local → hpc  1.21 MB")]):
        ry = yy + HEAD + ROW * i
        if ry + ROW > H - STATUS:
            break
        s.icon("check", px + 8, ry + 5, P["ok"], 14)
        s.text(px + 28, ry + 16, when, SMALL, P["muted"], mono=True)
        s.text(px + 112, ry + 16, m, SMALL, mono=True)
        s.line(px, ry + ROW, W, ry + ROW)
    return s.str()


def case_setup_3d(P):
    """examples/sphere_re300: the sphere is enclosed by the far-field zones; a clip plane exposes it."""
    s = Svg(P)
    z = P["zones"]
    chrome(s, [("Case", None), ("boundaries", None), ("sphere", None)], "case",
           [("Local", P["ok"], "laptop"), ("mallard 0.11.0 · 3D · double · Threads", None, None),
            ("850 000 cells · preview 186 000 triangles (decimated surfaces + cut)", None, None),
            ("rendered locally · vtk.js · 60 fps", None, "view"), ("check ok", P["ok"], "check"),
            ("input.toml", None, None)], project="sphere-re300")
    x0, tw0 = 44, 248
    head(s, x0, TOP, tw0, "Case", "input.toml", ("plus", "more"))
    tree(s, x0, TOP + HEAD, tw0, [
        (0, "mesh", "Mesh", True, None, "sphere.msh"),
        (1, None, "prisms", None, None, "10 layers"),
        (1, None, "tetrahedra", None, None, ""),
        (0, "physics", "Physics", False, None, "navier_stokes"),
        (0, "boundary", "Boundaries", True, None, "4"),
        (1, None, "sphere", None, None, "wall_adiabatic"),
        (1, None, "inflow", None, None, "farfield"),
        (1, None, "lateral", None, None, "farfield"),
        (1, None, "outflow", None, None, "farfield"),
        (0, "initial", "Initial state", False, None, "analytical"),
        (0, "numerics", "Numerics", False, None, "MUSCL · HLLC"),
        (0, "regions", "Regions", False, None, "6"),
        (0, "probes", "Probes", False, None, "0"),
        (0, "output", "Output", False, None, "3 writers"),
        (0, "run", "Run", False, None, "t_stop 750")], "sphere", hovered="lateral")
    s.line(x0 + tw0, TOP, x0 + tw0, H - STATUS)

    gx, gw = W - 352, 352
    vx, vw = x0 + tw0, gx - x0 - tw0
    tabs(s, vx, TOP, vw, ["Case view", "input.toml", "Mesh quality", "Check report"], "Case view")
    ty = TOP + TABS
    s.rect(vx, ty, vw, 30, P["bg"])
    bx = vx + 6

    def seg(labels, on, icons=None):
        nonlocal bx
        for i, lab in enumerate(labels):
            ic = icons[i] if icons else None
            w = 12 + tw(lab, SMALL) + (18 if ic else 0)
            act = lab in on
            s.rect(bx, ty + 4, w, 22, P["sel"] if act else P["bg"], None if act else P["border"], r=4)
            if ic:
                s.icon(ic, bx + 5, ty + 7, P["accent"] if act else P["icon"], 14)
            s.text(bx + (23 if ic else 6), ty + 19, lab, SMALL, P["accent"] if act else P["muted"], 500, mono=not ic)
            bx += w + 2
        bx += 8

    seg(["+X", "-X", "+Y", "-Y", "+Z", "-Z", "Iso"], {"Iso"})
    seg(["Persp", "Ortho"], {"Ortho"})
    seg(["2D lock"], set())
    seg(["Clip", "Box", "X-ray", "Isolate", "Selected", "Pick"], {"Clip", "X-ray", "Pick"},
        ["layers", "fit", "eye", "boundary", "filter", "pick"])
    s.line(vx, ty + 30, vx + vw, ty + 30)
    vy, vh = ty + 30, H - STATUS - ty - 30
    s.add(f'<clipPath id="v3"><rect x="{vx}" y="{vy}" width="{vw}" height="{vh}"/></clipPath><g clip-path="url(#v3)">')
    s.rect(vx, vy, vw, vh, P["view"])
    a, e = math.radians(-18), math.radians(18)
    k = 62.0
    ox, oy = vx + vw * 0.36, vy + vh * 0.48

    def pr(x, y, zz):
        xp = x * math.cos(a) + zz * math.sin(a)
        zp = -x * math.sin(a) + zz * math.cos(a)
        yp = y * math.cos(e) - zp * math.sin(e)
        return ox + k * xp, oy - k * yp

    def poly(pts, close=True):
        d = "M" + " L".join(f"{px:.1f} {py:.1f}" for px, py in pts)
        return d + (" Z" if close else "")

    X0, X1, Y0, Y1 = -2.6, 6.2, -2.4, 2.4
    cut = [pr(X0, Y0, 0), pr(X1, Y0, 0), pr(X1, Y1, 0), pr(X0, Y1, 0)]
    back = [pr(X0, Y0, -2.4), pr(X1, Y0, -2.4), pr(X1, Y1, -2.4), pr(X0, Y1, -2.4)]
    s.add(f'<path d="{poly(back)}" fill="{z[2]}" fill-opacity="0.07" stroke="{z[2]}" stroke-dasharray="3 3"/>')
    for i in range(4):
        s.line(*cut[i], *back[i], z[2], 1, "3 3")
    s.add(f'<clipPath id="cutp"><path d="{poly(cut)}"/></clipPath><g clip-path="url(#cutp)">')
    s.add(f'<path d="{poly(cut)}" fill="{P["bg"]}" fill-opacity="0.85"/>')
    nt = 72
    radii = [0.5]
    for i in range(10):
        radii.append(radii[-1] + 0.005 * 1.25 ** i * 1.6)
    while radii[-1] < 9:
        radii.append(radii[-1] * 1.13)
    for ri, rr in enumerate(radii):
        pts = [pr(rr * math.cos(2 * math.pi * j / 144), rr * math.sin(2 * math.pi * j / 144), 0) for j in range(145)]
        s.path(poly(pts, False), P["accent"] if ri <= 10 else P["mesh"], 0.6 if ri > 10 else 0.5)
    for j in range(nt):
        t0 = 2 * math.pi * j / nt
        s.line(*pr(0.5 * math.cos(t0), 0.5 * math.sin(t0), 0), *pr(9 * math.cos(t0), 9 * math.sin(t0), 0), P["mesh"], 0.6)
    for ri in range(10, len(radii) - 1):
        r0, r1 = radii[ri], radii[ri + 1]
        for j in range(nt):
            t0, t1 = 2 * math.pi * j / nt, 2 * math.pi * (j + 1) / nt
            s.line(*pr(r0 * math.cos(t0), r0 * math.sin(t0), 0), *pr(r1 * math.cos(t1), r1 * math.sin(t1), 0), P["mesh"], 0.5)
    s.add("</g>")
    s.add(f'<path d="{poly(cut)}" fill="none" stroke="{P["accent"]}" stroke-width="1.5"/>')
    bowl = [pr(0.5 * math.cos(t), 0.5 * math.sin(t), 0) for t in [2 * math.pi * j / 96 for j in range(97)]]
    s.add(f'<defs><radialGradient id="bw{P["name"]}" cx="0.6" cy="0.4" r="0.7"><stop offset="0" stop-color="{P["accent"]}" '
          f'stop-opacity="0.95"/><stop offset="1" stop-color="{P["accent"]}" stop-opacity="0.45"/></radialGradient></defs>')
    s.add(f'<path d="{poly(bowl)}" fill="url(#bw{P["name"]})" stroke="{P["accent"]}" stroke-width="2"/>')
    for lat in (-0.35, -0.2):
        rr = math.sqrt(0.25 - lat * lat)
        pts = [pr(rr * math.cos(t), rr * math.sin(t), lat) for t in [2 * math.pi * j / 64 for j in range(65)]]
        s.path(poly(pts, False), P["on_accent"], 0.6)
    sx, sy = pr(0.0, 0.0, 0)
    s.path(f"M{sx + 30} {sy - 50} l-24 42", P["fg"], 1)
    s.rect(sx + 30, sy - 70, 262, 34, P["bg"], P["accent"], r=4)
    s.text(sx + 38, sy - 56, "sphere · wall_adiabatic", SMALL, P["accent"], 600, mono=True)
    s.text(sx + 38, sy - 42, "picked through the cut · 31 248 faces", SMALL, P["muted"], mono=True)
    c0 = pr(X0 + 0.6, Y1 - 0.5, 0)
    s.circle(*c0, 6, P["bg"], P["accent"], 2)
    tip = pr(X0 + 0.6, Y1 - 0.5, 1.4)
    s.line(*c0, *tip, P["accent"], 2)
    s.circle(*tip, 4, P["accent"])
    s.text(tip[0] + 8, tip[1] + 4, "clip plane z = 0: drag to move, tip to tilt", SMALL, P["accent"], 600, mono=True)
    for (xx, lab) in ((X0, "inflow side"), (X1, "outflow side")):
        px, py = pr(xx, Y0, 0)
        s.text(px, py + 16, lab, SMALL, P["muted"], mono=True, anchor="middle")
    s.add("</g>")
    ix, iy, iw, ih = vx + 10, vy + vh - 150, 200, 140
    s.rect(ix, iy, iw, ih, P["bg"], P["border"], r=4)
    s.text(ix + 8, iy + 16, "Domain · x-ray 20 %", SMALL, P["muted"], 600)
    kk = 2.4
    ocx, ocy = ix + 80, iy + 80

    def pm(x, y, zz):
        xp = x * math.cos(a) + zz * math.sin(a)
        zp = -x * math.sin(a) + zz * math.cos(a)
        yp = y * math.cos(e) - zp * math.sin(e)
        return ocx + kk * xp, ocy - kk * yp

    B = [(-15, -15, -15), (30, -15, -15), (30, 15, -15), (-15, 15, -15), (-15, -15, 15), (30, -15, 15), (30, 15, 15), (-15, 15, 15)]
    faces = [((0, 3, 7, 4), z[1]), ((1, 2, 6, 5), z[3]), ((0, 1, 5, 4), z[2]), ((3, 2, 6, 7), z[2]), ((0, 1, 2, 3), z[2]),
             ((4, 5, 6, 7), z[2])]
    for f, c in faces:
        s.add(f'<path d="{poly([pm(*B[i]) for i in f])}" fill="{c}" fill-opacity="0.08" stroke="{c}" stroke-width="0.8"/>')
    cp = [pm(-15, -15, 0), pm(30, -15, 0), pm(30, 15, 0), pm(-15, 15, 0)]
    s.add(f'<path d="{poly(cp)}" fill="{P["accent"]}" fill-opacity="0.18" stroke="{P["accent"]}" stroke-width="1"/>')
    s.circle(*pm(0, 0, 0), 2, P["accent"])
    s.path(f"M{ix + iw - 40} {iy + ih - 16} l0 -22 M{ix + iw - 40} {iy + ih - 16} l22 0", P["muted"], 1.2)
    s.text(ix + iw - 14, iy + ih - 12, "x", SMALL, P["muted"], mono=True)
    s.text(ix + iw - 44, iy + ih - 42, "y", SMALL, P["muted"], mono=True)
    s.text(vx + vw - 8, vy + vh - 12, "x 0.31  y -0.18  z -0.40  ·  face 1 204 377", SMALL, P["muted"], mono=True, anchor="end")
    s.text(vx + vw - 8, vy + vh - 28, "drag orbit · shift pan · wheel zoom · F fit", SMALL, P["muted"], anchor="end")

    s.line(gx, TOP, gx, H - STATUS)
    head(s, gx, TOP, gw, "Zones", "in sync with the view", ("eye",))
    zl = [("sphere", z[0], "wall_adiabatic", "31 248", "eye", True),
          ("inflow", z[1], "farfield", "1 104", "xray", False),
          ("lateral", z[2], "farfield", "4 416", "xray", False),
          ("outflow", z[3], "farfield", "1 104", "hidden", False)]
    for i, (n, c, bc, faces_, vis, selz) in enumerate(zl):
        ry = TOP + HEAD + ROW * i
        if selz:
            s.rect(gx, ry, gw, ROW, P["sel"])
            s.rect(gx, ry, 2, ROW, P["accent"])
        s.rect(gx + 10, ry + 7, 10, 10, c, r=2)
        s.text(gx + 28, ry + 16, n, TEXT, P["accent"] if selz else P["fg"], 600 if selz else 400, mono=True)
        s.text(gx + 108, ry + 16, bc, SMALL, P["muted"])
        s.text(gx + 252, ry + 16, faces_, SMALL, P["muted"], mono=True, anchor="end")
        if vis == "eye":
            s.icon("eye", gx + gw - 70, ry + 5, P["icon"], 14)
        elif vis == "xray":
            s.icon("eye", gx + gw - 70, ry + 5, P["faint"], 14)
            s.text(gx + gw - 8, ry + 16, "x-ray", SMALL - 1, P["muted"], mono=True, anchor="end")
        else:
            s.icon("eye", gx + gw - 70, ry + 5, P["faint"], 14)
            s.line(gx + gw - 70, ry + 19, gx + gw - 56, ry + 5, P["faint"], 1.4)
            s.text(gx + gw - 8, ry + 16, "hidden", SMALL - 1, P["muted"], mono=True, anchor="end")
        s.line(gx, ry + ROW, gx + gw, ry + ROW)
    y = TOP + HEAD + ROW * 4
    head(s, gx, y, gw, "Boundary", "boundaries[0]")
    y = grid(s, gx, y + HEAD, gw, [
        ("group", "CONDITION"),
        ("type", "wall_adiabatic", "", "select"),
        ("wall_model", "none", "", "select"),
        ("group", "ZONE"),
        ("faces", "31 248 tri", ""), ("area", "3.1416", "nd²"), ("enclosed by", "inflow lateral outflow", ""),
    ], 96)
    head(s, gx, y, gw, "Section", "1 plane", ("plus",))
    grid(s, gx, y + HEAD, gw, [
        ("group", "CLIP PLANE 1"),
        ("origin", "[0.0, 0.0, 0.0]", "nd"), ("normal", "[0.0, 0.0, 1.0]", ""),
        ("keep", "z < 0 side", "", "select"), ("snap", "axis · 15°", "", "select"),
        ("group", "OTHERS"),
        ("box clip", "off", "", "select"), ("x-ray", "20 %", ""), ("pick", "through the cut", "", "select"),
        ("show", "all zones", "", "select"),
    ], 116)
    return s.str()


def figures():
    OUT.mkdir(parents=True, exist_ok=True)
    for P in (LIGHT, DARK):
        for name, draw in (("case-setup", case_setup), ("case-setup-3d", case_setup_3d), ("run-monitor", run_monitor),
                           ("data-manager", data_manager)):
            (OUT / f"gui-{name}-{P['name']}.svg").write_text(draw(P))


if __name__ == "__main__":
    figures()
