#!/usr/bin/env python3
"""Profile files for [[boundaries.data]], [[initialize.data]] and turbulence.profile.

    profiles.py law power OUT.csv --delta 0.02 --u-inf 30 [--exponent 7]
    profiles.py law poiseuille OUT.csv --half-height 0.05 --u-max 2
    profiles.py law spalding OUT.csv --delta 0.02 --u-inf 30 --u-tau 1.2 --nu 1.5e-5
    profiles.py describe FILE.csv [-o OUT.csv] [--coordinates d] [--at n_m] [--unit d=mm --unit u=m/s]
                        [--reference length=0.02 --reference velocity=30] [--source TEXT]
    profiles.py transform FILE.csv OUT.csv --scale length=0.05 --scale velocity=12
    profiles.py show FILE.csv [--plot OUT.png]

A profile file is a CSV file whose comment line "# mallard-profile {...}"
before the header describes it: the defaults of its entry's keys
(coordinates, at, units, reference) and where it comes from (source, and the
transforms that made it). See docs/guide/boundary_conditions.md#profile-files-metadata-and-rescaling.

law writes an analytic profile in the wall distance n_m [m] (read at the
inlet's n_m) with the velocity u [m/s]: the power law U (n / delta)^(1/N), the
developed laminar profile of a channel or pipe u_max (1 - (1 - n / h)^2), or
Spalding's law of the wall (kappa = 0.41, B = 5.0) up to the free stream.
describe adds or replaces the metadata of any CSV file. transform rescales a
described file as `scale` does in the input (each column by its dimension:
velocities by the velocity ratio, Reynolds stresses by its square, pressures
as rho U^2) and writes the new scales as its reference, appending the step
to its transforms. show prints the metadata and each column's range and
checks the file (a header, numbers, a coordinate that increases); --plot
draws each column against the first coordinate.
"""
import argparse
import json
import math
import sys

PREFIX = "# mallard-profile "
SCALES = ["length", "velocity", "density", "temperature"]

# Dimension exponents (mass, length, time, temperature, amount) of the units Mallard reads
BASE = {
    "m": (0, 1, 0, 0, 0), "s": (0, 0, 1, 0, 0), "g": (1, 0, 0, 0, 0), "K": (0, 0, 0, 1, 0), "mol": (0, 0, 0, 0, 1),
    "Pa": (1, -1, -2, 0, 0), "N": (1, 1, -2, 0, 0), "J": (1, 2, -2, 0, 0), "W": (1, 2, -3, 0, 0),
    "bar": (1, -1, -2, 0, 0), "atm": (1, -1, -2, 0, 0), "psi": (1, -1, -2, 0, 0), "Torr": (1, -1, -2, 0, 0),
    "L": (0, 3, 0, 0, 0), "min": (0, 0, 1, 0, 0), "h": (0, 0, 1, 0, 0), "in": (0, 1, 0, 0, 0), "ft": (0, 1, 0, 0, 0),
    "lb": (1, 0, 0, 0, 0), "lbf": (1, 1, -2, 0, 0), "cal": (1, 2, -2, 0, 0), "Hz": (0, 0, -1, 0, 0),
    "degR": (0, 0, 0, 1, 0), "rad": (0, 0, 0, 0, 0), "deg": (0, 0, 0, 0, 0),
}
AFFINE = {"degC", "degF"}
PREFIXES = "YZEPTGMkhdcmunpfazyµ"


def dimension(unit):
    """The dimension exponents of a unit as Mallard writes them ("m/s", "m^2/s^2", "kg m^-3", "1")."""
    unit = unit.strip()
    if unit in ("", "1"):
        return (0, 0, 0, 0, 0)
    if unit in AFFINE:
        return (0, 0, 0, 1, 0)
    total = [0] * 5

    def add(text, sign):
        for token in text.replace("*", " ").replace("(", " ").replace(")", " ").split():
            name, _, power = token.partition("^")
            p = sign * (int(power) if power else 1)
            base = BASE.get(name)
            if base is None and len(name) > 1 and name[0] in PREFIXES:
                base = BASE.get(name[1:])
            if base is None:
                raise ValueError(f"unknown unit {token} in {unit!r}")
            for i in range(5):
                total[i] += p * base[i]

    numerator, _, denominator = unit.partition("/")
    add(numerator if numerator.strip() != "1" else "", 1)
    add(denominator, -1)
    return tuple(total)


def read(path):
    """The metadata (a dict), the header and the rows of a CSV file."""
    metadata, header, rows = {}, None, []
    with open(path) as f:
        for number, line in enumerate(f, 1):
            text = line.strip()
            if not text:
                continue
            if text.startswith("#"):
                if header is None and text.startswith(PREFIX.strip()) and not metadata:
                    metadata = json.loads(text[len(PREFIX):])
                continue
            items = [item.strip().strip("\"'") for item in (text.split(",") if "," in text else text.split())]
            if header is None:
                header = items
                continue
            if len(items) != len(header):
                raise ValueError(f"{path}:{number}: {len(items)} values for {len(header)} columns")
            rows.append([float(v) for v in items])
    if header is None or not rows:
        raise ValueError(f"{path} has no rows")
    return metadata, header, rows


def write(path, metadata, header, rows):
    with open(path, "w") as f:
        f.write(PREFIX + json.dumps(metadata) + "\n")
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(repr(float(v)) for v in row) + "\n")


def wall_points(extent, n=64):
    """Wall distances from 0 to extent, clustered at the wall."""
    return [extent * (1 - math.cos(0.5 * math.pi * k / (n - 1))) for k in range(n)]


def spalding_u_plus(y_plus, kappa=0.41, b=5.0):
    """u+ of Spalding's law at y+, by Newton's method (the law is explicit in y+)."""
    u = min(y_plus, math.log(1 + kappa * y_plus) / kappa + b) if y_plus > 0 else 0.0
    for _ in range(100):
        k = kappa * u
        f = u + math.exp(-kappa * b) * (math.exp(k) - 1 - k - k * k / 2 - k ** 3 / 6) - y_plus
        df = 1 + math.exp(-kappa * b) * kappa * (math.exp(k) - 1 - k - k * k / 2)
        step = f / df
        u -= step
        if abs(step) < 1e-14 * max(1.0, abs(u)):
            break
    return u


def law(args):
    if args.law == "power":
        n = wall_points(args.delta * 1.25)
        u = [args.u_inf * min(x / args.delta, 1.0) ** (1.0 / args.exponent) for x in n]
        source, reference = f"power law u = U (n / delta)^(1/{args.exponent})", {"length": args.delta, "velocity": args.u_inf}
    elif args.law == "poiseuille":
        n = wall_points(args.half_height)
        u = [args.u_max * (1 - (1 - x / args.half_height) ** 2) for x in n]
        source, reference = "laminar channel or pipe u = u_max (1 - (1 - n / h)^2)", {"length": args.half_height, "velocity": args.u_max}
    else:
        n = wall_points(args.delta * 1.25)
        u = [min(args.u_tau * spalding_u_plus(x * args.u_tau / args.nu), args.u_inf) for x in n]
        source = f"Spalding's law of the wall, u_tau = {args.u_tau} m/s, nu = {args.nu} m^2/s, capped at U"
        reference = {"length": args.delta, "velocity": args.u_inf}
    metadata = {"coordinates": ["n_m"], "at": ["n_m"], "units": {"n_m": "m", "u": "m/s"}, "reference": reference,
                "source": source, "transforms": []}
    write(args.out, metadata, ["n_m", "u"], list(zip(n, u)))


def pairs(items, what):
    out = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep:
            sys.exit(f"--{what} takes NAME=VALUE, not {item!r}")
        out[key.strip()] = value.strip()
    return out


def describe(args):
    metadata, header, rows = read(args.file)
    if args.coordinates:
        metadata["coordinates"] = args.coordinates
    if args.at:
        metadata["at"] = args.at
    units = pairs(args.unit, "unit")
    for column, unit in units.items():
        if column not in header:
            sys.exit(f"{args.file} has no column {column}")
        dimension(unit)
        metadata.setdefault("units", {})[column] = unit
    for name, value in pairs(args.reference, "reference").items():
        if name not in SCALES:
            sys.exit(f"--reference {name}: the scales are {', '.join(SCALES)}")
        metadata.setdefault("reference", {})[name] = float(value)
    if args.source:
        metadata["source"] = args.source
    metadata.setdefault("transforms", [])
    write(args.out or args.file, metadata, header, rows)


def factors(metadata, scale):
    """Each column's factor, as the solver's scale computes it."""
    reference = metadata.get("reference", {})
    f = {}
    for name in SCALES:
        if name in scale:
            if name not in reference or not isinstance(reference[name], (int, float)):
                sys.exit(f"the file gives no reference {name} (a number in SI) to rescale from")
            f[name] = scale[name] / reference[name]
    fl, fu, fr, ft = (f.get(name, 1.0) for name in SCALES)
    f_mass, f_time = fr * fl ** 3, fl / fu
    out = {}
    for column, unit in metadata.get("units", {}).items():
        if unit.strip() in AFFINE and "temperature" in f:
            sys.exit(f"column {column} is in {unit}: give it in K to rescale its temperature")
        m, length, time, theta, amount = dimension(unit)
        out[column] = f_mass ** (m + amount) * fl ** length * f_time ** time * ft ** theta
    return f, out


def transform(args):
    metadata, header, rows = read(args.file)
    scale = {name: float(value) for name, value in pairs(args.scale, "scale").items()}
    unknown = set(scale) - set(SCALES)
    if unknown:
        sys.exit(f"--scale {', '.join(sorted(unknown))}: the scales are {', '.join(SCALES)}")
    f, by_column = factors(metadata, scale)
    missing = [c for c in header if c not in by_column]
    if missing:
        sys.exit(f"columns without a unit (give \"1\" if dimensionless): {', '.join(missing)}")
    rows = [[v * by_column[c] for v, c in zip(row, header)] for row in rows]
    for name in f:
        metadata.setdefault("reference", {})[name] = scale[name]
    step = "scale " + ", ".join(f"{name} x{f[name]:.6g}" for name in f)
    metadata["transforms"] = metadata.get("transforms", []) + [step]
    write(args.out, metadata, header, rows)


def show(args):
    metadata, header, rows = read(args.file)
    print(f"{args.file}: {len(rows)} rows, columns {', '.join(header)}")
    for key in ("source", "coordinates", "at", "units", "reference", "transforms"):
        if key in metadata:
            print(f"  {key}: {json.dumps(metadata[key])}")
    unknown = set(metadata) - {"source", "coordinates", "at", "units", "reference", "transforms"}
    problems = [f"unknown metadata key {k}" for k in sorted(unknown)]
    for column, unit in metadata.get("units", {}).items():
        if column not in header:
            problems.append(f"units: no column {column}")
        try:
            dimension(unit)
        except ValueError as e:
            problems.append(str(e))
    for j, name in enumerate(header):
        values = [row[j] for row in rows]
        print(f"  {name}: {min(values):.6g} to {max(values):.6g}")
    coordinates = metadata.get("coordinates", header[:1])
    if len(coordinates) == 1 and coordinates[0] in header:
        c = [row[header.index(coordinates[0])] for row in rows]
        if any(b <= a for a, b in zip(c, c[1:])):
            problems.append(f"the coordinate {coordinates[0]} does not increase from row to row")
    for p in problems:
        print("  problem: " + p)
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = header.index(coordinates[0]) if coordinates[0] in header else 0
        fig, ax = plt.subplots(figsize=(5, 4))
        for j, name in enumerate(header):
            if j != x:
                ax.plot([row[j] for row in rows], [row[x] for row in rows], label=name)
        ax.set_ylabel(header[x])
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot)
    return 1 if problems else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("law", help="an analytic profile in the wall distance n_m")
    p.add_argument("law", choices=["power", "poiseuille", "spalding"])
    p.add_argument("out")
    p.add_argument("--delta", type=float, help="boundary-layer thickness [m] (power, spalding)")
    p.add_argument("--u-inf", type=float, help="free stream [m/s] (power, spalding)")
    p.add_argument("--exponent", type=float, default=7.0, help="N of the power law (default 7)")
    p.add_argument("--half-height", type=float, help="channel half height or pipe radius [m] (poiseuille)")
    p.add_argument("--u-max", type=float, help="centerline velocity [m/s] (poiseuille)")
    p.add_argument("--u-tau", type=float, help="friction velocity [m/s] (spalding)")
    p.add_argument("--nu", type=float, help="kinematic viscosity [m^2/s] (spalding)")
    p = sub.add_parser("describe", help="add or replace a CSV file's metadata")
    p.add_argument("file")
    p.add_argument("-o", "--out")
    p.add_argument("--coordinates", nargs="+")
    p.add_argument("--at", nargs="+")
    p.add_argument("--unit", action="append", metavar="COLUMN=UNIT")
    p.add_argument("--reference", action="append", metavar="SCALE=VALUE")
    p.add_argument("--source")
    p = sub.add_parser("transform", help="rescale a described file")
    p.add_argument("file")
    p.add_argument("out")
    p.add_argument("--scale", action="append", required=True, metavar="SCALE=VALUE")
    p = sub.add_parser("show", help="print and check a profile file")
    p.add_argument("file")
    p.add_argument("--plot")
    args = parser.parse_args()
    if args.command == "law":
        needs = {"power": ["delta", "u_inf"], "poiseuille": ["half_height", "u_max"],
                 "spalding": ["delta", "u_inf", "u_tau", "nu"]}[args.law]
        missing = [n for n in needs if getattr(args, n) is None]
        if missing:
            parser.error(f"law {args.law} needs " + ", ".join("--" + n.replace("_", "-") for n in missing))
        law(args)
    elif args.command == "describe":
        describe(args)
    elif args.command == "transform":
        transform(args)
    else:
        return show(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
