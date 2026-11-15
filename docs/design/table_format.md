# Table format for tabulated chemistry (`.mtab`, version 1)

Status: specified (#631). The contract between Mallard and the programs that
write tables for it. Mallard reads tables and does not generate them: tables are
written by an external generator library, **`<generator, name TBD>`**, a
separate project that starts from FPVgen (IhmeGroup/FPVgen). Any other program
that writes this format works too. The reasons for the format are in
[flamelet.md](flamelet.md#3-table-format-decision-4); this page is the
specification.

The words *must*, *must not* and *may* are requirements on writers; the checks
Mallard makes are listed in [section 7](#7-what-mallard-checks).

## 1. Overview

A table holds stored quantities (fields) on the nodes of a tensor-product grid
whose axes are the model's control variables. A table also carries:

- a **model tag**, which says what the axes and the model's fields mean;
- a **thermodynamics tag**, which says how the solver gets temperature and
  pressure from the table;
- a **species block**, the mass fractions the table carries for output;
- the pressure it was generated at, model data and provenance.

The format is generic: FPV with Saghafian's compressible thermodynamics is the
first model Mallard reads. FGM/FPI, heat-loss axes, several streams and other
thermodynamics are new tags and fields in the same format, not new formats
([section 8](#8-versioning-and-new-models)).

## 2. File layout

A file (extension `.mtab`) is three parts, with nothing before, between or after
them:

1. **Magic line:** the ASCII bytes `mallard-table 1` and a line feed (0x0A). The
   number is the format version.
2. **Header:** one JSON object (RFC 8259, UTF-8) on one line, followed by a line
   feed. It must not contain a raw line feed (strings escape it as `\n`). It
   must not contain `NaN` or `Infinity`. Numbers must be written so that they
   read back to the same binary64 value (shortest round-trip form, as Python's
   `json` and `repr` write them).
3. **Data:** the values as IEEE 754 binary64, **little-endian**, exactly
   `n_0 n_1 ... n_(d-1) n_fields` values, where `n_a` is the node count of axis
   `a` and `n_fields` the number of fields. The file ends after the last value.

Order of the data: node-major, with the last axis varying fastest, and the
fields of a node contiguous, in the header's order. The value of field `f` at
node `(i_0, ..., i_(d-1))` is at index

```
((...((i_0 n_1 + i_1) n_2 + i_2)...) n_(d-1) + i_(d-1)) n_fields + f
```

of the data, counted in values (multiply by 8 for the byte offset from the start
of the data). This is NumPy's C order of an array of shape
`(n_0, ..., n_(d-1), n_fields)`.

## 3. Header members

| Member | Type | Required | Meaning |
|---|---|---|---|
| `format` | string | yes | `"mallard-table"` |
| `version` | integer | yes | `1`; must equal the magic line's number |
| `model` | string | yes | Model tag ([section 5](#5-models)): what the axes and the model's fields mean |
| `thermodynamics` | string | yes | Thermodynamics tag ([section 6](#6-thermodynamics)) |
| `axes` | array of objects | yes | The control variables, in data order: `{"name", "units", "values"}`. `values` holds at least 2 finite nodes, strictly increasing. Nodes need not be uniform |
| `fields` | array of objects | yes | The stored quantities, in data order: `{"name", "units"}`. Names are unique and match `[A-Za-z][A-Za-z0-9_]*` |
| `species` | array of strings | yes | The species block, possibly empty: species names as in the mechanism. Species `k` is stored as the field `Y_k` (mass fraction, units `"1"`). Every field whose name starts with `Y_` must be in this list |
| `pressure` | number | yes | The pressure the table was generated at, in Pa, positive |
| `model_data` | object | yes | Model-specific metadata ([section 5](#5-models)); `{}` if the model has none |
| `provenance` | object | yes | What made the table ([section 3.2](#32-provenance)) |

Writers may add other members. Readers ignore members they do not know, so
additions that old readers can ignore need no new version
([section 8](#8-versioning-and-new-models)).

### 3.1 Units

Units are strings in UDUNITS-2 syntax, with SI base and derived units, spaces
between factors and integer exponents: `"1"` for dimensionless quantities,
`"K"`, `"kg m-3"`, `"J kg-1 K-1"`, `"kg m-3 s-1"`. Mallard does not convert
units. A field it uses must carry exactly the units listed for it below, or the
table is refused. Fields Mallard does not use may have any units string.

### 3.2 Provenance

| Member | Type | Required | Meaning |
|---|---|---|---|
| `generator` | object | yes | `{"name": string, "version": string}` of the program that wrote the table |
| `created` | string | yes | Time of writing, RFC 3339 in UTC (`2026-11-15T12:00:00Z`) |
| `mechanism` | object | yes | `{"name": string, "sha256": string}`: the mechanism file's name and the SHA-256 of its bytes, as 64 lowercase hexadecimal digits |
| `input` | string | yes | The generator's input, verbatim, so that a table can be regenerated from its own header |
| anything else | any | no | For example the versions of the generator's dependencies (Cantera), the generator's source revision, the transport model of the flamelets |

Provenance must not hold personal data (user names, host names, paths below a
home directory). Mallard prints the generator, its version and the mechanism's
hash in the run log when it reads a table.

## 4. Fields every table holds

### 4.1 Transport

Every table that a solver couples to holds the transport properties of the
tabulated state. Mallard uses unity Lewis number: the diffusivity of every
control variable is `rho D = kappa / cp`.

| Field | Units | Meaning | Check |
|---|---|---|---|
| `mu` | `Pa s` | Dynamic viscosity | > 0 |
| `kappa` | `W m-1 K-1` | Thermal conductivity | > 0 |
| `cp` | `J kg-1 K-1` | Specific heat at constant pressure | > 0 |

### 4.2 Species block

Every species `k` in `species` has the field `Y_k`, units `"1"`, with values in
[0, 1]. The block may be partial, for example only the species that output or
radiation needs. Mallard does not require the mass fractions to sum to one.

## 5. Models

The model tag sets the axes (names, order, ranges) and the model's fields and
`model_data`.

### 5.1 `fpv`: flamelet/progress variable

Steady flamelets of a counterflow S-curve. The presumed PDF is beta in `Z` and
delta in the flamelet parameter, integrated by the generator
([flamelet.md](flamelet.md#2-table-generation-external)).

Axes, in this order, each in units `"1"`, each with first node exactly 0 and last
node exactly 1:

| Axis | Meaning |
|---|---|
| `Z` | Mean mixture fraction (Bilger) |
| `S` | Normalized variance `Zvar / (Z (1 - Z))` |
| `c` | Normalized mean progress variable: `(C - C_min) / (C_max - C_min)` at that `(Z, S)` |

Fields:

| Field | Units | Meaning | Check |
|---|---|---|---|
| `C` | `1` | Mean progress variable `sum_k w_k Y_k` | >= 0; linear in `c` at each `(Z, S)` (below) |
| `omega_C` | `kg m-3 s-1` | Mean source of `rho C` at the tabulated state | finite |
| `rho0` | `kg m-3` | Density of the tabulated state | > 0 |

**`C` must be linear in `c`** at every `(Z, S)` node:
`C(i, j, k) = C(i, j, 0) + c_k (C(i, j, last) - C(i, j, 0))`, to 1e-9 of the
largest `|C|` in the table, with `C(i, j, last) >= C(i, j, 0)`. Mallard then maps
a transported `C` to `c` from the two ends alone, and the lookup returns the
transported `C` exactly ([flamelet.md](flamelet.md#43-the-fpv-map-decision-3)).

`omega_C` is tabulated at the flamelet's density `rho0`. The compressible
solver scales it to the local state (stage 2 decides the scaling, for example
`(rho / rho0)^2` for a source that is second order in density). That is why
`rho0` is required.

`model_data`:

| Member | Type | Meaning |
|---|---|---|
| `Z_st` | number in (0, 1) | Stoichiometric mixture fraction |
| `progress_variable` | object | The weights `w_k` of `C`, by species name |
| `streams` | object | `{"fuel": {"T": K, "Y": {species: mass fraction}}, "oxidizer": {...}}`, which define `Z = 1` and `Z = 0` |

### 5.2 Reserved model tags

These tags are reserved for the models the generator will add. Their contracts
will be specified here when Mallard reads them. Until then, Mallard refuses
them by name:

- `fgm`, `fpi`: premixed and partially premixed manifolds, axes `Z`, `S`, `c`;
- `slfm`: steady laminar flamelets, axes `Z`, `S`, `chi_st`;
- `fpv-h`: FPV with an enthalpy-defect axis `h_defect` (heat loss);
- names for several streams (two mixture fractions) when they are designed.

## 6. Thermodynamics

The thermodynamics tag sets the fields from which the solver gets temperature
and pressure from the conservative state.

### 6.1 `saghafian2015`: compressible, linearized `gamma(T)`

After Saghafian, Terrapon & Pitsch (2015). The ratio of specific heats is linear
in temperature about the tabulated state, `gamma(T) = gamma0 + a_gamma (T - T0)`,
and with `cv = R / (gamma - 1)` the temperature follows from the internal energy
in closed form:

$$
T = T_0 + \frac{\gamma_0 - 1}{a_\gamma} \left( \exp\left( \frac{a_\gamma (e - e_0)}{R} \right) - 1 \right), \qquad p = \rho R T,
$$

with the limit `T = T0 + (gamma0 - 1)(e - e0) / R` as `a_gamma -> 0`.

| Field | Units | Meaning | Check |
|---|---|---|---|
| `T0` | `K` | Temperature of the tabulated state | > 0 |
| `e0` | `J kg-1` | Internal energy at `T0`, formation energies included, on Cantera's reference: species enthalpies of formation at 298.15 K | finite |
| `R` | `J kg-1 K-1` | Specific gas constant `R_u / W` | > 0 |
| `gamma0` | `1` | `gamma` at `T0` | > 1 |
| `a_gamma` | `K-1` | `d gamma / d T` at `T0` | finite |

### 6.2 Reserved thermodynamics tags

These are reserved and refused by name until specified: `low-mach` (tabulated
`T` and `rho`, pressure from the equation of state only), and tags for fuller
compressible models (for example a tabulated `cv(T)` per node).

## 7. What Mallard checks

Mallard reads a table once at setup and refuses it, with the file name and the
reason, unless:

1. The magic line is exactly `mallard-table 1`.
2. The header is one JSON object, `format` is `"mallard-table"` and `version`
   is 1.
3. `model` and `thermodynamics` are tags this version of Mallard supports. An
   unknown or reserved tag is refused by name, with the list of supported tags.
4. Every required member is present with its type. `pressure` is positive.
   `provenance.generator.name` and `version`, `created`, `input`,
   `mechanism.name` and `mechanism.sha256` (64 lowercase hexadecimal digits)
   are present. The model's `model_data` members are present and in range.
5. The axes are between 1 and 4 in number. The format sets no limit; this
   version of Mallard reads up to 4. Each axis has at least 2 finite,
   strictly increasing nodes. Field names are unique.
6. Every `Y_` field is a species in `species`.
7. The data holds exactly `prod(n_a) n_fields` values and every value is
   finite.
8. The model's axes have the required names, order, units and range.
9. Every field required by the model, the thermodynamics, transport and the
   species block is present, with exactly the listed units, and its values
   meet the listed checks.
10. Model-specific checks pass. For `fpv`, `C` is linear in `c`.

At run time (stage 2), Mallard also warns when the table's `pressure` differs
from the run's reference pressure by more than 10%.

## 8. Versioning and new models

- Version 1 is this page. The version changes when the layout changes, when a
  required member or field is added or removed, or when the meaning of an
  existing member, field or tag changes. Mallard refuses versions it does not
  know.
- New optional header members, new optional fields, and new provenance members
  need no new version: readers ignore what they do not know.
- New models and thermodynamics are new tags, specified here before Mallard
  reads them. They need no new version. A Mallard that does not support a tag
  refuses the table by name.
- The values are binary64. A binary32 variant would be a new version, decided
  when lookups show up in profiles.

## 9. Reading a table with NumPy

```python
import json
import numpy as np

with open("table.mtab", "rb") as f:
    assert f.readline() == b"mallard-table 1\n"
    header = json.loads(f.readline())
    data = np.frombuffer(f.read(), dtype="<f8")
shape = [len(a["values"]) for a in header["axes"]] + [len(header["fields"])]
data = data.reshape(shape)
T0 = data[..., [f["name"] for f in header["fields"]].index("T0")]
```

## 10. Reference fixture

[`test/data/flamelet/fixture_fpv.mtab`](../../test/data/flamelet/fixture_fpv.mtab)
is a small valid `fpv` / `saghafian2015` table (5 x 3 x 4 nodes, 15 fields,
species H2, O2, H2O and N2). Its fields are synthetic functions, multilinear in
`(Z, S, c)`, so lookups reproduce them exactly. They test the format, not
combustion. The script that wrote it is in its own `provenance.input`. A
generator can compare its writer against this file, and Mallard's tests
(`test/flamelet_table_test.cpp`) read it.

## References

- A. Saghafian, V. E. Terrapon and H. Pitsch, An efficient flamelet-based combustion model for compressible flows, *Combust. Flame* 162, 652–667 (2015).
- FPVgen: M. Bonanni, M. Ihme and the Laboratory of Fluids in Complex Environments, FPV flamelet table generation software, [doi:10.5281/zenodo.14728543](https://doi.org/10.5281/zenodo.14728543), MIT License.
