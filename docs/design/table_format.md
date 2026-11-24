# Table format for tabulated chemistry (HDF5, version 1)

Status: specified (#631). The contract between Mallard and the programs that
write tables for it. Mallard reads tables and does not generate them: tables are
written by an external generator library, **`<generator, name TBD>`**, a
separate project that starts from FPVgen (IhmeGroup/FPVgen). Any other program
that writes this format works too. The reasons for the format are in
[flamelet.md](flamelet.md#3-table-format-decision-4); this page is the
specification.

The words *must*, *must not*, *should* and *may* are requirements on writers;
the checks Mallard makes are listed in [section 7](#7-what-mallard-checks).

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

A table is an HDF5 file (any extension; `.h5` by convention), readable with
h5py alone ([section 9](#9-reading-a-table-with-h5py)). Mallard reads it in
builds with HDF5 (`-DMallard_ENABLE_HDF5=ON`; `-DUSE_SYSTEM_HDF5=OFF` builds
the bundled HDF5); a build without HDF5 refuses a table with an input error
that says so.

## 2. File layout

| Object | Kind | Required | Contents |
|---|---|---|---|
| `/` | root group | yes | The attributes of [section 3](#3-attributes-of-the-root-group) |
| `/values` | dataset | yes | The values: float64 or float32, shape `(n_0, ..., n_(d-1), n_fields)`, with attributes `axes`, `fields` and `units` (below) |
| `/axes/<name>` | dataset | yes, one per axis | The nodes of axis `<name>`: float64, one-dimensional, at least 2 finite nodes, strictly increasing, not necessarily uniform; attribute `units` (string) |
| `/model_data` | group | yes | Model-specific data ([section 5](#5-models)); empty if the model has none |
| `/provenance` | group | yes | What made the table ([section 3.2](#32-provenance)) |

Attributes of `/values`:

| Attribute | Type | Meaning |
|---|---|---|
| `axes` | array of strings | The axis names, in the order of the dataset's dimensions; each names a dataset `/axes/<name>` |
| `fields` | array of strings | The field names, in the order of the last dimension. Names are unique and match `[A-Za-z][A-Za-z0-9_]*` |
| `units` | array of strings | The units of each field, in the same order ([section 3.1](#31-units)) |

Axis names match `[A-Za-z][A-Za-z0-9_]*` too. Strings may be variable or fixed
length, ASCII or UTF-8 (h5py writes Python strings as variable-length UTF-8).
Numbers in attributes are scalars of any integer or floating-point type.

**The values.** `/values` holds IEEE 754 binary64 or binary32 values, and
**must** be stored contiguously and without filters (no chunking, no
compression), little-endian: h5py's and HDF5's default layout. The order is
node-major, with the last axis varying fastest and the fields of a node
contiguous: NumPy's C order of an array of shape
`(n_0, ..., n_(d-1), n_fields)`, where `n_a` is the node count of axis `a`.
The value of field `f` at node `(i_0, ..., i_(d-1))` is at index

```
((...((i_0 n_1 + i_1) n_2 + i_2)...) n_(d-1) + i_(d-1)) n_fields + f
```

counted in values from the dataset's start. Writers **should** align the
dataset to its value size (h5py: `alignment_threshold=1,
alignment_interval=8`; C: `H5Pset_alignment(fapl, 1, 8)`); HDF5 usually does
without being asked.

**Loading.** On CPU backends, where lookups run on the host copy, Mallard
reads the values into memory with `H5Dread`, which takes any layout HDF5
reads. Each rank then holds its own copy (192 MB for the design notes' large
example; the page cache's copy is reclaimable). Mapping the file instead would
keep one copy per node in the page cache, but lookups on it ran about 20%
slower: page-cache pages are 4 KiB on typical kernels (no transparent huge
pages for files, nor for shared memory where `shmem_enabled` is `never`),
while the read copy gets huge pages, and page-cache pages ran slower than
anonymous 4 KiB pages too. On GPU backends the host copy only stages the copy
to the device, so Mallard maps the values from the file (`mmap`, at the offset
`H5Dget_offset` returns) when the layout allows it: contiguous, unfiltered,
in the machine's byte order, aligned to the value size, and the first and last
values read the same through HDF5; otherwise it reads them. Both paths give
bitwise identical values. Do not overwrite a table while a run reads it: write
a new file and rename it.

**float32.** Values may be float32 to halve the memory and bandwidth of
lookups. The axes stay float64, and lookups sum in double precision, so a
float32 table looks up as its float64 source to float32 rounding (a relative
error of 6e-8 of the largest value in the lookup's cell). Mallard's checks
([section 7](#7-what-mallard-checks)) hold for float32 tables, with the
tolerance of the linearity of `C` in `Lambda` relaxed from 1e-9 to 1e-6.

## 3. Attributes of the root group

| Attribute | Type | Required | Meaning |
|---|---|---|---|
| `format` | string | yes | `"mallard-table"` |
| `version` | integer | yes | `1` |
| `model` | string | yes | Model tag ([section 5](#5-models)): what the axes and the model's fields mean |
| `thermodynamics` | string | yes | Thermodynamics tag ([section 6](#6-thermodynamics)) |
| `species` | array of strings | yes | The species block, possibly empty: species names as in the mechanism. Species `k` is stored as the field `Y_k` (mass fraction, units `"1"`). Every field whose name starts with `Y_` must be in this list |
| `pressure` | number | yes | The pressure the table was generated at, in Pa, positive |

Writers may add other attributes, groups and datasets. Readers ignore what they
do not know, so additions that old readers can ignore need no new version
([section 8](#8-versioning-and-new-models)).

### 3.1 Units

Units are strings in UDUNITS-2 syntax, with SI base and derived units, spaces
between factors and integer exponents: `"1"` for dimensionless quantities,
`"K"`, `"kg m-3"`, `"J kg-1 K-1"`, `"kg m-3 s-1"`. Mallard does not convert
units. A field it uses must carry exactly the units listed for it below, or the
table is refused. Fields Mallard does not use may have any units string.

### 3.2 Provenance

The group `/provenance`:

| Object | Kind | Required | Meaning |
|---|---|---|---|
| `generator` | group with string attributes `name`, `version` | yes | The program that wrote the table |
| `created` | string attribute | yes | Time of writing, RFC 3339 in UTC (`2026-11-15T12:00:00Z`) |
| `mechanism` | group with string attributes `name`, `sha256` | yes | The mechanism file's name and the SHA-256 of its bytes, as 64 lowercase hexadecimal digits |
| `input` | scalar string dataset | yes | The generator's input, verbatim, so that a table can be regenerated from its own file. A dataset rather than an attribute, because attributes are limited to 64 KiB |
| anything else | any | no | For example the versions of the generator's dependencies (Cantera), the generator's source revision, the transport model of the flamelets |

Provenance must not hold personal data (user names, host names, paths below a
home directory). When the solver reads a table (stage 2), it prints the
generator, its version and the mechanism's hash in the run log.

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

### 4.2 Optional fields

Fields Mallard reads when a table has them, checked as the others:

| Field | Units | Meaning | Check |
|---|---|---|---|
| `HRR` | `W m-3` | Heat release rate `-sum_k h_k omega_k` of the tabulated state, at its density `rho0` (output only) | finite |

### 4.3 Species block

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
| `Sz` | Normalized variance `Zvar / (Z (1 - Z))` |
| `Lambda` | Normalized mean progress variable: `(C - C_min) / (C_max - C_min)` at that `(Z, Sz)` |

The axis names are FPVgen's: `Sz` its normalized variance and `Lambda` its
normalized progress variable Λ (FPVgen's field `PROG_NORM`). `Lambda` is not
Ihme & Pitsch's flamelet parameter `C(Z_st)`
([flamelet.md](flamelet.md#1-models-and-control-variables)).

Fields:

| Field | Units | Meaning | Check |
|---|---|---|---|
| `C` | `1` | Mean progress variable `sum_k w_k Y_k` | >= 0; linear in `Lambda` at each `(Z, Sz)` (below) |
| `omega_C` | `kg m-3 s-1` | Mean source of `rho C` at the tabulated state | finite |
| `rho0` | `kg m-3` | Density of the tabulated state | > 0 |

**`C` must be linear in `Lambda`** at every `(Z, Sz)` node:
`C(i, j, k) = C(i, j, 0) + Lambda_k (C(i, j, last) - C(i, j, 0))`, to 1e-9 of the
largest `|C|` in the table, with `C(i, j, last) >= C(i, j, 0)`. Mallard then maps
a transported `C` to `Lambda` from the two ends alone, and the lookup returns the
transported `C` exactly ([flamelet.md](flamelet.md#43-the-fpv-map-decision-3)).

`omega_C` is tabulated at the flamelet's density `rho0`. The compressible
solver scales it to the local state (stage 2 decides the scaling, for example
`(rho / rho0)^2` for a source that is second order in density). That is why
`rho0` is required.

The fields of an `fpv` table at `Sz = 0` over `(Z, Lambda)`, from the sample
flamelets of the documentation (H2/N2 against air, `C = Y_H2O`,
`tools/docs/flamelet.py`; [flamelet.md](flamelet.md#2-table-generation-external)).
Above the dashed line the nodes come from the stable branch of the S-curve,
between it and the dotted line from the unstable branch, and below the dotted
line they are interpolated to the mixing solution:

![Temperature, density, source of C, heat release rate, C and OH mass fraction of a sample fpv table at zero variance, over Z and Lambda, with the extinction flamelet and the last unstable flamelet marked](../images/guide/flamelet_table-light.svg#gh-light-mode-only)
![Temperature, density, source of C, heat release rate, C and OH mass fraction of a sample fpv table at zero variance, over Z and Lambda, with the extinction flamelet and the last unstable flamelet marked](../images/guide/flamelet_table-dark.svg#gh-dark-mode-only)

`/model_data`:

| Object | Kind | Meaning |
|---|---|---|
| `Z_st` | number attribute, in (0, 1) | Stoichiometric mixture fraction |
| `progress_variable` | group, at least one attribute | The weights `w_k` of `C = sum_k w_k Y_k`: one number attribute per species, named by the species. Every weighted species is in the species block, and the field `C` equals `sum_k w_k Y_k` of the stored mass fractions at every node |
| `mixture_fraction` | string attribute | How `Z` is defined: `"bilger"`, Bilger's element-based mixture fraction between the two streams, `Z = (beta - beta_ox) / (beta_fuel - beta_ox)` with `beta = 2 Z_C / W_C + Z_H / (2 W_H) - Z_O / W_O` of the element mass fractions `Z_e` |
| `streams/fuel`, `streams/oxidizer` | groups | The streams that define `Z = 1` and `Z = 0`: number attribute `T` (K, positive) and a group `Y` with one number attribute per species, its mass fraction in [0, 1], summing to 1 |

Together, `progress_variable`, `mixture_fraction` and the streams define `Z`
and `C` of any composition, which is what converting a finite-rate solution to
the table needs ([flamelet.md](../guide/flamelet.md#converting-between-finite-rate-and-fpv)).

### 5.2 Reserved model tags

These tags are reserved for the models the generator will add. Their contracts
will be specified here when Mallard reads them. Until then, Mallard refuses
them by name:

- `fgm`, `fpi`: premixed and partially premixed manifolds, axes `Z`, `Sz`, `Lambda`;
- `slfm`: steady laminar flamelets, axes `Z`, `Sz`, `chi_st`;
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

1. The build has HDF5, and the file is an HDF5 file.
2. The root's `format` is `"mallard-table"` and `version` is 1.
3. `model` and `thermodynamics` are tags this version of Mallard supports. An
   unknown or reserved tag is refused by name, with the list of supported tags.
4. Every required attribute, group and dataset is present with its type.
   `pressure` is positive. `/provenance` has `generator` (`name`, `version`),
   `created`, `input` and `mechanism` (`name`, `sha256` of 64 lowercase
   hexadecimal digits). The model's `/model_data` is present and in range.
5. There is at least one axis; neither the format nor Mallard limits their
   number. Each axis has at least 2 finite, strictly increasing nodes. Axis and
   field names are of the form `[A-Za-z][A-Za-z0-9_]*`, and field names are
   unique; `fields` and `units` have the same length.
6. Every `Y_` field is a species in `species`.
7. `/values` is float64 or float32, of shape `(n_0, ..., n_(d-1), n_fields)`,
   and every value is finite.
8. The model's axes have the required names, order, units and range.
9. Every field required by the model, the thermodynamics, transport and the
   species block is present, with exactly the listed units, and its values
   meet the listed checks.
10. Model-specific checks pass. For `fpv`, `C` is linear in `Lambda`,
    `mixture_fraction` is `"bilger"`, every species
    `progress_variable` weighs is in the species block, `C` equals
    `sum_k w_k Y_k` at every node to `1e-9` (float64) or `1e-6` (float32) of
    the largest `|C|` or `sum_k |w_k Y_k|`, and each stream's mass fractions
    sum to 1 within `1e-6`.

At run time (stage 2), Mallard also warns when the table's `pressure` differs
from the run's reference pressure by more than 10%.

## 8. Versioning and new models

- Version 1 is this page. The version changes when the layout changes, when a
  required attribute, group, dataset or field is added or removed, or when the
  meaning of an existing one or of a tag changes. Mallard refuses versions it
  does not know.
- New optional attributes, groups, datasets and fields, and new provenance,
  need no new version: readers ignore what they do not know.
- New models and thermodynamics are new tags, specified here before Mallard
  reads them. They need no new version. A Mallard that does not support a tag
  refuses the table by name.
- Values are float64 or float32 within version 1 (section 2).

## 9. Reading a table with h5py

```python
import h5py

with h5py.File("table.h5", "r") as f:
    assert f.attrs["format"] == "mallard-table" and f.attrs["version"] == 1
    values = f["values"]
    axes = {a: f["axes"][a][()] for a in values.attrs["axes"]}
    fields = list(values.attrs["fields"])
    T0 = values[..., fields.index("T0")]  # shape (n_Z, n_Sz, n_Lambda)
    weights = dict(f["model_data/progress_variable"].attrs)
    assert f["model_data"].attrs["mixture_fraction"] == "bilger"
    fuel = dict(f["model_data/streams/fuel/Y"].attrs)
    writer = f["provenance/input"][()].decode()
```

Writing one takes the same calls in reverse; the fixture's writer below is a
complete example.

## 10. Reference fixtures

[`test/data/flamelet/fixture_fpv.h5`](../../test/data/flamelet/fixture_fpv.h5)
is a small valid `fpv` / `saghafian2015` table (5 x 3 x 4 nodes, 15 fields,
species H2, O2, H2O and N2) in float64, and
[`fixture_fpv_f32.h5`](../../test/data/flamelet/fixture_fpv_f32.h5) the same
table in float32. Their fields are synthetic functions, multilinear in
`(Z, Sz, Lambda)`, so lookups reproduce them exactly, and `C = Y_H2O` as its
`progress_variable` says. They test the format, not combustion. The h5py
script that wrote them is in their own `/provenance/input`. A generator can
compare its writer against these files, and Mallard's tests
(`test/flamelet_table_test.cpp`) read them.

[`convert_h2.h5`](../../test/data/flamelet/convert_h2.h5) is the table of the
conversion tests (`test/restart_convert_test.cpp`): H2/N2 against air, with
every species of `mechanisms/h2o2.yaml`, whose species conserve the streams'
elements, so that Bilger's `Z` of each node's composition is its `Z` and
`C = Y_H2O`. Its writer is also in its `/provenance/input`.

## References

- A. Saghafian, V. E. Terrapon and H. Pitsch, An efficient flamelet-based combustion model for compressible flows, *Combust. Flame* 162, 652–667 (2015).
- FPVgen: M. Bonanni, M. Ihme and the Laboratory of Fluids in Complex Environments, FPV flamelet table generation software, [doi:10.5281/zenodo.14728543](https://doi.org/10.5281/zenodo.14728543), MIT License.
