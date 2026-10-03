# Mechanisms

Cantera YAML mechanism files for `[physics] mechanism = "..."` and the tests.
They are copied unchanged from the data files of
[Cantera](https://cantera.org) 3.2.0 (BSD-3-Clause license, Copyright (c)
2001-2025, Cantera Developers); each file's header names its source.

| File | Phases | Species | Content |
|---|---|---|---|
| `h2o2.yaml` | `ohmech` | 10 | Hydrogen-oxygen submechanism of GRI-Mech 3.0 with Ar and N2; NASA-7 thermo, mixture-averaged transport |
| `gri30.yaml` | `gri30` | 53 | GRI-Mech 3.0 (natural gas); NASA-7 thermo, mixture-averaged transport |
| `airNASA9.yaml` | `airNASA9` | 11 | Air species with ions, NASA-9 thermo to 20,000 K (thermo only) |

Chemkin files can be converted with Cantera's `ck2yaml`.
