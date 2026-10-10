# Mechanisms

Cantera YAML mechanism files for `[physics] mechanism = "..."` and the tests.
Except `he_air.yaml`, `c3h8_2step.yaml` and `ch4_bfer.yaml` (assembled from them, see their headers), they are copied unchanged from the data files of
[Cantera](https://cantera.org) 3.2.0 (BSD-3-Clause license, Copyright (c)
2001-2025, Cantera Developers); each file's header names its source.

| File | Phases | Species | Content |
|---|---|---|---|
| `h2o2.yaml` | `ohmech` | 10 | Hydrogen-oxygen submechanism of GRI-Mech 3.0 with Ar and N2; NASA-7 thermo, mixture-averaged transport |
| `gri30.yaml` | `gri30` | 53 | GRI-Mech 3.0 (natural gas); NASA-7 thermo, mixture-averaged transport |
| `he_air.yaml` | `he_air` | 3 | Helium, N2 and O2 without reactions, for shock-bubble runs: HE (ATcT thermo, transport) from Cantera's `example_data/ammonia-CO-H2-Alzueta-2023.yaml`, N2 and O2 from `h2o2.yaml` |
| `c3h8_2step.yaml` | `c3h8_2step` | 6 | Two-step global propane-air mechanism (Westbrook & Dryer 1981 form, CO-CO2 equilibrium step), its fuel step tuned to GRI-Mech 3.0's flame speed at phi = 0.65, 288 K; species data from `gri30.yaml`. For LES of lean premixed propane flames (`examples/volvo_bluff_body`) |
| `ch4_bfer.yaml` | `gas` | 6 | Two-step lean CH4/air mechanism 2S-CH4-BFER (Franzelli et al. 2012, without the rich-side corrections), GRI-Mech 3.0 species data; for lean premixed flames |
| `airNASA9.yaml` | `airNASA9` | 11 | Air species with ions, NASA-9 thermo to 20,000 K (thermo only) |
| `nDodecane_Reitz.yaml` | `nDodecane_IG` (ideal gas), `nDodecane_RK` | 100 | Reduced n-dodecane-PAH mechanism of Wang, Ra, Jia & Reitz (2014), 432 reactions; the ~100-species chemistry benchmark (only the ideal-gas phase is supported) |

`external/` holds mechanisms too large to keep in the repository, copied from
the Cantera installation by `tools/cantera_mechanisms.py`: the n-hexane
mechanism of Zhang et al. (2015, NUIG; 1268 species, 5336 reactions) for the
large-mechanism benchmark and V3.

Chemkin files can be converted with Cantera's `ck2yaml`.
