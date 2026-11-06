# Documentation audit (proposal)

This is a proposal for review. It is not docs content, and it gets deleted before anything lands.

- **Audited:** `origin/main` at d080bb21, plus the website `mallard-website@origin/main` (`mkdocs.yml`, `pages.py`).
- **Taken as done:** #588 splits `guide/output.md` into Output / Probes and monitors / Statistics and renames "the log" / "the run log" to *terminal output* / *run record*. Website #47 adds those pages to the nav. Website #45 adds Running to the nav.
- **Line numbers** refer to `main` at d080bb21.

## Summary: the ten biggest problems

1. **The website's top level is organized by document type, not by what the reader wants.**
   - "Reference" holds ten unrelated things (`mkdocs.yml` nav).
   - Getting started and Tutorials are separate tabs.
   - The CLI and Running pages are appended to the guide alphabetically after the Performance dashboard (`pages.py:415-419`).
   - Contributing is not on the site at all.
2. **Each topic has several homes.**
   - Restarts are covered in 7 places: initialization 64-115, parallel 88-90 and 106, running 7-13, cli 76, input.md 149 and 461, channel_les 82.
   - `mallard reactor` is covered 4 times: thermochemistry 183, cli 139, input.md 1262, tools/README 42-69.
   - `mallard mesh convert` is covered 4 times: input.md 209, tools/README 71, meshes, cli 317.
   - `mallard check` is covered 3 times: input_files 357-400, cli 251, input.md 57.
   - Units are covered 3 times: input_files 106-197, output 195-259, input.md 24-55.
3. **There are two numerics references that disagree with each other.**
   - `numerics/overview.md` restates theory chapters 6, 7 and 9 (12-31, 238-398) and theory/time_integration 72-113 (32-118).
   - `theory/time_integration.md` 30-64 lists only SSPRK3, RK4 and FE. SSPRK54, CK45, RDPK3Sp35 and error control are covered only in overview 119-235.
4. **`docs/input.md` is part key reference, part second guide.**
   - Its order does not follow the guide: thermochemistry comes after numerics, Regions sits at 1113, `[output]`/`[[write_data]]` are separated from forces, probes and statistics by `[parallel]`, and `mallard reactor` is last.
   - About 400 lines are prose or derivations: LES/TFLES 371-455, spectra 793-836, probes 907-1036, VTKHDF layout 1217-1260.
   - Defaults are buried in descriptions. Only `[[sources]]` (1074) has a Default column, while the guide has about 20 Key / Default / Meaning tables.
5. **Some content is stale.** All of these were checked against `src/`.
   - geometry.md 13 and 266-278 say probes on boundary zones and surfaces are "Coming soon". output.md 451-540 documents both.
   - channel_les.md 143-149 says the wall model is "under development". les.md 176 and input.md 520 document it.
   - theory/boundaries_sources.md 25 and 40 say "`wall_isothermal` (single gas)" and "not available" for mixtures. It is implemented: `solver_mixture.cpp:656,688`, `mixture_viscous_flux.h:153`.
   - The status lines of design/les.md 3-6 and design/mpi.md 3 and 207 are stale.
   - design/chemistry.md is stale at 6, 619, 1754 and 1824 (`allow_missing_species` exists nowhere).
6. **Guide pages mix topics, and some titles don't describe their content.**
   - About half of geometry.md (210-309) is probes.
   - parallel.md runs threads/MPI, then cluster job scripts (92-149), then scaling benchmarks (151-205). `### Output` and `### Restarts` are nested under `## MPI` (67, 88).
   - running.md has four unrelated sections: control, Slurm, divergence and Kubernetes.
   - numerics.md "Time step and stopping" (96-140) holds `[run]` stop conditions.
   - thermochemistry.md 76 documents `[numerics] double_flux`.
7. **Beginners have no clear way in for several common tasks.**
   - There is no GPU page. A CUDA build appears only in first_simulation 16-23.
   - There is no ParaView how-to. The only one is in wedge.md 84-87.
   - quick_start.md and tutorials/first_simulation.md both build and run Sod with different workflows: `cd examples/sod` (qs 9) vs `mallard new` (fs 39-57). They also disagree on the minimum CMake version (3.16 vs 3.22; CMakeLists requires 3.16).
   - `upt` is never spelled out.
8. **The same thing has different names.**
   - A boundary zone is `name` (`[[boundaries]]`), `zone` (`[[forces]]`, `[[mesh.curved]]`), `boundary` (`[[probes]]`) and `geometry` (`[[write_data]]`).
   - Inflow turbulence is called "synthetic turbulence" (bc 265) and "synthetic inflow" (output 326, the design note).
   - Wall forces are called "force monitors" (initialization 89, parallel 86) and "Wall forces".
   - Pages are linked under several names: "Output and diagnostics" vs "Output"; "Running and controlling jobs" vs "run control".
9. **Design notes duplicate the theory manual and hold user docs.**
   - Equations are duplicated:
     - design/synthetic_inflow 42-106 repeats theory/les 324-380 almost verbatim.
     - design/wall_model 126-238 repeats theory/wall_models 107-301.
     - design/nscbc 15-118 repeats theory/boundaries_sources 99-184.
     - design/chemistry sections 4-6 repeat theory chapters 2 and 3.
   - User docs sit in design notes: chemistry 1741-1782 (a TOML usage example), wall_model 256-260 (keys), cli 6-39 (a command table).
   - Equations exist in only one design note: the inlet equations are only in design/inlets.md, and theory chapter 9 never mentions `stagnation_inlet` or `mass_flow_inlet`.
   - Two notes are far too long: design/chemistry is 2043 lines and design/les 1257 (480 of them validation, 701-1182).
10. **The theory manual's order and numbering are fragile.**
    - The chapters jump from fluxes to time integration and then back to boundaries (`theory/index.md` 13-38).
    - Model chapters hold discretization: chemistry_radiation has RODAS and splitting (179, 231); les has synthetic inflow (324); thermodynamics_transport has double flux (251).
    - 87 cross-references are written as "[chapter N]", so any reorder means renumbering link text.

Smaller defects:

- **Contradictions:**
  - Probes have "no `pivot`" (geometry 245), but output 449 and input.md list one.
  - The Slurm signal is `B:USR2@300` (parallel 108) vs `USR2@120` (running 35).
  - The 3D build directory is `build3d` (first_simulation 124) vs `build-3d` (tutorials).
  - cylinder.md 83 uses 4 threads, but its text at 87 says 3.
  - The `[les]` intro says Δ = V^(1/3), while the `filter_width` default is `scotti`.
  - `max_steps` is "per half step" in input.md but "per call" in thermochemistry 132.
- **Broken links:**
  - CONTRIBUTING.md 37 links `README.md#building`; the heading is "Building and installing".
  - guide/numerics.md 55 links `design/les.md#81-...-comte-bellot-corrsin-1971`; the "&" in the heading gives a double hyphen.
  - input.md 1082 "Expressions ([syntax](#regions))" points at the Regions anchor.
- **Old spelling:** input.md 461 `Mallard -i ... --restart`.
- **README:**
  - It describes VTU as the output format (258), but VTKHDF is the default.
  - It says "40 examples" (82); there are 47.
  - It mentions container tags `:0.9.0` (105-110) while the release is 0.8.0.
- **Website build:** `pages.py:121` cuts README from "## Building" to "## Contributing". As a result, the Installation page also carries README's Running, Testing and Postprocessing sections.
- **Stale rules:**
  - CONTRIBUTING.md 128-131 still says "the maintainer mirrors README changes on the website", but the site generates that page from README.
  - CLAUDE.md, an untracked local file, names `MallardTest`, `tools/device_changes.sh` and `only=cuda`. They are now `mallard-test2d`/`mallard`, `tools/dev/device_changes.sh` and `only=device`.
- **Checked and fine:**
  - There are no `upgrade-input` or `--quality-vtu` mentions, and no old BC names.
  - Every path in tools/README exists.
  - About 45 sampled defaults match the source.
  - All 268 catalog keys are documented.
  - `shape =` remains only for probes (`#554` renames it) and curved boundaries.

## 1. Page map

Audience key: **B** = beginner, **E** = expert user, **C** = contributor. Pages marked "(#588)" are as split by #588.

### Getting started

| Page | Lines | Audience | Contains |
|---|---|---|---|
| README.md | 284 | B | Features (16-87, about 70 lines), build: container / Spack / source (91-231), running, testing, postprocessing, citing. The site's Installation page is 91-263 of it |
| guide/quick_start.md | 30 | B | A three-command build and Sod run, then a link list that copies tutorials/index 11-18 |
| tutorials/index.md | 57 | B | The learning path table, including a placeholder tutorial 5 "(in preparation)" |
| tutorials/first_simulation.md | 173 | B | Build, `mallard new sod`, compare with the exact solution, 2D, 3D, templates |
| tutorials/wedge, cylinder, cone, channel_les | 143-172 | B→E | Problem, mesh, input, run, compare, things to try. Section sets vary; only cone and channel_les have "Monitor"; cone and channel_les have no "Next" link |

### User guide (`docs/guide`)

| Page | Lines | Audience | Contains |
|---|---|---|---|
| input_files | 430 | B | Anatomy, units (106-197), expressions (199-281), upgrades, unknown keys, errors, `mallard check` (357-400), key catalog (402, contributor), table of commands |
| meshes | 385 | E | Generated meshes, stretching, periodicity (94-174), Gmsh, other formats, transform, HDF5, curved boundaries, mesh quality |
| geometry | 315 | E | Regions (36-208), then probe shapes, sampling and reading probe files (210-309), plus stale "coming soon" stubs |
| physics | 141 | E | Euler/NS, viscosity laws, mixtures (a pointer), axisymmetric |
| thermochemistry | 261 | E | Mixtures, `double_flux`, chemistry, radiation, `mallard reactor` |
| les | 226 | E | SGS models, the low-dissipation flux, TFLES, PaSR, wall models |
| numerics | 299 | B/E | Presets (including validation numbers 30-65), time step **and run stopping**, reconstruction, Riemann solvers, integrators |
| boundary_conditions | 322 | B/E | Choosing a condition; stagnation and mass-flow inlets, characteristic boundaries and synthetic turbulence as peers of "Choosing"; Pitfalls |
| sources | 181 | E | `[source]`, `[[sources]]` (regions repeated from geometry), sponges |
| initialization | 193 | B/E | Uniform state, expressions, restarts (no `--restart`), initial states from scripts, remap, binary restart layout |
| output (#588) | ~300 | B/E | What a run writes, snapshots, terminal output, run record, units, provenance, formats, post-processing |
| probes (#588) | ~250 | E | `[[probes]]`, `[[forces]]`, `[integrals]` |
| statistics (#588) | ~100 | E | `[statistics]`, spectra, phase averages |
| parallel | 206 | E | Threads/GPUs (a pointer), MPI, job scripts, rank report, scaling 0.3.0/0.8.0 (benchmark history) |
| running | 83 | E | `mallard ctl`, Slurm, divergence, Kubernetes |
| performance | 81 | C | TENO 3D memory, chemistry: mostly PR history (#247, #265) |
| cli | 402 | E | Generated `--help` pages (tools/docs/cli.py) |

### Reference

| Page | Lines | Audience | Contains |
|---|---|---|---|
| input.md | 1287 | E | Every key, in a non-guide order, plus about 400 lines of prose (see problem 4) |
| tools/README.md | 432 | E | Commands (29-85, duplicates cli), readers, initial states (150-250, duplicates initialization 93-107), post-processing (261-356), developer scripts. Not listed: `dev/ci_status.sh`, `docs/inlets.py`, `docs/probes.py`, `docs/used_in.py` |
| examples/README.md | 112 | B/E | The case table (rendered as cards), plus out-of-place isotropic-turbulence validation prose (69-80) |
| references.md | 351 | E | Bibliography with "Used in" |

### Theory and numerics

| Page | Lines | Audience | Contains |
|---|---|---|---|
| theory/index | 64 | E | Chapter list, notation |
| theory/governing_equations | 170 | E | Conservation laws, single gas, axisymmetric |
| theory/thermodynamics_transport | 298 | E | Species thermodynamics, EOS, transport; also frozen surrogates (85), time-step diffusivity (236) and double flux (251), which are discretization |
| theory/chemistry_radiation | 358 | E | Kinetics, reactor; also RODAS (179) and coupling/splitting (231), which are discretization; radiation |
| theory/les | 380 | E | Filtered equations, SGS, filter width, dynamic procedure, TFLES, PaSR; synthetic inflow (324), which is a boundary topic |
| theory/wall_models | 372 | E | Models (107, 198); exchange point, coupling and restarts (42, 302, 335), which are discretization |
| theory/finite_volume | 464 | E | Semi-discrete form, quadrature, curved boundaries, reconstruction, low-Mach, gradients, axisymmetric cells |
| theory/fluxes | 390 | E | Riemann solvers, hybrid flux, mixtures, viscous fluxes |
| theory/time_integration | 127 | E | RHS, RK methods (only 3 of 6), time step, splitting |
| theory/boundaries_sources | 249 | E | Weak imposition, characteristic boundaries, sources. No inlets; stale on mixture walls |
| numerics/overview | 406 | E | Method summaries (duplicating theory) plus measured stability, carbuncle and low-Mach studies, and limitations |
| numerics/teno_e | 628 | E/C | TENO-E implementation reference |
| numerics/originality | 256 | E | Novelty claims ("What is new") |
| verification/index | 376 | E | Observed orders for every scheme |

### Developers

| Page | Lines | Audience | Contains |
|---|---|---|---|
| CONTRIBUTING.md | 177 | C | Build/test for development, style, the feature checklist, changelog, figures. Not on the website |
| design/documentation | 104 | C | The docs policy and style guide (not a design note) |
| design/chemistry | 2043 | C | Plan, plus "as implemented" milestone logs, plus validation (1882-2016) |
| design/les | 1257 | C | Proposal, plus validation (701-1182) |
| design/{axisymmetric, curved_boundaries, inlets, insitu_statistics, nscbc, periodic, remap, sources, synthetic_inflow, wall_model, mpi, cli} | 81-406 | C/E | Rationale; some have roadmaps, PR histories (periodic 200) or measurements |
| design/performance | 400 | C | GPU benchmark log ("in progress", baseline v0.5.0-92) |

The site also has `docs/builder.md` (input builder), `docs/dashboard.md` (performance dashboard, listed in the User guide), gallery and validation.

## 2. Problems with evidence (beyond the top ten)

**Hierarchy**
- boundary_conditions.md: `## Stagnation inlet` (125), `## Mass-flow inlet` (171), `## Characteristic boundaries` (231) and `## Synthetic turbulence` (265) sit at the same level as `## Choosing a condition` (57). #573 already restructures this as one section per condition, so leave it to #573.
- Wall models: a `[[boundaries]]` key, documented only in les.md 176-210. input.md puts it under boundaries (520). Keep the guide text in les.md and add a one-line pointer from boundary_conditions.
- Periodicity lives in meshes.md 94-174 (correct, since it is a mesh property). boundary_conditions 30-31 should link to it. #577 rewrites the syntax.
- input_files.md: the key catalog (402-416) is contributor material. Units and expressions (106-281) are a third of a beginner page.
- There are two Kubernetes manifest sets: `tools/jobs/kubernetes_job.yaml` (parallel 103) and `docs/guide/k8s/job.yaml` (running 60).

**Names and terms**
- The two "Running" pages ("Running in parallel" and "Running and controlling jobs") are far apart in the nav. Merge them (section 3).
- The tutorial is titled "Turbulent channel flow (LES)" but listed as "Channel LES" (tutorials/index 18, quick_start 26).
- Design-note titles mix forms: "Design: X", "Command line", "Writing documentation", "Source terms on the device, and regions". Issue numbers appear in headings (chemistry 662, performance 198/257, sources 43, les 996). `pages.py:410` strips some of these with regexes.
- `probe` in `[statistics.phase]` (output 375) is a trigger point, not a `[[probes]]` entry.
- "What is new" on the site reads as release notes; the page is about novel numerics.
- "References" (the bibliography) sits inside "Reference".

**Duplication** (canonical home in bold)
- Integrator table and CFL rules: overview 121-128 and guide/numerics 129-138 and 264-272. The **guide** keeps the rules of thumb; **theory** gets the equations; **verification** gets the measurements.
- Progress-row description: output 98-116 and input.md 1190-1201. Keep it in **output**; input.md keeps the keys.
- `mallard ctl` table: running 9-12 and cli 114-121. Keep it in **cli**, with running linking there.
- Performance tables: guide/performance copies design/performance 152, 179, 235 and 262. The **guide** keeps the headline numbers; design keeps the log.
- Python dependency lists differ in three places: README 258, CONTRIBUTING 56, tools/README 20.
- Changelog rules appear 3 times: CONTRIBUTING 123-127, design/documentation 83-104, changelog/README. Figure rules appear twice: CONTRIBUTING 115-117 and documentation 39-50.

**Design rationale or implementation detail in user pages**
- input_files 309 and 321-324 (`input.h`, `input_upgrade.cpp`, CI).
- numerics 30-65 and 93-94 (test names, validation numbers).
- parallel 151-205 (scaling studies, PR numbers).
- initialization 109-115 (binary restart layout).
- meshes 222, 266 and 319 ("Used in: the mesh-import tests").
- input.md 434-450 and 476-480 (formulas).

**Stale (in addition to problem 5)**
- design/les.md 401, 429 and 1205 use the key `phi_min`; it is `upwind_floor`.
- design/cli.md 48 has `--mesh-fields`; the option is `mallard check --quality`.
- design/chemistry 1732: PLOG/Chebyshev are listed as "planned" but are done. The section 8 BC table (1784-1808) is missing the mixture walls, NSCBC and inlets.
- theory/wall_models 357 and design/wall_model stage 5 still say isothermal mixture walls "will follow".
- theory/time_integration 90 and overview 46 omit the boundary exterior wave speed from the time step (`solver.cpp:2434`; guide/numerics 118-122 includes it).
- Provenance and units examples open the deprecated `.h5` output (output 190, 213). tools/README 263 and README 224 give HDF5/XDMF equal billing.
- `[benchmark]` keys (`chemistry_benchmark.cpp:47-124`) are documented only in `benchmarks/README.md`.
- The input catalog check (`tools/dev/input_catalog.py docs`) skips sections whose heading is not `` `[x]` `` (Wall models, Synthetic turbulence, Regions, `type = "remap"`, `mallard reactor`). It never parses the `[[sources]]` Default column.

**Tutorials**
- index 5 promises CPU, GPU, MPI and monitoring in every tutorial. wedge and cylinder have no GPU or monitoring section, and cone 101-104 gives no GPU command.
- Tutorial 1 uses `mallard new` into `runs/`. Tutorials 2-6 use `cp $MALLARD/tutorials/...` into `run/` (index 44), although `mallard new` makes cases from tutorials (cli 176).
- first_simulation 119-141 restates the meshes.md 3D mesh table.

## 3. Target structure

The principle: a beginner meets one path (Get started), and from there every topic has exactly one guide page, one reference entry and one theory section, cross-linked. Guide pages explain and show examples. `input.md` is a lookup table. Theory holds the equations. Design notes hold rationale and history only.

### Website tabs

This refines the user's proposed order, which I agree with. It has 10 tabs, so check that they fit at the Material tab breakpoint (1220 px). If not, merge Gallery into Examples, with Gallery as its first page.

1. **Home**
2. **Get started:** Installation (README build section only; see PR 9), Quick start (folded into the tab's landing page), then Tutorials 1…N in path order (one list, numbered).
3. **User guide** (order of writing an input):
   1. Input files (anatomy, numbers, errors, checking)
   2. Units and expressions (split from input_files)
   3. Meshes
   4. Geometry (regions only)
   5. Physics
   6. Thermochemistry
   7. Large-eddy simulation
   8. Initialization and restarts (all restart material: `--restart`, checkpoints, remap)
   9. Boundary conditions
   10. Sources and sponges
   11. Numerics (without run stopping)
   12. Output
   13. Probes and monitors
   14. Statistics
   15. Running (merged from `running.md` and the cluster part of `parallel.md`: stop conditions from numerics, `mallard ctl`, divergence, job scripts, Slurm, Kubernetes)
   16. Parallel and GPUs (threads, GPUs, MPI, rank report; a GPU build pointer, giving GPUs an entry point)
   17. Performance (user headline numbers only)

   The Performance dashboard moves to Validation.
4. **Input builder** (its own tab, as proposed).
5. **Examples** (the card gallery).
6. **Gallery.**
7. **Validation:** Validation cases, the Verification manual, the Performance dashboard (all three are "how good / how fast is it", measured).
8. **Theory:** the Theory manual (Part I and Part II, below); Numerical studies (the measurement part of `numerics/overview`, until PR 7 dissolves it); TENO-E details; "What's new in the numerics" (renamed from "What is new").
9. **Reference:** Input reference, Command line (`guide/cli.md` moves here, since it is generated lookup material), Tools, Bibliography (renamed from "References"), Release notes (if changelog releases are published), API reference.
10. **Developers:** Contributing (new on the site, from `CONTRIBUTING.md`), Writing documentation (`design/documentation.md`), Design notes (titles normalized to the plain topic), Performance log (`design/performance.md`).
    - Why the API reference goes in Reference rather than here: the user listed it in Reference. Doxygen of `src/` is contributor-facing, so it could go here instead; I lean Reference only for discoverability.

### Theory manual order

The fix is to order Part II the way the solver builds a step. Today, boundaries and sources (which feed the right-hand side) come after time integration, which consumes that right-hand side (`time_integration.md` 11-27 assembles it from every other Part II chapter).

- **Part I, the continuous model:**
  1. Governing equations
  2. Thermodynamics and transport
  3. Chemistry and radiation
  4. Large-eddy simulation
  5. Wall models
- **Part II, the discretization, in solver order:**
  6. Finite volume and reconstruction
  7. Numerical fluxes
  8. Boundary conditions and sources
  9. Time integration (Runge-Kutta, time step, splitting)

There are two steps:
- **Step A (small; PR 1):**
  - Reorder `theory/index.md` with Part headings; the site nav follows the index's link order, through `pages.py:437-441`.
  - Replace every "[chapter N](file.md#…)" link text with the chapter title, e.g. "[Numerical fluxes](fluxes.md#…)". There are 87 links in `docs/theory/*.md`. All are correct today (checked by script), and 13 would need renumbering under the swap. Using titles makes future moves free.
  - Change "Chapter 4 alone" (index 62) to "The LES chapter alone".
  - Equations stay numbered within each chapter, so "eq. (3) of [Governing equations]" still reads correctly.
  - The two "chapter 4" mentions in design/les.md cite books and are unaffected.
  - Neither #573 nor #588 touches any "chapter N" line, so this does not conflict with them.
- **Step B (content moves; PR 8, after #573):**
  - RODAS and splitting (chemistry_radiation 179-316) move to Time integration.
  - Synthetic inflow (les 324-380) and the wall-model exchange and coupling (wall_models 42-106, 302-354) move to Boundary conditions and sources.
  - Double flux and frozen surrogates (thermodynamics_transport 85, 251) move to Numerical fluxes.
  - Time-step diffusivity (236) moves to Time integration.
  - Add the missing integrators and inlets, and fix the mixture-wall statements.

### `docs/input.md`

- Reorder the sections to the guide order. Units and Checking an input shrink to one link each. `mallard reactor` and `[benchmark]` move into a "Other commands' inputs" section at the end.
- Use Key / Default / Meaning tables everywhere, and mark required keys.
- Move the prose (problem 4) to the guide or theory, leaving one line plus a link.
- Fix the catalog check so it parses every section and the Default column.

### Design notes

- Each note gets a status line ("Implemented in X.Y" / "Proposal" / "Log") and a plain title.
- Equations that theory already has are cut, with a link to theory instead.
- Usage examples and key tables move to the guide.
- Validation sections move to `verification/` or the examples.
- design/chemistry and design/les shrink to their rationale. Their milestone logs are dropped, since the changelog and git history hold them.

## 4. Change list, as reviewable PRs

Every PR touches docs only (no `src/`, so no device code). Conflict notes are against the open PRs as of today.

| # | PR | Size | Conflicts or waits for |
|---|---|---|---|
| 1 | **Theory manual order:** Part I/II, boundaries before time integration, chapter titles instead of "chapter N" in 87 links | S | None (#573 touches theory files but no "chapter N" lines). **Land first** |
| 2 | **Stale facts, one-liners:** geometry "coming soon" (13, 266-278); channel_les wall-model section (143-149, written as a real walkthrough or cut to a link); theory/boundaries_sources mixture walls (25, 40); theory/wall_models 357; input.md 461 `mallard`; input.md 1082 link; CONTRIBUTING 37 anchor; numerics.md 55 anchor; README 82/258; CMake 3.16; `build-3d`; Slurm signal; cylinder thread count; `pivot` | S | The geometry lines are touched by #554, #580, #582, #584, #586 and #500: do the geometry part after #586/#500, the rest now |
| 3 | **Design-note status and title cleanup:** status lines and titles for les, mpi, chemistry, cli, wall_model; `phi_min`→`upwind_floor`; `--mesh-fields`; cut `allow_missing_species` | S | design/chemistry, les, inlets, nscbc and synthetic_inflow are in #573, design/nscbc and sources in #574/#578/#580: wait for #573 |
| 4 | **Website nav (mallard-website):** the tabs of section 3; Get started merges Getting started and Tutorials; Input builder tab; Validation tab (validation, verification, dashboard); Theory tab; slimmer Reference; Developers tab (with a new Contributing page in `pages.py`); CLI to Reference; no more alphabetical `GUIDE_EXTRA` at the end; "Bibliography" and "What's new in the numerics" labels; landing `content/docs/index.md` cards to match | M | Website #45 (Running) and #47 (probes and statistics): base on #47. Merged by the user |
| 5 | **Running and parallel:** merge running.md with parallel.md 92-149 into "Running"; move stop conditions out of numerics.md 96-140; one Kubernetes manifest; scaling 151-205 to performance/dashboard; add a GPU section | M | #558 (cluster tutorial) and #481/#478 touch parallel.md: wait for #558 |
| 6 | **Restarts in one place:** initialization.md owns `--restart`, checkpoints and remap; the others link to it | S | #573, #576, #578, #579, #497 and #554 touch initialization.md: after #497/#578 |
| 7 | **Numerics references consolidated:** dissolve numerics/overview into theory (method text, the missing integrators in time_integration) and verification (stability, carbuncle, low-Mach studies); guide/numerics keeps the rules of thumb; leave a redirect stub | M | #573 touches overview and teno_e: after #573 |
| 8 | **Theory step B:** the content moves of section 3, plus inlets and the boundary wave speed | M | After #573 and PR 7 |
| 9 | **README and Get started:** README trimmed to pitch, build and links (feature list → site home; Running, Testing and Postprocessing → guide and CONTRIBUTING); `pages.py:121` cut adjusted; quick_start folded into first_simulation or the tab landing; tutorials use `mallard new` throughout | M | README is in most open PRs (one-line edits): do it after the #499 region stack (#554, #580, #582, #584, #586) lands |
| 10 | **input.md as a pure reference:** guide order, Default columns, required marks, prose moved out, catalog check extended to every section | L | Touched by about 12 open PRs (#497, #500, #554, #569, #571, #573, #574, #577, #578, #580, #582, #586, #588): **last**, after the queue drains; split it per section group if needed |
| 11 | **input_files split:** "Units and expressions" page; key catalog → CONTRIBUTING; `mallard check` → cli plus one example | S | #585, #587, #573, #577 and the region stack touch input_files.md: after them |
| 12 | **Design notes slimmed:** cut theory duplicates (synthetic_inflow, wall_model, nscbc, chemistry 4-6); usage → guide; validation → verification; drop the chemistry and les milestone logs | L | After #573/#574 |
| 13 | **Terms:** one name per concept in prose (boundary zone, synthetic inflow turbulence, wall forces, `upt` spelled out). Key renames (`zone`/`boundary`/`geometry`) are a code change for a separate issue, using `mallard input upgrade` | S (docs) | After #573 and #554 |
| 14 | **Docs policy in one place:** `design/documentation.md` becomes "Writing documentation" and covers README, tutorials, cli.md, mechanisms/benchmarks READMEs; CONTRIBUTING links to it instead of repeating the changelog and figure rules; remove the "mirror README on the website" sentence. The local CLAUDE.md fixes are untracked and the user's to make | S | #568 touches CONTRIBUTING (small) |

Suggested order:
1. PRs 1, 2 (partly) and 14 now.
2. PR 4 once #47 merges.
3. PRs 3, 7, 8 and 12 after #573.
4. PRs 5, 6 and 11 as their blockers land.
5. PRs 9, 10 and 13 last.

## Questions for the user

1. Should the API reference go in Reference or in Developers?
2. Should Gallery stay its own tab, or become the first page of Examples if 10 tabs crowd the bar?
3. Should `numerics/overview.md` be fully dissolved (PR 7), or kept as a "Numerical studies" page of measurements only?
4. Key-name unification (`zone` / `boundary` / `geometry` / `name` for a boundary zone): should it be a code issue now, or left alone?
