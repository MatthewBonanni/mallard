# The GUI

A graphical front end for Mallard
([#346](https://github.com/MatthewBonanni/mallard/issues/346)): case setup,
launching and monitoring runs locally, over SSH on Slurm, PBS and LSF clusters,
and on Kubernetes, moving data between those places, and basic post-processing.
This page is the design round: the decisions, the interfaces Mallard must
provide, what a throwaway prototype showed, the MVP and the follow-ups. Nothing
here is built yet.

The GUI lives in its own repository (working name `mallard-gui`). This page
stays in Mallard because most of it constrains Mallard's interfaces: the key
catalog, the run log, `mallard ctl`, the remote helpers and the capabilities
query. The GUI repository's README points here; its internal design (widgets,
state management, its own tests) lives there.

- [Decisions](#decisions)
- [Architecture](#architecture)
- [Versioned interfaces and compatibility](#versioned-interfaces-and-compatibility)
- [Where state lives](#where-state-lives)
- [Case setup](#case-setup)
- [Backends](#backends)
- [Launching, monitoring and control](#launching-monitoring-and-control)
- [Data: locations, manifest and transfers](#data-locations-manifest-and-transfers)
- [Viewing results: remote rendering by default](#viewing-results-remote-rendering-by-default)
- [Installation](#installation)
- [Desktop and web shells](#desktop-and-web-shells)
- [Design language](#design-language)
- [Testing and CI](#testing-and-ci)
- [Prerequisites against the current code](#prerequisites-against-the-current-code)
- [Prototype findings](#prototype-findings)
- [MVP and milestones](#mvp-and-milestones)
- [Follow-ups](#follow-ups)
- [Decisions on the design round's questions](#decisions-on-the-design-rounds-questions)

## Decisions

1. **A standalone application, not a ParaView plugin**, with VTK for 3D through
   [trame](https://kitware.github.io/trame/) (Kitware's Python framework for web
   UIs over VTK). One codebase serves a web UI in a browser and a desktop app.
2. **The GUI never duplicates solver logic.** Everything that needs to know
   physics, keys, defaults, units, file formats or validation asks the `mallard`
   of the place the case runs, through [`mallard session`](session.md): one
   resident process per case speaking JSON-RPC over stdio, whose every request
   is also a one-shot command. The GUI hard-codes no input key; new keys appear
   without a GUI release.
3. **`input.toml` is the single source of truth for a case.** The GUI edits it in
   place, keeping comments and layout, and its own project file never holds a
   solver key.
4. **A separate repository with its own releases**, pinned to Mallard through
   versioned interfaces (schema versions and a capabilities query), not through
   lockstep versions. Logic that must run next to the data (manifests, hashing,
   restart compatibility, install verification) is `mallard` subcommands, so it
   ships with the solver.
5. **Nothing installed or left running on a cluster beyond Mallard.** No daemon,
   no stored credentials. SSH with the user's own `~/.ssh/config`, keys and agent
   over one OpenSSH ControlMaster connection (MFA answered once); Kubernetes with
   the user's kubeconfig context. Jobs, helper pods and viewer jobs are
   short-lived and torn down by idle timeouts that do not depend on the GUI
   staying open.
6. **Every file shows where it lives, and is known by its content.** Local, or a
   named remote, always visible; no silent copies. Outputs are immutable objects
   identified by their SHA-256 (append-only streams by length and prefix hash),
   so a copy is identical or absent, never stale. Copies are explicit transfers
   with sizes, time estimates, resumption and checks.
7. **Results are rendered next to the data by default.** A short-lived viewer job
   (a Slurm job, or a pod on the volume) streams images or in-memory geometry to
   the UI through an SSH tunnel or a port-forward. Local copies are something the
   user does on purpose.
8. **One backend abstraction** covers local, SSH with Slurm (PBS and LSF through
   small adapters) and Kubernetes; remote-to-remote moves are a route of the same
   transfer engine.
9. **Remotes are global, projects reference them by name.** The project files
   are committed text with no secrets that record runs' identity only;
   per-user state stays in the user config directory and where copies exist in a
   per-machine cache.
10. **uv everywhere**, for the GUI and Mallard: `uv tool`, `uv venv`, `uv pip`,
    `uvx` in docs, scripts and installers, never plain pip, venv or pipx.

## Architecture

```text
 user's machine                                    a remote (cluster or Kubernetes)
 ┌──────────────────────────────────────┐         ┌───────────────────────────────────────┐
 │ browser or desktop shell (UI only)   │         │ login node / API server               │
 │        │ websocket, 127.0.0.1+token  │  stdio  │   mallard session  (one per case,     │
 │ mallard-gui backend (Python, trame)  │  over   │     JSON-RPC, idle timeout)           │
 │   backends: local | ssh+slurm | k8s  │  SSH /  │   mallard ctl | files (routed, short) │
 │   transfer engine, project model     │ kubectl │ solver job      mallard -i input.toml │
 │   local mallard session (setup)      │  exec   │   writes mallard.jsonl, outputs,      │
 │        │ stdio                       │ tunnel/ │   its endpoint record mallard.run.json│
 │ local mallard runs (detached)        │ port-fwd│ viewer job      trame/VTK, near data  │
 └──────────────────────────────────────┘         └───────────────────────────────────────┘
```

- **UI**: Vue components in the browser (trame's client). It holds no state that
  matters; closing it loses nothing.
- **Backend**: one Python process per user session, bound to `127.0.0.1` on a
  random port, with a per-session token in the URL, since other users of a shared
  machine can reach its localhost ports. It talks to Mallard through one
  [`mallard session`](session.md) per open case (locally, or through
  `ssh host mallard session` or `kubectl exec -i`), which keeps the case's mesh
  and setup between edits, owns the project model and the transfer engine, and
  renders local data with VTK.
- **Remote processes**: a `mallard session` per open case on the login node or
  in a helper pod (for small cases; a case whose mesh is large runs its session
  as a job, see [Scale](#scale)), ended by EOF or its idle timeout; short
  `mallard ctl` and `mallard files` commands; solver jobs; viewer jobs while
  viewing. All of them end on their own.
- **No in-process solver.** The programmatic interface of
  [#345](https://github.com/MatthewBonanni/mallard/issues/345) is not on the
  critical path: remote runs need a process boundary anyway, and
  `mallard session` is the programmatic interface the GUI sits on. The Python
  bindings become useful later for in-process previews of local cases.

Why a separate repository (user decision; the tradeoff, for the record): the GUI
has its own release cadence, heavy dependencies (VTK, trame, a desktop shell)
that Mallard's CI should not build, and UI tests unrelated to the solver.
The cost is drift between the GUI and the solver, which the versioned interfaces
below contain. An in-repository `gui/` would have released them together and
needed no compatibility matrix.

## Versioned interfaces and compatibility

The GUI talks to Mallard only through [`mallard session`](session.md) and the
run-side files and commands below, each with a schema name, an integer version
in its JSON and a JSON Schema in `schemas/`. Every session request is also a
one-shot command (the rows below), which runs the same handler:

| Interface | Command | Version today |
|---|---|---|
| Session | `mallard session [INPUT]`: JSON-RPC 2.0 over stdio, the requests of [The session](session.md#requests) | missing |
| Capabilities | `mallard version --format json`: version, build (dimension, precision, backends, MPI, HDF5, ...), installed dimensions, and the schema and version of every interface below | `mallard-capabilities` 1 (#1314) |
| Key catalog | `mallard list keys [INPUT] --format json` | `mallard-key-catalog` 1 |
| Check report | `mallard check INPUT --format json` | `mallard-check` 1 |
| Case view | `mallard check --view DIR`, `mallard info MESH --view DIR` | none |
| Run log | `mallard.jsonl` | `mallard-run-log` 2; its run block `mallard-run` 3 |
| Run directory | `mallard info RUN_DIR --format json` | `mallard-run-directory` 1 |
| Run control | `mallard ctl RUN_DIR status\|checkpoint\|stop\|set`, `STOP`/`CHECKPOINT` files | `mallard-run-control` 1, with its commands and files |
| Remote helpers | `mallard files`, `mallard install verify` (new, below) | missing |

Rules:

- **Minor changes are additive** (new fields, new record types); readers ignore
  what they do not know. A **major** change (a field removed or its meaning
  changed) bumps the version. Mallard publishes a JSON Schema per interface in
  `schemas/` and tests its output against it ([schemas and
  versions](session.md#schemas-versions-and-compatibility)).
- **The GUI declares the range it supports per interface.** On connecting to a
  `mallard`, local or remote, it runs the capabilities query once and caches it
  per install.
- **Older `mallard`** (an interface below the GUI's minimum, or no capabilities
  query): features that need the missing interface are disabled with a one-line
  reason and an "upgrade Mallard on hpc" action. Monitoring through the run log
  works back to run log version 3 (Mallard 0.10).
- **Newer `mallard`** (an interface above the GUI's maximum major): read-only for
  that interface (monitoring keeps working if the run log major is known), and a
  prompt to upgrade the GUI.
- **Compatibility matrix**, published in the GUI's README and tested in its CI:
  each GUI release against the current and previous minor release of Mallard and
  `main`. Mallard's own CI uploads the JSON of every interface from its examples
  as an artifact, which the GUI's CI uses as fixtures.

## Where state lives

Three places, by who owns the state and whether it is authoritative:

| Where | What | Shared | Authoritative |
|---|---|---|---|
| User config directory | remotes, settings | no | yes, for this user |
| Project directory (committed) | `mallard-project.toml`, `mallard-runs/<run>.toml`: identity only | yes, through git | yes |
| User cache directory | where copies of objects are, transfer state, measured throughput | no | never: rebuilt from the data |

### Global, per user: the user config directory

`~/.config/mallard-gui/` (the platform's config directory on macOS and Windows):

- `remotes.toml`: every remote, with a stable `id` (a UUID made when the remote
  is added) and a `name`; the connection (an SSH host alias from `~/.ssh/config`,
  or a kube context and namespace); the backend (`slurm`, `pbs`, `lsf`, `k8s`);
  the Mallard installs there (path or image, version, variant, verification
  result); defaults (partition or queue, account, GPU type); storage roots
  (scratch, project space, or a PVC and its mount path); viewer-job settings
  (partition, size, walltime, idle timeout); cached probe results (which
  remotes reach which directly, and how).
- `settings.toml`: theme, density, units, window layout, recent projects.
- `remote-ids.toml`: which local remote holds runs recorded under another remote
  id (see below).

No secrets. SSH keys and agents, kubeconfig and MFA stay with OpenSSH and
kubectl. The only secret the GUI could need (a registry token for a private
image, an object store key) goes in the OS keychain, never in a file.

### Per project: `mallard-project.toml` and `mallard-runs/`

A project is the directory that holds `input.toml` (a case folder or a
repository). The GUI writes two kinds of committed files:

- **`mallard-project.toml`**, next to `input.toml`: a visible, self-describing
  name rather than a hidden directory, so users see it, commit it and know what
  wrote it. It holds `schema = "mallard-project"` and `version = 1`; the remotes
  the project uses, by name, with only overrides that belong to the project
  (account, partition, run-directory root on that remote); the inputs (which
  `.toml` files are cases); saved views and plots.
- **`mallard-runs/<run>.toml`**, one file per run, committed: the run's durable
  identity only. Its `run_id` (from its run log), its home (the remote's name
  and stable id, and the run directory there), the SHA-256 of the input it ran,
  the Mallard version and install, the job id, and status pointers (the outcome
  and the last snapshot and restart, by path and SHA-256, as the run log's end
  record gives them). One file per run keeps diffs small and lets runs started
  by different people merge without conflicts.

They hold no copy or sync state: where copies of a run's files exist is
per-machine knowledge and lives in the [cache index](#the-manifest-and-the-cache-index). A run's
file changes when the run starts, when it ends (status pointers), and when it is
moved (its home); nothing else. Neither file duplicates solver keys or the run
log. Files are written with one item per line and sorted keys, so they diff and
merge cleanly.

```toml
# mallard-project.toml
schema = "mallard-project"
version = 1
inputs = ["input.toml"]

[remotes.hpc]                  # a name from the user's remotes.toml
account = "cfd-alloc"
run_root = "/scratch/project/cylinder"

[[views]]
name = "wake vorticity"
field = "VORTICITY_Z"
camera = {position = [3.0, 0.0, 12.0], focal_point = [3.0, 0.0, 0.0], up = [0.0, 1.0, 0.0]}
colormap = "vik"
```

```toml
# mallard-runs/re100-fine.toml
schema = "mallard-run-record"
version = 1
run_id = "3a256c99-a024-4845-8b46-5848d430e28b"
home = {remote = "hpc", remote_id = "5f0c1e2a-7d4b-4a51-9a52-0d2b9e4f6c11", directory = "/scratch/project/cylinder/runs/re100-fine"}
input_sha256 = "9b1f…"
mallard = "0.11.0-cuda"
job = "48213"
outcome = "finished"
last_snapshot = {path = "solut/cylinder_000160.vtkhdf", sha256 = "41d2…"}
last_restart = {path = "solut/restart_2066000.restart", sha256 = "c07a…"}
```

**Shared with a teammate.** Remote names are looked up in the teammate's own
`remotes.toml`. A run records its home remote's name and id: if the teammate
has no remote of that name, or one with another id, the GUI shows the run as "on
hpc (not one of your remotes)" and asks which of their remotes holds it; it
checks the answer by reading `mallard.jsonl` in the recorded directory there and
comparing `run_id`, then stores the mapping in the teammate's
`remote-ids.toml`, not in the project. The same happens when a user renames a
remote.

**Missing or damaged.** The run files can be rebuilt: the GUI scans the project
directory and each referenced remote's run root for `mallard.jsonl` files and
identifies runs by `run_id`. A run file whose run directory has disappeared is
shown as such, never deleted automatically.

## Case setup

### Driven by the key catalog

The forms come from the catalog the `mallard` that will run the case exports
([#1256](https://github.com/MatthewBonanni/mallard/issues/1256): keys declared
once in the code that reads them, exported by `mallard list keys --format json`):

- **Widgets from declarations**: type gives the widget (number, integer, bool,
  string, array, table); accepted values a select; shape a vector editor with
  one cell per component; units a unit-aware field (a plain number in the key's
  units, or a quantity with a unit, as the input accepts); range a validation
  rule; expressions a code field listing the variables they read (`x`, `y`, `t`,
  `u`, `v`, `n`); basic or expert a disclosure level; description and guide
  anchor the help.
- **What applies**: after each edit (debounced), the GUI sends the text to the
  case's session (`case.update`) and asks which keys this input reads
  (`case.keys`, the one-shot `mallard list keys INPUT --format json`), so the
  form shows exactly the keys that apply (the boundary fields of the chosen
  condition, the keys of the chosen reconstruction). This needs a partial-input
  mode: today the command fails until the input is complete (see the
  [prototype](#prototype-findings)).
- **Arrays of tables** (`[[boundaries]]`, `[[probes]]`, `[[write_data]]`) are
  lists of elements with a discriminator (`type`) and per-variant keys; the
  catalog must say which keys belong to which variant, or the GUI ends up
  hard-coding them, as the website's builder does today.
- **Roles of string keys**: which strings are files (mesh, mechanism, restart,
  data tables) and of what kind, so the GUI offers file pickers, previews and
  transfers for them.
- **Validation** is the session's `case.check` (the one-shot `mallard check
  INPUT --format json`), run on demand and after edits that change the setup;
  the session redoes only the setup stages the edit invalidates, so the mesh is
  never read again for an edit outside `[mesh]`. Its diagnostics are attached
  to fields by key, line and element, which needs structured fields (today they
  are only in the message text).
- **Editing the file**: a style-preserving TOML editor (tomlkit) changes values
  in place, so comments, order and layout survive; a side-by-side TOML view is
  two-way synchronized. `mallard input upgrade` is offered when the input uses
  renamed keys.
- **Starting points**: `mallard list templates` and `mallard new`, the same
  templates as the command line.

### The website's input builder

The builder (`content/assets/builder.js` in `MatthewBonanni/mallard-website`,
[website #62](https://github.com/MatthewBonanni/mallard-website/issues/62))
is about 370 lines of plain JavaScript that runs in the browser with no solver.
It reads the committed catalog (`input-keys.json`) but hard-codes the form: the
sections, the zones of each built-in mesh and the fields of each boundary
condition (`BC_FIELDS`). Decision: **share the contract, not the code.** Both
read the same exported catalog (#1256), and the catalog carries what both
hard-code today (variants, roles, order, basic or expert), so the builder can
drop `BC_FIELDS` too. The stacks differ (static plain JavaScript against trame
and Vue with a solver behind it), and the GUI's form depends on asking a live
`mallard` what applies, which the website cannot do. If the two form generators
later turn out to duplicate real logic, a shared Vue component is possible:
trame hosts custom Vue components.

### The case tree

The case is one tree, in the order a case is built: Mesh, Physics, Boundaries
(one node per zone), Initial state, Numerics, **Regions**, Probes, Output,
Statistics, Run. Each node shows its key value (`navier_stokes`, `TENO 3`), and a
badge with the count of errors or warnings `mallard check` attaches to it. The
3D view, the tree and the property grid are always visible together; selecting
in any one selects in the others.

![Case setup in 2D (examples/cylinder): the case tree with Regions as a section of its own, the case view with zones, regions and probes, the problems panel, and the selected region's properties with where it is used](../images/design/gui-case-setup-light.svg#gh-light-mode-only)
![Case setup in 2D (examples/cylinder): the case tree with Regions as a section of its own, the case view with zones, regions and probes, the problems panel, and the selected region's properties with where it is used](../images/design/gui-case-setup-dark.svg#gh-dark-mode-only)

### Regions: define once, preview, reuse by name

In Mallard a region is a general geometric primitive, not a probe setting
([Geometry](../guide/geometry.md#who-uses-what)): `[[sources]]`, `[[sponges]]`,
`[[probes]]`, `[[write_data]]`, `[[initialize.regions]]`, `[[initialize.data]]`,
`[[statistics.conditional]]`, `[[mesh.curved]]`, `mallard post extract`,
`mallard post average` and `mallard check --mesh-fields` all take one; regions
can be primitives, regions of the mesh (`domain`, boundary and cell zones),
booleans (`union`, `intersection`, `difference`), boundaries and interiors of
other regions, transformed, repeated as arrays of copies, moving, defined by the
solution, and (with [#1285](https://github.com/MatthewBonanni/mallard/issues/1285))
closed CAD surfaces. So Regions is a first-class section of the case tree:

- **Every region the case uses is listed**: the named ones of `[regions]`, the
  inline tables of each feature (named after their owner, such as
  `near_wake` from `write_data[2]`), and the mesh's own (`domain`, the zones).
  An inline region can be promoted to `[regions]` in one action, which rewrites
  the input to reference it by name.
- **Each region shows where it is used** ("used by": every feature that
  references it, with a link to that feature's node) and, from `mallard check`,
  its dimension, cells, measure and cut cells.
- **Every feature's form picks regions from the same list** (any region of the
  dimension that feature takes, as the catalog says), or defines a new one
  in place; a "use in…" action adds the selected region to another feature.
- **Previews**: every region is drawn in the case view (filled regions with
  their fractions, surfaces and curves with their normals), highlighted when
  selected in the tree, a form or the view.
- Probes are one consumer among others: the Probes section lists probes (points,
  curves, boundary zones, region reductions), each referencing a region or a
  probe type.

### Mesh import and the case view

- **Import**: a mesh file is read by `mallard info MESH --format json` (format,
  cells, cell types, bounds, boundary and cell zones with face counts, quality
  with the worst cells) and drawn from `mallard info MESH --view DIR`. Formats
  Mallard converts go through `mallard mesh convert`. A mesh on a remote is
  inspected there; nothing is copied to preview it.
- **Boundary assignment**: the case view has one block per zone, colored with a
  colorblind-safe categorical palette; picking a zone selects its
  `[[boundaries]]` element, and zones without a condition are highlighted until
  assigned, as `mallard check` reports them.
- **Boundary coordinates** (#346): on a selected patch, show `u`, `v` and set
  `origin` and `toward` by clicking on it.
- **Blocks link to the input**: `mallard check --view` draws every zone, region,
  probe and periodic pair as named blocks (and annotated SVG images,
  [website #61](https://github.com/MatthewBonanni/mallard-website/issues/61));
  linking each block to its input element needs block metadata, and rerunning on
  every edit of a large mesh needs a geometry-only view
  ([#1313](https://github.com/MatthewBonanni/mallard/issues/1313)).

**Navigation**, the same in 2D and 3D: zoom (wheel, pinch), pan (shift-drag or
middle button), orbit (drag), fit to view (F), view presets (+X, −X, +Y, −Y, +Z,
−Z, iso), perspective or orthographic projection, and a 2D mode that locks
rotation to the plane (on by default for 2D cases). Coordinates and the cell or
face under the cursor show in the view's corner.

**The section tool**, because the surfaces that need a condition are often hard
to reach: a sphere or a blade inside a far-field box, an internal baffle, a
periodic pair behind another zone.

- One or more **clip planes**, axis-aligned with snapping or arbitrary, dragged
  with a plane widget (origin handle, normal arrow to tilt), each keeping one
  side; a **box clip**.
- **Hide**, **isolate** and **show only selected** zones; **x-ray** draws the
  other zones translucent.
- **Picking through the cut**: picking works on interior and enclosed surfaces
  the section exposes, and ignores clipped geometry.
- The **zone list stays in sync** with the 3D selection, so a face that cannot be
  reached in the view is picked from the list, and selecting in the list
  highlights it in the view.

![Case setup in 3D (examples/sphere_re300): the sphere enclosed by the far-field zones, exposed by a clip plane and picked through the cut, with the zone list, the boundary condition and the section settings](../images/design/gui-case-setup-3d-light.svg#gh-light-mode-only)
![Case setup in 3D (examples/sphere_re300): the sphere enclosed by the far-field zones, exposed by a clip plane and picked through the cut, with the zone list, the boundary condition and the section settings](../images/design/gui-case-setup-3d-dark.svg#gh-dark-mode-only)

**On VTK and trame.** Clip planes are `vtkPlaneWidget`/`vtkImplicitPlaneWidget2`
driving a `vtkPlane` set on the mappers' clipping planes (no new geometry), the
box clip `vtkBoxWidget2` with `vtkPlanes`; x-ray is per-actor opacity with
depth peeling; isolate and hide toggle actors per block. Picking uses hardware
selection (`vtkHardwareSelector`, cell and block ids rendered to an offscreen
buffer), which honors clipping planes and opacity, so a click hits the first
visible surface, interior ones included; `vtkCellPicker` is the fallback. In
trame, local rendering (vtk.js in the browser) has its own widgets and picking
for small cases; remote rendering keeps them on the server, where picking is a
round trip.

**Large meshes.** The view never loads the volume mesh into the browser: it
shows zone surfaces, decimated above a few hundred thousand triangles, with the
full-resolution surface used for picking on the server, and a cut through the
volume computed on the server only for the plane shown. Meshes of hundreds of
millions of cells and more use levels of detail made by Mallard near the data;
see [Scale](#scale).

## Backends

One interface, three implementations (plus PBS and LSF as adapters of the
SSH backend, which differ only in submission and queue queries):

| Operation | Local | SSH + Slurm (PBS, LSF) | Kubernetes |
|---|---|---|---|
| Connect | none | `ssh -o ControlMaster=auto -o ControlPersist=…` to the host alias; MFA once | kube context and namespace from kubeconfig; `kubectl auth can-i` for jobs, pods, exec, port-forward |
| Probe | `mallard version --format json` | same, through the install's path or `module load` line; `sinfo` (partitions, GPU types), `module avail`, `apptainer`, `spack` | same in a short pod of the chosen image; node labels for GPU types |
| Stage inputs | none | transfer engine: rsync over the master connection | transfer engine into the PVC through a helper pod |
| Submit | the session's `run.launch`: a detached `mallard -i` (`mpirun -n N` optional) | job script in the run directory, `sbatch` (`qsub`, `bsub`) | a `Job` (or `MPIJob`) with GPU requests and node selectors, the image pinned by digest, the PVC mounted |
| Queue state | process status | `squeue`, `sacct` (`qstat`, `bjobs`) | `kubectl get job,pod -o json`, events |
| Logs | files | `mallard.jsonl` and stdout files on the shared filesystem | `mallard.jsonl` on the PVC; `kubectl logs` for the container's output |
| Control | `mallard ctl RUN` (routed by the run's endpoint record) | `mallard ctl RUN` on the login node, which routes through `srun --jobid=<id> --overlap` | `mallard ctl RUN` in the helper pod, or its `route_hint` (`kubectl exec` in the rank 0 pod) run by the backend |
| List files | `mallard files RUN_DIR` | `mallard files` over SSH | `mallard files` in the helper pod |
| Transfer | local copy | rsync, or streams with offsets | streams through `kubectl exec` (tar or offsets); `kubectl cp` only for small files |
| Remote view | local VTK | viewer job, SSH tunnel | viewer pod on the PVC, `kubectl port-forward` |
| Cleanup | none | `scancel` of viewer jobs; nothing else persists | delete helper and viewer pods (they also have `activeDeadlineSeconds` and idle exits) |

The helper pod is the pattern of `docs/guide/k8s/files.yaml`: the Mallard image
with the PVC mounted, created on demand, deleted after an idle minute or on
exit, with `activeDeadlineSeconds` as the backstop. Everything runs as the user
with what their kubeconfig allows; nothing cluster-wide is installed.

Where they differ: Kubernetes has no login node, so every remote command needs a
pod (startup of a few seconds, hidden by keeping one helper pod while the GUI is
active); the scheduler is Kubernetes itself (or Kueue where installed) rather
than a batch queue; multi-node runs need the MPI Operator, which comes after
the MVP (single-node `Job`s first). SSH clusters have a
shared filesystem the login node sees directly, and job-step attachment
(`srun --overlap`) instead of `exec`.

## Launching, monitoring and control

- **Launch**: `mallard check` on the target first (locally for local runs, on
  the login node or in a pod for remote ones, with `--ranks` and `--memory` for
  the job's size), then submit. The job script comes from the backend's
  template (the job scripts of `tools/jobs/` are the starting point) and is
  written into the run directory, so it is visible and reusable without the
  GUI.
- **Monitor**: the run log `mallard.jsonl` is followed by byte offset (the GUI
  reads only what is new), over SSH or the helper pod. Its start record carries
  the provenance and placement (rank 0's host included), progress records give
  step, `t`, `dt`, CFL, throughput, ETA, memory and the monitors, output records
  name every file written with its size and SHA-256, and the end record says how
  the run ended. Probes, forces and integrals are small CSV files read the same
  way. Monitoring survives the GUI closing, since it is only reading files.
- **Control**: one call, `mallard ctl <run_dir|run_id> stop|checkpoint|set|status`
  (the session's `run.control`), on whatever machine sees the run directory.
  It reads the endpoint record the run writes at start (host, PID, socket,
  scheduler job and step) and picks the transport: the socket on the same host,
  `srun --overlap` on the job's node, `kubectl exec` in the run's pod, SSH to a
  PBS or LSF job's node, a scheduler signal, and control files last. The reply
  carries the request's ID and returns when the run confirms ("stop accepted at
  step N, restart written: path"), and the request and its result are run-log
  records. The GUI never writes a control file
  ([Run control](session.md#run-control)).

## Data: locations, manifest and transfers

### Content addressing

Every file a run writes is identified by its content, not by its path or by
where a copy happens to be:

- **Objects**: snapshots, restart files, extracts and the files a run read
  (input, mesh, mechanism, data tables) are written once and identified by their
  SHA-256, which the run log's output records already hold (and `mallard pack`
  for the files read); large files are identified by a chunk tree of SHA-256
  hashes ([Scale](#scale)). A copy is either byte-identical to the object, or it does
  not exist; nothing can be "stale". Two runs that wrote the same bytes share
  one object.
- **Streams**: the run log, probe, force and integral CSV files and progress
  only ever grow while a run continues. A copy of a stream is a prefix of it,
  verified by the SHA-256 of that prefix: it is never stale, only shorter, and
  syncing it means appending the missing bytes.
- **Derived files** (the series indexes `.pvd`, `.vtkhdf.series`, `.xmf`, the
  run report `report.html`) are regenerated from objects and the run log where
  they are needed, not transferred as data.

What makes this hold, and where Mallard does not yet, checked against the
current code:

| Output | Behavior today | Fits |
|---|---|---|
| Restart files | written as `.partial`, renamed when complete | object |
| Snapshots with `series = false`, VTU, per-step VTP | one file per snapshot | object |
| Run log, CSV probes, forces, integrals | appended when a run continues from a restart; truncated by a fresh start in the same directory | stream (a fresh start is a new stream: the prefix hash no longer matches, and the run's `start` record has a new `run_id`) |
| VTKHDF series (`series = true`, the default; boundary and region writers always) | one file per snapshot and the mesh once, each written once; `PREFIX.vtkhdf` an index of HDF5 virtual datasets over them, rewritten after each snapshot | objects, and a derived index |
| HDF5 and VTKHDF probes | rows appended into one HDF5 file in place; cut back by a rewind, after a copy to `.rewound/<run_id>/` | no: live files, read on the remote ([#1317](https://github.com/MatthewBonanni/mallard/issues/1317)) |
| Snapshots and restart files after a rewind (resuming from an earlier restart) | the files of the later steps moved to `.rewound/<run_id>/` before the run writes, listed in its `start` record (`restart.superseded`) | objects: kept, at the path the record gives |
| Series indexes, `report.html` | rewritten | derived |

Why and how: [output objects](output_objects.md). A probe's HDF5 or VTKHDF
file stays one growing file: a file per sample, or an index rewritten at every
sample, would cost the step loop. The GUI shows it as a live file, viewable
remotely and not copyable as an object until its run has ended.

### The manifest and the cache index

A run's manifest is what `mallard files RUN_DIR --format json` returns where the
run lives: every object with its SHA-256, size, kind, step and `t`; every
stream with its length and the SHA-256 of its prefix up to that length; the
files present but not recorded (`.partial` files, a run still writing). Nothing
but `mallard` is needed on the remote.

Where copies of objects exist is a **cache index** in the user cache directory
(`~/.cache/mallard-gui/`): for each location (local, or a remote id), which
object hashes it holds at which paths and how long each stream copy is, plus
interrupted transfers and the throughput measured per route. It is per machine,
never committed and never authoritative: it is rebuilt by asking each location
for its manifest (`mallard files --hash` hashes what the run log does not
record) and by hashing local files, and an entry the location contradicts is
dropped.

The run's home (where it ran) is always shown with its path, as
`hpc:/scratch/…/re100-fine`; a copy shows as `local:runs/re100-fine/…`. Each
file has, per location, one of three states: present (an object there, or a
stream prefix with its length), absent, or partial (a transfer interrupted,
resumable).

### Selecting what to move

Selections are by what users think in, not by path: one snapshot; snapshots of
a time or step range; probes, forces and integrals only; restart files (the
newest, or one chosen); a decimated surface (made next to the data by the
viewer or `mallard post extract`, see [follow-ups](#follow-ups)); the case
inputs; or the whole run. Before anything moves the GUI shows the object count,
total size (objects the destination already holds are skipped and shown as
such) and an estimated time per route from the throughput measured on earlier
transfers of that route, or a short probe the first time.

### The transfer engine

A transfer is "fetch the object with SHA-256 H into location L, from any
location that has it" (and, for a stream, "append bytes N onward"). Sources come
from the cache index and are confirmed by their manifests; the engine picks the
cheapest route, which the user can override:

- **local ↔ remote**: rsync over the SSH master connection
  (`--partial --append-verify`) where the remote has rsync, else a stream with
  byte offsets; through the helper pod for Kubernetes.
- **remote ↔ remote, direct**, when one can reach the other: rsync from one
  login node to the other with the user's forwarded SSH agent (or a jump host);
  a shared filesystem mounted on both (no copy at all, the object is already
  there); a Kubernetes pod pulling from an SSH remote only if the user opts in to
  storing a key as a secret.
- **remote ↔ remote, funneled**: the source's stream piped into the
  destination's through the local machine, never written to local disk
  (`ssh A 'tail -c +N file' | ssh B 'cat >> file.partial'`, or the `kubectl
  exec` equivalent).

Every object lands as a `.partial` file, resumes from its length (from any
source holding the same hash, so an interrupted copy from one remote can finish
from another), and is renamed only after its SHA-256 matches H, computed by
`mallard files --hash` on the destination. Streams are appended and their new
prefix hash checked. The cache index records the result; nothing committed
changes.

**Detecting a direct route**: a probe from each remote to the other
(`ssh -A -o BatchMode=yes <other> true` through the master connection; for
shared filesystems, a marker file written in one storage root and looked for in
the other), cached with its date in the global remote config and rerun when a
direct transfer fails.

**Moving a case or a run to another remote.** The user picks the whole run, or
the inputs plus a chosen restart (to continue there), or selected outputs. For a
continuation, the GUI checks the restart on the destination before moving the
gigabytes: the destination's `mallard info RESTART --format json` must read it
(format version, dimension, precision, species); the rank count does not matter,
since a restart reads bitwise the same on any number of ranks
([Initialization](../guide/initialization.md)). A copy leaves the run's home as
it was: the objects are now present at both places. A move changes the run's
home in `mallard-runs/<run>.toml`, the only committed change; the old home is
cleaned only on explicit request, listing what will be deleted.

![The data manager: files of a run with their state at each location, and a transfer with its route and estimate](../images/design/gui-data-manager-light.svg#gh-light-mode-only)
![The data manager: files of a run with their state at each location, and a transfer with its route and estimate](../images/design/gui-data-manager-dark.svg#gh-dark-mode-only)

## Viewing results: remote rendering by default

### What needs no viewer job

The run monitor, plots of probes, forces and integrals, the run's health and
warnings, `mallard info RUN_DIR`, the case view's annotated SVG images and the
run report all come from small text and JSON files read over SSH or `exec`.
This covers most of watching a run.

![The run monitor: progress, the run log's numbers, probe and force plots, events, and a remote 3D view](../images/design/gui-run-monitor-light.svg#gh-light-mode-only)
![The run monitor: progress, the run log's numbers, probe and force plots, events, and a remote 3D view](../images/design/gui-run-monitor-dark.svg#gh-dark-mode-only)

### The viewer job

For fields in 3D, the GUI starts a viewer next to the data: a trame/VTK server
that reads Mallard's output (VTKHDF, VTU, PVD through VTK's readers) from the
shared filesystem or the PVC, as a Slurm job on a small CPU partition, or a pod
with the PVC mounted. The local UI reaches it through an SSH tunnel through the
master connection, or `kubectl port-forward`. It sends either images (remote
rendering, any data size) or decimated geometry that the browser renders with
vtk.js and keeps in memory only. Nothing is written to local disk.

**Budgets.**

| What | Target |
|---|---|
| Viewer job start | queue wait dominated; under a minute on a small CPU partition, with the queue estimate (`squeue --start`) shown while waiting |
| First image of a snapshot | under 5 s for a 1 GB snapshot (read at about 1 GB/s from a parallel filesystem) |
| Camera interaction, image mode | 15 to 30 frames/s at reduced quality while moving, full quality on release, for round trips up to 50 ms |
| Camera interaction, geometry mode | local frame rate; chosen automatically when the round trip exceeds about 100 ms and the extracted surface is under a few million triangles |
| Changing time step or field | under 2 s |

**GPU or CPU.** CPU rendering (OSMesa or llvmpipe) is the default: no GPU
allocation, short queues, and fast enough for surfaces and slices of a few
million triangles. EGL on a GPU node is an option per remote, for volume
rendering and large surfaces. VTK's wheels select EGL or OSMesa at run time
(to be confirmed for the VTK version pinned). One viewer process holds
surfaces and slices of any mesh, as levels of detail; data that must be rendered
whole uses a parallel viewer job ([Scale](#scale)).

**Which software.** The viewer is GUI code (it speaks the GUI's protocol), so it
is versioned with the GUI, not with Mallard: an image `mallard-viewer:<gui
version>` built by the GUI repository's CI from the Mallard image of a pinned
version (with VTK's offscreen build and trame added), used as is on Kubernetes
and through Apptainer on SSH clusters; where there is no container runtime, the
remote install puts the same pinned Python environment next to the Mallard
install: a uv-managed environment (`uv venv` with pinned, hashed requirements)
beside the Mallard install of the same version, created and removed with it
(see [Installation](#installation)).

**Cost and queue wait.** One viewer per remote per GUI session, reused for every
run and snapshot; started when the user opens a 3D view (or ahead of it, if the
user allows); an idle timeout (15 minutes without a client, configurable) after
which it exits on its own; a hard walltime as the backstop; and a cancel when
the GUI exits normally. The partition, account and size come from the remote's
viewer settings, so a site with a small interactive or debug partition can use
it.

**Where port forwarding is blocked**, in order:

1. a jump through the login node to the compute node (`ssh -J`), where compute
   nodes accept SSH from the login node;
2. a reverse tunnel from the job to the login node's localhost (`ssh -R`), then
   the normal tunnel to the login node;
3. the viewer's websocket carried over the SSH channel's standard input and
   output (`ssh login srun --jobid=<id> --overlap mallard-viewer --stdio`),
   relayed to a local port by the GUI: works wherever `srun --overlap` does;
4. the viewer on the login node itself, only as an explicit fallback the user
   chooses each time, with a warning that login nodes are shared, small and with
   a short timeout;
5. no live view: images rendered in a batch job (`mallard post render`,
   [#516](https://github.com/MatthewBonanni/mallard/issues/516)) and shown from
   the files.

On Kubernetes, `port-forward` goes through the API server and is rarely blocked;
`kubectl exec` with standard input and output is the same fallback as 3.

### Basic post-processing

A field on surfaces and slices with a colormap and range; probe time series,
spectra and their reductions; forces and integrals over time; comparisons of two
snapshots through `mallard diff`; averages through `mallard post average` run as
a job; extraction to CSV through `mallard post extract`; images exported from
the current view. Saved views and plots are in the project file. Heavier
analysis stays in ParaView or Python: the GUI opens the same files and says
where they are.

## Scale

The design must hold for meshes of 100 million to over a billion cells, run on
thousands of ranks, whose mesh, snapshots and restarts never fit on one machine
or in one viewer process. At a billion cells, roughly:

| What | Size |
|---|---|
| Volume mesh | 35 GB (tetrahedra) to 90 GB (hexahedra), read by the ranks in shares ([HDF5 meshes](../guide/meshes.md#hdf5-meshes-for-large-runs)) |
| Restart file | 40 GB for a single gas (5 variables × 8 B), 200 GB with 20 species; one file in global order, any rank count |
| Snapshot | 80 GB for ten double-precision fields; one shared VTKHDF file, or one VTU piece per rank (ranks × snapshots files: millions for a long run on thousands of ranks) |
| Boundary faces | about 6 million for a box, 10 to 100 million around a wall-resolved body |
| A run | many TB |

Four rules follow, and every feature is checked against them below:

1. **Nothing of size O(cells) reaches the GUI process, a browser, or a single
   viewer process.** The volume is never loaded for viewing; what is shown is
   bounded in size by a budget, whatever the mesh.
2. **Work goes to the data, in parallel, done by `mallard`**: the same parallel
   readers and partitions a run uses, under the remote's launcher, as a job.
   Short commands on a login node read only metadata; anything O(cells) is a
   job.
3. **Results arrive progressively**: coarse first, refined on demand and on
   zoom, each step cancellable.
4. **Nothing O(files) or O(bytes) runs in the UI thread or as per-file round
   trips over SSH.** Listings, hashing and estimates run on the remote, in one
   process, and come back as summaries and pages.

### Budgets

| What | Target at 10⁹ cells |
|---|---|
| GUI backend process | under 1 GB resident, independent of the case; manifests and the cache index in SQLite on disk, read by page |
| Browser tab | under 1 GB; at most about 2 million triangles of local geometry (vtk.js), otherwise image mode |
| UI thread | never blocked over 50 ms; every remote operation is asynchronous, with progress and a cancel |
| Open a case: zones, kinds, face counts, bounds | under 2 s (metadata only) |
| First 3D preview of the zones | under 10 s once the surface levels exist; making them is a one-off parallel job per mesh, cached by the mesh's hash |
| Boundary assignment change validated | under 1 s, against zone metadata |
| Region preview: geometry / cells and measures | under 1 s for analytic shapes over the surfaces / a geometry-only check as a job, minutes |
| Pick a zone / a face or cell at full resolution | under 0.2 s (hardware selection) / under 1 s (server query) |
| Section cut through the volume | coarse in under 3 s, full resolution in under 30 s |
| Manifest of a million files | first page under 2 s; full index in the background |
| Monitor poll | under 1 s; bytes read proportional to the new records |

### Viewer

**Surfaces, by Mallard, near the data.** The viewer never reads the volume to
draw a mesh. A `mallard` job extracts every boundary zone, and internal zones
and baffles, in parallel: each rank extracts its share of each zone's faces and
decimates it with its partition borders locked, and the pieces are merged into
**levels of detail** (for example 200 thousand, 2 million and 20 million
triangles per mesh) stored as spatial tiles with an index. The result is a
derived object named by the mesh's hash and the extraction parameters, so it is
made once per mesh and shared by every case and run on that mesh. For results,
the same extraction carries the fields of a snapshot onto the surfaces
([#1319](https://github.com/MatthewBonanni/mallard/issues/1319), extended in a
comment).

**Progressive refinement.** The view starts from the coarsest level of every
zone and loads finer tiles where the screen-space error asks for them (zooming
on the sphere loads its fine tiles, not the far field's), within the triangle
budget, evicting tiles that leave the view.

**Rendering.** One viewer process holds the tiles in view, which the budget
bounds, so it suffices for surfaces at any mesh size. Where the data itself must
be rendered whole (a volume rendering, a full-resolution iso-surface of a
billion-cell snapshot), the viewer job becomes parallel: the trame process
drives a ParaView server (`pvserver`) of N MPI ranks that reads the shared
VTKHDF file or the VTU pieces in parallel and composites the image with IceT,
so only images leave the job. N is proposed from the snapshot's size (about one
rank per 50 to 100 million cells, within the remote's viewer limits) and shown
with its cost before the job starts.

**Section cuts and picking at full resolution.** Clip planes and boxes on the
surfaces stay GPU clipping planes on the tiles shown (instant). The cut face
through the volume (the cells a plane crosses) is computed at full resolution
on the server: by the parallel `pvserver` for results, and for setup (a mesh
without a solution) by the case's [`mallard session`](session.md), which keeps
the mesh partitioned in memory and answers `mesh.cut` and `mesh.pick` until
idle ([#1319](https://github.com/MatthewBonanni/mallard/issues/1319)); each request
returns a decimated result first and the full one after. Picking a zone is exact
from hardware selection on any level, since decimation keeps zone membership;
the face or cell under the cursor is found by the same server from the picked
point on the full-resolution data. Reading only the cells near a plane or point
needs a spatial index of the mesh's chunks (the bounding box of each chunk of
the global order), which an HDF5 mesh can carry.

### Setup

- **Zones from metadata.** `mallard info MESH --format json` must return zone
  names, kinds, face counts, bounds and areas without reading the volume: from
  the zone tables of an HDF5 mesh directly, and for other formats from one
  parallel pass, cached beside the mesh by its hash. The GUI recommends
  `mallard mesh convert` for large meshes used more than once.
- **Boundary assignment** works on the zone list and the surface levels, and is
  validated instantly against the metadata (zones without a condition, names
  that are not zones); the full check follows as a job.
- **Region previews**: analytic regions (boxes, spheres, cylinders, their
  booleans and transforms) are drawn as their geometry over the surfaces at
  once; their cells, fractions and measures come from a geometry-only check near
  the data, returned as counts and the region's decimated boundary, never its
  cells. Regions defined by the solution need a snapshot and are previewed on
  the viewer.
- **`mallard check`** of a large case runs under the remote's launcher on the
  job's ranks (it already checks the run on N ranks under `mpirun -n N`), as a
  short job the GUI submits, never on a login node; the GUI estimates its memory
  from the metadata first. It needs the geometry-only mode, a case view that
  writes surfaces and region boundaries instead of a volume block, and region
  blocks written by each rank rather than gathered on rank 0
  ([#1313](https://github.com/MatthewBonanni/mallard/issues/1313), extended in a
  comment).

### Data and transfers

- **Hashes in parallel.** Object ids for large files are a **chunk tree**:
  the SHA-256 of every 64 MiB chunk, computed by all ranks in parallel once the
  file is closed (each reading its share of the chunks), and the object id is
  the SHA-256 of that list (files of one chunk keep their plain SHA-256),
  `sha256_tree` with `chunk_bytes` and `chunks` in the output record. Rank 0
  alone reading back every shared file (about 25 s for a 40 GB restart at
  1.6 GB/s, minutes for a 200 GB one, a queue that a run writing often
  outgrew) is gone. Chunks are also the unit of resume, verification and
  parallel transfer ([#1329](https://github.com/MatthewBonanni/mallard/issues/1329),
  [design](output_objects.md#chunk-tree-hashes)).
- **Manifests of millions of files.** `mallard files` streams JSON lines rather
  than one document, built from the run log (O(records)), listing directories
  only for what the log does not record, all in one process on the remote. It
  groups files into logical objects (a snapshot and its per-rank pieces is one
  entry with a piece count and total bytes), so the data manager lists
  snapshots, not pieces. A `--summary` with a selection (steps, kinds, writers)
  returns counts and bytes only, so estimates are computed on the remote and
  the file list never crosses the wire to price a transfer
  ([#1317](https://github.com/MatthewBonanni/mallard/issues/1317), extended in a
  comment). The GUI stores manifests in its SQLite cache index and pages them.
- **Transfers of many TB** run several streams at once (objects, chunks of a
  large object, or groups of pieces; 8 to 16 by default, per route), each chunk
  resumed and verified on its own, so a 200 GB restart moves at the route's
  aggregate bandwidth and an interruption costs at most one chunk per stream.
  Where both ends are Globus endpoints, Globus is a route the engine can choose
  (a GUI follow-up).
- **Partial subsets.** A Mallard restart is one file whatever the rank count,
  so there are no per-rank restart sets to move: continuing elsewhere moves the
  whole file, in parallel chunks. Its variables are contiguous byte ranges, so a
  subset by variable is a range read; a subset in space is
  `mallard post extract --region` near the data; a subset of ranks is their VTU
  pieces, selectable as such.

### Monitor

- The run log is already followed by byte offset. For long runs (millions of
  steps), each history is kept as min/max/last per bin at a few resolutions,
  bounded to about 10 thousand points per curve per level; zooming in reads the
  missing records by byte range, from an index of record offsets in the cache.
- Probes, forces and integrals in CSV are streams, read from the last offset.
  HDF5 and VTKHDF probe files are rewritten in place
  ([#1326](https://github.com/MatthewBonanni/mallard/issues/1326)), so they are
  read on the remote, new rows only and selected columns only
  ([#1317](https://github.com/MatthewBonanni/mallard/issues/1317)); a probe of
  thousands of points over a long run is never copied whole to plot one curve.

## Installation

### Locally: one command

The GUI is a Python application (trame), so the mechanism is a Python package
installed in its own environment:

```sh
uv tool install mallard-gui              # the GUI alone (it uses a mallard on PATH or a remote's)
uv tool install "mallard-gui[solver]"    # the GUI and a CPU mallard, 2D and 3D
```

plus a desktop installer per platform for users who do not use a terminal (see
[shells](#desktop-and-web-shells)), and a one-line shell installer that installs
uv first where it is missing. `[solver]` depends on the `mallard-python` wheel of
[#792](https://github.com/MatthewBonanni/mallard/issues/792) extended with the
`mallard` executables (CPU with OpenMP, no MPI, serial HDF5) for macOS arm64 and
Linux x86-64 and arm64; the reader libraries it already plans to carry come with
them. Why this one:

- **uv**: one command on every platform, an isolated environment, fast, no
  administrator rights, and the Python ecosystem trame and VTK already ship in
  (VTK's wheels are the supported binaries).
- **conda-forge**: good for VTK, but a second ecosystem to keep current and slow
  for new users to set up; a later channel, not the primary path.
- **Spack** ([#1027](https://github.com/MatthewBonanni/mallard/issues/1027)):
  right for the solver on clusters, wrong for a desktop GUI (build times, no VTK
  with its web stack).
- **Containers**: the GUI needs the user's SSH agent, kubeconfig and a browser;
  a container gets in the way of all three.

A workstation with a GPU uses the GUI with the container image or a Spack build
of Mallard, which the GUI finds on PATH or is pointed at.

### On a remote: one click

When a remote is added, the GUI offers to install Mallard there, versioned to
match what the GUI needs:

- **SSH clusters**, into the user's home or project space, without root:
    1. **Detect**: architecture and glibc (`uname -m`, `ldd --version`), the
       scheduler, partitions and GPU types (`sinfo`), modules (`module -t avail`
       for compilers, MPI, CUDA, ROCm), Apptainer, Spack.
    2. **Choose**, in this order: prebuilt portable binaries for the
       architecture, backend (CPU, CUDA, HIP; a new release artifact) and MPI
       family (MPICH ABI or Open MPI), so they run with the site's MPI through
       `module load`,
       downloaded by the login node from the GitHub release with its SHA-256
       checked; the release's container image through Apptainer (`apptainer pull`)
       for single-node runs; a Spack build in user space with the detected
       compilers, MPI and GPU toolkits as externals, run as a batch job: the
       fallback for clusters with neither family's MPI; slow.
    3. **Install** into `<root>/mallard/<version>-<variant>/` with an
       `install.toml` (version, variant, artifact and SHA-256, how it was
       installed), recorded in the global remote config. Versions live side by
       side; nothing points at "current", the GUI names the path.
    4. **Verify**: `mallard install verify` (new) as a short job on the chosen
       partition: the environment check (`mallard check` without input) and a
       quick template run (`mallard new <template> --quick`), with a JSON result
       recorded with the install.
    5. **Upgrade**: install the new version beside the old one, verify, then make
       it the remote's default; runs record the install they used, so a restart
       can continue on the same one. Old installs are removed only on request.
- **Kubernetes**: choose the image tag for the Mallard version and backend
  (`ghcr.io/matthewbonanni/mallard:<version>` or `:<version>-cuda`), recorded by
  digest; a short pod checks that the namespace can pull it (image pull secrets
  are the user's) and runs the same verification.

Where the remote has no container runtime, the install also creates the
viewer's uv environment beside it (uv itself is a single static binary placed in
the install). Installation leaves files only; nothing runs afterwards.

**Windows** runs the GUI only: the solver runs on remotes, or locally in WSL
(the GUI treats WSL as a local remote).

Once the GUI exists, `mallard gui` becomes a thin launcher in the solver that
starts an installed GUI (or says how to install it with uv).

### What the release pipeline must produce

Today `release.yml` builds and pushes the CPU (amd64, arm64) and CUDA images and
tests the Spack package. For the GUI it must also produce:

- `mallard-python` wheels with the `mallard` executables (CPU), for macOS arm64
  and manylinux x86-64 and arm64 (extends #792);
- portable binaries per architecture, backend (CPU, CUDA 12 and 13, ROCm) and MPI family
  with SHA-256 sums, as release assets;
- immutable version tags for every image variant (`:<version>-cpu-amd64`
  exists; `:<version>-cuda` too), with digests published;
- a `release-manifest.json` asset listing every artifact with its variant,
  SHA-256 and interface versions, which the GUI reads to choose what to install.

## Desktop and web shells

The UI is the same web application in every shell; the Python backend is the
same process.

| | trame desktop (pywebview) | Tauri + Python sidecar | Electron + Python sidecar |
|---|---|---|---|
| Install size | smallest shell (system webview) + Python, VTK (about 200 MB) | shell about 10 MB + the same backend | Chromium about 100 MB more |
| Startup | Python and VTK import, 2 to 4 s | same backend start; window at once | same |
| Native file dialogs | yes | yes | yes |
| OS keychain | Python `keyring` | plugin | `keytar`/safeStorage |
| Auto-update | none | built in, signed updates | `electron-updater` |
| Code signing | by hand around PyInstaller | built-in tooling for macOS notarization, Windows, Linux packages | mature tooling |
| WebGL (vtk.js geometry mode) | system webview: WebKitGTK on Linux is the weak spot | same as pywebview | consistent Chromium everywhere |

**Recommendation: Tauri** wrapping the web UI, with the Python backend bundled
as a sidecar (PyInstaller): a small shell, signed installers and updates on
the three platforms, and the same backend as `uv tool install`. `mallard-gui
--app` with pywebview is the zero-extra-dependency desktop mode for `uv tool` users,
and the interim one until the Tauri bundle exists. Electron's only advantage is
uniform WebGL, which matters only in geometry mode; image mode does not need it.
WebKitGTK's WebGL on Linux has to be measured in the first desktop milestone.

**Web app**: the same backend serves the UI to a browser on the same machine, or
on a host the user controls reached through an SSH tunnel (bound to
`127.0.0.1` there, token in the URL). Never a public multi-tenant service unless
the user asks for one.

## Design language

Contemporary, precise and calm: engineering software that is dense but legible,
with nothing decorative. The quality bar is that of the best current tools
(Linear and Figma for fit and finish, ParaView for what CFD users expect to be
able to do), not their look. The GUI shares its design tokens (colors, fonts,
spacing, radii) with the redesigned website: the token file lives in the website
repository and the GUI pins it by version. The mockups below use provisional
values until that file exists.

The mockups are drawn by `tools/docs/gui_mockups.py`; their numbers are
synthetic, built from the examples' inputs and reference results (cylinder:
St 0.164, Cd 1.36; sphere_re300: its mesh size and zones).

- **Typography**: a high-quality sans for the interface (Inter, or the shared
  token's choice) and a monospace for numbers, units, keys and paths (JetBrains
  Mono), with tabular figures so columns of numbers align; thin spaces as digit
  group separators (`1 284 516`). Fonts are bundled, never loaded from a CDN.
- **Color**: low-chroma neutrals, one accent (a teal, for actions and
  selection), and status colors used only for status (error, warning). Light and
  dark themes from the same tokens, following the system by default. Contrast
  meets WCAG AA (4.5:1 for text, 3:1 for controls and graphics).
- **Data colors**: perceptually uniform, colorblind-safe colormaps: `cividis`
  and `batlow` (sequential), `vik` (diverging, for signed fields such as
  vorticity), and Okabe-Ito for categories (zones, regions, curves). No rainbow
  colormaps by default.
- **Density**: compact by default, because a case setup is read more than it
  is browsed: the case tree, the 3D view and the property grid (label and value
  side by side, units inline) are all visible at once, with thin dividers
  instead of cards and minimal padding. A "comfortable" mode enlarges rows and
  text for touch and for beginners. Hit targets stay at least 24 px (WCAG 2.2
  2.5.8) in both. Provisional tokens:

    | Token | Compact | Comfortable |
    |---|---|---|
    | Base unit | 4 px | 4 px |
    | Row, input, tab height | 24 px (28 px tabs) | 32 px |
    | Body / secondary text | 12 / 11 px | 14 / 12 px |
    | Header bar / status bar | 36 / 24 px | 44 / 28 px |
    | Panel padding | 8 px | 12 px |
    | Radius | 4 px | 6 px |
    | Divider | 1 px `border` | 1 px `border` |
    | Accent (light / dark) | `#0f766e` / `#2dd4bf` | same |

    Secondary text uses the `muted` color, which meets 4.5:1 on every panel; the
    lighter `faint` is for dividers and disabled glyphs only, never for text that
    must be read.

- **Chrome**: a toolbar with the project, breadcrumbs (project / case / node)
  and the run controls; tabs for open cases and runs; a status bar with the
  remote, the job id, step, simulated time, wall time and ETA, and transfers in
  progress. Tree rows have expand glyphs, type icons, and validation badges
  (error and warning counts); hover and selection states differ by background,
  selection also by the accent bar.
- **Plots** always have labeled axes with units, ticks, a grid, and a legend
  when there is more than one curve; field views always have a colorbar with the
  field's name and units.
- **Units everywhere**: every quantity shows its unit, in the monospace, next to
  the value. Fields accept a plain number in the key's units or a quantity with
  a unit (`1 atm`), as Mallard's input does; hovering shows the SI value.
  Nondimensional cases show their reference scales.
- **Motion** only where it carries meaning: a progress bar moving, a panel
  opening from where it was invoked, a transfer's state changing; 120 to 200 ms,
  none with `prefers-reduced-motion`.
- **Icons**: one outline set at 1.5 px strokes on a 24-unit grid, shown at 16
  px: Lucide (ISC license, to be credited in the about box and the
  third-party notices), plus a few drawn in the same style for Mallard's
  concepts: boundary condition types, regions, probes, and the location badges
  (local machine, SSH remote, Kubernetes). The mockups use their own outline
  paths in that style.
- **Errors and validation**: `mallard check`'s diagnostics appear on the field
  they concern (red border, the message under it, its code and a link to its
  docs), as counts on the case tree node that holds them, in a problems panel
  (keyboard: next and previous problem), and in the 3D view (the zone without a
  condition, the region outside the domain, the cell that limits the time step).
  Warnings never block; errors block Run with the reason on the button.
- **Keyboard first, guided for beginners**: a command palette (Ctrl K / Cmd K)
  over every action and every input key ("numerics.face_reconstruction.order"),
  shortcuts for everything frequent, and the raw TOML beside the form. Beginners
  get the guided steps (mesh, physics, boundaries, initial state, numerics,
  output, probes and regions, check and run) with the catalog's basic keys only,
  and templates to start from.
- **Accessibility**: WCAG 2.2 AA: full keyboard operation with visible focus,
  labels and descriptions from the catalog for screen readers, status never
  carried by color alone (an icon and text with it), and resizable text.

## Testing and CI

In the GUI repository:

- **Contracts**: the parsers of every interface against fixtures of each
  supported Mallard version (the JSON Mallard's CI uploads), and against the
  JSON Schemas.
- **Integration**: against the CPU `mallard` of the `[solver]` extra: open a
  template, edit, check, run a few steps, monitor, stop.
- **Backends**: SSH and Slurm against a containerized Slurm cluster with sshd;
  Kubernetes against kind (Kubernetes in Docker); transfers interrupted and
  resumed on purpose; the funneled and direct remote-to-remote routes between
  two such clusters.
- **UI**: trame's browser tests (Playwright) for the main flows, screenshots in
  both themes and densities, and an automated accessibility check (axe).
- **No GPU** in the GUI's CI; the viewer's EGL path is tested by hand per
  release.

In Mallard: the JSON Schemas of the interfaces tested against the output of the
examples (the CLI tests already run every command), and the new subcommands
tested like any other.

## Prerequisites against the current code

| Prerequisite | State | Gap |
|---|---|---|
| Programmatic API (#345) | open | not needed for the MVP: `mallard session` is the interface |
| `mallard session` ([design](session.md), [#1340](https://github.com/MatthewBonanni/mallard/issues/1340)) | missing | the resident JSON-RPC process, its schemas, routed `mallard ctl` with acknowledgments, detached launch |
| `mallard check` (#321) | done: `--format json`, `--view` with block metadata, `--geometry-only`, `--ranks`, `--memory` | partial inputs fail; one setup error per run; `key` and `line` only in message text; setup warnings missing from the JSON |
| Key catalog (#1256) | open: `docs/input-keys/` is committed and inferred from the examples | no descriptions, no schema version; 204 of 511 general entries have no type; no variants for arrays of tables; no roles for file keys; the general catalog is read from a data directory, not the binary |
| `mallard ctl` (#322) | done: socket `mallard.sock`, `status`, `checkpoint`, `stop`, `set` | must run on rank 0's node |
| Control files | done: `STOP`, `CHECKPOINT`, `SET` (#1316) | off by default (`control_file_interval = 0`); `--set control_file_interval=N` turns them on |
| Run log (#326) | done: `mallard.jsonl` version 3 with start, progress, output (bytes, SHA-256), warning and end records | none for monitoring |
| Rank 0's host in the run log | done: the start record's `provenance` run block (`host`) and `placement` (one entry per rank with its host) | none |
| Capabilities query | done: `mallard version --format json` (#1314) | none |
| Remote helpers | partly: `info RUN_DIR`, `pack` list files with SHA-256 | `mallard files` (manifest, hashes of files, ranges and prefixes) |
| Write-once outputs | done ([#1326](https://github.com/MatthewBonanni/mallard/issues/1326)): snapshots, series and rewinds; probe HDF5 and VTKHDF files stay live | none |
| Install verification | done: `mallard install verify --format json` (#1320) | none |
| Decimated surfaces for viewing | missing | `mallard post extract` with decimation |

## Prototype findings

A throwaway prototype (about 180 lines of Python, not in the repository): a
trame app that loads the key catalog from a locally built 2D `mallard`, renders
a form for the cylinder example, writes the TOML, runs `mallard check` and shows
the case view. Built on macOS with trame 4.0, Vuetify 3, VTK 9.7.2 in a uv
environment, and Mallard `main` (2D, Threads, no HDF5), on a coarse cylinder mesh
of 1536 quadrilaterals.

What worked:

- The whole loop, catalog to form to TOML to check to 3D view, runs end to end;
  editing `run.cfl` to 5 in the form made `mallard check` report `W-CFL-LIMIT`
  in the UI.
- `mallard list keys input.toml --format json` (0.1 s) and `mallard check
  --format json --view view` (0.14 s) are fast enough to run after each edit at
  this size.
- The case view is directly usable as the GUI's preview: VTK's multiblock reader
  loads `case.vtm` with one named block per zone, region and probe.
- `mallard info MESH --format json` lists the boundary zones with face counts,
  enough to drive boundary assignment before an input exists.
- The run log of a short run has everything monitoring and the data manifest
  need: rank 0's host, output records with size and SHA-256, progress and end
  records.

What broke or is missing (each is a follow-up below):

1. **The general catalog needs the source tree**: `mallard list keys` without an
   input reads `docs/input-keys/` and fails unless `MALLARD_DATA_DIR` points at a
   source or install tree; its JSON has no schema version.
2. **The catalog is too thin for forms**: no descriptions; 204 of 511 general
   entries and 90 of the 198 keys of the cylinder case have no type; units on 63
   general entries only; defaults are TOML literals in strings (`"\"mesh.msh\""`).
3. **Arrays of tables are flattened**: `boundaries.T` is reported as required,
   though only some conditions take it, and nothing says which keys belong to
   which `type`. The prototype could only show `[[boundaries]]` as raw JSON.
4. **Partial inputs fail**: `list keys` and `check` stop on a missing mesh file
   or a missing stop condition, so a form cannot ask what applies while the case
   is being written.
5. **One error at a time, unstructured**: setup stops at the first semantic
   error (besides unused keys); the key and line are only in the message; a
   warning printed during setup (`W-FORMAT-VTU`) is on stderr and missing from
   the JSON diagnostics.
6. **No link from view blocks to input elements**: block names such as
   `write_data[prefix = "._solut_near_wake"].region` must be parsed to find the
   element.
7. **Rewriting the TOML loses the user's file**: writing it back with a plain
   serializer dropped all 28 comment lines and reformatted 153 lines into 256,
   so the GUI must edit in place with a style-preserving library.
8. **trame state**: edits to items of a list in shared state did not reach the
   server through `v-model`; explicit update events fixed it. A known trame
   pattern, noted for the GUI's state design.
9. **Control files are off by default** and there is no command-line override,
   so a launched run cannot be stopped through `STOP` without editing its input.

Not tested by the prototype: remote backends, transfers, the viewer job, the
desktop shell, 3D meshes and HDF5 builds.

## MVP and milestones

**MVP: one case from setup to results, locally and on one Slurm cluster, with
data that never moves silently.**

1. `uv tool install "mallard-gui[solver]"`; the web UI on localhost.
2. Open a folder with an `input.toml` (or start from a template); the project
   file is created.
3. Case setup from the catalog: the scalar sections and boundary assignment on a
   file mesh, with the case view in 3D and `mallard check` diagnostics inline;
   edits preserve the file.
4. Run locally; monitor from the run log; stop and checkpoint through control
   files.
5. Add one SSH + Slurm remote (global config, existing Mallard install chosen by
   path or module line, verified by the capabilities query); stage inputs, check
   on the login node, submit, follow the queue, the run log, probes and forces;
   stop and checkpoint.
6. The data manager on content addressing: the manifest, states per location,
   selections, sizes and estimates, resumable transfers of objects by SHA-256
   and of streams by prefix between local and the remote, tracked in the
   per-machine cache index.
7. The viewer job on a CPU partition through an SSH tunnel: one snapshot's field
   on surfaces and slices, image mode, idle timeout and cleanup.

Mallard prerequisites of the MVP: the catalog with types, units, descriptions,
variants and a schema version (#1256, at least for the keys of the guided
steps); partial-input `list keys` (#1311); structured diagnostics (#1312); the
capabilities query (#1314); `mallard session` with routed run control (#1340); `mallard files` (#1317); a way to turn control files
on from the command line (#1316).

**After the MVP:**

| Milestone | Scope |
|---|---|
| M2: more places | Kubernetes backend (helper pod, Jobs, viewer pod, port-forward); PBS and LSF adapters; remote-to-remote transfers, funneled then direct, with route probing; restart-compatibility check; moving a run |
| M3: installs | one-click remote install (archive, Apptainer, Spack), `mallard install verify`, versions side by side, upgrades; the release artifacts; the Tauri desktop app with signed installers |
| M4: full setup | regions and probes editing with previews, boundary coordinates by picking, every section from the catalog, geometry-only fast previews, command palette over keys, `ctl set` and `SET` file |
| M5: results | geometry mode (vtk.js), GPU viewer, fallbacks for blocked forwarding, extraction for large data, saved views, plots of probes with spectra, `diff` and `post average` as jobs |
| M6: scale | surface levels of detail made by Mallard and progressive refinement, the resident query process for cuts and picks, the parallel viewer job (`pvserver` with IceT), chunk-tree hashes and parallel transfers, streamed manifests, decimated histories; tested on a mesh of 10⁹ cells |

## Follow-ups

Filed, each part of #346.

**Mallard**

| What | Issue |
|---|---|
| The catalog for the GUI: schema version, types, units, descriptions, variants of arrays of tables, roles of file keys, display hints, typed defaults, exported from the binary | [#1256](https://github.com/MatthewBonanni/mallard/issues/1256) (extended in a comment) |
| `list keys` and `check` on partial inputs | [#1311](https://github.com/MatthewBonanni/mallard/issues/1311) |
| Structured check diagnostics, setup warnings in the JSON, several errors at once | [#1312](https://github.com/MatthewBonanni/mallard/issues/1312) |
| Case-view block metadata and a geometry-only `--view`; at scale, zone metadata without reading the volume, surface-only views, partitioned check | [#1313](https://github.com/MatthewBonanni/mallard/issues/1313) (extended in a comment) |
| `mallard session`: the resident JSON-RPC interface, schemas of every interface, routed and acknowledged `mallard ctl`, detached launch ([design](session.md)) | [#1340](https://github.com/MatthewBonanni/mallard/issues/1340) |
| `mallard version --format json`, the capabilities query | [#1314](https://github.com/MatthewBonanni/mallard/issues/1314) |
| JSON Schemas of the interfaces, tested in CI; the examples' JSON as a CI artifact | [#1315](https://github.com/MatthewBonanni/mallard/issues/1315) |
| `--set KEY=VALUE` (or control files on by default) and a `SET` control file | [#1316](https://github.com/MatthewBonanni/mallard/issues/1316) |
| `mallard files RUN_DIR`: objects by SHA-256, streams by length and prefix hash, `--hash` of files, ranges and prefixes; at scale, streamed and grouped manifests, remote summaries, incremental probe reads | [#1317](https://github.com/MatthewBonanni/mallard/issues/1317) (extended in a comment) |
| `mallard info RESTART`: whether this build can continue from it | [#1318](https://github.com/MatthewBonanni/mallard/issues/1318) |
| `mallard post extract` of decimated surfaces and slices; `post render` ([#516](https://github.com/MatthewBonanni/mallard/issues/516)) for batch images; at scale, parallel levels of detail per zone and a resident process for cuts and picks | [#1319](https://github.com/MatthewBonanni/mallard/issues/1319) (extended in a comment) |
| `mallard install verify` | [#1320](https://github.com/MatthewBonanni/mallard/issues/1320) |
| Write-once outputs: single-file series and rewinds modify files in place | [#1326](https://github.com/MatthewBonanni/mallard/issues/1326) |
| Chunk-tree hashes of output files, computed by all ranks in parallel | [#1329](https://github.com/MatthewBonanni/mallard/issues/1329) |

**Release pipeline**

| What | Issue |
|---|---|
| `mallard-python` wheels with the CPU `mallard` executables (extends #792) | [#1321](https://github.com/MatthewBonanni/mallard/issues/1321) |
| Portable binaries per architecture and backend, against the MPICH and Open MPI ABIs | [#1322](https://github.com/MatthewBonanni/mallard/issues/1322) |
| `release-manifest.json` of artifacts, digests and interface versions | [#1323](https://github.com/MatthewBonanni/mallard/issues/1323) |

**Website**: the builder generated from the exported catalog, without its
hard-coded boundary fields:
[mallard-website #85](https://github.com/MatthewBonanni/mallard-website/issues/85)
(supersedes website #62).

**Later**: `mallard gui`, a thin launcher, once the GUI exists. **GUI repository**
(once created): the MVP above, the viewer image and the compatibility matrix.

## Decisions on the design round's questions

1. **Viewer on clusters without containers**: a uv-managed environment pinned
   beside the Mallard install of the same version.
2. **Direct remote-to-remote copies** use SSH agent forwarding. No key is stored
   as a Kubernetes secret unless the user opts in.
3. **MPI in prebuilt archives**: the MPICH ABI where possible; Spack builds for
   clusters that need their own MPI.
4. **`mallard-runs/` is committed.** It holds only runs' durable identity, so it
   changes when a run starts, ends or moves.
5. **Windows**: the GUI only; the solver on remotes or in WSL.
6. **`mallard gui`**: a thin launcher in the solver, once the GUI exists.
7. **Kubernetes multi-node** (the MPI Operator): after the MVP.
8. **Login-node viewer**: only as an explicit fallback, with a warning that
   login nodes are shared.
9. **Design tokens**: the file lives in the website repository; the GUI pins it
   by version.
10. **Name**: `mallard-gui`, for the repository and on PyPI.
11. **Data model**: content addressing. Immutable objects by SHA-256, append-only
    streams by prefix, identity committed, copies in a per-machine cache.
