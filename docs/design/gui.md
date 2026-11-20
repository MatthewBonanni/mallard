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
- [Open questions](#open-questions)

## Decisions

1. **A standalone application, not a ParaView plugin**, with VTK for 3D through
   [trame](https://kitware.github.io/trame/) (Kitware's Python framework for web
   UIs over VTK). One codebase serves a web UI in a browser and a desktop app.
2. **The GUI never duplicates solver logic.** Everything that needs to know
   physics, keys, defaults, units, file formats or validation asks the `mallard`
   of the place the case runs, through its command line and JSON output. The GUI
   hard-codes no input key; new keys appear without a GUI release.
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
6. **Every file shows where it lives.** Local, or a named remote, always visible;
   no silent copies. Copies are explicit transfers with sizes, time estimates,
   resumption, SHA-256 checks and a record.
7. **Results are rendered next to the data by default.** A short-lived viewer job
   (a Slurm job, or a pod on the volume) streams images or in-memory geometry to
   the UI through an SSH tunnel or a port-forward. Local copies are something the
   user does on purpose.
8. **One backend abstraction** covers local, SSH with Slurm (PBS and LSF through
   small adapters) and Kubernetes; remote-to-remote moves are a route of the same
   transfer engine.
9. **Remotes are global, projects reference them by name.** The project file is
   shareable text with no secrets; per-user and per-machine state stays in the
   user config directory.

## Architecture

```text
 user's machine                                    a remote (cluster or Kubernetes)
 ┌──────────────────────────────────────┐         ┌───────────────────────────────────────┐
 │ browser or desktop shell (UI only)   │         │ login node / API server               │
 │        │ websocket, 127.0.0.1+token  │         │   mallard list keys | check | info |  │
 │ mallard-gui backend (Python, trame)  │  SSH    │   files | ctl   (short commands)      │
 │   backends: local | ssh+slurm | k8s  ├─────────┤                                       │
 │   transfer engine, project model     │ kubectl │ solver job      mallard -i input.toml │
 │   local mallard (check, previews)    │         │   writes mallard.jsonl, outputs       │
 │        │ subprocess                  │ tunnel/ │ viewer job      trame/VTK server,     │
 │ local mallard runs                   │ port-fwd│   idle-stops, next to the data        │
 └──────────────────────────────────────┘         └───────────────────────────────────────┘
```

- **UI**: Vue components in the browser (trame's client). It holds no state that
  matters; closing it loses nothing.
- **Backend**: one Python process per user session, bound to `127.0.0.1` on a
  random port, with a per-session token in the URL, since other users of a shared
  machine can reach its localhost ports. It runs `mallard` commands as
  subprocesses (locally, or through SSH or `kubectl exec`), owns the project
  model and the transfer engine, and renders local data with VTK.
- **Remote processes**: short `mallard` commands on the login node or in a
  helper pod; solver jobs; viewer jobs while viewing. All of them end on their
  own.
- **No in-process solver.** The programmatic interface of
  [#345](https://github.com/MatthewBonanni/mallard/issues/345) is not on the
  critical path: remote runs need a process boundary anyway, and the command
  line with JSON output is the programmatic interface the GUI sits on. The Python
  bindings become useful later for in-process previews of local cases.

Why a separate repository (user decision; the tradeoff, for the record): the GUI
has its own release cadence, heavy dependencies (VTK, trame, a desktop shell)
that Mallard's CI should not build, and UI tests unrelated to the solver.
The cost is drift between the GUI and the solver, which the versioned interfaces
below contain. An in-repository `gui/` would have released them together and
needed no compatibility matrix.

## Versioned interfaces and compatibility

The GUI talks to Mallard only through these, each with a schema name and an
integer version in its JSON:

| Interface | Command | Version today |
|---|---|---|
| Capabilities | `mallard version --format json` (new): version, build (dimension, precision, backend, MPI, HDF5), and the version of every interface below | missing |
| Key catalog | `mallard list keys [INPUT] --format json` | none (no schema field) |
| Check report | `mallard check INPUT --format json` | `version` field |
| Case view | `mallard check --view DIR`, `mallard info MESH --view DIR` | none |
| Run log | `mallard.jsonl` | `schema = "mallard-run"`, `version = 3` |
| Run directory | `mallard info RUN_DIR --format json` | `version` field |
| Run control | `mallard ctl RUN_DIR status\|checkpoint\|stop\|set`, `STOP`/`CHECKPOINT` files | none |
| Remote helpers | `mallard files`, `mallard install verify` (new, below) | missing |

Rules:

- **Minor changes are additive** (new fields, new record types); readers ignore
  what they do not know. A **major** change (a field removed or its meaning
  changed) bumps the version. Mallard publishes a JSON Schema per interface in
  `docs/schemas/` and tests its output against it.
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

### Global, per user: the user config directory

`~/.config/mallard-gui/` (the platform's config directory on macOS and Windows):

- `remotes.toml`: every remote, with a stable `id` (a UUID made when the remote
  is added) and a `name`; the connection (an SSH host alias from `~/.ssh/config`,
  or a kube context and namespace); the backend (`slurm`, `pbs`, `lsf`, `k8s`);
  the Mallard installs there (path or image, version, variant, verification
  result); defaults (partition or queue, account, GPU type); storage roots
  (scratch, project space, or a PVC and its mount path); viewer-job settings
  (partition, size, walltime, idle timeout); cached probe results (which
  remotes reach which directly, measured throughput per route).
- `settings.toml`: theme, density, units, window layout, recent projects.
- `remote-ids.toml`: which local remote holds runs recorded under another remote
  id (see below).

No secrets. SSH keys and agents, kubeconfig and MFA stay with OpenSSH and
kubectl. The only secret the GUI could need (a registry token for a private
image, an object store key) goes in the OS keychain, never in a file.

### Per project: `mallard-project.toml` and `mallard-runs/`

A project is the directory that holds `input.toml` (a case folder or a
repository). The GUI writes:

- **`mallard-project.toml`**, next to `input.toml`: a visible, self-describing
  name rather than a hidden directory, so users see it, commit it and know what
  wrote it. It holds `schema = "mallard-project"` and `version = 1`; the remotes
  the project uses, by name, with only overrides that belong to the project
  (account, partition, run-directory root on that remote); the inputs (which
  `.toml` files are cases); saved views and plots.
- **`mallard-runs/<run>.toml`**, one file per run: the run's name, `run_id`
  (from its run log), the remote's name and id, the run directory there, the job
  id, the Mallard install and version, and its data record: the copies that
  exist outside the run's home and the transfer records. One file per run keeps
  diffs small and lets runs started by different people merge without conflicts,
  as `docs/input-keys/` does for keys.

Neither file duplicates solver keys or the run log: what a run wrote is read
from its `mallard.jsonl` wherever it lives; the per-run file records only what
the run log cannot know (where copies are, what was moved, when). Files are
written with one item per line and sorted keys, so they diff and merge cleanly.

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
remote = {name = "hpc", id = "5f0c1e2a-7d4b-4a51-9a52-0d2b9e4f6c11"}
directory = "/scratch/project/cylinder/runs/re100-fine"
job = "48213"
mallard = "0.11.0-cuda"

[[copies]]                     # a file present somewhere other than the run's home
path = "solut/cylinder_000060.vtkhdf"
at = "local:runs/re100-fine/solut/cylinder_000060.vtkhdf"
sha256 = "cc3b0d4d…"
copied = 2026-11-20T13:40:00Z

[[transfers]]
started = 2026-11-20T13:40:00Z
from = "hpc"
to = "local"
route = "direct"
files = 2
bytes = 4200000
result = "verified"
```

**Shared with a teammate.** Remote names are looked up in the teammate's own
`remotes.toml`. A run records its remote's name and id: if the teammate has no
remote of that name, or one with another id, the GUI shows the run as "on hpc
(not one of your remotes)" and asks which of their remotes holds it; it checks
the answer by reading `mallard.jsonl` in the recorded directory there and
comparing `run_id`, then stores the mapping in the teammate's `remote-ids.toml`,
not in the project. The same happens when a user renames a remote.

**Missing or stale.** The project file and the per-run files can be rebuilt:
the GUI scans the project directory and each referenced remote's run root for
`mallard.jsonl` files, identifies runs by `run_id`, and identifies local copies
by SHA-256 against the run logs' output records. A per-run file whose run
directory has disappeared is shown as such, never deleted automatically.

**Committed or ignored**: `mallard-project.toml` is meant to be committed;
whether `mallard-runs/` is committed by default is an [open question](#open-questions).

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
- **What applies**: after each edit (debounced), the GUI asks the case's
  `mallard list keys INPUT --format json` which keys this input reads, so the
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
- **Validation** is `mallard check INPUT --format json`, run on demand and after
  edits that change the setup. Its diagnostics are attached to fields by key and
  line, which needs structured `key` and `line` fields (today they are only in
  the message text).
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

### Mesh import, boundary zones, regions and probes

- **Import**: a mesh file is read by `mallard info MESH --format json` (format,
  cells, cell types, bounds, boundary and cell zones with face counts, quality
  with the worst cells) and drawn from `mallard info MESH --view DIR`. Formats
  Mallard converts go through `mallard mesh convert`. A mesh on a remote is
  inspected there; nothing is copied to preview it.
- **Boundary assignment**: the case view has one block per zone. The 3D view
  colors zones (a colorblind-safe categorical palette), picking a zone selects
  its `[[boundaries]]` element, and zones without a condition are highlighted
  until assigned, as `mallard check` reports them.
- **Boundary coordinates** (#346): on a selected patch, show `u`, `v` and set
  `origin` and `toward` by clicking on it.
- **Regions and probes**: `mallard check --view` already draws every region
  (filled regions with their fractions, surfaces and curves with normals),
  probe and periodic pair as named blocks, plus annotated light and dark SVG
  images ([website #61](https://github.com/MatthewBonanni/mallard-website/issues/61)).
  The GUI shows the blocks and links each to its input element, which needs
  block metadata (the key path and element index of each block). For big meshes
  the full check is too slow to rerun per edit: a geometry-only view (mesh,
  zones, regions, probes, without stencils and numerics) is a follow-up.

## Backends

One interface, three implementations (plus PBS and LSF as adapters of the
SSH backend, which differ only in submission and queue queries):

| Operation | Local | SSH + Slurm (PBS, LSF) | Kubernetes |
|---|---|---|---|
| Connect | none | `ssh -o ControlMaster=auto -o ControlPersist=…` to the host alias; MFA once | kube context and namespace from kubeconfig; `kubectl auth can-i` for jobs, pods, exec, port-forward |
| Probe | `mallard version --format json` | same, through the install's path or `module load` line; `sinfo` (partitions, GPU types), `module avail`, `apptainer`, `spack` | same in a short pod of the chosen image; node labels for GPU types |
| Stage inputs | none | transfer engine: rsync over the master connection | transfer engine into the PVC through a helper pod |
| Submit | subprocess (`mpirun -n N` optional) | job script in the run directory, `sbatch` (`qsub`, `bsub`) | a `Job` (or `MPIJob`) with GPU requests and node selectors, the image pinned by digest, the PVC mounted |
| Queue state | process status | `squeue`, `sacct` (`qstat`, `bjobs`) | `kubectl get job,pod -o json`, events |
| Logs | files | `mallard.jsonl` and stdout files on the shared filesystem | `mallard.jsonl` on the PVC; `kubectl logs` for the container's output |
| Control | `STOP`/`CHECKPOINT` files; `mallard ctl` | the same files; `mallard ctl` on rank 0's node through `srun --jobid=<id> --overlap` | the same files through the helper pod; `mallard ctl` through `kubectl exec` in the rank 0 pod |
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
than a batch queue; multi-node runs need the MPI Operator. SSH clusters have a
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
- **Control**: `STOP` and `CHECKPOINT` files work from anywhere the run
  directory is visible; they need `[run] control_file_interval > 0`, which is
  0 by default, so a GUI-launched run needs a way to turn it on without editing
  the user's input (a `--set` option, listed in [Command line](cli.md) but not
  implemented, or a different default). `status` comes from the run log; `set`
  needs `mallard ctl` on rank 0's node until a `SET` control file exists.

## Data: locations, manifest and transfers

### The manifest

For each run, the GUI builds a manifest from:

- the run log's output records (path, kind, step, `t`, bytes, SHA-256 of every
  file written);
- the files the run read (input, mesh, mechanism, data tables), as
  `mallard pack` already lists them with their roles and SHA-256;
- the per-run record (copies elsewhere, transfers).

A new `mallard files RUN_DIR --format json` subcommand returns the first two
for a run directory, adding files present but not recorded (`.partial` files,
files a run still writing has not recorded yet) and, with `--hash`, the SHA-256
of chosen files or of a byte range of one (to resume and verify a transfer). It
runs wherever the run lives, so nothing but `mallard` is needed there.

Every file has one of these states per location:

| State | Meaning |
|---|---|
| present | the file is there and its SHA-256 matches the run log's record |
| absent | not there |
| partial | a transfer was interrupted; resumes from its offset |
| stale | a copy whose source has since changed (the run rewrote it, or a newer record exists) |
| local only | an input edited locally and not yet staged |

The run's home (where it ran) is always shown with its path, as
`hpc:/scratch/…/re100-fine`; a local copy shows as `local:runs/re100-fine/…`.

### Selecting what to move

Selections are by what users think in, not by path: one snapshot; snapshots of
a time or step range; probes, forces and integrals only; restart files (the
newest, or one chosen); a decimated surface (made next to the data by the
viewer or `mallard post extract`, see [follow-ups](#follow-ups)); the case
inputs; or the whole run. Before anything moves the GUI shows the file count,
total size and an estimated time per route from the throughput measured on
earlier transfers of that route (cached in the global config), or a short probe
the first time.

### The transfer engine

One engine for every route; a route is a pair of endpoints and a method:

- **local ↔ remote**: rsync over the SSH master connection
  (`--partial --append-verify`) where the remote has rsync, else a stream with
  byte offsets; through the helper pod for Kubernetes.
- **remote ↔ remote, funneled**: the source's stream is piped into the
  destination's through the local machine, never written to local disk
  (`ssh A 'tail -c +N file' | ssh B 'cat >> file.partial'`, or the `kubectl
  exec` equivalent).
- **remote ↔ remote, direct**: from one remote to the other when they can reach
  each other: rsync from one login node to the other through a jump host or the
  user's forwarded agent (opt-in per remote pair), a shared filesystem mounted on
  both (no copy at all), or a Kubernetes pod pulling from an SSH remote where the
  user has given it a key as a secret.

Every transfer is per file, resumable (a `.partial` file and its offset), and
ends with a SHA-256 check against the manifest (computed by `mallard files
--hash` on the destination), then the rename. The result is recorded in the
run's `mallard-runs/<run>.toml`: the copy (location, path, SHA-256, time) and
the transfer (route, files, bytes, result).

**Detecting a direct route**: a probe from each remote to the other
(`ssh -o BatchMode=yes <other> true`, with and without agent forwarding;
for shared filesystems, a marker file written in one storage root and looked for
in the other), cached with its date in the global remote config and rerun when a
direct transfer fails. The GUI picks the fastest route that works (direct over
funneled) and shows the choice, which the user can override.

**Moving a case or a run to another remote.** The user picks the whole run, or
the inputs plus a chosen restart (to continue there), or selected outputs. For a
continuation, the GUI checks the restart on the destination before moving the
gigabytes: the destination's `mallard info RESTART --format json` must read it
(format version, dimension, precision, species); the rank count does not matter,
since a restart reads bitwise the same on any number of ranks
([Initialization](../guide/initialization.md)). After the move the run shows as
present on both, or as moved if the user then asks to delete the source. The
source is cleaned only on explicit request, listing what will be deleted.

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
(to be confirmed for the VTK version pinned). The viewer is one process at
first; data larger than a node's memory is reduced first (a slice, a surface,
a region) by `mallard post extract` in a batch job.

**Which software.** The viewer is GUI code (it speaks the GUI's protocol), so it
is versioned with the GUI, not with Mallard: an image `mallard-viewer:<gui
version>` built by the GUI repository's CI from the Mallard image of a pinned
version (with VTK's offscreen build and trame added), used as is on Kubernetes
and through Apptainer on SSH clusters; where there is no container runtime, the
remote install puts the same pinned Python environment next to the Mallard
install (see [Installation](#installation)). Whether that counts as "Mallard
only" is an [open question](#open-questions).

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
4. the viewer on the login node itself, opt-in, small and with a short timeout,
   where the site's policy allows it;
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

- **uv/pip**: one command on every platform, an isolated environment, fast, no
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
  2. **Choose**, in this order: a prebuilt relocatable archive for the
     architecture and backend (CPU, CUDA, HIP; a new release artifact),
     downloaded by the login node from the GitHub release with its SHA-256
     checked; the release's container image through Apptainer (`apptainer pull`)
     for single-node runs; a Spack build in user space with the detected
     compilers, MPI and GPU toolkits as externals, run as a batch job (best for
     multi-node runs with the site's MPI; slow).
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

Installation leaves files only; nothing runs afterwards.

### What the release pipeline must produce

Today `release.yml` builds and pushes the CPU (amd64, arm64) and CUDA images and
tests the Spack package. For the GUI it must also produce:

- `mallard-python` wheels with the `mallard` executables (CPU), for macOS arm64
  and manylinux x86-64 and arm64 (extends #792);
- relocatable archives per architecture and backend (CPU, CUDA 12 and 13, ROCm)
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
--app` with pywebview is the zero-extra-dependency desktop mode for pip users,
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
spacing, radii) with the redesigned website, which records them; the mockups
below use provisional values until that file exists.

![Case setup: the setup steps with their status, the mesh with its zones, the problems panel, and the selected zone's boundary condition with an inline error](../images/design/gui-case-setup-light.svg#gh-light-mode-only)
![Case setup: the setup steps with their status, the mesh with its zones, the problems panel, and the selected zone's boundary condition with an inline error](../images/design/gui-case-setup-dark.svg#gh-dark-mode-only)

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
- **Grid and density**: a 4 px base unit on an 8 px grid. Two density modes:
  compact (28 px rows, 13 px text) for experts, comfortable (36 px rows, 14 px
  text) for beginners, switchable at any time.
- **Units everywhere**: every quantity shows its unit, in the monospace, next to
  the value. Fields accept a plain number in the key's units or a quantity with
  a unit (`1 atm`), as Mallard's input does; hovering shows the SI value.
  Nondimensional cases show their reference scales.
- **Motion** only where it carries meaning: a progress bar moving, a panel
  opening from where it was invoked, a transfer's state changing; 120 to 200 ms,
  none with `prefers-reduced-motion`.
- **Icons**: one outline set (Lucide, 1.5 px strokes, 16 and 20 px) plus a few
  drawn in the same style for Mallard's concepts: boundary condition types,
  regions, probes, and the location badges (local machine, SSH remote,
  Kubernetes).
- **Errors and validation**: `mallard check`'s diagnostics appear on the field
  they concern (red border, the message under it, its code and a link to its
  docs), as counts on the setup step that holds them, in a problems panel
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
| Programmatic API (#345) | open | not needed for the MVP: the command line with JSON is the interface |
| `mallard check` (#321) | done: `--format json`, `--view`, `--ranks`, `--memory` | partial inputs fail; one setup error per run; `key` and `line` only in message text; setup warnings missing from the JSON; no block metadata in the case view; no geometry-only view |
| Key catalog (#1256) | open: `docs/input-keys/` is committed and inferred from the examples | no descriptions, no schema version; 204 of 511 general entries have no type; no variants for arrays of tables; no roles for file keys; the general catalog is read from a data directory, not the binary |
| `mallard ctl` (#322) | done: socket `mallard.sock`, `status`, `checkpoint`, `stop`, `set` | must run on rank 0's node; no `SET` control file |
| Control files | done: `STOP`, `CHECKPOINT` | off by default (`control_file_interval = 0`); `--set` is listed in [Command line](cli.md) but not implemented |
| Run log (#326) | done: `mallard.jsonl` version 3 with start, progress, output (bytes, SHA-256), warning and end records | none for monitoring |
| Rank 0's host in the run log | done: the start record's `provenance` run block (`host`) and `placement` (one entry per rank with its host) | none |
| Capabilities query | missing | `mallard version --format json` |
| Remote helpers | partly: `info RUN_DIR`, `pack` list files with SHA-256 | `mallard files` (manifest, hashes of files and ranges) |
| Install verification | missing | `mallard install verify` |
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
6. The data manager: the manifest, states per location, selections, sizes and
   estimates, resumable verified transfers between local and the remote,
   recorded in `mallard-runs/`.
7. The viewer job on a CPU partition through an SSH tunnel: one snapshot's field
   on surfaces and slices, image mode, idle timeout and cleanup.

Mallard prerequisites of the MVP: the catalog with types, units, descriptions,
variants and a schema version (#1256, at least for the keys of the guided
steps); partial-input `list keys`; structured diagnostics; the capabilities
query; `mallard files`; a way to turn control files on from the command line.

**After the MVP:**

| Milestone | Scope |
|---|---|
| M2: more places | Kubernetes backend (helper pod, Jobs, viewer pod, port-forward); PBS and LSF adapters; remote-to-remote transfers, funneled then direct, with route probing; restart-compatibility check; moving a run |
| M3: installs | one-click remote install (archive, Apptainer, Spack), `mallard install verify`, versions side by side, upgrades; the release artifacts; the Tauri desktop app with signed installers |
| M4: full setup | regions and probes editing with previews, boundary coordinates by picking, every section from the catalog, geometry-only fast previews, command palette over keys, `ctl set` and `SET` file |
| M5: results | geometry mode (vtk.js), GPU viewer, fallbacks for blocked forwarding, extraction for large data, saved views, plots of probes with spectra, `diff` and `post average` as jobs |

## Follow-ups

Proposed issues, for approval before filing.

**Mallard**

1. **Catalog for the GUI** (extends #1256): a schema version; types, units,
   descriptions and guide anchors for every key; variants of arrays of tables by
   discriminator; roles of string keys (mesh, mechanism, restart, data file);
   basic or expert; display order; typed defaults; exported from the binary with
   no data directory.
2. **`list keys` and `check` on partial inputs**: report the keys that apply and
   the ones missing instead of failing on a missing mesh file or stop condition.
3. **Structured check diagnostics**: `key`, `line` and element index fields;
   setup warnings in the JSON; independent errors reported together where the
   setup can continue.
4. **Case-view block metadata and a geometry-only view**: each block's key path,
   element index and kind; a `--view` that skips stencils and numerics, measured
   on a large 3D mesh.
5. **Capabilities query**: `mallard version --format json` with the build and
   the version of every interface.
6. **JSON Schemas of the interfaces** in `docs/schemas/`, tested in CI, and the
   examples' JSON uploaded as a CI artifact for the GUI's fixtures.
7. **Control files from the command line**: implement `--set KEY=VALUE` (listed
   in [Command line](cli.md)), or turn control files on by default, and add the
   `SET` control file so `set` works without job-step attachment (#346).
8. **`mallard files RUN_DIR`**: the manifest (recorded outputs, files read,
   unrecorded files) and `--hash` of files and byte ranges.
9. **Restart readability in `info`**: a field saying whether this build can
   continue from a restart, and why not (dimension, precision, format version,
   species).
10. **Decimated extraction**: `mallard post extract` of surfaces and slices with
    decimation, for viewing over slow links and the no-viewer fallback;
    `post render` (#516) for images in batch jobs.
11. **`mallard install verify`**: the environment check plus a quick template
    run, with a JSON verdict.

**Release pipeline**

12. **Wheels with the solver**: `mallard-python` wheels carrying the CPU
    `mallard` executables for macOS arm64 and manylinux x86-64 and arm64
    (extends #792).
13. **Relocatable archives** per architecture and backend (CPU, CUDA 12, CUDA
    13, ROCm) with SHA-256 sums; the MPI strategy is an open question.
14. **`release-manifest.json`** listing every artifact, image digest and
    interface version.

**Website**

15. **Builder from the exported catalog**: once follow-up 1 lands, the builder
    reads the exported catalog and drops its hard-coded `BC_FIELDS` and zone
    lists (supersedes website #62).

**GUI repository** (once created): the MVP above, the viewer image, and the
compatibility matrix.

## Open questions

1. **Viewer software on clusters without containers**: is a pinned Python
   environment (trame, VTK) next to the Mallard install acceptable under
   "nothing installed beyond Mallard", or must the viewer be container-only
   (Apptainer) there?
2. **Remote-to-remote direct routes**: may the GUI use SSH agent forwarding to a
   login node (opt-in per remote pair), and may a Kubernetes pod be given an SSH
   key as a secret to pull from a cluster?
3. **MPI in prebuilt archives**: build against the MPICH ABI (Cray MPICH, Intel
   MPI, MVAPICH), wait for the MPI 5 standard ABI, ship per-MPI variants, or keep
   multi-node runs on Spack builds and archives single-node?
4. **`mallard-runs/` in git**: committed by default (shared record of where
   data is) or ignored (per-user state)?
5. **Windows**: GUI only with remote runs (and WSL for a local solver), or a
   native Windows solver build?
6. **A `mallard gui` subcommand** in the solver that starts the GUI when it is
   installed, for discoverability?
7. **Kubernetes multi-node**: is the MPI Operator in scope for the GUI, or
   single-node Jobs only at first?
8. **Login-node viewer** as a fallback: offer it at all, given site policies?
9. **Shared design tokens**: where the website's token file lives and how the
   GUI pins it (a published package, or a vendored copy per release).
10. **The `mallard-gui` name** on PyPI and as the repository name.
