# The session: Mallard's machine interface

`mallard session` is the one interface through which the GUI
([#346](https://github.com/MatthewBonanni/mallard/issues/346), [design](gui.md))
and scripts talk to Mallard: a process that loads a case once, stays resident,
and answers JSON-RPC 2.0 requests line by line on its standard input and
output. Every request also exists as a one-shot command that runs the same
code, and every request, response, event and run-side file follows a JSON
Schema checked into the repository. Runs never depend on a session, and run
control goes through one command, `mallard ctl`, that finds the run and picks
the transport itself, so no client ever writes a control file.

This page is the design round of [#1340](https://github.com/MatthewBonanni/mallard/issues/1340). Nothing here is built yet; the
[plan](#implementation-plan) lists the pull requests.

- [Why](#why)
- [Decisions](#decisions)
- [Protocol](#protocol)
- [Requests](#requests)
- [One code path: the CLI parity rule](#one-code-path-the-cli-parity-rule)
- [Lifecycle](#lifecycle)
- [Errors and diagnostics](#errors-and-diagnostics)
- [Schemas, versions and compatibility](#schemas-versions-and-compatibility)
- [Performance on massive meshes](#performance-on-massive-meshes)
- [Runs and sessions](#runs-and-sessions)
- [Run control](#run-control)
- [Conformance tests](#conformance-tests)
- [Security](#security)
- [Alternatives considered](#alternatives-considered)
- [Relation to open issues and pull requests](#relation-to-open-issues-and-pull-requests)
- [Implementation plan](#implementation-plan)

## Why

The GUI design first called `mallard` once per question: `mallard list keys
INPUT --format json` after each edit, `mallard check --format json`,
`mallard check --view DIR`, `mallard info MESH --format json`, and `STOP` files
or `mallard ctl` for control. The prototype worked at 1536 cells, but the model
does not scale and is not clean:

- **Every call starts over.** MPI and Kokkos start, the input is parsed, the
  mesh is read, partitioned and its geometry computed, all to answer whether
  `run.cfl = 5` is too large. On a 10⁸-cell mesh the mesh alone takes minutes;
  the answer needs milliseconds.
- **Output is a mixture.** JSON on stdout, warnings as text on stderr, some
  outputs with a schema field and some without, exit statuses that mean
  different things per command. A client must parse text to be robust.
- **Control has three transports and the client picks.** The socket works only
  on rank 0's node, control files only when `control_file_interval > 0`, signals
  only through a launcher that forwards them. The GUI would carry a decision
  tree per backend, and a `STOP` file gives no answer.

## Decisions

1. **`mallard session [INPUT]`** is a resident process speaking JSON-RPC 2.0,
   one message per line, on stdin and stdout. It holds one case and keeps what
   setup computed (the mesh above all) between requests.
2. **One code path.** Each request is a handler that takes JSON parameters and
   returns a JSON result. The one-shot commands (`check --format json`,
   `list keys --format json`, `info --format json`, `ctl`, ...) parse their
   arguments into the same parameters and call the same handler, so every
   session feature exists on the command line by construction, and the text
   output of a command is rendered from the same result.
3. **Structured machine output only.** Every request, response, event and
   run-side file follows a versioned JSON Schema in `schemas/`. Errors carry a
   code, key, line and element; warnings are data, never stderr text.
4. **Runs are independent of sessions.** A session launches a run as a detached
   `mallard -i ...` that outlives it. Cluster submission stays with the GUI's
   backends.
5. **One call routes all control.** `mallard ctl <run_dir|run_id>
   stop|checkpoint|set|status`, also the session request `run.control`, reads
   the endpoint record the run advertises and picks the transport: the socket,
   `srun --overlap`, `kubectl exec`, PBS and LSF equivalents, scheduler signals,
   and control files last. Every request gets an acknowledgment with its request
   ID, recorded in the run log.
6. **No network listeners.** stdio only; remote sessions go through
   `ssh host mallard session` or `kubectl exec -i pod -- mallard session`. No
   HTTP, no TCP port.

## Protocol

**Framing.** One JSON-RPC 2.0 message per line (UTF-8, `\n`-terminated, no
newline inside a message; JSON escapes newlines in strings). The client writes
requests and notifications to the session's stdin; the session writes
responses and notifications to stdout, and nothing else. stdout is flushed after
each message. stderr carries a human-readable log of the session process (setup
phases, timings) that clients may show but never parse; everything a client
needs is in a message.

**Messages** follow JSON-RPC 2.0 exactly:

```json
{"jsonrpc": "2.0", "id": 7, "method": "case.check", "params": {"partial": true}}
{"jsonrpc": "2.0", "id": 7, "result": {"schema": "mallard-check", "version": 1, "ok": false, "diagnostics": [...]}}
{"jsonrpc": "2.0", "id": 8, "error": {"code": -32002, "message": "...", "data": {...}}}
{"jsonrpc": "2.0", "method": "$/progress", "params": {"id": 7, "phase": "mesh", "done": 0.4}}
```

- `id` is an integer or a string chosen by the client; the session never reuses
  or reorders it: responses come in the order of the requests, except
  `$/cancel`, which is handled as soon as it is read.
- Batches (JSON arrays) are not supported: they answer -32600. A client that
  wants several answers sends several requests; they are processed in order.
- Notifications from the session: `$/progress` (phase and fraction of a long
  request), `$/log` (a structured warning or note raised outside any request,
  rare), `run.event` (records of a run the client follows, see
  [Runs](#runs-and-sessions)).
- Notifications from the client: `$/cancel {"id": 7}` stops request 7 at its
  next phase boundary; it then answers error -32800 (cancelled).

**Parameters** are objects, never positional arrays. Paths are relative to the
session's working directory unless the case is open, then relative to the
input's directory, as `mallard -i` resolves them.

## Requests

Every request has a result schema and a one-shot command. The one-shot column
is normative: a request without one is a bug (see
[the parity rule](#one-code-path-the-cli-parity-rule)). New one-shots are marked
*new*.

| Method | Parameters | Result (schema) | One-shot |
|---|---|---|---|
| `initialize` | `client` (name, version), `require` (schema names and the major versions the client reads) | `mallard-capabilities`: version, build, dimensions, every interface's schema and version, the session's methods | `mallard version --format json` |
| `shutdown` | none | `null`; the session exits after answering | none needed (a process exit) |
| `case.open` | `input` (path) or `text` with `path` (unsaved text, located for relative paths) | `mallard-case`: input path, dimension, sections, the mesh's identity (file, bytes, SHA-256 when hashed), what was set up and its timings | `mallard check INPUT --format json --setup-only` (*new*) |
| `case.update` | `text` (the whole input) | as `case.open`, plus `changed` (key paths that differ) and `invalidated` (setup stages to redo) | none needed (a one-shot always starts from the file) |
| `case.close` | none | `null`; frees the case | none needed |
| `case.check` | `partial`, `ranks`, `memory`, `throughput`, `steps`, `mesh_fields` | `mallard-check` | `mallard check INPUT --format json [--partial] [--ranks N] ...` |
| `case.keys` | `section` | `mallard-key-catalog` (the keys this case reads, with `complete`, `missing`, `diagnostics` of [#1311](https://github.com/MatthewBonanni/mallard/issues/1311)) | `mallard list keys [SECTION] INPUT --format json` |
| `keys.catalog` | `section` | `mallard-key-catalog` (every key there is) | `mallard list keys [SECTION] --format json` |
| `case.view` | `dir`, `geometry_only` | `mallard-case-view`: the files written in `dir` (`case.vtm` and its pieces, SVG images) and every block with its kind, key path, element and zone ([#1313](https://github.com/MatthewBonanni/mallard/issues/1313)) | `mallard check INPUT --view DIR [--geometry-only]` |
| `case.initial_state` | `dir` or `prefix`, `fields` | `mallard-snapshot` of the initial solution written, with field ranges | `mallard check INPUT --initial-state PREFIX` (*new*) |
| `mesh.info` | `file` (default: the case's mesh) | `mallard-mesh-info`: format, cells by type, bounds, boundary and cell zones with faces, areas and bounds | `mallard info MESH --format json` |
| `mesh.quality` | `n_worst`, `fields` (prefix of the mesh-field snapshot) | `mallard-mesh-quality`: per metric max, mean, limit, count above it, worst cells with ids and centers | `mallard info MESH --format json` (its `quality` member), `check --mesh-fields` |
| `mesh.cut` | `plane` (origin, normal) or `box`, `dir`, `target_triangles` | `mallard-mesh-cut`: the cut surface's file, triangle count, the cells it crosses | `mallard mesh cut INPUT --plane X,Y,Z,NX,NY,NZ --output FILE` (*new*) |
| `mesh.pick` | `point`, `kind` (`cell`, `face`, `zone`) | `mallard-mesh-pick`: the cell or face id, its zone, center, volume or area, quality metrics, owning rank | `mallard mesh pick INPUT X Y [Z] --format json` (*new*) |
| `run.launch` | `input` (default: the case's file, saved), `run_dir`, `ranks`, `launcher`, `set` (list of `KEY=VALUE`), `restart` | `mallard-run-launch`: run ID, run directory, PID, endpoint record path, the command line used | `mallard -i INPUT --detach [--run-dir DIR] ...` (*new*) |
| `run.control` | `run` (directory or run ID), `action` (`stop`, `checkpoint`, `set`, `status`), `settings`, `wait` (`accepted` or `done`), `timeout` | `mallard-run-control` reply (see [Run control](#run-control)) | `mallard ctl RUN action ... --format json` |
| `run.status` | `run` | `mallard-run-directory` (as `info RUN_DIR`), with the endpoint record and the last progress record | `mallard info RUN_DIR --format json` |
| `run.follow` | `run`, `from` (byte offset in `mallard.jsonl`) | `{offset}`; then `run.event` notifications with each new run-log record and its offset until `run.unfollow` or the run's `end` record | none needed (`tail -c +N mallard.jsonl`) |
| `run.unfollow` | `run` | `null` | none needed |

Rules for the table:

- **The input file is the client's.** The session never writes `input.toml`:
  the GUI edits it with a style-preserving editor and sends the text with
  `case.update`. `run.launch` without `input` requires the open case's text to
  equal the file on disk (else error -32010, `unsaved`), so a run always runs a
  file a human can read.
- **Bulk data goes to files, not to stdout.** Geometry, snapshots and cuts are
  written to a directory the client names (on the machine the session runs on,
  next to the data) and named in the result; the GUI's viewer reads them there.
  stdout carries only JSON.
- **Requests that need a case** answer -32001 (`no case`) before `case.open`.

## One code path: the CLI parity rule

Every request is a handler:

```cpp
namespace session {
struct Context;  // the open case and its setup cache; empty for a one-shot until it opens one
using Handler = json::Value (*)(Context &, const json::Value & params, Progress &);
struct Method {
    std::string_view name;    // "case.check"
    std::string_view command; // "check": the row of cli::commands() that is its one-shot
    interfaces::Schema params, result;
    Handler handler;
};
std::span<const Method> methods();
}
```

- **One-shot commands call handlers.** `mallard check INPUT --format json`
  builds `params` from its arguments, makes a fresh `Context`, calls
  `case.open` then `case.check`, and prints the result. With `--format text`
  it renders the same result as text. The text renderers read only the result
  object, so nothing reaches the text that the JSON lacks.
- **The method table links to the command table.** Each method names the
  command row (`cli::commands()`, [Command line](cli.md)) that is its one-shot;
  a test fails if a method has none (outside the four marked "none needed",
  which are session plumbing), if the command does not accept `--format json`,
  or if the JSON of the one-shot and the session's result differ for the same
  case (after removing timings).
- **Session-only state is a cache, never a behavior.** The session may answer
  faster than the one-shot, never differently. The conformance test above is
  what holds this.

## Lifecycle

```text
client                                     mallard session
  │ spawn: [ssh host | kubectl exec -i pod --] mallard [--dim 3] session [INPUT] [--kokkos-num-threads=N]
  │                                          MPI and Kokkos start once
  │ initialize ─────────────────────────────▶ capabilities, methods
  │ case.open {input} ──────────────────────▶ parse, read mesh, set up (progress notifications)
  │ case.check / case.keys / case.view ─────▶ answered from the setup cache
  │ case.update {text} ─────────────────────▶ diff, invalidate stages, redo only those
  │ run.launch ─────────────────────────────▶ detached mallard -i ... (outlives the session)
  │ run.control {run, action: stop} ────────▶ routed to the run, acknowledged
  │ shutdown, or EOF on stdin ──────────────▶ exits 0; runs keep going
```

- **Start.** `mallard session [INPUT]` with an input opens it at once
  (`case.open` without a request; its result is the response to
  `initialize`'s `case` member). The front end chooses the solver's dimension
  from the input as for `check`; without one, `--dim` or the default solver.
  A `case.open` of an input of the other dimension answers -32008
  (`wrong dimension`, with `dim`), and the client starts a session with
  `--dim`.
- **One case per session.** A GUI with two cases runs two sessions. This keeps
  memory accounting simple (a session's resident memory is its case's) and
  lets the OS reclaim everything on exit.
- **Order and concurrency.** A reader thread parses stdin; the main thread
  answers requests in order. `$/cancel` is handled by the reader: it marks the
  request, which stops at its next phase boundary (after the mesh, after the
  numerics, between regions) and leaves the cache as it was before the
  request.
- **End.** `shutdown`, EOF on stdin (the client exited, or the SSH connection
  dropped), SIGTERM or SIGHUP end the session with status 0 after the current
  phase; `--idle-timeout SECONDS` ends it after that long without a request
  (the GUI passes one for remote sessions, so a lost client never leaves a
  process holding a mesh on a login node). Runs it launched keep going.
- **Exit status.** 0 on a normal end, 1 if it could not start (bad option,
  MPI or Kokkos failure, reported as text on stderr and, if stdout is open, a
  `$/log` notification first), 2 on a usage error.
- **Parallel sessions** (a mesh too large for one process): `srun -n N mallard
  session` or `mpirun`. Rank 0 owns stdio and broadcasts each request; every
  rank runs the handler on its share; results are reduced to rank 0, as
  `mallard check` on N ranks does today. The protocol does not change. Not in
  the first implementation; the handler interface (collective, results on rank
  0) is designed for it.

## Errors and diagnostics

Two different things, never mixed:

- **A request that failed** answers a JSON-RPC `error`. Its `code` is a
  JSON-RPC standard code or one of Mallard's below; its `message` is one line
  for humans; its `data` holds `mallard_code` (a code of
  `diagnostic_code::ALL`, as `E-INPUT`) and, where they explain it,
  `diagnostics`.
- **A check that finds problems succeeded.** `case.check` returns a result with
  `ok = false` and its diagnostics; an input error during `case.open` makes the
  case open but not set up (`setup_done = false`), with the diagnostics in the
  result, so a client can still ask `case.keys` of a partial input.

| Code | Meaning |
|---|---|
| -32700, -32600, -32601, -32602, -32603 | JSON-RPC: parse error, invalid request (batches included), no such method, invalid parameters (with `data.errors`: the schema paths that failed), internal error (a bug: with the exception's text) |
| -32800 | Cancelled by `$/cancel` |
| -32001 | No case open |
| -32002 | The input cannot be parsed as TOML (`data.diagnostics` with line and column) |
| -32003 | A file the request needs cannot be read (`data.path`) |
| -32004 | Timeout (run control: `data.phase` is `accepted` or `done`, with the last step the run log shows) |
| -32005 | No run with this directory or ID |
| -32006 | Run unreachable: no transport worked (`data.attempts`: each route tried and why it failed; `data.route_hint` when the caller can run it, see [Run control](#run-control)) |
| -32007 | A schema the client requires is not served in that major version (`data.served`) |
| -32008 | Wrong dimension (`data.dim`) |
| -32009 | The run refused the request (`data.reply`: the run's own reply, e.g. a refused setting) |
| -32010 | `run.launch` of unsaved text |

**Diagnostics** are one object everywhere (check reports, key catalogs, run
log warnings, `$/log`, error `data`), extending
[#1312](https://github.com/MatthewBonanni/mallard/issues/1312)'s:

| Field | Meaning |
|---|---|
| `severity` | `error`, `warning`, `note` |
| `code` | `E-...`, `W-...`, one of `diagnostic_code::ALL`, with a row in the docs (`Diagnostics.EveryCodeIsExplainedInTheDocs`) |
| `message` | one paragraph for humans: what is wrong, the numbers, the fix |
| `key` | full key path with the element of an array of tables (`boundaries[1].T`, `boundaries[name = "inlet"].T`), or `null` |
| `line`, `column` | in the input, 1-based, or `null` |
| `element` | the element of an array of tables: `{"array": "boundaries", "index": 1, "name": "inlet"}`, or `null` |
| `location` | where on the mesh: `{"cell": 1204, "x": [0.3, 0.1, 0.0]}`, a zone or a face, or `null` |
| `missing` | a missing piece of a partial input (#1311) |
| `docs` | the page and anchor that explain the code |

**Warnings are data.** In a session and in every `--format json` command,
`logging::warning` during a handler is captured into the result's
`diagnostics` (`logging::WarningCapture` of #1332) and not printed. In text
mode the same diagnostics are printed by the renderer. A run's warnings go to
the run log's `warning` records as today, with the same object.

## Schemas, versions and compatibility

- **Where.** `schemas/<name>.v<major>.schema.json` at the repository's root,
  JSON Schema 2020-12, one file per interface and major version, with `$id`
  `https://matthewbonanni.github.io/mallard/schemas/<name>.v<major>.schema.json`
  (the website publishes the directory). They are installed with Mallard
  (`share/mallard/schemas/`), and `mallard version --format json` lists each
  interface's schema name, version and file. This is
  [#1315](https://github.com/MatthewBonanni/mallard/issues/1315)'s directory,
  moved from `docs/schemas/` because the schemas are a product of the build,
  installed and tested, not documentation.
- **What has one.** The session protocol itself (`mallard-session`: the
  envelope, the error object, the notifications); each request's parameters
  and result (the table above); the diagnostic object (shared through `$ref`);
  and the run side: the run log's records (`mallard-run-log`), the run block
  (`mallard-run`), the snapshot block, the endpoint record
  (`mallard-run-endpoint`), the control protocol (`mallard-run-control`), the
  control files and the pack manifest.
- **Every document names its schema**: `schema` and `version` members at its
  top (the existing convention of the run log and #1331). Run-log records other
  than `start` inherit the `start` record's.
- **Versions.** An integer major version per interface, as today. Adding a
  member, a record type, an enum value a reader may ignore, or a method keeps
  the version; removing or renaming a member, changing its meaning or units, or
  making an optional member required increments it. Readers ignore members
  they do not know. Schemas say `additionalProperties: true` at every level so
  that additive changes validate against the older schema.
- **Negotiation.** `initialize` takes `require`, the majors the client reads
  per interface; the session answers -32007 for one it cannot serve. A major
  increment keeps the previous major served (selected by `require`) for at
  least one minor release, so a GUI release and a Mallard release can be
  upgraded in either order.
- **The session's methods are versioned with `mallard-session`.** A new method
  is additive; a method removed or changed in its parameters increments it.
  `initialize` lists the methods, so a client tests for a method, not for a
  Mallard version.

## Performance on massive meshes

The session exists because nothing about a case should be computed twice. The
goals, on the machine the session runs on:

| Request | Goal at 10⁷ cells (3D) | Why it holds |
|---|---|---|
| `initialize` | under 0.5 s | Kokkos and MPI start once |
| `case.update` + `case.check` after an edit outside `[mesh]` | under 1 s | no mesh read, no partition, no geometry |
| `case.keys` after an edit | under 0.2 s | the keys come from the setup's read record |
| `case.check` after an edit to `[mesh]` | as the one-shot | the mesh changed |
| Boundary assignment validated (`[[boundaries]]` names against zones) | under 0.1 s | zone metadata in the cache |
| `mesh.pick` | under 0.1 s | a spatial index of the cells, built once |
| `mesh.cut` | under 2 s | the same index; only the crossed cells are touched |
| Resident memory | the one-shot `check`'s peak | the cache holds what setup built once |

**Stages and what they read.** Setup runs in stages (run parameters, mesh,
physics and boundaries, numerics, fields and regions, initial state). `Input`
already records every key a stage looks up; the session keeps, per stage, the
key paths it read and a fingerprint of their values, plus the identity of the
files it read (path, size, modification time, and the SHA-256 the run block
records). On `case.update`:

1. the new text is parsed and diffed against the cached tree, key by key;
2. a stage is invalid if a key it read changed, a file it read changed, or a
   stage it depends on is invalid (mesh → numerics → regions → initial state;
   run parameters and physics depend on nothing above them);
3. valid stages keep their objects; invalid ones are rebuilt, in order.

So an edit of `run.cfl` reruns only the run parameters and the time-step check,
an edit of a boundary's `T` reruns boundaries, the initial state and the
checks that read them, and an edit of `mesh.filename` reruns everything.

**The first implementation caches the mesh**, the stage that dominates at
scale: the mesh read and the built local mesh (connectivity, geometry, halo),
keyed by the keys `init_mesh` reads (`[mesh]`, `[parallel]`, `[periodic]`,
`[[mesh.curved]]`) and the mesh file's identity. Later stages rerun on each
`case.check`; they are cached one by one where a measurement shows they matter
(numerics' TENO tables are the next candidate, already cached on disk by
their own key). Whether a stage is valid is decided by its read record, never
by a hand-written list of keys, so a new key cannot be forgotten.

**Large cases** keep the [scale rules of the GUI](gui.md#scale): nothing
O(cells) crosses stdout; geometry goes to files near the data; cuts and picks
return bounded results. Above what one process holds, the session runs on N
ranks under the cluster's launcher (see [Lifecycle](#lifecycle)), as a job, not
on a login node.

**Measured** (to fill in with the implementation): `check` after an edit
outside `[mesh]`, session against one-shot, on the 3D meshes of
`benchmarks/` up to the largest a CPU pod holds.

## Runs and sessions

**Launch, locally.** `run.launch` (and `mallard -i INPUT --detach`) starts
`mallard -i INPUT [--restart ...] [--set ...]` as a new process group in its
own session (`setsid`), in the run directory, with stdin from `/dev/null` and
stdout and stderr to `mallard.out` there, then returns once the run has written
its endpoint record (within about a second; before setup). The run outlives the
session and the GUI. With `ranks > 1` it starts the run under `launcher`
(`mpirun -n N` by default, or the command the client gives, whose first word
must be on the session's allow list: `mpirun`, `mpiexec`, `srun`). The run
directory, if new, is created and the input copied in, as the GUI's local
backend expects.

**Cluster submission stays with the GUI's backends.** Job scripts, `sbatch`,
`qsub`, `bsub`, Kubernetes `Job`s and queue states belong to the GUI's backend
layer ([Backends](gui.md#backends)): they depend on the site, the account and
the user's choices, not on Mallard. The session on a login node checks the case
(`case.check` with `ranks` and `memory`), and once the job runs, `run.control`,
`run.status` and `run.follow` work on it as on a local run, because they go
through the run's endpoint record, not through the session that launched it.

**Runs never depend on a session.** A run reads nothing from a session and
answers control requests on its own channel; a session's crash, exit or
restart changes nothing for the run. A session can follow and control any run
the user can reach, whoever launched it.

## Run control

### The endpoint record

At start, before setup (so it also covers a run stuck in setup), rank 0 writes
`mallard.run.json` in the run directory, atomically (`.partial` and a rename),
schema `mallard-run-endpoint`, version 1:

```json
{"schema": "mallard-run-endpoint", "version": 1,
 "run_id": "3a256c99-a024-4845-8b46-5848d430e28b",
 "mallard": {"version": "0.12.0", "executable": "/opt/mallard/0.12.0/bin/mallard-3d"},
 "host": "node0412", "pid": 81234, "ranks": 512,
 "directory": "/scratch/proj/cyl/runs/re100",
 "socket": "/scratch/proj/cyl/runs/re100/mallard.sock",
 "scheduler": {"kind": "slurm", "job_id": "48213", "step_id": "0", "node": "node0412"},
 "started": "2026-11-20T14:03:11Z", "state": "setup"}
```

- `scheduler.kind` is `slurm` (`SLURM_JOB_ID`, `SLURM_STEP_ID`), `pbs`
  (`PBS_JOBID`), `lsf` (`LSB_JOBID`), `kubernetes` (the pod's name and
  namespace, from `HOSTNAME` and the service account's namespace file or the
  `MALLARD_POD`/`MALLARD_NAMESPACE` downward-API variables the guide's `Job`
  sets), or `none`.
- `state` is rewritten at `running` (the time loop and its socket started) and
  `ended` (with `outcome`, as the run log's end record). A record whose `state`
  is not `ended` and whose process is gone (same host, no such PID, or the
  scheduler says the job ended) is stale; `ctl` says so.
- The run ID is made at process start (today at setup), so the launcher knows
  it at once and every record of the run carries the same one.
- **Finding a run by ID.** The run also writes a copy of its endpoint record to
  `$XDG_STATE_HOME/mallard/runs/<run_id>.json` (`~/.local/state/...`; shared
  across the nodes of a cluster with a shared home), so `mallard ctl <run_id>`
  finds the directory. Entries of ended runs are removed after 30 days by the
  next run that writes one. `[output] run_registry = false` turns it off.

### Routing

`mallard ctl <run_dir|run_id> <action> [settings] [--wait accepted|done]
[--timeout S] [--format json]`, and the session's `run.control`, read the
endpoint record and try the routes in this order, taking the first that
reaches the run:

| Order | Route | When | How |
|---|---|---|---|
| 1 | Socket | this host is the record's `host` and the socket answers | the control protocol below, directly |
| 2 | Slurm step | `scheduler.kind = slurm` and `srun` is on PATH | `srun --jobid=<job> --overlap --nodes=1 --ntasks=1 -w <node> <executable> ctl --direct <dir> ...` |
| 3 | Kubernetes | `scheduler.kind = kubernetes` and `kubectl` reaches the pod | `kubectl exec -n <ns> <pod> -- <executable> ctl --direct <dir> ...` |
| 4 | PBS, LSF | `pbs` or `lsf` | `ssh <node> <executable> ctl --direct <dir> ...` where the site allows SSH to a job's nodes (most do: `pam_slurm_adopt` and its PBS and LSF equivalents); `blaunch -z <node>` inside the LSF job |
| 5 | Scheduler signal | `stop` or `checkpoint` only (no payload) | `scancel --signal=USR2` (stop) or `USR1` (checkpoint) `--full=no <job>`; `qsig -s USR2 <job>`; `bkill -s USR2 <job>`; `kill -USR2 <pid>` on the same host |
| 6 | Control file | the run directory is visible and `control_file_interval > 0` | `STOP`, `CHECKPOINT`, `SET` with the request's ID inside |

- **Signals.** Mallard already handles `SIGUSR1` (checkpoint) and `SIGTERM`,
  `SIGINT`, `SIGUSR2` (stop with a restart, exit status 3), and the wall-time
  limit `t_wall_stop` (stop with a restart). The signal route sends `SIGUSR2`
  rather than `SIGTERM` because Open MPI's `mpirun` ends its ranks on
  `SIGTERM` without letting them finish the step ([Running](../guide/running.md)).
  A signal carries no request ID: the acknowledgment is the next `control`
  record of that action in the run log after the request's time, marked
  `"source": "signal"`.
- **Control files** stay, documented for humans, scripts and setups where
  nothing else reaches the run; no Mallard client writes them unless every
  other route failed and the caller passed `--allow-files`.
- **`--direct`** skips routing and talks to the socket in `<dir>`; it is what
  the remote ends of routes 2 to 4 run, so a route never recurses.
- **A route the caller must run.** When `kubectl` or `srun` is not where the
  session runs (a session in a helper pod, a GUI on a laptop), `ctl` fails with
  -32006 and `data.route_hint`: the command to run elsewhere
  (`{"kind": "kubectl_exec", "namespace": ..., "pod": ..., "argv": [...]}`), so
  the GUI's backend runs it with the user's kubeconfig. The decision of which
  route stays in Mallard.
- **`status`** is answered by the socket when reachable; otherwise from the run
  log's last progress record and the endpoint record, with `"live": false`.

### The control protocol, version 2

The socket keeps accepting version 1's text lines (`stop`, `checkpoint`, `set
k=v`, `status`) for older clients. A line starting with `{` is version 2:

```json
{"schema": "mallard-run-control", "version": 2, "id": "c1a7…", "action": "stop"}
{"schema": "mallard-run-control", "version": 2, "id": "c1a7…", "action": "set", "settings": {"t_stop": "3.0", "write_data.flow.time_interval": "0.05"}}
```

The run answers on the same connection with up to two JSON lines and closes it:

```json
{"id": "c1a7…", "status": "accepted", "step": 20480, "t": 1.2741}
{"id": "c1a7…", "status": "done", "action": "stop", "step": 20481, "t": 1.2742, "restart": {"path": "solut/restart_20481.restart", "sha256": "c07a…"}, "outcome": "stopped"}
```

- `accepted`: rank 0 queued the request (immediately). `done`: every rank acted
  on it after the same step; for `stop` once the restart is written and the run
  is ending, for `checkpoint` once the restart is written, for `set` with each
  setting's result. A refused request answers `{"status": "refused", "error":
  {diagnostic}}`.
- `--wait accepted` returns after the first line; `--wait done` (the default)
  after the second. `--timeout` (default 30 s for `accepted`, none for `done`,
  since a step of a large run can take minutes) answers -32004 with the phase
  reached and the last step the run log shows.
- **Every request is recorded.** The run log gains `control` records, with the
  request's `id`, `action`, `source` (`socket`, `srun`, `kubectl`, `ssh`,
  `signal`, `file`, `--set`), `status`, `step`, `t` and the result (the restart
  file, the settings applied, the error). So a client that lost its connection,
  or used the signal or file route, finds the outcome by its request ID by
  following the run log. `mallard ctl` waiting on routes 5 and 6 reads the run
  log for its ID.
- The request ID is a UUID made by the client (`ctl` makes one if not given
  `--id`).

## Conformance tests

In Mallard's suite (`test/schema_test.cpp`, and the CLI tests):

- **A validator for the schemas' subset** (`type`, `properties`, `required`,
  `items`, `enum`, `const`, `$ref`, `oneOf`, `additionalProperties`, `minimum`,
  `pattern`) in the test code, and a test that every schema in `schemas/` uses
  only that subset and is valid JSON Schema 2020-12 (checked once by Python's
  `jsonschema` in `test/python/`).
- **Every method, both ways.** For each method of `session::methods()`, on the
  examples that apply: the session's result and the one-shot's JSON validate
  against the result schema and are equal (timings removed). A test drives a
  real `mallard session` process through a pipe for the protocol itself
  (framing, errors, cancel, EOF, order).
- **Run side.** The run log of a short run, its endpoint record and every
  `ctl` reply validate against their schemas; `ctl` with each route the test
  machine has (socket, signal, file) acknowledges with the request's ID.
- **Schema versions.** A test fails if a schema file changes without either an
  additive diff or a version increment (a removed `required` member or
  property, a changed `type`), comparing against the copy of the last release
  kept in `schemas/released/`.
- **Fixtures for the GUI.** CI uploads every interface's JSON from the examples
  as an artifact (#1315), which the GUI's CI tests against.

## Security

- **No listener.** stdio only. The session never binds a port or a socket; the
  run's control socket is the run's, mode 0600 as today.
- **What a session runs.** Only `mallard` itself (the session's own executable
  or the endpoint record's) and, for routes, `srun`, `kubectl`, `ssh`,
  `scancel`, `qsig`, `bkill` with arguments it builds. `run.launch`'s
  `launcher` is checked against the allow list. No request runs a shell.
- **Paths** are the client's; the session runs as the user, so it can read
  what the user can, nothing more.

## Alternatives considered

- **One process per call** (the GUI design's first model): simple, but every
  call pays MPI, Kokkos and the mesh, and the GUI multiplies parsers of
  heterogeneous outputs. Kept as the one-shots, which now share the handlers.
- **An HTTP or websocket server in Mallard**: a listener on a shared machine
  needs authentication, TLS or tokens; remote use needs port forwarding. stdio
  over SSH or `kubectl exec` gets authentication, encryption and reachability
  from tools the user already has.
- **Language Server Protocol framing** (`Content-Length` headers): robust for
  messages with newlines, but JSON never needs one, and line framing works with
  `jq`, `ssh` and a few lines of Python.
- **Python bindings in the GUI's process** (#345): no process boundary for
  local cases, but remote cases need one anyway, a crash takes the GUI down, and
  the bindings would duplicate this interface.
- **The client chooses the control transport**: what the GUI design did; it
  puts site knowledge (which node, which scheduler command) in every client.

## Relation to open issues and pull requests

| Item | Relation |
|---|---|
| [#1311](https://github.com/MatthewBonanni/mallard/issues/1311), [#1312](https://github.com/MatthewBonanni/mallard/issues/1312) (#1332) | Partial inputs and structured diagnostics: `case.keys` and `case.check` return them; the session builds on that pull request |
| [#1313](https://github.com/MatthewBonanni/mallard/issues/1313) | Block metadata and the geometry-only view: `case.view` |
| [#1314](https://github.com/MatthewBonanni/mallard/issues/1314) (#1331) | `version --format json` is `initialize`'s result; its interface list gains the session's methods and schema files |
| [#1315](https://github.com/MatthewBonanni/mallard/issues/1315) | JSON Schemas: done here, in `schemas/` |
| [#1316](https://github.com/MatthewBonanni/mallard/issues/1316) (#1334) | `--set` is how `run.launch` passes settings; the `SET` file is control route 6 |
| [#1317](https://github.com/MatthewBonanni/mallard/issues/1317) | `mallard files`: a later method, `run.files`, with the same parity |
| [#1319](https://github.com/MatthewBonanni/mallard/issues/1319) | The resident process for cuts and picks is the session (`mesh.cut`, `mesh.pick`) |
| [#1326](https://github.com/MatthewBonanni/mallard/issues/1326) (#1337) | Write-once outputs: the restart a `done` reply names is an object by SHA-256 |

## Implementation plan

Each step is a pull request with its tests, docs (`docs/guide/session.md` for
the protocol, the command pages, `docs/input.md` for keys) and changelog.

1. **This design** (docs only), with [The GUI](gui.md) pointing here.
2. **Schemas and conformance** for the existing interfaces: `schemas/`, the
   test validator, schemas of the check report, key catalog, capabilities, run
   log, run block, run directory; installed; listed by `version --format json`.
   After #1331.
3. **The session and the handlers**: `session::methods()`, `mallard session`
   with `initialize`, `shutdown`, `case.open`, `case.update`, `case.close`,
   `case.check`, `case.keys`, `keys.catalog`, `$/cancel`, `$/progress`; the
   one-shots `check --format json` and `list keys --format json` through the
   handlers; the mesh cache; `docs/guide/session.md`; latency measurements.
   After #1332.
4. **Run control**: the endpoint record and the run registry, the control
   protocol version 2 with acknowledgments, `control` records in the run log,
   routing in `mallard ctl` (socket, Slurm, Kubernetes, SSH for PBS and LSF,
   signals, files), `run.control`, `run.status`, `run.follow`. After #1334.
5. **Launch**: `mallard -i INPUT --detach`, `run.launch`.
6. **Case view and mesh**: `case.view` with block metadata and
   `--geometry-only` (#1313), `mesh.info`, `mesh.quality`,
   `case.initial_state`.
7. **Cuts and picks**: the spatial index, `mesh.cut`, `mesh.pick` and their
   one-shots.
8. **Parallel sessions** on N ranks, measured on the largest mesh of the
   benchmarks.
