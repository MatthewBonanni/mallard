# Sessions: Mallard for programs

`mallard session` holds a case and answers requests about it, so that a program (the GUI, an editor plugin, a script, an agent) can ask again after each edit without starting Mallard each time. Requests and answers are [JSON-RPC 2.0](https://www.jsonrpc.org/specification), one JSON object per line, on standard input and standard output. Nothing else is written to standard output: the setup's text goes to standard error.

```bash
$ mallard session cases/cylinder/input.toml
{"jsonrpc": "2.0", "id": 1, "method": "case.check"}
{"jsonrpc":"2.0","method":"$/progress","params":{"id":1,"phase":"mesh"}}
...
{"jsonrpc":"2.0","id":1,"result":{"version":1,"status":"ok",...}}
{"jsonrpc": "2.0", "id": 2, "method": "shutdown"}
{"jsonrpc":"2.0","id":2,"result":null}
```

A session listens on no network port. On another machine, run it through the connection you already have: `ssh hpc mallard session input.toml` or `kubectl exec -i pod -- mallard session input.toml`. It ends at `shutdown` or when its standard input closes.

## Requests

Each request with a one-shot command runs the same code as that command, so its result is exactly that command's JSON output.

| Method | Params | Result | Same as |
|---|---|---|---|
| `initialize` | | `mallard-capabilities`: the version, build and `dimension`, the schema of every interface, every method with its one-shot command and result schema, `devices` (the backend, each GPU visible with its `name` and `memory` in bytes, the host's `cpu` and `threads`), `control` (the socket's `commands`; under `settings` every key `mallard ctl set` takes, with its `type` (`count` or `time`), `units`, and `selector`, what names its element: `prefix_or_index`, `name` or `zone`; the control `files`; the `routes` `mallard ctl` tries, in order, each `available` if a program it runs is on `PATH` here), and the open case | `mallard version --format json` |
| `case.open` | `input` (path), `text` (optional: the input's text, if it differs from the file) | the case (`mallard-case`) | |
| `case.update` | `text`: the whole input | the case, with `changed`: the key paths whose values differ | |
| `case.close` | | `null` | |
| `case.check` | the options of `mallard check` by name: `partial`, `ranks`, `memory`, `throughput`, `steps`, `mesh_fields`, `stencils`, `cells`, `region`, `weights`, `catalog`, `view`, `restart` | `mallard-check` | `mallard check INPUT --format json` |
| `case.view` | `dir`: where to write it | `mallard-case-view`: the files written, every block, and the frame of each boundary's local coordinates ([The case view](input_files.md#the-case-view)) | `mallard check INPUT --view DIR --format json` (its `view`) |
| `case.boundary_values` | `dir`: where to write them | `mallard-boundary-values`: per `[[boundaries]]` entry, the file of its values at each face and each column's range; estimated inflow of stagnation and mass-flow inlets ([Boundary values](input_files.md#boundary-values)) | `mallard check INPUT --boundary-values DIR --format json` (its `boundary_values`) |
| `case.keys` | `section` (optional) | `mallard-key-catalog` | `mallard list keys [SECTION] INPUT --format json` |
| `case.files` | `sha256` (optional, default `false`) | `mallard-case-files`: the input and every file its setup read, with role, path as written and resolved, `exists`, `bytes`, `sha256` ([The other commands](input_files.md#the-other-commands)) | `mallard files INPUT --format json [--sha256]` |
| `keys.catalog` | `section` (optional) | `mallard-key-catalog`, with each table's `form` ([The key catalog](input_files.md#the-key-catalog)) | `mallard list keys [SECTION] --format json` |
| `file.info` | `file`: an output or restart file | `mallard-file-info`: what it holds; for a restart, `continuable` by this build and its `incompatibilities` ([Restarts](initialization.md#restarts)) | `mallard info FILE --format json` |
| `units.catalog` | none | `mallard-unit-catalog`: every unit with its dimension (as the key catalog's `dimension` names it), exponents and exact factor, the SI prefixes and the named dimensions ([Units](input_files.md#units)) | `mallard list units --format json` |
| `run.launch` | `input` (default: the open case's file, which must be saved), `run_dir` (default: the session's working directory), `restart`, `launcher` (e.g. `["mpirun", "-n", "4"]`: `mpirun`, `mpiexec` or `srun`), `options` (more arguments of `mallard -i`), `timeout` | `mallard-run-launch`: run ID, directory, PID, endpoint record | `mallard -i INPUT --detach` |
| `run.control` | `run` (directory or run ID), `action` (`status`, `checkpoint`, `stop`, `set`), `settings` (object), `id`, `wait` (`accepted` or `done`), `timeout`, `allow_files` | `mallard-run-control` | `mallard ctl RUN ACTION [KEY=VALUE...] --format json` |
| `shutdown` | | `null`, then the session exits | |

Params are always by name. A flag is `true` (`{"partial": true}`); other values are numbers or strings as on the command line. Relative paths are relative to the session's working directory, as for the one-shot commands.

The session never writes the input file: `case.update` takes the text of the editor, saved or not. A check after an edit reads the mesh file again only if the file itself changed (its size or modification time); the case's `mesh_files_read` counts the reads. An unchanged case answers a repeated request from memory.

While a request runs, the session sends `$/progress` notifications naming the request and the setup phase it starts. Warnings raised outside a result's diagnostics come as a `$/log` notification with a `diagnostics` list.

To stop a request, send the notification `{"jsonrpc": "2.0", "method": "$/cancel", "params": {"id": 3}}`. A request not yet started is answered with the canceled error; a running one stops at the start of its next setup phase.

## Errors

A problem of the case is not an error of the request: `case.check` succeeds and its result lists the problem as a diagnostic with its code, key, array element, line and column. Errors of a request:

| Code | Meaning |
|---|---|
| -32700 | The line is not JSON |
| -32600 | Not a request: a batch (send one request per line), no `"jsonrpc": "2.0"`, or no method |
| -32601 | No such method |
| -32602 | Wrong params: a name the method does not take, or a value of the wrong type |
| -32603 | An internal error of Mallard (please report it) |
| -32800 | Canceled |
| -32001 | No case is open: send `case.open` first |
| -32003 | A file cannot be read; `data.path` names it |
| -32004 | Timeout: `data.phase` says what was awaited (`accepted`, `done`) |
| -32005 | No such run, or it ended (`data.outcome`) |
| -32006 | No route reached the run: `data.tried` says why, and `data.route_hint.argv` is the command to run where it can (e.g. `kubectl exec ...`) |
| -32010 | `run.launch` of an open case whose text differs from its file: save it first |

## Schemas

Every result has its schema's `version`, the session's own results also its name (`"schema": "mallard-case"`), and `initialize` gives each method's result schema. The schemas are in [`schemas/`](https://github.com/MatthewBonanni/mallard/tree/main/schemas) (installed in `share/mallard/schemas`):

| Schema | Of |
|---|---|
| `mallard-capabilities.v1` | `mallard version --format json` and `initialize` (with `case`) |
| `mallard-session.v1` | any message the session writes |
| `mallard-case.v1` | the results of `case.open` and `case.update` |
| `mallard-check.v1` | `mallard check --format json` and `case.check` |
| `mallard-case-view.v1` | `case.view`, and the `view` of `mallard check --view DIR --format json` |
| `mallard-boundary-values.v1` | `case.boundary_values`, and the `boundary_values` of `mallard check --boundary-values DIR --format json` |
| `mallard-key-catalog.v1` | `mallard list keys --format json`, `case.keys` and `keys.catalog` |
| `mallard-case-files.v1` | `mallard files INPUT --format json` and `case.files` |
| `mallard-file-info.v1` | `mallard info FILE --format json` (an output or restart file) and `file.info`; its `version` is the file's format version, as it always was |
| `mallard-unit-catalog.v1` | `mallard list units --format json` and `units.catalog` |
| `mallard-diagnostic.v1` | one diagnostic, as the others list them |
| `mallard-run-control.v2` | `mallard ctl --format json` and `run.control`; `$defs` has the socket's requests and replies and the run log's `control` records |
| `mallard-run-launch.v1` | `mallard -i INPUT --detach` and `run.launch` |
| `mallard-run-endpoint.v1` | a run's `mallard.run.json` and its copy in the run registry |

A version changes only when a member is removed or changes meaning; new members may appear in any release, so a client should ignore members it does not know. The test suite checks every result, from the session and from the one-shot commands, against its schema.

The requests still to come (cuts and picks, following a run's log) are described in the design notes, `docs/design/session.md`.
