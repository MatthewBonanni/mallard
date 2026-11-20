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
| `initialize` | | the session: `mallard` version, `dimension`, every method with its one-shot command and result schema, and the open case | |
| `case.open` | `input` (path), `text` (optional: the input's text, if it differs from the file) | the case (`mallard-case`) | |
| `case.update` | `text`: the whole input | the case, with `changed`: the key paths whose values differ | |
| `case.close` | | `null` | |
| `case.check` | the options of `mallard check` by name: `partial`, `ranks`, `memory`, `throughput`, `steps`, `mesh_fields`, `stencils`, `cells`, `region`, `weights`, `catalog`, `view`, `restart` | `mallard-check` | `mallard check INPUT --format json` |
| `case.keys` | `section` (optional) | `mallard-key-catalog` | `mallard list keys [SECTION] INPUT --format json` |
| `keys.catalog` | `section` (optional) | `mallard-key-catalog` | `mallard list keys [SECTION] --format json` |
| `shutdown` | | `null`, then the session exits | |

Params are always by name. A flag is `true` (`{"partial": true}`); other values are numbers or strings as on the command line. Relative paths are relative to the session's working directory, as for the one-shot commands.

The session never writes the input file: `case.update` takes the text of the editor, saved or not. A check after an edit reads the mesh file again only if the file itself changed (its size or modification time); the case's `mesh_files_read` counts the reads. An unchanged case answers a repeated request from memory.

While a request runs, the session sends `$/progress` notifications naming the request and the setup phase it starts. Warnings raised outside a result's diagnostics come as a `$/log` notification with a `diagnostics` list.

To stop a request, send the notification `{"jsonrpc": "2.0", "method": "$/cancel", "params": {"id": 3}}`. A request not yet started is answered with the cancelled error; a running one stops at the start of its next setup phase.

## Errors

A problem of the case is not an error of the request: `case.check` succeeds and its result lists the problem as a diagnostic with its code, key, array element, line and column. Errors of a request:

| Code | Meaning |
|---|---|
| -32700 | The line is not JSON |
| -32600 | Not a request: a batch (send one request per line), no `"jsonrpc": "2.0"`, or no method |
| -32601 | No such method |
| -32602 | Wrong params: a name the method does not take, or a value of the wrong type |
| -32603 | An internal error of Mallard (please report it) |
| -32800 | Cancelled |
| -32001 | No case is open: send `case.open` first |
| -32003 | A file cannot be read; `data.path` names it |

## Schemas

Every result has its schema's `version`, the session's own results also its name (`"schema": "mallard-case"`), and `initialize` gives each method's result schema. The schemas are in [`schemas/`](https://github.com/MatthewBonanni/mallard/tree/main/schemas) (installed in `share/mallard/schemas`):

| Schema | Of |
|---|---|
| `mallard-session.v1` | the result of `initialize`; `$defs/message` is any message the session writes |
| `mallard-case.v1` | the results of `case.open` and `case.update` |
| `mallard-check.v1` | `mallard check --format json` and `case.check` |
| `mallard-key-catalog.v1` | `mallard list keys --format json`, `case.keys` and `keys.catalog` |
| `mallard-diagnostic.v1` | one diagnostic, as the others list them |

A version changes only when a member is removed or changes meaning; new members may appear in any release, so a client should ignore members it does not know. The test suite checks every result, from the session and from the one-shot commands, against its schema.

The requests still to come (run launch and control, case views, cuts and picks) are described in the design notes, `docs/design/session.md`.
