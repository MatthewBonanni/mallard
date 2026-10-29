# Command line

How the `mallard` command is laid out, and the rules every command follows. The
user-facing pages are in [Command line](../guide/cli.md).

## Layout

```text
Run       mallard -i input.toml [--restart latest] [--set k=v] | ctl | reactor
Set up    mallard new | input upgrade | input resolve
Inspect   mallard check [input.toml] | info <file> | list fields|species|templates|keys | diff
Mesh      mallard mesh convert | extrude
Post      mallard post extract|derive|average|convergence|render
Share     mallard pack
Maintain  mallard test | completion | --version
```

| Command | Group | Status |
|---|---|---|
| `mallard -i input.toml`, `ctl`, `reactor` | Run | available |
| `--restart latest` | Run | planned ([#510](https://github.com/MatthewBonanni/mallard/issues/510)) |
| `new`, `input upgrade` | Set up | available (`upgrade-input` is a deprecated alias of `input upgrade`) |
| `input resolve` | Set up | planned ([#509](https://github.com/MatthewBonanni/mallard/issues/509)) |
| `check input.toml` | Inspect | available |
| `check` with no input | Inspect | planned ([#508](https://github.com/MatthewBonanni/mallard/issues/508)) |
| `info <file>`, `info <run-dir> [--html]` (the run report) | Inspect | planned ([#504](https://github.com/MatthewBonanni/mallard/issues/504), [#515](https://github.com/MatthewBonanni/mallard/issues/515)) |
| `list fields\|species\|templates\|keys` | Inspect | planned ([#502](https://github.com/MatthewBonanni/mallard/issues/502)) |
| `diff` | Inspect | planned ([#506](https://github.com/MatthewBonanni/mallard/issues/506)) |
| `mesh convert` | Mesh | available |
| `mesh extrude` | Mesh | planned |
| `post extract` | Post | planned ([#505](https://github.com/MatthewBonanni/mallard/issues/505)) |
| `post derive` | Post | planned ([#511](https://github.com/MatthewBonanni/mallard/issues/511)) |
| `post average` | Post | planned ([#512](https://github.com/MatthewBonanni/mallard/issues/512)) |
| Reacting-flow analysis | Post | planned ([#513](https://github.com/MatthewBonanni/mallard/issues/513)) |
| `post convergence` | Post | planned ([#514](https://github.com/MatthewBonanni/mallard/issues/514)) |
| `post render` | Post | planned ([#516](https://github.com/MatthewBonanni/mallard/issues/516)) |
| `pack` | Share | planned |
| `test`, `--version` | Maintain | available |
| `completion` | Maintain | planned ([#507](https://github.com/MatthewBonanni/mallard/issues/507)) |

## Rules

- **Frequent actions are top-level verbs** (`check`, `new`, `ctl`); **families
  are noun groups** (`mesh`, `input`, `post`) whose verbs follow the noun:
  `mallard mesh convert`, `mallard post extract`. `mallard <family> --help`
  lists a family's commands.
- **One home per concept.** Mesh-quality and stencil fields come from
  `--mesh-fields`, not a command of their own; there is one field registry;
  regions describe all geometry; catalogs (fields, species, templates, keys)
  are under `mallard list`.
- **Renames keep the old name** as a deprecated alias for at least one minor
  release: it runs the command and prints `mallard <old> is deprecated: use
  mallard <new>` (on stderr; on stdout above the page for `--help`).
- **One table.** Every command is a row of the command table
  (`cli::commands()` in `src/commands.cpp`): its group, one-line summary,
  synopsis, description, options, arguments, examples, exit status, how the
  front end finds its dimension, and the function that runs it. The parser,
  the dispatcher, every help page, the top-level page (grouped as above), the
  user docs and, later, shell completion are made from it, so a command or
  option cannot exist without its help.
- **Help pages.** Each has a synopsis, a description, every option with its
  argument and default, examples, the exit status and a link to its section of
  the [Command line](../guide/cli.md) page. `-h` or `--help` anywhere in a
  command's arguments prints its page with exit status 0. Versions in help
  text are the build's (`{version}` in the table).
- **Usage errors** (an unknown option or command, a missing argument) print
  one line, `mallard <cmd>: <what> (see mallard <cmd> --help)`, with exit
  status 2. Other failures exit with 1; a run stopped on request with 3.
- **Pass-through options** are rows whose name ends in `*`: `--kokkos-*` for
  every command that starts Kokkos, `--gtest_*` for `mallard test`.
- **Tests** (`test/cli_test.cpp`) run every command's `--help` and fail if a
  command lacks a summary, a group or a page, or if its page misses an option
  the parser accepts; and check that `docs/guide/cli.md` shows every page as
  it is (`python tools/docs/cli.py --mallard build/src/mallard` regenerates
  it).

## Adding a command

1. A row in `src/commands.cpp` with every field, in its group of the layout
   above; a `Family` row for a new noun.
2. Its run function takes the parsed `cli::Args`; it reports misuse the parser
   cannot see with `cli::usage_error`.
3. `python tools/docs/cli.py --mallard build/src/mallard` adds its page to
   the user docs; move its row in the table above from planned to available.
