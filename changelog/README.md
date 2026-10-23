# Changelog fragments

A pull request that changes `src/` adds one file here instead of editing
`CHANGELOG.md`, so that pull requests landing in parallel do not conflict:

```
changelog/<branch>.<category>.md
```

with `<category>` one of `added`, `changed`, `deprecated`, `removed`, `fixed`,
`security`, holding one `- ` bullet per change (several files for several
categories). `python3 tools/dev/changelog.py notes Unreleased` previews them; at a
release, `python3 tools/dev/changelog.py release X.Y.Z` folds them into
`CHANGELOG.md` and deletes them. What to write:
[docs/design/documentation.md](../docs/design/documentation.md#changelog).
