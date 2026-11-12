<!-- See CONTRIBUTING.md. Delete what does not apply. -->

Closes #

## What and why

## Validation

<!-- The exact commands you ran (build options, test filters, cases, ranks, hardware) and their results. For bug fixes: the test or case that failed before the fix. For performance: numbers before and after. -->

```sh
```

## Checklist

- [ ] Tests that would fail on a plausible regression
- [ ] Docs: `docs/guide/`, every new or changed key in `docs/input.md`, figures from `tools/docs/` (light and dark)
- [ ] Example in `examples/` with a row in `examples/README.md` (new features)
- [ ] Changelog fragment in `changelog/` (any change to `src/`)
- [ ] README feature list updated, if it should mention this
- [ ] Builds without warnings with `-DMallard_WARNINGS_AS_ERRORS=ON`, 2D and 3D

## Device code

- [ ] This PR changes code that runs on the GPU (kernels, Kokkos functors or anything they call); a maintainer runs the GPU CI
