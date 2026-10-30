# Contributing to Mallard

Thank you for helping. Bug reports, questions, documentation fixes, examples and
new physics are all welcome, from first-time contributors and CFD experts alike.
This page is the source of truth for how changes land; if anything here is unclear,
that is a documentation bug worth reporting.

Everyone taking part follows the [Code of Conduct](CODE_OF_CONDUCT.md).

- [Reporting bugs and asking questions](#reporting-bugs-and-asking-questions)
- [Building and testing](#building-and-testing)
- [Code style](#code-style)
- [Adding a feature, end to end](#adding-a-feature-end-to-end)
- [Pull requests and CI](#pull-requests-and-ci)

## Reporting bugs and asking questions

- **Bugs:** open an issue with the *Bug report* template. The most useful reports
  have the smallest input file that shows the problem, the Mallard version or git
  commit, the build (2D/3D, backend, precision, MPI ranks) and the full error or
  log. Once Mallard writes a provenance block into its output files, paste that
  block: it records all of this.
- **Features:** open an issue with the *Feature request* template. Describe the
  problem first (the flow, the quantity, the case you cannot run), then the
  solution you have in mind and a reference (paper, other code) if there is one.
  For larger changes, agree on the design in the issue before writing code.
- **Questions:** check the [user guide](docs/guide/input_files.md), the
  [tutorials](docs/tutorials/index.md) and [every input key](docs/input.md) first;
  if they do not answer it, open an issue with the *Question* template. A question
  the docs should have answered is also a documentation bug.
- **Security issues:** do not open a public issue; contact
  [@MatthewBonanni](https://github.com/MatthewBonanni) privately as described in the
  [Code of Conduct](CODE_OF_CONDUCT.md#enforcement).

## Building and testing

The [README](README.md#building) covers building in full. For development:

```sh
git clone --recursive https://github.com/MatthewBonanni/mallard.git
cd mallard
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_SYSTEM_KOKKOS=OFF \
    -DKokkos_ENABLE_THREADS=ON -DMallard_WARNINGS_AS_ERRORS=ON
cmake --build build -j
./build/src/mallard test
```

- Build 3D in a separate directory with `-DMallard_DIM=3`; code that is not
  dimension specific must build and pass in both.
- Run a subset while iterating: `./build/src/mallard test --gtest_filter='TENO*'`.
  `mallard test` runs the test suite of a build (`build/test/mallard-test2d`, or
  `mallard-test3d` in 3D) or every one of an install tree (`--dim 2|3` picks one).
  Run the whole suite, 2D and 3D, before asking for review.
- With MPI (`-DMallard_ENABLE_MPI=ON`), run the multi-rank tests on 2 to 4 ranks:
  `mpirun -n 3 ./build/src/mallard test --gtest_filter='CommTest.*:MeshBlockTest.*:DistributedMeshTest.*:MPITest.*:MPI3DTest.*:ProbesTest.*'`.
  CI also runs the whole suite on 3 ranks, so every test must pass on any
  number: a test that inspects one rank's cells runs its solver undistributed
  (`set_distributed(false)`) or reduces over the ranks, loops over owned cells
  (halo cells hold stale values), and leaves shared files to rank 0 (files that
  every rank writes on its own go in `rank_scratch_dir()`).
- Python tools (`tools/`) need numpy, scipy, matplotlib, imageio and
  imageio-ffmpeg; a virtual environment works.

## Code style

- **Style:** the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html),
  in C++20. Match the surrounding code where the two differ.
- **Comments:** Doxygen (`/** ... */`, `@brief`, `@param`, `@return`) on every public
  interface: classes, functions and members declared in headers. Elsewhere, keep
  comments minimal: say *why* when it is not obvious, never restate *what* the code
  does. Rationale and measurements belong in the pull request or in `docs/design/`.
- **Warnings are errors.** CI builds with `-DMallard_WARNINGS_AS_ERRORS=ON` on GCC,
  Clang and nvcc, with `-Wall -Wextra -Wpedantic -Wshadow -Wconversion
  -Wsign-conversion -Wdouble-promotion -Wold-style-cast` and more (see
  `mallard_target_warnings` in [CMakeLists.txt](CMakeLists.txt)). Fix the cause;
  use explicit `static_cast`s for intended conversions, and `rtype` (not `double`)
  for floating-point values so single precision builds too.
- **Input keys** are read only through `Input` (`src/common/input.h`), never
  `toml::value` or `toml::table` directly (configuration fails if you do). Every
  key setup reads is then validated automatically: misspelled or unused keys are
  errors, with suggestions.
- **Error messages** teach: an invalid value throws an `InputError` built from a
  `Diagnostic` (key, line from `line_of`, what is wrong, the accepted values, the
  fix and the docs page); `missing_key`, `not_accepted` and `cannot_read` cover the
  common cases. A run and `mallard check` report it the same way, and
  `test/error_messages_test.cpp` holds an example of each kind.
- **Kokkos device code** must run on CUDA and HIP, not only on the CPU backends:
  - Kernels (`KOKKOS_LAMBDA`, functors) capture by value; never capture `this` or
    references to host objects. Copy the members a kernel needs into locals first.
  - Inside kernels: no `std::` containers or strings, no exceptions, no virtual
    calls, no allocation, no I/O. Functions called from kernels are
    `KOKKOS_INLINE_FUNCTION`; types captured by kernels are trivially copyable.
  - Data lives in `Kokkos::View`s in the default memory space; copy to the host
    with `Kokkos::deep_copy` (or a mirror) outside the time loop, not per step.
  - Results must not depend on the backend, thread count or rank count: reductions
    that feed the solution must be deterministic (the suite checks bitwise
    reproducibility across ranks and restarts).
  - nvcc compiles every source, including host-only code, so a host-only change can
    still break the CUDA build; the CUDA compile jobs in CI catch this.

## Adding a feature, end to end

A feature is done when a user can find it, use it and trust it. A pull request
that adds or changes behavior includes:

1. **Code**, in `src/`, following the style above. New input keys are read through
   `Input` with explicit defaults.
2. **Tests** in `test/` (GoogleTest, registered in
   [test/CMakeLists.txt](test/CMakeLists.txt)). Each test must fail for a plausible
   regression: prefer checks against an exact solution, a design order of
   accuracy, conservation, symmetry or an independent reference (an exact
   Riemann solver, Cantera) over restating the implementation. If you cannot name
   the bug a test would catch, it is not worth keeping.
3. **Documentation**, following [docs/design/documentation.md](docs/design/documentation.md):
   - the user-facing explanation (when to use it, a minimal TOML snippet, keys with
     defaults, pitfalls) in the topic's page of `docs/guide/`;
   - every new key with its default in [docs/input.md](docs/input.md);
   - equations and sources in `docs/theory/`, design rationale and measurements in
     `docs/design/`, and new references in [docs/references.md](docs/references.md);
   - figures drawn by a script in `tools/docs/` as a light/dark SVG pair
     (`python tools/docs/make_figures.py --mallard build/src/mallard`), never
     hand-edited images.
4. **An example** in `examples/<case>/` with its `input.toml` and a row in
   [examples/README.md](examples/README.md), ideally compared with a reference
   (exact solution, experiment, DNS). Add it to the example list in
   [test/example_inputs.h](test/example_inputs.h) where that applies, so the suite
   keeps it running.
5. **A changelog fragment**: a PR that changes `src/` adds
   `changelog/<branch>.<added|changed|deprecated|removed|fixed|security>.md` with one
   `- ` bullet per change (see [changelog/README.md](changelog/README.md)). Changes
   that alter results go under `changed`. CI fails without one; maintainers add the
   `no-changelog` label to PRs that do not need one.
6. **README and website:** if the feature belongs in the README's feature list, add
   it there; the maintainer mirrors README changes on the
   [website](https://matthewbonanni.github.io/mallard/), which builds its pages
   from `docs/`.

Small fixes need only what applies: a bug fix needs a test that failed before the
fix and a `fixed` fragment; a documentation fix needs neither.

## Pull requests and CI

- Branch from `main` (on a fork, or a branch if you have access) and open a pull
  request early as a draft if you want feedback. Keep each PR to one change.
- Fill in the PR template: what and why, the exact commands you ran to validate it
  and their results, and whether it touches device code.
- Pull requests are squash-merged: the PR title becomes the commit subject on
  `main`, so make it descriptive. Intermediate commits on the branch need not be
  tidy; push new commits rather than rewriting history once a review has started.
- **CI does not run on every push.** When a PR is ready to land, a maintainer (or
  you, with write access) runs `gh workflow run ci.yml --ref <branch>`. The run sets
  the commit status `CI` on the head commit: pending while it runs, then success or
  failure, and its *Details* link shows every job and its log. `main` requires a
  passing `CI` status on the PR's current head, so a new push needs a new run.
  CI builds with warnings as errors and runs the full suite in 2D and 3D, with
  OpenMP, serial, MPI on 1 to 4 ranks and single precision, plus a single-precision
  Clang build and the tutorials and case templates. Each job also shows on the PR as
  its own status, `CI / <job>`. Documentation-only changes skip the build jobs (the
  run takes about a minute). `main` runs CI on every merge.
- **After resolving a merge conflict** on a PR whose CI already passed, rerun only
  what the resolution can affect: `gh workflow run ci.yml --ref <branch> -f
  only=openmp,mpi-3d` (backend names, `tutorials`, `device`, `fuzz`). The `CI` status
  records the subset. A resolution in build files, shared headers or anything the
  whole suite exercises needs the full run.
- **Device builds**: CUDA, HIP (ROCm) and SYCL (oneAPI), 2D and 3D, compile only,
  run when a branch's changes touch device code: `tools/dev/device_changes.sh` decides (a
  changed `src/` file with Kokkos device constructs, or a Kokkos update). `main` also
  runs the CUDA builds on every push. `-f device=true` forces all three.
- **GPU CI.** Device code is also run on NVIDIA A100 GPUs by a maintainer-run GPU
  CI, outside GitHub Actions: CUDA 12 and 13 builds in 2D and 3D, the full test suite
  on the GPU (including the CUDA-graph and NCCL paths), the MPI tests on 2 ranks, and
  the [performance suite](benchmarks/README.md) against per-hardware baselines. It
  posts the commit statuses `gpu/tests` and `gpu/perf` on the PR's head commit, so
  its results show on the PR next to `CI`. A PR that changes device code merges only
  with both passing. You do not need a GPU to contribute: say in the PR template
  whether device code changed, and a maintainer runs it. Performance claims need
  numbers from the performance suite or a real case, before and after.
- Sanitizers (AddressSanitizer and UndefinedBehaviorSanitizer) run nightly on
  `main`.

By contributing, you agree that your contributions are licensed under the
[Apache License, Version 2.0](LICENSE).
