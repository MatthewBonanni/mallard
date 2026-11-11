# Running and controlling jobs

A running `mallard` takes requests from signals, from `mallard ctl` and, optionally, from files. Every rank acts on them after the same step: the requests travel with the time step's global reduction, so they cost no communication of their own.

| Request | Effect |
|---|---|
| `SIGUSR1` | Write a restart after the current step and continue |
| `SIGTERM`, `SIGINT`, `SIGUSR2` | Write a restart and the final output after the current step, then exit with status 3. A second `SIGINT` (Ctrl-C twice) exits at once. Before the time loop starts, these signals end the run at once |
| `mallard ctl <run-dir> status` | Step, `t`, `dt`, CFL number, progress, wall time per step, cells per second, ETA, memory of rank 0's process (and of its GPU), and the diagnostics of the last progress row |
| `mallard ctl <run-dir> checkpoint` | As `SIGUSR1`; replies with the restart file once it is written |
| `mallard ctl <run-dir> stop` | As `SIGTERM`; replies with the restart file once the run has ended |
| `mallard ctl <run-dir> set <key>=<value> ...` | Change settings from the next step on: `n_steps`, `t_stop`, `t_wall_stop` (0 removes a limit; one must remain; times in seconds or with a unit, e.g. `t_wall_stop=2h`), `check_interval`, `write_data.<prefix or index>.interval` or `.time_interval` (whichever the writer has), `probes.<name>.interval` or `.time_interval`, `forces.<zone>.interval`, `integrals.interval`. Other keys are refused |
| `STOP`, `CHECKPOINT` files | With `[run] control_file_interval`, as `stop` and `checkpoint` (and at the same step as they would act); the file is removed once seen |

```bash
$ mallard ctl runs/cylinder status
state: running
step: 12400
t: 0.8214
dt: 6.62e-05
cfl: 1
progress: 41.07%
...
$ mallard ctl runs/cylinder set t_stop=3.0 write_data.flow.time_interval=0.05
$ mallard ctl runs/cylinder stop
stopped at step 12415, t = 0.82239; restart: restart_012415.restart
```

Restarts on request go to the `[[write_data]]` restart writers, or to `restart_<step>.restart` in the working directory if there is none. `mallard ctl` talks to rank 0 through the Unix-domain socket `mallard.sock` in the directory the run was started from, readable and writable by its owner only, so it must run on rank 0's node (in Kubernetes: `kubectl exec`). `[run] control_socket = false` turns the socket off.

Exit statuses are 0 at a stop condition (`n_steps`, `t_stop`, `t_wall_stop`, a [trigger](#triggers)), 1 on errors (including divergence) and 3 when stopped on request. The run record's [end record](output.md#run-record-mallardjsonl) says which, as its `outcome`, with the last snapshot and restart file written; `mallard info <run-dir>` shows it. `t_wall_stop` is reached on whichever rank gets there first, and every rank stops after the same next step.

## Triggers

A `[[triggers]]` entry is a request the run makes of itself: a condition on
the run, and what to do once it is true. Use one to stop a run when it has
reached what it was for (a temperature, a force that has settled), to keep a
restart or a snapshot of the moment something happens, or to start the
statistics once the flow has developed.

```toml
[[probes]]
name = "outlet"
point = [0.9, 0.5]
fields = ["T"]
interval = 1

[[triggers]]
name = "ignition"
when = "outlet_T > 1500"          # a kexpr expression
do = ["checkpoint", "write flow"] # the prefix of a [[write_data]] entry

[[triggers]]
name = "settled"
when = "t > 0.02 and abs(forces_cylinder_Fx_pressure) < 1e-3"
for = 0.005                       # held this long (seconds of simulated time)
do = "stop"
```

| Key | Meaning |
|---|---|
| `when` | The condition, in [kexpr](expressions.md); true when not 0. Required |
| `do` | An action, or a list of them done in order: `"stop"`, `"checkpoint"`, `"write <prefix>"` (that `[[write_data]]` entry, now), `"start statistics"`. Required |
| `name` | In the log, the run record and restart files; letters, digits and `_`. Default `trigger_<index>` (from 0) |
| `for` | Seconds of simulated time the condition must hold before the trigger fires, counted from the step it last turned true. Default 0 |
| `repeat` | `false` fires once; `true` fires again each time the condition turns true after being false. Default `false` |

**Names in `when`.** `t` (s), `step`, `dt` (the last step's, s) and `wall`
(seconds since the solver started, rank 0's), and the latest value of every
monitored quantity, named after its file's column with the monitor's name in
front:

| Monitor | Names |
|---|---|
| `[integrals]` | `integrals_<column>`, e.g. `integrals_kinetic_energy` |
| `[[forces]]` | `forces_<zone>_<column>`, e.g. `forces_cylinder_Fx_pressure`, `forces_wall_Q` |
| `[[probes]]` reductions | `<probe>_<reduction>_<variable>`, e.g. `combustor_max_T`, `inlet_integral_MASS_FLUX` |
| `[[probes]]` of one point | `<probe>_<variable>`, e.g. `outlet_T` |

Characters other than letters, digits and `_` in these names become `_`.
A misspelled name is an input error (also of `mallard check`) that lists
the names the run has.

**When conditions are evaluated.** After every step, once that step's
monitors and probes are written, from the values they wrote: a monitor
written every 10 steps keeps its value in between. A condition that reads a
monitor before its first value is false. Evaluating costs nothing on the
device: on several ranks rank 0 evaluates every condition and sends the
outcome to the others, one small message per step (none when the run has no
triggers), so every rank acts after the same step.

**Actions.** `stop` ends the run after that step as `t_stop` would: the
final output and a restart are written, the exit status is 0 and the log
says which trigger stopped it. `checkpoint` writes a restart (as `mallard ctl
checkpoint`). `write <prefix>` writes that `[[write_data]]` entry's file at
once, besides its own schedule. `start statistics` discards the averages so
far and starts them at that step, in `[statistics]` and its `spectra` and
`phase`: they wait for the trigger, so `t_start` cannot be given as well.

**Record.** Each firing is a line of the log and an `event` record of kind
`trigger` in [`mallard.jsonl`](output.md#run-record-mallardjsonl), with the
trigger's `name`, its `firing` (1 for the first) and the `values` its
condition read. Restart files carry each trigger's state (fired, how many
times, since when its condition has held), so a resumed run neither fires a
`repeat = false` trigger again nor restarts a `for` clock.

Pitfalls: a run still needs `n_steps`, `t_stop` or `t_wall_stop` in
`[run]`, the limit if no trigger fires; `wall` is only as reproducible as
the machine's speed; `==` on `t` rarely holds exactly (use `>=`).

## MPI launchers and Slurm

`mpirun` and `mpiexec` forward `SIGUSR1` and `SIGUSR2` to every rank, but Open MPI ends the job on `SIGTERM` and `SIGINT` without letting the ranks finish their step: send `SIGUSR2` to `mpirun` to stop, or use `mallard ctl`. Some MPI distributions start the real launcher from a wrapper that does not pass signals on: the NVIDIA HPC SDK's `mpirun` runs HPC-X's `ompi/bin/.bin/mpirun` from a script, and a signal to the wrapper ends the wrapper and leaves the run going. Signal the launcher it starts, or use `mallard ctl`. Slurm's `scancel` signals every task directly, so `scancel --signal=TERM` works, as does `#SBATCH --signal=USR2@120` for a restart two minutes before the time limit.

## Divergence

Every step's time-step reduction also tells whether any owned cell holds a non-finite value. The run then stops with the number of such cells and the first ones by global cell id (variable, value, cell, centroid and rank), and the most likely cause with what to try:

```text
error: Solution diverged at step 9, t = 0.707496: 48 cells have a non-finite value: RHO = nan in cell 38 at
(0.15625, 0.40625); ... The last valid state (step 8, t = 0.632983) is in restart_000008.restart. Most likely
cause: time step too large (the failing step ran at CFL 3 (run.cfl = 3), above 1.2, the largest stable value of
SSPRK3 for smooth flow); try: lower run.cfl to 1. divergence_report.txt has the evidence and other causes.
```

It writes the state at the start of the step before, the last valid one (all `[[write_data]]` outputs and a restart; see [`keep_last_good`](../input.md#run)), and exits with status 1. The invalid state is found one step after it appears, or right after its step with `[numerics] check_nan`; without `check_nan`, the outputs and `[statistics]` samples of that step may hold it (resume from the last valid state with `[statistics] reset = true` if its averages are not finite).

### The divergence report

Before it stops, the run diagnoses the failure from the failing state and the last valid one, and prints the
report; it also writes it as `divergence_report.txt` and `divergence_report.json` (`[output] divergence_report`
sets the prefix, `""` none), as a `divergence` record of the [run record](output.md#run-record-mallardjsonl),
and a snapshot of the failure's neighborhood for ParaView. It is gathered only once the run has failed, from
what the run already keeps: a healthy run does nothing more per step, and its results do not change.

- **What and where:** the step, the cells that became invalid (non-finite, or a non-positive density or
  pressure) by global id, with their rank on several ranks, what failed first in them (density, pressure,
  temperature, a mass fraction, velocity), their extent, and the cell that was closest to failing in the last
  valid state.
- **Likely causes, most likely first**, each with a weight from 0 to 1, the evidence behind it and what to try:

  | Cause | Evidence |
  |---|---|
  | Time step | The failing step's CFL number against the integrator's largest stable one for smooth flow (1.2 for SSPRK3, 1.35 for RK4, 2.2 for SSPRK54, CK45 and RDPK3Sp35, 1 for FE; [table](../numerics/overview.md#time-step)), and how many cells exceeded it: many is the time step, a few is the cells (next row) |
  | Mesh | The [quality](meshes.md#mesh-quality) of the invalid cells and their neighbors, against the questionable values and the mesh's worst; stronger when the CFL excess or the failure is confined to those cells |
  | Boundary condition | Invalid cells within two cells of a boundary zone, and the exterior state of its faces there (inflow, far field, Dirichlet, outlet pressure) against the cell next to it: not a valid state, or a pressure or density ratio above 10, or a velocity jump above 3 sound speeds |
  | Initial state | Cells of the initial state (or the restart's) with a non-positive density or pressure, a failure within ten steps of the start, and neighbors that differ by more than a factor 100 |
  | Chemistry | Reacting runs: a temperature or mass fraction that failed first, a cell that needed ten times the mean chemistry sub-steps, rejected Rosenbrock sub-steps |
  | Scheme | [Repairs](output.md#numerical-repairs) in the invalid cells since the last progress row, TENO's troubled cells and the MUSCL limiter there; always listed last, as the general remedy (`numerics.preset = "second_order"`, or `"first_order"` as the last resort) |

- **Evidence:** the state of the cell closest to failing and of its face neighbors in the last valid state
  (density, speed, pressure, temperature, local CFL number), their quality metrics, the nearest boundary zone,
  repairs, TENO and limiter activity and chemistry sub-steps; the domain means; the last progress rows before the
  failure (`dt`, CFL number, minimum density and pressure, and temperature for mixtures).
- **Reproduce:** the restart of the last valid state to resume from (`mallard -i input.toml --restart FILE`,
  with `[numerics] check_nan = true` and `[output] check_interval = 1` to follow it step by step), and
  `divergence_report_neighborhood_<step>.vtkhdf` (`.vtu` in builds without HDF5): the invalid cells and three
  layers of cells around them, with the last valid state's density, velocity, pressure, temperature, local CFL
  number and mass fractions, `DIVERGED` (1 in the invalid cells) and the four mesh-quality fields.

An inflow that the flow next to it cannot take, for example:

```text
Likely causes, most likely first
  1. boundary condition of left (inlet_mass_flow) (0.90)
     - its exterior state is far from the flow next to it: p 11.2049 and rho 11.2049 outside, against p 1 and
       rho 1 in the cell next to it (ratio 11.2, velocity jump 7.29 sound speeds)
     - 12 of the 12 invalid cells are within two cells of left
     Try: check the values of left (units and signs), or start from a state closer to them
  2. initial state (0.35)
     - the run failed 1 step after its start, before the flow could develop
     Try: check [initialize]: units, signs and the regions of its expressions (density, pressure and
     temperature must be positive everywhere)
```

and a cell squeezed to a sliver, at a fixed `dt` that suits the rest of the mesh:

```text
  1. poor cells (0.98)
     - cell 135 at (0.439063, 0.53125), which became invalid, has volume ratio 39 (questionable above 10; the
       mesh's worst is 39)
     - only 6 cells exceeded the stable CFL number: the time step suits the rest of the mesh
     Try: improve cells 119, 134, 135, 136, 151 (mallard check input.toml --mesh-fields mesh writes the
     metrics for ParaView), or lower the time step to suit them
  2. time step too large (0.55)
     - the failing step ran at CFL 3.08, above 1.2, the largest stable value of SSPRK3 for smooth flow
     - 6 of 256 cells (2%) were above it: the excess is local, so look at those cells first
```

The weights rank the causes of one failure; they are not probabilities. A pressure that becomes negative
without becoming non-finite does not stop a run (the [run health](output.md#run-health) warns of it); the
report covers the cells that do.

## Kubernetes

The [container images](https://matthewbonanni.github.io/mallard/docs/installation/#container) run as Kubernetes workloads. Manifests, in [`docs/guide/k8s`](k8s):

| File | What |
|---|---|
| [`pvc.yaml`](k8s/pvc.yaml) | The volume for the cases and their output |
| [`job.yaml`](k8s/job.yaml) | A `Job` on one GPU node: `mallard`, or `mpirun` with one rank per GPU |
| [`mpijob.yaml`](k8s/mpijob.yaml) | An `MPIJob` over several GPU nodes |
| [`files.yaml`](k8s/files.yaml) | A pod to copy cases in and results out with `kubectl cp` |

```sh
kubectl apply -f pvc.yaml -f job.yaml
kubectl logs -f job/mallard
kubectl apply -f files.yaml && kubectl cp mallard-files:/data/examples/riemann_2d/solut solut
```

The `Job` copies the examples to the volume on its first start and runs the case in `CASE` (`riemann_2d`); set `CASE` to a case of your own copied there. `RANKS`, `nvidia.com/gpu` and `OMP_NUM_THREADS` set the ranks, GPUs and host threads per rank. The cluster needs the [NVIDIA device plugin](https://github.com/NVIDIA/k8s-device-plugin) (or the GPU Operator).

**Restarts.** When the pod is preempted, evicted or deleted, a `preStop` hook stops the run with `mallard ctl stop`, which writes a restart file (with `mpirun`, `SIGTERM` alone would end the ranks without one); `terminationGracePeriodSeconds` must exceed a step plus that write. Exit status 3 does not count as a failure (`podFailurePolicy`), so the `Job` starts a new pod, which resumes with `--restart latest` from the newest complete restart file the run wrote ([resuming](initialization.md#resuming-from-the-newest-restart)). To end the run for good, `kubectl delete job mallard`: the run still writes its restart file, from which a new `Job` resumes.

**Several nodes.** The images' Open MPI starts ranks on other pods over SSH, as the [Kubeflow MPI Operator](https://github.com/kubeflow/mpi-operator) (v2beta1) does: `mpijob.yaml` runs `mpirun` in a launcher pod and 4 ranks in each of 2 worker pods, with sshd as root. Halos go between GPUs with NCCL. The volume must be `ReadWriteMany` (e.g. NFS), at the same path in every pod. To stop, `SIGUSR2` to `mpirun` (the launcher's `preStop` hook) writes a restart file, from which the launcher's next attempt resumes. This manifest has not been tested yet.

**Pulling the images.** The packages on GHCR are public from 0.9.0 on. Before that, or for a private fork, pull with a token that has `read:packages`:

```sh
kubectl create secret docker-registry ghcr --docker-server=ghcr.io \
    --docker-username=<github-user> --docker-password=<token>
```

and uncomment `imagePullSecrets: [{name: ghcr}]` in the manifests.
