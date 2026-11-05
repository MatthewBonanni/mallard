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
| `STOP`, `CHECKPOINT` files | With `[run] control_file_interval`, as `stop` and `checkpoint`; the file is removed once seen |

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

Exit statuses are 0 at a stop condition (`n_steps`, `t_stop`, `t_wall_stop`, a [trigger](#triggers)), 1 on errors (including divergence) and 3 when stopped on request.

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
variables = ["T"]
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

Every step's time-step reduction also tells whether any owned cell holds a non-finite value. The run then stops with the number of such cells and the first ones by global cell id (variable, value, cell, centroid and rank):

```text
error: Solution diverged at step 611, t = 1.90583: 1 cell has a non-finite value: RHOE = nan in cell 19 at
(0.025, 0.975). The last valid state (step 610, t = 1.90328) is in restart_000610.restart. Common causes: ...
```

The message ends with the usual causes: a time step too large for the scheme
(lower `run.cfl`), initial or boundary values far from the flow (check units
and signs), or poor cells near the first one listed (`mallard check` reports
mesh quality); a more dissipative scheme (`numerics.preset = "robust"`) helps.

It writes the state at the start of the step before, the last valid one (all `[[write_data]]` outputs and a restart; see [`keep_last_good`](../input.md#run)), and exits with status 1. The invalid state is found one step after it appears, or right after its step with `[numerics] check_nan`; without `check_nan`, the outputs and `[statistics]` samples of that step may hold it (resume from the last valid state with `[statistics] reset = true` if its averages are not finite).

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
