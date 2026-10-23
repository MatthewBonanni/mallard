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

Exit statuses are 0 at a stop condition (`n_steps`, `t_stop`, `t_wall_stop`), 1 on errors (including divergence) and 3 when stopped on request.

## MPI launchers, Slurm and Kubernetes

`mpirun` and `mpiexec` forward `SIGUSR1` and `SIGUSR2` to every rank, but Open MPI ends the job on `SIGTERM` and `SIGINT` without letting the ranks finish their step: send `SIGUSR2` to `mpirun` to stop, or use `mallard ctl`. Some MPI distributions start the real launcher from a wrapper that does not pass signals on: the NVIDIA HPC SDK's `mpirun` runs HPC-X's `ompi/bin/.bin/mpirun` from a script, and a signal to the wrapper ends the wrapper and leaves the run going. Signal the launcher it starts, or use `mallard ctl`. Slurm's `scancel` signals every task directly, so `scancel --signal=TERM` works, as does `#SBATCH --signal=USR2@120` for a restart two minutes before the time limit.

In Kubernetes, give the pod a `terminationGracePeriodSeconds` longer than a step plus a restart write. A single `mallard` process receives the pod's `SIGTERM` itself; with `mpirun` as the container's command, add a `preStop` hook running `mallard ctl <run-dir> stop`, which returns once the restart is written.

## Divergence

Every step's time-step reduction also tells whether any owned cell holds a non-finite value. The run then stops with the number of such cells and the first ones by global cell id (variable, value, cell, centroid and rank):

```text
error: Solution diverged at step 611, t = 1.90583: 1 cell has a non-finite value: RHOE = nan in cell 19 at
(0.025, 0.975). The last valid state (step 610, t = 1.90328) is in restart_000610.restart.
```

It writes the state at the start of the step before, the last valid one (all `[[write_data]]` outputs and a restart; see [`keep_last_good`](../input.md#run)), and exits with status 1. The invalid state is found one step after it appears, or right after its step with `[numerics] check_nan`; without `check_nan`, the outputs and `[statistics]` samples of that step may hold it (resume from the last valid state with `[statistics] reset = true` if its averages are not finite).
