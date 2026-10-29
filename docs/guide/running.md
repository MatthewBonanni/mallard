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

## MPI launchers and Slurm

`mpirun` and `mpiexec` forward `SIGUSR1` and `SIGUSR2` to every rank, but Open MPI ends the job on `SIGTERM` and `SIGINT` without letting the ranks finish their step: send `SIGUSR2` to `mpirun` to stop, or use `mallard ctl`. Some MPI distributions start the real launcher from a wrapper that does not pass signals on: the NVIDIA HPC SDK's `mpirun` runs HPC-X's `ompi/bin/.bin/mpirun` from a script, and a signal to the wrapper ends the wrapper and leaves the run going. Signal the launcher it starts, or use `mallard ctl`. Slurm's `scancel` signals every task directly, so `scancel --signal=TERM` works, as does `#SBATCH --signal=USR2@120` for a restart two minutes before the time limit.

## Divergence

Every step's time-step reduction also tells whether any owned cell holds a non-finite value. The run then stops with the number of such cells and the first ones by global cell id (variable, value, cell, centroid and rank):

```text
error: Solution diverged at step 611, t = 1.90583: 1 cell has a non-finite value: RHOE = nan in cell 19 at
(0.025, 0.975). The last valid state (step 610, t = 1.90328) is in restart_000610.restart.
```

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

**Restarts.** When the pod is preempted, evicted or deleted, a `preStop` hook stops the run with `mallard ctl stop`, which writes a restart file (with `mpirun`, `SIGTERM` alone would end the ranks without one); `terminationGracePeriodSeconds` must exceed a step plus that write. Exit status 3 does not count as a failure (`podFailurePolicy`), so the `Job` starts a new pod, which resumes from the newest restart file the run wrote: it replaces the input's `[initialize]` with `type = "restart"` in `resume.toml`. To end the run for good, `kubectl delete job mallard`: the run still writes its restart file, from which a new `Job` resumes.

**Several nodes.** The images' Open MPI starts ranks on other pods over SSH, as the [Kubeflow MPI Operator](https://github.com/kubeflow/mpi-operator) (v2beta1) does: `mpijob.yaml` runs `mpirun` in a launcher pod and 4 ranks in each of 2 worker pods, with sshd as root. Halos go between GPUs with NCCL. The volume must be `ReadWriteMany` (e.g. NFS), at the same path in every pod. To stop, `SIGUSR2` to `mpirun` (the launcher's `preStop` hook) writes a restart file, from which the launcher's next attempt resumes. This manifest has not been tested yet.

**Pulling the images.** The packages on GHCR are public from 0.9.0 on. Before that, or for a private fork, pull with a token that has `read:packages`:

```sh
kubectl create secret docker-registry ghcr --docker-server=ghcr.io \
    --docker-username=<github-user> --docker-password=<token>
```

and uncomment `imagePullSecrets: [{name: ghcr}]` in the manifests.
