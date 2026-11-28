"""mallard.Session(comm=...): solvers on an mpi4py communicator. Runs under mpirun on 3 ranks:

    mpirun -n 3 python -m pytest -q -p no:cacheprovider test/python/test_solver_mpi.py

and is skipped otherwise. Ranks 0 and 1 run a case on their own communicator while rank 2 runs it alone, each
compared bitwise with the executable on as many ranks.
"""
import os
import subprocess

import pytest

from mallard import _solver

LAUNCHED = int(os.environ.get("OMPI_COMM_WORLD_SIZE") or os.environ.get("PMI_SIZE") or 1)
pytestmark = pytest.mark.skipif(LAUNCHED != 3 or not _solver._has_extension(2) and not _solver._has_extension(3),
                                reason="needs mpirun -n 3 and a build with MPI and the solver bindings")

if LAUNCHED == 3:
    MPI = pytest.importorskip("mpi4py.MPI")
    import mallard
    import test_solver

    DIM = test_solver.DIM


@pytest.fixture(scope="module")
def group():
    world = MPI.COMM_WORLD
    comm = world.Split(color=int(world.Get_rank() == 2), key=world.Get_rank())
    mallard.Session(["--kokkos-num-threads=1"], dim=DIM, comm=comm)
    return world, comm


def without_launcher():
    """The environment of this rank without what mpirun set, so that a command started from it is a job of its own."""
    return {k: v for k, v in os.environ.items() if not k.startswith(("OMPI_", "PMIX_", "PRTE_", "PMI_"))}


def run_executable(d, ranks):
    command = [mallard._lib.executable(), "-i", "input.toml", "--kokkos-num-threads=1"]
    if ranks > 1:
        command = ["mpirun", "-n", str(ranks), "--oversubscribe"] + command
    subprocess.run(command, cwd=d, check=True, capture_output=True, env=without_launcher())


def test_solvers_run_on_their_communicator_as_the_executable_on_as_many_ranks(group, tmp_path_factory, monkeypatch):
    world, comm = group
    root = world.bcast(str(tmp_path_factory.mktemp("mpi")) if world.Get_rank() == 0 else None)
    exe, py = (os.path.join(root, f"{kind}{comm.Get_size()}") for kind in ("executable", "python"))
    if comm.Get_rank() == 0:
        for d in (exe, py):
            os.makedirs(d)
            with open(os.path.join(d, "input.toml"), "w") as f:
                f.write(test_solver.case_text(8))
        run_executable(exe, comm.Get_size())
    comm.Barrier()
    os.chdir(py)
    solver = mallard.Solver(mallard.Case.from_toml_file("input.toml"))
    steps = []
    solver.on_step(lambda s: steps.append(s.step_count()))
    solver.setup()
    # The case's cells are shared among this communicator's ranks only
    cells = comm.allreduce(solver.field("RHO").shape[0])
    assert cells == comm.allreduce(int(solver.field("GLOBAL_ID").max()), op=MPI.MAX) + 1
    while solver.step():
        pass
    assert solver.finish() == 0
    assert steps == list(range(1, 9))
    if comm.Get_rank() == 0:
        # mallard.diff starts the executable: as a job of its own
        for k in set(os.environ) - set(without_launcher()):
            monkeypatch.delenv(k)
        files = sorted(os.listdir(os.path.join(exe, "solut")))
        assert files and files == sorted(os.listdir(os.path.join(py, "solut")))
        for name in files:
            report = mallard.diff(os.path.join(exe, "solut", name), os.path.join(py, "solut", name))
            assert report["result"] == "ok" and report["time"]["difference"] == 0, (name, report)
            assert report["fields"] and all(f["differing"] == 0 for f in report["fields"]), (name, report)
    world.Barrier()
