"""Mallard's C++ library API from Python: mallard.Case, mallard.Session and mallard.Solver.

The solver runs in this process through the extension module mallard._core2d or mallard._core3d (a build with
-DMallard_ENABLE_PYTHON_BINDINGS=ON), loaded only when a case or solver needs it, so the rest of the package works
without it. Fields are the solver's own memory: NumPy arrays on the host, DeviceArray (DLPack and
__cuda_array_interface__) on a GPU. Guide: docs/guide/python.md#running-cases.
"""
import atexit
import importlib
import importlib.machinery
import importlib.util
import os
import sys
import weakref

from . import _lib


class Error(RuntimeError):
    """A failure of the solver: str(e) is the message mallard prints, e.code its diagnostic code ("E-INPUT", ...)."""

    def __init__(self, message, code=""):
        super().__init__(message)
        self.code = code


_modules = {}


def _extension(dim):
    """The extension module of dimension dim: in this package, else next to the library the reader finds."""
    if dim in _modules:
        return _modules[dim]
    name = f"mallard._core{dim}d"
    try:
        module = importlib.import_module(name)
    except ImportError:
        module = None
        dirs = [os.path.dirname(lib.path) for lib in _libraries() if lib.dim == dim]
        for d in dirs:
            for suffix in importlib.machinery.EXTENSION_SUFFIXES:
                path = os.path.join(d, f"_core{dim}d{suffix}")
                if os.path.exists(path):
                    spec = importlib.util.spec_from_file_location(name, path)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[name] = module
                    spec.loader.exec_module(module)
                    break
            if module:
                break
        if module is None:
            raise ImportError(f"no {dim}D solver bindings (mallard._core{dim}d): build Mallard with "
                              "-DMallard_ENABLE_PYTHON_BINDINGS=ON (docs/guide/python.md#running-cases)") from None
    _modules[dim] = module
    return module


def _libraries():
    try:
        return _lib.libraries()
    except RuntimeError:
        return []


def _any_extension():
    for dim in (2, 3):
        try:
            return _extension(dim)
        except ImportError:
            pass
    raise ImportError("no solver bindings (mallard._core2d or mallard._core3d): build Mallard with "
                      "-DMallard_ENABLE_PYTHON_BINDINGS=ON (docs/guide/python.md#running-cases)")


class Case:
    """A case: the TOML input of a run, which the run records, so that mallard -i on it reruns it bit for bit."""

    def __init__(self, text, path="", name="input.toml"):
        self._text, self._path, self._name = text, os.fspath(path), name
        _any_extension().check_case(text, self._path or name)

    @classmethod
    def from_toml_file(cls, path):
        """The case of an input file, read now; paths inside it are relative to the working directory."""
        path = os.fspath(path)
        with open(path, "rb") as f:
            text = f.read().decode("utf-8")
        return cls(text, path, path)

    @classmethod
    def from_toml(cls, text, name="input.toml"):
        """The case of a TOML text; name is what error messages call it."""
        return cls(text, "", name)

    def toml(self):
        """The TOML text, exactly as read: write it to a file to rerun the case with mallard -i."""
        return self._text

    def path(self):
        """The input file of from_toml_file, else ""."""
        return self._path

    def dimension(self):
        """2 or 3, as mallard tells which solver runs the case."""
        return _any_extension().case_dimension(self._text, self._path or self._name)

    def __repr__(self):
        return f"Case({self._path!r})" if self._path else f"Case.from_toml(<{len(self._text)} characters>)"


_session = None
_solvers = weakref.WeakSet()


class Session:
    """MPI and Kokkos for the process, as mallard.Session in C++: Solver starts one with Kokkos' defaults if none
    is started. args are mallard's options (["--kokkos-num-threads=2"]). Kokkos starts once per process, so a
    session lasts until the interpreter exits or end() is called, after which no other can start.

    comm, an mpi4py communicator (uv pip install "mallard-python[mpi]"), runs the solvers on its processes instead
    of MPI_COMM_WORLD: every process of comm starts the session and runs the same cases."""

    def __init__(self, args=(), dim=None, comm=None):
        global _session
        if _session is not None:
            raise RuntimeError("a mallard.Session is already started in this process")
        self.dim = dim or _default_dimension()
        self._module = _extension(self.dim)
        self._module.start_session([str(a) for a in args], _comm_address(comm))
        self.comm = comm
        _session = self

    def end(self):
        """Closes every solver, then finalizes Kokkos (and MPI, if the session started it)."""
        global _session
        for solver in list(_solvers):
            solver.close()
        self._module.end_session()
        if _session is self:
            _session = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.end()


def _comm_address(comm):
    if comm is None:
        return 0
    try:
        from mpi4py import MPI
    except ImportError:
        message = 'a mallard.Session on an MPI communicator needs mpi4py: uv pip install "mallard-python[mpi]"'
        raise ImportError(message) from None
    if not isinstance(comm, MPI.Comm):
        raise TypeError(f"comm must be an mpi4py communicator (mpi4py.MPI.Comm), not {type(comm).__name__}")
    return MPI._addressof(comm)


def _default_dimension():
    dims = [d for d in (2, 3) if _has_extension(d)]
    return dims[0] if dims else 2


def _has_extension(dim):
    try:
        _extension(dim)
        return True
    except ImportError:
        return False


@atexit.register
def _end():
    if _session:
        _session.end()


class DeviceArray:
    """A field in a GPU's memory, without a copy: cupy.asarray(a), torch.from_dlpack(a) and numba take it through
    __cuda_array_interface__ or DLPack; Solver.field(name, memory="host") gives a NumPy copy."""

    def __init__(self, tensor, info):
        self._tensor, self._info = tensor, info
        self.shape = tuple(info["shape"])
        self.dtype = info["type"]
        self.device = info["device"]
        self.name, self.unit = info["name"], info["unit"]

    @property
    def __cuda_array_interface__(self):
        if self._info["device"] not in ("cuda", "hip"):
            raise AttributeError("__cuda_array_interface__ is for CUDA and HIP memory")
        size = int(self.dtype[1:])
        return {"shape": self.shape, "typestr": "<" + self.dtype, "data": (self._info["data"], not self._info["writable"]),
                "strides": tuple(s * size for s in self._info["strides"]), "version": 3, "stream": None}

    def __dlpack__(self, *args, **kwargs):
        return self._tensor.__dlpack__(*args, **kwargs)

    def __dlpack_device__(self):
        return self._tensor.__dlpack_device__()

    def __repr__(self):
        return f"DeviceArray({self.name!r}, shape={self.shape}, dtype={self.dtype}, device={self.device})"


class Solver:
    """Runs a case, as mallard::Solver in C++: setup(), then step() or advance_to(t), then finish(); run() is the
    run of mallard -i, with the same output and results bit for bit. Under MPI every call is collective."""

    def __init__(self, case):
        if not isinstance(case, Case):
            raise TypeError("mallard.Solver takes a mallard.Case (Case.from_toml_file or Case.from_toml)")
        dim = case.dimension()
        if not _session:
            if _session is False:
                raise RuntimeError("the mallard.Session has ended: Kokkos starts once per process")
            Session(dim=dim)
        elif _session.dim != dim:
            raise RuntimeError(f"a {dim}D case in a process whose session runs the {_session.dim}D solver: run it in "
                               "another process")
        self.case = case
        self._core = _session._module.Solver(case.toml(), case.path(), case._name)
        self._step_callbacks, self._stage_callbacks, self._next_id = {}, {}, 1
        self._hooked = (False, False)
        _solvers.add(self)

    def close(self):
        """Frees the solver; arrays of its fields are invalid after it."""
        self._core.close()

    def setup(self):
        """The setup of mallard -i, then the start of the run (its first output and progress row)."""
        self._core.setup()

    def step(self):
        """One step; False once the case's stop condition is reached."""
        return self._core.step()

    def advance_to(self, t):
        """Steps until the time is t, landing on it; False once the case's stop condition is reached."""
        return self._core.advance_to(float(t))

    def finish(self):
        """Ends the run (final output, summary, the run log's end record); the exit status mallard -i would give."""
        return self._core.finish()

    def run(self):
        """setup(), steps to the stop condition, finish(): the run of mallard -i."""
        return self._core.run()

    def finished(self):
        """Whether the case's stop condition is reached."""
        return self._core.finished()

    def time(self):
        """The solution time (s)."""
        return self._core.time()

    def step_count(self):
        """Steps taken since the initial state (a restart continues its count)."""
        return self._core.step_count()

    def monitors(self):
        """The latest value of every [integrals] column, [[forces]] component and [[probes]] value."""
        return self._core.monitors()

    def monitor(self, name):
        """The monitored quantity of this name (name, unit, value, step), or None."""
        return self._core.monitor(name)

    def field_names(self):
        """The names field() takes for this case (after setup)."""
        return self._core.field_names()

    def field(self, name, memory="solver"):
        """A read-only view of a field on this rank's owned cells: a NumPy array of the solver's memory on the host,
        a DeviceArray on a GPU (memory="host" copies it to a NumPy array). Views of the state stay the state's."""
        return self._wrap(*self._core.field(name, memory))

    def writable_field(self, name, memory="solver"):
        """A writable view of a conservative variable, partial density or block, to set the state between steps:
        write it, then commit_state(), or let the next step commit it."""
        return self._wrap(*self._core.writable_field(name, memory))

    def commit_state(self):
        """Takes what was written through writable fields as the state; the run log records it."""
        self._core.commit_state()

    def statistics(self):
        """The [statistics] means' averaging so far: samples, start, end and batches."""
        return self._core.statistics()

    def on_step(self, f):
        """Calls f(solver) after every step; returns an id for remove_callback."""
        return self._register(self._step_callbacks, f)

    def on_stage(self, f):
        """Calls f(solver, stage) at each stage of the time integrator; field() views the stage's state there.
        Steps run kernel by kernel on the host while one is registered."""
        return self._register(self._stage_callbacks, f)

    def remove_callback(self, callback_id):
        """Removes a callback; unknown ids are ignored."""
        self._step_callbacks.pop(callback_id, None)
        self._stage_callbacks.pop(callback_id, None)
        self._install()

    def _register(self, callbacks, f):
        if not callable(f):
            raise TypeError("a callback must be callable")
        callback_id = self._next_id
        self._next_id += 1
        callbacks[callback_id] = f
        self._install()
        return callback_id

    def _install(self):
        # Through a weak reference: callbacks that hold the solver leave no cycle through the extension
        me = weakref.ref(self)

        def steps():
            s = me()
            for f in list(s._step_callbacks.values()):
                f(s)

        def stages(stage):
            s = me()
            for f in list(s._stage_callbacks.values()):
                f(s, stage)

        hooked = (bool(self._step_callbacks), bool(self._stage_callbacks))
        if hooked[0] != self._hooked[0]:
            self._core.set_step_hook(steps if hooked[0] else None)
        if hooked[1] != self._hooked[1]:
            self._core.set_stage_hook(stages if hooked[1] else None)
        self._hooked = hooked

    @staticmethod
    def _wrap(array, info):
        return array if info["device"] == "host" else DeviceArray(array, info)

    def __repr__(self):
        return f"Solver({self.case!r}, step={self._core.step_count()})" if not self._core.closed() else "Solver(closed)"
