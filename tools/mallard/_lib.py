"""Loads libmallard2d/libmallard3d, the C library of Mallard's output reader (src/io/reader_c_api.h)."""
import ctypes
import functools
import os
import shutil
import sys

_SUFFIX = ".dylib" if sys.platform == "darwin" else ".dll" if sys.platform == "win32" else ".so"


class Library:
    def __init__(self, path):
        self.path = path
        lib = ctypes.CDLL(path)
        c = ctypes
        u64p = c.POINTER(c.c_uint64)
        for name, res, args in [
            ("mallard_error", c.c_char_p, []),
            ("mallard_library_version", c.c_char_p, []),
            ("mallard_dim", c.c_int, []),
            ("mallard_open", c.c_void_p, [c.c_char_p]),
            ("mallard_contents", c.c_char_p, [c.c_void_p]),
            ("mallard_close", None, [c.c_void_p]),
            ("mallard_read", c.c_void_p, [c.c_void_p, c.c_char_p, c.c_uint64]),
            ("mallard_read_all", c.c_void_p, [c.c_void_p, c.c_char_p]),
            ("mallard_read_geometry", c.c_void_p, [c.c_void_p, c.c_uint64]),
            ("mallard_run_files", c.c_void_p, [c.c_char_p]),
            ("mallard_array_values", c.POINTER(c.c_double), [c.c_void_p, u64p]),
            ("mallard_array_text", c.c_char_p, [c.c_void_p]),
            ("mallard_array_geometry", None,
             [c.c_void_p, c.POINTER(c.POINTER(c.c_double)), u64p, c.POINTER(u64p), u64p, c.POINTER(u64p), u64p,
              c.POINTER(c.POINTER(c.c_uint8)), u64p]),
            ("mallard_array_free", None, [c.c_void_p]),
        ]:
            f = getattr(lib, name)
            f.restype = res
            f.argtypes = args
        self.lib = lib
        self.dim = lib.mallard_dim()
        self.version = lib.mallard_library_version().decode()

    def check(self, result):
        if not result:
            raise RuntimeError(self.lib.mallard_error().decode())
        return result


def _search_dirs():
    here = os.path.dirname(os.path.abspath(__file__))
    dirs = []
    exe = executable(required=False)
    if exe:
        bin_dir = os.path.dirname(os.path.realpath(exe))
        dirs += [bin_dir, os.path.join(bin_dir, "..", "lib"), os.path.join(bin_dir, "..", "lib64")]
    # An install tree: PREFIX/share/mallard/tools/mallard
    prefix = os.path.join(here, "..", "..", "..", "..")
    dirs += [os.path.join(prefix, "lib"), os.path.join(prefix, "lib64")]
    return [os.path.normpath(d) for d in dirs]


@functools.lru_cache(maxsize=None)
def libraries():
    """The libraries found, 2D first: MALLARD_LIBRARY (paths separated by os.pathsep), else next to the
    mallard executable and in the install tree this package belongs to."""
    if os.environ.get("MALLARD_LIBRARY"):
        paths = os.environ["MALLARD_LIBRARY"].split(os.pathsep)
    else:
        paths = []
        for dim in (2, 3):
            for d in _search_dirs():
                p = os.path.join(d, f"libmallard{dim}d{_SUFFIX}")
                if os.path.exists(p):
                    paths.append(p)
                    break
    libs = sorted((Library(p) for p in paths if p), key=lambda lib: lib.dim)
    if not libs:
        raise RuntimeError("no libmallard2d or libmallard3d found: build Mallard with -DMallard_ENABLE_PYTHON=ON, "
                           "and put its mallard on PATH or set MALLARD_LIBRARY to the library's path")
    return libs


def executable(required=True):
    """The mallard executable: MALLARD_EXECUTABLE, else mallard on PATH."""
    exe = os.environ.get("MALLARD_EXECUTABLE") or shutil.which("mallard")
    if not exe and required:
        raise RuntimeError("no mallard executable: put it on PATH or set MALLARD_EXECUTABLE")
    return exe
