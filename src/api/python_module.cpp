/**
 * @file python_module.cpp
 * @brief mallard._core2d / mallard._core3d: the C++ library API (include/mallard/mallard.h) as a Python extension
 *        module of the mallard package, which wraps it in tools/mallard/_solver.py. Fields become NumPy arrays
 *        or DLPack tensors of the solver's memory, without copies.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

#include "mallard/mallard.h"

namespace nb = nanobind;
using namespace nb::literals;

#define MALLARD_STRINGIFY_(x) #x
#define MALLARD_STRINGIFY(x) MALLARD_STRINGIFY_(x)
#define MALLARD_MODULE_NAME(dim) _core##dim##d
#define MALLARD_MODULE(dim) MALLARD_MODULE_NAME(dim)

namespace {

/** @brief A mallard::Solver that Python can close before Kokkos is finalized, and the error a callback raised. */
struct PySolver {
    std::unique_ptr<mallard::Solver> solver;
    nb::object step_hook, stage_hook;
    mallard::CallbackId step_id = 0, stage_id = 0;
    std::unique_ptr<nb::python_error> callback_error;

    mallard::Solver & get() {
        if (!solver) throw std::runtime_error("mallard.Solver: the solver is closed");
        return *solver;
    }

    /** @brief f() without the GIL; a Python exception a callback raised during it is raised again as it was. */
    template <typename F>
    auto released(F && f) {
        try {
            nb::gil_scoped_release release;
            return f();
        } catch (const mallard::Error &) {
            rethrow_callback_error();
            throw;
        } catch (const std::exception &) {
            rethrow_callback_error();
            throw;
        }
    }

    void rethrow_callback_error() {
        if (!callback_error) return;
        std::unique_ptr<nb::python_error> e = std::move(callback_error);
        e->restore();
        throw nb::python_error();
    }

    /** @brief Calls hook(args...) with the GIL; a Python exception is kept for released() and ends the run. */
    template <typename... Args>
    void call(const nb::object & hook, Args &&... args) {
        nb::gil_scoped_acquire acquire;
        try {
            hook(std::forward<Args>(args)...);
        } catch (nb::python_error & e) {
            const std::string what = e.what();
            callback_error = std::make_unique<nb::python_error>(std::move(e));
            throw mallard::Error("E-RUN", "a Python callback raised: " + what);
        }
    }
};

nb::dlpack::dtype dtype_of(const mallard::ScalarType type) {
    switch (type) {
        case mallard::ScalarType::float32: return nb::dtype<float>();
        case mallard::ScalarType::float64: return nb::dtype<double>();
        case mallard::ScalarType::uint64: return nb::dtype<uint64_t>();
    }
    return nb::dtype<double>();
}

int device_type_of(const mallard::Device device) {
    switch (device) {
        case mallard::Device::host: return nb::device::cpu::value;
        case mallard::Device::cuda: return nb::device::cuda::value;
        case mallard::Device::hip: return nb::device::rocm::value;
        case mallard::Device::sycl: return nb::device::oneapi::value;
    }
    return nb::device::cpu::value;
}

/**
 * @brief The array of a Field: a NumPy array of host memory, else a DLPack tensor (nanobind's ndarray, which
 *        _solver.py wraps), owned by `owner`; a copy if `copy` (a host copy the solver may overwrite).
 */
nb::object array_of(const mallard::Field & f, nb::handle owner, bool own_copy) {
    std::vector<size_t> shape(f.shape.begin(), f.shape.end());
    const size_t ndim = shape.size();
    const nb::dlpack::dtype dtype = dtype_of(f.type);
    const int device = device_type_of(f.device);
    nb::rv_policy policy = nb::rv_policy::reference;
    if (own_copy) policy = nb::rv_policy::copy;
    if (f.device == mallard::Device::host) {
        if (f.writable) {
            return nb::cast(nb::ndarray<nb::numpy>(f.data, ndim, shape.data(), owner, f.strides.data(), dtype, device),
                            policy);
        }
        return nb::cast(nb::ndarray<nb::numpy, nb::ro>(f.data, ndim, shape.data(), owner, f.strides.data(), dtype, device),
                        policy);
    }
    if (f.writable) {
        return nb::cast(nb::ndarray<>(f.data, ndim, shape.data(), owner, f.strides.data(), dtype, device, f.device_id),
                        nb::rv_policy::reference);
    }
    return nb::cast(nb::ndarray<nb::ro>(f.data, ndim, shape.data(), owner, f.strides.data(), dtype, device, f.device_id),
                    nb::rv_policy::reference);
}

/** @brief A Field's description, for __cuda_array_interface__ and the DeviceArray's repr. */
nb::dict describe(const mallard::Field & f) {
    nb::dict d;
    d["name"] = f.name;
    d["unit"] = f.unit;
    d["data"] = reinterpret_cast<uintptr_t>(f.data);
    d["type"] = f.type == mallard::ScalarType::float32 ? "f4" : f.type == mallard::ScalarType::float64 ? "f8" : "u8";
    d["device"] = f.device == mallard::Device::host   ? "host"
                  : f.device == mallard::Device::cuda ? "cuda"
                  : f.device == mallard::Device::hip  ? "hip"
                                                      : "sycl";
    d["device_id"] = f.device_id;
    d["shape"] = f.shape;
    d["strides"] = f.strides;
    d["writable"] = f.writable;
    return d;
}

mallard::Memory memory_of(const std::string & name) {
    if (name == "solver") return mallard::Memory::solver;
    if (name == "host") return mallard::Memory::host;
    throw std::invalid_argument("memory must be \"solver\" or \"host\", not \"" + name + "\"");
}

nb::object field(nb::handle self, PySolver & s, const std::string & name, const std::string & memory, bool writable) {
    const mallard::Memory m = memory_of(memory);
    const mallard::Field f = writable ? s.get().writable_field(name, m) : s.get().field(name, m);
    // A host copy of device memory is the solver's buffer for the next copy: Python gets its own
    const bool copy = !writable && m == mallard::Memory::host && s.get().field(name).device != mallard::Device::host;
    nb::object array = array_of(f, self, copy);
    return nb::make_tuple(array, describe(f));
}

std::unique_ptr<mallard::Session> session;

} // namespace

NB_MODULE(MALLARD_MODULE(Mallard_DIM), m) {
    m.doc() = "Mallard's C++ library API for the " MALLARD_STRINGIFY(Mallard_DIM) "D solver; use it through mallard.Solver";

    nb::object error = nb::module_::import_("mallard._solver").attr("Error");
    nb::register_exception_translator(
        [](const std::exception_ptr & p, void * payload) {
            try {
                std::rethrow_exception(p);
            } catch (const mallard::Error & e) {
                nb::handle type(static_cast<PyObject *>(payload));
                nb::object value = type(e.what(), e.code());
                PyErr_SetObject(type.ptr(), value.ptr());
            }
        },
        error.release().ptr());

    m.def("api_version", &mallard::api_version);
    m.def("version", &mallard::version);
    m.def("dimension", &mallard::dimension);

    m.def("start_session", [](const std::vector<std::string> & args, uintptr_t mpi_comm) {
        if (session) throw std::runtime_error("mallard: a session is already started in this process");
        std::vector<std::string> storage = {"mallard"};
        storage.insert(storage.end(), args.begin(), args.end());
        std::vector<char *> argv;
        for (std::string & a : storage) argv.push_back(a.data());
        argv.push_back(nullptr);
        int argc = static_cast<int>(storage.size());
        char ** p = argv.data();
        session = std::make_unique<mallard::Session>(argc, p, reinterpret_cast<const void *>(mpi_comm));
    });
    m.def("end_session", [] { session.reset(); });
    m.def("has_session", [] { return session != nullptr; });

    m.def("case_dimension", [](const std::string & text, const std::string & name) {
        return mallard::Case::from_toml(text, name).dimension();
    });
    m.def("check_case", [](const std::string & text, const std::string & name) { mallard::Case::from_toml(text, name); });

    nb::class_<mallard::Monitor>(m, "Monitor")
        .def_ro("name", &mallard::Monitor::name)
        .def_ro("unit", &mallard::Monitor::unit)
        .def_ro("value", &mallard::Monitor::value)
        .def_ro("step", &mallard::Monitor::step)
        .def("__repr__", [](const mallard::Monitor & x) {
            return "Monitor(name='" + x.name + "', value=" + std::to_string(x.value) + ", unit='" + x.unit + "')";
        });

    nb::class_<mallard::Stage>(m, "Stage")
        .def_ro("step", &mallard::Stage::step)
        .def_ro("index", &mallard::Stage::index)
        .def_ro("time", &mallard::Stage::time)
        .def("__repr__", [](const mallard::Stage & x) {
            return "Stage(step=" + std::to_string(x.step) + ", index=" + std::to_string(x.index) +
                   ", time=" + std::to_string(x.time) + ")";
        });

    nb::class_<mallard::Statistics>(m, "Statistics")
        .def_ro("samples", &mallard::Statistics::samples)
        .def_ro("start", &mallard::Statistics::start)
        .def_ro("end", &mallard::Statistics::end)
        .def_ro("batches", &mallard::Statistics::batches);

    nb::class_<PySolver>(m, "Solver")
        .def("__init__",
             [](PySolver * self, const std::string & text, const std::string & path, const std::string & name) {
                 new (self) PySolver();
                 self->solver = std::make_unique<mallard::Solver>(path.empty() ? mallard::Case::from_toml(text, name)
                                                                               : mallard::Case::from_toml_file(path));
             },
             "text"_a, "path"_a, "name"_a)
        .def("close", [](PySolver & s) { s.solver.reset(); })
        .def("closed", [](PySolver & s) { return s.solver == nullptr; })
        .def("setup", [](PySolver & s) { s.released([&] { s.get().setup(); }); })
        .def("step", [](PySolver & s) { return s.released([&] { return s.get().step(); }); })
        .def("advance_to", [](PySolver & s, double t) { return s.released([&] { return s.get().advance_to(t); }); })
        .def("finish", [](PySolver & s) { return s.released([&] { return s.get().finish(); }); })
        .def("run", [](PySolver & s) { return s.released([&] { return s.get().run(); }); })
        .def("finished", [](PySolver & s) { return s.get().finished(); })
        .def("time", [](PySolver & s) { return s.get().time(); })
        .def("step_count", [](PySolver & s) { return s.get().step_count(); })
        .def("monitors", [](PySolver & s) { return s.released([&] { return s.get().monitors(); }); })
        .def("monitor", [](PySolver & s, const std::string & name) { return s.released([&] { return s.get().monitor(name); }); })
        .def("field_names", [](PySolver & s) { return s.get().field_names(); })
        .def("field", [](nb::handle self, const std::string & name, const std::string & memory) {
            return field(self, nb::cast<PySolver &>(self), name, memory, false);
        })
        .def("writable_field", [](nb::handle self, const std::string & name, const std::string & memory) {
            return field(self, nb::cast<PySolver &>(self), name, memory, true);
        })
        .def("commit_state", [](PySolver & s) { s.released([&] { s.get().commit_state(); }); })
        .def("statistics", [](PySolver & s) { return s.get().statistics(); })
        .def("set_step_hook",
             [](PySolver & s, nb::object hook) {
                 if (s.step_id) s.get().remove_callback(s.step_id);
                 s.step_id = 0;
                 s.step_hook = hook;
                 if (hook.is_none()) return;
                 s.step_id = s.get().on_step([&s](mallard::Solver &) { s.call(s.step_hook); });
             },
             nb::arg("hook").none())
        .def("set_stage_hook", [](PySolver & s, nb::object hook) {
            if (s.stage_id) s.get().remove_callback(s.stage_id);
            s.stage_id = 0;
            s.stage_hook = hook;
            if (hook.is_none()) return;
            s.stage_id =
                s.get().on_stage([&s](mallard::Solver &, const mallard::Stage & stage) { s.call(s.stage_hook, stage); });
        }, nb::arg("hook").none());
}
