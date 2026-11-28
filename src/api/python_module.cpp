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
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
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
    nb::object step_hook, stage_hook, progress_hook, output_hook, record_hook, log_hook;
    mallard::CallbackId step_id = 0, stage_id = 0, progress_id = 0, output_id = 0, record_id = 0;
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

/** @brief Replaces the callback `id` of hook `held` with `hook` (None: none), registered by `on`. */
template <typename On>
void set_hook(PySolver & s, nb::object & held, mallard::CallbackId & id, nb::object hook, On && on) {
    if (id) s.get().remove_callback(id);
    id = 0;
    held = hook;
    if (!hook.is_none()) id = on();
}

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
    m.def("case_set", [](const std::string & text, const std::string & name, const std::string & key,
                         const std::string & value) {
        return mallard::Case::from_toml(text, name).set(key, mallard::Value::toml(value)).toml();
    });

    m.def("case_check",
          [](const std::string & text, const std::string & path, const std::string & name, int ranks, double throughput,
             uint64_t steps, const std::string & memory, nb::object log) {
              const mallard::Case c = path.empty() ? mallard::Case::from_toml(text, name) : mallard::Case::from_toml_file(path);
              mallard::CheckOptions options;
              options.ranks = ranks;
              options.throughput = throughput;
              options.steps = steps;
              options.memory = memory;
              std::unique_ptr<nb::python_error> raised;
              if (!log.is_none()) {
                  options.log = [&log, &raised](const mallard::LogMessage & message) {
                      nb::gil_scoped_acquire acquire;
                      if (raised) return;
                      try {
                          log(message);
                      } catch (nb::python_error & e) {
                          raised = std::make_unique<nb::python_error>(std::move(e));
                      }
                  };
              }
              mallard::CheckReport report;
              {
                  nb::gil_scoped_release release;
                  report = c.check(options);
              }
              if (raised) {
                  raised->restore();
                  throw nb::python_error();
              }
              return report;
          },
          "text"_a, "path"_a, "name"_a, "ranks"_a, "throughput"_a, "steps"_a, "memory"_a, "log"_a.none());

    nb::class_<mallard::CheckDiagnostic>(m, "CheckDiagnostic")
        .def_prop_ro("level",
                     [](const mallard::CheckDiagnostic & x) {
                         return x.level == mallard::LogLevel::warning ? "warning" : "error";
                     })
        .def_ro("code", &mallard::CheckDiagnostic::code)
        .def_ro("message", &mallard::CheckDiagnostic::message)
        .def_ro("docs", &mallard::CheckDiagnostic::docs)
        .def("__repr__", [](const mallard::CheckDiagnostic & x) {
            return "CheckDiagnostic(code='" + x.code + "', message='" + x.message + "')";
        });

    nb::class_<mallard::CheckUnusedKey>(m, "CheckUnusedKey")
        .def_ro("key", &mallard::CheckUnusedKey::key)
        .def_ro("line", &mallard::CheckUnusedKey::line)
        .def_ro("suggestion", &mallard::CheckUnusedKey::suggestion)
        .def("__repr__", [](const mallard::CheckUnusedKey & x) {
            return "CheckUnusedKey(key='" + x.key + "', line=" + std::to_string(x.line) + ", suggestion='" +
                   x.suggestion + "')";
        });

    nb::class_<mallard::CheckBoundary>(m, "CheckBoundary")
        .def_ro("zone", &mallard::CheckBoundary::zone)
        .def_ro("condition", &mallard::CheckBoundary::condition)
        .def_ro("faces", &mallard::CheckBoundary::faces)
        .def_ro("details", &mallard::CheckBoundary::details)
        .def_ro("turbulence", &mallard::CheckBoundary::turbulence)
        .def("__repr__", [](const mallard::CheckBoundary & x) {
            return "CheckBoundary(zone='" + x.zone + "', condition='" + x.condition + "')";
        });

    nb::class_<mallard::CheckEstimate>(m, "CheckEstimate")
        .def_ro("id", &mallard::CheckEstimate::id)
        .def_ro("quantity", &mallard::CheckEstimate::quantity)
        .def_ro("value", &mallard::CheckEstimate::value)
        .def_ro("unit", &mallard::CheckEstimate::unit)
        .def_ro("target", &mallard::CheckEstimate::target)
        .def_ro("status", &mallard::CheckEstimate::status)
        .def_ro("basis", &mallard::CheckEstimate::basis)
        .def_ro("region", &mallard::CheckEstimate::region)
        .def("__repr__", [](const mallard::CheckEstimate & x) {
            return "CheckEstimate(id='" + x.id + "', value=" + std::to_string(x.value) + ")";
        });

    nb::class_<mallard::CheckTimeStep>(m, "CheckTimeStep")
        .def_ro("t", &mallard::CheckTimeStep::t)
        .def_ro("dt", &mallard::CheckTimeStep::dt)
        .def_ro("dt_cfl1", &mallard::CheckTimeStep::dt_cfl1)
        .def_ro("cfl", &mallard::CheckTimeStep::cfl)
        .def_ro("cfl_limit", &mallard::CheckTimeStep::cfl_limit)
        .def_ro("integrator", &mallard::CheckTimeStep::integrator)
        .def_ro("set_by", &mallard::CheckTimeStep::set_by);

    nb::class_<mallard::CheckMemory>(m, "CheckMemory")
        .def_ro("device_memory", &mallard::CheckMemory::device_memory)
        .def_ro("ranks", &mallard::CheckMemory::ranks)
        .def_ro("bytes", &mallard::CheckMemory::bytes)
        .def_ro("bytes_peak", &mallard::CheckMemory::bytes_peak)
        .def_ro("target_ranks", &mallard::CheckMemory::target_ranks)
        .def_ro("target_bytes_peak", &mallard::CheckMemory::target_bytes_peak)
        .def_ro("device_capacity", &mallard::CheckMemory::device_capacity);

    nb::class_<mallard::CheckCost>(m, "CheckCost")
        .def_ro("cells", &mallard::CheckCost::cells)
        .def_ro("ranks", &mallard::CheckCost::ranks)
        .def_ro("throughput", &mallard::CheckCost::throughput)
        .def_ro("throughput_measured", &mallard::CheckCost::throughput_measured)
        .def_ro("wall_per_time", &mallard::CheckCost::wall_per_time)
        .def_ro("flow_through_time", &mallard::CheckCost::flow_through_time)
        .def_ro("t_stop", &mallard::CheckCost::t_stop)
        .def_ro("n_steps", &mallard::CheckCost::n_steps);

    nb::class_<mallard::InputKey>(m, "InputKey")
        .def_ro("path", &mallard::InputKey::path)
        .def_ro("catalog_path", &mallard::InputKey::catalog_path)
        .def_ro("value", &mallard::InputKey::value)
        .def_ro("set", &mallard::InputKey::set)
        .def_ro("type", &mallard::InputKey::type)
        .def_ro("dimension", &mallard::InputKey::dimension)
        .def_ro("unit", &mallard::InputKey::unit)
        .def_ro("required", &mallard::InputKey::required)
        .def_ro("default", &mallard::InputKey::default_value)
        .def_ro("values", &mallard::InputKey::values)
        .def("__repr__", [](const mallard::InputKey & x) {
            return "InputKey(path='" + x.path + "', value='" + x.value + "', type='" + x.type + "', set=" +
                   (x.set ? "True" : "False") + ")";
        });

    nb::class_<mallard::CheckReport>(m, "CheckReport")
        .def_ro("status", &mallard::CheckReport::status)
        .def_ro("ok", &mallard::CheckReport::ok)
        .def_ro("setup_done", &mallard::CheckReport::setup_done)
        .def_ro("diagnostics", &mallard::CheckReport::diagnostics)
        .def_ro("unused_keys", &mallard::CheckReport::unused_keys)
        .def_ro("boundaries", &mallard::CheckReport::boundaries)
        .def_ro("time_step", &mallard::CheckReport::time_step)
        .def_ro("resolution", &mallard::CheckReport::resolution)
        .def_ro("memory", &mallard::CheckReport::memory)
        .def_ro("cost", &mallard::CheckReport::cost)
        .def_ro("resolved", &mallard::CheckReport::resolved)
        .def_ro("keys", &mallard::CheckReport::keys)
        .def_ro("text", &mallard::CheckReport::text)
        .def_ro("json", &mallard::CheckReport::json)
        .def("__repr__", [](const mallard::CheckReport & x) {
            return "CheckReport(status='" + x.status + "', diagnostics=" + std::to_string(x.diagnostics.size()) + ")";
        });

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

    nb::class_<mallard::Progress>(m, "Progress")
        .def_ro("step", &mallard::Progress::step)
        .def_ro("time", &mallard::Progress::time)
        .def_ro("dt", &mallard::Progress::dt)
        .def_ro("cfl", &mallard::Progress::cfl)
        .def_ro("fraction", &mallard::Progress::fraction)
        .def_ro("wall", &mallard::Progress::wall)
        .def_ro("wall_per_step", &mallard::Progress::wall_per_step)
        .def_ro("cells_per_second", &mallard::Progress::cells_per_second)
        .def_ro("eta", &mallard::Progress::eta)
        .def_ro("cells", &mallard::Progress::cells)
        .def_ro("ranks", &mallard::Progress::ranks)
        .def_ro("diagnostics", &mallard::Progress::diagnostics)
        .def_ro("written", &mallard::Progress::written)
        .def_ro("json", &mallard::Progress::json)
        .def("__repr__", [](const mallard::Progress & x) {
            return "Progress(step=" + std::to_string(x.step) + ", time=" + std::to_string(x.time) +
                   ", fraction=" + std::to_string(x.fraction) + ")";
        });

    nb::class_<mallard::OutputFile>(m, "OutputFile")
        .def_ro("path", &mallard::OutputFile::path)
        .def_ro("bytes", &mallard::OutputFile::bytes)
        .def("__repr__", [](const mallard::OutputFile & x) {
            return "OutputFile(path='" + x.path + "', bytes=" + std::to_string(x.bytes) + ")";
        });

    nb::class_<mallard::Output>(m, "Output")
        .def_ro("step", &mallard::Output::step)
        .def_ro("time", &mallard::Output::time)
        .def_ro("kind", &mallard::Output::kind)
        .def_ro("path", &mallard::Output::path)
        .def_ro("reason", &mallard::Output::reason)
        .def_ro("writer", &mallard::Output::writer)
        .def_ro("files", &mallard::Output::files)
        .def_ro("json", &mallard::Output::json)
        .def("__repr__", [](const mallard::Output & x) {
            return "Output(kind='" + x.kind + "', path='" + x.path + "', step=" + std::to_string(x.step) + ")";
        });

    nb::class_<mallard::Record>(m, "Record")
        .def_ro("type", &mallard::Record::type)
        .def_ro("json", &mallard::Record::json)
        .def("__repr__", [](const mallard::Record & x) { return "Record(type='" + x.type + "')"; });

    nb::class_<mallard::LogMessage>(m, "LogMessage")
        .def_prop_ro("level",
                     [](const mallard::LogMessage & x) {
                         return x.level == mallard::LogLevel::info      ? "info"
                                : x.level == mallard::LogLevel::warning ? "warning"
                                                                        : "error";
                     })
        .def_ro("code", &mallard::LogMessage::code)
        .def_ro("text", &mallard::LogMessage::text)
        .def("__repr__", [](const mallard::LogMessage & x) { return "LogMessage(text='" + x.text + "')"; });

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
        .def("step", [](PySolver & s, uint64_t n) { return s.released([&] { return s.get().step(n); }); }, "n"_a = 1)
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
        }, nb::arg("hook").none())
        .def("set_progress_hook", [](PySolver & s, nb::object hook) {
            set_hook(s, s.progress_hook, s.progress_id, hook, [&] {
                return s.get().on_progress([&s](const mallard::Progress & p) { s.call(s.progress_hook, p); });
            });
        }, nb::arg("hook").none())
        .def("set_output_hook", [](PySolver & s, nb::object hook) {
            set_hook(s, s.output_hook, s.output_id, hook, [&] {
                return s.get().on_output([&s](const mallard::Output & o) { s.call(s.output_hook, o); });
            });
        }, nb::arg("hook").none())
        .def("set_record_hook", [](PySolver & s, nb::object hook) {
            set_hook(s, s.record_hook, s.record_id, hook, [&] {
                return s.get().on_record([&s](const mallard::Record & r) { s.call(s.record_hook, r); });
            });
        }, nb::arg("hook").none())
        .def("set_log", [](PySolver & s, nb::object hook) {
            s.log_hook = hook;
            if (hook.is_none()) {
                s.get().set_log({});
                return;
            }
            s.get().set_log([&s](const mallard::LogMessage & message) { s.call(s.log_hook, message); });
        }, nb::arg("sink").none());
}
