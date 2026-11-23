/**
 * @file mallard_api.cpp
 * @brief The C++ library API (include/mallard/mallard.h) over Solver, as run_main.cpp drives it.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "mallard/mallard.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <new>
#include <random>

#include <Kokkos_Core.hpp>

#ifdef Mallard_HAS_MPI
#include <mpi.h>
#endif

#include "common_io.h"
#include "diagnostic_code.h"
#include "field_units.h"
#include "front_end.h"
#include "input.h"
#include "input_upgrade.h"
#include "json.h"
#include "log.h"
#include "parallel/binding.h"
#include "parallel/comm.h"
#include "run_log.h"
#include "run_status.h"
#include "solver/check.h"
#include "solver/solver.h"
#include "common/units.h"
#include "version.h"

namespace mallard {

namespace {

/** @brief The message the executable prints for e, without toml11's "[error] " prefix and trailing newlines. */
std::string error_text(const std::exception & e) {
    std::string text = e.what();
    if (text.rfind("[error] ", 0) == 0) text.erase(0, 8);
    while (!text.empty() && text.back() == '\n') text.pop_back();
    if (dynamic_cast<const std::bad_alloc *>(&e) || text.find("failed to allocate memory") != std::string::npos) {
        text += "\nThe run is out of memory. mallard check input.toml --ranks N estimates the memory per rank: "
                "run on more ranks or GPUs, or on a coarser mesh. See docs/guide/input_files.md#checking-an-input.";
    }
    return text;
}

/** @brief f(), with Mallard's exceptions as Error; input errors name the input as the executable does. */
template <typename F>
auto translating(const std::string & input_name, const char * code, F && f) {
    try {
        return f();
    } catch (const Error &) {
        throw;
    } catch (const InputError & e) {
        throw Error(e.code, input_name + ": " + error_text(e));
    } catch (const diagnostic_code::Error & e) {
        throw Error(e.code, error_text(e));
    } catch (const std::exception & e) {
        throw Error(code, error_text(e));
    }
}

/** @brief Rank 0's values on every rank, bit for bit. */
std::vector<double> broadcast(const std::vector<double> & values) {
    std::string bytes(values.size() * sizeof(double), '\0');
    if (!values.empty()) std::memcpy(bytes.data(), values.data(), bytes.size());
    bytes = comm::broadcast(bytes);
    std::vector<double> out(bytes.size() / sizeof(double));
    if (!out.empty()) std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
}

using DeviceMemory = Kokkos::DefaultExecutionSpace::memory_space;
constexpr bool DEVICE_IS_HOST = Kokkos::SpaceAccessibility<Kokkos::HostSpace, DeviceMemory>::accessible;

constexpr ScalarType SCALAR = sizeof(rtype) == 4 ? ScalarType::float32 : ScalarType::float64;

Device solver_device() {
    if constexpr (DEVICE_IS_HOST) return Device::host;
#if defined(KOKKOS_ENABLE_CUDA)
    return Device::cuda;
#elif defined(KOKKOS_ENABLE_HIP)
    return Device::hip;
#elif defined(KOKKOS_ENABLE_SYCL)
    return Device::sycl;
#else
    return Device::host;
#endif
}

int solver_device_id() {
#if defined(KOKKOS_ENABLE_CUDA)
    return Kokkos::DefaultExecutionSpace().cuda_device();
#elif defined(KOKKOS_ENABLE_HIP)
    return Kokkos::DefaultExecutionSpace().hip_device();
#else
    return 0;
#endif
}

/** @brief The UDUNITS units of a field, as the output files give them. */
std::string field_unit(const std::string & name) {
    if (units::system() != units::System::SI) return "1";
    if (name == "CENTROID") return "m";
    if (name == "VOLUME") return N_DIM == 2 ? "m2" : "m3";
    if (name == "GLOBAL_ID") return "1";
    const std::optional<FieldDescription> d = field_units::find(name);
    return d ? d->units() : std::string();
}

/** @brief A view's first n rows (all columns), in memory `device`. */
template <typename V>
Field view_field(const std::string & name, const V & v, uint32_t n, Device device, bool writable) {
    Field f;
    f.name = name;
    f.data = const_cast<void *>(static_cast<const void *>(v.data()));
    f.type = SCALAR;
    f.device = device;
    f.device_id = device == Device::host ? 0 : solver_device_id();
    f.shape.push_back(n);
    f.strides.push_back(static_cast<int64_t>(v.stride(0)));
    if constexpr (V::rank() == 2) {
        f.shape.push_back(static_cast<int64_t>(v.extent(1)));
        f.strides.push_back(static_cast<int64_t>(v.stride(1)));
    }
    f.writable = writable;
    return f;
}

/** @brief Column j of a block's field: its first element, one dimension. */
Field column(Field block, const std::string & name, int64_t j) {
    block.name = name;
    block.unit = field_unit(name);
    if (block.data) {
        block.data = static_cast<char *>(block.data) + j * block.strides[1] * static_cast<int64_t>(sizeof(rtype));
    }
    block.shape.resize(1);
    block.strides.resize(1);
    return block;
}

} // namespace

Error::~Error() = default;

int api_version() { return MALLARD_API_VERSION; }

std::string version() { return MALLARD_VERSION; }

int dimension() { return N_DIM; }

struct Session::Impl {
    std::unique_ptr<comm::Session> mpi;
    bool owns_kokkos = false;
};

namespace {
char program_name[] = "mallard";
char * no_args[] = {program_name, nullptr};
int no_argc = 1;
char ** no_argv = no_args;
} // namespace

Session::Session() : Session(no_argc, no_argv) {}

Session::Session(int & argc, char **& argv) : Session(argc, argv, nullptr) {}

Session::Session(int & argc, char **& argv, const void * mpi_comm) : impl(std::make_unique<Impl>()) {
    if (Kokkos::is_finalized()) {
        throw Error(diagnostic_code::SETUP, "Kokkos was finalized in this process and cannot initialize again: keep "
                                            "one mallard::Session for the whole program.");
    }
#ifdef Mallard_HAS_MPI
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    // MPI before Kokkos, so Kokkos can map each rank to its own device
    if (mpi_comm) {
        if (!mpi_initialized) {
            throw Error(diagnostic_code::SETUP, "mallard::Session on an MPI communicator: initialize MPI first");
        }
        impl->mpi = std::make_unique<comm::Session>(argc, argv, *static_cast<const MPI_Comm *>(mpi_comm));
    } else if (!mpi_initialized) {
        impl->mpi = std::make_unique<comm::Session>(argc, argv);
    }
#else
    if (mpi_comm) {
        throw Error(diagnostic_code::SETUP,
                    "mallard::Session on an MPI communicator: this library is built without MPI "
                    "(-DMallard_ENABLE_MPI=ON)");
    }
#endif
    if (!Kokkos::is_initialized()) {
        binding::select_device(argc, argv);
        Kokkos::initialize(argc, argv);
        impl->owns_kokkos = true;
    }
}

Session::~Session() {
    if (!impl) return;
    if (impl->owns_kokkos) Kokkos::finalize();
    impl->mpi.reset();
}

Value::Value(const bool b) : text_(b ? "true" : "false") {}
Value::Value(const int i) : text_(std::to_string(i)) {}
Value::Value(const long i) : text_(std::to_string(i)) {}
Value::Value(const long long i) : text_(std::to_string(i)) {}
Value::Value(const unsigned i) : text_(std::to_string(i)) {}
Value::Value(const unsigned long i) : Value(static_cast<unsigned long long>(i)) {}
Value::Value(const unsigned long long i) : text_(std::to_string(i)) {
    if (i > static_cast<unsigned long long>(std::numeric_limits<int64_t>::max())) {
        throw Error(diagnostic_code::INPUT, "mallard::Value: " + text_ + " is past TOML's largest integer, 2^63 - 1");
    }
}
Value::Value(const double x) : text_(input_detail::float_text(x)) {}
Value::Value(const char * s) : Value(std::string(s)) {}
Value::Value(const std::string & s) : text_(input_detail::string_text(s)) {}
Value::Value(const std::string_view s) : Value(std::string(s)) {}
Value::Value(const std::initializer_list<Value> items) : Value(array(std::vector<Value>(items))) {}

Value::Value(const Quantity & q)
    : text_(q.unit().empty() ? input_detail::float_text(q.value())
                             : input_detail::string_text(input_detail::float_text(q.value()) + " " + q.unit())) {}

Value Value::toml(std::string text) { return Value(Text{}, std::move(text)); }

Value Value::array(const std::vector<Value> & items) {
    std::string text = "[";
    for (size_t i = 0; i < items.size(); i++) text += (i ? ", " : "") + items[i].text_;
    return Value(Text{}, text + "]");
}

Case::Case(std::string text, std::string path, std::string name)
    : text_(std::move(text)), path_(std::move(path)), name_(std::move(name)) {
    translating(name_, diagnostic_code::INPUT, [&] { return parse_input_string(text_, name_); });
}

Case Case::from_toml_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw Error(diagnostic_code::INPUT, error_text(cannot_read("the input file", path)));
    return Case(std::string(std::istreambuf_iterator<char>(in), {}), path, path);
}

Case Case::from_toml(const std::string & text, const std::string & name) { return Case(text, "", name); }

const std::string & Case::toml() const { return text_; }

Case Case::set(const std::string_view key, const Value & value) const {
    std::string text;
    try {
        text = input_upgrade::set_value(text_, std::string(key), value.text());
    } catch (const std::invalid_argument & e) {
        throw Error(diagnostic_code::INPUT, name_ + ": " + e.what());
    }
    return Case(std::move(text), "", name_);
}

Case Case::set(const config::Table & table) const {
    if (table.path().empty()) {
        throw Error(diagnostic_code::INPUT, name_ + ": a table inside an array of tables or a named table is set "
                                                    "through its parent's setter, which places it");
    }
    Case out = *this;
    for (const auto & [key, value] : table.entries()) out = out.set(table.path() + "." + key, value);
    return out;
}

Case Case::save(const std::string & path) const {
    // Every rank of a run may save the case: each writes its own file and renames it, which replaces atomically
    const std::string own = path + ".tmp" + std::to_string(std::random_device{}());
    {
        std::ofstream out(own, std::ios::binary);
        out << text_;
        out.close();
        if (!out) throw Error(diagnostic_code::INPUT, error_text(cannot_write("the input file", own)));
    }
    std::error_code error;
    std::filesystem::rename(own, path, error);
    if (error) {
        std::filesystem::remove(own, error);
        throw Error(diagnostic_code::INPUT, error_text(cannot_write("the input file", path)));
    }
    return from_toml_file(path);
}

CheckReport Case::check() const { return check(CheckOptions{}); }

CheckReport Case::check(const CheckOptions & options) const {
    if (!Kokkos::is_initialized()) throw std::logic_error("mallard::Case::check: create a mallard::Session first");
    ::CheckOptions o;
    o.input_file = name_;
    o.ranks = options.ranks;
    o.throughput = options.throughput;
    o.steps = options.steps;
    if (!options.memory.empty()) {
        if (const std::string error = set_memory_target(options.memory, o); !error.empty()) {
            throw Error(diagnostic_code::INPUT, "mallard::CheckOptions::memory: " + error);
        }
    }
    logging::Sink sink;
    if (options.log) {
        sink = [log = options.log](logging::Level level, std::string_view code, std::string_view text) {
            LogMessage message;
            message.level = level == logging::Level::WARNING ? LogLevel::warning
                            : level == logging::Level::ERROR ? LogLevel::error
                                                             : LogLevel::info;
            message.code = code;
            message.text = text;
            log(message);
        };
    }
    struct Restore {
        logging::Sink previous;
        ~Restore() { logging::set_sink(std::move(previous)); }
    } restore{logging::set_sink(std::move(sink))};

    ::CheckReport r;
    try {
        ::Solver solver;
        r = solver.check(path_.empty() ? parse_input_string(text_, name_) : parse_input(path_), o);
    } catch (const InputError & e) {
        r = ::CheckReport{};
        r.input_file = name_;
        r.error(diagnostic_code::INPUT, name_ + ": " + error_text(e));
    }
    CheckReport out;
    out.status = report::status_name(r.status());
    out.ok = r.ok();
    out.setup_done = r.setup_done;
    for (const ::CheckDiagnostic & d : r.diagnostics) {
        out.diagnostics.push_back({d.severity == report::Status::WARNING ? LogLevel::warning : LogLevel::error, d.code,
                                   d.message, d.docs()});
    }
    for (const Diagnostic & d : r.unused) out.unused_keys.push_back({d.key, d.line, d.suggestion});
    for (const ::CheckBoundary & b : r.boundaries) {
        out.boundaries.push_back({b.zone, b.condition, b.faces, b.details, b.turbulence});
    }
    if (r.dt > 0.0) out.time_step = CheckTimeStep{r.t, r.dt, r.dt_cfl1, r.cfl, r.cfl_limit, r.integrator, r.dt_limit};
    for (const ::CheckEstimate & e : r.resolution) {
        out.resolution.push_back(
            {e.id, e.quantity, e.value, e.units, e.target, report::status_name(e.status), e.basis, e.region});
    }
    if (r.bytes > 0) {
        out.memory = CheckMemory{r.device_memory, r.ranks,          r.bytes, r.bytes_peak, r.target_ranks,
                                 r.target_bytes_peak, r.device_capacity};
    }
    if (r.setup_done) {
        out.cost = CheckCost{r.n_cells,       r.target_ranks,      r.throughput, r.throughput_measured,
                             r.wall_per_time, r.flow_through_time, r.t_stop,     r.n_steps};
    }
    out.resolved = r.resolved;
    for (const KeyRecord & k : r.keys) {
        const Dimension & d = k.use.dimension;
        out.keys.push_back({k.path, catalog_path(k.path), k.value, k.set, k.use.type,
                            d.same_as(dim::none) ? "" : d.describe(), d.symbol(), k.use.required,
                            k.use.required ? "" : k.use.fallback, k.values});
    }
    out.text = check_text(r, false);
    out.json = check_json(r);
    return out;
}

const std::string & Case::path() const { return path_; }

int Case::dimension() const {
    const front_end::Dimension d = translating(name_, diagnostic_code::INPUT, [&] {
        return front_end::parsed_input_dimension(parse_input_string(text_, name_));
    });
    if (d.dim == 0) throw Error(diagnostic_code::INPUT, name_ + ": cannot tell the case's dimension: " + d.why);
    return d.dim;
}

struct Solver::Impl {
    Case input;
    ::Solver solver;
    Solver * owner = nullptr;
    bool set_up = false;
    bool ended = false;

    struct Callback {
        CallbackId id;
        std::function<void(Solver &)> step;
        std::function<void(Solver &, const Stage &)> stage;
        std::function<void(const Progress &)> progress;
        std::function<void(const Output &)> output;
        std::function<void(const Record &)> record;
    };
    std::vector<Callback> callbacks;
    CallbackId next_id = 1;
    int in_callback = 0;
    int in_record = 0;  // inside a progress, output or record callback
    LogSink sink;
    int in_scope = 0;  // inside a call that runs the solver, with this solver's sink and record hook set
    bool stage_hooked = false;
    const ::State * stage_state = nullptr;  // inside a per-stage callback

    // Host copies, where the solver's memory is a device's, and what the caller writes into
    StateView::host_mirror_type h_conservatives;
    SpeciesView::host_mirror_type h_species;
    Kokkos::View<rtype *[N_PRIMITIVE]>::host_mirror_type h_primitives;
    Kokkos::View<rtype *[N_DIM]>::host_mirror_type h_centroid;
    std::vector<Kokkos::View<rtype *, Kokkos::HostSpace>> h_columns;
    std::vector<uint64_t> global_ids;
    bool writes_pending = false, conservatives_on_host = false, species_on_host = false;
    std::vector<std::string> written;

    explicit Impl(Case c) : input(std::move(c)) {}

    void not_in_callback(const char * call) const {
        if (in_callback) {
            throw std::logic_error(std::string("mallard::Solver::") + call + " inside a callback: callbacks may read fields "
                                   "and monitors and set the state, not step or end the run");
        }
    }

    void no_record_callback(const char * call) const {
        if (in_record) {
            throw std::logic_error(std::string("mallard::Solver::") + call + " inside a progress, output or record "
                                   "callback: these get their record only");
        }
    }

    template <typename M>
    bool any(M Callback::*member) const {
        return std::any_of(callbacks.begin(), callbacks.end(), [&](const Callback & c) { return bool(c.*member); });
    }

    /** @brief A progress, output or record callback's call: no Solver calls in it, and none that step. */
    template <typename F>
    void record_call(F && f) {
        in_callback++;
        in_record++;
        struct Leave {
            int & depth;
            int & record;
            ~Leave() {
                depth--;
                record--;
            }
        } leave{in_callback, in_record};
        f();
    }

    void progress_callbacks(const ::RunStatus & s, const std::string & files) {
        Progress p;
        p.step = s.step;
        p.time = s.t;
        p.dt = s.dt;
        p.cfl = s.cfl;
        p.fraction = s.progress;
        p.wall = s.wall;
        if (s.wall_per_step >= 0.0) p.wall_per_step = s.wall_per_step;
        if (s.cells_per_second >= 0.0) p.cells_per_second = s.cells_per_second;
        if (s.eta >= 0.0) p.eta = s.eta;
        p.cells = s.n_cells;
        p.ranks = s.n_ranks;
        p.diagnostics = s.monitors;
        p.written = files;
        p.json = s.record().str();
        record_call([&] {
            const std::vector<Callback> now = callbacks;
            for (const Callback & c : now) {
                if (c.progress) c.progress(p);
            }
        });
    }

    void record_callbacks(const std::string & line) {
        json::Value v;
        try {
            v = json::parse(line);
        } catch (const std::exception &) {
            return;
        }
        const json::Value * type = v.find("type");
        const Record record{type ? type->string : std::string(), line};
        std::optional<Output> output;
        if (record.type == "output") {
            Output & o = output.emplace();
            const auto number = [&](const char * key) { return v.find(key) ? v[key].number : 0.0; };
            const auto text = [&](const json::Value & from, const char * key) {
                return from.find(key) ? from[key].string : std::string();
            };
            o.step = static_cast<uint64_t>(number("step"));
            o.time = number("t");
            o.kind = text(v, "kind");
            o.path = text(v, "path");
            o.reason = text(v, "reason");
            o.writer = text(v, "writer");
            if (const json::Value * files = v.find("files")) {
                for (const json::Value & f : files->array) {
                    o.files.push_back({text(f, "path"), static_cast<uint64_t>(f.find("bytes") ? f["bytes"].number : 0.0)});
                }
            }
            o.json = line;
        }
        record_call([&] {
            const std::vector<Callback> now = callbacks;
            for (const Callback & c : now) {
                if (c.record) c.record(record);
                if (c.output && output) c.output(*output);
            }
        });
    }

    /** @brief While a call runs the solver: this solver's log sink and record callbacks, then those before. */
    struct Scope {
        Impl & m;
        logging::Sink sink;
        run_log::RecordHook hook;
        explicit Scope(Impl & m_) : m(m_) {
            sink = logging::set_sink(m.log_sink());
            hook = run_log::set_record_hook(m.record_hook());
            m.in_scope++;
        }
        ~Scope() {
            m.in_scope--;
            logging::set_sink(std::move(sink));
            run_log::set_record_hook(std::move(hook));
        }
        Scope(const Scope &) = delete;
        Scope & operator=(const Scope &) = delete;
    };

    logging::Sink log_sink() {
        if (!sink) return {};
        return [this](logging::Level level, std::string_view code, std::string_view text) {
            LogMessage message;
            message.level = level == logging::Level::WARNING ? LogLevel::warning
                            : level == logging::Level::ERROR ? LogLevel::error
                                                             : LogLevel::info;
            message.code = code;
            message.text = text;
            const LogSink f = sink;
            f(message);
        };
    }

    run_log::RecordHook record_hook() {
        if (!any(&Callback::output) && !any(&Callback::record)) return {};
        return [this](const std::string & line) { record_callbacks(line); };
    }

    void install_hooks() {
        if (any(&Callback::progress)) {
            solver.set_progress_hook([this](const ::RunStatus & s, const std::string & w) { progress_callbacks(s, w); });
        } else {
            solver.set_progress_hook({});
        }
        if (in_scope) {
            logging::set_sink(log_sink());
            run_log::set_record_hook(record_hook());
        }
        const bool steps = std::any_of(callbacks.begin(), callbacks.end(), [](const Callback & c) { return bool(c.step); });
        const bool stages = std::any_of(callbacks.begin(), callbacks.end(), [](const Callback & c) { return bool(c.stage); });
        solver.set_step_hook(steps ? std::function<void()>([this] { step_callbacks(); }) : std::function<void()>());
        if (stages == stage_hooked) return;
        stage_hooked = stages;
        solver.set_stage_hook(stages ? std::function<void(const ::State &, rtype, int)>(
                                           [this](const ::State & state, rtype t, int index) {
                                               stage_callbacks(state, t, index);
                                           })
                                     : std::function<void(const ::State &, rtype, int)>());
    }

    void step_callbacks() {
        in_callback++;
        struct Leave {
            int & depth;
            ~Leave() { depth--; }
        } leave{in_callback};
        // A callback registered or removed by another takes effect at the next step
        const std::vector<Callback> now = callbacks;
        for (const Callback & c : now) {
            if (c.step) c.step(*owner);
        }
        commit();
    }

    void stage_callbacks(const ::State & state, rtype t, int index) {
        in_callback++;
        stage_state = &state;
        struct Leave {
            int & depth;
            const ::State *& stage;
            ~Leave() {
                depth--;
                stage = nullptr;
            }
        } leave{in_callback, stage_state};
        const Stage stage{solver.get_step(), index, static_cast<double>(t)};
        const std::vector<Callback> now = callbacks;
        for (const Callback & c : now) {
            if (c.stage) c.stage(*owner, stage);
        }
    }

    void commit() {
        if (!writes_pending) return;
        const ::State state = solver.state();
        if (conservatives_on_host) Kokkos::deep_copy(state.flow, h_conservatives);
        if (species_on_host) Kokkos::deep_copy(state.species, h_species);
        std::string what = "the caller set";
        for (size_t i = 0; i < written.size(); i++) what += (i ? ", " : " ") + written[i];
        solver.state_set(what);
        writes_pending = conservatives_on_host = species_on_host = false;
        written.clear();
    }

    template <typename F>
    auto run(const char * code, F && f) {
        const Scope scope(*this);
        return translating(input.name_, code, std::forward<F>(f));
    }

    void ensure_setup() {
        if (ended) throw std::logic_error("mallard::Solver: the run has ended (finish)");
        if (set_up) return;
        not_in_callback("setup");
        if (set_up) return;
        if (!Kokkos::is_initialized()) {
            throw std::logic_error("mallard::Solver: create a mallard::Session (MPI and Kokkos) first");
        }
        run(diagnostic_code::SETUP, [&] {
            // From the file itself where there is one, as `mallard -i` reads it
            if (input.path_.empty()) {
                solver.init_from_text(input.text_, input.name_);
            } else {
                solver.init(input.path_);
            }
        });
        run(diagnostic_code::RUN, [&] { solver.start_run(); });
        set_up = true;
    }

    bool advance(uint64_t until_step, rtype until_time) {
        not_in_callback("step");
        ensure_setup();
        h_columns.clear();
        run(diagnostic_code::RUN, [&] {
            commit();
            solver.advance_run(until_step, until_time);
        });
        return !solver.run_stopped();
    }
};

Solver::Solver(Case c) {
    if (const int dim = c.dimension(); dim != N_DIM) {
        throw Error(diagnostic_code::INPUT, c.name_ + " is a " + std::to_string(dim) + "D case, which libmallard" +
                                                std::to_string(dim) + "d runs; this is libmallard" + std::to_string(N_DIM) + "d.");
    }
    impl = std::make_unique<Impl>(std::move(c));
    impl->owner = this;
}

Solver::~Solver() = default;

Solver::Solver(Solver && other) noexcept : impl(std::move(other.impl)) {
    if (impl) impl->owner = this;
}

Solver & Solver::operator=(Solver && other) noexcept {
    impl = std::move(other.impl);
    if (impl) impl->owner = this;
    return *this;
}

void Solver::setup() { impl->ensure_setup(); }

bool Solver::step() { return step(1); }

bool Solver::step(const uint64_t n) {
    impl->not_in_callback("step");
    impl->ensure_setup();
    if (impl->solver.run_stopped()) return false;
    if (n == 0) return true;
    const uint64_t now = impl->solver.get_step();
    const uint64_t until = n > std::numeric_limits<uint64_t>::max() - now ? std::numeric_limits<uint64_t>::max() : now + n;
    return impl->advance(until, std::numeric_limits<rtype>::infinity());
}

bool Solver::advance_to(const double t) {
    if (!std::isfinite(t)) throw std::invalid_argument("mallard::Solver::advance_to: t must be finite");
    return impl->advance(std::numeric_limits<uint64_t>::max(), static_cast<rtype>(t));
}

int Solver::finish() {
    impl->not_in_callback("finish");
    impl->ensure_setup();
    const int status = impl->run(diagnostic_code::RUN, [&] {
        impl->commit();
        return impl->solver.finish_run();
    });
    impl->ended = true;
    return status;
}

int Solver::run() {
    impl->advance(std::numeric_limits<uint64_t>::max(), std::numeric_limits<rtype>::infinity());
    return finish();
}

bool Solver::finished() const { return impl->set_up && impl->solver.run_stopped(); }

double Solver::time() const { return static_cast<double>(impl->solver.get_time()); }

uint64_t Solver::step_count() const { return impl->solver.get_step(); }

std::vector<Monitor> Solver::monitors() const {
    impl->no_record_callback("monitors");
    const ::Solver & s = impl->solver;
    if (!impl->set_up) return {};
    const auto monitored = s.monitored();
    std::vector<double> values = s.monitored_values();
    const std::vector<uint64_t> steps = s.monitored_steps();
    // Probes' values are on rank 0 alone in a distributed run
    if (s.is_distributed()) {
        const comm::SelfScope scope = s.communication_scope();
        values = broadcast(values);
    }
    const bool si = units::system() == units::System::SI;
    std::vector<Monitor> out;
    out.reserve(monitored.size());
    for (size_t i = 0; i < monitored.size(); i++) {
        Monitor m;
        const auto & [name, dimension] = monitored[i];
        m.name = name;
        m.unit = si ? dimension.udunits() : std::string("1");
        m.value = values[i];
        if (steps[i] != Probes::NO_VALUE) m.step = steps[i];
        out.push_back(std::move(m));
    }
    return out;
}

std::optional<Monitor> Solver::monitor(std::string_view name) const {
    for (Monitor & m : monitors()) {
        if (m.name == name) return std::move(m);
    }
    return std::nullopt;
}

std::vector<std::string> Solver::field_names() const {
    if (!impl->set_up) return {};
    std::vector<std::string> names(CONSERVATIVE_NAMES.begin(), CONSERVATIVE_NAMES.end());
    const ::Solver & s = impl->solver;
    for (const std::string & species : s.get_species_names()) names.push_back("RHOY_" + species);
    names.insert(names.end(), PRIMITIVE_NAMES.begin(), PRIMITIVE_NAMES.end());
    names.insert(names.end(), {"conservatives", "species", "primitives"});
    for (const std::string & name : s.get_statistics().variables()) {
        if (name.rfind("MEAN_", 0) == 0 || name.rfind("COV_", 0) == 0) names.push_back(name);
    }
    names.insert(names.end(), {"CENTROID", "VOLUME", "GLOBAL_ID"});
    return names;
}

namespace {

/** @brief Where a field name points: a column of a block, a whole block, an average or the geometry. */
struct FieldSource {
    enum Kind { CONSERVATIVE, SPECIES, PRIMITIVE, AVERAGE, CENTROID, VOLUME, GLOBAL_ID, NONE } kind = NONE;
    int column = -1;  // -1: the whole block
};

FieldSource find_field(const std::string & name, const std::vector<std::string> & species,
                       const std::vector<std::string> & averages) {
    for (int i = 0; i < N_CONSERVATIVE; i++) {
        if (name == CONSERVATIVE_NAMES[size_t(i)]) return {FieldSource::CONSERVATIVE, i};
    }
    for (size_t k = 0; k < species.size(); k++) {
        if (name == "RHOY_" + species[k]) return {FieldSource::SPECIES, int(k)};
    }
    for (int i = 0; i < N_PRIMITIVE; i++) {
        if (name == PRIMITIVE_NAMES[size_t(i)]) return {FieldSource::PRIMITIVE, i};
    }
    if (name == "conservatives") return {FieldSource::CONSERVATIVE, -1};
    if (name == "species") return {FieldSource::SPECIES, -1};
    if (name == "primitives") return {FieldSource::PRIMITIVE, -1};
    if (name == "CENTROID") return {FieldSource::CENTROID, -1};
    if (name == "VOLUME") return {FieldSource::VOLUME, -1};
    if (name == "GLOBAL_ID") return {FieldSource::GLOBAL_ID, -1};
    if (std::find(averages.begin(), averages.end(), name) != averages.end()) return {FieldSource::AVERAGE, -1};
    return {};
}

/** @brief host's copy of the device view v, allocated at the first call. */
template <typename H, typename V>
void copy_to_host(H & host, const V & v) {
    if (host.span() != v.span()) host = Kokkos::create_mirror(Kokkos::HostSpace(), v);
    Kokkos::deep_copy(host, v);
}

} // namespace

Field Solver::field(const std::string_view name_view, const Memory memory) {
    impl->no_record_callback("field");
    impl->ensure_setup();
    Impl & m = *impl;
    ::Solver & s = m.solver;
    const std::string name(name_view);
    std::vector<std::string> averages;
    for (const std::string & a : s.get_statistics().variables()) {
        if (a.rfind("MEAN_", 0) == 0 || a.rfind("COV_", 0) == 0) averages.push_back(a);
    }
    const FieldSource source = find_field(name, s.get_species_names(), averages);
    if (source.kind == FieldSource::NONE) {
        std::string known;
        for (const std::string & n : field_names()) known += (known.empty() ? "" : ", ") + n;
        throw Error(diagnostic_code::RUN, "mallard::Solver::field: no field " + name + " in this case; it has " + known + ".");
    }
    const uint32_t n = s.get_mesh()->n_owned();
    const bool host_copy = memory == Memory::host && !DEVICE_IS_HOST;
    const Device device = host_copy ? Device::host : solver_device();
    const bool in_stage = m.stage_state != nullptr;
    if (in_stage && (source.kind == FieldSource::PRIMITIVE || source.kind == FieldSource::AVERAGE)) {
        throw Error(diagnostic_code::RUN, "mallard::Solver::field: " + name + " is not computed at a stage; a per-stage "
                                          "callback reads the conservative variables, partial densities and geometry.");
    }
    Field block;
    switch (source.kind) {
        case FieldSource::CONSERVATIVE:
        case FieldSource::SPECIES: {
            const ::State state = in_stage ? *m.stage_state : s.boundary_state();
            const bool flow = source.kind == FieldSource::CONSERVATIVE;
            if (!host_copy) {
                block = flow ? view_field(name, state.flow, n, device, false) : view_field(name, state.species, n, device, false);
            } else if (flow) {
                // A host copy written by the caller stays theirs until committed
                if (!m.conservatives_on_host || in_stage) copy_to_host(m.h_conservatives, state.flow);
                block = view_field(name, m.h_conservatives, n, device, false);
            } else {
                if (!m.species_on_host || in_stage) copy_to_host(m.h_species, state.species);
                block = view_field(name, m.h_species, n, device, false);
            }
            break;
        }
        case FieldSource::PRIMITIVE:
            s.boundary_state();
            s.update_primitives();
            Kokkos::fence("mallard_api_primitives");
            if (host_copy) {
                copy_to_host(m.h_primitives, s.primitives);
                block = view_field(name, m.h_primitives, n, device, false);
            } else {
                block = view_field(name, s.primitives, n, device, false);
            }
            break;
        case FieldSource::AVERAGE: {
            const auto column_view = s.get_statistics().device_column(name).values;
            Kokkos::View<const rtype *, Kokkos::LayoutRight, DeviceMemory, Kokkos::MemoryUnmanaged> v(
                column_view.data(), column_view.extent(0));
            if (column_view.stride(0) != 1) throw std::logic_error("mallard::Solver::field: a strided average");
            if (host_copy) {
                m.h_columns.emplace_back("mallard_api_" + name, v.extent(0));
                Kokkos::deep_copy(m.h_columns.back(), v);
                block = view_field(name, m.h_columns.back(), n, device, false);
            } else {
                block = view_field(name, v, n, device, false);
            }
            break;
        }
        case FieldSource::CENTROID:
            if (host_copy) {
                copy_to_host(m.h_centroid, s.get_mesh()->cell_coords);
                block = view_field(name, m.h_centroid, n, device, false);
            } else {
                block = view_field(name, s.get_mesh()->cell_coords, n, device, false);
            }
            break;
        case FieldSource::VOLUME:
            if (host_copy) {
                block = view_field(name, s.get_mesh()->h_cell_volume, n, device, false);
            } else {
                block = view_field(name, s.get_mesh()->cell_volume, n, device, false);
            }
            break;
        case FieldSource::GLOBAL_ID: {
            const Mesh & mesh = *s.get_mesh();
            if (m.global_ids.size() != n) {
                m.global_ids.resize(n);
                for (uint32_t i = 0; i < n; i++) m.global_ids[i] = mesh.h_global_cell(i);
            }
            block.name = name;
            block.data = m.global_ids.data();
            block.type = ScalarType::uint64;
            block.shape = {int64_t(n)};
            block.strides = {1};
            break;
        }
        case FieldSource::NONE:
            break;
    }
    if (source.column >= 0) return column(block, name, source.column);
    const bool is_block = source.kind == FieldSource::CONSERVATIVE || source.kind == FieldSource::SPECIES ||
                          source.kind == FieldSource::PRIMITIVE;
    if (!is_block) block.unit = field_unit(name);
    return block;
}

Field Solver::writable_field(const std::string_view name_view, const Memory memory) {
    impl->no_record_callback("writable_field");
    impl->ensure_setup();
    Impl & m = *impl;
    if (m.stage_state) {
        throw Error(diagnostic_code::RUN, "mallard::Solver::writable_field inside a per-stage callback: the state is "
                                          "set between steps only.");
    }
    const std::string name(name_view);
    const FieldSource source = find_field(name, m.solver.get_species_names(), {});
    if (source.kind != FieldSource::CONSERVATIVE && source.kind != FieldSource::SPECIES) {
        throw Error(diagnostic_code::RUN, "mallard::Solver::writable_field: " + name + " is not part of the state; the "
                                          "state is the conservative variables (" + CONSERVATIVE_NAMES[0] + ", ...), "
                                          "the partial densities (RHOY_<species>) and their blocks (conservatives, "
                                          "species).");
    }
    Field f = field(name, memory);
    f.writable = true;
    const bool host_copy = memory == Memory::host && !DEVICE_IS_HOST;
    if (host_copy) (source.kind == FieldSource::CONSERVATIVE ? m.conservatives_on_host : m.species_on_host) = true;
    m.writes_pending = true;
    if (std::find(m.written.begin(), m.written.end(), name) == m.written.end()) m.written.push_back(name);
    return f;
}

void Solver::commit_state() {
    impl->no_record_callback("commit_state");
    impl->ensure_setup();
    impl->run(diagnostic_code::RUN, [&] { impl->commit(); });
}

Statistics Solver::statistics() const {
    Statistics out;
    out.start = out.end = std::numeric_limits<double>::quiet_NaN();
    if (!impl->set_up) return out;
    const ::Statistics & s = impl->solver.get_statistics();
    if (!s.means_enabled()) return out;
    out.samples = s.samples();
    out.start = s.average_start();
    out.end = s.average_end();
    out.batches = s.batches_enabled() ? s.batches() : 0;
    return out;
}

CallbackId Solver::on_step(std::function<void(Solver &)> f) {
    if (!f) throw std::invalid_argument("mallard::Solver::on_step: an empty function");
    const CallbackId id = impl->next_id++;
    impl->callbacks.push_back({id, std::move(f), {}, {}, {}, {}});
    impl->install_hooks();
    return id;
}

CallbackId Solver::on_stage(std::function<void(Solver &, const Stage &)> f) {
    if (!f) throw std::invalid_argument("mallard::Solver::on_stage: an empty function");
    const CallbackId id = impl->next_id++;
    impl->callbacks.push_back({id, {}, std::move(f), {}, {}, {}});
    impl->install_hooks();
    return id;
}

CallbackId Solver::on_progress(std::function<void(const Progress &)> f) {
    if (!f) throw std::invalid_argument("mallard::Solver::on_progress: an empty function");
    const CallbackId id = impl->next_id++;
    impl->callbacks.push_back({id, {}, {}, std::move(f), {}, {}});
    impl->install_hooks();
    return id;
}

CallbackId Solver::on_output(std::function<void(const Output &)> f) {
    if (!f) throw std::invalid_argument("mallard::Solver::on_output: an empty function");
    const CallbackId id = impl->next_id++;
    impl->callbacks.push_back({id, {}, {}, {}, std::move(f), {}});
    impl->install_hooks();
    return id;
}

CallbackId Solver::on_record(std::function<void(const Record &)> f) {
    if (!f) throw std::invalid_argument("mallard::Solver::on_record: an empty function");
    const CallbackId id = impl->next_id++;
    impl->callbacks.push_back({id, {}, {}, {}, {}, std::move(f)});
    impl->install_hooks();
    return id;
}

void Solver::set_log(LogSink sink) {
    impl->sink = std::move(sink);
    if (impl->in_scope) logging::set_sink(impl->log_sink());
}

void Solver::remove_callback(const CallbackId id) {
    auto & c = impl->callbacks;
    c.erase(std::remove_if(c.begin(), c.end(), [&](const Impl::Callback & k) { return k.id == id; }), c.end());
    impl->install_hooks();
}

} // namespace mallard
