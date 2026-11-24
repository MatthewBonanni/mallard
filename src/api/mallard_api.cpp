/**
 * @file mallard_api.cpp
 * @brief The C++ library API (include/mallard/mallard.h) over Solver, as run_main.cpp drives it.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "mallard/mallard.h"

#include <cmath>
#include <cstring>
#include <exception>
#include <fstream>
#include <iterator>
#include <limits>
#include <new>

#include <Kokkos_Core.hpp>

#ifdef Mallard_HAS_MPI
#include <mpi.h>
#endif

#include "diagnostic_code.h"
#include "input.h"
#include "parallel/binding.h"
#include "parallel/comm.h"
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

} // namespace

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

Session::Session(int & argc, char **& argv) : impl(std::make_unique<Impl>()) {
    if (Kokkos::is_finalized()) {
        throw Error(diagnostic_code::SETUP, "Kokkos was finalized in this process and cannot initialize again: keep "
                                            "one mallard::Session for the whole program.");
    }
#ifdef Mallard_HAS_MPI
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    // MPI before Kokkos, so Kokkos can map each rank to its own device
    if (!mpi_initialized) impl->mpi = std::make_unique<comm::Session>(argc, argv);
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

const std::string & Case::path() const { return path_; }

struct Solver::Impl {
    Case input;
    ::Solver solver;
    bool set_up = false;
    bool ended = false;

    explicit Impl(Case c) : input(std::move(c)) {}

    template <typename F>
    auto run(const char * code, F && f) {
        return translating(input.name_, code, std::forward<F>(f));
    }

    void ensure_setup() {
        if (ended) throw std::logic_error("mallard::Solver: the run has ended (finish)");
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
        ensure_setup();
        run(diagnostic_code::RUN, [&] { solver.advance_run(until_step, until_time); });
        return !solver.run_stopped();
    }
};

Solver::Solver(Case c) : impl(std::make_unique<Impl>(std::move(c))) {}

Solver::~Solver() = default;
Solver::Solver(Solver &&) noexcept = default;
Solver & Solver::operator=(Solver &&) noexcept = default;

void Solver::setup() { impl->ensure_setup(); }

bool Solver::step() {
    impl->ensure_setup();
    if (impl->solver.run_stopped()) return false;
    return impl->advance(impl->solver.get_step() + 1, std::numeric_limits<rtype>::infinity());
}

bool Solver::advance_to(const double t) {
    if (!std::isfinite(t)) throw std::invalid_argument("mallard::Solver::advance_to: t must be finite");
    return impl->advance(std::numeric_limits<uint64_t>::max(), static_cast<rtype>(t));
}

int Solver::finish() {
    impl->ensure_setup();
    const int status = impl->run(diagnostic_code::RUN, [&] { return impl->solver.finish_run(); });
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
    std::vector<Monitor> out;
    out.reserve(monitored.size());
    for (size_t i = 0; i < monitored.size(); i++) {
        Monitor m;
        const auto & [name, dimension] = monitored[i];
        m.name = name;
        m.unit = units::of(dimension);
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

} // namespace mallard
