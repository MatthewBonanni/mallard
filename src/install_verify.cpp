/**
 * @file install_verify.cpp
 * @brief `mallard install verify`: whether an install works where it is, with a verdict for scripts.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include "build_info.h"
#include "case_template.h"
#include "cli.h"
#include "commands.h"
#include "diagnostic_code.h"
#include "front_end.h"
#include "json.h"
#include "log.h"

namespace fs = std::filesystem;

namespace {

/** @brief One check of the verdict: what was run, whether it passed, how long it took and why. */
struct Check {
    std::string name;
    int dim = 0;
    bool passed = false;
    double seconds = 0.0;
    std::string detail;
    std::string extra;  ///< further members of its JSON object, as JSON text without braces
};

struct Exit {
    int status = -1;
    double seconds = 0.0;
    std::string output;
};

std::string read_text(const fs::path & path) {
    std::ifstream in(path);
    std::stringstream text;
    text << in.rdbuf();
    return text.str();
}

/** @brief Runs argv in dir, its standard output and error to log, and waits for it. */
Exit run_in(const std::vector<std::string> & argv, const fs::path & dir, const fs::path & log) {
    Exit e;
    std::vector<std::string> args = argv;
    std::vector<char *> c_args;
    for (auto & a : args) c_args.push_back(a.data());
    c_args.push_back(nullptr);
    const auto start = std::chrono::steady_clock::now();
    const pid_t pid = fork();
    if (pid == 0) {
        const int fd = open(log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0 || chdir(dir.c_str()) != 0) _exit(126);
        dup2(fd, 1);
        dup2(fd, 2);
        execvp(c_args[0], c_args.data());
        std::fprintf(stderr, "cannot run %s: %s\n", c_args[0], std::strerror(errno));
        _exit(127);
    }
    int wstatus = 0;
    if (pid > 0 && waitpid(pid, &wstatus, 0) == pid && WIFEXITED(wstatus)) e.status = WEXITSTATUS(wstatus);
    e.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    e.output = read_text(log);
    return e;
}

/** @brief The last lines of a command's output, for the reason a check failed. */
std::string tail(const std::string & text, size_t lines = 6) {
    std::vector<std::string> all;
    std::istringstream in(text);
    for (std::string line; std::getline(in, line);) {
        if (!line.empty()) all.push_back(line);
    }
    std::string out;
    for (size_t i = all.size() > lines ? all.size() - lines : 0; i < all.size(); i++) {
        out += (out.empty() ? "" : "\n") + all[i];
    }
    return out;
}

std::vector<std::string> split_words(const std::string & text) {
    std::vector<std::string> out;
    std::istringstream in(text);
    for (std::string w; in >> w;) out.push_back(w);
    return out;
}

/** @brief The environment check of one solver: `mallard check --format json` without an input, under the launcher. */
Check environment(int dim, const fs::path & solver, const std::vector<std::string> & launcher,
                  const std::vector<std::string> & kokkos, const fs::path & work) {
    Check c{"environment", dim, false, 0.0, "", ""};
    std::vector<std::string> argv = launcher;
    argv.push_back(solver.string());
    argv.insert(argv.end(), {"check", "--format", "json"});
    argv.insert(argv.end(), kokkos.begin(), kokkos.end());
    const Exit e = run_in(argv, work, work / ("environment_" + std::to_string(dim) + "d.log"));
    c.seconds = e.seconds;
    const size_t brace = e.output.find('{');
    const size_t close = e.output.rfind('}');
    try {
        if (brace == std::string::npos || close == std::string::npos || close < brace) {
            throw std::runtime_error("no report");
        }
        const json::Value report = json::parse(std::string_view(e.output).substr(brace, close - brace + 1));
        const json::Value * errors = report.find("errors");
        const json::Value * env = report.find("environment");
        const json::Value * build = env ? env->find("build") : nullptr;
        const int n_errors = errors ? static_cast<int>(errors->number) : 0;
        c.passed = e.status == 0 && n_errors == 0;
        std::vector<std::string> codes;
        if (const json::Value * d = report.find("diagnostics"); d && d->type == json::Value::Type::ARRAY) {
            for (const json::Value & diag : d->array) {
                const json::Value * code = diag.find("code");
                if (code) codes.push_back(code->string);
            }
        }
        std::string list;
        for (const std::string & code : codes) list += (list.empty() ? "" : ", ") + code;
        c.detail = c.passed ? (codes.empty() ? "ok" : "ok, with " + list) : std::to_string(n_errors) + " errors: " + list;
        if (build) {
            const json::Value * backend = build->find("backend");
            const json::Value * precision = build->find("precision");
            const json::Value * mpi = build->find("mpi");
            c.extra = "\"backend\":" + json::quote(backend ? backend->string : "") +
                      ",\"precision\":" + json::quote(precision ? precision->string : "") +
                      ",\"mpi\":" + std::string(mpi && mpi->boolean ? "true" : "false");
        }
    } catch (const std::exception &) {
        c.passed = false;
        c.detail = "exit " + std::to_string(e.status) + ": " + tail(e.output);
    }
    return c;
}

/**
 * @brief The Sod shock tube of `mallard new sod --quick`, made three-dimensional for 3D: extruded one cubic cell
 *        thick, so it takes the steps of the 2D case.
 */
bool make_case(int dim, const fs::path & front_end, const fs::path & dir, std::string & why) {
    // --dim: the template is 2D, and a build may have mallard3d alone
    const Exit e = run_in({front_end.string(), "--dim", std::to_string(dim), "new", "sod", dir.string(), "--quick",
                           "--force"},
                          dir.parent_path(),
                          dir.parent_path() / ("new_" + std::to_string(dim) + "d.log"));
    if (e.status != 0) {
        why = "mallard new sod --quick: exit " + std::to_string(e.status) + ": " + tail(e.output);
        return false;
    }
    if (dim == 2) return true;
    std::string text = read_text(dir / "input.toml");
    text = case_template::set_key(text, "mesh.Nz", "1");
    text = case_template::set_key(text, "mesh.Lz", "0.005");
    text = case_template::set_key(text, "initialize.u", "[\"0.0\", \"0.0\", \"0.0\"]");
    text += "\n[[boundaries]]\nname = \"back\"\ntype = \"symmetry\"\n\n[[boundaries]]\nname = \"front\"\ntype = \"symmetry\"\n";
    std::ofstream(dir / "input.toml") << text;
    return true;
}

/**
 * @brief The quick run: the Sod shock tube to t = 0.2 on 100 cells, through the run log's end record and its
 *        monitors against the exact solution: the undisturbed right state (rho 0.125, p 0.1) and the largest
 *        Mach number, 0.929 between the rarefaction's tail and the contact.
 */
Check quick_run(int dim, const fs::path & solver, const fs::path & front_end, const std::vector<std::string> & launcher,
                const std::vector<std::string> & kokkos, const fs::path & work) {
    Check c{"run", dim, false, 0.0, "", ""};
    const fs::path dir = work / ("sod_" + std::to_string(dim) + "d");
    if (!make_case(dim, front_end, dir, c.detail)) return c;
    std::vector<std::string> argv = launcher;
    argv.push_back(solver.string());
    argv.insert(argv.end(), {"-i", "input.toml"});
    argv.insert(argv.end(), kokkos.begin(), kokkos.end());
    const Exit e = run_in(argv, dir, dir / "run.log");
    c.seconds = e.seconds;
    std::ifstream log(dir / "mallard.jsonl");
    json::Value end, last;
    for (std::string line; std::getline(log, line);) {
        try {
            json::Value v = json::parse(line);
            const json::Value * type = v.find("type");
            if (!type) continue;
            if (type->string == "end") end = v;
            if (type->string == "progress") last = v;
        } catch (const std::exception &) {
        }
    }
    const json::Value * outcome = end.find("outcome");
    if (e.status != 0 || !outcome || outcome->string != "finished") {
        c.detail = "exit " + std::to_string(e.status) + (outcome ? ", outcome " + outcome->string : ", no end record") +
                   ": " + tail(e.output);
        return c;
    }
    const json::Value * monitors = last.find("monitors");
    auto monitor = [&](const char * name) {
        const json::Value * m = monitors ? monitors->find(name) : nullptr;
        return m && m->type == json::Value::Type::NUMBER ? m->number : std::nan("");
    };
    struct Reference {
        const char * name;
        double exact, tolerance;
    };
    const Reference references[] = {{"min_rho", 0.125, 0.0025}, {"min_p", 0.1, 0.002}, {"max_mach", 0.929, 0.05}};
    std::string values, wrong;
    for (const Reference & r : references) {
        const double v = monitor(r.name);
        values += std::string(values.empty() ? "" : ",") + json::quote(r.name) + ":" + json::number(v);
        if (!(std::abs(v - r.exact) <= r.tolerance)) {
            wrong += std::string(wrong.empty() ? "" : "; ") + r.name + " " + json::number(v) + ", exact " +
                     json::number(r.exact) + " +- " + json::number(r.tolerance);
        }
    }
    const json::Value * steps = end.find("steps");
    const json::Value * cps = end.find("cells_per_second");
    c.passed = wrong.empty();
    c.detail = c.passed ? "Sod to t = 0.2 in " + std::to_string(steps ? static_cast<long>(steps->number) : 0) +
                              " steps, as the exact solution"
                        : "the solution differs from the exact one: " + wrong;
    c.extra = "\"template\":\"sod\",\"steps\":" + json::number(steps ? steps->number : 0.0) +
              ",\"cells_per_second\":" + json::number(cps ? cps->number : std::nan("")) + ",\"monitors\":{" + values +
              "}";
    return c;
}

} // namespace

int install_verify_main(const cli::Args & args) {
    const std::string format = args.value_or("--format", "text");
    if (format != "text" && format != "json") {
        return cli::usage_error(*cli::command("install verify"), "--format takes text or json, not " + format);
    }
    const fs::path bin = executable_path().parent_path();
    const std::vector<int> dims = front_end::solvers();
    if (dims.empty()) {
        logging::error(diagnostic_code::EXECUTABLE, "mallard install verify: neither mallard2d nor mallard3d is in " +
                                                        bin.string() + ".");
        return 1;
    }
    const std::vector<std::string> launcher = split_words(args.value_or("--launcher", ""));
    std::vector<std::string> kokkos;
    for (const std::string & a : args.passthrough) {
        if (a.rfind("--kokkos-", 0) == 0) kokkos.push_back(a);
    }
    const bool keep = args.has("--keep");
    fs::path work;
    if (keep) {
        work = fs::absolute(args.value_or("--keep", ""));
        fs::create_directories(work);
    } else {
        std::string pattern = (fs::temp_directory_path() / "mallard-verify-XXXXXX").string();
        if (mkdtemp(pattern.data()) == nullptr) {
            logging::error(diagnostic_code::EXECUTABLE, "mallard install verify: cannot create a directory in " +
                                                            fs::temp_directory_path().string());
            return 1;
        }
        work = pattern;
    }
    const fs::path front = bin / "mallard";
    const auto start = std::chrono::steady_clock::now();
    std::vector<Check> checks;
    for (const int dim : dims) {
        const fs::path solver = bin / ("mallard" + std::to_string(dim) + "d");
        checks.push_back(environment(dim, solver, launcher, kokkos, work));
        checks.push_back(quick_run(dim, solver, fs::exists(front) ? front : solver, launcher, kokkos, work));
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    bool passed = true;
    for (const Check & c : checks) passed = passed && c.passed;
    if (!keep) {
        std::error_code ec;
        fs::remove_all(work, ec);
    }
    if (format == "json") {
        std::string list = "[";
        for (const Check & c : checks) {
            json::Object o;
            o.add("name", c.name).add("dim", c.dim).add("passed", c.passed).add("seconds", c.seconds).add("detail",
                                                                                                          c.detail);
            std::string text = o.str();
            if (!c.extra.empty()) text.insert(text.size() - 1, "," + c.extra);
            list += (list.size() > 1 ? "," : "") + text;
        }
        list += "]";
        json::Object out;
        out.add("schema", std::string("mallard-install-verify"))
            .add("version", 1)
            .add("passed", passed)
            .add("mallard", mallard_version())
            .add("prefix", bin.parent_path().string())
            .add("launcher", args.value_or("--launcher", ""))
            .add("seconds", seconds)
            .raw("checks", list);
        if (keep) out.add("directory", work.string());
        std::cout << out.str() << std::endl;
    } else {
        std::cout << "Mallard " << mallard_version() << " in " << bin.parent_path().string() << "\n";
        for (const Check & c : checks) {
            char time[32];
            std::snprintf(time, sizeof time, "%6.1f s", c.seconds);
            std::cout << "  " << (c.passed ? "ok    " : "FAILED") << "  " << c.dim << "D  " << c.name
                      << std::string(c.name.size() < 12 ? 12 - c.name.size() : 1, ' ') << time << "  " << c.detail
                      << "\n";
        }
        std::cout << (passed ? "The install works." : "The install does not work: see the checks that FAILED.")
                  << (keep ? " Its cases are in " + work.string() + "." : std::string()) << std::endl;
    }
    return passed ? 0 : 1;
}
