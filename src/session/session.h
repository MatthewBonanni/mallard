/**
 * @file session.h
 * @brief mallard session: Mallard's machine interface. A resident process that holds one case and answers
 *        JSON-RPC 2.0 requests, one per line, on stdin and stdout. The one-shot commands run the same handlers.
 *
 * Design: docs/design/session.md. Protocol: docs/guide/session.md.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef SESSION_H
#define SESSION_H

#include <cstdint>
#include <istream>
#include <memory>
#include <optional>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "check.h"
#include "input.h"
#include "json.h"

class MeshFileCache;
class SetupCache;

namespace session {

/** @brief Schema name and major version of the protocol: its messages, errors and notifications. */
inline constexpr const char * SCHEMA = "mallard-session";
inline constexpr int VERSION = 1;

/** @brief Schema of a case's description (case.open, case.update). */
inline constexpr const char * CASE_SCHEMA = "mallard-case";
inline constexpr int CASE_VERSION = 1;

/** @brief JSON-RPC 2.0 error codes, and Mallard's (docs/guide/session.md#errors). */
namespace code {
inline constexpr int PARSE = -32700;
inline constexpr int INVALID_REQUEST = -32600;
inline constexpr int NO_METHOD = -32601;
inline constexpr int INVALID_PARAMS = -32602;
inline constexpr int INTERNAL = -32603;
inline constexpr int CANCELED = -32800;
inline constexpr int NO_CASE = -32001;
inline constexpr int UNREADABLE = -32003;
} // namespace code

/** @brief A request that failed: a JSON-RPC error, with Mallard's data (a JSON object, or empty). */
struct Error : std::runtime_error {
    int code;
    std::string data;
    Error(int code_in, const std::string & message, std::string data_in = "")
        : std::runtime_error(message), code(code_in), data(std::move(data_in)) {}
};

/**
 * @brief Thrown at the start of a setup phase once the running request is canceled. Not a std::exception, so the
 *        setup's handlers of errors, which turn them into diagnostics, let it through.
 */
struct Canceled {};

/** @brief What mallard check finds of a case: its report, and the key catalog of its setup. */
struct CheckResult {
    CheckReport report;
    std::string catalog;  // key_catalog, as mallard check --catalog writes it
};

/** @brief The case a session holds, and what its setups keep between requests. */
class Context {
    public:
        /** @param resident keep mesh files and results between requests (the session), or not (a one-shot) */
        explicit Context(bool resident = false);
        ~Context();
        Context(const Context &) = delete;
        Context & operator=(const Context &) = delete;

        /** @brief Opens the input file, or text standing for it. @throws Error if the file cannot be read */
        void open(const std::string & path, const std::optional<std::string> & text = std::nullopt);
        /**
         * @brief Replaces the case's text. @return the key paths whose values differ ("run.cfl",
         *        "boundaries[1].T"); a whole table or array of tables when it is added or removed
         */
        std::vector<std::string> update(const std::string & text);
        void close();
        bool is_open() const { return opened; }
        const std::string & path() const { return input_path; }
        /** @brief The case's input, parsed anew with no key read. @throws InputError for TOML errors */
        Input fresh_input() const;
        /** @brief Mesh files read from disk by this context's setups; 0 for a one-shot's. */
        uint64_t mesh_files_read() const;
        bool resident() const { return keep; }
        /** @brief The stages of earlier setups, which a check sets up again only where its reads changed; nullptr
         *         for a one-shot. */
        SetupCache * setup_cache() const { return setups.get(); }

        /** @brief Results of requests on the current text, by request and parameters (resident only). */
        std::optional<std::string> remembered(const std::string & key) const;
        void remember(const std::string & key, const std::string & result);
        /** @brief The check of the current text with these options (key), before the environment's diagnostics, if
         *         one ran (resident only): case.check and case.keys share it. */
        const CheckResult * checked(const std::string & key) const;
        void remember_check(const std::string & key, const CheckResult & result);

    private:
        bool keep;
        bool opened = false;
        std::string input_path;
        std::optional<std::string> text;
        std::unique_ptr<MeshFileCache> mesh_files;
        std::unique_ptr<SetupCache> setups;
        std::vector<std::pair<std::string, std::string>> results;
        std::optional<std::pair<std::string, CheckResult>> last_check;
};

/**
 * @brief The work of mallard check and of case.check, case.keys and mallard list keys INPUT: sets the context's
 *        case up without running it, with the environment's diagnostics when environment. Collective; MPI and
 *        Kokkos started. Setup text goes to std::cout, setup warnings into the report.
 */
CheckResult check_case(Context & context, const CheckOptions & options, const std::string & restart = "",
                       bool environment = true);

/** @brief A request's method: its name, the one-shot command that runs the same handler, and its result's schema. */
struct Method {
    std::string_view name;     ///< "case.check"
    std::string_view command;  ///< its one-shot ("mallard check INPUT --format json"), or empty for session plumbing
    std::string_view result;   ///< schema of the result ("mallard-check"), or empty when null
    bool needs_case = false;
    /** @brief Runs the request. @return its result as JSON text. @throws Error */
    std::string (*handler)(Context &, const json::Value & params) = nullptr;
};

/** @brief Every method, in the order of docs/guide/session.md. */
std::span<const Method> methods();

/** @brief The method of this name, or nullptr. */
const Method * method(std::string_view name);

/**
 * @brief Answers requests read from in, one JSON-RPC message per line, with responses and notifications written
 *        to out, until shutdown or the end of in. MPI and Kokkos must be started; std::cout should not be out
 *        (setup text goes to it). @return the exit status
 */
int serve(std::istream & in, std::ostream & out, Context & context);

/** @brief A string member of params, else fallback. @throws Error for a member of another type */
std::string string_param(const json::Value & params, const std::string & name, const std::string & fallback = "");

/** @brief A boolean member of params, else fallback. @throws Error for a member of another type */
bool bool_param(const json::Value & params, const std::string & name, bool fallback = false);

} // namespace session

#endif // SESSION_H
