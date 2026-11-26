/**
 * @file session.cpp
 * @brief mallard session's case, its protocol loop and the work shared with mallard check.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "session.h"

#include <condition_variable>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>

#include "commands.h"
#include "diagnostic_code.h"
#include "interfaces.h"
#include "log.h"
#include "mesh_block.h"
#include "solver/solver.h"

namespace session {

namespace {

std::string error_text(const std::exception & e) {
    std::string text = e.what();
    if (text.rfind("[error] ", 0) == 0) text.erase(0, 8);
    while (!text.empty() && text.back() == '\n') text.pop_back();
    return text;
}

/** @brief Appends to out the key paths under path whose values differ between a and b. */
void differences(const Input & a, const Input & b, const std::string & path, std::vector<std::string> & out) {
    if (a.is_table() && b.is_table()) {
        const auto & ta = a.as_table().entries();
        const auto & tb = b.as_table().entries();
        std::set<std::string> keys;
        for (const auto & [k, v] : ta) keys.insert(k);
        for (const auto & [k, v] : tb) keys.insert(k);
        for (const std::string & k : keys) {
            const std::string sub = path.empty() ? k : path + "." + k;
            const auto ia = ta.find(k), ib = tb.find(k);
            if (ia == ta.end() || ib == tb.end()) {
                out.push_back(sub);
            } else {
                differences(ia->second, ib->second, sub, out);
            }
        }
        return;
    }
    if (a.is_array() && b.is_array() && a.as_array().size() == b.as_array().size()) {
        const auto & xa = a.as_array();
        const auto & xb = b.as_array();
        const bool tables = !xa.empty() && xa.front().is_table();
        if (tables) {
            for (size_t i = 0; i < xa.size(); i++) differences(xa[i], xb[i], path + "[" + std::to_string(i) + "]", out);
            return;
        }
    }
    if (!(a == b)) out.push_back(path);
}

} // namespace

Context::Context(bool resident) : keep(resident) {
    if (keep) mesh_files = std::make_unique<MeshFileCache>();
}

Context::~Context() = default;

void Context::open(const std::string & path, const std::optional<std::string> & text_in) {
    input_path = path;
    text = text_in;
    opened = true;
    results.clear();
}

std::vector<std::string> Context::update(const std::string & text_in) {
    std::vector<std::string> changed;
    try {
        const Input before = fresh_input();
        const Input after = parse_input_string(text_in, input_path);
        differences(before, after, "", changed);
    } catch (const std::exception &) {
        changed = {""};  // either side is not TOML: everything may differ
    }
    text = text_in;
    results.clear();
    return changed;
}

void Context::close() {
    opened = false;
    input_path.clear();
    text.reset();
    results.clear();
}

Input Context::fresh_input() const { return text ? parse_input_string(*text, input_path) : parse_input(input_path); }

bool Context::unsaved() const {
    if (!text) return false;
    std::ifstream in(input_path, std::ios::binary);
    return !in || std::string(std::istreambuf_iterator<char>(in), {}) != *text;
}

uint64_t Context::mesh_files_read() const { return mesh_files ? mesh_files->files_read() : 0; }

std::optional<std::string> Context::remembered(const std::string & key) const {
    for (const auto & [k, v] : results) {
        if (k == key) return v;
    }
    return std::nullopt;
}

void Context::remember(const std::string & key, const std::string & result) {
    if (keep) results.emplace_back(key, result);
}

CheckResult check_case(Context & context, const CheckOptions & options, const std::string & restart,
                       const bool environment) {
    CheckResult out;
    Solver solver;
    try {
        if (context.is_open()) {
            Input input = context.fresh_input();
            if (!restart.empty()) Solver::start_from_restart(input, restart);
            out.report = solver.check(input, options);
            out.catalog = key_catalog(resolve_input(solver.resolved_input()));
        }
    } catch (const InputError & e) {
        out.report.input_file = options.input_file;
        out.report.error(diagnostic_code::INPUT, options.input_file + ": " + error_text(e));
    }
    out.files = provenance::files_read();
    if (environment) check_environment(out.report);
    return out;
}

std::string case_files_json(const std::string & input, const CheckResult & result, const bool sha256) {
    namespace fs = std::filesystem;
    std::vector<provenance::FileRead> files = {{"input", input}};
    files.insert(files.end(), result.files.begin(), result.files.end());
    std::string list = "[";
    for (const provenance::FileRead & f : files) {
        std::error_code error;
        const bool exists = fs::is_regular_file(f.path, error);
        const std::uintmax_t bytes = exists ? fs::file_size(f.path, error) : 0;
        json::Object o;
        o.add("role", f.kind).add("path", f.path).add("resolved", fs::absolute(f.path, error).lexically_normal().string());
        o.add("exists", exists).raw("bytes", exists && !error ? std::to_string(bytes) : "null");
        o.raw("sha256", sha256 && exists ? json::quote(provenance::sha256_file(f.path)) : "null");
        list += (list.size() > 1 ? "," : "") + o.str();
    }
    return json::Object()
        .add("schema", interfaces::CASE_FILES.name)
        .add("version", interfaces::CASE_FILES.version)
        .add("input", input)
        .add("complete", result.report.setup_done)
        .raw("files", list + "]")
        .str();
}

std::string string_param(const json::Value & params, const std::string & name, const std::string & fallback) {
    const json::Value * v = params.find(name);
    if (!v || v->is_null()) return fallback;
    if (v->type != json::Value::Type::STRING) throw Error(code::INVALID_PARAMS, name + " takes a string");
    return v->string;
}

bool bool_param(const json::Value & params, const std::string & name, const bool fallback) {
    const json::Value * v = params.find(name);
    if (!v || v->is_null()) return fallback;
    if (v->type != json::Value::Type::BOOL) throw Error(code::INVALID_PARAMS, name + " takes true or false");
    return v->boolean;
}

namespace {

/** @brief A line read from the client, parsed, waiting to be answered. */
struct Pending {
    std::string id;  // JSON text of the request's id, empty for a notification
    std::string method;
    json::Value params;
    int error = 0;  // a message that is not a request: the error to answer
    std::string message;
};

/** @brief A line of the client as a request, or the error to answer it with. */
Pending parse(const std::string & line) {
    Pending p;
    json::Value message;
    try {
        message = json::parse(line);
    } catch (const std::exception & e) {
        p.id = "null";
        p.error = code::PARSE;
        p.message = std::string("not JSON: ") + e.what();
        return p;
    }
    const auto invalid = [&](const std::string & why) {
        p.id = p.id.empty() ? "null" : p.id;
        p.error = code::INVALID_REQUEST;
        p.message = why;
        return p;
    };
    if (message.type == json::Value::Type::ARRAY) return invalid("batches are not supported: one request per line");
    if (message.type != json::Value::Type::OBJECT) return invalid("a request is a JSON object");
    if (const json::Value * id = message.find("id")) {
        if (id->type != json::Value::Type::NUMBER && id->type != json::Value::Type::STRING) {
            return invalid("id is a number or a string");
        }
        p.id = json::dump(*id);
    }
    const json::Value * version = message.find("jsonrpc");
    if (!version || version->type != json::Value::Type::STRING || version->string != "2.0") {
        return invalid("jsonrpc must be \"2.0\"");
    }
    const json::Value * method = message.find("method");
    if (!method || method->type != json::Value::Type::STRING) return invalid("method is a string");
    p.method = method->string;
    if (const json::Value * params = message.find("params")) {
        if (params->type != json::Value::Type::OBJECT) return invalid("params is an object (by name)");
        p.params = *params;
    } else {
        p.params.type = json::Value::Type::OBJECT;
    }
    return p;
}

/**
 * @brief Requests read by a thread of their own, so that $/cancel reaches a request while it runs. The thread
 *        shares its state, and outlives the Reader if the input stays open after shutdown.
 */
class Reader {
    public:
        explicit Reader(std::istream & in) : state(std::make_shared<State>()) {
            std::thread([s = state, &in] { s->read(in); }).detach();
        }

        /** @brief The next request, or nothing at the end of the input. */
        std::optional<Pending> next() {
            State & s = *state;
            std::unique_lock lock(s.mutex);
            s.ready.wait(lock, [&] { return !s.queue.empty() || s.done; });
            if (s.queue.empty()) return std::nullopt;
            Pending p = std::move(s.queue.front());
            s.queue.pop_front();
            s.running = p.id;
            s.cancel_running = !p.id.empty() && s.canceled.erase(p.id) > 0;
            return p;
        }

        /** @brief Whether the request running was canceled. */
        bool canceled_now() {
            std::lock_guard lock(state->mutex);
            return state->cancel_running;
        }

    private:
        struct State {
            std::mutex mutex;
            std::condition_variable ready;
            std::deque<Pending> queue;
            std::set<std::string> canceled;
            std::string running;
            bool cancel_running = false;
            bool done = false;

            void read(std::istream & in) {
                std::string line;
                while (std::getline(in, line)) {
                    if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
                    Pending p = parse(line);
                    std::lock_guard lock(mutex);
                    if (p.method == "$/cancel" && p.id.empty() && p.error == 0) {
                        const json::Value * id = p.params.find("id");
                        if (!id) continue;
                        const std::string text = json::dump(*id);
                        if (text == running) {
                            cancel_running = true;
                        } else {
                            canceled.insert(text);
                        }
                        continue;
                    }
                    queue.push_back(std::move(p));
                    ready.notify_one();
                }
                std::lock_guard lock(mutex);
                done = true;
                ready.notify_one();
            }
        };

        std::shared_ptr<State> state;
};

void send(std::ostream & out, const std::string & message) { out << message << '\n' << std::flush; }

std::string error_message(const std::string & id, const int code, const std::string & message,
                          const std::string & data = "") {
    json::Object error;
    error.add("code", code).add("message", message);
    if (!data.empty()) error.raw("data", data);
    return json::Object().add("jsonrpc", "2.0").raw("id", id).raw("error", error.str()).str();
}

std::string notification(const std::string & method, const std::string & params) {
    return json::Object().add("jsonrpc", "2.0").add("method", method).raw("params", params).str();
}

/** @brief Warnings raised outside a result's diagnostics, as diagnostic objects. */
std::string warnings_json(const std::vector<std::pair<std::string, std::string>> & warnings) {
    std::vector<CheckDiagnostic> list;
    for (const auto & [c, m] : warnings) {
        CheckDiagnostic d;
        d.severity = report::Status::WARNING;
        d.code = c;
        d.message = m;
        list.push_back(d);
    }
    return diagnostics_json(list);
}

} // namespace

int serve(std::istream & in, std::ostream & out, Context & context) {
    Reader reader(in);
    while (std::optional<Pending> p = reader.next()) {
        const bool notification_only = p->id.empty();
        if (p->error != 0) {
            send(out, error_message(p->id, p->error, p->message));
            continue;
        }
        const Method * m = method(p->method);
        if (!m) {
            if (!notification_only) send(out, error_message(p->id, code::NO_METHOD, "no method " + p->method));
            continue;
        }
        if (reader.canceled_now()) {
            send(out, error_message(p->id, code::CANCELED, "canceled"));
            continue;
        }
        std::string result, failure;
        int failure_code = 0;
        std::string failure_data;
        logging::WarningCapture warnings;
        try {
            if (m->needs_case && !context.is_open()) throw Error(code::NO_CASE, "no case is open: send case.open first");
            const std::string id = p->id;
            logging::PhaseObserver phases([&](std::string_view phase) {
                if (reader.canceled_now()) throw Canceled{};
                if (!id.empty()) {
                    send(out, notification("$/progress",
                                           json::Object().raw("id", id).add("phase", std::string(phase)).str()));
                }
            });
            result = m->handler(context, p->params);
        } catch (const Canceled &) {
            failure_code = code::CANCELED;
            failure = "canceled";
        } catch (const Error & e) {
            failure_code = e.code;
            failure = e.what();
            failure_data = e.data;
        } catch (const std::exception & e) {
            failure_code = code::INTERNAL;
            failure = error_text(e);
        }
        if (!warnings.warnings.empty()) {
            send(out, notification("$/log", json::Object().raw("diagnostics", warnings_json(warnings.warnings)).str()));
        }
        if (!notification_only) {
            send(out, failure_code != 0 ? error_message(p->id, failure_code, failure, failure_data)
                                        : json::Object().add("jsonrpc", "2.0").raw("id", p->id).raw("result", result).str());
        }
        if (p->method == "shutdown" && failure_code == 0) break;
    }
    return 0;
}

} // namespace session
