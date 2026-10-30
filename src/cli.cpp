/**
 * @file cli.cpp
 * @brief Parser, dispatcher and help pages of the mallard commands.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "cli.h"

#include <algorithm>
#include <iostream>
#include <sstream>

#include "build_info.h"
#include "input.h"

namespace cli {

namespace {

constexpr size_t WIDTH = 80;
constexpr std::string_view DOCS = "https://matthewbonanni.github.io/mallard/docs/guide/cli/";
constexpr Option HELP = {"--help", "-h", "", "Show this page."};

bool is_help(std::string_view arg) { return arg == "-h" || arg == "--help"; }

std::vector<std::string_view> words(std::string_view name) {
    std::vector<std::string_view> out;
    while (!name.empty()) {
        const size_t space = name.find(' ');
        out.push_back(name.substr(0, space));
        name = space == std::string_view::npos ? std::string_view() : name.substr(space + 1);
    }
    return out;
}

/** @brief How many leading args spell name, or 0. */
size_t matches(std::string_view name, const std::vector<std::string> & args) {
    const auto w = words(name);
    if (w.empty() || w.size() > args.size()) return 0;
    for (size_t i = 0; i < w.size(); i++) {
        if (args[i] != w[i]) return 0;
    }
    return w.size();
}

bool is_prefix(const Option & o) { return o.name.ends_with('*'); }

const Option * find_option(const Command & command, std::string_view arg) {
    for (const Option & o : command.options) {
        if (arg == o.name || (!o.alias.empty() && arg == o.alias)) return &o;
        if (is_prefix(o) && arg.starts_with(o.name.substr(0, o.name.size() - 1))) return &o;
    }
    return nullptr;
}

std::string replace_version(std::string_view text) {
    std::string out(text);
    const std::string token = "{version}";
    for (size_t pos = out.find(token); pos != std::string::npos; pos = out.find(token, pos)) {
        out.replace(pos, token.size(), mallard_version());
    }
    return out;
}

/** @brief Text wrapped to WIDTH, every line indented; paragraphs (blank-line separated) kept apart. */
std::string wrap(std::string_view text, size_t indent, size_t first_column = 0) {
    std::string out;
    size_t column = first_column;
    bool first = true;
    std::istringstream lines{std::string(text)};
    std::string line;
    bool paragraph = false;
    while (std::getline(lines, line)) {
        if (line.empty()) {
            out += "\n\n";
            column = 0;
            first = true;
            paragraph = false;
            continue;
        }
        std::istringstream word_stream(line);
        std::string word;
        while (word_stream >> word) {
            if (first) {
                if (column < indent) out += std::string(indent - column, ' ');
                column = std::max(column, indent);
                first = false;
            } else if (column + 1 + word.size() > WIDTH) {
                out += "\n" + std::string(indent, ' ');
                column = indent;
            } else {
                out += ' ';
                column++;
            }
            out += word;
            column += word.size();
        }
        paragraph = true;
    }
    if (paragraph) out += "\n";
    return out;
}

std::string label(const Option & o) {
    std::string text = o.alias.empty() ? std::string(o.name) : std::string(o.alias) + ", " + std::string(o.name);
    if (!o.value.empty()) text += " " + std::string(o.value);
    return text;
}

void print_options(std::ostream & out, std::string_view heading, std::span<const Option> options) {
    if (options.empty()) return;
    size_t width = 0;
    for (const Option & o : options) width = std::max(width, label(o).size());
    width = std::min<size_t>(width, 24);
    out << "\n" << heading << ":\n";
    for (const Option & o : options) {
        const std::string text = "  " + label(o);
        out << text;
        const size_t column = width + 4;
        if (text.size() + 2 > column) {
            out << "\n";
            out << wrap(o.help, column);
        } else {
            out << wrap(o.help, column, text.size());
        }
    }
}

void print_lines(std::ostream & out, std::string_view heading, std::string_view lines) {
    if (lines.empty()) return;
    out << "\n" << heading << ":\n";
    std::istringstream in{replace_version(lines)};
    std::string line;
    while (std::getline(in, line)) out << (line.empty() ? "" : "  ") << line << "\n";
}

std::string command_label(const Command & c) { return c.name.empty() ? "mallard -i input.toml" : title(c); }

void print_commands(std::ostream & out, std::string_view family = {}) {
    size_t width = 0;
    for (const Command & c : commands()) width = std::max(width, command_label(c).size());
    out << "\nCommands:\n";
    for (const std::string_view group : groups()) {
        bool heading = false;
        for (const Command & c : commands()) {
            if (c.group != group) continue;
            if (!family.empty() && !c.name.starts_with(std::string(family) + " ")) continue;
            if (!heading && family.empty()) out << "  " << group << "\n";
            heading = true;
            const std::string text = (family.empty() ? "    " : "  ") + command_label(c);
            out << text << wrap(c.summary, width + 6, text.size());
        }
    }
}

void print_docs(std::ostream & out, const std::string & url) { out << "\nDocumentation: " << url << "\n"; }

int family_main(const Family & family, const std::vector<std::string> & rest) {
    const std::string name = "mallard " + std::string(family.name);
    if (!rest.empty() && std::none_of(rest.begin(), rest.end(), [](const std::string & a) { return is_help(a); })) {
        std::cerr << name << ": unknown command " << rest[0] << " (see " << name << " --help)\n";
        return 2;
    }
    print_help(rest.empty() ? std::cerr : std::cout, family);
    return rest.empty() ? 2 : 0;
}

} // namespace

std::string Args::value_or(std::string_view name, const std::string & fallback) const {
    const auto it = values.find(name);
    return it == values.end() ? fallback : it->second;
}

const Command * find(const std::vector<std::string> & args, size_t & words_used) {
    const Command * best = nullptr;
    words_used = 0;
    for (const Command & c : commands()) {
        const size_t n = matches(c.name, args);
        if (n > words_used) {
            best = &c;
            words_used = n;
        }
    }
    if (best) return best;
    if (!args.empty() && !args[0].starts_with('-')) return nullptr;
    return command("");
}

const Command * command(std::string_view name) {
    for (const Command & c : commands()) {
        if (c.name == name) return &c;
    }
    return nullptr;
}

bool parse(const Command & command, const std::vector<std::string> & args, Args & out, std::string & error) {
    out.help = std::any_of(args.begin(), args.end(), [](const std::string & a) { return is_help(a); });
    if (out.help) return true;
    for (size_t i = 0; i < args.size(); i++) {
        const std::string & arg = args[i];
        if (arg.size() < 2 || arg[0] != '-') {
            out.positional.push_back(arg);
            continue;
        }
        const size_t eq = arg.starts_with("--") ? arg.find('=') : std::string::npos;
        const std::string name = arg.substr(0, eq);
        const Option * option = find_option(command, name);
        if (!option) {
            error = "unknown option " + name;
            return false;
        }
        if (is_prefix(*option)) {
            out.passthrough.push_back(arg);
        } else if (option->value.empty()) {
            if (eq != std::string::npos) {
                error = "option " + std::string(option->name) + " takes no value";
                return false;
            }
            out.values[std::string(option->name)] = "";
        } else if (eq != std::string::npos) {
            out.values[std::string(option->name)] = arg.substr(eq + 1);
        } else if (i + 1 < args.size()) {
            out.values[std::string(option->name)] = args[++i];
        } else {
            error = "option " + std::string(option->name) + " needs a value, " + std::string(option->value);
            return false;
        }
    }
    const auto & spec = command.arguments;
    const bool variadic = !spec.empty() && spec.back().ends_with("...]");
    if (!variadic && out.positional.size() > spec.size()) {
        error = "unexpected argument " + out.positional[spec.size()];
        return false;
    }
    for (size_t i = out.positional.size(); i < spec.size(); i++) {
        if (!spec[i].starts_with('[')) {
            error = "missing " + std::string(spec[i]);
            return false;
        }
    }
    return true;
}

std::string title(const Command & command) {
    return command.name.empty() ? "mallard" : "mallard " + std::string(command.name);
}

std::string docs_url(const Command & command) {
    std::string anchor = "mallard";
    for (const std::string_view w : words(command.name)) {
        std::string_view word = w;
        while (word.starts_with('-')) word.remove_prefix(1);
        anchor += "-" + std::string(word);
    }
    return std::string(DOCS) + "#" + anchor;
}

void print_help(std::ostream & out, const Command & command) {
    std::istringstream synopsis{std::string(command.synopsis)};
    std::string line;
    bool first = true;
    while (std::getline(synopsis, line)) {
        out << (first ? "Usage: " : "       ") << line << "\n";
        first = false;
    }
    out << "\n" << replace_version(wrap(command.description, 0));
    if (command.name.empty()) print_commands(out);
    std::vector<Option> options(command.options.begin(), command.options.end());
    options.push_back(HELP);
    print_options(out, "Options", options);
    if (command.name.empty()) print_options(out, "Options of mallard before any command", global_options());
    print_lines(out, "Examples", command.examples);
    print_lines(out, "Exit status", command.exit_status);
    if (command.name.empty()) out << "\nmallard <command> --help shows the page of a command.\n";
    print_docs(out, docs_url(command));
}

void print_help(std::ostream & out, const Family & family) {
    const std::string name = "mallard " + std::string(family.name);
    out << "Usage: " << name << " <command> [options]\n\n" << wrap(family.summary, 0);
    print_commands(out, family.name);
    out << "\n" << name << " <command> --help shows the page of a command.\n";
    print_docs(out, std::string(DOCS) + "#mallard-" + std::string(family.name));
}

int usage_error(const Command & command, const std::string & message) {
    const std::string name = title(command);
    std::cerr << name << ": " << message << " (see " << name << " --help)\n";
    return 2;
}

int main(int argc, char ** argv) {
    const std::vector<std::string> args(argv + 1, argv + argc);
    size_t n = 0;
    const Command * command = find(args, n);
    if (!command) {
        for (const Family & f : families()) {
            if (args[0] == f.name) return family_main(f, {args.begin() + 1, args.end()});
        }
        std::vector<std::string> names;
        for (const Command & c : commands()) {
            const std::string first(words(c.name).empty() ? "" : words(c.name)[0]);
            if (!first.empty() && std::find(names.begin(), names.end(), first) == names.end()) names.push_back(first);
        }
        const std::string near = suggest_value(args[0], names);
        if (!near.empty()) {
            std::cerr << "mallard: unknown command " << args[0] << " (did you mean " << near << "?)\n";
        } else {
            std::cerr << "mallard: unknown command " << args[0] << "; to run an input file, give it with -i: mallard -i "
                      << args[0] << " (see mallard --help)\n";
        }
        return 2;
    }
    Args parsed;
    parsed.argc = argc;
    parsed.argv = argv;
    std::string error;
    if (!parse(*command, {args.begin() + std::ptrdiff_t(n), args.end()}, parsed, error)) {
        return usage_error(*command, error);
    }
    if (parsed.help) {
        print_help(std::cout, *command);
        if (!command->run_after_help) return 0;
        std::cout << std::flush;
    } else if (command->name.empty() && args.empty()) {
        print_help(std::cerr, *command);
        return 2;
    }
    return command->run(parsed);
}

} // namespace cli
