/**
 * @file cli.h
 * @brief The commands of the mallard executable: one table that the parser,
 *        the dispatcher, the help pages and the CLI docs all read.
 *
 * Layout and rules: docs/design/cli.md.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef CLI_H
#define CLI_H

#include <cstddef>
#include <map>
#include <ostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace cli {

/**
 * @brief An option a command accepts.
 */
struct Option {
    std::string_view name;   ///< "--ranks"; a name ending in '*' accepts every option with that prefix
    std::string_view alias;  ///< short form ("-l"), or empty
    std::string_view value;  ///< what the value is ("N"), or empty for a flag
    std::string_view help;   ///< what it does and its default
};

/**
 * @brief How the front end (`mallard`) finds the dimension of the solver a command runs on.
 */
enum class Case {
    none,           ///< any solver
    input,          ///< the input file: -i FILE, else the first argument
    template_name,  ///< the case template named by the first argument
    mesh,           ///< the mesh file, or input file, named by the first argument
    front_end,      ///< runs in the front end itself
};

/**
 * @brief The arguments of a command, parsed against its options.
 */
struct Args {
    std::vector<std::string> positional;                     ///< arguments that are not options
    std::map<std::string, std::string, std::less<>> values;  ///< by option name; "" for a flag
    std::vector<std::string> passthrough;                     ///< prefix options ("--gtest_*"), in order
    bool help = false;                                        ///< -h or --help was given
    int argc = 0;              ///< the process's arguments, for MPI and Kokkos
    char ** argv = nullptr;

    /** @brief Whether the option was given. */
    bool has(std::string_view name) const { return values.find(name) != values.end(); }
    /** @brief The option's value, else fallback. */
    std::string value_or(std::string_view name, const std::string & fallback) const;
};

/**
 * @brief A command: its help page, its options and positional arguments, and what runs it.
 */
struct Command {
    std::string_view name;         ///< "check", "mesh convert"; empty for the run (`mallard -i input.toml`)
    std::string_view group;        ///< heading of the top-level page (one of groups())
    std::string_view summary;      ///< one line on the top-level page
    std::string_view synopsis;     ///< usage lines, one per line
    std::string_view description;  ///< paragraphs, separated by blank lines; wrapped when printed
    std::span<const Option> options;
    std::span<const std::string_view> arguments;  ///< "INPUT", "[DIRECTORY]", "[KEY=VALUE...]"
    std::string_view examples;     ///< lines: "$ command" and comment lines
    std::string_view exit_status;  ///< lines
    Case dim = Case::none;
    int (*run)(const Args &) = nullptr;
    bool run_after_help = false;   ///< --help prints the page, then runs (`mallard test`: GoogleTest's page)
};

/**
 * @brief A noun that groups commands (`mallard mesh convert`).
 */
struct Family {
    std::string_view name;
    std::string_view summary;
};

/** @brief Every command, in the order of the top-level page within each group. */
std::span<const Command> commands();
/** @brief Command families. */
std::span<const Family> families();
/** @brief Groups of the top-level page, in order. */
std::span<const std::string_view> groups();
/** @brief Options of the front end, `mallard`, before any command. */
std::span<const Option> global_options();

/**
 * @brief The command the arguments name: the longest run of leading words
 *        that is a command name; the run command when they name none.
 * @param words set to the number of words naming it.
 * @return nullptr when the first argument is a word that names no command.
 */
const Command * find(const std::vector<std::string> & args, size_t & words);

/** @brief The command of this name, or nullptr. */
const Command * command(std::string_view name);

/**
 * @brief Parses the arguments after a command's name.
 * @param error set to a one-line message on a usage error.
 * @return false on a usage error.
 */
bool parse(const Command & command, const std::vector<std::string> & args, Args & out, std::string & error);

/** @brief "mallard check" (or "mallard" for the run). */
std::string title(const Command & command);

/** @brief The page of the command in the online docs. */
std::string docs_url(const Command & command);

/** @brief Prints the command's help page (the top-level page for the run). */
void print_help(std::ostream & out, const Command & command);

/** @brief Prints the page of a command family. */
void print_help(std::ostream & out, const Family & family);

/**
 * @brief Prints "<title>: <message> (see <title> --help)" to stderr.
 * @return 2, the exit status of a usage error.
 */
int usage_error(const Command & command, const std::string & message);

/**
 * @brief Runs the command that argv names: its help page, a usage error, or the command.
 * @return Exit status.
 */
int main(int argc, char ** argv);

} // namespace cli

#endif // CLI_H
