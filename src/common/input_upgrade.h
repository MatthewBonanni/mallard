/**
 * @file input_upgrade.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Renamed, moved and removed input keys and options: warnings while
 *        they are deprecated, errors once removed, and `mallard input upgrade`,
 *        which rewrites an input file keeping its comments and layout.
 * @version 0.9
 * @date 2026-10-16
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef INPUT_UPGRADE_H
#define INPUT_UPGRADE_H

#include <string>
#include <vector>

#include "input.h"

namespace input_upgrade {

/**
 * @brief A key renamed, moved to another table, or removed. Paths are dotted;
 *        `name[]` stands for every table of the array of tables `name`, and
 *        `to` has the same `[]` steps as `from` (the same element).
 */
struct KeyRule {
    std::string from;
    std::string to;       ///< Empty: removed without replacement
    std::string since;    ///< Release that deprecated it (still read, with a warning)
    std::string removed;  ///< First release that rejects it, at least one minor release after `since`
    std::string note;     ///< What changed, for the warning and the rewritten file
    /// Converts the value's TOML text (e.g. "3" to "4"); nullptr keeps it
    std::string (*convert)(const std::string & toml_text) = nullptr;
};

/**
 * @brief A string option renamed: `key = "from"` becomes `key = "to"`. A kept
 *        option still does what it did while deprecated (the setup reads it
 *        unchanged); only the rewritten file switches to `to`.
 */
struct OptionRule {
    std::string key;
    std::string from, to;
    std::string since, removed, note;
    bool kept = false;
};

/**
 * @brief A key holding an array of strings replaced by elements of an array of
 *        tables, one per string: `from = ["a", "b"]` becomes `[[to]] key =
 *        convert("a")`, `[[to]] key = convert("b")`, after the elements
 *        already there. `to` is dotted, without `[]` steps.
 */
struct ArrayRule {
    std::string from;
    std::string to;
    std::string key;      ///< The key each new element sets
    std::string context;  ///< A string key of from's table the conversion depends on, or empty
    std::string since, removed, note;
    /// The value of `key` for one string, given the context's value (empty if
    /// absent); throws std::invalid_argument, saying why, for strings it does not take
    std::vector<std::string> (*convert)(const std::string & item, const std::string & context) = nullptr;
};

/** @brief Every rule, oldest first; docs/guide/input_files.md lists them. */
struct Rules {
    std::vector<KeyRule> keys;
    std::vector<OptionRule> options;
    std::vector<ArrayRule> arrays;
};
const Rules & rules();

/** @brief One deprecated or removed key of an input, in the shape of the setup's diagnostics. */
struct Finding {
    std::string key;         ///< Where it is, e.g. "boundaries[2].u_in"
    std::size_t line = 0;    ///< 0 if unknown
    std::string message;     ///< What changed and since when
    std::string suggestion;  ///< The key or option to use instead, or empty
    bool removed = false;    ///< Rejected by this release (else deprecated: read, with a warning)
};

/** @brief "line 12: run.foo is deprecated since 0.9.0 ...". */
std::string describe(const Finding & finding);

/** @brief describe() plus how to update the file, for the run's warnings. */
std::string advice(const Finding & finding);

/**
 * @brief Rewrites the deprecated keys and options of `input` in place, as the
 *        setup reads them (kept options stay), and returns them. Throws an InputError listing the
 *        removed ones, with what to do (`mallard input upgrade`). `version`
 *        is this release's (MALLARD_VERSION when empty).
 */
std::vector<Finding> apply(Input & input, const std::string & version = "", const Rules & table = rules());

/**
 * @brief The text of an input file with every rule applied (also removed
 *        ones), changing only the lines of those keys: a renamed key keeps its
 *        line, value and comment; a moved key goes to the end of its new table;
 *        a removed key is commented out with a note. Where that is not
 *        possible (dotted keys, inline tables), the whole file is reformatted,
 *        and `reformatted` is set. `findings` gets every change.
 */
std::string upgrade_text(const std::string & text, std::vector<Finding> & findings,
                         bool & reformatted, const Rules & table = rules());

/**
 * @brief `mallard input upgrade [--check] old.toml > new.toml`: writes the
 *        upgraded file to stdout and the changes to stderr; with `check`, only
 *        the changes, and exit status 1 if there are any. Returns the exit status.
 */
int upgrade_file(const std::string & file, bool check);

} // namespace input_upgrade

#endif // INPUT_UPGRADE_H
