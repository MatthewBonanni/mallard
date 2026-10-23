/**
 * @file case_template.h
 * @brief `mallard new`: ready-to-run case directories from the examples and tutorials.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef CASE_TEMPLATE_H
#define CASE_TEMPLATE_H

#include <string>

namespace case_template {

/**
 * @brief Entry point of `mallard new`; argv holds the arguments after "new".
 */
int new_main(int argc, char ** argv);

/**
 * @brief Sets `key = value` in every [table] or [[table]] of a TOML text
 * that defines the key, keeping everything else (comments included); when
 * none does, adds it to each of them.
 * @param path Dotted path: "run.t_stop", "numerics.face_reconstruction.order".
 * @param value The new value as TOML text.
 */
std::string set_key(const std::string & text, const std::string & path, const std::string & value);

/**
 * @brief Rewrites an example's paths relative to the repository
 * (../../tools, ../../mechanisms/, ../../build/src/mallard) for a case
 * directory holding its own tools/ and mechanisms/.
 */
std::string relocate(const std::string & text);

} // namespace case_template

#endif // CASE_TEMPLATE_H
