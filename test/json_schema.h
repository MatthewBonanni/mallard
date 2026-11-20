/**
 * @file json_schema.h
 * @brief A validator of the JSON Schema subset that schemas/ uses, for the conformance tests of Mallard's
 *        machine-readable output.
 *
 * Keywords: type, properties, required, items, enum, const, $ref (to a file of schemas/, with an optional
 * #/$defs/NAME, or to #/$defs/NAME of the same file), $defs, oneOf, anyOf, additionalProperties (a boolean),
 * minimum, minItems. Annotations ($schema, $id, title, description, $comment) are ignored. Any other keyword is
 * an error of the schema, so a schema cannot ask for more than this validates.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef JSON_SCHEMA_H
#define JSON_SCHEMA_H

#include <map>
#include <string>
#include <vector>

#include "json.h"

namespace json_schema {

/** @brief The schemas of a directory, by file name; validates documents against one of them. */
class Schemas {
    public:
        /** @brief Reads every *.schema.json of dir. @throws std::runtime_error */
        explicit Schemas(const std::string & dir);
        /** @brief The file names read. */
        std::vector<std::string> files() const;
        /** @brief Where document breaks the schema of file, or of file#/$defs/NAME ("path: why"); empty if it conforms. */
        std::vector<std::string> validate(const std::string & file, const json::Value & document) const;
        /** @brief The keywords of file outside the subset, as "path: keyword"; empty if none. */
        std::vector<std::string> unsupported(const std::string & file) const;

    private:
        std::map<std::string, json::Value> schemas;
        void check(const std::string & file, const json::Value & schema, const json::Value & value,
                   const std::string & path, std::vector<std::string> & errors, int depth) const;
        const json::Value & resolve(const std::string & file, const std::string & ref, std::string & target) const;
};

} // namespace json_schema

#endif // JSON_SCHEMA_H
