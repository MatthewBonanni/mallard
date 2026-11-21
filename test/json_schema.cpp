/**
 * @file json_schema.cpp
 * @brief The JSON Schema subset validator of the conformance tests.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "json_schema.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

namespace json_schema {

namespace {

using Type = json::Value::Type;

const std::set<std::string> & keywords() {
    static const std::set<std::string> k = {"type", "properties", "required", "items", "enum", "const", "$ref",
                                            "$defs", "oneOf", "anyOf", "additionalProperties", "minimum",
                                            "minItems", "$schema", "$id", "title", "description", "$comment"};
    return k;
}

bool equal(const json::Value & a, const json::Value & b) { return json::dump(a) == json::dump(b); }

bool has_type(const json::Value & v, const std::string & type) {
    switch (v.type) {
        case Type::NUL: return type == "null";
        case Type::BOOL: return type == "boolean";
        case Type::NUMBER: return type == "number" || (type == "integer" && std::floor(v.number) == v.number);
        case Type::STRING: return type == "string";
        case Type::ARRAY: return type == "array";
        case Type::OBJECT: return type == "object";
    }
    return false;
}

void collect_unsupported(const json::Value & schema, const std::string & path, std::vector<std::string> & out) {
    if (schema.type != Type::OBJECT) return;
    for (size_t i = 0; i < schema.names.size(); i++) {
        const std::string & k = schema.names[i];
        const json::Value & v = schema.members[i];
        if (!keywords().count(k)) out.push_back(path + ": " + k);
        if (k == "properties" || k == "$defs") {
            for (size_t j = 0; j < v.names.size(); j++) collect_unsupported(v.members[j], path + "/" + k + "/" + v.names[j], out);
        } else if (k == "items") {
            collect_unsupported(v, path + "/items", out);
        } else if (k == "oneOf" || k == "anyOf") {
            for (size_t j = 0; j < v.array.size(); j++) collect_unsupported(v.array[j], path + "/" + k + "/" + std::to_string(j), out);
        } else if (k == "additionalProperties" && v.type != Type::BOOL) {
            out.push_back(path + ": additionalProperties other than a boolean");
        }
    }
}

} // namespace

Schemas::Schemas(const std::string & dir) {
    for (const auto & entry : std::filesystem::directory_iterator(dir)) {
        const std::string name = entry.path().filename().string();
        if (!name.ends_with(".schema.json")) continue;
        std::ifstream in(entry.path());
        std::stringstream text;
        text << in.rdbuf();
        try {
            schemas[name] = json::parse(text.str());
        } catch (const std::exception & e) {
            throw std::runtime_error(name + ": " + e.what());
        }
    }
    if (schemas.empty()) throw std::runtime_error("no *.schema.json in " + dir);
}

std::vector<std::string> Schemas::files() const {
    std::vector<std::string> out;
    for (const auto & [name, schema] : schemas) out.push_back(name);
    return out;
}

std::vector<std::string> Schemas::unsupported(const std::string & file) const {
    std::vector<std::string> out;
    collect_unsupported(schemas.at(file), "#", out);
    return out;
}

std::vector<std::string> Schemas::validate(const std::string & file, const json::Value & document) const {
    std::vector<std::string> errors;
    const std::string name = file.substr(0, file.find('#'));
    if (!schemas.count(name)) return {"no schema " + name};
    std::string target;
    check(name, resolve(name, file.substr(name.size()), target), document, "$", errors, 0);
    return errors;
}

const json::Value & Schemas::resolve(const std::string & file, const std::string & ref, std::string & target) const {
    const size_t hash = ref.find('#');
    target = hash == 0 || ref.empty() ? file : ref.substr(0, hash);
    const auto it = schemas.find(target);
    if (it == schemas.end()) throw std::runtime_error(file + ": $ref to an unknown file " + ref);
    if (hash == std::string::npos || hash + 1 == ref.size()) return it->second;
    const std::string pointer = ref.substr(hash + 1);
    const std::string prefix = "/$defs/";
    if (pointer.rfind(prefix, 0) != 0) throw std::runtime_error(file + ": $ref other than #/$defs/NAME: " + ref);
    const json::Value * defs = it->second.find("$defs");
    const json::Value * def = defs ? defs->find(pointer.substr(prefix.size())) : nullptr;
    if (!def) throw std::runtime_error(file + ": $ref to a missing definition " + ref);
    return *def;
}

void Schemas::check(const std::string & file, const json::Value & schema, const json::Value & value,
                    const std::string & path, std::vector<std::string> & errors, const int depth) const {
    if (depth > 64) throw std::runtime_error(file + ": $ref nested too deeply");
    if (schema.type == Type::BOOL) {
        if (!schema.boolean) errors.push_back(path + ": no value is allowed here");
        return;
    }
    if (const json::Value * ref = schema.find("$ref")) {
        std::string target;
        const json::Value & referenced = resolve(file, ref->string, target);
        check(target, referenced, value, path, errors, depth + 1);
    }
    if (const json::Value * type = schema.find("type")) {
        bool ok = false;
        if (type->type == Type::STRING) {
            ok = has_type(value, type->string);
        } else {
            for (const json::Value & t : type->array) ok = ok || has_type(value, t.string);
        }
        if (!ok) {
            errors.push_back(path + ": not of type " + json::dump(*type) + ": " + json::dump(value).substr(0, 80));
            return;
        }
    }
    if (const json::Value * c = schema.find("const"); c && !equal(*c, value)) {
        errors.push_back(path + ": not " + json::dump(*c));
    }
    if (const json::Value * e = schema.find("enum")) {
        if (std::none_of(e->array.begin(), e->array.end(), [&](const json::Value & x) { return equal(x, value); })) {
            errors.push_back(path + ": " + json::dump(value) + " is none of " + json::dump(*e));
        }
    }
    if (const json::Value * m = schema.find("minimum"); m && value.type == Type::NUMBER && value.number < m->number) {
        errors.push_back(path + ": below " + json::number(m->number));
    }
    if (value.type == Type::OBJECT) {
        if (const json::Value * required = schema.find("required")) {
            for (const json::Value & key : required->array) {
                if (!value.has(key.string)) errors.push_back(path + ": no member " + key.string);
            }
        }
        const json::Value * properties = schema.find("properties");
        const json::Value * additional = schema.find("additionalProperties");
        for (size_t i = 0; i < value.names.size(); i++) {
            const json::Value * sub = properties ? properties->find(value.names[i]) : nullptr;
            if (sub) {
                check(file, *sub, value.members[i], path + "." + value.names[i], errors, depth + 1);
            } else if (additional && additional->type == Type::BOOL && !additional->boolean) {
                errors.push_back(path + ": member " + value.names[i] + " is not allowed");
            }
        }
    }
    if (value.type == Type::ARRAY) {
        if (const json::Value * n = schema.find("minItems"); n && double(value.array.size()) < n->number) {
            errors.push_back(path + ": fewer than " + json::number(n->number) + " items");
        }
        if (const json::Value * items = schema.find("items")) {
            for (size_t i = 0; i < value.array.size(); i++) {
                check(file, *items, value.array[i], path + "[" + std::to_string(i) + "]", errors, depth + 1);
            }
        }
    }
    for (const char * combinator : {"oneOf", "anyOf"}) {
        const json::Value * options = schema.find(combinator);
        if (!options) continue;
        size_t matched = 0;
        for (const json::Value & option : options->array) {
            std::vector<std::string> sub;
            check(file, option, value, path, sub, depth + 1);
            matched += sub.empty();
        }
        const bool one = std::string(combinator) == "oneOf";
        if (one ? matched != 1 : matched == 0) {
            errors.push_back(path + ": matches " + std::to_string(matched) + " of the " + combinator + " schemas");
        }
    }
}

} // namespace json_schema
