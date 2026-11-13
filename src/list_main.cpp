/**
 * @file list_main.cpp
 * @brief mallard list fields|species|keys|templates: the catalogs, in general or for one case.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "build_info.h"
#include "case_template.h"
#include "check.h"
#include "cli.h"
#include "comm.h"
#include "commands.h"
#include "derived_catalog.h"
#include "diagnostic_code.h"
#include "field_units.h"
#include "input.h"
#include "json.h"
#include "log.h"
#include "mesh_quality.h"
#include "solver/solver.h"
#include "stencils.h"

namespace {

/** @brief What a case's setup gives: its output fields, species or key catalog, or the errors that stopped it. */
struct CaseFields {
    std::vector<std::string> fields, species, errors;
    std::string keys;  // key_catalog
};

/** @brief Runs body(solver, out) on a new Solver, with Kokkos, without its output, and collects its errors. */
template <typename F>
CaseFields quietly(const std::string & input_file, const cli::Args & args, F && body) {
    int argc = args.argc;
    char ** argv = args.argv;
    Kokkos::ScopeGuard kokkos(argc, argv);
    CaseFields out;
    // The setup summary is not part of the list
    std::ostringstream quiet;
    std::streambuf * cout = std::cout.rdbuf(quiet.rdbuf());
    try {
        Solver solver;
        body(solver, out);
    } catch (const InputError & e) {
        out.errors.push_back(input_file + ": " + e.what());
    } catch (const std::exception & e) {
        out.errors.push_back(e.what());
    }
    std::cout.rdbuf(cout);
    return out;
}

/** @brief The case's fields and species, from its input and mechanism without its mesh. */
CaseFields set_up(const std::string & input_file, const cli::Args & args) {
    return quietly(input_file, args, [&](Solver & solver, CaseFields & out) {
        out.fields = solver.init_fields(parse_input(input_file));
        out.species = solver.get_species_names();
    });
}

/** @brief The keys the case's setup reads: sets it up as mallard check does, without its report. */
CaseFields set_up_keys(const std::string & input_file, const cli::Args & args) {
    return quietly(input_file, args, [&](Solver & solver, CaseFields & out) {
        CheckOptions options;
        options.input_file = input_file;
        const CheckReport report = solver.check(parse_input(input_file), options);
        out.errors = report.errors();
        if (report.setup_done) out.keys = key_catalog(resolve_input(solver.resolved_input()));
    });
}

int report_errors(const CaseFields & c) {
    for (const auto & e : c.errors) logging::error(diagnostic_code::LIST, "mallard list: " + e);
    return 1;
}

std::string pattern_prefix(const std::string & name) { return name.substr(0, name.find('<')); }

/** @brief The catalog entries the case does not have, with what they need. */
std::vector<CatalogField> missing(const std::vector<std::string> & fields) {
    std::vector<CatalogField> out;
    for (const CatalogField & f : field_units::catalog()) {
        if (f.needs.empty() || f.group == "surface probes" || f.group == "mesh quality" || f.name == "REGION_FRACTION" ||
            f.name == "DIVERGED") {
            continue;
        }
        const std::string prefix = pattern_prefix(f.name);
        const bool pattern = prefix.size() < f.name.size();
        const bool present = std::any_of(fields.begin(), fields.end(), [&](const std::string & name) {
            return pattern ? name.rfind(prefix, 0) == 0 : name == f.name;
        });
        if (!present) out.push_back(f);
    }
    return out;
}

/**
 * @brief Where a field (or a pattern, <s> for a species) can be had: written by [[write_data]], sampled by
 *        [[probes]] or averaged by [statistics], as each of them decides, and derived by mallard post derive.
 */
std::vector<std::string> available(const std::string & pattern, const std::string & group,
                                   std::vector<std::string> species) {
    std::string name = pattern;
    if (const size_t at = name.find("<s>"); at != std::string::npos) {
        if (species.empty()) species = {"S"};
        name.replace(at, 3, species[0]);
    }
    const auto either = [&](auto && takes) { return takes(name) || takes(name + "_X"); };
    std::vector<std::string> out;
    const auto in = [&](const auto & names) { return std::find(names.begin(), names.end(), name) != names.end(); };
    const bool mesh_field = group == "mesh quality" || group == "stencils";
    if (group != "surface probes" && name != "DIVERGED" &&
        (!mesh_field || in(MeshQuality::FIELDS) || in(stencils::FIELDS))) {
        out.emplace_back("write_data");
    }
    if (group == "statistics" ? name.rfind("SOURCE_", 0) != 0
                              : either([&](const std::string & n) { return probe::samples(n, species); })) {
        out.emplace_back("probes");
    }
    if (either([&](const std::string & n) {
            try {
                CellSampler::code(n, species, "");
                return true;
            } catch (const InputError &) {
                return false;
            }
        })) {
        out.emplace_back("statistics");
    }
    if (derived::find(name.c_str()) != derived::N_FIELDS) out.emplace_back("post derive");
    return out;
}

std::string joined(const std::vector<std::string> & words) {
    std::string out;
    for (const std::string & w : words) out += (out.empty() ? "" : ", ") + w;
    return out.empty() ? "-" : out;
}

std::string units_text(const std::string & units) { return units == "1" ? "dimensionless" : units; }

bool json_format(const cli::Args & args) { return args.value_or("--format", "text") == "json"; }

/** @brief A usage error for a --format other than text or json, else 0. */
int format_error(const cli::Args & args, const char * command) {
    const std::string f = args.value_or("--format", "text");
    if (f == "text" || f == "json") return 0;
    return cli::usage_error(*cli::command(command), "--format takes text or json, not " + f);
}

std::string count_text(size_t n) { return logging::count(n, "field"); }

std::string field_json(const std::string & name, const std::string & units, const std::string & long_name,
                       const std::string & group, const std::string & needs, bool post_derive,
                       const std::vector<std::string> & species) {
    json::Object o;
    o.add("name", name).add("units", units).add("long_name", long_name).add("group", group);
    if (!needs.empty()) o.add("needs", needs);
    if (post_derive) o.add("post_derive", true);
    std::string where = "[";
    for (const std::string & w : available(name, group, species)) where += (where.size() > 1 ? "," : "") + json::quote(w);
    o.raw("available", where + "]");
    return o.str();
}

/** @brief The line naming the fields of names that mallard post derive writes, or "" if none. */
std::string post_derive_text(const std::vector<std::string> & names) {
    std::string list;
    for (const std::string & name : names) {
        if (field_units::post_derive(name)) list += (list.empty() ? "" : ", ") + name;
    }
    return list.empty() ? "" : "mallard post derive writes, from snapshots and restart files: " + list + ".\n";
}

template <typename T, typename F>
std::string json_array(const std::vector<T> & items, F && entry) {
    std::string out = "[";
    for (const T & item : items) out += (out.size() > 1 ? "," : "") + entry(item);
    return out + "]";
}

int list_catalog(const cli::Args & args) {
    const std::vector<CatalogField> catalog = field_units::catalog();
    if (json_format(args)) {
        std::cout << json::Object()
                         .raw("input", "null")
                         .raw("fields", json_array(catalog, [](const CatalogField & f) {
                                  return field_json(f.name, f.units, f.long_name, f.group, f.needs, f.post_derive, {});
                              }))
                         .str()
                  << "\n";
        return 0;
    }
    std::vector<std::vector<std::string>> rows;
    for (size_t i = 0; i < catalog.size(); i++) {
        const CatalogField & f = catalog[i];
        rows.push_back({f.name, units_text(f.units), f.long_name, f.needs.empty() ? "every case" : f.needs,
                        joined(available(f.name, f.group, {}))});
        if (i + 1 == catalog.size() || catalog[i + 1].group != f.group) {
            logging::section("Fields: " + f.group, count_text(rows.size()));
            logging::table({"field", "units", "description", "needs", "available in"}, rows);
            rows.clear();
        }
    }
    std::vector<std::string> names;
    for (const CatalogField & f : catalog) names.push_back(f.name);
    std::cout << "\n" << post_derive_text(names) << "mallard list fields input.toml lists the fields of one case.\n";
    return 0;
}

} // namespace

int list_fields_main(const cli::Args & args) {
    int argc = args.argc;
    char ** argv = args.argv;
    comm::Session session(argc, argv);
    if (const int e = format_error(args, "list fields")) return e;
    if (args.positional.empty()) return list_catalog(args);
    const std::string & input_file = args.positional[0];
    const CaseFields c = set_up(input_file, args);
    if (!c.errors.empty() && c.fields.empty()) return report_errors(c);
    const std::vector<CatalogField> absent = missing(c.fields);
    if (json_format(args)) {
        std::cout << json::Object()
                         .add("input", input_file)
                         .add("dim", N_DIM)
                         .raw("fields", json_array(c.fields, [&](const std::string & name) {
                                  const FieldDescription d = field_units::describe(name);
                                  return field_json(name, d.units(), d.long_name, field_units::group(name), "",
                                                    field_units::post_derive(name), c.species);
                              }))
                         .raw("missing", json_array(absent, [&](const CatalogField & f) {
                                  return field_json(f.name, f.units, f.long_name, f.group, f.needs, f.post_derive,
                                                    c.species);
                              }))
                         .str()
                  << "\n";
        return 0;
    }
    std::vector<std::string> fields = c.fields;
    std::stable_sort(fields.begin(), fields.end(), [](const std::string & a, const std::string & b) {
        return field_units::group_rank(field_units::group(a)) < field_units::group_rank(field_units::group(b));
    });
    std::vector<std::vector<std::string>> rows;
    for (size_t i = 0; i < fields.size(); i++) {
        const FieldDescription d = field_units::describe(fields[i]);
        const std::string g = field_units::group(fields[i]);
        rows.push_back({fields[i], units_text(d.units()), d.long_name, joined(available(fields[i], g, c.species))});
        if (i + 1 == fields.size() || field_units::group(fields[i + 1]) != g) {
            logging::section(input_file + ": " + g, count_text(rows.size()));
            logging::table({"field", "units", "description", "available in"}, rows);
            rows.clear();
        }
    }
    if (!absent.empty()) {
        logging::section("Not in this case", count_text(absent.size()));
        for (const CatalogField & f : absent) rows.push_back({f.name, units_text(f.units), f.needs});
        logging::table({"field", "units", "needs"}, rows);
    }
    std::cout << "\n" << post_derive_text(fields)
              << "Vectors also go by their name without _X, _Y, _Z (U); a trailing * selects every field it starts "
                 "(Y_*).\n";
    return 0;
}

int list_species_main(const cli::Args & args) {
    int argc = args.argc;
    char ** argv = args.argv;
    comm::Session session(argc, argv);
    if (const int e = format_error(args, "list species")) return e;
    const std::string & input_file = args.positional[0];
    const CaseFields c = set_up(input_file, args);
    if (!c.errors.empty() && c.fields.empty()) return report_errors(c);
    if (json_format(args)) {
        std::cout << json::Object()
                         .add("input", input_file)
                         .raw("species", json_array(c.species, [](const std::string & s) { return json::quote(s); }))
                         .str()
                  << "\n";
        return 0;
    }
    if (c.species.empty()) {
        std::cout << input_file << ": a single gas, no species (a mixture has [physics] gas = \"mixture\").\n";
        return 0;
    }
    logging::section(input_file + ": species", std::to_string(c.species.size()) + " species");
    std::vector<std::vector<std::string>> rows;
    for (size_t i = 0; i < c.species.size(); i++) rows.push_back({std::to_string(i), c.species[i]});
    logging::table({"#", "species"}, rows);
    return 0;
}

int list_templates_main(const cli::Args &) { return case_template::new_case({}, true, false, false); }

namespace {

/** @brief A key path from its catalog file name, which percent-encodes it (tools/dev/input_catalog.py). */
std::string key_of_file(const std::string & stem) {
    std::string key;
    for (size_t i = 0; i < stem.size(); i++) {
        if (stem[i] == '%' && i + 2 < stem.size()) {
            key += static_cast<char>(std::stoi(stem.substr(i + 1, 2), nullptr, 16));
            i += 2;
        } else {
            key += stem[i];
        }
    }
    return key;
}

}  // namespace

json::Value general_key_catalog() {
    for (const std::filesystem::path & dir : data_dirs()) {
        const std::filesystem::path catalog = dir / "docs" / "input-keys";
        if (!std::filesystem::is_directory(catalog)) continue;
        std::vector<std::pair<std::string, std::filesystem::path>> files;
        for (const auto & entry : std::filesystem::directory_iterator(catalog)) {
            if (entry.path().extension() != ".json") continue;
            files.emplace_back(key_of_file(entry.path().stem().string()), entry.path());
        }
        std::sort(files.begin(), files.end());
        json::Value out;
        out.type = json::Value::Type::OBJECT;
        for (const auto & [key, path] : files) {
            std::ifstream in(path);
            std::stringstream text;
            text << in.rdbuf();
            out.names.push_back(key);
            out.members.push_back(json::parse(text.str()));
        }
        return out;
    }
    throw std::runtime_error("the key catalog docs/input-keys is in none of Mallard's data directories (set "
                             "MALLARD_DATA_DIR to the source tree, or to the install tree's share/mallard)");
}

namespace {

std::string section_of(const std::string & key) { return key.substr(0, key.find('.')); }

bool in_section(const std::string & key, const std::string & section) {
    return section.empty() || key == section || key.starts_with(section + ".");
}

/** @brief A plain unit, else the unit of a dimension ("m" of "length (m)", "m^-1" as it is). */
std::string key_units(const json::Value & k) {
    if (const json::Value * unit = k.find("unit")) return unit->string;
    const json::Value * d = k.find("dimension");
    if (!d) return "";
    const size_t open = d->string.find('(');
    return open == std::string::npos ? d->string : d->string.substr(open + 1, d->string.size() - open - 2);
}

std::string key_default(const json::Value & k) {
    if (const json::Value * r = k.find("required"); r && r->boolean) return "required";
    if (const json::Value * d = k.find("default")) return d->string;
    std::string out;
    if (const json::Value * by_dim = k.find("defaults_by_dim")) {
        for (const std::string & dim : by_dim->keys()) {
            out += (out.empty() ? "" : ", ") + dim + "D: " + (*by_dim)[dim].string;
        }
    }
    return out;
}

std::string key_values(const json::Value & k) {
    std::string out;
    if (const json::Value * values = k.find("values")) {
        for (const json::Value & v : values->array) out += (out.empty() ? "" : ", ") + v.string;
    }
    return out;
}

/** @brief " (2D)" or " (3D)" for a key only one dimension's setup read. */
std::string only_dim(const json::Value & k) {
    const json::Value * dims = k.find("dims");
    return dims && dims->array.size() == 1 ? " (" + json::number(dims->array[0].number) + "D)" : "";
}

std::string value_text(const json::Value & v) {
    switch (v.type) {
        case json::Value::Type::BOOL: return v.boolean ? "true" : "false";
        case json::Value::Type::NUMBER: return json::number(v.number);
        case json::Value::Type::STRING: return json::quote(v.string);
        case json::Value::Type::ARRAY: return json_array(v.array, value_text);
        case json::Value::Type::OBJECT: {
            json::Object o;
            for (size_t i = 0; i < v.names.size(); i++) o.raw(v.names[i], value_text(v.members[i]));
            return o.str();
        }
        case json::Value::Type::NUL: break;
    }
    return "null";
}

std::string null_or(const std::string & text) { return text.empty() ? "null" : json::quote(text); }

} // namespace

int list_keys_main(const cli::Args & args) {
    int argc = args.argc;
    char ** argv = args.argv;
    comm::Session session(argc, argv);
    const cli::Command & command = *cli::command("list keys");
    if (const int e = format_error(args, "list keys")) return e;
    std::string section, input_file;
    for (const std::string & a : args.positional) {
        const bool file = a.ends_with(".toml") || std::filesystem::is_regular_file(a);
        std::string & slot = file ? input_file : section;
        if (!slot.empty()) return cli::usage_error(command, file ? "give one input file" : "give one section");
        slot = a;
    }
    section.erase(0, section.find_first_not_of('['));
    section.erase(section.find_last_not_of(']') + 1);

    json::Value catalog;
    if (input_file.empty()) {
        try {
            catalog = general_key_catalog();
        } catch (const std::exception & e) {
            logging::error(diagnostic_code::LIST, std::string("mallard list keys: ") + e.what());
            return 1;
        }
    } else {
        const CaseFields c = set_up_keys(input_file, args);
        if (c.keys.empty()) return report_errors(c);
        catalog = json::parse(c.keys);
    }
    std::vector<std::string> keys, sections;
    for (const std::string & key : catalog.keys()) {
        if (std::find(sections.begin(), sections.end(), section_of(key)) == sections.end()) {
            sections.push_back(section_of(key));
        }
        if (in_section(key, section)) keys.push_back(key);
    }
    if (keys.empty()) {
        const std::string near = suggest_value(section, sections);
        return cli::usage_error(command, (input_file.empty() ? "the catalog" : input_file) + " has no key in " +
                                             section + (near.empty() ? "" : " (did you mean " + near + "?)"));
    }

    const bool per_case = !input_file.empty();
    if (json_format(args)) {
        json::Object selected;
        for (const std::string & key : keys) selected.raw(key, value_text(catalog[key]));
        std::cout << json::Object()
                         .raw("input", null_or(input_file))
                         .raw("section", null_or(section))
                         .raw("keys", selected.str())
                         .str()
                  << "\n";
        return 0;
    }
    std::vector<std::vector<std::string>> rows;
    for (size_t i = 0; i < keys.size(); i++) {
        const json::Value & k = catalog[keys[i]];
        const json::Value * type = k.find("type");
        std::vector<std::string> row = {keys[i] + only_dim(k), type ? type->string : "", key_units(k), key_default(k),
                                        key_values(k)};
        if (per_case) row.insert(row.begin() + 1, k.find("set") && k["set"].boolean ? "yes" : "");
        rows.push_back(row);
        if (i + 1 == keys.size() || section_of(keys[i + 1]) != section_of(keys[i])) {
            logging::section((per_case ? input_file + ": [" : "Keys: [") + section_of(keys[i]) + "]",
                             logging::count(rows.size(), "key"));
            if (per_case) {
                logging::table({"key", "set", "type", "units", "default", "values"}, rows);
            } else {
                logging::table({"key", "type", "units", "default", "values"}, rows);
            }
            rows.clear();
        }
    }
    std::cout << (per_case ? "\nThe keys the setup of this case reads; mallard input resolve " + input_file +
                                 " writes the values it uses.\n"
                           : std::string("\nThe keys the examples and tutorials read; mallard list keys input.toml "
                                         "lists those of one case, and docs/input.md describes each.\n"));
    return 0;
}
