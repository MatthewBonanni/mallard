/**
 * @file list_main.cpp
 * @brief mallard list fields|species|templates: the catalogs, in general or for one case.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "case_template.h"
#include "check.h"
#include "cli.h"
#include "comm.h"
#include "commands.h"
#include "field_units.h"
#include "input.h"
#include "json.h"
#include "log.h"
#include "solver/solver.h"

namespace {

/** @brief What a case's setup gives: its output fields and species, or the errors that stopped it. */
struct CaseFields {
    std::vector<std::string> fields, species, errors;
};

/** @brief Sets the case up as mallard check does, without its report, and collects what it writes. */
CaseFields set_up(const std::string & input_file, const cli::Args & args) {
    int argc = args.argc;
    char ** argv = args.argv;
    Kokkos::ScopeGuard kokkos(argc, argv);
    CaseFields out;
    // The setup summary is not part of the list
    std::ostringstream quiet;
    std::streambuf * cout = std::cout.rdbuf(quiet.rdbuf());
    try {
        Solver solver;
        CheckOptions options;
        options.input_file = input_file;
        const CheckReport report = solver.check(parse_input(input_file), options);
        out.errors = report.errors;
        if (report.setup_done) {
            out.fields = solver.output_fields();
            out.species = solver.get_species_names();
        }
    } catch (const InputError & e) {
        out.errors.push_back(input_file + ": " + e.what());
    } catch (const std::exception & e) {
        out.errors.push_back(e.what());
    }
    std::cout.rdbuf(cout);
    return out;
}

int report_errors(const CaseFields & c) {
    for (const auto & e : c.errors) std::cerr << "mallard list: " << e << "\n";
    return 1;
}

std::string pattern_prefix(const std::string & name) { return name.substr(0, name.find('<')); }

/** @brief The catalog entries the case does not have, with what they need. */
std::vector<CatalogField> missing(const std::vector<std::string> & fields) {
    std::vector<CatalogField> out;
    for (const CatalogField & f : field_units::catalog()) {
        if (f.needs.empty() || f.group == "surface probes" || f.group == "mesh quality" || f.name == "REGION_FRACTION") {
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

std::string padded(std::string text, size_t width) {
    if (text.size() < width) text.resize(width, ' ');
    return text;
}

std::string field_json(const std::string & name, const std::string & units, const std::string & long_name,
                       const std::string & group, const std::string & needs) {
    json::Object o;
    o.add("name", name).add("units", units).add("long_name", long_name).add("group", group);
    if (!needs.empty()) o.add("needs", needs);
    return o.str();
}

template <typename T, typename F>
std::string json_array(const std::vector<T> & items, F && entry) {
    std::string out = "[";
    for (const T & item : items) out += (out.size() > 1 ? "," : "") + entry(item);
    return out + "]";
}

int list_catalog(const cli::Args & args) {
    const std::vector<CatalogField> catalog = field_units::catalog();
    if (args.has("--json")) {
        std::cout << json::Object()
                         .raw("input", "null")
                         .raw("fields", json_array(catalog, [](const CatalogField & f) {
                                  return field_json(f.name, f.units, f.long_name, f.group, f.needs);
                              }))
                         .str()
                  << "\n";
        return 0;
    }
    std::string group;
    for (const CatalogField & f : catalog) {
        if (f.group != group) logging::section("Fields: " + (group = f.group));
        std::string text = padded(f.units, 16) + f.long_name;
        if (!f.needs.empty()) text += "; needs " + f.needs;
        logging::item(f.name, text);
    }
    std::cout << "\nmallard list fields input.toml lists the fields of one case.\n";
    return 0;
}

} // namespace

int list_fields_main(const cli::Args & args) {
    int argc = args.argc;
    char ** argv = args.argv;
    comm::Session session(argc, argv);
    if (args.positional.empty()) return list_catalog(args);
    const std::string & input_file = args.positional[0];
    const CaseFields c = set_up(input_file, args);
    if (!c.errors.empty() && c.fields.empty()) return report_errors(c);
    const std::vector<CatalogField> absent = missing(c.fields);
    if (args.has("--json")) {
        std::cout << json::Object()
                         .add("input", input_file)
                         .add("dim", N_DIM)
                         .raw("fields", json_array(c.fields, [](const std::string & name) {
                                  const FieldDescription d = field_units::describe(name);
                                  return field_json(name, d.units(), d.long_name, field_units::group(name), "");
                              }))
                         .raw("missing", json_array(absent, [](const CatalogField & f) {
                                  return field_json(f.name, f.units, f.long_name, f.group, f.needs);
                              }))
                         .str()
                  << "\n";
        return 0;
    }
    std::vector<std::string> fields = c.fields;
    std::stable_sort(fields.begin(), fields.end(), [](const std::string & a, const std::string & b) {
        return field_units::group_rank(field_units::group(a)) < field_units::group_rank(field_units::group(b));
    });
    std::string group = "-";
    for (const std::string & name : fields) {
        const FieldDescription d = field_units::describe(name);
        const std::string g = field_units::group(name);
        if (g != group) logging::section(input_file + ": " + (group = g));
        logging::item(name, padded(d.units(), 16) + d.long_name);
    }
    if (!absent.empty()) logging::section("Not in this case");
    for (const CatalogField & f : absent) logging::item(f.name, "needs " + f.needs);
    std::cout << "\nVectors also go by their name without _X, _Y, _Z (U); a trailing * selects every field it starts "
                 "(Y_*).\n";
    return 0;
}

int list_species_main(const cli::Args & args) {
    int argc = args.argc;
    char ** argv = args.argv;
    comm::Session session(argc, argv);
    const std::string & input_file = args.positional[0];
    const CaseFields c = set_up(input_file, args);
    if (!c.errors.empty() && c.fields.empty()) return report_errors(c);
    if (args.has("--json")) {
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
    for (const std::string & s : c.species) std::cout << s << "\n";
    return 0;
}

int list_templates_main(const cli::Args &) { return case_template::new_case({}, true, false, false); }
