/**
 * @file particle_quantities.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Per-particle quantities: reading [[particles.quantities]] and
 *        compiling particle expressions.
 * @version 0.1
 * @date 2026-11-20
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "particle_quantities.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <limits>
#include <regex>

#include "log.h"
#include "region_input.h"

namespace particles {

std::vector<std::string> base_variables() {
    std::vector<std::string> v = {"x", "y"};
    if (N_DIM == 3) v.push_back("z");
    v.push_back("t");
    v.push_back("RHO");
    for (const char * u : {"U_X", "U_Y", "U_Z"}) {
        if (v.size() < VAR_U + N_DIM) v.push_back(u);
    }
    v.push_back("P");
    v.push_back("T");
    for (const char * u : {"V_X", "V_Y", "V_Z"}) {
        if (v.size() < VAR_V + N_DIM) v.push_back(u);
    }
    v.push_back("T_P");
    return v;
}

namespace {

constexpr const char * KINDS[] = {"integral", "max", "min", "first_time", "winding"};

const std::string DOCS = "docs/guide/particles.md#per-particle-quantities";

std::array<double, 3> read_vector(const Input & table, const std::string & key, const std::string & where,
                                  const Dimension & dimension) {
    const std::vector<rtype> v = find_real_vector(table, key, dimension);
    if (v.size() != N_DIM) {
        throw InputError(Diagnostic{.key = where + "." + key,
                                    .line = line_of(table, key),
                                    .message = "must have " + std::to_string(N_DIM) + " components",
                                    .docs = DOCS});
    }
    std::array<double, 3> out{0.0, 0.0, 0.0};
    for (size_t d = 0; d < N_DIM; d++) out[d] = static_cast<double>(v[d]);
    return out;
}

std::string upper(std::string s) {
    for (char & c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
}

/** @brief A quantity's name is a field name: an uppercase identifier, none of the variables (in any case) or the quantities before it. */
void check_name(const Input & e, const std::string & where, const std::string & name, const std::vector<std::string> & base,
                const std::vector<std::string> & before) {
    const auto error = [&](const std::string & message, const std::string & suggestion, const std::string & fix) {
        return InputError(Diagnostic{.key = where + ".name",
                                     .line = line_of(e, "name"),
                                     .message = message,
                                     .suggestion = suggestion,
                                     .fix = fix,
                                     .docs = DOCS});
    };
    const std::string upgrade = "; `mallard input upgrade old.toml > new.toml` rewrites a lowercase name and its references";
    if (!std::regex_match(name, std::regex("[A-Z][A-Z0-9_]*"))) {
        const std::string up = upper(name);
        const bool fixable = std::regex_match(up, std::regex("[A-Z][A-Z0-9_]*"));
        throw error("\"" + name + "\" is not an uppercase name: quantities are fields, named like RHO and U_X",
                    fixable ? "\"" + up + "\"" : "",
                    fixable ? "Name it \"" + up + "\"" + upgrade
                            : "Use uppercase letters, digits and _, starting with a letter");
    }
    for (const auto & b : base) {
        if (upper(b) == name) {
            throw error("\"" + name + "\" is the particle variable " + b, "",
                        "Choose another name: " + b + " is already a variable of particle expressions");
        }
    }
    if (std::find(before.begin(), before.end(), name) != before.end()) {
        throw error("\"" + name + "\" names two quantities", "", "Give each quantity its own name");
    }
}

} // namespace

std::string Quantities::substitute(const std::string & where, const std::string & text) {
    static const std::regex in_call(R"(\bin\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\))");
    std::string out;
    auto last = text.cbegin();
    for (std::sregex_iterator it(text.begin(), text.end(), in_call), end; it != end; ++it) {
        const std::string name = (*it)[1].str();
        auto known = std::find(m_region_names.begin(), m_region_names.end(), name);
        if (known == m_region_names.end()) {
            if (m_region_names.size() == MAX_REGIONS) {
                throw InputError(Diagnostic{.key = where,
                                            .message = "reads in() of more than " + std::to_string(MAX_REGIONS) +
                                                       " regions in all particle expressions",
                                            .docs = DOCS});
            }
            const region::Region & r = m_regions->get(Input(name), where);
            if (!r.analytic() || r.moving()) {
                throw InputError(Diagnostic{.key = where,
                                            .message = "in(" + name + ") needs a region of shapes that does not move",
                                            .fix = "Define " + name + " in [regions] from primitives (box, sphere, ...)",
                                            .docs = DOCS});
            }
            m_region_names.push_back(name);
            m_region_views.push_back(r.device());
            known = m_region_names.end() - 1;
        }
        out.append(last, (*it)[0].first);
        out += "in_" + *known;
        last = (*it)[0].second;
    }
    out.append(last, text.cend());
    return out;
}

std::vector<std::string> Quantities::variables(const uint32_t visible) const {
    std::vector<std::string> v = base_variables();
    // Every quantity has its slot, so that the regions' variables come after all of them
    for (uint32_t k = 0; k < m_count; k++) v.push_back(k < visible ? m_names[k] : "#" + std::to_string(k));
    for (const auto & r : m_region_names) v.push_back("in_" + r);
    return v;
}

kexpr::Program Quantities::compile(const std::string & where, const std::string & text) {
    const std::string substituted = substitute(where, text);
    kexpr::Program p(where, substituted, variables(m_count));
    if (!m_temperature && (p.uses(VAR_TEMPERATURE) || p.uses(VAR_TP))) {
        throw InputError(Diagnostic{.key = where,
                                    .message = "reads T or T_P, which particle expressions have for a single gas only",
                                    .docs = DOCS});
    }
    m_time = m_time || p.uses(VAR_T);
    return p;
}

void Quantities::init(const Input & table, region::Regions & regions, const uint32_t first_column, const bool temperature) {
    m_regions = &regions;
    m_temperature = temperature;
    if (!contains_optional(table, "quantities")) return;
    const auto entries = find_value<std::vector<Input>>(table, "quantities");
    if (entries.size() > MAX_QUANTITIES) {
        throw InputError(Diagnostic{.key = "particles.quantities",
                                    .line = line_of(table, "quantities"),
                                    .message = "has " + std::to_string(entries.size()) + " quantities; at most " +
                                               std::to_string(MAX_QUANTITIES) + " are allowed",
                                    .docs = DOCS});
    }
    m_count = static_cast<uint32_t>(entries.size());
    const std::vector<std::string> base = base_variables();
    // Every program is compiled before any is stored, with the names of the quantities before it
    for (size_t k = 0; k < entries.size(); k++) {
        const Input & e = entries[k];
        const std::string where = "particles.quantities[" + std::to_string(k) + "]";
        const std::string name = find_value<std::string>(e, "name");
        check_name(e, where, name, base, m_names);
        std::vector<std::string> given;
        for (const char * kind : KINDS) {
            if (contains_optional(e, kind)) given.push_back(kind);
        }
        if (given.size() != 1) {
            throw InputError(Diagnostic{.key = where,
                                        .line = line_of(e),
                                        .message = given.empty() ? "gives no kind"
                                                                 : "gives " + std::to_string(given.size()) + " kinds",
                                        .valid = {std::begin(KINDS), std::end(KINDS)},
                                        .fix = "Give exactly one of integral, max, min, first_time, winding",
                                        .docs = DOCS});
        }
        const std::string kind = given[0];
        QuantitySpec q;
        const uint32_t visible = static_cast<uint32_t>(k);
        const auto compile_visible = [&](const std::string & key, const std::string & text) {
            const std::string substituted = substitute(where + "." + key, text);
            kexpr::Program p(where + "." + key, substituted, variables(visible));
            if (!m_temperature && (p.uses(VAR_TEMPERATURE) || p.uses(VAR_TP))) {
                throw InputError(Diagnostic{.key = where + "." + key,
                                            .line = line_of(e, key),
                                            .message = "reads T or T_P, which particle expressions have for a single gas only",
                                            .docs = DOCS});
            }
            m_time = m_time || p.uses(VAR_T);
            m_programs.push_back(std::move(p));
            return static_cast<int32_t>(m_programs.size() - 1);
        };
        std::string text;
        if (kind == "winding") {
            q.kind = Accumulate::WINDING;
            const Input & w = e.at("winding");
            const std::array<double, 3> point = read_vector(w, "point", where + ".winding", dim::length);
            std::array<double, 3> axis{0.0, 0.0, 1.0};
            if (N_DIM == 3) axis = read_vector(w, "axis", where + ".winding", dim::none);
            const double norm = std::sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
            if (!(norm > 0.0)) {
                throw InputError(Diagnostic{.key = where + ".winding.axis", .line = line_of(w, "axis"),
                                            .message = "is zero", .docs = DOCS});
            }
            for (size_t i = 0; i < 3; i++) {
                q.point[i] = point[i];
                q.axis[i] = axis[i] / norm;
            }
            text = "winding about (" + logging::real(point[0]) + ", " + logging::real(point[1]) +
                   (N_DIM == 3 ? ", " + logging::real(point[2]) + "), axis (" + logging::real(q.axis[0]) + ", " +
                                     logging::real(q.axis[1]) + ", " + logging::real(q.axis[2])
                               : std::string()) +
                   ")";
        } else {
            text = find_value<std::string>(e, kind);
            q.program = compile_visible(kind, text);
            q.kind = kind == "integral" ? Accumulate::INTEGRAL
                     : kind == "max"    ? Accumulate::MAX
                     : kind == "min"    ? Accumulate::MIN
                                        : Accumulate::FIRST_TIME;
            if (q.kind == Accumulate::FIRST_TIME) m_time = true;
        }
        std::string gate;
        if (contains_optional(e, "while")) {
            gate = find_value<std::string>(e, "while");
            q.gate = compile_visible("while", gate);
        }
        if (q.kind == Accumulate::INTEGRAL || q.kind == Accumulate::WINDING) {
            q.column = first_column + m_integrated++;
        } else {
            q.column = m_aux++;
            m_initial_aux.push_back(q.kind == Accumulate::MAX   ? -std::numeric_limits<double>::infinity()
                                    : q.kind == Accumulate::MIN ? std::numeric_limits<double>::infinity()
                                                                : std::numeric_limits<double>::quiet_NaN());
        }
        m_specs.push_back(q);
        m_names.push_back(name);
        m_summary.push_back(name + ": " + kind + (kind == "winding" ? " " + text.substr(8) : " of " + text) +
                            (gate.empty() ? std::string() : " while " + gate));
    }
}

QuantityFunctor Quantities::functor() {
    if (m_built) return m_functor;
    m_built = true;
    std::vector<const kexpr::Program *> programs;
    for (const auto & p : m_programs) programs.push_back(&p);
    m_functor.programs = kexpr::ProgramTable<>(programs);
    Kokkos::View<QuantitySpec *> specs("particle_quantity_specs", m_specs.size());
    Kokkos::deep_copy(specs, Kokkos::View<const QuantitySpec *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
                                 m_specs.data(), m_specs.size()));
    m_functor.specs = specs;
    m_functor.n_regions = static_cast<uint32_t>(m_region_views.size());
    for (size_t r = 0; r < m_region_views.size(); r++) m_functor.regions[r] = m_region_views[r];
    return m_functor;
}

std::vector<std::string> Quantities::summary() const { return m_summary; }

} // namespace particles
