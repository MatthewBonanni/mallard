/**
 * @file source_terms.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Source terms of [[sources]] and [source].
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "source_terms.h"

#include <array>
#include <iomanip>
#include <set>
#include <sstream>

#include "comm.h"
#include "input.h"
#include "region.h"

namespace sources {

namespace {

const std::array<std::string, N_FIXED> FIXED_NAMES = {
    "x", "y", "z", "t", "rho", "u_x", "u_y", "u_z", "p", "T",
    "mean_rho", "mean_rhou_x", "mean_rhou_y", "mean_rhou_z", "mean_rhoE", "mean_ke"};

/** @brief A species name as an expression variable: other characters than letters, digits and _ become _. */
std::string identifier(const std::string & name) {
    std::string s = name;
    for (char & c : s) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') c = '_';
    }
    return s;
}

std::string text_of(const toml::value & v, const std::string & where) {
    if (v.is_string()) return v.as_string();
    std::ostringstream s;
    s << std::setprecision(17) << as_double(v, where);
    return s.str();
}

bool reads_solution(const expr::Program & p, const uint16_t n_variables) {
    for (uint16_t i = RHO; i < n_variables; i++) {
        if (i >= MEAN_RHO && i < N_FIXED) continue;
        if (p.uses(i)) return true;
    }
    return false;
}

bool reads_means(const expr::Program & p) {
    for (uint16_t i = MEAN_RHO; i < N_FIXED; i++) {
        if (p.uses(i)) return true;
    }
    return false;
}

/**
 * @brief Adds the static field (if any) and the terms of every owned cell to
 *        a sink(c, column, value per unit volume); species terms also add
 *        their mass to column 0.
 */
template <typename Sink>
struct TermKernel {
    Kokkos::View<rtype **, Kokkos::LayoutRight> field;
    Kokkos::View<Term *> terms;
    expr::ProgramTable<> programs;
    Kokkos::View<double **, Kokkos::LayoutRight> weights;
    CellFields fields;
    Kokkos::View<double *> means;
    Kokkos::View<rtype> time;
    uint32_t n_species;
    Sink sink;

    KOKKOS_INLINE_FUNCTION void operator()(const uint32_t c) const {
        for (uint32_t col = 0; col < field.extent(1); col++) sink(c, col, double(field(c, col)));
        const CellVariables vars{fields, c, time.data() ? double(time()) : 0.0, means.data(), n_species};
        for (uint32_t j = 0; j < terms.extent(0); j++) {
            const Term & term = terms(j);
            const double w = term.weights < 0 ? 1.0 : weights(term.weights, c);
            if (w == 0.0) continue;
            const double v = programs(term.program, vars) * w * term.scale;
            sink(c, term.column, v);
            if (term.column >= N_CONSERVATIVE) sink(c, 0, v);
        }
    }
};

/** @brief Adds to the RHS: the flow block times the cell measure, species per unit volume. */
struct AddToRhs {
    StateView flow;
    SpeciesView rhoY;
    Kokkos::View<rtype *> measure;

    KOKKOS_INLINE_FUNCTION void operator()(const uint32_t c, const uint32_t col, const double v) const {
        if (col < N_CONSERVATIVE) {
            flow(c, col) += static_cast<rtype>(v) * measure(c);
        } else {
            rhoY(c, col - N_CONSERVATIVE) += static_cast<rtype>(v);
        }
    }
};

/** @brief Sums the values per unit volume into a (cell, column) field. */
struct Accumulate {
    Kokkos::View<rtype **, Kokkos::LayoutRight> out;

    KOKKOS_INLINE_FUNCTION void operator()(const uint32_t c, const uint32_t col, const double v) const {
        out(c, col) += static_cast<rtype>(v);
    }
};

} // namespace

SourceTerms::SourceTerms(const toml::value & input, const std::shared_ptr<Mesh> & mesh_,
                         const std::vector<std::string> & species, const CellFields & fields_) :
    mesh(mesh_), fields(fields_), n_species(static_cast<uint32_t>(species.size())) {
    std::vector<std::string> names(FIXED_NAMES.begin(), FIXED_NAMES.end());
    if constexpr (N_DIM == 2) names[U_Z] = names[MEAN_RHOU_Z] = "";
    for (const auto & s : species) names.push_back("Y_" + identifier(s));
    for (const auto & s : species) names.push_back("X_" + identifier(s));
    for (size_t i = N_FIXED; i < names.size(); i++) {
        for (size_t j = N_FIXED; j < i; j++) {
            if (names[i] == names[j]) throw InputError("sources: species " + names[i] + " are ambiguous as variables.");
        }
    }
    const uint16_t n_variables = static_cast<uint16_t>(names.size());
    const bool mixture = n_species > 0;
    const uint32_t n_owned = mesh->n_owned();

    std::vector<std::pair<const toml::value *, std::string>> entries;
    if (input.contains("source")) {
        const toml::value & s = input.at("source");
        if (s.contains("rho") || s.contains("rhou") || s.contains("rhoE")) entries.emplace_back(&s, "source");
    }
    if (input.contains("sources")) {
        if (!input.at("sources").is_array()) throw InputError("sources must be an array of tables: [[sources]].");
        const auto & list = input.at("sources").as_array();
        for (size_t i = 0; i < list.size(); i++) entries.emplace_back(&list[i], "sources[" + std::to_string(i) + "]");
    }
    m_empty = entries.empty();
    if (m_empty) return;

    double volume = 0.0;
    for (uint32_t c = 0; c < n_owned; c++) volume += double(mesh->h_cell_measure(c));
    volume = comm::allreduce(volume, comm::Op::SUM);

    std::vector<Term> static_terms, dyn_terms;
    std::vector<std::vector<double>> weight_rows;
    for (const auto & [entry, where] : entries) {
        const bool legacy = where == "source";
        if (!entry->is_table()) throw InputError(where + " must be a table.");
        if (!legacy) {
            static const std::set<std::string> keys = {"name", "region", "total", "rho", "rhou", "rhoE", "rhoY"};
            for (const auto & [key, value] : entry->as_table()) {
                if (!keys.count(key)) {
                    throw InputError(where + ": unknown key " + key +
                                     " (expected name, region, total, rho, rhou, rhoE, rhoY).");
                }
            }
        }
        const std::string label = entry->contains("name") ? toml::find<std::string>(*entry, "name") : where;
        const bool total = !legacy && toml::find_or<bool>(*entry, "total", false);
        int32_t row = -1;
        double integral = volume;
        std::string region_text = "everywhere";
        if (!legacy && entry->contains("region")) {
            const region::Region r(entry->at("region"), where + ".region");
            std::vector<double> fractions = region::volume_fractions(*mesh, r);
            double sum = 0.0;
            for (uint32_t c = 0; c < n_owned; c++) sum += fractions[c] * double(mesh->h_cell_measure(c));
            integral = comm::allreduce(sum, comm::Op::SUM);
            if (!(integral > 0.0)) throw InputError(where + ".region contains no part of any cell.");
            row = static_cast<int32_t>(weight_rows.size());
            weight_rows.push_back(std::move(fractions));
            std::ostringstream s;
            s << r.summary() << ", measure " << logging::real(integral);
            region_text = s.str();
        }

        // (column, expression, name)
        std::vector<std::tuple<uint32_t, std::string, std::string>> values;
        if (entry->contains("rho")) {
            if (mixture) {
                throw InputError(where + ".rho: in gas mixtures mass enters through rhoY (rho gains their sum).");
            }
            values.emplace_back(0u, text_of(entry->at("rho"), where + ".rho"), "rho");
        }
        if (entry->contains("rhou")) {
            const toml::value & v = entry->at("rhou");
            if (!v.is_array() || v.as_array().size() != N_DIM) {
                throw InputError(where + ".rhou must have " + std::to_string(N_DIM) + " components.");
            }
            FOR_I_DIM values.emplace_back(1u + i, text_of(v.as_array()[i], where + ".rhou"),
                                          "rhou_" + std::string(1, "xyz"[i]));
        }
        if (entry->contains("rhoE")) {
            values.emplace_back(uint32_t(N_DIM + 1), text_of(entry->at("rhoE"), where + ".rhoE"), "rhoE");
        }
        if (entry->contains("rhoY")) {
            if (!mixture) throw InputError(where + ".rhoY needs a gas mixture.");
            if (!entry->at("rhoY").is_table()) throw InputError(where + ".rhoY must be a table of species.");
            for (const auto & [name, value] : entry->at("rhoY").as_table()) {
                const auto k = std::find(species.begin(), species.end(), name);
                if (k == species.end()) throw InputError(where + ".rhoY: no species " + name + " in the mechanism.");
                values.emplace_back(uint32_t(N_CONSERVATIVE + (k - species.begin())), text_of(value, where + ".rhoY"),
                                    "rhoY_" + name);
            }
        }
        if (values.empty()) throw InputError(where + " sets none of rho, rhou, rhoE, rhoY.");

        std::string kinds;
        for (const auto & [column, text, name] : values) {
            expr::Program p(where + "." + name, text, names);
            if (total) {
                for (uint16_t i = 0; i < n_variables; i++) {
                    if (i != TIME && p.uses(i)) {
                        throw InputError(where + "." + name + ": with total = true, values may depend on t only (got " +
                                         names[i] + "); the region gives the distribution.");
                    }
                }
            }
            const bool time = p.uses(TIME), solution = reads_solution(p, n_variables), mean = reads_means(p);
            m_time = m_time || time;
            m_means = m_means || mean;
            Term term;
            term.program = static_cast<uint32_t>(programs.size());
            term.column = column;
            term.weights = row;
            term.scale = total ? 1.0 / integral : 1.0;
            (time || solution || mean ? dyn_terms : static_terms).push_back(term);
            programs.push_back(std::move(p));
            const std::string kind = mean ? "domain averages" : (solution ? "solution" : (time ? "t" : ""));
            if (!kind.empty() && kinds.find(kind) == std::string::npos) kinds += (kinds.empty() ? "" : ", ") + kind;
        }
        std::string text;
        for (const auto & v : values) text += (text.empty() ? "" : ", ") + std::get<2>(v);
        text += total ? " (integrated totals)" : " (per unit volume)";
        text += kinds.empty() ? ", steady" : ", of " + kinds;
        m_summary.emplace_back("Source " + label, text + "; " + region_text);
    }

    std::vector<const expr::Program *> pointers;
    for (const auto & p : programs) pointers.push_back(&p);
    table = expr::ProgramTable<>(pointers);
    const size_t n_rows = weight_rows.size();
    weights = Kokkos::View<double **, Kokkos::LayoutRight>("source_weights", n_rows, mesh->n_cells);
    auto h_weights = Kokkos::create_mirror_view(weights);
    for (size_t r = 0; r < n_rows; r++) {
        for (uint32_t c = 0; c < mesh->n_cells; c++) h_weights(r, c) = weight_rows[r][c];
    }
    Kokkos::deep_copy(weights, h_weights);
    means = Kokkos::View<double *>("source_means", N_MEANS);
    auto upload = [](const std::vector<Term> & terms, const std::string & label) {
        Kokkos::View<Term *> v(label, terms.size());
        Kokkos::deep_copy(v, Kokkos::View<const Term *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(terms.data(),
                                                                                                    terms.size()));
        return v;
    };
    dynamic_terms = upload(dyn_terms, "source_dynamic_terms");
    if (!static_terms.empty()) {
        const Kokkos::View<Term *> terms = upload(static_terms, "source_static_terms");
        static_field = Kokkos::View<rtype **, Kokkos::LayoutRight>("source_static_field", mesh->n_cells,
                                                                   N_CONSERVATIVE + n_species);
        Kokkos::parallel_for("source_static_field", n_owned,
                             TermKernel<Accumulate>{{}, terms, table, weights, fields, means, {}, n_species,
                                                    Accumulate{static_field}});
    }
}

void SourceTerms::update_means(const State & solution) {
    const StateView U = solution.flow;
    const Kokkos::View<rtype *> V = mesh->cell_measure;
    std::array<double, N_MEANS + 1> sums{};
    Kokkos::parallel_reduce("source_means", mesh->n_owned(),
                            KOKKOS_LAMBDA(const uint32_t c, double & v_sum, double & rho_sum, double & mx, double & my,
                                          double & mz, double & e_sum, double & k_sum) {
        const double v = double(V(c)), rho = double(U(c, 0));
        double m[3] = {0.0, 0.0, 0.0};
        FOR_I_DIM m[i] = double(U(c, 1 + i));
        v_sum += v;
        rho_sum += rho * v;
        mx += m[0] * v;
        my += m[1] * v;
        mz += m[2] * v;
        e_sum += double(U(c, N_DIM + 1)) * v;
        k_sum += 0.5 * (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]) / rho * v;
    }, sums[0], sums[1], sums[2], sums[3], sums[4], sums[5], sums[6]);
    sums = comm::allreduce(sums, comm::Op::SUM);
    auto h_means = Kokkos::create_mirror_view(means);
    for (int i = 0; i < N_MEANS; i++) h_means(i) = sums[size_t(i) + 1] / sums[0];
    Kokkos::deep_copy(means, h_means);
}

void SourceTerms::add(const State & solution, const State & rhs, const Kokkos::View<rtype> & stage_time) {
    if (m_means) update_means(solution);
    Kokkos::parallel_for("rhs_source_terms", mesh->n_owned(),
                         TermKernel<AddToRhs>{static_field, dynamic_terms, table, weights, fields, means, stage_time,
                                              n_species, AddToRhs{rhs.flow, rhs.species, mesh->cell_measure}});
}

Kokkos::View<rtype **, Kokkos::LayoutRight> SourceTerms::evaluate(const State & solution, const rtype t) {
    Kokkos::View<rtype **, Kokkos::LayoutRight> out("source_values", mesh->n_cells, N_CONSERVATIVE + n_species);
    if (m_empty) return out;
    if (m_means) update_means(solution);
    Kokkos::View<rtype> time("source_time");
    Kokkos::deep_copy(time, t);
    Kokkos::parallel_for("source_values", mesh->n_owned(),
                         TermKernel<Accumulate>{static_field, dynamic_terms, table, weights, fields, means, time,
                                                n_species, Accumulate{out}});
    return out;
}

} // namespace sources
