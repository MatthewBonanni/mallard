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

#include <algorithm>
#include <array>
#include <sstream>

#include "comm.h"
#include "device_sums.h"
#include "input.h"
#include "region_input.h"
#include "region_motion.h"

namespace sources {

namespace {

const std::array<std::string, N_FIXED> FIXED_NAMES = {
    "x", "y", "z", "t", "rho", "u_x", "u_y", "u_z", "p", "T",
    "mean_rho", "mean_rhou_x", "mean_rhou_y", "mean_rhou_z", "mean_rhoE", "mean_ke", "copy"};

/** @brief A species name as an expression variable: other characters than letters, digits and _ become _. */
std::string identifier(const std::string & name) {
    std::string s = name;
    for (char & c : s) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') c = '_';
    }
    return s;
}

/**
 * @brief Sum of values over the owned cells of every rank (of this rank's
 *        cells if the mesh is not distributed) in global cell order, so that
 *        it is the same bit for bit on any number of ranks.
 */
double ordered_sum(const Mesh & mesh, const std::vector<double> & values) {
    const uint32_t n_owned = mesh.n_owned();
    std::vector<uint64_t> ids;
    std::vector<double> local;
    for (uint32_t c = 0; c < n_owned; c++) {
        if (values[c] == 0.0) continue;
        ids.push_back(mesh.h_global_cell(c));
        local.push_back(values[c]);
    }
    if (mesh.n_global_cells > 0) {
        ids = comm::allgatherv(ids);
        local = comm::allgatherv(local);
    }
    std::vector<size_t> order(ids.size());
    for (size_t i = 0; i < order.size(); i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&ids](const size_t a, const size_t b) { return ids[a] < ids[b]; });
    double sum = 0.0;
    for (const size_t i : order) sum += local[i];
    return sum;
}

bool reads_solution(const kexpr::Program & p, const uint16_t n_variables) {
    for (uint16_t i = RHO; i < n_variables; i++) {
        if (i >= MEAN_RHO && i < N_FIXED) continue;
        if (p.uses(i)) return true;
    }
    return false;
}

bool reads_means(const kexpr::Program & p) {
    for (uint16_t i = MEAN_RHO; i < COPY; i++) {
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
    kexpr::ProgramTable<> programs;
    Kokkos::View<double **, Kokkos::LayoutRight> weights;
    CellFields fields;
    Kokkos::View<double *> means;
    Kokkos::View<rtype> time;
    Kokkos::View<double *> integrals;
    uint32_t n_species;
    Sink sink;

    KOKKOS_INLINE_FUNCTION void operator()(const uint32_t c) const {
        for (uint32_t col = 0; col < field.extent(1); col++) sink(c, col, double(field(c, col)));
        CellVariables vars{fields, c, time.data() ? double(time()) : 0.0, means.data(), n_species};
        for (uint32_t j = 0; j < terms.extent(0); j++) {
            const Term & term = terms(j);
            const double w = term.weights < 0 ? 1.0 : weights(term.weights, c);
            if (w == 0.0) continue;
            vars.copy = term.copy;
            double scale = term.scale;
            if (term.integral >= 0) {
                const double integral = integrals(term.integral);
                scale = integral > 0.0 ? 1.0 / integral : 0.0;
            }
            const double v = programs(term.program, vars) * w * scale;
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

SourceTerms::SourceTerms(const Input & input, const std::shared_ptr<Mesh> & mesh_,
                         const std::vector<std::string> & species, const CellFields & fields_,
                         region::Regions & regions, const bool use_nccl) :
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

    std::vector<std::pair<const Input *, std::string>> entries;
    if (input.contains("source")) {
        const Input & s = input.at("source");
        if (s.contains("rho") || s.contains("rhou") || s.contains("rhoE")) entries.emplace_back(&s, "source");
    }
    if (input.contains("sources")) {
        if (!input.at("sources").is_array()) throw InputError("sources must be an array of tables: [[sources]].");
        const auto & list = input.at("sources").as_array();
        for (size_t i = 0; i < list.size(); i++) entries.emplace_back(&list[i], "sources[" + std::to_string(i) + "]");
    }
    m_empty = entries.empty();
    if (m_empty) return;

    std::vector<double> measure(n_owned);
    for (uint32_t c = 0; c < n_owned; c++) measure[c] = double(mesh->h_cell_measure(c));
    const double volume = ordered_sum(*mesh, measure);

    std::vector<Term> static_terms, dyn_terms;
    std::vector<std::vector<double>> weight_rows;
    int32_t n_totals = 0;
    for (const auto & [entry, where] : entries) {
        const bool legacy = where == "source";
        if (!entry->is_table()) throw InputError(where + " must be a table.");
        if (legacy && entry->contains("time_dependent")) {
            if (!entry->at("time_dependent").is_boolean()) throw InputError("source.time_dependent must be true or false.");
            logging::warning("source.time_dependent is ignored: expressions of t are re-evaluated at every stage.");
        }
        const std::string label = !legacy && entry->contains("name") ? find_value<std::string>(*entry, "name") : where;
        const bool total = !legacy && find_value_or<bool>(*entry, "total", false);
        // A total is integrated over the region: per unit volume times a volume
        const auto per = [total](const Dimension & d) { return total ? d.times(dim::length, 3) : d; };
        // (column, expression, name)
        std::vector<std::tuple<uint32_t, std::string, std::string>> values;
        if (entry->contains("rho")) {
            if (mixture) {
                throw InputError(where + ".rho: in gas mixtures mass enters through rhoY (rho gains their sum).");
            }
            values.emplace_back(0u, quantity_or_expression(entry->at("rho"), where + ".rho", per(dim::mass_source)), "rho");
        }
        if (entry->contains("rhou")) {
            const Input & v = entry->at("rhou");
            if (!v.is_array() || v.as_array().size() != N_DIM) {
                throw InputError(where + ".rhou must have " + std::to_string(N_DIM) + " components.");
            }
            FOR_I_DIM values.emplace_back(1u + i,
                                          quantity_or_expression(v.as_array()[i], where + ".rhou",
                                                                 per(dim::momentum_source)),
                                          "rhou_" + std::string(1, "xyz"[i]));
        }
        if (entry->contains("rhoE")) {
            values.emplace_back(uint32_t(N_DIM + 1),
                                quantity_or_expression(entry->at("rhoE"), where + ".rhoE", per(dim::power_density)),
                                "rhoE");
        }
        if (entry->contains("rhoY")) {
            if (!mixture) throw InputError(where + ".rhoY needs a gas mixture.");
            if (!entry->at("rhoY").is_table()) throw InputError(where + ".rhoY must be a table of species.");
            for (const auto & [name, value] : entry->at("rhoY").as_table()) {
                const auto k = std::find(species.begin(), species.end(), name);
                if (k == species.end()) throw InputError(where + ".rhoY: no species " + name + " in the mechanism.");
                values.emplace_back(uint32_t(N_CONSERVATIVE + (k - species.begin())),
                                    quantity_or_expression(value, where + ".rhoY." + name, per(dim::mass_source)),
                                    "rhoY_" + name);
            }
        }
        if (values.empty()) throw InputError(where + " sets none of rho, rhou, rhoE, rhoY.");

        std::vector<kexpr::Program> compiled;
        bool per_copy = false;
        for (const auto & [column, text, name] : values) {
            kexpr::Program p(where + "." + name, text, names, expression_unit_hint);
            if (total) {
                for (uint16_t i = 0; i < n_variables; i++) {
                    if (i != TIME && i != COPY && p.uses(i)) {
                        throw InputError(where + "." + name + ": with total = true, values may depend on t and copy only (got " +
                                         names[i] + "); the region gives the distribution.");
                    }
                }
            }
            per_copy = per_copy || p.uses(COPY);
            compiled.push_back(std::move(p));
        }

        // The parts the terms apply to: the region, or each of its copies if they read copy
        struct Part {
            int32_t row = -1, integral = -1;
            double scale = 1.0;
        };
        std::vector<Part> parts(1);
        std::string region_text = "everywhere";
        bool moves = false;
        if (!legacy && entry->contains("region")) {
            const Input & value = entry->at("region");
            const region::Region & r = regions.get(value, where + ".region");
            region::require_full_dimension(r, value, where + ".region");
            moves = r.moving() || r.solution();
            const uint32_t n_parts = per_copy ? std::max<uint32_t>(1, static_cast<uint32_t>(r.copies().size())) : 1;
            parts.assign(n_parts, Part{});
            double integral = 0.0;
            std::shared_ptr<const region::SolutionLevels> levels;
            if (r.solution()) levels = std::make_shared<region::SolutionLevels>(*mesh, r.conditions(), species, fields);
            for (uint32_t k = 0; k < n_parts; k++) {
                std::vector<double> fractions = r.solution()  ? std::vector<double>(mesh->n_cells, 0.0)
                                                : n_parts > 1 ? region::cell_fractions(*mesh, r.copy(k))
                                                              : regions.fractions(value, where + ".region");
                std::vector<double> weighted(n_owned);
                for (uint32_t c = 0; c < n_owned; c++) weighted[c] = fractions[c] * measure[c];
                const double part = ordered_sum(*mesh, weighted);
                if (!moves && !(part > 0.0)) {
                    throw InputError(where + ".region " + (n_parts > 1 ? "copy " + std::to_string(k) + " " : "") +
                                     "contains no part of any cell.");
                }
                integral += part;
                parts[k].row = static_cast<int32_t>(weight_rows.size());
                parts[k].scale = total && part > 0.0 ? 1.0 / part : 1.0;
                if (moves) {
                    Moving m;
                    m.fractions = n_parts > 1 ? std::make_unique<region::MovingFractions>(*mesh, r.copy(k), levels)
                                              : std::make_unique<region::MovingFractions>(*mesh, r, levels);
                    m.row = static_cast<uint32_t>(parts[k].row);
                    if (total) m.integral = parts[k].integral = n_totals++;
                    moving.push_back(std::move(m));
                }
                weight_rows.push_back(std::move(fractions));
            }
            std::ostringstream text;
            text << r.summary();
            if (r.solution()) {
                text << ", from the solution";
            } else if (moves) {
                text << ", moving";
            } else {
                text << ", measure " << logging::real(integral);
            }
            if (n_parts > 1) text << ", each of " << n_parts << " copies";
            region_text = text.str();
        } else if (per_copy) {
            throw InputError(where + ": copy is the index of a region's copy, but there is no region.");
        }

        std::string kinds;
        for (size_t v = 0; v < compiled.size(); v++) {
            kexpr::Program & p = compiled[v];
            const bool time = p.uses(TIME) || moves, solution = reads_solution(p, n_variables), mean = reads_means(p);
            m_time = m_time || time;
            m_means = m_means || mean;
            for (uint32_t k = 0; k < parts.size(); k++) {
                Term term;
                term.program = static_cast<uint32_t>(programs.size());
                term.column = std::get<0>(values[v]);
                term.weights = parts[k].row;
                term.scale = total ? (parts[k].row < 0 ? 1.0 / volume : parts[k].scale) : 1.0;
                term.integral = total ? parts[k].integral : -1;
                term.copy = static_cast<double>(k);
                (time || solution || mean ? dyn_terms : static_terms).push_back(term);
            }
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

    std::vector<const kexpr::Program *> pointers;
    for (const auto & p : programs) pointers.push_back(&p);
    table = kexpr::ProgramTable<>(pointers);
    const size_t n_rows = weight_rows.size();
    weights = Kokkos::View<double **, Kokkos::LayoutRight>("source_weights", n_rows, mesh->n_cells);
    auto h_weights = Kokkos::create_mirror_view(weights);
    for (size_t r = 0; r < n_rows; r++) {
        for (uint32_t c = 0; c < mesh->n_cells; c++) h_weights(r, c) = weight_rows[r][c];
    }
    Kokkos::deep_copy(weights, h_weights);
    nccl_sums = use_nccl;
    if (n_totals > 0) {
        integral_sums = std::make_unique<DeviceSums>(static_cast<uint32_t>(n_totals), n_owned, mesh->n_global_cells > 0);
        integrands = Kokkos::View<double **, Kokkos::LayoutLeft>("source_integrands", n_owned, size_t(n_totals));
    }
    means = Kokkos::View<double *>("source_means", N_MEANS);
    if (m_means) {
        mean_terms = Kokkos::View<double *[N_MEANS + 1]>("source_mean_terms", mesh->n_owned());
        mean_sums = ExactSums<N_MEANS + 1>(mesh->n_owned(), mesh->n_global_cells > 0, use_nccl);
    }
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
                             TermKernel<Accumulate>{{}, terms, table, weights, fields, means, {}, {}, n_species,
                                                    Accumulate{static_field}});
    }
    check_loops();
}

void SourceTerms::update_means(const State & solution) {
    const StateView U = solution.flow;
    const Kokkos::View<rtype *> V = mesh->cell_measure;
    const Kokkos::View<double *[N_MEANS + 1]> terms = mean_terms;
    Kokkos::parallel_for("source_mean_terms", mesh->n_owned(), KOKKOS_LAMBDA(const uint32_t c) {
        const double v = double(V(c)), rho = double(U(c, 0));
        double m[3] = {0.0, 0.0, 0.0};
        FOR_I_DIM m[i] = double(U(c, 1 + i));
        terms(c, 0) = v;
        terms(c, 1) = rho * v;
        for (int i = 0; i < 3; i++) terms(c, 2 + i) = m[i] * v;
        terms(c, 5) = double(U(c, N_DIM + 1)) * v;
        terms(c, 6) = 0.5 * (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]) / rho * v;
    });
    mean_sums.sum(terms);
    const Kokkos::View<double *> sums = mean_sums.sums(), m = means;
    Kokkos::parallel_for("source_means", 1, KOKKOS_LAMBDA(const int) {
        for (int i = 0; i < N_MEANS; i++) m(i) = sums(i + 1) / sums(0);
    });
}

SourceTerms::~SourceTerms() = default;

void SourceTerms::move(const Kokkos::View<rtype> & time) {
    const Kokkos::View<double **, Kokkos::LayoutRight> w = weights;
    const Kokkos::View<double **, Kokkos::LayoutLeft> terms = integrands;
    const Kokkos::View<rtype *> V = mesh->cell_measure;
    const uint32_t n_owned = mesh->n_owned();
    for (const Moving & m : moving) {
        m.fractions->update(time);
        const Kokkos::View<double *> f = m.fractions->fractions();
        const uint32_t row = m.row;
        const int32_t k = m.integral;
        Kokkos::parallel_for("source_moving_weights", mesh->n_cells, KOKKOS_LAMBDA(const uint32_t c) {
            w(row, c) = f(c);
            if (k >= 0 && c < n_owned) terms(c, k) = f(c) * double(V(c));
        });
    }
    if (integral_sums) integral_sums->sum(integrands, nccl_sums);
}

void SourceTerms::add(const State & solution, const State & rhs, const Kokkos::View<rtype> & stage_time) {
    if (m_means) update_means(solution);
    if (!moving.empty()) move(stage_time);
    Kokkos::parallel_for("rhs_source_terms", mesh->n_owned(),
                         TermKernel<AddToRhs>{static_field, dynamic_terms, table, weights, fields, means, stage_time,
                                              integral_sums ? integral_sums->totals() : Kokkos::View<double *>(), n_species,
                                              AddToRhs{rhs.flow, rhs.species, mesh->cell_measure}});
}

Kokkos::View<rtype **, Kokkos::LayoutRight> SourceTerms::evaluate(const State & solution, const rtype t) {
    Kokkos::View<rtype **, Kokkos::LayoutRight> out("source_values", mesh->n_cells, N_CONSERVATIVE + n_species);
    if (m_empty) return out;
    if (m_means) update_means(solution);
    Kokkos::View<rtype> time("source_time");
    Kokkos::deep_copy(time, t);
    if (!moving.empty()) move(time);
    Kokkos::parallel_for("source_values", mesh->n_owned(),
                         TermKernel<Accumulate>{static_field, dynamic_terms, table, weights, fields, means, time,
                                                integral_sums ? integral_sums->totals() : Kokkos::View<double *>(), n_species,
                                                Accumulate{out}});
    check_loops();
    return out;
}

void SourceTerms::check_loops() const {
    if (!table.loops) return;
    const uint32_t stopped = comm::allreduce(table.stopped(), comm::Op::MAX);
    if (stopped > 0) throw InputError(programs[stopped - 1].runaway_error());
}

} // namespace sources
