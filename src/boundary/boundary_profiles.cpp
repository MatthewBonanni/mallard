/**
 * @file boundary_profiles.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Profiles from data at boundaries.
 * @version 0.1
 * @date 2026-11-17
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "boundary_profiles.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "boundary_coordinates.h"
#include "comm.h"
#include "log.h"
#include "mesh.h"

namespace {

const std::string DOCS = "docs/guide/boundary_conditions.md#profiles-from-data";

InputError invalid(const Input & table, const std::string & key, const std::string & leaf, const std::string & message,
                   const std::string & fix) {
    return InputError({.key = key, .line = line_of(table, leaf), .message = message, .fix = fix, .docs = DOCS});
}

} // namespace

BoundaryProfiles::BoundaryProfiles(const Input & entry, const std::string & where_) : where(where_) {
    if (!entry.contains("data")) return;
    if (!entry.at("data").is_array()) {
        throw invalid(entry, where + ".data", "data", "must be an array of tables",
                      "Write each entry as [[boundaries.data]] after its [[boundaries]] entry");
    }
    namespace bv = boundary_variable;
    const std::vector<std::string> & base = BoundaryValue::variables();
    const std::vector<std::string> variables = BoundaryValue::variables(entry);
    std::vector<std::string> names(base);
    const auto & list = entry.at("data").as_array();
    for (size_t i = 0; i < list.size(); i++) {
        const Input & e = list[i];
        const std::string key = where + ".data[" + std::to_string(i) + "]";
        Mapped m;
        m.entry = tabulated::read_entry(e, key, DOCS, data);
        m.first = static_cast<uint16_t>(names.size());
        for (const std::string & var : m.entry.variables) {
            if (std::find(names.begin(), names.end(), var) != names.end()) {
                throw invalid(e, key + ".name", "name", "makes the variable " + var + ", which is already defined",
                              "Give each [[boundaries.data]] entry of a boundary its own name");
            }
            names.push_back(var);
        }
        const std::vector<std::string> & coordinates = m.entry.coordinates;
        std::vector<std::string> at;
        if (contains_optional(e, "at")) {
            at = find_value<std::vector<std::string>>(e, "at");
        } else {
            for (const auto & c : coordinates) {
                if (c == "t" || std::find(base.begin(), base.end(), c) == base.end()) {
                    throw invalid(e, key + ".at", "at",
                                  "is missing, and the coordinate " + c + " is not x, y, z or a local coordinate",
                                  "Add at = [...], an expression of x, y, z and the local coordinates for each "
                                  "coordinate, e.g. at = [\"n\"] for the distance to the wall");
                }
            }
            at = coordinates;
        }
        if (at.size() != coordinates.size()) {
            throw invalid(e, key + ".at", "at",
                          "has " + std::to_string(at.size()) + " expressions for " +
                              std::to_string(coordinates.size()) + " coordinates",
                          "Give one expression of x, y, z and the local coordinates per coordinate");
        }
        for (size_t d = 0; d < at.size(); d++) {
            const std::string name = key + ".at[" + std::to_string(d) + "]";
            m.at.push_back(std::make_shared<kexpr::Program>(name, at[d], base, BoundaryValue::hint, bv::FIRST_LOCAL));
            if (m.at.back()->uses(bv::T)) {
                throw invalid(e, key + ".at", "at", "depends on t, but the data are read once, at setup",
                              "Read the data at a fixed place and combine them with t in the boundary's "
                              "expressions, e.g. u = \"" + m.entry.name + "_u * (1 + 0.1 * sin(t))\"");
            }
            m.at_text.push_back(at[d]);
        }
        sets.push_back(std::move(m));
    }
    if (names != variables) throw std::logic_error(where + ": the data variables differ from their files' headers");
    const std::vector<bool> read = variables_read(entry);
    for (size_t i = 0; i < sets.size(); i++) {
        bool used = false;
        for (size_t f = 0; f < sets[i].entry.variables.size(); f++) used = used || read[sets[i].first + f];
        if (!used) {
            throw invalid(list[i], where + ".data[" + std::to_string(i) + "].name", "name",
                          "no expression of the boundary reads its variables",
                          "Read them as <name>_<column> in an expression of the entry, or remove the data");
        }
    }
}

std::vector<uint16_t> BoundaryProfiles::coordinates_used() const {
    std::vector<uint16_t> used;
    for (uint16_t i = boundary_variable::FIRST_LOCAL; i < boundary_variable::FIRST_DATA; i++) {
        for (const Mapped & m : sets) {
            for (const auto & p : m.at) {
                if (p->uses(i) && std::find(used.begin(), used.end(), i) == used.end()) used.push_back(i);
            }
        }
    }
    return used;
}

std::string BoundaryProfiles::evaluate(const Mesh & mesh, const std::vector<uint32_t> & faces,
                                   std::vector<FacePoint> & points, const bool distributed) const {
    const tabulated::DataView<Kokkos::HostSpace> host = data.host();
    const double inf = std::numeric_limits<double>::infinity();
    std::string summary;
    for (const Mapped & m : sets) {
        const tabulated::Set & set = data.set(m.entry.set);
        const size_t dims = m.at.size(), n_fields = m.entry.variables.size();
        std::vector<double> lo(dims + n_fields, inf), hi(dims + n_fields, -inf);
        std::vector<uint64_t> outside(dims, 0);
        std::string error;
        for (size_t k = 0; k < points.size(); k++) {
            const auto vars = points[k].at(0.0);
            double q[tabulated::MAX_DIMS] = {0.0, 0.0, 0.0};
            for (size_t d = 0; d < dims; d++) {
                q[d] = (*m.at[d])(vars.data());
                if (!std::isfinite(q[d]) && error.empty()) {
                    std::string at;
                    for (size_t j = 0; j < N_DIM; j++) at += (j ? ", " : "(") + logging::real(points[k][j]);
                    error = m.entry.name + ".at[" + std::to_string(d) + "] = \"" + m.at_text[d] + "\" is " +
                            logging::real(q[d]) + " at the face at " + at + ")";
                }
                lo[d] = std::min(lo[d], q[d]);
                hi[d] = std::max(hi[d], q[d]);
                if (!set.points && static_cast<uint32_t>(mesh.h_cells_of_face(faces[k], 0)) < mesh.n_owned()) {
                    const double * axis = host.axes.data() + set.axes[d];
                    if (q[d] < axis[0] || q[d] > axis[set.n_axis[d] - 1]) outside[d]++;
                }
            }
            for (size_t f = 0; f < n_fields; f++) {
                const double v = host(m.entry.set, static_cast<uint32_t>(f), q);
                points[k][m.first + f] = v;
                lo[dims + f] = std::min(lo[dims + f], v);
                hi[dims + f] = std::max(hi[dims + f], v);
            }
        }
        if (const std::string first = distributed ? comm::first_error(error) : error; !first.empty()) {
            throw InputError({.key = where + ".data",
                              .message = first + ", not a finite number",
                              .fix = "Give an expression that is finite at every face of the boundary",
                              .docs = DOCS});
        }
        if (distributed) {
            for (size_t j = 0; j < lo.size(); j++) {
                lo[j] = comm::allreduce(lo[j], comm::Op::MIN);
                hi[j] = comm::allreduce(hi[j], comm::Op::MAX);
            }
            for (uint64_t & n : outside) n = comm::allreduce(n, comm::Op::SUM);
        }
        auto range = [&](const size_t j) { return logging::real(lo[j]) + " to " + logging::real(hi[j]); };
        std::string text = m.entry.name + " (" + m.entry.file + ": " + m.entry.kind;
        for (size_t d = 0; d < dims; d++) {
            text += "; " + m.entry.coordinates[d] + " = " + m.at_text[d] + " " + range(d);
            if (outside[d] > 0) {
                const double * axis = host.axes.data() + set.axes[d];
                text += ", " + logging::count(outside[d], "face") + " outside the table's " +
                        logging::real(axis[0]) + " to " + logging::real(axis[set.n_axis[d] - 1]) +
                        " take its end values";
            }
        }
        text += "):";
        for (size_t f = 0; f < n_fields; f++) text += (f ? ", " : " ") + m.entry.variables[f] + " " + range(dims + f);
        summary += (summary.empty() ? "" : "; ") + text;
    }
    return summary;
}
