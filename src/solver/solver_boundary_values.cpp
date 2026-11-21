/**
 * @file solver_boundary_values.cpp
 * @brief The values each [[boundaries]] entry prescribes at its faces, as the boundary itself evaluates them, and
 *        the inflow that stagnation and mass-flow inlets would make of the initial state: mallard check
 *        --boundary-values and the session's case.boundary_values.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "solver.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>

#include <Kokkos_Core.hpp>

#include "comm.h"
#include "interfaces.h"
#include "json.h"
#include "log.h"

namespace {

struct Column {
    std::string name, units;
    bool estimated = false;
};

/** @brief The SI unit of a value slot of the entry's fields (BoundaryFields::Slot). */
std::string slot_units(const BoundaryType type, const int slot) {
    if (slot < BoundaryFields::SLOT_P) return type == BoundaryType::INLET_MASS_FLOW ? "kg/(m^2 s)" : "m/s";
    if (slot == BoundaryFields::SLOT_P) return "Pa";
    if (slot == BoundaryFields::SLOT_T) return "K";
    return type == BoundaryType::WALL_HEAT_FLUX ? "W/m^2" : "kg/(m^2 s)";
}

/** @brief A file name of the entry: no slashes or spaces. */
std::string file_name(const size_t index, std::string zone) {
    for (char & c : zone) {
        if (c == '/' || c == '\\' || c == ' ') c = '_';
    }
    return "boundaries_" + std::to_string(index) + "_" + zone + ".csv";
}

} // namespace

std::string Solver::boundary_values(const std::string & dir) const {
    namespace bv = boundary_variable;
    constexpr uint8_t E = N_DIM + 1;
    const auto h_bcs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundary_data.bcs);
    if (comm::is_root()) std::filesystem::create_directories(dir);
    const char * axes[] = {"x", "y", "z"};
    std::string entries = "[";
    for (size_t entry = 0; entry < boundary_summary.size(); entry++) {
        const BoundaryCondition & condition = h_bcs(entry);
        const BoundaryFields * fields = entry < boundary_fields.size() ? boundary_fields[entry].get() : nullptr;
        std::vector<uint32_t> faces;
        for (const uint32_t f : boundary_entry_faces[entry]) {
            if (static_cast<uint32_t>(mesh->h_cells_of_face(f, 0)) < mesh->n_owned()) faces.push_back(f);
        }

        std::vector<Column> columns;
        for (size_t d = 0; d < N_DIM; d++) columns.push_back({axes[d], "m"});
        for (size_t d = 0; d < N_DIM; d++) columns.push_back({std::string("n_") + axes[d], "1"});
        // The local coordinates the entry's expressions read, as its setup computed them
        std::vector<uint16_t> coordinates;
        for (uint16_t k = bv::FIRST_LOCAL; k < bv::FIRST_DATA; k++) {
            int computed = 0;
            for (const uint32_t f : faces) computed = computed || std::isfinite(boundary_point(f)[k]);
            if (comm::allreduce(computed, comm::Op::MAX) == 0) continue;
            coordinates.push_back(k);
            const bool metres = k == bv::N_M || k == bv::U_M
#if N_DIM == 3
                                || k == bv::V_M || k == bv::R_M
#endif
                ;
            columns.push_back({BoundaryValue::variables()[k], metres ? "m" : "1"});
        }
        std::vector<int> slots;
        for (int s = 0; fields && s < BoundaryFields::N_SLOTS; s++) {
            if (!fields->slot(s)) continue;
            slots.push_back(s);
            columns.push_back({fields->slot_key(s), slot_units(condition.type, s)});
        }
        const bool velocity = fields && fields->get_vector() != BoundaryFields::Vector::NONE &&
                              (condition.type == BoundaryType::INLET_STATE || condition.type == BoundaryType::FARFIELD ||
                               condition.is_wall());
        if (velocity) {
            for (size_t d = 0; d < N_DIM; d++) columns.push_back({std::string("u_") + axes[d], "m/s"});
        }
        const bool inflow =
            condition.type == BoundaryType::INLET_STAGNATION || condition.type == BoundaryType::INLET_MASS_FLOW;
        if (inflow) {
            columns.push_back({"rho", "kg/m^3", true});
            for (size_t d = 0; d < N_DIM; d++) columns.push_back({std::string("u_") + axes[d], "m/s", true});
            columns.push_back({"speed", "m/s", true});
            columns.push_back({"rho_u_n", "kg/(m^2 s)", true});
            columns.push_back({"p", "Pa", true});
        }

        // The ghost state of the flux, against the cell's initial state
        Kokkos::View<double **>::host_mirror_type h_ghost;
        if (inflow && !faces.empty()) {
            Kokkos::View<uint32_t *> d_faces("boundary_values_faces", faces.size());
            const auto h_faces = Kokkos::create_mirror_view(d_faces);
            for (size_t k = 0; k < faces.size(); k++) h_faces(k) = faces[k];
            Kokkos::deep_copy(d_faces, h_faces);
            Kokkos::View<double **> ghost("boundary_values_ghost", faces.size(), N_DIM + 2);
            const BoundaryData boundaries = boundary_data;
            const auto W = W_cells;
            const auto normals = mesh->face_normals;
            const auto cells = mesh->cells_of_face;
            Kokkos::parallel_for("boundary_values_ghost", faces.size(), KOKKOS_LAMBDA(const size_t k) {
                const uint32_t f = d_faces(k);
                rtype n_vec[N_DIM], n[N_DIM], W_i[N_DIM + 2], W_g[N_DIM + 2], th_g[2];
                FOR_I_DIM n_vec[i] = normals(f, i);
                unit<N_DIM>(n_vec, n);
                for (uint8_t j = 0; j < N_DIM + 2; j++) W_i[j] = W(cells(f, 0), j);
                const int32_t i_bc = boundaries.face_bc(f);
                if (boundaries.mixture_inflow(i_bc)) {
                    boundaries.mixture_inflow_W(i_bc, W_i, n, W_g, th_g);
                } else {
                    boundaries.ghost_W(f, W_i, n, W_g, true);
                }
                for (uint8_t j = 0; j < N_DIM + 2; j++) ghost(k, j) = static_cast<double>(W_g[j]);
            });
            h_ghost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ghost);
        }

        const size_t width = columns.size();
        std::vector<uint64_t> keys;
        std::vector<double> rows, areas;
        for (size_t k = 0; k < faces.size(); k++) {
            const uint32_t f = faces[k];
            const uint32_t c = static_cast<uint32_t>(mesh->h_cells_of_face(f, 0));
            uint32_t j = 0;
            while (mesh->h_face_of_cell(c, j) != f) j++;
            keys.push_back(mesh->h_global_cell(c) * 8 + j);
            const double area = static_cast<double>(mesh->h_face_area(f));
            areas.push_back(area);
            const FacePoint x = boundary_point(f);
            Point n_in{};
            FOR_I_DIM n_in[i] = -static_cast<double>(mesh->h_face_normals(f, i)) / area;
            for (size_t d = 0; d < N_DIM; d++) rows.push_back(x[d]);
            for (size_t d = 0; d < N_DIM; d++) rows.push_back(n_in[d]);
            for (const uint16_t c_k : coordinates) rows.push_back(x[c_k]);
            for (const int s : slots) rows.push_back(fields->slot(s)->at(x, 0.0));
            if (velocity) {
                BoundaryCondition prescribed = condition;
                fields->apply(prescribed, x, 0.0, double(physics.R), n_in);
                for (size_t d = 0; d < N_DIM; d++) rows.push_back(static_cast<double>(prescribed.data[1 + d]));
            }
            if (inflow) {
                double speed2 = 0.0, into = 0.0;
                rows.push_back(h_ghost(k, 0));
                for (size_t d = 0; d < N_DIM; d++) {
                    const double u = h_ghost(k, 1 + d);
                    rows.push_back(u);
                    speed2 += u * u;
                    into += u * n_in[d];
                }
                rows.push_back(std::sqrt(speed2));
                rows.push_back(h_ghost(k, 0) * into);
                rows.push_back(h_ghost(k, E));
            }
        }
        if (is_distributed()) {
            keys = comm::allgatherv(keys);
            rows = comm::allgatherv(rows);
            areas = comm::allgatherv(areas);
        }
        std::vector<size_t> order(keys.size());
        std::iota(order.begin(), order.end(), size_t(0));
        std::sort(order.begin(), order.end(), [&](const size_t a, const size_t b) { return keys[a] < keys[b]; });

        const std::string zone = boundary_summary[entry].zone;
        const std::string file = (std::filesystem::path(dir) / file_name(entry, zone)).string();
        if (comm::is_root()) {
            std::ofstream out(file);
            out << "face";
            for (const Column & col : columns) out << "," << col.name;
            out << "\n";
            for (const size_t k : order) {
                out << keys[k];
                for (size_t c = 0; c < width; c++) out << "," << json::number(rows[k * width + c]);
                out << "\n";
            }
            if (!out) throw std::runtime_error("cannot write " + file);
        }
        std::string list = "[";
        for (size_t c = 0; c < width; c++) {
            const double inf = std::numeric_limits<double>::infinity();
            double lo = inf, hi = -inf, sum = 0.0, total = 0.0;
            for (const size_t k : order) {
                const double v = rows[k * width + c];
                lo = std::min(lo, v);
                hi = std::max(hi, v);
                sum += v * areas[k];
                total += areas[k];
            }
            list += (list.size() > 1 ? "," : "") + json::Object()
                                                         .add("name", columns[c].name)
                                                         .add("units", columns[c].units)
                                                         .add("estimated", columns[c].estimated)
                                                         .add("min", lo)
                                                         .add("mean", total > 0.0 ? sum / total : std::nan(""))
                                                         .add("max", hi)
                                                         .str();
        }
        entries += (entries.size() > 1 ? "," : "") + json::Object()
                                                         .add("entry", "boundaries[name = \"" + zone + "\"]")
                                                         .add("index", static_cast<uint64_t>(entry))
                                                         .add("zone", zone)
                                                         .add("condition", boundary_summary[entry].condition)
                                                         .add("file", file)
                                                         .add("faces", static_cast<uint64_t>(keys.size()))
                                                         .raw("columns", list + "]")
                                                         .str();
    }
    return json::Object()
        .add("schema", interfaces::BOUNDARY_VALUES.name)
        .add("version", interfaces::BOUNDARY_VALUES.version)
        .add("t", static_cast<double>(t))
        .raw("entries", entries + "]")
        .str();
}
