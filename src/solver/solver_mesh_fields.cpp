/**
 * @file solver_mesh_fields.cpp
 * @brief The mesh and stencil fields of a run (docs/guide/output.md#mesh-and-stencil-fields):
 *        computed once at setup, outside the time loop.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "solver.h"

#include "common_io.h"
#include "input.h"
#include "log.h"
#include "mesh_quality.h"

std::vector<int> Solver::cell_owners() const {
    std::vector<int> owner(mesh->n_cells, comm::rank());
    if (!is_distributed()) return owner;
    for (size_t n = 0; n < distribution.neighbors.size() && n < distribution.recv_cells.size(); n++) {
        for (const uint32_t c : distribution.recv_cells[n]) owner[c] = distribution.neighbors[n];
    }
    return owner;
}

stencils::Table Solver::stencil_table(const std::vector<uint32_t> & targets, const bool weights) const {
    return stencils::build(*face_reconstruction, *mesh, boundary_data, cell_owners(), targets, weights);
}

void Solver::init_mesh_fields() {
    const std::string stencil_file = find_value_or<std::string>(input, "output", "write_stencils", "");
    mesh_field_names.clear();
    for (const char * name : MeshQuality::FIELDS) {
        if (writes_variable(input, name)) mesh_field_names.emplace_back(name);
    }
    const size_t n_quality = mesh_field_names.size();
    for (const char * name : stencils::FIELDS) {
        if (writes_variable(input, name)) mesh_field_names.emplace_back(name);
    }
    const bool write_file = !stencil_file.empty() && output_files_enabled();
    h_mesh_fields = {};
    if (mesh_field_names.empty() && !write_file) return;

    std::vector<std::pair<std::string, std::vector<double>>> values;
    if (n_quality > 0) {
        const MeshQuality quality = mesh_quality(*mesh);
        for (size_t m = 0; m < MeshQuality::N_METRICS; m++) values.emplace_back(MeshQuality::FIELDS[m], quality.cells[m]);
    }
    if (mesh_field_names.size() > n_quality || write_file) {
        std::vector<uint32_t> targets;
        if (write_file) {
            targets.resize(mesh->n_owned());
            for (uint32_t c = 0; c < mesh->n_owned(); c++) targets[c] = c;
        }
        const stencils::Table table = stencil_table(targets, false);
        if (write_file) logging::line("Stencils: " + stencils::write(table, *mesh, cell_owners(), stencil_file));
        for (auto & field : table.summary_fields()) values.push_back(std::move(field));
    }
    h_mesh_fields = Kokkos::View<rtype **, Kokkos::LayoutRight, Kokkos::HostSpace>("mesh_fields", mesh->n_cells,
                                                                                    mesh_field_names.size());
    for (size_t k = 0; k < mesh_field_names.size(); k++) {
        for (const auto & [name, cells] : values) {
            if (name != mesh_field_names[k]) continue;
            for (uint32_t c = 0; c < mesh->n_owned(); c++) h_mesh_fields(c, k) = static_cast<rtype>(cells[c]);
        }
    }
}
