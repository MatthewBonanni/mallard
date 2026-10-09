/**
 * @file solver.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Solver class implementation.
 * @version 0.2
 * @date 2023-12-20
 *
 * @copyright Copyright (c) 2023 Matthew Bonanni
 *
 */

#include "solver.h"

#include "input.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>

#include <Kokkos_Core.hpp>

#include "comm.h"
#include "common.h"
#include "log.h"
#include "mesh_block.h"
#include "partition.h"
#include "expression.h"
#include "gradient.h"
#include "viscous_flux.h"

Solver::Solver() {
    // Empty
}

Solver::~Solver() {
    Kokkos::fence();
}

int Solver::init(const std::string & input_file_name) {
    return init(toml::parse(input_file_name));
}

template <typename F>
void Solver::timed_phase(const std::string & name, F && f) {
    logging::begin_phase(name);
    Kokkos::Timer phase_timer;
    f();
    Kokkos::fence();
    const double seconds = phase_timer.seconds();
    t_wall_setup += seconds;
    logging::end_phase(seconds);
}

int Solver::init(const toml::value & input_in) {
    this->input = input_in;

    t = 0.0;
    step = 0;
    t_wall_setup = 0.0;

    logging::section("Setup");
    init_run_parameters();
    timed_phase("mesh", [&] { init_mesh(); });
    timed_phase("physics and boundaries", [&] {
        init_physics();
        init_boundaries();
    });
    timed_phase("numerics", [&] { init_numerics(); });
    while (halo_too_shallow()) {
        timed_phase("halo rebuild (" + std::to_string(halo_layers) + " layers)", [&] {
            init_mesh();
            init_boundaries();
            init_numerics();
        });
    }
    // The cache describes the local mesh at this halo depth
    if (auto * teno = dynamic_cast<TENO *>(face_reconstruction.get())) teno->save_cache(halo_layers);
    setup.reset();
    timed_phase("fields and output", [&] {
        allocate_memory();
        init_les();
        statistics.init(input, species_names, mesh->n_cells);
        init_rhs_split();
        init_sources();
        init_sponges();
        register_data();
        init_output();
    });
    timed_phase("initial solution", [&] { init_solution(); });
    logging::begin_phase("total");
    logging::end_phase(t_wall_setup);
    print_setup();
    return 0;
}

namespace {

/**
 * @brief Cell counts by type, faces, extent and cell size range over all ranks.
 *        Each rank counts its owned cells; a face between two ranks is counted by
 *        the rank owning the side with the lower global cell id.
 */
logging::Items describe_mesh(const Mesh & mesh) {
    using logging::count;
    using logging::format;
    const bool distributed = !mesh.h_global_cell_id.empty();
    const uint32_t n_owned = mesh.n_owned();
    auto owned = [&](int32_t c) { return c >= 0 && static_cast<uint32_t>(c) < n_owned; };

    constexpr int MAX_NODES = 9;
    std::array<uint64_t, MAX_NODES + 2> counts{};  // cells by node count, then faces, boundary faces
    rtype v_min = std::numeric_limits<rtype>::max(), v_max = 0.0;
    for (uint32_t i = 0; i < n_owned; i++) {
        counts[std::min<uint32_t>(mesh.h_n_nodes_of_cell(i), MAX_NODES - 1)]++;
        v_min = std::min(v_min, mesh.h_cell_volume(i));
        v_max = std::max(v_max, mesh.h_cell_volume(i));
    }
    for (uint32_t f = 0; f < mesh.n_faces; f++) {
        const int32_t a = mesh.h_cells_of_face(f, 0), b = mesh.h_cells_of_face(f, 1);
        if (b < 0) {
            counts[MAX_NODES] += owned(a);
            counts[MAX_NODES + 1] += owned(a);
        } else if (owned(a) && owned(b)) {
            counts[MAX_NODES]++;
        } else if (owned(a) != owned(b) && distributed) {
            const int32_t mine = owned(a) ? a : b, other = owned(a) ? b : a;
            counts[MAX_NODES] += mesh.h_global_cell_id[mine] < mesh.h_global_cell_id[other];
        }
    }
    counts = comm::allreduce(counts, comm::Op::SUM);

    std::array<rtype, 2 * N_DIM + 2> lo_hi;  // -min and max per dimension, -min and max volume
    lo_hi.fill(std::numeric_limits<rtype>::lowest());
    for (uint32_t n = 0; n < mesh.n_nodes; n++) {
        for (int d = 0; d < N_DIM; d++) {
            lo_hi[d] = std::max(lo_hi[d], -mesh.h_node_coords(n, d));
            lo_hi[N_DIM + d] = std::max(lo_hi[N_DIM + d], mesh.h_node_coords(n, d));
        }
    }
    lo_hi[2 * N_DIM] = -v_min;
    lo_hi[2 * N_DIM + 1] = v_max;
    lo_hi = comm::allreduce(lo_hi, comm::Op::MAX);
    v_min = -lo_hi[2 * N_DIM];
    v_max = lo_hi[2 * N_DIM + 1];

    const std::map<uint32_t, const char *> names = N_DIM == 2
        ? std::map<uint32_t, const char *>{{3, "tri"}, {4, "quad"}}
        : std::map<uint32_t, const char *>{{4, "tet"}, {5, "pyramid"}, {6, "prism"}, {8, "hex"}};
    std::string types;
    uint64_t n_cells = 0;
    int n_types = 0;
    for (int k = 0; k < MAX_NODES; k++) {
        if (counts[k] == 0) continue;
        const auto it = names.find(k);
        types += (types.empty() ? "" : ", ") + count(counts[k]) + " " +
                 (it != names.end() ? std::string(it->second) : std::to_string(k) + "-node");
        n_cells += counts[k];
        n_types++;
    }
    std::string extent;
    for (int d = 0; d < N_DIM; d++) {
        extent += (d ? " x " : "") +
                  format("[%.4g, %.4g]", static_cast<double>(-lo_hi[d]), static_cast<double>(lo_hi[N_DIM + d]));
    }
    return {
        {"Cells", n_types == 1 ? types : count(n_cells) + ": " + types},
        {"Faces", count(counts[MAX_NODES]) + " (" + count(counts[MAX_NODES + 1]) + " boundary)"},
        {"Extent", extent},
        {N_DIM == 2 ? "Cell area" : "Cell volume",
         format("%.3e to %.3e (ratio %.3g)", static_cast<double>(v_min), static_cast<double>(v_max),
                static_cast<double>(v_max / v_min))},
    };
}

bool writes_variable(const toml::value & input, const std::string & name) {
    if (!input.contains("write_data")) return false;
    const auto outputs = toml::find<std::vector<toml::value>>(input, "write_data");
    for (const auto & output : outputs) {
        if (!output.contains("variables")) continue;
        const auto variables = toml::find<std::vector<std::string>>(output, "variables");
        if (std::find(variables.begin(), variables.end(), name) != variables.end()) return true;
    }
    return false;
}

} // namespace

void Solver::init_mesh() {
    const std::string type = toml::find_or<std::string>(input, "mesh", "type", "file");
    if (!is_distributed()) {
        mesh = std::make_shared<Mesh>();
        mesh->init(input);
    } else {
        if (halo_layers == 0) {
            // A TENO cache records the halo depth its stencils need, which spares
            // the setup pass that would otherwise find it out
            const toml::value reconstruction =
                toml::find_or(input, "numerics", "face_reconstruction", toml::value(toml::table{}));
            const bool teno = toml::find_or<std::string>(reconstruction, "type", "FO") == "TENO";
            const int cached =
                comm::allreduce(teno ? int(TENO::cached_halo_layers(reconstruction)) : 0, comm::Op::MAX);
            halo_layers = std::max(base_halo_layers(), cached);
        }
        // Partition once; deeper halos (see halo_too_shallow) grow the existing layers
        if (!setup) {
            partitioner = toml::find_or<std::string>(input, "parallel", "partitioner",
                                                     have_graph_partitioner() ? "graph" : "hilbert");
            if (partitioner != "graph" && partitioner != "hilbert") {
                throw InputError("parallel.partitioner = \"" + partitioner + "\" is not one of: graph, hilbert.");
            }
            setup = std::make_unique<DistributedMesh>(read_mesh_block(input), Mesh::periodic_pairs(input));
            setup->distribute(partitioner == "graph" ? partition_graph(*setup, comm::size())
                                                     : partition_hilbert(*setup, comm::size()));
        }
        mesh = setup->build_local_mesh(halo_layers, distribution);
        halo = HaloExchange(distribution);
    }
    mesh_summary = describe_mesh(*mesh);
    mesh_summary.insert(mesh_summary.begin(),
                        {"Type", type == "file" ? "file " + toml::find_or<std::string>(input, "mesh", "filename",
                                                                                         "mesh.msh")
                                                : type});
    const Stretching stretching = mesh_stretching(input);
    if (stretching != Stretching{}) {
        if (type == "file") throw InputError("[mesh] stretching applies to generated meshes.");
        std::string text;
        FOR_I_DIM text += (i ? ", " : "[") + logging::real(double(stretching[i]));
        mesh_summary.insert(mesh_summary.begin() + 1, {"Stretching", text + "] (tanh, toward both ends)"});
    }
    n_cells_global = comm::allreduce(uint64_t(mesh->n_owned()), comm::Op::SUM);
    if (is_distributed()) {
        const uint64_t n_owned = distribution.n_owned;
        const uint64_t n_halo = mesh->n_cells - distribution.n_owned;
        const uint64_t min_owned = comm::allreduce(n_owned, comm::Op::MIN);
        const uint64_t max_owned = comm::allreduce(n_owned, comm::Op::MAX);
        const uint64_t max_halo = comm::allreduce(n_halo, comm::Op::MAX);
        const double mean = static_cast<double>(n_cells_global) / comm::size();
        mesh_summary.emplace_back("Partition", logging::format("%s, %d ranks, %d halo layers",
                                                               partitioner == "graph" ? "graph (KaMinPar)"
                                                                                      : "Hilbert curve",
                                                               comm::size(), halo_layers));
        mesh_summary.emplace_back("Cells per rank",
                                  logging::format("%s to %s owned (imbalance %.3f), up to %s halo",
                                                  logging::count(min_owned).c_str(),
                                                  logging::count(max_owned).c_str(), max_owned / mean,
                                                  logging::count(max_halo).c_str()));
    }
    axisymmetric = toml::find_or<bool>(input, "physics", "axisymmetric", false);
    if (axisymmetric) {
        for (const auto & translation : mesh->periodic_translations) {
            if (translation[1] != 0.0_r) {
                throw InputError("physics.axisymmetric: the mesh cannot be periodic in y (the radius).");
            }
        }
        mesh->make_axisymmetric();
        mesh_summary.emplace_back("Geometry", "axisymmetric about the x axis (y = r), per radian");
    }
    mesh->copy_host_to_device();
}

int Solver::base_halo_layers() const {
    const std::string type = toml::find_or<std::string>(
        toml::find_or(input, "numerics", "face_reconstruction", toml::value(toml::table{})), "type", "FO");
    // Faces between owned and halo-layer-1 cells need the layer-1 reconstruction,
    // which reads one more layer (MUSCL gradients and limiters, viscous gradients)
    const bool viscous = toml::find_or<std::string>(input, "physics", "type", "euler") == "navier_stokes";
    // The hybrid flux's sensor reads the velocity gradients of halo-layer-1 cells too
    const bool hybrid = toml::find_or<std::string>(input, "numerics", "convective_flux", "riemann") == "hybrid";
    return (type == "FO" && !viscous && !hybrid) ? 1 : 2;
}

bool Solver::halo_too_shallow() {
    if (!is_distributed()) return false;
    auto * teno = dynamic_cast<TENO *>(face_reconstruction.get());
    if (teno == nullptr) return false;
    // Owned and halo-layer-1 cells are reconstructed; their stencil searches must
    // see every layer they visited, so stencils match the serial ones exactly
    int needed = 0;
    for (uint32_t c = 0; c < mesh->n_cells; c++) {
        if (distribution.layer[c] <= 1) needed = std::max(needed, distribution.layer[c] + teno->gather_depth[c] + 1);
    }
    needed = comm::allreduce(needed, comm::Op::MAX);
    if (needed <= halo_layers) return false;
    halo_layers = needed;
    return true;
}

void Solver::init_rhs_split() {
    auto to_device = [](const std::vector<uint32_t> & v, const char * label) {
        Kokkos::View<uint32_t *> d(label, v.size());
        Kokkos::deep_copy(d, Kokkos::View<const uint32_t *, Kokkos::HostSpace>(v.data(), v.size()));
        return d;
    };
    const uint32_t n_owned = mesh->n_owned();
    std::vector<uint32_t> faces;
    for (uint32_t f = 0; f < mesh->n_faces; f++) {
        const int32_t c0 = mesh->h_cells_of_face(f, 0), c1 = mesh->h_cells_of_face(f, 1);
        if (c0 < static_cast<int32_t>(n_owned) || (c1 >= 0 && c1 < static_cast<int32_t>(n_owned))) faces.push_back(f);
    }
    rhs_faces = to_device(faces.size() < mesh->n_faces ? faces : std::vector<uint32_t>(), "rhs_faces");

    std::vector<uint32_t> cells;
    if (halo.active()) cells = face_reconstruction->cells_independent_of_halo(n_owned);
    n_early_cells = cells.size();
    if (n_early_cells == 0) return;
    std::vector<bool> early(mesh->n_cells, false);
    for (uint32_t c : cells) early[c] = true;
    for (uint32_t c = 0; c < mesh->n_reconstructed(); c++) {
        if (!early[c]) cells.push_back(c);
    }
    rhs_cells = to_device(cells, "rhs_cells");
    // Host backends run kernels to completion anyway; on devices the early cells get their own stream
    if constexpr (!Kokkos::SpaceAccessibility<Kokkos::HostSpace,
                                              Kokkos::DefaultExecutionSpace::memory_space>::accessible) {
        overlap_space = Kokkos::Experimental::partition_space(Kokkos::DefaultExecutionSpace(), 1)[0];
    }
}

void Solver::init_physics() {
    if (!input.contains("physics")) throw InputError("missing [physics] table.");
    const std::string gas = toml::find_or<std::string>(input, "physics", "gas", "perfect");
    if (gas == "perfect") {
        if (input.contains("chemistry")) throw InputError("[chemistry] needs physics.gas = \"mixture\".");
        physics = Euler::from_input(input);
        return;
    }
    if (gas != "mixture") {
        throw InputError("physics.gas = \"" + gas + "\" is not one of: perfect, mixture.");
    }
    mixture_model = std::make_shared<MixtureModel>(MixtureModel::from_input(input));
    mixture = mixture_model->device();
    species_names = mixture_model->species_names();
    physics = Euler();
    init_chemistry();
    init_radiation();
}

void Solver::init_boundaries() {
    // A fully periodic mesh has no boundary faces
    const std::vector<toml::value> input_boundaries =
        input.contains("boundaries") ? toml::find<std::vector<toml::value>>(input, "boundaries")
                                     : std::vector<toml::value>{};
    std::vector<int32_t> face_bc(mesh->n_faces, -1);
    std::vector<BoundaryCondition> bcs;
    boundary_summary.clear();
    dirichlet_boundaries.clear();
    average_pressure_outlets.clear();
    // Faces of upt boundaries whose composition varies along them, as (boundary, face):
    // each face gets its own copy of the condition, appended after the input's
    std::vector<std::array<uint32_t, 2>> profiled_faces;
    // Faces of each nscbc_inlet on this rank, by input boundary
    std::vector<std::pair<size_t, std::vector<uint32_t>>> inlets;

    for (size_t i_bc = 0; i_bc < input_boundaries.size(); i_bc++) {
        const toml::value & bound = input_boundaries[i_bc];
        if (!bound.contains("name")) {
            throw std::runtime_error("Boundary name not specified.");
        }
        if (!bound.contains("type")) {
            throw std::runtime_error("Boundary type not specified.");
        }
        const std::string name = toml::find<std::string>(bound, "name");
        const auto & periodic = mesh->periodic_zones;
        if (std::find(periodic.begin(), periodic.end(), name) != periodic.end()) {
            throw InputError("boundaries: zone \"" + name + "\" is periodic and takes no condition.");
        }
        FaceZone * zone = mesh->get_face_zone(name);
        if (zone != nullptr && zone->get_type() != FaceZoneType::BOUNDARY) zone = nullptr;
        // A rank's part of the mesh may not touch every zone
        if (comm::allreduce(uint32_t(zone != nullptr), comm::Op::SUM) == 0) {
            std::vector<std::string> names;
            for (const FaceZone & z : *mesh->face_zones()) {
                if (z.get_type() == FaceZoneType::BOUNDARY) names.push_back(z.get_name());
            }
            std::sort(names.begin(), names.end());
            std::string zones;
            for (const auto & z : names) zones += (zones.empty() ? "" : ", ") + z;
            throw InputError("boundaries: no boundary zone \"" + name + "\" in the mesh (zones: " + zones + ").");
        }
        bcs.push_back(BoundaryCondition::from_input(bound, physics));
        // Optional filter selecting part of the zone by face centroid
        std::unique_ptr<Expression> where;
        if (bound.contains("where")) {
            where = std::make_unique<Expression>(name + ".where", toml::find<std::string>(bound, "where"));
        }
        DirichletBoundary dirichlet;
        if (bcs.back().type == BoundaryType::DIRICHLET) {
            for (const char * key : {"rho", "u", "p"}) {
                if (!bound.contains(key)) {
                    throw std::runtime_error(std::string("Missing ") + key + " for boundary: " + name + ".");
                }
            }
            std::vector<std::string> u = toml::find<std::vector<std::string>>(bound, "u");
            if (u.size() != N_DIM) {
                throw std::runtime_error("Invalid u for boundary: " + name + ".");
            }
            dirichlet.W.emplace_back(name + ".rho", toml::find<std::string>(bound, "rho"));
            FOR_I_DIM dirichlet.W.emplace_back(name + ".u[" + std::to_string(i) + "]", u[i]);
            dirichlet.W.emplace_back(name + ".p", toml::find<std::string>(bound, "p"));
        }
        const bool profiled = is_mixture() &&
                              (bcs.back().type == BoundaryType::UPT || bcs.back().type == BoundaryType::NSCBC_INLET) &&
                              MixtureModel::composition_varies(bound);
        uint32_t n_selected = 0;
        uint64_t n_owned_selected = 0;
        for (uint32_t i = 0; zone && i < zone->n_faces(); i++) {
            const uint32_t i_face = zone->h_faces(i);
            if (where && where->at(Kokkos::subview(mesh->h_face_coords, i_face, Kokkos::ALL()), N_DIM) == 0.0) {
                continue;
            }
            if (face_bc[i_face] != -1) {
                throw std::runtime_error("Boundary " + name + " assigned more than once.");
            }
            face_bc[i_face] = i_bc;
            if (profiled) profiled_faces.push_back({static_cast<uint32_t>(i_bc), i_face});
            dirichlet.faces.push_back(i_face);
            n_selected++;
            n_owned_selected += static_cast<uint32_t>(mesh->h_cells_of_face(i_face, 0)) < mesh->n_owned();
        }
        if (comm::allreduce(n_selected, comm::Op::SUM) == 0) {
            throw std::runtime_error("Boundary " + name + " selects no faces.");
        }
        if (bcs.back().type == BoundaryType::NSCBC_INLET) inlets.emplace_back(i_bc, dirichlet.faces);
        if (bcs.back().type == BoundaryType::DIRICHLET) {
            dirichlet_boundaries.push_back(std::move(dirichlet));
        } else if (bcs.back().type == BoundaryType::P_OUT_AVERAGE) {
            // The area average runs over faces of owned cells only, so no face counts twice
            std::vector<uint32_t> owned;
            for (uint32_t f : dirichlet.faces) {
                if (static_cast<uint32_t>(mesh->h_cells_of_face(f, 0)) < mesh->n_owned()) owned.push_back(f);
            }
            Kokkos::View<uint32_t *> faces("average_pressure_faces", owned.size());
            auto h_faces = Kokkos::create_mirror_view(faces);
            for (size_t i = 0; i < owned.size(); i++) h_faces(i) = owned[i];
            Kokkos::deep_copy(faces, h_faces);
            // Sum in an order every rank count agrees on: by global cell, then by
            // the face's position in that cell
            std::vector<uint64_t> keys;
            for (uint32_t f : owned) {
                const uint32_t c = mesh->h_cells_of_face(f, 0);
                uint64_t k = 0;
                while (mesh->h_face_of_cell(c, k) != f) k++;
                keys.push_back(mesh->h_global_cell(c) * 8 + k);  // No cell has 8 faces
            }
            if (is_distributed()) keys = comm::allgatherv(keys);
            std::vector<uint32_t> order(keys.size());
            std::iota(order.begin(), order.end(), 0u);
            std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return keys[a] < keys[b]; });
            average_pressure_outlets.push_back({static_cast<int32_t>(i_bc), faces, std::move(order)});
        }
        n_owned_selected = comm::allreduce(n_owned_selected, comm::Op::SUM);
        std::string text = BOUNDARY_NAMES.at(bcs.back().type) + ", " + logging::count(n_owned_selected) + " faces";
        if (where) text += ", where " + toml::find<std::string>(bound, "where");
        if (profiled) text += ", varying composition";
        boundary_summary.emplace_back(name, text);
    }
    for (const auto & [i_bc, i_face] : profiled_faces) {
        const BoundaryCondition copy = bcs[i_bc];
        face_bc[i_face] = static_cast<int32_t>(bcs.size());
        bcs.push_back(copy);
    }

    if (FaceZone * partition = mesh->get_face_zone(PARTITION_ZONE)) {
        BoundaryCondition bc;
        bc.type = BoundaryType::PARTITION;
        bcs.push_back(bc);
        for (uint32_t i = 0; i < partition->n_faces(); i++) face_bc[partition->h_faces(i)] = bcs.size() - 1;
    }
    for (uint32_t i_face = 0; i_face < mesh->n_faces; i_face++) {
        if (mesh->h_cells_of_face(i_face, 1) < 0 && face_bc[i_face] < 0) {
            throw std::runtime_error("Boundary face " + std::to_string(i_face) +
                                     " has no boundary condition.");
        }
        // Faces on the axis carry no flux (zero revolved area); their ghost
        // states feed reconstruction and gradients, which need the mirror image
        if (axisymmetric && mesh->h_cells_of_face(i_face, 1) < 0 && mesh->h_face_measure(i_face) == 0.0_r) {
            const BoundaryType type = bcs[face_bc[i_face]].type;
            if (type != BoundaryType::SYMMETRY && type != BoundaryType::PARTITION) {
                throw InputError("physics.axisymmetric: boundary faces on the axis (y = 0) need type = "
                                 "\"symmetry\", not \"" + BOUNDARY_NAMES.at(type) + "\".");
            }
        }
    }
    characteristic_transverse = false;
    for (const BoundaryCondition & bc : bcs) {
        characteristic_transverse |= bc.is_characteristic() && bc.relax[BoundaryCondition::BETA] != 0.0_r;
    }
    t_characteristic = -1.0;
    characteristic_state_set = false;
    if (is_mixture()) init_mixture_boundaries(input_boundaries, profiled_faces, bcs);
    boundary_data = make_boundary_data(*mesh, face_bc, bcs, physics.gamma, physics.R, is_viscous(), physics);
    if (is_mixture()) {
        const uint32_t n_species = mixture.n_species;
        boundary_data.bc_Y = Kokkos::View<rtype **, Kokkos::LayoutRight>("bc_Y", bcs.size(), n_species);
        boundary_data.bc_thermo = Kokkos::View<rtype *[2]>("bc_thermo", bcs.size());
        auto h_bc_Y = Kokkos::create_mirror_view(boundary_data.bc_Y);
        auto h_thermo = Kokkos::create_mirror_view(boundary_data.bc_thermo);
        for (size_t i_bc = 0; i_bc < bcs.size(); i_bc++) {
            for (uint32_t k = 0; k < n_species; k++) {
                h_bc_Y(i_bc, k) = bc_mass_fractions[i_bc].empty() ? 0.0_r : static_cast<rtype>(bc_mass_fractions[i_bc][k]);
            }
            h_thermo(i_bc, 0) = static_cast<rtype>(bc_surrogates[i_bc][0]);
            h_thermo(i_bc, 1) = static_cast<rtype>(bc_surrogates[i_bc][1]);
        }
        Kokkos::deep_copy(boundary_data.bc_Y, h_bc_Y);
        Kokkos::deep_copy(boundary_data.bc_thermo, h_thermo);
        if (is_viscous()) {
            bc_transport_values = Kokkos::View<rtype **, Kokkos::LayoutRight>("bc_transport_values", bcs.size(),
                                                                              n_species + 1);
            auto h_values = Kokkos::create_mirror_view(bc_transport_values);
            for (size_t i_bc = 0; i_bc < bcs.size(); i_bc++) {
                if (bc_mass_fractions[i_bc].empty()) continue;
                const std::vector<double> X = mixture_model->mole_fractions(bc_mass_fractions[i_bc]);
                h_values(i_bc, 0) = static_cast<rtype>(bc_temperatures[i_bc]);
                for (uint32_t k = 0; k < n_species; k++) h_values(i_bc, 1 + k) = static_cast<rtype>(X[k]);
            }
            Kokkos::deep_copy(bc_transport_values, h_values);
        }
    }
    h_face_state = Kokkos::create_mirror_view(boundary_data.face_state);
    h_face_state_index = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundary_data.face_state_index);
    t_boundary_states = -1.0;
    init_inlets(input_boundaries, inlets);
}

void Solver::init_sources() {
    if (!input.contains("source")) {
        return;
    }
    const toml::value & source = input.at("source");
    if (source.contains("gravity")) {
        std::vector<rtype> g = find_real_vector(input, "source", "gravity");
        if (g.size() != N_DIM) {
            throw std::runtime_error("source.gravity must have " + std::to_string(N_DIM) + " components.");
        }
        has_gravity = true;
        FOR_I_DIM gravity[i] = g[i];
        FOR_I_DIM boundary_data.gravity[i] = g[i];
        face_reconstruction->set_boundaries(boundary_data);
        std::string text;
        FOR_I_DIM text += (i ? ", " : "[") + logging::real(double(gravity[i]));
        source_summary.emplace_back("Gravity", text + "]");
    }
    if (source.contains("mass_flow")) {
        const std::vector<rtype> m = find_real_vector(input, "source", "mass_flow");
        if (m.size() != N_DIM) {
            throw InputError("source.mass_flow must have " + std::to_string(N_DIM) + " components.");
        }
        rtype norm = 0.0_r;
        FOR_I_DIM norm += m[i] * m[i];
        norm = std::sqrt(norm);
        if (!(norm > 0.0_r) || !std::isfinite(norm)) {
            throw InputError("source.mass_flow must be a finite, nonzero vector.");
        }
        hold_mass_flow = true;
        mass_flow_target = double(norm);
        FOR_I_DIM mass_flow_direction[i] = m[i] / norm;
        double volume = 0.0;
        for (uint32_t i_cell = 0; i_cell < mesh->n_owned(); i_cell++) {
            volume += static_cast<double>(mesh->h_cell_volume(i_cell));
        }
        domain_volume = comm::allreduce(volume, comm::Op::SUM);
        std::string text;
        FOR_I_DIM text += (i ? ", " : "[") + logging::real(double(m[i]));
        source_summary.emplace_back("Mass flow", text + "] (volume average of rho u), held by a uniform body force");
    }
    const bool any_expression = source.contains("rho") || source.contains("rhou") || source.contains("rhoE");
    if (!any_expression) {
        return;
    }
    std::vector<std::string> texts(N_CONSERVATIVE, "0");
    texts[0] = toml::find_or<std::string>(input, "source", "rho", "0");
    texts[N_DIM + 1] = toml::find_or<std::string>(input, "source", "rhoE", "0");
    if (source.contains("rhou")) {
        std::vector<std::string> rhou = toml::find<std::vector<std::string>>(input, "source", "rhou");
        if (rhou.size() != N_DIM) {
            throw std::runtime_error("source.rhou must have " + std::to_string(N_DIM) + " components.");
        }
        FOR_I_DIM texts[1 + i] = rhou[i];
    }
    for (size_t i = 0; i < texts.size(); i++) {
        source_expressions.emplace_back("source[" + CONSERVATIVE_NAMES[i] + "]", texts[i]);
    }
    source_time_dependent = toml::find_or<bool>(input, "source", "time_dependent", false);
    source_field = StateView("source_field", mesh->n_cells);
    h_source_field = Kokkos::create_mirror_view(source_field);
    source_summary.emplace_back("Source terms", source_time_dependent ? "expressions, time dependent" : "expressions, steady");
}

void Solver::add_mass_flow_force(StateView solution, StateView rhs) {
    const uint32_t n_owned = mesh->n_owned();
    Kokkos::View<rtype *> vol = mesh->cell_volume;
    Kokkos::Array<rtype, N_DIM> e;
    FOR_I_DIM e[i] = mass_flow_direction[i];
    // Along e: the volume integral of rho u and the rate of change of that integral from all other terms
    double momentum = 0.0, rate = 0.0;
    Kokkos::parallel_reduce("mass_flow_sums", n_owned,
                            KOKKOS_LAMBDA(const uint32_t i_cell, double & sum_momentum, double & sum_rate) {
        rtype m = 0.0_r, r = 0.0_r;
        FOR_I_DIM {
            m += solution(i_cell, 1 + i) * e[i];
            r += rhs(i_cell, 1 + i) * e[i];
        }
        sum_momentum += static_cast<double>(m * vol(i_cell));
        sum_rate += static_cast<double>(r);
    }, momentum, rate);
    const std::array<double, 2> sums = comm::allreduce(std::array<double, 2>{momentum, rate}, comm::Op::SUM);
    // Cancel the other terms, and close the gap to the target within a step
    double force = -sums[1] / domain_volume;
    if (dt > 0.0_r) force += (mass_flow_target - sums[0] / domain_volume) / static_cast<double>(dt);
    mass_flow_force = force;
    Kokkos::Array<rtype, N_DIM> f;
    FOR_I_DIM f[i] = static_cast<rtype>(force) * e[i];
    Kokkos::parallel_for("rhs_mass_flow_force", n_owned, KOKKOS_LAMBDA(const uint32_t i_cell) {
        const rtype V = vol(i_cell);
        rtype work = 0.0_r;
        FOR_I_DIM {
            rhs(i_cell, 1 + i) += f[i] * V;
            work += f[i] * solution(i_cell, 1 + i);
        }
        rhs(i_cell, N_DIM + 1) += work / solution(i_cell, 0) * V;
    });
}

void Solver::update_source_field(rtype t_eval) {
    if (source_expressions.empty() || (t_source >= 0.0_r && (!source_time_dependent || t_eval == t_source))) {
        return;
    }
    for (uint32_t i_cell = 0; i_cell < mesh->n_cells; i_cell++) {
        const auto x = Kokkos::subview(mesh->h_cell_coords, i_cell, Kokkos::ALL());
        FOR_I_CONSERVATIVE h_source_field(i_cell, i) = source_expressions[i].at(x, N_DIM, double(t_eval));
    }
    Kokkos::deep_copy(source_field, h_source_field);
    t_source = t_eval;
}

void Solver::update_average_pressure_outlets(StateView solution) {
    const Euler phys = physics;
    for (const auto & outlet : average_pressure_outlets) {
        Kokkos::View<uint32_t *> faces = outlet.faces;
        Kokkos::View<int32_t *[2]> cells_of_face = mesh->cells_of_face;
        Kokkos::View<rtype *> face_area = mesh->face_measure;
        Kokkos::View<rtype *[2], Kokkos::LayoutRight> pA_A("outlet_pA_A", faces.extent(0));
        Kokkos::parallel_for("outlet_average_pressure", faces.extent(0), KOKKOS_LAMBDA(const uint32_t k) {
            const uint32_t f = faces(k);
            const int32_t c = cells_of_face(f, 0);
            rtype U[N_CONSERVATIVE], W[N_CONSERVATIVE];
            FOR_I_CONSERVATIVE U[i] = solution(c, i);
            phys.compute_W_from_conservatives(W, U);
            pA_A(k, 0) = W[N_DIM + 1] * face_area(f);
            pA_A(k, 1) = face_area(f);
        });
        auto h_pA_A = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pA_A);
        std::vector<rtype> local(h_pA_A.data(), h_pA_A.data() + h_pA_A.size());
        const std::vector<rtype> all = is_distributed() ? comm::allgatherv(local) : local;
        rtype pA = 0.0, A = 0.0;
        for (uint32_t k : outlet.sum_order) {
            pA += all[2 * k];
            A += all[2 * k + 1];
        }
        auto bc = Kokkos::subview(boundary_data.bcs, outlet.i_bc);
        auto h_bc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bc);
        h_bc().data[0] = h_bc().data[N_DIM + 1] - pA / A;
        Kokkos::deep_copy(bc, h_bc);
    }
}

void Solver::update_boundary_states(rtype t_eval) {
    update_inflow(t_eval);
    if (dirichlet_boundaries.empty() || t_eval == t_boundary_states) {
        return;
    }
    for (const auto & bc : dirichlet_boundaries) {
        for (uint32_t i_face : bc.faces) {
            const auto x = Kokkos::subview(mesh->h_face_coords, i_face, Kokkos::ALL());
            const int32_t k = h_face_state_index(i_face);
            if (k < 0) throw std::logic_error("Dirichlet boundary face " + std::to_string(i_face) + " has no state.");
            for (uint8_t i = 0; i < N_DIM + 2; i++) h_face_state(k, i) = bc.W[i].at(x, N_DIM, double(t_eval));
        }
    }
    Kokkos::deep_copy(boundary_data.face_state, h_face_state);
    t_boundary_states = t_eval;
}

void Solver::init_numerics() {
    const toml::value face_reconstruction_input =
        toml::find_or(input, "numerics", "face_reconstruction", toml::value(toml::table{}));
    const std::string face_reconstruction_str = toml::find_or<std::string>(face_reconstruction_input, "type", "FO");
    const std::string riemann_solver_str = toml::find_or<std::string>(input, "numerics", "riemann_solver", "HLLC");
    const std::string time_integrator_str = toml::find_or<std::string>(input, "numerics", "time_integrator", "SSPRK3");

    auto it_face = FACE_RECONSTRUCTION_TYPES.find(face_reconstruction_str);
    if (it_face == FACE_RECONSTRUCTION_TYPES.end()) {
        throw unknown_option(FACE_RECONSTRUCTION_TYPES, "numerics.face_reconstruction.type", face_reconstruction_str);
    }
    auto it_riemann = RIEMANN_SOLVER_TYPES.find(riemann_solver_str);
    if (it_riemann == RIEMANN_SOLVER_TYPES.end()) {
        throw unknown_option(RIEMANN_SOLVER_TYPES, "numerics.riemann_solver", riemann_solver_str);
    }
    auto it_time = TIME_INTEGRATOR_TYPES.find(time_integrator_str);
    if (it_time == TIME_INTEGRATOR_TYPES.end()) {
        throw unknown_option(TIME_INTEGRATOR_TYPES, "numerics.time_integrator", time_integrator_str);
    }

    switch (it_face->second) {
        case FaceReconstructionType::FIRST_ORDER:
            face_reconstruction = std::make_unique<FirstOrder>();
            break;
        case FaceReconstructionType::MUSCL:
            face_reconstruction = std::make_unique<MUSCL>();
            break;
        case FaceReconstructionType::TENO:
            face_reconstruction = std::make_unique<TENO>();
            break;
    }

    riemann_solver_type = it_riemann->second;

    switch (it_time->second) {
        case TimeIntegratorType::FE:
            time_integrator = std::make_unique<FE>();
            break;
        case TimeIntegratorType::RK4:
            time_integrator = std::make_unique<RK4>();
            break;
        case TimeIntegratorType::SSPRK3:
            time_integrator = std::make_unique<SSPRK3>();
            break;
    }

    face_reconstruction->set_mesh(mesh);
    face_reconstruction->set_boundaries(boundary_data);
    face_reconstruction->init(face_reconstruction_input);
    flux_weights = face_reconstruction->face_quad_weights;
    if (axisymmetric) init_axisymmetric_weights();
    if (is_mixture()) {
        scalar_reconstruction.init(mesh, boundary_data, mixture.n_species, *face_reconstruction);
    }
    double_flux = toml::find_or<bool>(input, "numerics", "double_flux", false);
    if (double_flux && !is_mixture()) {
        throw InputError("numerics.double_flux needs gas = \"mixture\".");
    }
    if (double_flux && simpler) {
        throw InputError("numerics.double_flux is not available with chemistry.coupling = \"simpler\" (its "
                         "thermodynamics are frozen over a step, which the reaction substep would leave stale).");
    }

    rhs_func = [this](State solution, State rhs, rtype t_stage) { calc_rhs(solution, rhs, t_stage); };
    check_nan = toml::find_or<bool>(input, "numerics", "check_nan", false);
    low_mach_cutoff = find_real_or(input, "numerics", "low_mach_cutoff", 0.1);
    if (!(low_mach_cutoff > 0.0_r)) {
        throw std::runtime_error("numerics: low_mach_cutoff must be positive (1 disables the low-Mach correction).");
    }
    const std::string flux = toml::find_or<std::string>(input, "numerics", "convective_flux", "riemann");
    if (flux != "riemann" && flux != "hybrid") {
        throw InputError("numerics.convective_flux = \"" + flux + "\" is not one of: riemann, hybrid.");
    }
    hybrid_flux = flux == "hybrid";
    const toml::value hybrid_input = toml::find_or(input, "numerics", "hybrid", toml::value(toml::table{}));
    if (!hybrid_flux && !hybrid_input.as_table().empty()) {
        throw InputError("[numerics.hybrid] needs numerics.convective_flux = \"hybrid\".");
    }
    hybrid_threshold = find_real_or(hybrid_input, "sensor_threshold", 0.65_r);
    hybrid_floor = find_real_or(hybrid_input, "upwind_floor", 0.0_r);
    if (!(hybrid_threshold >= 0.0_r && hybrid_threshold <= 1.0_r) || !(hybrid_floor >= 0.0_r && hybrid_floor <= 1.0_r)) {
        throw InputError("numerics.hybrid: sensor_threshold and upwind_floor must be in [0, 1].");
    }
}

void Solver::init_axisymmetric_weights() {
    // int_f F r dl = L / 2 sum_q w_q r_q F_q, exact in r for any rule (r is linear along a face)
    const auto & points = face_reconstruction->quadrature_face.h_points;
    const auto & weights = face_reconstruction->quadrature_face.h_weights;
    const uint32_t n_q = points.extent(0);
    flux_weights = Kokkos::View<rtype **>("flux_weights", mesh->n_faces, n_q);
    auto h_weights = Kokkos::create_mirror_view(flux_weights);
    for (uint32_t f = 0; f < mesh->n_faces; f++) {
        const uint32_t a = mesh->h_node_of_face(f, 0), b = mesh->h_node_of_face(f, 1);
        const rtype dy = mesh->h_node_coords(b, 1) - mesh->h_node_coords(a, 1);
        for (uint32_t q = 0; q < n_q; q++) {
            const rtype r_q = mesh->h_face_coords(f, 1) + 0.5_r * points(q, 0) * dy;
            h_weights(f, q) = weights(q) * std::max(r_q, 0.0_r);
        }
    }
    Kokkos::deep_copy(flux_weights, h_weights);
    geometric_source = Kokkos::View<rtype *>("geometric_source", mesh->n_cells);
    if (!is_mixture() && physics.is_viscous()) cell_mu = Kokkos::View<rtype *>("cell_mu", mesh->n_cells);
}

void Solver::init_run_parameters() {
    if (!input.contains("run")) {
        throw InputError("missing [run] table.");
    }
    const toml::value & run = input.at("run");
    if (run.contains("dt") == run.contains("cfl")) {
        throw InputError("run: specify exactly one of dt and cfl.");
    }
    if (!run.contains("n_steps") && !run.contains("t_stop") && !run.contains("t_wall_stop")) {
        throw InputError("run: specify at least one of n_steps, t_stop and t_wall_stop.");
    }
    use_cfl = run.contains("cfl");
    if (use_cfl) {
        cfl = find_real(input, "run", "cfl");
    } else {
        dt_fixed = find_real(input, "run", "dt");
        dt = dt_fixed;
    }
    n_steps = toml::find_or<uint64_t>(input, "run", "n_steps", 0);
    t_stop = find_real_or(input, "run", "t_stop", -1.0);
    t_wall_stop = find_real_or(input, "run", "t_wall_stop", -1.0);
}

void Solver::init_output() {
    check_interval = toml::find_or<uint32_t>(input, "output", "check_interval", 1);
    if (check_interval == 0) {
        throw InputError("output.check_interval must be positive.");
    }
    if (input.contains("forces")) {
        for (const auto & entry : toml::find<std::vector<toml::value>>(input, "forces")) {
            ForceMonitor monitor;
            monitor.zone = toml::find<std::string>(entry, "zone");
            FaceZone * zone = mesh->get_face_zone(monitor.zone);
            if (zone != nullptr && zone->get_type() != FaceZoneType::BOUNDARY) zone = nullptr;
            if (comm::allreduce(uint32_t(zone != nullptr), comm::Op::SUM) == 0) {
                throw std::runtime_error("forces: unknown boundary zone " + monitor.zone + ".");
            }
            std::vector<uint32_t> owned;
            for (uint32_t i = 0; zone && i < zone->n_faces(); i++) {
                const uint32_t f = zone->h_faces(i);
                if (static_cast<uint32_t>(mesh->h_cells_of_face(f, 0)) < mesh->n_owned()) owned.push_back(f);
            }
            monitor.faces = Kokkos::View<uint32_t *>("force_faces", owned.size());
            auto h_faces = Kokkos::create_mirror_view(monitor.faces);
            for (size_t i = 0; i < owned.size(); i++) h_faces(i) = owned[i];
            Kokkos::deep_copy(monitor.faces, h_faces);
            monitor.interval = toml::find_or<uint64_t>(entry, "interval", 1);
            if (monitor.interval == 0) {
                throw std::runtime_error("forces: interval must be positive.");
            }
            const std::string file = toml::find_or<std::string>(entry, "file", "forces_" + monitor.zone + ".csv");
            monitor.file = file;
            if (comm::is_root()) {
                const std::filesystem::path parent = std::filesystem::path(file).parent_path();
                if (!parent.empty()) std::filesystem::create_directories(parent);
                // init_output runs before the restart state is read, so check the input
                const bool resume = toml::find_or<std::string>(input, "initialize", "type", "") == "restart" &&
                                    std::filesystem::exists(file) && std::filesystem::file_size(file) > 0;
                monitor.out = std::make_shared<std::ofstream>(file, resume ? std::ios::app : std::ios::trunc);
                if (!resume) {
                    *monitor.out << "step,t";
                    for (const char * kind : {"pressure", "viscous"}) {
                        FOR_I_DIM *monitor.out << ",F" << "xyz"[i] << "_" << kind;
                    }
                    *monitor.out << "\n";
                }
            }
            force_monitors.push_back(monitor);
        }
    }
    if (input.contains("integrals")) {
        integral_monitor.interval = toml::find_or<uint64_t>(input, "integrals", "interval", 1);
        if (integral_monitor.interval == 0) {
            throw std::runtime_error("integrals: interval must be positive.");
        }
        if (!viscous_gradients.is_allocated()) {
            viscous_gradients = Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]>("viscous_gradients", mesh->n_cells);
            viscous_gradient = make_vertex_gradient(make_gradient(*mesh, boundary_data, W_cells, viscous_gradients), *mesh);
        }
        integral_monitor.budget = toml::find_or<bool>(input, "integrals", "budget", false);
        if (integral_monitor.budget && axisymmetric) {
            throw InputError("integrals.budget is for planar runs (the geometric source of axisymmetric runs is not "
                             "in it).");
        }
        const std::string file = toml::find_or<std::string>(input, "integrals", "file", "integrals.csv");
        integral_monitor.file = file;
        if (comm::is_root()) {
            const std::filesystem::path parent = std::filesystem::path(file).parent_path();
            if (!parent.empty()) std::filesystem::create_directories(parent);
            const bool resume = toml::find_or<std::string>(input, "initialize", "type", "") == "restart" &&
                                std::filesystem::exists(file) && std::filesystem::file_size(file) > 0;
            integral_monitor.out = std::make_shared<std::ofstream>(file, resume ? std::ios::app : std::ios::trunc);
            if (!resume) {
                *integral_monitor.out << "step,t,kinetic_energy,enstrophy,dilatation_squared,pressure_dilatation,"
                                          "velocity_squared,vorticity_squared,density_squared,temperature,"
                                          "temperature_squared"
                                       << (integral_monitor.budget ? ",ke_rate_convective,ke_rate_viscous,ke_rate_sgs,"
                                                                     "eps_numerical,pressure_work"
                                                                   : "")
                                       << (les_on && les.dynamic ? ",les_C" : "") << "\n";
            }
        }
    }
    probes.init(input, *mesh, species_names, toml::find_or<std::string>(input, "initialize", "type", "") == "restart");
    if (!input.contains("write_data")) {
        return;
    }
    std::vector<toml::value> outputs = toml::find<std::vector<toml::value>>(input, "write_data");
    for (const auto & output : outputs) {
        data_writers.push_back(std::make_unique<DataWriter>());
        data_writers.back()->init(output, data, mesh, restart_variables());
    }
}

void Solver::allocate_memory() {
    const uint32_t n_species = species_names.size();
    conservatives = StateView("conservatives", mesh->n_cells);
    species = SpeciesView("species", mesh->n_cells, n_species);
    primitives = Kokkos::View<rtype *[N_PRIMITIVE]>("primitives", mesh->n_cells);
    W_cells = Kokkos::View<rtype *[N_CONSERVATIVE]>("W_cells", mesh->n_cells);
    face_solution = Kokkos::View<rtype **[2][N_CONSERVATIVE]>("face_solution",
                                                              mesh->n_faces,
                                                              face_reconstruction->n_face_quadrature_points());
    face_flux = Kokkos::View<rtype *[N_CONSERVATIVE]>("face_flux", mesh->n_faces);
    cfl_local = Kokkos::View<rtype *>("cfl_local", mesh->n_cells);
    if (writes_variable(input, "P_MAX")) {
        p_max = Kokkos::View<rtype *>("p_max", mesh->n_cells);
        h_p_max = Kokkos::create_mirror_view(p_max);
    }
    bool vortex_output = false;
    for (const char * name : {"Q", "VORTICITY", "VORTICITY_X", "VORTICITY_Y", "VORTICITY_Z"}) {
        vortex_output = vortex_output || writes_variable(input, name);
    }
    if (vortex_output) {
        vortex_fields = Kokkos::View<rtype **>("vortex_fields", mesh->n_cells, N_DIM == 2 ? 2 : 4);
        h_vortex_fields = Kokkos::create_mirror_view(vortex_fields);
    }
    if (is_viscous() || vortex_output) {
        viscous_gradients = Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]>("viscous_gradients", mesh->n_cells);
        viscous_gradient = make_vertex_gradient(make_gradient(*mesh, boundary_data, W_cells, viscous_gradients), *mesh);
    }
    if (hybrid_flux) {
        cell_upwind = Kokkos::View<rtype *>("cell_upwind", mesh->n_cells);
        if (!viscous_gradients.is_allocated()) {
            viscous_gradients = Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]>("viscous_gradients", mesh->n_cells);
            viscous_gradient = make_vertex_gradient(make_gradient(*mesh, boundary_data, W_cells, viscous_gradients), *mesh);
        }
    }
    if (is_mixture() && is_viscous()) {
        cell_transport =Kokkos::View<rtype *[3]>("cell_transport", mesh->n_cells);
        h_cell_transport = Kokkos::create_mirror_view(cell_transport);
        const uint32_t n_diffusion = (mixture.transport.thermal_diffusion ? 2 : 1) * n_species;
        cell_diffusion = Kokkos::View<double **, Kokkos::LayoutRight>("cell_diffusion", mesh->n_cells, n_diffusion);
        transport_values = Kokkos::View<rtype **, Kokkos::LayoutRight>("transport_values", mesh->n_cells,
                                                                       N_DIM + 1 + n_species);
        transport_gradients = Kokkos::View<rtype ***, Kokkos::LayoutRight>("transport_gradients", mesh->n_cells,
                                                                           N_DIM + 1 + n_species, N_DIM);
        h_D = Kokkos::View<rtype **, Kokkos::LayoutRight, Kokkos::HostSpace>("D", mesh->n_cells, n_diffusion);
    }
    if (is_mixture()) {
        const uint32_t n_quad = face_reconstruction->n_face_quadrature_points();
        cell_scalars = ScalarView("cell_scalars", mesh->n_cells, n_species + 2);
        T_seed = Kokkos::View<rtype *>("T_seed", mesh->n_cells);
        h_T_seed = Kokkos::create_mirror_view(T_seed);
        face_thermo = Kokkos::View<rtype **[2][2]>("face_thermo", mesh->n_faces, n_quad);
        face_mdot = Kokkos::View<rtype **>("face_mdot", mesh->n_faces, n_quad);
        species_slots = Kokkos::View<rtype ***, Kokkos::LayoutRight>("species_slots", mesh->n_faces, 2, n_species);
        if (double_flux) {
            frozen_thermo = Kokkos::View<rtype *[2]>("frozen_thermo", mesh->n_cells);
            face_energy_1 = Kokkos::View<rtype *>("face_energy_1", mesh->n_faces);
        }
        if (auto * teno = dynamic_cast<TENO *>(face_reconstruction.get())) {
            cell_molar_mass = Kokkos::View<rtype *>("cell_molar_mass", mesh->n_cells);
            teno->set_mixture(Kokkos::subview(cell_scalars, Kokkos::ALL(), n_species), cell_molar_mass);
        }
        h_Y = Kokkos::View<rtype **, Kokkos::LayoutRight, Kokkos::HostSpace>("Y", mesh->n_cells, n_species);
        h_X = Kokkos::View<rtype **, Kokkos::LayoutRight, Kokkos::HostSpace>("X", mesh->n_cells, n_species);
        if (reacting) allocate_chemistry();
        if (radiating) h_qrad = Kokkos::View<rtype *, Kokkos::HostSpace>("QRAD", mesh->n_cells);
    }
    h_conservatives = Kokkos::create_mirror_view(conservatives);
    h_species = Kokkos::create_mirror_view(species);
    h_primitives = Kokkos::create_mirror_view(primitives);
    h_cfl_local = Kokkos::create_mirror_view(cfl_local);

    solution_vec.clear();
    rhs_vec.clear();
    solution_vec.push_back(state());
    for (uint8_t i = 1; i < time_integrator->get_n_solution_vectors(); i++) {
        solution_vec.emplace_back("solution", mesh->n_cells, n_species);
    }
    for (uint8_t i = 0; i < time_integrator->get_n_rhs_vectors(); i++) {
        rhs_vec.emplace_back("rhs", mesh->n_cells, n_species);
    }
}

void Solver::copy_host_to_device() {
    Kokkos::deep_copy(conservatives, h_conservatives);
    if (species.span() > 0) Kokkos::deep_copy(species, h_species);
    if (is_mixture()) Kokkos::deep_copy(T_seed, h_T_seed);
    if (reacting) Kokkos::deep_copy(chem_h, h_chem_h);
    if (p_max.is_allocated()) Kokkos::deep_copy(p_max, h_p_max);
    Kokkos::deep_copy(primitives, h_primitives);
}

void Solver::copy_device_to_host() {
    Kokkos::deep_copy(h_conservatives, conservatives);
    if (species.span() > 0) Kokkos::deep_copy(h_species, species);
    if (is_mixture()) {
        Kokkos::deep_copy(h_T_seed, T_seed);
        if (is_viscous()) {
            // Coefficients of the current state; W and the scalars are work arrays the next RHS refills
            update_cell_states(state(), false);
            update_transport();
            Kokkos::deep_copy(h_cell_transport, cell_transport);
            auto h_c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cell_diffusion);
            const auto & mech = mixture_model->mechanism();
            for (uint32_t c = 0; c < mesh->n_cells; c++) {
                const double rho = static_cast<double>(h_conservatives(c, 0));
                double n = 0.0;
                for (size_t k = 0; k < species_names.size(); k++) {
                    n += static_cast<double>(h_species(c, k)) / (rho * mech.species[k].molecular_weight);
                }
                for (size_t k = 0; k < species_names.size(); k++) {
                    h_D(c, k) = static_cast<rtype>(h_c(c, k) / (rho * mech.species[k].molecular_weight * n));
                }
                for (size_t k = species_names.size(); k < h_D.extent(1); k++) h_D(c, k) = static_cast<rtype>(h_c(c, k));
            }
        }
        if (reacting) {
            update_heat_release_rate();
            Kokkos::deep_copy(h_hrr, hrr);
            Kokkos::deep_copy(h_production, production);
            Kokkos::deep_copy(h_chem_h, chem_h);
            Kokkos::deep_copy(h_chem_cost, chem_cost);
        }
        std::vector<double> Y(species_names.size());
        for (uint32_t c = 0; c < mesh->n_cells; c++) {
            const double rho = static_cast<double>(h_conservatives(c, 0));
            for (size_t k = 0; k < Y.size(); k++) Y[k] = static_cast<double>(h_species(c, k)) / rho;
            const std::vector<double> X = mixture_model->mole_fractions(Y);
            for (size_t k = 0; k < Y.size(); k++) {
                h_Y(c, k) = static_cast<rtype>(Y[k]);
                h_X(c, k) = static_cast<rtype>(X[k]);
            }
        }
    }
    Kokkos::deep_copy(h_primitives, primitives);
    if (radiating) {
        for (uint32_t c = 0; c < mesh->n_cells; c++) {
            const PartialDensities rho_k{CellSpecies{&h_species(c, 0), h_species.stride(1)}, 1.0};
            h_qrad(c) = static_cast<rtype>(radiation.loss(static_cast<double>(h_primitives(c, N_DIM + 1)), rho_k));
        }
    }
    Kokkos::deep_copy(h_cfl_local, cfl_local);
    if (les_on) {
        halo.exchange(state());
        eddy_viscosity_of_state(mesh->n_cells);
        Kokkos::deep_copy(h_les_coefficients, les_coefficients);
    }
    if (tfles_on) Kokkos::deep_copy(h_tfles_fields, tfles_fields);
    if (pasr_on) Kokkos::deep_copy(h_chem_time_scale, chem_time_scale);
    statistics.copy_device_to_host();
    if (p_max.is_allocated()) Kokkos::deep_copy(h_p_max, p_max);
    if (vortex_fields.is_allocated()) {
        update_vortex_fields();
        Kokkos::deep_copy(h_vortex_fields, vortex_fields);
    }
    if (auto * teno = dynamic_cast<TENO *>(face_reconstruction.get())) {
        Kokkos::deep_copy(h_teno_sigma, teno->troubled);
    }
}

void Solver::register_data() {
    data.clear();
    data.reserve(CONSERVATIVE_NAMES.size() + species_names.size() + PRIMITIVE_NAMES.size() + 7);
    for (size_t i = 0; i < CONSERVATIVE_NAMES.size(); i++) {
        data.push_back(Data(CONSERVATIVE_NAMES[i], Kokkos::subview(h_conservatives, Kokkos::ALL(), i)));
    }
    for (size_t k = 0; k < species_names.size(); k++) {
        data.push_back(Data("RHOY_" + species_names[k], Kokkos::subview(h_species, Kokkos::ALL(), k)));
    }
    if (is_mixture()) {
        data.reserve(data.size() + 4 * species_names.size() + 6 + PRIMITIVE_NAMES.size() + 2);
        for (size_t k = 0; k < species_names.size(); k++) {
            data.push_back(Data("Y_" + species_names[k], Kokkos::subview(h_Y, Kokkos::ALL(), k)));
            data.push_back(Data("X_" + species_names[k], Kokkos::subview(h_X, Kokkos::ALL(), k)));
        }
        data.push_back(Data("T_SEED", h_T_seed));
        if (is_viscous()) {
            data.push_back(Data("MU", Kokkos::subview(h_cell_transport, Kokkos::ALL(), 0)));
            data.push_back(Data("LAMBDA", Kokkos::subview(h_cell_transport, Kokkos::ALL(), 1)));
            for (size_t k = 0; k < species_names.size(); k++) {
                data.push_back(Data("D_" + species_names[k], Kokkos::subview(h_D, Kokkos::ALL(), k)));
            }
            if (h_D.extent(1) > species_names.size()) {
                data.reserve(data.size() + species_names.size());
                for (size_t k = 0; k < species_names.size(); k++) {
                    data.push_back(Data("DT_" + species_names[k],
                                        Kokkos::subview(h_D, Kokkos::ALL(), species_names.size() + k)));
                }
            }
        }
        if (reacting) {
            data.push_back(Data("CHEM_H", h_chem_h));
            data.push_back(Data("CHEM_COST", h_chem_cost));
            data.push_back(Data("HRR", h_hrr));
            for (size_t k = 0; k < species_names.size(); k++) {
                data.push_back(Data("OMEGA_" + species_names[k], Kokkos::subview(h_production, Kokkos::ALL(), k)));
            }
        }
        if (radiating) data.push_back(Data("QRAD", h_qrad));
    }
    for (size_t i = 0; i < PRIMITIVE_NAMES.size(); i++) {
        data.push_back(Data(PRIMITIVE_NAMES[i], Kokkos::subview(h_primitives, Kokkos::ALL(), i)));
    }
    data.push_back(Data("CFL", h_cfl_local));
    if (les_on) data.push_back(Data("MU_T", Kokkos::subview(h_les_coefficients, Kokkos::ALL(), 0)));
    if (tfles_on) {
        data.push_back(Data("TF_F", Kokkos::subview(h_tfles_fields, Kokkos::ALL(), 0)));
        data.push_back(Data("TF_E", Kokkos::subview(h_tfles_fields, Kokkos::ALL(), 1)));
        data.push_back(Data("TF_OMEGA", Kokkos::subview(h_tfles_fields, Kokkos::ALL(), 2)));
    }
    if (pasr_on) data.push_back(Data("PASR_KAPPA", h_chem_time_scale));
    statistics.register_data(data);
    if (p_max.is_allocated()) data.push_back(Data("P_MAX", h_p_max));
    if (vortex_fields.is_allocated()) {
        data.push_back(Data("Q", Kokkos::subview(h_vortex_fields, Kokkos::ALL(), 0)));
        if constexpr (N_DIM == 2) {
            data.push_back(Data("VORTICITY", Kokkos::subview(h_vortex_fields, Kokkos::ALL(), 1)));
        } else {
            FOR_I_DIM {
                data.push_back(Data(std::string("VORTICITY_") + "XYZ"[i],
                                    Kokkos::subview(h_vortex_fields, Kokkos::ALL(), 1 + i)));
            }
        }
    }
    if (auto * teno = dynamic_cast<TENO *>(face_reconstruction.get())) {
        // Troubled-cell indicator: TENO stencil selection is active where it exceeds the threshold
        h_teno_sigma = Kokkos::create_mirror_view(teno->troubled);
        data.push_back(Data("TENO_SIGMA", h_teno_sigma));
    }
}

std::vector<std::string> Solver::restart_variables() const {
    std::vector<std::string> names(CONSERVATIVE_NAMES.begin(), CONSERVATIVE_NAMES.end());
    for (const auto & name : species_names) names.push_back("RHOY_" + name);
    if (is_mixture()) names.push_back("T_SEED");
    if (reacting) names.push_back("CHEM_H");
    if (p_max.is_allocated()) names.push_back("P_MAX");
    for (const auto & name : statistics.variables()) names.push_back(name);
    return names;
}

namespace {

/**
 * @brief Per-check solution diagnostics over owned cells: min density, min
 *        pressure, max Mach number and the number of TENO-troubled cells.
 */
struct DiagnosticsFunctor {
    StateView U;
    Kokkos::View<rtype *> sigma;  // TENO indicator, empty for other reconstructions
    rtype sigma_threshold;
    Euler physics;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t i_cell, rtype & min_rho, rtype & min_p, rtype & max_mach,
                    uint64_t & n_troubled) const {
        rtype cons[N_CONSERVATIVE], W[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE cons[i] = U(i_cell, i);
        physics.compute_W_from_conservatives(W, cons);
        const rtype rho = W[0], p = W[N_DIM + 1];
        rtype u2 = 0.0;
        FOR_I_DIM u2 += W[1 + i] * W[1 + i];
        min_rho = Kokkos::fmin(min_rho, rho);
        min_p = Kokkos::fmin(min_p, p);
        max_mach = Kokkos::fmax(max_mach, Kokkos::sqrt(u2) / physics.get_sound_speed_from_pressure_density(p, rho));
        if (sigma.extent(0) > 0 && sigma(i_cell) >= sigma_threshold) n_troubled++;
    }
};

constexpr int HEADER_EVERY = 25;

} // namespace

int Solver::run() {
    logging::section("Run");
    step_run_start = step;
    step_last_check = step;
    t_wall_run_start = timer.seconds();
    t_stepping_last_check = t_wall_stepping;
    progress_run_start = progress();
    n_progress_rows = 0;

    calc_dt();
    print_progress();
    if (step == 0) {
        Kokkos::Timer output_timer;
        copy_device_to_host();
        write_data(true);
        write_integrals();
        write_probes();
        t_wall_output += output_timer.seconds();
    }
    std::string stop;
    // Strang splitting with the half steps of consecutive steps fused into one
    // chemistry call, except where output, checks or the end read the state
    defer_chemistry = reacting && fuse_chemistry;
    while ((stop = stop_reason()).empty()) {
        Kokkos::Timer step_timer;
        calc_dt();
        take_step();
        if (chemistry_pending > 0.0 && state_needed()) {
            advance_chemistry(chemistry_pending);
            chemistry_pending = 0.0;
            halo_current = false;
        }
        if (p_max.is_allocated()) update_p_max();
        if (statistics.due(step, t)) {
            update_primitives();
            statistics.sample(t, cell_sampler(), mesh->n_owned());
        }
        check_fields();
        t_wall_stepping += step_timer.seconds();
        if (step % check_interval == 0) {
            Kokkos::Timer check_timer;
            print_progress();
            t_wall_checks += check_timer.seconds();
        }
        Kokkos::Timer output_timer;
        write_data();
        write_forces();
        write_integrals();
        write_probes();
        t_wall_output += output_timer.seconds();
    }
    defer_chemistry = false;
    if (step != step_last_check) print_progress();
    Kokkos::Timer output_timer;
    copy_device_to_host();
    write_data(true);
    t_wall_output += output_timer.seconds();
    print_summary(stop);
    return 0;
}

std::string Solver::stop_reason() const {
    if (n_steps > 0 && step >= n_steps) return "n_steps = " + logging::count(n_steps) + " reached";
    if (t_stop > 0 && t >= t_stop) return "t_stop = " + logging::real(double(t_stop)) + " reached";
    if (t_wall_stop > 0 && timer.seconds() >= double(t_wall_stop)) {
        return "t_wall_stop = " + logging::duration(double(t_wall_stop)) + " reached";
    }
    return "";
}

bool Solver::state_needed() const {
    if (!stop_reason().empty() || step % check_interval == 0) return true;
    for (const auto & writer : data_writers) {
        if (writer->due(step, t)) return true;
    }
    for (const auto & monitor : force_monitors) {
        if (step % monitor.interval == 0) return true;
    }
    return (integral_monitor.interval > 0 && step % integral_monitor.interval == 0) || statistics.due(step, t) ||
           probes.due(step);
}

double Solver::progress() const {
    double f = 0.0;
    if (n_steps > 0) f = std::max(f, static_cast<double>(step) / n_steps);
    if (t_stop > 0) f = std::max(f, static_cast<double>(t / t_stop));
    if (t_wall_stop > 0) f = std::max(f, timer.seconds() / double(t_wall_stop));
    return std::min(f, 1.0);
}

void Solver::print_progress() {
    auto * teno = dynamic_cast<TENO *>(face_reconstruction.get());
    DiagnosticsFunctor functor{conservatives, teno ? teno->troubled : Kokkos::View<rtype *>(),
                               teno ? teno->sigma_threshold : rtype(0), physics};
    rtype min_rho = 0.0, min_p = 0.0, max_mach = 0.0;
    uint64_t n_troubled = 0;
    Kokkos::parallel_reduce("diagnostics", Kokkos::RangePolicy<>(0, mesh->n_owned()), functor,
                            Kokkos::Min<rtype>(min_rho), Kokkos::Min<rtype>(min_p), Kokkos::Max<rtype>(max_mach),
                            Kokkos::Sum<uint64_t>(n_troubled));
    std::array<rtype, 6> mix = {};  // min rho, min p, -max Ma, min T, -max T, -max |sum Y - 1|
    if (is_mixture()) {
        mix = mixture_diagnostics();
        min_rho = mix[0];
        min_p = mix[1];
        max_mach = -mix[2];
    }
    const auto mins = comm::allreduce(std::array<rtype, 3>{min_rho, min_p, -max_mach}, comm::Op::MIN);
    n_troubled = comm::allreduce(n_troubled, comm::Op::SUM);

    using logging::format;
    if (n_progress_rows % HEADER_EVERY == 0) {
        std::string header = format("%8s %10s %8s %6s  %9s %7s %7s  %9s %9s %6s", "step", "t", "dt", "done",
                                    "wall/step", "cells/s", "ETA", "min rho", "min p", "max Ma");
        if (teno) header += format(" %8s", "troubled");
        if (is_mixture()) header += format(" %8s %8s %9s", "min T", "max T", "sum Y-1");
        if (reacting) header += format(" %8s %6s", "reacting", "chem/c");
        logging::line(logging::style(header, logging::Style::BOLD));
    }
    n_progress_rows++;

    const double now = timer.seconds();
    const uint64_t steps = step - step_last_check;
    const double stepping = t_wall_stepping - t_stepping_last_check;
    const bool timed = steps > 0 && stepping > 0.0;
    const double f = progress();
    const double df = f - progress_run_start;
    const double eta = df > 0.0 ? (now - t_wall_run_start) * (1.0 - f) / df : -1.0;
    const std::string eta_text = f >= 1.0 ? "0 s" : (eta >= 0.0 && eta < 1.0 ? "<1 s" : logging::duration(eta));
    std::string row = format("%8llu %10.4e %8.2e %5.1f%%  %9s %7s %7s  %9.2e %9.2e %6.3f",
                             static_cast<unsigned long long>(step), static_cast<double>(t), static_cast<double>(dt),
                             100.0 * f, timed ? logging::duration(stepping / steps).c_str() : "--",
                             timed ? logging::si(n_cells_global * steps / stepping).c_str() : "--",
                             eta_text.c_str(), static_cast<double>(mins[0]),
                             static_cast<double>(mins[1]), static_cast<double>(-mins[2]));
    if (teno) row += format(" %7.2f%%", 100.0 * n_troubled / n_cells_global);
    if (is_mixture()) {
        row += format(" %8.1f %8.1f %9.2e", static_cast<double>(mix[3]), static_cast<double>(-mix[4]),
                      static_cast<double>(-mix[5]));
    }
    if (reacting) {
        const auto [active, max_cost] = chemistry_statistics();
        row += format(" %7.2f%% %6.0f", 100.0 * active / n_cells_global, max_cost);
    }
    logging::line(row);

    step_last_check = step;
    t_stepping_last_check = t_wall_stepping;
}

void Solver::print_setup() const {
    using logging::real;
    logging::section("Mesh");
    logging::items(mesh_summary);

    logging::section("Physics");
    logging::items(is_mixture() ? mixture_model->summary() : physics.summary());
    logging::items(source_summary);
    logging::item("Initial state", initial_state);

    logging::section("Numerics");
    logging::items(face_reconstruction->summary());
    logging::item("Riemann solver", RIEMANN_SOLVER_NAMES.at(riemann_solver_type));
    if (hybrid_flux) {
        logging::item("Convective flux", "hybrid: KEEP central, Riemann solver where the compression sensor exceeds " +
                                             real(double(hybrid_threshold)) + " (upwind floor " +
                                             real(double(hybrid_floor)) + ")");
    }
    logging::item("Time integrator", TIME_INTEGRATOR_NAMES.at(time_integrator->get_type()));
    logging::item("Time step", use_cfl ? "CFL " + real(double(cfl)) : "dt " + real(double(dt_fixed)) + " (fixed)");
    std::string stop;
    if (n_steps > 0) stop += "n_steps = " + logging::count(n_steps);
    if (t_stop > 0) stop += (stop.empty() ? "" : ", ") + std::string("t = ") + real(double(t_stop));
    if (t_wall_stop > 0) stop += (stop.empty() ? "" : ", ") + std::string("wall ") + logging::duration(double(t_wall_stop));
    logging::item("Stop at", stop);
    if (check_nan) logging::item("NaN check", "every step");
    if (les_on) logging::items(les.summary());
    if (tfles_on) logging::items(thickened_flame.summary());
    if (pasr_on) logging::items(pasr.summary());
    if (double_flux) logging::item("Double flux", "frozen gamma and e0 per cell and step (not energy conservative)");
    if (reacting) {
        logging::item("Chemistry", std::to_string(kinetics.n_reactions) + " reactions, " +
                                       (simpler ? "SIMPLER balanced splitting" : "Strang splitting") +
                                       (fuse_chemistry ? " (half steps fused)" : "") + ", RODAS (rtol " +
                                       real(chemistry_options.integrator.rtol) + ", atol " +
                                       real(chemistry_options.atol_Y) + ")" +
                                       (T_frozen > 0.0 ? ", frozen below " + real(T_frozen) + " K" : "") + ", " +
                                       (cell_chemistry.lanes() == 1 ? std::string("one thread per cell")
                                                                    : std::to_string(cell_chemistry.lanes()) +
                                                                          " lanes per cell"));
    }

    logging::section("Boundaries");
    logging::items(boundary_summary);

    logging::section("Output");
    logging::item("Progress", "every " + logging::count(check_interval) + " steps");
    for (const auto & writer : data_writers) {
        const auto [kind, text] = writer->summary();
        logging::item(kind, text);
    }
    for (const auto & monitor : force_monitors) {
        logging::item("forces", monitor.file + " (" + monitor.zone + ") every " + logging::count(monitor.interval) +
                                    " steps");
    }
    if (integral_monitor.interval > 0) {
        logging::item("integrals", integral_monitor.file + " every " + logging::count(integral_monitor.interval) +
                                       " steps");
    }
    if (statistics.enabled()) logging::item("statistics", statistics.summary());
    logging::items(probes.summary());
}

void Solver::print_summary(const std::string & stop) const {
    using logging::duration;
    const double total = timer.seconds();
    const uint64_t steps = step - step_run_start;
    logging::section("Summary");
    logging::item("Stopped", stop + " at step " + logging::count(step) + ", t = " + logging::real(double(t)));
    logging::item("Wall time", duration(total) + ": setup " + duration(t_wall_setup) + ", time stepping " +
                                   duration(t_wall_stepping) + ", diagnostics " + duration(t_wall_checks) +
                                   ", output " + duration(t_wall_output));
    if (reacting && t_wall_stepping > 0.0) {
        logging::item("Chemistry", duration(t_wall_chemistry) + " (" +
                                       logging::format("%.0f%%", 100.0 * t_wall_chemistry / t_wall_stepping) +
                                       " of time stepping)");
    }
    if (steps > 0 && t_wall_stepping > 0.0) {
        logging::item("Throughput", logging::si(n_cells_global * steps / t_wall_stepping) + " cells/s, " +
                                        duration(t_wall_stepping / steps) + " per step over " +
                                        logging::count(steps) + " steps");
    }
    std::string files;
    auto add = [&](const std::string & text) { files += (files.empty() ? "" : ", ") + text; };
    for (const auto & writer : data_writers) {
        const uint64_t n = writer->files_written();
        if (writer->get_format() == DataFormat::RESTART) {
            add(logging::count(n) + " restart");
        } else {
            add(logging::count(n) + " vtu (" + writer->get_prefix() + ".pvd)");
        }
    }
    for (const auto & monitor : force_monitors) add(monitor.file);
    if (integral_monitor.interval > 0) add(integral_monitor.file);
    for (const auto & file : probes.files()) add(file);
    if (!files.empty()) logging::item("Files", files);
}

void Solver::check_fields() {
    if (!check_nan) {
        return;
    }
    StateView U = conservatives;
    uint32_t n_bad = 0;
    Kokkos::parallel_reduce("check_nan", mesh->n_owned(), KOKKOS_LAMBDA(const uint32_t i_cell, uint32_t & bad) {
        FOR_I_CONSERVATIVE {
            if (!Kokkos::isfinite(U(i_cell, i))) bad++;
        }
    }, n_bad);
    n_bad = comm::allreduce(n_bad, comm::Op::SUM);
    if (species.span() > 0) {
        SpeciesView Y = species;
        const uint32_t n_species = Y.extent(1);
        uint32_t n_bad_species = 0;
        Kokkos::parallel_reduce("check_nan_species", mesh->n_owned(), KOKKOS_LAMBDA(const uint32_t i_cell, uint32_t & bad) {
            for (uint32_t k = 0; k < n_species; k++) {
                if (!Kokkos::isfinite(Y(i_cell, k))) bad++;
            }
        }, n_bad_species);
        n_bad += comm::allreduce(n_bad_species, comm::Op::SUM);
    }
    if (n_bad > 0) {
        std::stringstream msg;
        msg << "Non-finite values found in solution at step " << step << ", t = " << t << ".";
        throw std::runtime_error(msg.str());
    }
}

void Solver::write_data(bool force) {
    bool any_due = force;
    for (auto & writer : data_writers) {
        any_due = any_due || writer->due(step, t);
    }
    if (!any_due) {
        return;
    }
    update_primitives();
    copy_device_to_host();
    const RestartAttributes attributes = statistics.attributes();
    const RestartFaces faces = characteristic_restart_faces();
    for (auto & writer : data_writers) {
        writer->write(step, t, force, attributes, faces);
    }
}

void Solver::write_probes() {
    if (!probes.due(step)) return;
    update_primitives();
    probes.write(step, t, cell_sampler());
}

void Solver::take_step() {
    if (simpler) {
        take_simpler_step();
        halo_current = false;
        Kokkos::fence();
        step++;
        t += dt;
        return;
    }
    if (reacting) {
        Kokkos::deep_copy(chem_cost, 0.0_r);
        advance_chemistry(chemistry_pending + 0.5 * static_cast<double>(dt));
        chemistry_pending = 0.0;
        halo_current = false;
    }
    if (double_flux) {
        freeze_thermodynamics();
        cells_frozen = true;
    }
    time_integrator->take_step(t, dt, solution_vec, rhs_vec, rhs_func);
    if (double_flux) {
        cells_frozen = false;
        reset_energy();
    }
    if (reacting) {
        if (defer_chemistry) {
            chemistry_pending = 0.5 * static_cast<double>(dt);
        } else {
            advance_chemistry(0.5 * static_cast<double>(dt));
        }
    }
    halo_current = false;
    Kokkos::fence();
    step++;
    t += dt;
}

void Solver::update_primitives() {
    if (is_mixture()) {
        update_primitives_mixture();
        return;
    }
    const Euler phys = physics;
    StateView U = conservatives;
    Kokkos::View<rtype *[N_PRIMITIVE]> P = primitives;
    Kokkos::parallel_for("update_primitives", mesh->n_cells, KOKKOS_LAMBDA(const uint32_t i_cell) {
        rtype cons[N_CONSERVATIVE];
        rtype prim[N_PRIMITIVE];
        FOR_I_CONSERVATIVE cons[i] = U(i_cell, i);
        phys.compute_primitives_from_conservatives(prim, cons);
        FOR_I_PRIMITIVE P(i_cell, i) = prim[i];
    });
}

void Solver::update_p_max() {
    update_primitives();
    Kokkos::View<rtype *> peak = p_max;
    Kokkos::View<rtype *[N_PRIMITIVE]> P = primitives;
    Kokkos::parallel_for("update_p_max", mesh->n_cells, KOKKOS_LAMBDA(const uint32_t i_cell) {
        peak(i_cell) = Kokkos::max(peak(i_cell), P(i_cell, N_DIM));
    });
}

void Solver::calc_dt() {
    // Halo values are stale after the last stage of the previous step; the
    // first stage of the next one reuses them
    halo.exchange(state());
    halo_current = true;
    const rtype dt_cfl1 = calc_dt_cfl1();
    dt = use_cfl ? cfl * dt_cfl1 : dt_fixed;
    dt = std::min(dt, sponge_dt_max);
    // Land exactly on t_stop and on time-based output times
    rtype t_target = (t_stop > 0) ? t_stop : std::numeric_limits<rtype>::infinity();
    for (const auto & writer : data_writers) {
        t_target = std::min(t_target, writer->next_time());
    }
    if (t + dt > t_target && t_target > t) {
        dt = t_target - t;
    }
    if (!(dt > 0.0_r)) {
        throw std::runtime_error("Invalid dt: " + std::to_string(dt) + ".");
    }
    // cfl_local holds dt_cfl1 per cell; convert to local CFL number
    Kokkos::View<rtype *> c = cfl_local;
    const rtype dt_ = dt;
    Kokkos::parallel_for("local_cfl", mesh->n_cells, KOKKOS_LAMBDA(const uint32_t i) {
        c(i) = dt_ / c(i);
    });
}

/**
 * @brief Per-cell stable time step for CFL = 1 (Blazek eqs. 6.20-6.21, C = 4):
 *        dt_i = 2 V_i / (sum_f (|u_n| + a)_f A_f + 4 nu_eff sum_f A_f^2 / V_i),
 *        with the face wave speed taken as the max over the two adjacent cells
 *        and nu_eff = max(4/3, gamma/Pr) mu / rho for viscous flow. Half the
 *        sums over faces stand for Blazek's sums of projected areas.
 */
struct TimeStepFunctor {
    Kokkos::View<uint32_t *> offsets_faces_of_cell;
    Kokkos::View<uint32_t *> faces_of_cell;
    Kokkos::View<int32_t *[2]> cells_of_face;
    Kokkos::View<rtype *[N_DIM]> face_normals;
    Kokkos::View<rtype *> face_area;
    Kokkos::View<rtype *> cell_volume;
    StateView conservatives;
    Kokkos::View<rtype *> dt_local;
    Euler physics;
    Kokkos::View<rtype *[N_DIM]> radius_coords;  // axisymmetric runs: cell_coords, else empty
    Kokkos::View<rtype *[3]> sgs;                // LES: (cell, [mu_t, ...]), else empty
    rtype Pr_t = 1.0_r;

    KOKKOS_INLINE_FUNCTION
    rtype wave_speed(const int32_t i_cell, const rtype * n) const {
        rtype cons[N_CONSERVATIVE], W[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE cons[i] = conservatives(i_cell, i);
        physics.compute_W_from_conservatives(W, cons);
        const rtype u_n = dot<N_DIM>(W + 1, n);
        return Kokkos::fabs(u_n) + physics.get_sound_speed_from_pressure_density(W[N_DIM + 1], W[0]);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t i_cell, rtype & dt_min) const {
        rtype sum = 0.0;
        rtype sum_area2 = 0.0;
        for (uint32_t k = offsets_faces_of_cell(i_cell); k < offsets_faces_of_cell(i_cell + 1); k++) {
            const uint32_t i_face = faces_of_cell(k);
            rtype n[N_DIM];
            rtype n_vec[N_DIM];
            FOR_I_DIM n_vec[i] = face_normals(i_face, i);
            unit<N_DIM>(n_vec, n);
            const int32_t c0 = cells_of_face(i_face, 0);
            const int32_t c1 = cells_of_face(i_face, 1);
            rtype lambda = wave_speed(c0, n);
            if (c1 >= 0) lambda = Kokkos::fmax(lambda, wave_speed(c1, n));
            sum += lambda * face_area(i_face);
            sum_area2 += face_area(i_face) * face_area(i_face);
        }
        if (physics.is_viscous()) {
            rtype cons[N_CONSERVATIVE], W[N_CONSERVATIVE];
            FOR_I_CONSERVATIVE cons[i] = conservatives(i_cell, i);
            physics.compute_W_from_conservatives(W, cons);
            const rtype T = W[N_DIM + 1] / (W[0] * physics.R);
            const rtype mu = physics.viscosity(T);
            rtype coeff = Kokkos::fmax(4.0_r / 3.0_r, physics.gamma / physics.Pr) * mu / W[0];
            if (sgs.extent(0) > 0) {
                const rtype mu_t = sgs(i_cell, 0);
                coeff = Kokkos::fmax(4.0_r / 3.0_r * (mu + mu_t), physics.gamma * (mu / physics.Pr + mu_t / Pr_t)) /
                        W[0];
            }
            sum += 4.0_r * coeff * sum_area2 / cell_volume(i_cell);
            if (radius_coords.extent(0) > 0) {
                // Decay of u_r by the hoop stress
                const rtype r = radius_coords(i_cell, 1);
                sum += coeff * cell_volume(i_cell) / (r * r);
            }
        }
        const rtype dt_i = 2.0_r * cell_volume(i_cell) / sum;
        dt_local(i_cell) = dt_i;
        dt_min = Kokkos::fmin(dt_min, dt_i);
    }
};

rtype Solver::calc_dt_cfl1() {
    if (is_mixture()) return calc_dt_cfl1_mixture();
    TimeStepFunctor functor{mesh->offsets_faces_of_cell,
                            mesh->faces_of_cell,
                            mesh->cells_of_face,
                            mesh->face_normals,
                            mesh->face_measure,
                            mesh->cell_measure,
                            conservatives,
                            cfl_local,
                            physics,
                            axisymmetric ? mesh->cell_coords : Kokkos::View<rtype *[N_DIM]>(),
                            les_coefficients,
                            les.Pr_t};
    if (les_on) {
        eddy_viscosity_of_state(mesh->n_owned());
        if (les.dynamic) {
            update_dynamic_constant();
            update_eddy_viscosity(mesh->n_owned());
        }
    }
    rtype dt_min = std::numeric_limits<rtype>::max();
    Kokkos::parallel_reduce("time_step", mesh->n_owned(), functor, Kokkos::Min<rtype>(dt_min));
    return comm::allreduce(dt_min, comm::Op::MIN);
}

/**
 * @brief Traction of the fluid on boundary faces, t = p n - tau . n with n
 *        pointing out of the fluid. Pressure is the adjacent cell's; on walls
 *        the velocity gradient is corrected with the wall velocity as in the
 *        viscous flux.
 */
struct ForceFunctor {
    Kokkos::View<uint32_t *> faces;
    Kokkos::View<rtype *[N_DIM]> normals;
    Kokkos::View<rtype *[N_DIM]> face_coords;
    Kokkos::View<rtype *[N_DIM]> cell_coords;
    Kokkos::View<int32_t *[2]> cells_of_face;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W;
    Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients;
    BoundaryData boundaries;
    Euler physics;
    bool viscous;
    Kokkos::View<rtype *[3]> mixture_transport;  // viscous mixtures: (cell, [mu, ...]), else empty
    bool axisymmetric = false;  // per radian: tractions weighted by the face radius

    struct value_type {
        rtype v[2 * N_DIM];
    };

    KOKKOS_INLINE_FUNCTION
    void init(value_type & sum) const {
        for (int i = 0; i < 2 * N_DIM; i++) sum.v[i] = 0.0;
    }

    KOKKOS_INLINE_FUNCTION
    void join(value_type & dst, const value_type & src) const {
        for (int i = 0; i < 2 * N_DIM; i++) dst.v[i] += src.v[i];
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t k, value_type & total) const {
        rtype * sum = total.v;
        const uint32_t f = faces(k);
        const int32_t c = cells_of_face(f, 0);
        rtype n_A[N_DIM];
        FOR_I_DIM n_A[i] = normals(f, i);
        rtype n[N_DIM];
        unit<N_DIM>(n_A, n);
        if (axisymmetric) FOR_I_DIM n_A[i] *= face_coords(f, 1);
        FOR_I_DIM sum[i] += W(c, N_DIM + 1) * n_A[i];
        if (!viscous) return;
        rtype g[N_DIM][N_DIM];
        for (uint8_t v = 0; v < N_DIM; v++) {
            FOR_I_DIM g[v][i] = gradients(c, 1 + v, i);
        }
        const BoundaryCondition & bc = boundaries.bcs(boundaries.face_bc(f));
        if (bc.is_wall()) {
            rtype dn = 0.0;
            FOR_I_DIM dn += (face_coords(f, i) - cell_coords(c, i)) * n[i];
            for (uint8_t v = 0; v < N_DIM; v++) {
                const rtype correction = (bc.data[1 + v] - W(c, 1 + v)) / dn - dot<N_DIM>(g[v], n);
                FOR_I_DIM g[v][i] += correction * n[i];
            }
        }
        const rtype mu = mixture_transport.extent(0) > 0 ? mixture_transport(c, 0)
                                                         : physics.viscosity(W(c, N_DIM + 1) / (W(c, 0) * physics.R));
        rtype tau_n[N_DIM];
        rtype u[N_DIM];
        FOR_I_DIM u[i] = bc.is_wall() ? bc.data[1 + i] : W(c, 1 + i);
        viscous_traction(mu, g, n_A, tau_n, axisymmetric ? hoop_divergence(face_coords(f, 1), u, g) : 0.0_r);
        FOR_I_DIM sum[N_DIM + i] -= tau_n[i];
    }
};

std::array<rtype, 2 * N_DIM> Solver::calc_force(const Kokkos::View<uint32_t *> & faces) {
    const Euler phys = physics;
    StateView U = conservatives;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W = W_cells;
    if (is_mixture()) {
        update_cell_states(state(), false);
    } else Kokkos::parallel_for("force_W", mesh->n_cells, KOKKOS_LAMBDA(const uint32_t i_cell) {
        rtype cons[N_CONSERVATIVE], W_c[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE cons[i] = U(i_cell, i);
        phys.compute_W_from_conservatives(W_c, cons);
        FOR_I_CONSERVATIVE W(i_cell, i) = W_c[i];
    });
    if (is_viscous()) {
        Kokkos::parallel_for("force_gradients", mesh->n_cells, viscous_gradient);
        if (is_mixture()) update_transport();
    }
    ForceFunctor functor{faces, mesh->face_normals, mesh->face_coords, mesh->cell_coords, mesh->cells_of_face,
                         W_cells, viscous_gradients, boundary_data, physics, is_viscous(),
                         is_mixture() ? cell_transport : Kokkos::View<rtype *[3]>(), axisymmetric};
    ForceFunctor::value_type result;
    Kokkos::parallel_reduce("force", faces.extent(0), functor, result);
    std::array<rtype, 2 * N_DIM> F;
    for (int i = 0; i < 2 * N_DIM; i++) F[i] = result.v[i];
    return comm::allreduce(F, comm::Op::SUM);
}

void Solver::write_forces() {
    for (auto & monitor : force_monitors) {
        if (step % monitor.interval != 0) continue;
        const auto F = calc_force(monitor.faces);
        if (!monitor.out) continue;
        *monitor.out << step << "," << std::setprecision(12) << t;
        for (const rtype f : F) *monitor.out << "," << f;
        *monitor.out << "\n";
        monitor.out->flush();
    }
}

std::array<rtype, N_CONSERVATIVE> Solver::integrate_conservatives() {
    std::array<rtype, N_CONSERVATIVE> total;
    StateView U = conservatives;
    Kokkos::View<rtype *> vol = mesh->cell_measure;
    for (uint8_t i_var = 0; i_var < N_CONSERVATIVE; i_var++) {
        rtype sum = 0.0;
        Kokkos::parallel_reduce("integrate", mesh->n_owned(), KOKKOS_LAMBDA(const uint32_t i, rtype & s) {
            s += U(i, i_var) * vol(i);
        }, sum);
        total[i_var] = sum;
    }
    return comm::allreduce(total, comm::Op::SUM);
}

/**
 * @brief Per-cell contributions to Solver::integrate_flow_statistics.
 */
struct FlowStatisticsFunctor {
    Kokkos::View<rtype *[N_CONSERVATIVE]> W;
    Kokkos::View<rtype *[N_PRIMITIVE]> primitives;
    Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients;
    Kokkos::View<rtype *> volume;
    Kokkos::View<rtype *[N_DIM]> radius_coords;  // axisymmetric runs: cell_coords, else empty

    struct value_type {
        rtype v[N_FLOW_STATISTICS];
    };

    KOKKOS_INLINE_FUNCTION
    void init(value_type & sum) const {
        for (int i = 0; i < N_FLOW_STATISTICS; i++) sum.v[i] = 0.0;
    }

    KOKKOS_INLINE_FUNCTION
    void join(value_type & dst, const value_type & src) const {
        for (int i = 0; i < N_FLOW_STATISTICS; i++) dst.v[i] += src.v[i];
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c, value_type & sum) const {
        const rtype rho = W(c, 0);
        const rtype T = primitives(c, N_DIM + 1);
        const rtype V = volume(c);
        rtype u2 = 0.0, div = 0.0;
        FOR_I_DIM {
            u2 += W(c, 1 + i) * W(c, 1 + i);
            div += gradients(c, 1 + i, i);
        }
        if (radius_coords.extent(0) > 0) div += W(c, 2) / radius_coords(c, 1);
        // gradients(c, 1 + k, i) = d u_k / d x_i
        rtype omega2 = 0.0;
        if constexpr (N_DIM == 2) {
            const rtype w = gradients(c, 2, 0) - gradients(c, 1, 1);
            omega2 = w * w;
        } else {
            for (uint8_t k = 0; k < 3; k++) {
                const uint8_t a = (k + 1) % 3, b = (k + 2) % 3;
                const rtype w = gradients(c, 1 + b, a) - gradients(c, 1 + a, b);
                omega2 += w * w;
            }
        }
        sum.v[0] += 0.5_r * rho * u2 * V;
        sum.v[1] += 0.5_r * rho * omega2 * V;
        sum.v[2] += div * div * V;
        sum.v[3] += W(c, N_DIM + 1) * div * V;
        sum.v[4] += u2 * V;
        sum.v[5] += omega2 * V;
        sum.v[6] += rho * rho * V;
        sum.v[7] += T * V;
        sum.v[8] += T * T * V;
    }
};

void Solver::update_velocity_gradients() {
    halo.exchange(state());
    update_boundary_states(t);
    const Euler phys = physics;
    StateView U = conservatives;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W = W_cells;
    if (is_mixture()) {
        update_cell_states(state(), false);
    } else Kokkos::parallel_for("statistics_W", mesh->n_cells, KOKKOS_LAMBDA(const uint32_t i_cell) {
        rtype cons[N_CONSERVATIVE], W_c[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE cons[i] = U(i_cell, i);
        phys.compute_W_from_conservatives(W_c, cons);
        FOR_I_CONSERVATIVE W(i_cell, i) = W_c[i];
    });
    update_primitives();
    if (!face_reconstruction->cell_gradients(W_cells, viscous_gradients, mesh->n_owned())) {
        Kokkos::parallel_for("statistics_gradients", mesh->n_owned(), viscous_gradient);
    }
}

std::array<rtype, N_FLOW_STATISTICS> Solver::integrate_flow_statistics() {
    update_velocity_gradients();
    FlowStatisticsFunctor functor{W_cells, primitives, viscous_gradients, mesh->cell_measure,
                                  axisymmetric ? mesh->cell_coords : Kokkos::View<rtype *[N_DIM]>()};
    FlowStatisticsFunctor::value_type result;
    Kokkos::parallel_reduce("statistics", mesh->n_owned(), functor, result);
    std::array<rtype, N_FLOW_STATISTICS> sums;
    for (int i = 0; i < N_FLOW_STATISTICS; i++) sums[i] = result.v[i];
    return comm::allreduce(sums, comm::Op::SUM);
}

void Solver::update_vortex_fields() {
    update_velocity_gradients();
    Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> g = viscous_gradients;
    Kokkos::View<rtype **> out = vortex_fields;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W = W_cells;
    Kokkos::View<rtype *[N_DIM]> radius_coords = axisymmetric ? mesh->cell_coords : Kokkos::View<rtype *[N_DIM]>();
    Kokkos::parallel_for("vortex_fields", mesh->n_owned(), KOKKOS_LAMBDA(const uint32_t c) {
        // g(c, 1 + k, i) = d u_k / d x_i, and |S|^2 - |Omega|^2 = d u_k / d x_i d u_i / d x_k,
        // plus the hoop strain (u_r / r)^2 in axisymmetric runs
        rtype gg = 0.0;
        FOR_I_DIM {
            for (uint8_t k = 0; k < N_DIM; k++) gg += g(c, 1 + k, i) * g(c, 1 + i, k);
        }
        if (radius_coords.extent(0) > 0) {
            const rtype hoop = W(c, 2) / radius_coords(c, 1);
            gg += hoop * hoop;
        }
        out(c, 0) = -0.5_r * gg;
        if constexpr (N_DIM == 2) {
            out(c, 1) = g(c, 2, 0) - g(c, 1, 1);
        } else {
            for (uint8_t k = 0; k < 3; k++) {
                const uint8_t a = (k + 1) % 3, b = (k + 2) % 3;
                out(c, 1 + k) = g(c, 1 + b, a) - g(c, 1 + a, b);
            }
        }
    });
}

void Solver::write_integrals() {
    if (integral_monitor.interval == 0 || step % integral_monitor.interval != 0) return;
    const auto sums = integrate_flow_statistics();
    KineticEnergyBudget rates;
    if (integral_monitor.budget) rates = kinetic_energy_budget();
    if (!integral_monitor.out) return;
    *integral_monitor.out << step << "," << std::setprecision(12) << t;
    for (const rtype s : sums) *integral_monitor.out << "," << s;
    if (integral_monitor.budget) {
        // The exact convective rate is the pressure work; the rest is numerical
        *integral_monitor.out << "," << rates.convective << "," << rates.viscous << "," << rates.sgs << ","
                              << rates.pressure_work - rates.convective << "," << rates.pressure_work;
    }
    if (les_on && les.dynamic) *integral_monitor.out << "," << les.C;
    *integral_monitor.out << "\n";
    integral_monitor.out->flush();
}
