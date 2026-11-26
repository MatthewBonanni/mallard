/**
 * @file solver_mesh.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief The mesh of the Solver: reading and distributing it, the halo, the partition-face
 *        plans of the RHS and the axisymmetric weights.
 * @version 0.1
 * @date 2026-11-24
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "solver.h"
#include "boundary_coordinates.h"
#include "boundary_profiles.h"
#include "gated.h"

#include "diagnostic_code.h"
#include "input.h"

#include <algorithm>
#include <unordered_set>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>

#include <Kokkos_Core.hpp>

#include "comm.h"
#include "device_comm.h"
#include "log.h"
#include "mesh_block.h"
#include "region_input.h"
#include "region_motion.h"
#include "partition.h"
#include "provenance.h"
#include "run_report.h"
#include "gradient.h"

namespace {

/**
 * @brief Cell counts by type, faces, extent and cell size range over all ranks.
 *        Each rank counts its owned cells; a face between two ranks is counted by
 *        the rank owning the side with the lower global cell id.
 */
MeshSetup describe_mesh(const Mesh & mesh) {
    const bool distributed = mesh.n_global_cells > 0;
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
            counts[MAX_NODES] += mesh.h_global_cell_id[static_cast<size_t>(mine)] < mesh.h_global_cell_id[static_cast<size_t>(other)];
        }
    }
    counts = comm::allreduce(counts, comm::Op::SUM);

    std::array<rtype, 2 * N_DIM + 2> lo_hi;  // -min and max per dimension, -min and max volume
    lo_hi.fill(std::numeric_limits<rtype>::lowest());
    for (uint32_t n = 0; n < mesh.n_nodes; n++) {
        for (size_t d = 0; d < N_DIM; d++) {
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
    MeshSetup m;
    for (uint32_t k = 0; k < MAX_NODES; k++) {
        if (counts[k] == 0) continue;
        const auto it = names.find(k);
        m.cells.emplace_back(it != names.end() ? std::string(it->second) : std::to_string(k) + "-node", counts[k]);
    }
    m.faces = counts[MAX_NODES];
    m.boundary_faces = counts[MAX_NODES + 1];
    for (size_t d = 0; d < N_DIM; d++) {
        m.lo[d] = static_cast<double>(-lo_hi[d]);
        m.hi[d] = static_cast<double>(lo_hi[N_DIM + d]);
    }
    m.size_min = static_cast<double>(v_min);
    m.size_max = static_cast<double>(v_max);
    return m;
}

} // namespace

void Solver::init_mesh() {
    const std::string type = find_value_or<std::string>(input, "mesh", "type", "file");
    // Read in every run, so that an input runs unchanged on any number of ranks
    partitioner = find_value_or<std::string>(input, "parallel", "partitioner", "multijagged");
    record_options("parallel.partitioner", {"graph", "hilbert", "multijagged"});
    if (partitioner != "graph" && partitioner != "hilbert" && partitioner != "multijagged") {
        throw not_accepted(input.at("parallel"), "parallel.partitioner", partitioner, {"graph", "hilbert", "multijagged"});
    }
    const std::string exchange =
        find_value_or<std::string>(input, "parallel", "halo_exchange", comm::nccl_available() ? "nccl" : "mpi");
    record_options("parallel.halo_exchange", {"nccl", "mpi"});
    if (exchange != "nccl" && exchange != "mpi") {
        throw not_accepted(input.at("parallel"), "parallel.halo_exchange", exchange, {"mpi", "nccl"});
    }
    curved::Surfaces surfaces;
    if (!is_distributed()) {
        mesh = std::make_shared<Mesh>();
        mesh->init(input);
        surfaces = curved::surfaces_of_mesh(*mesh, input);
    } else {
        if (halo_layers == 0) {
            // A TENO cache records the halo depth its stencils need, which spares
            // the setup pass that would otherwise find it out
            const Input reconstruction =
                toml::find_or(input, "numerics", "face_reconstruction", Input(Input::table_type{}));
            const bool teno = find_value_or<std::string>(reconstruction, "type", "FO") == "TENO";
            const int cached =
                comm::allreduce(teno ? int(TENO::cached_halo_layers(reconstruction)) : 0, comm::Op::MAX);
            halo_layers = std::max(base_halo_layers(), cached);
        }
        // The block is freed before the Mesh is built, so a deeper halo (see
        // halo_too_shallow) reads it again; it keeps the partition
        DistributedMesh setup(read_mesh_block(input), Mesh::periodic_pairs(input));
        if (!block_owners) {
            block_owners = partitioner == "graph"         ? partition_graph(setup, comm::size())
                         : partitioner == "multijagged" ? partition_multijagged(setup, comm::size())
                                                        : partition_hilbert(setup, comm::size());
        }
        setup.distribute(*block_owners);
        surfaces = curved::surfaces_of_block(setup.mesh_block(), input);
        mesh = std::move(setup).build_local_mesh(halo_layers, distribution, cell_numbering_of_input(input));
        if (exchange == "nccl" && !comm::nccl_available()) {
            throw InputError({.key = "parallel.halo_exchange",
                              .line = line_of(input.at("parallel"), "halo_exchange"),
                              .message = "\"nccl\" needs a build with NCCL, and this one has none",
                              .fix = "Set halo_exchange = \"mpi\", or rebuild with -DMallard_ENABLE_NCCL=ON",
                              .docs = "docs/guide/parallel.md#mpi"});
        }
        halo = HaloExchange(distribution, exchange == "nccl");
    }
    mesh->apply_curved(surfaces);
    mesh_setup = describe_mesh(*mesh);
    Settings & mesh_settings = mesh_setup.settings;
    mesh_settings.emplace_back("Type", type == "file" ? "file " + find_value_or<std::string>(input, "mesh", "filename",
                                                                                               "mesh.msh")
                                                      : type);
    const MeshTransform transform = mesh_transform(input);
    if (!transform.is_identity()) mesh_settings.emplace_back("Transform", transform.description);
    const Stretching stretching = mesh_stretching(input);
    if (stretching != Stretching{}) {
        if (type == "file") {
            throw InputError({.key = "mesh.stretching",
                              .line = line_of(input.at("mesh"), "stretching"),
                              .message = "applies only to generated meshes, not to a mesh file",
                              .fix = "Remove stretching and grade the mesh in the program that made the file",
                              .docs = "docs/guide/meshes.md#stretching"});
        }
        std::string text;
        FOR_I_DIM text += (i ? ", " : "[") + logging::real(double(stretching[i]));
        mesh_settings.emplace_back("Stretching", text + "] (tanh, toward both ends)");
    }
    n_cells_global = comm::allreduce(uint64_t(mesh->n_owned()), comm::Op::SUM);
    if (is_distributed()) {
        const uint64_t n_owned = distribution.n_owned;
        const uint64_t n_halo = mesh->n_cells - distribution.n_owned;
        const uint64_t min_owned = comm::allreduce(n_owned, comm::Op::MIN);
        const uint64_t max_owned = comm::allreduce(n_owned, comm::Op::MAX);
        const uint64_t max_halo = comm::allreduce(n_halo, comm::Op::MAX);
        mesh_setup.partition = MeshSetup::Partition{.partitioner = partitioner == "graph"         ? "graph (KaMinPar)"
                                                                   : partitioner == "multijagged" ? "multi-jagged"
                                                                                                  : "Hilbert curve",
                                                    .exchange = halo.uses_nccl() ? "NCCL" : "MPI",
                                                    .ranks = comm::size(),
                                                    .halo_layers = halo_layers,
                                                    .owned_min = min_owned,
                                                    .owned_max = max_owned,
                                                    .halo_max = max_halo};
    }
    {
        const CellOrderChoice & choice = mesh->cell_order_choice;
        const bool hilbert = choice.order == CellOrder::HILBERT;
        std::string text = hilbert ? "Hilbert curve" : "input order";
        if (is_distributed()) {
            const int n_hilbert = comm::allreduce(int(hilbert), comm::Op::SUM);
            if (n_hilbert > 0 && n_hilbert < comm::size()) {
                text = logging::format("Hilbert curve on %d of %d ranks, input order on the others", n_hilbert, comm::size());
            } else {
                text = n_hilbert > 0 ? "Hilbert curve" : "input order";
            }
        } else if (choice.input_lines > 0.0) {
            text += logging::format(" (auto: %.3g cache lines per cell, %.3g %s)",
                                    hilbert ? choice.hilbert_lines : choice.input_lines,
                                    hilbert ? choice.input_lines : choice.hilbert_lines,
                                    hilbert ? "in the input order" : "along the Hilbert curve");
        }
        std::array<uint64_t, 2> blocks = {mesh->structured_blocks.size(), 0};
        for (const StructuredBlock & block : mesh->structured_blocks) {
            uint64_t cells = 1;
            for (uint32_t e : block.n) cells *= e;
            blocks[1] += cells;
        }
        blocks = comm::allreduce(blocks, comm::Op::SUM);
        if (blocks[0] > 0) {
            text += logging::format(", structured blocks first (%llu %s, %.1f%% of the cells)",
                                    static_cast<unsigned long long>(blocks[0]), blocks[0] == 1 ? "block" : "blocks",
                                    100.0 * double(blocks[1]) / double(n_cells_global));
        }
        mesh_settings.emplace_back("Cell order", text);
    }
    if (mesh->curved_geometry) {
        // Curved boundary faces of owned cells, and curved faces in all
        uint64_t counts[2] = {0, 0};
        for (uint32_t f = 0; f < mesh->n_faces; f++) {
            if (!mesh->curved_geometry->face_is_curved(f) || uint32_t(mesh->h_cells_of_face(f, 0)) >= mesh->n_owned()) continue;
            counts[0] += mesh->h_cells_of_face(f, 1) < 0;
            counts[1]++;
        }
        const auto total = comm::allreduce(std::array<uint64_t, 2>{counts[0], counts[1]}, comm::Op::SUM);
        std::string text = logging::count(total[0], "boundary face");
        if (N_DIM == 3) text += " (" + logging::count(total[1], "curved face") + ")";
        uint64_t n_high_order = 0;
        for (const auto & face : surfaces.faces) n_high_order += face.shape < 0;
        if (n_high_order > 0) text += ", high-order faces from the mesh file";
        for (const auto & shape : surfaces.shapes) text += ", " + shape.zone + " projected";
        mesh_settings.emplace_back("Curved boundary", text);
    }
    axisymmetric = find_value_or<bool>(input, "physics", "axisymmetric", false);
    if (axisymmetric) {
        if (mesh->is_rotational()) {
            throw InputError({.key = "physics.axisymmetric",
                              .line = line_of(input.at("physics"), "axisymmetric"),
                              .message = "rotational periodicity is not supported in axisymmetric runs",
                              .fix = "Remove the angle of the [[periodic]] pair, or run in 3D",
                              .docs = "docs/input.md#axisymmetric-flows"});
        }
        for (const auto & translation : mesh->periodic_translations) {
            if (translation[1] != 0.0_r) {
                throw InputError({.key = "physics.axisymmetric",
                                  .line = line_of(input.at("physics"), "axisymmetric"),
                                  .message = "the mesh is periodic in y, which is the radius of an axisymmetric run",
                                  .fix = "Remove the periodicity in y",
                                  .docs = "docs/input.md#axisymmetric-flows"});
            }
        }
        mesh->make_axisymmetric();
        mesh_setup.settings.emplace_back("Geometry", "axisymmetric about the x axis (y = r), per radian");
    }
    if (mesh->curved_geometry && is_distributed()) exchange_halo_geometry();
    mesh->copy_host_to_device();
}

void Solver::exchange_halo_geometry() {
    // Centroid and volume in the flow block, the axisymmetric measure and covariance as species columns
    const uint32_t n_revolved = mesh->axisymmetric ? 4 : 0;
    const State geometry("halo_geometry", mesh->n_cells, n_revolved);
    auto h_flow = Kokkos::create_mirror_view(geometry.flow);
    auto h_revolved = Kokkos::create_mirror_view(geometry.species);
    for (uint32_t c = 0; c < mesh->n_owned(); c++) {
        FOR_I_DIM h_flow(c, i) = mesh->h_cell_coords(c, i);
        h_flow(c, N_DIM) = mesh->h_cell_volume(c);
        if (n_revolved == 0) continue;
        h_revolved(c, 0) = mesh->h_cell_measure(c);
        for (uint32_t k = 0; k < 3; k++) h_revolved(c, k + 1) = mesh->h_cell_covariance(c, k);
    }
    Kokkos::deep_copy(geometry.flow, h_flow);
    Kokkos::deep_copy(geometry.species, h_revolved);
    halo.exchange(geometry);
    Kokkos::deep_copy(h_flow, geometry.flow);
    Kokkos::deep_copy(h_revolved, geometry.species);
    for (uint32_t c = mesh->n_owned(); c < mesh->n_cells; c++) {
        FOR_I_DIM mesh->h_cell_coords(c, i) = h_flow(c, i);
        mesh->h_cell_volume(c) = h_flow(c, N_DIM);
        if (n_revolved == 0) continue;
        mesh->h_cell_measure(c) = h_revolved(c, 0);
        for (uint32_t k = 0; k < 3; k++) mesh->h_cell_covariance(c, k) = h_revolved(c, k + 1);
    }
}

int Solver::base_halo_layers() const {
    const std::string type = find_value_or<std::string>(
        toml::find_or(input, "numerics", "face_reconstruction", Input(Input::table_type{})), "type", "FO");
    // Faces between owned and halo-layer-1 cells need the layer-1 reconstruction,
    // which reads one more layer (MUSCL gradients and limiters, viscous gradients)
    const bool viscous = find_value_or<std::string>(input, "physics", "type", "euler") == "navier_stokes";
    // The hybrid flux's sensor reads the velocity gradients of halo-layer-1 cells too
    const bool hybrid = find_value_or<std::string>(input, "numerics", "convective_flux", "riemann") == "hybrid";
    // Fields of the velocity gradient in conditions read it in the cells' vertex neighbors
    return (type == "FO" && !viscous && !hybrid && !region::conditions_read_gradients(input)) ? 1 : 2;
}

void Solver::trim_halo() {
    // The deep TENO halo holds every cell the stencil searches visited, so that
    // stencils match the serial ones; the time loop needs only the cells the
    // stencils kept, and the layers the other kernels read
    const auto * teno = dynamic_cast<const TENO *>(face_reconstruction.get());
    if (!is_distributed() || teno == nullptr) return;
    // Owned cells' stencils, or with exchanges_partition_faces() and halo layer 1 not reconstructed, owned cells'
    // and those of the halo image cells of transmissive faces without an image face (image_point_values)
    const bool own_faces = exchanges_partition_faces();
    std::vector<uint8_t> reconstructed(mesh->n_cells, 0);
    std::fill_n(reconstructed.begin(), own_faces ? mesh->n_owned() : mesh->n_reconstructed(), uint8_t(1));
    if (own_faces) {
        for (const uint32_t c : face_reconstruction->halo_image_cells()) reconstructed[c] = 1;
    }
    std::vector<uint8_t> needed = teno->stencil_cells(reconstructed);
    const int base = base_halo_layers();
    for (uint32_t c = 0; c < mesh->n_cells; c++) needed[c] = needed[c] || distribution.layer[c] <= base;
    // Image neighbors of the cells whose gradients are read
    if (boundary_data.image_neighbors.extent(0) > 0) {
        auto offsets = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundary_data.image_neighbor_offsets);
        auto neighbors = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundary_data.image_neighbors);
        for (uint32_t c = 0; c < mesh->n_cells; c++) {
            if (distribution.layer[c] > 1) continue;
            for (uint32_t k = offsets(c); k < offsets(c + 1); k++) needed[neighbors(k)] = 1;
        }
    }
    // Wall-model points: their cells and the vertex neighbors their gradients read
    for (const uint32_t c : wall_model_points.cells) {
        needed[c] = 1;
        const VertexNeighbors & neighbors = mesh->vertex_neighbors();
        for (uint32_t k = neighbors.h_offsets(c); k < neighbors.h_offsets(c + 1); k++) needed[neighbors.h_cells(k)] = 1;
    }
    trim_halo_exchange(distribution, needed);
    halo = HaloExchange(distribution, halo.uses_nccl());
}

void Solver::describe_partition_surface() {
    if (!is_distributed()) return;
    const uint32_t n_owned = distribution.n_owned;
    uint64_t cut = 0;
    for (uint32_t f = 0; f < mesh->n_faces; f++) {
        const int32_t a = mesh->h_cells_of_face(f, 0), b = mesh->h_cells_of_face(f, 1);
        cut += b >= 0 && (static_cast<uint32_t>(a) < n_owned) != (static_cast<uint32_t>(b) < n_owned);
    }
    uint64_t layer1 = 0;
    for (uint32_t c = n_owned; c < mesh->n_cells; c++) layer1 += distribution.layer[c] == 1;
    uint64_t received = 0;
    for (const auto & cells : distribution.recv_cells) received += cells.size();
    const std::array<uint64_t, 3> local{cut, layer1, received};
    const auto total = comm::allreduce(local, comm::Op::SUM);
    const auto most = comm::allreduce(local, comm::Op::MAX);
    MeshSetup::Partition & p = *mesh_setup.partition;
    p.cut_faces = total[0] / 2;
    p.layer1_max = most[1];
    p.layer1_all = total[1];
    p.received_max = most[2];
    p.received_all = total[2];
}

bool Solver::halo_too_shallow() {
    if (!is_distributed()) return false;
    auto * teno = dynamic_cast<TENO *>(face_reconstruction.get());
    // Every rank takes part in the reduction below, also those without image neighbors
    const bool images = comm::allreduce(int(boundary_data.image_neighbors.extent(0) > 0), comm::Op::MAX) > 0;
    if (teno == nullptr && !wall_model_on && !images) return false;
    // The stencil searches of the cells with TENO tables must see every layer
    // they visited, so stencils match the serial ones exactly
    int needed = 0;
    for (uint32_t r = 0; teno && r < teno->n_rows(); r++) {
        needed = std::max(needed, distribution.layer[teno->cell_of_row(r)] + teno->gather_depth[r] + 1);
    }
    // The same for the gradient stencils that continue across extrapolation boundaries
    for (uint32_t c = 0; !image_reach.empty() && c < mesh->n_cells; c++) {
        if (distribution.layer[c] <= 1) needed = std::max(needed, distribution.layer[c] + image_reach[c]);
    }
    // The wall model's exchange and edge points must lie in cells with complete gradient stencils
    needed = std::max(needed, wall_model_points.needed_layers);
    needed = comm::allreduce(needed, comm::Op::MAX);
    if (needed <= halo_layers) return false;
    halo_layers = needed;
    return true;
}

bool Solver::exchanges_partition_faces() const {
    return is_distributed() && !is_mixture() && face_reconstruction->exchanges_partition_faces();
}

void Solver::init_rhs_split() {
    const uint32_t n_owned = mesh->n_owned();
    auto owned = [&](int32_t c) { return c >= 0 && c < static_cast<int32_t>(n_owned); };
    std::vector<uint32_t> faces;
    for (uint32_t f = 0; f < mesh->n_faces; f++) {
        if (owned(mesh->h_cells_of_face(f, 0)) || owned(mesh->h_cells_of_face(f, 1))) faces.push_back(f);
    }
    rhs_faces = IndexSet::of(faces, "rhs_faces");

    const bool own_faces = exchanges_partition_faces();
    if (own_faces) {
        // Fluxes read the halo side of partition faces, and transmissive
        // boundary faces read a side of their image face, which may be a halo cell's
        const BoundaryData::HostImages images = boundary_data.host_images();
        std::vector<uint32_t> local, remote;
        std::vector<RemoteFaceSide> needed;
        std::vector<bool> requested(2 * size_t(mesh->n_faces), false);
        auto need = [&](uint32_t f, int32_t c) {
            const size_t k = 2 * size_t(f) + (mesh->h_cells_of_face(f, 0) == c ? 0 : 1);
            if (requested[k]) return;
            requested[k] = true;
            needed.push_back({f, static_cast<uint32_t>(c)});
        };
        for (const uint32_t f : faces) {
            const int32_t c0 = mesh->h_cells_of_face(f, 0), c1 = mesh->h_cells_of_face(f, 1);
            bool waits = c1 >= 0 && owned(c0) != owned(c1);
            if (waits) need(f, owned(c0) ? c1 : c0);
            if (c1 < 0 && images.face[f] >= 0) {
                const uint32_t g = static_cast<uint32_t>(images.face[f]);
                const int32_t c = mesh->h_cells_of_face(g, images.side[f]);
                if (!owned(c)) {
                    need(g, c);
                    waits = true;
                }
            }
            (waits ? remote : local).push_back(f);
        }
        local_faces = IndexSet::of(local, "local_faces");
        partition_faces = IndexSet::of(remote, "partition_faces");
        const FacePlan plan = plan_face_exchange(distribution, *mesh, needed);
        face_exchange = FaceExchange(plan, static_cast<uint8_t>(face_solution.extent(1)), halo.uses_nccl());
        if (polynomial_viscous) init_polynomial_exchange(plan, needed);
    } else if (polynomial_viscous) {
        allocate_polynomials(mesh->n_reconstructed());
    }

    // Early: the cells independent of the halo among [0, n_interior()), where the numbering puts them
    std::vector<uint32_t> early, late;
    if (halo.active()) early = face_reconstruction->cells_independent_of_halo(n_owned);
    early.erase(std::lower_bound(early.begin(), early.end(), mesh->n_interior()), early.end());
    if (early.empty() && !(own_faces && halo.active())) return;
    std::vector<bool> is_early(mesh->n_cells, false);
    for (uint32_t c : early) is_early[c] = true;
    for (uint32_t c = 0; c < (own_faces ? n_owned : mesh->n_reconstructed()); c++) {
        if (!is_early[c]) late.push_back(c);
    }
    early_cells = IndexSet::of(early, "rhs_cells");
    late_cells = IndexSet::of(late, "rhs_cells");
    // Host backends run kernels to completion anyway; on devices the early cells get their own stream
    if constexpr (!Kokkos::SpaceAccessibility<Kokkos::HostSpace,
                                              Kokkos::DefaultExecutionSpace::memory_space>::accessible) {
        // Without early cells nothing forks overlap_space into a step graph's capture, and joining a stream
        // outside the capture invalidates it
        overlap_space = !early.empty() ? Kokkos::Experimental::partition_space(Kokkos::DefaultExecutionSpace(), 1)[0]
                                       : Kokkos::DefaultExecutionSpace();
        if (own_faces) comm_space = comm::high_priority_space();
    }
}

void Solver::allocate_polynomials(const uint32_t n_cells) {
    auto & teno = dynamic_cast<TENO &>(*face_reconstruction);
    teno.polynomials = Kokkos::View<rtype **[2][N_CONSERVATIVE]>("viscous_polynomials", (n_cells + 1) / 2,
                                                                 teno.n_dof_large);
}

void Solver::init_polynomial_exchange(const FacePlan & faces, const std::vector<RemoteFaceSide> & needed) {
    // Rows of cell c are (c / 2, side c % 2) of the polynomials: a FaceExchange of NK points per side sends them
    uint32_t n = mesh->n_reconstructed();
    for (const RemoteFaceSide & r : needed) n = std::max(n, r.cell + 1);
    allocate_polynomials(n);
    auto rows = [&](const std::vector<std::vector<uint32_t>> & face_lists,
                    const std::vector<std::vector<uint8_t>> & side_lists, std::vector<std::vector<uint32_t>> & row_lists,
                    std::vector<std::vector<uint8_t>> & lane_lists, std::vector<std::vector<uint64_t>> & global) {
        row_lists.assign(face_lists.size(), {});
        lane_lists.assign(face_lists.size(), {});
        global.assign(face_lists.size(), {});
        // Both ends list a neighbor's cells in the same order, so each keeps the first of repeated cells
        for (size_t k = 0; k < face_lists.size(); k++) {
            std::unordered_set<uint32_t> seen;
            for (size_t j = 0; j < face_lists[k].size(); j++) {
                const uint32_t c = static_cast<uint32_t>(mesh->h_cells_of_face(face_lists[k][j], side_lists[k][j]));
                if (!seen.insert(c).second) continue;
                row_lists[k].push_back(c >> 1);
                lane_lists[k].push_back(static_cast<uint8_t>(c & 1u));
                global[k].push_back(mesh->h_global_cell(c));
            }
        }
    };
    FacePlan plan;
    plan.neighbors = faces.neighbors;
    std::vector<std::vector<uint64_t>> send_cells, recv_cells;
    rows(faces.send_faces, faces.send_sides, plan.send_faces, plan.send_sides, send_cells);
    rows(faces.recv_faces, faces.recv_sides, plan.recv_faces, plan.recv_sides, recv_cells);
    // That order is the plan's: check once that each neighbor sends the cells, in the order, this rank expects
    std::vector<std::vector<uint64_t>> to_rank(static_cast<size_t>(comm::size()));
    for (size_t k = 0; k < plan.neighbors.size(); k++) to_rank[static_cast<size_t>(plan.neighbors[k])] = send_cells[k];
    const auto from_rank = comm::alltoallv(to_rank);
    for (size_t k = 0; k < plan.neighbors.size(); k++) {
        if (from_rank[static_cast<size_t>(plan.neighbors[k])] != recv_cells[k]) {
            throw std::logic_error("init_polynomial_exchange: rank " + std::to_string(plan.neighbors[k]) +
                                   " sends other cells than rank " + std::to_string(comm::rank()) + " expects");
        }
    }
    const auto & teno = dynamic_cast<const TENO &>(*face_reconstruction);
    polynomial_exchange = FaceExchange(plan, teno.n_dof_large, halo.uses_nccl(), 2);
}

void Solver::init_axisymmetric_weights() {
    // int_f F r dl = L / 2 sum_q w_q r_q F_q, exact in r for any rule (r is linear along a face)
    const auto & points = face_reconstruction->quadrature_face.h_points;
    const auto & weights = face_reconstruction->quadrature_face.h_weights;
    const uint32_t n_q = static_cast<uint32_t>(points.extent(0));
    Kokkos::View<rtype **, Kokkos::HostSpace> h_weights("flux_weights_host", mesh->n_state_faces(), n_q);
    for (uint32_t f = 0; f < mesh->n_state_faces(); f++) {
        const uint32_t a = mesh->h_node_of_face(f, 0), b = mesh->h_node_of_face(f, 1);
        const rtype dy = mesh->h_node_coords(b, 1) - mesh->h_node_coords(a, 1);
        for (uint32_t q = 0; q < n_q; q++) {
            const rtype r_q = mesh->h_face_coords(f, 1) + 0.5_r * points(q, 0) * dy;
            h_weights(f, q) = weights(q) * std::max(r_q, 0.0_r);
        }
    }
    if (mesh->curved_geometry) revolve_curved_flux_points(h_weights);
    flux_weights = share_face_weights("flux_weights", mesh->n_state_faces(), n_q, [&](const uint32_t f, rtype * w_f) {
        for (uint32_t q = 0; q < n_q; q++) w_f[q] = h_weights(f, q);
    });
    geometric_source = Kokkos::View<rtype *>("geometric_source", mesh->n_cells);
    if (!is_mixture() && physics.is_viscous()) cell_mu = Kokkos::View<rtype *>("cell_mu", mesh->n_cells);
}

void Solver::revolve_curved_flux_points(const Kokkos::View<rtype **, Kokkos::HostSpace> & h_weights) {
    // The points' area vectors r_q w_q n_q, shifted by a common vector so that they sum to the
    // face's int r n dS: uniform states then stay uniform, as the geometric source integrates p
    // over the curved cell
    const curved::Geometry & geometry = *mesh->curved_geometry;
    const auto & points = face_reconstruction->quadrature_face.h_points;
    const auto & weights = face_reconstruction->quadrature_face.h_weights;
    const uint32_t n_q = static_cast<uint32_t>(points.extent(0));
    std::vector<std::array<double, 2>> ref(n_q);
    std::vector<double> ref_w(n_q);
    double w_sum = 0.0;
    for (uint32_t q = 0; q < n_q; q++) {
        ref[q] = {double(points(q, 0)), 0.0};
        ref_w[q] = double(weights(q));
        w_sum += ref_w[q];
    }
    flux_normals = Kokkos::View<rtype ***>("flux_normals", mesh->n_state_faces(), n_q, N_DIM);
    Kokkos::deep_copy(flux_normals, face_reconstruction->face_quad_normals);
    auto h_normals = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), flux_normals);
    std::vector<curved::Vec3> x, n, V(n_q);
    std::vector<double> w;
    for (uint32_t f = 0; f < mesh->n_state_faces(); f++) {
        if (!geometry.face_is_curved(f)) continue;
        geometry.face_rule(f, ref, ref_w, x, n, w);
        const curved::Vec3 R = geometry.revolved_area_vector(f);
        curved::Vec3 shift = R;
        for (uint32_t q = 0; q < n_q; q++) {
            for (size_t i = 0; i < 2; i++) {
                V[q][i] = std::max(x[q][1], 0.0) * w[q] * n[q][i];
                shift[i] -= V[q][i];
            }
        }
        for (uint32_t q = 0; q < n_q; q++) {
            for (size_t i = 0; i < 2; i++) V[q][i] += ref_w[q] / w_sum * shift[i];
            const double length = std::sqrt(V[q][0] * V[q][0] + V[q][1] * V[q][1]);
            FOR_I_DIM h_normals(f, q, i) = rtype(V[q][i] / length);
            h_weights(f, q) = rtype(2.0 * length / double(mesh->h_face_area(f)));
        }
    }
    Kokkos::deep_copy(flux_normals, h_normals);
}
