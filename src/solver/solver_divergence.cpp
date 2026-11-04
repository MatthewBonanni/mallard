/**
 * @file solver_divergence.cpp
 * @brief The divergence report's evidence, gathered from the solver once a
 *        run has diverged: nothing here runs while the run is healthy.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "solver.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>

#include <Kokkos_Core.hpp>

#include "comm.h"
#include "common_io.h"
#include "distribution.h"
#include "log.h"
#include "mesh_block.h"
#include "mesh_quality.h"
#include "run_log.h"

namespace {

using divergence::Cell;
using divergence::Evidence;

constexpr size_t N_SHOWN = 3;          // invalid cells shown besides the focus
constexpr int SNAPSHOT_LAYERS = 3;     // cells around the invalid ones in the snapshot
constexpr int AMBIGUOUS = -2;          // every conservative variable is non-finite

/** @brief cfl_local holds each cell's stable time step at CFL 1: make it the local CFL number of dt. */
void to_local_cfl(const Kokkos::View<rtype *> & c, const rtype dt, const uint32_t n) {
    Kokkos::parallel_for("divergence_local_cfl", n, KOKKOS_LAMBDA(const uint32_t i) { c(i) = dt / c(i); });
}

/** @brief The exterior state [rho, u, p] of the flux at each boundary face, from the cell's state. */
Kokkos::View<rtype **>::host_mirror_type ghost_states(const BoundaryData & boundaries, const std::vector<uint32_t> & list,
                                                     const Mesh & mesh, const StateView & U,
                                                     const Kokkos::View<rtype *[N_PRIMITIVE]> & P) {
    Kokkos::View<uint32_t *> faces("divergence_faces", list.size());
    Kokkos::deep_copy(faces, Kokkos::View<const uint32_t *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(list.data(), list.size()));
    Kokkos::View<rtype **> out("divergence_ghosts", list.size(), N_DIM + 2);
    const Kokkos::View<int32_t *[2]> cells = mesh.cells_of_face;
    const Kokkos::View<rtype *[N_DIM]> normals = mesh.face_normals;
    const BoundaryData bd = boundaries;
    Kokkos::parallel_for("divergence_ghosts", list.size(), KOKKOS_LAMBDA(const uint32_t k) {
        const uint32_t f = faces(k);
        const int32_t c = cells(f, 0);
        rtype W[N_DIM + 2], W_g[N_DIM + 2], n[N_DIM], n_vec[N_DIM];
        W[0] = U(c, 0);
        FOR_I_DIM W[1 + i] = P(c, i);
        W[N_DIM + 1] = P(c, N_DIM);
        FOR_I_DIM n_vec[i] = normals(f, i);
        unit<N_DIM>(n_vec, n);
        bd.ghost_W(f, W, n, W_g, true);
        for (uint32_t i = 0; i < N_DIM + 2; i++) out(k, i) = W_g[i];
    });
    return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
}

bool prescribes_state(const BoundaryType type, const bool mixture) {
    switch (type) {
        case BoundaryType::INLET_STATE:
        case BoundaryType::FARFIELD:
        case BoundaryType::DIRICHLET:
        case BoundaryType::OUTLET_PRESSURE:
            return true;
        case BoundaryType::INLET_STAGNATION:
        case BoundaryType::INLET_MASS_FLOW:
            return !mixture;  // a mixture's inflow is solved apart from ghost_W
        default:
            return false;
    }
}

bool finite(double v) { return std::isfinite(v); }

} // namespace

divergence::Evidence Solver::diagnose_divergence(const bool has_valid, std::vector<int> & failed, MeshQuality & quality) {
    Evidence e;
    const uint32_t n_owned = mesh->n_owned(), n_local = mesh->n_cells;
    const uint32_t n_species = static_cast<uint32_t>(species_names.size());
    const bool mix = is_mixture();
    e.step = step;
    e.t = double(t);
    e.step_start = step_run_start;
    e.from_restart = find_value_or<std::string>(input, "initialize", "type", "") == "restart";
    e.has_valid = has_valid;
    e.step_valid = last_good_step;
    e.t_valid = double(last_good_t);
    e.mixture = mix;
    e.n_cells = n_cells_global;
    e.n_ranks = comm::size();
    e.integrator = TIME_INTEGRATOR_NAMES.at(time_integrator->get_type());
    e.limit = divergence::cfl_limit(time_integrator->get_type());
    e.use_cfl = use_cfl;
    e.error_control = step_controller.enabled();
    e.cfl_input = double(cfl);
    e.dt = double(dt);
    e.history.assign(progress_history.begin(), progress_history.end());
    const Euler phys = physics;

    // What failed in each cell of the failing state, halo cells from their owners
    halo.exchange(state());
    update_primitives();
    failed.assign(n_local, -1);
    {
        const auto U = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), conservatives);
        const auto Y = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), species);
        const auto P = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), primitives);
        for (uint32_t c = 0; c < n_local; c++) {
            bool all = true, momentum = true;
            for (uint32_t v = 0; v < N_CONSERVATIVE; v++) all = all && !finite(double(U(c, v)));
            for (uint32_t v = 1; v <= N_DIM; v++) momentum = momentum && finite(double(U(c, v)));
            const double rho = double(U(c, 0)), p = double(P(c, N_DIM)), T = double(P(c, N_DIM + 1));
            bool species_ok = true;
            for (uint32_t k = 0; k < n_species; k++) species_ok = species_ok && finite(double(Y(c, k)));
            if (all) {
                failed[c] = AMBIGUOUS;
            } else if (!(rho > 0.0)) {
                failed[c] = divergence::DENSITY;
            } else if (!momentum) {
                failed[c] = divergence::VELOCITY;
            } else if (!finite(double(U(c, N_DIM + 1))) || !(p > 0.0) || !finite(p)) {
                failed[c] = divergence::PRESSURE;
            } else if (mix && (!(T > 0.0) || !finite(T))) {
                failed[c] = divergence::TEMPERATURE;
            } else if (!species_ok) {
                failed[c] = divergence::MASS_FRACTION;
            }
        }
    }

    // The last valid state, and the local CFL number of the failing step there
    if (has_valid) restore_last_good();
    update_primitives();
    e.dt_cfl1 = double(calc_dt_cfl1());
    const bool cfl_known = e.dt > 0.0 && e.dt_cfl1 > 0.0;
    if (cfl_known) to_local_cfl(cfl_local, dt, n_owned);
    const auto U = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), conservatives);
    const auto P = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), primitives);
    const auto h_cfl = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cfl_local);
    // The last valid state is the failing one if none was kept: only the run's first state then says something
    const bool state_known = has_valid || step == step_run_start;

    // Means, and cells of the last valid state that are not admissible
    quality = mesh_quality(*mesh);
    std::array<double, 4> sums{};  // V rho, V p, V, inadmissible cells
    std::vector<double> inadmissible;
    for (uint32_t c = 0; c < n_owned; c++) {
        const double V = double(mesh->h_cell_measure(c)), rho = double(U(c, 0)), p = double(P(c, N_DIM));
        if (finite(rho) && finite(p)) {
            sums[0] += V * rho;
            sums[1] += V * p;
            sums[2] += V;
        }
        if (state_known && !(rho > 0.0 && p > 0.0)) {
            sums[3] += 1.0;
            const uint64_t id = mesh->h_global_cell(c);
            if (inadmissible.empty() || id < static_cast<uint64_t>(inadmissible[0])) {
                Cell cell;
                cell.id = id;
                cell.rank = comm::rank();
                for (int d = 0; d < N_DIM; d++) cell.x[size_t(d)] = double(mesh->h_cell_coords(c, d));
                cell.rho = rho;
                cell.p = p;
                inadmissible = cell.pack();
            }
        }
    }
    sums = comm::allreduce(sums, comm::Op::SUM);
    e.rho_mean = sums[2] > 0.0 ? sums[0] / sums[2] : 0.0;
    e.p_mean = sums[2] > 0.0 ? sums[1] / sums[2] : 0.0;
    e.n_inadmissible = static_cast<uint64_t>(sums[3]);
    {
        const std::vector<double> all = comm::allgatherv(inadmissible);
        for (size_t r = 0; r + Cell::WIDTH <= all.size(); r += Cell::WIDTH) {
            const Cell c = Cell::unpack(&all[r]);
            if (r == 0 || c.id < e.inadmissible.id) e.inadmissible = c;
        }
    }
    auto margin = [&](uint32_t c) {
        const double rho = double(U(c, 0)), p = double(P(c, N_DIM));
        if (!state_known || !finite(rho) || !finite(p)) return 0.0;
        return std::min(e.rho_mean > 0.0 ? rho / e.rho_mean : rho, e.p_mean > 0.0 ? p / e.p_mean : p);
    };
    for (uint32_t c = 0; c < n_local; c++) {
        if (failed[c] != AMBIGUOUS) continue;
        const double rho = double(U(c, 0)), p = double(P(c, N_DIM));
        failed[c] = state_known && e.rho_mean > 0.0 && e.p_mean > 0.0 && p / e.p_mean < rho / e.rho_mean
                        ? divergence::PRESSURE
                        : divergence::DENSITY;
    }

    // Boundary zones, by name over all ranks, with their condition
    std::vector<std::string> zone_names;
    {
        std::string local;
        for (const FaceZone & z : mesh->h_face_zones()) {
            if (z.get_type() == FaceZoneType::BOUNDARY && z.get_name() != PARTITION_ZONE) local += z.get_name() + '\n';
        }
        for (const std::string & part : comm::size() > 1 ? comm::allgather(local) : std::vector<std::string>{local}) {
            std::istringstream lines(part);
            for (std::string name; std::getline(lines, name);) zone_names.push_back(name);
        }
        std::sort(zone_names.begin(), zone_names.end());
        zone_names.erase(std::unique(zone_names.begin(), zone_names.end()), zone_names.end());
    }
    const size_t n_zones = zone_names.size();
    const auto face_bc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundary_data.face_bc);
    const auto bcs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundary_data.bcs);
    std::vector<int32_t> face_zone(mesh->n_faces, -1);
    std::vector<int32_t> zone_type(n_zones, -1);
    for (const FaceZone & z : mesh->h_face_zones()) {
        const auto it = std::lower_bound(zone_names.begin(), zone_names.end(), z.get_name());
        if (z.get_type() != FaceZoneType::BOUNDARY || it == zone_names.end() || *it != z.get_name()) continue;
        const auto index = static_cast<int32_t>(it - zone_names.begin());
        for (uint32_t i = 0; i < z.n_faces(); i++) {
            const uint32_t f = z.h_faces(i);
            face_zone[f] = index;
            if (face_bc(f) >= 0) zone_type[size_t(index)] = static_cast<int32_t>(bcs(face_bc(f)).type);
        }
    }
    if (n_zones > 0) comm::allreduce(std::span<int32_t>(zone_type), comm::Op::MAX);
    e.zones.resize(n_zones);
    for (size_t z = 0; z < n_zones; z++) {
        e.zones[z].name = zone_names[z];
        e.zones[z].type = zone_type[z] >= 0 ? BOUNDARY_NAMES.at(static_cast<BoundaryType>(zone_type[z])) : "none";
    }

    auto faces_of = [&](uint32_t c) {
        std::vector<uint32_t> f;
        for (uint32_t k = mesh->h_offsets_faces_of_cell(c); k < mesh->h_offsets_faces_of_cell(c + 1); k++) {
            f.push_back(mesh->h_faces_of_cell(k));
        }
        return f;
    };
    auto neighbors_of = [&](uint32_t c) {
        std::vector<uint32_t> n;
        for (const uint32_t f : faces_of(c)) {
            const int32_t c0 = mesh->h_cells_of_face(f, 0), c1 = mesh->h_cells_of_face(f, 1);
            const int32_t other = static_cast<uint32_t>(c0) == c ? c1 : c0;
            if (other >= 0 && static_cast<uint32_t>(other) != c) n.push_back(static_cast<uint32_t>(other));
        }
        std::sort(n.begin(), n.end());
        n.erase(std::unique(n.begin(), n.end()), n.end());
        return n;
    };
    // The nearest boundary zone within two face layers (the lowest index among the nearest); within: all of them
    auto nearest_zone = [&](uint32_t c, std::set<int> * within = nullptr) -> std::pair<int, int> {
        std::vector<uint32_t> frontier = {c};
        std::set<uint32_t> seen = {c};
        std::pair<int, int> nearest = {-1, -1};
        for (int layer = 0; layer <= 2; layer++) {
            int zone = -1;
            std::vector<uint32_t> next;
            for (const uint32_t cell : frontier) {
                for (const uint32_t f : faces_of(cell)) {
                    if (mesh->h_cells_of_face(f, 1) < 0 && face_zone[f] >= 0 && (zone < 0 || face_zone[f] < zone)) {
                        zone = face_zone[f];
                    }
                    if (within && mesh->h_cells_of_face(f, 1) < 0 && face_zone[f] >= 0) within->insert(face_zone[f]);
                }
                for (const uint32_t n : neighbors_of(cell)) {
                    if (seen.insert(n).second) next.push_back(n);
                }
            }
            if (zone >= 0 && nearest.first < 0) nearest = {zone, layer};
            if (nearest.first >= 0 && !within) break;
            frontier = std::move(next);
        }
        return nearest;
    };

    // Repairs, TENO and the limiter, chemistry: what each cell saw in the failing step
    std::vector<std::pair<std::string, Kokkos::View<uint8_t *>::host_mirror_type>> repaired;
    for (const auto & [site, name] : {std::make_pair(run_health.clip_site, "composition_clip"),
                                      std::make_pair(run_health.reset_site, "energy_reset")}) {
        if (site == HealthState::NONE) continue;
        repaired.emplace_back(name, Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                                        run_health.repairs.recorder(site).touched));
    }
    auto * teno = dynamic_cast<TENO *>(face_reconstruction.get());
    auto * muscl = dynamic_cast<MUSCL *>(face_reconstruction.get());
    e.teno = teno != nullptr;
    e.limiter = muscl && muscl->limiter != LimiterType::NONE;
    Kokkos::View<rtype *>::host_mirror_type sigma;
    if (teno && teno->troubled.extent(0) > 0) sigma = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), teno->troubled);
    Kokkos::View<rtype *[N_CONSERVATIVE]>::host_mirror_type limiters;
    if (e.limiter && muscl->limiters.extent(0) > 0) {
        limiters = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), muscl->limiters);
    }
    e.reacting = reacting;
    Kokkos::View<rtype *>::host_mirror_type cost;
    if (reacting && chem_cost.is_allocated()) cost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), chem_cost);

    auto make_cell = [&](uint32_t c) {
        Cell cell;
        cell.id = mesh->h_global_cell(c);
        cell.rank = comm::rank();
        for (int d = 0; d < N_DIM; d++) cell.x[size_t(d)] = double(mesh->h_cell_coords(c, d));
        cell.failed = failed[c];
        cell.rho = double(U(c, 0));
        double u2 = 0.0;
        for (int d = 0; d < N_DIM; d++) u2 += double(P(c, d)) * double(P(c, d));
        cell.speed = std::sqrt(u2);
        cell.p = double(P(c, N_DIM));
        cell.T = double(P(c, N_DIM + 1));
        if (!mix && cell.rho > 0.0 && cell.p > 0.0) cell.a = std::sqrt(double(phys.gamma) * cell.p / cell.rho);
        if (c < n_owned) {
            if (cfl_known) cell.cfl = double(h_cfl(c));
            for (size_t m = 0; m < MeshQuality::N_METRICS; m++) cell.quality[m] = quality.cells[m][c];
        }
        std::tie(cell.zone, cell.layers) = nearest_zone(c);
        for (const auto & [name, touched] : repaired) cell.repaired = cell.repaired || (touched(c) & 1u);
        if (sigma.extent(0) > c) cell.troubled = sigma(c) >= teno->sigma_threshold;
        if (limiters.extent(0) > c) {
            for (uint32_t v = 0; v < N_CONSERVATIVE; v++) cell.limited = cell.limited || limiters(c, v) < 1.0_r;
        }
        if (cost.extent(0) > c) cell.chemistry_cost = double(cost(c));
        for (const uint32_t n : neighbors_of(c)) cell.neighbors.push_back(mesh->h_global_cell(n));
        return cell;
    };

    // The invalid owned cells: counts, extent, nearby zones, and the candidates for the report
    std::vector<double> counts(divergence::N_QUANTITIES + 6 + repaired.size() + n_zones, 0.0);
    enum { N_INVALID = divergence::N_QUANTITIES, N_OVER, N_REPAIRED, N_TROUBLED, N_LIMITED, COST };
    std::array<double, 6> box;
    box.fill(-std::numeric_limits<double>::infinity());
    std::vector<std::pair<double, uint32_t>> candidates;
    std::set<uint32_t> near_faces;
    for (uint32_t c = 0; c < n_owned; c++) {
        if (cfl_known && double(h_cfl(c)) > e.limit) counts[N_OVER] += 1.0;
        if (cost.extent(0) > c) counts[COST] += double(cost(c));
        if (failed[c] < 0) continue;
        counts[size_t(failed[c])] += 1.0;
        counts[N_INVALID] += 1.0;
        for (int d = 0; d < N_DIM; d++) {
            box[size_t(d)] = std::max(box[size_t(d)], -double(mesh->h_cell_coords(c, d)));
            box[3 + size_t(d)] = std::max(box[3 + size_t(d)], double(mesh->h_cell_coords(c, d)));
        }
        const Cell cell = make_cell(c);
        counts[N_REPAIRED] += cell.repaired;
        counts[N_TROUBLED] += cell.troubled;
        counts[N_LIMITED] += cell.limited;
        for (size_t s = 0; s < repaired.size(); s++) counts[COST + 1 + s] += (repaired[s].second(c) & 1u);
        std::set<int> within;
        nearest_zone(c, &within);
        for (const int z : within) counts[COST + 1 + repaired.size() + size_t(z)] += 1.0;
        candidates.emplace_back(margin(c), c);
        // Boundary faces of the cell and its neighbors, to compare their exterior states with the flow
        std::vector<uint32_t> around = neighbors_of(c);
        around.push_back(c);
        for (const uint32_t a : around) {
            for (const uint32_t f : faces_of(a)) {
                if (mesh->h_cells_of_face(f, 1) < 0 && face_bc(f) >= 0 && face_zone[f] >= 0 &&
                    prescribes_state(bcs(face_bc(f)).type, mix)) {
                    near_faces.insert(f);
                }
            }
        }
    }
    comm::allreduce(std::span<double>(counts), comm::Op::SUM);
    box = comm::allreduce(box, comm::Op::MAX);
    for (int d = 0; d < 3; d++) {
        e.lo[size_t(d)] = d < N_DIM ? -box[size_t(d)] : 0.0;
        e.hi[size_t(d)] = d < N_DIM ? box[3 + size_t(d)] : 0.0;
    }
    for (size_t q = 0; q < divergence::N_QUANTITIES; q++) e.failed[q] = static_cast<uint64_t>(counts[q]);
    e.n_invalid = static_cast<uint64_t>(counts[N_INVALID]);
    e.n_over_limit = static_cast<uint64_t>(counts[N_OVER]);
    e.n_repaired = static_cast<uint64_t>(counts[N_REPAIRED]);
    e.n_troubled = static_cast<uint64_t>(counts[N_TROUBLED]);
    e.n_limited = static_cast<uint64_t>(counts[N_LIMITED]);
    e.chemistry_cost_mean = n_cells_global ? counts[COST] / double(n_cells_global) : 0.0;
    for (size_t s = 0; s < repaired.size(); s++) {
        if (counts[COST + 1 + s] > 0.0) e.repairs.push_back(repaired[s].first);
    }
    for (size_t z = 0; z < n_zones; z++) e.zones[z].near = static_cast<uint64_t>(counts[COST + 1 + repaired.size() + z]);
    if (reacting) {
        double substeps = 0.0, rejected = 0.0;
        for (const auto & [name, value] : run_health.last.chemistry) {
            if (name == "substeps") substeps = value;
            if (name == "rejected_substeps") rejected = value;
        }
        if (substeps > 0.0) e.rejected_fraction = rejected / substeps;
    }
    for (size_t m = 0; m < MeshQuality::N_METRICS; m++) {
        for (const double v : quality.cells[m]) e.mesh_max[m] = std::max(e.mesh_max[m], v);
    }
    e.mesh_max = comm::allreduce(e.mesh_max, comm::Op::MAX);

    // Exterior states of the boundary faces next to the invalid cells
    {
        const std::vector<uint32_t> list(near_faces.begin(), near_faces.end());
        std::vector<double> local;  // per face: zone, key, inadmissible, ratio, jump, p_g, p_c, rho_g, rho_c
        if (comm::allreduce(static_cast<uint64_t>(list.size()), comm::Op::SUM) > 0) {
            const auto W_g = ghost_states(boundary_data, list, *mesh, conservatives, primitives);
            for (size_t k = 0; k < list.size(); k++) {
                const uint32_t f = list[k];
                const auto c = static_cast<uint32_t>(mesh->h_cells_of_face(f, 0));
                const double rho_g = double(W_g(k, 0)), p_g = double(W_g(k, N_DIM + 1));
                const double rho_c = double(U(c, 0)), p_c = double(P(c, N_DIM));
                const bool bad = !(rho_g > 0.0) || !(p_g > 0.0) || !finite(rho_g) || !finite(p_g);
                double ratio = 1.0, jump = 0.0;
                if (!bad && rho_c > 0.0 && p_c > 0.0) {
                    ratio = std::max({p_g / p_c, p_c / p_g, rho_g / rho_c, rho_c / rho_g});
                    if (!mix) {
                        double du2 = 0.0;
                        for (int d = 0; d < N_DIM; d++) du2 += std::pow(double(W_g(k, 1 + d)) - double(P(c, d)), 2);
                        jump = std::sqrt(du2) / std::sqrt(double(phys.gamma) * p_c / rho_c);
                    }
                }
                const double key = bad ? std::numeric_limits<double>::max() : std::max(ratio, jump);
                local.insert(local.end(), {double(face_zone[f]), key, double(bad), ratio, jump, p_g, p_c, rho_g, rho_c});
            }
        }
        const std::vector<double> all = comm::size() > 1 ? comm::allgatherv(local) : local;
        constexpr size_t W = 9;
        std::vector<double> best_key(n_zones, -1.0);
        for (size_t r = 0; r + W <= all.size(); r += W) {
            divergence::Zone & z = e.zones[static_cast<size_t>(all[r])];
            z.prescribed = true;
            z.inadmissible = z.inadmissible || all[r + 2] != 0.0;
            z.ratio = std::max(z.ratio, all[r + 3]);
            z.jump = std::max(z.jump, all[r + 4]);
            if (all[r + 1] > best_key[static_cast<size_t>(all[r])]) {
                best_key[static_cast<size_t>(all[r])] = all[r + 1];
                z.p_ghost = all[r + 5], z.p_cell = all[r + 6], z.rho_ghost = all[r + 7], z.rho_cell = all[r + 8];
            }
        }
    }

    // The invalid cell closest to failing at the last valid state, the next ones, and its neighbors
    std::sort(candidates.begin(), candidates.end(), [&](const auto & a, const auto & b) {
        return a.first < b.first || (a.first == b.first && mesh->h_global_cell(a.second) < mesh->h_global_cell(b.second));
    });
    std::vector<double> local;
    for (size_t k = 0; k < std::min(candidates.size(), N_SHOWN + 1); k++) {
        std::vector<double> r = make_cell(candidates[k].second).pack();
        r[23] = candidates[k].first;
        local.insert(local.end(), r.begin(), r.end());
    }
    const std::vector<double> all = comm::size() > 1 ? comm::allgatherv(local) : local;
    std::vector<std::pair<double, Cell>> shown;
    for (size_t r = 0; r + Cell::WIDTH <= all.size(); r += Cell::WIDTH) shown.emplace_back(all[r + 23], Cell::unpack(&all[r]));
    std::stable_sort(shown.begin(), shown.end(), [](const auto & a, const auto & b) {
        return a.first < b.first || (a.first == b.first && a.second.id < b.second.id);
    });
    if (!shown.empty()) e.focus = shown[0].second;
    for (size_t k = 1; k < std::min(shown.size(), N_SHOWN + 1); k++) e.others.push_back(shown[k].second);

    std::vector<double> mine;
    if (!shown.empty()) {
        std::unordered_map<uint64_t, uint32_t> owned;
        for (uint32_t c = 0; c < n_owned; c++) owned[mesh->h_global_cell(c)] = c;
        for (const uint64_t id : e.focus.neighbors) {
            const auto it = owned.find(id);
            if (it == owned.end()) continue;
            const std::vector<double> r = make_cell(it->second).pack();
            mine.insert(mine.end(), r.begin(), r.end());
        }
    }
    const std::vector<double> neighbors = comm::size() > 1 ? comm::allgatherv(mine) : mine;
    for (size_t r = 0; r + Cell::WIDTH <= neighbors.size(); r += Cell::WIDTH) e.neighbors.push_back(Cell::unpack(&neighbors[r]));
    std::sort(e.neighbors.begin(), e.neighbors.end(), [](const Cell & a, const Cell & b) { return a.id < b.id; });

    return e;
}

std::vector<divergence::Cause> Solver::write_divergence_report(Evidence & e, const std::vector<int> & failed,
                                                               const MeshQuality & quality) {
    const bool files = output_files_enabled() && !divergence_prefix.empty();
    if (files) {
        create_parent_directories("the divergence report", divergence_prefix);
        // The invalid cells and the cells within a few layers of them, of the state in place (the last valid one)
        const uint32_t n_owned = mesh->n_owned(), n_local = mesh->n_cells;
        std::vector<int> layer(n_local, -1);
        std::vector<uint32_t> frontier;
        for (uint32_t c = 0; c < n_local; c++) {
            if (c < failed.size() && failed[c] >= 0) {
                layer[c] = 0;
                frontier.push_back(c);
            }
        }
        for (int l = 1; l <= SNAPSHOT_LAYERS; l++) {
            std::vector<uint32_t> next;
            for (const uint32_t c : frontier) {
                for (uint32_t k = mesh->h_offsets_faces_of_cell(c); k < mesh->h_offsets_faces_of_cell(c + 1); k++) {
                    const uint32_t f = mesh->h_faces_of_cell(k);
                    for (int s = 0; s < 2; s++) {
                        const int32_t n = mesh->h_cells_of_face(f, s);
                        if (n >= 0 && layer[static_cast<uint32_t>(n)] < 0) {
                            layer[static_cast<uint32_t>(n)] = l;
                            next.push_back(static_cast<uint32_t>(n));
                        }
                    }
                }
            }
            frontier = std::move(next);
        }
        CellSubset subset;
        subset.name = "divergence neighborhood";
        subset.fractions.assign(n_local, 0.0);
        for (uint32_t c = 0; c < n_owned; c++) {
            if (layer[c] < 0) continue;
            subset.cells.push_back(c);
            subset.fractions[c] = 1.0;
        }
        copy_device_to_host();
        std::vector<Data> fields;
        std::vector<std::string> names;
        std::vector<std::string> wanted = {"RHO", "CFL"};
        for (size_t i = 0; i + 1 < PRIMITIVE_NAMES.size(); i++) wanted.push_back(PRIMITIVE_NAMES[i]);
        for (const auto & s : species_names) wanted.push_back("Y_" + s);
        for (const Data & d : data) {
            if (std::find(wanted.begin(), wanted.end(), d.name()) == wanted.end()) continue;
            fields.push_back(d);
            names.push_back(d.name());
        }
        Kokkos::View<rtype *, Kokkos::HostSpace> diverged("DIVERGED", n_local);
        for (uint32_t c = 0; c < n_local; c++) diverged(c) = c < failed.size() && failed[c] >= 0;
        fields.emplace_back("DIVERGED", diverged);
        names.emplace_back("DIVERGED");
        std::vector<Kokkos::View<rtype *, Kokkos::HostSpace>> metrics;
        for (size_t m = 0; m < MeshQuality::N_METRICS; m++) {
            metrics.emplace_back(MeshQuality::FIELDS[m], n_local);
            for (uint32_t c = 0; c < n_owned && c < quality.cells[m].size(); c++) {
                metrics.back()(c) = static_cast<rtype>(quality.cells[m][c]);
            }
            fields.emplace_back(MeshQuality::FIELDS[m], metrics.back());
            names.emplace_back(MeshQuality::FIELDS[m]);
        }
        Input table(Input::table_type{});
        table.as_table()["prefix"] = divergence_prefix + "_neighborhood";
        table.as_table()["interval"] = 1;
        table.as_table()["variables"] = names;
        table.as_table()["format"] = std::string(have_hdf5() && (comm::size() == 1 || have_parallel_hdf5()) ? "vtkhdf" : "vtu");
        DataWriter writer;
        writer.init(table, fields, mesh, {}, &subset);
        writer.write(step, t, true);
        writer.finish();
        e.snapshot_file = writer.last_written();
        e.report_file = divergence_prefix + ".txt";
    }

    const std::vector<divergence::Cause> causes = divergence::rank_causes(e);
    const std::string text = divergence::text(e, causes);
    logging::line();
    for (std::istringstream lines(text); !lines.eof();) {
        std::string line;
        std::getline(lines, line);
        logging::line(line);
    }
    const json::Object record = divergence::record(e, causes);
    run_log::write(record);
    if (files && comm::is_root()) {
        std::ofstream(divergence_prefix + ".txt") << text;
        std::ofstream(divergence_prefix + ".json") << record.str() << "\n";
    }
    return causes;
}
