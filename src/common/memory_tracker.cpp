/**
 * @file memory_tracker.cpp
 * @brief Bytes held in Kokkos allocations, by memory space and by category of
 *        their labels, from Kokkos' tool hooks.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "memory_tracker.h"

#include <algorithm>
#include <cstring>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include <Kokkos_Core.hpp>

namespace memory {

namespace {

using C = Category;

struct Rule {
    const char * label;
    bool prefix;
    Category category;
};

// First match wins: exact labels, then prefixes from the most specific
constexpr Rule RULES[] = {
    // Solution, stages and cell fields
    {"conservatives", false, C::STATE}, {"species", false, C::STATE}, {"primitives", false, C::STATE},
    {"W_cells", false, C::STATE}, {"cfl_local", false, C::STATE}, {"solution", true, C::STATE},
    {"rhs_faces", false, C::MESH}, {"rhs", true, C::STATE}, {"last_good", true, C::STATE}, {"T_seed", false, C::STATE},
    {"cell_scalars", false, C::STATE}, {"cell_molar_mass", false, C::STATE}, {"p_max", false, C::STATE},
    {"vortex_fields", false, C::STATE}, {"derived_fields", false, C::STATE},
    {"positivity_kappa", false, C::STATE}, {"cell_upwind", false, C::STATE},
    {"frozen_thermo", false, C::STATE}, {"stage_time", false, C::STATE}, {"step_", true, C::STATE},
    {"geometric_source", false, C::STATE}, {"cell_mu", false, C::STATE}, {"rhoY", false, C::STATE},
    // Faces, gradients and reconstruction buffers
    {"face_solution", false, C::FACES}, {"face_flux", false, C::FACES}, {"face_thermo", false, C::FACES},
    {"face_mdot", false, C::FACES}, {"face_energy_1", false, C::FACES}, {"species_slots", false, C::FACES},
    {"gradients", false, C::FACES}, {"viscous_gradients", false, C::FACES}, {"limiters", false, C::FACES},
    {"scalar_", true, C::FACES}, {"image_point_W", false, C::FACES}, {"teno_troubled", true, C::FACES},
    {"teno_n_troubled", false, C::FACES}, {"teno_selection", false, C::FACES}, {"cell_transport", false, C::FACES},
    {"cell_diffusion", false, C::FACES}, {"transport_values", false, C::FACES},
    {"transport_gradients", false, C::FACES},
    // TENO setup's workspace
    {"teno_batch_", true, C::SETUP}, {"teno_setup_", true, C::SETUP}, {"teno_moments_rows", false, C::SETUP},
    {"teno_rows", false, C::SETUP}, {"teno_boxes", false, C::SETUP}, {"teno_cell_rule", false, C::SETUP},
    {"teno_pack_", true, C::SETUP}, {"teno_large_size_rows", false, C::SETUP},
    {"teno_small_size_rows", false, C::SETUP}, {"teno_scale_rows", false, C::SETUP},
    {"teno_basis_mean_rows", false, C::SETUP}, {"teno_si_rows", false, C::SETUP},
    {"teno_entry_rotations", false, C::SETUP}, {"teno_boundary_faces", false, C::SETUP},
    {"teno_", true, C::TABLES},
    // Mesh, geometry and quadrature
    {"node_coords", false, C::MESH}, {"nodes_of_", true, C::MESH}, {"cells_of_", true, C::MESH},
    {"faces_of_cell", false, C::MESH}, {"offsets_", true, C::MESH}, {"cell_coords", false, C::MESH},
    {"cell_volume", false, C::MESH}, {"cell_measure", false, C::MESH}, {"face_area", false, C::MESH},
    {"face_normals", false, C::MESH}, {"face_coords", false, C::MESH}, {"face_measure", false, C::MESH},
    {"face_quad_", true, C::MESH}, {"face_image", true, C::MESH}, {"image_", true, C::MESH},
    {"face_bc", false, C::MESH}, {"face_shift", false, C::MESH}, {"shift", true, C::MESH},
    {"rotated_faces", false, C::MESH}, {"zone_", true, C::MESH}, {"flux_weights", false, C::MESH},
    {"vertex_gradient_weights", true, C::MESH}, {"points", false, C::MESH}, {"weights", false, C::MESH},
    // Boundaries, sources, sponges, LES and wall models
    {"face_char", false, C::MODELS}, {"face_state", true, C::MODELS}, {"bc_", true, C::MODELS},
    {"bcs", false, C::MODELS}, {"char_", true, C::MODELS}, {"dirichlet_", true, C::MODELS},
    {"timed_", true, C::MODELS}, {"average_pressure_faces", false, C::MODELS}, {"outlet_", true, C::MODELS},
    {"sponge_", true, C::MODELS}, {"source_", true, C::MODELS}, {"SOURCE", false, C::MODELS},
    {"les_", true, C::MODELS}, {"dynamic_halo", true, C::HALO}, {"dynamic_", true, C::MODELS},
    {"tfles_halo", true, C::HALO}, {"tfles_", true, C::MODELS}, {"budget_", true, C::MODELS}, {"mass_flow_", true, C::MODELS},
    {"inflow_", true, C::MODELS}, {"wall_", true, C::MODELS}, {"expr_", true, C::MODELS},
    // Chemistry, thermodynamics and transport
    {"kinetics_", true, C::CHEMISTRY}, {"lu_", true, C::CHEMISTRY}, {"chem_work", false, C::WORK}, {"chem_pivot", false, C::WORK}, {"chem_", true, C::CHEMISTRY},
    {"chemistry_", true, C::CHEMISTRY}, {"thermo_", true, C::CHEMISTRY}, {"transport_", true, C::CHEMISTRY},
    {"production", false, C::CHEMISTRY}, {"hrr", false, C::CHEMISTRY}, {"radiation_", true, C::CHEMISTRY},
    // Statistics, probes and run health
    {"probe_", true, C::DIAGNOSTICS}, {"statistics_", true, C::DIAGNOSTICS}, {"spectra_", true, C::DIAGNOSTICS},
    {"ledger_", true, C::DIAGNOSTICS}, {"phase_", true, C::DIAGNOSTICS}, {"force_faces", false, C::DIAGNOSTICS},
    {"cell_covariance", false, C::DIAGNOSTICS}, {"repair_", true, C::DIAGNOSTICS},
    {"reset_energy_", true, C::DIAGNOSTICS}, {"clip_flags", false, C::DIAGNOSTICS},
    {"integral", true, C::DIAGNOSTICS}, {"exact_sums", true, C::DIAGNOSTICS},
    // Output staging (host fields and mirrors)
    {"D", false, C::OUTPUT}, {"X", false, C::OUTPUT}, {"Y", false, C::OUTPUT}, {"QRAD", false, C::OUTPUT},
    {"REGION_FRACTION", false, C::OUTPUT}, {"output_", true, C::OUTPUT},
    // Halo exchange
    {"halo_", true, C::HALO},
    // Setup
    {"init_", true, C::SETUP}, {"data_", true, C::SETUP}, {"region_", true, C::SETUP}, {"expression_", true, C::SETUP},
    // Kokkos
    {"Kokkos::", true, C::RUNTIME},
};

constexpr const char * NAMES[N_CATEGORIES] = {
    "solution and stages",  "faces and gradients",  "reconstruction tables", "mesh and geometry",
    "boundaries and models", "chemistry",           "diagnostics",           "output staging",
    "halo buffers",          "work memory",         "setup",                 "runtime and pages",
    "other",
};

struct Space {
    Usage usage;
    uint64_t phase_peak = 0;
    Bytes at_phase_peak{};
};

struct Allocation {
    int space;
    uint64_t size;
    Category category;
    std::string label;
};

struct State {
    bool active = false;
    Space spaces[2];  // device, host
    std::unordered_map<const void *, Allocation> live;
    std::set<std::string> unclassified;
    Kokkos::Tools::Experimental::EventSet previous{};
};

State state;

int space_index(const Kokkos::Profiling::SpaceHandle handle) {
    return std::strncmp(handle.name, "Host", 4) == 0 || std::strstr(handle.name, "HostPinned") != nullptr ? 1 : 0;
}

bool ends_with(const std::string_view s, const std::string_view tail) {
    return s.size() >= tail.size() && s.substr(s.size() - tail.size()) == tail;
}

void on_allocate(const Kokkos::Profiling::SpaceHandle handle, const char * label, const void * ptr,
                 const uint64_t size) {
    if (state.previous.allocate_data) state.previous.allocate_data(handle, label, ptr, size);
    const int s = space_index(handle);
    const Category c = category(label);
    if (c == Category::OTHER) state.unclassified.insert(label);
    state.live[ptr] = {s, size, c, label ? label : ""};
    Space & space = state.spaces[s];
    Usage & u = space.usage;
    u.current += size;
    u.by_category[static_cast<size_t>(c)] += size;
    if (u.current > u.peak) {
        u.peak = u.current;
        u.at_peak = u.by_category;
    }
    if (u.current > space.phase_peak) {
        space.phase_peak = u.current;
        space.at_phase_peak = u.by_category;
    }
}

void on_deallocate(const Kokkos::Profiling::SpaceHandle handle, const char * label, const void * ptr,
                   const uint64_t size) {
    if (state.previous.deallocate_data) state.previous.deallocate_data(handle, label, ptr, size);
    const auto it = state.live.find(ptr);
    if (it == state.live.end()) return;
    Usage & u = state.spaces[it->second.space].usage;
    u.current -= it->second.size;
    u.by_category[static_cast<size_t>(it->second.category)] -= it->second.size;
    state.live.erase(it);
}

} // namespace

const char * category_name(const Category c) { return NAMES[static_cast<size_t>(c)]; }

bool scales_with_cells(const Category c) { return c != Category::SETUP && c != Category::WORK && c != Category::RUNTIME; }

Category category(const std::string_view label) {
    // Host copies of device views (create_mirror, create_mirror_view in device builds)
    if (ends_with(label, "_mirror")) return Category::OUTPUT;
    // GrowingBuffer: the rest of its last page
    if (ends_with(label, "_pages")) return Category::RUNTIME;
    for (const Rule & rule : RULES) {
        if (rule.prefix ? label.substr(0, std::strlen(rule.label)) == rule.label : label == rule.label) {
            return rule.category;
        }
    }
    return Category::OTHER;
}

Tracker::Tracker() {
    if (state.active) throw std::logic_error("memory::Tracker: one at a time.");
    state = State();
    state.active = true;
    state.previous = Kokkos::Tools::Experimental::get_callbacks();
    Kokkos::Tools::Experimental::set_allocate_data_callback(on_allocate);
    Kokkos::Tools::Experimental::set_deallocate_data_callback(on_deallocate);
}

Tracker::~Tracker() {
    Kokkos::Tools::Experimental::set_allocate_data_callback(state.previous.allocate_data);
    Kokkos::Tools::Experimental::set_deallocate_data_callback(state.previous.deallocate_data);
    state = State();
}

namespace {

Usage usage(const Space & space) {
    Usage u = space.usage;
    u.phases.push_back(space.at_phase_peak);
    return u;
}

void start_phase(Space & space) {
    space.phase_peak = space.usage.current;
    space.at_phase_peak = space.usage.by_category;
}

} // namespace

void end_phase() {
    if (!state.active) return;
    for (Space & space : state.spaces) {
        space.usage.phases.push_back(space.at_phase_peak);
        start_phase(space);
    }
}

Usage Tracker::device() const { return usage(state.spaces[0]); }

Usage Tracker::host() const { return usage(state.spaces[1]); }

void Tracker::reset_peak() {
    for (Space & space : state.spaces) {
        space.usage.peak = space.usage.current;
        space.usage.at_peak = space.usage.by_category;
        space.usage.phases.clear();
        start_phase(space);
    }
}

std::map<std::string, uint64_t> Tracker::live(const bool device) const {
    std::map<std::string, uint64_t> bytes;
    for (const auto & [ptr, a] : state.live) {
        if ((a.space == 0) == device) bytes[a.label] += a.size;
    }
    return bytes;
}

std::vector<std::string> Tracker::unclassified() const {
    return std::vector<std::string>(state.unclassified.begin(), state.unclassified.end());
}

} // namespace memory
