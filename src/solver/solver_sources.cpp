/**
 * @file solver_sources.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Source terms of the Solver: [[sources]] and the mass-flow forcing.
 * @version 0.1
 * @date 2026-11-24
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "solver.h"
#include "gated.h"

#include "input.h"

#include <cmath>
#include <memory>
#include <optional>

#include <Kokkos_Core.hpp>

#include "comm.h"
#include "device_comm.h"
#include "log.h"
#include "region_input.h"
#include "region_motion.h"
#include "provenance.h"

region::CellFields Solver::cell_fields() const {
    region::CellFields fields;
    fields.coords = mesh->cell_coords;
    fields.W = W_cells;
    if (is_mixture()) {
        fields.Y = cell_scalars;
        fields.inv_W = mixture.thermo.inv_W;
        fields.R_universal = chemistry::GAS_CONSTANT;
    } else {
        fields.R = double(physics.R);
        fields.gamma = double(physics.get_gamma());
    }
    return fields;
}

void Solver::init_sources(region::Regions & regions) {
    auto terms = std::make_unique<sources::SourceTerms>(input, mesh, species_names, cell_fields(), regions,
                                                         is_distributed() && halo.uses_nccl());
    if (!terms->empty()) {
        source_summary.add(terms->summary());
        source_terms = std::move(terms);
        stage_time = Kokkos::View<rtype>("stage_time");
    }
    if (!input.contains("source")) {
        return;
    }
    const Input & source = input.at("source");
    if (contains_optional(source, "gravity")) {
        std::vector<rtype> g = find_real_vector(input, "source", "gravity", dim::acceleration);
        if (g.size() != N_DIM) {
            throw InputError({.key = "source.gravity",
                              .line = line_of(source, "gravity"),
                              .message = "has " + logging::count(g.size(), "component") + ", but this build is " +
                                         std::to_string(N_DIM) + "D",
                              .fix = N_DIM == 3 ? "Write gravity = [gx, gy, gz]" : "Write gravity = [gx, gy]"});
        }
        has_gravity = true;
        FOR_I_DIM gravity[i] = g[i];
        FOR_I_DIM boundary_data.gravity[i] = g[i];
        face_reconstruction->set_boundaries(boundary_data);
        FOR_I_DIM source_summary.quantity(std::string("gravity ") + "xyz"[i], logging::real(double(gravity[i])),
                                          units::column("m/s^2"), "");
    }
    if (contains_optional(source, "mass_flow")) {
        const std::vector<rtype> m = find_real_vector(input, "source", "mass_flow", dim::mass_flux);
        if (m.size() != N_DIM) {
            throw InputError({.key = "source.mass_flow",
                              .line = line_of(source, "mass_flow"),
                              .message = "has " + logging::count(m.size(), "component") + ", but this build is " +
                                         std::to_string(N_DIM) + "D",
                              .fix = N_DIM == 3 ? "Write mass_flow = [mx, my, mz]" : "Write mass_flow = [mx, my]"});
        }
        rtype norm = 0.0_r;
        FOR_I_DIM norm += m[i] * m[i];
        norm = std::sqrt(norm);
        if (!(norm > 0.0_r) || !std::isfinite(norm)) {
            throw InputError({.key = "source.mass_flow",
                              .line = line_of(source, "mass_flow"),
                              .message = "must be a finite, nonzero vector: its direction is the flow's",
                              .fix = "Give the target mass flux per unit area along the flow, e.g. [1.0, 0.0]"});
        }
        hold_mass_flow = true;
        mass_flow_target = double(norm);
        FOR_I_DIM mass_flow_direction[i] = m[i] / norm;
        const uint64_t n_global = is_distributed() ? n_cells_global : mesh->n_owned();
        const int bits = 62 - static_cast<int>(std::ceil(std::log2(static_cast<double>(n_global) + 1.0)));
        mass_flow_unit = std::ldexp(1.0, bits);
        mass_flow_terms = Kokkos::View<double *[2]>("mass_flow_terms", mesh->n_owned());
        mass_flow_largest = Kokkos::View<double *>("mass_flow_largest", 2);
        mass_flow_sums = Kokkos::View<int64_t *>("mass_flow_sums", 2);
        mass_flow_totals = Kokkos::View<double *>("mass_flow_totals", 3);
        init_mass_flow_volume();
        source_summary.setting("Mass flow", "volume average of rho u, held by a uniform body force");
        FOR_I_DIM source_summary.quantity(std::string("mass flow ") + "xyz"[i], logging::real(double(m[i])),
                                          units::column("kg/(m^2 s)"), "volume average of rho u");
    }
}

void Solver::init_mass_flow_volume() {
    const Kokkos::View<double *[2]> terms = mass_flow_terms;
    const Kokkos::View<rtype *> vol = mesh->cell_volume;
    gated::parallel_for("mass_flow_volume", mesh->n_owned(), KOKKOS_LAMBDA(const uint32_t i_cell) {
        terms(i_cell, 0) = static_cast<double>(vol(i_cell));
        terms(i_cell, 1) = 0.0;
    });
    sum_mass_flow_terms();
    Kokkos::deep_copy(domain_volume, Kokkos::subview(mass_flow_totals, 0));
}

void Solver::sum_mass_flow_terms() {
    const uint32_t n_owned = mesh->n_owned();
    const Kokkos::View<double *[2]> terms = mass_flow_terms;
    using DeviceMemory = Kokkos::DefaultExecutionSpace::memory_space;
    for (int k = 0; k < 2; k++) {
        gated::parallel_reduce("mass_flow_largest", n_owned, KOKKOS_LAMBDA(const uint32_t c, double & m) {
            m = Kokkos::fmax(m, Kokkos::fabs(terms(c, k)));
        }, Kokkos::Max<double, DeviceMemory>(Kokkos::View<double, DeviceMemory>(mass_flow_largest.data() + k)));
    }
    if (is_distributed()) comm::allreduce_device(mass_flow_largest, comm::Op::MAX, halo.uses_nccl());
    const Kokkos::View<double *> largest = mass_flow_largest;
    const double unit = mass_flow_unit;
    for (int k = 0; k < 2; k++) {
        gated::parallel_reduce("mass_flow_sum", n_owned, KOKKOS_LAMBDA(const uint32_t c, int64_t & sum) {
            if (largest(k) > 0.0) sum += static_cast<int64_t>(Kokkos::round(terms(c, k) * (unit / largest(k))));
        }, Kokkos::Sum<int64_t, DeviceMemory>(Kokkos::View<int64_t, DeviceMemory>(mass_flow_sums.data() + k)));
    }
    if (is_distributed()) comm::allreduce_device(mass_flow_sums, comm::Op::SUM, halo.uses_nccl());
    const Kokkos::View<int64_t *> sums = mass_flow_sums;
    const Kokkos::View<double *> totals = mass_flow_totals;
    gated::parallel_for("mass_flow_totals", 1, KOKKOS_LAMBDA(const int) {
        for (int k = 0; k < 2; k++) totals(k) = largest(k) > 0.0 ? static_cast<double>(sums(k)) / (unit / largest(k)) : 0.0;
    });
}

void Solver::add_mass_flow_force(StateView solution, StateView rhs) {
    const uint32_t n_owned = mesh->n_owned();
    const Kokkos::View<rtype *> vol = mesh->cell_volume;
    Kokkos::Array<rtype, N_DIM> e;
    FOR_I_DIM e[i] = mass_flow_direction[i];
    // Along e: the volume integral of rho u and the rate of change of that integral from all other terms
    const Kokkos::View<double *[2]> terms = mass_flow_terms;
    gated::parallel_for("mass_flow_terms", n_owned, KOKKOS_LAMBDA(const uint32_t i_cell) {
        rtype m = 0.0_r, r = 0.0_r;
        FOR_I_DIM {
            m += solution(i_cell, 1 + i) * e[i];
            r += rhs(i_cell, 1 + i) * e[i];
        }
        terms(i_cell, 0) = static_cast<double>(m * vol(i_cell));
        terms(i_cell, 1) = static_cast<double>(r);
    });
    sum_mass_flow_terms();
    // Cancel the other terms, and close the gap to the target within the step, whose dt a captured step has only on the device
    const Kokkos::View<double *> totals = mass_flow_totals;
    const double target = mass_flow_target, volume = domain_volume;
    const rtype * dt_device = recording_step ? step_dt_device() : nullptr;
    const rtype dt_host = dt;
    gated::parallel_for("mass_flow_force", 1, KOKKOS_LAMBDA(const int) {
        const rtype dt_ = dt_device ? *dt_device : dt_host;
        double force = -totals(1) / volume;
        if (dt_ > 0.0_r) force += (target - totals(0) / volume) / static_cast<double>(dt_);
        totals(2) = force;
    });
    gated::parallel_for("rhs_mass_flow_force", n_owned, KOKKOS_LAMBDA(const uint32_t i_cell) {
        Kokkos::Array<rtype, N_DIM> f;
        FOR_I_DIM f[i] = static_cast<rtype>(totals(2)) * e[i];
        const rtype V = vol(i_cell);
        rtype work = 0.0_r;
        FOR_I_DIM {
            rhs(i_cell, 1 + i) += f[i] * V;
            work += f[i] * solution(i_cell, 1 + i);
        }
        rhs(i_cell, N_DIM + 1) += work / solution(i_cell, 0) * V;
    });
}

double Solver::get_mass_flow_force() const {
    const comm::SelfScope scope = communication_scope();
    if (!hold_mass_flow) return 0.0;
    double force = 0.0;
    Kokkos::deep_copy(force, Kokkos::subview(mass_flow_totals, 2));
    return force;
}
