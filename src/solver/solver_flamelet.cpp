/**
 * @file solver_flamelet.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Flamelet-gas parts of the Solver (gas = "flamelet", docs/design/flamelet.md):
 *        the models it refuses, the progress-variable source and, in LES, the
 *        subgrid variance of the mixture fraction.
 * @version 0.1
 * @date 2026-11-20
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "solver.h"

#include <Kokkos_Core.hpp>

#include "comm.h"
#include "diagnostic_code.h"
#include "input.h"
#include "log.h"
#include "mixture_viscous_flux.h"

namespace {

/** @brief Adds the table's source of rho C at each cell's [Z, C] and density to the scalars' rates (per unit volume). */
struct ProgressSourceFunctor {
    flamelet::Gas<> gas;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W;
    ScalarView scalars;
    SpeciesView rhs;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c) const {
        using G = flamelet::Gas<>;
        const flamelet::Point<3> p = gas.locate(static_cast<double>(scalars(c, G::Z)), gas.cell_variance(c),
                                                static_cast<double>(scalars(c, G::C)));
        rhs(c, G::C) += static_cast<rtype>(gas.source(p, static_cast<double>(W(c, 0))));
    }
};

/**
 * @brief The algebraic subgrid variance of Z (Pierce & Moin 1998), C_Z Delta^2 |grad Z|^2 with Delta
 *        the eddy viscosity's filter width, at most Z (1 - Z), the variance of a mixture of the pure
 *        streams. grad Z is the least-squares gradient of Z = rho Z / rho of the state over the viscous
 *        stencil, with the ghost values of MixtureGradientFunctor: inlet_state faces add their Z, other
 *        boundaries nothing.
 */
struct AlgebraicVarianceFunctor {
    LSQVertexGradientFunctor stencil;
    StateView U;
    SpeciesView rhoY;
    Kokkos::View<rtype **, Kokkos::LayoutRight> bc_values;  // (condition, [conducted, Z, C]) of inlet_state
    Kokkos::View<rtype *> delta;
    double C_Z;
    Kokkos::View<rtype *> variance;

    KOKKOS_INLINE_FUNCTION double Z(const uint32_t c) const {
        return static_cast<double>(rhoY(c, flamelet::Gas<>::Z)) / static_cast<double>(U(c, 0));
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c) const {
        const double Z_c = Z(c);
        double g[N_DIM] = {};
        for (uint32_t k = stencil.offsets_cells_of_cell(c); k < stencil.offsets_cells_of_cell(c + 1); k++) {
            const double dZ = Z(stencil.cells_of_cell(k)) - Z_c;
            FOR_I_DIM g[i] += static_cast<double>(stencil.weights.cells(k, i)) * dZ;
        }
        stencil.for_each_image(c, [&](const uint32_t j, const uint32_t k) {
            const double dZ = Z(j) - Z_c;
            FOR_I_DIM g[i] += static_cast<double>(stencil.weights.images(k, i)) * dZ;
        });
        const LSQGradientFunctor & f = stencil.faces;
        stencil.for_each_ghost(c, [&](const uint32_t i_face, const uint32_t k) {
            const int32_t i_bc = f.boundaries.face_bc(i_face);
            const BoundaryCondition & bc = f.boundaries.bcs(i_bc);
            if (bc.type != BoundaryType::INLET_STATE || bc.is_characteristic()) return;
            const double dZ = static_cast<double>(bc_values(i_bc, 1 + flamelet::Gas<>::Z)) - Z_c;
            FOR_I_DIM g[i] += static_cast<double>(stencil.weights.faces(k, i)) * dZ;
        });
        double g2 = 0.0;
        FOR_I_DIM g2 += g[i] * g[i];
        const double d = static_cast<double>(delta(c));
        const double z = Kokkos::fmin(1.0, Kokkos::fmax(0.0, Z_c));
        variance(c) = static_cast<rtype>(Kokkos::fmin(C_Z * d * d * g2, z * (1.0 - z)));
    }
};

/** @brief The subgrid variance from the transported second moment, Z2 - Z^2, within [0, Z (1 - Z)]. */
struct SecondMomentVarianceFunctor {
    StateView U;
    SpeciesView rhoY;
    Kokkos::View<rtype *> variance;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c) const {
        using G = flamelet::Gas<>;
        const double rho = static_cast<double>(U(c, 0));
        const double Z = Kokkos::fmin(1.0, Kokkos::fmax(0.0, static_cast<double>(rhoY(c, G::Z)) / rho));
        const double Zvar = static_cast<double>(rhoY(c, G::Z2)) / rho - Z * Z;
        variance(c) = static_cast<rtype>(Kokkos::fmin(Kokkos::fmax(Zvar, 0.0), Z * (1.0 - Z)));
    }
};

/**
 * @brief Sink of the second moment rho Z2: its filtered dissipation rate rho chi = 2 rho D |grad Z|^2 (the
 *        resolved part) + rho chi_sgs, chi_sgs = C_chi (D + D_t) Zvar / Delta^2. rho D is the conducted
 *        coefficient kappa / cp, rho D_t = mu_t / Sc_t of the SGS coefficients, grad Z the transport gradient.
 */
struct SecondMomentSinkFunctor {
    flamelet::Gas<> gas;
    Kokkos::View<rtype *[3]> transport;                          // (cell, [mu, kappa / cp, nu_eff])
    Kokkos::View<rtype *[3]> sgs;                                // (cell, [mu_t, mu_t / Pr_t, mu_t / Sc_t])
    Kokkos::View<rtype ***, Kokkos::LayoutRight> gradients;      // (cell, [u, h, Z, C, Z2], dimension)
    Kokkos::View<rtype *> delta;
    double C_chi;
    SpeciesView rhs;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c) const {
        using G = flamelet::Gas<>;
        double g2 = 0.0;
        FOR_I_DIM {
            const double g = static_cast<double>(gradients(c, N_DIM + 1 + G::Z, i));
            g2 += g * g;
        }
        const double rho_D = static_cast<double>(transport(c, 1)), rho_D_t = static_cast<double>(sgs(c, 2));
        const double d = static_cast<double>(delta(c));
        const double chi = 2.0 * rho_D * g2 + C_chi * (rho_D + rho_D_t) * gas.cell_variance(c) / (d * d);
        rhs(c, G::Z2) -= static_cast<rtype>(chi);
    }
};

/**
 * @brief An initial state converted at zero variance (p, T, u and the scalars given) brought to the
 *        same p, T, u and scalars at the cell's subgrid variance: rho = p / (R T) and e = e(T) of the
 *        table at (Z, Zvar, C).
 */
struct InitialVarianceFunctor {
    Mixture gas;
    StateView U;
    SpeciesView rhoY;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c) const {
        constexpr uint8_t E = N_DIM + 1;
        rtype cons[N_CONSERVATIVE], W[N_CONSERVATIVE], gamma, e0, T;
        FOR_I_CONSERVATIVE cons[i] = U(c, i);
        gas.cell_state(cons, cell_species(rhoY, c), 0.0_r, 0.0, W, gamma, e0, T);
        const PartialDensities y{cell_species(rhoY, c), 1.0 / static_cast<double>(cons[0])};
        const flamelet::Thermo th = gas.table_thermo(y, gas.flamelet.cell_variance(c));
        const double T_c = static_cast<double>(T);
        const double rho = static_cast<double>(W[E]) / (th.R * T_c), scale = rho / static_cast<double>(cons[0]);
        double u2 = 0.0;
        FOR_I_DIM u2 += static_cast<double>(W[1 + i]) * static_cast<double>(W[1 + i]);
        U(c, 0) = static_cast<rtype>(rho);
        FOR_I_DIM U(c, 1 + i) = static_cast<rtype>(rho * static_cast<double>(W[1 + i]));
        U(c, E) = static_cast<rtype>(rho * (th.energy(T_c) + 0.5 * u2));
        for (uint32_t k = 0; k < gas.n_species; k++) {
            rhoY(c, k) = static_cast<rtype>(static_cast<double>(rhoY(c, k)) * scale);
        }
    }
};

} // namespace

void Solver::update_flamelet_variance(const State & s) {
    if (!flamelet_variance.is_allocated()) return;
    if (mixture.flamelet.second_moment) {
        Kokkos::parallel_for("flamelet_variance", mesh->n_cells,
                             SecondMomentVarianceFunctor{s.flow, s.species, flamelet_variance});
        return;
    }
    // Owned cells have complete stencils; halo cells take their owners' variances
    const uint32_t n_owned = mesh->n_owned();
    Kokkos::parallel_for("flamelet_variance", halo.active() ? n_owned : mesh->n_cells,
                         AlgebraicVarianceFunctor{viscous_gradient, s.flow, s.species, bc_transport_values, les_width,
                                                  mixture_model->flamelet().variance_constant(), flamelet_variance});
    if (!halo.active()) return;
    const SpeciesView packed = flamelet_halo.species;
    const Kokkos::View<rtype *> variance = flamelet_variance;
    Kokkos::parallel_for(
        "flamelet_variance_pack", n_owned, KOKKOS_LAMBDA(const uint32_t c) { packed(c, 0) = variance(c); });
    halo.exchange(flamelet_halo);
    Kokkos::parallel_for("flamelet_variance_unpack", Kokkos::RangePolicy<>(n_owned, mesh->n_cells),
                         KOKKOS_LAMBDA(const uint32_t c) { variance(c) = packed(c, 0); });
}

void Solver::apply_initial_variance() {
    if (!flamelet_variance.is_allocated()) return;
    if (mixture.flamelet.second_moment) {
        // Zero variance, as at inlets: the cell averages of Z^2 would hold the profile's variance within each cell
        const SpeciesView rhoY = species;
        const StateView U = conservatives;
        Kokkos::parallel_for(
            "flamelet_initial_second_moment", mesh->n_cells, KOKKOS_LAMBDA(const uint32_t c) {
                using G = flamelet::Gas<>;
                rhoY(c, G::Z2) = rhoY(c, G::Z) * rhoY(c, G::Z) / U(c, 0);
            });
        return;
    }
    update_flamelet_variance(state());
    Kokkos::parallel_for("flamelet_initial_variance", mesh->n_cells,
                         InitialVarianceFunctor{mixture, conservatives, species});
}

void Solver::check_flamelet_input() const {
    auto refuse = [&](const char * key, const std::string & message, const std::string & fix) {
        throw InputError({.key = key,
                          .line = line_of(input, key),
                          .message = message,
                          .fix = fix,
                          .docs = "docs/guide/flamelet.md"});
    };
    if (input.contains("chemistry")) {
        refuse("chemistry", "finite-rate chemistry does not combine with gas = \"flamelet\": the table holds the chemistry",
               "Remove [chemistry], or use gas = \"mixture\" with a mechanism");
    }
    if (input.contains("radiation")) {
        refuse("radiation", "radiation with gas = \"flamelet\" needs tables with an enthalpy-defect axis, not yet read",
               "Remove [radiation]");
    }
    if (les_on && input.at("les").contains("combustion")) {
        throw InputError({.key = "les.combustion",
                          .line = line_of(input.at("les"), "combustion"),
                          .message = "the flamelet table already represents the subgrid flame: neither thickening "
                                     "(TFLES) nor PaSR combines with gas = \"flamelet\"",
                          .fix = "Remove [les.combustion]",
                          .docs = "docs/guide/flamelet.md"});
    }
}

void Solver::add_progress_source(SpeciesView rhs) {
    Kokkos::parallel_for("progress_source", mesh->n_owned(),
                         ProgressSourceFunctor{mixture.flamelet, W_cells, cell_scalars, rhs});
    if (mixture.flamelet.second_moment) {
        Kokkos::parallel_for("second_moment_sink", mesh->n_owned(),
                             SecondMomentSinkFunctor{mixture.flamelet, cell_transport, les_coefficients,
                                                     transport_gradients, les_width,
                                                     mixture_model->flamelet().dissipation_constant(), rhs});
    }
}

void Solver::check_table_pressure() const {
    const double p_table = mixture_model->flamelet().table().pressure();
    const auto P = primitives;
    double lo = 0.0, hi = 0.0;
    Kokkos::parallel_reduce(
        "table_pressure", mesh->n_owned(),
        KOKKOS_LAMBDA(const uint32_t c, double & l, double & h) {
            const double p = static_cast<double>(P(c, N_DIM));
            l = Kokkos::fmin(l, p);
            h = Kokkos::fmax(h, p);
        },
        Kokkos::Min<double>(lo), Kokkos::Max<double>(hi));
    lo = comm::allreduce(lo, comm::Op::MIN);
    hi = comm::allreduce(hi, comm::Op::MAX);
    std::vector<std::string> where;
    if (lo < 0.9 * p_table || hi > 1.1 * p_table) {
        where.push_back("the initial pressure (" + logging::real(lo) + " to " + logging::real(hi) + " Pa)");
    }
    const std::vector<Input> boundaries = contains_optional(input, "boundaries")
                                              ? find_value<std::vector<Input>>(input, "boundaries")
                                              : std::vector<Input>{};
    const auto h_bcs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundary_data.bcs);
    for (size_t i = 0; i < boundaries.size() && i < h_bcs.extent(0); i++) {
        const BoundaryCondition & bc = h_bcs(i);
        const bool prescribed = bc.type == BoundaryType::INLET_STATE || bc.type == BoundaryType::INLET_STAGNATION ||
                                bc.type == BoundaryType::OUTLET_PRESSURE;
        const double p = static_cast<double>(bc.data[N_DIM + 1]);
        if (prescribed && (p < 0.9 * p_table || p > 1.1 * p_table)) {
            where.push_back("boundary " + find_value<std::string>(boundaries[i], "name") + " (" + logging::real(p) + " Pa)");
        }
    }
    if (where.empty()) return;
    std::string list;
    for (const std::string & w : where) list += (list.empty() ? "" : ", ") + w;
    logging::warning(diagnostic_code::TABLE_PRESSURE,
                     "the flamelet table was generated at " + logging::real(p_table) + " Pa, and " + list +
                         " differs from it by more than 10%: its flamelets' chemistry and transport hold at their own "
                         "pressure. Generate the table at the run's pressure");
}
