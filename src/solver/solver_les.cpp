/**
 * @file solver_les.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Large-eddy simulation parts of the Solver: filter width, eddy
 *        viscosity and the kinetic-energy budget (docs/design/les.md).
 * @version 0.1
 * @date 2026-10-05
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "solver.h"

#include <cmath>

#include <Kokkos_Core.hpp>

#include "input.h"
#include "launch_bounds.h"
#include "mixture_viscous_flux.h"

namespace {

/** @brief The 3x3 velocity gradient g[i][j] = d u_i / d x_j of a cell from grad(i, j) over N_DIM components. */
template <typename Gradient>
KOKKOS_INLINE_FUNCTION void velocity_gradient(const Gradient & grad, double g[3][3]) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) g[i][j] = 0.0;
    }
    FOR_I_DIM {
        for (uint8_t j = 0; j < N_DIM; j++) g[i][j] = static_cast<double>(grad(i, j));
    }
}

/** @brief SGS coefficients of a calorically perfect gas: mu_t and cp mu_t / Pr_t. */
struct EddyViscosityFunctor {
    LES les;
    rtype cp;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W;
    Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients;
    Kokkos::View<rtype *> delta;
    Kokkos::View<rtype *[3]> sgs;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c) const {
        double g[3][3];
        velocity_gradient([&](int i, int j) { return gradients(c, 1 + i, j); }, g);
        const rtype mu_t = W(c, 0) * les.nu_t(g, delta(c));
        sgs(c, 0) = mu_t;
        sgs(c, 1) = cp * mu_t / les.Pr_t;
        sgs(c, 2) = 0.0_r;
    }
};

/**
 * @brief SGS coefficients of a gas mixture, [mu_t, cp mu_t / Pr_t,
 *        mu_t / (Sc_t W)], from the transport gradients; the time-step
 *        diffusivity NU_EFF grows by max(4/3 nu_t, lambda_t / (rho cv),
 *        nu_t / Sc_t), a bound on the growth of the largest diffusivity.
 */
struct MixtureEddyViscosityFunctor {
    LES les;
    Mixture gas;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W;
    ScalarView scalars;
    Kokkos::View<rtype **, Kokkos::LayoutRight> values;
    Kokkos::View<rtype ***, Kokkos::LayoutRight> gradients;
    Kokkos::View<rtype *> delta;
    Kokkos::View<rtype *[3]> coefficients;
    Kokkos::View<rtype *[3]> sgs;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c) const {
        double g[3][3];
        velocity_gradient([&](int i, int j) { return gradients(c, i, j); }, g);
        const double rho = static_cast<double>(W(c, 0));
        const double nu_t = static_cast<double>(les.nu_t(g, delta(c)));
        const ScalarMassFractions y{&scalars(c, 0)};
        const double R = gas.thermo.gas_constant(y);
        const double cp = gas.thermo.cp_mass(static_cast<double>(values(c, N_DIM)), y);
        const double mu_t = rho * nu_t;
        const double lambda_t = cp * mu_t / static_cast<double>(les.Pr_t);
        const double Sc_t = static_cast<double>(les.Sc_t);
        sgs(c, 0) = static_cast<rtype>(mu_t);
        sgs(c, 1) = static_cast<rtype>(lambda_t);
        sgs(c, 2) = static_cast<rtype>(mu_t / Sc_t * R / chemistry::GAS_CONSTANT);
        const double growth = Kokkos::fmax(Kokkos::fmax(4.0 / 3.0 * nu_t, lambda_t / (rho * (cp - R))), nu_t / Sc_t);
        coefficients(c, NU_EFF) += static_cast<rtype>(growth);
    }
};

/** @brief u . R_m - |u|^2 / 2 R_rho per owned cell, R the face fluxes summed over the cell. */
struct KineticEnergyRateFunctor {
    Kokkos::View<uint32_t *> offsets_faces_of_cell;
    Kokkos::View<uint32_t *> faces_of_cell;
    Kokkos::View<int32_t *[2]> cells_of_face;
    Kokkos::View<rtype *[N_CONSERVATIVE]> face_flux;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c, rtype & sum) const {
        rtype R[N_DIM + 1] = {};
        for (uint32_t k = offsets_faces_of_cell(c); k < offsets_faces_of_cell(c + 1); k++) {
            const uint32_t f = faces_of_cell(k);
            const rtype sign = (cells_of_face(f, 0) == static_cast<int32_t>(c)) ? 1.0_r : -1.0_r;
            for (uint8_t i = 0; i <= N_DIM; i++) R[i] += sign * face_flux(f, i);
        }
        rtype u2 = 0.0_r, work = 0.0_r;
        FOR_I_DIM {
            u2 += W(c, 1 + i) * W(c, 1 + i);
            work += W(c, 1 + i) * R[1 + i];
        }
        sum += work - 0.5_r * u2 * R[0];
    }
};

} // namespace

void Solver::init_les() {
    les_on = LES::in_input(input);
    if (!les_on) return;
    les = LES::from_input(input);
    if (axisymmetric) throw InputError("[les] is not available for axisymmetric runs.");
    les_delta = Kokkos::View<rtype *>("les_delta", mesh->n_cells);
    auto h_delta = Kokkos::create_mirror_view(les_delta);
    for (uint32_t c = 0; c < mesh->n_cells; c++) h_delta(c) = std::pow(mesh->h_cell_volume(c), 1.0_r / N_DIM);
    Kokkos::deep_copy(les_delta, h_delta);
    les_coefficients = Kokkos::View<rtype *[3]>("les_coefficients", mesh->n_cells);
    h_les_coefficients = Kokkos::create_mirror_view(les_coefficients);
}

void Solver::update_eddy_viscosity(const uint32_t n) {
    if (is_mixture()) {
        Kokkos::parallel_for("eddy_viscosity", n,
                             MixtureEddyViscosityFunctor{les, mixture, W_cells, cell_scalars, transport_values,
                                                         transport_gradients, les_delta, cell_transport,
                                                         les_coefficients});
    } else {
        Kokkos::parallel_for("eddy_viscosity", n,
                             EddyViscosityFunctor{les, physics.cp, W_cells, viscous_gradients, les_delta, les_coefficients});
    }
}

void Solver::eddy_viscosity_of_state(const uint32_t n) {
    update_boundary_states(t);
    if (is_mixture()) {
        update_cell_states(state(), false);
        Kokkos::parallel_for("mixture_transport", mesh->n_cells,
                             MixtureTransportFunctor{mixture, W_cells, cell_scalars, cell_transport, cell_diffusion,
                                                     transport_values});
        Kokkos::parallel_for("mixture_gradients", n,
                             MixtureGradientFunctor{viscous_gradient, transport_values, transport_gradients,
                                                    bc_transport_values, mixture.n_species});
    } else {
        const Euler phys = physics;
        StateView U = conservatives;
        Kokkos::View<rtype *[N_CONSERVATIVE]> W = W_cells;
        Kokkos::parallel_for("les_W", mesh->n_cells, KOKKOS_LAMBDA(const uint32_t c) {
            rtype cons[N_CONSERVATIVE], W_c[N_CONSERVATIVE];
            FOR_I_CONSERVATIVE cons[i] = U(c, i);
            phys.compute_W_from_conservatives(W_c, cons);
            FOR_I_CONSERVATIVE W(c, i) = W_c[i];
        });
        Kokkos::parallel_for("les_gradients", HeavyRange<>(0, n), viscous_gradient);
    }
    update_eddy_viscosity(n);
}

rtype Solver::kinetic_energy_rate() const {
    rtype sum = 0.0_r;
    Kokkos::parallel_reduce("kinetic_energy_rate", mesh->n_owned(),
                            KineticEnergyRateFunctor{mesh->offsets_faces_of_cell, mesh->faces_of_cell,
                                                     mesh->cells_of_face, face_flux, W_cells},
                            sum);
    return sum;
}

KineticEnergyBudget Solver::kinetic_energy_budget() {
    State scratch("budget_rhs", mesh->n_cells, static_cast<uint32_t>(species_names.size()));
    budget = KineticEnergyBudget{};
    budget_pass = true;
    calc_rhs(state(), scratch, t);
    budget_pass = false;
    const std::array<rtype, 3> local = {budget.convective, budget.viscous, budget.sgs};
    const std::array<rtype, 3> total = comm::allreduce(local, comm::Op::SUM);
    return KineticEnergyBudget{total[0], total[1], total[2]};
}
