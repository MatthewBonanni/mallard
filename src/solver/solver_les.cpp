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
#include "tfles.h"

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

/** @brief Vorticity of each cell from the mixture transport gradients (2D: its z component only). */
struct VorticityFunctor {
    Kokkos::View<rtype ***, Kokkos::LayoutRight> gradients;  // (cell, [u, T, X], dimension)
    Kokkos::View<rtype *[3]> vorticity;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c) const {
        if constexpr (N_DIM == 2) {
            vorticity(c, 0) = vorticity(c, 1) = 0.0_r;
            vorticity(c, 2) = gradients(c, 1, 0) - gradients(c, 0, 1);
        } else {
            for (uint8_t k = 0; k < 3; k++) {
                const uint8_t a = (k + 1) % 3, b = (k + 2) % 3;
                vorticity(c, k) = gradients(c, b, a) - gradients(c, a, b);
            }
        }
    }
};

/** @brief Least-squares gradient of the vorticity, on the stencil of the viscous gradients (zero-gradient ghosts). */
struct VorticityGradientFunctor {
    LSQVertexGradientFunctor stencil;
    Kokkos::View<rtype *[3]> vorticity;
    Kokkos::View<rtype *[3][N_DIM]> gradients;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c) const {
        for (uint8_t v = 0; v < 3; v++) {
            rtype g[N_DIM] = {};
            for (uint32_t k = stencil.offsets_cells_of_cell(c); k < stencil.offsets_cells_of_cell(c + 1); k++) {
                const rtype dq = vorticity(stencil.cells_of_cell(k), v) - vorticity(c, v);
                FOR_I_DIM g[i] += stencil.weights.cells(k, i) * dq;
            }
            FOR_I_DIM gradients(c, v, i) = g[i];
        }
    }
};

/**
 * @brief Thickened flame fields of an owned cell: [F, E, Omega] and the
 *        chemistry time scale E / F. The SGS velocity is Colin et al.'s
 *        u' = 2 Delta^3 |lap(curl u)|, with the Laplacian of the vorticity the
 *        divergence of its face gradients (cell means corrected along the
 *        line of centroids, as the viscous fluxes); boundary faces carry none.
 */
struct ThickenedFlameFunctor {
    ThickenedFlame tf;
    Kokkos::View<uint32_t *> offsets_faces_of_cell;
    Kokkos::View<uint32_t *> faces_of_cell;
    Kokkos::View<int32_t *[2]> cells_of_face;
    Kokkos::View<rtype *[N_DIM]> normals;
    Kokkos::View<rtype *> face_area;
    Kokkos::View<rtype *> cell_volume;
    Kokkos::View<rtype *[N_DIM]> cell_coords;
    Kokkos::View<rtype *[N_DIM]> shifts;
    Kokkos::View<uint8_t *> face_shift;
    Kokkos::View<rtype *[3]> vorticity;
    Kokkos::View<rtype *[3][N_DIM]> gradients;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W;
    Kokkos::View<rtype **, Kokkos::LayoutRight> values;  // (cell, [u, T, X])
    Kokkos::View<rtype *[3]> transport;
    Kokkos::View<rtype *> delta;
    Kokkos::View<rtype *[3]> fields;
    Kokkos::View<rtype *> time_scale;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c) const {
        double lap[3] = {};
        for (uint32_t k = offsets_faces_of_cell(c); k < offsets_faces_of_cell(c + 1); k++) {
            const uint32_t f = faces_of_cell(k);
            const int32_t c0 = cells_of_face(f, 0), c1 = cells_of_face(f, 1);
            if (c1 < 0) continue;
            rtype n[N_DIM], n_vec[N_DIM], d[N_DIM];
            FOR_I_DIM n_vec[i] = normals(f, i);
            unit<N_DIM>(n_vec, n);
            FOR_I_DIM d[i] = (cell_coords(c1, i) + shifts(face_shift(f), i)) - cell_coords(c0, i);
            const rtype d_n = dot<N_DIM>(d, n);
            const rtype sign = c0 == static_cast<int32_t>(c) ? 1.0_r : -1.0_r;
            for (uint8_t v = 0; v < 3; v++) {
                rtype g[N_DIM];
                FOR_I_DIM g[i] = 0.5_r * (gradients(c0, v, i) + gradients(c1, v, i));
                const rtype dq = vorticity(c1, v) - vorticity(c0, v);
                const rtype g_n = dot<N_DIM>(g, n) + (dq - dot<N_DIM>(g, d)) / d_n;
                lap[v] += static_cast<double>(sign * face_area(f) * g_n);
            }
        }
        const double V = static_cast<double>(cell_volume(c)), D = static_cast<double>(delta(c));
        const double lap_norm = Kokkos::sqrt(lap[0] * lap[0] + lap[1] * lap[1] + lap[2] * lap[2]) / V;
        const double u_prime = 2.0 * D * D * D * lap_norm;
        const double omega = tf.sensor(static_cast<double>(values(c, N_DIM)));
        const double F = tf.thickening(D, omega);
        const double nu = static_cast<double>(transport(c, MU)) / static_cast<double>(W(c, 0));
        const double E = tf.wrinkling(F, u_prime, nu);
        fields(c, 0) = static_cast<rtype>(F);
        fields(c, 1) = static_cast<rtype>(E);
        fields(c, 2) = static_cast<rtype>(omega);
        time_scale(c) = static_cast<rtype>(E / F);
    }
};

/**
 * @brief Thickened transport of each cell: conductivity and species
 *        diffusivities times E F; the SGS heat and species fluxes times
 *        1 - Omega (the thickened flame is resolved); the time step's
 *        diffusivity times E F.
 */
struct ThickenFunctor {
    Kokkos::View<rtype *[3]> fields;
    Kokkos::View<rtype *[3]> coefficients;
    Kokkos::View<double **, Kokkos::LayoutRight> diffusion;
    Kokkos::View<rtype *[3]> sgs;
    uint32_t n_species;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c) const {
        const rtype EF = fields(c, 0) * fields(c, 1), resolved = 1.0_r - fields(c, 2);
        coefficients(c, LAMBDA) *= EF;
        coefficients(c, NU_EFF) *= EF;
        for (uint32_t k = 0; k < n_species; k++) diffusion(c, k) *= static_cast<double>(EF);
        if (sgs.extent(0) > 0) {
            sgs(c, 1) *= resolved;
            sgs(c, 2) *= resolved;
        }
    }
};

} // namespace

void Solver::update_thickened_flame() {
    // calc_dt has filled the halo of the state
    eddy_viscosity_of_state(mesh->n_cells);
    Kokkos::parallel_for("tfles_vorticity", mesh->n_cells, VorticityFunctor{transport_gradients, tfles_vorticity});
    // The vorticity of outer halo cells misses neighbors: take the owners' before differentiating it again
    exchange_cell_vectors(tfles_vorticity);
    Kokkos::parallel_for("tfles_vorticity_gradient", mesh->n_cells,
                         VorticityGradientFunctor{viscous_gradient, tfles_vorticity, tfles_gradients});
    Kokkos::parallel_for("tfles_fields", mesh->n_owned(),
                         ThickenedFlameFunctor{thickened_flame, mesh->offsets_faces_of_cell, mesh->faces_of_cell,
                                               mesh->cells_of_face, mesh->face_normals, mesh->face_area,
                                               mesh->cell_volume, mesh->cell_coords, mesh->shifts, mesh->face_shift,
                                               tfles_vorticity, tfles_gradients, W_cells, transport_values,
                                               cell_transport, les_delta, tfles_fields, chem_time_scale});
    // Halo cells take their owners' fields: the face diffusivities need them
    exchange_cell_vectors(tfles_fields);
}

void Solver::update_partially_stirred_reactor() {
    const uint32_t n_owned = mesh->n_owned();
    cell_chemistry.heat_release(conservatives, species, T_seed, hrr, production, n_owned);
    const PartiallyStirredReactor model = pasr;
    const Mixture gas = mixture;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W = W_cells;
    ScalarView scalars = cell_scalars;
    Kokkos::View<rtype **, Kokkos::LayoutRight> values = transport_values;
    Kokkos::View<rtype *[3]> transport = cell_transport, sgs = les_coefficients;
    Kokkos::View<rtype *> q = hrr, delta = les_delta, kappa = chem_time_scale;
    Kokkos::parallel_for("pasr_fraction", n_owned, KOKKOS_LAMBDA(const uint32_t c) {
        const double rho = static_cast<double>(W(c, 0));
        const double T = static_cast<double>(values(c, N_DIM));
        const ScalarMassFractions y{&scalars(c, 0)};
        const double rho_cp_T = rho * gas.thermo.cp_mass(T, y) * T;
        kappa(c) = static_cast<rtype>(model.fraction(rho_cp_T, static_cast<double>(q(c)), static_cast<double>(delta(c)),
                                                     static_cast<double>(transport(c, 0)) / rho,
                                                     static_cast<double>(sgs(c, 0)) / rho));
    });
}

void Solver::exchange_cell_vectors(const Kokkos::View<rtype *[3]> & v) {
    if (!halo.active()) return;
    SpeciesView packed = tfles_halo.species;
    Kokkos::parallel_for("halo_pack", mesh->n_cells, KOKKOS_LAMBDA(const uint32_t c) {
        for (int k = 0; k < 3; k++) packed(c, k) = v(c, k);
    });
    halo.exchange(tfles_halo);
    Kokkos::parallel_for("halo_unpack", mesh->n_cells, KOKKOS_LAMBDA(const uint32_t c) {
        for (int k = 0; k < 3; k++) v(c, k) = packed(c, k);
    });
}

void Solver::thicken_transport() {
    Kokkos::parallel_for("tfles_thicken", mesh->n_cells,
                         ThickenFunctor{tfles_fields, cell_transport, cell_diffusion, les_coefficients,
                                        mixture.n_species});
}

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
    if (!input.at("les").contains("combustion")) return;
    if (!reacting || !is_viscous()) {
        throw InputError("[les.combustion] needs a reacting viscous mixture (gas = \"mixture\", navier_stokes, [chemistry]).");
    }
    const toml::value & combustion = input.at("les").at("combustion");
    if (toml::find_or<std::string>(combustion, "model", "") == "pasr") {
        pasr = PartiallyStirredReactor::from_input(combustion);
        pasr_on = true;
        chem_time_scale = Kokkos::View<rtype *>("chem_time_scale", mesh->n_cells);
        h_chem_time_scale = Kokkos::create_mirror_view(chem_time_scale);
        Kokkos::deep_copy(chem_time_scale, 1.0_r);
        return;
    }
    thickened_flame = ThickenedFlame::from_input(input.at("les").at("combustion"));
    tfles_on = true;
    tfles_fields = Kokkos::View<rtype *[3]>("tfles_fields", mesh->n_cells);
    h_tfles_fields = Kokkos::create_mirror_view(tfles_fields);
    chem_time_scale = Kokkos::View<rtype *>("chem_time_scale", mesh->n_cells);
    tfles_vorticity = Kokkos::View<rtype *[3]>("tfles_vorticity", mesh->n_cells);
    tfles_gradients = Kokkos::View<rtype *[3][N_DIM]>("tfles_gradients", mesh->n_cells);
    tfles_halo = State("tfles_halo", mesh->n_cells, 3);
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
