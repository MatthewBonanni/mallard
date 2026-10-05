/**
 * @file solver_mixture.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Gas-mixture parts of the Solver: cell states, RHS, time step,
 *        primitives and boundaries.
 * @version 0.3
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "solver.h"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>

#include <Kokkos_Core.hpp>

#include "flux_functor.h"
#include "input.h"
#include "launch_bounds.h"
#include "mixture_flux.h"
#include "mixture_viscous_flux.h"

namespace {

/** @brief Applies a per-face functor to the entries of a list. */
template <typename F>
struct OverFaces {
    F f;
    Kokkos::View<uint32_t *> faces;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t k) const { f(faces(k)); }
};

/** @brief W, the scalars [Y, gamma, e0] and (optionally) the temperature seed of each cell. */
struct MixtureCellFunctor {
    Mixture gas;
    StateView U;
    SpeciesView rhoY;
    Kokkos::View<rtype *> T_seed;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W;
    ScalarView scalars;
    Kokkos::View<rtype *> molar_mass;  // empty unless TENO needs it
    Kokkos::View<rtype *[2]> frozen;   // double flux within a step: [gamma, e0] in place of the EOS
    bool update_seed;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c) const {
        rtype cons[N_CONSERVATIVE], W_c[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE cons[i] = U(c, i);
        const CellSpecies rhoY_c = cell_species(rhoY, c);
        rtype gamma, e0, T = 0.0_r;
        if (frozen.extent(0) > 0) {
            gamma = frozen(c, 0);
            e0 = frozen(c, 1);
            W_c[0] = cons[0];
            FOR_I_DIM W_c[1 + i] = cons[1 + i] / cons[0];
            W_c[N_DIM + 1] = (gamma - 1.0_r) * (cons[N_DIM + 1] - cons[0] * e0 -
                                                0.5_r * cons[0] * dot<N_DIM>(W_c + 1, W_c + 1));
        } else {
            gas.cell_state(cons, rhoY_c, T_seed(c), W_c, gamma, e0, T);
        }
        FOR_I_CONSERVATIVE W(c, i) = W_c[i];
        if (molar_mass.extent(0) > 0) {
            double n = 0.0;
            for (uint32_t k = 0; k < gas.n_species; k++) n += static_cast<double>(rhoY_c[k]) * gas.thermo.inv_W(k);
            molar_mass(c) = static_cast<rtype>(static_cast<double>(cons[0]) / n);
        }
        const rtype inv_rho = 1.0_r / cons[0];
        for (uint32_t k = 0; k < gas.n_species; k++) scalars(c, k) = rhoY_c[k] * inv_rho;
        scalars(c, gas.n_species) = gamma;
        scalars(c, gas.n_species + 1) = e0;
        if (update_seed && frozen.extent(0) == 0) T_seed(c) = T;
    }
};

/** @brief Double flux: each cell's [gamma, e0] from the true EOS at the start of a step. */
struct FreezeFunctor {
    Mixture gas;
    StateView U;
    SpeciesView rhoY;
    Kokkos::View<rtype *> T_seed;
    Kokkos::View<rtype *[2]> frozen;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c) const {
        rtype cons[N_CONSERVATIVE], W[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE cons[i] = U(c, i);
        rtype gamma, e0, T;
        gas.cell_state(cons, cell_species(rhoY, c), T_seed(c), W, gamma, e0, T);
        T_seed(c) = T;
        frozen(c, 0) = gamma;
        frozen(c, 1) = e0;
    }
};

/**
 * @brief Double flux: rho E from the true EOS at the pressure that the
 *        frozen [gamma, e0] give, keeping rho, rho u and the composition.
 */
struct ResetEnergyFunctor {
    Mixture gas;
    StateView U;
    SpeciesView rhoY;
    Kokkos::View<rtype *[2]> frozen;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c) const {
        constexpr uint8_t E = N_DIM + 1;
        const double rho = static_cast<double>(U(c, 0));
        double rhou2 = 0.0;
        FOR_I_DIM rhou2 += static_cast<double>(U(c, 1 + i)) * static_cast<double>(U(c, 1 + i));
        const double kinetic = 0.5 * rhou2 / rho;
        const double p = (static_cast<double>(frozen(c, 0)) - 1.0) *
                         (static_cast<double>(U(c, E)) - rho * static_cast<double>(frozen(c, 1)) - kinetic);
        const PartialDensities y{cell_species(rhoY, c), 1.0 / rho};
        const double T = p / (rho * gas.thermo.gas_constant(y));
        double e, cv;
        gas.thermo.e_cv(T, y, e, cv);
        U(c, E) = static_cast<rtype>(rho * e + kinetic);
    }
};

/** @brief Per-cell stable time step for CFL = 1 of a mixture, as TimeStepFunctor. */
struct MixtureTimeStepFunctor {
    Kokkos::View<uint32_t *> offsets_faces_of_cell;
    Kokkos::View<uint32_t *> faces_of_cell;
    Kokkos::View<int32_t *[2]> cells_of_face;
    Kokkos::View<rtype *[N_DIM]> face_normals;
    Kokkos::View<rtype *> face_area;
    Kokkos::View<rtype *> cell_volume;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W;
    ScalarView scalars;
    Kokkos::View<rtype *> dt_local;
    uint32_t n_species;
    Kokkos::View<rtype *[3]> transport;  // viscous: (cell, [mu, lambda, nu_eff]), else empty

    KOKKOS_INLINE_FUNCTION
    rtype wave_speed(const int32_t c, const rtype * n) const {
        rtype u[N_DIM];
        FOR_I_DIM u[i] = W(c, 1 + i);
        return Kokkos::fabs(dot<N_DIM>(u, n)) + Kokkos::sqrt(scalars(c, n_species) * W(c, N_DIM + 1) / W(c, 0));
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c, rtype & dt_min) const {
        rtype sum = 0.0_r, sum_area2 = 0.0_r;
        for (uint32_t k = offsets_faces_of_cell(c); k < offsets_faces_of_cell(c + 1); k++) {
            const uint32_t f = faces_of_cell(k);
            rtype n_vec[N_DIM], n[N_DIM];
            FOR_I_DIM n_vec[i] = face_normals(f, i);
            unit<N_DIM>(n_vec, n);
            rtype lambda = wave_speed(cells_of_face(f, 0), n);
            if (cells_of_face(f, 1) >= 0) lambda = Kokkos::fmax(lambda, wave_speed(cells_of_face(f, 1), n));
            sum += lambda * face_area(f);
            sum_area2 += face_area(f) * face_area(f);
        }
        // As TimeStepFunctor, with nu_eff = max(4/3 mu / rho, lambda / (rho cv), max_k D_k)
        if (transport.extent(0) > 0) sum += 4.0_r * transport(c, NU_EFF) * sum_area2 / cell_volume(c);
        const rtype dt_c = cell_volume(c) / sum;
        dt_local(c) = dt_c;
        dt_min = Kokkos::fmin(dt_min, dt_c);
    }
};

/** @brief Primitives [u, p, T, h] of a mixture cell. */
struct MixturePrimitivesFunctor {
    Mixture gas;
    StateView U;
    SpeciesView rhoY;
    Kokkos::View<rtype *> T_seed;
    Kokkos::View<rtype *[N_PRIMITIVE]> P;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c) const {
        rtype cons[N_CONSERVATIVE], W[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE cons[i] = U(c, i);
        rtype gamma, e0, T;
        gas.cell_state(cons, cell_species(rhoY, c), T_seed(c), W, gamma, e0, T);
        FOR_I_DIM P(c, i) = W[1 + i];
        P(c, N_DIM) = W[N_DIM + 1];
        P(c, N_DIM + 1) = T;
        // h = e + p / rho, with e = p / (rho (gamma - 1)) + e0
        P(c, N_DIM + 2) = W[N_DIM + 1] / W[0] * gamma / (gamma - 1.0_r) + e0;
    }
};

/** @brief Temperature of each cell from its energy, without a seed. */
struct TemperatureFunctor {
    Mixture gas;
    StateView U;
    SpeciesView rhoY;
    Kokkos::View<rtype *> T_seed;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c) const {
        rtype cons[N_CONSERVATIVE], W[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE cons[i] = U(c, i);
        rtype gamma, e0, T;
        gas.cell_state(cons, cell_species(rhoY, c), 0.0_r, W, gamma, e0, T);
        T_seed(c) = T;
    }
};

/** @brief Partial density times volume, for the species integrals. */
struct SpeciesIntegralFunctor {
    SpeciesView rhoY;
    Kokkos::View<rtype *> volume;
    uint32_t k;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c, rtype & sum) const { sum += rhoY(c, k) * volume(c); }
};

/** @brief Minima of rho, p, -Ma, T, -T and -|sum Y - 1| over cells. */
struct MixtureDiagnosticsFunctor {
    Mixture gas;
    StateView U;
    SpeciesView rhoY;
    Kokkos::View<rtype *> T_seed;

    struct value_type {
        rtype v[6];
    };

    KOKKOS_INLINE_FUNCTION
    void init(value_type & m) const {
        for (int i = 0; i < 6; i++) m.v[i] = Kokkos::Experimental::finite_max_v<rtype>;
    }

    KOKKOS_INLINE_FUNCTION
    void join(value_type & dst, const value_type & src) const {
        for (int i = 0; i < 6; i++) dst.v[i] = Kokkos::fmin(dst.v[i], src.v[i]);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c, value_type & m) const {
        rtype cons[N_CONSERVATIVE], W[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE cons[i] = U(c, i);
        rtype gamma, e0, T;
        gas.cell_state(cons, cell_species(rhoY, c), T_seed(c), W, gamma, e0, T);
        rtype u2 = 0.0_r, sum_Y = 0.0_r;
        FOR_I_DIM u2 += W[1 + i] * W[1 + i];
        for (uint32_t k = 0; k < gas.n_species; k++) sum_Y += rhoY(c, k);
        const rtype values[6] = {W[0], W[N_DIM + 1], -Kokkos::sqrt(u2 * W[0] / (gamma * W[N_DIM + 1])), T, -T,
                                 -Kokkos::fabs(sum_Y / W[0] - 1.0_r)};
        for (int i = 0; i < 6; i++) m.v[i] = Kokkos::fmin(m.v[i], values[i]);
    }
};

/**
 * @brief A composition given as X or Y by species, each value a number or an
 *        expression in x, y, z, with an optional balance species taking
 *        1 - sum of the others; without one the values are normalized.
 *        Negative values count as zero.
 */
class CompositionExpressions {
    public:
        CompositionExpressions(const MixtureModel & mixture_model, const toml::value & table, const std::string & name) :
            mixture(mixture_model), where(name) {
            if (table.contains("X") == table.contains("Y")) {
                throw InputError(where + ": give the composition as exactly one of X and Y.");
            }
            mole = table.contains("X");
            const std::string key = mole ? "X" : "Y";
            if (!table.at(key).is_table()) throw InputError(where + "." + key + " must be a table of species.");
            for (const auto & [species, value] : table.at(key).as_table()) {
                const int32_t k = mixture.mechanism().species_index(species);
                if (k < 0) throw InputError(where + "." + key + ": no species " + species + " in the mechanism.");
                listed.push_back(k);
                std::ostringstream text;
                if (value.is_string()) {
                    text << value.as_string();
                } else {
                    text << std::setprecision(17) << static_cast<double>(as_real(value, species));
                }
                values.emplace_back(where + "." + key + "." + species, text.str());
            }
            if (table.contains("balance")) {
                const std::string species = toml::find<std::string>(table, "balance");
                balance = mixture.mechanism().species_index(species);
                if (balance < 0) throw InputError(where + ".balance: no species " + species + " in the mechanism.");
            }
        }

        /** @brief Mass fractions at the point with coordinates x[0..N_DIM). */
        template <typename T>
        std::vector<double> mass_fractions(const T & x) const {
            std::vector<double> f(mixture.n_species(), 0.0);
            double sum = 0.0;
            for (size_t i = 0; i < listed.size(); i++) {
                f[listed[i]] = std::max(values[i].at(x, N_DIM), 0.0);
                if (listed[i] != balance) sum += f[listed[i]];
            }
            if (balance >= 0) {
                f[balance] = std::max(1.0 - sum, 0.0);
                sum += f[balance];
            }
            if (!(sum > 0.0)) throw InputError(where + ": the composition is zero at a face.");
            for (double & v : f) v /= sum;
            return mole ? mixture.mass_fractions_from_mole(f) : f;
        }

    private:
        const MixtureModel & mixture;
        std::string where;
        bool mole = false;
        int32_t balance = -1;
        std::vector<int32_t> listed;
        std::vector<Expression> values;
};

} // namespace

std::array<rtype, 6> Solver::mixture_diagnostics() {
    MixtureDiagnosticsFunctor::value_type result;
    Kokkos::parallel_reduce("mixture_diagnostics", mesh->n_owned(),
                            MixtureDiagnosticsFunctor{mixture, conservatives, species, T_seed}, result);
    std::array<rtype, 6> m;
    for (int i = 0; i < 6; i++) m[i] = result.v[i];
    return comm::allreduce(m, comm::Op::MIN);
}

void Solver::update_cell_states(const State & solution, const bool update_seed) {
    MixtureCellFunctor functor{mixture, solution.flow, solution.species, T_seed, W_cells, cell_scalars,
                               cell_molar_mass, cells_frozen ? frozen_thermo : Kokkos::View<rtype *[2]>(),
                               update_seed};
    Kokkos::parallel_for("mixture_cells", mesh->n_cells, functor);
}

void Solver::update_transport() {
    Kokkos::parallel_for("mixture_transport", mesh->n_cells,
                         MixtureTransportFunctor{mixture, W_cells, cell_scalars, cell_transport, cell_diffusion,
                                                 transport_values});
    Kokkos::parallel_for("mixture_gradients", mesh->n_cells,
                         MixtureGradientFunctor{viscous_gradient, transport_values, transport_gradients,
                                                bc_transport_values, mixture.n_species});
}

void Solver::freeze_thermodynamics() {
    Kokkos::parallel_for("freeze_thermodynamics", mesh->n_cells,
                         FreezeFunctor{mixture, conservatives, species, T_seed, frozen_thermo});
}

void Solver::reset_energy() {
    Kokkos::parallel_for("reset_energy", mesh->n_owned(), ResetEnergyFunctor{mixture, conservatives, species, frozen_thermo});
}

void Solver::init_temperature_seed() {
    Kokkos::parallel_for("temperature_seed", mesh->n_cells, TemperatureFunctor{mixture, conservatives, species, T_seed});
}

void Solver::calc_rhs_mixture(State state, State rhs_state, rtype t_stage) {
    const bool exchange = halo.active() && !(halo_current && state.flow.data() == conservatives.data());
    halo_current = false;
    update_boundary_states(t_stage);
    if (exchange) halo.exchange(state);
    update_cell_states(state, true);
    face_reconstruction->calc_face_values(W_cells, face_solution);
    scalar_reconstruction.calc(cell_scalars, W_cells, face_thermo);

    if (double_flux && !cells_frozen) {
        throw std::logic_error("Double flux needs the cells' thermodynamics frozen (take_step).");
    }
    switch (riemann_solver_type) {
        case RiemannSolverType::RUSANOV:
            double_flux ? launch_double_flux_functor<riemann::Rusanov>() : launch_mixture_flux_functor<riemann::Rusanov>();
            break;
        case RiemannSolverType::HLL:
            double_flux ? launch_double_flux_functor<riemann::HLL>() : launch_mixture_flux_functor<riemann::HLL>();
            break;
        case RiemannSolverType::HLLC:
            double_flux ? launch_double_flux_functor<riemann::HLLC>() : launch_mixture_flux_functor<riemann::HLLC>();
            break;
        default:
            throw std::logic_error("Riemann solver without a mixture flux.");
    }

    const uint32_t n_species = mixture.n_species;
    scalar_reconstruction.species_slots(cell_scalars, face_mdot, face_reconstruction->quadrature_face.weights,
                                        face_reconstruction->face_quad_weights, species_slots);
    if (is_viscous()) {
        update_transport();
        MixtureViscousFluxFunctor viscous_functor{mesh->face_normals, mesh->face_area, mesh->face_coords,
                                                  mesh->cell_coords, mesh->cells_of_face, mesh->shifts,
                                                  mesh->face_shift, boundary_data, mixture, cell_scalars,
                                                  transport_values, transport_gradients, cell_transport,
                                                  cell_diffusion, face_flux, species_slots};
        if (rhs_faces.extent(0) == 0) {
            Kokkos::parallel_for("mixture_viscous_flux", mesh->n_faces, viscous_functor);
        } else {
            Kokkos::View<uint32_t *> list = rhs_faces;
            Kokkos::parallel_for("mixture_viscous_flux", list.extent(0),
                                 OverFaces<MixtureViscousFluxFunctor>{viscous_functor, list});
        }
    }

    const uint32_t n_owned = mesh->n_owned();
    StateView rhs = rhs_state.flow;
    if (double_flux) {
        DoubleFluxSumFunctor sum_functor{mesh->offsets_faces_of_cell, mesh->faces_of_cell, mesh->cells_of_face,
                                         face_flux, face_energy_1, rhs};
        Kokkos::parallel_for("face_flux_sum", n_owned, sum_functor);
    } else {
        FaceFluxSumFunctor sum_functor{mesh->offsets_faces_of_cell, mesh->faces_of_cell, mesh->cells_of_face,
                                       face_flux, rhs};
        Kokkos::parallel_for("face_flux_sum", n_owned, sum_functor);
    }
    SpeciesSumFunctor species_sum{mesh->offsets_faces_of_cell, mesh->faces_of_cell, mesh->cells_of_face,
                                  mesh->cell_volume, species_slots, rhs_state.species, n_species};
    Kokkos::parallel_for("species_sum", n_owned, species_sum);

    StateView solution = state.flow;
    Kokkos::View<rtype *> vol = mesh->cell_volume;
    if (has_gravity || !source_expressions.empty()) {
        update_source_field(t_stage);
        const bool gravity_on = has_gravity;
        const bool field_on = !source_expressions.empty();
        Kokkos::Array<rtype, N_DIM> g;
        FOR_I_DIM g[i] = gravity[i];
        StateView S = source_field;
        Kokkos::parallel_for("rhs_sources", n_owned, KOKKOS_LAMBDA(const uint32_t i_cell) {
            const rtype V = vol(i_cell);
            if (gravity_on) {
                rtype rhou[N_DIM];
                FOR_I_DIM {
                    rhs(i_cell, 1 + i) += solution(i_cell, 0) * g[i] * V;
                    rhou[i] = solution(i_cell, 1 + i);
                }
                rhs(i_cell, N_DIM + 1) += dot<N_DIM>(rhou, g.data()) * V;
            }
            if (field_on) {
                FOR_I_CONSERVATIVE rhs(i_cell, i) += S(i_cell, i) * V;
            }
        });
    }
    Kokkos::parallel_for("rhs_divide_volume", n_owned, KOKKOS_LAMBDA(const uint32_t i_cell) {
        FOR_I_CONSERVATIVE rhs(i_cell, i) /= vol(i_cell);
    });
}

template <typename T_riemann_solver>
void Solver::launch_mixture_flux_functor() {
    const uint32_t n_species = mixture.n_species;
    MixtureFluxFunctor<T_riemann_solver> functor{
        mesh->face_normals,
        mesh->face_area,
        mesh->cells_of_face,
        face_reconstruction->quadrature_face.weights,
        face_reconstruction->face_quad_weights,
        face_solution,
        face_thermo,
        boundary_data,
        W_cells,
        Kokkos::subview(cell_scalars, Kokkos::ALL(), Kokkos::make_pair(n_species, n_species + 2)),
        face_flux,
        face_mdot,
        low_mach_cutoff};
    // Faces of owned cells, as the single-gas flux
    if (rhs_faces.extent(0) == 0) {
        Kokkos::parallel_for("mixture_flux", HeavyRange<>(0, mesh->n_faces), functor);
    } else {
        Kokkos::View<uint32_t *> list = rhs_faces;
        Kokkos::parallel_for("mixture_flux", HeavyRange<>(0, list.extent(0)), OverFaces<MixtureFluxFunctor<T_riemann_solver>>{functor, list});
    }
}

template <typename T_riemann_solver>
void Solver::launch_double_flux_functor() {
    const uint32_t n_species = mixture.n_species;
    MixtureDoubleFluxFunctor<T_riemann_solver> functor{
        mesh->face_normals,
        mesh->face_area,
        mesh->cells_of_face,
        face_reconstruction->quadrature_face.weights,
        face_reconstruction->face_quad_weights,
        face_solution,
        boundary_data,
        W_cells,
        frozen_thermo,
        face_thermo,
        Kokkos::subview(cell_scalars, Kokkos::ALL(), Kokkos::make_pair(n_species, n_species + 2)),
        face_flux,
        face_energy_1,
        face_mdot,
        low_mach_cutoff};
    if (rhs_faces.extent(0) == 0) {
        Kokkos::parallel_for("double_flux", HeavyRange<>(0, mesh->n_faces), functor);
    } else {
        Kokkos::View<uint32_t *> list = rhs_faces;
        Kokkos::parallel_for("double_flux", HeavyRange<>(0, list.extent(0)),
                             OverFaces<MixtureDoubleFluxFunctor<T_riemann_solver>>{functor, list});
    }
}

rtype Solver::calc_dt_cfl1_mixture() {
    update_cell_states(state(), false);
    if (is_viscous()) {
        Kokkos::parallel_for("mixture_transport", mesh->n_owned(),
                             MixtureTransportFunctor{mixture, W_cells, cell_scalars, cell_transport, cell_diffusion,
                                                     transport_values});
    }
    MixtureTimeStepFunctor functor{mesh->offsets_faces_of_cell, mesh->faces_of_cell, mesh->cells_of_face,
                                   mesh->face_normals, mesh->face_area, mesh->cell_volume, W_cells,
                                   cell_scalars, cfl_local, mixture.n_species, cell_transport};
    rtype dt_min = std::numeric_limits<rtype>::max();
    Kokkos::parallel_reduce("time_step", mesh->n_owned(), functor, Kokkos::Min<rtype>(dt_min));
    return comm::allreduce(dt_min, comm::Op::MIN);
}

void Solver::update_primitives_mixture() {
    Kokkos::parallel_for("update_primitives", mesh->n_cells,
                         MixturePrimitivesFunctor{mixture, conservatives, species, T_seed, primitives});
}

std::vector<rtype> Solver::integrate_species() {
    std::vector<rtype> total(species_names.size());
    for (uint32_t k = 0; k < total.size(); k++) {
        rtype sum = 0.0_r;
        Kokkos::parallel_reduce("integrate_species", mesh->n_owned(),
                                SpeciesIntegralFunctor{species, mesh->cell_volume, k}, sum);
        total[k] = sum;
    }
    if (total.empty()) return total;
    comm::allreduce(std::span<rtype>(total), comm::Op::SUM);
    return total;
}

void Solver::init_mixture_boundaries(const std::vector<toml::value> & input_boundaries,
                                     const std::vector<std::array<uint32_t, 2>> & profiled_faces,
                                     std::vector<BoundaryCondition> & bcs) {
    bc_mass_fractions.assign(bcs.size(), {});
    bc_surrogates.assign(bcs.size(), {1.4, 0.0});
    bc_temperatures.assign(bcs.size(), 0.0);
    // Prescribed state of condition i_bc
    auto prescribe = [&](size_t i_bc, const std::vector<double> & Y, double T) {
        BoundaryCondition & bc = bcs[i_bc];
        const double p = static_cast<double>(bc.data[N_DIM + 1]);
        bc.data[0] = static_cast<rtype>(p / (mixture_model->gas_constant(Y) * T));
        bc_mass_fractions[i_bc] = Y;
        bc_temperatures[i_bc] = T;
        mixture_model->surrogates(T, Y, bc_surrogates[i_bc][0], bc_surrogates[i_bc][1]);
    };
    for (size_t i_bc = 0; i_bc < input_boundaries.size(); i_bc++) {
        const toml::value & bound = input_boundaries[i_bc];
        const std::string name = toml::find<std::string>(bound, "name");
        const std::string where = "boundaries[name = \"" + name + "\"]";
        switch (bcs[i_bc].type) {
            case BoundaryType::EXTRAPOLATION:
            case BoundaryType::SYMMETRY:
            case BoundaryType::WALL_ADIABATIC:
            case BoundaryType::P_OUT:
                break;
            case BoundaryType::UPT: {
                const double T = static_cast<double>(find_real(bound, "T"));
                if (!MixtureModel::composition_varies(bound)) {
                    prescribe(i_bc, mixture_model->mass_fractions(bound, where), T);
                    break;
                }
                // Each face's own copy of the condition, at the composition of its center
                const CompositionExpressions composition(*mixture_model, bound, where);
                for (size_t k = 0; k < profiled_faces.size(); k++) {
                    if (profiled_faces[k][0] != i_bc) continue;
                    const auto center = Kokkos::subview(mesh->h_face_coords, profiled_faces[k][1], Kokkos::ALL());
                    prescribe(input_boundaries.size() + k, composition.mass_fractions(center), T);
                }
                break;
            }
            default:
                throw InputError(where + ": type \"" + BOUNDARY_NAMES.at(bcs[i_bc].type) +
                                 "\" is not yet supported with gas = \"mixture\" (extrapolation, symmetry, "
                                 "wall_adiabatic, upt, p_out).");
        }
    }
}
