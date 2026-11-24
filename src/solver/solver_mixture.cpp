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
#include "gated.h"

#include <algorithm>
#include <limits>

#include <Kokkos_Core.hpp>

#include "flux_functor.h"
#include "input.h"
#include "launch_bounds.h"
#include "mixture_flux.h"
#include "mixture_viscous_flux.h"

namespace {

/** @brief The primitives [u, p, T, h] of cell c from its W, gamma, e0 and T. */
KOKKOS_INLINE_FUNCTION
void set_primitives(const Kokkos::View<rtype *[N_PRIMITIVE]> & P, const uint32_t c, const rtype * W, const rtype gamma,
                    const rtype e0, const rtype T) {
    FOR_I_DIM P(c, i) = W[1 + i];
    P(c, N_DIM) = W[N_DIM + 1];
    P(c, N_DIM + 1) = T;
    // h = e + p / rho, with e = p / (rho (gamma - 1)) + e0
    P(c, N_DIM + 2) = W[N_DIM + 1] / W[0] * gamma / (gamma - 1.0_r) + e0;
}

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
    Kokkos::View<rtype *[N_PRIMITIVE]> primitives;  // also set from the same state, unless empty (never with frozen)

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
            gas.cell_state(cons, rhoY_c, T_seed(c), gas.flamelet.cell_variance(c), W_c, gamma, e0, T);
            if (primitives.extent(0) > 0) set_primitives(primitives, c, W_c, gamma, e0, T);
        }
        FOR_I_CONSERVATIVE W(c, i) = W_c[i];
        if (molar_mass.extent(0) > 0) {
            if (gas.tabulated()) {
                const PartialDensities y{rhoY_c, 1.0 / static_cast<double>(cons[0])};
                molar_mass(c) = static_cast<rtype>(chemistry::GAS_CONSTANT / gas.gas_constant(y, gas.flamelet.cell_variance(c)));
            } else {
                double n = 0.0;
                for (uint32_t k = 0; k < gas.n_species; k++) n += static_cast<double>(rhoY_c[k]) * gas.thermo.inv_W(k);
                molar_mass(c) = static_cast<rtype>(static_cast<double>(cons[0]) / n);
            }
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
        gas.cell_state(cons, cell_species(rhoY, c), T_seed(c), gas.flamelet.cell_variance(c), W, gamma, e0, T);
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
    Kokkos::View<rtype *> volume;      // repair audit: the change of rho E V and the changed cells, else empty
    Kokkos::View<uint8_t *> touched;

    // One array reduction, [change, changed cells], into a device view: combined reducers wait for the host
    using value_type = double[];
    const unsigned value_count = 2;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c, double sums[]) const {
        const rtype before = U(c, N_DIM + 1);
        (*this)(c);
        const rtype after = U(c, N_DIM + 1);
        if (after != before) {
            sums[0] += static_cast<double>(volume(c)) * (static_cast<double>(after) - static_cast<double>(before));
            sums[1] += 1.0;
            touched(c) = 3;
        }
    }

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
        const double Zvar = gas.flamelet.cell_variance(c);
        const double T = p / (rho * gas.gas_constant(y, Zvar));
        double e, cv;
        gas.e_cv(T, y, e, cv, Zvar);
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
    Kokkos::View<rtype *[N_DIM]> radius_coords;  // axisymmetric runs: cell_coords, else empty
    BoundaryData boundaries;
    PeriodicMaps periodic;
    Kokkos::View<uint8_t *> face_shift;

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
            const int32_t c0 = cells_of_face(f, 0);
            rtype lambda = wave_speed(c0, n);
            if (cells_of_face(f, 1) >= 0) {
                // The normal in cell 1's frame
                if (periodic.rotated()) periodic.unrotate(face_shift(f), n);
                lambda = Kokkos::fmax(lambda, wave_speed(cells_of_face(f, 1), n));
            } else {
                rtype W_c[N_CONSERVATIVE];
                FOR_I_CONSERVATIVE W_c[i] = W(c0, i);
                lambda = Kokkos::fmax(lambda, boundaries.exterior_wave_speed(f, W_c, n, scalars(c0, n_species)));
            }
            sum += lambda * face_area(f);
            sum_area2 += face_area(f) * face_area(f);
        }
        // As TimeStepFunctor, with nu_eff = max(4/3 mu / rho, lambda / (rho cv), max_k D_k)
        if (transport.extent(0) > 0) {
            sum += 4.0_r * transport(c, NU_EFF) * sum_area2 / cell_volume(c);
            // Decay of u_r by the hoop stress
            if (radius_coords.extent(0) > 0) {
                const rtype r = radius_coords(c, 1);
                sum += transport(c, NU_EFF) * cell_volume(c) / (r * r);
            }
        }
        const rtype dt_c = 2.0_r * cell_volume(c) / sum;
        dt_local(c) = dt_c;
        dt_min = Kokkos::fmin(dt_min, finite(c) ? dt_c : -1.0_r);
    }

    /** @brief Whether the cell's state and mass fractions are finite; else the run stops. */
    KOKKOS_INLINE_FUNCTION
    bool finite(const uint32_t c) const {
        bool all = true;
        FOR_I_CONSERVATIVE all = all && Kokkos::isfinite(W(c, i));
        for (uint32_t k = 0; k <= n_species; k++) all = all && Kokkos::isfinite(scalars(c, k));
        return all;
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
        gas.cell_state(cons, cell_species(rhoY, c), T_seed(c), gas.flamelet.cell_variance(c), W, gamma, e0, T);
        set_primitives(P, c, W, gamma, e0, T);
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
        gas.cell_state(cons, cell_species(rhoY, c), 0.0_r, gas.flamelet.cell_variance(c), W, gamma, e0, T);
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

    using value_type = step_queue::MixtureDiagnostics;

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
        gas.cell_state(cons, cell_species(rhoY, c), T_seed(c), gas.flamelet.cell_variance(c), W, gamma, e0, T);
        rtype u2 = 0.0_r, sum_Y = 0.0_r;
        FOR_I_DIM u2 += W[1 + i] * W[1 + i];
        for (uint32_t k = 0; k < gas.n_species; k++) sum_Y += rhoY(c, k);
        // A flamelet gas's scalars [Z, C] need not sum to one
        const rtype values[6] = {W[0], W[N_DIM + 1], -Kokkos::sqrt(u2 * W[0] / (gamma * W[N_DIM + 1])), T, -T,
                                 gas.tabulated() ? 0.0_r : -Kokkos::fabs(sum_Y / W[0] - 1.0_r)};
        for (int i = 0; i < 6; i++) m.v[i] = Kokkos::fmin(m.v[i], values[i]);
    }
};

} // namespace

/**
 * @brief A composition given as X or Y by species, each value a number or an
 *        expression in x, y, z, with an optional balance species taking
 *        1 - sum of the others; without one the values are normalized.
 *        Negative values count as zero.
 */
class CompositionExpressions {
    public:
        /** @param variables Those of the expressions: x, y, z, t, or at boundaries BoundaryValue::variables(entry). */
        CompositionExpressions(const MixtureModel & mixture_model, const Input & table, const std::string & name,
                               const std::vector<std::string> & variables = {"x", "y", "z", "t"},
                               const size_t first_shadowable = SIZE_MAX) :
            mixture(mixture_model), where(name), names(variables), shadowable(first_shadowable) {
            if (mixture.tabulated()) {
                // Z and C (default 0), each a number or an expression; C = "burnt" follows Z
                for (const char * key : {"X", "Y", "balance"}) {
                    if (table.contains(key)) {
                        throw InputError(where + "." + key + ": a flamelet gas takes its composition as Z and C, "
                                         "not species.");
                    }
                }
                const char * keys[2] = {"Z", "C"};
                for (int32_t k = 0; k < 2; k++) {
                    const bool given = table.contains(keys[k]);
                    if (k == 1 && given && table.at("C").is_string() && table.at("C").as_string() == "burnt") {
                        burnt = true;
                        continue;
                    }
                    listed.push_back(k);
                    texts.push_back(given ? quantity_or_expression(table.at(keys[k]), keys[k], dim::none) : "0");
                    values.emplace_back(where + "." + keys[k], line_of(table, keys[k]), texts.back(), variables,
                                        shadowable);
                }
                return;
            }
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
                texts.push_back(quantity_or_expression(value, species, dim::none));
                values.emplace_back(where + "." + key + "." + species, line_of(value), texts.back(), variables,
                                    shadowable);
            }
            if (table.contains("balance")) {
                const std::string species = find_value<std::string>(table, "balance");
                balance = mixture.mechanism().species_index(species);
                if (balance < 0) throw InputError(where + ".balance: no species " + species + " in the mechanism.");
            }
        }

        /** @brief Whether C is the table's burnt value at Z (a flamelet gas), which the device does not evaluate. */
        bool burnt_progress() const { return burnt; }

        /** @brief The fractions compiled as expressions of x, y, z, t, for the device. */
        std::shared_ptr<CompositionPrograms> compiled() const {
            auto c = std::make_shared<CompositionPrograms>();
            for (size_t i = 0; i < listed.size(); i++) {
                c->species.push_back(static_cast<uint32_t>(listed[i]));
                c->fractions.push_back(std::make_shared<kexpr::Program>(where + ".composition", texts[i], names,
                                                                       BoundaryValue::hint, shadowable));
            }
            c->balance = balance;
            c->mole = mole;
            return c;
        }

        /** @brief Mass fractions at the points coords(rows[k]), the expressions evaluated on the device. */
        std::vector<std::vector<double>> mass_fractions(const Kokkos::View<rtype *[N_DIM]> & coords,
                                                        const std::vector<uint32_t> & rows) const {
            std::vector<const Expression *> expressions;
            for (const Expression & e : values) expressions.push_back(&e);
            const std::vector<double> v = evaluate_at(expressions, coords, rows);
            std::vector<std::vector<double>> Y;
            Y.reserve(rows.size());
            for (size_t k = 0; k < rows.size(); k++) Y.push_back(mass_fractions(v.data() + k * values.size()));
            return Y;
        }

        /** @brief Mass fractions at boundary faces, at t = 0 (expressions of BoundaryValue::variables(entry)). */
        std::vector<std::vector<double>> mass_fractions(const std::vector<FacePoint> & points) const {
            std::vector<std::vector<double>> Y;
            Y.reserve(points.size());
            std::vector<double> v(values.size());
            for (const FacePoint & p : points) {
                const auto vars = p.at(0.0);
                for (size_t i = 0; i < values.size(); i++) v[i] = values[i].value(vars);
                Y.push_back(mass_fractions(v.data()));
            }
            return Y;
        }

    private:
        /** @brief Mass fractions from the listed species' values. */
        std::vector<double> mass_fractions(const double * value) const {
            std::vector<double> f(mixture.n_species(), 0.0);
            if (mixture.tabulated()) {
                for (size_t i = 0; i < listed.size(); i++) f[static_cast<size_t>(listed[i])] = std::max(value[i], 0.0);
                if (burnt) f[flamelet::Gas<>::C] = mixture.flamelet().burnt_progress(f[flamelet::Gas<>::Z]);
                mixture.flamelet().host().complete(f);
                return f;
            }
            double sum = 0.0;
            for (size_t i = 0; i < listed.size(); i++) {
                f[static_cast<size_t>(listed[i])] = std::max(value[i], 0.0);
                if (listed[i] != balance) sum += f[static_cast<size_t>(listed[i])];
            }
            if (balance >= 0) {
                f[static_cast<size_t>(balance)] = std::max(1.0 - sum, 0.0);
                sum += f[static_cast<size_t>(balance)];
            }
            if (!(sum > 0.0)) throw InputError(where + ": the composition is zero at a face.");
            for (double & v : f) v /= sum;
            return mole ? mixture.mass_fractions_from_mole(f) : f;
        }

        const MixtureModel & mixture;
        std::string where;
        std::vector<std::string> names;  // the variables of the expressions
        size_t shadowable;
        bool mole = false;
        bool burnt = false;
        int32_t balance = -1;
        std::vector<int32_t> listed;
        std::vector<std::string> texts;
        std::vector<Expression> values;
};

namespace {

/**
 * @brief GeometricSourceFunctor (solver_rhs.cpp) for mixtures: the viscosity
 *        and velocity gradients come from the transport update.
 */
struct MixtureGeometricSourceFunctor {
    Kokkos::View<rtype *[N_CONSERVATIVE]> W;
    Kokkos::View<rtype *[3]> transport;                         // viscous, else empty
    Kokkos::View<rtype ***, Kokkos::LayoutRight> gradients;    // (cell, [u, T, X], dimension)
    Kokkos::View<rtype *> area;
    Kokkos::View<rtype *[N_DIM]> cell_coords;
    Kokkos::View<rtype *> source;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c) const {
        rtype p_eff = W(c, N_DIM + 1);
        if (transport.extent(0) > 0) {
            const rtype hoop = W(c, 2) / cell_coords(c, 1);
            const rtype div = gradients(c, 0, 0) + gradients(c, 1, 1) + hoop;
            p_eff -= transport(c, MU) * (2.0_r * hoop - 2.0_r / 3.0_r * div);
        }
        source(c) = p_eff * area(c);
    }
};

} // namespace

void Solver::mixture_reductions(step_queue::MixtureDiagnostics * out) {
    using Memory = Kokkos::DefaultExecutionSpace::memory_space;
    gated::parallel_reduce("mixture_diagnostics", mesh->n_owned(),
                            MixtureDiagnosticsFunctor{mixture, conservatives, species, T_seed},
                            Kokkos::View<step_queue::MixtureDiagnostics, Memory>(out));
}

void Solver::update_cell_states(const State & solution, const bool update_seed) {
    update_flamelet_variance(solution);
    MixtureCellFunctor functor{mixture, solution.flow, solution.species, T_seed, W_cells, cell_scalars,
                               cell_molar_mass, cells_frozen ? frozen_thermo : Kokkos::View<rtype *[2]>(),
                               update_seed, {}};
    gated::parallel_for("mixture_cells", mesh->n_cells, functor);
}

void Solver::update_cell_states_and_primitives() {
    if (cells_frozen) {
        update_cell_states(state(), false);
        update_primitives();
        return;
    }
    update_flamelet_variance(state());
    MixtureCellFunctor functor{mixture, conservatives, species, T_seed, W_cells, cell_scalars, cell_molar_mass,
                               Kokkos::View<rtype *[2]>(), false, primitives};
    gated::parallel_for("mixture_cells", mesh->n_cells, functor);
}

void Solver::update_transport() {
    gated::parallel_for("mixture_transport", mesh->n_cells,
                         MixtureTransportFunctor{mixture, W_cells, cell_scalars, cell_transport, cell_diffusion,
                                                 transport_values});
    with_rotations(mesh->periodic_maps(), [&](auto rotations) {
        gated::parallel_for("mixture_gradients", n_gradient_cells(),
                             MixtureGradientFunctor<decltype(rotations)::value>{
                                 viscous_gradient, transport_values, transport_gradients, bc_transport_values,
                                 mixture.n_species, cell_transport, mixture.flamelet});
    });
}

void Solver::freeze_thermodynamics() {
    update_flamelet_variance(state());
    gated::parallel_for("freeze_thermodynamics", mesh->n_cells,
                         FreezeFunctor{mixture, conservatives, species, T_seed, frozen_thermo});
}

void Solver::reset_energy() {
    ResetEnergyFunctor functor{mixture, conservatives, species, frozen_thermo, {}, {}};
    if (run_health.reset_site == HealthState::NONE) {
        gated::parallel_for("reset_energy", mesh->n_owned(), functor);
        return;
    }
    const RepairRecorder recorder = run_health.repairs.recorder(run_health.reset_site);
    functor.volume = mesh->cell_measure;
    functor.touched = recorder.touched;
    const Kokkos::View<double *> sums = run_health.reset_sums;
    gated::parallel_reduce("reset_energy", Kokkos::RangePolicy<>(0, mesh->n_owned()), functor, sums);
    gated::parallel_for("reset_energy_record", 1, KOKKOS_LAMBDA(const int) {
        recorder.change(0) += sums(0);
        recorder.activations(0) += static_cast<uint64_t>(sums(1));
    });
}

void Solver::init_temperature_seed() {
    update_flamelet_variance(state());
    gated::parallel_for("temperature_seed", mesh->n_cells, TemperatureFunctor{mixture, conservatives, species, T_seed});
}

void Solver::calc_rhs_mixture(State state, State rhs_state, rtype t_stage, rtype c) {
    const bool exchange = halo.active() && !(halo_current && state.flow.data() == conservatives.data());
    halo_current = false;
    update_boundary_states(t_stage, c);
    if (exchange) halo.exchange(state);
    update_cell_states(state, !budget_pass);
    face_reconstruction->calc_face_values(W_cells, face_solution);
    face_reconstruction->rotate_periodic_faces(face_solution);
    face_reconstruction->image_point_values(W_cells, face_solution);
    // The characteristic faces read the reconstructed gamma of this stage
    scalar_reconstruction.calc(cell_scalars, W_cells, face_solution, face_thermo);
    if (!budget_pass) update_characteristic_boundaries(t_stage);
    if (hybrid_flux) update_upwind_sensor();

    if (double_flux && !cells_frozen) {
        throw std::logic_error("Double flux needs the cells' thermodynamics frozen (take_step).");
    }
    launch_mixture_convective_flux();

    const uint32_t n_species = mixture.n_species;
    scalar_reconstruction.species_slots(cell_scalars, face_mdot, face_reconstruction->quadrature_face.weights,
                                        flux_weights, species_slots);
    if (budget_pass) {
        budget_rates_of_fluxes(0);
    }
    bool sgs_only = false;
    auto viscous_flux = [&] {
        with_rotations(mesh->periodic_maps(), [&](auto rotations) {
            MixtureViscousFluxFunctor<decltype(rotations)::value> functor{
                mesh->face_normals, mesh->face_measure, mesh->face_coords, mesh->cell_coords, mesh->cells_of_face,
                mesh->periodic_maps(), mesh->face_shift, boundary_data, mixture, cell_scalars, transport_values,
                transport_gradients, cell_transport, cell_diffusion, face_flux, species_slots, axisymmetric,
                mesh->cell_covariance, les_coefficients, sgs_only,
                double_flux ? face_energy_1 : Kokkos::View<rtype *>()};
            parallel_for_set("mixture_viscous_flux", rhs_faces, functor);
        });
    };
    if (is_viscous()) {
        update_transport();
        if (les_on) update_eddy_viscosity(mesh->n_cells);
        if (tfles_on) thicken_transport();
        viscous_flux();
        if (budget_pass) budget_rates_of_fluxes(1);
    }

    const uint32_t n_owned = mesh->n_owned();
    StateView rhs = rhs_state.flow;
    if (double_flux) {
        DoubleFluxSumFunctor sum_functor{mesh->offsets_faces_of_cell, mesh->faces_of_cell, mesh->cells_of_face,
                                         face_flux, face_energy_1, rhs, mesh->periodic_maps(), mesh->face_shift};
        gated::parallel_for("face_flux_sum", n_owned, sum_functor);
    } else {
        FaceFluxSumFunctor sum_functor{mesh->offsets_faces_of_cell, mesh->faces_of_cell, mesh->cells_of_face,
                                       face_flux, rhs, mesh->periodic_maps(), mesh->face_shift};
        gated::parallel_for("face_flux_sum", n_owned, sum_functor);
    }
    SpeciesSumFunctor species_sum{mesh->offsets_faces_of_cell, mesh->faces_of_cell, mesh->cells_of_face,
                                  mesh->cell_measure, species_slots, rhs_state.species, n_species};
    gated::parallel_for("species_sum", n_owned, species_sum);
    if (mixture.tabulated()) add_progress_source(rhs_state.species);

    if (axisymmetric) {
        const bool viscous = is_viscous();
        gated::parallel_for("geometric_source", n_owned,
                             MixtureGeometricSourceFunctor{W_cells, viscous ? cell_transport : Kokkos::View<rtype *[3]>(),
                                                           transport_gradients, mesh->cell_volume, mesh->cell_coords,
                                                           geometric_source});
        add_geometric_source(rhs, viscous ? Kokkos::View<rtype *, Kokkos::LayoutStride>(
                                                Kokkos::subview(cell_transport, Kokkos::ALL(), int(MU)))
                                          : Kokkos::View<rtype *, Kokkos::LayoutStride>());
    }

    StateView solution = state.flow;
    Kokkos::View<rtype *> vol = mesh->cell_measure;
    add_sources(state, rhs_state, t_stage, c);
    if (hold_mass_flow) add_mass_flow_force(solution, rhs);
    if (radiating) add_radiation(rhs, state.species);

    gated::parallel_for("rhs_divide_volume", n_owned, KOKKOS_LAMBDA(const uint32_t i_cell) {
        FOR_I_CONSERVATIVE rhs(i_cell, i) /= vol(i_cell);
    });
    apply_sponges(state, rhs_state, t_stage, c);
    if (!budget_pass && run_health.ledger.on()) {
        ledger_stage(state, nullptr);
    }
    if (budget_pass && les_on) {
        // The SGS fluxes alone, into the face fluxes this evaluation no longer needs
        gated::fill(face_flux, 0.0_r);
        sgs_only = true;
        viscous_flux();
        budget_rates_of_fluxes(2);
    }
}

rtype Solver::calc_dt_cfl1_mixture(const Kokkos::View<rtype> & on_device) {
    if (les_on && les.dynamic) {
        eddy_viscosity_of_state(mesh->n_owned());
        update_dynamic_constant();
    }
    if (tfles_on) {
        update_thickened_flame();
        thicken_transport();
    } else if (pasr_on) {
        eddy_viscosity_of_state(mesh->n_owned());
        update_partially_stirred_reactor();
    } else if (les_on) {
        eddy_viscosity_of_state(mesh->n_owned());
    } else {
        update_cell_states(state(), false);
        if (is_viscous()) {
            gated::parallel_for("mixture_transport", mesh->n_owned(),
                                 MixtureTransportFunctor{mixture, W_cells, cell_scalars, cell_transport, cell_diffusion,
                                                         transport_values});
        }
    }
    MixtureTimeStepFunctor functor{mesh->offsets_faces_of_cell, mesh->faces_of_cell, mesh->cells_of_face,
                                   mesh->face_normals, mesh->face_measure, mesh->cell_measure, W_cells,
                                   cell_scalars, cfl_local, mixture.n_species, cell_transport,
                                   axisymmetric ? mesh->cell_coords : Kokkos::View<rtype *[N_DIM]>(), boundary_data,
                                   mesh->periodic_maps(), mesh->face_shift};
    gated::parallel_reduce("time_step", mesh->n_owned(), functor,
                            Kokkos::Min<rtype, Kokkos::DefaultExecutionSpace::memory_space>(
                                on_device.data() ? on_device : dt_reduce_slot()));
    return on_device.data() ? 0.0_r : reduce_dt();
}

void Solver::update_primitives_mixture() {
    // At the subgrid variances of the last update of the cell states: computing them takes an exchange
    gated::parallel_for("update_primitives", mesh->n_cells,
                         MixturePrimitivesFunctor{mixture, conservatives, species, T_seed, primitives});
}

std::vector<rtype> Solver::integrate_species() {
    const comm::SelfScope scope = communication_scope();
    std::vector<rtype> total(species_names.size());
    for (uint32_t k = 0; k < total.size(); k++) {
        rtype sum = 0.0_r;
        gated::parallel_reduce("integrate_species", mesh->n_owned(),
                                SpeciesIntegralFunctor{species, mesh->cell_measure, k}, sum);
        total[k] = sum;
    }
    if (total.empty() || !is_distributed()) return total;
    comm::allreduce(std::span<rtype>(total), comm::Op::SUM);
    return total;
}

void Solver::read_composition(const Input & table, const std::string & where) const {
    CompositionExpressions(*mixture_model, table, where);
}

std::vector<std::vector<double>> Solver::composition_at_cells(const Input & table, const std::string & where,
                                                              const std::vector<uint32_t> & cells) const {
    return CompositionExpressions(*mixture_model, table, where).mass_fractions(mesh->cell_coords, cells);
}

void Solver::read_mixture_boundary(const Input & bound, BoundaryEntry & entry) const {
    const std::string & where = entry.where_name;
    auto unsupported = [&](const std::string & what) {
        throw InputError(where + ": " + what + " is not yet supported with gas = \"" +
                         (mixture_model->tabulated() ? "flamelet" : "mixture") +
                         "\" (it is with outlet_extrapolation, symmetry, the wall_* types, inlet_state, "
                         "inlet_stagnation, inlet_mass_flow and outlet_pressure without average).");
    };
    switch (entry.bc.type) {
        case BoundaryType::OUTLET_EXTRAPOLATION:
        case BoundaryType::SYMMETRY:
        case BoundaryType::WALL_ADIABATIC:
        case BoundaryType::WALL_ISOTHERMAL:
        case BoundaryType::WALL_HEAT_FLUX:
        case BoundaryType::OUTLET_PRESSURE:
            if (entry.bc.has(BoundaryCondition::AVERAGE)) unsupported("average = true");
            return;
        case BoundaryType::INLET_STATE:
        case BoundaryType::INLET_STAGNATION:
        case BoundaryType::INLET_MASS_FLOW: {
            const char * T_key = entry.bc.type == BoundaryType::INLET_STAGNATION ? "T0" : "T";
            // The temperature and composition of the condition, or of each face where they are expressions
            entry.T_varies = BoundaryValue::is_expression(bound.at(T_key));
            if (!entry.T_varies) entry.T = static_cast<double>(find_real(bound, T_key, dim::temperature));
            if (MixtureModel::composition_varies(bound)) {
                entry.composition = std::make_shared<CompositionExpressions>(
                    *mixture_model, bound, where, BoundaryValue::variables(bound), boundary_variable::FIRST_LOCAL);
                if (entry.composition->burnt_progress() && entry.composition->compiled()->depends_on_time()) {
                    throw InputError(where + ".C: \"burnt\" with Z varying in time is not supported; give C "
                                     "as an expression of t too.");
                }
            } else {
                entry.Y = mixture_model->mass_fractions(bound, where);
            }
            return;
        }
        default:
            unsupported("type = \"" + BOUNDARY_NAMES.at(entry.bc.type) + "\"");
    }
}

void Solver::init_mixture_boundaries(const std::vector<std::array<uint32_t, 2>> & profiled_faces,
                                     std::vector<BoundaryCondition> & bcs) {
    bc_mass_fractions.assign(bcs.size(), {});
    bc_surrogates.assign(bcs.size(), {1.4, 0.0});
    bc_temperatures.assign(bcs.size(), 0.0);
    // Prescribed state of condition i_bc
    auto prescribe = [&](size_t i_bc, const std::vector<double> & Y, double T) {
        BoundaryCondition & bc = bcs[i_bc];
        if (bc.type == BoundaryType::INLET_STATE) {
            const double p = static_cast<double>(bc.data[N_DIM + 1]);
            bc.data[0] = static_cast<rtype>(p / (mixture_model->gas_constant(Y) * T));
        }
        bc_mass_fractions[i_bc] = Y;
        bc_temperatures[i_bc] = T;
        mixture_model->surrogates(T, Y, bc_surrogates[i_bc][0], bc_surrogates[i_bc][1]);
    };
    for (size_t i_bc = 0; i_bc < boundary_entries.size(); i_bc++) {
        const BoundaryEntry & entry = boundary_entries[i_bc];
        const BoundaryType type = bcs[i_bc].type;
        if (type != BoundaryType::INLET_STATE && type != BoundaryType::INLET_STAGNATION &&
            type != BoundaryType::INLET_MASS_FLOW) {
            continue;
        }
        const BoundaryFields * fields = boundary_fields[i_bc].get();
        const std::shared_ptr<CompositionExpressions> & composition = entry.composition;
        if (composition) {
            const auto programs = composition->compiled();
            if (programs->depends_on_time()) boundary_composition[i_bc] = programs;
        } else if (!entry.T_varies) {
            prescribe(i_bc, entry.Y, entry.T);
        }
        on_every_rank([&] {
            std::vector<size_t> profiled;
            std::vector<FacePoint> points;
            for (size_t k = 0; k < profiled_faces.size(); k++) {
                if (profiled_faces[k][0] != i_bc) continue;
                profiled.push_back(k);
                points.push_back(boundary_point(profiled_faces[k][1]));
            }
            const std::vector<std::vector<double>> Y =
                composition ? composition->mass_fractions(points) : std::vector<std::vector<double>>();
            for (size_t j = 0; j < profiled.size(); j++) {
                const double T = entry.T_varies ? fields->temperature(points[j], 0.0) : entry.T;
                prescribe(boundary_entries.size() + profiled[j], composition ? Y[j] : entry.Y, T);
            }
        });
    }
}
