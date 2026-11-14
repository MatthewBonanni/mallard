/**
 * @file derived_fields.cpp
 * @brief Derived output fields of a flow state.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "derived_fields.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "mixture.h"
#include "probes.h"

namespace derived {

namespace {

const char * const DOCS = "docs/guide/output.md#derived-fields";

/** @brief Atoms of element e in species s. */
double atoms(const chemistry::Species & s, const char * e) {
    for (const auto & [element, n] : s.composition) {
        if (element == e) return n;
    }
    return 0.0;
}

/**
 * @brief Bilger's coupling function per unit mass fraction of species s,
 *        (2 C + H / 2 - O) / W [kmol/kg]: beta = sum_k b_k Y_k is conserved
 *        by reactions (Bilger, Starner & Kee 1990), so Z = (beta - beta_O) / (beta_F - beta_O).
 */
double bilger_coefficient(const chemistry::Species & s) {
    return (2.0 * atoms(s, "C") + 0.5 * atoms(s, "H") - atoms(s, "O")) / s.molecular_weight;
}

/** @brief The mass fractions of a stream of [output.mixture]: a composition table, or a [[boundaries]] name. */
std::vector<double> stream(const Input & input, const Input & table, const std::string & key,
                           const MixtureModel & mixture) {
    const std::string path = "output.mixture." + key;
    const Input & value = table.at(key);
    if (!value.is_string()) {
        if (!value.is_table() || (!value.contains("X") && !value.contains("Y"))) {
            throw InputError(Diagnostic{.key = path,
                                        .line = line_of(table, key),
                                        .message = "is a composition {X = {...}} or {Y = {...}}, or the name of a "
                                                   "[[boundaries]] entry whose composition it takes",
                                        .fix = "e.g. " + key + (key == "fuel" ? " = {X = {CH4 = 1}}" : " = {X = {O2 = 1, N2 = 3.76}}"),
                                        .docs = DOCS});
        }
        if (MixtureModel::composition_varies(value)) throw InputError(path + " must be uniform: give numbers.");
        return mixture.mass_fractions(value, path);
    }
    const std::string name = find_value<std::string>(table, key);
    std::vector<std::string> names;
    std::vector<double> Y;
    if (input.contains("boundaries") && input.at("boundaries").is_array()) {
        for (const Input & b : input.at("boundaries").as_array()) {
            if (!b.is_table() || !b.contains("name") || !b.at("name").is_string()) continue;
            names.push_back(b.at("name").as_string());
            if (names.back() != name) continue;
            if (!b.contains("X") && !b.contains("Y")) {
                throw InputError(path + ": the boundary \"" + name + "\" has no composition (X or Y): give the " +
                                 key + " stream's composition, {X = {...}}.");
            }
            if (MixtureModel::composition_varies(b)) {
                throw InputError(path + ": the composition of the boundary \"" + name + "\" varies in space or " +
                                 "time: give the " + key + " stream's composition, {X = {...}}.");
            }
            const std::vector<double> here = mixture.mass_fractions(b, "boundaries[name = \"" + name + "\"]");
            if (!Y.empty() && here != Y) {
                throw InputError(path + ": the [[boundaries]] entries named \"" + name + "\" have different " +
                                 "compositions: give the " + key + " stream's composition, {X = {...}}.");
            }
            Y = here;
        }
    }
    if (!Y.empty()) return Y;
    throw InputError(Diagnostic{.key = path,
                                .line = line_of(table, key),
                                .message = "no [[boundaries]] entry is named \"" + name + "\"",
                                .valid = names,
                                .suggestion = suggest_value(name, names),
                                .docs = DOCS});
}

/** @brief The species of a stream that counts most by mass among those that qualify(species), or -1. */
template <typename F>
int32_t most_abundant(const chemistry::Mechanism & mech, const std::vector<double> & Y, F && qualifies) {
    int32_t best = -1;
    for (uint32_t k = 0; k < mech.n_species(); k++) {
        if (Y[k] > 0.0 && qualifies(mech.species[k]) && (best < 0 || Y[k] > Y[static_cast<uint32_t>(best)])) {
            best = static_cast<int32_t>(k);
        }
    }
    return best;
}

int32_t species_of(const Input & table, const std::string & key, const MixtureModel & mixture, int32_t fallback) {
    const std::string name = find_value_or<std::string>(table, key, "");
    if (name.empty()) return fallback;
    const int32_t k = mixture.mechanism().species_index(name);
    if (k < 0) throw mixture.unknown_species(table, key, name);
    return k;
}

/** @brief Mass fractions Y(k) of cell c, for the thermodynamic tables. */
struct CellY {
    Kokkos::View<rtype **, Kokkos::LayoutRight> Y;
    uint32_t c;
    KOKKOS_INLINE_FUNCTION double operator()(const uint32_t k) const { return static_cast<double>(Y(c, k)); }
};

} // namespace

Settings Settings::from_input(const Input & input, const Euler & physics, const MixtureModel * mixture,
                              const bool viscous) {
    Settings s;
    s.mixture = mixture != nullptr;
    s.viscous = viscous;
    if (!mixture) {
        s.cp = physics.cp;
        s.R = physics.R;
        s.T_ref = find_real(input, "physics", "T_ref", dim::temperature);
        s.p_ref = find_real(input, "physics", "p_ref", dim::pressure);
    }
    const Input * output = input.contains("output") && input.at("output").is_table() ? &input.at("output") : nullptr;
    if (output && output->contains("reference")) {
        const ProbeReference ref = probe::read_reference(output->at("reference"), "output.reference");
        s.reference = true;
        s.p_inf = ref.p;
        s.q_inf = ref.q;
    }
    if (!output || !output->contains("mixture")) return s;
    const Input & table = output->at("mixture");
    if (!mixture) {
        throw InputError(Diagnostic{.key = "output.mixture",
                                    .line = line_of(*output, "mixture"),
                                    .message = "describes the streams and species of a gas mixture, and this case "
                                               "is a perfect gas",
                                    .fix = "Remove [output.mixture], or set physics.gas = \"mixture\"",
                                    .docs = DOCS});
    }
    const chemistry::Mechanism & mech = mixture->mechanism();
    const uint32_t n = mech.n_species();
    s.scalar_weights = Kokkos::View<rtype **, Kokkos::LayoutRight>("derived_scalar_weights", N_SCALARS, n);
    auto weights = Kokkos::create_mirror_view(s.scalar_weights);

    const bool fuel = table.contains("fuel"), oxidizer = table.contains("oxidizer");
    if (fuel != oxidizer) {
        throw missing_key(table, std::string("output.mixture.") + (fuel ? "oxidizer" : "fuel"),
                          "The mixture fraction needs both streams, fuel and oxidizer");
    }
    int32_t fuel_default = -1, oxidizer_default = -1;
    if (fuel) {
        const std::vector<double> Y_F = stream(input, table, "fuel", *mixture);
        const std::vector<double> Y_O = stream(input, table, "oxidizer", *mixture);
        double beta_F = 0.0, beta_O = 0.0, scale = 0.0;
        std::vector<double> b(n);
        for (uint32_t k = 0; k < n; k++) {
            b[k] = bilger_coefficient(mech.species[k]);
            beta_F += b[k] * Y_F[k];
            beta_O += b[k] * Y_O[k];
            scale = std::max(scale, std::abs(b[k]));
        }
        if (!(std::abs(beta_F - beta_O) > 1e-6 * scale)) {
            throw InputError(Diagnostic{.key = "output.mixture",
                                        .line = line_of(output->at("mixture")),
                                        .message = "the fuel and oxidizer streams have the same Bilger coupling "
                                                   "function (2 C + H/2 - O per unit mass), so no mixture fraction "
                                                   "separates them",
                                        .fix = "Give the fuel stream (C and H) and the oxidizer stream (O)",
                                        .docs = DOCS});
        }
        s.streams = true;
        for (uint32_t k = 0; k < n; k++) weights(Z, k) = static_cast<rtype>(b[k] / (beta_F - beta_O));
        s.z_offset = static_cast<rtype>(-beta_O / (beta_F - beta_O));
        const auto burns = [](const chemistry::Species & sp) { return atoms(sp, "C") + atoms(sp, "H") > 0.0; };
        fuel_default = most_abundant(mech, Y_F, burns);
        oxidizer_default = most_abundant(mech, Y_O, [&](const chemistry::Species & sp) {
            return atoms(sp, "O") > 0.0 && !burns(sp);
        });
    }
    s.fuel_species = species_of(table, "fuel_species", *mixture, fuel_default);
    s.oxidizer_species = species_of(table, "oxidizer_species", *mixture, oxidizer_default);
    if (s.fuel_species >= 0) weights(FUEL, static_cast<uint32_t>(s.fuel_species)) = 1.0_r;
    if (s.oxidizer_species >= 0) weights(OXIDIZER, static_cast<uint32_t>(s.oxidizer_species)) = 1.0_r;
    Kokkos::deep_copy(s.scalar_weights, weights);

    if (table.contains("progress")) {
        const Input & progress = table.at("progress");
        const std::string key = "output.mixture.progress";
        const bool by_species = progress.is_table() && progress.contains("species");
        const bool by_T = progress.is_table() && (progress.contains("T_unburnt") || progress.contains("T_burnt"));
        if (by_species == by_T) {
            throw InputError(Diagnostic{.key = key,
                                        .line = line_of(table, "progress"),
                                        .message = "is either {species = [...], burnt = ...} (a sum of mass "
                                                   "fractions) or {T_unburnt = ..., T_burnt = ...}",
                                        .fix = "e.g. progress = {species = [\"CO2\", \"CO\"], burnt = 0.15}",
                                        .docs = DOCS});
        }
        if (by_species) {
            const auto names = find_value<std::vector<std::string>>(progress, "species");
            if (names.empty()) throw InputError(key + ".species lists no species.");
            const rtype burnt = find_real_or(progress, "burnt", dim::none, 1.0_r);
            if (!(burnt > 0.0_r)) throw InputError(key + ".burnt must be positive.");
            s.progress = Progress::SPECIES;
            s.progress_weights = Kokkos::View<rtype *>("derived_progress_weights", n);
            auto h = Kokkos::create_mirror_view(s.progress_weights);
            for (const std::string & name : names) {
                const int32_t k = mech.species_index(name);
                if (k < 0) throw mixture->unknown_species(progress, "species", name);
                h(static_cast<uint32_t>(k)) = 1.0_r / burnt;
            }
            Kokkos::deep_copy(s.progress_weights, h);
        } else {
            for (const char * k : {"T_unburnt", "T_burnt"}) {
                if (!progress.contains(k)) throw missing_key(progress, key + "." + k);
            }
            s.T_unburnt = find_real(progress, "T_unburnt", dim::temperature);
            s.T_burnt = find_real(progress, "T_burnt", dim::temperature);
            if (!(s.T_burnt != s.T_unburnt)) throw InputError(key + ": T_burnt must differ from T_unburnt.");
            s.progress = Progress::TEMPERATURE;
        }
    }
    return s;
}

bool Settings::available(const Field f) const {
    const uint16_t n = FIELDS[f].needs;
    const auto needs = [&](uint16_t bit) { return (n & bit) != 0; };
    return !(needs(need::THREE_D) && N_DIM != 3) && !(needs(need::MIXTURE) && !mixture) &&
           !(needs(need::VISCOUS) && !viscous) && !(needs(need::REFERENCE) && !reference) &&
           !(needs(need::STREAMS) && !streams) && !(needs(need::PROGRESS) && progress == Progress::NONE) &&
           !(needs(need::SPECIES_PAIR) && (fuel_species < 0 || oxidizer_species < 0));
}

void Settings::require(const Field f, const Input & input) const {
    if (available(f)) return;
    const uint16_t n = FIELDS[f].needs;
    const auto needs = [&](uint16_t bit) { return (n & bit) != 0; };
    const std::string name = FIELDS[f].name;
    Diagnostic d{.key = "write_data.fields", .docs = DOCS};
    for (const Input & entry : input.at("write_data").as_array()) {
        if (!entry.is_table() || !entry.contains("fields") || !entry.at("fields").is_array()) continue;
        for (const Input & v : entry.at("fields").as_array()) {
            if (v.is_string() && v.as_string() == name && d.line == 0) d.line = line_of(entry, "fields");
        }
    }
    if (needs(need::THREE_D) && N_DIM != 3) {
        d.message = name + " is zero in 2D, where the vorticity is normal to the velocity";
        d.fix = "Run the case with mallard3d, or remove " + name;
    } else if (needs(need::MIXTURE) && !mixture) {
        d.message = name + " is a field of gas mixtures, and this case is a perfect gas";
        d.fix = "Remove " + name + ", or set physics.gas = \"mixture\"";
    } else if (needs(need::VISCOUS) && !viscous) {
        d.message = name + " needs the mixture's diffusivity, which only navier_stokes has";
        d.fix = "Set physics.type = \"navier_stokes\", or remove " + name;
    } else if (needs(need::REFERENCE) && !reference) {
        d.message = name + " needs the free stream it is relative to";
        d.fix = "Add [output] reference = {rho = ..., u = [...], p = ...}";
    } else if (needs(need::STREAMS) && !streams) {
        d.message = name + " needs the fuel and oxidizer streams of the mixture fraction";
        d.fix = "Add [output.mixture] fuel = {X = {...}} (or a [[boundaries]] name) and oxidizer = ...";
    } else if (needs(need::PROGRESS)) {
        d.message = name + " needs its definition";
        d.fix = "Add [output.mixture] progress = {species = [\"CO2\", \"CO\"], burnt = ...} or "
                "progress = {T_unburnt = ..., T_burnt = ...}";
    } else {
        d.message = name + " needs a fuel and an oxidizer species";
        d.fix = streams ? "The streams have none that qualifies: add [output.mixture] fuel_species and "
                          "oxidizer_species"
                        : "Add [output.mixture] fuel_species and oxidizer_species, or the fuel and oxidizer streams";
    }
    throw InputError(d);
}

Workspace::Workspace(const uint32_t n_cells)
    : values("derived_scalars", n_cells, N_SCALARS), gradients("derived_scalar_gradients", n_cells, N_SCALARS, N_DIM) {}

void scalar_gradients(const Settings & settings, const Kokkos::View<rtype **, Kokkos::LayoutRight> & Y,
                      const LSQVertexGradientFunctor & stencil, const Kokkos::View<rtype **, Kokkos::LayoutRight> & bc_Y,
                      const uint32_t n_cells, const uint32_t n_owned, Workspace & workspace) {
    const Kokkos::View<rtype **, Kokkos::LayoutRight> weights = settings.scalar_weights;
    const Kokkos::View<rtype **, Kokkos::LayoutRight> values = workspace.values;
    const Kokkos::View<rtype ***, Kokkos::LayoutRight> gradients = workspace.gradients;
    const uint32_t n_species = static_cast<uint32_t>(weights.extent(1));
    Kokkos::parallel_for("derived_scalars", n_cells, KOKKOS_LAMBDA(const uint32_t c) {
        for (uint32_t v = 0; v < N_SCALARS; v++) {
            rtype sum = 0.0;
            for (uint32_t k = 0; k < n_species; k++) sum += weights(v, k) * Y(c, k);
            values(c, v) = sum;
        }
    });
    const LSQVertexGradientFunctor s = stencil;
    Kokkos::parallel_for("derived_scalar_gradients", n_owned, KOKKOS_LAMBDA(const uint32_t c) {
        const LSQGradientFunctor & f = s.faces;
        rtype g[N_SCALARS][N_DIM] = {};
        for (uint32_t k = s.offsets_cells_of_cell(c); k < s.offsets_cells_of_cell(c + 1); k++) {
            const uint32_t j = s.cells_of_cell(k);
            for (uint32_t v = 0; v < N_SCALARS; v++) {
                const rtype dq = values(j, v) - values(c, v);
                FOR_I_DIM g[v][i] += s.weights.cells(k, i) * dq;
            }
        }
        s.for_each_image(c, [&](const uint32_t j, const uint32_t k) {
            for (uint32_t v = 0; v < N_SCALARS; v++) {
                const rtype dq = values(j, v) - values(c, v);
                FOR_I_DIM g[v][i] += s.weights.images(k, i) * dq;
            }
        });
        s.for_each_ghost(c, [&](const uint32_t i_face, const uint32_t k) {
            const int32_t i_bc = f.boundaries.face_bc(i_face);
            const BoundaryCondition & bc = f.boundaries.bcs(i_bc);
            if (bc.type != BoundaryType::INLET_STATE || bc.is_characteristic() || bc_Y.extent(0) == 0) return;
            rtype total = 0.0;
            for (uint32_t m = 0; m < n_species; m++) total += bc_Y(i_bc, m);
            if (!(total > 0.5_r)) return;
            for (uint32_t v = 0; v < N_SCALARS; v++) {
                rtype ghost = 0.0;
                for (uint32_t m = 0; m < n_species; m++) ghost += weights(v, m) * bc_Y(i_bc, m);
                const rtype dq = ghost - values(c, v);
                FOR_I_DIM g[v][i] += s.weights.faces(k, i) * dq;
            }
        });
        for (uint32_t v = 0; v < N_SCALARS; v++) {
            FOR_I_DIM gradients(c, v, i) = g[v][i];
        }
    });
}

void compute(const Selection & selection, const Settings & settings, const Inputs & inputs, const uint32_t n_cells,
             Kokkos::View<rtype **> out) {
    if (!selection.any()) return;
    const Selection sel = selection;
    const Kokkos::View<rtype *[N_CONSERVATIVE]> W = inputs.W;
    const Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> g = inputs.gradients;
    const rtype gamma = inputs.gamma;
    const Kokkos::View<rtype *, Kokkos::LayoutStride> cell_gamma = inputs.cell_gamma;
    const Kokkos::View<rtype *[N_DIM]> radius_coords = inputs.radius_coords;
    const Kokkos::View<rtype *[N_PRIMITIVE]> P = inputs.primitives;
    const Kokkos::View<rtype **, Kokkos::LayoutRight> Y = inputs.Y;
    const chemistry::ThermoTable<> thermo = inputs.thermo;
    const Kokkos::View<rtype *[3]> transport = inputs.transport;
    const Kokkos::View<rtype ***, Kokkos::LayoutRight> sg = inputs.scalar_gradients;
    const Kokkos::View<rtype **, Kokkos::LayoutRight> weights = settings.scalar_weights;
    const Kokkos::View<rtype *> progress_weights = settings.progress_weights;
    const bool mixture = settings.mixture;
    const bool progress_T = settings.progress == Settings::Progress::TEMPERATURE;
    const rtype cp = settings.cp, R = settings.R, T_ref = settings.T_ref, p_ref = settings.p_ref;
    const rtype p_inf = settings.p_inf, q_inf = settings.q_inf, z_offset = settings.z_offset;
    const rtype T_unburnt = settings.T_unburnt, T_burnt = settings.T_burnt;
    const uint32_t n_species = mixture ? thermo.n_species : 0;
    Kokkos::parallel_for("derived_fields", n_cells, KOKKOS_LAMBDA(const uint32_t c) {
        const auto put = [&](Field f, rtype value) {
            if (sel.column[f] >= 0) out(c, static_cast<uint32_t>(sel.column[f])) = value;
        };
        const auto wants = [&](Field f) { return sel.column[f] >= 0; };
        const rtype rho = W(c, 0), p = W(c, N_DIM + 1);
        const rtype gamma_c = cell_gamma.extent(0) > 0 ? cell_gamma(c) : gamma;
        rtype u2 = 0.0;
        FOR_I_DIM u2 += W(c, 1 + i) * W(c, 1 + i);
        put(MACH, Kokkos::sqrt(u2 * rho / (gamma_c * p)));
        if (wants(P0) || wants(T0)) {
            const rtype ratio = 1.0_r + 0.5_r * (gamma_c - 1.0_r) * u2 * rho / (gamma_c * p);
            put(P0, p * Kokkos::pow(ratio, gamma_c / (gamma_c - 1.0_r)));
            put(T0, P(c, N_DIM + 1) * ratio);
        }
        if (wants(H0)) put(H0, P(c, N_DIM + 2) + 0.5_r * u2);
        put(SOUND_SPEED, Kokkos::sqrt(gamma_c * p / rho));
        if (wants(ENTROPY)) {
            const rtype T = P(c, N_DIM + 1);
            if (mixture) {
                // s = sum_k Y_k (s_k(T) - R_k ln(X_k p / p_atm)): the third-law entropy with that of mixing
                const CellY y{Y, c};
                double sum_n = 0.0;
                for (uint32_t k = 0; k < n_species; k++) sum_n += y(k) * thermo.inv_W(k);
                double mixing = 0.0;
                for (uint32_t k = 0; k < n_species; k++) {
                    const double n = y(k) * thermo.inv_W(k);
                    if (n > 0.0) mixing += n * Kokkos::log(n / sum_n);
                }
                const double s = thermo.s0_mass(static_cast<double>(T), y) -
                                 chemistry::GAS_CONSTANT * (sum_n * Kokkos::log(static_cast<double>(p) /
                                                                                 chemistry::ONE_ATM) + mixing);
                put(ENTROPY, static_cast<rtype>(s));
            } else {
                put(ENTROPY, cp * Kokkos::log(T / T_ref) - R * Kokkos::log(p / p_ref));
            }
        }
        put(CP, (p - p_inf) / q_inf);
        if (wants(Q_CRITERION) || wants(LAMBDA2) || wants(DILATATION) || wants(HELICITY) || wants(ENSTROPHY) ||
            wants(STRAIN_RATE)) {
            const Tensor a = velocity_gradient(g, W, radius_coords, c);
            put(Q_CRITERION, q_criterion(a));
            if (wants(LAMBDA2)) put(LAMBDA2, lambda2(a));
            put(DILATATION, dilatation(a));
            put(ENSTROPHY, enstrophy(a));
            put(STRAIN_RATE, strain_rate(a));
            if (wants(HELICITY)) {
                rtype w[3];
                vorticity(a, w);
                rtype h = 0.0;
                FOR_I_DIM h += W(c, 1 + i) * w[i];
                put(HELICITY, h);
            }
        }
        if (wants(SCHLIEREN)) {
            rtype g2 = 0.0;
            FOR_I_DIM g2 += g(c, 0, i) * g(c, 0, i);
            put(SCHLIEREN, Kokkos::sqrt(g2));
        }
        if (wants(MIXTURE_FRACTION)) {
            rtype z = z_offset;
            for (uint32_t k = 0; k < n_species; k++) z += weights(Z, k) * Y(c, k);
            put(MIXTURE_FRACTION, z);
        }
        if (wants(PROGRESS_VARIABLE)) {
            rtype value = 0.0;
            if (progress_T) {
                value = (P(c, N_DIM + 1) - T_unburnt) / (T_burnt - T_unburnt);
            } else {
                for (uint32_t k = 0; k < n_species; k++) value += progress_weights(k) * Y(c, k);
            }
            put(PROGRESS_VARIABLE, value);
        }
        if (wants(FLAME_INDEX)) {
            rtype g_F[N_DIM], g_O[N_DIM];
            FOR_I_DIM {
                g_F[i] = sg(c, FUEL, i);
                g_O[i] = sg(c, OXIDIZER, i);
            }
            put(FLAME_INDEX, flame_index(g_F, g_O));
        }
        if (wants(SCALAR_DISSIPATION)) {
            // chi = 2 D |grad Z|^2 with D = lambda / (rho cp), cp the cell's frozen one
            const rtype T = P(c, N_DIM + 1);
            const rtype cp_c = gamma_c / (gamma_c - 1.0_r) * p / (rho * T);
            rtype gz2 = 0.0;
            FOR_I_DIM gz2 += sg(c, Z, i) * sg(c, Z, i);
            put(SCALAR_DISSIPATION, 2.0_r * transport(c, 1) / (rho * cp_c) * gz2);
        }
    });
}

} // namespace derived
