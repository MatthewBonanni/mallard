/**
 * @file boundary_fields.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Boundary numbers given as expressions.
 * @version 0.1
 * @date 2026-10-16
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "boundary_fields.h"

#include <iomanip>
#include <sstream>

#include "input.h"

namespace {

/** @brief A constant program holding v exactly. */
std::shared_ptr<expr::Program> constant(const double v, const std::string & name) {
    std::ostringstream text;
    text << std::setprecision(17) << v;
    return std::make_shared<expr::Program>(name, text.str(), BoundaryValue::variables());
}

std::string real(const double v) {
    std::ostringstream s;
    s << std::setprecision(6) << v;
    return s.str();
}

} // namespace

const std::vector<std::string> & BoundaryValue::variables() {
    static const std::vector<std::string> names = {"x", "y", "z", "t"};
    return names;
}

BoundaryValue::BoundaryValue(const Input & v, const std::string & name, const Dimension & dimension) {
    if (is_expression(v)) {
        program = std::make_shared<expr::Program>(name, v.as_string(), variables());
    } else {
        program = constant(as_double(v, name, dimension), name);
    }
}

bool BoundaryValue::is_expression(const Input & v) { return v.is_string() && !has_unit(v.as_string()); }

double BoundaryValue::at(const Point & x, const double t) const {
    const double vars[4] = {x[0], x[1], N_DIM > 2 ? x[N_DIM - 1] : 0.0, t};
    return (*program)(vars);
}

BoundaryFields::BoundaryFields(const Input & input, const BoundaryType type_, const std::string & where_)
    : type(type_), where(where_) {
    auto read = [&](const char * key, const Dimension & dimension, BoundaryValue & v, bool & has) {
        if (!input.contains(key)) return;
        v = BoundaryValue(input.at(key), where + "." + key, dimension);
        has = true;
    };
    const bool walls = type == BoundaryType::WALL_ADIABATIC || type == BoundaryType::WALL_ISOTHERMAL ||
                       type == BoundaryType::WALL_HEAT_FLUX;
    if ((type == BoundaryType::UPT || walls) && input.contains("u")) {
        const Input & u_in = input.at("u");
        if (!u_in.is_array() || u_in.as_array().size() != N_DIM) {
            throw InputError(where + ".u must have " + std::to_string(N_DIM) + " components (numbers or expressions).");
        }
        FOR_I_DIM u.emplace_back(u_in.as_array()[i], where + ".u[" + std::to_string(i) + "]", dim::velocity);
    }
    if (type == BoundaryType::UPT || type == BoundaryType::NSCBC_INLET) {
        read("p", dim::pressure, p, has_p);
        read("T", dim::temperature, T, has_T);
    } else if (type == BoundaryType::STAGNATION_INLET) {
        read("p0", dim::pressure, p, has_p);
        read("T0", dim::temperature, T, has_T);
    } else if (type == BoundaryType::MASS_FLOW_INLET) {
        read("T", dim::temperature, T, has_T);
        read("mass_flux", dim::mass_flux, flux, has_flux);
    }
}

std::vector<const BoundaryValue *> BoundaryFields::values() const {
    std::vector<const BoundaryValue *> all;
    for (const BoundaryValue & v : u) all.push_back(&v);
    for (const BoundaryValue * v : {&p, &T, &flux}) all.push_back(v);
    return all;
}

bool BoundaryFields::varies() const {
    for (const BoundaryValue * v : values()) {
        if (v->varies()) return true;
    }
    return false;
}

bool BoundaryFields::depends_on_time() const {
    for (const BoundaryValue * v : values()) {
        if (v->depends_on_time()) return true;
    }
    return false;
}

const BoundaryValue * BoundaryFields::slot(const int s) const {
    if (s < SLOT_P) return static_cast<size_t>(s) < u.size() ? &u[static_cast<size_t>(s)] : nullptr;
    if (s == SLOT_P) return has_p ? &p : nullptr;
    if (s == SLOT_T) return has_T ? &T : nullptr;
    return has_flux ? &flux : nullptr;
}

std::string BoundaryFields::slot_key(const int s) const {
    const bool stagnation = type == BoundaryType::STAGNATION_INLET;
    if (s < SLOT_P) return "u[" + std::to_string(s) + "]";
    if (s == SLOT_P) return stagnation ? "p0" : "p";
    if (s == SLOT_T) return stagnation ? "T0" : "T";
    return "mass_flux";
}

double BoundaryFields::temperature(const Point & x, const double t) const { return has_T ? T.at(x, t) : 0.0; }

void BoundaryFields::apply(BoundaryCondition & bc, const Point & x, const double t, const double R,
                           const bool numbers_only) const {
    constexpr uint8_t E = N_DIM + 1;
    auto use = [&](const BoundaryValue & v) { return !numbers_only || !v.varies(); };
    auto positive = [&](const BoundaryValue & v, const char * key) {
        const double value = v.at(x, t);
        if (!(value > 0.0)) {
            throw InputError(where + "." + key + " must be positive (" + real(value) + " at t = " + real(t) + ").");
        }
        return value;
    };
    bool u_varies = false;
    for (const BoundaryValue & v : u) u_varies = u_varies || v.varies();
    if (!u.empty() && (!numbers_only || !u_varies)) {
        FOR_I_DIM bc.data[1 + i] = static_cast<rtype>(u[i].at(x, t));
    }
    switch (type) {
        case BoundaryType::UPT:
        case BoundaryType::NSCBC_INLET:
            if (!has_p || !has_T || !use(p) || !use(T)) break;
            {
                const double p_v = positive(p, "p"), T_v = positive(T, "T");
                bc.data[0] = static_cast<rtype>(p_v / (R * T_v));
                bc.data[E] = static_cast<rtype>(p_v);
                if (type == BoundaryType::NSCBC_INLET) bc.relax[BoundaryCondition::T_TARGET] = static_cast<rtype>(T_v);
            }
            break;
        case BoundaryType::STAGNATION_INLET:
            if (has_p && use(p)) bc.data[E] = static_cast<rtype>(positive(p, "p0"));
            if (has_T && use(T)) bc.data[0] = static_cast<rtype>(positive(T, "T0"));
            break;
        case BoundaryType::MASS_FLOW_INLET:
            if (has_T && use(T)) bc.data[0] = static_cast<rtype>(positive(T, "T"));
            if (has_flux && use(flux)) {
                const double G = flux.at(x, t);
                if (!(G >= 0.0) || (!flux.varies() && !(G > 0.0))) {
                    throw InputError(where + ".mass_flux must be positive (non-negative at faces of an expression).");
                }
                bc.data[E] = static_cast<rtype>(G);
            }
            break;
        default:
            break;
    }
}

bool CompositionPrograms::depends_on_time() const {
    for (const auto & f : fractions) {
        if (f->uses(3)) return true;
    }
    return false;
}

void TimedBoundaryFields::add(const uint32_t i_bc, const Point & x, const std::shared_ptr<const BoundaryFields> & fields,
                              const std::shared_ptr<const CompositionPrograms> & composition) {
    uint32_t set = 0;
    while (set < sets.size() && sets[set] != fields) set++;
    if (set == sets.size()) {
        sets.push_back(fields);
        compositions.push_back(composition);
    }
    faces_host.push_back({i_bc, set, x});
}

void TimedBoundaryFields::finish() {
    const size_t n = faces_host.size(), n_sets = sets.size();
    face_bc = Kokkos::View<uint32_t *>("timed_face_bc", n);
    face_set = Kokkos::View<uint32_t *>("timed_face_set", n);
    face_x = Kokkos::View<double *[3]>("timed_face_x", n);
    auto h_bc = Kokkos::create_mirror_view(face_bc);
    auto h_set = Kokkos::create_mirror_view(face_set);
    auto h_x = Kokkos::create_mirror_view(face_x);
    for (size_t k = 0; k < n; k++) {
        h_bc(k) = faces_host[k].i_bc;
        h_set(k) = faces_host[k].set;
        for (size_t d = 0; d < 3; d++) h_x(k, d) = d < N_DIM ? faces_host[k].x[d] : 0.0;
    }
    std::vector<const expr::Program *> all;
    set_type = Kokkos::View<uint8_t *>("timed_set_type", n_sets);
    set_slot = Kokkos::View<int32_t *[BoundaryFields::N_SLOTS]>("timed_set_slot", n_sets);
    set_composition = Kokkos::View<int32_t *[4]>("timed_set_composition", n_sets);
    auto h_type = Kokkos::create_mirror_view(set_type);
    auto h_slot = Kokkos::create_mirror_view(set_slot);
    auto h_comp = Kokkos::create_mirror_view(set_composition);
    std::vector<uint32_t> species, program;
    for (size_t s = 0; s < n_sets; s++) {
        h_type(s) = static_cast<uint8_t>(sets[s]->get_type());
        for (int k = 0; k < BoundaryFields::N_SLOTS; k++) {
            const BoundaryValue * v = sets[s]->slot(k);
            h_slot(s, k) = v ? static_cast<int32_t>(all.size()) : -1;
            if (v) all.push_back(v->compiled().get());
        }
        const CompositionPrograms * c = compositions[s].get();
        h_comp(s, 0) = static_cast<int32_t>(species.size());
        h_comp(s, 1) = c ? static_cast<int32_t>(c->species.size()) : 0;
        h_comp(s, 2) = c ? c->balance : -1;
        h_comp(s, 3) = c && c->mole ? 1 : 0;
        for (size_t i = 0; c && i < c->species.size(); i++) {
            species.push_back(c->species[i]);
            program.push_back(static_cast<uint32_t>(all.size()));
            all.push_back(c->fractions[i].get());
        }
    }
    composition_species = Kokkos::View<uint32_t *>("timed_composition_species", species.size());
    composition_program = Kokkos::View<uint32_t *>("timed_composition_program", program.size());
    auto h_species = Kokkos::create_mirror_view(composition_species);
    auto h_program = Kokkos::create_mirror_view(composition_program);
    for (size_t i = 0; i < species.size(); i++) {
        h_species(i) = species[i];
        h_program(i) = program[i];
    }
    Kokkos::deep_copy(face_bc, h_bc);
    Kokkos::deep_copy(face_set, h_set);
    Kokkos::deep_copy(face_x, h_x);
    Kokkos::deep_copy(set_type, h_type);
    Kokkos::deep_copy(set_slot, h_slot);
    Kokkos::deep_copy(set_composition, h_comp);
    Kokkos::deep_copy(composition_species, h_species);
    Kokkos::deep_copy(composition_program, h_program);
    programs = expr::ProgramTable<>(all);
    error_face = Kokkos::View<int32_t>("timed_error_face");
    error = Kokkos::View<double[3]>("timed_error");
}

namespace {

struct TimedFieldsFunctor {
    Kokkos::View<uint32_t *> face_bc;
    Kokkos::View<uint32_t *> face_set;
    Kokkos::View<double *[3]> face_x;
    Kokkos::View<uint8_t *> set_type;
    Kokkos::View<int32_t *[BoundaryFields::N_SLOTS]> set_slot;
    Kokkos::View<int32_t *[4]> set_composition;
    Kokkos::View<uint32_t *> composition_species;
    Kokkos::View<uint32_t *> composition_program;
    expr::ProgramTable<> programs;
    Kokkos::View<rtype> time;
    Kokkos::View<BoundaryCondition *> bcs;
    TimedBoundaryFields::Gas gas;
    Kokkos::View<int32_t> error_face;
    Kokkos::View<double[3]> error;

    KOKKOS_INLINE_FUNCTION void fail(const uint32_t k, const int slot, const double t, const double v) const {
        if (Kokkos::atomic_compare_exchange(&error_face(), 0, static_cast<int32_t>(k) + 1) == 0) {
            error(0) = t;
            error(1) = v;
            error(2) = slot;
        }
    }

    KOKKOS_INLINE_FUNCTION void operator()(const uint32_t k) const {
        constexpr uint8_t E = N_DIM + 1;
        const uint32_t i_bc = face_bc(k), s = face_set(k);
        const double t = static_cast<double>(time());
        const double vars[4] = {face_x(k, 0), face_x(k, 1), face_x(k, 2), t};
        const expr::ArrayVariables variables{vars};
        double value[BoundaryFields::N_SLOTS] = {};
        bool has[BoundaryFields::N_SLOTS] = {};
        for (int j = 0; j < BoundaryFields::N_SLOTS; j++) {
            const int32_t p = set_slot(s, j);
            has[j] = p >= 0;
            if (has[j]) value[j] = programs(static_cast<uint32_t>(p), variables);
        }
        BoundaryCondition & bc = bcs(i_bc);
        if (has[BoundaryFields::SLOT_U]) FOR_I_DIM bc.data[1 + i] = static_cast<rtype>(value[BoundaryFields::SLOT_U + i]);
        double R = gas.R;
        if (gas.mixture) {
            const int32_t first = set_composition(s, 0), count = set_composition(s, 1);
            if (count > 0) {
                const int32_t balance = set_composition(s, 2);
                const uint32_t n_species = static_cast<uint32_t>(gas.bc_Y.extent(1));
                const auto Y = Kokkos::subview(gas.bc_Y, i_bc, Kokkos::ALL());
                for (uint32_t i = 0; i < n_species; i++) Y(i) = 0.0_r;
                double sum = 0.0;
                for (int32_t i = 0; i < count; i++) {
                    const uint32_t sp = composition_species(static_cast<uint32_t>(first + i));
                    const double v = programs(composition_program(static_cast<uint32_t>(first + i)), variables);
                    Y(sp) = static_cast<rtype>(v > 0.0 ? v : 0.0);
                    if (static_cast<int32_t>(sp) != balance) sum += v > 0.0 ? v : 0.0;
                }
                if (balance >= 0) {
                    const double rest = sum < 1.0 ? 1.0 - sum : 0.0;
                    Y(static_cast<uint32_t>(balance)) = static_cast<rtype>(rest);
                    sum += rest;
                }
                if (!(sum > 0.0)) fail(k, BoundaryFields::N_SLOTS, t, sum);
                double norm = sum;
                if (set_composition(s, 3)) {
                    norm = 0.0;
                    for (uint32_t i = 0; i < n_species; i++) norm += double(Y(i)) / gas.thermo.inv_W(i);
                    for (uint32_t i = 0; i < n_species; i++) Y(i) = static_cast<rtype>(double(Y(i)) / gas.thermo.inv_W(i));
                }
                for (uint32_t i = 0; i < n_species; i++) Y(i) = static_cast<rtype>(double(Y(i)) / norm);
            }
            R = gas.thermo.gas_constant(BoundaryComposition{gas.bc_Y, static_cast<int32_t>(i_bc)});
        }
        auto positive = [&](const int j) {
            if (!(value[j] > 0.0)) fail(k, j, t, value[j]);
            return value[j];
        };
        const BoundaryType type = static_cast<BoundaryType>(set_type(s));
        if (type == BoundaryType::UPT || type == BoundaryType::NSCBC_INLET) {
            if (has[BoundaryFields::SLOT_P] && has[BoundaryFields::SLOT_T]) {
                const double p = positive(BoundaryFields::SLOT_P), T = positive(BoundaryFields::SLOT_T);
                bc.data[0] = static_cast<rtype>(p / (R * T));
                bc.data[E] = static_cast<rtype>(p);
                if (type == BoundaryType::NSCBC_INLET) bc.relax[BoundaryCondition::T_TARGET] = static_cast<rtype>(T);
            }
        } else if (type == BoundaryType::STAGNATION_INLET) {
            if (has[BoundaryFields::SLOT_P]) bc.data[E] = static_cast<rtype>(positive(BoundaryFields::SLOT_P));
            if (has[BoundaryFields::SLOT_T]) bc.data[0] = static_cast<rtype>(positive(BoundaryFields::SLOT_T));
        } else if (type == BoundaryType::MASS_FLOW_INLET) {
            if (has[BoundaryFields::SLOT_T]) bc.data[0] = static_cast<rtype>(positive(BoundaryFields::SLOT_T));
            if (has[BoundaryFields::SLOT_FLUX]) {
                if (!(value[BoundaryFields::SLOT_FLUX] >= 0.0)) fail(k, BoundaryFields::SLOT_FLUX, t, value[BoundaryFields::SLOT_FLUX]);
                bc.data[E] = static_cast<rtype>(value[BoundaryFields::SLOT_FLUX]);
            }
        }
        if (!gas.mixture || !has[BoundaryFields::SLOT_T]) return;
        const BoundaryComposition y{gas.bc_Y, static_cast<int32_t>(i_bc)};
        const double T = value[BoundaryFields::SLOT_T];
        double e, cv;
        gas.thermo.e_cv(T, y, e, cv);
        gas.bc_thermo(i_bc, 0) = static_cast<rtype>((cv + R) / cv);
        gas.bc_thermo(i_bc, 1) = static_cast<rtype>(e - cv * T);
        if (gas.transport.extent(0) == 0) return;
        const uint32_t n_species = static_cast<uint32_t>(gas.bc_Y.extent(1));
        double moles = 0.0;
        for (uint32_t i = 0; i < n_species; i++) moles += y(i) * gas.thermo.inv_W(i);
        gas.transport(i_bc, 0) = static_cast<rtype>(T);
        for (uint32_t i = 0; i < n_species; i++) {
            gas.transport(i_bc, 1 + i) = static_cast<rtype>(y(i) * gas.thermo.inv_W(i) / moles);
        }
    }
};

struct InletTargetsFunctor {
    Kokkos::View<int32_t *> chars;
    Kokkos::View<double *[3]> x;
    Kokkos::View<uint32_t *[N_DIM]> slots;
    expr::ProgramTable<> programs;
    Kokkos::View<rtype *[N_DIM]> target;
    double t;

    KOKKOS_INLINE_FUNCTION void operator()(const uint32_t k) const {
        const double vars[4] = {x(k, 0), x(k, 1), x(k, 2), t};
        for (uint8_t d = 0; d < N_DIM; d++) {
            target(chars(k), d) = static_cast<rtype>(programs(slots(k, d), expr::ArrayVariables{vars}));
        }
    }
};

} // namespace

void TimedBoundaryFields::apply(const Kokkos::View<rtype> & time, const Kokkos::View<BoundaryCondition *> & bcs,
                                const Gas & gas) const {
    const TimedFieldsFunctor functor{face_bc,  face_set, face_x, set_type, set_slot, set_composition,
                                     composition_species, composition_program, programs, time, bcs, gas,
                                     error_face, error};
    Kokkos::parallel_for("timed_boundary_fields", static_cast<uint32_t>(faces_host.size()), functor);
}

void TimedBoundaryFields::check() const {
    if (faces_host.empty()) return;
    int32_t face = 0;
    Kokkos::deep_copy(face, error_face);
    if (face == 0) return;
    const auto h_error = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), error);
    const FaceHost & f = faces_host[static_cast<size_t>(face - 1)];
    const BoundaryFields & fields = *sets[f.set];
    const int slot = static_cast<int>(h_error(2));
    std::string at = " at t = " + real(h_error(0)) + ", face (";
    for (size_t d = 0; d < N_DIM; d++) at += (d ? ", " : "") + real(f.x[d]);
    at += ")";
    if (slot == BoundaryFields::N_SLOTS) {
        throw InputError(fields.get_where() + ": the composition is zero" + at + ".");
    }
    const std::string key = fields.slot_key(slot);
    throw InputError(fields.get_where() + "." + key + " must be " +
                     (slot == BoundaryFields::SLOT_FLUX ? "non-negative" : "positive") + " (" + real(h_error(1)) + at + ").");
}

bool TimedInletTargets::depends_on_time(const Input & u, const std::string & where) {
    if (!u.is_array()) return false;
    for (size_t d = 0; d < u.as_array().size(); d++) {
        const Input & v = u.as_array()[d];
        if (!BoundaryValue::is_expression(v)) continue;
        const expr::Program p(where + ".u[" + std::to_string(d) + "]", v.as_string(), BoundaryValue::variables());
        if (p.uses(3)) return true;
    }
    return false;
}

void TimedInletTargets::add(const Input & u, const std::string & where, const std::vector<int32_t> & chars_in,
                            const std::vector<Point> & x_in) {
    std::array<uint32_t, N_DIM> slot{};
    FOR_I_DIM {
        slot[i] = static_cast<uint32_t>(host_programs.size());
        host_programs.push_back(
            BoundaryValue(u.as_array()[i], where + ".u[" + std::to_string(i) + "]", dim::velocity).compiled());
    }
    for (size_t k = 0; k < chars_in.size(); k++) {
        host_chars.push_back(chars_in[k]);
        host_x.push_back(x_in[k]);
        host_slots.push_back(slot);
    }
}

void TimedInletTargets::finish() {
    const size_t n = host_chars.size();
    chars = Kokkos::View<int32_t *>("timed_inlet_chars", n);
    x = Kokkos::View<double *[3]>("timed_inlet_x", n);
    slots = Kokkos::View<uint32_t *[N_DIM]>("timed_inlet_slots", n);
    auto h_chars = Kokkos::create_mirror_view(chars);
    auto h_x = Kokkos::create_mirror_view(x);
    auto h_slots = Kokkos::create_mirror_view(slots);
    for (size_t k = 0; k < n; k++) {
        h_chars(k) = host_chars[k];
        for (size_t d = 0; d < 3; d++) h_x(k, d) = d < N_DIM ? host_x[k][d] : 0.0;
        FOR_I_DIM h_slots(k, i) = host_slots[k][i];
    }
    Kokkos::deep_copy(chars, h_chars);
    Kokkos::deep_copy(x, h_x);
    Kokkos::deep_copy(slots, h_slots);
    std::vector<const expr::Program *> all;
    for (const auto & p : host_programs) all.push_back(p.get());
    programs = expr::ProgramTable<>(all);
}

void TimedInletTargets::fill(const double t, const Kokkos::View<rtype *[N_DIM]> & target) const {
    if (host_chars.empty()) return;
    Kokkos::parallel_for("timed_inlet_targets", static_cast<uint32_t>(host_chars.size()),
                         InletTargetsFunctor{chars, x, slots, programs, target, t});
}
