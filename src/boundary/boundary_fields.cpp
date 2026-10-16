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

#include "input.h"

BoundaryValue::BoundaryValue(const toml::value & v, const std::string & name) {
    if (v.is_string()) {
        expression = std::make_shared<Expression>(name, v.as_string());
    } else {
        value = as_double(v, name);
    }
}

double BoundaryValue::at(const Point & x, const double t) const {
    if (!expression) return value;
    return (*expression)(x[0], x[1], N_DIM > 2 ? x[N_DIM - 1] : 0.0, t);
}

BoundaryFields::BoundaryFields(const toml::value & input, const BoundaryType type_, const std::string & where_)
    : type(type_), where(where_) {
    auto read = [&](const char * key, BoundaryValue & v, bool & has) {
        if (!input.contains(key)) return;
        v = BoundaryValue(input.at(key), where + "." + key);
        has = true;
    };
    const bool walls = type == BoundaryType::WALL_ADIABATIC || type == BoundaryType::WALL_ISOTHERMAL ||
                       type == BoundaryType::WALL_HEAT_FLUX;
    if ((type == BoundaryType::UPT || walls) && input.contains("u")) {
        const toml::value & u_in = input.at("u");
        if (!u_in.is_array() || u_in.as_array().size() != N_DIM) {
            throw InputError(where + ".u must have " + std::to_string(N_DIM) + " components (numbers or expressions).");
        }
        FOR_I_DIM u.emplace_back(u_in.as_array()[i], where + ".u[" + std::to_string(i) + "]");
    }
    if (type == BoundaryType::UPT || type == BoundaryType::NSCBC_INLET) {
        read("p", p, has_p);
        read("T", T, has_T);
    } else if (type == BoundaryType::STAGNATION_INLET) {
        read("p0", p, has_p);
        read("T0", T, has_T);
    } else if (type == BoundaryType::MASS_FLOW_INLET) {
        read("T", T, has_T);
        read("mass_flux", flux, has_flux);
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

double BoundaryFields::temperature(const Point & x, const double t) const { return has_T ? T.at(x, t) : 0.0; }

void BoundaryFields::apply(BoundaryCondition & bc, const Point & x, const double t, const double R,
                           const bool numbers_only) const {
    constexpr uint8_t E = N_DIM + 1;
    auto use = [&](const BoundaryValue & v) { return !numbers_only || !v.varies(); };
    auto positive = [&](const BoundaryValue & v, const char * key) {
        const double value = v.at(x, t);
        if (!(value > 0.0)) {
            throw InputError(where + "." + key + " must be positive (" + std::to_string(value) + " at t = " +
                             std::to_string(t) + ").");
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
