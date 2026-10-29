/**
 * @file field_units.cpp
 * @brief The table of output fields' dimensions and descriptions.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "field_units.h"

#include "common_typedef.h"

#include <cctype>
#include <sstream>
#include <stdexcept>

namespace {

struct Entry {
    const char * name;
    Dimension dimension;
    const char * long_name;
};

constexpr Dimension momentum_density = dim::mass_flux;
constexpr Dimension energy_density = dim::pressure;
constexpr Dimension thermal_conductivity = dim::of(1, 1, -3, -1);
constexpr Dimension diffusivity = dim::of(0, 2, -1);
constexpr Dimension thermal_diffusion = dim::of(1, -1, -1);

// Vectors (RHOU, U, VORTICITY in 3D) also name their components <name>_X, _Y, _Z
const Entry VECTORS[] = {
    {"RHOU", momentum_density, "momentum density"},
    {"U", dim::velocity, "velocity"},
    {"VORTICITY", dim::rate, "vorticity"},
};

const Entry NAMED[] = {
    {"RHO", dim::density, "density"},
    {"RHOE", energy_density, "total energy density"},
    {"P", dim::pressure, "pressure"},
    {"T", dim::temperature, "temperature"},
    {"H", dim::velocity_squared, "specific enthalpy"},
    {"CFL", dim::none, "local CFL number"},
    {"Q", dim::rate.times(dim::rate), "Q-criterion"},
    {"TENO_SIGMA", dim::none, "TENO troubled-cell indicator"},
    {"MU", dim::dynamic_viscosity, "dynamic viscosity"},
    {"LAMBDA", thermal_conductivity, "thermal conductivity"},
    {"T_SEED", dim::temperature, "temperature guess of the thermodynamic state"},
    {"HRR", dim::power_density, "heat release rate"},
    {"CHEM_H", dim::time, "last chemistry sub-step"},
    {"CHEM_COST", dim::none, "chemistry sub-steps in the last step"},
    {"QRAD", dim::power_density, "radiative heat source"},
    {"MU_T", dim::dynamic_viscosity, "eddy viscosity"},
    {"TF_F", dim::none, "flame thickening factor"},
    {"TF_E", dim::none, "flame wrinkling efficiency"},
    {"TF_OMEGA", dim::none, "flame sensor"},
    {"PASR_KAPPA", dim::none, "PaSR reacting fraction"},
    {"P_MAX", dim::pressure, "largest pressure seen"},
};

// <prefix><species>
const Entry SPECIES[] = {
    {"Y_", dim::none, "mass fraction of "},
    {"X_", dim::none, "mole fraction of "},
    {"RHOY_", dim::density, "partial density of "},
    {"D_", diffusivity, "diffusion coefficient of "},
    {"DT_", thermal_diffusion, "thermal diffusion coefficient of "},
    {"OMEGA_", dim::mass_source, "net production rate of "},
};

bool starts_with(const std::string & s, const std::string & prefix) { return s.rfind(prefix, 0) == 0; }

std::optional<FieldDescription> find_base(const std::string & name) {
    for (const auto & e : NAMED) {
        if (name == e.name) return FieldDescription{e.dimension, e.long_name, ""};
    }
    for (const auto & e : VECTORS) {
        const std::string stem = e.name;
        if (name == stem) return FieldDescription{e.dimension, e.long_name, ""};
        for (const char * axis : {"X", "Y", "Z"}) {
            if (name == stem + "_" + axis) {
                const char lower = static_cast<char>(std::tolower(static_cast<unsigned char>(axis[0])));
                return FieldDescription{e.dimension, std::string(1, lower) + " component of " + e.long_name, ""};
            }
        }
    }
    for (const auto & e : SPECIES) {
        if (starts_with(name, e.name) && name.size() > std::string(e.name).size()) {
            return FieldDescription{e.dimension, e.long_name + name.substr(std::string(e.name).size()), ""};
        }
    }
    return std::nullopt;
}

/** @brief A field that statistics and probes sample: RHO, a primitive or Y_<species>. */
std::optional<FieldDescription> find_sampled(const std::string & name) {
    bool sampled = name == "RHO" || (starts_with(name, "Y_") && name.size() > 2);
    for (const auto & primitive : PRIMITIVE_NAMES) sampled = sampled || name == primitive;
    return sampled ? find_base(name) : std::nullopt;
}

} // namespace

namespace field_units {

std::optional<FieldDescription> find(const std::string & name) {
    if (auto base = find_base(name)) return base;
    if (starts_with(name, "SOURCE_")) {
        auto of = find_base(name.substr(7));
        if (!of) return std::nullopt;
        return FieldDescription{of->dimension.times(dim::time, -1), "source of " + of->long_name, ""};
    }
    if (starts_with(name, "MEAN_")) {
        auto of = find_sampled(name.substr(5));
        if (!of) return std::nullopt;
        return FieldDescription{of->dimension, "time average of " + of->long_name, ""};
    }
    if (starts_with(name, "COV_")) {
        // COV_<A>_<B>: the first split whose halves are both sampled fields (names hold underscores)
        const std::string pair = name.substr(4);
        for (size_t at = pair.find('_'); at != std::string::npos; at = pair.find('_', at + 1)) {
            const auto a = find_sampled(pair.substr(0, at)), b = find_sampled(pair.substr(at + 1));
            if (a && b) {
                return FieldDescription{a->dimension.times(b->dimension),
                                        "covariance of " + a->long_name + " and " + b->long_name, ""};
            }
        }
        return std::nullopt;
    }
    if (starts_with(name, "PHASE_")) {
        const size_t at = name.find('_', 6);
        if (at == std::string::npos || at == 6) return std::nullopt;
        const std::string bin = name.substr(6, at - 6);
        for (const char c : bin) {
            if (!std::isdigit(static_cast<unsigned char>(c))) return std::nullopt;
        }
        auto of = find_sampled(name.substr(at + 1));
        if (!of) return std::nullopt;
        return FieldDescription{of->dimension, "phase average of " + of->long_name + " in bin " + bin, ""};
    }
    return std::nullopt;
}

FieldDescription describe(const std::string & name) {
    if (auto d = find(name)) return *d;
    throw std::logic_error("field_units: no units for the output field \"" + name +
                           "\"; add it to the table in src/io/field_units.cpp.");
}

std::string csv_line(const std::vector<Dimension> & columns) {
    std::string line = "# units: ";
    for (size_t i = 0; i < columns.size(); i++) line += (i ? "," : "") + columns[i].udunits();
    return line;
}

std::string xml_escape(const std::string & text) {
    std::string out;
    for (const char c : text) {
        switch (c) {
            case '&': out += "&amp;"; break;
            case '<': out += "&lt;"; break;
            case '>': out += "&gt;"; break;
            case '"': out += "&quot;"; break;
            default: out += c;
        }
    }
    return out;
}

std::string vtk_field_data(const std::vector<std::string> & names, const std::vector<FieldDescription> & fields,
                           const std::string & indent) {
    // VTK writes ASCII string arrays as their bytes, each string ended by a 0
    auto array = [&](const char * array_name, auto && entry) {
        std::ostringstream out;
        out << indent << "<Array type=\"String\" Name=\"" << array_name << "\" NumberOfTuples=\"" << names.size()
            << "\" format=\"ascii\">";
        for (size_t i = 0; i < names.size(); i++) {
            for (const char c : entry(i)) out << static_cast<int>(static_cast<unsigned char>(c)) << " ";
            out << (i + 1 < names.size() ? "0 " : "0");
        }
        out << "</Array>\n";
        return out.str();
    };
    return array("MALLARD_FIELD_NAMES", [&](size_t i) { return names[i]; }) +
           array("MALLARD_FIELD_UNITS", [&](size_t i) { return fields[i].units(); }) +
           array("MALLARD_FIELD_LONG_NAMES", [&](size_t i) { return fields[i].long_name; });
}

} // namespace field_units
