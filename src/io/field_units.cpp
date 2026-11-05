/**
 * @file field_units.cpp
 * @brief The table of output fields' dimensions and descriptions.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "field_units.h"

#include "common_typedef.h"

#include <algorithm>
#include <cctype>
#include <iterator>
#include <sstream>
#include <stdexcept>

namespace {

struct Entry {
    const char * name;
    Dimension dimension;
    const char * long_name;
    const char * group = "";  // where mallard list fields files it; empty is an error (FieldUnitsTest)
    const char * needs = "";  // what a case needs for the field, "" if every case has it
    const char * non_si_units = "";
};

constexpr Dimension momentum_density = dim::mass_flux;
constexpr Dimension energy_density = dim::pressure;
constexpr Dimension thermal_conductivity = dim::of(1, 1, -3, -1);
constexpr Dimension diffusivity = dim::of(0, 2, -1);
constexpr Dimension thermal_diffusion = dim::of(1, -1, -1);

constexpr const char * CONSERVATIVE = "conservative";
constexpr const char * PRIMITIVE = "primitive";
constexpr const char * DERIVED = "derived";
constexpr const char * TRANSPORT = "transport";
constexpr const char * MIXTURE = "mixture";
constexpr const char * CHEMISTRY = "chemistry";
constexpr const char * LES = "LES";
constexpr const char * SURFACE = "surface probes";
constexpr const char * QUALITY = "mesh quality";
constexpr const char * WALLS = "[[probes]] on a wall zone";
constexpr const char * SURFACES = "[[probes]] on a surface or boundary zone";

// Vectors (RHOU, U, VORTICITY in 3D) also name their components <name>_X, _Y, _Z
const Entry VECTORS[] = {
    {"RHOU", momentum_density, "momentum density", CONSERVATIVE, ""},
    {"U", dim::velocity, "velocity", PRIMITIVE, ""},
    {"VORTICITY", dim::rate, "vorticity", DERIVED, ""},
    {"MOMENTUM_FLUX", dim::pressure, "normal momentum flux", SURFACE, SURFACES},
    {"TRACTION_PRESSURE", dim::pressure, "pressure traction on the boundary", SURFACE, WALLS},
    {"TRACTION_VISCOUS", dim::pressure, "viscous traction on the boundary", SURFACE, WALLS},
    {"TAU_W", dim::pressure, "wall shear stress", SURFACE, WALLS},
};

const Entry NAMED[] = {
    {"RHO", dim::density, "density", CONSERVATIVE, ""},
    {"RHOE", energy_density, "total energy density", CONSERVATIVE, ""},
    {"P", dim::pressure, "pressure", PRIMITIVE, ""},
    {"T", dim::temperature, "temperature", PRIMITIVE, ""},
    {"H", dim::velocity_squared, "specific enthalpy", PRIMITIVE, ""},
    {"CFL", dim::none, "local CFL number", DERIVED, ""},
    {"MACH", dim::none, "Mach number", DERIVED, ""},
    {"Q_CRITERION", dim::rate.times(dim::rate), "Q-criterion", DERIVED, ""},
    {"LAMBDA2", dim::rate.times(dim::rate), "lambda2 vortex criterion", DERIVED, ""},
    {"SCHLIEREN", dim::of(1, -4, 0), "density gradient magnitude", DERIVED, ""},
    {"P_MAX", dim::pressure, "largest pressure seen", DERIVED, ""},
    {"TENO_SIGMA", dim::none, "TENO troubled-cell indicator", DERIVED, "face_reconstruction type = \"TENO\""},
    {"MU", dim::dynamic_viscosity, "dynamic viscosity", TRANSPORT, "a mixture with navier_stokes"},
    {"LAMBDA", thermal_conductivity, "thermal conductivity", TRANSPORT,
     "a mixture with navier_stokes"},
    {"T_SEED", dim::temperature, "temperature guess of the thermodynamic state", MIXTURE, "a mixture"},
    {"HRR", dim::power_density, "heat release rate", CHEMISTRY, "a mixture with [chemistry]"},
    {"CHEM_H", dim::time, "last chemistry sub-step", CHEMISTRY, "a mixture with [chemistry]"},
    {"CHEM_COST", dim::none, "chemistry sub-steps in the last step", CHEMISTRY, "a mixture with [chemistry]"},
    {"QRAD", dim::power_density, "radiative heat source", CHEMISTRY, "[radiation]"},
    {"MU_T", dim::dynamic_viscosity, "eddy viscosity", LES, "[les]"},
    {"TF_F", dim::none, "flame thickening factor", LES, "[les.combustion] (TFLES)"},
    {"TF_E", dim::none, "flame wrinkling efficiency", LES, "[les.combustion] (TFLES)"},
    {"TF_OMEGA", dim::none, "flame sensor", LES, "[les.combustion] (TFLES)"},
    {"PASR_KAPPA", dim::none, "PaSR reacting fraction", LES, "[les.combustion] model = \"pasr\""},
    {"REGION_FRACTION", dim::none, "volume fraction of the region in the cell", DERIVED,
     "a [[write_data]] region, which writes it"},
    {"MASS_FLUX", dim::mass_flux, "normal mass flux", SURFACE, SURFACES},
    {"ENTHALPY_FLUX", dim::heat_flux, "normal total enthalpy flux", SURFACE, SURFACES},
    {"TAU_W", dim::pressure, "wall shear stress magnitude", SURFACE, WALLS},
    {"CF", dim::none, "skin friction coefficient", SURFACE, WALLS},
    {"CP", dim::none, "pressure coefficient", SURFACE, WALLS},
    {"Q_W", dim::heat_flux, "wall heat flux into the fluid", SURFACE, WALLS},
    {"U_TAU", dim::velocity, "friction velocity", SURFACE, WALLS},
    {"Y_PLUS", dim::none, "first-cell y+", SURFACE, WALLS},
    {"NON_ORTHOGONALITY", dim::of(0, 0, 0, 0, 0, 1), "largest face non-orthogonality", QUALITY,
     "mallard check or info --quality", "degree"},
    {"SKEWNESS", dim::none, "largest face skewness", QUALITY, "mallard check or info --quality"},
    {"ASPECT_RATIO", dim::none, "cell aspect ratio", QUALITY, "mallard check or info --quality"},
    {"VOLUME_RATIO", dim::none, "largest volume ratio to a neighbor", QUALITY, "mallard check or info --quality"},
    {"DELTA_OVER_ETA", dim::none, "cell size over the estimated Kolmogorov length", QUALITY,
     "mallard check --quality (resolution estimates)"},
    {"CELLS_PER_DELTA_L", dim::none, "cells across the laminar flame thickness", QUALITY,
     "mallard check --quality (resolution estimates) with [les.combustion] (TFLES)"},
    {"DIVERGED", dim::none, "1 in the cells that became invalid", DERIVED,
     "a run that diverges, whose divergence report snapshot writes it"},
};

// <prefix><species>
const Entry SPECIES[] = {
    {"Y_", dim::none, "mass fraction of ", MIXTURE, "a mixture"},
    {"X_", dim::none, "mole fraction of ", MIXTURE, "a mixture"},
    {"RHOY_", dim::density, "partial density of ", CONSERVATIVE, "a mixture"},
    {"D_", diffusivity, "diffusion coefficient of ", TRANSPORT, "a mixture with navier_stokes"},
    {"DT_", thermal_diffusion, "thermal diffusion coefficient of ", TRANSPORT,
     "a mixture with navier_stokes and soret = true"},
    {"OMEGA_", dim::mass_source, "net production rate of ", CHEMISTRY, "a mixture with [chemistry]"},
    {"SPECIES_FLUX_", dim::mass_flux, "normal mass flux of ", SURFACE,
     "a mixture; [[probes]] on a surface or boundary zone"},
};

bool starts_with(const std::string & s, const std::string & prefix) { return s.rfind(prefix, 0) == 0; }

std::optional<FieldDescription> find_base(const std::string & name) {
    for (const auto & e : NAMED) {
        if (name == e.name) return FieldDescription{e.dimension, e.long_name, "", e.non_si_units};
    }
    for (const auto & e : VECTORS) {
        const std::string stem = e.name;
        if (name == stem) return FieldDescription{e.dimension, e.long_name, "", ""};
        for (const char * axis : {"X", "Y", "Z"}) {
            if (name == stem + "_" + axis) {
                const char lower = static_cast<char>(std::tolower(static_cast<unsigned char>(axis[0])));
                return FieldDescription{e.dimension, std::string(1, lower) + " component of " + e.long_name, "", ""};
            }
        }
    }
    for (const auto & e : SPECIES) {
        if (starts_with(name, e.name) && name.size() > std::string(e.name).size()) {
            return FieldDescription{e.dimension, e.long_name + name.substr(std::string(e.name).size()), "", ""};
        }
    }
    return std::nullopt;
}

/**
 * @brief A field that statistics and probes sample (a conservative, a primitive,
 *        Y_<species> or RHOY_<species>), or the vector U or RHOU.
 */
std::optional<FieldDescription> find_sampled(const std::string & name) {
    bool sampled = name == "U" || name == "RHOU" || (starts_with(name, "Y_") && name.size() > 2) ||
                   (starts_with(name, "RHOY_") && name.size() > 5);
    for (const auto & conservative : CONSERVATIVE_NAMES) sampled = sampled || name == conservative;
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
        return FieldDescription{of->dimension.times(dim::time, -1), "source of " + of->long_name, "", ""};
    }
    if (starts_with(name, "MEAN_")) {
        // Of a [statistics] field, or the time average of a probe variable
        auto of = find_sampled(name.substr(5));
        if (!of) of = find_base(name.substr(5));
        if (!of) return std::nullopt;
        return FieldDescription{of->dimension, "time average of " + of->long_name, "", ""};
    }
    if (starts_with(name, "COV_")) {
        // COV_<A>_<B>: the first split whose halves are both sampled fields (names hold underscores)
        const std::string pair = name.substr(4);
        for (size_t at = pair.find('_'); at != std::string::npos; at = pair.find('_', at + 1)) {
            const auto a = find_sampled(pair.substr(0, at)), b = find_sampled(pair.substr(at + 1));
            if (a && b) {
                return FieldDescription{a->dimension.times(b->dimension),
                                        "covariance of " + a->long_name + " and " + b->long_name, "", ""};
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
        return FieldDescription{of->dimension, "phase average of " + of->long_name + " in bin " + bin, "", ""};
    }
    return std::nullopt;
}

size_t group_rank(const std::string & group) {
    const char * order[] = {CONSERVATIVE, PRIMITIVE, DERIVED, TRANSPORT, MIXTURE, CHEMISTRY, LES,
                            "sources", "statistics", SURFACE, QUALITY};
    for (size_t i = 0; i < std::size(order); i++) {
        if (group == order[i]) return i;
    }
    return std::size(order);
}

std::string group(const std::string & name) {
    for (const auto & e : NAMED) {
        if (name == e.name) return e.group;
    }
    for (const auto & e : VECTORS) {
        const std::string stem = e.name;
        for (const char * suffix : {"", "_X", "_Y", "_Z"}) {
            if (name == stem + suffix) return e.group;
        }
    }
    for (const auto & e : SPECIES) {
        if (starts_with(name, e.name) && name.size() > std::string(e.name).size()) return e.group;
    }
    if (starts_with(name, "SOURCE_")) return "sources";
    for (const char * prefix : {"MEAN_", "COV_", "PHASE_"}) {
        if (starts_with(name, prefix)) return "statistics";
    }
    return "";
}

std::vector<CatalogField> catalog() {
    std::vector<CatalogField> out;
    auto add = [&](const std::string & name, const Entry & e, const std::string & long_name) {
        const std::string units = e.non_si_units[0] ? e.non_si_units : e.dimension.udunits();
        out.push_back({name, units, long_name, e.group, e.needs});
    };
    for (const auto & e : NAMED) add(e.name, e, e.long_name);
    for (const auto & e : VECTORS) {
        if (find_base(e.name)->long_name == e.long_name) add(e.name, e, e.long_name + std::string(" (vector)"));
    }
    for (const auto & e : SPECIES) add(e.name + std::string("<s>"), e, e.long_name + std::string("species s"));
    const Entry derived[] = {
        {"SOURCE_<A>", dim::none, "", "sources", "[[sources]] or [source]"},
        {"MEAN_<A>", dim::none, "", "statistics", "[statistics] fields with A, or [[probes]] averaging A"},
        {"COV_<A>_<B>", dim::none, "", "statistics", "[statistics] products with A*B"},
        {"PHASE_<bin>_<A>", dim::none, "", "statistics", "[statistics.phase]"},
    };
    const char * long_names[] = {"source of A, per second", "time average of A", "covariance of A and B",
                                 "phase average of A in a bin"};
    const char * units[] = {"A's s-1", "A's", "A's times B's", "A's"};
    for (size_t i = 0; i < std::size(derived); i++) {
        out.push_back({derived[i].name, units[i], long_names[i], derived[i].group, derived[i].needs});
    }
    std::stable_sort(out.begin(), out.end(), [](const CatalogField & a, const CatalogField & b) {
        return group_rank(a.group) < group_rank(b.group);
    });
    return out;
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
