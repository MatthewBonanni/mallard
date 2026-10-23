/**
 * @file mechanism.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Gas-phase mechanisms read from Cantera YAML files.
 * @version 0.3
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "mechanism.h"

#include <algorithm>
#include <cstdlib>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

#include "units.h"

namespace chemistry {

namespace {

struct ElementWeight {
    const char * symbol;
    double weight;
};

// Cantera 3.2.0, ct.Element(symbol).weight (elements without one are omitted)
constexpr ElementWeight ELEMENT_WEIGHTS[] = {
#include "element_weights.inc"
};

std::string lower(std::string s) {
    for (char & c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

/** @brief A parsed YAML file and its default units. */
struct Source {
    std::string path;
    YAML::Node root;
    UnitSystem units;
};

/**
 * @brief One side of a reaction equation: species with coefficients, and the
 *        third body ("M" for "+ M", the species or "M" of "(+ X)").
 */
struct EquationSide {
    std::vector<std::pair<std::string, double>> species;
    std::string third_body;
    bool falloff = false;
};

EquationSide parse_side(const std::string & text, const std::string & where) {
    EquationSide side;
    // Tokens separated by spaces; "+" alone separates terms; "(+M)" may be split as "(+" "M)"
    std::vector<std::string> tokens;
    std::string token;
    for (size_t i = 0; i <= text.size(); i++) {
        if (i == text.size() || std::isspace(static_cast<unsigned char>(text[i]))) {
            if (!token.empty()) tokens.push_back(token);
            token.clear();
        } else {
            token += text[i];
        }
    }
    double coefficient = 1.0;
    bool have_coefficient = false;
    for (size_t i = 0; i < tokens.size(); i++) {
        std::string t = tokens[i];
        if (t == "+") continue;
        if (t.rfind("(+", 0) == 0) {
            // "(+M)" or "(+X)", possibly split into "(+" and "M)"
            std::string body = t.substr(2);
            if (body.empty() && i + 1 < tokens.size()) body = tokens[++i];
            if (body.empty() || body.back() != ')') throw std::runtime_error(where + ": malformed falloff third body");
            side.third_body = body.substr(0, body.size() - 1);
            side.falloff = true;
            continue;
        }
        if (t == "M") {
            side.third_body = "M";
            continue;
        }
        char * stop = nullptr;
        const double value = std::strtod(t.c_str(), &stop);
        if (stop != t.c_str() && *stop == '\0') {
            coefficient = value;
            have_coefficient = true;
            continue;
        }
        if (!have_coefficient && stop != t.c_str() && std::isalpha(static_cast<unsigned char>(*stop))) {
            // "2O2": a coefficient written against the species name
            coefficient = value;
            t = std::string(stop);
        }
        side.species.emplace_back(t, coefficient);
        coefficient = 1.0;
        have_coefficient = false;
    }
    return side;
}

void merge(std::vector<std::pair<uint32_t, double>> & terms, uint32_t k, double nu) {
    for (auto & [species, coefficient] : terms) {
        if (species == k) {
            coefficient += nu;
            return;
        }
    }
    terms.emplace_back(k, nu);
}

const Dimension PRESSURE{0, 0, 0, 0, 0, 0, 1.0};

class Reader {
    public:
        explicit Reader(const std::string & file) { main = load(file); }

        Mechanism read(const std::string & phase_name);

    private:
        std::shared_ptr<Source> load(const std::string & path);
        std::shared_ptr<Source> source_of(const std::string & spec, std::string & section);
        void add_species(const YAML::Node & entry, const Source & source, const std::string & section);
        Species parse_species(const YAML::Node & node, const Source & source) const;
        SpeciesThermo parse_thermo(const YAML::Node & node, const Source & source, const std::string & where) const;
        void add_reactions(const YAML::Node & entry, const Source & source, const std::string & section);
        bool parse_reaction(const YAML::Node & node, const Source & source, bool declared_only, Reaction & r) const;

        std::map<std::string, std::shared_ptr<Source>> sources;
        std::shared_ptr<Source> main;
        std::map<std::string, double> custom_weights;  // by lower-case symbol
        Mechanism mech;
};

std::shared_ptr<Source> Reader::load(const std::string & path) {
    const std::string key = std::filesystem::weakly_canonical(path).string();
    if (auto it = sources.find(key); it != sources.end()) return it->second;
    auto source = std::make_shared<Source>();
    source->path = path;
    try {
        source->root = YAML::LoadFile(path);
        source->units = UnitSystem(source->root["units"]);
    } catch (const YAML::BadFile &) {
        throw std::runtime_error("Could not open mechanism file " + path + ".");
    } catch (const YAML::Exception & e) {
        throw std::runtime_error("Mechanism file " + path + ": " + e.what());
    }
    if (const YAML::Node elements = source->root["elements"]) {
        for (const auto & e : elements) {
            if (!e["symbol"] || !e["atomic-weight"]) {
                throw std::runtime_error("Mechanism file " + path + ": elements need a symbol and an atomic-weight.");
            }
            custom_weights[lower(e["symbol"].as<std::string>())] = e["atomic-weight"].as<double>();
        }
    }
    sources[key] = source;
    return source;
}

/**
 * @brief The file of a species section reference "section" or "file.yaml/section",
 *        relative to the main file's directory.
 */
std::shared_ptr<Source> Reader::source_of(const std::string & spec, std::string & section) {
    const size_t slash = spec.rfind('/');
    if (slash == std::string::npos) {
        section = spec;
        return main;
    }
    section = spec.substr(slash + 1);
    std::filesystem::path path = spec.substr(0, slash);
    if (path.is_relative()) path = std::filesystem::path(main->path).parent_path() / path;
    return load(path.string());
}

void Reader::add_species(const YAML::Node & entry, const Source & source, const std::string & section) {
    const YAML::Node list = source.root[section];
    if (!list || !list.IsSequence()) {
        throw std::runtime_error("Mechanism file " + source.path + " has no species section \"" + section + "\".");
    }
    auto add = [&](const YAML::Node & node) {
        Species sp = parse_species(node, source);
        if (mech.species_index(sp.name) >= 0) {
            throw std::runtime_error("Mechanism " + mech.file + ": species " + sp.name + " is listed twice.");
        }
        mech.species.push_back(std::move(sp));
    };
    if (entry.IsScalar() && entry.Scalar() == "all") {
        for (const auto & node : list) add(node);
        return;
    }
    if (!entry.IsSequence()) {
        throw std::runtime_error("Mechanism " + mech.file + ": a species list must be \"all\" or a list of names.");
    }
    for (const auto & name : entry) {
        bool found = false;
        for (const auto & node : list) {
            if (node["name"] && node["name"].as<std::string>() == name.as<std::string>()) {
                add(node);
                found = true;
                break;
            }
        }
        if (!found) {
            throw std::runtime_error("Mechanism file " + source.path + ": no species " + name.as<std::string>() +
                                     " in section \"" + section + "\".");
        }
    }
}

void Reader::add_reactions(const YAML::Node & entry, const Source & source, const std::string & section) {
    std::string mode = "all";
    if (entry && entry.IsScalar()) mode = entry.Scalar();
    if (mode == "none") return;
    if (mode != "all" && mode != "declared-species") {
        throw std::runtime_error("Mechanism " + mech.file + ": reactions \"" + mode +
                                 "\" is not one of all, declared-species, none.");
    }
    const YAML::Node list = source.root[section];
    if (!list) {
        if (section == "reactions") return;
        throw std::runtime_error("Mechanism file " + source.path + " has no reaction section \"" + section + "\".");
    }
    for (const auto & node : list) {
        Reaction r;
        if (parse_reaction(node, source, mode == "declared-species", r)) mech.reactions.push_back(std::move(r));
    }
}

bool Reader::parse_reaction(const YAML::Node & node, const Source & source, const bool declared_only,
                            Reaction & r) const {
    if (!node["equation"]) throw std::runtime_error("Mechanism file " + source.path + ": a reaction has no equation.");
    r.equation = node["equation"].as<std::string>();
    const std::string where = "Mechanism file " + source.path + ", reaction \"" + r.equation + "\"";
    std::string arrow;
    size_t at = std::string::npos;
    for (const char * a : {"<=>", "=>", "="}) {
        at = r.equation.find(a);
        if (at != std::string::npos) {
            arrow = a;
            break;
        }
    }
    if (at == std::string::npos) throw std::runtime_error(where + ": no <=>, => or = in the equation.");
    r.reversible = arrow != "=>";
    const EquationSide lhs = parse_side(r.equation.substr(0, at), where);
    const EquationSide rhs = parse_side(r.equation.substr(at + arrow.size()), where);
    if (lhs.third_body != rhs.third_body || lhs.falloff != rhs.falloff) {
        throw std::runtime_error(where + ": the third body differs between the sides.");
    }
    for (const auto & [side, terms] : {std::make_pair(&lhs, &r.reactants), std::make_pair(&rhs, &r.products)}) {
        for (const auto & [name, nu] : side->species) {
            const int32_t k = mech.species_index(name);
            if (k < 0) {
                if (declared_only) return false;
                throw std::runtime_error(where + ": no species " + name + " in the phase.");
            }
            merge(*terms, static_cast<uint32_t>(k), nu);
        }
    }

    const std::string type = node["type"] ? node["type"].as<std::string>() : "elementary";
    if (type == "falloff" || lhs.falloff) {
        if (!lhs.falloff || type != "falloff") throw std::runtime_error(where + ": falloff reactions need (+M).");
        r.type = ReactionType::FALLOFF;
    } else if (type == "three-body" || !lhs.third_body.empty()) {
        if (lhs.third_body.empty()) throw std::runtime_error(where + ": three-body reactions need + M.");
        r.type = ReactionType::THREE_BODY;
    } else if (type == "pressure-dependent-Arrhenius") {
        r.type = ReactionType::PLOG;
    } else if (type == "Chebyshev") {
        r.type = ReactionType::CHEBYSHEV;
    } else if (type != "elementary" && type != "reaction") {
        throw std::runtime_error(where + ": reaction type \"" + type + "\" is not supported (elementary, three-body, "
                                 "falloff, pressure-dependent-Arrhenius, Chebyshev).");
    }
    if (node["duplicate"]) r.duplicate = node["duplicate"].as<bool>();

    // Forward orders: the reactants' coefficients unless given
    r.orders = r.reactants;
    if (node["orders"]) {
        for (const auto & kv : node["orders"]) {
            const int32_t k = mech.species_index(kv.first.as<std::string>());
            bool found = false;
            for (auto & [species, order] : r.orders) {
                if (k >= 0 && species == static_cast<uint32_t>(k)) {
                    order = kv.second.as<double>();
                    found = true;
                }
            }
            if (!found) throw std::runtime_error(where + ": orders are only supported for reactants.");
        }
    }
    double order = 0.0;
    for (const auto & term : r.orders) order += term.second;

    // A has units concentration^(1 - n) / time for a rate of total order n
    auto arrhenius = [&](const YAML::Node & k, double n, const std::string & what) {
        if (!k || !k["A"]) throw std::runtime_error(where + ": missing " + what + ".");
        Arrhenius a;
        const Dimension dimension{0, 3.0 * (n - 1.0), -1.0, 0, 1.0 - n, 0, 0};
        a.A = source.units.convert(k["A"], dimension, where + " " + what + ".A");
        a.b = k["b"] ? k["b"].as<double>() : 0.0;
        a.Ea_R = k["Ea"] ? source.units.convert_activation_energy(k["Ea"], where + " " + what + ".Ea") : 0.0;
        if (a.A < 0.0 && !(node["negative-A"] && node["negative-A"].as<bool>())) {
            throw std::runtime_error(where + ": negative " + what + ".A.");
        }
        return a;
    };
    if (r.type == ReactionType::FALLOFF) {
        r.rate = arrhenius(node["high-P-rate-constant"], order, "high-P-rate-constant");
        r.low = arrhenius(node["low-P-rate-constant"], order + 1.0, "low-P-rate-constant");
        if (node["Troe"]) {
            const YAML::Node troe = node["Troe"];
            r.falloff = FalloffType::TROE;
            r.falloff_params = {troe["A"].as<double>(), troe["T3"].as<double>(), troe["T1"].as<double>(),
                                troe["T2"] ? troe["T2"].as<double>() : 0.0, 0.0};
        } else if (node["SRI"]) {
            const YAML::Node sri = node["SRI"];
            r.falloff = FalloffType::SRI;
            r.falloff_params = {sri["A"].as<double>(), sri["B"].as<double>(), sri["C"].as<double>(),
                                sri["D"] ? sri["D"].as<double>() : 1.0, sri["E"] ? sri["E"].as<double>() : 0.0};
        }
    } else if (r.type == ReactionType::PLOG) {
        const YAML::Node list = node["rate-constants"];
        if (!list || !list.IsSequence() || list.size() == 0) throw std::runtime_error(where + ": no rate-constants.");
        for (const auto & entry : list) {
            if (!entry["P"]) throw std::runtime_error(where + ": a rate-constants entry has no P.");
            const double p = source.units.convert(entry["P"], PRESSURE, where + " P");
            if (!(p > 0.0)) throw std::runtime_error(where + ": pressures must be positive.");
            r.plog.emplace_back(p, arrhenius(entry, order, "rate-constants entry"));
        }
        std::stable_sort(r.plog.begin(), r.plog.end(),
                         [](const auto & a, const auto & b) { return a.first < b.first; });
    } else if (r.type == ReactionType::CHEBYSHEV) {
        const YAML::Node T_range = node["temperature-range"], p_range = node["pressure-range"], data = node["data"];
        if (!T_range || T_range.size() != 2 || !p_range || p_range.size() != 2 || !data || !data.IsSequence() ||
            data.size() == 0) {
            throw std::runtime_error(where + ": Chebyshev reactions need temperature-range, pressure-range and data.");
        }
        r.chebyshev_range = {T_range[0].as<double>(), T_range[1].as<double>(),
                             source.units.convert(p_range[0], PRESSURE, where + " pressure-range"),
                             source.units.convert(p_range[1], PRESSURE, where + " pressure-range")};
        if (!(r.chebyshev_range[0] > 0.0 && r.chebyshev_range[1] > r.chebyshev_range[0] &&
              r.chebyshev_range[2] > 0.0 && r.chebyshev_range[3] > r.chebyshev_range[2])) {
            throw std::runtime_error(where + ": invalid Chebyshev ranges.");
        }
        r.chebyshev_n_T = static_cast<uint32_t>(data.size());
        r.chebyshev_n_p = static_cast<uint32_t>(data[0].size());
        // The rate constant's units (those of A for this order) enter through the first coefficient
        const Dimension dimension{0, 3.0 * (order - 1.0), -1.0, 0, 1.0 - order, 0, 0};
        const double factor = source.units.convert(YAML::Node(1.0), dimension, where + " data");
        for (const auto & row : data) {
            if (row.size() != r.chebyshev_n_p) throw std::runtime_error(where + ": Chebyshev data rows differ in length.");
            for (const auto & value : row) r.chebyshev.push_back(value.as<double>());
        }
        r.chebyshev[0] += std::log10(factor);
    } else {
        if (node["rate-constant"] && node["rate-constant"].IsMap() && !node["rate-constant"]["A"]) {
            throw std::runtime_error(where + ": unsupported rate parameterization.");
        }
        r.rate = arrhenius(node["rate-constant"], r.type == ReactionType::THREE_BODY ? order + 1.0 : order,
                           "rate-constant");
    }
    if (r.type == ReactionType::THREE_BODY || r.type == ReactionType::FALLOFF) {
        if (lhs.third_body != "M") {
            // A specific collider, "(+AR)"
            const int32_t k = mech.species_index(lhs.third_body);
            if (k < 0) throw std::runtime_error(where + ": no third-body species " + lhs.third_body + ".");
            r.default_efficiency = 0.0;
            r.efficiencies = {{static_cast<uint32_t>(k), 1.0}};
        } else {
            if (node["default-efficiency"]) r.default_efficiency = node["default-efficiency"].as<double>();
            if (node["efficiencies"]) {
                for (const auto & kv : node["efficiencies"]) {
                    const int32_t k = mech.species_index(kv.first.as<std::string>());
                    if (k < 0) continue;  // as Cantera: efficiencies of species outside the phase are ignored
                    r.efficiencies.emplace_back(static_cast<uint32_t>(k), kv.second.as<double>());
                }
            }
        }
    }
    return true;
}

Mechanism Reader::read(const std::string & phase_name) {
    mech.file = main->path;
    const YAML::Node phases = main->root["phases"];
    if (!phases || !phases.IsSequence() || phases.size() == 0) {
        throw std::runtime_error("Mechanism file " + main->path + " defines no phases.");
    }
    std::string names;
    size_t index = phases.size();
    for (size_t i = 0; i < phases.size(); i++) {
        const std::string name = phases[i]["name"] ? phases[i]["name"].as<std::string>() : "";
        names += (names.empty() ? "" : ", ") + name;
        if (index == phases.size() && (phase_name.empty() || name == phase_name)) index = i;
    }
    if (index == phases.size()) {
        throw std::runtime_error("Mechanism file " + main->path + " has no phase \"" + phase_name + "\" (phases: " +
                                 names + ").");
    }
    const YAML::Node phase = phases[index];
    mech.phase = phase["name"] ? phase["name"].as<std::string>() : "";
    const std::string thermo = phase["thermo"] ? phase["thermo"].as<std::string>() : "";
    if (thermo != "ideal-gas") {
        throw std::runtime_error("Mechanism " + main->path + ", phase " + mech.phase + ": thermo \"" + thermo +
                                 "\" is not supported (only ideal-gas).");
    }
    if (const YAML::Node elements = phase["elements"]) {
        for (const auto & e : elements) {
            if (!e.IsScalar()) {
                throw std::runtime_error("Mechanism " + main->path + ", phase " + mech.phase +
                                         ": elements must be a list of symbols.");
            }
            mech.elements.push_back(e.as<std::string>());
        }
    }

    const YAML::Node species = phase["species"];
    if (!species) {
        add_species(YAML::Node("all"), *main, "species");
    } else if (species.IsScalar() || (species.IsSequence() && species.size() > 0 && species[0].IsScalar())) {
        add_species(species, *main, "species");
    } else if (species.IsSequence()) {
        for (const auto & block : species) {
            if (!block.IsMap() || block.size() != 1) {
                throw std::runtime_error("Mechanism " + main->path + ", phase " + mech.phase +
                                         ": a species entry must map one section to its species.");
            }
            for (const auto & kv : block) {
                std::string section;
                const auto source = source_of(kv.first.as<std::string>(), section);
                add_species(kv.second, *source, section);
            }
        }
    } else {
        throw std::runtime_error("Mechanism " + main->path + ", phase " + mech.phase + ": malformed species list.");
    }
    if (mech.species.empty()) {
        throw std::runtime_error("Mechanism " + main->path + ", phase " + mech.phase + " has no species.");
    }
    if (phase["kinetics"]) {
        const YAML::Node reactions = phase["reactions"];
        if (!reactions || reactions.IsScalar()) {
            add_reactions(reactions, *main, "reactions");
        } else {
            for (const auto & item : reactions) {
                if (item.IsScalar()) {
                    add_reactions(YAML::Node("all"), *main, item.Scalar());
                    continue;
                }
                for (const auto & kv : item) {
                    std::string section;
                    const auto source = source_of(kv.first.as<std::string>(), section);
                    add_reactions(kv.second, *source, section);
                }
            }
        }
    }
    return mech;
}

Species Reader::parse_species(const YAML::Node & node, const Source & source) const {
    Species sp;
    if (!node["name"]) throw std::runtime_error("Mechanism file " + source.path + ": a species has no name.");
    sp.name = node["name"].as<std::string>();
    const std::string where = "Mechanism file " + source.path + ", species " + sp.name;
    if (!node["composition"] || !node["composition"].IsMap()) {
        throw std::runtime_error(where + ": missing composition.");
    }
    for (const auto & kv : node["composition"]) {
        const std::string element = kv.first.as<std::string>();
        const double atoms = kv.second.as<double>();
        if (!mech.elements.empty()) {
            const bool declared = std::any_of(mech.elements.begin(), mech.elements.end(),
                                              [&](const std::string & e) { return lower(e) == lower(element); });
            if (!declared) {
                throw std::runtime_error(where + ": element " + element + " is not among the phase's elements.");
            }
        }
        double weight;
        if (auto it = custom_weights.find(lower(element)); it != custom_weights.end()) {
            weight = it->second;
        } else {
            weight = atomic_weight(element);
        }
        if (!(weight > 0.0)) throw std::runtime_error(where + ": unknown element " + element + ".");
        sp.composition.emplace_back(element, atoms);
        sp.molecular_weight += atoms * weight;
    }
    if (!node["thermo"]) throw std::runtime_error(where + ": missing thermo.");
    sp.thermo = parse_thermo(node["thermo"], source, where);
    if (const YAML::Node tr = node["transport"]) {
        const std::string model = tr["model"] ? tr["model"].as<std::string>() : "";
        if (model != "gas") throw std::runtime_error(where + ": transport model \"" + model + "\" is not supported (gas).");
        if (!tr["geometry"] || !tr["well-depth"] || !tr["diameter"]) {
            throw std::runtime_error(where + ": gas transport data need geometry, well-depth and diameter.");
        }
        double atoms = 0.0;
        for (const auto & [element, n] : sp.composition) {
            if (lower(element) != "e") atoms += n;
        }
        const std::string geometry = tr["geometry"].as<std::string>();
        SpeciesTransport & t = sp.transport;
        if (geometry == "atom" && atoms <= 1.0) {
            t.geometry = MoleculeGeometry::ATOM;
        } else if (geometry == "linear" && atoms >= 2.0) {
            t.geometry = MoleculeGeometry::LINEAR;
        } else if (geometry == "nonlinear" && atoms >= 3.0) {
            t.geometry = MoleculeGeometry::NONLINEAR;
        } else {
            throw std::runtime_error(where + ": invalid transport geometry \"" + geometry + "\" for its atoms.");
        }
        auto value = [&](const char * key) { return tr[key] ? tr[key].as<double>() : 0.0; };
        // Customary units: Angstrom, K, Debye, Angstrom^3
        t.diameter = 1e-10 * value("diameter");
        t.well_depth = BOLTZMANN * value("well-depth");
        t.dipole = 1e-21 / LIGHT_SPEED * value("dipole");
        t.polarizability = 1e-30 * value("polarizability");
        t.rotational_relaxation = value("rotational-relaxation");
        if (!(t.diameter > 0.0) || t.well_depth < 0.0 || t.dipole < 0.0 || t.polarizability < 0.0 ||
            t.rotational_relaxation < 0.0) {
            throw std::runtime_error(where + ": invalid transport data (non-positive diameter or negative values).");
        }
        sp.has_transport = true;
    }
    return sp;
}

SpeciesThermo Reader::parse_thermo(const YAML::Node & node, const Source & source, const std::string & where) const {
    SpeciesThermo th;
    const std::string model = node["model"] ? node["model"].as<std::string>() : "";
    const Dimension temperature{0, 0, 0, 1, 0, 0, 0};
    if (model == "NASA7" || model == "NASA9") {
        th.model = model == "NASA7" ? ThermoModel::NASA7 : ThermoModel::NASA9;
        const size_t n_coeffs = model == "NASA7" ? 7 : 9;
        const YAML::Node ranges = node["temperature-ranges"];
        const YAML::Node data = node["data"];
        if (!ranges || !data || !ranges.IsSequence() || !data.IsSequence() || ranges.size() != data.size() + 1 ||
            data.size() == 0) {
            throw std::runtime_error(where + ": " + model + " needs temperature-ranges and one data list per range.");
        }
        for (size_t i = 0; i < ranges.size(); i++) {
            th.T_bounds.push_back(source.units.convert(ranges[i], temperature, where + " temperature-ranges"));
            if (i > 0 && !(th.T_bounds[i] > th.T_bounds[i - 1])) {
                throw std::runtime_error(where + ": temperature-ranges must increase.");
            }
        }
        for (const auto & row : data) {
            if (!row.IsSequence() || row.size() != n_coeffs) {
                throw std::runtime_error(where + ": " + model + " data need " + std::to_string(n_coeffs) +
                                         " coefficients per range.");
            }
            std::array<double, 9> a = {};
            const size_t offset = 9 - n_coeffs;
            for (size_t j = 0; j < n_coeffs; j++) a[offset + j] = row[j].as<double>();
            th.coeffs.push_back(a);
        }
    } else if (model == "constant-cp") {
        th.model = ThermoModel::CONSTANT_CP;
        const Dimension molar_energy{0, 0, 0, 0, -1, 1, 0};
        const Dimension molar_entropy{0, 0, 0, -1, -1, 1, 0};
        auto get = [&](const char * key, const Dimension & d, double fallback) {
            return node[key] ? source.units.convert(node[key], d, where + " " + key) : fallback;
        };
        const double T0 = get("T0", temperature, 298.15);
        const double h0 = get("h0", molar_energy, 0.0);
        const double s0 = get("s0", molar_entropy, 0.0);
        const double cp0 = get("cp0", molar_entropy, 0.0);
        th.constant_cp = {T0, h0, s0, cp0};
        th.T_bounds = {get("T-min", temperature, 0.0),
                       get("T-max", temperature, std::numeric_limits<double>::infinity())};
        std::array<double, 9> a = {};
        a[2] = cp0 / GAS_CONSTANT;
        a[7] = (h0 - cp0 * T0) / GAS_CONSTANT;
        a[8] = (s0 - cp0 * std::log(T0)) / GAS_CONSTANT;
        th.coeffs.push_back(a);
    } else {
        throw std::runtime_error(where + ": thermo model \"" + model +
                                 "\" is not supported (NASA7, NASA9, constant-cp).");
    }
    return th;
}

} // namespace

double atomic_weight(const std::string & symbol) {
    const std::string s = lower(symbol);
    for (const auto & e : ELEMENT_WEIGHTS) {
        if (lower(e.symbol) == s) return e.weight;
    }
    return -1.0;
}

size_t SpeciesThermo::range(const double T) const {
    size_t r = 0;
    const bool upper_closed = model == ThermoModel::NASA7;
    while (r + 1 < coeffs.size() && (upper_closed ? T > T_bounds[r + 1] : T >= T_bounds[r + 1])) r++;
    return r;
}

int32_t Mechanism::species_index(const std::string & name) const {
    for (size_t k = 0; k < species.size(); k++) {
        if (species[k].name == name) return static_cast<int32_t>(k);
    }
    return -1;
}

std::vector<std::string> Mechanism::species_names() const {
    std::vector<std::string> names;
    for (const auto & sp : species) names.push_back(sp.name);
    return names;
}

Mechanism read_mechanism(const std::string & file, const std::string & phase) {
    return Reader(file).read(phase);
}

} // namespace chemistry
