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
