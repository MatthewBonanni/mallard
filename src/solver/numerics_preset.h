/**
 * @file numerics_preset.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Numerics presets: validated combinations of the numerics keys,
 *        selected by `[numerics] preset`, under any key the input sets.
 * @version 0.1
 * @date 2026-10-16
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef NUMERICS_PRESET_H
#define NUMERICS_PRESET_H

#include <string>
#include <vector>

#include "input.h"

namespace numerics_preset {

/** @brief One key a preset covers, and who set it; its value is the key's record (resolve_input). */
struct Setting {
    std::string key;   ///< Full key, e.g. "numerics.face_reconstruction.type" or "run.cfl"
    bool from_preset;  ///< false: the input sets the key, overriding the preset
};

/** @brief The preset of a run and what became of each of its keys. */
struct Resolution {
    std::string name;               ///< Empty without a preset
    bool is_default = false;        ///< The input sets no preset and no reconstruction: DEFAULT applies
    std::vector<Setting> settings;  ///< Every key the preset covers, in the preset's order

    /** @brief "high_order", "(default)" if so, plus the keys the input overrides with their values, for the log. */
    std::string summary(const ResolvedInput & resolved) const;
};

/** @brief The preset of an input that sets neither `[numerics] preset` nor `[numerics.face_reconstruction]`. */
inline constexpr const char * DEFAULT = "high_order";

/** @brief The preset names, in the order the documentation lists them; DEFAULT first. */
std::vector<std::string> names();

/**
 * @brief Writes the keys of the input's `[numerics] preset` into the input
 *        wherever it leaves them unset, so the rest of the setup reads one
 *        resolved input. Reconstruction options (`order`, `limiter`) apply only
 *        when the resolved reconstruction type is the preset's; `run.cfl` only
 *        when the input sets neither `cfl` nor `dt`. An input that sets neither
 *        a preset nor `[numerics.face_reconstruction]` gets DEFAULT; one with
 *        that table but no preset is unchanged, and its resolution empty.
 * @throws InputError for a preset that is not a string or not a known name.
 */
Resolution apply(Input & input);

} // namespace numerics_preset

#endif // NUMERICS_PRESET_H
