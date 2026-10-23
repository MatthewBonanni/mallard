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

/** @brief One key a preset sets, as resolved for a run. */
struct Setting {
    std::string key;    ///< Full key, e.g. "numerics.face_reconstruction.type" or "run.cfl"
    std::string value;  ///< Resolved value, e.g. "TENO", "5", "0.8"
    bool from_preset;   ///< false: the input sets the key, overriding the preset
};

/** @brief The preset of a run and what became of each of its keys. */
struct Resolution {
    std::string name;               ///< Empty without a preset
    std::vector<Setting> settings;  ///< Every key the preset covers, in the preset's order

    /** @brief "accurate", plus the keys the input overrides, for the log. */
    std::string summary() const;
};

/** @brief The preset names, in the order the documentation lists them. */
std::vector<std::string> names();

/**
 * @brief Writes the keys of the input's `[numerics] preset` into the input
 *        wherever it leaves them unset, so the rest of the setup reads one
 *        resolved input. Reconstruction options (`order`, `limiter`) apply only
 *        when the resolved reconstruction type is the preset's; `run.cfl` only
 *        when the input sets neither `cfl` nor `dt`. Without a preset the input
 *        is unchanged and the resolution is empty.
 * @throws InputError for a preset that is not a string or not a known name.
 */
Resolution apply(Input & input);

} // namespace numerics_preset

#endif // NUMERICS_PRESET_H
