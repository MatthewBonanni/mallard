/**
 * @file diagnostic_code.cpp
 * @brief Where the docs explain each diagnostic code.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "diagnostic_code.h"

#include <algorithm>
#include <cctype>

namespace diagnostic_code {

std::string docs(const std::string_view code) {
    const auto entry = std::find_if(ALL.begin(), ALL.end(), [&](const Entry & e) { return code == e.code; });
    if (entry == ALL.end()) throw std::logic_error("no diagnostic code " + std::string(code));
    std::string anchor(code);
    std::transform(anchor.begin(), anchor.end(), anchor.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    return std::string(entry->page) + "#" + anchor;
}

} // namespace diagnostic_code
