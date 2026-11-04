/**
 * @file json.cpp
 * @brief JSON text implementation.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "json.h"

#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>

namespace json {

std::string quote(std::string_view text) {
    std::string out;
    out.reserve(text.size() + 2);
    out += '"';
    for (const char c : text) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char code[8];
                    std::snprintf(code, sizeof(code), "\\u%04x", static_cast<unsigned>(c));
                    out += code;
                } else {
                    out += c;
                }
        }
    }
    out += '"';
    return out;
}

std::string number(double value) {
    if (!std::isfinite(value)) return "null";
    std::array<char, 32> buffer{};
    const auto result = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
    return std::string(buffer.data(), result.ptr);
}

Object & Object::raw(std::string_view key, std::string_view value) {
    if (text.size() > 1) text += ',';
    text += quote(key);
    text += ':';
    text += value;
    return *this;
}

Object & Object::add_numbers(std::string_view key, const std::vector<std::pair<std::string, double>> & values) {
    Object inner;
    for (const auto & [name, value] : values) inner.add(name, value);
    return raw(key, inner.str());
}

} // namespace json
