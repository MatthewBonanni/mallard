/**
 * @file json.h
 * @brief JSON text for the run log and provenance: string quoting, numbers
 *        that read back exactly, and objects built key by key.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef JSON_H
#define JSON_H

#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace json {

/** @brief A JSON string literal of text (UTF-8 passed through, control characters escaped). */
std::string quote(std::string_view text);

/** @brief The shortest text that reads back as value; null if it is not finite. */
std::string number(double value);

/** @brief A JSON object, built key by key in order. */
class Object {
    public:
        /** @brief Adds a value given as JSON text (an object, array, number, ...). */
        Object & raw(std::string_view key, std::string_view value);

        template <typename T>
        Object & add(std::string_view key, const T & value) {
            if constexpr (std::is_same_v<T, bool>) {
                return raw(key, value ? "true" : "false");
            } else if constexpr (std::is_integral_v<T>) {
                return raw(key, std::to_string(value));
            } else if constexpr (std::is_floating_point_v<T>) {
                return raw(key, number(static_cast<double>(value)));
            } else {
                return raw(key, quote(value));
            }
        }

        /** @brief A number, or null where it is negative (unknown). */
        Object & add_known(std::string_view key, double value) {
            return value < 0.0 ? raw(key, "null") : add(key, value);
        }

        /** @brief An object of numbers, in order. */
        Object & add_numbers(std::string_view key, const std::vector<std::pair<std::string, double>> & values);

        bool empty() const { return text.size() == 1; }
        std::string str() const { return text + "}"; }

    private:
        std::string text = "{";
};

} // namespace json

#endif // JSON_H
