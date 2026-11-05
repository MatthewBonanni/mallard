/**
 * @file json.h
 * @brief JSON text for the run log and provenance: string quoting, numbers
 *        that read back exactly, objects built key by key, and a strict parser
 *        for reading them back (mallard info, tests).
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef JSON_H
#define JSON_H

#include <cstdint>
#include <stdexcept>
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

/**
 * @brief A parsed JSON value. Objects keep their members in order.
 */
struct Value {
    enum class Type { NUL, BOOL, NUMBER, STRING, ARRAY, OBJECT };
    Type type = Type::NUL;
    bool boolean = false;
    double number = 0.0;
    std::string string;
    std::vector<Value> array;
    std::vector<std::string> names;  ///< of the object's members, in order
    std::vector<Value> members;      ///< (a vector of an incomplete type is allowed, a pair is not)

    /** @brief Whether the object has this member. */
    bool has(const std::string & key) const;
    /** @brief The member of this name. @throws std::out_of_range if there is none */
    const Value & operator[](const std::string & key) const;
    /** @brief The member of this name, or nullptr (also for a value that is not an object). */
    const Value * find(const std::string & key) const;
    std::vector<std::string> keys() const { return names; }
    bool is_null() const { return type == Type::NUL; }
};

/**
 * @brief Parses JSON text (RFC 8259; \u escapes become UTF-8).
 * @throws std::runtime_error saying what is wrong and where
 */
Value parse(std::string_view text);

} // namespace json

#endif // JSON_H
