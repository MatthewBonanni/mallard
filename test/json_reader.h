/**
 * @file json_reader.h
 * @brief A strict JSON parser for checking the run log and provenance in tests.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef JSON_READER_H
#define JSON_READER_H

#include <cctype>
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace json_reader {

struct Value {
    enum class Type { NUL, BOOL, NUMBER, STRING, ARRAY, OBJECT };
    Type type = Type::NUL;
    bool boolean = false;
    double number = 0.0;
    std::string string;
    std::vector<Value> array;
    std::vector<std::string> names;  // of the object's members, in order
    std::vector<Value> members;      // (a vector of an incomplete type is allowed, a pair is not)

    bool has(const std::string & key) const {
        for (const auto & k : names) {
            if (k == key) return true;
        }
        return false;
    }
    const Value & operator[](const std::string & key) const {
        for (size_t i = 0; i < names.size(); i++) {
            if (names[i] == key) return members[i];
        }
        throw std::out_of_range("no key " + key);
    }
    std::vector<std::string> keys() const { return names; }
};

class Parser {
    public:
        explicit Parser(std::string_view text_in) : text(text_in) {}

        Value document() {
            Value v = value();
            space();
            if (at != text.size()) fail("trailing characters");
            return v;
        }

    private:
        std::string_view text;
        size_t at = 0;

        [[noreturn]] void fail(const std::string & what) const {
            throw std::runtime_error("JSON: " + what + " at " + std::to_string(at));
        }
        void space() {
            while (at < text.size() && (text[at] == ' ' || text[at] == '\t' || text[at] == '\n' || text[at] == '\r')) at++;
        }
        bool take(std::string_view word) {
            if (text.substr(at, word.size()) != word) return false;
            at += word.size();
            return true;
        }
        void expect(char c) {
            space();
            if (at >= text.size() || text[at] != c) fail(std::string("expected ") + c);
            at++;
        }

        Value value() {
            space();
            if (at >= text.size()) fail("unexpected end");
            Value v;
            const char c = text[at];
            if (c == '{') {
                v.type = Value::Type::OBJECT;
                at++;
                space();
                if (take("}")) return v;
                do {
                    space();
                    std::string key = string();
                    if (v.has(key)) fail("duplicate key " + key);
                    expect(':');
                    v.names.push_back(std::move(key));
                    v.members.push_back(value());
                    space();
                } while (take(","));
                expect('}');
            } else if (c == '[') {
                v.type = Value::Type::ARRAY;
                at++;
                space();
                if (take("]")) return v;
                do {
                    v.array.push_back(value());
                    space();
                } while (take(","));
                expect(']');
            } else if (c == '"') {
                v.type = Value::Type::STRING;
                v.string = string();
            } else if (take("true")) {
                v.type = Value::Type::BOOL;
                v.boolean = true;
            } else if (take("false")) {
                v.type = Value::Type::BOOL;
            } else if (take("null")) {
                v.type = Value::Type::NUL;
            } else {
                v.type = Value::Type::NUMBER;
                const size_t start = at;
                if (text[at] == '-') at++;
                if (at >= text.size() || !std::isdigit(static_cast<unsigned char>(text[at]))) fail("bad number");
                while (at < text.size() && (std::isdigit(static_cast<unsigned char>(text[at])) || text[at] == '.' ||
                                            text[at] == 'e' || text[at] == 'E' || text[at] == '+' || text[at] == '-')) {
                    at++;
                }
                v.number = std::strtod(std::string(text.substr(start, at - start)).c_str(), nullptr);
            }
            return v;
        }

        std::string string() {
            if (at >= text.size() || text[at] != '"') fail("expected a string");
            at++;
            std::string out;
            while (at < text.size() && text[at] != '"') {
                const char c = text[at++];
                if (static_cast<unsigned char>(c) < 0x20) fail("control character in a string");
                if (c != '\\') {
                    out += c;
                    continue;
                }
                if (at >= text.size()) fail("unterminated escape");
                const char e = text[at++];
                switch (e) {
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case '/': out += '/'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case 'n': out += '\n'; break;
                    case 'r': out += '\r'; break;
                    case 't': out += '\t'; break;
                    case 'u': {
                        if (at + 4 > text.size()) fail("short \\u escape");
                        const unsigned long code = std::stoul(std::string(text.substr(at, 4)), nullptr, 16);
                        at += 4;
                        if (code >= 0x80) fail("non-ASCII \\u escape");
                        out += static_cast<char>(code);
                        break;
                    }
                    default: fail("bad escape");
                }
            }
            if (at >= text.size()) fail("unterminated string");
            at++;
            return out;
        }
};

inline Value parse(std::string_view text) { return Parser(text).document(); }

} // namespace json_reader

#endif // JSON_READER_H
