/**
 * @file json.cpp
 * @brief JSON text implementation.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "json.h"

#include <array>
#include <charconv>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>

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

bool Value::has(const std::string & key) const { return find(key) != nullptr; }

const Value * Value::find(const std::string & key) const {
    for (size_t i = 0; i < names.size(); i++) {
        if (names[i] == key) return &members[i];
    }
    return nullptr;
}

const Value & Value::operator[](const std::string & key) const {
    const Value * v = find(key);
    if (!v) throw std::out_of_range("no key " + key);
    return *v;
}

namespace {

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
            throw std::runtime_error("JSON: " + what + " at character " + std::to_string(at));
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

        unsigned long hex4() {
            if (at + 4 > text.size()) fail("short \\u escape");
            const std::string digits(text.substr(at, 4));
            char * end = nullptr;
            const unsigned long code = std::strtoul(digits.c_str(), &end, 16);
            if (end != digits.c_str() + 4) fail("bad \\u escape");
            at += 4;
            return code;
        }

        static void utf8(std::string & out, unsigned long code) {
            auto byte = [&](unsigned long b) { out += static_cast<char>(static_cast<unsigned char>(b)); };
            if (code < 0x80) {
                byte(code);
            } else if (code < 0x800) {
                byte(0xc0 | (code >> 6));
                byte(0x80 | (code & 0x3f));
            } else if (code < 0x10000) {
                byte(0xe0 | (code >> 12));
                byte(0x80 | ((code >> 6) & 0x3f));
                byte(0x80 | (code & 0x3f));
            } else {
                byte(0xf0 | (code >> 18));
                byte(0x80 | ((code >> 12) & 0x3f));
                byte(0x80 | ((code >> 6) & 0x3f));
                byte(0x80 | (code & 0x3f));
            }
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
                        unsigned long code = hex4();
                        if (code >= 0xd800 && code < 0xdc00 && take("\\u")) {
                            const unsigned long low = hex4();
                            if (low < 0xdc00 || low >= 0xe000) fail("bad surrogate pair");
                            code = 0x10000 + ((code - 0xd800) << 10) + (low - 0xdc00);
                        }
                        utf8(out, code);
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

} // namespace

Value parse(std::string_view text) { return Parser(text).document(); }

} // namespace json
