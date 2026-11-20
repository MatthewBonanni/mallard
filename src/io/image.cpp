/**
 * @file image.cpp
 * @brief Palette images, their text and lines, and their PNG and GIF files.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "image.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "compression.h"

namespace image {

namespace {

/** @brief The built-in font, drawn for Mallard: printable ASCII, a row of 5 bits per byte, top row first. */
constexpr uint8_t FONT[95][GLYPH_HEIGHT] = {
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},  // ' '
    {0x04, 0x04, 0x04, 0x04, 0x04, 0x00, 0x04, 0x00, 0x00},  // '!'
    {0x0a, 0x0a, 0x0a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},  // '"'
    {0x0a, 0x0a, 0x1f, 0x0a, 0x1f, 0x0a, 0x0a, 0x00, 0x00},  // '#'
    {0x04, 0x0f, 0x14, 0x0e, 0x05, 0x1e, 0x04, 0x00, 0x00},  // '$'
    {0x18, 0x19, 0x02, 0x04, 0x08, 0x13, 0x03, 0x00, 0x00},  // '%'
    {0x0c, 0x12, 0x14, 0x08, 0x15, 0x12, 0x0d, 0x00, 0x00},  // '&'
    {0x04, 0x04, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},  // "'"
    {0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02, 0x00, 0x00},  // '('
    {0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08, 0x00, 0x00},  // ')'
    {0x00, 0x04, 0x15, 0x0e, 0x15, 0x04, 0x00, 0x00, 0x00},  // '*'
    {0x00, 0x04, 0x04, 0x1f, 0x04, 0x04, 0x00, 0x00, 0x00},  // '+'
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x0c, 0x04, 0x08},  // ','
    {0x00, 0x00, 0x00, 0x1f, 0x00, 0x00, 0x00, 0x00, 0x00},  // '-'
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x0c, 0x00, 0x00},  // '.'
    {0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x00, 0x00, 0x00},  // '/'
    {0x0e, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0e, 0x00, 0x00},  // '0'
    {0x04, 0x0c, 0x04, 0x04, 0x04, 0x04, 0x0e, 0x00, 0x00},  // '1'
    {0x0e, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1f, 0x00, 0x00},  // '2'
    {0x1f, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0e, 0x00, 0x00},  // '3'
    {0x02, 0x06, 0x0a, 0x12, 0x1f, 0x02, 0x02, 0x00, 0x00},  // '4'
    {0x1f, 0x10, 0x1e, 0x01, 0x01, 0x11, 0x0e, 0x00, 0x00},  // '5'
    {0x06, 0x08, 0x10, 0x1e, 0x11, 0x11, 0x0e, 0x00, 0x00},  // '6'
    {0x1f, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08, 0x00, 0x00},  // '7'
    {0x0e, 0x11, 0x11, 0x0e, 0x11, 0x11, 0x0e, 0x00, 0x00},  // '8'
    {0x0e, 0x11, 0x11, 0x0f, 0x01, 0x02, 0x0c, 0x00, 0x00},  // '9'
    {0x00, 0x0c, 0x0c, 0x00, 0x0c, 0x0c, 0x00, 0x00, 0x00},  // ':'
    {0x00, 0x0c, 0x0c, 0x00, 0x0c, 0x0c, 0x04, 0x08, 0x00},  // ';'
    {0x02, 0x04, 0x08, 0x10, 0x08, 0x04, 0x02, 0x00, 0x00},  // '<'
    {0x00, 0x00, 0x1f, 0x00, 0x1f, 0x00, 0x00, 0x00, 0x00},  // '='
    {0x08, 0x04, 0x02, 0x01, 0x02, 0x04, 0x08, 0x00, 0x00},  // '>'
    {0x0e, 0x11, 0x01, 0x02, 0x04, 0x00, 0x04, 0x00, 0x00},  // '?'
    {0x0e, 0x11, 0x17, 0x15, 0x16, 0x10, 0x0f, 0x00, 0x00},  // '@'
    {0x0e, 0x11, 0x11, 0x1f, 0x11, 0x11, 0x11, 0x00, 0x00},  // 'A'
    {0x1e, 0x11, 0x11, 0x1e, 0x11, 0x11, 0x1e, 0x00, 0x00},  // 'B'
    {0x0e, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0e, 0x00, 0x00},  // 'C'
    {0x1c, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1c, 0x00, 0x00},  // 'D'
    {0x1f, 0x10, 0x10, 0x1e, 0x10, 0x10, 0x1f, 0x00, 0x00},  // 'E'
    {0x1f, 0x10, 0x10, 0x1e, 0x10, 0x10, 0x10, 0x00, 0x00},  // 'F'
    {0x0e, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0f, 0x00, 0x00},  // 'G'
    {0x11, 0x11, 0x11, 0x1f, 0x11, 0x11, 0x11, 0x00, 0x00},  // 'H'
    {0x0e, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0e, 0x00, 0x00},  // 'I'
    {0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0c, 0x00, 0x00},  // 'J'
    {0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11, 0x00, 0x00},  // 'K'
    {0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1f, 0x00, 0x00},  // 'L'
    {0x11, 0x1b, 0x15, 0x15, 0x11, 0x11, 0x11, 0x00, 0x00},  // 'M'
    {0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x00, 0x00},  // 'N'
    {0x0e, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0e, 0x00, 0x00},  // 'O'
    {0x1e, 0x11, 0x11, 0x1e, 0x10, 0x10, 0x10, 0x00, 0x00},  // 'P'
    {0x0e, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0d, 0x00, 0x00},  // 'Q'
    {0x1e, 0x11, 0x11, 0x1e, 0x14, 0x12, 0x11, 0x00, 0x00},  // 'R'
    {0x0f, 0x10, 0x10, 0x0e, 0x01, 0x01, 0x1e, 0x00, 0x00},  // 'S'
    {0x1f, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x00, 0x00},  // 'T'
    {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0e, 0x00, 0x00},  // 'U'
    {0x11, 0x11, 0x11, 0x11, 0x11, 0x0a, 0x04, 0x00, 0x00},  // 'V'
    {0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0a, 0x00, 0x00},  // 'W'
    {0x11, 0x11, 0x0a, 0x04, 0x0a, 0x11, 0x11, 0x00, 0x00},  // 'X'
    {0x11, 0x11, 0x0a, 0x04, 0x04, 0x04, 0x04, 0x00, 0x00},  // 'Y'
    {0x1f, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1f, 0x00, 0x00},  // 'Z'
    {0x0e, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0e, 0x00, 0x00},  // '['
    {0x00, 0x10, 0x08, 0x04, 0x02, 0x01, 0x00, 0x00, 0x00},  // '\\'
    {0x0e, 0x02, 0x02, 0x02, 0x02, 0x02, 0x0e, 0x00, 0x00},  // ']'
    {0x04, 0x0a, 0x11, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},  // '^'
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1f, 0x00},  // '_'
    {0x08, 0x04, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},  // '`'
    {0x00, 0x00, 0x0e, 0x01, 0x0f, 0x11, 0x0f, 0x00, 0x00},  // 'a'
    {0x10, 0x10, 0x16, 0x19, 0x11, 0x11, 0x1e, 0x00, 0x00},  // 'b'
    {0x00, 0x00, 0x0e, 0x10, 0x10, 0x11, 0x0e, 0x00, 0x00},  // 'c'
    {0x01, 0x01, 0x0d, 0x13, 0x11, 0x11, 0x0f, 0x00, 0x00},  // 'd'
    {0x00, 0x00, 0x0e, 0x11, 0x1f, 0x10, 0x0e, 0x00, 0x00},  // 'e'
    {0x06, 0x09, 0x08, 0x1c, 0x08, 0x08, 0x08, 0x00, 0x00},  // 'f'
    {0x00, 0x00, 0x0f, 0x11, 0x11, 0x0f, 0x01, 0x11, 0x0e},  // 'g'
    {0x10, 0x10, 0x16, 0x19, 0x11, 0x11, 0x11, 0x00, 0x00},  // 'h'
    {0x04, 0x00, 0x0c, 0x04, 0x04, 0x04, 0x0e, 0x00, 0x00},  // 'i'
    {0x02, 0x00, 0x06, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0c},  // 'j'
    {0x10, 0x10, 0x12, 0x14, 0x18, 0x14, 0x12, 0x00, 0x00},  // 'k'
    {0x0c, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0e, 0x00, 0x00},  // 'l'
    {0x00, 0x00, 0x1a, 0x15, 0x15, 0x11, 0x11, 0x00, 0x00},  // 'm'
    {0x00, 0x00, 0x16, 0x19, 0x11, 0x11, 0x11, 0x00, 0x00},  // 'n'
    {0x00, 0x00, 0x0e, 0x11, 0x11, 0x11, 0x0e, 0x00, 0x00},  // 'o'
    {0x00, 0x00, 0x1e, 0x11, 0x11, 0x1e, 0x10, 0x10, 0x10},  // 'p'
    {0x00, 0x00, 0x0f, 0x11, 0x11, 0x0f, 0x01, 0x01, 0x01},  // 'q'
    {0x00, 0x00, 0x16, 0x19, 0x10, 0x10, 0x10, 0x00, 0x00},  // 'r'
    {0x00, 0x00, 0x0f, 0x10, 0x0e, 0x01, 0x1e, 0x00, 0x00},  // 's'
    {0x08, 0x08, 0x1c, 0x08, 0x08, 0x09, 0x06, 0x00, 0x00},  // 't'
    {0x00, 0x00, 0x11, 0x11, 0x11, 0x13, 0x0d, 0x00, 0x00},  // 'u'
    {0x00, 0x00, 0x11, 0x11, 0x11, 0x0a, 0x04, 0x00, 0x00},  // 'v'
    {0x00, 0x00, 0x11, 0x11, 0x15, 0x15, 0x0a, 0x00, 0x00},  // 'w'
    {0x00, 0x00, 0x11, 0x0a, 0x04, 0x0a, 0x11, 0x00, 0x00},  // 'x'
    {0x00, 0x00, 0x11, 0x11, 0x11, 0x0f, 0x01, 0x11, 0x0e},  // 'y'
    {0x00, 0x00, 0x1f, 0x02, 0x04, 0x08, 0x1f, 0x00, 0x00},  // 'z'
    {0x02, 0x04, 0x04, 0x08, 0x04, 0x04, 0x02, 0x00, 0x00},  // '{'
    {0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x00, 0x00},  // '|'
    {0x08, 0x04, 0x04, 0x02, 0x04, 0x04, 0x08, 0x00, 0x00},  // '}'
    {0x00, 0x00, 0x08, 0x15, 0x02, 0x00, 0x00, 0x00, 0x00},  // '~'
};

const std::array<uint32_t, 256> & crc_table() {
    static const std::array<uint32_t, 256> table = [] {
        std::array<uint32_t, 256> t{};
        for (uint32_t n = 0; n < 256; n++) {
            uint32_t c = n;
            for (int k = 0; k < 8; k++) c = c & 1u ? 0xedb88320u ^ (c >> 1) : c >> 1;
            t[n] = c;
        }
        return t;
    }();
    return table;
}

uint32_t crc32(const uint8_t * data, size_t n, uint32_t crc = 0) {
    const auto & t = crc_table();
    crc = ~crc;
    for (size_t i = 0; i < n; i++) crc = t[(crc ^ data[i]) & 0xffu] ^ (crc >> 8);
    return ~crc;
}

uint32_t adler32(const std::vector<uint8_t> & data) {
    uint32_t a = 1, b = 0;
    size_t i = 0;
    while (i < data.size()) {
        // 5552 bytes keep the sums below 2^32 before the modulus (zlib's NMAX)
        const size_t end = std::min(data.size(), i + 5552);
        for (; i < end; i++) {
            a += data[i];
            b += a;
        }
        a %= 65521u;
        b %= 65521u;
    }
    return (b << 16) | a;
}

void put32(std::vector<uint8_t> & out, uint32_t v) {
    for (int s = 24; s >= 0; s -= 8) out.push_back(static_cast<uint8_t>(v >> s));
}

void put16le(std::vector<uint8_t> & out, uint32_t v) {
    out.push_back(static_cast<uint8_t>(v & 0xffu));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xffu));
}

/** @brief Bits into bytes, least significant first, as DEFLATE and GIF pack them. */
class BitWriter {
    public:
        explicit BitWriter(std::vector<uint8_t> & out_in) : out(out_in) {}
        void put(uint32_t bits, int n) {
            buffer |= uint64_t(bits) << used;
            used += n;
            while (used >= 8) {
                out.push_back(static_cast<uint8_t>(buffer & 0xffu));
                buffer >>= 8;
                used -= 8;
            }
        }
        /** @brief A Huffman code, which DEFLATE stores most significant bit first. */
        void put_code(uint32_t code, int n) {
            uint32_t reversed = 0;
            for (int i = 0; i < n; i++) reversed |= ((code >> i) & 1u) << (n - 1 - i);
            put(reversed, n);
        }
        void flush() {
            if (used > 0) out.push_back(static_cast<uint8_t>(buffer & 0xffu));
            buffer = 0;
            used = 0;
        }

    private:
        std::vector<uint8_t> & out;
        uint64_t buffer = 0;
        int used = 0;
};

void put_literal(BitWriter & w, uint32_t v) {
    if (v < 144) {
        w.put_code(0x30 + v, 8);
    } else if (v < 256) {
        w.put_code(0x190 + v - 144, 9);
    } else if (v < 280) {
        w.put_code(v - 256, 7);
    } else {
        w.put_code(0xc0 + v - 280, 8);
    }
}

constexpr uint16_t LENGTH_BASE[29] = {3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
                                      31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
constexpr uint8_t LENGTH_EXTRA[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};

void put_repeat(BitWriter & w, uint32_t length) {
    int k = 28;
    while (LENGTH_BASE[k] > length) k--;
    put_literal(w, 257 + uint32_t(k));
    if (LENGTH_EXTRA[k] > 0) w.put(length - LENGTH_BASE[k], LENGTH_EXTRA[k]);
    w.put_code(0, 5);  // distance 1
}

} // namespace

Rgb parse_color(const std::string & hex) {
    if (hex.size() != 7 || hex[0] != '#' ||
        !std::all_of(hex.begin() + 1, hex.end(), [](char c) { return std::isxdigit(static_cast<unsigned char>(c)); })) {
        throw std::invalid_argument("not a color #rrggbb: " + hex);
    }
    const auto byte = [&](size_t i) { return static_cast<uint8_t>(std::stoul(hex.substr(i, 2), nullptr, 16)); };
    return {byte(1), byte(3), byte(5)};
}

void Indexed::fill(int64_t x0, int64_t y0, int64_t x1, int64_t y1, uint8_t color) {
    x0 = std::max<int64_t>(x0, 0);
    y0 = std::max<int64_t>(y0, 0);
    x1 = std::min<int64_t>(x1, width);
    y1 = std::min<int64_t>(y1, height);
    for (int64_t y = y0; y < y1; y++) {
        std::fill_n(pixels.begin() + y * int64_t(width) + x0, std::max<int64_t>(x1 - x0, 0), color);
    }
}

void Indexed::line(double x0, double y0, double x1, double y1, uint8_t color, int thickness) {
    const double length = std::max(std::abs(x1 - x0), std::abs(y1 - y0));
    const auto n = static_cast<int64_t>(std::ceil(length));
    const int64_t lo = -(thickness - 1) / 2, hi = thickness / 2;
    for (int64_t i = 0; i <= n; i++) {
        const double s = n > 0 ? double(i) / double(n) : 0.0;
        const auto x = static_cast<int64_t>(std::floor(x0 + s * (x1 - x0)));
        const auto y = static_cast<int64_t>(std::floor(y0 + s * (y1 - y0)));
        fill(x + lo, y + lo, x + hi + 1, y + hi + 1, color);
    }
}

std::vector<uint8_t> Indexed::rgb() const {
    std::vector<uint8_t> out(pixels.size() * 3);
    for (size_t i = 0; i < pixels.size(); i++) {
        const Rgb c = pixels[i] < palette.size() ? palette[pixels[i]] : Rgb{};
        out[3 * i] = c.r;
        out[3 * i + 1] = c.g;
        out[3 * i + 2] = c.b;
    }
    return out;
}

int64_t text_width(const std::string & text, int scale) {
    return text.empty() ? 0 : (int64_t(text.size()) * GLYPH_ADVANCE - (GLYPH_ADVANCE - GLYPH_WIDTH)) * scale;
}

void text(Indexed & image, int64_t x, int64_t y, const std::string & text, int scale, uint8_t color, int align) {
    if (align == 1) x -= text_width(text, scale) / 2;
    if (align == 2) x -= text_width(text, scale);
    for (const char ch : text) {
        const auto c = static_cast<unsigned char>(ch);
        const uint8_t * glyph = FONT[(c >= 32 && c < 127 ? c : '?') - 32];
        for (int row = 0; row < GLYPH_HEIGHT; row++) {
            for (int col = 0; col < GLYPH_WIDTH; col++) {
                if ((glyph[row] >> (GLYPH_WIDTH - 1 - col)) & 1u) {
                    image.fill(x + col * scale, y + row * scale, x + (col + 1) * scale, y + (row + 1) * scale, color);
                }
            }
        }
        x += GLYPH_ADVANCE * scale;
    }
}

std::vector<uint8_t> zlib_stream_builtin(const std::vector<uint8_t> & data) {
    std::vector<uint8_t> out = {0x78, 0x01};
    out.reserve(data.size() / 4 + 64);
    BitWriter w(out);
    w.put(1, 1);  // the last block
    w.put(1, 2);  // fixed Huffman codes
    size_t i = 0;
    while (i < data.size()) {
        size_t run = 0;
        if (i > 0) {
            while (run < 258 && i + run < data.size() && data[i + run] == data[i - 1]) run++;
        }
        if (run >= 3) {
            put_repeat(w, static_cast<uint32_t>(run));
            i += run;
        } else {
            put_literal(w, data[i]);
            i++;
        }
    }
    put_literal(w, 256);
    w.flush();
    put32(out, adler32(data));
    return out;
}

std::vector<uint8_t> zlib_stream(const std::vector<uint8_t> & data) {
    if (!compression::have_zlib()) return zlib_stream_builtin(data);
    const std::vector<char> z = compression::deflate(data.data(), data.size(), 1, 6);
    return {z.begin(), z.end()};
}

std::vector<uint8_t> png(const Indexed & image) {
    const size_t w = image.width;
    std::vector<uint8_t> rows;
    rows.reserve((w + 1) * image.height);
    const auto breaks = [](const uint8_t * p, size_t n) {
        size_t k = 0;
        for (size_t i = 1; i < n; i++) k += p[i] != p[i - 1];
        return k;
    };
    std::vector<uint8_t> up(w);
    for (size_t y = 0; y < image.height; y++) {
        const uint8_t * row = image.pixels.data() + y * w;
        if (y > 0) {
            for (size_t x = 0; x < w; x++) up[x] = static_cast<uint8_t>(row[x] - (row - w)[x]);
        }
        const bool use_up = y > 0 && breaks(up.data(), w) < breaks(row, w);
        rows.push_back(use_up ? 2 : 0);
        rows.insert(rows.end(), use_up ? up.data() : row, (use_up ? up.data() : row) + w);
    }
    std::vector<uint8_t> out = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
    const auto chunk = [&](const char * type, const std::vector<uint8_t> & body) {
        put32(out, static_cast<uint32_t>(body.size()));
        const size_t start = out.size();
        out.insert(out.end(), type, type + 4);
        out.insert(out.end(), body.begin(), body.end());
        put32(out, crc32(out.data() + start, out.size() - start));
    };
    std::vector<uint8_t> header;
    put32(header, image.width);
    put32(header, image.height);
    header.insert(header.end(), {8, 3, 0, 0, 0});  // 8 bits, palette, deflate, adaptive filters, no interlace
    chunk("IHDR", header);
    std::vector<uint8_t> palette;
    for (const Rgb & c : image.palette) palette.insert(palette.end(), {c.r, c.g, c.b});
    chunk("PLTE", palette);
    chunk("IDAT", zlib_stream(rows));
    chunk("IEND", {});
    return out;
}

std::vector<uint8_t> gif_image_data(const std::vector<uint8_t> & pixels) {
    constexpr int MIN_CODE_SIZE = 8;
    constexpr uint32_t CLEAR = 1u << MIN_CODE_SIZE, END = CLEAR + 1, MAX_CODE = 4095;
    // next[code * 256 + byte]: the code of the string code + byte, 0 if none yet
    std::vector<uint16_t> next(size_t(MAX_CODE + 1) * 256, 0);
    std::vector<size_t> used;
    std::vector<uint8_t> codes;
    BitWriter w(codes);
    int size = MIN_CODE_SIZE + 1;
    uint32_t last = END;
    w.put(CLEAR, size);
    int64_t current = -1;
    for (const uint8_t p : pixels) {
        if (current < 0) {
            current = p;
            continue;
        }
        const size_t key = size_t(current) * 256 + p;
        if (next[key] != 0) {
            current = next[key];
            continue;
        }
        w.put(static_cast<uint32_t>(current), size);
        next[key] = static_cast<uint16_t>(++last);
        used.push_back(key);
        if (last >= (1u << size)) size++;
        if (last == MAX_CODE) {
            w.put(CLEAR, size);
            for (const size_t k : used) next[k] = 0;
            used.clear();
            size = MIN_CODE_SIZE + 1;
            last = END;
        }
        current = p;
    }
    if (current >= 0) w.put(static_cast<uint32_t>(current), size);
    w.put(CLEAR, size);
    w.put(END, MIN_CODE_SIZE + 1);
    w.flush();
    std::vector<uint8_t> out = {MIN_CODE_SIZE};
    for (size_t i = 0; i < codes.size(); i += 255) {
        const size_t n = std::min<size_t>(255, codes.size() - i);
        out.push_back(static_cast<uint8_t>(n));
        out.insert(out.end(), codes.begin() + std::ptrdiff_t(i), codes.begin() + std::ptrdiff_t(i + n));
    }
    out.push_back(0);
    return out;
}

Gif::Gif(const std::string & path_in, uint32_t width_in, uint32_t height_in, const std::vector<Rgb> & palette,
         uint16_t delay_in)
    : path(path_in), width(width_in), height(height_in), delay(delay_in) {
    if (width > 65535 || height > 65535) throw std::runtime_error(path + ": GIF images are at most 65535 pixels wide");
    file = std::fopen(path.c_str(), "wb");
    if (!file) throw std::runtime_error("cannot write " + path);
    std::vector<uint8_t> head = {'G', 'I', 'F', '8', '9', 'a'};
    put16le(head, width);
    put16le(head, height);
    head.insert(head.end(), {0xf7, 0, 0});  // a global color table of 256 colors
    for (size_t i = 0; i < 256; i++) {
        const Rgb c = i < palette.size() ? palette[i] : Rgb{};
        head.insert(head.end(), {c.r, c.g, c.b});
    }
    // Loop forever (the NETSCAPE2.0 application extension)
    head.insert(head.end(), {0x21, 0xff, 11, 'N', 'E', 'T', 'S', 'C', 'A', 'P', 'E', '2', '.', '0', 3, 1, 0, 0, 0});
    std::fwrite(head.data(), 1, head.size(), file);
}

Gif::~Gif() {
    if (file) std::fclose(file);
}

void Gif::add(const Indexed & frame) {
    if (frame.width != width || frame.height != height) throw std::logic_error("Gif: frames of another size");
    std::vector<uint8_t> out = {0x21, 0xf9, 4, 0};
    put16le(out, delay);
    out.insert(out.end(), {0, 0, 0x2c, 0, 0, 0, 0});
    put16le(out, width);
    put16le(out, height);
    out.push_back(0);
    const std::vector<uint8_t> data = gif_image_data(frame.pixels);
    out.insert(out.end(), data.begin(), data.end());
    std::fwrite(out.data(), 1, out.size(), file);
}

void Gif::close() {
    if (!file) return;
    const uint8_t trailer = 0x3b;
    std::fwrite(&trailer, 1, 1, file);
    const bool failed = std::ferror(file) != 0;
    const bool closed = std::fclose(file) == 0;
    file = nullptr;
    if (failed || !closed) throw std::runtime_error("cannot write " + path);
}

} // namespace image
