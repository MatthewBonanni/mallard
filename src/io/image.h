/**
 * @file image.h
 * @brief Palette images for mallard post render: text, lines and rectangles
 *        drawn into them, and the PNG and animated GIF files they are written as,
 *        with no library outside Mallard (zlib compresses PNG where the build has it).
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef IMAGE_H
#define IMAGE_H

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace image {

/** @brief An sRGB color. */
struct Rgb {
    uint8_t r = 0, g = 0, b = 0;
    bool operator==(const Rgb &) const = default;
};

/** @brief "#rrggbb" as a color. @throws std::invalid_argument otherwise */
Rgb parse_color(const std::string & hex);

/** @brief An image of indices into a palette of up to 256 colors, rows from the top. */
struct Indexed {
    uint32_t width = 0, height = 0;
    std::vector<uint8_t> pixels;  ///< width * height
    std::vector<Rgb> palette;

    Indexed() = default;
    Indexed(uint32_t w, uint32_t h, uint8_t fill = 0) : width(w), height(h), pixels(size_t(w) * h, fill) {}

    /** @brief Sets the pixel, if it is in the image. */
    void set(int64_t x, int64_t y, uint8_t color) {
        if (x >= 0 && y >= 0 && x < int64_t(width) && y < int64_t(height)) pixels[size_t(y) * width + size_t(x)] = color;
    }
    /** @brief Fills [x0, x1) x [y0, y1), clipped to the image. */
    void fill(int64_t x0, int64_t y0, int64_t x1, int64_t y1, uint8_t color);
    /** @brief A line of this thickness in pixels between pixel coordinates (centers at half-integers). */
    void line(double x0, double y0, double x1, double y1, uint8_t color, int thickness = 1);
    /** @brief The pixels as RGB, three bytes each. */
    std::vector<uint8_t> rgb() const;
};

/** @brief Glyphs of the built-in font: 5 x 9 pixels (7 above the baseline, 2 below), 6 apart. */
inline constexpr int GLYPH_WIDTH = 5, GLYPH_HEIGHT = 9, GLYPH_ADVANCE = 6, GLYPH_ASCENT = 7;

/** @brief Width in pixels of text drawn at this scale (printable ASCII; others draw as '?'). */
int64_t text_width(const std::string & text, int scale);

/**
 * @brief Draws text with the top of its capitals at y, each font pixel a scale x scale square.
 * @param align 0: x is the left end, 1: the center, 2: the right end
 */
void text(Indexed & image, int64_t x, int64_t y, const std::string & text, int scale, uint8_t color, int align = 0);

/**
 * @brief A zlib stream of the bytes (RFC 1950), with zlib where the build has it, else
 *        with Mallard's own encoder (fixed Huffman codes, repeats of the previous byte).
 */
std::vector<uint8_t> zlib_stream(const std::vector<uint8_t> & data);

/** @brief zlib_stream by Mallard's own encoder, whatever the build. */
std::vector<uint8_t> zlib_stream_builtin(const std::vector<uint8_t> & data);

/** @brief The image as a PNG file (8-bit palette, each row filtered as the smaller of none and up). */
std::vector<uint8_t> png(const Indexed & image);

/**
 * @brief An animated GIF written frame by frame, every frame of one size and palette,
 *        looping forever.
 */
class Gif {
    public:
        /**
         * @param delay the time each frame shows, in hundredths of a second
         * @throws std::runtime_error if the file cannot be written
         */
        Gif(const std::string & path, uint32_t width, uint32_t height, const std::vector<Rgb> & palette,
            uint16_t delay);
        ~Gif();
        Gif(const Gif &) = delete;
        Gif & operator=(const Gif &) = delete;
        /** @brief Appends a frame of the size and palette given. */
        void add(const Indexed & frame);
        /** @brief Ends the file. @throws std::runtime_error if writing failed */
        void close();

    private:
        std::FILE * file = nullptr;
        std::string path;
        uint32_t width, height;
        uint16_t delay;
};

/** @brief The frame's pixels as GIF image data: the LZW minimum code size, then sub-blocks, then 0. */
std::vector<uint8_t> gif_image_data(const std::vector<uint8_t> & pixels);

} // namespace image

#endif // IMAGE_H
