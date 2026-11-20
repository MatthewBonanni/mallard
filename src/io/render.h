/**
 * @file render.h
 * @brief The images of mallard post render: the cells of a snapshot on a plane
 *        (2D cells as they are, 3D cells cut by a slice), sampled into pixels and
 *        colored by a field, with a colorbar, the time and a scale bar.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef RENDER_H
#define RENDER_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "image.h"
#include "output_reader.h"

namespace render {

/** @brief The colormaps, by name; each also reversed, as NAME_r. */
std::vector<std::string> colormap_names();

/** @brief 256 colors of a colormap, from the low end. @throws std::invalid_argument for an unknown name */
std::array<image::Rgb, 256> colormap(const std::string & name);

/** @brief How a field is colored by default. */
struct FieldStyle {
    std::string colormap;
    bool symmetric = false;  ///< a range centered on zero
    bool log = false;        ///< a logarithmic scale
};

/**
 * @brief The colormap and scale of a field by its name and the range of its values: coolwarm centered on
 *        zero for values of both signs (each beyond a tenth of the largest magnitude) and for signed
 *        fields (vorticity components, DILATATION, HELICITY, Q_CRITERION, FLAME_INDEX), inferno for
 *        temperatures and heat release, plasma for species and mixture fractions, gray reversed on a
 *        logarithmic scale for SCHLIEREN, viridis for the rest.
 */
FieldStyle default_style(const std::string & field, double min, double max);

/** @brief Whether the cells include volume cells (tetrahedra, hexahedra, ...), which a slice cuts. */
bool has_volume_cells(const output::Geometry & geometry);

/** @brief The plane coordinates of a slice normal to axis (0: x, 1: y, 2: z): (y, z), (x, z) or (x, y). */
std::array<int, 2> plane_axes(int axis);

/** @brief The cells drawn on the plane: a polygon each, in plane coordinates. */
struct Polygons {
    std::vector<std::array<double, 2>> points;
    std::vector<uint64_t> offsets{0};  ///< polygon i's points are points[offsets[i], offsets[i + 1])
    std::vector<uint64_t> cells;       ///< the cell of each polygon, in the file's order
    std::array<double, 4> bounds{};    ///< x0, x1, y0, y1 of every point
    size_t size() const { return cells.size(); }
};

/**
 * @brief The cells of a geometry as polygons: 2D cells (triangles, quadrilaterals, polygons, the outline
 *        of quadratic cells) in x and y; with volume cells, the polygon each cell's vertices cut on the plane
 *        normal to axis at that position (exact for convex cells), a cell touching the plane only from
 *        below not cut, so that a plane through faces cuts each place once.
 */
Polygons polygons(const output::Geometry & geometry, int axis, double at);

/** @brief A rectangle of the plane: x0 < x1, y0 < y1. */
struct Window {
    double x0 = 0.0, x1 = 1.0, y0 = 0.0, y1 = 1.0;
};

/**
 * @brief Which cells each pixel of an image of the window shows, with their weights: the cells whose
 *        centroid lies in the pixel, weighted by their area (an area average where cells are smaller than
 *        pixels, so fine meshes do not alias), else the cell at its center; none outside the cells.
 */
struct PixelMap {
    uint32_t width = 0, height = 0;
    std::vector<uint64_t> offsets;  ///< pixel p's cells are cells[offsets[p], offsets[p + 1]), rows from the top
    std::vector<uint64_t> cells;
    std::vector<float> weights;     ///< summing to one over each pixel

    /** @brief The value each pixel shows; NaN where it shows no cell. */
    std::vector<float> apply(const std::vector<double> & cell_values, unsigned threads) const;
};

/** @brief The pixel map of the polygons in an image of width x height pixels of the window. */
PixelMap pixel_map(const Polygons & polygons, const Window & window, uint32_t width, uint32_t height, unsigned threads);

/** @brief Where the parts of an image go, in pixels. */
struct Layout {
    uint32_t width = 0, height = 0;
    int scale = 1;  ///< pixels per pixel of the font
    int64_t plot_x = 0, plot_y = 0;
    uint32_t plot_width = 0, plot_height = 0;
    bool bare = false;   ///< the plot alone
    bool title = false;  ///< a line for a title
};

/**
 * @brief The layout of an image of the window: the plot with the window's aspect ratio, and around it
 *        (unless bare) the time and the field's label above, the colorbar to the right and the scale bar
 *        below; the font scale grows with the width. height 0: as the plot needs, at most 1.5 x width;
 *        else the plot is centered in what is left. Width and height are made even, as video codecs need.
 */
Layout layout(const Window & window, uint32_t width, uint32_t height, bool bare, bool title);

/** @brief The colors of a theme besides the colormap. */
struct Theme {
    image::Rgb background, foreground, muted, outside, mesh, overlay;
};

/** @brief The documentation's light or dark palette (tools/docs/figstyle.py). */
Theme theme(bool dark);

/** @brief How values map to colors. */
struct Scale {
    double lo = 0.0, hi = 1.0;
    bool log = false;
};

/** @brief Up to about six round values for the colorbar's ticks: multiples of 1, 2 or 5 times a power of
 *         ten, or powers of ten on a logarithmic scale; the ends if there are fewer than two. */
std::vector<double> ticks(const Scale & scale);

/** @brief A drawing over the plot: segments and points in plane coordinates, labeled. */
struct Overlay {
    std::vector<std::array<std::array<double, 2>, 2>> segments;
    std::vector<std::array<double, 2>> points;
    std::string label;
};

/** @brief Everything an image shows. */
struct Frame {
    Window window;
    const PixelMap * map = nullptr;
    std::vector<float> values;  ///< per pixel of the map
    std::array<image::Rgb, 256> colormap{};
    Scale scale;
    Theme colors;
    std::string title, time, label;  ///< above the plot: a title line, the time, the field and units
    const Polygons * mesh = nullptr;  ///< whose edges to draw, or none
    std::vector<Overlay> overlays;
};

/** @brief Palette indices of an image: 0 to 239 the colormap, then these. */
enum Color : uint8_t { BACKGROUND = 240, FOREGROUND, MUTED, OUTSIDE, MESH, OVERLAY, N_COLORS };

/** @brief The image of a frame in a layout. */
image::Indexed draw(const Layout & layout, const Frame & frame, unsigned threads);

/** @brief A number for labels: %g with up to four significant digits. */
std::string number(double x);

} // namespace render

#endif // RENDER_H
