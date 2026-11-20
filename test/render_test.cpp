/**
 * @file render_test.cpp
 * @brief mallard post render: every pixel shows the cells under it, slices cut each cell once with its
 *        cross section, and the PNG and GIF files decode to the images drawn.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "cli.h"
#include "commands.h"
#include "compression.h"
#include "image.h"
#include "render.h"
#include "scratch.h"
#include "solver.h"
#include "test_fixtures.h"

namespace {

namespace fs = std::filesystem;

/** @brief An nx x ny grid of the unit square, each square split into two triangles along alternating diagonals. */
output::Geometry triangles(uint64_t nx, uint64_t ny) {
    output::Geometry g;
    for (uint64_t j = 0; j <= ny; j++) {
        for (uint64_t i = 0; i <= nx; i++) g.points.insert(g.points.end(), {double(i) / double(nx), double(j) / double(ny), 0.0});
    }
    for (uint64_t j = 0; j < ny; j++) {
        for (uint64_t i = 0; i < nx; i++) {
            const uint64_t a = j * (nx + 1) + i, b = a + 1, c = a + nx + 2, d = a + nx + 1;
            const bool flip = (i + j) % 2 == 1;
            for (const auto & t : flip ? std::vector<std::array<uint64_t, 3>>{{a, b, d}, {b, c, d}}
                                       : std::vector<std::array<uint64_t, 3>>{{a, b, c}, {a, c, d}}) {
                g.connectivity.insert(g.connectivity.end(), t.begin(), t.end());
                g.offsets.push_back(g.connectivity.size());
                g.types.push_back(5);
            }
        }
    }
    return g;
}

/** @brief An nx x ny x nz grid of hexahedra of the box [0, 1] x [0, 1] x [0, 1], z layer by layer. */
output::Geometry hexahedra(uint64_t nx, uint64_t ny, uint64_t nz) {
    output::Geometry g;
    const auto id = [&](uint64_t i, uint64_t j, uint64_t k) { return (k * (ny + 1) + j) * (nx + 1) + i; };
    for (uint64_t k = 0; k <= nz; k++) {
        for (uint64_t j = 0; j <= ny; j++) {
            for (uint64_t i = 0; i <= nx; i++) {
                g.points.insert(g.points.end(), {double(i) / double(nx), double(j) / double(ny), double(k) / double(nz)});
            }
        }
    }
    for (uint64_t k = 0; k < nz; k++) {
        for (uint64_t j = 0; j < ny; j++) {
            for (uint64_t i = 0; i < nx; i++) {
                g.connectivity.insert(g.connectivity.end(), {id(i, j, k), id(i + 1, j, k), id(i + 1, j + 1, k), id(i, j + 1, k),
                                                             id(i, j, k + 1), id(i + 1, j, k + 1), id(i + 1, j + 1, k + 1),
                                                             id(i, j + 1, k + 1)});
                g.offsets.push_back(g.connectivity.size());
                g.types.push_back(12);
            }
        }
    }
    return g;
}

double polygon_area(const render::Polygons & p, size_t i) {
    double a = 0.0;
    for (uint64_t k = p.offsets[i]; k < p.offsets[i + 1]; k++) {
        const auto & u = p.points[k];
        const auto & v = p.points[k + 1 < p.offsets[i + 1] ? k + 1 : p.offsets[i]];
        a += u[0] * v[1] - v[0] * u[1];
    }
    return 0.5 * std::abs(a);
}

/** @brief The pixels of LZW image data, as a GIF decoder reads them. */
std::vector<uint8_t> lzw_decode(const std::vector<uint8_t> & data, size_t & at) {
    const int min = data[at++];
    std::vector<uint8_t> bytes;
    for (uint8_t n; (n = data[at++]) != 0; at += n) bytes.insert(bytes.end(), data.begin() + std::ptrdiff_t(at), data.begin() + std::ptrdiff_t(at + n));
    const uint32_t clear = 1u << min, end = clear + 1;
    std::vector<std::vector<uint8_t>> table;
    const auto reset = [&] {
        table.assign(end + 1, {});
        for (uint32_t i = 0; i < clear; i++) table[i] = {uint8_t(i)};
    };
    reset();
    int size = min + 1;
    size_t bit = 0;
    int64_t previous = -1;
    std::vector<uint8_t> out;
    while (bit + size_t(size) <= bytes.size() * 8) {
        uint32_t code = 0;
        for (int i = 0; i < size; i++, bit++) code |= uint32_t((bytes[bit / 8] >> (bit % 8)) & 1u) << i;
        if (code == clear) {
            reset();
            size = min + 1;
            previous = -1;
            continue;
        }
        if (code == end) break;
        std::vector<uint8_t> entry;
        if (code < table.size()) {
            entry = table[code];
        } else {
            EXPECT_EQ(code, table.size());
            EXPECT_GE(previous, 0);
            if (code != table.size() || previous < 0) return out;
            entry = table[size_t(previous)];
            entry.push_back(entry[0]);
        }
        out.insert(out.end(), entry.begin(), entry.end());
        if (previous >= 0 && table.size() < 4096) {
            std::vector<uint8_t> added = table[size_t(previous)];
            added.push_back(entry[0]);
            table.push_back(added);
            if (table.size() == (1u << size) && size < 12) size++;
        }
        previous = code;
    }
    return out;
}

/** @brief The frames of a GIF file. */
std::vector<std::vector<uint8_t>> gif_frames(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    const std::vector<uint8_t> d((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    EXPECT_GT(d.size(), 13u + 768u);
    EXPECT_EQ(std::string(d.begin(), d.begin() + 6), "GIF89a");
    std::vector<std::vector<uint8_t>> frames;
    size_t at = 13 + 768;
    while (at < d.size() && d[at] != 0x3b) {
        if (d[at] == 0x21) {
            at += 2;
            for (uint8_t n; (n = d[at++]) != 0;) at += n;
        } else {
            EXPECT_EQ(d[at], 0x2c);
            at += 10;
            frames.push_back(lzw_decode(d, at));
        }
    }
    return frames;
}

uint32_t be32(const std::vector<uint8_t> & d, size_t at) {
    return uint32_t(d[at]) << 24 | uint32_t(d[at + 1]) << 16 | uint32_t(d[at + 2]) << 8 | uint32_t(d[at + 3]);
}

/** @brief The image of a PNG file of 8-bit palette indices (filters none and up), by zlib. */
image::Indexed png_image(const std::vector<uint8_t> & d) {
    image::Indexed img;
    std::vector<uint8_t> z;
    for (size_t at = 8; at < d.size();) {
        const uint32_t n = be32(d, at);
        const std::string type(d.begin() + std::ptrdiff_t(at + 4), d.begin() + std::ptrdiff_t(at + 8));
        const std::vector<uint8_t> body(d.begin() + std::ptrdiff_t(at + 8), d.begin() + std::ptrdiff_t(at + 8 + n));
        if (type == "IHDR") {
            img.width = be32(body, 0);
            img.height = be32(body, 4);
            EXPECT_EQ(body[8], 8);
            EXPECT_EQ(body[9], 3);
        } else if (type == "PLTE") {
            for (size_t i = 0; i < n; i += 3) img.palette.push_back({body[i], body[i + 1], body[i + 2]});
        } else if (type == "IDAT") {
            z.insert(z.end(), body.begin(), body.end());
        }
        at += 12 + n;
    }
    std::vector<uint8_t> rows((size_t(img.width) + 1) * img.height);
    compression::inflate(reinterpret_cast<const char *>(z.data()), z.size(), rows.data(), rows.size());
    img.pixels.resize(size_t(img.width) * img.height);
    for (size_t y = 0; y < img.height; y++) {
        const uint8_t filter = rows[y * (img.width + 1)];
        EXPECT_TRUE(filter == 0 || (filter == 2 && y > 0));
        for (size_t x = 0; x < img.width; x++) {
            const uint8_t v = rows[y * (img.width + 1) + 1 + x];
            img.pixels[y * img.width + x] = filter == 2 ? uint8_t(v + img.pixels[(y - 1) * img.width + x]) : v;
        }
    }
    return img;
}

image::Indexed test_image() {
    std::mt19937 rng(7);
    image::Indexed img(301, 97);
    img.palette.resize(256);
    for (size_t i = 0; i < 256; i++) img.palette[i] = {uint8_t(i), uint8_t(255 - i), uint8_t(i / 2)};
    // Noise, runs and stripes: every code width, table resets and both PNG filters
    for (size_t i = 0; i < img.pixels.size(); i++) {
        const size_t y = i / img.width;
        img.pixels[i] = y < 30 ? uint8_t(rng() % 256) : y < 60 ? uint8_t(y % 5) : uint8_t((i % img.width) / 40);
    }
    return img;
}

} // namespace

TEST(RenderTest, EveryPixelOfATiledSquareShowsACell) {
    const render::Polygons p = render::polygons(triangles(7, 5), 2, 0.0);
    ASSERT_EQ(p.size(), 70u);
    const render::Window w{0.0, 1.0, 0.0, 1.0};
    // Pixels much smaller than cells, and about their size: pixel centers on the cells' edges and diagonals too
    for (const uint32_t n : {140u, 35u, 14u}) {
        const render::PixelMap map = render::pixel_map(p, w, n, n, 2);
        for (size_t px = 0; px < size_t(n) * n; px++) {
            ASSERT_LT(map.offsets[px], map.offsets[px + 1]) << n << " pixels: pixel " << px << " shows no cell";
            double total = 0.0;
            for (uint64_t k = map.offsets[px]; k < map.offsets[px + 1]; k++) total += double(map.weights[k]);
            EXPECT_NEAR(total, 1.0, 1e-6);
        }
    }
}

TEST(RenderTest, CellsSmallerThanPixelsAreAveragedByArea) {
    // A checkerboard of 0 and 1, four by four cells to a pixel: each pixel shows its mean, 0.5, not one cell
    const output::Geometry g = triangles(64, 64);
    std::vector<double> values(g.types.size());
    for (size_t c = 0; c < values.size(); c++) {
        const size_t square = c / 2;
        values[c] = double((square % 64 + square / 64) % 2);
    }
    const render::Polygons p = render::polygons(g, 2, 0.0);
    const render::PixelMap map = render::pixel_map(p, {0.0, 1.0, 0.0, 1.0}, 16, 16, 2);
    for (const float v : map.apply(values, 2)) EXPECT_NEAR(v, 0.5f, 1e-6f);
}

TEST(RenderTest, SlicesCutEachCellOnceWithItsCrossSection) {
    const output::Geometry g = hexahedra(4, 3, 2);
    // Within a layer, and through the faces between the layers: each column of cells cut once
    for (const double z : {0.3, 0.5}) {
        const render::Polygons p = render::polygons(g, 2, z);
        ASSERT_EQ(p.size(), 12u) << "z = " << z;
        for (size_t i = 0; i < p.size(); i++) {
            EXPECT_EQ(p.cells[i], i) << "z = " << z << ": the lower layer's cells";
            EXPECT_NEAR(polygon_area(p, i), 1.0 / 12.0, 1e-12);
        }
    }
    // A plane normal to x: polygons in (y, z), the cells of one column of x
    const render::Polygons p = render::polygons(g, 0, 0.6);
    ASSERT_EQ(p.size(), 6u);
    EXPECT_NEAR(p.bounds[1] - p.bounds[0], 1.0, 1e-12);
    EXPECT_NEAR(p.bounds[3] - p.bounds[2], 1.0, 1e-12);
    for (size_t i = 0; i < p.size(); i++) EXPECT_EQ(p.cells[i] % 4, 2u);

    // The unit cube as six tetrahedra: any slice's cross sections fill the unit square
    output::Geometry t = hexahedra(1, 1, 1);
    const std::vector<uint64_t> hex(t.connectivity.begin(), t.connectivity.end());
    t.connectivity.clear();
    t.offsets = {0};
    t.types.clear();
    for (const auto & tet : std::vector<std::array<int, 4>>{{0, 1, 2, 6}, {0, 2, 3, 6}, {0, 3, 7, 6},
                                                           {0, 7, 4, 6}, {0, 4, 5, 6}, {0, 5, 1, 6}}) {
        for (const int v : tet) t.connectivity.push_back(hex[size_t(v)]);
        t.offsets.push_back(t.connectivity.size());
        t.types.push_back(10);
    }
    for (const double z : {0.13, 0.5, 0.91}) {
        const render::Polygons q = render::polygons(t, 2, z);
        double area = 0.0;
        for (size_t i = 0; i < q.size(); i++) area += polygon_area(q, i);
        EXPECT_NEAR(area, 1.0, 1e-12) << "z = " << z;
    }
}

TEST(RenderTest, ColorbarTicksAreRoundAndWithinTheRange) {
    std::mt19937 rng(3);
    std::uniform_real_distribution<double> u(-3.0, 3.0);
    for (int trial = 0; trial < 200; trial++) {
        const double a = std::pow(10.0, u(rng)) * (trial % 3 == 0 ? -1.0 : 1.0);
        const double b = a + std::pow(10.0, u(rng));
        const std::vector<double> t = render::ticks({a, b, false});
        ASSERT_GE(t.size(), 2u);
        EXPECT_LE(t.size(), 11u);
        const std::vector<std::string> labels = render::tick_labels(t);
        EXPECT_EQ(std::adjacent_find(labels.begin(), labels.end()), labels.end()) << a << " " << b;
        for (const double x : t) {
            EXPECT_GE(x, a - 1e-9 * std::abs(b - a));
            EXPECT_LE(x, b + 1e-9 * std::abs(b - a));
        }
        if (t.size() > 2) {
            const double step = t[1] - t[0];
            const double mantissa = step / std::pow(10.0, std::floor(std::log10(step) + 1e-9));
            EXPECT_TRUE(std::abs(mantissa - 1.0) < 1e-6 || std::abs(mantissa - 2.0) < 1e-6 ||
                        std::abs(mantissa - 5.0) < 1e-6) << step;
        }
    }
    EXPECT_EQ(render::ticks({1e-3, 10.0, true}), (std::vector<double>{1e-3, 1e-2, 1e-1, 1.0, 10.0}));
}

TEST(RenderTest, PngFilesHoldTheImage) {
    if (!compression::have_zlib()) GTEST_SKIP() << "decoding needs zlib";
    const image::Indexed img = test_image();
    const image::Indexed back = png_image(image::png(img));
    EXPECT_EQ(back.width, img.width);
    EXPECT_EQ(back.height, img.height);
    EXPECT_EQ(back.palette, img.palette);
    EXPECT_EQ(back.pixels, img.pixels);
    // The encoder of builds without zlib writes streams zlib reads
    std::vector<uint8_t> bytes(img.pixels.begin(), img.pixels.end());
    const std::vector<uint8_t> z = image::zlib_stream_builtin(bytes);
    std::vector<uint8_t> out(bytes.size());
    compression::inflate(reinterpret_cast<const char *>(z.data()), z.size(), out.data(), out.size());
    EXPECT_EQ(out, bytes);
}

TEST(RenderTest, GifFilesHoldTheFrames) {
    const image::Indexed img = test_image();
    const std::string path = (rank_scratch_dir() / "render_frames.gif").string();
    image::Indexed second = img;
    std::reverse(second.pixels.begin(), second.pixels.end());
    image::Gif gif(path, img.width, img.height, img.palette, 5);
    gif.add(img);
    gif.add(second);
    gif.close();
    const std::vector<std::vector<uint8_t>> frames = gif_frames(path);
    ASSERT_EQ(frames.size(), 2u);
    EXPECT_EQ(frames[0], img.pixels);
    EXPECT_EQ(frames[1], second.pixels);
}

namespace {

/** @brief Runs mallard post render; its exit status. */
int run_render(const std::vector<std::string> & words, std::string * errors = nullptr) {
    cli::Args args;
    std::string error;
    EXPECT_TRUE(cli::parse(*cli::command("post render"), words, args, error)) << error;
    std::ostringstream report, messages;
    std::streambuf * out = std::cout.rdbuf(report.rdbuf());
    std::streambuf * err = std::cerr.rdbuf(messages.rdbuf());
    const int status = post_render(args);
    std::cout.rdbuf(out);
    std::cerr.rdbuf(err);
    if (errors) *errors = messages.str();
    return status;
}

std::string file_text(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

} // namespace

TEST(RenderTest, ASeriesDrawsEveryFrameAndAPngItsLast) {
    // Each rank runs the case on its own and draws its own files
    const std::string dir = (rank_scratch_dir() / "render_series").string();
    fs::remove_all(dir);
    std::ostringstream s;
    s << "[run]\nn_steps = 4\ndt = 2.0e-2\n[mesh]\ntype = \"cartesian\"\nNx = 12\nNy = 6\nLx = 1.0\nLy = 0.5\n";
    if (N_DIM == 3) s << "Nz = 3\nLz = 0.5\n[[periodic]]\nzones = [\"back\", \"front\"]\n";
    s << "[[periodic]]\nzones = [\"left\", \"right\"]\n[[periodic]]\nzones = [\"bottom\", \"top\"]\n"
      << "[initialize]\ntype = \"analytical\"\nrho = \"1.0 + 0.5 * sin(2 * pi * x)\"\n"
      << "u = " << (N_DIM == 2 ? "[1.0, 0.0]" : "[1.0, 0.0, 0.0]") << "\np = 1.0\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
      << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "[output]\ncheck_interval = 1000000\nrun_log = \"\"\n"
      << "[[write_data]]\nprefix = \"" << dir << "/flow\"\nformat = \"vtu\"\nfields = [\"RHO\", \"U\", \"P\"]\ninterval = 2\n";
    {
        Solver solver;
        solver.set_distributed(false);
        solver.init(parse_toml(s.str()));
        solver.run();
    }
    const std::string files = dir + "/flow_*.vtu";
    std::string errors;
    ASSERT_EQ(run_render({files, "-o", dir + "/rho.gif", "--size", "160", "--threads", "2"}, &errors), 0) << errors;
    EXPECT_EQ(gif_frames(dir + "/rho.gif").size(), 3u);

    ASSERT_EQ(run_render({files, "-o", dir + "/series.png", "--size", "160"}, &errors), 0) << errors;
    ASSERT_EQ(run_render({files, "--latest", "-o", dir + "/last.png", "--size", "160"}, &errors), 0) << errors;
    ASSERT_EQ(run_render({files, "--times", "0", "-o", dir + "/first.png", "--size", "160"}, &errors), 0) << errors;
    EXPECT_EQ(file_text(dir + "/series.png"), file_text(dir + "/last.png"));
    EXPECT_NE(file_text(dir + "/series.png"), file_text(dir + "/first.png"));

    EXPECT_EQ(run_render({files, "--field", "MACH", "-o", dir + "/x.png"}, &errors), 2);
    EXPECT_NE(errors.find("has no field MACH"), std::string::npos) << errors;
}
