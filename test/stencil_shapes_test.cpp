/**
 * @file stencil_shapes_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Stencil shapes of the cells of structured blocks.
 * @version 0.1
 * @date 2026-11-13
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "face_reconstruction.h"
#include "stencil_shapes.h"
#include "test_fixtures.h"

namespace {

constexpr uint32_t M = N_DIM == 2 ? 14 : 10;

std::shared_ptr<Mesh> box(const std::string & extra = "") {
    std::string text = "[mesh]\ntype = \"cartesian\"\nNx = " + std::to_string(M) + "\nNy = " + std::to_string(M) + "\n";
    if constexpr (N_DIM == 3) text += "Nz = " + std::to_string(M) + "\n";
    auto mesh = std::make_shared<Mesh>();
    mesh->init(parse_toml(text + extra));
    mesh->copy_host_to_device();
    return mesh;
}

uint32_t interior(uint32_t reach) {
    uint32_t n = 1;
    for (uint32_t a = 0; a < N_DIM; a++) n *= M - 2 * reach;
    return n;
}

} // namespace

/** @brief A uniform box has one shape; cells within reach of its faces have none. */
TEST(StencilShapes, AUniformBoxHasOneShape) {
    const auto mesh = box();
    ASSERT_EQ(mesh->structured_blocks.size(), 1u);
    for (uint32_t reach : {1u, 2u}) {
        const StencilShapes shapes = stencil_shapes(*mesh, reach);
        ASSERT_EQ(shapes.first.size(), 1u);
        EXPECT_EQ(shapes.first[0], reach * (1 + M + (N_DIM == 3 ? M * M : 0)));
        EXPECT_EQ(uint32_t(std::count(shapes.shape.begin(), shapes.shape.end(), 0u)), interior(reach));
    }
}

/** @brief Stretched along y, a box has one shape per plane of constant j, mirror planes included. */
TEST(StencilShapes, AStretchedBoxHasOneShapePerPlane) {
    const auto mesh = box(N_DIM == 2 ? "stretching = [0.0, 1.7]\n" : "stretching = [0.0, 1.7, 0.0]\n");
    const uint32_t reach = 2;
    const StencilShapes shapes = stencil_shapes(*mesh, reach);
    EXPECT_EQ(shapes.first.size(), M - 2 * reach);
    std::map<double, std::set<uint32_t>> by_plane;
    for (uint32_t c = 0; c < shapes.shape.size(); c++) {
        if (shapes.shape[c] != StencilShapes::NONE) by_plane[std::round(1e9 * double(mesh->h_cell_coords(c, 1)))].insert(shapes.shape[c]);
    }
    EXPECT_EQ(by_plane.size(), M - 2 * reach);
    for (const auto & [j, set] : by_plane) EXPECT_EQ(set.size(), 1u) << "plane " << j;
}

#if Mallard_DIM == 3
namespace {

/** (i, j[, k]) of block cell c of the one-block mesh. */
std::array<int32_t, N_DIM> index_of(const Mesh & mesh, uint32_t c) {
    const StructuredBlock & block = mesh.structured_blocks.at(0);
    std::array<int32_t, N_DIM> x;
    for (uint32_t a = 0, r = c - block.first; a < N_DIM; r /= block.n[a], a++) x[a] = int32_t(r % block.n[a]);
    return x;
}

} // namespace

/**
 * @brief On a stretched box, TENO5 reads cells up to 3 away along an axis
 *        (one more than its vertex layers), and the central pseudo-inverses of
 *        two cells of one shape at that reach agree to round-off once their
 *        entries are matched by offset.
 */
TEST(StencilShapes, CellsOfOneShapeHaveOneTENOTable) {
    const auto mesh = box("stretching = [0.0, 1.7, 0.0]\n");
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::SYMMETRY, 1.4_r);
    TENO teno;
    teno.set_mesh(mesh);
    teno.set_boundaries(bd);
    teno.init(parse_toml("type = \"TENO\"\norder = 5\n"));
    const StencilShapes shapes = stencil_shapes(*mesh, 3);

    using Offset = std::array<int32_t, N_DIM>;
    std::vector<std::map<Offset, std::vector<double>>> table(shapes.first.size());
    double largest = 0.0, worst = 0.0;
    uint32_t compared = 0;
    int32_t reach_seen = 0;
    teno.for_each_stencil([&](uint32_t c, const TENO::CellStencils & s) {
        if (c >= shapes.shape.size() || shapes.shape[c] == StencilShapes::NONE) return;
        const Offset x = index_of(*mesh, c);
        std::map<Offset, std::vector<double>> entries;
        for (uint32_t slot = 0; slot < s.cells.size(); slot++) {
            ASSERT_GE(s.cells[slot], 0);
            Offset o = index_of(*mesh, uint32_t(s.cells[slot]));
            for (uint32_t a = 0; a < N_DIM; a++) {
                o[a] -= x[a];
                reach_seen = std::max(reach_seen, std::abs(o[a]));
            }
            if (slot < s.central) {
                const auto * p = s.pinv.data() + size_t(slot) * teno.n_dof_large;
                entries[o].assign(p, p + teno.n_dof_large);
            }
        }
        auto & ref = table[shapes.shape[c]];
        if (ref.empty()) {
            ref = std::move(entries);
            return;
        }
        ASSERT_EQ(entries.size(), ref.size());
        for (const auto & [o, column] : entries) {
            const auto it = ref.find(o);
            ASSERT_NE(it, ref.end());
            for (size_t l = 0; l < column.size(); l++) {
                largest = std::max(largest, std::abs(it->second[l]));
                worst = std::max(worst, std::abs(column[l] - it->second[l]));
            }
        }
        compared++;
    }, true);
    EXPECT_GT(compared, 0u);
    EXPECT_EQ(reach_seen, 3);
    EXPECT_LT(worst, 1e-12 * largest);
}
#endif
