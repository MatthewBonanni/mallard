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
#include <limits>
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

std::shared_ptr<Mesh> box(const std::string & extra = "", uint32_t m = M) {
    std::string text = "[mesh]\ntype = \"cartesian\"\nNx = " + std::to_string(m) + "\nNy = " + std::to_string(m) + "\n";
    if constexpr (N_DIM == 3) text += "Nz = " + std::to_string(m) + "\n";
    auto mesh = std::make_shared<Mesh>();
    mesh->init(parse_toml(text + extra));
    mesh->copy_host_to_device();
    return mesh;
}

uint32_t power(uint32_t base) {
    uint32_t n = 1;
    for (uint32_t a = 0; a < N_DIM; a++) n *= base;
    return n;
}

} // namespace

/**
 * @brief In a uniform box every cell has a shape: one for the cells whose
 *        reach has no boundary face, and one per distance (up to the reach)
 *        from each face along each axis.
 */
TEST(StencilShapes, AUniformBoxHasAShapePerDistanceFromItsFaces) {
    const auto mesh = box();
    ASSERT_EQ(mesh->structured_blocks.size(), 1u);
    for (uint32_t reach : {1u, 2u}) {
        const StencilShapes shapes = stencil_shapes(*mesh, reach);
        EXPECT_EQ(std::count(shapes.shape.begin(), shapes.shape.end(), StencilShapes::NONE), 0);
        EXPECT_EQ(shapes.first.size(), power(2 * reach + 3));
        const uint32_t center = (reach + 1) * (1 + M + (N_DIM == 3 ? M * M : 0));
        EXPECT_EQ(uint32_t(std::count(shapes.shape.begin(), shapes.shape.end(), shapes.shape[center])),
                  power(M - 2 * reach - 2));
    }
}

/** @brief Stretched along y, a box has a shape per plane of constant y and distance from the other faces. */
TEST(StencilShapes, AStretchedBoxHasAShapePerPlane) {
    const auto mesh = box(N_DIM == 2 ? "stretching = [0.0, 1.7]\n" : "stretching = [0.0, 1.7, 0.0]\n");
    const uint32_t reach = 2;
    const StencilShapes shapes = stencil_shapes(*mesh, reach);
    EXPECT_EQ(shapes.first.size(), M * power(2 * reach + 3) / (2 * reach + 3));
    std::map<double, std::set<uint32_t>> by_plane;
    for (uint32_t c = 0; c < shapes.shape.size(); c++) {
        by_plane[std::round(1e4 * double(mesh->h_cell_coords(c, 1)))].insert(shapes.shape[c]);
    }
    EXPECT_EQ(by_plane.size(), M);
    for (const auto & [y, set] : by_plane) EXPECT_EQ(set.size(), power(2 * reach + 3) / (2 * reach + 3)) << "plane " << y;
}

/**
 * @brief A periodic box has two shapes, one per way of keying, across its
 *        periodic faces. A node moved 1e-3 of a cell changes the shapes of the
 *        cells whose reach includes a cell around it, and only those.
 */
TEST(StencilShapes, APeriodicBoxHasTwoShapesAndANodeMovedChangesTheShapesWithinReach) {
    const uint32_t m = 16, reach = 5;
    std::string periodic = "[[periodic]]\nzones = [\"left\", \"right\"]\n[[periodic]]\nzones = [\"bottom\", \"top\"]\n";
    if constexpr (N_DIM == 3) periodic += "[[periodic]]\nzones = [\"back\", \"front\"]\n";
    const auto mesh = box(periodic, m);
    // One shape keyed through the block's indices, one through the vertex neighbors across the periodic faces
    const StencilShapes whole = stencil_shapes(*mesh, reach);
    EXPECT_EQ(std::count(whole.shape.begin(), whole.shape.end(), StencilShapes::NONE), 0);
    ASSERT_EQ(whole.first.size(), 2u);
    // The node at (8, 8[, 8]) / m, in the cells (7..8, 7..8[, 7..8])
    uint32_t middle = 0;
    for (uint32_t v = 0; v < mesh->n_nodes; v++) {
        bool at = true;
        FOR_I_DIM at = at && std::abs(mesh->h_node_coords(v, i) - 0.5_r) < 1e-6_r;
        if (at) middle = v;
    }
    mesh->h_node_coords(middle, 0) += 1e-3_r / rtype(m);
    const StructuredBlock & block = mesh->structured_blocks.at(0);
    auto cell_at = [&](uint32_t i, uint32_t j) {
        uint32_t l = i + block.n[0] * j;
        if constexpr (N_DIM == 3) l += block.n[0] * block.n[1] * j;
        return block.first + l;
    };
    for (const uint32_t r : {reach, reach - 1}) {
        const StencilShapes shapes = stencil_shapes(*mesh, r);
        const uint32_t unchanged = shapes.shape[cell_at(0, 0)];
        EXPECT_EQ(shapes.shape[cell_at(7 - r - 1, 7)], unchanged) << "reach " << r;
        EXPECT_NE(shapes.shape[cell_at(7 - r, 7)], unchanged) << "reach " << r;
    }
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
 * @brief On a stretched box, TENO5's stencils read cells up to 3 away along an
 *        axis, within its gather depth, and the central pseudo-inverses of two
 *        cells of one shape over the gather depth agree to round-off once their
 *        entries are matched by offset.
 */
TEST(StencilShapes, CellsOfOneShapeHaveOneTENOTable) {
    const auto mesh = box("stretching = [0.0, 1.7, 0.0]\n", 14);
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::SYMMETRY, 1.4_r);
    TENO teno;
    teno.set_mesh(mesh);
    teno.set_boundaries(bd);
    teno.init(parse_toml("type = \"TENO\"\norder = 5\n"));
    uint32_t depth = 0;
    for (const uint8_t d : teno.gather_depth) depth = std::max<uint32_t>(depth, d);
    const StencilShapes shapes = stencil_shapes(*mesh, depth);

    using Offset = std::array<int32_t, N_DIM>;
    std::vector<std::map<Offset, std::vector<double>>> table(shapes.first.size());
    double largest = 0.0, worst = 0.0;
    uint32_t compared = 0;
    int32_t reach_seen = 0;
    teno.for_each_stencil([&](uint32_t c, const TENO::CellStencils & s) {
        if (c >= shapes.shape.size()) return;
        const Offset x = index_of(*mesh, c);
        for (uint32_t a = 0; a < N_DIM; a++) {
            if (uint32_t(x[a]) <= depth || uint32_t(x[a]) + depth + 1 >= 14u) return;  // images near the walls
        }
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
    EXPECT_LT(worst, 1e3 * double(std::numeric_limits<rtype>::epsilon()) * largest);
}
#endif
