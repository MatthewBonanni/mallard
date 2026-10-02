/**
 * @file mesh_block_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Mesh blocks: HDF5 mesh files and generated blocks.
 * @version 0.4
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "comm.h"
#include "mesh.h"
#include "mesh_block.h"
#include "test_fixtures.h"

namespace {

/**
 * @brief A synthetic mesh (connectivity need not be geometric for the file
 *        format) with variable-length cells and faces, split at the given
 *        fractions of each array.
 */
MeshBlock synthetic_block(const std::vector<double> & split) {
    const uint64_t n_cells = 37, n_nodes = 53, n_faces = 23;
    const int r = comm::rank();
    auto range = [&](uint64_t n) { return std::pair<uint64_t, uint64_t>(n * split[r], n * split[r + 1]); };
    MeshBlock block;
    block.zone_names = {"inlet", "wall", "outlet"};
    const auto [c0, c1] = range(n_cells);
    const auto [n0, n1] = range(n_nodes);
    const auto [f0, f1] = range(n_faces);
    block.first_cell = c0;
    block.first_node = n0;
    static const uint32_t cell_sizes[2][4] = {{3, 4, 3, 4}, {4, 5, 6, 8}};
    for (uint64_t c = c0; c < c1; c++) {
        std::vector<uint64_t> nodes;
        for (uint32_t k = 0; k < cell_sizes[N_DIM - 2][c % 4]; k++) nodes.push_back((c * 7 + k * 11) % n_nodes);
        block.add_cell(nodes);
    }
    for (uint64_t g = n0; g < n1; g++) {
        std::array<double, N_DIM> x;
        FOR_I_DIM x[i] = (i == 0) ? 0.5 * g : (i == 1) ? 1e-3 * g * g : -double(g);
        block.node_coords.push_back(x);
    }
    for (uint64_t f = f0; f < f1; f++) {
        std::vector<uint64_t> nodes;
        for (uint32_t k = 0; k < (N_DIM == 2 ? 2 : 3 + f % 2); k++) nodes.push_back((f * 5 + k * 3) % n_nodes);
        block.add_face(nodes, f % 3);
    }
    return block;
}

std::vector<double> fractions(bool even) {
    const int p = comm::size();
    std::vector<double> split(p + 1);
    for (int r = 0; r <= p; r++) split[r] = even ? double(r) / p : double(r * r) / (p * p);
    return split;
}

} // namespace

TEST(MeshBlockTest, HDF5FileReadsBackInEvenBlocksWhateverTheWrittenSplit) {
    if (!have_hdf5()) GTEST_SKIP() << "built without HDF5";
    if (comm::size() > 1 && !have_parallel_hdf5()) GTEST_SKIP() << "needs parallel HDF5";
    const std::string file = (std::filesystem::temp_directory_path() / "mallard_block_test.h5").string();
    // Written from uneven blocks, so every offset in the file depends on the other ranks
    write_mesh_h5(file, synthetic_block(fractions(false)));
    comm::barrier();
    const MeshBlock read = read_mesh_h5(file);
    // Even blocks: block_begin(n, r, p) is n * r / p
    const MeshBlock expected = synthetic_block(fractions(true));
    EXPECT_EQ(read.first_cell, expected.first_cell);
    EXPECT_EQ(read.first_node, expected.first_node);
    EXPECT_EQ(read.cell_offsets, expected.cell_offsets);
    EXPECT_EQ(read.cell_nodes, expected.cell_nodes);
    EXPECT_EQ(read.node_coords, expected.node_coords);
    EXPECT_EQ(read.face_offsets, expected.face_offsets);
    EXPECT_EQ(read.face_nodes, expected.face_nodes);
    EXPECT_EQ(read.face_zone, expected.face_zone);
    EXPECT_EQ(read.zone_names, expected.zone_names);
    comm::barrier();
}

namespace {

std::vector<std::string> generated_types() {
    if constexpr (N_DIM == 2) return {"cartesian", "cartesian_tri", "wedge"};
    return {"cartesian", "cartesian_tet", "cartesian_prism", "cartesian_pyramid", "cartesian_mixed"};
}

} // namespace

/**
 * @brief Global ids are what restart files and rank-count comparisons key on,
 *        so each rank's generated block must number cells and nodes as the
 *        serial generator does.
 */
TEST(MeshBlockTest, GeneratedBlocksNumberCellsAndNodesAsTheSerialMesh) {
    for (const std::string & type : generated_types()) {
        SCOPED_TRACE(type);
        const toml::value input = parse_toml("[mesh]\ntype = \"" + type +
                                             "\"\nNx = 6\nNy = 5\nNz = 3\nLx = 1.5\nLy = 1.0\nLz = 0.7\n");
        Mesh serial;
        serial.init(input);
        const MeshBlock block = read_mesh_block(input);
        EXPECT_EQ(comm::allreduce(block.n_cells(), comm::Op::SUM), serial.n_cells);
        EXPECT_EQ(comm::allreduce(block.n_nodes(), comm::Op::SUM), serial.n_nodes);
        for (uint64_t n = 0; n < block.n_nodes(); n++) {
            FOR_I_DIM ASSERT_EQ(rtype(block.node_coords[n][i]), serial.h_node_coords(block.first_node + n, i));
        }
        for (uint64_t c = 0; c < block.n_cells(); c++) {
            // The serial 3D mesh reorients cells, which permutes their nodes
            std::vector<uint64_t> nodes(block.cell_nodes.begin() + block.cell_offsets[c],
                                        block.cell_nodes.begin() + block.cell_offsets[c + 1]);
            std::vector<uint64_t> expected;
            for (uint32_t k = 0; k < serial.h_n_nodes_of_cell(block.first_cell + c); k++) {
                expected.push_back(serial.h_node_of_cell(block.first_cell + c, k));
            }
            std::sort(nodes.begin(), nodes.end());
            std::sort(expected.begin(), expected.end());
            ASSERT_EQ(nodes, expected) << "cell " << block.first_cell + c;
        }
        // Every boundary face of the block, in the zone the serial mesh puts it in
        std::map<std::string, std::set<std::vector<uint64_t>>> serial_faces;
        for (FaceZone & zone : *serial.face_zones()) {
            if (zone.get_type() != FaceZoneType::BOUNDARY) continue;
            for (uint32_t i = 0; i < zone.n_faces(); i++) {
                std::vector<uint64_t> nodes;
                for (uint32_t k = 0; k < serial.h_n_nodes_of_face(zone.h_faces(i)); k++) {
                    nodes.push_back(serial.h_node_of_face(zone.h_faces(i), k));
                }
                std::sort(nodes.begin(), nodes.end());
                serial_faces[zone.get_name()].insert(nodes);
            }
        }
        uint64_t n_serial = 0;
        for (const auto & [name, faces] : serial_faces) n_serial += faces.size();
        EXPECT_EQ(comm::allreduce(block.n_faces(), comm::Op::SUM), n_serial);
        for (uint64_t f = 0; f < block.n_faces(); f++) {
            std::vector<uint64_t> nodes(block.face_nodes.begin() + block.face_offsets[f],
                                        block.face_nodes.begin() + block.face_offsets[f + 1]);
            std::sort(nodes.begin(), nodes.end());
            EXPECT_TRUE(serial_faces[block.zone_names[block.face_zone[f]]].count(nodes))
                << "boundary face " << f << " of zone " << block.zone_names[block.face_zone[f]];
        }
    }
}
