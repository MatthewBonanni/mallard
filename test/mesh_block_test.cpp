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
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <map>
#include <random>
#include <sstream>
#include <set>
#include <string>
#include <vector>

#include "comm.h"
#include "gmsh_fixtures.h"
#include "input.h"
#include "mesh.h"
#include "mesh_block.h"
#include "mesh_file.h"
#include "test_fixtures.h"
#include "scratch.h"

namespace {

/**
 * @brief A synthetic mesh (connectivity need not be geometric for the file
 *        format) with variable-length cells and faces, split at the given
 *        fractions of each array.
 */
MeshBlock synthetic_block(const std::vector<double> & split) {
    const uint64_t n_cells = 37, n_nodes = 53, n_faces = 23;
    const int r = comm::rank();
    auto range = [&](uint64_t n) { return std::pair<uint64_t, uint64_t>(static_cast<uint64_t>(static_cast<double>(n) * split[static_cast<size_t>(r)]),
                                                                    static_cast<uint64_t>(static_cast<double>(n) * split[static_cast<size_t>(r) + 1])); };
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
        FOR_I_DIM x[i] = (i == 0) ? 0.5 * double(g) : (i == 1) ? 1e-3 * double(g) * double(g) : -double(g);
        block.node_coords.push_back(x);
    }
    for (uint64_t f = f0; f < f1; f++) {
        std::vector<uint64_t> nodes;
        for (uint32_t k = 0; k < (N_DIM == 2 ? 2 : 3 + f % 2); k++) nodes.push_back((f * 5 + k * 3) % n_nodes);
        block.add_face(nodes, static_cast<uint32_t>(f % 3));
    }
    return block;
}

std::vector<double> fractions(bool even) {
    const int p = comm::size();
    std::vector<double> split(static_cast<size_t>(p + 1));
    for (int r = 0; r <= p; r++) split[static_cast<size_t>(r)] = even ? double(r) / p : double(r * r) / (p * p);
    return split;
}

} // namespace

TEST(MeshBlockTest, HDF5FileReadsBackInEvenBlocksWhateverTheWrittenSplit) {
    if (!have_hdf5()) GTEST_SKIP() << "built without HDF5";
    if (comm::size() > 1 && !have_parallel_hdf5()) GTEST_SKIP() << "needs parallel HDF5";
    const std::string file = (scratch_dir() / "mallard_block_test.h5").string();
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
    const std::string stretched = N_DIM == 2 ? "stretching = [0.8, 1.7]\n" : "stretching = [0.8, 1.7, 0.0]\n";
    for (const std::string & type : generated_types()) {
      for (const std::string & stretching : {std::string(), stretched}) {
        SCOPED_TRACE(type + " " + stretching);
        const Input input = parse_toml("[mesh]\ntype = \"" + type +
                                             "\"\nNx = 6\nNy = 5\nNz = 3\nLx = 1.5\nLy = 1.0\nLz = 0.7\n" +
                                             stretching);
        Mesh serial;
        serial.init(input);
        const MeshBlock block = read_mesh_block(input);
        EXPECT_EQ(comm::allreduce(block.n_cells(), comm::Op::SUM), serial.n_cells);
        EXPECT_EQ(comm::allreduce(block.n_nodes(), comm::Op::SUM), serial.n_nodes);
        for (uint64_t n = 0; n < block.n_nodes(); n++) {
            FOR_I_DIM ASSERT_EQ(rtype(block.node_coords[n][i]), serial.h_node_coords(block.first_node + n, i));
        }
        std::vector<uint32_t> local(serial.n_cells);
        for (uint32_t c = 0; c < serial.n_cells; c++) local[serial.h_global_cell(c)] = c;
        for (uint64_t c = 0; c < block.n_cells(); c++) {
            // The serial 3D mesh reorients cells, which permutes their nodes
            std::vector<uint64_t> nodes(block.cell_nodes.begin() + static_cast<std::ptrdiff_t>(block.cell_offsets[c]),
                                        block.cell_nodes.begin() + static_cast<std::ptrdiff_t>(block.cell_offsets[c + 1]));
            std::vector<uint64_t> expected;
            const uint32_t s = local[block.first_cell + c];
            for (uint32_t k = 0; k < serial.h_n_nodes_of_cell(s); k++) {
                expected.push_back(serial.h_node_of_cell(s, k));
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
            std::vector<uint64_t> nodes(block.face_nodes.begin() + static_cast<std::ptrdiff_t>(block.face_offsets[f]),
                                        block.face_nodes.begin() + static_cast<std::ptrdiff_t>(block.face_offsets[f + 1]));
            std::sort(nodes.begin(), nodes.end());
            EXPECT_TRUE(serial_faces[block.zone_names[block.face_zone[f]]].count(nodes))
                << "boundary face " << f << " of zone " << block.zone_names[block.face_zone[f]];
        }
      }
    }
}

/**
 * @brief mallard post average weighs the cells of a block by their volumes
 *        (of revolution, per radian, in an axisymmetric run) and places them
 *        at their centroids without building faces; these are bit for bit
 *        the mesh's, whatever the orientation of the block's cells.
 */
TEST(MeshBlockTest, CellGeometryOfABlockIsTheMeshs) {
    std::vector<std::pair<std::string, bool>> meshes;
    for (const std::string & type : generated_types()) {
        const std::string toml = "[mesh]\ntype = \"" + type + "\"\nNx = 6\nNy = 5\nNz = 3\nLx = 1.5\nLy = 1.0\nLz = 0.7\n" +
                                 (N_DIM == 2 ? "stretching = [0.8, 1.7]\n" : "stretching = [0.8, 1.7, 0.0]\n");
        meshes.emplace_back(toml, false);
        if (N_DIM == 2 && type != "wedge") meshes.emplace_back(toml, true);
    }
    const std::string file = N_DIM == 2 ? write_temp_shared("block_geometry.msh", jittered_mixed_mesh(6))
                                        : write_temp_shared("block_geometry.msh", jittered_tetrahedral_mesh(3));
    meshes.emplace_back("[mesh]\nfilename = \"" + file + "\"\n", false);
    if (N_DIM == 2) meshes.emplace_back("[mesh]\nfilename = \"" + file + "\"\n", true);
    for (const auto & [toml, axisymmetric] : meshes) {
        SCOPED_TRACE(toml + (axisymmetric ? "axisymmetric" : ""));
        const Input input = parse_toml(toml);
        Mesh serial;
        serial.init(input);
        if (axisymmetric) serial.make_axisymmetric();
        std::vector<uint32_t> local(serial.n_cells);
        for (uint32_t c = 0; c < serial.n_cells; c++) local[serial.h_global_cell(c)] = c;
        const MeshBlock block = read_mesh_block(input);
        const std::vector<CellGeometry> geometry = block_cell_geometry(block, axisymmetric);
        ASSERT_EQ(geometry.size(), block.n_cells());
        for (uint64_t c = 0; c < block.n_cells(); c++) {
            const uint32_t s = local[block.first_cell + c];
            ASSERT_EQ(geometry[c].volume, serial.h_cell_volume(s)) << "cell " << block.first_cell + c;
            ASSERT_EQ(geometry[c].measure, serial.h_cell_measure(s)) << "cell " << block.first_cell + c;
            FOR_I_DIM ASSERT_EQ(geometry[c].centroid[i], serial.h_cell_coords(s, i)) << "cell " << block.first_cell + c;
        }
    }
}

/**
 * @brief Wall-resolved channel grids: tanh stretching clusters the nodes of a
 *        direction toward both ends, y_j = L/2 (1 + tanh(beta (2j/n - 1)) / tanh(beta)),
 *        leaves the other directions uniform, and the cells still tile the box
 *        with the periodic seams and walls in place.
 */
TEST(MeshBlockTest, StretchingClustersNodesTowardBothEnds) {
    const double beta = 1.7, Ly = 2.0;
    const uint32_t ny = 8;
    const Input input = parse_toml(
        std::string("[mesh]\ntype = \"cartesian\"\nNx = 3\nNy = 8\nNz = 4\nLx = 1.5\nLy = 2.0\nLz = 0.5\n") +
        (N_DIM == 2 ? "stretching = [0.0, 1.7]\n[[periodic]]\nzones = [\"left\", \"right\"]\n" : "stretching = [0.0, 1.7, 0.0]\n[[periodic]]\nzones = [\"left\", \"right\"]\n[[periodic]]\nzones = [\"back\", \"front\"]\n"));
    Mesh mesh;
    mesh.init(input);
    std::set<double> ys, xs;
    for (uint32_t n = 0; n < mesh.n_nodes; n++) {
        ys.insert(double(mesh.h_node_coords(n, 1)));
        xs.insert(double(mesh.h_node_coords(n, 0)));
    }
    ASSERT_EQ(ys.size(), ny + 1);
    uint32_t j = 0;
    for (double y : ys) {
        const double expected = 0.5 * Ly * (1.0 + std::tanh(beta * (2.0 * j / ny - 1.0)) / std::tanh(beta));
        EXPECT_NEAR(y, expected, precision_tol<double>(1e-14, 1e-6)) << "node " << j;
        j++;
    }
    // Uniform in x: 4 nodes 0.5 apart
    ASSERT_EQ(xs.size(), 4u);
    EXPECT_NEAR(*std::next(xs.begin()), 0.5, precision_tol<double>(1e-14, 1e-6));
    double volume = 0.0;
    for (uint32_t c = 0; c < mesh.n_cells; c++) volume += double(mesh.h_cell_volume(c));
    EXPECT_NEAR(volume, N_DIM == 2 ? 3.0 : 1.5, precision_tol<double>(1e-12, 1e-5));
    std::set<std::string> zones;
    for (FaceZone & zone : *mesh.face_zones()) {
        if (zone.get_type() == FaceZoneType::BOUNDARY) zones.insert(zone.get_name());
    }
    EXPECT_EQ(zones, (std::set<std::string>{"bottom", "top"}));
}

namespace {

/** @brief The same mesh as a Gmsh 2.2 and a Gmsh 4.1 file, and the whole block both must read as. */
struct GmshCase {
    std::string v22, v41;
    MeshBlock whole;
};

/**
 * @brief A box of n^N_DIM quadrilaterals or hexahedra, its nodes in shuffled
 *        order with shuffled, sparse tags (4.1: in three blocks) and an
 *        unused node first. Elements, in order: a point; boundary faces
 *        "low" (x = 0); cells of zone "b"; a curve (3D); faces at x = 1
 *        without a physical group; cells without a zone; cells of zone "a";
 *        faces "high" (y = 0). Points, curves (3D) and faces without a
 *        physical group are not part of the mesh.
 */
GmshCase scrambled_gmsh(uint32_t n) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> jitter(-0.3, 0.3);
    const uint32_t m = n + 1, n_grid = N_DIM == 2 ? m * m : m * m * m;
    std::vector<uint32_t> order(n_grid);
    for (uint32_t k = 0; k < n_grid; k++) order[k] = k;
    std::shuffle(order.begin(), order.end(), rng);
    std::vector<uint64_t> position(n_grid), tag(n_grid + 1);  // grid node -> file position; position -> tag
    for (uint32_t k = 0; k < n_grid; k++) position[order[k]] = k + 1;
    for (uint32_t k = 0; k <= n_grid; k++) tag[k] = 1000 + 7 * uint64_t(k);
    std::shuffle(tag.begin(), tag.end(), rng);
    std::vector<std::array<double, 3>> x(n_grid + 1, {0.5, 0.5, 0.5});
    for (uint32_t g = 0; g < n_grid; g++) {
        const uint32_t i = g % m, j = (g / m) % m, k = g / (m * m);
        x[position[g]] = {(i + jitter(rng)) / n, (j + jitter(rng)) / n, N_DIM == 3 ? (k + jitter(rng)) / n : 0.0};
    }
    auto node = [&](uint32_t i, uint32_t j, uint32_t k) { return position[(k * m + j) * m + i]; };
    struct Block {
        int dim, entity, type, physical;
        std::vector<std::vector<uint64_t>> elements;  // node positions
    };
    std::vector<Block> blocks;
    blocks.push_back({0, 1, 15, 0, {{0}}});
    auto faces = [&](int entity, int physical, auto corners) {
        Block b{N_DIM - 1, entity, N_DIM == 2 ? 1 : 3, physical, {}};
        for (uint32_t a = 0; a < n; a++) {
            for (uint32_t c = 0; c < (N_DIM == 2 ? 1 : n); c++) b.elements.push_back(corners(a, c));
        }
        blocks.push_back(b);
    };
    auto cells = [&](int entity, int physical, uint32_t from, uint32_t to) {
        Block b{N_DIM, entity, N_DIM == 2 ? 3 : 5, physical, {}};
        for (uint32_t c = from; c < to; c++) {
            const uint32_t i = c % n, j = (c / n) % n, k = c / (n * n);
            if constexpr (N_DIM == 2) {
                b.elements.push_back({node(i, j, 0), node(i + 1, j, 0), node(i + 1, j + 1, 0), node(i, j + 1, 0)});
            } else {
                b.elements.push_back({node(i, j, k), node(i + 1, j, k), node(i + 1, j + 1, k), node(i, j + 1, k),
                                      node(i, j, k + 1), node(i + 1, j, k + 1), node(i + 1, j + 1, k + 1),
                                      node(i, j + 1, k + 1)});
            }
        }
        blocks.push_back(b);
    };
    const uint32_t n_cells = N_DIM == 2 ? n * n : n * n * n;
    if constexpr (N_DIM == 2) {
        faces(1, 2, [&](uint32_t a, uint32_t) { return std::vector<uint64_t>{node(0, a + 1, 0), node(0, a, 0)}; });
        cells(1, 12, 0, n_cells / 3);
        faces(2, 0, [&](uint32_t a, uint32_t) { return std::vector<uint64_t>{node(n, a, 0), node(n, a + 1, 0)}; });
        cells(2, 0, n_cells / 3, n_cells / 2);
        cells(3, 11, n_cells / 2, n_cells);
        faces(3, 1, [&](uint32_t a, uint32_t) { return std::vector<uint64_t>{node(a, 0, 0), node(a + 1, 0, 0)}; });
    } else {
        faces(1, 2, [&](uint32_t a, uint32_t c) {
            return std::vector<uint64_t>{node(0, a, c), node(0, a, c + 1), node(0, a + 1, c + 1), node(0, a + 1, c)};
        });
        cells(1, 12, 0, n_cells / 3);
        blocks.push_back({1, 1, 1, 0, {{node(0, 0, 0), node(1, 0, 0)}}});
        faces(2, 0, [&](uint32_t a, uint32_t c) {
            return std::vector<uint64_t>{node(n, a, c), node(n, a + 1, c), node(n, a + 1, c + 1), node(n, a, c + 1)};
        });
        cells(2, 0, n_cells / 3, n_cells / 2);
        cells(3, 11, n_cells / 2, n_cells);
        faces(3, 1, [&](uint32_t a, uint32_t c) {
            return std::vector<uint64_t>{node(a, 0, c), node(a + 1, 0, c), node(a + 1, 0, c + 1), node(a, 0, c + 1)};
        });
    }

    GmshCase out;
    MeshBlock & whole = out.whole;
    whole.zone_names = {"low", "high"};
    whole.cell_zone_names = {"b", "a"};
    for (const auto & xyz : x) {
        std::array<double, N_DIM> y;
        FOR_I_DIM y[i] = double(static_cast<rtype>(xyz[i]));
        whole.node_coords.push_back(y);
    }
    for (const Block & b : blocks) {
        for (const auto & e : b.elements) {
            if (b.dim == N_DIM) {
                whole.add_cell(e);
                whole.cell_zone.push_back(b.physical == 12 ? 0 : b.physical == 11 ? 1 : Mesh::NO_CELL_ZONE);
            } else if (b.dim == N_DIM - 1 && b.physical != 0) {
                whole.add_face(e, b.physical == 2 ? 0u : 1u);
            }
        }
    }

    const std::string bd = std::to_string(N_DIM - 1), cd = std::to_string(N_DIM);
    const std::string names = "$PhysicalNames\n4\n" + bd + " 1 \"high\"\n" + bd + " 2 \"low\"\n" + cd + " 11 \"a\"\n" +
                              cd + " 12 \"b\"\n$EndPhysicalNames\n";
    size_t n_elements = 0;
    for (const Block & b : blocks) n_elements += b.elements.size();
    std::ostringstream v22, v41;
    v22 << std::setprecision(17) << "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n" << names << "$Nodes\n" << x.size() << "\n";
    for (size_t k = 0; k < x.size(); k++) v22 << tag[k] << " " << x[k][0] << " " << x[k][1] << " " << x[k][2] << "\n";
    v22 << "$EndNodes\n$Elements\n" << n_elements << "\n";
    size_t id = 1;
    for (const Block & b : blocks) {
        for (const auto & e : b.elements) {
            v22 << id++ << " " << b.type << " 2 " << b.physical << " " << b.entity;
            for (const uint64_t k : e) v22 << " " << tag[k];
            v22 << "\n";
        }
    }
    v22 << "$EndElements\n";
    out.v22 = v22.str();

    // Entities: a point, and in each dimension 1 to N_DIM entities 1 to 3 with their blocks' physical groups
    std::map<std::pair<int, int>, int> physical;
    for (const Block & b : blocks) physical[{b.dim, b.entity}] = b.physical;
    v41 << std::setprecision(17) << "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n" << names << "$Entities\n1";
    for (int d = 1; d <= 3; d++) v41 << " " << (d <= N_DIM ? 3 : 0);
    v41 << "\n1 0 0 0 0\n";
    for (int d = 1; d <= N_DIM; d++) {
        for (int e = 1; e <= 3; e++) {
            const int ph = physical.count({d, e}) ? physical[{d, e}] : 0;
            v41 << e << " 0 0 0 1 1 1 " << (ph ? "1 " + std::to_string(ph) : "0") << " 0\n";
        }
    }
    const size_t cut[4] = {0, 1, 1 + x.size() / 3, x.size()};
    v41 << "$EndEntities\n$Nodes\n3 " << x.size() << " 1000 " << 1000 + 7 * n_grid << "\n";
    for (int b = 0; b < 3; b++) {
        v41 << N_DIM << " " << b + 1 << " 0 " << cut[b + 1] - cut[b] << "\n";
        for (size_t k = cut[b]; k < cut[b + 1]; k++) v41 << tag[k] << "\n";
        for (size_t k = cut[b]; k < cut[b + 1]; k++) v41 << x[k][0] << " " << x[k][1] << " " << x[k][2] << "\n";
    }
    v41 << "$EndNodes\n$Elements\n" << blocks.size() << " " << n_elements << " 1 " << n_elements << "\n";
    id = 1;
    for (const Block & b : blocks) {
        v41 << b.dim << " " << b.entity << " " << b.type << " " << b.elements.size() << "\n";
        for (const auto & e : b.elements) {
            v41 << id++;
            for (const uint64_t k : e) v41 << " " << tag[k];
            v41 << "\n";
        }
    }
    v41 << "$EndElements\n";
    out.v41 = v41.str();
    return out;
}

/** @brief Expect block to be part r of p of whole (see MeshBlock). */
void expect_part(const MeshBlock & whole, const MeshBlock & block, int r, int p) {
    auto slice = [](const auto & values, uint64_t a, uint64_t b) {
        return std::vector<typename std::decay_t<decltype(values)>::value_type>(values.begin() + std::ptrdiff_t(a),
                                                                               values.begin() + std::ptrdiff_t(b));
    };
    const uint64_t c0 = block_begin(whole.n_cells(), r, p), c1 = block_begin(whole.n_cells(), r + 1, p);
    ASSERT_EQ(block.first_cell, c0);
    ASSERT_EQ(block.n_cells(), c1 - c0);
    EXPECT_EQ(block.cell_nodes, slice(whole.cell_nodes, whole.cell_offsets[c0], whole.cell_offsets[c1]));
    for (uint64_t c = 0; c <= block.n_cells(); c++) {
        EXPECT_EQ(block.cell_offsets[c], whole.cell_offsets[c0 + c] - whole.cell_offsets[c0]);
    }
    EXPECT_EQ(block.cell_zone, slice(whole.cell_zone, c0, c1));
    EXPECT_EQ(block.cell_zone_names, whole.cell_zone_names);
    const uint64_t n0 = block_begin(whole.n_nodes(), r, p), n1 = block_begin(whole.n_nodes(), r + 1, p);
    EXPECT_EQ(block.first_node, n0);
    EXPECT_EQ(block.node_coords, slice(whole.node_coords, n0, n1));
    const uint64_t f0 = block_begin(whole.n_faces(), r, p), f1 = block_begin(whole.n_faces(), r + 1, p);
    ASSERT_EQ(block.n_faces(), f1 - f0);
    EXPECT_EQ(block.face_nodes, slice(whole.face_nodes, whole.face_offsets[f0], whole.face_offsets[f1]));
    EXPECT_EQ(block.face_zone, slice(whole.face_zone, f0, f1));
    EXPECT_EQ(block.zone_names, whole.zone_names);
    EXPECT_TRUE(block.face_high_order_offsets.empty());
}

} // namespace

/**
 * @brief Each rank's block of a Gmsh file, which the ranks read in parallel,
 *        is its part of the mesh, whatever the order and tags of the nodes,
 *        the entity blocks of format 4.1, and the elements that are not part
 *        of the mesh between those that are.
 */
TEST(MeshBlockTest, EachRankReadsItsPartOfAGmshFile) {
    const GmshCase mesh = scrambled_gmsh(N_DIM == 2 ? 7 : 4);
    const int r = comm::rank(), p = comm::size();
    for (const auto & [name, content] : {std::pair<std::string, std::string>{"mallard_scrambled_22.msh", mesh.v22},
                                         std::pair<std::string, std::string>{"mallard_scrambled_41.msh", mesh.v41}}) {
        SCOPED_TRACE(name);
        expect_part(mesh.whole, read_file_block(write_temp_shared(name, content)), r, p);
    }
}

/**
 * @brief Every rank's block of a mesh file is its part of the whole file as
 *        one rank reads it: cells, nodes, boundary faces and their zones,
 *        cell zones and high-order nodes.
 */
TEST(MeshBlockTest, EachRankGetsItsPartOfTheMeshFile) {
#if N_DIM == 2
    const std::vector<std::string> files = {write_temp_shared("mallard_part_zones.msh", jittered_mixed_mesh(6, true)),
                                            write_temp_shared("mallard_part_curved.msh", annulus_gmsh(3, 16, 2))};
#else
    const std::vector<std::string> files = {write_temp_shared("mallard_part_curved.msh", shell_gmsh(2, 2, false, true))};
#endif
    const int r = comm::rank(), p = comm::size();
    for (const std::string & file : files) {
        SCOPED_TRACE(file);
        const MeshFileData data = read_mesh_file(file);
        const MeshBlock block = read_file_block(file);
        ASSERT_EQ(block.first_cell, block_begin(data.cells.size(), r, p));
        ASSERT_EQ(block.n_cells(), block_begin(data.cells.size(), r + 1, p) - block.first_cell);
        for (uint64_t c = 0; c < block.n_cells(); c++) {
            const auto cell = data.cells[block.first_cell + c];
            ASSERT_EQ(std::vector<uint64_t>(block.cell_nodes.begin() + std::ptrdiff_t(block.cell_offsets[c]),
                                            block.cell_nodes.begin() + std::ptrdiff_t(block.cell_offsets[c + 1])),
                      std::vector<uint64_t>(cell.begin(), cell.end()));
        }
        ASSERT_EQ(block.first_node, block_begin(data.nodes.size(), r, p));
        ASSERT_EQ(block.n_nodes(), block_begin(data.nodes.size(), r + 1, p) - block.first_node);
        for (uint64_t n = 0; n < block.n_nodes(); n++) {
            FOR_I_DIM EXPECT_EQ(block.node_coords[n][i], double(data.nodes[block.first_node + n][i]));
        }
        const uint64_t first_face = block_begin(data.boundary_faces.size(), r, p);
        ASSERT_EQ(block.n_faces(), block_begin(data.boundary_faces.size(), r + 1, p) - first_face);
        std::vector<std::array<double, N_DIM>> high_order;
        for (uint64_t f = 0; f < block.n_faces(); f++) {
            const Mesh::BoundaryFace & face = data.boundary_faces[first_face + f];
            EXPECT_EQ(block.zone_names[block.face_zone[f]], face.zone);
            EXPECT_EQ(std::vector<uint64_t>(block.face_nodes.begin() + std::ptrdiff_t(block.face_offsets[f]),
                                            block.face_nodes.begin() + std::ptrdiff_t(block.face_offsets[f + 1])),
                      std::vector<uint64_t>(face.nodes.begin(), face.nodes.end()));
            for (const curved::Vec3 & x : data.boundary_high_order[first_face + f]) {
                std::array<double, N_DIM> y;
                FOR_I_DIM y[i] = x[i];
                high_order.push_back(y);
            }
        }
        EXPECT_EQ(block.face_high_order_nodes, high_order);
        EXPECT_EQ(block.cell_zone_names, data.cell_zone_names);
        ASSERT_EQ(block.cell_zone.size(), data.cell_zone_names.empty() ? 0 : block.n_cells());
        for (uint64_t c = 0; c < block.cell_zone.size(); c++) {
            const uint64_t g = block.first_cell + c;
            EXPECT_EQ(block.cell_zone[c], g < data.cell_zone.size() ? data.cell_zone[g] : Mesh::NO_CELL_ZONE);
        }
    }
}

TEST(MeshBlockTest, TransformOnReadMapsNodesAndHighOrderNodes) {
    // 2D: scale [2, 3], rotate 90 deg, translate [1, -1]: (x, y) -> (1 - 3y, 2x - 1).
    // 3D: scale 2, rotate 120 deg about [1, 1, 1] (cycles the axes), translate [1, 2, 3]: (x, y, z) -> (2z + 1, 2x + 2, 2y + 3)
#if N_DIM == 2
    const std::string file = write_temp_shared("mallard_transform.msh", annulus_gmsh(3, 16, 2));
    const std::string transform = "[mesh.transform]\nscale = [2, 3]\nrotate = 90\ntranslate = [1, -1]\n";
    auto expected = [](const auto & x) {
        return std::array<double, N_DIM>{1.0 - 3.0 * double(x[1]), 2.0 * double(x[0]) - 1.0};
    };
    const double det = 6.0;
#else
    const std::string file = write_temp_shared("mallard_transform_3d.msh", shell_gmsh(2, 2, false, true));
    const std::string transform =
        "[mesh.transform]\nscale = 2\nrotate = {axis = [1, 1, 1], angle = 120}\ntranslate = [1, 2, 3]\n";
    auto expected = [](const auto & x) {
        return std::array<double, N_DIM>{2.0 * double(x[2]) + 1.0, 2.0 * double(x[0]) + 2.0, 2.0 * double(x[1]) + 3.0};
    };
    const double det = 8.0;
#endif
    const double tol = precision_tol<double>(1e-12, 1e-5);
    // Cells keep the file's order, so that b's cells pair with a's
    const Input input = parse_toml("[mesh]\ntype = \"file\"\nfilename = \"" + file + "\"\ncell_order = \"input\"\n" + transform);

    const MeshBlock plain = read_file_block(file), moved = read_mesh_block(input);
    ASSERT_EQ(moved.n_nodes(), plain.n_nodes());
    for (uint64_t n = 0; n < plain.n_nodes(); n++) {
        FOR_I_DIM EXPECT_NEAR(moved.node_coords[n][i], expected(plain.node_coords[n])[i], tol);
    }
    ASSERT_EQ(moved.face_high_order_nodes.size(), plain.face_high_order_nodes.size());
    ASSERT_FALSE(plain.face_high_order_nodes.empty());
    for (size_t k = 0; k < plain.face_high_order_nodes.size(); k++) {
        FOR_I_DIM EXPECT_NEAR(moved.face_high_order_nodes[k][i], expected(plain.face_high_order_nodes[k])[i], tol);
    }

    if (comm::size() > 1) return;
    Mesh a, b;
    a.init_file(file);
    b.init(input);
    ASSERT_EQ(b.n_cells, a.n_cells);
    ASSERT_EQ(b.n_faces, a.n_faces);
    for (uint32_t c = 0; c < a.n_cells; c++) {
        EXPECT_NEAR(double(b.h_cell_volume(c)), det * double(a.h_cell_volume(c)), tol * det);
        std::array<rtype, N_DIM> x;
        FOR_I_DIM x[i] = a.h_cell_coords(c, i);
        FOR_I_DIM EXPECT_NEAR(double(b.h_cell_coords(c, i)), expected(x)[i], 10 * tol);
    }
    for (FaceZone & zone : *a.face_zones()) {
        ASSERT_NE(b.get_face_zone(zone.get_name()), nullptr) << zone.get_name();
        EXPECT_EQ(b.get_face_zone(zone.get_name())->n_faces(), zone.n_faces()) << zone.get_name();
    }
    ASSERT_EQ(b.high_order_faces.size(), a.high_order_faces.size());
    for (size_t f = 0; f < a.high_order_faces.size(); f++) {
        for (size_t k = 0; k < a.high_order_faces[f].nodes.size(); k++) {
            const auto y = expected(a.high_order_faces[f].nodes[k]);
            FOR_I_DIM EXPECT_NEAR(b.high_order_faces[f].nodes[k][i], y[i], tol);
        }
    }
}

TEST(MeshBlockTest, TransformIsRejectedForGeneratedMeshesAndUnknownKeys) {
    EXPECT_THROW(mesh_transform(parse_toml("[mesh]\ntype = \"cartesian\"\n[mesh.transform]\nscale = 2\n")),
                 InputError);
    EXPECT_THROW(mesh_transform(parse_toml("[mesh]\ntype = \"file\"\n[mesh.transform]\nshift = [1, 1]\n")),
                 InputError);
    EXPECT_THROW(mesh_transform(parse_toml("[mesh]\ntype = \"file\"\n[mesh.transform]\nscale = 0\n")), InputError);
}
