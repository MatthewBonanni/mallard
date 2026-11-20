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
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "comm.h"
#include "data_paths.h"
#include "gmsh_fixtures.h"
#include "input.h"
#include "mesh.h"
#include "mesh_block.h"
#include "mesh_file.h"
#include "read_parallel.h"
#include "renumbered_mesh.h"
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

/**
 * @brief Every rank's block of a mesh file is its part of the mesh one rank
 *        reads alone, whatever the number of ranks: cells, nodes, boundary
 *        faces and their zones, cell zones and high-order nodes, for every
 *        format (CGNS zones joined by 1-to-1 connectivities included).
 */
TEST(MeshBlockTest, EachRankGetsItsPartOfTheMeshFile) {
    const std::string meshes = source_dir() + "/test/data/meshes/";
#if N_DIM == 2
    std::vector<std::string> files = {write_temp_shared("mallard_part_zones.msh", jittered_mixed_mesh(6, true)),
                                      write_temp_shared("mallard_part_curved.msh", annulus_gmsh(3, 16, 2))};
    for (const char * name : {"cyl2d_fluent.msh", "cyl2d_fluent_bin.msh", "cyl2d_foam", "cyl2d_foam_bin"}) files.push_back(meshes + name);
    if (have_cgns()) {
        for (const char * name : {"cyl2d.cgns", "cyl2d_zones.cgns", "cyl2d_o2.cgns"}) files.push_back(meshes + name);
    }
#else
    std::vector<std::string> files = {write_temp_shared("mallard_part_curved.msh", shell_gmsh(2, 2, false, true))};
    for (const char * name : {"cyl3d_fluent.msh", "cyl3d_fluent_bin.msh", "cyl3d_foam", "cyl3d_foam_bin"}) files.push_back(meshes + name);
    if (have_cgns()) {
        for (const char * name : {"cyl3d.cgns", "cyl3d_zones.cgns", "cyl3d_tet.cgns"}) files.push_back(meshes + name);
    }
#endif
    const int r = comm::rank(), p = comm::size();
    for (const std::string & file : files) {
        SCOPED_TRACE(file);
        MeshBlock whole;
        {
            comm::SelfScope self;
            whole = read_file_block(file);
        }
        const MeshBlock block = read_file_block(file);
        ASSERT_EQ(block.first_cell, block_begin(whole.n_cells(), r, p));
        ASSERT_EQ(block.n_cells(), block_begin(whole.n_cells(), r + 1, p) - block.first_cell);
        const uint64_t first_corner = whole.cell_offsets[block.first_cell];
        for (uint64_t c = 0; c <= block.n_cells(); c++) {
            ASSERT_EQ(block.cell_offsets[c], whole.cell_offsets[block.first_cell + c] - first_corner);
        }
        ASSERT_TRUE(std::equal(block.cell_nodes.begin(), block.cell_nodes.end(), whole.cell_nodes.begin() + std::ptrdiff_t(first_corner)));
        ASSERT_EQ(block.first_node, block_begin(whole.n_nodes(), r, p));
        ASSERT_EQ(block.n_nodes(), block_begin(whole.n_nodes(), r + 1, p) - block.first_node);
        EXPECT_TRUE(std::equal(block.node_coords.begin(), block.node_coords.end(),
                               whole.node_coords.begin() + std::ptrdiff_t(block.first_node)));
        const uint64_t first_face = block_begin(whole.n_faces(), r, p);
        ASSERT_EQ(block.n_faces(), block_begin(whole.n_faces(), r + 1, p) - first_face);
        EXPECT_EQ(block.zone_names, whole.zone_names);
        ASSERT_EQ(block.face_high_order_offsets.empty(), whole.face_high_order_offsets.empty());
        for (uint64_t f = 0; f < block.n_faces(); f++) {
            const uint64_t g = first_face + f;
            EXPECT_EQ(block.face_zone[f], whole.face_zone[g]);
            EXPECT_EQ(std::vector<uint64_t>(block.face_nodes.begin() + std::ptrdiff_t(block.face_offsets[f]),
                                            block.face_nodes.begin() + std::ptrdiff_t(block.face_offsets[f + 1])),
                      std::vector<uint64_t>(whole.face_nodes.begin() + std::ptrdiff_t(whole.face_offsets[g]),
                                            whole.face_nodes.begin() + std::ptrdiff_t(whole.face_offsets[g + 1])));
            if (whole.face_high_order_offsets.empty()) continue;
            EXPECT_TRUE(std::equal(block.face_high_order_nodes.begin() + std::ptrdiff_t(block.face_high_order_offsets[f]),
                                   block.face_high_order_nodes.begin() + std::ptrdiff_t(block.face_high_order_offsets[f + 1]),
                                   whole.face_high_order_nodes.begin() + std::ptrdiff_t(whole.face_high_order_offsets[g]),
                                   whole.face_high_order_nodes.begin() + std::ptrdiff_t(whole.face_high_order_offsets[g + 1])));
        }
        EXPECT_EQ(block.cell_zone_names, whole.cell_zone_names);
        ASSERT_EQ(block.cell_zone.size(), whole.cell_zone.empty() ? 0 : block.n_cells());
        EXPECT_TRUE(std::equal(block.cell_zone.begin(), block.cell_zone.end(), whole.cell_zone.begin() + std::ptrdiff_t(block.first_cell)));
    }
}

namespace {

/** @brief A whole block's boundary faces in order of zone, in block order within a zone. */
std::vector<uint64_t> sorted_faces(const MeshBlock & block) {
    std::vector<uint64_t> faces(block.n_faces());
    std::iota(faces.begin(), faces.end(), 0);
    std::stable_sort(faces.begin(), faces.end(), [&](uint64_t a, uint64_t b) { return block.face_zone[a] < block.face_zone[b]; });
    return faces;
}

/**
 * @brief A whole block as a Gmsh file, its nodes in order with tags 1, 2, ...
 *        or a random permutation of 5, 7, 9, ... (sparse), its boundary faces
 *        sorted by zone, and every third cell in the cell zone "third". Gmsh
 *        4.1 splits the nodes into three blocks, and the cells into a block
 *        per run of cells with the same zone.
 */
std::string block_gmsh(const MeshBlock & block, bool v41, bool sparse) {
    const uint64_t n_nodes = block.n_nodes(), n_cells = block.n_cells();
    std::vector<uint64_t> tag(n_nodes);
    std::iota(tag.begin(), tag.end(), 1);
    if (sparse) {
        std::shuffle(tag.begin(), tag.end(), std::mt19937(11));
        for (uint64_t & t : tag) t = 2 * t + 3;
    }
    const auto faces = sorted_faces(block);
    const uint64_t n_zones = block.zone_names.size(), third = n_zones + 1;
    auto type = [](int dim, size_t n) {
        if (dim == 1) return 1;
        if (dim == 2) return n == 3 ? 2 : 3;
        return n == 4 ? 4 : n == 5 ? 7 : n == 6 ? 6 : 5;
    };
    auto nodes = [&](std::ostringstream & s, const std::vector<uint64_t> & offsets, const std::vector<uint64_t> & all, uint64_t e) {
        for (uint64_t k = offsets[e]; k < offsets[e + 1]; k++) s << " " << tag[all[k]];
        s << "\n";
    };
    std::ostringstream s;
    s << std::setprecision(17) << "$MeshFormat\n" << (v41 ? "4.1" : "2.2") << " 0 8\n$EndMeshFormat\n$PhysicalNames\n"
      << n_zones + 1 << "\n";
    for (uint64_t z = 0; z < n_zones; z++) s << N_DIM - 1 << " " << z + 1 << " \"" << block.zone_names[z] << "\"\n";
    s << N_DIM << " " << third << " \"third\"\n$EndPhysicalNames\n";
    auto coordinates = [&](uint64_t n) {
        const auto & x = block.node_coords[n];
        s << x[0] << " " << x[1] << " " << (N_DIM == 3 ? x[N_DIM - 1] : 0.0) << "\n";
    };
    if (!v41) {
        s << "$Nodes\n" << n_nodes << "\n";
        for (uint64_t n = 0; n < n_nodes; n++) {
            s << tag[n] << " ";
            coordinates(n);
        }
        s << "$EndNodes\n$Elements\n" << faces.size() + n_cells << "\n";
        uint64_t id = 1;
        for (uint64_t f : faces) {
            s << id++ << " " << type(N_DIM - 1, block.face_offsets[f + 1] - block.face_offsets[f]) << " 2 "
              << block.face_zone[f] + 1 << " 1";
            nodes(s, block.face_offsets, block.face_nodes, f);
        }
        for (uint64_t c = 0; c < n_cells; c++) {
            s << id++ << " " << type(N_DIM, block.cell_offsets[c + 1] - block.cell_offsets[c]) << " 2 "
              << (c % 3 == 0 ? third : 0) << " 1";
            nodes(s, block.cell_offsets, block.cell_nodes, c);
        }
        s << "$EndElements\n";
        return s.str();
    }
    s << "$Entities\n0 " << (N_DIM == 2 ? n_zones : 0) << " " << (N_DIM == 2 ? 2 : n_zones) << " "
      << (N_DIM == 2 ? 0 : 2) << "\n";
    for (uint64_t z = 0; z < n_zones; z++) s << z + 1 << " 0 0 0 1 1 1 1 " << z + 1 << " 0\n";
    s << "1 0 0 0 1 1 1 1 " << third << " 0\n2 0 0 0 1 1 1 0 0\n$EndEntities\n";
    const uint64_t splits[] = {0, n_nodes / 3, n_nodes / 2, n_nodes};
    s << "$Nodes\n3 " << n_nodes << " 1 " << *std::max_element(tag.begin(), tag.end()) << "\n";
    for (int b = 0; b < 3; b++) {
        s << N_DIM << " 1 0 " << splits[b + 1] - splits[b] << "\n";
        for (uint64_t n = splits[b]; n < splits[b + 1]; n++) s << tag[n] << "\n";
        for (uint64_t n = splits[b]; n < splits[b + 1]; n++) coordinates(n);
    }
    // One block per zone of faces, of cell type and of zoned or unzoned cells, in file order
    std::vector<std::array<uint64_t, 4>> runs;  // dim, entity, type, first element (index into faces or cells)
    for (uint64_t k = 0; k < faces.size(); k++) {
        const uint64_t f = faces[k];
        const std::array<uint64_t, 3> key = {uint64_t(N_DIM - 1), block.face_zone[f] + 1,
                                             uint64_t(type(N_DIM - 1, block.face_offsets[f + 1] - block.face_offsets[f]))};
        if (runs.empty() || runs.back()[0] != key[0] || runs.back()[1] != key[1] || runs.back()[2] != key[2]) {
            runs.push_back({key[0], key[1], key[2], k});
        }
    }
    const size_t n_face_runs = runs.size();
    for (uint64_t c = 0; c < n_cells; c++) {
        const std::array<uint64_t, 3> key = {uint64_t(N_DIM), c % 3 == 0 ? 1u : 2u,
                                             uint64_t(type(N_DIM, block.cell_offsets[c + 1] - block.cell_offsets[c]))};
        if (runs.size() == n_face_runs || runs.back()[1] != key[1] || runs.back()[2] != key[2]) {
            runs.push_back({key[0], key[1], key[2], c});
        }
    }
    s << "$EndNodes\n$Elements\n" << runs.size() << " " << faces.size() + n_cells << " 1 " << faces.size() + n_cells << "\n";
    uint64_t id = 1;
    for (size_t b = 0; b < runs.size(); b++) {
        const bool cells = b >= n_face_runs;
        const uint64_t end = b + 1 < runs.size() && (b + 1 < n_face_runs) == !cells ? runs[b + 1][3]
                                                                                   : (cells ? n_cells : faces.size());
        s << runs[b][0] << " " << runs[b][1] << " " << runs[b][2] << " " << end - runs[b][3] << "\n";
        for (uint64_t k = runs[b][3]; k < end; k++) {
            s << id++;
            if (cells) {
                nodes(s, block.cell_offsets, block.cell_nodes, k);
            } else {
                nodes(s, block.face_offsets, block.face_nodes, faces[k]);
            }
        }
    }
    s << "$EndElements\n";
    return s.str();
}

} // namespace

/**
 * @brief Each rank's block of a Gmsh file, which the ranks parse together, is
 *        its part of the mesh written to the file, whatever the file's format
 *        and node tags: consecutive or sparse and shuffled tags, nodes and
 *        elements split into many Gmsh 4.1 blocks, cell zones that leave some
 *        cells out, a section marker after spaces. Reading the file in chunks
 *        of 64 bytes, with lines and markers across chunks, and indexing every
 *        other line changes nothing.
 */
TEST(MeshBlockTest, EachRankGetsItsPartOfTheMeshWrittenToAGmshFile) {
#if N_DIM == 2
    const MeshBlock mesh = jittered_box(MeshType::CARTESIAN_MIXED, {7, 5}, {1.0, 0.8});
#else
    const MeshBlock mesh = jittered_box(MeshType::CARTESIAN_MIXED, {4, 3, 3}, {1.0, 0.8, 0.6});
#endif
    const std::vector<uint64_t> faces = sorted_faces(mesh);
    const int r = comm::rank(), p = comm::size();
    for (const bool v41 : {false, true}) {
        for (const bool sparse : {false, true}) {
            const std::string name = std::string("mallard_written_") + (v41 ? "41" : "22") + (sparse ? "_sparse" : "") + ".msh";
            SCOPED_TRACE(name);
            std::string text = block_gmsh(mesh, v41, sparse);
            if (v41 && sparse) text.replace(text.find("\n$Elements\n"), 11, "\n  $Elements\n");
            const std::string file = write_temp_shared(name, text);
            const MeshBlock block = read_file_block(file), chunked = read_gmsh_block(file, 64, 2);
            EXPECT_EQ(chunked.first_cell, block.first_cell);
            EXPECT_EQ(chunked.cell_offsets, block.cell_offsets);
            EXPECT_EQ(chunked.cell_nodes, block.cell_nodes);
            EXPECT_EQ(chunked.cell_zone, block.cell_zone);
            EXPECT_EQ(chunked.first_node, block.first_node);
            EXPECT_EQ(chunked.node_coords, block.node_coords);
            EXPECT_EQ(chunked.face_offsets, block.face_offsets);
            EXPECT_EQ(chunked.face_nodes, block.face_nodes);
            EXPECT_EQ(chunked.face_zone, block.face_zone);
            EXPECT_EQ(chunked.zone_names, block.zone_names);
            ASSERT_EQ(block.first_cell, block_begin(mesh.n_cells(), r, p));
            ASSERT_EQ(block.n_cells(), block_begin(mesh.n_cells(), r + 1, p) - block.first_cell);
            for (uint64_t c = 0; c < block.n_cells(); c++) {
                const uint64_t g = block.first_cell + c;
                ASSERT_EQ(std::vector<uint64_t>(block.cell_nodes.begin() + std::ptrdiff_t(block.cell_offsets[c]),
                                                block.cell_nodes.begin() + std::ptrdiff_t(block.cell_offsets[c + 1])),
                          std::vector<uint64_t>(mesh.cell_nodes.begin() + std::ptrdiff_t(mesh.cell_offsets[g]),
                                                mesh.cell_nodes.begin() + std::ptrdiff_t(mesh.cell_offsets[g + 1])))
                    << "cell " << g;
            }
            EXPECT_EQ(block.cell_zone_names, std::vector<std::string>{"third"});
            ASSERT_EQ(block.cell_zone.size(), block.n_cells());
            for (uint64_t c = 0; c < block.n_cells(); c++) {
                EXPECT_EQ(block.cell_zone[c], (block.first_cell + c) % 3 == 0 ? 0u : Mesh::NO_CELL_ZONE);
            }
            ASSERT_EQ(block.first_node, block_begin(mesh.n_nodes(), r, p));
            ASSERT_EQ(block.n_nodes(), block_begin(mesh.n_nodes(), r + 1, p) - block.first_node);
            for (uint64_t n = 0; n < block.n_nodes(); n++) {
                FOR_I_DIM EXPECT_EQ(block.node_coords[n][i],
                                    double(static_cast<rtype>(mesh.node_coords[block.first_node + n][i])));
            }
            const uint64_t first_face = block_begin(faces.size(), r, p);
            ASSERT_EQ(block.n_faces(), block_begin(faces.size(), r + 1, p) - first_face);
            for (uint64_t f = 0; f < block.n_faces(); f++) {
                const uint64_t g = faces[first_face + f];
                EXPECT_EQ(block.zone_names[block.face_zone[f]], mesh.zone_names[mesh.face_zone[g]]);
                EXPECT_EQ(std::vector<uint64_t>(block.face_nodes.begin() + std::ptrdiff_t(block.face_offsets[f]),
                                                block.face_nodes.begin() + std::ptrdiff_t(block.face_offsets[f + 1])),
                          std::vector<uint64_t>(mesh.face_nodes.begin() + std::ptrdiff_t(mesh.face_offsets[g]),
                                                mesh.face_nodes.begin() + std::ptrdiff_t(mesh.face_offsets[g + 1])));
            }
            EXPECT_TRUE(block.face_high_order_offsets.empty());
        }
    }
}

/**
 * @brief An element whose node no other rank parses, in the last line of
 *        $Elements, fails the read on every rank with the same message,
 *        naming the node, whether node tags map to positions directly
 *        (consecutive tags) or through other ranks (sparse tags).
 */
TEST(MeshBlockTest, AnUnlistedNodeFailsTheReadOnEveryRank) {
#if N_DIM == 2
    const MeshBlock mesh = jittered_box(MeshType::CARTESIAN_TRI, {4, 3}, {1.0, 0.8});
#else
    const MeshBlock mesh = jittered_box(MeshType::CARTESIAN_TET, {2, 2, 2}, {1.0, 0.8, 0.6});
#endif
    for (const bool sparse : {false, true}) {
        std::string text = block_gmsh(mesh, false, sparse);
        const size_t end = text.find("\n$EndElements"), last = text.rfind(' ', end);
        text.replace(last + 1, end - last - 1, "999999");
        const std::string file = write_temp_shared(sparse ? "mallard_unlisted_sparse.msh" : "mallard_unlisted.msh", text);
        std::string error;
        try {
            read_file_block(file);
        } catch (const std::runtime_error & e) {
            error = e.what();
        }
        EXPECT_NE(error.find("node 999999, which $Nodes does not list"), std::string::npos) << error;
        for (const std::string & other : comm::allgather(error)) EXPECT_EQ(other, error);
    }
}

/**
 * @brief Node tags anywhere in 0 ... 2^64 - 1 find their nodes, and a tag
 *        listed twice is its last node, as when the nodes were read in order.
 */
TEST(MeshBlockTest, NodeTagsFindTheirNodesAcrossTheirWholeRange) {
    constexpr uint64_t MAX = ~uint64_t(0);
    struct Case {
        std::vector<uint64_t> tags, corners, expected;
    };
#if N_DIM == 2
    const int type = 2;
    const std::vector<Case> cases = {{{MAX, 0, 7}, {0, 7, MAX}, {1, 2, 0}}, {{1, 2, 3, 1}, {1, 2, 3}, {3, 1, 2}}};
#else
    const int type = 4;
    const std::vector<Case> cases = {{{MAX, 0, 7, 9}, {0, 7, MAX, 9}, {1, 2, 0, 3}},
                                     {{1, 2, 3, 4, 1}, {1, 2, 3, 4}, {4, 1, 2, 3}}};
#endif
    for (size_t k = 0; k < cases.size(); k++) {
        std::ostringstream s;
        s << "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n" << cases[k].tags.size() << "\n";
        for (size_t n = 0; n < cases[k].tags.size(); n++) s << cases[k].tags[n] << " " << n << " " << n * n << " " << (N_DIM == 3 ? n % 2 : 0) << "\n";
        s << "$EndNodes\n$Elements\n1\n1 " << type << " 2 0 1";
        for (uint64_t t : cases[k].corners) s << " " << t;
        s << "\n$EndElements\n";
        const MeshBlock block = read_file_block(write_temp_shared("mallard_tags_" + std::to_string(k) + ".msh", s.str()));
        EXPECT_EQ(comm::allreduce(block.n_cells(), comm::Op::SUM), 1u);
        if (block.n_cells() == 1) {
            EXPECT_EQ(block.cell_nodes, cases[k].expected) << "case " << k;
        }
    }
}

/**
 * @brief Sparse node tags clustered in value (most in a narrow band, two near
 *        2^64, unevenly many per rank) split into ranges that each hold their
 *        rank's share of the tags, so no rank holds most of the tag table.
 */
TEST(MeshBlockTest, ClusteredNodeTagsSplitIntoRangesOfEqualCounts) {
    const int r = comm::rank(), p = comm::size();
    std::vector<uint64_t> tags;
    for (uint64_t k = 0; k < uint64_t(40 + 7 * r); k++) tags.push_back(uint64_t(r) + uint64_t(p) * k + 1);
    if (r == 0) tags.insert(tags.end(), {~uint64_t(0), ~uint64_t(0) - 5});
    const uint64_t n = comm::allreduce(uint64_t(tags.size()), comm::Op::SUM);
    const std::vector<uint64_t> boundaries = parallel_read::balanced_boundaries(tags);
    ASSERT_EQ(boundaries.size(), size_t(p - 1));
    std::vector<uint64_t> count(size_t(p), 0);
    for (uint64_t t : tags) count[size_t(std::lower_bound(boundaries.begin(), boundaries.end(), t) - boundaries.begin())]++;
    comm::allreduce(std::span<uint64_t>(count), comm::Op::SUM);
    for (int q = 0; q < p; q++) EXPECT_EQ(count[size_t(q)], block_begin(n, q + 1, p) - block_begin(n, q, p)) << "range " << q;
}

/**
 * @brief Cells rebuilt from faces are the same whether the faces travel as
 *        32-bit words or, with a node past 2^32, as 64-bit ones (no mesh file
 *        in the tests is that large).
 */
TEST(MeshBlockTest, CellsOfFacesAreTheSameWithWideNodeIds) {
    // Two quadrilaterals (2D) or hexahedra (3D) sharing a side, by their faces, all on rank 0
    const std::vector<std::vector<uint64_t>> cells = N_DIM == 2 ? std::vector<std::vector<uint64_t>>{{0, 1, 4, 3}, {1, 2, 5, 4}}
                                                                : std::vector<std::vector<uint64_t>>{{0, 1, 4, 3, 6, 7, 10, 9}, {1, 2, 5, 4, 7, 8, 11, 10}};
    std::map<std::vector<uint64_t>, size_t> face_of;
    std::vector<std::vector<uint64_t>> faces;
    std::vector<std::array<int64_t, 2>> face_cells;
    if (comm::rank() == 0) {
        for (size_t c = 0; c < cells.size(); c++) {
            for (const auto & local : cell_local_faces(uint32_t(cells[c].size()))) {
                std::vector<uint64_t> face;
                for (uint8_t k : local) face.push_back(cells[c][k]);
                std::vector<uint64_t> key = face;
                std::sort(key.begin(), key.end());
                const auto [it, added] = face_of.emplace(key, faces.size());
                if (added) {
                    faces.push_back(face);
                    face_cells.push_back({int64_t(c), -1});
                } else {
                    face_cells[it->second][1] = int64_t(c);
                }
            }
        }
    }
    auto read = [&](uint64_t offset) {
        parallel_read::Records records;
        std::vector<uint64_t> ids;
        for (size_t f = 0; f < faces.size(); f++) {
            for (uint64_t n : faces[f]) records.words.push_back(n + offset);
            records.end_record();
            ids.push_back(f);
        }
        MeshBlock block;
        parallel_read::cells_of_faces(cells.size(), std::move(ids), std::move(records),
                                      std::vector<std::array<int64_t, 2>>(face_cells), "test", block);
        return block;
    };
    const MeshBlock narrow = read(0), wide = read(uint64_t(1) << 32);
    EXPECT_EQ(comm::allreduce(narrow.n_cells(), comm::Op::SUM), cells.size());
    EXPECT_EQ(wide.cell_offsets, narrow.cell_offsets);
    ASSERT_EQ(wide.cell_nodes.size(), narrow.cell_nodes.size());
    for (size_t k = 0; k < narrow.cell_nodes.size(); k++) EXPECT_EQ(wide.cell_nodes[k], narrow.cell_nodes[k] + (uint64_t(1) << 32));
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
