/**
 * @file mesh_import_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Third-party mesh formats against the Gmsh export of the same mesh
 *        (fixtures from tools/make_import_meshes.py).
 * @version 0.9
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifdef Mallard_HAS_CGNS
#include <cgnslib.h>
#endif

#include "comm.h"
#include "gmsh_fixtures.h"
#include "mesh.h"
#include "mesh_block.h"
#include "mesh_file.h"

namespace {

std::string fixture(const std::string & name) { return std::string(MALLARD_SOURCE_DIR) + "/test/data/meshes/" + name; }

using Key = std::array<double, N_DIM + 1>;  // centroid, then volume or area

/** @brief Sort by the keys rounded to 1e-6, so round-off cannot reorder them. */
void sort_keys(std::vector<Key> & keys) {
    auto rounded = [](const Key & k) {
        std::array<long long, N_DIM + 1> r;
        for (size_t i = 0; i <= N_DIM; i++) r[i] = std::llround(k[i] * 1e6);
        return r;
    };
    std::sort(keys.begin(), keys.end(), [&](const Key & a, const Key & b) { return rounded(a) < rounded(b); });
}

/** @brief Sorted (centroid, measure) of the cells, and of the faces of each zone. */
std::vector<Key> cell_keys(const Mesh & m) {
    std::vector<Key> keys;
    for (uint32_t c = 0; c < m.n_cells; c++) {
        Key k;
        FOR_I_DIM k[i] = double(m.h_cell_coords(c, i));
        k[N_DIM] = double(m.h_cell_volume(c));
        keys.push_back(k);
    }
    sort_keys(keys);
    return keys;
}

std::vector<Key> zone_keys(Mesh & m, const std::string & zone) {
    std::vector<Key> keys;
    const FaceZone * z = m.get_face_zone(zone);
    if (z == nullptr) return keys;
    for (uint32_t n = 0; n < z->n_faces(); n++) {
        const uint32_t f = z->h_faces(n);
        Key k;
        FOR_I_DIM k[i] = double(m.h_face_coords(f, i));
        k[N_DIM] = double(m.h_face_area(f));
        keys.push_back(k);
    }
    sort_keys(keys);
    return keys;
}

void expect_keys_near(const std::vector<Key> & a, const std::vector<Key> & b, double tol, const std::string & what) {
    ASSERT_EQ(a.size(), b.size()) << what;
    for (size_t n = 0; n < a.size(); n++) {
        for (size_t i = 0; i <= N_DIM; i++) ASSERT_NEAR(a[n][i], b[n][i], tol) << what << " " << n;
    }
}

/** @brief The same cells (by centroid and volume, any order), face count and boundary zones (by name and faces). */
void expect_same_mesh(const std::string & reference, const std::string & other, double tol = 1e-12) {
    Mesh a, b;
    a.init_file(fixture(reference));
    b.init_file(fixture(other));
    EXPECT_EQ(b.n_cells, a.n_cells);
    EXPECT_EQ(b.n_faces, a.n_faces);
    EXPECT_EQ(b.n_nodes, a.n_nodes);
    expect_keys_near(cell_keys(a), cell_keys(b), tol, "cells");
    std::vector<std::string> names_a, names_b;
    for (const FaceZone & z : a.h_face_zones()) names_a.push_back(z.get_name());
    for (const FaceZone & z : b.h_face_zones()) names_b.push_back(z.get_name());
    std::sort(names_a.begin(), names_a.end());
    std::sort(names_b.begin(), names_b.end());
    EXPECT_EQ(names_b, names_a);
    for (const std::string & name : names_a) expect_keys_near(zone_keys(a, name), zone_keys(b, name), tol, name);
}

const double TOL = precision_tol<double>(1e-12, 1e-5);

} // namespace

TEST(MeshImportTest, CgnsMatchesGmsh) {
    if (!have_cgns()) GTEST_SKIP() << "built without CGNS";
    if constexpr (N_DIM == 2) {
        expect_same_mesh("cyl2d.msh", "cyl2d.cgns", TOL);
    } else {
        expect_same_mesh("cyl3d.msh", "cyl3d.cgns", TOL);
        expect_same_mesh("cyl3d_tet.msh", "cyl3d_tet.cgns", TOL);
    }
}

TEST(MeshImportTest, CgnsZonesJoinedOneToOneMatchOneZone) {
    if (!have_cgns()) GTEST_SKIP() << "built without CGNS";
    expect_same_mesh(N_DIM == 2 ? "cyl2d.msh" : "cyl3d.msh", N_DIM == 2 ? "cyl2d_zones.cgns" : "cyl3d_zones.cgns", TOL);
}

TEST(MeshImportTest, CgnsHighOrderFacesMatchGmsh) {
    if (!have_cgns()) GTEST_SKIP() << "built without CGNS";
    if (N_DIM != 2) GTEST_SKIP() << "2D fixture";
    expect_same_mesh("cyl2d_o2.msh", "cyl2d_o2.cgns", TOL);
    const MeshFileData a = read_mesh_file(fixture("cyl2d_o2.msh")), b = read_mesh_file(fixture("cyl2d_o2.cgns"));
    // Every curved face's full node list (corners first, Gmsh order), any face order
    auto curved = [](const MeshFileData & d) {
        std::vector<std::vector<double>> out;
        for (const auto & ho : d.boundary_high_order) {
            if (ho.empty()) continue;
            std::vector<double> x;
            for (const auto & p : ho) x.insert(x.end(), p.begin(), p.end());
            out.push_back(x);
        }
        std::sort(out.begin(), out.end());
        return out;
    };
    const auto ca = curved(a), cb = curved(b);
    ASSERT_EQ(cb.size(), ca.size());
    ASSERT_FALSE(ca.empty());
    for (size_t f = 0; f < ca.size(); f++) {
        ASSERT_EQ(cb[f].size(), ca[f].size());
        for (size_t k = 0; k < ca[f].size(); k++) EXPECT_NEAR(cb[f][k], ca[f][k], 1e-14);
    }
}

TEST(MeshImportTest, CgnsOfTheOtherDimensionIsRejected) {
    if (!have_cgns()) GTEST_SKIP() << "built without CGNS";
    try {
        read_mesh_file(fixture(N_DIM == 2 ? "cyl3d.cgns" : "cyl2d.cgns"));
        FAIL() << "read a mesh of the other dimension";
    } catch (const std::runtime_error & e) {
        EXPECT_NE(std::string(e.what()).find("Mallard_DIM"), std::string::npos) << e.what();
    }
}

TEST(MeshImportTest, FluentMatchesGmsh) {
    // foamMeshToFluent writes 10 significant digits
    const double tol = precision_tol<double>(1e-9, 1e-5);
    const std::string gmsh = N_DIM == 2 ? "cyl2d.msh" : "cyl3d.msh";
    expect_same_mesh(gmsh, N_DIM == 2 ? "cyl2d_fluent.msh" : "cyl3d_fluent.msh", tol);
    expect_same_mesh(gmsh, N_DIM == 2 ? "cyl2d_fluent_bin.msh" : "cyl3d_fluent_bin.msh", tol);
}

TEST(MeshImportTest, OpenFoamMatchesGmsh) {
    // 2D: the Gmsh mesh extruded one cell thick between empty patches
    const std::string gmsh = N_DIM == 2 ? "cyl2d.msh" : "cyl3d.msh";
    const std::string foam = N_DIM == 2 ? "cyl2d_foam" : "cyl3d_foam";
    expect_same_mesh(gmsh, foam, TOL);
    expect_same_mesh(gmsh, foam + "_bin/constant/polyMesh", TOL);
}

TEST(MeshImportTest, PolygonsAndPolyhedraAreRejected) {
    // A pentagon (2D) or pentagonal prism (3D), by its faces
    std::vector<std::vector<uint32_t>> faces;
    if constexpr (N_DIM == 2) {
        for (uint32_t k = 0; k < 5; k++) faces.push_back({k, (k + 1) % 5});
    } else {
        faces = {{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}};
        for (uint32_t k = 0; k < 5; k++) faces.push_back({k, (k + 1) % 5, (k + 1) % 5 + 5, k + 5});
    }
    const std::vector<std::array<int64_t, 2>> face_cells(faces.size(), {0, -1});
    try {
        cells_from_faces(1, faces, face_cells, "test");
        FAIL() << "accepted a polygon or polyhedron";
    } catch (const std::runtime_error & e) {
        EXPECT_NE(std::string(e.what()).find("no general polygons or polyhedra"), std::string::npos) << e.what();
    }
}

// Malformed files found by the fuzzers (test/fuzz): each is rejected with an
// error instead of being read out of bounds, looping or exhausting memory

namespace {

void expect_rejected(const std::string & path) {
    EXPECT_THROW(Mesh().init_file(path), std::runtime_error) << path;
}

const std::string FOAM_HEADER = "FoamFile\n{\n    format ascii;\n}\n";

/** @brief An OpenFOAM polyMesh directory of the given files' data (after their header). */
std::string write_foam(const std::string & name, const std::array<std::string, 5> & data) {
    const auto dir = std::filesystem::temp_directory_path() / name / "constant" / "polyMesh";
    std::filesystem::create_directories(dir);
    const char * files[5] = {"points", "faces", "owner", "neighbour", "boundary"};
    for (size_t k = 0; k < 5; k++) {
        std::ofstream(dir / files[k]) << (data[k].rfind("FoamFile", 0) == 0 ? "" : FOAM_HEADER) << data[k];
    }
    return dir.string();
}

} // namespace

TEST(MeshImportTest, GmshCountsPastTheEndAreRejected) {
    expect_rejected(write_temp("mallard_fuzz_count.msh",
                               "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n1000000000000\n1 0 0 0\n"));
    expect_rejected(write_temp("mallard_fuzz_tags.msh", "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n1\n1 0 0 0\n"
                                                        "$EndNodes\n$Elements\n1\n1 2 -1 1 1 1\n$EndElements\n"));
}

TEST(MeshImportTest, FluentIdsOutOfRangeAreRejected) {
    expect_rejected(write_temp("mallard_fuzz_node0.cas", "(2 2)\n(10 (1 0 1 1 2)(\n0 0\n1 1\n))\n"));
    expect_rejected(write_temp("mallard_fuzz_face0.cas", "(2 2)\n(13 (3 0 0 3 2)(\n1 2 1 0\n))\n"));
    expect_rejected(write_temp("mallard_fuzz_nodes.cas", "(2 2)\n(10 (0 1 ffffffffff 0 2))\n"));
}

TEST(MeshImportTest, MalformedOpenFoamIsRejected) {
    const std::string points = "3\n(\n(0 0 0)\n(1 0 0)\n(0 1 0)\n)\n";
    const std::string boundary = "1\n(\nback\n{\n    type empty;\n    nFaces 2;\n    startFace 0;\n}\n)\n";
    // A NUL byte in the header (read as an empty token, forever)
    const std::string nul = std::string("FoamFile\n{\n    format") + '\0' + " ascii;\n}\n" + points;
    expect_rejected(write_foam("mallard_fuzz_nul", {nul, "", "", "", ""}));
    // A list longer than the file
    expect_rejected(write_foam("mallard_fuzz_list", {"1000000000000\n(\n)\n", "", "", "", ""}));
    // Faces owned by a negative cell
    expect_rejected(write_foam("mallard_fuzz_owner",
                               {points, "2\n(\n3(0 1 2)\n3(0 2 1)\n)\n", "2\n(\n0\n-5\n)\n", "0\n(\n)\n", boundary}));
}

TEST(MeshImportTest, MalformedHdf5MeshIsRejected) {
    if (!have_hdf5()) GTEST_SKIP() << "built without HDF5";
    auto block_with = [](std::vector<uint64_t> offsets, std::vector<uint64_t> nodes) {
        MeshBlock block;
        block.node_coords.assign(N_DIM + 1, {});
        block.zone_names = {"wall"};
        block.cell_offsets = std::move(offsets);
        block.cell_nodes = std::move(nodes);
        return block;
    };
    std::vector<uint64_t> cell(N_DIM + 1);
    for (size_t k = 0; k < cell.size(); k++) cell[k] = k;
    std::vector<uint64_t> missing_node = cell;
    missing_node.back() = 99;
    const std::string h5 = (std::filesystem::temp_directory_path() / "mallard_fuzz.h5").string();
    const std::vector<MeshBlock> blocks = {
        block_with({0, cell.size()}, missing_node),     // a node that is not there
        block_with({0, cell.size(), 1}, cell),          // decreasing offsets
        block_with({0, uint64_t(1) << 40}, cell),       // offsets past the cells' nodes
    };
    for (const MeshBlock & block : blocks) {
        write_mesh_h5(h5, block);
        expect_rejected(h5);
    }
}

#ifdef Mallard_HAS_CGNS
TEST(MeshImportTest, CgnsBadElementRangeIsRejected) {
    const std::string file = (std::filesystem::temp_directory_path() / "mallard_fuzz_range.cgns").string();
    int fn = 0, B = 0, Z = 0, C = 0, S = 0;
    ASSERT_EQ(cg_open(file.c_str(), CG_MODE_WRITE, &fn), CG_OK);
    ASSERT_EQ(cg_base_write(fn, "base", N_DIM, 3, &B), CG_OK);
    const cgsize_t size[3] = {N_DIM + 1, 1, 0};
    ASSERT_EQ(cg_zone_write(fn, B, "zone", size, CGNS_ENUMV(Unstructured), &Z), CG_OK);
    const double x[4] = {0.0, 1.0, 0.0, 0.0}, y[4] = {0.0, 0.0, 1.0, 0.0}, z[4] = {0.0, 0.0, 0.0, 1.0};
    ASSERT_EQ(cg_coord_write(fn, B, Z, CGNS_ENUMV(RealDouble), "CoordinateX", x, &C), CG_OK);
    ASSERT_EQ(cg_coord_write(fn, B, Z, CGNS_ENUMV(RealDouble), "CoordinateY", y, &C), CG_OK);
    ASSERT_EQ(cg_coord_write(fn, B, Z, CGNS_ENUMV(RealDouble), "CoordinateZ", z, &C), CG_OK);
    // The one cell numbered -2
    const cgsize_t conn[4] = {1, 2, 3, 4};
    const auto type = N_DIM == 2 ? CGNS_ENUMV(TRI_3) : CGNS_ENUMV(TETRA_4);
    ASSERT_EQ(cg_section_write(fn, B, Z, "cells", type, -2, -2, 0, conn, &S), CG_OK);
    cg_close(fn);
    expect_rejected(file);
}
#endif
