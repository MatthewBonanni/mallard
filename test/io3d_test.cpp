/**
 * @file io3d_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief 3D Gmsh reading, VTU output and restart files.
 * @version 0.1
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "gmsh_fixtures.h"
#include "test_fixtures.h"
#include "data.h"
#include "data_writer.h"

namespace {

// A hex [0,1]^3 with a prism (x in [1, 2], triangle extruded along y) on its
// x = 1 face, a pyramid (apex (0.5, 0.5, 2)) on its top and a tetrahedron on
// the pyramid's x-facing side. Volume 1 + 1/2 + 1/3 + 1/10. The tetrahedron is
// given inverted. A named physical curve, a point, and an untagged surface
// triangle on the pyramid-tet interface must all be ignored.
const char * MSH22 = R"($MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
5
2 1 "bottom"
2 2 "sides"
2 3 "top"
3 4 "fluid"
1 5 "edge"
$EndPhysicalNames
$Nodes
12
1 0 0 0
2 1 0 0
3 1 1 0
4 0 1 0
5 0 0 1
6 1 0 1
7 1 1 1
8 0 1 1
9 0.5 0.5 2
10 2 0 0
11 2 1 0
12 -0.5 0.5 1.2
$EndNodes
$Elements
21
1 15 2 0 1 1
2 1 2 5 1 1 2
3 3 2 1 1 1 2 3 4
4 3 2 1 1 2 10 11 3
5 3 2 2 2 1 4 8 5
6 3 2 2 2 1 2 6 5
7 3 2 2 2 4 3 7 8
8 3 2 2 2 10 11 7 6
9 2 2 2 2 2 6 10
10 2 2 2 2 3 7 11
11 2 2 3 3 5 6 9
12 2 2 3 3 6 7 9
13 2 2 3 3 7 8 9
14 2 2 3 3 5 8 12
15 2 2 3 3 5 9 12
16 2 2 3 3 8 9 12
17 2 2 0 4 5 8 9
18 5 2 4 1 1 2 3 4 5 6 7 8
19 6 2 4 1 2 6 10 3 7 11
20 7 2 4 1 5 6 7 8 9
21 4 2 4 1 5 8 9 12
$EndElements
)";

const char * MSH41 = R"($MeshFormat
4.1 0 8
$EndMeshFormat
$PhysicalNames
5
2 1 "bottom"
2 2 "sides"
2 3 "top"
3 4 "fluid"
1 5 "edge"
$EndPhysicalNames
$Entities
1 1 4 1
1 0 0 0 1 6
1 0 0 0 1 0 0 1 5 2 1 -2
1 0 0 0 2 1 0 1 1 0
2 -0.5 0 0 2 1 1 1 2 0
3 -0.5 0 1 1 1 2 1 3 0
4 0 0 1 0.5 1 2 0 0
1 -0.5 0 0 2 1 2 1 4 0
$EndEntities
$Nodes
1 12 1 12
3 1 0 12
1
2
3
4
5
6
7
8
9
10
11
12
0 0 0
1 0 0
1 1 0
0 1 0
0 0 1
1 0 1
1 1 1
0 1 1
0.5 0.5 2
2 0 0
2 1 0
-0.5 0.5 1.2
$EndNodes
$Elements
11 21 1 21
0 1 15 1
1 1
1 1 1 1
2 1 2
2 1 3 2
3 1 2 3 4
4 2 10 11 3
2 2 3 4
5 1 4 8 5
6 1 2 6 5
7 4 3 7 8
8 10 11 7 6
2 2 2 2
9 2 6 10
10 3 7 11
2 3 2 6
11 5 6 9
12 6 7 9
13 7 8 9
14 5 8 12
15 5 9 12
16 8 9 12
2 4 2 1
17 5 8 9
3 1 5 1
18 1 2 3 4 5 6 7 8
3 1 6 1
19 2 6 10 3 7 11
3 1 7 1
20 5 6 7 8 9
3 1 4 1
21 5 8 9 12
$EndElements
)";

void check_mixed_mesh(const std::string & file) {
    Mesh mesh;
    mesh.init_file(file);
    ASSERT_EQ(mesh.n_nodes, 12u);
    ASSERT_EQ(mesh.n_cells, 4u);
    std::map<uint32_t, int> cells_with_n_nodes;
    for (uint32_t c = 0; c < mesh.n_cells; c++) cells_with_n_nodes[mesh.h_n_nodes_of_cell(c)]++;
    EXPECT_EQ(cells_with_n_nodes, (std::map<uint32_t, int>{{4, 1}, {5, 1}, {6, 1}, {8, 1}}));
    EXPECT_EQ(mesh.n_faces, 17u);

    const std::pair<const char *, uint32_t> zones[] = {{"interior", 3}, {"bottom", 2}, {"sides", 6}, {"top", 6}};
    for (const auto & [name, n] : zones) {
        FaceZone * zone = mesh.get_face_zone(name);
        ASSERT_NE(zone, nullptr) << name;
        EXPECT_EQ(zone->n_faces(), n) << name;
    }
    EXPECT_EQ(mesh.n_face_zones(), 4u);

    rtype total = 0.0;
    for (uint32_t c = 0; c < mesh.n_cells; c++) {
        EXPECT_GT(mesh.h_cell_volume(c), 0.0);
        total += mesh.h_cell_volume(c);
        rtype closure[N_DIM] = {};
        for (uint32_t k = 0; k < mesh.h_n_faces_of_cell(c); k++) {
            const uint32_t f = mesh.h_face_of_cell(c, k);
            const rtype sign = (mesh.h_cells_of_face(f, 0) == static_cast<int32_t>(c)) ? 1.0 : -1.0;
            FOR_I_DIM closure[i] += sign * mesh.h_face_normals(f, i);
        }
        FOR_I_DIM EXPECT_NEAR(closure[i], 0.0, 1e-14) << "cell " << c;
    }
    EXPECT_NEAR(total, 1.0 + 0.5 + 1.0 / 3.0 + 0.1, roundoff(1e-14));
}

/**
 * @brief Raw appended arrays of a VTU file, by name ("Points" for the points).
 */
std::map<std::string, std::string> read_appended_arrays(const std::string & file) {
    std::ifstream in(file, std::ios::binary);
    const std::string raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const size_t base = raw.find('_', raw.find("<AppendedData")) + 1;
    std::map<std::string, std::string> arrays;
    size_t pos = 0;
    while ((pos = raw.find("<DataArray", pos)) < base) {
        const size_t end = raw.find('>', pos);
        const std::string tag = raw.substr(pos, end - pos);
        pos = end;
        auto attr = [&](const std::string & key) {
            size_t a = tag.find(" " + key + "=\"");
            if (a == std::string::npos) return std::string();
            a += key.size() + 3;
            return tag.substr(a, tag.find('"', a) - a);
        };
        if (attr("offset").empty()) continue;
        const size_t offset = base + std::stoull(attr("offset"));
        uint64_t n_bytes;
        std::memcpy(&n_bytes, raw.data() + offset, sizeof(n_bytes));
        const std::string name = attr("Name").empty() ? "Points" : attr("Name");
        arrays[name] = raw.substr(offset + sizeof(n_bytes), n_bytes);
        arrays[name + "@components"] = attr("NumberOfComponents");
    }
    return arrays;
}

template <typename T>
std::vector<T> as_vector(const std::string & bytes) {
    std::vector<T> v(bytes.size() / sizeof(T));
    std::memcpy(v.data(), bytes.data(), v.size() * sizeof(T));
    return v;
}

/**
 * @brief Cell data for a DataWriter: column v of the host view is named names[v].
 */
struct CellFields {
    Kokkos::View<rtype **>::host_mirror_type values;
    std::vector<Data> data;
    CellFields(uint32_t n_cells, const std::vector<std::string> & names)
        : values(Kokkos::create_mirror_view(Kokkos::View<rtype **>("values", n_cells, names.size()))) {
        for (size_t v = 0; v < names.size(); v++) {
            data.push_back(Data(names[v], Kokkos::subview(values, Kokkos::ALL(), v)));
            for (uint32_t c = 0; c < n_cells; c++) values(c, v) = 100.0 * v + c;
        }
    }
};

std::shared_ptr<Mesh> mixed_mesh() {
    auto mesh = std::make_shared<Mesh>();
    mesh->init_file(write_temp("mallard_mixed3d_22.msh", MSH22));
    return mesh;
}

} // namespace

TEST(IO3DTest, ReadsGmsh22MixedMesh) {
    check_mixed_mesh(write_temp("mallard_mixed3d_22.msh", MSH22));
}

TEST(IO3DTest, ReadsGmsh41MixedMesh) {
    check_mixed_mesh(write_temp("mallard_mixed3d_41.msh", MSH41));
}

TEST(IO3DTest, HigherOrderElementsAreRejected) {
    // A 10-node tetrahedron (type 11) in place of the linear one
    std::string msh = MSH22;
    msh.replace(msh.find("21 4 2 4 1 5 8 9 12"), 19, "21 11 2 4 1 5 8 9 12 1 2 3 4 6 7 10");
    try {
        Mesh().init_file(write_temp("mallard_tet10.msh", msh));
        ADD_FAILURE() << "a 10-node tetrahedron was accepted";
    } catch (const std::runtime_error & e) {
        EXPECT_NE(std::string(e.what()).find("element type 11"), std::string::npos) << e.what();
    }
}

TEST(IO3DTest, VolumeOutputFollowsVTKCellConventions) {
    const std::string dir = (std::filesystem::temp_directory_path() / "mallard_vtu3d").string();
    std::filesystem::remove_all(dir);
    auto mesh = mixed_mesh();
    CellFields fields(mesh->n_cells, {"RHO", "U_X", "U_Y", "U_Z"});
    DataWriter writer;
    writer.init(parse_toml("prefix = \"" + dir + "/flow\"\nformat = \"vtu\"\ninterval = 1\n"
                           "variables = [\"RHO\", \"U\"]\n"), fields.data, mesh);
    writer.write(0, 0.0);

    auto arrays = read_appended_arrays(dir + "/flow_000000.vtu");
    const auto points = as_vector<rtype>(arrays["Points"]);
    const auto connectivity = as_vector<int64_t>(arrays["connectivity"]);
    const auto offsets = as_vector<int64_t>(arrays["offsets"]);
    const auto types = as_vector<uint8_t>(arrays["types"]);
    ASSERT_EQ(points.size(), 3u * mesh->n_nodes);
    ASSERT_EQ(types.size(), mesh->n_cells);
    ASSERT_EQ(offsets.size(), mesh->n_cells);
    ASSERT_EQ(connectivity.size(), 4u + 5u + 6u + 8u);
    EXPECT_EQ(offsets.back(), static_cast<int64_t>(connectivity.size()));

    // VTK: the base face's right-hand normal points toward the opposite
    // node(s) for tetra (10), hexahedron (12) and pyramid (14), and away from
    // them for the wedge (13)
    struct Convention { uint32_t n_nodes; uint8_t type; std::vector<int> base, opposite; double sign; };
    const std::vector<Convention> conventions = {
        {4, 10, {0, 1, 2}, {3}, 1.0},
        {5, 14, {0, 1, 2, 3}, {4}, 1.0},
        {6, 13, {0, 1, 2}, {3, 4, 5}, -1.0},
        {8, 12, {0, 1, 2, 3}, {4, 5, 6, 7}, 1.0},
    };
    std::map<uint8_t, int> n_of_type;
    for (uint32_t c = 0; c < mesh->n_cells; c++) {
        const int64_t start = c == 0 ? 0 : offsets[c - 1];
        const uint32_t n = offsets[c] - start;
        n_of_type[types[c]]++;
        auto x = [&](int k, int d) { return static_cast<double>(points[3 * connectivity[start + k] + d]); };
        for (const auto & conv : conventions) {
            if (conv.n_nodes != n) continue;
            EXPECT_EQ(types[c], conv.type);
            const auto & b = conv.base;
            const bool tri = b.size() == 3;
            // Two edges of a triangle, or the diagonals of a quad, span its normal
            double e1[3], e2[3], normal[3], to_opposite[3] = {};
            for (int d = 0; d < 3; d++) {
                e1[d] = x(b[tri ? 1 : 2], d) - x(b[0], d);
                e2[d] = x(b[tri ? 2 : 3], d) - x(b[tri ? 0 : 1], d);
                double base_center = 0.0, opposite_center = 0.0;
                for (int k : b) base_center += x(k, d) / b.size();
                for (int k : conv.opposite) opposite_center += x(k, d) / conv.opposite.size();
                to_opposite[d] = opposite_center - base_center;
            }
            for (int d = 0; d < 3; d++) normal[d] = e1[(d + 1) % 3] * e2[(d + 2) % 3] - e1[(d + 2) % 3] * e2[(d + 1) % 3];
            const double dot = normal[0] * to_opposite[0] + normal[1] * to_opposite[1] + normal[2] * to_opposite[2];
            EXPECT_GT(conv.sign * dot, 0.0) << "VTK type " << int(conv.type);
        }
    }
    EXPECT_EQ(n_of_type, (std::map<uint8_t, int>{{10, 1}, {12, 1}, {13, 1}, {14, 1}}));

    // U is one 3-component array, interleaved per cell
    EXPECT_EQ(arrays["U@components"], "3");
    const auto u = as_vector<rtype>(arrays["U"]);
    ASSERT_EQ(u.size(), 3u * mesh->n_cells);
    for (uint32_t c = 0; c < mesh->n_cells; c++) {
        for (uint32_t k = 0; k < 3; k++) EXPECT_EQ(u[3 * c + k], fields.values(c, 1 + k));
    }
    std::filesystem::remove_all(dir);
}

TEST(IO3DTest, BoundaryZoneOutputWritesTrianglesAndQuads) {
    const std::string dir = (std::filesystem::temp_directory_path() / "mallard_vtu3d_faces").string();
    std::filesystem::remove_all(dir);
    auto mesh = mixed_mesh();
    CellFields fields(mesh->n_cells, {"RHO"});
    DataWriter writer;
    writer.init(parse_toml("prefix = \"" + dir + "/sides\"\nformat = \"vtu\"\ninterval = 1\n"
                           "geometry = \"sides\"\nvariables = [\"RHO\"]\n"), fields.data, mesh);
    writer.write(0, 0.0);

    std::ifstream in(dir + "/sides_000000.vtu");
    std::stringstream buffer;
    buffer << in.rdbuf();
    const std::string text = buffer.str();
    auto array = [&](const std::string & name) {
        const size_t start = text.find(">", text.find("Name=\"" + name + "\"")) + 1;
        std::istringstream values(text.substr(start, text.find("</DataArray>", start) - start));
        return std::vector<double>(std::istream_iterator<double>(values), std::istream_iterator<double>());
    };
    // 4 quads and 2 triangles on the hex and prism sides, 4 * 4 + 2 * 3 nodes
    std::map<int, int> n_of_type;
    for (double t : array("types")) n_of_type[static_cast<int>(t)]++;
    EXPECT_EQ(n_of_type, (std::map<int, int>{{5, 2}, {9, 4}}));
    EXPECT_EQ(array("offsets").back(), 22.0);
    EXPECT_NE(text.find("NumberOfPoints=\"10\""), std::string::npos);
    // Points keep their z coordinate (the prism top edge 6-7 is at z = 1)
    const size_t start = text.find(">", text.find("<Points>") + 9) + 1;
    std::istringstream coords(text.substr(start, text.find("</DataArray>", start) - start));
    double x, y, z, z_max = 0.0;
    while (coords >> x >> y >> z) z_max = std::max(z_max, z);
    EXPECT_EQ(z_max, 1.0);
    std::filesystem::remove_all(dir);
}

TEST(IO3DTest, RestartRoundTripCarriesAllConservatives) {
    const std::string dir = (std::filesystem::temp_directory_path() / "mallard_restart3d").string();
    std::filesystem::remove_all(dir);
    auto mesh = mixed_mesh();
    std::vector<std::string> names(CONSERVATIVE_NAMES.begin(), CONSERVATIVE_NAMES.end());
    CellFields fields(mesh->n_cells, names);
    DataWriter writer;
    writer.init(parse_toml("prefix = \"" + dir + "/r\"\nformat = \"restart\"\ninterval = 7\n"), fields.data, mesh);
    writer.write(7, 0.25);

    RestartData restart = read_restart(dir + "/r_000007.restart");
    EXPECT_EQ(restart.step, 7u);
    EXPECT_EQ(restart.t, 0.25);
    ASSERT_EQ(restart.n_cells, mesh->n_cells);
    EXPECT_EQ(restart.names, names);
    ASSERT_EQ(restart.fields.size(), 5u);
    for (uint32_t v = 0; v < 5; v++) {
        for (uint32_t c = 0; c < mesh->n_cells; c++) EXPECT_EQ(restart.fields[v][c], fields.values(c, v));
    }

    // A file holding a 2D run's variables is refused
    const std::vector<std::string> names_2d = {"RHO", "RHOU_X", "RHOU_Y", "RHOE"};
    CellFields fields_2d(mesh->n_cells, names_2d);
    DataWriter writer_2d;
    writer_2d.init(parse_toml("prefix = \"" + dir + "/r2d\"\nformat = \"restart\"\ninterval = 7\n"), fields_2d.data,
                   mesh, names_2d);
    writer_2d.write(7, 0.25);
    try {
        read_restart(dir + "/r2d_000007.restart");
        ADD_FAILURE() << "a 2D restart file was accepted";
    } catch (const std::runtime_error & e) {
        EXPECT_NE(std::string(e.what()).find("Mallard_DIM"), std::string::npos) << e.what();
    }
    std::filesystem::remove_all(dir);
}
