/**
 * @file mesh_file_formats_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Each rank's block of a mesh written as a Fluent, OpenFOAM or CGNS
 *        file, which the ranks read together, against the mesh written.
 * @version 0.1
 * @date 2026-11-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#ifdef Mallard_HAS_CGNS
#include <cgnslib.h>
#endif

#include "comm.h"
#include "mesh.h"
#include "mesh_block.h"
#include "mesh_file.h"
#include "renumbered_mesh.h"
#include "scratch.h"

namespace {

std::vector<uint64_t> cell_of(const MeshBlock & m, uint64_t c) {
    return {m.cell_nodes.begin() + std::ptrdiff_t(m.cell_offsets[c]), m.cell_nodes.begin() + std::ptrdiff_t(m.cell_offsets[c + 1])};
}

std::vector<uint64_t> face_of(const MeshBlock & m, uint64_t f) {
    return {m.face_nodes.begin() + std::ptrdiff_t(m.face_offsets[f]), m.face_nodes.begin() + std::ptrdiff_t(m.face_offsets[f + 1])};
}

std::vector<uint64_t> sorted(std::vector<uint64_t> v) {
    std::sort(v.begin(), v.end());
    return v;
}

/** @brief The test mesh, and its boundary faces in order of zone, in mesh order within a zone. */
struct TestMesh {
    MeshBlock mesh;
    std::vector<uint64_t> faces;
};

TestMesh test_mesh() {
    TestMesh t;
#if N_DIM == 2
    t.mesh = jittered_box(MeshType::CARTESIAN_MIXED, {9, 7}, {1.0, 0.8});
#else
    t.mesh = jittered_box(MeshType::CARTESIAN_MIXED, {6, 5, 4}, {1.0, 0.8, 0.6});
#endif
    t.faces.resize(t.mesh.n_faces());
    std::iota(t.faces.begin(), t.faces.end(), 0);
    std::stable_sort(t.faces.begin(), t.faces.end(),
                     [&](uint64_t a, uint64_t b) { return t.mesh.face_zone[a] < t.mesh.face_zone[b]; });
    return t;
}

/**
 * @brief The faces of a mesh: interior ones (zone -1, from each cell's local
 *        faces) and then the boundary faces of each zone in TestMesh order, as
 *        the mesh lists them; each with its one or two cells.
 */
struct Faces {
    std::vector<std::vector<uint64_t>> nodes;
    std::vector<std::array<int64_t, 2>> cells;
    std::vector<int> zone;
};

Faces faces_of(const TestMesh & t) {
    const MeshBlock & m = t.mesh;
    std::map<std::vector<uint64_t>, size_t> index;
    Faces all;
    for (uint64_t c = 0; c < m.n_cells(); c++) {
        const std::vector<uint64_t> cell = cell_of(m, c);
        std::vector<std::vector<uint64_t>> local;
        if constexpr (N_DIM == 2) {
            for (size_t k = 0; k < cell.size(); k++) local.push_back({cell[k], cell[(k + 1) % cell.size()]});
        } else {
            for (const auto & lf : cell_local_faces(uint32_t(cell.size()))) {
                local.emplace_back();
                for (uint8_t k : lf) local.back().push_back(cell[k]);
            }
        }
        for (const auto & face : local) {
            const auto [it, added] = index.emplace(sorted(face), all.nodes.size());
            if (added) {
                all.nodes.push_back(face);
                all.cells.push_back({int64_t(c), -1});
            } else {
                all.cells[it->second][1] = int64_t(c);
            }
        }
    }
    Faces out;
    for (size_t f = 0; f < all.nodes.size(); f++) {
        if (all.cells[f][1] < 0) continue;
        out.nodes.push_back(all.nodes[f]);
        out.cells.push_back(all.cells[f]);
        out.zone.push_back(-1);
    }
    for (uint64_t f : t.faces) {
        const std::vector<uint64_t> face = face_of(m, f);
        out.nodes.push_back(face);
        out.cells.push_back(all.cells.at(index.at(sorted(face))));
        out.zone.push_back(int(m.face_zone[f]));
    }
    return out;
}

/**
 * @brief Every rank's block is its part of the mesh written: its cells (with
 *        their corners in any order unless exact_cells), nodes, boundary
 *        faces (TestMesh order) and zones, and cell zones.
 */
void expect_part(const TestMesh & t, const MeshBlock & block, bool exact_cells,
                 const std::vector<std::string> & cell_zone_names, const std::vector<uint32_t> & cell_zone) {
    const MeshBlock & m = t.mesh;
    const int r = comm::rank(), p = comm::size();
    ASSERT_EQ(block.first_cell, block_begin(m.n_cells(), r, p));
    ASSERT_EQ(block.n_cells(), block_begin(m.n_cells(), r + 1, p) - block.first_cell);
    for (uint64_t c = 0; c < block.n_cells(); c++) {
        const uint64_t g = block.first_cell + c;
        if (exact_cells) {
            ASSERT_EQ(cell_of(block, c), cell_of(m, g)) << "cell " << g;
        } else {
            ASSERT_EQ(sorted(cell_of(block, c)), sorted(cell_of(m, g))) << "cell " << g;
        }
    }
    EXPECT_EQ(block.cell_zone_names, cell_zone_names);
    ASSERT_EQ(block.cell_zone.size(), cell_zone.empty() ? 0 : block.n_cells());
    for (uint64_t c = 0; c < block.cell_zone.size(); c++) EXPECT_EQ(block.cell_zone[c], cell_zone[block.first_cell + c]) << c;
    ASSERT_EQ(block.first_node, block_begin(m.n_nodes(), r, p));
    ASSERT_EQ(block.n_nodes(), block_begin(m.n_nodes(), r + 1, p) - block.first_node);
    for (uint64_t n = 0; n < block.n_nodes(); n++) {
        FOR_I_DIM EXPECT_EQ(block.node_coords[n][i], double(static_cast<rtype>(m.node_coords[block.first_node + n][i])));
    }
    const uint64_t first_face = block_begin(t.faces.size(), r, p);
    ASSERT_EQ(block.n_faces(), block_begin(t.faces.size(), r + 1, p) - first_face);
    for (uint64_t f = 0; f < block.n_faces(); f++) {
        const uint64_t g = t.faces[first_face + f];
        EXPECT_EQ(block.zone_names[block.face_zone[f]], m.zone_names[m.face_zone[g]]);
        EXPECT_EQ(face_of(block, f), face_of(m, g));
    }
    EXPECT_TRUE(block.face_high_order_offsets.empty());
}

/** @brief Write files from rank 0 into a new directory of the scratch directory, visible to every rank on return. */
std::filesystem::path write_shared(const std::string & dir, const std::map<std::string, std::string> & files) {
    const std::filesystem::path path = scratch_dir() / dir;
    if (comm::rank() == 0) {
        std::filesystem::create_directories(path);
        for (const auto & [name, content] : files) std::ofstream(path / name, std::ios::binary) << content;
    }
    comm::barrier();
    return path;
}

template <typename T>
void put(std::ostringstream & s, T value) {
    s.write(reinterpret_cast<const char *>(&value), sizeof(T));
}

/** @brief The cells of zone "fluid" (first) and "solid" (second half) of Fluent and CGNS files. */
uint64_t half(const TestMesh & t) { return t.mesh.n_cells() / 2; }

/**
 * @brief The mesh as a Fluent file: the interior faces, then a face zone per
 *        boundary zone, and the cell zones "fluid" and "solid" (each half the
 *        cells); binary with single-precision nodes and faces of mixed sizes.
 */
std::string fluent_file(const TestMesh & t, bool binary) {
    const MeshBlock & m = t.mesh;
    const Faces faces = faces_of(t);
    const uint64_t n_nodes = m.n_nodes(), n_cells = m.n_cells(), n_faces = faces.nodes.size();
    std::ostringstream s;
    s << std::hex << "(0 \"written by mesh_file_formats_test\")\n(2 " << N_DIM << ")\n(10 (0 1 " << n_nodes << " 0 " << N_DIM << "))\n";
    if (binary) {
        s << "(2010 (1 1 " << n_nodes << " 1 " << N_DIM << ")(";
        for (const auto & x : m.node_coords) FOR_I_DIM put(s, float(x[i]));
        s << ")\nEnd of Binary Section   2010)\n";
    } else {
        s << "(10 (1 1 " << n_nodes << " 1 " << N_DIM << ")(\n" << std::setprecision(17);
        for (const auto & x : m.node_coords) {
            s << std::defaultfloat;
            FOR_I_DIM s << (i ? " " : "") << x[i];
            s << "\n" << std::hex;
        }
        s << "))\n";
    }
    s << "(12 (0 1 " << n_cells << " 0 0))\n(12 (5 1 " << half(t) << " 1 0))\n(12 (6 " << half(t) + 1 << " " << n_cells << " 1 0))\n";
    s << "(13 (0 1 " << n_faces << " 0 0))\n";
    for (size_t f = 0; f < n_faces;) {
        size_t end = f;
        while (end < n_faces && faces.zone[end] == faces.zone[f]) end++;
        const int zone = faces.zone[f] < 0 ? 2 : 10 + faces.zone[f];
        s << (binary ? "(2013 (" : "(13 (") << zone << " " << f + 1 << " " << end << " " << (faces.zone[f] < 0 ? 2 : 3) << " 0)(";
        if (!binary) s << "\n";
        for (; f < end; f++) {
            const int64_t c0 = faces.cells[f][0] + 1, c1 = faces.cells[f][1] + 1;
            if (binary) {
                put(s, int32_t(faces.nodes[f].size()));
                for (uint64_t n : faces.nodes[f]) put(s, int32_t(n + 1));
                put(s, int32_t(c0));
                put(s, int32_t(c1));
            } else {
                s << faces.nodes[f].size();
                for (uint64_t n : faces.nodes[f]) s << " " << n + 1;
                s << " " << c0 << " " << c1 << "\n";
            }
        }
        s << (binary ? ")\nEnd of Binary Section   2013)\n" : "))\n");
    }
    s << std::dec << "(45 (2 interior default-interior)())\n(45 (5 fluid fluid)())\n(45 (6 fluid solid)())\n";
    for (size_t z = 0; z < m.zone_names.size(); z++) s << "(45 (" << 10 + z << " wall " << m.zone_names[z] << ")())\n";
    return s.str();
}

std::string foam_header(bool binary, const std::string & cls) {
    return "/* written by mesh_file_formats_test */\nFoamFile\n{\n    version     2.0;\n    format      " +
           std::string(binary ? "binary" : "ascii") + ";\n    arch        \"LSB;label=32;scalar=64\";\n    class       " + cls +
           ";\n    object      x;\n}\n\n";
}

/** @brief A list of labels, ASCII one per line (or on one line if short, as OpenFOAM writes) or binary. */
std::string foam_labels(const std::vector<int64_t> & values, bool binary) {
    std::ostringstream s;
    s << values.size();
    if (binary) {
        s << "(";
        for (int64_t v : values) put(s, int32_t(v));
        s << ")\n";
    } else if (values.size() <= 10) {
        s << "(";
        for (size_t k = 0; k < values.size(); k++) s << (k ? " " : "") << values[k];
        s << ")\n";
    } else {
        s << "\n(\n";
        for (int64_t v : values) s << v << "\n";
        s << ")\n";
    }
    return s.str();
}

/**
 * @brief The mesh as an OpenFOAM polyMesh: a patch per boundary zone, and the
 *        cell zones "third" (every third cell) and then "early" (the first
 *        five, so cells 0 and 3 end up in "early"). 2D: extruded one cell thick
 *        in z between the empty patch frontAndBack, each cell's face at z = 0
 *        its nodes in order.
 */
std::map<std::string, std::string> foam_files(const TestMesh & t, bool binary) {
    const MeshBlock & m = t.mesh;
    Faces faces = faces_of(t);
    std::vector<std::array<double, 3>> points;
    for (const auto & x : m.node_coords) points.push_back({x[0], x[1], N_DIM == 3 ? x[N_DIM - 1] : 0.0});
    std::vector<std::string> patch_names = m.zone_names;
    std::vector<std::string> patch_types(m.zone_names.size(), "wall");
    if constexpr (N_DIM == 2) {
        const uint64_t n = m.n_nodes();
        for (const auto & x : m.node_coords) points.push_back({x[0], x[1], 0.1});
        for (auto & face : faces.nodes) face = {face[0], face[1], face[1] + n, face[0] + n};
        for (uint64_t c = 0; c < m.n_cells(); c++) {
            const std::vector<uint64_t> cell = cell_of(m, c);
            std::vector<uint64_t> front;
            for (auto it = cell.rbegin(); it != cell.rend(); ++it) front.push_back(*it + n);
            for (const auto & face : {cell, front}) {
                faces.nodes.push_back(face);
                faces.cells.push_back({int64_t(c), -1});
                faces.zone.push_back(int(m.zone_names.size()));
            }
        }
        patch_names.push_back("frontAndBack");
        patch_types.push_back("empty");
    }
    std::map<std::string, std::string> files;
    std::ostringstream pts;
    pts << foam_header(binary, "vectorField") << points.size();
    if (binary) {
        pts << "(";
        for (const auto & x : points) {
            for (double c : x) put(pts, c);
        }
        pts << ")\n";
    } else {
        pts << "\n(\n" << std::setprecision(17);
        for (const auto & x : points) pts << "(" << x[0] << " " << x[1] << " " << x[2] << ")\n";
        pts << ")\n";
    }
    files["points"] = pts.str();
    std::ostringstream fs;
    if (binary) {
        std::vector<int64_t> offsets = {0}, nodes;
        for (const auto & face : faces.nodes) {
            nodes.insert(nodes.end(), face.begin(), face.end());
            offsets.push_back(int64_t(nodes.size()));
        }
        fs << foam_header(true, "faceCompactList") << foam_labels(offsets, true) << "\n" << foam_labels(nodes, true);
    } else {
        fs << foam_header(false, "faceList") << faces.nodes.size() << "\n(\n";
        for (const auto & face : faces.nodes) {
            fs << face.size() << "(";
            for (size_t k = 0; k < face.size(); k++) fs << (k ? " " : "") << face[k];
            fs << ")\n";
        }
        fs << ")\n";
    }
    files["faces"] = fs.str();
    std::vector<int64_t> owner, neighbour;
    for (size_t f = 0; f < faces.nodes.size(); f++) {
        owner.push_back(faces.cells[f][0]);
        if (faces.zone[f] < 0) neighbour.push_back(faces.cells[f][1]);
    }
    files["owner"] = foam_header(binary, "labelList") + foam_labels(owner, binary);
    files["neighbour"] = foam_header(binary, "labelList") + foam_labels(neighbour, binary);
    std::ostringstream b;
    b << foam_header(false, "polyBoundaryMesh") << patch_names.size() << "\n(\n";
    size_t start = neighbour.size();
    for (size_t z = 0; z < patch_names.size(); z++) {
        const size_t n = size_t(std::count(faces.zone.begin(), faces.zone.end(), int(z)));
        b << "    " << patch_names[z] << "\n    {\n        type " << patch_types[z] << ";\n        inGroups List<word> 1(wall);\n"
          << "        nFaces " << n << ";\n        startFace " << start << ";\n    }\n";
        start += n;
    }
    b << ")\n";
    files["boundary"] = b.str();
    std::vector<int64_t> third, early = {0, 1, 2, 3, 4};
    for (uint64_t c = 0; c < m.n_cells(); c += 3) third.push_back(int64_t(c));
    files["cellZones"] = foam_header(binary, "regIOobject") + "2\n(\nthird\n{\n    type cellZone;\ncellLabels      List<label> " +
                         foam_labels(third, binary) + ";\n}\n\nearly\n{\n    type cellZone;\n    cellLabels List<label> " +
                         foam_labels(early, binary) + ";\n}\n)\n";
    return files;
}

} // namespace

TEST(MeshFileFormatsTest, EachRankGetsItsPartOfAFluentFile) {
    const TestMesh t = test_mesh();
    std::vector<uint32_t> cell_zone(t.mesh.n_cells());
    for (uint64_t c = 0; c < cell_zone.size(); c++) cell_zone[c] = c < half(t) ? 0 : 1;
    for (const bool binary : {false, true}) {
        SCOPED_TRACE(binary ? "binary" : "ASCII");
        TestMesh written = t;
        if (binary) {
            for (auto & x : written.mesh.node_coords) FOR_I_DIM x[i] = double(float(x[i]));
        }
        const std::string name = std::string("mallard_formats_fluent") + (binary ? "_bin" : "");
        const auto dir = write_shared(name, {{"mesh.msh", fluent_file(t, binary)}});
        const MeshBlock block = read_file_block((dir / "mesh.msh").string());
        expect_part(written, block, false, {"fluid", "solid"}, cell_zone);
    }
}

TEST(MeshFileFormatsTest, EachRankGetsItsPartOfAnOpenFoamMesh) {
    const TestMesh t = test_mesh();
    std::vector<uint32_t> cell_zone(t.mesh.n_cells(), Mesh::NO_CELL_ZONE);
    for (uint64_t c = 0; c < cell_zone.size(); c++) cell_zone[c] = c < 5 ? 1 : c % 3 == 0 ? 0 : Mesh::NO_CELL_ZONE;
    for (const bool binary : {false, true}) {
        SCOPED_TRACE(binary ? "binary" : "ASCII");
        const auto dir = write_shared(std::string("mallard_formats_foam") + (binary ? "_bin" : ""), foam_files(t, binary));
        const MeshBlock block = read_file_block(dir.string());
        expect_part(t, block, N_DIM == 2, {"third", "early"}, cell_zone);
    }
}

#ifdef Mallard_HAS_CGNS
/**
 * @brief The mesh as a CGNS file: the cells in two MIXED sections, the second
 *        a cell zone through a boundary condition in the family "solid", then a
 *        section per boundary zone.
 */
TEST(MeshFileFormatsTest, EachRankGetsItsPartOfACgnsFile) {
    const TestMesh t = test_mesh();
    const MeshBlock & m = t.mesh;
    const std::string file = (scratch_dir() / "mallard_formats.cgns").string();
    if (comm::rank() == 0) {
        int fn = 0, B = 0, Z = 0, C = 0, S = 0, F = 0, BC = 0;
        ASSERT_EQ(cg_open(file.c_str(), CG_MODE_WRITE, &fn), CG_OK);
        ASSERT_EQ(cg_base_write(fn, "base", N_DIM, N_DIM, &B), CG_OK);
        ASSERT_EQ(cg_family_write(fn, B, "solid", &F), CG_OK);
        const cgsize_t size[3] = {cgsize_t(m.n_nodes()), cgsize_t(m.n_cells()), 0};
        ASSERT_EQ(cg_zone_write(fn, B, "zone", size, CGNS_ENUMV(Unstructured), &Z), CG_OK);
        const char * names[3] = {"CoordinateX", "CoordinateY", "CoordinateZ"};
        for (size_t d = 0; d < N_DIM; d++) {
            std::vector<double> x;
            for (const auto & node : m.node_coords) x.push_back(node[d]);
            ASSERT_EQ(cg_coord_write(fn, B, Z, CGNS_ENUMV(RealDouble), names[d], x.data(), &C), CG_OK);
        }
        auto type_of = [](size_t n, int dim) {
            if (dim == 1) return CGNS_ENUMV(BAR_2);
            if (dim == 2) return n == 3 ? CGNS_ENUMV(TRI_3) : CGNS_ENUMV(QUAD_4);
            return n == 4 ? CGNS_ENUMV(TETRA_4) : n == 5 ? CGNS_ENUMV(PYRA_5) : n == 6 ? CGNS_ENUMV(PENTA_6) : CGNS_ENUMV(HEXA_8);
        };
        cgsize_t next = 1;
        auto write_mixed = [&](const std::string & name, const std::vector<std::vector<uint64_t>> & elements, int dim) {
            std::vector<cgsize_t> conn, offsets = {0};
            for (const auto & e : elements) {
                conn.push_back(cgsize_t(type_of(e.size(), dim)));
                for (uint64_t n : e) conn.push_back(cgsize_t(n + 1));
                offsets.push_back(cgsize_t(conn.size()));
            }
            const cgsize_t end = next + cgsize_t(elements.size()) - 1;
            ASSERT_EQ(cg_poly_section_write(fn, B, Z, name.c_str(), CGNS_ENUMV(MIXED), next, end, 0, conn.data(), offsets.data(), &S), CG_OK);
            next = end + 1;
        };
        std::vector<std::vector<uint64_t>> first, second;
        for (uint64_t c = 0; c < m.n_cells(); c++) (c < half(t) ? first : second).push_back(cell_of(m, c));
        write_mixed("cells", first, N_DIM);
        const cgsize_t solid[2] = {next, next + cgsize_t(second.size()) - 1};
        write_mixed("solid_cells", second, N_DIM);
        ASSERT_EQ(cg_boco_write(fn, B, Z, "solid_cells", CGNS_ENUMV(FamilySpecified), CGNS_ENUMV(PointRange), 2, solid, &BC), CG_OK);
        ASSERT_EQ(cg_boco_gridlocation_write(fn, B, Z, BC, CGNS_ENUMV(CellCenter)), CG_OK);
        ASSERT_EQ(cg_goto(fn, B, "Zone_t", Z, "ZoneBC_t", 1, "BC_t", BC, "end"), CG_OK);
        ASSERT_EQ(cg_famname_write("solid"), CG_OK);
        for (size_t z = 0; z < m.zone_names.size(); z++) {
            std::vector<std::vector<uint64_t>> faces;
            for (uint64_t f : t.faces) {
                if (m.face_zone[f] == z) faces.push_back(face_of(m, f));
            }
            write_mixed(m.zone_names[z], faces, N_DIM - 1);
        }
        cg_close(fn);
    }
    comm::barrier();
    std::vector<uint32_t> cell_zone(m.n_cells());
    for (uint64_t c = 0; c < cell_zone.size(); c++) cell_zone[c] = c < half(t) ? Mesh::NO_CELL_ZONE : 0;
    expect_part(t, read_file_block(file), true, {"solid"}, cell_zone);
}
#endif
