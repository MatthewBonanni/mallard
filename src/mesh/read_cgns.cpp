/**
 * @file read_cgns.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Unstructured CGNS mesh files, each rank reading its own part of the
 *        nodes and elements.
 * @version 0.9
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "mesh_file.h"

#include <stdexcept>

#ifdef Mallard_HAS_CGNS

#include <algorithm>
#include <bit>
#include <filesystem>
#include <map>
#include <memory>
#include <set>

#include <cgnslib.h>

#include "comm.h"
#include "read_parallel.h"

namespace {

using parallel_read::collectively;
using parallel_read::part_of;
using parallel_read::Records;

constexpr int BOUNDARY_DIM = N_DIM - 1;
constexpr size_t FAMILY_NAME_SIZE = 4096;  // a family path, up to CG_MAX_GOTO_DEPTH names

struct ElementInfo {
    int dim;          // -1: unsupported
    int n_corners;
    bool curved_face;  // a high-order face in Gmsh node order, which curves its face
};

ElementInfo element_info(CGNS_ENUMT(ElementType_t) type) {
    switch (type) {
        case CGNS_ENUMV(NODE): return {0, 1, false};
        case CGNS_ENUMV(BAR_2): return {1, 2, false};
        case CGNS_ENUMV(BAR_3): case CGNS_ENUMV(BAR_4): return {1, 2, true};
        case CGNS_ENUMV(TRI_3): return {2, 3, false};
        case CGNS_ENUMV(TRI_6): case CGNS_ENUMV(TRI_9): case CGNS_ENUMV(TRI_10): return {2, 3, true};
        case CGNS_ENUMV(QUAD_4): return {2, 4, false};
        case CGNS_ENUMV(QUAD_8): case CGNS_ENUMV(QUAD_9): case CGNS_ENUMV(QUAD_16): return {2, 4, true};
        case CGNS_ENUMV(TETRA_4): case CGNS_ENUMV(TETRA_10): case CGNS_ENUMV(TETRA_16): case CGNS_ENUMV(TETRA_20): case CGNS_ENUMV(TETRA_22):
        case CGNS_ENUMV(TETRA_34): case CGNS_ENUMV(TETRA_35): return {3, 4, false};
        case CGNS_ENUMV(PYRA_5): case CGNS_ENUMV(PYRA_13): case CGNS_ENUMV(PYRA_14): case CGNS_ENUMV(PYRA_21): case CGNS_ENUMV(PYRA_29): case CGNS_ENUMV(PYRA_30):
        case CGNS_ENUMV(PYRA_P4_29): case CGNS_ENUMV(PYRA_50): case CGNS_ENUMV(PYRA_55): return {3, 5, false};
        case CGNS_ENUMV(PENTA_6): case CGNS_ENUMV(PENTA_15): case CGNS_ENUMV(PENTA_18): case CGNS_ENUMV(PENTA_24): case CGNS_ENUMV(PENTA_38):
        case CGNS_ENUMV(PENTA_40): case CGNS_ENUMV(PENTA_33): case CGNS_ENUMV(PENTA_66): case CGNS_ENUMV(PENTA_75): return {3, 6, false};
        case CGNS_ENUMV(HEXA_8): case CGNS_ENUMV(HEXA_20): case CGNS_ENUMV(HEXA_27): case CGNS_ENUMV(HEXA_32): case CGNS_ENUMV(HEXA_56): case CGNS_ENUMV(HEXA_64):
        case CGNS_ENUMV(HEXA_44): case CGNS_ENUMV(HEXA_98): case CGNS_ENUMV(HEXA_125): return {3, 8, false};
        default: return {-1, 0, false};
    }
}

class CgnsFile {
  public:
    explicit CgnsFile(const std::string & filename) : name(filename) {
        if (cg_open(filename.c_str(), CG_MODE_READ, &fn) != CG_OK) {
            throw std::runtime_error("Could not open CGNS file " + filename + ": " + cg_get_error());
        }
    }
    ~CgnsFile() { cg_close(fn); }
    CgnsFile(const CgnsFile &) = delete;
    CgnsFile & operator=(const CgnsFile &) = delete;

    void check(int status, const std::string & what) const {
        if (status != CG_OK) throw std::runtime_error("CGNS file " + name + ": " + what + ": " + cg_get_error());
    }
    [[noreturn]] void fail(const std::string & what) const {
        throw std::runtime_error("CGNS file " + name + ": " + what);
    }

    int fn = 0;
    std::string name;
};

/** @brief An element of a zone, by element id: a cell, a boundary face or neither. */
struct ElementRef {
    enum Kind : uint8_t { NONE, CELL, FACE } kind = NONE;
    uint32_t index = 0;
};

/** @brief Name of the zone a BC or section belongs to: its family's FamilyName (as Gmsh writes physical names), or the family name. */
std::string family_zone_name(const CgnsFile & file, int base, const std::string & family) {
    int n_families = 0;
    file.check(cg_nfamilies(file.fn, base, &n_families), "reading families");
    for (int f = 1; f <= n_families; f++) {
        char name[33] = {};
        int n_bocos = 0, n_geos = 0;
        file.check(cg_family_read(file.fn, base, f, name, &n_bocos, &n_geos), "reading family");
        if (family != name) continue;
        int n_names = 0;
        if (cg_nfamily_names(file.fn, base, f, &n_names) == CG_OK && n_names > 0) {
            char node[33] = {}, value[FAMILY_NAME_SIZE] = {};
            file.check(cg_family_name_read(file.fn, base, f, 1, node, value), "reading family name");
            return value[0] ? value : node;
        }
        return family;
    }
    return family;
}

bool read_famname(std::string & out) {
    char name[FAMILY_NAME_SIZE] = {};
    if (cg_famname_read(name) != CG_OK) return false;
    out = name;
    return !out.empty();
}

/** @brief A section of a zone, and this rank's part [from, to] of its elements (by id; from > to: none). */
struct SectionInfo {
    std::string name, family;  // family: the zone name of the section's family, or empty
    CGNS_ENUMT(ElementType_t) type;
    cgsize_t start = 0, end = 0, from = 1, to = 0;
    std::vector<ElementRef> refs;  // this rank's elements, from `from` on
};

struct ZoneInfo {
    std::string name;
    cgsize_t n_nodes = 0;
    uint64_t first_node = 0;  // in the file-wide numbering of every zone's nodes
    cgsize_t max_id = 0;      // the greatest element id
    std::vector<SectionInfo> sections;

    /** @brief This rank's element of that id, NONE kind if it is another rank's or none at all; the last section listing it decides. */
    ElementRef element(cgsize_t id) const {
        for (size_t s = sections.size(); s-- > 0;) {
            const SectionInfo & x = sections[s];
            if (id < x.start || id > x.end) continue;
            if (id < x.from || id > x.to) return {};
            return x.refs[size_t(id - x.from)];
        }
        return {};
    }
};

/** @brief Union-find over the file-wide node numbering, joining nodes that zone connectivities identify; roots are the least nodes. */
class NodeClasses {
  public:
    uint64_t root(uint64_t n) {
        auto it = parent.find(n);
        if (it == parent.end()) return n;
        const uint64_t r = root(it->second);
        it->second = r;
        return r;
    }
    void join(uint64_t a, uint64_t b) {
        const uint64_t ra = root(a), rb = root(b);
        if (ra != rb) parent[std::max(ra, rb)] = std::min(ra, rb);
    }

  private:
    std::map<uint64_t, uint64_t> parent;
};

} // namespace

bool have_cgns() { return true; }

int cgns_dimension(const std::string & filename) {
    CgnsFile file(filename);
    int n_bases = 0;
    file.check(cg_nbases(file.fn, &n_bases), "reading bases");
    if (n_bases < 1) file.fail("no CGNSBase_t node.");
    char base_name[33] = {};
    int cell_dim = 0, phys_dim = 0;
    file.check(cg_base_read(file.fn, 1, base_name, &cell_dim, &phys_dim), "reading the base");
    return cell_dim;
}

MeshBlock read_cgns_block(const std::string & filename) {
    const int p = comm::size(), r = comm::rank();
    std::unique_ptr<CgnsFile> opened;
    collectively([&] { opened = std::make_unique<CgnsFile>(filename); });
    const CgnsFile & file = *opened;
    const int fn = file.fn;
    // Every node, element or point a count announces takes at least a byte of the file
    const cgsize_t file_size = cgsize_t(std::filesystem::file_size(filename));
    auto check_count = [&](cgsize_t n, const std::string & what) {
        if (n < 0 || n > file_size) file.fail("bad " + what + " count " + std::to_string(n) + ".");
    };
    const int B = 1;
    std::vector<ZoneInfo> zones;
    uint64_t n_nodes = 0, n_elements = 0;
    collectively([&] {
        int n_bases = 0;
        file.check(cg_nbases(fn, &n_bases), "reading bases");
        if (n_bases < 1) file.fail("no CGNSBase_t node.");
        if (n_bases > 1) file.fail("has " + std::to_string(n_bases) + " bases; Mallard reads files with one.");
        char base_name[33] = {};
        int cell_dim = 0, phys_dim = 0;
        file.check(cg_base_read(fn, B, base_name, &cell_dim, &phys_dim), "reading the base");
        if (cell_dim != N_DIM) {
            file.fail("has " + std::to_string(cell_dim) + "D cells, but Mallard was built with Mallard_DIM = " +
                      std::to_string(N_DIM) + ".");
        }
        int n_zones = 0;
        file.check(cg_nzones(fn, B, &n_zones), "reading zones");
        if (n_zones < 1) file.fail("no zones.");
        zones.resize(size_t(n_zones));
        for (int z = 1; z <= n_zones; z++) {
            ZoneInfo & zone = zones[size_t(z - 1)];
            CGNS_ENUMT(ZoneType_t) type;
            file.check(cg_zone_type(fn, B, z, &type), "reading zone type");
            char zone_name[33] = {};
            cgsize_t size[9] = {};
            file.check(cg_zone_read(fn, B, z, zone_name, size), "reading zone");
            zone.name = zone_name;
            if (type != CGNS_ENUMV(Unstructured)) {
                file.fail("zone " + zone.name + " is structured; Mallard reads unstructured zones.");
            }
            zone.n_nodes = size[0];
            check_count(zone.n_nodes, "node");
            zone.first_node = n_nodes;
            n_nodes += uint64_t(zone.n_nodes);
            int n_sections = 0;
            file.check(cg_nsections(fn, B, z, &n_sections), "reading sections");
            for (int s = 1; s <= n_sections; s++) {
                SectionInfo x;
                char section_name[33] = {};
                int n_boundary = 0, parent_flag = 0;
                file.check(cg_section_read(fn, B, z, s, section_name, &x.type, &x.start, &x.end, &n_boundary, &parent_flag),
                           "reading section");
                x.name = section_name;
                if (x.type == CGNS_ENUMV(NGON_n) || x.type == CGNS_ENUMV(NFACE_n)) {
                    file.fail("section " + x.name +
                              " holds polyhedra (NGON_n/NFACE_n); Mallard reads element-based sections (triangles, "
                              "quadrilaterals, tetrahedra, pyramids, prisms, hexahedra, and MIXED).");
                }
                if (x.start < 1 || x.end < x.start || x.end > file_size) {
                    file.fail("section " + x.name + " has bad element range " + std::to_string(x.start) + " to " +
                              std::to_string(x.end) + ".");
                }
                std::string family;
                if (cg_goto(fn, B, "Zone_t", z, "Elements_t", s, "end") == CG_OK && read_famname(family)) {
                    x.name = x.family = family_zone_name(file, B, family);
                }
                n_elements += uint64_t(x.end - x.start + 1);
                zone.max_id = std::max(zone.max_id, x.end);
                zone.sections.push_back(std::move(x));
            }
        }
    });

    // This rank's share of the nodes, by their file-wide numbers
    const uint64_t first_node = block_begin(n_nodes, r, p), end_node = block_begin(n_nodes, r + 1, p);
    std::vector<std::array<double, N_DIM>> xyz(end_node - first_node);
    collectively([&] {
        const char * names[3] = {"CoordinateX", "CoordinateY", "CoordinateZ"};
        std::vector<double> c;
        for (size_t z = 0; z < zones.size(); z++) {
            const ZoneInfo & zone = zones[z];
            const uint64_t from = std::max(first_node, zone.first_node);
            const uint64_t to = std::min(end_node, zone.first_node + uint64_t(zone.n_nodes));
            if (from >= to) continue;
            const cgsize_t lo = cgsize_t(from - zone.first_node + 1), hi = cgsize_t(to - zone.first_node);
            c.resize(to - from);
            for (size_t d = 0; d < N_DIM; d++) {
                file.check(cg_coord_read(fn, B, int(z + 1), names[d], CGNS_ENUMV(RealDouble), &lo, &hi, c.data()),
                           std::string("reading ") + names[d] + " of zone " + zone.name);
                for (uint64_t n = from; n < to; n++) xyz[n - first_node][d] = c[n - from];
            }
        }
    });

    // This rank's share of the elements, in file order: cells, and boundary faces. General reads, since
    // cg_elements_partial_read reads and keeps a whole section stored as integers of another size than cgsize_t
    const CGNS_ENUMT(DataType_t) SIZE_TYPE = sizeof(cgsize_t) == 8 ? CGNS_ENUMV(LongInteger) : CGNS_ENUMV(Integer);
    const uint64_t first_element = block_begin(n_elements, r, p), end_element = block_begin(n_elements, r + 1, p);
    Records cells, faces, face_all;  // corners (file-wide nodes); every node of curved faces
    std::vector<uint32_t> face_zone_of;  // the CGNS zone of each
    std::vector<std::array<uint32_t, 2>> cell_section, face_section;
    int curved_faces = 0;
    collectively([&] {
        uint64_t seq = 0;
        std::vector<cgsize_t> conn, offsets;
        for (size_t z = 0; z < zones.size(); z++) {
            ZoneInfo & zone = zones[z];
            for (size_t s = 0; s < zone.sections.size(); s++) {
                SectionInfo & x = zone.sections[s];
                const uint64_t n = uint64_t(x.end - x.start + 1);
                const uint64_t from = std::max(first_element, seq), to = std::min(end_element, seq + n);
                if (from < to) {
                    x.from = x.start + cgsize_t(from - seq);
                    x.to = x.start + cgsize_t(to - seq) - 1;
                }
                seq += n;
                if (from >= to) continue;
                const int Z = int(z + 1), S = int(s + 1);
                if (x.type == CGNS_ENUMV(MIXED)) {
                    cgsize_t size = 0;
                    file.check(cg_ElementPartialSize(fn, B, Z, S, x.from, x.to, &size), "reading section size");
                    check_count(size, "element data");
                    conn.resize(size_t(size));
                    offsets.resize(size_t(x.to - x.from + 2));
                    file.check(cg_poly_elements_general_read(fn, B, Z, S, x.from, x.to, SIZE_TYPE, conn.data(), offsets.data()),
                               "reading elements");
                } else {
                    int npe = 0;
                    if (cg_npe(x.type, &npe) != CG_OK || npe <= 0) npe = 0;
                    conn.resize(size_t(x.to - x.from + 1) * size_t(npe));
                    file.check(cg_elements_general_read(fn, B, Z, S, x.from, x.to, SIZE_TYPE, conn.data()), "reading elements");
                }
                x.refs.resize(size_t(x.to - x.from + 1));
                size_t pos = 0;
                for (cgsize_t e = x.from; e <= x.to; e++) {
                    CGNS_ENUMT(ElementType_t) type_e = x.type;
                    if (x.type == CGNS_ENUMV(MIXED)) {
                        const cgsize_t offset = offsets[size_t(e - x.from)] - offsets[0];
                        if (offset < 0 || size_t(offset) >= conn.size()) file.fail("bad offsets in section " + x.name + ".");
                        pos = size_t(offset);
                        type_e = static_cast<CGNS_ENUMT(ElementType_t)>(conn[pos++]);
                    }
                    int npe = 0;
                    const ElementInfo info = element_info(type_e);
                    if (info.dim < 0 || cg_npe(type_e, &npe) != CG_OK || npe <= 0) {
                        file.fail("unsupported element type " + std::string(cg_ElementTypeName(type_e)) + " in section " +
                                  x.name + "; nodes, bars, triangles, quadrilaterals, tetrahedra, pyramids, prisms "
                                  "and hexahedra are supported (high-order faces up to BAR_4, TRI_10 and QUAD_16).");
                    }
                    if (info.dim > N_DIM) {
                        file.fail("has " + std::to_string(info.dim) + "D elements, but Mallard was built with Mallard_DIM = " +
                                  std::to_string(N_DIM) + ".");
                    }
                    if (pos + size_t(npe) > conn.size()) file.fail("section " + x.name + " ends early.");
                    const size_t first = pos;
                    for (int k = 0; k < npe; k++) {
                        const cgsize_t local = conn[pos++];
                        if (local < 1 || local > zone.n_nodes) file.fail("node id out of range in section " + x.name + ".");
                    }
                    auto node = [&](size_t k) { return zone.first_node + uint64_t(conn[first + k] - 1); };
                    ElementRef & ref = x.refs[size_t(e - x.from)];
                    if (info.dim == N_DIM) {
                        ref = {ElementRef::CELL, uint32_t(cells.size())};
                        for (int k = 0; k < info.n_corners; k++) cells.words.push_back(node(size_t(k)));
                        cells.end_record();
                        cell_section.push_back({uint32_t(z), uint32_t(s)});
                    } else if (info.dim == BOUNDARY_DIM) {
                        ref = {ElementRef::FACE, uint32_t(faces.size())};
                        for (int k = 0; k < info.n_corners; k++) faces.words.push_back(node(size_t(k)));
                        faces.end_record();
                        if (info.curved_face) {
                            curved_faces = 1;
                            for (int k = 0; k < npe; k++) face_all.words.push_back(node(size_t(k)));
                        }
                        face_all.end_record();
                        face_zone_of.push_back(uint32_t(z));
                        face_section.push_back({uint32_t(z), uint32_t(s)});
                    }
                }
            }
        }
    });
    const std::vector<uint64_t> counts = comm::allgatherv(std::vector<uint64_t>{cells.size(), faces.size()});
    uint64_t n_cells = 0, n_faces = 0, my_first_cell = 0, my_first_face = 0;
    for (int q = 0; q < p; q++) {
        if (q == r) {
            my_first_cell = n_cells;
            my_first_face = n_faces;
        }
        n_cells += counts[2 * size_t(q)];
        n_faces += counts[2 * size_t(q) + 1];
    }
    if (n_cells == 0) file.fail("contains no " + std::to_string(N_DIM) + "D elements.");
    curved_faces = comm::allreduce(curved_faces, comm::Op::MAX);

    // Zones joined by 1-to-1 connectivities share their interface nodes
    NodeClasses classes;
    collectively([&] {
        for (int z = 1; z <= int(zones.size()); z++) {
            int n_conns = 0;
            file.check(cg_nconns(fn, B, z, &n_conns), "reading connectivities");
            for (int i = 1; i <= n_conns; i++) {
                char conn_name[33] = {}, donor_name[33] = {};
                CGNS_ENUMT(GridLocation_t) location;
                CGNS_ENUMT(GridConnectivityType_t) type;
                CGNS_ENUMT(PointSetType_t) ptset, donor_ptset;
                CGNS_ENUMT(ZoneType_t) donor_zone_type;
                CGNS_ENUMT(DataType_t) donor_type;
                cgsize_t n_points = 0, n_donor = 0;
                file.check(cg_conn_info(fn, B, z, i, conn_name, &location, &type, &ptset, &n_points, donor_name,
                                        &donor_zone_type, &donor_ptset, &donor_type, &n_donor),
                           "reading connectivity");
                const ZoneInfo & zone = zones[size_t(z - 1)];
                const std::string what = "connectivity " + std::string(conn_name) + " of zone " + zone.name;
                if (type != CGNS_ENUMV(Abutting1to1)) file.fail(what + " is not Abutting1to1; Mallard joins zones only 1-to-1.");
                if (location != CGNS_ENUMV(Vertex) || ptset != CGNS_ENUMV(PointList) || n_donor != n_points) {
                    file.fail(what + ": Mallard reads vertex PointList/PointListDonor connectivities.");
                }
                auto donor = std::find_if(zones.begin(), zones.end(), [&](const ZoneInfo & zz) { return zz.name == donor_name; });
                if (donor == zones.end()) file.fail(what + ": unknown donor zone " + donor_name + ".");
                check_count(n_points, "connectivity point");
                std::vector<cgsize_t> points(static_cast<size_t>(n_points)), donors(static_cast<size_t>(n_donor));
                file.check(cg_conn_read(fn, B, z, i, points.data(),
                                        sizeof(cgsize_t) == 8 ? CGNS_ENUMV(LongInteger) : CGNS_ENUMV(Integer), donors.data()),
                           "reading " + what);
                for (size_t k = 0; k < points.size(); k++) {
                    if (points[k] < 1 || points[k] > zone.n_nodes || donors[k] < 1 || donors[k] > donor->n_nodes) {
                        file.fail(what + ": node id out of range.");
                    }
                    classes.join(zone.first_node + uint64_t(points[k] - 1), donor->first_node + uint64_t(donors[k] - 1));
                }
            }
        }
    });

    // Names of boundary faces and cells: BCs (through their family) first; then sections' names or families
    constexpr uint32_t NO_BC = ~uint32_t(0);
    std::vector<std::string> face_name(faces.size()), bc_names;
    std::vector<uint32_t> cell_bc(cells.size(), NO_BC);  // index into bc_names
    collectively([&] {
        for (int z = 1; z <= int(zones.size()); z++) {
            const ZoneInfo & zone = zones[size_t(z - 1)];
            int n_bocos = 0;
            file.check(cg_nbocos(fn, B, z, &n_bocos), "reading boundary conditions");
            for (int bc = 1; bc <= n_bocos; bc++) {
                char bc_name[33] = {};
                CGNS_ENUMT(BCType_t) bc_type;
                CGNS_ENUMT(PointSetType_t) ptset;
                cgsize_t n_points = 0, normal_size = 0;
                int normal_index[3] = {}, n_datasets = 0;
                CGNS_ENUMT(DataType_t) normal_type;
                file.check(cg_boco_info(fn, B, z, bc, bc_name, &bc_type, &ptset, &n_points, normal_index, &normal_size,
                                        &normal_type, &n_datasets),
                           "reading boundary condition");
                CGNS_ENUMT(GridLocation_t) location;
                file.check(cg_boco_gridlocation_read(fn, B, z, bc, &location), "reading boundary condition location");
                check_count(n_points, "boundary condition point");
                std::vector<cgsize_t> points(static_cast<size_t>(n_points));
                file.check(cg_boco_read(fn, B, z, bc, points.data(), nullptr), "reading boundary condition " + std::string(bc_name));
                std::string name = bc_name, family;
                if (cg_goto(fn, B, "Zone_t", z, "ZoneBC_t", 1, "BC_t", bc, "end") == CG_OK && read_famname(family)) {
                    name = family_zone_name(file, B, family);
                }
                cgsize_t range_lo = 0, range_hi = -1;
                if (ptset == CGNS_ENUMV(PointRange) || ptset == CGNS_ENUMV(ElementRange)) {
                    if (points.size() != 2 || points[0] < 1 || points[1] < points[0] || points[1] > file_size) {
                        file.fail("boundary condition " + std::string(bc_name) + ": bad range.");
                    }
                    range_lo = points[0];
                    range_hi = points[1];
                    points.clear();
                } else if (ptset != CGNS_ENUMV(PointList) && ptset != CGNS_ENUMV(ElementList)) {
                    file.fail("boundary condition " + std::string(bc_name) + ": unsupported point set type.");
                }
                auto name_face = [&](uint32_t f) {
                    if (face_name[f].empty()) face_name[f] = name;
                };
                if (location == CGNS_ENUMV(Vertex)) {
                    std::vector<uint64_t> vertices;
                    auto add_vertex = [&](cgsize_t v) {
                        if (v < 1 || v > zone.n_nodes) file.fail("boundary condition " + std::string(bc_name) + ": node id out of range.");
                        vertices.push_back(zone.first_node + uint64_t(v - 1));
                    };
                    for (cgsize_t v : points) add_vertex(v);
                    for (cgsize_t v = range_lo; v <= range_hi; v++) add_vertex(v);
                    std::sort(vertices.begin(), vertices.end());
                    for (uint32_t f = 0; f < faces.size(); f++) {
                        if (face_zone_of[f] != uint32_t(z - 1)) continue;
                        const auto corners = faces[f];
                        if (std::all_of(corners.begin(), corners.end(),
                                        [&](uint64_t n) { return std::binary_search(vertices.begin(), vertices.end(), n); })) {
                            name_face(f);
                        }
                    }
                    continue;
                }
                auto name_element = [&](cgsize_t e) {
                    const ElementRef ref = zone.element(e);
                    if (ref.kind == ElementRef::FACE) name_face(ref.index);
                    if (ref.kind == ElementRef::CELL && cell_bc[ref.index] == NO_BC && !name.empty()) {
                        if (bc_names.empty() || bc_names.back() != name) bc_names.push_back(name);
                        cell_bc[ref.index] = uint32_t(bc_names.size() - 1);
                    }
                };
                for (cgsize_t e : points) {
                    if (e < 1 || e > zone.max_id) file.fail("boundary condition " + std::string(bc_name) + ": element id out of range.");
                    name_element(e);
                }
                if (range_hi < range_lo) continue;
                if (range_hi > zone.max_id) file.fail("boundary condition " + std::string(bc_name) + ": element id out of range.");
                // A range: only the ids of this rank's elements
                for (const SectionInfo & x : zone.sections) {
                    for (cgsize_t e = std::max(range_lo, x.from); e <= std::min(range_hi, x.to); e++) name_element(e);
                }
            }
        }
    });
    std::vector<std::string> local_names;
    std::set<std::string> seen;
    for (uint32_t f = 0; f < faces.size(); f++) {
        if (face_name[f].empty()) face_name[f] = zones[face_section[f][0]].sections[face_section[f][1]].name;
        if (seen.insert(face_name[f]).second) local_names.push_back(face_name[f]);
    }
    for (ZoneInfo & zone : zones) {
        for (SectionInfo & x : zone.sections) release(x.refs);
    }
    MeshBlock block;
    block.zone_names = parallel_read::names_in_order(local_names);
    const std::string none;
    auto cell_name = [&](uint32_t c) -> const std::string & {
        const std::string & family = zones[cell_section[c][0]].sections[cell_section[c][1]].family;
        return !family.empty() ? family : cell_bc[c] == NO_BC ? none : bc_names[cell_bc[c]];
    };
    local_names.clear();
    seen.clear();
    for (uint32_t c = 0; c < cells.size(); c++) {
        const std::string & name = cell_name(c);
        if (!name.empty() && seen.insert(name).second) local_names.push_back(name);
    }
    block.cell_zone_names = parallel_read::names_in_order(local_names);
    auto index_of = [](const std::vector<std::string> & names, const std::string & name) {
        return uint64_t(std::find(names.begin(), names.end(), name) - names.begin());
    };

    // Merged nodes, and every node of curved faces
    for (uint64_t & n : cells.words) n = classes.root(n);
    for (uint64_t & n : faces.words) n = classes.root(n);
    const std::vector<uint64_t> face_all_xyz = parallel_read::ask(
        face_all.words, N_DIM, [&](uint64_t n) { return part_of(n, n_nodes, p); },
        [&](uint64_t n, uint64_t * out) { FOR_I_DIM out[i] = std::bit_cast<uint64_t>(xyz[n - first_node][i]); });

    {
        std::vector<uint64_t> index(cells.size());
        Records records;
        for (size_t c = 0; c < cells.size(); c++) {
            index[c] = my_first_cell + c;
            const std::string & name = cell_name(uint32_t(c));
            records.words.push_back(name.empty() ? Mesh::NO_CELL_ZONE : index_of(block.cell_zone_names, name));
            records.words.insert(records.words.end(), cells[c].begin(), cells[c].end());
            records.end_record();
        }
        release(cells.words);
        release(cells.offsets);
        release(cell_bc);
        release(cell_section);
        const Records mine = parallel_read::to_owners(index, std::move(records), n_cells);
        block.first_cell = block_begin(n_cells, r, p);
        for (size_t c = 0; c < mine.size(); c++) {
            block.add_cell(mine[c].subspan(1));
            if (!block.cell_zone_names.empty()) block.cell_zone.push_back(static_cast<uint32_t>(mine[c][0]));
        }
    }
    {
        std::vector<uint64_t> index(faces.size());
        Records records;
        for (size_t f = 0; f < faces.size(); f++) {
            index[f] = my_first_face + f;
            records.words.push_back(index_of(block.zone_names, face_name[f]));
            records.words.push_back(faces[f].size());
            records.words.insert(records.words.end(), faces[f].begin(), faces[f].end());
            records.words.insert(records.words.end(), face_all_xyz.begin() + std::ptrdiff_t(face_all.offsets[f] * N_DIM),
                                 face_all_xyz.begin() + std::ptrdiff_t(face_all.offsets[f + 1] * N_DIM));
            records.end_record();
        }
        parallel_read::place_faces(index, std::move(records), n_faces, curved_faces != 0, block);
    }
    parallel_read::place_used_nodes(n_nodes, std::move(xyz), {&block.cell_nodes, &block.face_nodes}, block);
    return block;
}

#else

bool have_cgns() { return false; }

int cgns_dimension(const std::string & filename) {
    throw std::runtime_error("Cannot read " + filename + ": Mallard was built without CGNS (-DMallard_ENABLE_CGNS=ON).");
}

MeshBlock read_cgns_block(const std::string & filename) {
    throw std::runtime_error("Cannot read " + filename + ": Mallard was built without CGNS (-DMallard_ENABLE_CGNS=ON).");
}

#endif
