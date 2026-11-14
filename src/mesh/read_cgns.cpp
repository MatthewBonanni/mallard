/**
 * @file read_cgns.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Unstructured CGNS mesh files.
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
#include <filesystem>
#include <map>
#include <numeric>
#include <set>

#include <cgnslib.h>

namespace {

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

struct ZoneInfo {
    std::string name;
    cgsize_t n_nodes = 0;
    uint64_t first_node = 0;  // in the file-wide numbering of every zone's nodes
    std::vector<ElementRef> elements;  // by element id - 1
    std::vector<int> face_section;     // per boundary face of this zone: its section
    std::vector<std::string> section_names;
    std::vector<std::string> section_families;  // a section's family zone name, or empty
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

/** @brief Union-find over the file-wide node numbering, joining nodes that zone connectivities identify. */
uint64_t find_root(std::vector<uint64_t> & parent, uint64_t n) {
    while (parent[n] != n) n = parent[n] = parent[parent[n]];
    return n;
}

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

MeshFileData read_cgns(const std::string & filename) {
    CgnsFile file(filename);
    const int fn = file.fn;
    // Every node, element or point a count announces takes at least a byte of the file
    const cgsize_t file_size = cgsize_t(std::filesystem::file_size(filename));
    auto check_count = [&](cgsize_t n, const std::string & what) {
        if (n < 0 || n > file_size) file.fail("bad " + what + " count " + std::to_string(n) + ".");
    };
    int n_bases = 0;
    file.check(cg_nbases(fn, &n_bases), "reading bases");
    if (n_bases < 1) file.fail("no CGNSBase_t node.");
    if (n_bases > 1) file.fail("has " + std::to_string(n_bases) + " bases; Mallard reads files with one.");
    const int B = 1;
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

    std::vector<ZoneInfo> zones(static_cast<size_t>(n_zones));
    std::vector<std::array<double, 3>> coords;  // every node of every zone
    MeshFileData data;
    std::vector<std::vector<uint64_t>> cell_nodes, face_nodes, face_all_nodes;  // file-wide node ids
    std::vector<int> face_zone;  // zone index of every boundary face
    std::vector<std::pair<size_t, size_t>> cell_section;  // zone and section of every cell
    std::vector<std::string> cell_bc;  // zone name of a cell from a boundary condition on cells (Gmsh's volume groups)

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
        zone.first_node = coords.size();
        coords.resize(coords.size() + size_t(zone.n_nodes), {0.0, 0.0, 0.0});
        std::vector<double> c(static_cast<size_t>(zone.n_nodes));
        const cgsize_t lo = 1, hi = zone.n_nodes;
        const char * names[3] = {"CoordinateX", "CoordinateY", "CoordinateZ"};
        for (size_t d = 0; d < N_DIM; d++) {
            file.check(cg_coord_read(fn, B, z, names[d], CGNS_ENUMV(RealDouble), &lo, &hi, c.data()),
                       std::string("reading ") + names[d] + " of zone " + zone.name);
            for (size_t n = 0; n < c.size(); n++) coords[zone.first_node + n][d] = c[n];
        }

        int n_sections = 0;
        file.check(cg_nsections(fn, B, z, &n_sections), "reading sections");
        for (int s = 1; s <= n_sections; s++) {
            char section_name[33] = {};
            CGNS_ENUMT(ElementType_t) section_type;
            cgsize_t start = 0, end = 0;
            int n_boundary = 0, parent_flag = 0;
            file.check(cg_section_read(fn, B, z, s, section_name, &section_type, &start, &end, &n_boundary, &parent_flag),
                       "reading section");
            zone.section_names.push_back(section_name);
            if (section_type == CGNS_ENUMV(NGON_n) || section_type == CGNS_ENUMV(NFACE_n)) {
                file.fail("section " + std::string(section_name) +
                          " holds polyhedra (NGON_n/NFACE_n); Mallard reads element-based sections (triangles, "
                          "quadrilaterals, tetrahedra, pyramids, prisms, hexahedra, and MIXED).");
            }
            if (start < 1 || end < start || end > file_size) {
                file.fail("section " + std::string(section_name) + " has bad element range " + std::to_string(start) +
                          " to " + std::to_string(end) + ".");
            }
            cgsize_t data_size = 0;
            file.check(cg_ElementDataSize(fn, B, z, s, &data_size), "reading section size");
            check_count(data_size, "element data");
            std::vector<cgsize_t> conn(static_cast<size_t>(data_size));
            std::vector<cgsize_t> offsets;
            if (section_type == CGNS_ENUMV(MIXED)) {
                offsets.resize(static_cast<size_t>(end - start + 2));
                file.check(cg_poly_elements_read(fn, B, z, s, conn.data(), offsets.data(), nullptr), "reading elements");
            } else {
                file.check(cg_elements_read(fn, B, z, s, conn.data(), nullptr), "reading elements");
            }
            if (zone.elements.size() < size_t(end)) zone.elements.resize(static_cast<size_t>(end));
            size_t pos = 0;
            for (cgsize_t e = start; e <= end; e++) {
                CGNS_ENUMT(ElementType_t) type_e = section_type;
                if (section_type == CGNS_ENUMV(MIXED)) {
                    const cgsize_t offset = offsets[size_t(e - start)];
                    if (offset < 0 || size_t(offset) >= conn.size()) file.fail("bad offsets in section " + std::string(section_name) + ".");
                    pos = size_t(offset);
                    type_e = static_cast<CGNS_ENUMT(ElementType_t)>(conn[pos++]);
                }
                int npe = 0;
                const ElementInfo info = element_info(type_e);
                if (info.dim < 0 || cg_npe(type_e, &npe) != CG_OK || npe <= 0) {
                    file.fail("unsupported element type " + std::string(cg_ElementTypeName(type_e)) + " in section " +
                              section_name + "; nodes, bars, triangles, quadrilaterals, tetrahedra, pyramids, prisms "
                              "and hexahedra are supported (high-order faces up to BAR_4, TRI_10 and QUAD_16).");
                }
                if (info.dim > N_DIM) {
                    file.fail("has " + std::to_string(info.dim) + "D elements, but Mallard was built with Mallard_DIM = " +
                              std::to_string(N_DIM) + ".");
                }
                if (pos + size_t(npe) > conn.size()) file.fail("section " + std::string(section_name) + " ends early.");
                std::vector<uint64_t> nodes(static_cast<size_t>(npe));
                for (auto & n : nodes) {
                    const cgsize_t local = conn[pos++];
                    if (local < 1 || local > zone.n_nodes) file.fail("node id out of range in section " + std::string(section_name) + ".");
                    n = zone.first_node + uint64_t(local - 1);
                }
                ElementRef & ref = zone.elements[size_t(e - 1)];
                if (info.dim == N_DIM) {
                    ref = {ElementRef::CELL, uint32_t(cell_nodes.size())};
                    cell_nodes.emplace_back(nodes.begin(), nodes.begin() + info.n_corners);
                    cell_section.emplace_back(size_t(z - 1), size_t(s - 1));
                } else if (info.dim == BOUNDARY_DIM) {
                    ref = {ElementRef::FACE, uint32_t(face_nodes.size())};
                    face_nodes.emplace_back(nodes.begin(), nodes.begin() + info.n_corners);
                    face_all_nodes.push_back(info.curved_face ? nodes : std::vector<uint64_t>());
                    face_zone.push_back(z - 1);
                    zone.face_section.push_back(s - 1);
                }
            }
        }
    }
    if (cell_nodes.empty()) file.fail("contains no " + std::to_string(N_DIM) + "D elements.");

    // Zones joined by 1-to-1 connectivities share their interface nodes
    std::vector<uint64_t> parent(coords.size());
    std::iota(parent.begin(), parent.end(), uint64_t(0));
    for (int z = 1; z <= n_zones; z++) {
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
            const std::string what = "connectivity " + std::string(conn_name) + " of zone " + zones[size_t(z - 1)].name;
            if (type != CGNS_ENUMV(Abutting1to1)) file.fail(what + " is not Abutting1to1; Mallard joins zones only 1-to-1.");
            if (location != CGNS_ENUMV(Vertex) || ptset != CGNS_ENUMV(PointList) || n_donor != n_points) {
                file.fail(what + ": Mallard reads vertex PointList/PointListDonor connectivities.");
            }
            auto donor = std::find_if(zones.begin(), zones.end(), [&](const ZoneInfo & zz) { return zz.name == donor_name; });
            if (donor == zones.end()) file.fail(what + ": unknown donor zone " + donor_name + ".");
            check_count(n_points, "connectivity point");
            std::vector<cgsize_t> points(static_cast<size_t>(n_points)), donors(static_cast<size_t>(n_donor));
            file.check(cg_conn_read(fn, B, z, i, points.data(), sizeof(cgsize_t) == 8 ? CGNS_ENUMV(LongInteger) : CGNS_ENUMV(Integer),
                                    donors.data()),
                       "reading " + what);
            for (size_t k = 0; k < points.size(); k++) {
                if (points[k] < 1 || points[k] > zones[size_t(z - 1)].n_nodes || donors[k] < 1 || donors[k] > donor->n_nodes) {
                    file.fail(what + ": node id out of range.");
                }
                const uint64_t a = zones[size_t(z - 1)].first_node + uint64_t(points[k] - 1);
                const uint64_t b = donor->first_node + uint64_t(donors[k] - 1);
                const uint64_t ra = find_root(parent, a), rb = find_root(parent, b);
                if (ra != rb) parent[std::max(ra, rb)] = std::min(ra, rb);
            }
        }
    }

    // Boundary zone names: BCs (through their family) take precedence over the sections' names
    std::vector<std::string> face_name(face_nodes.size());
    for (int z = 1; z <= n_zones; z++) {
        ZoneInfo & zone = zones[size_t(z - 1)];
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
            std::vector<uint32_t> faces;
            if (ptset == CGNS_ENUMV(PointRange) || ptset == CGNS_ENUMV(ElementRange)) {
                if (points.size() != 2 || points[0] < 1 || points[1] < points[0] || points[1] > file_size) {
                    file.fail("boundary condition " + std::string(bc_name) + ": bad range.");
                }
                for (cgsize_t e = points[0]; e <= points[1]; e++) points.push_back(e);
                points.erase(points.begin(), points.begin() + 2);
            } else if (ptset != CGNS_ENUMV(PointList) && ptset != CGNS_ENUMV(ElementList)) {
                file.fail("boundary condition " + std::string(bc_name) + ": unsupported point set type.");
            }
            if (location == CGNS_ENUMV(Vertex)) {
                std::set<uint64_t> vertices;
                for (cgsize_t p : points) {
                    if (p < 1 || p > zone.n_nodes) file.fail("boundary condition " + std::string(bc_name) + ": node id out of range.");
                    vertices.insert(zone.first_node + uint64_t(p - 1));
                }
                for (size_t f = 0; f < face_nodes.size(); f++) {
                    if (face_zone[f] != z - 1) continue;
                    if (std::all_of(face_nodes[f].begin(), face_nodes[f].end(), [&](uint64_t n) { return vertices.count(n); })) {
                        faces.push_back(uint32_t(f));
                    }
                }
            } else {
                for (cgsize_t e : points) {
                    if (e < 1 || size_t(e) > zone.elements.size()) {
                        file.fail("boundary condition " + std::string(bc_name) + ": element id out of range.");
                    }
                    const ElementRef & ref = zone.elements[size_t(e - 1)];
                    if (ref.kind == ElementRef::FACE) faces.push_back(ref.index);
                    if (ref.kind == ElementRef::CELL) {
                        if (cell_bc.size() <= ref.index) cell_bc.resize(ref.index + 1);
                        if (cell_bc[ref.index].empty()) cell_bc[ref.index] = name;
                    }
                }
            }
            for (uint32_t f : faces) {
                if (face_name[f].empty()) face_name[f] = name;
            }
        }
        zone.section_families.assign(zone.section_names.size(), "");
        for (size_t s = 0; s < zone.section_names.size(); s++) {
            std::string family;
            if (cg_goto(fn, B, "Zone_t", z, "Elements_t", int(s + 1), "end") == CG_OK && read_famname(family)) {
                zone.section_names[s] = family_zone_name(file, B, family);
                zone.section_families[s] = zone.section_names[s];
            }
        }
    }
    {
        std::vector<size_t> next(zones.size(), 0);
        for (size_t f = 0; f < face_nodes.size(); f++) {
            const size_t z = size_t(face_zone[f]);
            const int s = zones[z].face_section[next[z]++];
            if (face_name[f].empty()) face_name[f] = zones[z].section_names[size_t(s)];
        }
    }

    // File-wide nodes -> merged nodes
    std::vector<uint32_t> index(coords.size(), ~uint32_t(0));
    std::vector<uint64_t> merged;
    for (uint64_t n = 0; n < coords.size(); n++) {
        const uint64_t r = find_root(parent, n);
        if (r == n) {
            index[n] = uint32_t(merged.size());
            merged.push_back(n);
        } else {
            index[n] = index[r];
        }
    }
    for (uint64_t n : merged) {
        std::array<rtype, N_DIM> x;
        for (size_t d = 0; d < N_DIM; d++) x[d] = static_cast<rtype>(coords[n][d]);
        data.nodes.push_back(x);
    }
    auto renumber = [&](const std::vector<uint64_t> & nodes) {
        std::vector<uint32_t> out;
        for (uint64_t n : nodes) out.push_back(index[n]);
        return out;
    };
    for (const auto & c : cell_nodes) data.cells.push_back(renumber(c));
    for (size_t c = 0; c < cell_section.size(); c++) {
        const std::string & family = zones[cell_section[c].first].section_families[cell_section[c].second];
        if (!family.empty()) {
            data.set_cell_zone(c, family);
        } else if (c < cell_bc.size() && !cell_bc[c].empty()) {
            data.set_cell_zone(c, cell_bc[c]);
        }
    }
    for (size_t f = 0; f < face_nodes.size(); f++) {
        data.boundary_faces.push_back({renumber(face_nodes[f]), face_name[f]});
        std::vector<curved::Vec3> ho;
        for (uint64_t n : face_all_nodes[f]) ho.push_back({coords[n][0], coords[n][1], N_DIM == 3 ? coords[n][2] : 0.0});
        data.boundary_high_order.push_back(std::move(ho));
    }
    drop_unused_nodes(data);
    return data;
}

#else

bool have_cgns() { return false; }

int cgns_dimension(const std::string & filename) {
    throw std::runtime_error("Cannot read " + filename + ": Mallard was built without CGNS (-DMallard_ENABLE_CGNS=ON).");
}

MeshFileData read_cgns(const std::string & filename) {
    throw std::runtime_error("Cannot read " + filename + ": Mallard was built without CGNS (-DMallard_ENABLE_CGNS=ON).");
}

#endif
