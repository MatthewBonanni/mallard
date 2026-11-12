// CGNS mesh files. Raw bytes would mostly exercise libhdf5's and libcgns's own
// parsers, so the input describes a file written with the CGNS API instead:
// zones, coordinates, element sections, zone connectivities, boundary
// conditions and families, consistent or not.
#include <array>
#include <string>
#include <vector>

#include <cgnslib.h>
#include <fuzzer/FuzzedDataProvider.h>

#include "fuzz_common.h"

namespace {

constexpr std::array<CGNS_ENUMT(ElementType_t), 15> ELEMENT_TYPES = {
    CGNS_ENUMV(NODE),    CGNS_ENUMV(BAR_2),   CGNS_ENUMV(BAR_3),  CGNS_ENUMV(TRI_3),   CGNS_ENUMV(TRI_6),
    CGNS_ENUMV(QUAD_4),  CGNS_ENUMV(QUAD_9),  CGNS_ENUMV(TETRA_4), CGNS_ENUMV(PYRA_5), CGNS_ENUMV(PENTA_6),
    CGNS_ENUMV(HEXA_8),  CGNS_ENUMV(HEXA_27), CGNS_ENUMV(MIXED),  CGNS_ENUMV(NGON_n),  CGNS_ENUMV(TETRA_10)};
constexpr std::array<CGNS_ENUMT(PointSetType_t), 4> POINT_SETS = {
    CGNS_ENUMV(PointList), CGNS_ENUMV(PointRange), CGNS_ENUMV(ElementList), CGNS_ENUMV(ElementRange)};
constexpr std::array<CGNS_ENUMT(GridLocation_t), 4> LOCATIONS = {
    CGNS_ENUMV(Vertex), CGNS_ENUMV(FaceCenter), CGNS_ENUMV(EdgeCenter), CGNS_ENUMV(CellCenter)};

std::vector<cgsize_t> ids(FuzzedDataProvider & in, size_t n, cgsize_t n_valid) {
    std::vector<cgsize_t> v(n);
    for (auto & id : v) {
        id = in.ConsumeBool() ? in.ConsumeIntegralInRange<cgsize_t>(1, std::max<cgsize_t>(n_valid, 1))
                              : in.ConsumeIntegral<cgsize_t>();
    }
    return v;
}

int npe(CGNS_ENUMT(ElementType_t) type) {
    int n = 0;
    return cg_npe(type, &n) == CG_OK ? n : 0;
}

void write_zone(FuzzedDataProvider & in, int fn, int B, int z, int n_zones, int & n_families) {
    const std::string name = "zone" + std::to_string(z);
    const cgsize_t n_nodes = in.ConsumeIntegralInRange<cgsize_t>(1, 40);
    const cgsize_t size[3] = {n_nodes, in.ConsumeIntegralInRange<cgsize_t>(1, 40), 0};
    int Z = 0;
    if (cg_zone_write(fn, B, name.c_str(), size, CGNS_ENUMV(Unstructured), &Z) != CG_OK) return;
    const char * coord_names[3] = {"CoordinateX", "CoordinateY", "CoordinateZ"};
    for (const char * coord : coord_names) {
        std::vector<double> x(static_cast<size_t>(n_nodes));
        for (double & c : x) c = in.ConsumeFloatingPoint<double>();
        int C = 0;
        cg_coord_write(fn, B, Z, CGNS_ENUMV(RealDouble), coord, x.data(), &C);
    }
    cgsize_t next = 1;
    const int n_sections = in.ConsumeIntegralInRange<int>(0, 4);
    for (int s = 0; s < n_sections; s++) {
        const auto type = in.PickValueInArray(ELEMENT_TYPES);
        const cgsize_t start = in.ConsumeBool() ? next : in.ConsumeIntegralInRange<cgsize_t>(-2, 50);
        const cgsize_t end = start + in.ConsumeIntegralInRange<cgsize_t>(-1, 8);
        next = end + 1;
        const size_t n_elements = end >= start ? size_t(end - start + 1) : 0;
        const std::string section = "section" + std::to_string(s);
        int S = 0;
        if (type == CGNS_ENUMV(MIXED) || type == CGNS_ENUMV(NGON_n)) {
            std::vector<cgsize_t> conn, offsets{0};
            for (size_t e = 0; e < n_elements; e++) {
                const auto t = in.PickValueInArray(ELEMENT_TYPES);
                conn.push_back(in.ConsumeBool() ? cgsize_t(t) : in.ConsumeIntegral<cgsize_t>());
                const auto nodes = ids(in, size_t(std::max(npe(t), 1)), n_nodes);
                conn.insert(conn.end(), nodes.begin(), nodes.end());
                offsets.push_back(cgsize_t(conn.size()));
            }
            cg_poly_section_write(fn, B, Z, section.c_str(), type, start, end, 0, conn.data(), offsets.data(), &S);
        } else {
            const auto conn = ids(in, n_elements * size_t(npe(type)), n_nodes);
            cg_section_write(fn, B, Z, section.c_str(), type, start, end, 0, conn.data(), &S);
        }
        if (S > 0 && in.ConsumeBool() && cg_goto(fn, B, "Zone_t", Z, "Elements_t", S, "end") == CG_OK) {
            cg_famname_write(("family" + std::to_string(in.ConsumeIntegralInRange<int>(0, n_families))).c_str());
        }
    }
    const int n_bocos = in.ConsumeIntegralInRange<int>(0, 3);
    for (int bc = 0; bc < n_bocos; bc++) {
        const auto ptset = in.PickValueInArray(POINT_SETS);
        const bool range = ptset == CGNS_ENUMV(PointRange) || ptset == CGNS_ENUMV(ElementRange);
        const auto points = ids(in, range ? 2 : in.ConsumeIntegralInRange<size_t>(1, 16), std::max(n_nodes, next));
        int BC = 0;
        const std::string boco = "bc" + std::to_string(bc);
        if (cg_boco_write(fn, B, Z, boco.c_str(), CGNS_ENUMV(BCWall), ptset, cgsize_t(points.size()), points.data(),
                          &BC) != CG_OK) {
            continue;
        }
        cg_boco_gridlocation_write(fn, B, Z, BC, in.PickValueInArray(LOCATIONS));
        if (in.ConsumeBool() && cg_goto(fn, B, "Zone_t", Z, "ZoneBC_t", 1, "BC_t", BC, "end") == CG_OK) {
            const int F = in.ConsumeIntegralInRange<int>(0, n_families);
            cg_famname_write(("family" + std::to_string(F)).c_str());
            if (F == n_families) {
                int written = 0;
                if (cg_family_write(fn, B, ("family" + std::to_string(F)).c_str(), &written) == CG_OK) n_families++;
            }
        }
    }
    if (n_zones > 1 && in.ConsumeBool()) {
        const size_t n = in.ConsumeIntegralInRange<size_t>(1, 8);
        const auto points = ids(in, n, n_nodes);
        const auto donors = ids(in, in.ConsumeBool() ? n : n + 1, 40);
        const std::string donor = "zone" + std::to_string(in.ConsumeIntegralInRange<int>(1, n_zones + 1));
        int I = 0;
        cg_conn_write(fn, B, Z, "interface", CGNS_ENUMV(Vertex), CGNS_ENUMV(Abutting1to1), CGNS_ENUMV(PointList),
                      cgsize_t(n), points.data(), donor.c_str(), CGNS_ENUMV(Unstructured), CGNS_ENUMV(PointListDonor),
                      sizeof(cgsize_t) == 8 ? CGNS_ENUMV(LongInteger) : CGNS_ENUMV(Integer), cgsize_t(donors.size()),
                      donors.data(), &I);
    }
}

} // namespace

extern "C" int LLVMFuzzerInitialize(int *, char ***) {
    fuzz::init_kokkos();
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    FuzzedDataProvider in(data, size);
    const auto path = fuzz::scratch("mesh.cgns");
    int fn = 0;
    if (cg_open(path.c_str(), CG_MODE_WRITE, &fn) != CG_OK) return -1;
    int B = 0;
    const int cell_dim = in.ConsumeBool() ? N_DIM : in.ConsumeIntegralInRange<int>(1, 3);
    if (cg_base_write(fn, "base", cell_dim, 3, &B) == CG_OK) {
        const int n_zones = in.ConsumeIntegralInRange<int>(1, 3);
        int n_families = 0;
        for (int z = 1; z <= n_zones; z++) write_zone(in, fn, B, z, n_zones, n_families);
    }
    cg_close(fn);
    fuzz::read_mesh(path);
    return 0;
}
