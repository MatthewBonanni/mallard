/**
 * @file region_snapshots_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief [[write_data]] of regions: volumes as subsets of the mesh; surfaces,
 *        curves, points and boundary zones as the probes of them write them.
 * @version 0.1
 * @date 2026-10-30
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "comm.h"
#include "input.h"
#include "mesh_block.h"
#include "region.h"
#include "scratch.h"
#include "solver.h"
#include "test_fixtures.h"
#include "test_utils.h"

#ifdef Mallard_HAS_HDF5
#include "hdf5_output.h"
#include "hdf5_util.h"
#endif

namespace {

std::string point(double x, double y, double z) {
    std::ostringstream s;
    s << "[" << x << ", " << y << (N_DIM == 3 ? ", " + std::to_string(z) : "") << "]";
    return s.str();
}

std::string fresh_dir(const std::string & name) {
    const std::string dir = (scratch_dir() / ("region_snapshots_" + name)).string();
    comm::barrier();
    if (comm::is_root()) std::filesystem::remove_all(dir);
    comm::barrier();
    return dir;
}

std::string read_all(const std::string & file) {
    std::ifstream in(file);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

/** @brief A box of extrapolation boundaries but a symmetric bottom, with smooth fields. */
std::string box_case(const std::string & mesh, const std::string & recon, const std::string & extra,
                     const std::string & run = "n_steps = 4\ndt = 2.0e-3\n") {
    std::ostringstream s;
    s << "[run]\n" << run << "[mesh]\ntype = \"" << mesh << "\"\nNx = 8\nNy = 6\nLx = 1.0\nLy = 1.0\n"
      << (N_DIM == 3 ? "Nz = 4\nLz = 1.0\n" : "")
      << "[initialize]\ntype = \"analytical\"\nrho = \"1.0 + 0.2 * sin(2 * pi * x) * cos(pi * y)\"\n"
      << "u = " << (N_DIM == 3 ? "[\"0.3\", \"-0.1 * x\", \"0.2\"]" : "[\"0.3\", \"-0.1 * x\"]")
      << "\np = \"1.0 + 0.1 * x * y\"\n";
    const char * zones[6] = {"left", "right", "bottom", "top", "back", "front"};
    for (int k = 0; k < 2 * N_DIM; k++) {
        s << "[[boundaries]]\nname = \"" << zones[k] << "\"\ntype = \""
          << (k == 2 ? "symmetry" : "extrapolation") << "\"\n";
    }
    s << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n[numerics.face_reconstruction]\n" << recon
      << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "[output]\ncheck_interval = 1000000\n" << extra;
    return s.str();
}

void run_case(const std::string & input) {
    Solver solver;
    solver.init(parse_toml(input));
    solver.run();
    comm::barrier();
}

/** @brief The keys of the regions of lower dimension the tests write, by name. */
std::map<std::string, std::string> sampled_regions() {
    std::map<std::string, std::string> r = {
        {"slice", "type = \"cut_plane\"\npoint = " + point(0.43, 0.5, 0.5) + "\nnormal = " + point(1.0, 0.4, 0.2) + "\n"},
        {"line", "type = \"line\"\nstart = " + point(0.1, 0.2, 0.3) + "\nend = " + point(0.9, 0.7, 0.6) + "\nn_points = 9\n"},
        {"pts", "type = \"points\"\npoints = [" + point(0.3, 0.3, 0.3) + ", " + point(0.7, 0.6, 0.4) + "]\n"},
    };
    if (N_DIM == 3) {
        r["surf"] = "type = \"surface\"\nx = \"0.2 + 0.6 * u\"\ny = \"0.3 + 0.4 * v\"\nz = \"0.5 + 0.1 * u * v\"\n"
                    "n_points = [4, 5]\n";
    }
    return r;
}


/**
 * @brief The appended raw arrays of an uncompressed VTU file, by name (Points,
 *        connectivity, offsets, types and the cell data).
 */
std::map<std::string, std::string> vtu_arrays(const std::string & filename) {
    const std::string text = read_all(filename);
    const std::string marker = "<AppendedData encoding=\"raw\">\n_";
    const size_t base = text.find(marker) + marker.size();
    std::map<std::string, std::string> arrays;
    for (size_t at = 0; (at = text.find("<DataArray", at)) < base; at++) {
        const size_t end = text.find("/>", at);
        const std::string tag = text.substr(at, end - at);
        const size_t name = tag.find("Name=\""), offset = tag.find("offset=\"");
        if (offset == std::string::npos) continue;
        const std::string key = name == std::string::npos ? "Points" : tag.substr(name + 6, tag.find('"', name + 6) - name - 6);
        size_t p = base + std::stoull(tag.substr(offset + 8));
        uint64_t n = 0;
        std::memcpy(&n, text.data() + p, sizeof(n));
        arrays[key] = text.substr(p + sizeof(n), n);
    }
    return arrays;
}

template <typename T>
std::vector<T> as(const std::string & bytes) {
    std::vector<T> values(bytes.size() / sizeof(T));
    std::memcpy(values.data(), bytes.data(), values.size() * sizeof(T));
    return values;
}

/** @brief Each cell's node coordinates, in the order of its connectivity. */
std::vector<std::vector<double>> cell_points(const std::vector<double> & points, const std::vector<int64_t> & connectivity,
                                             const std::vector<int64_t> & offsets, bool leading_zero) {
    std::vector<std::vector<double>> cells;
    int64_t first = 0;
    for (size_t c = leading_zero ? 1 : 0; c < offsets.size(); c++) {
        std::vector<double> x;
        for (int64_t k = first; k < offsets[c]; k++) {
            for (int d = 0; d < 3; d++) x.push_back(points[3 * size_t(connectivity[size_t(k)]) + size_t(d)]);
        }
        cells.push_back(x);
        first = offsets[c];
    }
    return cells;
}

/** @brief The region's cells (fraction > 0) of the solver's mesh, in global order, and every cell's fraction. */
std::pair<std::vector<uint32_t>, std::vector<double>> region_cells(const Solver & solver, const std::string & table) {
    const Input value = parse_toml("r = " + table + "\n").at("r");
    const region::Region region(value, "test");
    const std::vector<double> fractions = region::cell_fractions(*solver.get_mesh(), region);
    std::vector<uint32_t> cells;
    for (uint32_t c = 0; c < solver.get_mesh()->n_owned(); c++) {
        if (fractions[c] > 0.0) cells.push_back(c);
    }
    return {cells, fractions};
}

} // namespace

TEST(RegionSnapshotsTest, ZonesCutPlanesSurfacesCurvesAndPointsAreWrittenAsTheirPointProbesWriteThem) {
    // The snapshot and the probe of a region share their sampling code: the
    // same point values (of the reconstruction) on the same geometry
    const std::string dir = fresh_dir("sampled");
    std::string names = "\"bottom\"";
    std::ostringstream s;
    s << "[regions]\n";
    for (const auto & [name, keys] : sampled_regions()) {
        std::string inline_keys = keys;
        for (size_t k; (k = inline_keys.find('\n')) != std::string::npos;) inline_keys.replace(k, 1, ", ");
        s << name << " = {" << inline_keys.substr(0, inline_keys.size() - 2) << "}\n";
        names += ", \"" + name + "\"";
    }
    const std::string variables = "variables = [\"RHO\", \"U\", \"P\"]\ninterval = 2\n";
    s << "[[write_data]]\nprefix = \"" << dir << "/s/s\"\nformat = \"vtu\"\nregions = [" << names << "]\n" << variables;
    auto probe = [&](const std::string & name, const std::string & format, const std::string & where) {
        const std::string sub = format == "vtp" ? "p" : "pk";
        s << "[[probes]]\nname = \"" << sub << "_" << name << "\"\nsampling = \"point\"\nformat = \"" << format
          << "\"\nfile = \"" << dir << "/" << sub << "/" << name << (format == "vtp" ? ".pvd" : ".vtkhdf") << "\"\n"
          << where << variables;
    };
    probe("named_slice", "vtp", "region = \"slice\"\n");
    probe("bottom", "vtp", "boundary = \"bottom\"\n");
    for (const auto & [name, keys] : sampled_regions()) probe(name, "vtp", keys);
#ifdef Mallard_HAS_HDF5
    s << "[[write_data]]\nprefix = \"" << dir << "/k/k\"\nformat = \"vtkhdf\"\nregions = [" << names << "]\n" << variables;
    probe("bottom", "vtkhdf", "boundary = \"bottom\"\n");
    for (const auto & [name, keys] : sampled_regions()) probe(name, "vtkhdf", keys);
#endif
    run_case(box_case("cartesian", "type = \"TENO\"\norder = 3\n", s.str()));
    if (!comm::is_root()) return;

    std::vector<std::string> all = {"bottom"};
    for (const auto & [name, keys] : sampled_regions()) all.push_back(name);
    for (const auto & name : all) {
        for (const char * step : {"_000000.vtp", "_000002.vtp", "_000004.vtp"}) {
            const std::string snapshot = read_all(dir + "/s/s_" + name + step);
            EXPECT_GT(snapshot.size(), 500u) << name << step;
            EXPECT_EQ(without_provenance(snapshot), without_provenance(read_all(dir + "/p/" + name + step))) << name << step;
        }
        EXPECT_TRUE(std::filesystem::exists(dir + "/s/s_" + name + ".pvd")) << name;
#ifdef Mallard_HAS_HDF5
        const auto snapshot = h5_all_datasets(dir + "/k/k_" + name + ".vtkhdf");
        EXPECT_GT(snapshot.size(), 8u) << name;
        EXPECT_EQ(snapshot, h5_all_datasets(dir + "/pk/" + name + ".vtkhdf")) << name;
#endif
    }
    EXPECT_EQ(without_provenance(read_all(dir + "/p/named_slice_000002.vtp")),
              without_provenance(read_all(dir + "/p/slice_000002.vtp")))
        << "a probe of a named region of lower dimension samples it as its type keys do";
}

TEST(RegionSnapshotsTest, ZonesCutPlanesSurfacesCurvesAndPointsDoNotDependOnTheRankCount) {
    const std::string dir = fresh_dir("ranks");
    auto input = [&](const std::string & sub) {
        std::ostringstream s;
        std::string names = "\"bottom\"";
        s << "[regions]\n";
        for (const auto & [name, keys] : sampled_regions()) {
            std::string inline_keys = keys;
            for (size_t k; (k = inline_keys.find('\n')) != std::string::npos;) inline_keys.replace(k, 1, ", ");
            s << name << " = {" << inline_keys.substr(0, inline_keys.size() - 2) << "}\n";
            names += ", \"" + name + "\"";
        }
        s << "[[write_data]]\nprefix = \"" << dir << "/" << sub << "/v\"\nformat = \"vtu\"\ninterval = 2\n"
          << "variables = [\"RHO\", \"U\", \"MASS_FLUX\", \"CFL\"]\nregions = [\"bottom\", \"slice\"" << (N_DIM == 3 ? ", \"surf\"" : "")
          << "]\n[[write_data]]\nprefix = \"" << dir << "/" << sub << "/c\"\nformat = \"vtu\"\ninterval = 2\n"
          << "variables = [\"RHO\", \"P\"]\nregions = [" << names << "]\n";
#ifdef Mallard_HAS_HDF5
        s << "[[write_data]]\nprefix = \"" << dir << "/" << sub << "/k\"\nformat = \"vtkhdf\"\ninterval = 2\n"
          << "variables = [\"RHO\", \"P\"]\nregions = [" << names << "]\n";
#endif
        return box_case(N_DIM == 3 ? "cartesian_tet" : "cartesian_tri", "type = \"MUSCL\"\n", s.str());
    };
    {
        Solver distributed;
        distributed.init(parse_toml(input("d")));
        distributed.run();
        Solver serial;
        serial.set_distributed(false);
        serial.init(parse_toml(input("s")));
        serial.run();
    }
    comm::barrier();
    if (!comm::is_root()) return;
    std::vector<std::string> files = {"v_bottom_000004.vtp", "v_slice_000004.vtp"};
    if (N_DIM == 3) files.push_back("v_surf_000004.vtp");
    files.push_back("c_bottom_000002.vtp");
    for (const auto & [name, keys] : sampled_regions()) files.push_back("c_" + name + "_000002.vtp");
    for (const auto & f : files) {
        const std::string serial = read_all(dir + "/s/" + f);
        EXPECT_GT(serial.size(), 500u) << f;
        EXPECT_EQ(without_provenance(read_all(dir + "/d/" + f)), without_provenance(serial)) << f << " on " << comm::size()
                                                                                             << " ranks";
    }
#ifdef Mallard_HAS_HDF5
    for (const auto & [name, keys] : sampled_regions()) {
        EXPECT_EQ(h5_all_datasets(dir + "/d/k_" + name + ".vtkhdf"), h5_all_datasets(dir + "/s/k_" + name + ".vtkhdf"))
            << name;
    }
    EXPECT_EQ(h5_all_datasets(dir + "/d/k_bottom.vtkhdf"), h5_all_datasets(dir + "/s/k_bottom.vtkhdf"));
#endif
}

TEST(RegionSnapshotsTest, AVolumeIsTheFullSnapshotRestrictedToTheCellsItCovers) {
    // Serial, where the files hold the cells in the mesh's order (MPITest compares ranks)
    if (comm::size() > 1) GTEST_SKIP() << "serial";
    const std::string dir = fresh_dir("volume");
    const std::string ball = std::string("{type = \"sphere\", center = ") + point(0.45, 0.5, 0.5) +
                             ", radius = 0.3, edge = 0.1}";
    const std::string variables = "variables = [\"RHO\", \"U\", \"P\"]\ninterval = 2\n";
    std::string writers = "[[write_data]]\nprefix = \"" + dir + "/flow\"\nformat = \"vtu\"\n" + variables +
                          "[[write_data]]\nprefix = \"" + dir + "/near\"\nformat = \"vtu\"\nregion = " + ball + "\n" + variables;
#ifdef Mallard_HAS_HDF5
    writers += "[[write_data]]\nprefix = \"" + dir + "/kflow\"\nformat = \"vtkhdf\"\n" + variables +
               "[[write_data]]\nprefix = \"" + dir + "/knear\"\nformat = \"vtkhdf\"\nregion = " + ball + "\n" + variables;
#endif
    const std::string input = box_case(N_DIM == 3 ? "cartesian_tet" : "cartesian_tri", "type = \"MUSCL\"\n", writers);
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(input));
    solver.run();
    const auto [cells, fractions] = region_cells(solver, ball);
    ASSERT_GT(cells.size(), 4u);
    ASSERT_LT(cells.size(), size_t(solver.get_mesh()->n_owned()));
    size_t n_partial = 0;
    for (uint32_t c : cells) n_partial += fractions[c] < 1.0;
    EXPECT_GT(n_partial, 0u) << "cells across the edge";

    // Every array of the subset is the full one's at the region's cells, which keep their nodes
    auto check = [&](const std::vector<std::vector<double>> & full_cells, const std::vector<std::vector<double>> & near_cells,
                     const std::map<std::string, std::vector<rtype>> & full, const std::map<std::string, std::vector<rtype>> & near,
                     const std::vector<rtype> & fraction, const std::string & what) {
        ASSERT_EQ(near_cells.size(), cells.size()) << what;
        ASSERT_EQ(fraction.size(), cells.size()) << what;
        for (size_t k = 0; k < cells.size(); k++) {
            EXPECT_EQ(near_cells[k], full_cells[cells[k]]) << what << " " << k;
            EXPECT_EQ(double(fraction[k]), double(static_cast<rtype>(fractions[cells[k]]))) << what << " " << k;
            for (const auto & [name, values] : full) {
                const size_t width = values.size() / full_cells.size();
                for (size_t i = 0; i < width; i++) {
                    EXPECT_EQ(near.at(name)[width * k + i], values[width * cells[k] + i]) << what << " " << name << " " << k;
                }
            }
        }
    };
    for (const char * step : {"_000000", "_000004"}) {
        auto full = vtu_arrays(dir + "/flow" + step + ".vtu"), near = vtu_arrays(dir + "/near" + step + ".vtu");
        auto geometry = [](std::map<std::string, std::string> & a) {
            const std::vector<rtype> points = as<rtype>(a.at("Points"));
            return cell_points(std::vector<double>(points.begin(), points.end()), as<int64_t>(a.at("connectivity")), as<int64_t>(a.at("offsets")),
                               false);
        };
        std::map<std::string, std::vector<rtype>> full_values, near_values;
        for (const char * v : {"RHO", "U", "P"}) {
            full_values[v] = as<rtype>(full.at(v));
            near_values[v] = as<rtype>(near.at(v));
        }
        check(geometry(full), geometry(near), full_values, near_values, as<rtype>(near.at("REGION_FRACTION")),
              std::string("vtu") + step);
        const std::vector<int64_t> connectivity = as<int64_t>(near.at("connectivity"));
        const std::set<int64_t> used(connectivity.begin(), connectivity.end());
        EXPECT_EQ(as<rtype>(near.at("Points")).size(), 3 * used.size()) << "only the region's nodes";
#ifdef Mallard_HAS_HDF5
        const std::string kf = dir + "/kflow" + step + ".vtkhdf", kn = dir + "/knear" + step + ".vtkhdf";
        auto h5_geometry = [](const std::string & f) {
            return cell_points(h5_values<double>(f, "VTKHDF/Points"), h5_values<int64_t>(f, "VTKHDF/Connectivity"),
                               h5_values<int64_t>(f, "VTKHDF/Offsets"), true);
        };
        std::map<std::string, std::vector<rtype>> kfull, knear;
        for (const char * v : {"RHO", "U", "P"}) {
            kfull[v] = h5_values<rtype>(kf, std::string("VTKHDF/CellData/") + v);
            knear[v] = h5_values<rtype>(kn, std::string("VTKHDF/CellData/") + v);
        }
        check(h5_geometry(kf), h5_geometry(kn), kfull, knear, h5_values<rtype>(kn, "VTKHDF/CellData/REGION_FRACTION"),
              std::string("vtkhdf") + step);
        std::set<int64_t> k_used;
        for (int64_t i : h5_values<int64_t>(kn, "VTKHDF/Connectivity")) k_used.insert(i);
        EXPECT_EQ(h5_values<int64_t>(kn, "VTKHDF/NumberOfPoints"), std::vector<int64_t>{int64_t(k_used.size())});
        h5::Handle file(H5Fopen(kn.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose, kn);
        h5::Handle fraction(H5Dopen2(file, "VTKHDF/CellData/REGION_FRACTION", H5P_DEFAULT), H5Dclose, "REGION_FRACTION");
        EXPECT_EQ(h5::read_string_attribute(fraction, "units"), "1");
#endif
    }
}

TEST(RegionSnapshotsTest, RegionsWriteOneSeriesEachAndTheLastStateToo) {
    // regions = [...] writes a series per region, after its name or place;
    // "domain" is the whole mesh; the run's end writes every region once more
    const std::string dir = fresh_dir("list");
    const std::string writers =
        "[regions]\nslice = {type = \"cut_plane\", point = " + point(0.5, 0.5, 0.5) + ", normal = " + point(0.0, 1.0, 0.0) +
        "}\n[[write_data]]\nprefix = \"" + dir + "/r\"\nformat = \"vtu\"\ninterval = 3\nvariables = [\"P\"]\n"
        "regions = [\"slice\", {type = \"" + std::string(N_DIM == 3 ? "box" : "rectangle") + "\", center = " + point(0.25, 0.25, 0.25) + ", size = " + point(0.5, 0.5, 0.5) +
        "}, \"domain\", \"top\"]\n";
    Solver solver;
    solver.init(parse_toml(box_case("cartesian", "type = \"MUSCL\"\n", writers)));
    solver.run();
    comm::barrier();
    if (!comm::is_root()) return;
    const bool pieces = solver.get_mesh()->n_global_cells > 0;
    for (const char * step : {"_000000", "_000003", "_000004"}) {
        EXPECT_TRUE(std::filesystem::exists(dir + "/r_slice" + std::string(step) + ".vtp")) << step;
        EXPECT_TRUE(std::filesystem::exists(dir + "/r_top" + std::string(step) + ".vtp")) << step;
        EXPECT_TRUE(std::filesystem::exists(dir + "/r_1" + std::string(step) + (pieces ? ".pvtu" : ".vtu"))) << step;
        EXPECT_TRUE(std::filesystem::exists(dir + "/r_domain" + std::string(step) + (pieces ? ".pvtu" : ".vtu"))) << step;
    }
    EXPECT_FALSE(std::filesystem::exists(dir + "/r_slice_000001.vtp"));
    if (!pieces) {
        const std::string box = read_all(dir + "/r_1_000003.vtu"), domain = read_all(dir + "/r_domain_000003.vtu");
        EXPECT_NE(box.find("NumberOfCells=\"" + std::to_string(4 * 3 * (N_DIM == 3 ? 2 : 1)) + "\""), std::string::npos);
        EXPECT_NE(domain.find("NumberOfCells=\"" + std::to_string(8 * 6 * (N_DIM == 3 ? 4 : 1)) + "\""), std::string::npos);
        EXPECT_NE(box.find("Name=\"REGION_FRACTION\""), std::string::npos);
        EXPECT_EQ(domain.find("Name=\"REGION_FRACTION\""), std::string::npos);
    }
}

TEST(RegionSnapshotsTest, KeysThatDoNotFitTheRegionAreErrors) {
    const std::string dir = fresh_dir("errors");
    const std::string writer = "[[write_data]]\nprefix = \"" + dir + "/e\"\ninterval = 1\nvariables = [\"P\"]\n";
    const std::string slice = "{type = \"cut_plane\", point = " + point(0.5, 0.5, 0.5) + ", normal = " + point(1, 0, 0) + "}";
    const std::string ball = "{type = \"sphere\", center = " + point(0.5, 0.5, 0.5) + ", radius = 0.2}";
    auto message = [&](const std::string & keys) {
        Solver solver;
        try {
            solver.init(parse_toml(box_case("cartesian", "type = \"MUSCL\"\n", writer + keys)));
        } catch (const InputError & e) {
            return std::string(e.what());
        }
        return std::string();
    };
    EXPECT_NE(message("region = \"bottom\"\nregions = [\"top\"]\n").find("write_data.regions"), std::string::npos);
    const std::string unknown = message("region = \"botom\"\n");
    EXPECT_NE(unknown.find("no region named \"botom\""), std::string::npos) << unknown;
    EXPECT_NE(unknown.find("bottom"), std::string::npos) << unknown;
    EXPECT_NE(message("format = \"restart\"\nregion = " + slice + "\n").find("does not hold surfaces"), std::string::npos);
    EXPECT_NE(message("format = \"vtu\"\nseries = false\nregion = " + slice + "\n").find("write_data.series"),
              std::string::npos);
    EXPECT_NE(message("format = \"restart\"\nregion = " + ball + "\n").find("only as vtkhdf or vtu"), std::string::npos);
    EXPECT_NE(message("format = \"restart\"\nregion = \"top\"\n").find("does not hold surfaces"), std::string::npos);
    EXPECT_NE(message("format = \"vtu\"\nregion = {type = \"sphere\", center = " + point(5, 5, 5) + ", radius = 0.1}\n")
                  .find("part of any cell"),
              std::string::npos);
    EXPECT_EQ(message("format = \"vtu\"\nregion = " + slice + "\n"), "");
}
