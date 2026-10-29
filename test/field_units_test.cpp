/**
 * @file field_units_test.cpp
 * @brief Units and descriptions of output fields: the table, and the metadata
 *        every output format carries.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "comm.h"
#include "field_units.h"
#include "scratch.h"
#include "solver.h"
#include "test_fixtures.h"
#ifdef Mallard_HAS_HDF5
#include "hdf5_util.h"
#endif

namespace {

std::string read_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::vector<std::string> split(const std::string & text, char delimiter) {
    std::vector<std::string> out;
    std::stringstream ss(text);
    for (std::string item; std::getline(ss, item, delimiter);) out.push_back(item);
    return out;
}

std::string units_of(const std::string & name) { return field_units::describe(name).units(); }

/** @brief The VTK XML String arrays of a file, decoded: name -> strings. */
std::map<std::string, std::vector<std::string>> string_arrays(const std::string & text) {
    std::map<std::string, std::vector<std::string>> out;
    const std::regex array_re("<Array type=\"String\" Name=\"([^\"]+)\" NumberOfTuples=\"(\\d+)\"[^>]*>([^<]*)</Array>");
    for (auto it = std::sregex_iterator(text.begin(), text.end(), array_re); it != std::sregex_iterator(); ++it) {
        std::vector<std::string> strings(1);
        std::stringstream codes((*it)[3].str());
        for (int c; codes >> c;) {
            if (c == 0) {
                strings.emplace_back();
            } else {
                strings.back() += static_cast<char>(c);
            }
        }
        strings.pop_back();
        EXPECT_EQ(strings.size(), std::stoul((*it)[2].str())) << (*it)[1];
        out[(*it)[1].str()] = strings;
    }
    return out;
}

/** @brief The cell or point arrays a VTK XML file lists, other than the mesh's. */
std::vector<std::string> vtk_arrays(const std::string & text) {
    std::vector<std::string> names;
    const std::regex re("<DataArray[^>]* Name=\"([^\"]+)\"");
    for (auto it = std::sregex_iterator(text.begin(), text.end(), re); it != std::sregex_iterator(); ++it) {
        const std::string name = (*it)[1];
        if (name != "TIME" && name != "MALLARD_FORMAT_VERSION" && name != "Points" && name != "connectivity" &&
            name != "offsets" && name != "types") {
            names.push_back(name);
        }
    }
    return names;
}

/** @brief Every array of a VTK XML file has its table units and long name in the field-data record. */
void expect_vtk_units(const std::string & path, const std::map<std::string, std::string> & extra = {}) {
    const std::string text = read_file(path);
    // VTK reads the field data of the data set, not of a piece
    EXPECT_LT(text.find("<FieldData>"), text.find("<Piece")) << path;
    auto arrays = string_arrays(text.substr(0, text.find("<AppendedData")));
    const auto & names = arrays["MALLARD_FIELD_NAMES"];
    const auto & units = arrays["MALLARD_FIELD_UNITS"];
    const auto & long_names = arrays["MALLARD_FIELD_LONG_NAMES"];
    EXPECT_EQ(names, vtk_arrays(text.substr(0, text.find("<AppendedData")))) << path;
    ASSERT_EQ(units.size(), names.size()) << path;
    ASSERT_EQ(long_names.size(), names.size()) << path;
    for (size_t i = 0; i < names.size(); i++) {
        if (extra.count(names[i])) {
            EXPECT_EQ(units[i], extra.at(names[i])) << path << ": " << names[i];
            continue;
        }
        EXPECT_EQ(units[i], units_of(names[i])) << path << ": " << names[i];
        EXPECT_EQ(long_names[i], field_units::describe(names[i]).long_name) << path << ": " << names[i];
    }
}

/** @brief A CSV file's units line follows its header, one entry per column. */
std::map<std::string, std::string> csv_units(const std::string & path) {
    std::ifstream in(path);
    std::string header, units;
    std::getline(in, header);
    std::getline(in, units);
    EXPECT_EQ(units.rfind("# units: ", 0), 0u) << path;
    const auto names = split(header, ','), values = split(units.substr(9), ',');
    EXPECT_EQ(names.size(), values.size()) << path;
    std::map<std::string, std::string> out;
    for (size_t i = 0; i < names.size() && i < values.size(); i++) out[names[i]] = values[i];
    return out;
}

const std::string SIDE_POINT = N_DIM == 3 ? "[0.5, 0.5, 0.5]" : "[0.5, 0.5]";

/** @brief A channel (periodic in x and z, walls in y) with every kind of output that has units. */
std::string channel(const std::string & dir) {
    std::ostringstream s;
    s << "[run]\nn_steps = 2\ncfl = 0.25\n"
      << "[mesh]\ntype = \"cartesian\"\nNx = 4\nNy = 4\nLx = 1.0\nLy = 1.0\n"
      << (N_DIM == 3 ? "Nz = 2\nLz = 1.0\nperiodic = [\"x\", \"z\"]\n" : "periodic = [\"x\"]\n")
      << "[[boundaries]]\nname = \"bottom\"\ntype = \"wall_adiabatic\"\n"
      << "[[boundaries]]\nname = \"top\"\ntype = \"wall_adiabatic\"\n"
      << "[initialize]\ntype = \"analytical\"\nrho = \"1.0 + 0.1 * x\"\np = \"1.0\"\nu = "
      << (N_DIM == 3 ? "[\"0.5\", \"0.0\", \"0.0\"]" : "[\"0.5\", \"0.0\"]") << "\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\n[numerics.face_reconstruction]\ntype = \"FO\"\n"
      << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "[output]\nrun_log = \"\"\n"
      << "[statistics]\nfields = [\"RHO\", \"U_X\", \"P\"]\nproducts = [\"U_X*P\", \"RHO*RHO\"]\n"
      << "[integrals]\nfile = \"" << dir << "/integrals.csv\"\n"
      << "[[forces]]\nzone = \"bottom\"\nfile = \"" << dir << "/forces.csv\"\n"
      << "[[probes]]\nname = \"point\"\nfile = \"" << dir << "/point.csv\"\npoint = " << SIDE_POINT
      << "\nvariables = [\"RHO\", \"U\", \"P\", \"T\", \"H\"]\nreductions = [\"integral\", \"mean\"]\n"
      << "reductions_file = \"" << dir << "/point_reductions.csv\"\n"
      << "[[probes]]\nname = \"line\"\nfile = \"" << dir << "/line.csv\"\nstart = "
      << (N_DIM == 3 ? "[0.1, 0.5, 0.5]" : "[0.1, 0.5]") << "\nend = " << (N_DIM == 3 ? "[0.9, 0.5, 0.5]" : "[0.9, 0.5]")
      << "\nn_points = 4\nvariables = [\"RHO\", \"P\"]\nreductions = [\"integral\", \"rms\"]\n"
      << "reductions_file = \"" << dir << "/line_reductions.csv\"\n"
      << "[[probes]]\nname = \"vtp\"\nformat = \"vtp\"\nfile = \"" << dir << "/vtp.pvd\"\nstart = "
      << (N_DIM == 3 ? "[0.1, 0.5, 0.5]" : "[0.1, 0.5]") << "\nend = " << (N_DIM == 3 ? "[0.9, 0.5, 0.5]" : "[0.9, 0.5]")
      << "\nn_points = 4\nvariables = [\"RHO\", \"T\"]\n";
    const std::string variables =
        "variables = [\"RHO\", \"RHOU\", \"RHOE\", \"U\", \"U_X\", \"P\", \"T\", \"H\", \"CFL\", \"Q\", \"VORTICITY\", "
        "\"MEAN_*\", \"COV_*\"]\n";
    s << "[[write_data]]\nprefix = \"" << dir << "/f\"\nformat = \"vtu\"\ninterval = 2\n" << variables
      << "[[write_data]]\nprefix = \"" << dir << "/wall\"\nformat = \"vtu\"\ngeometry = \"bottom\"\ninterval = 2\n"
      << "variables = [\"P\", \"U\"]\n";
#ifdef Mallard_HAS_HDF5
    if (comm::size() == 1 || have_parallel_hdf5()) {
        s << "[[write_data]]\nprefix = \"" << dir << "/h\"\nformat = \"hdf5\"\ninterval = 2\n" << variables
          << "[[probes]]\nname = \"h5\"\nformat = \"hdf5\"\nfile = \"" << dir << "/probe.h5\"\nstart = "
          << (N_DIM == 3 ? "[0.1, 0.5, 0.5]" : "[0.1, 0.5]") << "\nend = "
          << (N_DIM == 3 ? "[0.9, 0.5, 0.5]" : "[0.9, 0.5]") << "\nn_points = 4\nvariables = [\"RHO\", \"U\"]\n";
    }
#endif
    return s.str();
}

} // namespace

TEST(FieldUnitsTest, UnitsFollowFromTheDimensions) {
    EXPECT_EQ(units_of("RHO"), "kg m-3");
    EXPECT_EQ(units_of("RHOU_Y"), "kg m-2 s-1");
    EXPECT_EQ(units_of("RHOE"), "kg m-1 s-2");
    EXPECT_EQ(units_of("U"), "m s-1");
    EXPECT_EQ(units_of("P"), "kg m-1 s-2");
    EXPECT_EQ(units_of("T"), "K");
    EXPECT_EQ(units_of("H"), "m2 s-2");
    EXPECT_EQ(units_of("Q"), "s-2");
    EXPECT_EQ(units_of("VORTICITY_Z"), "s-1");
    EXPECT_EQ(units_of("CFL"), "1");
    EXPECT_EQ(units_of("LAMBDA"), "kg m s-3 K-1");
    EXPECT_EQ(units_of("DT_H2O"), "kg m-1 s-1");
    EXPECT_EQ(units_of("OMEGA_CH4"), "kg m-3 s-1");
    EXPECT_EQ(units_of("Y_O2"), "1");
    EXPECT_EQ(field_units::describe("X_N2").long_name, "mole fraction of N2");
    EXPECT_EQ(units_of("NON_ORTHOGONALITY"), "degree") << "mallard check writes it in degrees";
    EXPECT_EQ(field_units::describe("U_Y").long_name, "y component of velocity");
}

TEST(FieldUnitsTest, DerivedFieldsTakeTheirUnitsFromTheirFields) {
    EXPECT_EQ(units_of("SOURCE_RHOE"), "kg m-1 s-3");
    EXPECT_EQ(units_of("SOURCE_RHOY_OH"), "kg m-3 s-1");
    EXPECT_EQ(units_of("MEAN_T"), "K");
    EXPECT_EQ(units_of("MEAN_Y_H2"), "1");
    EXPECT_EQ(units_of("MEAN_U"), "m s-1") << "the vector of MEAN_U_X, MEAN_U_Y";
    EXPECT_EQ(units_of("SOURCE_RHOU"), "kg m-2 s-2");
    // Covariances have the product of the units, also where names hold underscores
    EXPECT_EQ(units_of("COV_U_X_U_Y"), "m2 s-2");
    EXPECT_EQ(units_of("COV_RHO_RHO"), "kg2 m-6");
    EXPECT_EQ(units_of("COV_U_X_P"), "kg s-3");
    EXPECT_EQ(units_of("COV_Y_O2_T"), "K");
    EXPECT_EQ(field_units::describe("COV_U_X_P").long_name, "covariance of x component of velocity and pressure");
    EXPECT_EQ(units_of("PHASE_03_P"), "kg m-1 s-2");
    EXPECT_EQ(field_units::describe("PHASE_03_P").long_name, "phase average of pressure in bin 03");
}

TEST(FieldUnitsTest, UnknownFieldsHaveNoUnits) {
    for (const char * name : {"RHOX", "MEAN_", "COV_U_X", "COV_U_X_NOPE", "PHASE_P", "PHASE_x1_P", "SOURCE_P_X", "Y_"}) {
        EXPECT_FALSE(field_units::find(name).has_value()) << name;
    }
    EXPECT_THROW(field_units::describe("NOPE"), std::logic_error);
}

TEST(FieldUnitsTest, EveryWrittenFieldCarriesTheTablesUnits) {
    const std::string dir = (scratch_dir() / "mallard_field_units").string();
    if (comm::is_root()) std::filesystem::remove_all(dir);
    comm::barrier();
    {
        Solver solver;
        solver.init(parse_toml(channel(dir)));
        solver.run();
    }
    comm::barrier();
    if (!comm::is_root()) return;

    // VTU (each piece of a PVTU), boundary-zone VTU and VTP
    for (const std::string & vtu : comm::size() > 1 ? std::vector<std::string>{"f_000002_p0000.vtu", "wall_000002_p0000.vtu"}
                                                    : std::vector<std::string>{"f_000002.vtu", "wall_000002.vtu"}) {
        expect_vtk_units(dir + "/" + vtu);
    }
    const auto record = string_arrays(read_file(dir + (comm::size() > 1 ? "/f_000002_p0000.vtu" : "/f_000002.vtu")));
    const auto & names = record.at("MALLARD_FIELD_NAMES");
    for (const char * name : {"MEAN_RHO", "COV_U_X_P", "COV_RHO_RHO", "VORTICITY"}) {
        EXPECT_NE(std::find(names.begin(), names.end(), name), names.end()) << name;
    }
    expect_vtk_units(dir + "/vtp_000002.vtp", {{"arc_length", "m"}});

    // CSV: the columns' own units
    const std::string measure = dim::of(0, N_DIM, 0).udunits();
    auto point = csv_units(dir + "/point.csv");
    EXPECT_EQ(point["t"], "s");
    EXPECT_EQ(point["x"], "m");
    for (const char * v : {"RHO", "U_X", "U_Y", "P", "T", "H"}) EXPECT_EQ(point[v], units_of(v)) << v;
    auto point_reductions = csv_units(dir + "/point_reductions.csv");
    EXPECT_EQ(point_reductions["count"], "1");
    EXPECT_EQ(point_reductions["integral_P"], units_of("P")) << "a sum over points";
    EXPECT_EQ(point_reductions["mean_U_Y"], units_of("U_Y"));
    auto line_reductions = csv_units(dir + "/line_reductions.csv");
    EXPECT_EQ(line_reductions["length"], "m");
    EXPECT_EQ(line_reductions["integral_RHO"], "kg m-2") << "per unit length";
    EXPECT_EQ(line_reductions["rms_P"], units_of("P"));
    EXPECT_EQ(csv_units(dir + "/line.csv")["RHO"], "kg m-3");
    auto forces = csv_units(dir + "/forces.csv");
    EXPECT_EQ(forces["Fx_pressure"], N_DIM == 3 ? "kg m s-2" : "kg s-2") << "per unit depth in 2D";
    EXPECT_EQ(forces["Fy_viscous"], forces["Fx_pressure"]);
    auto integrals = csv_units(dir + "/integrals.csv");
    EXPECT_EQ(integrals["kinetic_energy"], N_DIM == 3 ? "kg m2 s-2" : "kg m s-2");
    EXPECT_EQ(integrals["temperature"], N_DIM == 3 ? "m3 K" : "m2 K");
    EXPECT_EQ(integrals["dilatation_squared"], measure + " s-2");
    // The provenance follows the units
    std::ifstream in(dir + "/integrals.csv");
    std::string line;
    for (int i = 0; i < 3; i++) std::getline(in, line);
    EXPECT_EQ(line.rfind("# provenance ", 0), 0u);

#ifdef Mallard_HAS_HDF5
    if (comm::size() == 1 || have_parallel_hdf5()) {
        // HDF5: CF attributes on each dataset
        h5::Handle file(H5Fopen((dir + "/h_000002.h5").c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose, "h5");
        h5::Handle group(H5Gopen2(file, "fields", H5P_DEFAULT), H5Gclose, "fields");
        H5G_info_t info;
        h5::check(H5Gget_info(group, &info), "H5Gget_info");
        EXPECT_EQ(info.nlinks, names.size()) << "the same fields as the VTU";
        for (const auto & name : names) {
            h5::Handle dset(H5Dopen2(group, name.c_str(), H5P_DEFAULT), H5Dclose, name);
            EXPECT_EQ(h5::read_string_attribute(dset, "units"), units_of(name)) << name;
            EXPECT_EQ(h5::read_string_attribute(dset, "long_name"), field_units::describe(name).long_name) << name;
            EXPECT_LE(H5Aexists(dset, "standard_name"), 0) << name;
        }
        h5::Handle mesh(H5Fopen((dir + "/h_mesh.h5").c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose, "mesh");
        h5::Handle coordinates(H5Dopen2(mesh, "nodes/coordinates", H5P_DEFAULT), H5Dclose, "coordinates");
        EXPECT_EQ(h5::read_string_attribute(coordinates, "units"), "m");

        h5::Handle probe(H5Fopen((dir + "/probe.h5").c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose, "probe");
        for (const auto & [name, units] : std::map<std::string, std::string>{
                 {"time", "s"}, {"coordinates", "m"}, {"arc_length", "m"}, {"fields/RHO", units_of("RHO")},
                 {"fields/U_Y", units_of("U_Y")}}) {
            h5::Handle dset(H5Dopen2(probe, name.c_str(), H5P_DEFAULT), H5Dclose, name);
            EXPECT_EQ(h5::read_string_attribute(dset, "units"), units) << name;
        }

        // XDMF: an Information element per Attribute, in both indexes
        for (const char * xmf : {"/h_000002.xmf", "/h.xmf"}) {
            const std::string text = read_file(dir + xmf);
            const std::regex attribute_re("<Attribute Name=\"([^\"]+)\"[^>]*>\\s*<Information Name=\"units\" Value=\"([^\"]*)\"/>"
                                          "\\s*<Information Name=\"long_name\" Value=\"([^\"]*)\"/>");
            size_t n = 0;
            for (auto it = std::sregex_iterator(text.begin(), text.end(), attribute_re); it != std::sregex_iterator(); ++it) {
                EXPECT_EQ((*it)[2].str(), units_of((*it)[1])) << xmf;
                EXPECT_EQ((*it)[3].str(), field_units::describe((*it)[1]).long_name) << xmf;
                n++;
            }
            const std::regex any_attribute("<Attribute ");
            const size_t attributes = static_cast<size_t>(std::distance(
                std::sregex_iterator(text.begin(), text.end(), any_attribute), std::sregex_iterator()));
            EXPECT_EQ(n, attributes) << xmf;
            EXPECT_GE(n, names.size()) << xmf;
        }
    }
#endif
    if (!::testing::Test::HasFailure()) std::filesystem::remove_all(dir);
}
