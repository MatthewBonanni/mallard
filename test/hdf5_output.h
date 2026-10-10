/**
 * @file hdf5_output.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Reading back HDF5 solution output in tests.
 * @version 0.5
 * @date 2026-10-04
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef HDF5_OUTPUT_H
#define HDF5_OUTPUT_H

#ifdef Mallard_HAS_HDF5

#include <gtest/gtest.h>

#include <filesystem>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "comm.h"
#include "hdf5_util.h"
#include "solver.h"
#include "test_fixtures.h"

/** @brief The bytes, as stored, of every dataset of an HDF5 output file (mesh or snapshot), by path. */
inline std::map<std::string, std::vector<char>> h5_datasets(const std::string & filename) {
    using h5::Handle;
    Handle file(H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose, "opening " + filename);
    std::map<std::string, std::vector<char>> datasets;
    for (const char * group_name : {"nodes", "cells", "fields"}) {
        if (H5Lexists(file, group_name, H5P_DEFAULT) <= 0) continue;
        Handle group(H5Gopen2(file, group_name, H5P_DEFAULT), H5Gclose, group_name);
        H5G_info_t info;
        h5::check(H5Gget_info(group, &info), "H5Gget_info");
        for (hsize_t k = 0; k < info.nlinks; k++) {
            char name[256] = {};
            H5Lget_name_by_idx(group, ".", H5_INDEX_NAME, H5_ITER_INC, k, name, sizeof(name), H5P_DEFAULT);
            Handle dset(H5Dopen2(group, name, H5P_DEFAULT), H5Dclose, name);
            Handle type(H5Dget_type(dset), H5Tclose, "H5Dget_type");
            Handle space(H5Dget_space(dset), H5Sclose, "H5Dget_space");
            std::vector<char> bytes(static_cast<size_t>(H5Sget_simple_extent_npoints(space)) * H5Tget_size(type) + 1);
            h5::check(H5Dread(dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, bytes.data()), name);
            bytes.pop_back();
            datasets[std::string(group_name) + "/" + name] = std::move(bytes);
        }
    }
    return datasets;
}

/**
 * @brief Run the input distributed and serially, each writing HDF5 output with
 *        writer (a [[write_data]] body without prefix and format); the mesh
 *        and snapshot files must hold the same bytes.
 */
inline void expect_hdf5_output_matches_serial(const std::string & input, const std::string & writer,
                                              const std::string & dir, const std::vector<uint64_t> & steps) {
    if (comm::is_root()) std::filesystem::remove_all(dir);
    comm::barrier();
    auto with_writer = [&](const std::string & name) {
        return input + "[[write_data]]\nprefix = \"" + dir + "/" + name + "\"\nformat = \"hdf5\"\n" + writer;
    };
    Solver distributed;
    distributed.init(parse_toml(with_writer("d")));
    distributed.run();
    Solver serial;
    serial.set_distributed(false);
    serial.init(parse_toml(with_writer("s")));
    serial.run();
    comm::barrier();
    if (comm::is_root()) {
        std::vector<std::string> stems = {"mesh"};
        for (uint64_t step : steps) {
            std::ostringstream stem;
            stem << std::setw(6) << std::setfill('0') << step;
            stems.push_back(stem.str());
        }
        for (const auto & stem : stems) {
            const auto a = h5_datasets(dir + "/d_" + stem + ".h5");
            const auto b = h5_datasets(dir + "/s_" + stem + ".h5");
            EXPECT_FALSE(b.empty()) << stem;
            EXPECT_EQ(a.size(), b.size()) << stem;
            for (const auto & [name, bytes] : b) {
                EXPECT_FALSE(bytes.empty()) << stem << " " << name;
                EXPECT_TRUE(a.count(name) && a.at(name) == bytes) << stem << " " << name << " on " << comm::size()
                                                                   << " ranks";
            }
        }
    }
    comm::barrier();
}

#endif // Mallard_HAS_HDF5

#endif // HDF5_OUTPUT_H
